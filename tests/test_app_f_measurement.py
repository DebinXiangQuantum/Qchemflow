from types import SimpleNamespace

import numpy as np
import pytest

from chemflow.quantum.pauli import allocate_shots, estimate, qwc_groups


def test_covariance_and_allocation():
    terms = {"ZI": 1., "IZ": 1., "II": 3.}
    groups = qwc_groups(terms)
    result = estimate(terms, groups, [{"00": 50, "11": 50}])
    assert result["energy"] == 3
    assert result["stat_err"] == pytest.approx(np.sqrt(4/99))
    assert result["per_group_stat_err"] == [result["stat_err"]]
    assert allocate_shots(11, [1, 2, 3]) == [4, 4, 3]
    assert allocate_shots(10, [1, 2], [1, 3]) == [3, 7]


@pytest.mark.parametrize("ansatz", ["uccsd", "lucj"])
def test_circuit_matches_sector_ansatz(ansatz):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from chemflow.backends.circuits import build_circuit
    from chemflow.impls.ansatz import uccsd_pool, uccsd_state
    from chemflow.impls.fci import sector_states
    from chemflow.impls.lucj import LucjParams, lucj_state
    from chemflow.types import CircuitSpec, Sector

    sector = Sector(n_orb=3, n_alpha=1, n_beta=1)
    generators = uccsd_pool(sector)[1]
    size = len(generators) if ansatz == "uccsd" else LucjParams.zeros(3).size
    values = np.random.default_rng(9).normal(0, .2, size)
    expected = (uccsd_state(sector, values, generators) if ansatz == "uccsd" else
                lucj_state(sector, [LucjParams.from_vector(values, 3)]))
    circuit = build_circuit(CircuitSpec(ansatz=ansatz, n_orb=3, n_alpha=1, n_beta=1,
                                       layers=(tuple(values),)))
    actual = Statevector.from_instruction(circuit).data[sector_states(sector)]
    assert abs(np.vdot(expected, actual)) == pytest.approx(1, abs=1e-10)


def test_offline_cloud_jobs_and_replay(tmp_path, monkeypatch):
    pytest.importorskip("qiskit")
    from chemflow import build
    from chemflow.backends import lqcloud
    from chemflow.ops.molecular import molecular_registry
    from chemflow.quantum.vqe import register_sampled_vqe
    from chemflow.runtime import Executor

    submissions = []

    def submit(*args, **kwargs):
        submissions.append(kwargs)
        return SimpleNamespace(job_id=f"offline-{len(submissions)}",
            result=lambda **kwargs: SimpleNamespace(success=True, data={"memory": ["0000"]*8}),
            status_info=lambda: {"status": "completed", "shots": 8, "execution_time": .1})

    backend = lqcloud.LqcloudBackend(state_dir=tmp_path,
        provider_factory=lambda key: SimpleNamespace(get_backend=lambda name:
            SimpleNamespace(config={"status": "active", "billing_multiplier": 1}, run=submit)))
    monkeypatch.setenv(backend.api_key_env, "offline")
    monkeypatch.setattr(lqcloud, "best_layout", lambda config, size: list(range(size)))
    monkeypatch.setattr(lqcloud, "to_lqcloud", lambda circuit: "offline-program")
    registry = molecular_registry()
    register_sampled_vqe(registry, backend=backend)
    with build.program("offline") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                  builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_orb=2, n_alpha=1, n_beta=1)
        energy = builder.ground_energy(ham, sector)
        builder.refine(energy, "vqe_sampled", max_evaluations=3, shots_per_group=8, seed=3)
        builder.bind(energy, "vqe_lqcloud")
    first = Executor(registry).run(builder.to_ir())[energy.id]
    count = len(submissions)
    assert count == first.diagnostics["evaluations"] * first.diagnostics["groups"]
    assert first.cost.qpu_shots == count * 8
    second = Executor(registry).run(builder.to_ir())[energy.id]
    assert len(submissions) == count
    assert second.cost.qpu_shots == 0
    assert first.value == second.value


@pytest.mark.parametrize("params", [{"max_evaluations": True}, {"shots_per_group": 0},
                                   {"sampler": "lqcloud"}, {"optimizer": "unknown"}])
def test_static_rejects_invalid_sampled_parameters(params):
    from chemflow.check.static import parameter_diagnostics
    from chemflow.domains import load_registry
    from chemflow.ir.schema import Node, Program

    node = Node(id="energy", op="ground_energy", binding="vqe_shots_ideal",
                refinement="vqe_sampled", params=params)
    assert parameter_diagnostics(Program(name="invalid", nodes=[node]), load_registry(["quantum"]),
                                 require_complete=False)


@pytest.mark.parametrize("binding,optimizer", [("vqe_shots_ideal", "spsa"),
                                             ("vqe_aer_mq02", "cobyla")])
def test_seeded_emulators_and_final_measurement(binding, optimizer):
    pytest.importorskip("qiskit")
    if binding == "vqe_aer_mq02":
        pytest.importorskip("qiskit_aer")
    from chemflow import build
    from chemflow.domains import load_registry
    from chemflow.runtime import Executor

    with build.program("seeded") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                  builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_orb=2, n_alpha=1, n_beta=1)
        energy = builder.ground_energy(ham, sector)
        builder.refine(energy, "vqe_sampled", optimizer=optimizer, max_evaluations=5,
                       shots_per_group=100, seed=23)
        builder.bind(energy, binding)
    registry = load_registry(["quantum"])
    first = Executor(registry).run(builder.to_ir())[energy.id]
    second = Executor(registry).run(builder.to_ir())[energy.id]
    assert first.value == second.value
    assert 1 <= first.diagnostics["evaluations"] <= 5
    assert first.cost.sim_shots == (first.diagnostics["evaluations"] *
                                    first.diagnostics["groups"] * 100)
    assert first.diagnostics["relation_status"] == "proved"


def test_sampled_template_is_discoverable_and_deployable():
    from chemflow import build
    from chemflow.agent.orchestrator import Orchestrator
    from chemflow.check.static import check
    from chemflow.domains import load_registry

    with build.program("screen") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                  builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_orb=2, n_alpha=1, n_beta=1)
        energy = builder.ground_energy(ham, sector)
    plan = builder.to_ir()
    plan.references = [energy.id]
    plan.targets = []
    trial, identifiers = Orchestrator._candidate_plan(plan, [{"template": "vqe_sampled"}])
    candidate = next(node for node in trial.nodes if node.id in identifiers.values())
    registry = load_registry(["quantum"])
    assert registry.search("vqe_sampled", kind="refinement")
    assert candidate.binding == "vqe_shots_ideal"
    assert not check(trial, registry, purpose="deploy")
    candidate.binding = "vqe_lqcloud"
    assert not check(trial, registry, purpose="production")
    candidate.params["shots_per_group"] = 50001
    assert any(issue.code == "CF-E140" for issue in check(trial, registry, purpose="deploy"))
