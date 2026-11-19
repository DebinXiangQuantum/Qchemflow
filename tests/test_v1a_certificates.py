import copy
import json
import random

import numpy as np
import pytest

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.check.static import infer_relations
from chemflow.domains import load_registry
from chemflow.ids import canonical_json, content_hash
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation
from chemflow.runtime import Executor
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.runtime.task import freeze_task
from chemflow.verify import check_certificate, prove

MODEL = {"U": 2., "eps_bath": [0.], "t_bath": [0.5], "unit": "hartree"}
SECTOR = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
SPEC = {"question": "H2", "target": {
    "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
    "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
    "sector": SECTOR, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": .0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 10000},
    "allowed_effects": ["pure", "simulator"], "decision": "choose"}


def spectral_plan(sampled=False):
    registry = load_registry(["molecular", "algorithms", "dynamics"])
    with build.program("certificate-response") as builder:
        hamiltonian = builder.node("impurity_model", **MODEL)
        spectra = []
        for index, alpha in enumerate((1, 2, 0)):
            sector = builder.sector(n_alpha=alpha, n_beta=1, n_orb=2)
            if sampled and index == 0:
                prep = builder.node("prepare_state", hamiltonian=hamiltonian, sector=sector,
                                    ansatz="hf")
                samples = builder.node("sample_configurations", prep=prep, shots=1, seed=7)
                subspace = builder.node("subspace_from_samples", samples=samples, sector=sector)
            else:
                subspace = builder.node("full_subspace", sector=sector)
            spectra.append(builder.node("rayleigh_ritz", hamiltonian=hamiltonian,
                                        subspace=subspace))
        response = builder.node("spectral_function", ground=spectra[0], particle=spectra[1],
                                hole=spectra[2], orbital=0, spin="alpha",
                                omega={"start": -4., "stop": 4., "n": 51, "unit": "hartree"},
                                eta=.1)
        builder.refine(response, "lehmann")
        builder.bind(response, "lehmann_from_spectra")
    return builder.to_ir(), registry, spectra, response


@pytest.mark.parametrize("sampled", [False, True])
def test_ritz_and_lehmann_require_complete_coverage(sampled):
    program, registry, spectra, response = spectral_plan(sampled)
    computed = Executor(registry).run(program)
    for identifier, expected in (
        (spectra[0].id, "variational_upper_bound" if sampled else "equivalent"),
        (response.id, "approximate" if sampled else "equivalent"),
    ):
        certificate = prove(program, identifier, registry=registry, computed=computed)
        assert certificate["relation"] == expected
        assert check_certificate(json.loads(canonical_json(certificate)), program,
                                 registry=registry, computed=computed).valid
    assert computed[response.id].diagnostics["relation_status"] == "proved"


def test_hand_built_composition_downgrades_a_false_declaration():
    registry = molecular_registry()
    registry.refinements["mp2"] = registry.refinements["mp2"].model_copy(
        update={"relation": TargetRelation.equivalent})
    with build.program("false-declaration") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(**SECTOR)
        exact = builder.ground_energy(hamiltonian, sector)
        approximation = builder.ground_energy(hamiltonian, sector)
        builder.refine(approximation, "mp2")
        builder.bind(approximation, "mp2")
        difference = builder.node("energy_difference", left=exact, right=approximation)
    program = builder.to_ir()
    assert infer_relations(program, registry)[difference.id] == "equivalent"
    computed = Executor(registry).run(program)
    assert computed[difference.id].relation == "approximate"
    assert computed[difference.id].diagnostics["declared_relation"] == "equivalent"
    assert computed[difference.id].diagnostics["relation_status"] == "proved"


def test_missing_resolution_downgrades_and_old_certificate_is_rejected():
    program, registry, spectra, response = spectral_plan()
    computed = Executor(registry).run(program)
    honest = prove(program, response.id, registry=registry, computed=computed)
    original = computed[spectra[1].id]
    truncated = computed["_artifacts"].put(np.zeros((2, 1)))
    computed[spectra[1].id] = original.model_copy(update={"resolution_vectors_ref": truncated})
    assert not check_certificate(honest, program, registry=registry, computed=computed).valid
    weaker = prove(program, response.id, registry=registry, computed=computed)
    assert weaker["relation"] == "approximate"
    assert check_certificate(weaker, program, registry=registry, computed=computed).valid


def test_nonsector_basis_and_nonhermitian_integrals_cannot_prove_bounds():
    program, registry, spectra, _ = spectral_plan()
    computed = Executor(registry).run(program)
    root = spectra[0].id
    honest = prove(program, root, registry=registry, computed=computed)
    node = next(node for node in program.nodes if node.id == root)
    space_id = node.inputs["subspace"]
    original = computed[space_id]
    computed[space_id] = original.model_copy(update={"determinants": (0,)})
    assert not check_certificate(honest, program, registry=registry, computed=computed).valid
    assert prove(program, root, registry=registry, computed=computed)["relation"] == "approximate"
    computed[space_id] = original
    data = copy.copy(computed[node.inputs["hamiltonian"]])
    bad = computed["_artifacts"].put(np.array([[0., 1.], [0., 0.]]))
    data.hamiltonian = data.hamiltonian.model_copy(update={"h1_ref": bad})
    computed[node.inputs["hamiltonian"]] = data
    assert prove(program, root, registry=registry, computed=computed)["relation"] == "approximate"


def test_kernel_single_field_fuzz():
    program, registry, _, response = spectral_plan()
    computed = Executor(registry).run(program)
    honest = json.loads(canonical_json(prove(program, response.id, registry=registry,
                                            computed=computed)))

    def leaves(value, path=()):
        if isinstance(value, dict):
            return [item for key, child in value.items() for item in leaves(child, (*path, key))]
        if isinstance(value, list):
            return [item for index, child in enumerate(value)
                    for item in leaves(child, (*path, index))]
        return [path]

    paths = leaves(honest)
    generator = random.Random(728)
    for _ in range(250):
        path = generator.choice(paths)
        forged = copy.deepcopy(honest)
        destination = forged
        for key in path[:-1]:
            destination = destination[key]
        previous = destination[path[-1]]
        destination[path[-1]] = (not previous if isinstance(previous, bool) else
                                previous + 1 if type(previous) is int else "mutated")
        verdict = check_certificate(forged, program, registry=registry, computed=computed)
        assert not verdict.valid or forged == honest


def test_kernel_rejects_missing_premise_and_unknown_axiom():
    program, registry, spectra, _ = spectral_plan()
    computed = Executor(registry).run(program)
    honest = prove(program, spectra[0].id, registry=registry, computed=computed)
    for field, value in (("uses", []), ("lemma", "binding:unreviewed"),
                         ("node", spectra[1].id)):
        forged = copy.deepcopy(honest)
        forged["steps"][-1][field] = value
        assert not check_certificate(forged, program, registry=registry, computed=computed).valid


def test_prove_tool_executes_only_closure_and_persists_artifacts(tmp_path):
    store = TaskStore("v1a-test")
    record = store.freeze(TaskSpec.model_validate(SPEC), "v1a-test")
    registry = molecular_registry()
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store, plan_directory=tmp_path)
    empty = tools.call("designer", "plan.new", {"identifier": "proof-tool"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty})["data"]
    plan = tools.plans.get(base["plan_id"])
    target = next(node.id for node in plan.nodes if node.refinement == "hf")
    result = tools.call("critic", "algorithm.prove", {
        "plan_id": base["plan_id"], "node_id": target})["data"]
    assert result["kernel_verdict"]["valid"], result
    assert result["certificate"]["program_hash"] == plan.hash
    identifier = result["certificate_id"]
    certificate = json.loads((tmp_path / "certificates" / f"{identifier}.json").read_text())
    assert identifier == content_hash(certificate)
    assert {node.refinement for node in plan.nodes if node.id in
            {step["node"] for step in certificate["steps"]}} == {None, "hf"}
    assert list((tmp_path / "artifacts").iterdir())
    assert tools.call("designer", "algorithm.prove", {
        "plan_id": base["plan_id"], "node_id": target})["data"] == result


def test_artifact_store_persistence_round_trip(tmp_path):
    store = ArtifactStore()
    reference = store.put(np.array([5, 6], dtype=np.int64))
    store.persist(tmp_path)
    assert ArtifactStore(tmp_path).get(reference).tolist() == [5, 6]


def test_unknown_binding_cannot_satisfy_a_relation_gate():
    registry = molecular_registry()
    registry.register(registry.implementations["fci_pyscf"].model_copy(update={"id": "unknown_fci"}))
    with build.program("unknown-binding") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(**SECTOR)
        energy = builder.ground_energy(hamiltonian, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "unknown_fci")
        builder.accept(energy, relation="equivalent")
    program = builder.to_ir()
    computed = Executor(registry).run(program, freeze_task(program))
    assert computed[energy.id].relation == "equivalent"
    assert computed[energy.id].diagnostics["relation_status"] == "declared"
    assert not computed["gates"][0].passed
    assert not check_certificate(computed["relation_certificates"][energy.id], program,
                                 registry=registry, computed=computed).valid


@pytest.mark.parametrize("reverse,expected", [(False, "variational_upper_bound"),
                                             (True, "approximate")])
def test_difference_sign_rule(reverse, expected):
    registry = molecular_registry()
    with build.program("difference-signs") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(**SECTOR)
        exact = builder.ground_energy(hamiltonian, sector)
        trial = builder.ground_energy(hamiltonian, sector)
        builder.refine(trial, "hf")
        builder.bind(trial, "hf")
        difference = builder.node("energy_difference", left=exact if reverse else trial,
                                  right=trial if reverse else exact)
    program = builder.to_ir()
    computed = Executor(registry).run(program)
    assert computed[difference.id].relation == expected
    assert computed[difference.id].diagnostics["relation_status"] == "proved"


def test_vqe_requires_the_actual_nonzero_sector_state():
    registry = molecular_registry()
    with build.program("trial-witness") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(**SECTOR)
        energy = builder.ground_energy(hamiltonian, sector)
        builder.refine(energy, "vqe")
        builder.bind(energy, "vqe_statevector")
    program = builder.to_ir()
    computed = Executor(registry).run(program)
    honest = prove(program, energy.id, registry=registry, computed=computed)
    assert honest["relation"] == "variational_upper_bound"
    original = computed[energy.id]
    for vector in (np.zeros(4), np.ones(4), np.ones(16)):
        state_ref = computed["_artifacts"].put(vector)
        computed[energy.id] = original.model_copy(update={"diagnostics": {
            **original.diagnostics, "state_ref": state_ref}})
        assert not check_certificate(honest, program, registry=registry, computed=computed).valid
        weaker = prove(program, energy.id, registry=registry, computed=computed)
        assert weaker["relation"] == "approximate"
        assert check_certificate(weaker, program, registry=registry, computed=computed).valid


def test_missing_artifact_downgrades_instead_of_trusting_a_count():
    program, registry, spectra, _ = spectral_plan()
    computed = Executor(registry).run(program)
    computed.pop("_artifacts")
    certificate = prove(program, spectra[0].id, registry=registry, computed=computed)
    assert certificate["relation"] == "approximate"
    assert check_certificate(certificate, program, registry=registry, computed=computed).valid


def test_kernel_rejects_artifact_content_substitution():
    program, registry, spectra, response = spectral_plan()
    computed = Executor(registry).run(program)
    honest = prove(program, response.id, registry=registry, computed=computed)
    store = computed["_artifacts"]
    reference = computed[spectra[0].id].resolution_vectors_ref
    forged = store.put(np.zeros_like(store.get(reference)))
    store.memory[reference] = store.memory[forged]
    assert not check_certificate(honest, program, registry=registry, computed=computed).valid
    weaker = prove(program, response.id, registry=registry, computed=computed)
    assert weaker["relation"] == "approximate"
    assert check_certificate(weaker, program, registry=registry, computed=computed).valid


def test_prove_tool_cannot_bypass_the_shot_budget():
    from chemflow.diagnostics import AdapterError

    spec = copy.deepcopy(SPEC)
    spec["budget"]["sim_shots"] = 0
    store = TaskStore("v1a-budget")
    record = store.freeze(TaskSpec.model_validate(spec), "v1a-budget")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    empty = tools.call("designer", "plan.new", {"identifier": "budget-proof"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty})["data"]
    candidate = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"],
        "candidates": [{"template": "sqd", "shots": 2048}]})["data"]
    with pytest.raises(AdapterError, match="CF-E070"):
        tools.call("designer", "algorithm.prove", {"plan_id": candidate["plan_id"],
                    "node_id": next(iter(candidate["ids"].values()))})


def test_linear_observable_estimates_remain_stochastic():
    registry = molecular_registry()
    with build.program("sample-difference") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        sampled = builder.node("sample", operator=encoding, prep=prep, shots=32, seed=7)
        exact = builder.node("expectation", operator=encoding, prep=prep)
        difference = builder.node("energy_difference", left=sampled, right=exact)
    program = builder.to_ir()
    computed = Executor(registry).run(program)
    certificate = prove(program, difference.id, registry=registry, computed=computed)
    assert certificate["relation"] == "stochastic_estimate"
    assert check_certificate(certificate, program, registry=registry, computed=computed).valid


def test_exact_lattice_solver_certifies_a_nondefault_particle_sector():
    registry = load_registry(["molecular", "dynamics"])
    with build.program("off-half-filling") as builder:
        hamiltonian = builder.node("impurity_model", **MODEL)
        sector = builder.sector(n_alpha=1, n_beta=0, n_orb=2)
        energy = builder.ground_energy(hamiltonian, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "fci_pyscf")
    program = builder.to_ir()
    computed = Executor(registry).run(program)
    certificate = prove(program, energy.id, registry=registry, computed=computed)
    assert certificate["relation"] == "equivalent"
    assert check_certificate(certificate, program, registry=registry, computed=computed).valid


def test_sci_exports_exact_spin_strings_not_fractional_occupations():
    registry = molecular_registry()
    with build.program("sci-strings") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        hamiltonian = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(**SECTOR)
        energy = builder.ground_energy(hamiltonian, sector, select_cutoff=1e-8)
        builder.refine(energy, "sci")
        builder.bind(energy, "sci_pyscf")
    program = builder.to_ir()
    computed = Executor(registry).run(program)
    estimate = computed[energy.id]
    store = computed["_artifacts"]
    assert store.get(estimate.diagnostics["determinants_ref"]).shape == (4, 2)
    honest = prove(program, energy.id, registry=registry, computed=computed)
    assert honest["relation"] == "variational_upper_bound"
    computed[energy.id] = estimate.model_copy(update={"diagnostics": {
        **estimate.diagnostics, "determinants_ref": store.put(np.array([[1.5, 1.5]]))}})
    assert not check_certificate(honest, program, registry=registry, computed=computed).valid
    assert prove(program, energy.id, registry=registry, computed=computed)["relation"] == "approximate"


def test_certificate_survives_canonical_program_round_trip():
    program, registry, _, response = spectral_plan()
    computed = Executor(registry).run(program)
    certificate = prove(program, response.id, registry=registry, computed=computed)
    restored = type(program).from_json(program.to_json())
    assert restored.hash == program.hash
    assert check_certificate(certificate, restored, registry=registry, computed=computed).valid
    assert content_hash(prove(restored, response.id, registry=registry, computed=computed)) == (
        content_hash(certificate))
