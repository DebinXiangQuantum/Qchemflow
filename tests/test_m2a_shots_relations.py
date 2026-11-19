from chemflow.check.static import check, infer_relations
from chemflow.impls.pyscf_impls import build_molecule, fci_pyscf
from chemflow.ir.builder import Builder
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.runtime.executor import Executor
from chemflow.runtime.task import freeze_task
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def test_shot_vqe_relation_and_cost():
    builder = Builder("shot-h2")
    molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
    model = builder.electronic_model(basis="sto-3g")
    hamiltonian = builder.hamiltonian(molecule, model)
    sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
    energy = builder.ground_energy(hamiltonian, sector)
    builder.refine(energy, "vqe", ansatz="uccsd", measurement="shots", shots=32,
                   maxiter=10, seed=42)
    builder.bind(energy, "vqe_statevector")
    program = builder.to_ir()
    registry = molecular_registry()
    assert infer_relations(program, registry)[energy.id] == TargetRelation.approximate
    result = Executor(registry).run(program)[energy.id]
    assert result.relation == TargetRelation.approximate
    assert result.cost.sim_shots > 0
    assert result.cost.qpu_shots in (None, 0)
    builder.accept(energy, relation="equivalent")
    program = builder.to_ir()
    assert "CF-E050" in [issue.code for issue in check(program, registry, freeze_task(program))]


def test_spin_target_weakens_only_on_drift():
    def run(spin_adapted, initial=None, maxiter=100):
        builder = Builder("o2-spin")
        molecule = builder.molecule("O 0 0 0; O 0 0 1.21")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model, active={"n_elec": 2, "n_orb": 2})
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        energy = builder.ground_energy(hamiltonian, sector)
        params = {"spin_adapted": spin_adapted, "s": 0, "maxiter": maxiter}
        if initial is not None:
            params["initial"] = initial
        builder.refine(energy, "vqe", **params)
        builder.bind(energy, "vqe_statevector")
        return Executor(molecular_registry()).run(builder.to_ir())[energy.id]

    drift = run(False, [0.3, 0, 0], 1)
    assert drift.diagnostics["s2"] > 1e-4
    assert drift.relation == TargetRelation.approximate
    assert drift.provenance.relation_weakening_reason
    singlet = run(True)
    assert singlet.relation == TargetRelation.variational_upper_bound
    store = ArtifactStore()
    data = build_molecule("O 0 0 0; O 0 0 1.21", ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=2, n_orb=2))
    exact, _ = fci_pyscf(data, store, 0)
    assert abs(singlet.value.value - exact) < 1e-6
