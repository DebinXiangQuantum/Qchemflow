import pytest

from chemflow import Q, build
from chemflow.check import check
from chemflow.diagnostics import CheckError
from chemflow.ids import content_hash
from chemflow.ir.schema import Program
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import RELATION_TABLE, TargetRelation, compose
from chemflow.runtime import Executor, freeze_task
from chemflow.types import (
    BackendKind,
    ConstantLedger,
    CostRecord,
    Estimate,
    Evidence,
    Provenance,
    QuantityKind,
)


def test_units() -> None:
    assert Q(1, "hartree").to("eV").value == pytest.approx(27.211386, rel=1e-6)
    assert Q(1, "angstrom").to("bohr").to("angstrom").value == pytest.approx(1)
    with pytest.raises(ValueError):
        _ = Q(1, "hartree") + Q(1, "angstrom")


def test_ir_hash_and_gate() -> None:
    with build.program("identity") as builder:
        node = builder.molecule("H 0 0 0", charge=0, spin=0)
        builder.accept(node, relation="equivalent")
    program = builder.to_ir()
    original_hash = program.hash
    assert Program.from_json(program.to_json()).hash == program.hash
    assert content_hash({"x": 1.0, "y": {"b": 2, "a": 3}}) == content_hash(
        {"y": {"a": 3, "b": 2}, "x": 1.0})
    old_hash = program.gates[0].frozen_hash
    program.gates[0].relation = "approximate"
    assert program.hash != original_hash
    assert not program.gates[0].intact()
    assert program.gates[0].frozen_hash == old_hash


def test_relation_table_exhaustive() -> None:
    assert len(RELATION_TABLE) == 25
    for upstream in TargetRelation:
        for downstream in TargetRelation:
            actual = compose(upstream, downstream)
            if TargetRelation.model_change in (upstream, downstream):
                assert actual == TargetRelation.model_change
            elif upstream == TargetRelation.equivalent:
                assert actual == downstream
            elif downstream == TargetRelation.equivalent:
                assert actual == upstream
            elif upstream == downstream == TargetRelation.variational_upper_bound:
                assert actual == TargetRelation.variational_upper_bound
            elif upstream == downstream == TargetRelation.stochastic_estimate:
                assert actual == TargetRelation.stochastic_estimate
            else:
                assert actual == TargetRelation.approximate


def test_reject_mutations() -> None:
    registry = molecular_registry()

    def rejection(program: Program, code: str) -> None:
        assert code in [item.code for item in check(program, registry, task=freeze_task(program))]
        with pytest.raises(CheckError):
            Executor(registry).run(program, task=freeze_task(program))

    with build.program("context") as builder:
        model = builder.electronic_model(basis="sto-3g")
        first = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.8"), model)
        second = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 1.0"), model)
        first_encoding = builder.node("encode", hamiltonian=first)
        second_encoding = builder.node("encode", hamiltonian=second)
        state = builder.node("prepare", hamiltonian=second, encoding=second_encoding)
        builder.node("expectation", operator=first_encoding, prep=state)
    rejection(builder.to_ir(), "CF-E010")

    with build.program("encoding") as builder:
        model = builder.electronic_model(basis="sto-3g")
        first = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.8"), model)
        second = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 1.0"), model)
        builder.node("prepare", hamiltonian=second,
                     encoding=builder.node("encode", hamiltonian=first))
    rejection(builder.to_ir(), "CF-E090")

    with build.program("models") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        first = builder.ground_energy(builder.hamiltonian(
            molecule, builder.electronic_model(basis="sto-3g")), sector)
        second = builder.ground_energy(builder.hamiltonian(
            molecule, builder.electronic_model(basis="6-31g")), sector)
        builder.node("energy_difference", left=first, right=second)
    rejection(builder.to_ir(), "CF-E011")

    with build.program("kinds") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        ham = builder.hamiltonian(molecule, builder.electronic_model(basis="sto-3g"))
        energy = builder.ground_energy(ham, builder.sector(n_alpha=1, n_beta=1, n_orb=2))
        difference = builder.node("energy_difference", left=energy, right=energy)
        builder.node("energy_difference", left=difference, right=energy)
    rejection(builder.to_ir(), "CF-E002")

    with build.program("spin") as builder:
        mol = builder.molecule("H 0 0 0; H 0 0 0.74")
        ham = builder.hamiltonian(mol, builder.electronic_model(basis="sto-3g"))
        builder.ground_energy(ham, builder.sector(n_alpha=2, n_beta=0, n_orb=2), s=0)
    rejection(builder.to_ir(), "CF-E020")

    with build.program("refinement") as builder:
        mol = builder.molecule("H 0 0 0; H 0 0 0.74")
        ham = builder.hamiltonian(mol, builder.electronic_model(basis="sto-3g"))
        energy = builder.ground_energy(ham, builder.sector(n_alpha=1, n_beta=1, n_orb=2))
        builder.refine(energy, "hf")
        builder.bind(energy, "fci_pyscf")
    rejection(builder.to_ir(), "CF-E050")

    with build.program("units") as builder:
        builder.molecule("H 0 0 0", unit="furlong")
    rejection(builder.to_ir(), "CF-E001")

    with build.program("evidence") as builder:
        builder.molecule("H 0 0 0", evidence=[{"status": "certified_bound"}])
    rejection(builder.to_ir(), "CF-E040")

    with build.program("unknown_effect") as builder:
        node = builder.molecule("H 0 0 0")
        node.effects = "unknown"
    rejection(builder.to_ir(), "CF-E080")


def test_evidence_and_constants() -> None:
    with pytest.raises(ValueError, match="CF-E030"):
        ConstantLedger().add("nuclear_repulsion", Q(1, "hartree")).add(
            "nuclear_repulsion", Q(2, "hartree"))
    with pytest.raises(ValueError, match="CF-E040"):
        Evidence(status="certified_bound")
    with pytest.raises(ValueError, match="CF-E040"):
        Evidence._runtime(status="certified_bound", artifact_ref="invented")


def test_budget_before_execution_and_cache() -> None:
    registry = molecular_registry()
    calls = []
    registry.implementations["molecule"].resource_model = lambda **kwargs: {"qpu_shots": 2}
    registry.implementations["molecule"].adapter = lambda store, **kwargs: calls.append(1)
    with build.program("budget") as builder:
        builder.molecule("H 0 0 0")
        builder.budget(qpu_shots=1)
    executor = Executor(registry)
    with pytest.raises(CheckError):
        executor.run(builder.to_ir())
    assert not calls
    builder.budget(qpu_shots=2)
    executor.run(builder.to_ir())
    executor.run(builder.to_ir())
    assert calls == [1] and executor.cache_hits == 1


def test_unknown_effect_remains_invalid_with_frozen_task() -> None:
    with build.program("unknown_effect") as builder:
        node = builder.molecule("H 0 0 0")
        node.effects = "unknown"
    program = builder.to_ir()
    assert "CF-E080" in [issue.code for issue in check(program, molecular_registry())]
    assert "CF-E080" in [issue.code for issue in check(
        program, molecular_registry(), task=freeze_task(program))]


def test_derived_estimates_and_runtime_backend_provenance() -> None:
    with build.program("derived") as builder:
        model = builder.electronic_model(basis="sto-3g")
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        first = builder.ground_energy(builder.hamiltonian(
            builder.molecule("H 0 0 0; H 0 0 0.74"), model), sector)
        second = builder.ground_energy(builder.hamiltonian(
            builder.molecule("H 0 0 0; H 0 0 0.80"), model), sector)
        difference = builder.node("energy_difference", left=first, right=second)
        comparison = builder.node("compare", left=first, right=second)
    program = builder.to_ir()
    registry = molecular_registry()
    results = Executor(registry).run(program, task=freeze_task(program))
    for node in (difference, comparison):
        assert isinstance(results[node.id], Estimate)
        assert results[node.id].kind == QuantityKind.energy_difference
        assert results[node.id].value.value == pytest.approx(
            results[first.id].value.value - results[second.id].value.value)
    registry.implementations["compare"].adapter = lambda store, left, right: Estimate(
        value=Q(0, "hartree"), kind=QuantityKind.energy_difference, target=comparison.id,
        relation=TargetRelation.equivalent, context=left.context,
        provenance=Provenance(impl_id="compare", impl_version="1", backend_kind=BackendKind.qpu,
                              device="fake"), cost=CostRecord())
    with pytest.raises(CheckError) as error:
        Executor(registry).run(program, task=freeze_task(program))
    assert "CF-E060" in {diagnostic.code for diagnostic in error.value.diagnostics}
