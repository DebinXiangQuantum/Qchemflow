import pytest
from pyscf import gto, scf
from scipy.optimize import minimize_scalar

import qcl


@qcl.workflow
def optimized_energy(molecule, cpu):
    optimized = qcl.optimize(molecule).to(cpu)
    return qcl.scf.hf(optimized).to(cpu).energy


def test_optimized_geometry_flows_to_scf():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = optimized_energy.compile(molecule, cpu)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "geometry.optimize", "scf.hf",
    ]
    energy = plan.run().value
    reference = minimize_scalar(
        lambda distance: scf.RHF(gto.M(
            atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g",
            verbose=0)).run().e_tot,
        bounds=(0.6, 0.9), method="bounded",
    )
    assert reference.success
    assert energy == pytest.approx(reference.fun, abs=1e-6)

    @qcl.workflow
    def geometry(mol, device):
        return qcl.optimize(mol).to(device)

    result = geometry.compile(molecule, cpu).run()
    assert isinstance(result, qcl.GeometryResult)
    assert result.molecule.basis == "sto-3g"
    assert result.provenance["implementation"] == "geometry.optimize/chemflow-geometric"


def test_saddle_requires_matching_endpoint_composition_and_cpu():
    reactant = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    product = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    with pytest.raises(qcl.CompileError):
        @qcl.workflow
        def invalid(cpu):
            return qcl.engines.sella.saddle(
                reactant, product, qcl.Molecule("He 0 0 0", "sto-3g")).to(cpu)

        invalid.compile(qcl.Cluster.local().cpus[0])
    @qcl.workflow
    def saddle(cpu):
        return qcl.engines.sella.saddle(reactant, product, reactant).to(cpu)

    with pytest.raises(qcl.CompileError) as error:
        saddle.compile(qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_analytic_pes_optimizes_with_fixed_bond():
    @qcl.workflow
    def constrained(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.Analytic())
        guess = molecule.set_bond(0, 1, 1.2)
        optimized = qcl.optimize(pes, guess, fixed=[qcl.Bond(0, 1)]).to(cpu)
        return qcl.scf.hf(optimized).to(cpu).energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = constrained.compile(molecule, cpu)
    assert plan.explain(format="json")["operations"][0]["op"] == (
        "geometry.constrained_optimize")
    result = plan.run()
    reference = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 1.2", basis="sto-3g",
                              verbose=0)).run().e_tot
    assert result.value == pytest.approx(reference, abs=1e-7)


def test_pes_rejects_callback_that_changes_the_question():
    @qcl.workflow
    def invalid(molecule, other, cpu):
        pes = qcl.PES(lambda _geometry, lane: qcl.scf.hf(other).to(lane),
                      gradient=qcl.Analytic())
        return qcl.optimize(pes, molecule, fixed=[qcl.Bond(0, 1)]).to(cpu)

    first = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    second = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(first, second, qcl.Cluster.local().cpus[0])
    assert any(item["code"] == "QCL-E103" for item in error.value.diagnostics)


def test_pes_rejects_open_shell_reference_mismatch():
    @qcl.workflow
    def open_shell(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.Analytic())
        return qcl.optimize(pes, molecule, fixed=[qcl.Bond(0, 1)]).to(cpu)

    with pytest.raises(qcl.CompileError) as error:
        open_shell.compile(qcl.Molecule("H 0 0 0; He 0 0 1", "sto-3g", spin=1),
                           qcl.Cluster.local().cpus[0])
    assert any(item["code"] == "QCL-E103" for item in error.value.diagnostics)


def test_sequential_constrained_scan_keeps_bounded_geometry_provenance():
    @qcl.workflow
    def scan(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.Analytic())

        def point(previous, distance, lane):
            guess = previous.set_bond(0, 1, distance)
            optimized = qcl.optimize(pes, guess, fixed=[qcl.Bond(0, 1)]).to(lane)
            return optimized, qcl.scf.hf(optimized).to(lane).energy

        return qcl.scan(point, init=molecule, over=[0.8, 0.9]).to(cpu)

    result = scan.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Cluster.local().cpus[0]).run()
    for geometry, energy in result:
        assert set(geometry.provenance["question"]["starting_molecule"]) == {
            "atoms", "basis", "charge", "spin", "unit"}
        assert energy.value < 0


def test_finite_difference_pes_optimizes_h2_on_cpu():
    @qcl.workflow
    def relax(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.FiniteDifference(step=1e-3))
        return qcl.optimize(pes, molecule).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.95", "sto-3g")
    result = relax.compile(molecule, qcl.Cluster.local().cpus[0]).run(timeout_s=120)
    from chemflow.domains.paths import coordinates

    _, positions = coordinates({"atoms": result.molecule.atoms,
                                "unit": result.molecule.unit})
    assert abs(positions[1, 2] - positions[0, 2]) == pytest.approx(0.735, abs=0.03)
    assert result.provenance["method"]["gradient"] == "finite_difference"


def test_finite_difference_pes_rejects_invalid_step_and_gpu():
    @qcl.workflow
    def relax(molecule, device, step):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.FiniteDifference(step=step))
        return qcl.optimize(pes, molecule).to(device)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.95", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        relax.compile(molecule, cpu, 0)
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        relax.compile(molecule, qcl.Device("gpu0", "gpu"), 1e-3)


def test_finite_difference_pes_respects_fixed_bond():
    @qcl.workflow
    def fixed(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.FiniteDifference(step=1e-3))
        return qcl.optimize(pes, molecule, fixed=[qcl.Bond(0, 1)]).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.9", "sto-3g")
    result = fixed.compile(molecule, qcl.Cluster.local().cpus[0]).run(timeout_s=120)
    from chemflow.domains.paths import coordinates

    _, positions = coordinates({"atoms": result.molecule.atoms,
                                "unit": result.molecule.unit})
    assert abs(positions[1, 2] - positions[0, 2]) == pytest.approx(0.9, abs=1e-3)
    assert result.provenance["method"]["constraints"][0]["type"] == "bond"


def test_finite_difference_pes_relaxes_other_coordinates():
    @qcl.workflow
    def fixed(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.FiniteDifference(step=1e-3))
        return qcl.optimize(pes, molecule, fixed=[qcl.Bond(0, 1)]).to(cpu)

    molecule = qcl.Molecule("O 0 0 0; H 0 0 1.0; H 0.8 0 -0.6", "sto-3g")
    result = fixed.compile(molecule, qcl.Cluster.local().cpus[0]).run(timeout_s=180)
    import numpy as np

    from chemflow.domains.paths import coordinates

    _, positions = coordinates({"atoms": result.molecule.atoms,
                                "unit": result.molecule.unit})
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(1.0, abs=1e-3)
    assert result.provenance["method"]["constraints"][0]["value"] == pytest.approx(1.0)
