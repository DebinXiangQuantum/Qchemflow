import importlib.util

import pytest

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def gfn2_energy(molecule, device):
    return qcl.engines.tblite.gfn2.energy(molecule).to(device)


@qcl.workflow
def gfn2_optimize(molecule, device):
    geometry = qcl.engines.tblite.gfn2.optimize(molecule).to(device)
    return geometry, qcl.engines.tblite.gfn2.energy(geometry).to(device)


@qcl.workflow
def gfn2_solvation(molecule, device, solvent):
    gas = qcl.engines.tblite.gfn2.energy(molecule).to(device)
    liquid = qcl.engines.tblite.gfn2.energy(molecule, solvent=solvent).to(device)
    return liquid - gas


@qcl.workflow
def gfn2_solvated_optimize(molecule, device, solvent):
    geometry = qcl.engines.tblite.gfn2.optimize(
        molecule, solvent=solvent).to(device)
    return geometry, qcl.engines.tblite.gfn2.energy(
        geometry, solvent=solvent).to(device)


def test_gfn2_solvated_optimization_matches_single_point():
    pytest.importorskip("tblite")
    stretched = qcl.Molecule("H 0 0 0; H 0 0 1.1", "sto-3g")
    solvent = qcl.ALPB("water")
    geometry, energy = gfn2_solvated_optimize.compile(stretched, CPU, solvent).run()
    assert geometry.provenance["method"]["solvent"] == {
        "model": "alpb", "name": "water"}
    starting = (gfn2_solvation.compile(stretched, CPU, solvent).run().value +
                gfn2_energy.compile(stretched, CPU).run().value)
    assert energy.value < starting - 0.01
    from pyscf import gto
    from tblite.interface import Calculator

    molecule = gto.M(atom=geometry.molecule.atoms, basis="sto-3g", verbose=0)
    reference = Calculator("GFN2-xTB", molecule.atom_charges(), molecule.atom_coords())
    reference.add("alpb-solvation", "water")
    assert energy.value == pytest.approx(float(reference.singlepoint().get("energy")), abs=1e-8)


@pytest.mark.parametrize("model", ["alpb", "gbsa"])
def test_gfn2_named_solvent_matches_tblite(model):
    pytest.importorskip("tblite")
    solvent = qcl.ALPB("water") if model == "alpb" else qcl.GBSA("water")
    plan = gfn2_solvation.compile(MOLECULE, CPU, solvent)
    assert plan.explain(format="json")["operations"][1]["implementation"] == (
        "gfn2.energy/chemflow-tblite")
    result = plan.run()
    from pyscf import gto
    from tblite.interface import Calculator

    molecule = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    calc = Calculator("GFN2-xTB", molecule.atom_charges(), molecule.atom_coords())
    calc.add(f"{model}-solvation", "water")
    reference = float(calc.singlepoint().get("energy"))
    gas = gfn2_energy.compile(MOLECULE, CPU).run().value
    assert result.value == pytest.approx(reference - gas, abs=1e-8)
    assert result.provenance["operands"][0]["method"]["solvent"] == {
        "model": model, "name": "water"}


def test_gfn2_solvent_rejects_invalid_descriptor_and_gpu():
    with pytest.raises(ValueError):
        qcl.ALPB("")
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        gfn2_solvation.compile(MOLECULE, CPU, qcl.PCM("water"))
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        gfn2_solvation.compile(MOLECULE, qcl.Device("gpu0", "gpu"), qcl.ALPB("water"))
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        gfn2_solvated_optimize.compile(MOLECULE, CPU, qcl.PCM("water"))


def test_gfn2_optimized_geometry_flows_to_energy():
    pytest.importorskip("tblite")
    stretched = qcl.Molecule("H 0 0 0; H 0 0 1.1", "sto-3g")
    plan = gfn2_optimize.compile(stretched, CPU)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "gfn2.optimize", "gfn2.energy"]
    geometry, energy = plan.run()
    assert geometry.provenance["method"]["family"] == "GFN2-xTB"
    assert geometry.provenance["implementation"] == "gfn2.optimize/chemflow-tblite"
    assert energy.value < gfn2_energy.compile(stretched, CPU).run().value - 0.01
    from pyscf import gto
    from tblite.interface import Calculator

    optimized = gto.M(atom=geometry.molecule.atoms, basis="sto-3g", verbose=0)
    reference = Calculator("GFN2-xTB", optimized.atom_charges(),
                           optimized.atom_coords()).singlepoint().get("energy")
    assert energy.value == pytest.approx(float(reference), abs=1e-8)


def test_gfn2_optimize_rejects_gpu_and_missing_engine(monkeypatch):
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        gfn2_optimize.compile(MOLECULE, qcl.Device("gpu0", "gpu"))
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "tblite" else original(name))
    with pytest.raises(qcl.CompileError, match="QCL-E401"):
        gfn2_optimize.compile(MOLECULE, CPU)


def test_gfn2_cpu_reference_and_method_identity():
    pytest.importorskip("tblite")
    plan = gfn2_energy.compile(MOLECULE, CPU)
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "gfn2.energy/chemflow-tblite")
    result = plan.run()
    assert result.value == pytest.approx(-0.9819836946654522, abs=1e-9)
    assert result.provenance["method"]["family"] == "GFN2-xTB"
    assert "basis" not in result.provenance["method"]


def test_gfn2_rejects_gpu_and_incompatible_hf_comparison():
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        gfn2_energy.compile(MOLECULE, qcl.Device("gpu0", "gpu"))

    @qcl.workflow
    def mixed(molecule, cpu):
        baseline = qcl.engines.tblite.gfn2.energy(molecule).to(cpu)
        reference = qcl.scf.hf(molecule).to(cpu).energy
        return qcl.compare(baseline, reference)

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        mixed.compile(MOLECULE, CPU)


def test_gfn2_reports_missing_engine_at_compile_time(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "tblite" else original(name))
    with pytest.raises(qcl.CompileError, match="QCL-E401"):
        gfn2_energy.compile(MOLECULE, CPU)
