import pytest

import qcl


@qcl.workflow
def pbe_energy(molecule, cpu):
    return qcl.dft.rks(molecule, xc="pbe").to(cpu).energy


def test_rks_pbe_matches_pyscf_reference():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = pbe_energy.compile(molecule, qcl.Device("cpu0", "cpu"))
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "dft.rks/chemflow-pyscf")
    result = plan.run()
    assert result.value == pytest.approx(-1.1520727951590923, abs=1e-8)
    assert result.provenance["method"]["functional"] == "pbe"

    @qcl.workflow
    def orbitals(mol, cpu):
        mean_field = qcl.dft.rks(mol, xc="pbe").to(cpu)
        return mean_field.homo, mean_field.lumo

    homo, lumo = orbitals.compile(molecule, qcl.Device("cpu0", "cpu")).run()
    assert homo.value < lumo.value
    assert homo.provenance["method"]["functional"] == "pbe"


def test_rks_rejects_spin_and_unknown_functional():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        pbe_energy.compile(qcl.Molecule("H 0 0 0", "sto-3g", spin=1),
                           qcl.Device("cpu0", "cpu"))
    assert pbe_energy.compile(molecule, qcl.Device("gpu0", "gpu")).explain(
        format="json")["operations"][0]["implementation"] == "dft.rks/gpu4pyscf"

    @qcl.workflow
    def invalid(mol, cpu):
        return qcl.dft.rks(mol, xc="not-a-functional").to(cpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"))


def test_rks_orbitals_are_not_silently_treated_as_hf_active_space():
    @qcl.workflow
    def incorrect(molecule, cpu):
        mean_field = qcl.dft.rks(molecule, xc="pbe").to(cpu)
        return qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        incorrect.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Device("cpu0", "cpu"))
