import pytest
from pyscf import fci, gto, scf

import qcl


@qcl.workflow
def ground(molecule, active, cpu, engine):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector,
                         engine=engine).to(cpu).energy


@pytest.mark.parametrize("atoms,spin,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 0, 2, 2),
    ("Li 0 0 0", 1, 3, 5),
    ("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4", 0, 4, 4),
    (("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4; "
      "H 0 0 3.2; H 0 0 4.0"), 0, 6, 6),
])
def test_native_ci_matches_pyscf(atoms, spin, electrons, orbitals):
    molecule = qcl.Molecule(atoms, "sto-3g", spin=spin)
    active = qcl.Active(electrons, orbitals)
    cpu = qcl.Device("cpu0", "cpu")
    plan = ground.compile(molecule, active, cpu, "native")
    assert plan.explain(format="json")["operations"][-1]["implementation"] == (
        "fci.solve/qcl-rust")
    native = plan.run()
    wrapped = ground.compile(molecule, active, cpu, "pyscf").run()
    system = gto.M(atom=atoms, basis="sto-3g", spin=spin, verbose=0)
    mean_field = (scf.ROHF(system) if spin else scf.RHF(system)).run()
    reference = fci.FCI(mean_field).kernel()[0]
    assert native.value == pytest.approx(reference, abs=1e-10)
    assert native.value == pytest.approx(wrapped.value, abs=1e-10)
    assert native.provenance["implementation"] == "fci.solve/qcl-rust"
    assert native.provenance["versions"]["qcl-ci"] == "0.1.0"


def test_native_ci_rejects_unsupported_engine_and_device():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    active = qcl.Active(2, 2)
    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        ground.compile(molecule, active, qcl.Device("cpu0", "cpu"), "unknown")
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        ground.compile(molecule, active, qcl.Device("gpu0", "gpu"), "native")


def test_default_fci_keeps_legacy_ir_parameters():
    plan = ground.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Active(2, 2), qcl.Device("cpu0", "cpu"), "pyscf")
    node = next(item for item in plan.graph_json()["nodes"]
                if item["op"] == "fci.solve")
    assert node["params"] == {}


def test_native_ci_rejects_large_sector_at_compile_time():
    atoms = "; ".join(f"H 0 0 {index * 0.8}" for index in range(10))
    with pytest.raises(qcl.CompileError, match="QCL-E209"):
        ground.compile(qcl.Molecule(atoms, "sto-3g"), qcl.Active(10, 10),
                       qcl.Device("cpu0", "cpu"), "native")


def test_native_ci_h8_sparse_lanczos_matches_pyscf():
    atoms = "; ".join(f"H 0 0 {index * 0.8}" for index in range(8))
    molecule = qcl.Molecule(atoms, "sto-3g")
    plan = ground.compile(molecule, qcl.Active(8, 8),
                          qcl.Device("cpu0", "cpu"), "native")
    native = plan.run(timeout_s=120)
    system = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    reference = fci.FCI(scf.RHF(system).run()).kernel()[0]
    assert native.value == pytest.approx(reference, abs=1e-10)


def test_native_ci_shifted_sectors_match_pyscf():
    from pyscf import ao2mo, mcscf

    @qcl.workflow
    def charged(molecule, cpu):
        mean_field = qcl.scf.rhf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        return {str(delta): qcl.fci.solve(
            space.hamiltonian, space.sector.add_electrons(delta),
            engine="native").to(cpu).energy for delta in (-1, 0, 1)}

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    actual = charged.compile(molecule, qcl.Device("cpu0", "cpu")).run()
    system = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    mean_field = scf.RHF(system).run()
    cas = mcscf.CASCI(mean_field, 2, 2)
    h1, core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), 2)
    for delta, sector in ((-1, (1, 0)), (0, (1, 1)), (1, (2, 1))):
        reference = fci.direct_spin1.FCI().kernel(h1, eri, 2, sector)[0] + core
        assert actual[str(delta)].value == pytest.approx(reference, abs=1e-10)


def test_native_ci_tracks_shifted_sector_through_explicit_copy():
    from pyscf import ao2mo, mcscf

    @qcl.workflow
    def copied(molecule, cpu):
        mean_field = qcl.scf.rhf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        sector = space.sector.add_electrons(1).to(cpu)
        return qcl.fci.solve(space.hamiltonian, sector,
                             engine="native").to(cpu).energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = copied.compile(molecule, qcl.Device("cpu0", "cpu")).run()
    mean_field = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                              verbose=0)).run()
    cas = mcscf.CASCI(mean_field, 2, 2)
    h1, core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), 2)
    reference = fci.direct_spin1.FCI().kernel(h1, eri, 2, (2, 1))[0] + core
    assert result.value == pytest.approx(reference, abs=1e-10)
