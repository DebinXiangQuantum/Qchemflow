import numpy as np
import pytest
from pyscf import fci, gto, scf

import qcl


@qcl.workflow
def fci_rdm1(molecule, active, device, engine="pyscf"):
    mean_field = qcl.scf.hf(molecule).to(device)
    space = qcl.active_space(mean_field, active).to(device)
    state = qcl.fci.solve(space.hamiltonian, space.sector, engine=engine).to(device)
    return qcl.rdm1(state).to(device)


@qcl.workflow
def fci_rdm2(molecule, active, device, engine="pyscf"):
    mean_field = qcl.scf.hf(molecule).to(device)
    space = qcl.active_space(mean_field, active).to(device)
    state = qcl.fci.solve(space.hamiltonian, space.sector, engine=engine).to(device)
    return qcl.rdm2(state).to(device)


def test_fci_rdm1_matches_independent_pyscf():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    plan = fci_rdm1.compile(molecule, qcl.Active(2, 2), cpu)
    result = plan.run()

    mol = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    mf = scf.RHF(mol).run()
    solver = fci.FCI(mf)
    _, vector = solver.kernel()
    expected = solver.make_rdm1(vector, 2, mol.nelec)
    np.testing.assert_allclose(result.data, expected, atol=1e-9)
    assert result.shape == (2, 2)
    assert np.trace(result.data) == pytest.approx(2.0)
    assert result.unit == "dimensionless"
    assert result.device == "cpu0"
    assert result.provenance["implementation"] == "rdm1/chemflow-pyscf"


def test_rdm1_requires_fci_result():
    @qcl.workflow
    def invalid(molecule, device):
        mean_field = qcl.scf.hf(molecule).to(device)
        return qcl.rdm1(mean_field).to(device)

    with pytest.raises(qcl.CompileError, match="rdm1"):
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"))


@pytest.mark.parametrize("atoms,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 2, 2),
    ("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4", 4, 4),
    (("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4; "
      "H 0 0 3.2; H 0 0 4.0"), 6, 6),
    ("; ".join(f"H 0 0 {index * 0.8}" for index in range(8)), 8, 8),
])
def test_native_rdm1_matches_pyscf(atoms, electrons, orbitals):
    molecule = qcl.Molecule(atoms, "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    native = fci_rdm1.compile(molecule, qcl.Active(electrons, orbitals), cpu,
                              engine="native").run()
    reference = fci_rdm1.compile(molecule, qcl.Active(electrons, orbitals), cpu).run()
    np.testing.assert_allclose(native.data, reference.data, atol=1e-9)
    assert native.provenance["implementation"] == "rdm1/qcl-rust"


def test_native_rdm1_survives_cpu_copy_and_shifted_sector():
    @qcl.workflow
    def shifted(molecule, cpu, engine):
        mean_field = qcl.scf.rhf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        sector = space.sector.add_electrons(1).to(cpu)
        state = qcl.fci.solve(space.hamiltonian, sector, engine=engine).to(cpu)
        return qcl.rdm1(state.to(cpu)).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    native_plan = shifted.compile(molecule, cpu, "native")
    assert native_plan.explain(format="json")["operations"][-1]["implementation"] == (
        "rdm1/qcl-rust")
    native = native_plan.run()
    reference = shifted.compile(molecule, cpu, "pyscf").run()
    np.testing.assert_allclose(native.data, reference.data, atol=1e-9)
    assert np.trace(native.data) == pytest.approx(3.0)


def test_rdm1_rejects_gpu_placement():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device("gpu0", "gpu")

    @qcl.workflow
    def invalid(molecule, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        state = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu)
        return qcl.rdm1(state).to(gpu)

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, cpu, gpu)
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


@pytest.mark.parametrize("engine", ["pyscf", "native", "sci"])
def test_gpu_density_requires_gpu_fci_source(engine):
    @qcl.workflow
    def invalid(molecule, cpu, gpu, engine):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        state = (qcl.sci.solve(space.hamiltonian, space.sector).to(cpu)
                 if engine == "sci" else
                 qcl.fci.solve(space.hamiltonian, space.sector,
                               engine=engine).to(cpu))
        return qcl.rdm1(state).to(gpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"),
                        qcl.Device("gpu0", "gpu"), engine)
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


def test_gpu_density_rejects_a_different_gpu_than_its_fci_source():
    @qcl.workflow
    def invalid(molecule, cpu, source_gpu, target_gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        state = qcl.fci.solve(space.hamiltonian.to(source_gpu),
                              space.sector.to(source_gpu)).to(source_gpu)
        return qcl.rdm1(state).to(target_gpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"),
                        qcl.Device("gpu0", "gpu"), qcl.Device("gpu1", "gpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


def test_gpu_fci_rdm_requires_explicit_cpu_copy_under_strict_transfers():
    @qcl.workflow
    def invalid(molecule, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        operator = space.hamiltonian.to(gpu)
        sector = space.sector.to(gpu)
        state = qcl.fci.solve(operator, sector).to(gpu)
        return qcl.rdm1(state).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"),
                        qcl.Device("gpu0", "gpu"), strict_transfers=True)
    assert error.value.diagnostics[0]["code"] == "QCL-E205"


@pytest.mark.parametrize("density", ["rdm1", "rdm2"])
def test_gpu_fci_density_after_explicit_cpu_copy(density):
    pytest.importorskip("cupy")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")

    @qcl.workflow
    def transferred(molecule, cpu, gpu, density):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        state = qcl.fci.solve(space.hamiltonian.to(gpu),
                              space.sector.to(gpu)).to(gpu)
        state = state.to(cpu)
        return getattr(qcl, density)(state).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu, gpu = qcl.Device("cpu0", "cpu"), qcl.Device("gpu0", "gpu")
    result = transferred.compile(molecule, cpu, gpu, density,
                                 strict_transfers=True).run(timeout_s=120)
    reference = (fci_rdm1 if density == "rdm1" else fci_rdm2).compile(
        molecule, qcl.Active(2, 2), cpu).run()
    np.testing.assert_allclose(result.data, reference.data, atol=1e-8)


@pytest.mark.parametrize("density", ["rdm1", "rdm2"])
@pytest.mark.parametrize("gpu_index", [0, 1])
@pytest.mark.parametrize("atoms,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 2, 2),
    ("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4", 4, 4),
])
def test_gpu_fci_density_stays_on_selected_gpu(
        density, gpu_index, atoms, electrons, orbitals):
    pytest.importorskip("cupy")
    if len(qcl.Cluster.local().gpus) <= gpu_index:
        pytest.skip("Requested CUDA GPU unavailable")

    @qcl.workflow
    def resident(molecule, active, cpu, gpu, density):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        state = qcl.fci.solve(space.hamiltonian.to(gpu),
                              space.sector.to(gpu)).to(gpu)
        return getattr(qcl, density)(state).to(gpu)

    molecule = qcl.Molecule(atoms, "sto-3g")
    active = qcl.Active(electrons, orbitals)
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device(f"gpu{gpu_index}", "gpu")
    plan = resident.compile(molecule, active, cpu, gpu, density,
                            strict_transfers=True)
    assert plan.explain(format="json")["operations"][-1]["implementation"] == (
        f"{density}/chemflow-cupy")
    result = plan.run(timeout_s=120)
    reference = (fci_rdm1 if density == "rdm1" else fci_rdm2).compile(
        molecule, active, cpu).run()
    import cupy as cp

    assert result.device == gpu.id
    assert isinstance(result.data, cp.ndarray)
    cp.testing.assert_allclose(result.data, cp.asarray(reference.data), atol=1e-8)


@pytest.mark.parametrize("atoms,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 2, 2),
    ("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4", 4, 4),
])
def test_fci_rdm2_matches_independent_pyscf(atoms, electrons, orbitals):
    molecule = qcl.Molecule(atoms, "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    solver = fci.FCI(mf)
    _, vector = solver.kernel()
    expected = solver.make_rdm12(vector, orbitals, mol.nelec)[1]
    for engine, implementation in (("pyscf", "rdm2/chemflow-pyscf"),
                                   ("native", "rdm2/qcl-rust")):
        plan = fci_rdm2.compile(molecule, qcl.Active(electrons, orbitals), cpu,
                                engine=engine)
        result = plan.run()
        np.testing.assert_allclose(result.data, expected, atol=1e-9)
        assert result.shape == (orbitals,) * 4
        assert result.unit == "dimensionless"
        assert result.provenance["implementation"] == implementation


def test_rdm2_requires_fci_result():
    @qcl.workflow
    def invalid(molecule, cpu):
        return qcl.rdm2(qcl.scf.hf(molecule).to(cpu)).to(cpu)

    with pytest.raises(qcl.CompileError, match="rdm2"):
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"))


@pytest.mark.parametrize("density", ["rdm1", "rdm2"])
@pytest.mark.parametrize("atoms,spin,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 0, 2, 2),
    ("H 0 0 0", 1, 1, 1),
])
def test_sci_density_matches_fci_in_complete_small_space(
        density, atoms, spin, electrons, orbitals):
    @qcl.workflow
    def selected(molecule, active, cpu, density):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        state = qcl.sci.solve(space.hamiltonian, space.sector,
                              select_cutoff=1e-9).to(cpu)
        return getattr(qcl, density)(state).to(cpu)

    molecule = qcl.Molecule(atoms, "sto-3g", spin=spin)
    active = qcl.Active(electrons, orbitals)
    cpu = qcl.Device("cpu0", "cpu")
    actual = selected.compile(molecule, active, cpu, density).run()
    reference = (fci_rdm1 if density == "rdm1" else fci_rdm2).compile(
        molecule, active, cpu).run()
    np.testing.assert_allclose(actual.data, reference.data, atol=1e-9)
    assert actual.provenance["method"]["family"] == "SCI"
