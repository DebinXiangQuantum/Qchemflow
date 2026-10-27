import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf

import qcl


@qcl.workflow
def ci_sigma(molecule, active, device, engine="pyscf"):
    mean_field = qcl.scf.hf(molecule).to(device)
    space = qcl.active_space(mean_field, active).to(device)
    state = qcl.fci.solve(space.hamiltonian, space.sector, engine=engine).to(device)
    coefficients = state.coefficients
    return coefficients, qcl.sigma(space.hamiltonian, coefficients,
                                   engine=engine).to(device), state.energy


@pytest.mark.parametrize("atoms,electrons,orbitals", [
    ("H 0 0 0; H 0 0 0.74", 2, 2),
    ("H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4", 4, 4),
])
@pytest.mark.parametrize("engine", ["pyscf", "native"])
def test_ci_sigma_matches_independent_pyscf_and_eigenvalue(
        atoms, electrons, orbitals, engine):
    molecule = qcl.Molecule(atoms, "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    plan = ci_sigma.compile(molecule, qcl.Active(electrons, orbitals), cpu, engine)
    sigma_operation = next(item for item in plan.explain(format="json")["operations"]
                           if item["op"] == "sigma")
    assert sigma_operation["implementation"] == (
        "sigma/qcl-rust" if engine == "native" else "sigma/chemflow-numpy")
    coefficients, product, energy = plan.run()

    system = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mean_field = scf.RHF(system).run()
    h1 = mean_field.mo_coeff.T @ mean_field.get_hcore() @ mean_field.mo_coeff
    eri = ao2mo.kernel(system, mean_field.mo_coeff)
    absorbed = fci.direct_spin1.absorb_h1e(h1, eri, orbitals,
                                           (electrons // 2, electrons // 2), 0.5)
    expected = fci.direct_spin1.contract_2e(
        absorbed, coefficients.data, orbitals,
        (electrons // 2, electrons // 2)) + system.energy_nuc() * coefficients.data
    np.testing.assert_allclose(product.data, expected, atol=1e-8)
    np.testing.assert_allclose(product.data, energy.value * coefficients.data, atol=1e-8)
    assert coefficients.shape == product.shape
    assert coefficients.unit == "dimensionless"
    assert product.unit == "hartree"
    assert product.provenance["implementation"] == (
        "sigma/qcl-rust" if engine == "native" else "sigma/chemflow-numpy")


def test_sigma_rejects_a_non_ci_tensor():
    @qcl.workflow
    def invalid(molecule, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        return qcl.sigma(space.hamiltonian, qcl.rdm1(
            qcl.fci.solve(space.hamiltonian, space.sector).to(cpu)).to(cpu)).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError, match="CI coefficient"):
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"))


def test_sigma_rejects_coefficients_from_another_active_space():
    @qcl.workflow
    def invalid(molecule, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        other = qcl.active_space(mean_field, qcl.Active(2, 1)).to(cpu)
        coefficients = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).coefficients
        return qcl.sigma(other.hamiltonian, coefficients).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


@pytest.mark.parametrize("engine", ["pyscf", "native"])
def test_sigma_supports_shifted_electron_sector(engine):
    @qcl.workflow
    def shifted(molecule, cpu, engine):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        sector = space.sector.add_electrons(1).to(cpu)
        state = qcl.fci.solve(space.hamiltonian, sector, engine=engine).to(cpu)
        return state.coefficients, qcl.sigma(space.hamiltonian, state.coefficients).to(cpu), state.energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    coefficients, product, energy = shifted.compile(
        molecule, qcl.Device("cpu0", "cpu"), engine).run()
    assert coefficients.shape == (1, 2)
    np.testing.assert_allclose(product.data, energy.value * coefficients.data, atol=1e-8)


def test_user_ci_vector_sigma_matches_pyscf_for_random_coefficients():
    @qcl.workflow
    def random_action(molecule, coefficients, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(4, 4)).to(cpu)
        vector = qcl.ci_vector(coefficients, space.sector).to(cpu)
        return vector, qcl.sigma(space.hamiltonian, vector,
                                 engine="native").to(cpu)

    atoms = "H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4"
    coefficients = np.random.default_rng(21).normal(size=(6, 6))
    molecule = qcl.Molecule(atoms, "sto-3g")
    vector, product = random_action.compile(
        molecule, coefficients, qcl.Device("cpu0", "cpu"),
        strict_transfers=True).run()

    system = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mean_field = scf.RHF(system).run()
    h1 = mean_field.mo_coeff.T @ mean_field.get_hcore() @ mean_field.mo_coeff
    eri = ao2mo.kernel(system, mean_field.mo_coeff)
    absorbed = fci.direct_spin1.absorb_h1e(h1, eri, 4, (2, 2), 0.5)
    expected = (fci.direct_spin1.contract_2e(absorbed, coefficients, 4, (2, 2)) +
                system.energy_nuc() * coefficients)
    np.testing.assert_allclose(vector.data, coefficients, atol=1e-15)
    np.testing.assert_allclose(product.data, expected, atol=1e-10)
    assert product.provenance["implementation"] == "sigma/qcl-rust"


def test_native_sigma_rejects_sector_beyond_rust_limit_at_compile_time():
    @qcl.workflow
    def large_action(molecule, cpu, engine):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(8, 9)).to(cpu)
        vector = qcl.ci_vector(np.zeros((126, 126)), space.sector).to(cpu)
        return qcl.sigma(space.hamiltonian, vector, engine=engine).to(cpu)

    atoms = "; ".join(f"H 0 0 {index}" for index in range(10))
    molecule = qcl.Molecule(atoms, "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    large_action.compile(molecule, cpu, "pyscf")
    with pytest.raises(qcl.CompileError, match="QCL-E209"):
        large_action.compile(molecule, cpu, "native")


def test_user_ci_vector_rejects_wrong_sector_shape():
    @qcl.workflow
    def invalid(molecule, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        return qcl.ci_vector([[1, 2, 3]], space.sector).to(cpu)

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E101"


def test_user_ci_vector_sigma_requires_placement():
    @qcl.workflow
    def invalid(molecule, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        vector = qcl.ci_vector([[1.0, 0.0], [0.0, 0.0]], space.sector).to(cpu)
        return qcl.sigma(space.hamiltonian, vector)

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E201"


def test_sigma_rejects_qpu_placement():
    @qcl.workflow
    def invalid(molecule, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        vector = qcl.ci_vector([[1.0, 0.0], [0.0, 0.0]], space.sector).to(cpu)
        return qcl.sigma(space.hamiltonian, vector).to(qpu)

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"),
                        qcl.Device("sim0", "qpu", simulator_host="cpu0"))
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


def test_native_sigma_rejects_gpu_placement():
    @qcl.workflow
    def invalid(molecule, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        vector = qcl.ci_vector([[1.0, 0.0], [0.0, 0.0]], space.sector).to(gpu)
        return qcl.sigma(space.hamiltonian.to(gpu), vector,
                         engine="native").to(gpu)

    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Device("cpu0", "cpu"), qcl.Device("gpu0", "gpu"))


def test_user_ci_vector_sigma_runs_on_gpu():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA is unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CUDA is unavailable")

    @qcl.workflow
    def random_action(molecule, vector, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        coefficients = qcl.ci_vector(vector, space.sector).to(gpu)
        operator = space.hamiltonian.to(gpu)
        return qcl.sigma(operator, coefficients).to(gpu)

    coefficients = np.random.default_rng(6).normal(size=(2, 2))
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    product = random_action.compile(molecule, coefficients,
                                    qcl.Device("cpu0", "cpu"),
                                    qcl.Device("gpu0", "gpu"),
                                    strict_transfers=True).run()
    assert product.device == "gpu0"
    system = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    mean_field = scf.RHF(system).run()
    h1 = mean_field.mo_coeff.T @ mean_field.get_hcore() @ mean_field.mo_coeff
    eri = ao2mo.kernel(system, mean_field.mo_coeff)
    absorbed = fci.direct_spin1.absorb_h1e(h1, eri, 2, (1, 1), 0.5)
    expected = (fci.direct_spin1.contract_2e(absorbed, coefficients, 2, (1, 1)) +
                system.energy_nuc() * coefficients)
    np.testing.assert_allclose(cp.asnumpy(product.data), expected, atol=1e-10)


@pytest.mark.parametrize("gpu_id", ["gpu0", "gpu1"])
def test_gpu_sigma_matches_cpu(gpu_id):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= int(gpu_id[3:]):
            pytest.skip("Selected CUDA device is unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CUDA is unavailable")

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device(gpu_id, "gpu")

    @qcl.workflow
    def on_gpu(molecule, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        operator = space.hamiltonian.to(gpu)
        sector = space.sector.to(gpu)
        state = qcl.fci.solve(operator, sector).to(gpu)
        return state.coefficients, qcl.sigma(operator, state.coefficients).to(gpu), state.energy

    coefficients, product, energy = on_gpu.compile(molecule, cpu, gpu).run()
    assert coefficients.device == product.device == gpu_id
    assert product.__dlpack_device__() == (2, int(gpu_id[3:]))
    np.testing.assert_allclose(cp.asnumpy(product.data),
                               energy.value * cp.asnumpy(coefficients.data), atol=1e-8)
    assert product.provenance["implementation"] == "sigma/chemflow-cupy"
    assert energy.provenance["method"]["solver"].startswith("cupy.")


def test_gpu_fci_coefficients_can_be_copied_to_cpu():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA is unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CUDA is unavailable")

    @qcl.workflow
    def copied(molecule, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        state = qcl.fci.solve(space.hamiltonian.to(gpu),
                              space.sector.to(gpu)).to(gpu)
        coefficients = state.to(cpu).coefficients
        return coefficients, qcl.sigma(space.hamiltonian, coefficients).to(cpu), state.energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = copied.compile(molecule, qcl.Device("cpu0", "cpu"),
                          qcl.Device("gpu0", "gpu"), strict_transfers=True)
    coefficients, product, energy = plan.run()
    assert coefficients.device == product.device == "cpu0"
    assert coefficients.__dlpack_device__() == (1, 0)
    np.testing.assert_allclose(product.data, energy.value * coefficients.data, atol=1e-8)
