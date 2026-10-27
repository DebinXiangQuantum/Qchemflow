import numpy as np
import pytest
from pyscf import gto

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def one_electron(molecule, device, kind):
    return qcl.int1e(molecule, kind).to(device)


@qcl.workflow
def generalized_eigh(molecule, cpu):
    kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
    overlap = qcl.int1e(molecule, "overlap").to(cpu)
    spectrum = qcl.eigh(kinetic, overlap).to(cpu)
    return spectrum.values, spectrum.vectors


@qcl.workflow
def contracted_integrals(molecule, cpu):
    kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
    overlap = qcl.int1e(molecule, "overlap").to(cpu)
    scalar = qcl.contract("ij,ji->", kinetic, overlap).to(cpu)
    matrix = qcl.contract("ij,jk->ik", kinetic, overlap).to(cpu)
    return scalar, matrix


def test_tensor_contraction_matches_numpy_and_records_path():
    scalar, matrix = contracted_integrals.compile(MOLECULE, CPU).run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    kinetic, overlap = mol.intor("int1e_kin"), mol.intor("int1e_ovlp")
    assert scalar.shape == ()
    assert scalar.unit == "hartree"
    assert scalar.data.item() == pytest.approx(np.einsum("ij,ji->", kinetic, overlap))
    np.testing.assert_allclose(matrix.data, kinetic @ overlap, atol=1e-12)
    np.testing.assert_allclose(np.from_dlpack(matrix), kinetic @ overlap, atol=1e-12)
    assert scalar.provenance["method"]["path"]


def test_contract_rejects_wrong_operands_and_implicit_gpu_transfer():
    @qcl.workflow
    def invalid(molecule, cpu):
        matrix = qcl.int1e(molecule, "overlap").to(cpu)
        return qcl.contract("ij,jk->ik", matrix).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        invalid.compile(MOLECULE, CPU)

    @qcl.workflow
    def on_gpu(molecule, cpu, gpu):
        matrix = qcl.int1e(molecule, "overlap").to(cpu)
        return qcl.contract("ij->ji", matrix).to(gpu)

    with pytest.raises(qcl.CompileError, match="QCL-E205"):
        on_gpu.compile(MOLECULE, CPU, qcl.Device("gpu0", "gpu"),
                       strict_transfers=True)


def test_gpu_tensor_contract_and_dlpack_on_real_device():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")

    @qcl.workflow
    def gpu_contract(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        overlap = qcl.int1e(molecule, "overlap").to(cpu).to(gpu)
        return qcl.contract("ij,jk->ik", kinetic, overlap).to(gpu)

    plan = gpu_contract.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                                strict_transfers=True)
    assert [transfer["kind"] for transfer in plan.explain(format="json")["transfers"]][-2:] == [
        "worker_transport", "worker_restore"]
    result = plan.run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                      verbose=0)
    expected = reference.intor("int1e_kin") @ reference.intor("int1e_ovlp")
    assert isinstance(result.data, cupy.ndarray)
    assert result.__dlpack_device__()[0] == 2
    assert result.provenance["implementation"] == "contract/opt-einsum-cupy"
    assert result.provenance["result_transport"] == "gpu-host-gpu"
    cupy.testing.assert_allclose(cupy.from_dlpack(result), cupy.asarray(expected), atol=1e-12)


def test_gpu_generalized_eigh_matches_scipy_on_real_device():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")

    @qcl.workflow
    def gpu_eigh(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        overlap = qcl.int1e(molecule, "overlap").to(cpu).to(gpu)
        return qcl.eigh(kinetic, overlap).to(gpu)

    plan = gpu_eigh.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                            strict_transfers=True)
    result = plan.run()
    from scipy.linalg import eigh

    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    expected, _ = eigh(reference.intor("int1e_kin"), reference.intor("int1e_ovlp"))
    assert result.provenance["implementation"] == "eigh/cupy-cusolver"
    assert isinstance(result.values.data, cupy.ndarray)
    cupy.testing.assert_allclose(result.values.data, cupy.asarray(expected), atol=1e-11)
    overlap = cupy.asarray(reference.intor("int1e_ovlp"))
    cupy.testing.assert_allclose(result.vectors.data.T @ overlap @ result.vectors.data,
                                 cupy.eye(len(expected)), atol=1e-10)

    @qcl.workflow
    def copied_back(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        overlap = qcl.int1e(molecule, "overlap").to(cpu).to(gpu)
        return qcl.eigh(kinetic, overlap).to(gpu).values.to(cpu)

    copied = copied_back.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                                 strict_transfers=True).run()
    assert isinstance(copied.data, np.ndarray)
    np.testing.assert_allclose(copied.data, expected, atol=1e-11)


def test_gpu_tensor_uses_second_visible_device():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) < 2:
        pytest.skip("Two CUDA GPUs are required")

    @qcl.workflow
    def second_gpu(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        return qcl.eigh(kinetic).to(gpu).values

    values = second_gpu.compile(MOLECULE, cluster.cpus[0], cluster.gpus[1],
                                strict_transfers=True).run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    assert values.device == "gpu1"
    assert values.__dlpack_device__() == (2, 1)
    cupy.testing.assert_allclose(values.data,
                                 cupy.asarray(np.linalg.eigvalsh(reference.intor("int1e_kin"))),
                                 atol=1e-11)


def test_explicit_tensor_copy_between_two_gpus():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) < 2:
        pytest.skip("Two CUDA GPUs are required")

    @qcl.workflow
    def two_gpus(molecule, cpu, first, second):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(first)
        eigenvalues = qcl.eigh(kinetic).to(first).values
        moved = eigenvalues.to(second)
        return qcl.contract("i->i", moved).to(second)

    plan = two_gpus.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                            cluster.gpus[1], strict_transfers=True)
    assert any(item["source_device"] == "gpu0" and item["to"] == "gpu1"
               for item in plan.explain(format="json")["transfers"])
    result = plan.run()
    assert result.device == "gpu1"
    assert result.__dlpack_device__() == (2, 1)
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                      verbose=0).intor("int1e_kin")
    cupy.testing.assert_allclose(result.data,
                                 cupy.asarray(np.linalg.eigvalsh(reference)), atol=1e-11)


def test_generalized_eigh_matches_scipy():
    from scipy.linalg import eigh

    plan = generalized_eigh.compile(MOLECULE, CPU)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "int1e", "int1e", "eigh"]
    values, vectors = plan.run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    reference_values, reference_vectors = eigh(mol.intor("int1e_kin"),
                                               mol.intor("int1e_ovlp"))
    np.testing.assert_allclose(values.data, reference_values, atol=1e-12)
    np.testing.assert_allclose(abs(vectors.data), abs(reference_vectors), atol=1e-12)
    np.testing.assert_allclose(np.from_dlpack(values), reference_values, atol=1e-12)


def test_eigh_composite_result_and_metric_units():
    @qcl.workflow
    def spectrum(molecule, cpu):
        matrix = qcl.int1e(molecule, "kinetic").to(cpu)
        return qcl.eigh(matrix).to(cpu)

    result = spectrum.compile(MOLECULE, CPU).run()
    assert isinstance(result, qcl.Eigensystem)
    assert result.values.shape == (2,)
    assert result.vectors.shape == (2, 2)
    assert result.values.unit == "hartree"

    @qcl.workflow
    def wrong_metric(molecule, cpu):
        matrix = qcl.int1e(molecule, "kinetic").to(cpu)
        return qcl.eigh(matrix, matrix).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E102"):
        wrong_metric.compile(MOLECULE, CPU)


@pytest.mark.parametrize("kind,integral", [
    ("overlap", "int1e_ovlp"),
    ("kinetic", "int1e_kin"),
    ("nuclear", "int1e_nuc"),
])
def test_int1e_tensor_matches_pyscf_and_exports_dlpack(kind, integral):
    result = one_electron.compile(MOLECULE, CPU, kind).run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0).intor(integral)
    assert result.device == "cpu0"
    assert result.shape == reference.shape
    assert result.dtype == "float64"
    assert result.provenance["implementation"] == "int1e/chemflow-pyscf"
    np.testing.assert_allclose(result.data, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.from_dlpack(result), reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind,integral,shape,unit", [
    ("dipole", "int1e_r", (3, 2, 2), "bohr"),
    ("quadrupole", "int1e_rr", (3, 3, 2, 2), "bohr^2"),
])
def test_int1e_position_moments_match_pyscf(kind, integral, shape, unit):
    result = one_electron.compile(MOLECULE, CPU, kind).run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                      verbose=0).intor(integral).reshape(shape)
    assert result.shape == shape
    assert result.unit == unit
    np.testing.assert_allclose(result.data, reference, rtol=0, atol=1e-12)


def test_int1e_has_explicit_cpu_copy_and_rejects_unsupported_gpu():
    @qcl.workflow
    def copied(molecule, cpu):
        return qcl.int1e(molecule, "overlap").to(cpu).to(cpu)

    plan = copied.compile(MOLECULE, CPU)
    assert [node["op"] for node in plan.graph_json()["nodes"]][-2:] == [
        "int1e", "copy"]
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                      verbose=0).intor("int1e_ovlp")
    np.testing.assert_allclose(plan.run().data, reference, rtol=0, atol=1e-12)
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        one_electron.compile(MOLECULE, qcl.Device("gpu0", "gpu"), "overlap")


def test_int1e_rejects_unknown_kind_and_non_molecule():
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        one_electron.compile(MOLECULE, CPU, "invalid")
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        one_electron.compile("not a molecule", CPU, "overlap")


def test_orthogonalize_and_solve_match_dense_references():
    @qcl.workflow
    def linear_algebra(molecule, cpu):
        overlap = qcl.int1e(molecule, "overlap").to(cpu)
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        orthogonalizer = qcl.orthogonalize(overlap).to(cpu)
        rhs = qcl.eigh(kinetic).to(cpu).values
        answer = qcl.solve(overlap, rhs).to(cpu)
        return orthogonalizer, answer

    orthogonalizer, answer = linear_algebra.compile(MOLECULE, CPU).run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    overlap, kinetic = reference.intor("int1e_ovlp"), reference.intor("int1e_kin")
    rhs = np.linalg.eigvalsh(kinetic)
    np.testing.assert_allclose(orthogonalizer.data.T @ overlap @ orthogonalizer.data,
                               np.eye(len(rhs)), atol=1e-11)
    np.testing.assert_allclose(answer.data, np.linalg.solve(overlap, rhs), atol=1e-11)
    assert orthogonalizer.unit == "dimensionless"
    assert answer.unit == "hartree"


def test_orthogonalize_rejects_energy_matrix():
    @qcl.workflow
    def invalid(molecule, cpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        return qcl.orthogonalize(kinetic).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E102"):
        invalid.compile(MOLECULE, CPU)


def test_gpu_orthogonalize_and_solve_on_real_device():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")

    @qcl.workflow
    def linear_algebra(molecule, cpu, gpu):
        overlap = qcl.int1e(molecule, "overlap").to(cpu).to(gpu)
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        orthogonalizer = qcl.orthogonalize(overlap).to(gpu)
        rhs = qcl.eigh(kinetic).to(gpu).values
        answer = qcl.solve(overlap, rhs).to(gpu)
        return orthogonalizer, answer

    orthogonalizer, answer = linear_algebra.compile(
        MOLECULE, cluster.cpus[0], cluster.gpus[0], strict_transfers=True).run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    overlap = cupy.asarray(reference.intor("int1e_ovlp"))
    kinetic = cupy.asarray(reference.intor("int1e_kin"))
    cupy.testing.assert_allclose(orthogonalizer.data.T @ overlap @ orthogonalizer.data,
                                 cupy.eye(overlap.shape[0]), atol=1e-10)
    cupy.testing.assert_allclose(overlap @ answer.data, cupy.linalg.eigvalsh(kinetic),
                                 atol=1e-10)


def test_svd_reconstructs_cpu_integral_and_projects_singular_values():
    @qcl.workflow
    def decomposition(molecule, cpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        result = qcl.svd(kinetic).to(cpu)
        return result, result.s

    result, singular = decomposition.compile(MOLECULE, CPU).run()
    kinetic = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                    verbose=0).intor("int1e_kin")
    np.testing.assert_allclose((result.u.data * result.s.data) @ result.vh.data,
                               kinetic, atol=1e-12)
    np.testing.assert_allclose(singular.data, np.linalg.svd(kinetic)[1], atol=1e-12)
    assert result.s.unit == "hartree"
    assert result.u.unit == result.vh.unit == "dimensionless"


def test_svd_rejects_non_tensor_input():
    @qcl.workflow
    def invalid(molecule, cpu):
        return qcl.svd(qcl.scf.hf(molecule).to(cpu)).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        invalid.compile(MOLECULE, CPU)


def test_gpu_svd_result_and_explicit_cpu_copy():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")

    @qcl.workflow
    def decomposition(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        return qcl.svd(kinetic).to(gpu)

    plan = decomposition.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                                 strict_transfers=True)
    result = plan.run()
    assert isinstance(result.u.data, cupy.ndarray)
    assert isinstance(result.s.data, cupy.ndarray)
    assert isinstance(result.vh.data, cupy.ndarray)
    kinetic = cupy.asarray(gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                                 verbose=0).intor("int1e_kin"))
    cupy.testing.assert_allclose((result.u.data * result.s.data) @ result.vh.data,
                                 kinetic, atol=1e-11)

    @qcl.workflow
    def cpu_copy(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        return qcl.svd(kinetic).to(gpu).to(cpu)

    copied = cpu_copy.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                              strict_transfers=True).run()
    assert isinstance(copied.s.data, np.ndarray)
    np.testing.assert_allclose(copied.s.data, cupy.asnumpy(result.s.data), atol=1e-11)


def test_ao_to_mo_transform_matches_pyscf_and_einsum():
    from pyscf import ao2mo

    @qcl.workflow
    def transformed(molecule, cpu):
        overlap = qcl.int1e(molecule, "overlap").to(cpu)
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        coefficients = qcl.eigh(kinetic, overlap).to(cpu).vectors
        matrix = qcl.transform(kinetic, coefficients).to(cpu)
        dipole = qcl.transform(qcl.int1e(molecule, "dipole").to(cpu),
                               coefficients).to(cpu)
        quadrupole = qcl.transform(qcl.int1e(molecule, "quadrupole").to(cpu),
                                   coefficients).to(cpu)
        eri = qcl.transform(qcl.int2e(molecule).to(cpu), coefficients).to(cpu)
        return coefficients, matrix, dipole, quadrupole, eri

    coefficients, matrix, dipole, quadrupole, eri = transformed.compile(
        MOLECULE, CPU, strict_transfers=True).run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    orbitals = coefficients.data
    np.testing.assert_allclose(matrix.data, orbitals.T @ mol.intor("int1e_kin") @ orbitals,
                               atol=1e-12)
    np.testing.assert_allclose(dipole.data, np.einsum(
        "xpq,pi,qj->xij", mol.intor("int1e_r"), orbitals, orbitals), atol=1e-12)
    np.testing.assert_allclose(quadrupole.data, np.einsum(
        "xypq,pi,qj->xyij", mol.intor("int1e_rr").reshape(3, 3, 2, 2),
        orbitals, orbitals), atol=1e-12)
    np.testing.assert_allclose(eri.data, ao2mo.incore.full(
        mol.intor("int2e"), orbitals, compact=False).reshape(2, 2, 2, 2), atol=1e-12)
    assert (matrix.unit, dipole.unit, quadrupole.unit, eri.unit) == (
        "hartree", "bohr", "bohr^2", "hartree")
    assert eri.provenance["implementation"] == "transform/opt-einsum"


def test_transform_rejects_non_tensor_coefficients():
    @qcl.workflow
    def invalid(molecule, cpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        return qcl.transform(kinetic, molecule).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        invalid.compile(MOLECULE, CPU)


def test_transform_rejects_dimensional_coefficients():
    @qcl.workflow
    def invalid(molecule, cpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        return qcl.transform(kinetic, kinetic).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E102"):
        invalid.compile(MOLECULE, CPU)


def test_transform_requires_explicit_gpu_transfers():
    @qcl.workflow
    def invalid(molecule, cpu, gpu):
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
        coefficients = qcl.eigh(kinetic).to(cpu).vectors
        return qcl.transform(kinetic, coefficients).to(gpu)

    with pytest.raises(qcl.CompileError, match="QCL-E205"):
        invalid.compile(MOLECULE, CPU, qcl.Device("gpu0", "gpu"),
                        strict_transfers=True)


def test_gpu_ao_to_mo_transform_and_explicit_eri_copy():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")

    @qcl.workflow
    def transformed(molecule, cpu, gpu):
        overlap = qcl.int1e(molecule, "overlap").to(cpu).to(gpu)
        kinetic = qcl.int1e(molecule, "kinetic").to(cpu).to(gpu)
        coefficients = qcl.eigh(kinetic, overlap).to(gpu).vectors
        eri = qcl.int2e(molecule).to(cpu).to(gpu)
        transformed_eri = qcl.transform(eri, coefficients).to(gpu)
        transformed_kinetic = qcl.transform(kinetic, coefficients).to(gpu)
        return transformed_eri, transformed_kinetic

    plan = transformed.compile(MOLECULE, cluster.cpus[0], cluster.gpus[0],
                               strict_transfers=True)
    assert any(item["kind"] == "worker_restore" for item in
               plan.explain(format="json")["transfers"])
    eri, kinetic = plan.run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    overlap = mol.intor("int1e_ovlp")
    ao_kinetic = mol.intor("int1e_kin")
    from scipy.linalg import eigh

    _, orbitals = eigh(ao_kinetic, overlap)
    from pyscf import ao2mo

    reference_eri = ao2mo.incore.full(mol.intor("int2e"), orbitals,
                                      compact=False).reshape(2, 2, 2, 2)
    assert isinstance(eri.data, cupy.ndarray)
    assert eri.__dlpack_device__()[0] == 2
    cupy.testing.assert_allclose(eri.data, cupy.asarray(reference_eri), atol=1e-10)
    cupy.testing.assert_allclose(kinetic.data,
                                 cupy.asarray(orbitals.T @ ao_kinetic @ orbitals),
                                 atol=1e-10)
    assert eri.provenance["implementation"] == "transform/opt-einsum-cupy"
