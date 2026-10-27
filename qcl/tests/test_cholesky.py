"""Pivoted Cholesky factors of molecular four-center ERIs."""

import numpy as np
import pytest
from pyscf import gto

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def factors(molecule, tolerance, device):
    return qcl.cholesky(molecule, tolerance).to(device)


def test_cholesky_reconstructs_eri_and_records_residual():
    result = factors.compile(MOLECULE, 1e-10, CPU).run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    eri = mol.intor("int2e")
    reconstructed = np.einsum("Pij,Pkl->ijkl", result.data, result.data)
    assert result.shape[1:] == (mol.nao_nr(), mol.nao_nr())
    assert result.provenance["implementation"] == "cholesky/pyscf-pivoted"
    assert result.provenance["method"]["tolerance"] == 1e-10
    assert result.provenance["method"]["max_abs_reconstruction_error"] == pytest.approx(
        float(np.max(np.abs(reconstructed - eri))), abs=1e-14)
    np.testing.assert_allclose(reconstructed, eri, rtol=0, atol=1e-10)


def test_cholesky_rank_responds_to_tolerance():
    tight = factors.compile(MOLECULE, 1e-10, CPU).run()
    loose = factors.compile(MOLECULE, 1e-2, CPU).run()
    assert loose.shape[0] < tight.shape[0]
    assert loose.provenance["method"]["max_abs_reconstruction_error"] <= 1e-2


def test_cholesky_reconstruction_has_energy_unit():
    @qcl.workflow
    def reconstruct(molecule, cpu):
        vectors = qcl.cholesky(molecule, 1e-10).to(cpu)
        return qcl.contract("Pij,Pkl->ijkl", vectors, vectors).to(cpu)

    result = reconstruct.compile(MOLECULE, CPU).run()
    assert result.unit == "hartree"


def test_cholesky_rejects_bad_tolerance_and_gpu_placement():
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(qcl.CompileError, match="QCL-E104"):
            factors.compile(MOLECULE, value, CPU)
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        factors.compile(MOLECULE, 1e-10, qcl.Device("gpu0", "gpu"))


def test_cholesky_requires_explicit_placement():
    @qcl.workflow
    def unplaced(molecule):
        return qcl.cholesky(molecule, 1e-8)

    with pytest.raises(qcl.CompileError, match="QCL-E201"):
        unplaced.compile(MOLECULE)


@pytest.mark.parametrize("ordinal", [0, 1])
def test_cholesky_factors_support_gpu_contraction(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip(f"GPU {ordinal} is unavailable")

    @qcl.workflow
    def reconstructed_eri(molecule, cpu, gpu):
        vectors = qcl.cholesky(molecule, 1e-10).to(cpu).to(gpu)
        return qcl.contract("Pij,Pkl->ijkl", vectors, vectors).to(gpu)

    plan = reconstructed_eri.compile(MOLECULE, cluster.cpus[0],
                                     cluster.gpus[ordinal], strict_transfers=True)
    result = plan.run()
    reference = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis,
                      verbose=0).intor("int2e")
    assert result.device == f"gpu{ordinal}"
    assert result.unit == "hartree"
    assert isinstance(result.data, cupy.ndarray)
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference), atol=1e-10)
