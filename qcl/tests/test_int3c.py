"""Three-center density-fitting integrals through the placed QCL tensor API."""

import numpy as np
import pytest
from pyscf import df, gto

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def three_center(molecule, auxbasis, device):
    return qcl.int3c(molecule, auxbasis).to(device)


def test_int3c_matches_pyscf_and_records_basis():
    plan = three_center.compile(MOLECULE, "weigend", CPU)
    assert plan.explain(format="json")["operations"][-1]["implementation"] == (
        "int3c/chemflow-pyscf")
    result = plan.run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    auxmol = df.make_auxmol(mol, "weigend")
    reference = df.incore.aux_e2(mol, auxmol, aosym="s1")
    assert result.shape == (mol.nao_nr(), mol.nao_nr(), auxmol.nao_nr())
    assert result.device == "cpu0"
    assert result.provenance["method"]["auxbasis"] == "weigend"
    np.testing.assert_allclose(result.data, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.data, result.data.swapaxes(0, 1), atol=1e-12)


def test_int3c_requires_placement_and_rejects_unsupported_gpu():
    @qcl.workflow
    def unplaced(molecule):
        return qcl.int3c(molecule, "weigend")

    with pytest.raises(qcl.CompileError, match="QCL-E201"):
        unplaced.compile(MOLECULE)
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        three_center.compile(MOLECULE, "weigend", qcl.Device("gpu0", "gpu"))


def test_int3c_rejects_invalid_inputs():
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        three_center.compile("not a molecule", "weigend", CPU)
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        three_center.compile(MOLECULE, "", CPU)


@pytest.mark.parametrize("ordinal", [0, 1])
def test_int3c_tensor_copies_to_gpu_for_contraction(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip(f"GPU {ordinal} is unavailable")

    @qcl.workflow
    def gram(molecule, cpu, gpu):
        integrals = qcl.int3c(molecule, "weigend").to(cpu).to(gpu)
        return qcl.contract("ijp,klp->ijkl", integrals, integrals).to(gpu)

    plan = gram.compile(MOLECULE, cluster.cpus[0], cluster.gpus[ordinal],
                        strict_transfers=True)
    result = plan.run()
    mol = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    auxiliary = df.make_auxmol(mol, "weigend")
    integrals = df.incore.aux_e2(mol, auxiliary, aosym="s1")
    reference = np.einsum("ijp,klp->ijkl", integrals, integrals)
    assert result.device == f"gpu{ordinal}"
    assert isinstance(result.data, cupy.ndarray)
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference), rtol=1e-12)
