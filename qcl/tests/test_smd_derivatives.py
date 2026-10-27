import numpy as np
import pytest
from pyscf import gto, scf

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
SOLVENT = qcl.SMD("water")


@qcl.workflow
def smd_derivatives(molecule, device, finite=False):
    surface = qcl.PES(
        lambda geometry, lane: qcl.scf.rhf(geometry, solvent=SOLVENT).to(lane),
        gradient=qcl.FiniteDifference(step=1e-3) if finite else qcl.Analytic())
    gradient = qcl.gradient(surface, molecule).to(device)
    if finite:
        return gradient
    return gradient, qcl.hessian(surface, molecule).to(device)


def reference_derivatives():
    system = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    reference = scf.RHF(system).SMD()
    reference.with_solvent.solvent = "water"
    reference.conv_tol = 1e-10
    try:
        reference.kernel()
    except RuntimeError as error:
        if "SMD module is not available" in str(error):
            pytest.skip("PySCF was built without SMD")
        raise
    return reference.nuc_grad_method().kernel(), reference.Hessian().kernel()


def test_smd_derivatives_compile_with_model_identity():
    plan = smd_derivatives.compile(MOLECULE, qcl.Device("cpu0", "cpu"))
    assert [item["implementation"] for item in
            plan.explain(format="json")["operations"]] == [
                "gradient/smd-pyscf", "hessian/smd-pyscf"]


def test_smd_cpu_derivatives_match_pyscf():
    reference_gradient, reference_hessian = reference_derivatives()
    gradient, hessian = smd_derivatives.compile(
        MOLECULE, qcl.Device("cpu0", "cpu")).run()
    np.testing.assert_allclose(gradient.data, reference_gradient, atol=1e-9)
    np.testing.assert_allclose(hessian.data, reference_hessian, atol=1e-9)
    assert gradient.provenance["method"]["solvent_model"] == "SMD"
    assert hessian.provenance["method"]["solvent"] == "water"
    finite = smd_derivatives.compile(MOLECULE, qcl.Device("cpu0", "cpu"), True).run()
    np.testing.assert_allclose(finite.data, reference_gradient, atol=1e-5)
    assert finite.provenance["implementation"] == "gradient/fd-smd-pyscf"


@pytest.mark.parametrize("ordinal", [0, 1])
def test_smd_gpu_derivatives_match_cpu(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    reference_gradient, reference_hessian = reference_derivatives()
    gradient, hessian = smd_derivatives.compile(
        MOLECULE, cluster.gpus[ordinal]).run(timeout_s=120)
    cupy.testing.assert_allclose(gradient.data, cupy.asarray(reference_gradient), atol=1e-8)
    cupy.testing.assert_allclose(hessian.data, cupy.asarray(reference_hessian), atol=1e-8)
    assert gradient.__dlpack_device__() == (2, ordinal)
    assert hessian.provenance["implementation"] == "hessian/smd-gpu4pyscf"
