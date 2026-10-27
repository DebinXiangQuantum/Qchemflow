import numpy as np
import pytest
from pyscf import gto, scf

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
SOLVENT = qcl.PCM("water")


@qcl.workflow
def pcm_derivatives(molecule, device, solvent):
    surface = qcl.PES(lambda geometry, lane:
                      qcl.scf.rhf(geometry, solvent=solvent).to(lane),
                      gradient=qcl.Analytic())
    return (qcl.gradient(surface, molecule).to(device),
            qcl.hessian(surface, molecule).to(device))


@qcl.workflow
def pcm_finite_gradient_and_modes(molecule, device, solvent):
    callback = lambda geometry, lane: qcl.scf.rhf(geometry, solvent=solvent).to(lane)
    finite_surface = qcl.PES(callback, gradient=qcl.FiniteDifference(step=1e-3))
    analytic_surface = qcl.PES(callback, gradient=qcl.Analytic())
    gradient = qcl.gradient(finite_surface, molecule).to(device)
    modes = qcl.frequencies(qcl.hessian(analytic_surface, molecule).to(device)).to(device)
    return gradient, modes


def reference_derivatives():
    system = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    mean_field = scf.RHF(system).PCM()
    mean_field.with_solvent.method = "IEF-PCM"
    mean_field.with_solvent.eps = SOLVENT.eps
    mean_field.conv_tol = 1e-10
    mean_field.kernel()
    return mean_field.nuc_grad_method().kernel(), mean_field.Hessian().kernel()


def test_cpu_pcm_derivatives_match_pyscf():
    plan = pcm_derivatives.compile(MOLECULE, qcl.Device("cpu0", "cpu"), SOLVENT)
    implementations = [item["implementation"]
                       for item in plan.explain(format="json")["operations"]]
    assert implementations == ["gradient/pcm-pyscf", "hessian/pcm-pyscf"]
    gradient, hessian = plan.run()
    reference_gradient, reference_hessian = reference_derivatives()
    np.testing.assert_allclose(gradient.data, reference_gradient, atol=1e-9)
    np.testing.assert_allclose(hessian.data, reference_hessian, atol=1e-9)
    assert gradient.provenance["method"]["solvent"] == "water"
    assert hessian.provenance["method"]["pcm_eps"] == SOLVENT.eps


def test_pcm_finite_gradient_and_frequency_keep_solvent_provenance():
    finite, modes = pcm_finite_gradient_and_modes.compile(
        MOLECULE, qcl.Device("cpu0", "cpu"), SOLVENT).run()
    reference_gradient, _ = reference_derivatives()
    np.testing.assert_allclose(finite.data, reference_gradient, atol=1e-5)
    assert finite.provenance["implementation"] == "gradient/fd-pcm-pyscf"
    assert finite.provenance["method"]["solvent"] == "water"
    assert modes.provenance["method"]["solvent"] == "water"


@pytest.mark.parametrize("ordinal", [0, 1])
def test_gpu_pcm_derivatives_match_cpu(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("The requested CUDA GPU is unavailable")
    gradient, hessian = pcm_derivatives.compile(
        MOLECULE, cluster.gpus[ordinal], SOLVENT).run(timeout_s=120)
    reference_gradient, reference_hessian = reference_derivatives()
    assert gradient.__dlpack_device__() == (2, ordinal)
    assert hessian.__dlpack_device__() == (2, ordinal)
    assert gradient.provenance["implementation"] == "gradient/pcm-gpu4pyscf"
    assert hessian.provenance["implementation"] == "hessian/pcm-gpu4pyscf"
    cupy.testing.assert_allclose(gradient.data, cupy.asarray(reference_gradient),
                                 atol=1e-8)
    cupy.testing.assert_allclose(hessian.data, cupy.asarray(reference_hessian),
                                 atol=1e-8)
