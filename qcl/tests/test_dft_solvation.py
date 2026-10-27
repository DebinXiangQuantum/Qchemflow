import numpy as np
import pytest
from pyscf import dft, gto

import qcl

H2 = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
H = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)


@qcl.workflow
def solvated_dft(molecule, device, solvent):
    if molecule.spin:
        return qcl.dft.uks(molecule, xc="pbe", solvent=solvent).to(device).energy
    return qcl.dft.rks(molecule, xc="pbe", solvent=solvent).to(device).energy


@qcl.workflow
def solvated_rks_derivatives(molecule, device, solvent):
    surface = qcl.PES(
        lambda geometry, lane: qcl.dft.rks(
            geometry, xc="pbe", solvent=solvent).to(lane),
        gradient=qcl.Analytic())
    return (qcl.gradient(surface, molecule).to(device),
            qcl.hessian(surface, molecule).to(device))


def test_rks_pcm_on_cpu_matches_pyscf():
    solvent = qcl.PCM("water")
    plan = solvated_dft.compile(H2, qcl.Device("cpu0", "cpu"), solvent)
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "dft.rks/chemflow-pyscf-pcm")
    result = plan.run()
    system = gto.M(atom=H2.atoms, basis=H2.basis, verbose=0)
    reference = dft.RKS(system)
    reference.xc = "pbe"
    reference = reference.PCM()
    reference.with_solvent.method = "IEF-PCM"
    reference.with_solvent.eps = solvent.eps
    reference.kernel()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-8)
    assert result.provenance["method"]["pcm_eps"] == solvent.eps


def test_rks_pcm_derivatives_on_cpu_match_pyscf():
    solvent = qcl.PCM("water")
    system = gto.M(atom=H2.atoms, basis=H2.basis, verbose=0)
    reference = dft.RKS(system)
    reference.xc = "pbe"
    reference = reference.PCM()
    reference.with_solvent.method = "IEF-PCM"
    reference.with_solvent.eps = solvent.eps
    reference.conv_tol = 1e-10
    reference.kernel()
    plan = solvated_rks_derivatives.compile(H2, qcl.Device("cpu0", "cpu"), solvent)
    assert [item["implementation"] for item in
            plan.explain(format="json")["operations"]] == [
                "gradient/dft-pcm-pyscf", "hessian/dft-pcm-pyscf"]
    gradient, hessian = plan.run()
    np.testing.assert_allclose(gradient.data, reference.nuc_grad_method().kernel(), atol=1e-9)
    np.testing.assert_allclose(hessian.data, reference.Hessian().kernel(), atol=1e-9)
    assert hessian.provenance["method"]["xc"] == "pbe"
    assert gradient.provenance["method"]["solvent"] == "water"


@pytest.mark.parametrize("ordinal", [0, 1])
def test_gpu_rks_pcm_derivatives_match_cpu(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    gpu_gradient, gpu_hessian = solvated_rks_derivatives.compile(
        H2, cluster.gpus[ordinal], qcl.PCM("water")).run(timeout_s=120)
    cpu_gradient, cpu_hessian = solvated_rks_derivatives.compile(
        H2, cluster.cpus[0], qcl.PCM("water")).run(timeout_s=120)
    cupy.testing.assert_allclose(gpu_gradient.data, cupy.asarray(cpu_gradient.data), atol=1e-7)
    cupy.testing.assert_allclose(gpu_hessian.data, cupy.asarray(cpu_hessian.data), atol=1e-7)
    assert gpu_gradient.__dlpack_device__() == (2, ordinal)
    assert gpu_hessian.provenance["implementation"] == "hessian/dft-pcm-gpu4pyscf"


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("molecule,solvent", [(H2, qcl.PCM("water")),
                                              (H, qcl.SMD("water"))])
def test_gpu_solvated_dft_matches_cpu(molecule, solvent, ordinal):
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    gpu = solvated_dft.compile(molecule, cluster.gpus[ordinal], solvent).run(timeout_s=120)
    cpu = solvated_dft.compile(molecule, cluster.cpus[0], solvent).run(timeout_s=120)
    assert gpu.value == pytest.approx(cpu.value, abs=1e-8)
    assert gpu.provenance["device"] == f"gpu{ordinal}"
    assert gpu.provenance["implementation"].endswith(
        "-smd" if isinstance(solvent, qcl.SMD) else "-pcm")
