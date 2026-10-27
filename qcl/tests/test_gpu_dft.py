import pytest

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
GPU = qcl.Device("gpu0", "gpu")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def gpu_pbe(molecule, gpu, cpu=None):
    mean_field = qcl.dft.rks(molecule, xc="pbe").to(gpu)
    if cpu is not None:
        mean_field = mean_field.to(cpu)
    return mean_field.energy, mean_field.homo, mean_field.lumo


def test_gpu_rks_has_registered_implementation_and_explicit_copy():
    plan = gpu_pbe.compile(MOLECULE, GPU, CPU, strict_transfers=True)
    explanation = plan.explain(format="json")
    assert explanation["operations"][0]["implementation"] == "dft.rks/gpu4pyscf"
    assert any(item["kind"] == "explicit" and item["to"] == "cpu0"
               for item in explanation["transfers"])


def test_gpu_rks_matches_cpu_pbe_reference():
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    gpu_energy, gpu_homo, gpu_lumo = gpu_pbe.compile(MOLECULE, GPU).run(timeout_s=120)
    copied_energy, copied_homo, copied_lumo = gpu_pbe.compile(
        MOLECULE, GPU, CPU, strict_transfers=True).run(timeout_s=120)
    assert gpu_energy.value == pytest.approx(-1.1520727951590923, abs=1e-8)
    assert copied_energy.value == pytest.approx(gpu_energy.value, abs=1e-10)
    assert copied_homo.value == pytest.approx(gpu_homo.value, abs=1e-10)
    assert copied_lumo.value == pytest.approx(gpu_lumo.value, abs=1e-10)
    assert gpu_energy.provenance["implementation"] == "dft.rks/gpu4pyscf"
