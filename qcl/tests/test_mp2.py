import pytest

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def mp2_energy(molecule, device):
    reference = qcl.scf.hf(molecule).to(device)
    return qcl.mp2(reference).to(device).energy


@qcl.workflow
def transferred_mp2_energy(molecule, gpu, cpu):
    reference = qcl.scf.hf(molecule).to(gpu).to(cpu)
    return qcl.mp2(reference).to(cpu).energy


def test_mp2_cpu_matches_pyscf_reference():
    plan = mp2_energy.compile(MOLECULE, CPU)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "scf.hf", "mp2"]
    result = plan.run()
    assert result.value == pytest.approx(-1.1298973809859585, abs=1e-9)
    assert result.provenance["method"]["family"] == "MP2"


def test_mp2_requires_rhf_and_supported_placement():
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        mp2_energy.compile(MOLECULE, qcl.Device("qpu0", "qpu"))

    @qcl.workflow
    def from_dft(molecule, cpu):
        return qcl.mp2(qcl.dft.rks(molecule, xc="pbe").to(cpu)).to(cpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        from_dft.compile(MOLECULE, CPU)

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        mp2_energy.compile(qcl.Molecule("H 0 0 0", "sto-3g", spin=1), CPU)


def test_gpu_mp2_matches_cpu_reference():
    gpu = qcl.Device("gpu0", "gpu")
    assert mp2_energy.compile(MOLECULE, gpu).explain(
        format="json")["operations"][-1]["implementation"] == "mp2/gpu4pyscf"
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    cpu_value = mp2_energy.compile(MOLECULE, CPU).run().value
    gpu_result = mp2_energy.compile(MOLECULE, gpu).run(timeout_s=120)
    assert gpu_result.value == pytest.approx(cpu_value, abs=1e-8)
    assert gpu_result.provenance["implementation"] == "mp2/gpu4pyscf"
    transferred = transferred_mp2_energy.compile(
        MOLECULE, gpu, CPU, strict_transfers=True)
    assert any(item["kind"] == "explicit" for item in transferred.explain(
        format="json")["transfers"])
    assert transferred.run(timeout_s=120).value == pytest.approx(cpu_value, abs=1e-8)
