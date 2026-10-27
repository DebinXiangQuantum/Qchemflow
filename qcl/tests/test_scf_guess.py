import pytest
from pyscf import gto, scf

import qcl

START = qcl.Molecule("H 0 0 0; H 0 0 0.75", "sto-3g")
TARGET = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def seeded_rhf(start, target, device):
    previous = qcl.scf.rhf(start).to(device)
    return qcl.scf.rhf(target, guess=previous).to(device).energy


def test_rhf_guess_reuses_converged_density_without_changing_energy():
    plan = seeded_rhf.compile(START, TARGET, CPU, strict_transfers=True)
    operations = plan.explain(format="json")["operations"]
    assert [item["op"] for item in operations] == ["scf.rhf", "scf.rhf"]
    assert plan.explain(format="json")["transfers"] == []
    result = plan.run()
    reference = scf.RHF(gto.M(atom=TARGET.atoms, basis=TARGET.basis,
                              verbose=0)).run(conv_tol=1e-10)
    assert result.value == pytest.approx(reference.e_tot, abs=1e-9)
    assert result.provenance["method"]["guess"] == "RHF density"
    assert result.provenance["method"]["guess_molecule"]["atoms"] == START.atoms


def test_rhf_guess_requires_matching_basis_and_reference():
    different_basis = qcl.Molecule(START.atoms, "6-31g")
    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        seeded_rhf.compile(different_basis, TARGET, CPU)

    @qcl.workflow
    def wrong_reference(molecule, device):
        previous = qcl.dft.rks(molecule, xc="pbe").to(device)
        return qcl.scf.rhf(molecule, guess=previous).to(device).energy

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        wrong_reference.compile(TARGET, CPU)


def test_rhf_guess_requires_explicit_cross_device_copy():
    @qcl.workflow
    def copied(start, target, cpu, gpu, explicit):
        previous = qcl.scf.rhf(start).to(cpu)
        if explicit:
            previous = previous.to(gpu)
        return qcl.scf.rhf(target, guess=previous).to(gpu).energy

    gpu = qcl.Device("gpu0", "gpu")
    with pytest.raises(qcl.CompileError, match="QCL-E205"):
        copied.compile(START, TARGET, CPU, gpu, False, strict_transfers=True)
    plan = copied.compile(START, TARGET, CPU, gpu, True, strict_transfers=True)
    assert any(item["kind"] == "explicit" and item["to"] == "gpu0"
               for item in plan.explain(format="json")["transfers"])


def test_rhf_guess_on_gpu_matches_cpu():
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("CUDA GPU unavailable")
    gpu = cluster.gpus[-1]
    gpu_result = seeded_rhf.compile(START, TARGET, gpu).run(timeout_s=120)
    cpu_result = seeded_rhf.compile(START, TARGET, cluster.cpus[0]).run()
    assert gpu_result.value == pytest.approx(cpu_result.value, abs=1e-8)

    @qcl.workflow
    def copied(start, target, cpu, gpu):
        previous = qcl.scf.rhf(start).to(cpu).to(gpu)
        return qcl.scf.rhf(target, guess=previous).to(gpu).energy

    crossed = copied.compile(START, TARGET, cluster.cpus[0], gpu,
                             strict_transfers=True).run(timeout_s=120)
    assert crossed.value == pytest.approx(cpu_result.value, abs=1e-8)
