import pytest
from pyscf import gto, scf

import qcl


@qcl.workflow
def solvation_scan(molecule, solvents, cpu):
    integrals = qcl.int2e(molecule).to(cpu)
    gas = qcl.scf.rhf(molecule, eri=integrals).to(cpu).energy

    def solvated(name, lane):
        result = qcl.scf.rhf(molecule, eri=integrals,
                             solvent=qcl.PCM(name)).to(lane)
        return result.energy - gas

    return qcl.map(solvated, solvents).to([cpu])


def test_pcm_scan_reuses_integrals_and_matches_pyscf():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = solvation_scan.compile(molecule, ["water", "toluene"], cpu,
                                  strict_transfers=True)
    operations = plan.explain(format="json")["operations"]
    assert [item["op"] for item in operations] == [
        "int2e", "scf.rhf", "scf.rhf", "scf.rhf",
    ]
    assert plan.explain(format="json")["transfers"] == []
    result = plan.run()
    system = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    gas = scf.RHF(system).run().e_tot
    for name, estimate in zip(("water", "toluene"), result, strict=True):
        reference = scf.RHF(system).PCM()
        reference.with_solvent.method = "IEF-PCM"
        reference.with_solvent.eps = qcl.PCM(name).eps
        reference.kernel()
        assert estimate.value == pytest.approx(reference.e_tot - gas, abs=1e-8)
        assert estimate.provenance["operands"][0]["method"]["solvent"] == name


def test_invalid_solvent_and_gpu_integral_placement_are_rejected():
    with pytest.raises(ValueError):
        qcl.PCM("not-a-solvent")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        solvation_scan.compile(molecule, ["water"], qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


@qcl.workflow
def smd_energy(molecule, device, solvent):
    return qcl.scf.rhf(molecule, solvent=solvent).to(device).energy


def test_smd_descriptor_and_cpu_plan():
    with pytest.raises(ValueError):
        qcl.SMD("not-a-solvent")
    plan = smd_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Device("cpu0", "cpu"), qcl.SMD("water"))
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "scf.rhf/chemflow-pyscf-smd")


@pytest.mark.parametrize("ordinal", [0, 1])
def test_smd_gpu_matches_pyscf_cpu_when_available(ordinal):
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    solvent = qcl.SMD("water")
    result = smd_energy.compile(molecule, cluster.gpus[ordinal], solvent).run(timeout_s=120)
    system = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    reference = scf.RHF(system).SMD()
    reference.with_solvent.solvent = "water"
    reference.kernel()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-8)
    assert result.provenance["implementation"] == "scf.rhf/gpu4pyscf-smd"
    assert result.provenance["method"]["solvent"] == "water"
