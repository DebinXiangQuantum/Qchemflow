import pytest

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")
GPU = qcl.Device("gpu0", "gpu")


@qcl.workflow
def gpu_hf(molecule, gpu):
    return qcl.scf.hf(molecule).to(gpu).energy


@qcl.workflow
def gpu_rhf_to_cpu_fci(molecule, gpu, cpu, explicit):
    mean_field = qcl.scf.rhf(molecule).to(gpu)
    if explicit:
        mean_field = mean_field.to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy


def test_gpu_scf_compiles_pcm_but_rejects_eri_override():
    plan = gpu_hf.compile(MOLECULE, GPU)
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "scf.hf/gpu4pyscf")
    open_shell = gpu_hf.compile(qcl.Molecule("H 0 0 0", "sto-3g", spin=1), GPU)
    assert open_shell.explain(format="json")["operations"][0]["implementation"] == (
        "scf.hf/gpu4pyscf")

    @qcl.workflow
    def solvated(molecule, gpu):
        return qcl.scf.rhf(molecule, solvent=qcl.PCM("water")).to(gpu).energy

    pcm = solvated.compile(MOLECULE, GPU)
    assert pcm.explain(format="json")["operations"][0]["implementation"] == (
        "scf.rhf/gpu4pyscf-pcm")

    @qcl.workflow
    def solvated_with_eri(molecule, gpu):
        eri = qcl.int2e(molecule).to(CPU)
        return qcl.scf.rhf(molecule, eri=eri,
                           solvent=qcl.PCM("water")).to(gpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        solvated_with_eri.compile(MOLECULE, GPU)


@pytest.mark.parametrize("solvent", ["water", "toluene"])
@pytest.mark.parametrize("gpu_index", [0, 1])
def test_gpu_pcm_matches_cpu_reference(solvent, gpu_index):
    pytest.importorskip("gpu4pyscf")
    if len(qcl.Cluster.local().gpus) <= gpu_index:
        pytest.skip("Requested CUDA GPU unavailable")
    from pyscf import gto, scf

    @qcl.workflow
    def solvated(molecule, device, name):
        return qcl.scf.rhf(molecule, solvent=qcl.PCM(name)).to(device).energy

    device = qcl.Device(f"gpu{gpu_index}", "gpu")
    plan = solvated.compile(MOLECULE, device, solvent)
    result = plan.run(timeout_s=120)
    system = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    reference = scf.RHF(system).PCM()
    reference.with_solvent.method = "IEF-PCM"
    reference.with_solvent.eps = qcl.PCM(solvent).eps
    reference.kernel()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-8)
    assert result.provenance["implementation"] == "scf.rhf/gpu4pyscf-pcm"
    assert result.provenance["device"] == device.id
    assert result.provenance["method"]["solvent"] == solvent


@pytest.mark.parametrize("explicit", [False, True])
def test_gpu_scf_matches_cpu_fci_after_transfer(explicit):
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    plan = gpu_rhf_to_cpu_fci.compile(MOLECULE, GPU, CPU, explicit,
                                       strict_transfers=explicit)
    assert any(item["kind"] == ("explicit" if explicit else "implicit")
               for item in plan.explain(format="json")["transfers"])
    result = plan.run(timeout_s=120)
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)


def test_gpu_scf_matches_cpu_energy():
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    result = gpu_hf.compile(MOLECULE, GPU).run(timeout_s=120)
    assert result.value == pytest.approx(-1.1167593073964, abs=1e-8)
    assert result.provenance["implementation"] == "scf.hf/gpu4pyscf"


def test_two_gpu_scf_operations_in_one_plan():
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) < 2:
        pytest.skip("Two CUDA GPUs are required")
    from pyscf import gto, scf

    @qcl.workflow
    def independent(molecules, devices):
        return [qcl.scf.rhf(molecule).to(device).energy
                for molecule, device in zip(molecules, devices, strict=True)]

    second = qcl.Molecule("H 0 0 0; H 0 0 0.9", "sto-3g")
    plan = independent.compile((MOLECULE, second), cluster.gpus[:2],
                               strict_transfers=True)
    results = plan.run(timeout_s=120)
    for result, molecule, device in zip(results, (MOLECULE, second),
                                        cluster.gpus[:2], strict=True):
        reference = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                                 verbose=0)).run()
        assert result.value == pytest.approx(reference.e_tot, abs=1e-8)
        assert result.provenance["device"] == device.id


def test_scf_guess_can_move_between_two_gpus():
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) < 2:
        pytest.skip("Two CUDA GPUs are required")
    from pyscf import gto, scf

    @qcl.workflow
    def continued(start, finish, first, second):
        previous = qcl.scf.rhf(start).to(first).to(second)
        return qcl.scf.rhf(finish, guess=previous).to(second).energy

    second = qcl.Molecule("H 0 0 0; H 0 0 0.9", "sto-3g")
    plan = continued.compile(MOLECULE, second, *cluster.gpus[:2],
                             strict_transfers=True)
    result = plan.run(timeout_s=120)
    reference = scf.RHF(gto.M(atom=second.atoms, basis=second.basis,
                             verbose=0)).run()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-8)
    assert result.provenance["device"] == cluster.gpus[1].id
