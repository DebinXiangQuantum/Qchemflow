import pytest

import qcl

ATOM = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
CPU = qcl.Device("cpu0", "cpu")
GPU = qcl.Device("gpu0", "gpu")


@qcl.workflow
def uhf_energy(molecule, device):
    return qcl.scf.uhf(molecule).to(device).energy


@qcl.workflow
def uks_energy(molecule, device):
    return qcl.dft.uks(molecule, xc="pbe").to(device).energy


@qcl.workflow
def transferred_uhf_energy(molecule, gpu, cpu):
    return qcl.scf.uhf(molecule).to(gpu).to(cpu).energy


@qcl.workflow
def transferred_uks_energy(molecule, gpu, cpu):
    return qcl.dft.uks(molecule, xc="pbe").to(gpu).to(cpu).energy


@qcl.workflow
def uhf_orbitals(molecule, device):
    mean_field = qcl.scf.uhf(molecule).to(device)
    return (mean_field.orbital_energy("homo", spin="alpha"),
            mean_field.orbital_energy("homo", spin="beta"),
            mean_field.orbital_energy("lumo", spin="alpha"))


@qcl.workflow
def uks_orbitals(molecule, device):
    mean_field = qcl.dft.uks(molecule, xc="pbe").to(device)
    return (mean_field.orbital_energy("homo", spin="alpha"),
            mean_field.orbital_energy("homo", spin="beta"))


@qcl.workflow
def copied_uhf_orbital(molecule, gpu, cpu):
    mean_field = qcl.scf.uhf(molecule).to(gpu).to(cpu)
    return mean_field.orbital_energy("homo", spin="alpha")


def test_unrestricted_cpu_reference_and_diagnostics():
    assert uhf_energy.compile(ATOM, CPU).run().value == pytest.approx(
        -0.4665818495572752, abs=1e-9)
    assert uks_energy.compile(ATOM, CPU).run().value == pytest.approx(
        -0.46437566325051144, abs=1e-8)
    closed = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        uhf_energy.compile(closed, CPU)
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        uks_energy.compile(closed, CPU)

    @qcl.workflow
    def unsupported(molecule, cpu):
        return qcl.scf.uhf(molecule).to(cpu).homo

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        unsupported.compile(ATOM, CPU)


def test_unrestricted_gpu_reference():
    assert uhf_energy.compile(ATOM, GPU).explain(
        format="json")["operations"][0]["implementation"] == "scf.uhf/gpu4pyscf"
    assert uks_energy.compile(ATOM, GPU).explain(
        format="json")["operations"][0]["implementation"] == "dft.uks/gpu4pyscf"
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    assert uhf_energy.compile(ATOM, GPU).run(timeout_s=120).value == pytest.approx(
        -0.4665818495572752, abs=1e-8)
    assert uks_energy.compile(ATOM, GPU).run(timeout_s=120).value == pytest.approx(
        -0.46437566325051144, abs=1e-8)
    assert transferred_uhf_energy.compile(ATOM, GPU, CPU).run(
        timeout_s=120).value == pytest.approx(-0.4665818495572752, abs=1e-8)
    assert transferred_uks_energy.compile(ATOM, GPU, CPU).run(
        timeout_s=120).value == pytest.approx(-0.46437566325051144, abs=1e-8)
    lithium = qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)
    for workflow in (uhf_energy, uks_energy):
        cpu_value = workflow.compile(lithium, CPU).run().value
        gpu_value = workflow.compile(lithium, GPU).run(timeout_s=120).value
        assert gpu_value == pytest.approx(cpu_value, abs=1e-8)


def test_spin_resolved_orbitals_require_matching_reference():
    lithium = qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)
    alpha, beta, lumo = uhf_orbitals.compile(lithium, CPU).run()
    assert alpha.value > beta.value
    assert lumo.value > alpha.value
    assert alpha.provenance["question"]["property"] == "homo_alpha"
    uks_alpha, uks_beta = uks_orbitals.compile(lithium, CPU).run()
    assert uks_alpha.value > uks_beta.value

    @qcl.workflow
    def invalid(molecule, cpu):
        return qcl.scf.uhf(molecule).to(cpu).orbital_energy("homo", spin="wrong")

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        invalid.compile(lithium, CPU)

    @qcl.workflow
    def restricted(molecule, cpu):
        return qcl.scf.rhf(molecule).to(cpu).orbital_energy("homo", spin="alpha")

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        restricted.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), CPU)


def test_spin_resolved_orbitals_match_on_gpu():
    assert uhf_orbitals.compile(qcl.Molecule("Li 0 0 0", "sto-3g", spin=1),
                                GPU).output_type == "composite"
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    lithium = qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)
    for workflow in (uhf_orbitals, uks_orbitals):
        cpu_values = workflow.compile(lithium, CPU).run()
        gpu_values = workflow.compile(lithium, GPU).run(timeout_s=120)
        for cpu_value, gpu_value in zip(cpu_values, gpu_values, strict=True):
            assert gpu_value.value == pytest.approx(cpu_value.value, abs=1e-7)
    copied = copied_uhf_orbital.compile(
        lithium, GPU, CPU, strict_transfers=True).run(timeout_s=120)
    assert copied.value == pytest.approx(uhf_orbitals.compile(lithium, CPU).run()[0].value,
                                       abs=1e-7)
