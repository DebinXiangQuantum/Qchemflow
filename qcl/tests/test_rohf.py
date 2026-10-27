import pytest

import qcl


@qcl.workflow
def atomic_rohf(molecule, cpu):
    return qcl.scf.rohf(molecule).to(cpu).energy


@qcl.workflow
def rohf_fci_after_gpu_copy(molecule, active, gpu, cpu):
    mean_field = qcl.scf.rohf(molecule).to(gpu).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy


def test_explicit_rohf_matches_open_shell_hf():
    atom = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
    cpu = qcl.Device("cpu0", "cpu")
    plan = atomic_rohf.compile(atom, cpu)
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "scf.rohf/chemflow-pyscf")
    assert plan.run().value == pytest.approx(-0.466581849557, abs=1e-9)


def test_rohf_rejects_closed_shell():
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        atomic_rohf.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                            qcl.Device("cpu0", "cpu"))


def test_gpu_rohf_matches_cpu_for_open_shell_atoms():
    gpu = qcl.Device("gpu0", "gpu")
    cpu = qcl.Device("cpu0", "cpu")
    for atom in (qcl.Molecule("H 0 0 0", "sto-3g", spin=1),
                 qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)):
        assert atomic_rohf.compile(atom, gpu).explain(
            format="json")["operations"][0]["implementation"] == "scf.rohf/gpu4pyscf"
    pytest.importorskip("gpu4pyscf")
    if not qcl.Cluster.local().gpus:
        pytest.skip("CUDA GPU unavailable")
    for atom in (qcl.Molecule("H 0 0 0", "sto-3g", spin=1),
                 qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)):
        expected = atomic_rohf.compile(atom, cpu).run().value
        assert atomic_rohf.compile(atom, gpu).run(timeout_s=120).value == (
            pytest.approx(expected, abs=1e-8))

        @qcl.workflow
        def automatic_reference(molecule, device):
            return qcl.scf.hf(molecule).to(device).energy

        assert automatic_reference.compile(atom, gpu).run(timeout_s=120).value == (
            pytest.approx(expected, abs=1e-8))
    atom = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
    plan = rohf_fci_after_gpu_copy.compile(
        atom, qcl.Active(1, 1), gpu, cpu, strict_transfers=True)
    assert plan.run(timeout_s=120).value == pytest.approx(-0.4665818495572752,
                                                         abs=1e-8)
