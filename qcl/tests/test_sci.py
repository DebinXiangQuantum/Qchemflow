import pytest

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def selected_energy(molecule, active, cpu, cutoff=1e-10):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sci.solve(space.hamiltonian, space.sector,
                         select_cutoff=cutoff, total_spin=0).to(cpu).energy


def test_sci_cpu_matches_fci_on_complete_small_space():
    plan = selected_energy.compile(MOLECULE, qcl.Active(2, 2), CPU)
    assert plan.explain(format="json")["operations"][-1]["implementation"] == (
        "sci.solve/chemflow-pyscf")
    result = plan.run()
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)
    assert result.provenance["method"]["family"] == "SCI"
    assert result.provenance["method"]["select_cutoff"] == 1e-10


def test_sci_cutoff_controls_variational_candidate():
    lithium_hydride = qcl.Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g")
    active = qcl.Active(4, 5)
    tight = selected_energy.compile(lithium_hydride, active, CPU,
                                    cutoff=1e-10).run()
    loose = selected_energy.compile(lithium_hydride, active, CPU,
                                    cutoff=0.01).run()
    assert loose.value >= tight.value - 1e-8
    assert (loose.provenance["method"]["determinant_count"] <=
            tight.provenance["method"]["determinant_count"])


def test_sci_rejects_wrong_inputs_and_device():
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        selected_energy.compile(MOLECULE, qcl.Active(2, 2),
                                qcl.Device("gpu0", "gpu"))
    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        selected_energy.compile(MOLECULE, qcl.Active(2, 2), CPU, cutoff=-1)

    @qcl.workflow
    def wrong_operator(molecule, cpu):
        return qcl.sci.solve(qcl.scf.hf(molecule).to(cpu),
                             qcl.Sector(2, 0)).to(cpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        wrong_operator.compile(MOLECULE, CPU)
