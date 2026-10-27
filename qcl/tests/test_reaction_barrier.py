import pytest

import qcl

HCN = "C 0 0 0; N 0 0 1.16; H 0 0 -1.07"
HNC = "C 0 0 0; N 0 0 1.17; H 0 0 2.17"
SADDLE = "C 0 0 0; N 0 0 1.2; H 1.1 0 .5"


@qcl.workflow
def reaction_barrier(reactant, ts_guess, product, active, cpu):
    minimum = qcl.optimize(reactant).to(cpu)
    transition = qcl.engines.sella.saddle(ts_guess, reactant, product).to(cpu)

    def energy(molecule):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy

    return energy(transition) - energy(minimum)


def test_hcn_barrier_requires_connected_transition_state():
    molecules = tuple(qcl.Molecule(atoms, "sto-3g") for atoms in (HCN, SADDLE, HNC))
    plan = reaction_barrier.compile(*molecules, qcl.Active(2, 2),
                                    qcl.Cluster.local().cpus[0])
    operations = [item["op"] for item in plan.explain(format="json")["operations"]]
    assert operations == ["geometry.optimize", "sella.saddle", "scf.hf", "active_space",
                          "fci.solve", "scf.hf", "active_space", "fci.solve"]
    barrier = plan.run(timeout_s=1800)
    assert barrier.value == pytest.approx(0.14, abs=0.1)
    assert barrier.provenance["operands"][0]["question"]["property"] == "electronic_energy"


@qcl.workflow
def thermal_reaction_barrier(reactant, ts_guess, product, active, cpu, temperature):
    minimum = qcl.optimize(reactant).to(cpu)
    transition = qcl.engines.sella.saddle(ts_guess, reactant, product).to(cpu)

    def free_energy(molecule):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        electronic = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        modes = qcl.frequencies(molecule).to(cpu)
        correction = qcl.thermo.rrho(
            molecule, modes, temperature_K=temperature).to(cpu)
        return electronic + correction

    barrier = free_energy(transition) - free_energy(minimum)
    return barrier, qcl.thermo.eyring(barrier, temperature).to(cpu)


def test_hcn_free_energy_barrier_and_eyring_rate():
    from chemflow.domains.paths import rate_constant

    molecules = tuple(qcl.Molecule(atoms, "sto-3g") for atoms in (HCN, SADDLE, HNC))
    plan = thermal_reaction_barrier.compile(
        *molecules, qcl.Active(2, 2), qcl.Cluster.local().cpus[0], 298.15)
    barrier, rate = plan.run(timeout_s=1800)
    assert barrier.unit == "hartree"
    assert barrier.value > 0
    assert rate.unit == "1/s"
    assert rate.value == pytest.approx(
        rate_constant(None, barrier.value, 298.15)["k_per_s"], rel=1e-10)
