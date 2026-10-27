"""Connected HCN to HNC FCI/RRHO barrier and Eyring rate."""

import qcl


@qcl.workflow
def reaction_barrier(reactant, guess, product, active, cpu, *,
                     temperature_K=None, pressure_Pa=101325.0,
                     quantity="gibbs", rate_constant=False):
    if rate_constant and (temperature_K is None or quantity != "gibbs"):
        raise ValueError("Eyring rate requires a Gibbs barrier and temperature")
    minimum = qcl.optimize(reactant).to(cpu)
    transition = qcl.engines.sella.saddle(guess, reactant, product).to(cpu)

    def energy(molecule):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        electronic = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        if temperature_K is None:
            return electronic
        modes = qcl.frequencies(molecule).to(cpu)
        correction = qcl.thermo.rrho(
            molecule, modes, temperature_K=temperature_K,
            pressure_Pa=pressure_Pa, quantity=quantity).to(cpu)
        return electronic + correction

    barrier = energy(transition) - energy(minimum)
    if rate_constant:
        return barrier, qcl.thermo.eyring(barrier, temperature_K).to(cpu)
    return barrier


if __name__ == "__main__":
    plan = reaction_barrier.compile(
        qcl.Molecule("C 0 0 0; N 0 0 1.16; H 0 0 -1.07", "sto-3g"),
        qcl.Molecule("C 0 0 0; N 0 0 1.2; H 1.1 0 .5", "sto-3g"),
        qcl.Molecule("C 0 0 0; N 0 0 1.17; H 0 0 2.17", "sto-3g"),
        qcl.Active(2, 2), qcl.Cluster.local().cpus[0],
        temperature_K=298.15, rate_constant=True,
    )
    print(plan.explain())
    print(plan.run(timeout_s=1800))
