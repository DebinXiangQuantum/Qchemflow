"""Stoichiometric electronic reaction energy from QCL ground states."""

import qcl


@qcl.workflow
def reaction_energy(species, cpu):
    terms = []
    for molecule, active, coefficient in species:
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        energy = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        terms.append(coefficient * energy)
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


if __name__ == "__main__":
    cpu = qcl.Cluster.local().cpus[0]
    plan = reaction_energy.compile([
        (qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), qcl.Active(2, 2), -1),
        (qcl.Molecule("H 0 0 0", "sto-3g", spin=1), qcl.Active(1, 1), 2),
    ], cpu)
    print(plan.explain())
    print(plan.run())
