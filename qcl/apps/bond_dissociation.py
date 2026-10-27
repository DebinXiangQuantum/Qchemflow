"""Rigid homolytic bond dissociation using a species-sized active space."""

import qcl


@qcl.workflow
def bond_dissociation(molecule, bonds, cpu, active=None):
    active = active if active is not None else qcl.Active(max_orbitals=2)
    def energy(species, device):
        mean_field = qcl.scf.hf(species).to(device)
        space = qcl.active_space(mean_field, active).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    parent = energy(molecule, cpu)

    def point(bond, lane):
        first, second = molecule.cleave(bond)
        return energy(first, lane) + energy(second, lane) - parent

    return qcl.map(point, bonds).to([cpu])


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = bond_dissociation.compile(molecule, molecule.bonds(),
                                     qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
