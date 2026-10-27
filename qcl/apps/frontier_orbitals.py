"""Frontier orbital energies and vertical IP/EA from one active Hamiltonian."""

import qcl


@qcl.workflow
def frontier_orbitals(molecule, active, cpu):
    mean_field = qcl.scf.rhf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    energies = {
        delta: qcl.fci.solve(space.hamiltonian,
                             space.sector.add_electrons(delta)).to(cpu).energy
        for delta in (-1, 0, 1)
    }
    ip = energies[-1] - energies[0]
    ea = energies[0] - energies[1]
    return mean_field.homo, mean_field.lumo, ip, ea, ip - ea


if __name__ == "__main__":
    plan = frontier_orbitals.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
        qcl.Active(2, 2), qcl.Cluster.local().cpus[0],
    )
    print(plan.explain())
    print(plan.run())
