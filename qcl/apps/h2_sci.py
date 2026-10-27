"""Selected CI energy with an explicit spin target and selection threshold."""

import qcl


@qcl.workflow
def selected_energy(molecule, active, cpu):
    reference = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(reference, active).to(cpu)
    return qcl.sci.solve(space.hamiltonian, space.sector,
                         select_cutoff=1e-10, total_spin=0).to(cpu).energy


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    print(selected_energy.compile(molecule, qcl.Active(2, 2),
                                  qcl.Cluster.local().cpus[0]).run())
