"""H2 full-CI ground energy on the CPU reference implementation."""

import qcl


@qcl.workflow
def ground_energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, sector=space.sector).to(cpu).energy


if __name__ == "__main__":
    device = qcl.Cluster.local().cpus[0]
    plan = ground_energy.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
        qcl.Active(electrons=2, orbitals=2), device,
    )
    print(plan.explain())
    print(plan.run())
