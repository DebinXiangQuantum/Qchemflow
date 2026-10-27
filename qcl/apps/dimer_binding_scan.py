"""Rigid dimer counterpoise interaction curve."""

import qcl


@qcl.workflow
def dimer_binding_scan(dimer, frag_a, frag_b, separations, cpu,
                       active=None):
    active = active if active is not None else qcl.Active(max_orbitals=4)
    def energy(molecule, device):
        mean_field = qcl.scf.hf(molecule).to(device)
        space = qcl.active_space(mean_field, active).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    def point(distance, lane):
        geometry = dimer.set_separation(frag_a, frag_b, distance)
        return (energy(geometry, lane) - energy(geometry.ghost(frag_b), lane) -
                energy(geometry.ghost(frag_a), lane))

    return qcl.map(point, separations).to([cpu])


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", "sto-3g")
    plan = dimer_binding_scan.compile(molecule, [0, 1], [2, 3], [2.0, 3.0],
                                      qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
