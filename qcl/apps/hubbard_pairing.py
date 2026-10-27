"""Hubbard finite-cluster energy and site correlations."""

import qcl


@qcl.workflow
def hubbard_pairing(lattice, U, sector, i, j, device):
    operator = qcl.models.hubbard(lattice, t=1.0, U=U)
    ground = qcl.ed.ground_state(operator, sector=sector).to(device)
    pair = qcl.expect(ground, qcl.ops.onsite_pair(i, j)).to(device)
    spin = qcl.expect(ground, qcl.ops.spin_spin(i, j)).to(device)
    return ground.energy, pair, spin


if __name__ == "__main__":
    plan = hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                   qcl.Sector(n=2, sz=0), 0, 1,
                                   qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
