"""H2 small-space FCI using QCL's explicit native Rust CPU engine."""

import qcl


@qcl.workflow
def ground(molecule, active, cpu):
    mean_field = qcl.scf.rhf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector,
                         engine="native").to(cpu).energy


if __name__ == "__main__":
    plan = ground.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Active(2, 2), qcl.Cluster.local().cpus[0])
    result = plan.run()
    print(result.value, result.unit, result.provenance["implementation"])
