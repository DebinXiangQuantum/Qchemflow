"""H2 FCI energy and spin-summed active-orbital density matrices."""

import numpy as np

import qcl


@qcl.workflow
def h2_density(molecule, active, cpu, engine="native"):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    state = qcl.fci.solve(space.hamiltonian, space.sector,
                          engine=engine).to(cpu)
    return {"energy": state.energy,
            "rdm1": qcl.rdm1(state).to(cpu),
            "rdm2": qcl.rdm2(state).to(cpu)}


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = h2_density.compile(molecule, qcl.Active(2, 2),
                                qcl.Cluster.local().cpus[0]).run()
    print("Energy (hartree):", result["energy"].value)
    print("1-RDM trace:", np.trace(result["rdm1"].data))
    print("2-RDM shape:", result["rdm2"].shape)
