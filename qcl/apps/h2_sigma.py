"""Apply the active-space Hamiltonian to an H2 FCI coefficient tensor."""

import numpy as np

import qcl


@qcl.workflow
def h2_sigma(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    state = qcl.fci.solve(space.hamiltonian, space.sector,
                          engine="native").to(cpu)
    coefficients = state.coefficients
    return state.energy, coefficients, qcl.sigma(
        space.hamiltonian, coefficients).to(cpu)


if __name__ == "__main__":
    energy, coefficients, product = h2_sigma.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
        qcl.Active(2, 2), qcl.Cluster.local().cpus[0]).run()
    print("Energy (hartree):", energy.value)
    print("Hamiltonian residual:",
          np.linalg.norm(product.data - energy.value * coefficients.data))
