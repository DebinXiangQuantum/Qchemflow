"""Open-shell UHF/UKS energy and spin-resolved orbital projections."""

import qcl


@qcl.workflow
def atomic_orbitals(molecule, cpu):
    mean_field = qcl.dft.uks(molecule, xc="pbe").to(cpu)
    return (mean_field.energy,
            mean_field.orbital_energy("homo", spin="alpha"),
            mean_field.orbital_energy("homo", spin="beta"))


if __name__ == "__main__":
    atom = qcl.Molecule("Li 0 0 0", "sto-3g", spin=1)
    energy, alpha_homo, beta_homo = atomic_orbitals.compile(
        atom, qcl.Cluster.local().cpus[0]).run()
    print(energy, alpha_homo, beta_homo)
