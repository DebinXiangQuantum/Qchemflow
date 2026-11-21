"""H2 potential-energy curve: FCI, HF and CCSD at five bond lengths in one QCL plan."""

import qcl

RADII = (0.6, 0.8, 1.0, 1.4, 2.0)
METHODS = ("fci", "hf", "ccsd")


@qcl.workflow
def h2_pes(cpu):
    rows = {}
    for radius in RADII:
        molecule = qcl.Molecule(f"H 0 0 0; H 0 0 {radius}", basis="sto-3g")
        space = qcl.active_space(qcl.scf.hf(molecule).to(cpu), qcl.Active(2, 2)).to(cpu)
        rows[f"{radius:.1f}"] = {
            "fci": qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy,
            **{method: qcl.classical.solve(space.hamiltonian, space.sector,
                                           method=method).to(cpu).energy
               for method in METHODS[1:]}}
    return rows


def main() -> dict:
    plan = h2_pes.compile(qcl.Cluster.local().cpus[0])
    energies, relations = plan.run(), plan.relations()
    print("R/angstrom | method | E/hartree | relation")
    for radius, row in energies.items():
        for method in METHODS:
            print(f"{radius} | {method} | {row[method].value:.10f} | "
                  f"{relations[radius][method]}")
    return energies


if __name__ == "__main__":
    main()
