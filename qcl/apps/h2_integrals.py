"""Inspect one-electron AO integrals and their generalized eigensystem."""

import qcl


@qcl.workflow
def integrals(molecule, cpu):
    kinetic = qcl.int1e(molecule, "kinetic").to(cpu)
    overlap = qcl.int1e(molecule, "overlap").to(cpu)
    spectrum = qcl.eigh(kinetic, overlap).to(cpu)
    trace = qcl.contract("ij,ji->", kinetic, overlap).to(cpu)
    orthogonalizer = qcl.orthogonalize(overlap).to(cpu)
    solution = qcl.solve(overlap, spectrum.values).to(cpu)
    singular_values = qcl.svd(kinetic).to(cpu).s
    molecular_kinetic = qcl.transform(kinetic, spectrum.vectors).to(cpu)
    molecular_eri = qcl.transform(qcl.int2e(molecule).to(cpu),
                                  spectrum.vectors).to(cpu)
    return (spectrum, trace, orthogonalizer, solution, singular_values,
            molecular_kinetic, molecular_eri)


if __name__ == "__main__":
    plan = integrals.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                             qcl.Cluster.local().cpus[0])
    print(plan.explain())
    (spectrum, trace, orthogonalizer, solution, singular_values,
     molecular_kinetic, molecular_eri) = plan.run()
    print(spectrum.values.data)
    print(spectrum.vectors.data)
    print(trace.data)
    print(orthogonalizer.data)
    print(solution.data)
    print(singular_values.data)
    print(molecular_kinetic.data)
    print(molecular_eri.data)
