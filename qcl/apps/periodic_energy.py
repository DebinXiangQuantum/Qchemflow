"""A periodic KRHF energy on a CPU."""

import qcl

SYSTEM = {
    "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
    "atoms": "C 0 0 0; C 0.8917 0.8917 0.8917",
    "basis": "gth-szv", "pseudo": "gth-pade", "unit": "angstrom",
}


@qcl.workflow
def energy(system, cpu):
    return qcl.periodic.scf(system, kmesh=(1, 1, 1)).to(cpu)


if __name__ == "__main__":
    result = energy.compile(SYSTEM, qcl.Cluster.local().cpus[0]).run(timeout_s=180)
    print(f"{result.value:.12f} {result.unit}")
