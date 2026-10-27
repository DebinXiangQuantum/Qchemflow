from chemflow import Q, build
from chemflow.ir.render import render
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def main() -> None:
    with build.program("h2_pes") as builder:
        radius = builder.scan("R", Q([0.6, 0.8, 1.0, 1.4, 2.0], "angstrom"))
        molecule = builder.molecule("H 0 0 0; H 0 0 {R}", params={"R": radius})
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        results = []
        for method in ("fci_pyscf", "hf", "ccsd"):
            energy = builder.ground_energy(hamiltonian, sector)
            builder.refine(energy, method)
            builder.bind(energy, method)
            results.append((method, energy.id))
        builder.budget(qpu_shots=0)
    program = builder.to_ir()
    print(render(program))
    print("R/angstrom | method | E/hartree | relation | evidence | wall_s")
    for radius, point in zip(program.scans[0].values.value,
                             Executor(molecular_registry()).run(program)["scan"]):
        for method, node_id in results:
            estimate = point[node_id]
            print(f"{radius:.1f} | {method} | {estimate.value.value:.10f} | "
                  f"{estimate.relation} | {estimate.evidence[0].status} | "
                  f"{estimate.cost.wall_s:.4f}")


if __name__ == "__main__":
    main()
