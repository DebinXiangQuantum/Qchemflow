"""Small frozen tasks with independently computed scientific oracles."""

from itertools import combinations

import numpy as np

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.domains import get_pack, load_registry, pack_names
from chemflow.ir.schema import Node
from chemflow.runtime import Executor


def list_benchmarks():
    return [(pack, name) for pack in pack_names() for name in get_pack(pack).benchmarks]


def _species(name, atoms, coefficient, electrons, orbitals, spin=0, role="minimum"):
    return {"name": name, "molecule": {"atoms": atoms, "spin": spin},
            "coefficient": coefficient, "active": {"n_elec": electrons, "n_orb": orbitals},
            "sector": {"n_alpha": (electrons + spin) // 2,
                       "n_beta": (electrons - spin) // 2, "n_orb": orbitals},
            "target_spin": spin / 2, "role": role}


def _task(pack, name):
    if pack == "reactions":
        species = [_species("H2", "H 0 0 0; H 0 0 0.74", -1, 2, 2),
                   _species("H", "H 0 0 0", 2, 1, 1, 1)]
        if name == "h3_exchange_barrier":
            species[1]["coefficient"] = -1
            species.append(_species("H3_TS", "H 0 0 -0.93; H 0 0 0; H 0 0 0.93",
                                    1, 3, 3, 1, "ts"))
        target = {"kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
                  "species": species}
    elif name.startswith("hubbard"):
        size = 2 if name == "hubbard_dimer" else 6
        target = {"kind": "lattice_ground_energy", "lattice": {"model": "hubbard",
                  "geometry": "chain" if size == 2 else "ring", "n_sites": size,
                  "t": 1., "U": 4., "unit": "eV"},
                  "sector": {"n_alpha": size // 2, "n_beta": size // 2, "n_orb": size},
                  "target_spin": 0}
    elif name == "h6_ewf_full_bath":
        species = _species("H6", "; ".join(f"H 0 0 {index}" for index in range(6)), 1, 6, 6)
        target = {"kind": "molecular_ground_energy", "electronic_model": {"basis": "sto-3g"},
                  **{key: species[key] for key in ("molecule", "active", "sector", "target_spin")}}
    else:
        target = {"kind": "periodic_energy", "system": {
            "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
            "atoms": "C 0 0 0; C 0.8917 0.8917 0.8917", "unit": "angstrom",
            "basis": "gth-szv", "pseudo": "gth-pade"}, "kmesh": [1, 1, 2]}
    return TaskSpec.model_validate({"question": name, "target": target,
        "gates": [] if target["kind"] == "periodic_energy" else [
            {"kind": "abs_error_vs_exact", "threshold": 1e-6, "unit": "hartree"}],
        "budget": {"wall_s": 600}, "allowed_effects": ["pure"], "decision": "benchmark"})


def _hubbard_oracle(size, hopping, repulsion, ring):
    basis = [sum(1 << site for site in sites) for sites in combinations(range(size), size // 2)]
    addresses = {mask: index for index, mask in enumerate(basis)}
    kinetic = np.zeros((len(basis), len(basis)))
    bonds = [(site, site + 1) for site in range(size - 1)]
    if ring:
        bonds.append((size - 1, 0))
    for column, mask in enumerate(basis):
        for left, right in bonds:
            for source, destination in ((left, right), (right, left)):
                if mask & (1 << source) and not mask & (1 << destination):
                    lower, upper = sorted((source, destination))
                    between = sum(1 << site for site in range(lower + 1, upper))
                    sign = (-1) ** ((mask & between).bit_count())
                    moved = mask ^ (1 << source) ^ (1 << destination)
                    kinetic[addresses[moved], column] -= hopping * sign
    matrix = np.kron(kinetic, np.eye(len(basis))) + np.kron(np.eye(len(basis)), kinetic)
    matrix += np.diag([repulsion * (alpha & beta).bit_count()
                       for alpha in basis for beta in basis])
    return float(np.linalg.eigvalsh(matrix)[0])


def _oracle(name, spec, plan, results):
    from pyscf import fci, gto, scf

    if name == "hubbard_dimer":
        return (4 - np.sqrt(4 ** 2 + 16)) / 2, "eV", "Analytic Hubbard dimer eigenvalue"
    if name == "hubbard_ring6":
        return _hubbard_oracle(6, 1., 4., True), "eV", "Independent NumPy occupation-basis ED"
    if name == "diamond_kpoint_supercell":
        from pyscf.pbc import gto as pbcgto
        from pyscf.pbc import scf as pbcscf

        system = spec.target.system
        cell = pbcgto.M(a=system["lattice"], atom=system["atoms"], unit=system["unit"],
                        basis=system["basis"], pseudo=system["pseudo"], verbose=0)
        mf = pbcscf.KRHF(cell, cell.make_kpts([1, 1, 2]), exxdiv="ewald")
        mf.conv_tol = 1e-10
        return float(mf.kernel()), "hartree", "Direct PySCF KRHF; translational supercell identity"
    if name == "h6_ewf_full_bath":
        mol = gto.M(atom=spec.target.molecule["atoms"], basis="sto-3g", verbose=0)
        return float(fci.FCI(scf.RHF(mol).run()).kernel()[0]), "hartree", "Direct PySCF full FCI"
    energy = 0.
    optimized = [results[node.id] for node in plan.nodes if node.op == "geometry_optimize"]
    for species in spec.target.species:
        molecule = optimized[0] if species.role == "ts" else species.molecule
        mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                    spin=molecule.get("spin", 0), charge=molecule.get("charge", 0),
                    basis=spec.target.electronic_model["basis"], verbose=0)
        mf = (scf.UHF if mol.spin else scf.RHF)(mol).run()
        value = mf.e_tot if mol.nelectron == 1 else fci.FCI(mf).kernel()[0]
        energy += species.coefficient * value
    return float(energy), "hartree", "Direct PySCF FCI/UHF at the reported species geometries"


def run_benchmark(pack, name):
    if (pack, name) == ("dynamics", "aim-design-bath5"):
        from chemflow.domains.design_benchmark import run

        return run()
    if pack in ("qmmm", "drug", "synthesis") and name in get_pack(pack).benchmarks:
        from chemflow.domains.m6b_benchmarks import run

        return run(pack, name)
    if (pack, name) not in list_benchmarks():
        raise ValueError(f"Unknown benchmark {pack}/{name}")
    spec = _task(pack, name)
    store = TaskStore("chemflow in-memory benchmark")
    record = store.freeze(spec, "benchmark")
    owner = get_pack("molecular" if name == "h6_ewf_full_bath" else pack)
    plan, controls = owner.base_plan(record)
    selected = [*controls, *plan.references]
    required = plan.references[:]
    if name == "h6_ewf_full_bath":
        reference = next(node for node in plan.nodes if node.id == plan.references[0])
        for bath in ("full", "dmet", "bno"):
            node = Node(id=f"ewf_{bath}", op="ground_energy", inputs=dict(reference.inputs),
                        params={"s": 0, "fragments": "atoms", "bath": bath, "solver": "fci"},
                        refinement="ewf", binding="ewf_embedding")
            plan.nodes.append(node)
            selected.append(node.id)
            if bath == "full":
                required.append(node.id)
                plan.targets.append(node.id)
    if name == "diamond_kpoint_supercell":
        selected, required = plan.targets[:], plan.targets[:]
    result = Executor(load_registry(["molecular", pack])).run_plan(plan, record, store)
    oracle, unit, source = _oracle(name, spec, plan, result)
    gates = {gate.node: gate.model_dump(mode="json") for gate in result.get("gates", [])}
    rows = [{"node": identifier, "value": result[identifier].value.to(unit).value,
             "unit": unit, "abs_error": abs(result[identifier].value.to(unit).value - oracle),
             "relation": result[identifier].relation.value, "gate": gates.get(identifier),
             "diagnostics": result[identifier].diagnostics} for identifier in selected]
    tolerance = 1e-7 if name == "diamond_kpoint_supercell" else 1e-6
    return {"name": name, "pack": pack, "task_hash": record.hash,
            "oracle": {"value": oracle, "unit": unit, "source": source}, "results": rows,
            "passed": bool(required) and all(row["abs_error"] < tolerance and
                       (row["gate"] is None or row["gate"]["passed"])
                       for row in rows if row["node"] in required)}
