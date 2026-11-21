"""Reproducible QCL benchmarks against independent oracles (``chemflow bench``).

Signed-task benchmarks compile the task's QCL screen (classical controls and the exact
reference, no candidates) and compare the reference with a direct PySCF or NumPy oracle.
The QM/MM, ligand and synthesis benchmarks run one QCL primitive each.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

HARTREE_KCAL = 627.5094740631
WATER = [("O", -1.551007, -0.114520, 0.0), ("H", -1.934259, 0.762503, 0.0),
         ("H", -0.599677, 0.040712, 0.0), ("O", 1.350625, 0.111469, 0.0),
         ("H", 1.680398, -0.373741, -0.758561), ("H", 1.680398, -0.373741, 0.758561)]
LIGANDS = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
           "paracetamol": "CC(=O)Nc1ccc(O)cc1", "rhodanine": "O=C1NC(=S)SC1=Cc1ccccc1",
           "quinone": "O=C1C=CC(=O)C=C1"}
BENCHMARKS = (
    ("reactions", "h2_atomization"), ("reactions", "h3_exchange_barrier"),
    ("materials", "hubbard_dimer"), ("materials", "hubbard_ring6"),
    ("materials", "h6_ewf_full_bath"), ("materials", "diamond_kpoint_supercell"),
    ("qmmm", "water_dimer_qmmm"), ("drug", "ligand_funnel"), ("drug", "h2_dimer_cp"),
    ("synthesis", "paracetamol_routes"),
)


def list_benchmarks() -> list[tuple[str, str]]:
    return list(BENCHMARKS)


def _species(name, atoms, coefficient, electrons, orbitals, spin=0, role="minimum"):
    return {"name": name, "molecule": {"atoms": atoms, "spin": spin},
            "coefficient": coefficient, "active": {"n_elec": electrons, "n_orb": orbitals},
            "sector": {"n_alpha": (electrons + spin) // 2,
                       "n_beta": (electrons - spin) // 2, "n_orb": orbitals},
            "target_spin": spin / 2, "role": role}


def _task(name: str) -> Any:
    from chemflow.agent.task import TaskSpec

    if name in ("h2_atomization", "h3_exchange_barrier"):
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
    elif name == "h2_dimer_cp":
        target = {"kind": "interaction_energy", "electronic_model": {"basis": "6-31g"},
                  "complex": {"atoms": "H 0 0 0; H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"},
                  "fragments": [[0, 1], [2, 3]], "counterpoise": True}
    else:
        target = {"kind": "periodic_energy", "system": {
            "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
            "atoms": "C 0 0 0; C 0.8917 0.8917 0.8917", "unit": "angstrom",
            "basis": "gth-szv", "pseudo": "gth-pade"}, "kmesh": [1, 1, 2]}
    return TaskSpec.model_validate({"question": name, "target": target,
        "gates": [] if target["kind"] == "periodic_energy" else [
            {"kind": "abs_error_vs_exact", "threshold": 1e-6, "unit": "hartree"}],
        "budget": {"wall_s": 600}, "allowed_effects": ["pure"], "decision": "benchmark"})


def _hubbard_oracle(size: int, hopping: float, repulsion: float, ring: bool) -> float:
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


def _oracle(name: str, spec: Any) -> tuple[float, str, str]:
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
    if name == "h2_dimer_cp":
        energies = []
        for atoms in (spec.target.complex["atoms"],
                      "H 0 0 0; H 0 0 0.74; ghost-H 0 0 3.0; ghost-H 0 0 3.74",
                      "ghost-H 0 0 0; ghost-H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"):
            reference = scf.RHF(gto.M(atom=atoms, basis="6-31g", verbose=0)).run(conv_tol=1e-12)
            energies.append(fci.FCI(reference).kernel()[0])
        return (float(energies[0] - sum(energies[1:])), "hartree",
                "Direct PySCF exact FCI with explicit ghost-H atoms")
    energy = 0.
    for species in spec.target.species:
        molecule = species.molecule
        mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                    spin=molecule.get("spin", 0), charge=molecule.get("charge", 0),
                    basis=spec.target.electronic_model["basis"], verbose=0)
        mf = (scf.UHF if mol.spin else scf.RHF)(mol).run()
        value = mf.e_tot if mol.nelectron == 1 else fci.FCI(mf).kernel()[0]
        energy += species.coefficient * value
    return float(energy), "hartree", "Direct PySCF FCI/UHF at the signed species geometries"


def _in(value: Any, unit: str) -> float:
    from chemflow.agent.screen import HARTREE_EV

    if value.unit == unit:
        return float(value.value)
    if (value.unit, unit) == ("hartree", "eV"):
        return float(value.value) * HARTREE_EV
    if (value.unit, unit) == ("eV", "hartree"):
        return float(value.value) / HARTREE_EV
    raise ValueError(f"Cannot compare {value.unit} with {unit}")


def _screen(name: str) -> dict:
    """Run a signed task's QCL screen (controls and exact reference) against its oracle."""
    import qcl
    from chemflow.agent import screen as screens

    spec = _task(name)
    built = screens.build(spec, qcl.Cluster.local(), proposed=(), simulator=False)
    plan = built.workflow.compile(*built.args)
    outputs = plan.run()
    if not isinstance(outputs, dict):
        outputs = {built.reference or "model": outputs}
    relations = plan.relations()
    oracle, unit, source = _oracle(name, spec)
    rows = []
    for output, value in outputs.items():
        energy = value[0] if isinstance(value, tuple) else value
        number = _in(energy, unit)
        rows.append({"node": output, "value": number, "unit": unit,
                     "abs_error": abs(number - oracle),
                     "relation": (relations.get(output) if isinstance(relations, dict)
                                  else relations)})
    reference = built.reference if built.reference in outputs else next(iter(outputs))
    tolerance = 1e-7 if name == "diamond_kpoint_supercell" else 1e-6
    required = [row for row in rows if row["node"] == reference]
    return {"oracle": {"value": oracle, "unit": unit, "source": source}, "results": rows,
            "plan_hash": plan.hash,
            "passed": bool(required) and all(row["abs_error"] < tolerance for row in required)}


def _ewf() -> dict:
    """EWF with a full bath and FCI fragments reproduces full FCI (H6 chain, STO-3G)."""
    from pyscf import fci, gto, scf

    import qcl

    atoms = "; ".join(f"H 0 0 {index}" for index in range(6))
    baths = ("full", "dmet", "bno")

    @qcl.workflow
    def ewf(mol: Any, cpu: Any) -> Any:
        hf = qcl.scf.hf(mol).to(cpu)
        space = qcl.active_space(hf, qcl.Active(6, 6)).to(cpu)
        operator, sector = space.hamiltonian, space.sector
        energies = {bath: qcl.embedding.ewf(operator, sector, bath=bath).to(cpu)
                    for bath in baths}
        energies["fci"] = qcl.fci.solve(operator, sector).to(cpu).energy
        return energies

    cpu = qcl.Cluster.local().cpus[0]
    plan = ewf.compile(qcl.Molecule(atoms, "sto-3g"), cpu)
    outputs = plan.run()
    relations = plan.relations()
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    oracle = float(fci.FCI(scf.RHF(mol).run()).kernel()[0])
    rows = [{"node": name, "value": float(value.value), "unit": "hartree",
             "abs_error": abs(float(value.value) - oracle), "relation": relations[name]}
            for name, value in outputs.items()]
    passed = all(row["abs_error"] < 1e-6 for row in rows if row["node"] in ("full", "fci"))
    return {"oracle": {"value": oracle, "unit": "hartree", "source": "Direct PySCF full FCI"},
            "results": rows, "passed": passed}


def _water_dimer() -> dict:
    from pyscf import gto, qmmm, scf

    import qcl

    lines = []
    for index, (element, xpos, ypos, zpos) in enumerate(WATER):
        atom = ("O", "H1", "H2")[index % 3]
        lines.append(f"HETATM{index + 1:5d} {atom:<4} HOH A{index // 3 + 1:4d}    "
                     f"{xpos:8.3f}{ypos:8.3f}{zpos:8.3f}  1.00  0.00          {element:>2}")
    positions = np.array([atom[1:] for atom in WATER]) * 0.1

    @qcl.workflow
    def water(cpu: Any) -> Any:
        return qcl.qmmm.energy("\n".join(lines) + "\nEND\n", positions.tolist(), [0, 1, 2],
                               forcefields=("amber14/tip3p.xml",), basis="sto-3g",
                               method="rhf").to(cpu)

    plan = water.compile(qcl.Cluster.local().cpus[0])
    value = float(plan.run().value)
    reference = qmmm.mm_charge(scf.RHF(gto.M(atom=WATER[:3], basis="sto-3g", verbose=0)),
        [atom[1:] for atom in WATER[3:]], [-0.834, 0.417, 0.417],
        unit="angstrom").run(conv_tol=1e-11)
    ratio = 0.31507524065751241 / np.linalg.norm(positions[0] - positions[3])
    lennard_jones = 4 * 0.635968 / 2625.4996394799 * (ratio ** 12 - ratio ** 6)
    oracle = float(reference.e_tot + lennard_jones)
    return {"oracle": {"value": oracle, "unit": "hartree",
                       "source": "Independent embedded PySCF RHF + analytic TIP3P O-O LJ"},
            "results": [{"value": value, "relation": plan.relations(),
                         "abs_error": abs(value - oracle)}],
            "passed": abs(value - oracle) < 1e-7}


def _funnel() -> dict:
    import qcl

    @qcl.workflow
    def funnel(cpu: Any) -> Any:
        return qcl.drug.filter(LIGANDS, rules=("lipinski", "pains")).to(cpu)

    result = funnel.compile(qcl.Cluster.local().cpus[0]).run()
    expected = {"rhodanine", "quinone"}
    return {"oracle": {"flagged": sorted(expected),
                       "source": "Literature PAINS families: rhodanines and quinones"},
            "results": result.data,
            "passed": set(result.data["flagged"]) == expected and
            set(result.data["passed"]) == {"aspirin", "ibuprofen", "paracetamol"}}


def _paracetamol() -> dict:
    import qcl

    @qcl.workflow
    def routes(cpu: Any) -> Any:
        return qcl.synthesis.routes(LIGANDS["paracetamol"], stock="zinc", iterations=100,
                                    time_limit_s=120).to(cpu)

    result = routes.compile(qcl.Cluster.local().cpus[0]).run(timeout_s=600).data
    leaves = {"CC(=O)OC(C)=O", "Nc1ccc(O)cc1"}
    return {"oracle": {"leaves": sorted(leaves), "source": "Textbook aminophenol acetylation"},
            "results": {"solved": result["solved"], "route_count": len(result["routes"]),
                        "top_route": result["routes"][0] if result["routes"] else None,
                        "inventory": result["inventory"]},
            "passed": result["solved"] and leaves <= set(result["routes"][0]["leaves"])}


def run_benchmark(pack: str, name: str) -> dict:
    if (pack, name) not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {pack}/{name}")
    runner = {"h6_ewf_full_bath": _ewf, "water_dimer_qmmm": _water_dimer,
              "ligand_funnel": _funnel, "paracetamol_routes": _paracetamol}.get(name)
    report = runner() if runner is not None else _screen(name)
    return {"name": name, "pack": pack, **report}
