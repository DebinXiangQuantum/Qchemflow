"""Independent small-system and literature-label oracles for the M6b adapters."""

import numpy as np

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.domains import get_pack, load_registry
from chemflow.runtime import Executor

WATER = [("O", -1.551007, -0.114520, 0.0), ("H", -1.934259, 0.762503, 0.0),
         ("H", -0.599677, 0.040712, 0.0), ("O", 1.350625, 0.111469, 0.0),
         ("H", 1.680398, -0.373741, -0.758561), ("H", 1.680398, -0.373741, 0.758561)]
LIGANDS = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
           "paracetamol": "CC(=O)Nc1ccc(O)cc1", "rhodanine": "O=C1NC(=S)SC1=Cc1ccccc1",
           "quinone": "O=C1C=CC(=O)C=C1"}


def water_dimer():
    from pyscf import gto, qmmm, scf

    lines = []
    for index, (element, xpos, ypos, zpos) in enumerate(WATER):
        name = ("O", "H1", "H2")[index % 3]
        lines.append(f"HETATM{index + 1:5d} {name:<4} HOH A{index // 3 + 1:4d}    "
                     f"{xpos:8.3f}{ypos:8.3f}{zpos:8.3f}  1.00  0.00          {element:>2}")
    positions = np.array([atom[1:] for atom in WATER]) * 0.1
    with build.program("water-dimer-qmmm") as builder:
        system = builder.node("mm_system", pdb="\n".join(lines) + "\nEND\n",
                              forcefields=["amber14/tip3p.xml"], nonbonded="NoCutoff")
        node = builder.node("qmmm_energy", system=system, positions_nm=positions.tolist(),
                            qm_atoms=[0, 1, 2], basis="sto-3g", method="rhf")
    result = Executor(load_registry(["qmmm"])).run(builder.to_ir())[node.id]
    reference = qmmm.mm_charge(scf.RHF(gto.M(atom=WATER[:3], basis="sto-3g", verbose=0)),
        [atom[1:] for atom in WATER[3:]], [-0.834, 0.417, 0.417], unit="angstrom").run(conv_tol=1e-11)
    ratio = 0.31507524065751241 / np.linalg.norm(positions[0] - positions[3])
    lennard_jones = 4 * 0.635968 / 2625.4996394799 * (ratio ** 12 - ratio ** 6)
    oracle = float(reference.e_tot + lennard_jones)
    value = result.value.value
    return {"oracle": {"value": oracle, "unit": "hartree",
                       "source": "Independent embedded PySCF RHF + analytic TIP3P O-O LJ"},
            "results": [{"value": value, "relation": result.relation.value,
                         "abs_error": abs(value - oracle)}], "passed": abs(value - oracle) < 1e-7}


def interaction():
    from pyscf import fci, gto, scf

    target = {"kind": "interaction_energy", "electronic_model": {"basis": "6-31g"},
        "complex": {"atoms": "H 0 0 0; H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"},
        "fragments": [[0, 1], [2, 3]], "counterpoise": True}
    spec = TaskSpec.model_validate({"question": "H2 dimer CP", "target": target,
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 1e-9, "unit": "hartree"}],
        "decision": "benchmark"})
    store = TaskStore("m6b-benchmark")
    record = store.freeze(spec, "benchmark")
    plan, controls = get_pack("drug").base_plan(record)
    results = Executor(load_registry(["drug"])).run_plan(plan, record, store)
    exact, hartree_fock = [], []
    for atoms in (target["complex"]["atoms"],
                  "H 0 0 0; H 0 0 0.74; ghost-H 0 0 3.0; ghost-H 0 0 3.74",
                  "ghost-H 0 0 0; ghost-H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"):
        reference = scf.RHF(gto.M(atom=atoms, basis="6-31g", verbose=0)).run(conv_tol=1e-12)
        exact.append(fci.FCI(reference).kernel()[0])
        hartree_fock.append(reference.e_tot)
    oracle = float(exact[0] - sum(exact[1:]))
    hf_oracle = float(hartree_fock[0] - sum(hartree_fock[1:]))
    rows = [{"method": method, "value": results[identifier].value.value,
             "kcal_mol": results[identifier].value.value * 627.5094740631,
             "relation": results[identifier].relation.value}
            for method, identifier in zip(("hf", "mp2", "ccsd", "ccsd_t", "fci"),
                                          [*controls, *plan.references])]
    return {"task_hash": record.hash,
            "oracle": {"value": oracle, "unit": "hartree", "hf": hf_oracle,
                       "source": "Direct PySCF exact FCI and RHF with explicit ghost-H atoms"},
            "results": rows, "passed": abs(rows[-1]["value"] - oracle) < 1e-9 and
            abs(rows[0]["value"] - hf_oracle) < 1e-9}


def funnel():
    with build.program("literature-funnel") as builder:
        library = builder.node("ligand_library", ligands=LIGANDS)
        node = builder.node("filter_ligands", library=library, rules=["lipinski", "pains"])
    result = Executor(load_registry(["drug"])).run(builder.to_ir())[node.id]
    expected = {"rhodanine", "quinone"}
    return {"oracle": {"flagged": sorted(expected),
                       "source": "Literature PAINS families: rhodanines and quinones"},
            "results": result.model_dump(), "passed": set(result.flagged) == expected and
            set(result.passed) == {"aspirin", "ibuprofen", "paracetamol"}}


def paracetamol():
    with build.program("paracetamol") as builder:
        node = builder.node("search_routes", target=LIGANDS["paracetamol"], stock="zinc",
                            iterations=100, time_limit_s=120)
    result = Executor(load_registry(["synthesis"])).run(builder.to_ir())[node.id]
    leaves = {"CC(=O)OC(C)=O", "Nc1ccc(O)cc1"}
    return {"oracle": {"leaves": sorted(leaves), "source": "Textbook aminophenol acetylation"},
            "results": {"solved": result.solved, "route_count": len(result.routes),
                        "top_route": result.routes[0] if result.routes else None,
                        "inventory": result.inventory},
            "passed": result.solved and leaves <= set(result.routes[0]["leaves"])}


def run(pack, name):
    benchmark = {"water_dimer_qmmm": water_dimer, "h2_dimer_cp": interaction,
                 "ligand_funnel": funnel, "paracetamol_routes": paracetamol}[name]
    return {"name": name, "pack": pack, **benchmark()}
