"""Reviewer tests for APP batch C: bonds, BDE, metabolic sites, ranking, conformers (written by
Claude; implementers must not edit).

Applications covered:
- 键解离能 (bond dissociation energies);
- 代谢位点预测 (metabolic sites);
- 多分子稳定性排序 (stability ranking);
- 多构象能量排序 (conformer ranking).

Plan: docs/research/09-application-operators.md, batch C.

Contract:
- **Bond operators.**
  - `enumerate_bonds(molecule; elements=None)`: RDKit bond perception
    (`rdDetermineBonds`, with the molecule's charge). Returns `bonds` as
    [{atoms, elements, symmetry_class}], where symmetry_class is from RDKit canonical ranks
    so that equivalent bonds share a class. Optionally filtered by an element pair such as
    "C-H".
  - `homolysis(molecule; atoms)`: two radical fragments at the parent geometry (rigid),
    each neutral with the lowest multiplicity for its electron count. A lone H atom is a
    doublet.
- **Target kind `bond_dissociation`.**
  - Schema: `{kind, molecule, electronic_model, bonds: "all" | "<A>-<B>" | [[i, j], ...],
    fragments: "rigid"}`.
  - Value: one electronic BDE per symmetry class, E(frag1) + E(frag2) − E(parent), in
    canonical class order (sorted by the smallest atom index in each class).
  - The screening reference is FCI. The eval harness scores `expected.values_hartree` per
    class. Category "bond" is added.
- **`site_reactivity(molecule, model; level)`.** Per hydrogen: its C–H BDE (as above, at
  `level`), SASA in Å² (FreeSASA with RDKit radii, probe 1.4 Å), and
  `score = sasa_fraction / (1 + exp((bde_kcal − 90) / 3))`. The output names `top_site`.
  The evidence is empirical (heuristic), and is labelled that way.
- **Target kind `ranking`.**
  - Schema: `{kind, items: [{name, molecule}], electronic_model, quantity: "energy"}`. The
    items share a formula, i.e. isomers or conformers.
  - Output: the items ordered by energy, plus `certified_pairs`. A pair is certified only
    when the energy gap exceeds the sum of both items' error bounds; the bounds come from
    certificates, validity maps or the task's bar.
  - `expected.ranking` (a list of names) is scored by exact order.
- **`boltzmann_weights(energies; temperature_K)`** returns normalised populations.
- **`conformer_search(molecule_smiles; level, count, rmsd)`.** ETKDG conformers, optimised
  at `level` (gfn2 via tblite, or mmff), deduplicated by heavy-atom RMSD, and returned
  sorted by energy.
"""
from __future__ import annotations

import math
from argparse import Namespace

import numpy as np
import pytest
import yaml

ETHANOL = ("C -0.0007 -0.0080 0.0000; C 1.5150 0.0107 0.0000; O 1.9668 1.3577 0.0000; "
           "H -0.3846 -1.0321 0.0000; H -0.3958 0.4987 0.8811; H -0.3958 0.4987 -0.8811; "
           "H 1.8974 -0.5038 0.8798; H 1.8974 -0.5038 -0.8798; H 2.9336 1.3423 0.0000")
METHANE = ("C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; "
           "H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276")


def _execute(builder_fn):
    from chemflow import build
    from chemflow.domains import load_registry, pack_names
    from chemflow.runtime import Executor

    with build.program("c") as builder:
        node = builder_fn(builder)
    return Executor(load_registry(pack_names())).run(builder.to_ir())[node.id]


def _get(result, key):
    return result[key] if isinstance(result, dict) else getattr(result, key)


def test_c1_bond_enumeration_with_symmetry_classes() -> None:
    result = _execute(lambda b: b.node("enumerate_bonds", molecule=b.molecule(ETHANOL)))
    bonds = _get(result, "bonds")
    assert len(bonds) == 8
    ch = [bond for bond in bonds if sorted(bond["elements"]) == ["C", "H"]]
    assert len(ch) == 5 and len({bond["symmetry_class"] for bond in ch}) == 2
    filtered = _get(_execute(lambda b: b.node("enumerate_bonds", molecule=b.molecule(ETHANOL),
                                              elements="O-H")), "bonds")
    assert [sorted(bond["atoms"]) for bond in filtered] == [[2, 8]]


def test_c1_homolysis_gives_radical_fragments() -> None:
    result = _execute(lambda b: b.node("homolysis", molecule=b.molecule(METHANE), atoms=[0, 1]))
    fragments = _get(result, "fragments")
    spins = sorted(fragment["spin"] for fragment in fragments)
    assert spins == [1, 1]  # 2S: CH3 doublet + H doublet
    assert sorted(len(fragment["atoms"].split(";")) for fragment in fragments) == [1, 4]


def _fci(atoms, spin):
    from pyscf import fci, gto, scf

    mol = gto.M(atom=atoms, basis="sto-3g", spin=spin, verbose=0)
    mf = (scf.ROHF(mol) if spin else scf.RHF(mol)).run()
    return fci.FCI(mf).kernel()[0]


def _run_single(tmp_path, task):
    from chemflow.eval import runner

    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "app-c", "tasks": [task]}))
    results = runner.run(Namespace(suite=None, suite_file=path, tasks=None, policy="scripted",
                                   model=None, roles="designer", jobs=1, out=tmp_path / "out"))
    return results["rows"][0]


def _task(identifier, category, target, expected, threshold=0.0016):
    return {"id": identifier, "category": category, "spec": {
        "question": f"Evaluate {identifier}", "target": target,
        "gates": [{"kind": "abs_error_vs_exact", "threshold": threshold, "unit": "hartree"},
                  {"kind": "relation_at_least", "relation": "approximate"}],
        "budget": {"wall_s": 1800}, "allowed_effects": ["pure"],
        "decision": "Choose a supported method without changing the signed target"},
        "expected": expected,
        "reference_source": "Reviewer PySCF FCI, independent of ChemFlow"}


def test_c2_methane_ch_bde_through_the_agent(tmp_path) -> None:
    parent = _fci(METHANE, 0)
    methyl = _fci("; ".join(METHANE.split("; ")[:1] + METHANE.split("; ")[2:]), 1)
    hydrogen = _fci("H 0.6276 0.6276 0.6276", 1)
    target = {"kind": "bond_dissociation", "molecule": {"atoms": METHANE},
              "electronic_model": {"basis": "sto-3g"}, "bonds": "C-H", "fragments": "rigid"}
    row = _run_single(tmp_path, _task("methane-bde", "bond", target,
                                      {"values_hartree": [float(methyl + hydrogen - parent)]}))
    assert row["correct"] and row["completed"], row


def test_c3_site_reactivity_reports_bde_sasa_and_the_formula() -> None:
    result = _execute(lambda b: b.node(
        "site_reactivity", molecule=b.molecule(ETHANOL), model=b.electronic_model(basis="sto-3g"),
        level={"method": "hf", "functional": None, "dispersion": None}))
    sites = _get(result, "sites")
    hydrogens = [site for site in sites]
    assert {site["atom"] for site in hydrogens} == {3, 4, 5, 6, 7}  # C-H hydrogens only
    for site in hydrogens:
        expected = site["sasa_fraction"] / (1 + math.exp((site["bde_kcal"] - 90) / 3))
        assert site["score"] == pytest.approx(expected)
    top = max(hydrogens, key=lambda site: site["score"])
    assert _get(result, "top_site") == top["atom"]
    assert _get(result, "evidence") == "empirical"


def test_c4_isomer_ranking_through_the_agent(tmp_path) -> None:
    hcn, hnc = "C 0 0 0; N 0 0 1.16; H 0 0 -1.07", "C 0 0 0; N 0 0 1.17; H 0 0 2.17"
    energies = {"HCN": _fci(hcn, 0), "HNC": _fci(hnc, 0)}
    order = sorted(energies, key=energies.get)
    target = {"kind": "ranking", "items": [{"name": "HNC", "molecule": {"atoms": hnc}},
                                           {"name": "HCN", "molecule": {"atoms": hcn}}],
              "electronic_model": {"basis": "sto-3g"}, "quantity": "energy"}
    row = _run_single(tmp_path, _task("hcn-hnc-ranking", "ranking", target,
                                      {"ranking": order}))
    assert row["correct"], row


def test_c4_boltzmann_weights() -> None:
    energies = [0.0, 0.001, 0.003]
    result = _execute(lambda b: b.node("boltzmann_weights", energies=energies,
                                       temperature_K=298.15))
    beta = 4.3597447222071e-18 / (1.380649e-23 * 298.15)
    raw = np.exp(-beta * np.array(energies))
    assert np.allclose(_get(result, "weights"), raw / raw.sum(), atol=1e-12)


def test_c4_butane_conformer_search_finds_anti_first() -> None:
    result = _execute(lambda b: b.node("conformer_search", molecule_smiles="CCCC",
                                       level={"method": "gfn2", "functional": None,
                                              "dispersion": None}, count=20, rmsd=0.3))
    conformers = _get(result, "conformers")
    assert len(conformers) >= 2
    energies = [item["energy_hartree"] for item in conformers]
    assert energies == sorted(energies)
    points = np.array([[float(value) for value in atom.split()[1:]]
                       for atom in conformers[0]["atoms"].split(";")])
    b0, b1, b2 = points[0] - points[1], points[2] - points[1], points[3] - points[2]
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    dihedral = abs(math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
    assert dihedral > 150  # anti is the lowest butane conformer
