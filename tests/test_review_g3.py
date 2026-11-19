"""Reviewer regression tests for G3, molecule names to structures (written by Claude;
implementers must not edit).

From the ChemGraph comparison (user priority 2026-10-13). Contract: drug-pack operator
`ligands_from_names` (param `names`, a list) returns a `LigandLibrary` whose entries carry the
RDKit-canonical SMILES and InChIKey plus provenance: `source` ("pubchem"), `cid`,
`retrieved_at`, `cached`. It resolves names through the module function
`chemflow.domains.drug.fetch_pubchem(name) -> {"CID", "SMILES", "InChIKey"}` (PubChem PUG
REST) and caches resolutions in the JSON file named by `CHEMFLOW_NAME_CACHE`, so replays are
offline and deterministic.
- New effect `network`: a node that may reach an external service needs `network` in the
  signed task's `allowed_effects`, otherwise it is refused before anything is fetched.
- A cache miss without connectivity is CF-E080, naming the molecule.
- A PubChem InChIKey that disagrees with the InChIKey of its own SMILES is refused (CF-E060).
- The library feeds the existing drug operators (filters, descriptors, conformers, docking).
"""
from __future__ import annotations

import json

import pytest

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.runtime import Executor

ASPIRIN = {"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
           "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}


def _registry():
    from chemflow.domains import load_registry

    return load_registry(["molecular", "drug"])


def _program(names=("aspirin",)):
    with build.program("names") as builder:
        library = builder.node("ligands_from_names", names=list(names))
        verdict = builder.node("filter_ligands", library=library, rules=["lipinski"])
    return builder.to_ir(), library.id, verdict.id


@pytest.fixture
def pubchem(monkeypatch, tmp_path):
    from chemflow.domains import drug

    calls = []
    monkeypatch.setenv("CHEMFLOW_NAME_CACHE", str(tmp_path / "names.json"))

    def fetch(name):
        calls.append(name)
        if name != "aspirin":
            raise ConnectionError("network unreachable")
        return dict(ASPIRIN)

    monkeypatch.setattr(drug, "fetch_pubchem", fetch)
    return calls, tmp_path / "names.json"


def test_a_name_resolves_to_a_canonical_structure_with_provenance(pubchem) -> None:
    from rdkit import Chem

    calls, cache = pubchem
    program, library, verdict = _program()
    computed = Executor(_registry()).run(program)
    entry = computed[library].ligands["aspirin"]
    assert entry["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN["SMILES"]))
    assert entry["inchikey"] == ASPIRIN["InChIKey"]
    assert entry["source"] == "pubchem" and entry["cid"] == 2244 and entry["retrieved_at"]
    assert entry["cached"] is False and calls == ["aspirin"]
    assert "aspirin" in computed[verdict].passed  # feeds the existing drug operators
    assert json.loads(cache.read_text())["aspirin"]["CID"] == 2244


def test_replays_use_the_cache_without_the_network(pubchem, monkeypatch) -> None:
    from chemflow.domains import drug

    calls, _ = pubchem
    program, library, _ = _program()
    Executor(_registry()).run(program)
    monkeypatch.setattr(drug, "fetch_pubchem",
                        lambda name: (_ for _ in ()).throw(ConnectionError("offline")))
    replay = Executor(_registry()).run(_program()[0])
    assert replay[library].ligands["aspirin"]["cached"] is True and calls == ["aspirin"]


def test_an_unresolvable_name_offline_is_refused_by_name(pubchem) -> None:
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run(_program(("unobtainium",))[0])
    assert "CF-E080" in str(info.value) and "unobtainium" in str(info.value)


def test_a_record_whose_inchikey_contradicts_its_smiles_is_refused(pubchem, monkeypatch) -> None:
    from chemflow.domains import drug

    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: {
        **ASPIRIN, "InChIKey": "RZVAJINKPMORJF-UHFFFAOYSA-N"})  # paracetamol's key
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run(_program()[0])
    assert "CF-E060" in str(info.value)


def test_network_access_needs_the_tasks_permission(pubchem) -> None:
    calls, _ = pubchem
    spec = TaskSpec.model_validate({
        "question": "aspirin filter", "target": {
            "kind": "molecular_ground_energy", "operator": "ground_energy",
            "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"}, "electronic_model": {"basis": "sto-3g"},
            "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
        "gates": [{"kind": "relation_at_least", "relation": "approximate"}],
        "budget": {"wall_s": 60}, "allowed_effects": ["pure"], "decision": "d"})
    store = TaskStore("reviewer")
    record = store.freeze(spec, "reviewer")
    program, library, _ = _program()
    program.targets = [library]
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run_plan(program, record, store)
    message = str(info.value).lower()
    assert "cf-e120" in message and ("network" in message or "effect" in message)
    assert calls == []  # refused for the missing permission, before anything was fetched
