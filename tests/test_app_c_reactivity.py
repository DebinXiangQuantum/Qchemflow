"""APP C edge cases, proof boundaries, identities and discovery."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemflow.agent.task import RankingTarget, TaskSpec
from chemflow.domains import reactivity as app

METHANE = ("C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; "
           "H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276")
HYDROGEN = {"atoms": "H 0 0 0", "spin": 1}


def ranking_target():
    return {"kind": "ranking", "items": [{"name": "first", "molecule": HYDROGEN},
            {"name": "second", "molecule": {"atoms": "H 0 0 1", "spin": 1}}],
            "electronic_model": {"basis": "sto-3g"}, "quantity": "energy"}


def record(target=None):
    return SimpleNamespace(spec=TaskSpec.model_validate({
        "question": "Compare energies", "target": target or ranking_target(),
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "decision": "Use controls and an exact screening reference"}))


def test_bond_symmetry_representatives_and_bohr():
    molecule = {"atoms": METHANE}
    assert len(app.representatives(molecule, "C-H")) == 1
    assert app.representatives(molecule, [[0, 4], [0, 2]])[0]["atoms"] == [0, 2]
    entries = app._entries(molecule)
    bohr = {"atoms": "; ".join(symbol + " " + " ".join(map(str, position))
                              for symbol, position in entries), "unit": "bohr"}
    assert app.enumerate_bonds(None, bohr) == app.enumerate_bonds(None, molecule)
    with pytest.raises(ValueError, match="does not exist"):
        app.representatives(molecule, [[1, 2]])


@pytest.mark.parametrize("atoms", [[0, 0], [-1, 0], [0, 8], [1, 2], [0], [False, 1]])
def test_homolysis_rejects_invalid_bonds(atoms):
    with pytest.raises(ValueError):
        app.homolysis(None, {"atoms": METHANE}, atoms)


def test_homolysis_multiple_bond_and_charged_parent():
    result = app.homolysis(None, {"atoms": "N 0 0 0; N 0 0 1.1"}, [0, 1])
    assert [fragment["spin"] for fragment in result["fragments"]] == [1, 1]
    with pytest.raises(ValueError, match="charged"):
        app.homolysis(None, {"atoms": METHANE, "charge": 1}, [0, 1])


def test_ranking_refuses_formula_charge_and_extra_fields():
    target = ranking_target()
    target["items"][1]["molecule"] = {"atoms": "He 0 0 0"}
    with pytest.raises(ValidationError, match="shared formula"):
        RankingTarget.model_validate(target)
    target = ranking_target()
    target["items"][1]["molecule"]["charge"] = 1
    with pytest.raises(ValidationError):
        RankingTarget.model_validate(target)
    with pytest.raises(ValidationError):
        RankingTarget.model_validate({**ranking_target(), "unapproved": True})


def test_fci_determinant_cap_checked_before_solve(monkeypatch):
    monkeypatch.setattr(app, "FCI_MAX_DETERMINANTS", 1)
    with pytest.raises(ValueError, match="determinant cap"):
        app._solve({"atoms": METHANE}, {"basis": "sto-3g"}, "fci")


def test_mmff_conformers_deterministic_and_sorted():
    arguments = {"molecule_smiles": "CCCC", "level": {"method": "mmff"}, "count": 8}
    result = app.conformer_search(None, **arguments)
    assert result == app.conformer_search(None, **arguments)
    energies = [item["energy_hartree"] for item in result["conformers"]]
    assert len(energies) >= 2 and energies == sorted(energies)
