"""APP C edge cases, proof boundaries, identities and discovery."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemflow import build
from chemflow.agent.task import RankingTarget, TaskSpec
from chemflow.check import check
from chemflow.domains import load_registry, pack_names
from chemflow.domains import reactivity as app
from chemflow.runtime import Executor
from chemflow.verify.kernel import check_certificate
from chemflow.verify.prover import prove

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


@pytest.mark.parametrize("energies,temperature", [([], 298), ([float("nan")], 298),
                                                ([0], 0), ([0], -1), ([[0]], 298)])
def test_boltzmann_rejects_invalid_input(energies, temperature):
    with pytest.raises(ValueError):
        app.boltzmann_weights(None, energies, temperature)


def test_boltzmann_offset_invariance_and_underflow():
    expected = app.boltzmann_weights(None, [0, 0.001, 10])["weights"]
    shifted = app.boltzmann_weights(None, [-100, -99.999, -90])["weights"]
    assert np.allclose(expected, shifted, atol=1e-12)
    assert sum(expected) == pytest.approx(1)
    assert app.boltzmann_weights(None, [0, 1], 1e-320)["weights"] == [1., 0.]


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


def test_certified_pairs_strict_sum_of_bars():
    from chemflow.domains._adapters import estimate

    def result(gap):
        inputs = {f"component{index}": estimate(value, "total_electronic_energy", "approximate",
                  {"hamiltonian_id": "test"}, "app_component_energy")
                  for index, value in enumerate([0., gap])}
        return app.energy_collection(None, ranking_target(), 0.001, **inputs).diagnostics

    assert result(0.002)["certified_pairs"] == []
    assert result(0.0021)["certified_pairs"] == [["first", "second"]]
    assert result(0.)["ranking"] == ["first", "second"]


def test_base_plan_identity_reference_and_proofs():
    registry = load_registry(pack_names())
    frozen = record()
    plan, controls = app.base_plan(frozen)
    assert len(controls) == 3 and len(plan.references) == 1
    assert not check(plan, registry)
    result = Executor(registry).run(plan)
    reference = plan.references[0]
    assert result[reference].relation == "equivalent"
    assert result[controls[0]].relation == "approximate"
    certificate = prove(plan, reference, registry=registry, computed=result)
    assert check_certificate(certificate, plan, registry=registry, computed=result).valid
    result[reference].diagnostics["values_hartree"][1] += 0.1
    assert not check_certificate(certificate, plan, registry=registry, computed=result).valid
    assert app.identity(plan, frozen, {}) == []
    changed = plan.model_copy(deep=True)
    changed.nodes[-1].params["bar_hartree"] = 0.1
    assert app.identity(changed, frozen, {})[0].code == "CF-E120"


def test_fci_determinant_cap_checked_before_solve(monkeypatch):
    monkeypatch.setattr(app, "FCI_MAX_DETERMINANTS", 1)
    with pytest.raises(ValueError, match="determinant cap"):
        app._solve({"atoms": METHANE}, {"basis": "sto-3g"}, "fci")


def test_optional_level_none_and_invalid_parameter_checks():
    registry = load_registry(pack_names())
    with build.program("levels") as builder:
        builder.node("conformer_search", molecule_smiles="CCCC", count=2, rmsd=0.3,
                     level={"method": "mmff", "functional": None, "dispersion": None})
    assert not check(builder.to_ir(), registry)
    builder.ir.nodes[-1].params["count"] = -1
    assert any(issue.code == "CF-E080" for issue in check(builder.to_ir(), registry))


def test_cards_and_resources_discoverable():
    from chemflow.cards import load_cards, validate_cards

    registry = load_registry(pack_names())
    cards = load_cards(registry)
    assert any(card.target == "chemflow.ranking@1" for card in cards)
    assert registry.search("homolysis")
    assert validate_cards(registry, Path.cwd(), cards).ok
    for name in ("enumerate_bonds", "homolysis", "site_reactivity", "boltzmann_weights",
                 "conformer_search", "bond_dissociation", "ranking", "app_component_energy"):
        implementation = registry.implementations[name]
        assert implementation.execution_role == "classical"
        assert implementation.resource_model()["qpu_shots"] == 0


def test_geometry_family_updates_nested_components_and_target():
    from chemflow.agent.families import instantiate

    target = ranking_target()
    target["items"][0]["molecule"] = {"atoms": "H 0 0 0; H 0 0 0.74"}
    target["items"][1]["molecule"] = {"atoms": "H 0 0 0; H 0 0 0.80"}
    plan, _ = app.base_plan(record(target))
    scaled = instantiate(plan, "geometry_scale", 1.2)
    assert scaled.nodes[1].params["molecule"] != plan.nodes[1].params["molecule"]
    target = scaled.nodes[-1].params["target"]
    assert target["items"][1]["molecule"] == scaled.nodes[1].params["molecule"]
    assert app._entries(target["items"][1]["molecule"])[1][1][2] * 0.52917721092 == pytest.approx(0.96)
    assert all(node.params["target"] == target for node in scaled.nodes if node.op == "ranking")


def test_mmff_conformers_deterministic_and_sorted():
    arguments = {"molecule_smiles": "CCCC", "level": {"method": "mmff"}, "count": 8}
    result = app.conformer_search(None, **arguments)
    assert result == app.conformer_search(None, **arguments)
    energies = [item["energy_hartree"] for item in result["conformers"]]
    assert len(energies) >= 2 and energies == sorted(energies)
