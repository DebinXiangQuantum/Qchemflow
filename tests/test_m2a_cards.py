from pathlib import Path

import pytest

from chemflow.cards import CardEvidence, validate_cards
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation


def test_seed_cards():
    report = validate_cards(molecular_registry(), Path(__file__).resolve().parents[1])
    assert report.ok
    cards = report.cards
    assert {card.id for card in cards} == {
        "fci", "hf", "mp2", "ccsd", "ccsd_t", "vqe_uccsd", "vqe_lucj", "adapt_vqe",
        "sqd_lucj", "sci_pyscf"}


def test_invalid_cards(tmp_path):
    registry = molecular_registry()
    cards = validate_cards(registry, tmp_path).cards
    for changes in ({"template": "missing"}, {"relation": TargetRelation.equivalent},
                    {"evidence": [CardEvidence(path="../elsewhere", summary="test",
                                               outcome="failed", date="2026-10-11",
                                               sha256="0" * 64)]}):
        altered = cards[1].model_copy(update=changes)
        with pytest.raises(ValueError):
            validate_cards(registry, tmp_path, [altered])
