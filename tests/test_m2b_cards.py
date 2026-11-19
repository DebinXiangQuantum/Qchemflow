from hashlib import sha256
from pathlib import Path

from chemflow.cards import CardEvidence, load_cards, validate_cards
from chemflow.ops.molecular import molecular_registry

ROOT = Path(__file__).resolve().parents[1]


def test_evidence_pins_and_quotes():
    report = validate_cards(molecular_registry(), ROOT)
    assert report.ok
    assert {card.id for card in report.cards} >= {"sqd_lucj", "sci_pyscf"}
    for card in report.cards:
        for evidence in card.evidence:
            path = ROOT / evidence.path
            if path.is_file():
                assert sha256(path.read_bytes()).hexdigest() == evidence.sha256
                if evidence.quote:
                    assert evidence.quote in path.read_text()


def test_missing_and_mismatched_evidence(tmp_path):
    card = load_cards()[0]
    evidence = CardEvidence(path="report.md", date="2026-10-11", outcome="failed",
                            summary="negative result", sha256="0" * 64)
    card = card.model_copy(update={"evidence": [evidence]})
    registry = molecular_registry()
    assert validate_cards(registry, tmp_path, [card]).evidence[0].status == "unavailable"
    (tmp_path / "report.md").write_text("other")
    assert validate_cards(registry, tmp_path, [card]).evidence[0].status == "error"
