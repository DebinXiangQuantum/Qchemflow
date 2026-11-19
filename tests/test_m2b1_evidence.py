"""The ADAPT card pins a self-contained benchmark artifact."""

from hashlib import sha256
from pathlib import Path

from chemflow.cards import load_cards


def test_adapt_evidence_is_immutable_benchmark():
    card = next(card for card in load_cards() if card.id == "adapt_vqe")
    evidence = card.evidence[0]
    assert evidence.path == "benchmarks/results/adapt_h4_plateau.json"
    path = Path(__file__).resolve().parents[1] / evidence.path
    assert sha256(path.read_bytes()).hexdigest() == evidence.sha256
    assert evidence.quote in path.read_text()
