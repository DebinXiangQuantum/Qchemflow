import pytest

from chemflow.agent.report import ReportLintError, lint_report, write_report


def test_claims_require_support():
    headline = {"node": "n", "value": "-1 hartree", "relation": "approximate",
                "evidence": "diagnostic_only", "gate_verdict": "FAIL",
                "cost": {"wall_s": 1}, "backend_kind": "quantum_simulator"}
    for claim in ("quantum advantage", "speedup", "certified", "exact", "QPU result"):
        with pytest.raises(ReportLintError):
            lint_report(claim, [headline])
    assert "approximate" in write_report("question", "stop", "no pass", [headline])
