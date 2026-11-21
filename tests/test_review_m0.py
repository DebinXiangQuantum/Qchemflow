"""Reviewer regression tests for M0/M1 (written by Claude; implementers must not edit).

Every program here is built only through the public builder, with no hand-planted
context flags: the checker/runtime has to infer context itself.
"""
import pytest

from chemflow.types import Evidence


def test_certified_evidence_cannot_be_constructed_outside_runtime() -> None:
    with pytest.raises(Exception):
        Evidence(status="certified_bound", artifact_ref="anything")
    with pytest.raises(Exception):
        Evidence(status="certified_bound", artifact_ref="anything", source="implementation")
