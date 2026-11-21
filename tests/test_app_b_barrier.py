from argparse import Namespace

import pytest
import yaml

from chemflow.agent.task import ReactionBarrierTarget, TaskSpec
from chemflow.domains.paths import connected_ts
from chemflow.eval.runner import run
from chemflow.types import ElectronicModel

HCN = {"atoms": "C 0 0 0; N 0 0 1.16; H 0 0 -1.07"}
HNC = {"atoms": "C 0 0 0; N 0 0 1.17; H 0 0 2.17"}
SADDLE = {"atoms": "C 0 0 0; N 0 0 1.2; H 1.1 0 .5"}


def barrier_target():
    return ReactionBarrierTarget(kind="reaction_barrier", product=HNC,
        electronic_model={"basis": "sto-3g"}, species=[
            {"name": name, "molecule": molecule, "coefficient": coefficient, "role": role,
             "active": {"n_elec": 2, "n_orb": 2},
             "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}}
            for name, molecule, coefficient, role in
            (("reactant", HCN, -1, "minimum"), ("saddle", SADDLE, 1, "ts"))])


def barrier_spec():
    return TaskSpec(question="Verified HCN isomerization barrier", target=barrier_target(),
        gates=[{"kind": "abs_error_vs_exact", "threshold": .1, "unit": "hartree"},
               {"kind": "relation_at_least", "relation": "approximate"}],
        budget={"wall_s": 1800}, allowed_effects=["pure"], decision="Choose a classical control")


def test_barrier_rejects_unconnected_irc(monkeypatch):
    monkeypatch.setattr("chemflow.domains.paths.irc", lambda *args, **kwargs: {
        "ts": SADDLE, "connects": False})
    with pytest.raises(ValueError, match="do not connect"):
        connected_ts(None, SADDLE, ElectronicModel(basis="sto-3g"), HCN, HNC, {"method": "rhf"})


def test_barrier_agent_runs_verified_hf_path(tmp_path):
    path = tmp_path / "barrier.yaml"
    path.write_text(yaml.safe_dump({"name": "barrier-integration", "tasks": [{
        "id": "verified-barrier", "category": "path", "spec": barrier_spec().model_dump(),
        "expected": {"value_hartree": .14},
        "reference_source": "Broad HCN barrier range; integration test, not precision calibration"}]}))
    results = run(Namespace(suite=None, suite_file=path, tasks=None, policy="scripted", model=None,
                            roles="designer", jobs=1, out=tmp_path / "out"))
    row = results["rows"][0]
    assert row["completed"] and row["gate_passed"] and row["relation_ok"], row
    assert row["answer_value"] > 0
