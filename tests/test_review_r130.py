"""Reviewer tests for R130 (written by Claude; implementers must not edit).

User requirement (2026-10-14): "there contains some state vector simulator. Please ensure that
the simulator is only used for verification, and cannot be used for the algorithm deployed in
hybrid quantum-classical nodes."

What the reviewer found:
- `vqe_statevector` and `adapt_statevector` compute exact energies from a simulated state, and
  were offered as candidate templates. glm-5.3 proposed 13–17 VQE candidates per run.
- The exact `expectation` op and the `ideal_energy_projective_sampler` produce results no
  hardware can.
- Only `sample_configurations` has a QPU implementation (lqcloud MQ02), and SQD uses it.

Contract:
- `ImplementationSpec` gains `execution_role`, one of "classical", "qpu", "emulator" or
  "verification_only", and `emulates` (an op name or None).
  - cpu_numeric and gpu_numeric default to "classical"; qpu means "qpu".
  - Every quantum_simulator or fault_tolerant_projection implementation must declare
    "emulator" or "verification_only".
  - An emulator stands in for an op that has a QPU implementation in
    `chemflow.backends.QPU_CATALOG` (op → QPU implementation ids). Its `emulates` names that
    op, and its observable semantics match the QPU's (samples of a prepared circuit).
- `check(program, registry, task=None, purpose=...)`, with purpose "screen" (the default),
  "deploy" or "production":
  - "deploy": CF-E140 for every verification-only binding in the dependency closure of a
    target that is not one of the plan's references. References, and the verification
    graphs behind certificates, falsification, validity maps and `algorithm.verify`, may use
    any simulator.
  - "production": additionally CF-E140 for every emulator binding, since production quantum
    steps must be bound to QPU implementations.
- `plan.candidates` refuses templates with no deployable implementation (vqe today). The
  error says the statevector implementation is verification-only. `design.submit` and the
  orchestrator's candidate path check with purpose "deploy". `run` at scale "production"
  checks with purpose "production".
- The scripted designer proposes only deployable candidates.
- Candidate items gain `deployable` and `deployment` (node → QPU implementation id from the
  catalog, for each emulated quantum step). The critic never escalates a candidate whose
  `deployable` is false.
"""
from __future__ import annotations

import json
from argparse import Namespace

VERIFICATION_ONLY = ("vqe_statevector", "adapt_statevector", "expectation",
                     "ideal_energy_projective_sampler")


def test_r130_scripted_candidates_are_deployable(tmp_path) -> None:
    from chemflow.backends import QPU_CATALOG
    from chemflow.eval import runner

    out = tmp_path / "out"
    runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="scripted",
                         model=None, roles="designer", jobs=1, out=out))
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    candidates = [item for entry in entries if entry["state"] == "CRITIQUE"
                  for item in entry["observation"]["candidates"]]
    assert candidates
    assert {item["method"] for item in candidates} <= {"sci", "sqd"}
    for item in candidates:
        assert item["deployable"] is True, item["id"]
        if item["method"] == "sqd":
            assert item["deployment"] and all(
                implementation in QPU_CATALOG["sample_configurations"]
                for implementation in item["deployment"].values())


def test_r130_the_critic_never_escalates_an_undeployable_candidate() -> None:
    from chemflow.agent.critic import review

    candidate = {"id": "c", "method": "sqd", "relation": "variational_upper_bound",
                 "relation_status": "proved", "evidence": True, "gate_passed": True,
                 "error": 1e-6, "cost": {"wall_s": 0.0}, "selection_work": 1}
    control = {"id": "n6", "method": "ccsd_t", "relation": "approximate",
               "relation_status": "proved", "evidence": True, "gate_passed": False,
               "error": 1e-2, "cost": {"wall_s": 1.0}, "selection_work": 10 ** 6}
    assert review([candidate | {"deployable": False}], [control]).decision != "escalate"
    assert review([candidate | {"deployable": True}], [control]).decision == "escalate"
