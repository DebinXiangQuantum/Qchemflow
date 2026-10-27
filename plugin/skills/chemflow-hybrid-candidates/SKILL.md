---
name: chemflow-hybrid-candidates
description: Compare classical controls with SQD, SCI and VQE candidates under frozen budgets.
---
Read `cards_search` and `cards_get` validity and cost limits. Build candidates
with `plan_candidates`, inspect `estimate_cost`, run `check`, and screen using
`run` only within the signed budget. Use `results_query` for comparable evidence.
CPU, GPU, simulators and QPU all consume the signed task resource budget.
The screen/production scale is informational only.
Pass capped `text_sha256` objects back as plan parameter values to reuse exact frozen text.

When no registered refinement fits, or all applicable candidates fail their gates
or budget, inspect `algorithm_list` and assemble a new composition of primitives.
Use `algorithm_verify` against random family instances and the frozen target's
exact reference, then `algorithm_register` only with a passing server report id.
Registration is experimental: retain the evidence id, rerun task gates, and keep
the classical control comparison. See `chemflow-design-loop` for the workflow.
