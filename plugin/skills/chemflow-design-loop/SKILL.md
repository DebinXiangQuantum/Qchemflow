---
name: chemflow-design-loop
description: Design and screen task-bound ChemFlow plans through role-scoped MCP tools.
---
To start the complete loop from a human-approved task, use
`chemflow agent run --task task.json --public-key /trusted/public.pem --out /tmp/chemflow-run`.
This verifies the original signature and runs every role scripted by default.
Keep the same output directory for reruns so its task ledger accumulates costs.
Read `summary.json`, `report.md` and `journal.jsonl` for the decision and evidence.
Live `--policy anthropic|openai` requires `CHEMFLOW_LIVE_LLM=1` and the provider API
key; OpenAI also requires `--model`. `--roles designer,critic` selects LLM roles;
all other roles stay scripted. Never sign or change the task to bypass a refusal.

Start with `registry_search` and `cards_search`. Create a handle with `plan_new`, then
use `plan_base`, `plan_candidates`, or `plan_add_node`, `plan_refine`, and `plan_bind`.
Inspect with `plan_render` and `plan_diff`; call `check` before `run` at screen scale.
Query `results_query` and `results_get` for evidence. Do not alter task gates or
claim unsupported accuracy. Humans alone freeze tasks and resource budgets.

Design a composition of algorithm primitives only when no registered refinement fits,
or all applicable candidates fail their gates or budget. Inspect `algorithm_list`
before inventing another composition. Use `algorithm_verify` (designer/orchestrator)
with the plan handle, target node, instance count and seed. The accuracy bar is the
frozen task's matching error gate, never a tool input or an agent choice. L1 gates
are dimensionless; energy bars use energy units. A task without an error gate
cannot verify a design, and plans must not contain gates. The server uses the
frozen task target, not the candidate's observable parameters, to construct
the exact reference on random family instances. Keep failing evidence as negative
results; never assert a report passed yourself. `algorithm_register` accepts only a
passing server report id and a unique name. Registered designs remain experimental,
with evidence and recorded accuracy attached. A stricter task bar requires new
verification and registration before execution. Verification is not promotion to
stable or a waiver of task
gates, exact-reference comparisons, classical controls, or resource budgets.
