# ChemFlow integrity rules

The signed task defines target, gates, effects and budget. Never modify it through
an agent tool. Use the `qcl` MCP tools only. Humans freeze tasks out of band.
All devices consume the task budget; effects describe determinism only.
Ground every report claim in recorded relations, evidence, gate verdicts, cost and
device; `screen_run` returns the report.

For the complete CLI loop, run `chemflow agent run --task task.json
--public-key /trusted/public.pem --out /tmp/chemflow-run` with a human-signed task.
The CLI verifies and preserves the signed record; never re-sign it or delete the
output ledger to reset budgets. Scripted policies are the default. Live policies
require explicit `CHEMFLOW_LIVE_LLM=1`, the provider API key, and `--model` for
OpenAI; missing authorization must refuse rather than silently fall back.
