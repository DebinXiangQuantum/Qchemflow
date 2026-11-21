---
name: chemflow-rules
description: Enforce ChemFlow task integrity, evidence and resource budget boundaries.
---
Only use the tools of the bound `qcl` MCP server. Treat signed tasks and budgets as
immutable. Humans run the CLI to freeze tasks and resource budgets.
Start an approved run with `chemflow agent run --task task.json
--public-key /trusted/public.pem --out /tmp/chemflow-run`. Preserve the output directory
and its ledger across reruns; inspect the generated summary, report and journal. The
default policy is scripted; live policies require `CHEMFLOW_LIVE_LLM=1` and provider
credentials (plus `--model` for OpenAI), with no silent scripted fallback.
D-R2: the only dimensions are wall_s, cpu_core_s, gpu_s, sim_shots, qpu_s and
qpu_shots. QPU resources are device seconds and shots, not billing or job counts.
Unknown amounts are not zeros; exhausted dimensions refuse unknown consumption.
QPU sampling is stochastic, just like simulator sampling. Never edit task, key or journal
files with framework shell or file tools. `screen_run` writes the report.
