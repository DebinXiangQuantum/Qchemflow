---
name: chemflow-rules
description: Enforce ChemFlow task integrity, role isolation, evidence and resource budget boundaries.
---
Only use tools exposed by the bound MCP role. Treat signed tasks and budgets as
immutable. Humans run the CLI to freeze tasks and resource budgets.
Start an already approved loop with `chemflow agent run --task task.json
--public-key /trusted/public.pem --out /tmp/chemflow-run`. Preserve the task ledger
across reruns; inspect the generated summary, report and journal. The default is
scripted; live policies require `CHEMFLOW_LIVE_LLM=1` and provider credentials
(plus `--model` for OpenAI), with no silent scripted fallback.
D-R2: the only dimensions are wall_s, cpu_core_s, gpu_s, sim_shots, qpu_s and
qpu_shots. QPU resources are device seconds and shots, not billing or job counts.
Unknown amounts are not zeros; exhausted dimensions refuse unknown consumption.
Completed results survive overruns, which must be reported in budget_overruns.
QPU sampling is stochastic, just like simulator sampling. Never edit task, key or journal
files with framework shell or file tools. Screen before reporting and use `report_write`.
