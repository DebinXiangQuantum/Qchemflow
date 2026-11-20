# QCL screens

Use only the bound `qcl` MCP server profile. The task is signed by a human: never freeze,
edit or reinterpret it, and never assert relations, evidence or gates yourself.

Allowed MCP tools: `task_get`, `methods_list`, `diagnose_model`, `screen_review`, `screen_run`, `results_query`.

`methods_list` gives the classical ladder (HF, MP2, CCSD, CCSD(T)), the exact in-model
reference and the candidate templates: `sci {select_cutoff}` and `sqd {shots}` (closed-shell
molecules). `screen_review {candidates}` compiles a proposal into one QCL plan without
running it and reports the compiler-inferred relations, predicted time and simulator shots.
`screen_run` runs the screen (the scripted rounds, or one round of your candidates) and
returns the deterministic decision, every graded output and the report. Each run spends the
signed budget; a run beyond it is refused with CF-E070.

For programs beyond the typed candidates, write a QCL program and ask the human to run
`chemflow agent run --program <file>`: authored programs are question-checked against the
signed task.
