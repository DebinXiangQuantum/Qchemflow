---
name: chemflow-screens
description: Review and run QCL screens of a signed ChemFlow task through the qcl MCP tools.
---
A screen compiles the signed task into one QCL plan: the classical ladder (HF, MP2,
CCSD, CCSD(T)), candidate methods and the exact in-model reference, sharing SCF and
integrals. Read the task with `task_get` and the ladder, reference and candidate
templates with `methods_list`. Candidates are `sci {select_cutoff}` and
`sqd {shots}` (closed-shell species only).

Compile a proposal with `screen_review {candidates}` before running it: it reports the
QCL plan hash, the compiler-inferred relation of every output, the predicted time and
the simulator shots, and spends nothing. `screen_run` runs the scripted candidate
rounds, or one round of your candidates, and returns the deterministic critic's
decision (stop, escalate or revise), every graded output and the report. Each run
spends the signed budget; a run beyond it is refused with CF-E070 before any work.
Query `results_query` for everything graded so far.

For a method outside the typed candidates, write a QCL program and ask the human to
run `chemflow agent run --program <file>` or `--screen --candidate-program <file>`:
authored programs are checked against the signed task's question.
