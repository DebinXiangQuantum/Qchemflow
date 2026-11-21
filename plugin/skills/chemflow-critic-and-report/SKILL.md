---
name: chemflow-critic-and-report
description: Read a QCL screen's graded results and report them without overclaiming.
---
`screen_run` returns the decision, the graded outputs and `report.md`; `results_query`
returns every graded output so far. Each output carries its value, the compiler's
relation to the exact answer (with a certificate), its gate verdict, cost and device.
The verdict is computed deterministically: classical controls that meet the gates at
no greater cost dominate a candidate, and a candidate escalates only with a supported
relation, passing gates and evidence. Quote only recorded values. Escalation is not
advantage: never claim an exact, certified, QPU or advantage result the record does
not support. Humans alone freeze task revisions and resource budgets.
