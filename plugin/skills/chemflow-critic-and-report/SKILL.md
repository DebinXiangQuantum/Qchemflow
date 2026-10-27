---
name: chemflow-critic-and-report
description: Critique supported evidence and produce a linted ChemFlow report.
---
Inspect `results_query` and `results_get`, compare relation strength, gate verdicts,
costs and classical dominance. Use `report_write` (`report.write`) with headlines_json
containing a JSON list of `{"result_id": "<id from run output results>"}` objects.
The server supplies the recorded value, relation, evidence, gate verdict, cost and backend.
Missing or unknown ids and supplied fields that differ from the record are rejected.
Never supply advantage_evidence: advantage or speedup claims require a supporting
critic verdict recorded by the server, not an agent assertion. Escalation alone is not advantage.
Never claim an exact, certified, QPU or advantage result without supporting evidence.
Humans alone freeze task revisions and resource budgets.
