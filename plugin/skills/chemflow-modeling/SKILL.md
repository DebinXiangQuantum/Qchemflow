---
name: chemflow-modeling
description: Inspect frozen molecular targets and select consistent active space and spin sectors.
---
Use `diagnose_model` for orbital diagnostics; compare the suggested active space
with the signed task target. Use `registry_get`, `plan_new`, `plan_add_node`, and
`check` to validate ports, sector and spin. Never edit signed task records;
request a human to freeze a new version if the target must change.
