---
name: chemflow-modeling
description: Inspect a signed molecular target and its active space before screening.
---
Use `diagnose_model` for the RHF energy, HOMO-LUMO gap, MP2 natural occupations and a
suggested active space, and compare it with the signed target. Tasks with
`active: auto` take the modeler's choice or that suggestion; a fixed active space is
never changed. Never edit signed task records; ask a human to freeze a new version if
the target must change.
