# ChemFlow applications

Applications are **unsigned task generators**, not accuracy guarantees. Select an
application, inspect its finite model, error/relation gates and resource budget, and
ask a human approver to freeze the task. Neither the catalogue, planner nor MCP
drafting tools have signing authority. No network is used to construct examples.

## Catalogue

All energy outputs are in hartree except MM/GBSA (kcal/mol). A gate against a
model-defined reference measures reproducibility of that model, not experimental
accuracy. STO-3G examples are inexpensive software checks, not production chemistry.

| ID / Chinese name | Inputs | Outputs and methods | Reference | Validity and limitations |
|---|---|---|---|---|
| `ground_state_energy` 分子基态能量 | XYZ/SDF/SMILES, basis, active space, sector | Molecular ground energy; HF/correlation, SCI/SQD, optional sampled VQE request | Small-space PySCF FCI; Szabo–Ostlund | Fixed geometry/model/sector. MQ02 candidates must pass R130; statevector VQE is not deployed. |
| `stability_ranking` 多分子稳定性排序 | Named same-formula, same-charge isomers | Electronic energy vector, order and supported rank pairs | Component FCI; APP C | Not Gibbs stability; ordering needs resolved energy differences. Example: HCN/HNC. |
| `conformer_ranking` 多构象能量排序 | Connected closed-shell SMILES; count, RMSD | Seed-42 ETKDG/MMFF (or GFN2) inputs, then electronic ranking | Electronic FCI; Riniker–Landrum ETKDG, Halgren MMFF | Draft optimization is deterministic input preparation only. Search is incomplete; deduplication can leave one conformer. |
| `pes_scan` 势能面扫描 | Molecule, bond/angle/dihedral atoms and values | Rigid or constrained-relaxed energy curve | Pointwise FCI; APP B | Explicit `internal_coordinate` validity axis; valid sampled points do not prove all geometries accurate. |
| `metabolic_sites` 代谢位点预测 | Explicit-H molecule, basis, HF level | C-H BDE/FreeSASA site scores and top site | Independent PySCF ROHF/RHF plus FreeSASA formula recomputation | Empirical, uncalibrated model; no enzyme specificity, kinetics or clinical prediction. |
| `bond_dissociation` 键解离能预测 | Molecule, all bonds / element-pair / index filter | Rigid homolytic BDE vector | Fragment-minus-parent FCI; APP C | Neutral rigid radicals; not relaxed/thermal BDE, not universal bond-breaking chemistry. |
| `solvation_scan` 溶剂化自由能扫描 | Molecule and solvent list | One energy difference per solvent; GFN2/ALPB or HF/DFT PCM | Direct tblite/PySCF; Cossi et al. PCM, Ehlert et al. ALPB | Same geometry. PCM is electrostatic only. SMD requires compiled support and is not the default. |
| `frontier_orbitals` 前线轨道能量预测 | Molecule, HF/DFT level | HOMO, LUMO, orbital gap; optional vertical IP and EA tasks | SCF eigenvalues; charge-sector FCI | Orbital gap is not the fundamental gap. Physical finite-model gap is IP−EA; finite basis/spin limitations apply. |
| `aqueous_solubility` 水溶度快速评估 | SMILES; optional melting point in °C | ESOL by default, GSE when melting point supplied; log10(mol/L) | Independent RDKit descriptor fit; Delaney 2004, Yalkowsky–Valvani 1980 | Empirical domain and calibration labels retained; no pH/temperature correction; dataset RMSE is not an individual error bar. |
| `reaction_barrier` 反应能垒估算 | Ordered-atom reactant/TS/product; species active spaces | Verified connected TS barrier, optional RRHO Gibbs/Eyring rate | HF saddle/IRC and active-space FCI; Sella, Eyring | One imaginary mode and IRC connectivity required. No tunnelling/dynamical recrossing by default. HCN example uses CAS(4,4), retaining both π orbitals. |
| `dimer_binding_scan` 二聚体结合能(刚性扫描) | Dimer, two atom-index fragments, separations | Counterpoise-corrected rigid interaction curve | Direct component FCI; Boys–Bernardi | Centroid separation, fixed monomers; not binding free energy or solvent/entropy sampling. |
| `protein_ligand_binding` 蛋白质-配体结合能 | Receptor PDB plus residue/SMILES/SDF ligand | Amber14/GAFF2/GBn2 MM/GBSA; optional frames/minimization | Independent OpenMM Reference calculation; PDB 181L | Preparation/protonation dependent; no configurational entropy. Requires side environment; missing tools give CF-E080. SMILES alone is not docking. |
| `lattice_superconductivity` 超导材料模拟(格点模型) | Hubbard lattice and sector, site pair | s-wave pair and spin correlations plus ground energy | Exact finite-sector diagonalization; APP G | Finite clusters only. **Unconventional Tc is not predicted. Allen–Dynes is only for user/EPW electron–phonon inputs.** |

See `app-a-properties.md` through `app-g-lattice-correlations.md` for operator
schemas, literature and proof assumptions; `app-h-applications.md` describes the
new integration and metabolic target. The catalogue's `parameters` field lists
defaults. Additional optional knobs are `functional`, `dispersion`, `final_spin`,
`melting_point_C`, `model`, and `method`, where applicable.

## CLI workflow

```sh
chemflow app list
chemflow app describe frontier_orbitals
chemflow app draft frontier_orbitals --example --param physical_gap=true --out drafts/frontier
chemflow app draft ground_state_energy --example --param qpu=true --out drafts/mq02
chemflow app draft conformer_ranking --input molecule.smi --param count=8 --out drafts/conformers
chemflow app draft stability_ranking --input isomer-a.xyz --input isomer-b.xyz --out drafts/ranking
chemflow app draft solvation_scan --example --out drafts/alpb
chemflow app draft solvation_scan --example \
  --param 'solvents=[{"model":"pcm","eps":78.3553},{"model":"pcm","eps":9.8629}]' \
  --out drafts/pcm
chemflow app draft aqueous_solubility --example --param melting_point_C=40.0 --out drafts/logs
chemflow app draft reaction_barrier --example --param rate_constant=true --out drafts/barrier
chemflow app draft protein_ligand_binding --example --out drafts/binding
chemflow eval list
chemflow eval run --suite apps --policy scripted --jobs 3 --out runs/apps
```

`--input` is repeatable. A JSON input contains the application's named inputs
(use `describe` to inspect an example); XYZ/SDF inputs describe molecules, PDB
describes a receptor, and `.smi`/`.smiles` contains one SMILES. Binding SDF inputs
must already be positioned relative to the receptor. Drafting refuses to overwrite
existing task JSON files. It never freezes a task: inspect each file and use the
existing human-operated `chemflow task freeze` flow before `chemflow agent run`.

`draft(id, inputs=None, parameters=None, example=False)` is also available from
`chemflow.apps`. Examples and defaults are copied before overrides. Every returned
TaskSpec includes an absolute-error gate in the target's unit and a relation gate.
Defaults are suggestions requiring human approval, not evidence that a method passes.
The default solvation draft includes water/octanol ALPB and water/octanol PCM
(dielectric constants 78.3553 and 9.8629 respectively), as four separate tasks.
The QPU option only requests `vqe_sampled` in the decision and allocates QPU time/shots;
it does not choose a binding or weaken deployment checks.

## Planner and MCP

Planner tools are `app.list` and `app.draft`; their MCP names are `app_list` and
`app_draft`. Listing is available in every profile, drafting only in planner/full.
`app.draft` accepts `app_id`, `inputs`, `parameters`, and `example`. Inputs and
parameters can be JSON strings for strict-schema LLM providers; direct tool calls
also accept dictionaries. The tool returns TaskSpec dictionaries, never approvals.

## Evaluation and reproducibility

`src/chemflow/eval/suites/apps.yaml` contains 13 tasks. Recompute references with:

```sh
python scripts/eval_reference_values.py --suite apps --print-values
python scripts/eval_reference_values.py --suite apps
```

These scripts call PySCF, RDKit, tblite, FreeSASA and a separate OpenMM oracle
directly, never ChemFlow execution or domain adapters. The HCN TS reference solves
the independent HF gradient root and checks its Hessian before CASCI. The binding
oracle independently prepares **tests/data/181L.pdb** with the declared protocol.
Preparation is deterministic to numerical tolerance, not guaranteed bitwise across
platforms; the binding bar is 0.01 kcal/mol. Missing PROPKA is explicitly labelled
as standard-pKa fallback; changing protonation protocols requires new references.

Install existing extras with `chemflow[apps]`. The `binding-tools` extra records
side-environment dependencies; use the existing conda tools environment when these
packages are unavailable from pip. No package download occurs during application use.
Suite wall time is recorded in `results.json` at `metadata.elapsed_s`; the batch
validation report records the measured run on this worktree.
