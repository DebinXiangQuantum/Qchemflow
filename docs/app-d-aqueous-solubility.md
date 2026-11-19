# APP D: empirical evidence and aqueous solubility

## Operators and use

Load the `solubility` domain pack (included in `pack_names()`). RDKit is already
declared in the `drug` optional dependency group; NumPy is a core dependency.
No network access is needed. `registry.search("solubility")` discovers the operators.

- `logs_esol(molecule_smiles)` returns a record with numeric `value`,
  `unit: log10(mol/L)`, `model`, `validity_status`, and `max_tanimoto`.
- `logs_gse(molecule_smiles, melting_point_C)` returns the same record for GSE:
  `logS = 0.5 - 0.01 * (max(25, melting_point_C) - 25) - clogP`.
- `empirical_estimate(molecule_smiles, model, melting_point_C=None)` is the
  typed `Estimate` bridge to the existing agent, units, evidence, and gate pipeline.
  Its diagnostics retain the model card and applicability information.

All implementations are classical, pure, and cacheable. GPU time, simulated shots,
QPU time, and QPU shots are known zeros. CPU and wall time are measured, not invented.
Each operator has a typed spec, refinement, implementation axiom and resource model.

## Frozen target

```yaml
question: Estimate paracetamol aqueous solubility
target:
  kind: aqueous_solubility
  molecule_smiles: CC(=O)Nc1ccc(O)cc1
  model: esol
  melting_point_C: null
gates:
  - {kind: abs_error_vs_exact, threshold: 0.000001, unit: dimensionless}
  - {kind: relation_at_least, relation: equivalent}
budget: {wall_s: 300}
allowed_effects: [pure]
decision: Report the model-defined estimate and applicability
```

The target forbids extra fields. `model` is `esol` or `gse`; GSE requires a finite
melting point. Invalid/empty SMILES and nonfinite parameters are rejected before
execution. Pack identity checks bind the molecule, model, and melting point to the
frozen task, including the screening reference. The base plan has a classical
control and a separate model-evaluation screening reference, not an experimental oracle.
Scripted agent runs choose the classical control without quantum escalation.

Eval-suite rows use category `property` and `expected: {value: ..., unit: dimensionless}`.
Correctness means `abs(answer - expected.value) <= task bar`, after converting both
the answer and gate threshold to `expected.unit`. LogS is dimensionless with the
explicit mol/L reference convention, so `log10(mol/L)` and `dimensionless` convert
one-to-one. This is not a conversion of logS to concentration.

## Dataset, fitting, and reproducibility

The package bundles `chemflow/data/esol/delaney-processed.csv`, 1128 molecules,
SHA-256 `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`.
Source: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv

Licence/attribution note: **Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000;
via MoleculeNet. Redistributed research dataset; no separate dataset licence is
asserted.** This is attribution, not a claim that the source grants a particular
licence. Review redistribution terms for your intended use.

`chemflow.empirical.model_card("esol")` computes and caches the ordinary least
squares refit against measured log solubility. It returns defensive copies.
Features are an intercept, RDKit Crippen MolLogP, MolWt, CalcNumRotatableBonds,
and aromatic-heavy-atom fraction. Five shuffled folds use NumPy default_rng seed
42; each held-out prediction uses coefficients fit on the other four folds.
CV RMSE is pooled over all held-out residuals. Train RMSE is in-sample. Neither
metric is a per-molecule bound. Reproducibility assumes the same RDKit descriptor
implementation and numerical environment; this is a refit, not Delaney's original
published coefficients.

## Evidence, relations, and applicability

The V1 `empirical` denotation describes the declared model, not an experimental
property. `empirical_model_evaluation` is definitional equality: the proof kernel
recomputes the model value and applicability witness. A modified result cannot
retain an equivalent proof. The experimental relation is always `model_change`.
Verified ESOL results inside the domain carry runtime-issued `empirical_calibration`
evidence with a content-addressed card/applicability artifact. GSE and outside-domain
results carry diagnostic evidence only. Equivalence remains about the formula.

Inside-domain requires all three:

1. Every element occurs in the training set.
2. Every descriptor is inside its inclusive training min/max range.
3. Maximum training-set Morgan (radius 2, 2048 bits) Tanimoto similarity is at least 0.3.

Results and candidate summaries expose `inside_domain` or `outside_domain`.
The V1c critic rejects outside-domain candidates for escalation. It may still
report a classical estimate with its outside-domain warning.

GSE uses the ESOL structural domain only as a proxy. The CSV has no melting
points, so a GSE train/CV RMSE cannot honestly be computed: its card stores null
metrics and the report explains their absence. No pH/speciation, salt, polymorph,
temperature dependence, or experimental uncertainty is modeled. Disconnected
SMILES are parsed as supplied; applicability is a screen, not chemical validation.

## References

- Delaney, J. S., *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000–1005 (ESOL).
- Yalkowsky and Valvani, *J. Pharm. Sci.* **1980**, 69, 912–922 (GSE basis).
- MoleculeNet's processed Delaney copy, source URL and hash above.
