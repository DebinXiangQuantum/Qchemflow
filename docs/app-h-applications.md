# APP H: unsigned application layer and metabolic site model

## Architecture and authority

`chemflow.apps.catalog()` enumerates the 13 fixed identifiers and Chinese names.
`draft()` composes the A–G target schemas, never signs or runs an electronic
calculation. The sole preparatory optimization is the requested deterministic
conformer search (seed 42, default count 8). Ranking accepts a nonempty collection:
deduplication of a small molecule can legitimately yield a singleton. No duplicate
or artificial conformers are fabricated to satisfy a two-item minimum.

CLI and planner/MCP delegate to the same function. Task validation is performed
before writing drafts. All tasks carry unit-aware absolute-error/relation gates
and wall/simulator/QPU resource limits; only the human signing flow can freeze them.
Sampled VQE requests are decision text and resources only, preserving R130.

## New target and operator

```yaml
kind: metabolic_sites
operator: metabolic_sites
molecule:
  atoms: 'C 0 0 0; H .63 .63 .63; H -.63 -.63 .63; H -.63 .63 -.63; H .63 -.63 -.63'
electronic_model: {basis: sto-3g}
level: {method: hf, functional: null, dispersion: null}
```

The frozen target forbids extra fields. The thin `metabolism` pack wraps APP C
`site_reactivity` using its explicit-H graph, rigid neutral radicals, HF BDE and
FreeSASA accessible area. HF is the default; the existing APP C `mp2`, `ccsd_t`,
and `fci` levels are also accepted. The operator returns a typed `Estimate` of kind
`metabolic_sites`, with dimensionless maximum score and per-site diagnostics:

`score(H) = SASA_fraction(H) * sigmoid((90 - BDE(H)[kcal/mol]) / 3)`.

`SASA_fraction` divides FreeSASA area by `4π(r_vdw + 1.4 Å)^2`.
The diagnostics retain `sites`, `top_site`, the scoring formula,
`evidence: empirical`, `validity_status: uncalibrated`, and
`experimental_relation: model_change`. Reports explicitly label the result as
model-defined and diagnostic-only. There is no calibrated empirical evidence token.

The catalogue's C-H abstraction heuristic is not SMARTCyp, a CYP model, an enzyme
selectivity model, a pharmacokinetic model, or a claim of biological accuracy.
Zero sites means maximum score zero and no top site; it does not establish metabolic
stability. Unsupported chemistry/perception fails rather than generating a score.

## Verification, resources and identity

The pack supplies separate model-evaluation control and screening reference nodes.
Frozen identity checks cover molecule, electronic model, level, all targets and
references. R136 validation rejects unsupported/nonfinite/extra parameters, while
nullable level fields remain accepted by the existing checker rule.

The operator has an explicit classical implementation and resource model: unknown
CPU/wall time, known-zero GPU, simulator shots and QPU time/shots. Its V1a
`declared_model` axiom states equality **only to the specified scoring model**.
It reuses the existing model-evaluation lemma; no new physical inequality is added.
Equality of repeat evaluations is not equality to experimental metabolism.

The eval harness scores the dimensionless headline and, when declared, every
atom-indexed site score. The independent methane oracle recomputes parent/radical
HF energies and FreeSASA areas without calling ChemFlow. Other app references use
direct FCI, SCF/ALPB, RDKit descriptor fits and OpenMM in the tools environment.

## Optional rate

`ReactionBarrierTarget.rate_constant` defaults false. When true it requires a Gibbs
thermochemistry target with declared temperature. Report generation evaluates the
existing Eyring `rate_constant` model from each computed Gibbs barrier, preserving
TS/IRC verification. The reported rate has no separate accuracy guarantee or
tunnelling correction. Existing reaction tasks remain unchanged by default.

## References and limitations

- APP C `docs/app-c-reactivity.md`: rigid homolysis, graph perception and heuristic.
- Mitternacht, *F1000Research* 5:189 (2016): FreeSASA.
- Riniker and Landrum, *J. Chem. Inf. Model.* 55, 2562–2574 (2015): ETKDG.
- Halgren, *J. Comput. Chem.* 17, 490–519 (1996): MMFF94.
- Delaney, *J. Chem. Inf. Comput. Sci.* 44, 1000–1005 (2004): ESOL.
- Eyring, *J. Chem. Phys.* 3, 107 (1935): transition-state rate model.
- Existing APP A–G docs contain electronic, solvent, MM/GBSA and lattice references.

The independent reference routes are deliberately scoped to the small bundled
fixtures; unsupported general molecules/protocols raise errors. FCI caps and model
limitations still apply to user inputs. SMILES embedding is not conformer coverage,
SDF/PDB ligand placement is not docking, and finite-cluster pairing is not a Tc prediction.
