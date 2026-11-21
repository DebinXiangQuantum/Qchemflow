# V1a: relation certificates

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

V1a verifies **what an executed result is relative to its graph-defined target**.
It does not prove an adapter's implementation correct, certify floating-point
roundoff, or establish that a chemical model describes an experiment accurately.
The mathematical implication is conditional on explicit implementation contracts.
V1b will falsify those contracts; V1d can replace the documented mathematics with
machine-checked lemmas.

Previously a refinement supplied a relation label and an algebra propagated it.
Static `infer_relations` remains useful for planning and reference discovery, but
is not evidence. Execution now constructs a proof, independently checks it, and
only marks the relation `proved` after acceptance by the kernel. Unsupported
bindings and model changes retain their label with status `declared`; they cannot
satisfy relation gates or cause critic escalation. Legacy hand-built critic items
without a status key retain their previous behavior.

## Denotations

`verify/semantics.py` maps an operator and its registered binding/version to a
term kind, independently of the refinement's claimed strength:

| Binding family | Denotation |
| --- | --- |
| FCI / sector diagonalization | Lowest eigenvalue of the supplied Hamiltonian in the requested sector |
| SCI / SQD / Ritz | Eigenvalue of the Hamiltonian projected on an explicit determinant set |
| VQE / ADAPT | Rayleigh quotient of a nonzero state in the fixed-particle SectorEngine |
| HF | Physical Hamiltonian expectation in an HF determinant; global HF convergence is unnecessary for the bound |
| Reaction / interaction / difference | Signed linear combination of input energies, in common units |
| Lehmann | Retarded projected spectral sum at the specified grid and broadening |
| Expectation / sample | Exact or sampled expectation of the specified observable in the specified state, **not** its ground energy |
| Thermochemistry | RRHO ideal-gas correction at the specified temperature, pressure and frequencies |
| MP2 / CC / embedding / periodic SCF / docking | Named approximation with no ordering law |
| Typed intermediate | Model, Hamiltonian, sector, state, samples or subspace construction |

Unlisted bindings have no denotation. An arbitrary new binding does not acquire
an axiom by choosing the same refinement label as a built-in method.

## Lemma table

`LEMMAS` includes full statements, named hypotheses and literature references.
The table is deliberately separate from `AXIOMS`.

| Lemma | Mathematical rule and principal hypotheses |
| --- | --- |
| `exact_diagonalization` | Complete diagonalization of Hermitian H restricted to S returns its lowest eigenvalue; Courant–Fischer. |
| `sector_decomposition` | Spin-free number-conserving H commutes with both particle-number operators, so their joint sectors are invariant; Szabo–Ostlund ch. 2. |
| `rayleigh_ritz` | For a nonempty orthonormal determinant space V contained in S, ordered Ritz root k bounds sector root k from above; Hylleraas–Undheim–MacDonald. |
| `rayleigh_ritz_full` | Distinct valid sector determinants numbering dim(S) span S, giving equality of corresponding roots; Rayleigh–Ritz. |
| `variational_principle` | A nonzero trial state in S has Rayleigh quotient at least the lowest sector eigenvalue; Szabo–Ostlund ch. 3. |
| `positive_linear_combination` | Exact terms allow either sign; upper-bound terms require nonnegative coefficients to preserve an upper bound. |
| `linear_combination_exact` | A signed linear combination of exact nonzero-coefficient terms is exact. |
| `stochastic_linear_combination` | Deterministic signed sums of exact values and sampled observable estimates remain stochastic; linearity of expectation. |
| `lehmann_completeness` | Exact ground and complete particle/hole resolutions with the correct ladder amplitudes give the retarded response; Fetter–Walecka. |
| `stochastic_estimate` | Finite-sample observable estimates are stochastic, without a deterministic ordering or accuracy guarantee. |
| `expectation_evaluation` | Evaluating an observable exactly equals the specified expectation target, not necessarily the ground eigenvalue. |
| `rrho_evaluation` | Evaluation of the specified RRHO partition-function correction equals that model's correction; McQuarrie, Statistical Mechanics. |
| `no_ordering_law` | Forget an unsupported ordering/equality claim for a known approximation; `approximate` asserts no error bound. |

SCI and SQD certify their requested upper-bound claim, even when selection happens
to cover the full particle sector. The full-equality search rule is used for the
explicit `rayleigh_ritz` operator. This preserves the existing recipe contract
without pretending that a full particle basis is a full *spin-selected* basis.
The kernel can check a full-Ritz argument when its hypotheses actually hold.

## Implementation axiom table

All axiom identifiers have the prefix `binding:`. They bind a specific registered
implementation, operator and version (currently `1`). Contracts describe what
the adapter must do, **not** something the symbolic kernel has independently
established by comparing numerical energies.

| Axiom suffix | Contract |
| --- | --- |
| `fci_pyscf` | Lowest physical eigenvalue in the requested particle/spin sector, including constants. |
| `sector_diag_numpy` | Lowest physical eigenvalue in the requested particle/spin sector, including constants. |
| `sci_pyscf` | Lowest projected physical energy on the exported final SCI determinant set, with supported spin selection. |
| `sqd_sector` | Lowest projected physical energy on the exported final recovered/augmented SQD set, with supported spin selection. |
| `vqe_statevector` | Exported nonzero SectorEngine state; exact mode evaluates its quotient, shot mode samples its observable. |
| `adapt_statevector` | Exported nonzero SectorEngine state and its exact physical energy. |
| `hf` | Physical energy of a normalized determinant with the requested spin occupations. |
| `reaction_hf` | Species HF energy in the specified particle sector. |
| `lattice_hf` | Site-Hamiltonian HF energy in the specified particle sector. |
| `mp2` | Named MP2 approximation, without an ordering law. |
| `ccsd` | Named CCSD approximation, without an ordering law. |
| `ccsd_t` | Named CCSD(T) approximation, without an ordering law. |
| `reaction_mp2` | Species MP2 approximation, without an ordering law. |
| `reaction_ccsd` | Species CCSD approximation, without an ordering law. |
| `reaction_ccsd_t` | Species CCSD(T) approximation, without an ordering law. |
| `ewf_embedding` | Specified embedding approximation, without an ordering law. |
| `rayleigh_ritz` | Projected Hermitian eigensolve on the supplied basis, retaining the full projected resolution. |
| `lehmann_from_spectra` | Retarded sum using the supplied resolutions and fermionic ladder amplitudes. |
| `spectral_exact` | Full ground and feasible neighboring-sector resolutions followed by the retarded sum. |
| `reaction_energy` | Specified stoichiometric sum, including matching thermal corrections when requested. |
| `interaction_energy` | Complex minus fragments, with the declared ghost-basis counterpoise convention. |
| `energy_difference` | Left energy minus right energy in common units. |
| `compare` | Left energy minus right energy in common units. |
| `pauli_shot_estimator` | Finite-shot Pauli estimate of the specified observable/state. |
| `ideal_energy_projective_sampler` | Finite-shot projective estimate of the specified observable/state. |
| `expectation` | Exact expectation of the specified observable in the specified state. |
| `molecule` | Preserves atoms, coordinate unit, charge and spin. |
| `geometry` | Preserves the supplied geometry parameters. |
| `electronic_model` | Constructs the specified basis/reference/electronic model. |
| `scf` | Constructs spin-free spatial integrals in the SCF orbital basis. |
| `integrals` | Extracts the SCF Hamiltonian without changing integrals or constants. |
| `active_space` | Restricts to the specified active space with frozen-core constants accounted for. |
| `hamiltonian` | Constructs the declared molecular Hamiltonian and counts constants once. |
| `sector` | Preserves integer particle counts and the optional spin target. |
| `encode` | Implements the alpha-first Jordan–Wigner representation, including constants. |
| `prepare` | Prepares the occupied-spin HF bitstring in the given encoding. |
| `prepare_state` | Constructs the requested HF/UCCSD/LUCJ state in the particle sector. |
| `sample_configurations` | Returns computational-basis counts with the requested sampling model and shots. |
| `full_subspace` | Enumerates every determinant in the particle sector. |
| `subspace_from_samples` | Recovers occupations and takes the declared product closure of spin strings. |
| `extend_subspace` | Applies the specified ladder and excitation closure in the destination sector. |
| `lattice_model` | Constructs the declared spin-independent Hubbard integrals. |
| `impurity_model` | Constructs the declared Anderson impurity and bath integrals. |
| `periodic_scf` | Specified periodic mean-field approximation, without a universal ordering claim. |
| `dock` | Specified docking approximation, without an ordering law. |
| `component_energy` | Generic correlated component approximation, without an ordering law. |
| `component_energy:fci` | FCI on the exported closed-shell component Hamiltonian, including ghost functions. |
| `component_energy:hf` | HF on the exported closed-shell component Hamiltonian, including ghost functions. |
| `thermochemistry` | Evaluates the specified RRHO ideal-gas thermal correction. |

## Witnesses and exactness

The kernel recomputes predicates from the program and computed objects; recorded
booleans alone are never evidence.

- Every determinant is checked using integer masks and population counts, with
  range and uniqueness checks. Full coverage uses exact binomial dimensions,
  never rounded `sector_fraction`/`coverage` diagnostics.
- Linear coefficients are converted through `Fraction(str(value))`; sign and zero
  tests are exact for the decimal value carried by the JSON program.
- Spin-independent, number-conserving spatial-integral representation establishes
  conservation of N_alpha and N_beta. Integral shape, finiteness and Hermitian
  symmetries are also checked. Lattice/impurity Hamiltonians may be restricted to
  any feasible particle sector; their default filling is not a restriction on H.
  Integral symmetry checks use absolute tolerance
  `1e-10`, no relative tolerance: these are numerical consistency checks, not an
  exact-arithmetic proof about floating-point diagonalization.
- SCI and SQD export `determinants_ref`. VQE and ADAPT export `state_ref`, engine
  and evaluation mode. The kernel checks the finite state shape against the
  sector dimension and normalization to `1e-10`. Shot VQE remains `approximate` relative to the
  ground-energy target, not an unbiased ground-energy estimator.
  SCI stores separate integer alpha/beta strings so combining both spins cannot
  overflow a signed 64-bit packed mask; fractional strings are rejected.
- Requested total spin must be supported by the binding, the returned S² check
  (`1e-4`) and successful spin completion where needed. This is an implementation
  contract plus a numerical consistency check, not a formal spin-purity proof.
- Lehmann equality requires every actual sector basis and the full projected
  energy/vector arrays, not merely the leading-root view or a coverage label.
  Leading views must agree exactly with the retained resolution; orthonormality
  is checked numerically to `1e-10`. Particle/hole counts, orbital, spin and Hamiltonian identities must agree.
  An absent neighbor is allowed only at an exactly empty/full spin filling.
- RRHO equivalence is relative to the declared RRHO model and fixed frequencies,
  not to exact molecular thermodynamics. Reaction coefficients are applied to
  both electronic terms and matching corrections.

Large arrays live in `ArtifactStore`, not in certificate JSON. Run tools persist
them in `<plan_directory>/artifacts`; the kernel verifies their SHA-256 contents before use.
Exact-response diagnostics retain small
resolution descriptors with determinant/vector/energy references. A fresh checker
needs the program, computed objects and artifact store; a certificate is not a
standalone proof of the contents of missing arrays. Missing witnesses weaken
known denotations to `approximate`, rather than trusting counts or old labels.

## Certificate format and kernel

The public API is:

```python
from chemflow.verify import prove, check_certificate

certificate = prove(program, node_id, registry=registry, computed=computed)
verdict = check_certificate(certificate, program, registry=registry, computed=computed)
```

`computed` is the executor result mapping, including its `_artifacts` store.
Certificates contain exactly `version`, `node`, `relation`, `program_hash` and
`steps`. Each step contains `id`, `lemma`, `node`, `uses`, `side_conditions` and
`conclusion: {node, relation}`. IDs use `node:axiom`, `node:sector` and
`node:result`; cited steps must precede the citing step. Intermediate conclusions
`denotation` and `sector_invariant` are not result-to-target relations.
Dependency ports are visited in sorted order, so canonical program serialization
cannot change the certificate or invalidate its ordered citations.

The prover traverses the dependency closure bottom-up and chooses applicable
rules. The kernel never imports the prover. Its small `RULES` table specifies
which denotations and conclusions each lemma permits. The checker independently:

1. Checks the exact schema, version and whole-program hash.
2. Resolves the node's closure and recomputes concrete witnesses.
3. Checks the binding axiom, version, step identity and ordered citations.
4. Compares the entire recorded witness mapping to its recomputation.
5. Requires every hypothesis of the selected mathematical rule to be true.
6. Checks sector premises, signed-combination premises and complete-response
   premises, then matches the final conclusion to the certificate header.

Unknown axioms, unknown rules, missing/forward premises, wrong-node conclusions
and altered side conditions are rejected. `Verdict` contains `valid` and `reasons`.
Predicate helpers in `semantics.py`, the lemma/axiom tables, and the kernel are
the explicit trusted base; proof-search decisions are not trusted.

The certificate ID is SHA-256 of `chemflow.ids.canonical_json(certificate)` (UTF-8,
sorted keys, compact separators, no NaN). Files are written to
`<plan_directory>/certificates/<id>.json`. A trailing file newline is not part of
the canonical hash. Executor certificates initially name the executable program;
the plan tool rechecks certificates against the stored gate-free plan before
publishing them.

Every executed estimate/spectrum gets `relation_status` and
`relation_certificate` in diagnostics. Rated items and recorded headlines expose
both fields. A weaker proved result replaces the old label and preserves
`declared_relation`. An unproved attempt can still have a stored certificate ID,
but its status is `declared` and the kernel verdict is not valid. Relation gates,
including signed-task R127 requirements, require accepted proofs; exact-reference
accuracy gates also require a proved exact reference.

## Designer/critic tool

`algorithm.prove {plan_id, node_id}` is available to designer and critic roles,
including their MCP profiles. It validates the stored plan and task identity,
executes only the requested dependency closure at screen scale when needed,
checks effect permissions and uses the same persistent task budget/lock as `run`.
Repeated requests can reuse the already executed closure. The response contains
`certificate`, `certificate_id`, `kernel_verdict: {valid, reasons}`, and `scale`.
No LLM, network request, symbolic-algebra dependency or exact reference solve is
required merely to prove a Ritz bound.

## Registry coverage and deliberate limitations

The all-domain-pack registry audit distinguishes unsupported implementations from
supported approximations:

- **Not proved:** `qmmm_energy`, `qmmm_rhf`, `qmmm_casci`, `qmmm_gpu_fci`,
  `qmmm_sqd_statevector`, `qmmm_sqd_aer`, `qmmm_sqd_gpu`, `qmmm_sqd_qpu`.
  The external QM/MM calculator does not expose the embedded-Hamiltonian and
  solver-basis/state witnesses needed by this kernel. Its old equality/upper-bound
  labels remain **declared**, not silently upgraded to proofs.
- **Not proved:** optional `fci_cupy` and unknown/custom or experimental-design
  bindings; they have no reviewed implementation axiom in this version. GPU/QPU
  samplers are likewise not themselves certified. Their concrete recovered
  determinant outputs can still support a downstream CPU Ritz proof, which does
  not rely on the distribution that selected the subspace.
- **Deliberately declared:** `potential_energy` and explicit model changes. No
  equality with a different Hamiltonian is inferred.
- **No denotation for intermediate utilities:** geometry optimization,
  frequencies, periodic-system/MM-system construction, ligand utilities, route
  search and `map`. Downstream mathematics can be conditional on their concrete
  outputs; this is not a proof of the optimization, Hessian or cheminformatics
  implementation.
- **Proved only as approximate:** MP2, CCSD, CCSD(T), EWF, periodic SCF, docking,
  incomplete Lehmann response, shot VQE, mixed-sign combinations of nonexact
  energies, and known solvers with missing/failed witnesses. No quantitative
  accuracy, convergence, spin purity, statistical confidence or validity domain
  follows from `no_ordering_law`.
- **Conditionally proved:** HF/VQE/ADAPT/SCI/SQD require the concrete sector,
  Hamiltonian and applicable spin witnesses. Unsupported spin/ansatz/measurement
  cases do not inherit a bound just because their refinement declared one.

The old `fci_pyscf` refinement alias has no separately advertised implementation
under that alias; the actual `fci_pyscf` binding is covered by its implementation
contract. Full Lehmann responses can be proved equivalent even though static
inference conservatively labels the generic `lehmann` refinement approximate.

## Validation

`tests/test_v1a_certificates.py` covers false-declaration composition downgrades,
full and sampled Ritz, complete/incomplete/truncated Lehmann resolutions,
invalid sector bases and Hamiltonians, missing artifacts, VQE state witnesses,
signed differences, unknown-binding gates, JSON round trips, closure-only tool
execution, artifact persistence and 250 deterministic random single-field
certificate mutations. The immutable `tests/test_review_v1a.py` remains unchanged.

Final validation on branch `v1a` (2026-10-14):

| Check | Result |
| --- | --- |
| Full pytest, supplied interpreter/environment, cache provider disabled | **1354 passed, 11 skipped**, 354.11 seconds. |
| Dedicated V1a tests plus immutable V1a reviewer tests | **27 passed**: 20 new cases and all 7 reviewer cases. |
| Ruff on `src` and `tests/test_v1a_certificates.py` | **All checks passed.** |
| Full-repository Ruff | **One existing I001**, `tests/test_review_v1a.py:193`; the immutable reviewer's imports are not sorted. Not modified or suppressed. |
| SHA-256 comparison for all 47 `tests/test_review_*.py` files | **All unchanged.** |
| `git diff --check` | **Passed.** |

No reviewer assertion is considered mathematically wrong. Legacy SCI/SQD tests
require the conservative recipe upper-bound label even when the selected set is
full; explicit full-Ritz proofs expose equality. There are no new dependencies,
network/LLM calls or commits in this implementation.

## Changed files

| File | Change |
| --- | --- |
| `src/chemflow/verify/__init__.py` | Public lemma/axiom, prover and checker API. |
| `src/chemflow/verify/lemmas.py` | Thirteen mathematical rules and forty-nine binding contracts. |
| `src/chemflow/verify/semantics.py` | Binding denotations and concrete witness predicates. |
| `src/chemflow/verify/prover.py` | Deterministic bottom-up proof construction. |
| `src/chemflow/verify/kernel.py` | Independent rule and certificate checking. |
| `src/chemflow/verify/integration.py` | Production checking, relation diagnostics and certificate persistence. |
| `src/chemflow/runtime/executor.py` | Certify executed outputs before evaluating gates; require proved relations. |
| `src/chemflow/runtime/artifacts.py` | Persist artifacts and optionally verify their SHA-256 contents. |
| `src/chemflow/impls/sci.py` | Export the actual final alpha/beta determinant strings by reference. |
| `src/chemflow/impls/sqd.py` | Export the actual final recovered/augmented determinant set by reference. |
| `src/chemflow/ops/molecular.py` | Export VQE/ADAPT states and exact-evaluation engine metadata. |
| `src/chemflow/domains/dynamics.py` | Retain full exact-Lehmann resolution witnesses by reference. |
| `src/chemflow/domains/drug.py` | Expose counterpoise-component Hamiltonian and sector witnesses. |
| `src/chemflow/agent/tools.py` | Budgeted closure-only proof tool, artifact and certificate persistence. |
| `src/chemflow/agent/critic.py` | Never escalate a declared candidate. |
| `src/chemflow/agent/report.py` | Certificate/status fields on items and headlines; proved relation gates. |
| `src/chemflow/agent/llm.py` | Describe the proof tool in the designer prompt. |
| `plugin/agents/designer.md` | Designer MCP proof-tool allowlist and instructions. |
| `plugin/agents/critic.md` | Critic MCP proof-tool allowlist and declared-relation policy. |
| `tests/test_v1a_certificates.py` | Dedicated mathematical, adversarial, persistence and budget regressions. |
| `docs/v1a-relation-certificates.md` | Design, trust boundary, tables, coverage and implementation report. |

The pre-existing untracked research design and reviewer test are not implementation
changes. No reviewer test, dependency declaration or lint configuration is modified.
