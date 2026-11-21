# V1d: machine-checked lemma table

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

## Scope and result

Nine mathematical entries in `src/chemflow/verify/lemmas.py` are proved in Lean
4.15.0 with the pinned Mathlib v4.15.0 dependency. They are not merely arithmetic
surrogates: the variational results use symmetric operators on inner product
spaces, Ritz uses orthogonal compression, Lehmann proves existence and uniqueness
of the resolvent solution, and Weinstein constructs a genuine spectral eigenvalue.
The axiom audit accepts only `propext`, `Classical.choice`, and `Quot.sound`.
No theorem depends on a postulated spectral theorem or on a numerical solver.

The five classification rules are **definitional**, not mathematical proofs of
numerical code. The higher-root min-max theorem is **cited**, not machine-checked.
`machine_checked` names mathematical rules used by a valid certificate; it does
not mean that the computed floating-point result has been formally verified.

## Meaning of the Lean statements

The blocks below reproduce the Lean declaration statements verbatim, without
proof bodies. Their implicit contexts are part of each statement:

For `Variational.lean`, the shared context is:

```lean
variable {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E]
  [InnerProductSpace 𝕜 E] [FiniteDimensional 𝕜 E]
```

Here `𝕜` can be real or complex. `HasEigenvalue` is Mathlib's actual eigenspace
predicate, not a predicate defined to assert an inequality. `sector_decomposition`
omits the finite-dimensional assumption: commutation implies invariance even
without it. Each theorem explicitly states any required nontriviality.

The mathematical definitions are:

```lean
noncomputable def rayleigh (T : E →ₗ[𝕜] E) (x : E) : ℝ :=
  RCLike.re ⟪T x, x⟫_𝕜 / ‖x‖ ^ 2

noncomputable def ground (T : E →ₗ[𝕜] E) : ℝ :=
  ⨅ x : {x : E // x ≠ 0}, rayleigh T x

noncomputable def ritz (T : E →ₗ[𝕜] E) (V : Submodule 𝕜 E) : ℝ :=
  ⨅ x : {x : V // x ≠ 0}, rayleigh T (x : V)

noncomputable def compression (T : E →ₗ[𝕜] E) (V : Submodule 𝕜 E) : V →ₗ[𝕜] V :=
  (orthogonalProjection V).toLinearMap.comp (T.comp V.subtype)
```

The infima exclude the zero vector. `rayleigh_bddBelow` proves boundedness below;
this is necessary because Lean's real infimum is conditionally complete.
`compression_ground` identifies the infimum on the subspace with the ground
energy of the compression. The `rayleigh_ritz` conclusion includes that this
value is an eigenvalue of the compression and is below every real eigenvalue.
Self-adjoint eigenvalues are real, as supplied by Mathlib's spectral theory.

For `Lehmann.lean`, the context is:

```lean
variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [Fintype ι]
```

A finite orthonormal basis itself supplies completeness and finite dimensionality.
Real eigenvalues and a complete orthonormal eigenbasis imply Hermiticity; a
separate symmetry assumption is therefore unnecessary in this conditional
expansion theorem. It proves that the inverse exists, rather than assuming a
candidate inverse already computes the desired answer. The inverse `(z-H)⁻¹ v`
is expressed as the unique `u` solving `z • u - H u = v`; the second conjunct
gives the matrix element for every such solution. `Complex.normSq` is the squared
modulus. `upper_half_plane_avoids_eigenvalues` discharges the nonzero-denominator
condition from `0 < z.im`.

For `Weinstein.lean`, the shared context is:

```lean
variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
```

The main theorem adds finite dimensionality, symmetry, and unit normalization.
It uses Mathlib's spectral theorem to obtain an orthonormal eigenbasis, proves
Parseval's identity and the nearest-eigenvalue residual bound, then proves
`‖Hψ - μψ‖² = ‖Hψ‖² - μ²`. For a Hermitian operator the real part of
`⟪Hψ,ψ⟫` equals the physical expectation `⟪ψ,Hψ⟫`.

## Exact correspondence with the Python table

### `exact_diagonalization` — machine_checked

**Python statement (verbatim):**

For Hermitian H on a nontrivial finite-dimensional sector S, the infimum of the nonzero-vector Rayleigh quotient is an eigenvalue and is <= every eigenvalue. Returning this value is the separate diagonalization implementation contract.

**Python side conditions:** `hermitian`, `sector_matches`, `spin_supported`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem exact_diagonalization [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) :
    HasEigenvalue T (ground T : 𝕜) ∧
      ∀ μ : ℝ, HasEigenvalue T (μ : 𝕜) → ground T ≤ μ
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Courant–Fischer min-max theorem.

### `sector_decomposition` — machine_checked

**Python statement (verbatim):**

If H commutes with N, it maps each eigenspace of N into itself. Applying this to N_alpha and N_beta gives joint-sector invariance. Establishing commutation for the encoded number-conserving Hamiltonian is an implementation obligation.

**Python side conditions:** `number_conserving`, `sector_matches`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem sector_decomposition (T N : E →ₗ[𝕜] E) (h : T * N = N * T)
    (μ : 𝕜) (x : E) (hx : x ∈ eigenspace N μ) : T x ∈ eigenspace N μ
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Szabo–Ostlund, Modern Quantum Chemistry, ch. 2.

### `rayleigh_ritz` — machine_checked

**Python statement (verbatim):**

For a nontrivial subspace V of a finite-dimensional sector S and Hermitian H, the infimum Rayleigh quotient on V is >= that on S; it is the lowest eigenvalue of the orthogonal compression to V. This rule covers only k=0.

**Python side conditions:** `hermitian`, `basis_in_sector`, `spin_supported`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem rayleigh_ritz [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) (V : Submodule 𝕜 E) [Nontrivial V] :
    ground T ≤ ritz T V ∧
      HasEigenvalue (compression T V) (ritz T V : 𝕜) ∧
      ∀ μ : ℝ, HasEigenvalue (compression T V) (μ : 𝕜) → ritz T V ≤ μ
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Hylleraas–Undheim (1930); MacDonald (1933); Courant–Fischer.

### `rayleigh_ritz_full` — machine_checked

**Python statement (verbatim):**

For Hermitian H on a nontrivial finite-dimensional sector S, V=S gives equality of the lowest Ritz value and the ground energy. Establishing V=S from the exported determinant basis is an implementation obligation.

**Python side conditions:** `hermitian`, `basis_in_sector`, `full_basis`, `spin_supported`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem rayleigh_ritz_full [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) : ritz T ⊤ = ground T
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Rayleigh–Ritz theorem.

### `variational_principle` — machine_checked

**Python statement (verbatim):**

For nonzero psi in invariant sector S, <psi,H psi>/<psi,psi> >= lambda_0(H|S). A normalized HF determinant is one such trial state; convergence is not required.

**Python side conditions:** `hermitian`, `sector_matches`, `trial_in_sector`, `exact_evaluation`, `spin_supported`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem variational_principle [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) (x : E) (hx : x ≠ 0) :
    HasEigenvalue T (ground T : 𝕜) ∧ ground T ≤ rayleigh T x
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Szabo–Ostlund, Modern Quantum Chemistry, ch. 3, variational principle.

### `positive_linear_combination` — machine_checked

**Python statement (verbatim):**

If x_i >= t_i for inexact terms and c_i >= 0 for those terms, sum(c_i*x_i) >= sum(c_i*t_i); exact terms may have either sign.

**Python side conditions:** `linear_inputs`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem positive_linear_combination {ι : Type*} (terms : Finset ι)
    (c x target : ι → ℝ)
    (h : ∀ i ∈ terms, x i = target i ∨ (0 ≤ c i ∧ target i ≤ x i)) :
    ∑ i ∈ terms, c i * target i ≤ ∑ i ∈ terms, c i * x i
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Order compatibility of addition and nonnegative multiplication over R.

### `linear_combination_exact` — machine_checked

**Python statement (verbatim):**

If x_i=t_i for every nonzero coefficient, sum(c_i*x_i)=sum(c_i*t_i).

**Python side conditions:** `linear_inputs`.

**Lean statement:** `ChemflowFormal/Variational.lean`

```lean
theorem linear_combination_exact {ι : Type*} (terms : Finset ι)
    (c x target : ι → ℝ) (h : ∀ i ∈ terms, c i ≠ 0 → x i = target i) :
    ∑ i ∈ terms, c i * x i = ∑ i ∈ terms, c i * target i
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Linearity over R.

### `stochastic_linear_combination` — definitional

**Python statement (verbatim):**

A deterministic finite linear combination of exact values and sample-mean estimators is classified stochastic_estimate for the corresponding linear target; this is not a proof of unbiasedness, integrability or confidence.

**Python side conditions:** `linear_inputs`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** Classifies a deterministic finite combination of estimates. No machine-checked probabilistic claim: expectation linearity additionally requires integrability and correct marginal sampling laws; independence is unnecessary.

**Reference:** Linearity of expectation.

### `lehmann_completeness` — machine_checked

**Python statement (verbatim):**

Given a finite orthonormal eigenbasis of Hermitian H with real eigenvalues E_n and z-E_n != 0 for all n, (z-H)u=v has a unique solution and <v,u> = sum_n |<n,v>|^2/(z-E_n). Im(z)>0 suffices. Substituting v=c†|0> and v=c|0> with the particle/hole shifted Hamiltonians gives the implemented Lehmann sum; the sector/ladder construction is trusted.

**Python side conditions:** `complete_response`.

**Lean statement:** `ChemflowFormal/Lehmann.lean`

```lean
theorem lehmann_completeness (H : E →ₗ[ℂ] E) (basis : OrthonormalBasis ι ℂ E)
    (energy : ι → ℝ) (heigen : ∀ i, H (basis i) = (energy i : ℂ) • basis i)
    (z : ℂ) (hz : ∀ i, z - (energy i : ℂ) ≠ 0) (v : E) :
    (∃! u : E, z • u - H u = v) ∧
    ∀ u : E, z • u - H u = v →
      ⟪v, u⟫_ℂ = ∑ i, (Complex.normSq ⟪basis i, v⟫_ℂ : ℂ) / (z - energy i)
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Fetter–Walecka, Quantum Theory of Many-Particle Systems, Lehmann representation.

### `stochastic_estimate` — definitional

**Python statement (verbatim):**

Finite sample averages of the specified observable are stochastic estimates, not deterministic variational bounds; no accuracy or confidence level follows.

**Python side conditions:** `sampled_evaluation`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** Classifies a finite sample mean; asserts no accuracy, unbiasedness or deterministic ordering. Sampling-law correctness is an implementation obligation.

**Reference:** Sample-mean definition; linearity of expectation.

### `expectation_evaluation` — definitional

**Python statement (verbatim):**

Exact evaluation of the specified observable in the specified state equals that expectation target; it does not equal the ground energy in general.

**Python side conditions:** `known_denotation`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** Equality to the specified state/observable expectation is the denotation and binding contract, not a theorem about the numerical evaluator.

**Reference:** Definition of the quantum expectation value.

### `rrho_evaluation` — definitional

**Python statement (verbatim):**

Evaluating the specified rigid-rotor harmonic-oscillator ideal-gas correction equals that model's correction at fixed temperature, pressure and frequencies; this does not assert equality to experimental thermodynamics.

**Python side conditions:** `thermal_contract`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** Equality to the fixed RRHO model is the denotation and binding contract, not a proof of physical accuracy or of the numerical partition function.

**Reference:** McQuarrie, Statistical Mechanics, ideal-gas partition functions.

### `no_ordering_law` — definitional

**Python statement (verbatim):**

A denoted approximation without applicable equality or order hypotheses is classified approximate. This asserts neither an error bound nor an ordering.

**Python side conditions:** `known_denotation`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** Conservative classification: absence of certified equality/order yields approximate, without asserting that an ordering is mathematically impossible.

**Reference:** ChemFlow relation calculus: conservative forgetting rule.

### `rayleigh_ritz_excited` — cited

**Python statement (verbatim):**

For Hermitian H and nontrivial V subset S, each ordered Ritz root k>0 with k<dim(V) is >= the corresponding sector eigenvalue; when V=S all roots agree.

**Python side conditions:** `hermitian`, `basis_in_sector`, `spin_supported`, `excited_roots`.

**Lean statement:** none; this rule is not reported as machine-checked.

**Lean note:** The higher-root min-max theorem and full-spectrum equality are cited, not formalized here.

**Reference:** Courant–Fischer; Hylleraas–Undheim (1930); MacDonald (1933).

### `weinstein_bound` — machine_checked

**Python statement (verbatim):**

For Hermitian H on a finite-dimensional complex inner product space and unit psi, with mu=Re< H psi,psi >, some eigenvalue lambda satisfies |lambda-mu| <= sqrt(||H psi||^2-mu^2). This does not identify a ground eigenvalue.

**Python side conditions:** `hermitian`, `normalized_trial`, `exact_moments`.

**Lean statement:** `ChemflowFormal/Weinstein.lean`

```lean
theorem weinstein_bound [FiniteDimensional ℂ E] (H : E →ₗ[ℂ] E)
    (hH : H.IsSymmetric) (ψ : E) (hψ : ‖ψ‖ = 1) :
    let μ := Complex.re ⟪H ψ, ψ⟫_ℂ
    ∃ eigenvalue : ℝ, HasEigenvalue H (eigenvalue : ℂ) ∧
      |eigenvalue - μ| ≤ Real.sqrt (‖H ψ‖ ^ 2 - μ ^ 2)
```

**Lean note:** Finite-dimensional exact mathematics; numerical binding contracts and floating-point side-condition checks remain trusted.

**Reference:** Weinstein residual bound.

## Restricted sectors and the Lehmann composition

The invariance theorem can be applied separately to each number operator. A
vector in their joint eigenspace remains in each eigenspace, hence in the
intersection. Mathlib's `IsSymmetric.restrict_invariant` supplies a symmetric
restriction. The variational and Ritz theorems can then be instantiated with
the sector as the ambient inner product space. The following explicit helper
proves the sector variational conclusion:

```lean
theorem sector_variational [Nontrivial E] (T : E →ₗ[𝕜] E) (hT : T.IsSymmetric)
    (V : Submodule 𝕜 E) [Nontrivial V] (hV : ∀ v ∈ V, T v ∈ V)
    (x : V) (hx : x ≠ 0) :
    ground (T.restrict hV) ≤ rayleigh (T.restrict hV) x
```

The Lehmann theorem's docstring specifies the physical composition: use
`v = c†|0⟩` for the particle sector and `v = c|0⟩` for the hole sector, with the
appropriate energy-shifted Hamiltonians and denominator signs. This formalizes
the finite resolvent identity, not the second-quantized implementation of the
ladder operators, reference energy, spin conventions, or frequency grid.
Those remain obligations of the response binding contract.

## Deliberate scope changes from V1a

1. **exact_diagonalization:** replaced the claim about an algorithm returning
   the ground energy with the spectral theorem that the Rayleigh infimum is the
   least eigenvalue. The solver-return claim belongs to the existing binding axiom.
2. **rayleigh_ritz:** restricted the machine-checked rule to the lowest root.
   Higher-root ordering is now the separate cited `rayleigh_ritz_excited` rule.
3. **rayleigh_ritz_full:** the formal statement now assumes `V = ⊤` and proves
   lowest-root equality. The determinant-count argument establishing full span
   stays in the trusted witness predicate; full-spectrum equality is included
   in the separate cited rule.
4. **sector_decomposition:** commutation is an explicit premise, not a proved
   consequence of the Python encoding of a spin-free number-conserving Hamiltonian.
   Invariance itself is proved. The kernel's construction checks are trusted.
5. **lehmann_completeness:** made the finite orthonormal eigenbasis and resolvent
   hypotheses explicit and isolated the resolvent identity. The numerical
   particle/hole/ladder composition remains an implementation contract.
6. **stochastic_linear_combination:** explicitly a classification rule. It does
   not claim unbiasedness or prove expectation linearity without integrability
   and sampling-law hypotheses. No independence assumption is needed for the
   usual integrable expectation-linearity theorem, which is not formalized here.

There is no weakening of the mathematical variational or finite-sum claims.
`weinstein_bound` is new in the table: it bounds distance to **some** eigenvalue,
not necessarily the ground eigenvalue, and is not an absolute ground-energy
interval certificate. Its side conditions are not fabricated as a new proof
rule for numerical intervals.

## Prover and kernel behavior

`Verdict.machine_checked` and `Verdict.trusted` are sorted tuples partitioning
the unique lemma/axiom names used by a valid certificate. Invalid verdicts have
empty tuples. Implementation axioms, definitions and citations remain in
`trusted`; the mathematical entries appear in `machine_checked` only after
successful manifest integrity checks.

An exported `Spectrum` carries its complete projected resolution, even when
only its first displayed root was requested. When its basis has more than one
vector, the prover emits an `excited` step citing `rayleigh_ritz_excited` before
the Ritz result. The kernel independently recomputes this fact and requires
that step as a dependency of both partial and full Ritz results. This conservative
policy also covers resolutions subsequently consumed by a Lehmann certificate.
Ground-energy-only SCI/SQD values do not acquire an unnecessary excited-root claim.

The runtime checks the complete source hash map, the root import file, toolchain,
Lake configuration and lockfile, Mathlib revision, theorem name and source file,
the verbatim declaration statement, the Python lemma metadata hash, and the
allowed axiom list. Missing, stale or mismatching evidence rejects the certificate;
there is no fallback that labels an unaudited theorem machine-checked. Runtime
checking does not invoke Lean and does not cache away subsequent file changes.

`formal/manifest.json` records these checks' inputs and audit timings.
`formal/Audit.lean` is the generated `#print axioms` input and
`formal/axiom-audit.txt` contains the actual output. Hashes detect stale or
inconsistent artifacts; they are **not** a cryptographic signature defending
against an attacker replacing both the manifest and its sources. Re-run the
Lean audit when accepting a new source/manifest pair.

## Trusted base

- Lean's kernel and its standard logical axioms; the installed Lean toolchain,
  pinned Mathlib sources/compiled dependencies, and the local build/audit process.
- The Python checker, manifest loader, and artifact-to-mathematics interpretation.
  The Lean proofs do not prove the correctness of Python execution.
- Every `binding:*` axiom actually listed in a certificate: diagonalization,
  projected solve, state preparation/evaluation, sample generation, linear
  combination, response/ladder construction, or model evaluation as applicable.
- The cited higher-root min-max/full-spectrum rule and the five explicitly
  definitional classifications when used. Their precise notes appear above.
- Floating-point arithmetic, artifact integrity and tolerance-based predicates:
  Hermiticity, normalization, orthogonality, spin selection, basis coverage and
  exact-moment interpretation are not converted into rigorous error bounds by
  these Lean theorems. Input-data correctness and physical-model validity also
  remain outside the formalization.

## Re-running offline

From this worktree (packages must already exist under `formal/.lake/packages`):

```sh
PYTHONPATH=$PWD/src env -u DYLD_LIBRARY_PATH \
  .venv/bin/python \
  scripts/formal_audit.py

(cd formal && lake build && lake env lean Audit.lean)

PYTHONPATH=$PWD/src env -u DYLD_LIBRARY_PATH \
  .venv/bin/python \
  -m pytest -q -p no:cacheprovider

PYTHONPATH=$PWD/src env -u DYLD_LIBRARY_PATH \
  .venv/bin/python \
  -m ruff check .
```

The audit script runs in the foreground, calls only `lake build` and
`lake env lean Audit.lean`, removes an old manifest before building, rejects
nonstandard axioms, and checks that sources did not change during the audit.
It never calls `lake update` or a package-fetch command. No new dependencies
are required. Keep the generated manifest and audit evidence with the source
changes. Reviewer tests are not modified; additional integrity tests exercise
source, theorem, metadata and axiom mismatches and missing excited-root citations.

## Validation on 2026-10-15

- Audit invocation: incremental `lake build` completed in **2.788 seconds**;
  `#print axioms` audit completed in **47.038 seconds**. These are the measured
  audit-run timings, not a claim about a clean rebuild of all Mathlib dependencies.
- All nine audited theorems use only `propext`, `Classical.choice`, and `Quot.sound`.
- Focused V1d and certificate tests: **37 passed** in **38.75 seconds**.
- Full prescribed pytest command: **1504 passed, 11 skipped** in **398.66 seconds**.
- Full interpreter-provided `ruff check .`: **All checks passed**.
- Every existing `tests/test_review_*.py` SHA-256 matches the pre-edit snapshot,
  including the initially untracked V1d reviewer test. No reviewer test was edited
  or judged incorrect. No commit, network operation, or dependency addition was made.
