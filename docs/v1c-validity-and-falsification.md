# V1c validity domains and V1b falsification

Implemented on `v1bc`, following the signed-task A1c rule. The user supplies
accuracy bars and validity domains; neither designer nor critic can select them.
The concrete reviewer contracts supersede the exploratory V1b/V1c examples in
`research/07-verified-design.md`: accuracy failures belong to validity mapping,
not to falsification of an approximate method's mathematical relation.

## Signed format and tool interfaces

```json
{
  "validity": [
    {"axis": "geometry_scale", "range": [0.7, 2.5], "points": 7, "resolution": 0.1}
  ]
}
```

`TaskSpec.validity` is an immutable tuple of validated axes. Each range has two
finite endpoints, lo < hi, and contains the signed task point. There are 3–33
initial points and 0 < resolution < hi − lo. Duplicate axes are rejected.
Supported axes are:

| Axis | Target | Signed point / transformation |
| --- | --- | --- |
| `geometry_scale` | Molecular ground energy | 1.0; scale every coordinate about the first atom, preserving the original length unit, charge, basis, active space and sector |
| `geometry_scale` | Reaction energy | 1.0; scale each species about its own first atom, retaining each signed species active space, sector, charge, spin, length unit and reaction coefficient |
| `hubbard_u` | Lattice ground energy | Signed lattice U (default 4); replace U in the signed lattice's unit |
| `impurity_u` | Impurity spectral function | Signed impurity U; replace U, retaining baths, sector, observable, frequency grid and broadening |

Validity is included in the canonical signed content and changes its hash.
The canonical schema includes an empty validity tuple when omitted. Records signed
with the old serialization therefore need fresh human approval rather than an
automatic signature migration; mapping tools never rewrite signatures.
Previously Pydantic silently ignored extra task fields. `TaskSpec`, its typed
targets, gates, reaction species/thermochemistry and validity axes now use
`extra="forbid"`. Open dictionary-valued model parameters remain dictionaries:
their contents are preserved in the signature, not silently discarded.

Both designer and critic receive these strict tools:

```json
{"name": "algorithm.validity", "arguments": {"plan_id": "<handle>", "node_id": "<node>"}}
{"name": "algorithm.falsify", "arguments": {"plan_id": "<handle>", "node_id": "<node>"}}
```

Their argument schemas contain exactly `plan_id` and `node_id`. `algorithm.validity`
is offered only when the frozen task declares a validity domain, analogous to the
target-support filtering for `algorithm.verify` (R118). Static plugin allowlists
retain the capability; task-bound schemas and MCP listings filter it. Direct calls without
a signed domain or accuracy bar are refused. Effects, signature and target identity
are checked before mapping. The selected dependency closure and a fresh reference
closure from the domain's `plan.base` route are executed separately. The oracle
has its own executor/cache and shares only the artifact store and task ledger.
No human task is re-signed. Derived instances carry `derived_from` (the signed
task content hash) and `instance_hash`; low-level IR freezing is an execution
integrity check, not creation of a new signed task.

All executions, including exact references, charge the existing task ledger.
Independent variance/projection checks charge measured wall time too. Persistent
tools use the existing run and file locks. Budget refusal is not a clean report.

### R132–R134 corrections

Reaction targets use `geometry_scale` both for signed validity and, without a
signed domain, as the default falsification family (range 0.5–3.0). Reference
construction leaves active-space selection to the reaction pack, which uses each
species' signed space; the first candidate Hamiltonian's space is not the reaction
aggregate. Derived geometries do not alter the frozen task. Interaction-energy
targets remain unsupported and may refuse with `No falsification family`.

R133 originated in the independent CPU state checks: their bare `CostRecord`
left QPU dimensions unknown, so R99 correctly refused them when the signed QPU
budget was zero. Those checks now use the existing `cpu_numeric` resource defaults
for both forecasts and measured records, with measured wall and CPU-core time.
GPU time, simulator shots and QPU time/shots are known zero for these checks.
Execution resource models continue to account for emulator simulator shots while
CPU/emulator backends declare zero QPU use. No budget exemption is introduced:
unknown or positive use of an exhausted resource still refuses execution.
Regressions cover molecular, lattice (`hubbard_u`) and impurity (`impurity_u`)
families, classical and available SQD emulator routes, `algorithm.prove`, and
impurity `algorithm.verify` including its automatic falsification pass.

## Validity map

The response contains `node`, `method`, `bar`, `axes`, `task_point`, `assumption`,
`task_hash` and measured `wall_s`. Energy bars/errors are in hartree; spectral
errors are the integrated L1 metric on the signed grid, dimensionless.
Each axis contains its signed settings plus:

- `points`: sorted records with `value`, `result`, `exact`, signed `error` for
  energy (nonnegative L1 error for spectra), `status`, `certified_bound`, relation
  and derived-instance provenance;
- `regions`: adjacent `{lo, hi, status}` intervals covering the declared range;
- `counterexamples`: `{value, error}` for sampled accuracy failures.

The initial points are evenly spaced, and the signed task point is evaluated
explicitly if necessary. Adjacent unequal statuses are bisected until their gap
is at most the signed resolution. The returned points include these extra samples.
If the requested resolution is below floating-point spacing at a transition,
the tool refuses rather than looping forever or claiming the tolerance was met.
Equal neighboring intervals are merged. Shared endpoints have the status in their
point record; region labels describe the interiors between samples.

| Status | Meaning |
| --- | --- |
| `certified` | Sample error passes and the conditional variance bound is no larger than the bar, with independent state checks passing; both bounding samples meet this rule for a region |
| `sampled_valid` | Absolute error passes at the sample / both bounding samples, without that certificate |
| `invalid` | Sampled error exceeds the signed bar; equal-status bounding samples label an invalid region |
| `unknown` | Transition bracket with differing bounding statuses |

**Every region label assumes no unobserved status reversal between samples.**
This is stated in the map, not hidden behind the word “certified.” The map is not
an interval-arithmetic proof or a theorem that every interior point passes.

### Weinstein bound and state witnesses

For a normalized state ψ in the fixed-particle sector, construct that sector's
physical Hamiltonian independently from its integral artifacts. Compute

```
μ = ψ† H ψ
σ = ||Hψ − μψ|| = sqrt(ψ† H² ψ − μ²)
```

The residual form avoids cancellation in subtracting nearly equal second moments;
it is the same variance, without sampling H² or truncating Hψ to the selected
subspace. Evaluation is floating-point numerical linear algebra, not outward-rounded
interval arithmetic. Sigma bounds distance to **some** eigenvalue. Identifying
that eigenvalue as the desired ground state is an additional assumption, recorded
in `assumption`; an upper-bound relation alone does not establish it.

SCI (including polarized sectors) and SQD export their actual solver state as
content-addressed `state_ref` artifacts in full-sector determinant ordering. VQE
and ADAPT already export states. Ritz uses its returned subspace eigenvectors,
embedded into the full sector. No bound is invented by diagonalizing a new state
in place of the returned state. CCSD(T), other methods without states, and results
without a proved variational/equality relation get null bounds. Shot evaluations
do not inherit a deterministic Rayleigh/variance certificate. Dense independent
checks are limited to 4096 sector determinants; exceeding that limit is refused,
not silently labeled certified. Inconclusive independent spin resolution is also
refused rather than called clean.

## Falsification

Responses contain `verdict` (`clean` or `falsified`), all tested `instances`,
`generators`, `seed`, `counterexamples`, `refuted_axioms`, `program_hash`,
`task_hash`, `domain`, `wall_s`, `assumption` and `report_id`.

The domain is the signed validity declaration when present. Otherwise only these
documented one-dimensional defaults are used, with all other target parameters
held fixed:

| Family | Default domain | Default resolution |
| --- | --- | --- |
| Molecular geometry scale | [0.5, 3.0] | 0.025 |
| Hubbard U | [min(0, signed U), max(8, 2 × signed U)] in the model's unit | range width / 100 |
| Impurity U | [min(0, signed U), max(8, 2 × signed U)] in the model's unit | range width / 100 |

Each search evaluates endpoints, the signed point, midpoint, and **12 random draws
with seed 1729**, deduplicating coincident points. Endpoints include compressed and
stretched geometries or zero/strong interactions in the defaults; the midpoint
and signed point retain the family's geometry/bath symmetries. Signed domains are
never expanded to reach another edge case. No sector, bath or grid is secretly
changed. `generators` records `edge` and `random`; even a clean search persists all
its instances and observations.

Properties are independent of the accuracy bar:

- Certified equality: absolute energy or spectral L1 difference ≤ 1e-8.
- Certified variational upper bound: energy difference ≥ −1e-9.
- SCI/SQD/Ritz: independently rebuild the Hamiltonian projected on the returned
  determinant basis and compare its lowest compatible-spin eigenvalue (tolerance
  1e-8) with the returned energy.
- State-bearing variational implementations: state normalization and the Rayleigh
  quotient (energy tolerance 1e-8). Cheap checks also cover state-bearing dependencies
  of the root certificate, including the Ritz spectra in a Lehmann design.

The relation to falsify is taken from the signed task point and held fixed during
the search; a weaker certificate on a derived instance cannot hide a violation.

Each observed failing property is shrunk from the closest failing sample on each
side toward the signed point by bisection. Reports include the failing instance,
property, observed/expected values, and `shrink_bracket` of width ≤ resolution.
This is a local shrink: it cannot exclude a disconnected, unsampled failure
closer to the task point. A task-point failure cannot shrink closer than itself.

Reports use the A1b `DesignStore` canonical payload and content-addressed storage.
Only relevant cited implementation axioms are named as refuted; mathematical
lemmas are not refuted by an implementation counterexample. Refutations survive
later runs/proofs and reopening the same task's store. Historical result records
receive a persistent refutation overlay rather than rewriting the original
content-addressed evidence.

`algorithm.register` requires both the existing passing verification and clean
falsification for the same candidate-closure program hash and task hash. Any stored
falsification for that program blocks registration. To retain the existing
verify→register workflow, successful `algorithm.verify` automatically performs
the budgeted falsification search; callers can also invoke it explicitly.

## Orchestration and R129

When validity is declared, normal runs map all available target-operator results:
screened candidates, controls, references and the eventual answer. Candidate items
include the task-point `validity_status`; critic escalation refuses invalid,
unknown, declared or refuted candidates. Recorded headlines include maps;
`report.md` has a Validity section for each mapped headline, with the bar, ranges,
counterexamples and assumptions. `summary.json` carries these maps both in result
headlines and in its `validity` list. Report linting recognizes the exact generated
validity section, not arbitrary unsupported certification claims.

For R129, `design.decline.plan_id` treats only literal `"null"`, `""`, or JSON null
as absent. Other strings remain ordinary plan handles and are validated normally.

## Measured H4 CCSD(T) reviewer example

Linear H4, signed spacings 1 Å, STO-3G, four active electrons/orbitals, singlet,
accuracy bar **0.0016 hartree (1.6 mHa)**. The signed geometry-scale range is
[0.7, 2.5], seven initial samples, resolution 0.1. Because the signed spacing is
1 Å, the numerical scale equals the interatomic spacing in Å in this example.

| Scale / spacing (Å) | CCSD(T) − FCI (mHa) | Status |
| ---: | ---: | --- |
| 0.700 | −0.002016284 | sampled_valid |
| 1.000 | −0.042947219 | sampled_valid |
| 1.300 | −0.505706567 | sampled_valid |
| 1.450 | −1.484731750 | sampled_valid |
| 1.525 | −2.420578531 | invalid |
| 1.600 | −3.804268247 | invalid |
| 1.900 | −15.058855107 | invalid |
| 2.200 | −29.056956686 | invalid |
| 2.500 | −35.112332292 | invalid |

The extra bisection points are 1.450 and 1.525. Regions are:

- **sampled_valid:** [0.700, 1.450];
- **unknown transition:** (1.450, 1.525), width 0.075 ≤ 0.1;
- **invalid:** [1.525, 2.500], under the stated interpolation assumption.

All CCSD(T) certified bounds are null. The task point is sampled_valid:
CCSD(T) = −2.166430395853476 Ha, FCI = −2.166387448634762 Ha.
The negative errors demonstrate that CCSD(T) is not a variational upper bound.
This is an accuracy failure, not falsification of its `approximate` relation.

Measured locally on 2026-10-14 with the supplied interpreter, fresh toolsets,
no LLM/network calls, elapsed tool-call wall times:

| Reviewer case | Seconds | Result |
| --- | ---: | --- |
| H4 CCSD(T) validity | 0.3787 | Nine points; transition [1.450, 1.525] |
| H4 SCI validity | 0.2535 | Maximum sigma 1.81 × 10⁻¹⁴ Ha |
| H4 correct SCI falsification | 0.5426 | clean; 16 instances recorded |
| H4 injected SCI bug falsification | 0.7692 | falsified; 23 instances including shrinking |

The injected −0.002 Ha bug activates below 1.30 Ha nuclear repulsion. The shrunk
relation counterexample is scale **1.7990601971**, with relation bracket
[1.7264183610, 1.7990601971]; observed error −0.002 Ha violates the upper bound.
The independent projection and Rayleigh checks also refute the implementation.
Timings are observations, not performance guarantees.

## What is not certified

- An unsampled interval, a domain outside the signed range, a different target,
  basis, bath, sector, geometry family or production size.
- Universal implementation correctness from a clean finite falsification search.
- The ground-state-identification assumption, monotonicity between samples, or
  globally minimal counterexamples in a disconnected failure set.
- Temple/gap bounds, a priori T1/D1 predicates, arbitrary multi-parameter families,
  sum-rule/size-consistency axioms without an implemented independent check,
  formal proof-assistant mathematics, or hardware/shot-confidence certificates.
- Correctness of PySCF, the shared integral artifacts, the independent determinant
  Hamiltonian builder, or numerical linear algebra itself. They remain trusted
  computational components, not formally verified foundations.

The immutable reviewer tests are not considered wrong and are not modified.
Dedicated regressions are in `tests/test_v1c_domains.py` and
`tests/test_v1b_falsification.py`. No dependencies or commits are added.

## Changed files

| File | Change |
| --- | --- |
| `src/chemflow/agent/task.py` | Signed validity axes, strict constraints, unknown-field rejection |
| `src/chemflow/agent/families.py` | Family instantiation, independent references, variance/projection checks, maps, generators and shrinking |
| `src/chemflow/agent/tools.py` | Tool registration, ledger locking, automatic mapping, registration guard, refutation propagation and R129 |
| `src/chemflow/agent/critic.py` | Validity and refutation escalation guards |
| `src/chemflow/agent/report.py` | Headline maps/statuses and qualified Validity sections |
| `src/chemflow/agent/cli.py` | Summary validity maps |
| `src/chemflow/agent/llm.py` | Designer instructions for validity, falsification and proof |
| `src/chemflow/agent/memory.py` | Persistent refutation overlays for historical results |
| `src/chemflow/design.py` | Content-checked falsification report lookup |
| `src/chemflow/impls/sci.py` | Export actual SCI solver states in sector ordering |
| `src/chemflow/impls/pyscf_impls.py` | Optional FCI state return for polarized SCI |
| `src/chemflow/ops/molecular.py` | Export actual SQD states |
| `src/chemflow/verify/integration.py` | Preserve previously refuted status during certification |
| `plugin/agents/designer.md` | Tool allowlist and bounded-claim guidance |
| `plugin/agents/critic.md` | Tool allowlist and escalation policy |
| `tests/test_v1c_domains.py` | Signed fields, lattice/impurity axes, bisection, budgets, state bounds and R129 |
| `tests/test_v1b_falsification.py` | Clean-instance persistence, default domains, budget refusal and historical/proof refutations |
| `docs/v1c-validity-and-falsification.md` | Formats, assumptions, measured examples, limitations and implementation report |

## Original V1b/V1c validation

On 2026-10-14, using the supplied interpreter and environment:

```sh
PYTHONPATH=$PWD/src env -u DYLD_LIBRARY_PATH .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff check src tests
```

- **1395 passed, 11 skipped**, 325.93 seconds, including all six V1c and four V1b
  reviewer cases. No reviewer tests were edited, deselected or newly skipped.
- Both Ruff invocations: **All checks passed**.
- `git diff --check`: passed. All **49** reviewer-file SHA-256 hashes unchanged.
- One earlier full run encountered an intermittent external AiZynthFinder SIGSEGV
  in `test_review_m6b.py::test_route_search_finds_the_textbook_paracetamol_synthesis`
  after the subprocess had found and saved the route. That test passed unchanged
  on isolated retry and in the final full run. No synthesis code or test was altered.
- Branch remains `v1bc`, HEAD remains `e2ec46d`; no commit, network research, live
  LLM call, dependency change, or lint-configuration change was made.

## R132–R134 validation

On 2026-10-14, on `r132` at `d290d30`, using the same supplied interpreter and
commands:

- Focused verification tests: **44 passed**, including all three R132–R134 reviewer
  tests and ten new regressions in `tests/test_r132_verification.py`.
- Updated plugin/MCP conformance tests: **5 passed**. Their expected task-bound
  listings now omit validity when the task declares no domain.
- Final full suite: **1 failed, 1456 passed, 11 skipped**, 282.79 seconds.
- The remaining failure is an immutable contract conflict:
  `tests/test_review_v1c.py::test_v1c_the_domain_comes_from_the_signed_task` requires
  `algorithm.validity` in the critic schema for `_h4(validity=False)`, whereas
  `tests/test_review_r132_134.py::test_r134_validity_is_offered_only_with_a_signed_domain`
  requires its absence for the same task. The implementation follows R134; neither
  reviewer test was edited, bypassed, or newly skipped.
- Both Ruff checks passed; all **52** reviewer-file SHA-256 hashes are unchanged.
- Original R133 failures were also reproduced on lattice and impurity tasks for
  both falsification and validity using the baseline family evaluator; the new
  regressions pass with the corrected evaluator and zero QPU budgets.
