# G5: whole-agent evaluation

G5 measures **ChemFlow agent runs**, not isolated operator benchmarks. Every
task is signed and passed directly to `chemflow.agent.cli.run`, the same entry
point as `chemflow agent run`. This includes the QCL screens (controls,
candidates and exact reference), candidate design, critic, reporting, journal and
budget ledger. The harness does not substitute its own chemistry execution pipeline.

## Offline scripted baseline

From the repository, with its existing environment:

```sh
export PYTHONDONTWRITEBYTECODE=1
EVAL_ROOT="$PWD/runs/g5-evaluation"
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval list
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite core --policy scripted --out "$EVAL_ROOT/core"
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite core --policy scripted --tasks h2-ground,h2-atomization \
  --out "$EVAL_ROOT/small"
env -u DYLD_LIBRARY_PATH .venv/bin/python scripts/eval_reference_values.py
```

Write evaluation outputs to an explicit, permitted path (inside this worktree
when filesystem restrictions require it). Leave previously generated `runs/g5-*` artifacts
alone. Use an empty or nonexistent output directory for each evaluation. Reusing
artifacts is refused, so prior journals and resource ledgers cannot contaminate
a new score. `--suite-file path.yaml` replaces `--suite core` for custom suites;
exactly one is required. `--tasks` selects comma-separated ids in **suite order**;
unknown, empty and duplicate ids fail before creating outputs.

The core suite uses existing PySCF/NumPy/SciPy implementations, and the existing
geomeTRIC reaction-geometry adapter for the thermo task. No dependencies are
added. There are no name-resolver, remote model, QPU or network tasks. The
scripted policy requires no provider keys and never creates a provider client.
The small core suite is intended to finish within a few minutes on a CPU.

## Core tasks and independent references

All molecular geometries are in angstrom, and all electronic calculations use
STO-3G. Accuracy bars come exclusively from the signed task, not the suite's
answer key (A1c). Every numerical task declares `abs_error_vs_exact`, or
`l1_error_vs_exact` for a spectral target. Core energy gates use 0.0016 hartree
(1.6 mHa chemical accuracy); the dimensionless spectral L1 gate uses 0.1.
Both the **chosen answer** and the independent screening-reference sanity check
use this same declared bar. Energy thresholds are converted to hartree;
spectral thresholds must be dimensionless. When multiple applicable error
gates exist, all must hold, so the strictest threshold applies. Missing,
incompatible, negative or nonfinite accuracy bars are rejected.
Molecular, lattice and selection tasks require at least a variational upper
bound, while reaction differences allow approximate relations. The spectral
task requires equivalence within the declared finite model.

| Task | Category | Independent route / scalar |
| --- | --- | --- |
| `h2-ground` | molecular | Direct PySCF RHF/FCI, H₂ at 0.74 Å, singlet 2e/2o |
| `h2-stretched` | molecular | Direct PySCF RHF/FCI, H₂ at 1.5 Å, singlet 2e/2o |
| `h2-compressed` | molecular | Direct PySCF RHF/FCI, H₂ at 0.6 Å, singlet 2e/2o |
| `h2-atomization` | reaction | Twice direct isolated H UHF/FCI minus H₂ FCI at 0.74 Å |
| `hubbard-ring6` | lattice | Independent dense 400-dimensional Fock-sector diagonalization; 6-site periodic ring, 3α/3β, t=1, U=4 hartree |
| `impurity-spectral` | spectral | Independent 16-dimensional full-Fock Jordan–Wigner matrices, sector diagonalization and Lehmann sum |
| `h2-counterpoise` | interaction | Direct FCI of collinear (H₂)₂ minus both ghost-basis H₂ fragment FCIs; bond lengths 0.74 Å, molecule origins separated by 3 Å |
| `hcn-hnc-enthalpy` | thermo | HCN → HNC ΔH(298.15 K), canonical RHF CASCI(4e,4o) with five frozen occupied orbitals, RHF geometries/frequencies and ideal-gas RRHO corrections |
| `h4-selection-1p5` | selection | Direct PySCF RHF/FCI, linear H₄ at 1.5 Å adjacent spacing, singlet 4e/4o |
| `h4-selection-2p0` | selection | Direct PySCF RHF/FCI, linear H₄ at 2 Å adjacent spacing, singlet 4e/4o |
| `h2-budget-refusal` | refusal | H₂ at 0.74 Å with zero wall-time budget; CF-E070 before execution. Its unused energy is independently recomputed for provenance. |

The eleven-task suite includes two selection tasks where the scripted SCI/SQD
candidates pass the ordinary 1.6 mHa gate and short-run VQE candidates fail it.
Tests inspect actual candidate gate outcomes, rather than merely asserting the
category label. At 2 Å every classical control also fails the gate, so selecting
a supported passing candidate should yield `escalate`; at 1.5 Å CCSD passes,
and either `stop` with a recommended control or `escalate` can be justified by
the critic's existing candidate-dominance rule. Other tasks specify
`decision_in` where appropriate. Candidate recipes and task gates are not
modified to make the suite pass.

For `impurity-spectral`, the legacy field `expected.value_hartree` contains a
**dimensionless scalar**, not an energy. Its accuracy bar comes from the task's
dimensionless L1 gate. The scalar is the trapezoidal integral of the impurity alpha spectral
density over the specified 101-point [-4,4] hartree grid, with eta=0.1 hartree.
The model has U=2, impurity onsite energy -U/2, one bath orbital at zero and
hybridization 0.5 hartree. The finite-window integral is approximately 0.983,
not the infinite-window sum rule of one. The harness reads the `sum_rule`
scalar already stored in `results.sqlite`, tied separately to each answer or
reference result id, node id and signed record hash. For this task `answer_value`
is also dimensionless. It never tries to parse an energy out of “spectral
density (101 points)”. The signed L1 gate additionally checks the full curve;
the scalar score alone cannot distinguish all spectral shapes.

The independent reference script imports no ChemFlow implementation or
reviewer-test helper. For thermo it minimizes both linear bond lengths with
SciPy Nelder–Mead, rather than using ChemFlow's geometry pipeline, then calls
PySCF directly for the Hessian, harmonic frequencies and CASCI. It independently
adds linear-molecule enthalpy corrections (7kT/2 plus harmonic vibrational
energy including ZPE). `reference_source` in the YAML records each route.
The script recomputes **every** task and checks stored values to 1e-8;
it exits nonzero on discrepancies or missing reference routes. `--print-values`
prints newly calculated numbers without modifying the suite.
This numerical reproducibility check is not an agent accuracy bar; only the
task's declared gate sets the scoring threshold.

## Full benchmark and parallel runs

`full` is the broader **55-task** evaluation; `core` remains the eleven-task
harness smoke test: 13 easy, 19 medium, and 23 hard tasks. Every full task has a difficulty
tag, independently recomputable `reference_source`, and its own signed accuracy
gate. Difficulty is optional for older/custom suites, but required for `full`.

| Category | Tasks | Coverage |
| --- | ---: | --- |
| Molecular | 16 | H2 compression/stretching and triplet; He, Li, LiH, H2O, H5, HF, NH, HeH+; singlet/doublet/triplet; STO-3G/6-31G; full and frozen-core spaces through 2e/10o |
| Reaction | 8 | Six H2 atomizations across geometries/bases, 2H2 to H4 assembly, frozen-core LiH atomization |
| Thermo | 5 | HCN to HNC enthalpy/Gibbs; 200–1200 K, 10,000–1,000,000 Pa; independently optimized RHF geometries and harmonic RRHO |
| Lattice | 7 | Hubbard chains/rings, 2–8 sites, weak/strong U/t, half/doped fillings, spin imbalance, hartree/eV |
| Spectral | 4 | One/two-bath impurities, weak/strong U, asymmetric bath, alpha/beta channels, impurity/bath orbitals, varied broadening |
| Interaction | 5 | Collinear/parallel/crossed H2 dimers, stretched bonds, counterpoise on/off, He dimer in 6-31G |
| Selection | 6 | H4 spacings 1.3, 1.4, 1.5, 1.7, 1.8, 2.0 Å; actual SCI/SQD passes and short-run VQE failures |
| Refusal | 4 | Zero wall time, zero CPU-core time, insufficient shot budget for the scripted bundle, no permitted effects |

Hard tasks stress stretched bonds, open shells, large active spaces, strong
correlation, and high-temperature/free-energy conditions. They are not filtered
out when the current agent cannot answer: failures belong in the benchmark.
The reference script uses declared molecular active spaces, direct Hubbard FCI,
dense full-Fock impurity eigensystems and ghost-basis fragment FCIs. Thermo uses
independent two-bond Nelder–Mead optimization plus direct PySCF Hessians/CASCI
and independent linear-rotor ideal-gas RRHO expressions. For these 1:1 isomers,
pressure cancels from the reaction free energy; varying it tests plumbing, not
a pressure-dependent equilibrium shift.

Full energy gates range from 0.0001 to 0.003 hartree: tighter for weak
interactions, 0.0016 for most small-molecule/reaction/selection tasks, and
0.003 for larger correlated models. Spectral L1 gates are 0.05 or 0.1.
These are model-specific accuracy targets, not errors against experiment.
The spectral scalar stored in the legacy `value_hartree` field is actually
dimensionless integrated weight; the pipeline's spectral L1 gate additionally
checks the curve, as described for core.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
mkdir -p "$PWD/runs/eval-tmp"
export TMPDIR="$PWD/runs/eval-tmp"
env -u DYLD_LIBRARY_PATH .venv/bin/python scripts/eval_reference_values.py --suite full
env -u DYLD_LIBRARY_PATH .venv/bin/python scripts/eval_reference_values.py --suite core
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite full --policy scripted --out "$EVAL_ROOT/full-sequential"
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite full --policy scripted --jobs 4 --out "$EVAL_ROOT/full-parallel"
```

`--jobs N` must be a positive integer and defaults to 1. Each concurrent task
runs in a fresh Python **process**, with its own output directory, database,
journal, ledger and provider clients. A bounded thread pool only supervises
subprocesses; chemistry and agent execution do not run in its threads. This
avoids macOS semaphore restrictions on `multiprocessing` pools. Signing stays
in the parent; workers receive only signed tasks and the public key. Results
remain in suite order, regardless of completion or `--tasks` argument order.
`agent.log` preserves each task's stdout/stderr, including errors that prevent
a summary from being written. Worker-launch/protocol errors fail the command;
ordinary agent failures remain scored rows and do not stop subsequent tasks.

### Provider quota stops

A provider quota or usage-window stop ends evaluation scheduling cleanly rather
than retrying every remaining task. This includes GLM code `1308`, exhausted
balance codes such as `1113`, and usage-window messages (`使用上限`, `usage limit`,
or `limit will reset`). Ordinary throttling, such as code `1302`, remains retryable.
The agent records `LLM_FAILED` with `quota: true` and `reset_at` when the provider
message contains a reset timestamp. Timestamps are preserved as supplied; no
timezone is inferred.

After the first quota-stopped task, neither sequential nor `--jobs N` runs start
new tasks. Already-running workers finish normally. Results remain in suite
order, with one row per selected task. Unstarted tasks have `decision: "not_run"`,
`valid: false`, `completed: false`, `correct: false`, and an `error` naming the
provider quota and reset time (or stating that the reset time is unknown).
They have no LLM calls and are not themselves marked `llm_failed`; their task
directories contain the signed input but no agent outputs. The quota-stopped
task retains its diagnostic scores and is invalid because of `LLM_FAILED`.

`aggregate.not_run` counts unstarted tasks, and `report.md` states that the run
stopped on the provider quota and when it resets, if known. `results.json` and
`report.md` are still written, and the command exits with status **0**. Unstarted
rows are excluded from policy accuracy, just like other invalid rows. The
harness does not wait for the window to reset or automatically resume; a later
run must use a fresh output directory.

Scripted decisions must not depend on noisy microsecond timings. Agent
observations retain measured `cost`. For ground-energy solver selection they
also expose a reproducible `selection_work` heuristic: spatial-orbital count
to powers 3/5/6/7 for HF/MP2/CCSD/CCSD(T), or squared alpha/beta-sector
determinant dimension for SCI/SQD/FCI. The scripted critic's `decision_cost`
compares this work proxy plus simulator/QPU shot counts, **not measured time**.
An unsupported work proxy remains unknown and cannot establish dominance.
These asymptotic rankings ignore prefactors, sparsity, iterations and cache
hits: they are an explicit deterministic baseline-policy heuristic, not
calibrated resource bounds or evidence of a speedup. Classical controls may
still win (for example CCSD(T) versus a large N2 CI space), unlike simply
discarding all cost comparisons. Gates, evidence and supported relations
still decide eligibility; no-candidate control selection retains its stable
error/method/id ordering. Actual timings and budget enforcement are unchanged.
This makes scripted rows reproducible across process counts, except timing;
it does not promise reproducible live model responses or equivalence when
timing budgets are close enough to be exhausted under load.

### Refusal interpretation

Wall/CPU-zero tasks cannot begin numerical execution. The effect task permits
no effects, so the screen is refused with CF-E120 before any plan runs. Only that
structured effect refusal is recorded as an expected refusal: other CF-E120 errors
(bad signatures/target identities) are still failures. The expected denial remains
visible as an integrity event.

**Shot-budget limitation:** no nonnegative shot budget can rule out *every*
current candidate, because SCI uses zero shots. The 2048-shot task instead
requires refusal of the scripted batch containing 1024- and 2048-shot SQD
candidates (3072 total), before any shots are spent. Earlier classical controls
may consume wall time. This tests aggregate reservation rather than proving
the target unsolvable under its budget. A live agent that proposes a smaller
or shot-free plan may legitimately answer, despite not matching this task's
scripted refusal expectation. A universal shot-only refusal would require an
additional task constraint excluding shot-free methods; the current TaskSpec
does not express that constraint. Do not interpret this row as a universal
policy-safety score.

### Comparing policies

Use the **same suite revision, task ids, gates, roles, thread settings, and
resource limits**, with distinct empty output directories. Compare answer
correctness, completion, gate/relation/decision validity, integrity events,
tool/LLM calls and resource usage separately, stratified by category and
difficulty. Exclude expected refusals from energy-answer denominators and
report their matching-code/no-consumption rate separately. Do not substitute
`reference_correct` for answer correctness, drop failed tasks, or silently
loosen gates. Report the shot-bundle caveat above when comparing live policies.

For a configured OpenAI-compatible GLM provider, the existing live-policy
environment/explicit model requirements still apply; add `--jobs N` to the
ordinary `--policy openai --model MODEL --roles designer` evaluation command.
Choose concurrency within the provider's rate limits; each job can create its
own clients. Record model/roles, suite/commit, process count, and repeated-run
variation. No live calls are needed to validate the suite. See the recorded
offline measurements and all unanswered cases in `g5-full-baseline.md`.

## Scores and artifacts

Each `<out>/<task id>/` contains the signed `task.json` and normal agent outputs:
`summary.json`, `journal.jsonl`, `results.sqlite`, report, plans and ledger.
At the root, `results.json` contains `metadata`, `rows` and `aggregate`, and
`report.md` displays every row and the aggregate, policy/model/roles and git
commit. Metadata also records end-to-end harness `elapsed_s`; this differs
from ledger wall time, which excludes some orchestration and setup overhead.

`summary.json` records `candidate` verbatim from `CriticDecision.candidate`.
The harness selects only that node's result, including a recommended classical
control on `stop`. A null, missing, or ambiguous choice has no answer. There is
**no fallback** to the reference, the best passing candidate, the cheapest
control, the sole headline or a number inferred from the critic's reason text.
`screening_reference` is checked separately as `reference_correct`, never as a
substitute for agent correctness.

| Field | Meaning |
| --- | --- |
| `id`, `category` | Validated suite task identity |
| `difficulty` | Full-suite difficulty; optional and absent on legacy core rows |
| `completed` | The chosen answer has a usable finite scalar, or a refusal task reached a recorded refusal |
| `answer_node`, `answer_value` | Critic's candidate node id and that result's scalar; null value for absent/unusable choices; energies converted to hartree |
| `correct` | Chosen-answer absolute error is within the task's declared error gate; for refusal tasks, the requested refusal code and no-consumption evidence match |
| `reference_correct` | Screening-reference scalar agrees with the independent expected value using the same task error bar; false when no reference was computed |
| `gate_passed` | Chosen answer's gate verdict is `PASS`, not the reference's or every candidate's verdict |
| `relation` | Selected headline's recorded relation |
| `relation_ok` | Null when there is no answer; otherwise a valid relation satisfying every `spec.gates` entry of kind `relation_at_least` through ChemFlow's existing partial-order rules; without a minimum, any recognized relation qualifies |
| `design` | Latest journal design outcome: `screened` (`SCREEN`), `declined`, `refused`, or null |
| `answer_from_candidate` | True when the chosen answer node occurs among a `SCREEN` entry's outputs |
| `integrity_events` | CF-E060, CF-E120, CF-E130 found anywhere in journal entries; each code counted once per entry, repeated events retained |
| `decision` | Agent summary decision (`stop`, `escalate`, etc.); `failed` if no summary |
| `decision_ok` | Membership in `expected.decision_in`, or null when no decision expectation is declared |
| `refusal_code` | Recorded structured refusal code when decision is `refused`, otherwise null |
| `wall_s`, `sim_shots`, `qpu_s`, `qpu_shots` | `summary.json`'s `spent` values; missing/unknown resources remain null, never inferred from elapsed time or summed headline costs |
| `llm_calls` | Number of `LLM_CALL` plus `LLM_ERROR` journal entries, including malformed-response attempts |
| `llm_failed` | True if any role journaled `LLM_FAILED`, even if later controls produced a correct answer |
| `valid` | False for LLM failures; only valid rows contribute to policy accuracy |
| `tool_calls` | Total lengths of journal entries' top-level `tool_calls` lists; proposed-but-unexecuted calls inside LLM responses are not counted |

Aggregate fields sum completion, answer correctness, reference correctness,
true decision expectations, gates, relations, integrity-event
occurrences, calls and resources, plus `n`. A resource total is null if any
task's corresponding value is unknown. Core tasks explicitly budget all four
reported resources, with zero QPU budgets. A normal agent failure is scored
and subsequent tasks continue; input/configuration failures exit nonzero.
Incorrect answers do not make the evaluation command fail: inspect the scores.

`relation_ok` counts only true verdicts; `relation_rated` counts non-null verdicts.
`design_screened`, `design_declined`, `design_refused` and `answer_from_candidate`
count the corresponding row outcomes. All appear in `report.md` alongside the
existing totals; control fallback correctness is not candidate success.

### Designer protocol

The designer reads `observation.controls`: the graded classical controls and exact
reference of the first screening round, which it cannot change. `methods.list` gives
the candidate templates and their ranges, and `screen.review {candidates}` compiles a
proposal into the next round's QCL plan without running it, reporting the inferred
relation of each output, the predicted time and the simulator shots. Candidates are
typed: `sci {select_cutoff}` (1e-8 to 0.1) and `sqd {shots}` (1 to 1 000 000;
closed-shell species only), at most eight distinct ones. They apply to ground-energy
targets and componentwise to reaction energies, barriers, rankings, scans, bond
dissociation and IP/EA species. SQD keeps `seed=11`, `n_batches=2`, `max_iter=2` and
LUCJ preparation from CCSD amplitudes. These are screening recipes, not accuracy
guarantees; the signed gates and budget still apply.

The designer finishes with `design.submit {candidates}` or `design.decline {reason}`.
Either stops the turn immediately, including any later calls in the same response.
Submitted candidates are validated again by the trusted screen builder; an invalid
proposal records `DESIGN_REFUSED` with the reason and the run keeps the controls. A
decline records `DESIGN_DECLINED`. Without either terminal tool, `DESIGN_REFUSED` names
the ending: text reply, consecutive tool errors, turn limit or `LLM_FAILED`. No agent
output is executed as code: the builder turns typed candidates into one more QCL plan.

Model-facing parameter schemas inline local definitions and references while
retaining closed objects and required nullable fields. Object/array argument fields
may arrive as JSON strings: they are decoded once if they encode the expected
container type. Invalid JSON remains a structured validation error. The tool-call
journal preserves the original arguments, not the decoded copy.

### Modeler protocol

Declared molecular active spaces, spaces derived from lattice/spectral sectors or
reaction species, and full-molecular-space tasks do not consult the LLM modeler.
`MODEL` records the fixed choice and its reason, with no modeler provider call.
For `active: auto`, the modeler calls `model.choose {n_elec, n_orb, reason}`. This
modeler-only tool ends the turn immediately; the driver checks positive
integer counts, the 10-orbital screening cap, electron capacity and spin parity.
Without a choice, or with an infeasible choice, it uses the diagnostics'
`suggested_active` and records `MODEL_FALLBACK` with a reason. The diagnostic fallback must itself pass the feasibility checks.
Free text is never parsed as a model choice.

Planner prose is advisory: the original task still goes to the human approver.
Critic prose does not replace the deterministic review verdict. Analyst prose is
journaled, while the final report is generated from recorded results. None of
these roles' free text is parsed into task-critical data.

`invalid` counts rows with `valid: false`; `correct_valid` counts correct answers
among valid rows only. The report summary uses `correct_valid / (n - invalid)`
and explicitly shows excluded rows, including when no valid rows remain.
Raw `correct`, `reference_correct`, other diagnostic totals and resource costs
are retained for auditing, not for attributing fallback decisions to an LLM.

For `expected.outcome: refused`, `refusal_code` is mandatory; numeric reference
fields are optional. The budget task sets wall time to zero, so the screen's
budget preflight raises CF-E070 without executing any node. The existing error gains
`nothing_consumed=True` metadata; execution/selection behavior is unchanged.
The agent writes a structured `REFUSED` journal entry and summary with node,
code, no-consumption evidence and a resource snapshot, then retains its nonzero
agent-command exit status. The harness continues and scores the refusal.
It requires the matching code, positive no-consumption evidence, a known finite
resource snapshot and final spending identical to that snapshot. Earlier
legitimate work is allowed, but spending beyond the refused node is not. The
core refusal starts before any execution and all four resources are zero.
An error-code string alone, absent summary, unknown resources or later spending
cannot earn correctness. A refusal is not a passed energy gate.

Correctness, gates, relation and integrity are separate dimensions. A run can
have a correct reference and still contain rejected or unsafe agent actions.
Likewise `stop` can be a legitimate decision, not a run failure. The reference
score is a **pipeline sanity check, not an agent answer score**. When no candidates
remain, the critic selects the lowest-error gate-passing classical control.
Controls within `1e-9` of the minimum error are ordered by method:
`hf < mp2 < ccsd < ccsd_t < fci`, then by node id. Unknown methods sort after
the ladder. Summaries carry the node refinement as `method`; reaction and
interaction controls inherit the common electronic method of their components.
This same ordering applies to classical dominance. Measured time never breaks
control-error ties. The band is anchored to the minimum error, avoiding
order-dependent chains of pairwise near-equal comparisons.
The exact screening reference is a fallback only when no other control passes.
With no passing control, the critic requests revision without naming
an answer. This applies to every task kind, including refused LLM designs.
With candidates present, the existing dominance and escalation rules still apply.
The independent expected value is never replaced by the screening reference,
and an unanswered task cannot gain correctness from a correct reference.
Compare decisions, candidate reports, tool use, integrity events
and resources alongside correctness. Report model defaults explicitly and pin
the model for meaningful live comparisons.

One fresh Ed25519 keypair is generated per evaluation, used for all selected
tasks with approver `eval-harness`. Only `public.pem` is persisted. Private-key
material and its random encryption passphrase stay in memory; no private key
is written to any task/output directory. This signer is an evaluation authority,
not a replacement for human production approval, and is never given to a policy.

Scripted numerical methods use fixed seeds; compare repeated rows after removing
fields ending in `_s`. Signatures, record ids and journals can vary across runs;
those are intentionally absent from rows. Control-error ties use method and id,
not measured timing. Candidate affordability and resource-budget checks retain
their existing rules; method ordering does not waive resource constraints.
Live policies are not expected to be deterministic either.

## Opt-in live evaluations and costs

Live evaluation reuses the **same agent preflight and provider factory**.
It requires an already installed provider adapter, an API key and explicit
opt-in. The harness never enables the opt-in itself. For example:

```sh
export CHEMFLOW_LIVE_LLM=1
EVAL_ROOT="$(mktemp -d)"
export OPENAI_API_KEY='your-key'
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite core --tasks h2-ground --policy openai --model YOUR_MODEL \
  --roles designer --out "$EVAL_ROOT/openai"

export ANTHROPIC_API_KEY='your-key'
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite core --policy anthropic --model YOUR_MODEL \
  --roles designer,critic --out "$EVAL_ROOT/anthropic"
```

OpenAI requires `--model`; Anthropic can use its existing role-specific defaults.
GLM (Zhipu BigModel) reuses the `openai` extra and the same OpenAI-compatible
chat-completions adapter. Supply `GLM_API_KEY` through your environment or secret
manager (not a command-line argument or checked-in file), then run:

```sh
export CHEMFLOW_LIVE_LLM=1
env -u DYLD_LIBRARY_PATH .venv/bin/chemflow eval run \
  --suite core --tasks h2-ground --policy glm \
  --roles designer --out "$EVAL_ROOT/glm"
```

The default model is `glm-5.3`; use `--model glm-4.7` or `--model glm-5.3-flash`
to override it. The default endpoint is `https://open.bigmodel.cn/api/paas/v4/`,
overridable with `CHEMFLOW_GLM_BASE_URL`. GLM Coding Plan packages are served at
`https://open.bigmodel.cn/api/coding/paas/v4`; select that endpoint explicitly:

```sh
export CHEMFLOW_GLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
```

The general endpoint can report insufficient balance despite an active Coding
Plan package. Requests use a 120-second timeout; SDK retries are disabled so
`LLMPolicy` owns the complete provider-attempt budget. Missing live opt-in or an
empty `GLM_API_KEY` fails preflight; an OpenAI key does not substitute for it.
If the agent continues with scripted controls after a model failure, the
evaluation row is invalid rather than scored as the LLM's policy decision.
`LLM_CALL` journal decisions include `provider: "glm"`, `model`, token `usage`
(`input_tokens`, `output_tokens`) and `latency_s` (including policy backoff).
Credentials are not serialized; GLM response text and errors are redacted if
they echo the key. Evaluation metadata records the effective GLM model.

All providers retry errors marked `retryable` using `retry_backoff_s` (default
2 seconds), doubling to a 60-second cap, with `max_provider_attempts` (default 10)
total attempts per model call. The default delays are 2, 4, 8, 16, 32, 60, 60, 60
and 60 seconds (302 seconds total); there is no sleep after the last failure.
These retries count neither as model
turns nor as tool-error correction retries. Rate limits, connection failures,
timeouts and HTTP 5xx are retryable; authentication, permission, bad requests and
balance/quota exhaustion are not. GLM code 1113 (余额不足或无可用资源包) is
non-retryable even under HTTP 429; GLM 1302 is a retryable rate limit.
Every failed provider attempt journals `LLM_ERROR` with its 1-based attempt and
error code. Unrecovered failures journal `LLM_FAILED` with role, code and attempt
count. An eventual success does not invalidate a row merely because retries
occurred; any `LLM_FAILED` from any role does invalidate it.

Valid roles are planner, modeler, designer, critic and analyst; unselected roles
remain scripted. The default role selection is designer. Agent screening is
bounded to two rounds, exactly the default for `agent run`; provider-policy
retry limits remain unchanged. Both commands accept `--max-turns N` (default 24),
a positive model-call limit per LLM role decision. It applies to every selected
LLM role, including evaluation workers, and is recorded as `metadata.max_turns`
in `results.json`. **Live calls incur provider charges**. Resource
budgets are not monetary/token spending caps, and `llm_calls` is not a dollar
estimate. Start with one task; inspect journals for usage and provider failures
before broadening the suite. No live provider is contacted by the G5 tests;
fake clients exercise provider paths through the harness.

Before each of their last two model calls, roles with terminal tools receive a
reminder counting the remaining turns including that call and naming their terminal
tools (`design.submit`/`design.decline` or `model.choose`). Other roles receive no reminder.

`design.decline {reason}` keeps the controls-only behavior. Every screen is checked
against the remaining signed budget before it runs: a round whose simulator shots or
predicted wall time exceed it ends the run as CF-E070 with `nothing_consumed: true`.

## Adding tasks

1. Add a uniquely named task to a suite YAML under `src/chemflow/eval/suites/`,
   or use an external `--suite-file`. Built-in YAML files ship as package data.
2. Use a lower-case alphanumeric/hyphen id, one of the six domain categories or
   `selection`/`refusal`, a
   valid `TaskSpec`, finite `expected.value_hartree`, optional signed relation gates
   (`spec.gates: [{kind: relation_at_least, relation: variational_upper_bound}]`), and a nonblank
   `reference_source`. Optionally declare nonempty `decision_in`. Refusal tasks
   instead require `outcome: refused` and `refusal_code`; numeric fields are
   optional. Every non-refusal task must declare its own target-compatible
   error gate with a finite nonnegative threshold. Unknown schema fields are
   rejected to catch typos; the answer key cannot override task accuracy or relation
   requirements. `expected.relation_at_least` is rejected, including by the independent
   reference script. Both built-in suites carry their relation requirements in signed gates;
   their independently computed expected values are unchanged.
3. Compute the answer independently using direct PySCF or a separately coded
   dense oracle. Never copy a ChemFlow pipeline result into the answer key.
   Document geometry, model, spin/sector, active space, constants and scalar
   interpretation. Size the task's error gate to its scientific accuracy needs.
4. Add the independent route to the reference script when extending core.
   Choose cheap targets with offline effects and explicit resource budgets.
5. Recompute references, run the task twice with scripted policy, compare rows
   excluding `_s` fields, then run the full tests and lint:

```sh
env -u DYLD_LIBRARY_PATH .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache
```

Keep `PYTHONDONTWRITEBYTECODE=1` set for cache-free validation. Do not weaken
the task's gates to make an agent look successful.
