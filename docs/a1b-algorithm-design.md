# A1b: experimental algorithm design

Use the design loop when no registered refinement fits, or all applicable candidates
fail task gates or budgets. A1b implements static checks, small-instance exact
comparisons and native circuit resource measurement. It does not certify device
limits, calibrated noise performance, quantum advantage, or promotion to stable.

## Verification and evidence

`chemflow.design.verify_design(program, target_node, target=..., registry=...,
store=DesignStore(), instances=3, seed=5)` records a report, whether
it passes or fails. The target is a task-style dictionary. Exactly one
`impurity_model` node is required. Scans and loops are rejected.
Each run replaces that node's parameters in a deep copy; the input remains unchanged.

Declare the accuracy bar in the program with
`builder.accept(response, l1_error_le=Q(0.15, "dimensionless"))`, or an
`abs_error_le` energy quantity for an energy target. Verification requires exactly
one error gate on the target node, no gates on other nodes, and no named reference:
the verifier supplies its independent exact reference. Thresholds must be finite,
nonnegative and dimensionally correct. A relation required by any gate on the target
node, or by the task's `relation_at_least` gate, is enforced against the inferred relation. There is no caller-supplied `tolerance` argument.

With `task=record`, the program must contain no gates. The matching error gate in
the frozen task supplies the bar; a task without that gate cannot verify a design.
Agent plans remain gate-free, and `algorithm.verify` accepts only the plan handle,
target node, instance count and seed. Agents cannot choose or change the bar.

`DomainPack.random_instance("impurity_model", rng, n_bath=3)` in the dynamics pack
draws independent uniform repulsive interactions U in [1, 4] hartree, sorted bath
energies in [-1.5, 1.5] hartree, and positive hybridisations in [0.3, 0.8] hartree.
The impurity energy remains -U/2. These are small, interacting AIM benchmarks,
not a claim of coverage of all physical regimes. The supplied RNG alone determines
the draws; seed, original target and actual parameters are retained in the report.
Other families/packs return `NotImplemented`.

The exact reference is built independently from the target's sector, observable,
orbital, spin, grid and broadening, never the design's response parameters. Only the
sampled Hamiltonian parameters are shared. Spectral error is the dimensionless L1
integral on an identical grid and energy unit; incompatible grids are refused.
Energy error is absolute error in hartree, including the first Ritz root for a
ground-energy target. Unsupported target kinds or missing family references are
refused. An equivalent result additionally requires error <= 1e-8; an upper bound
requires E >= E_exact - 1e-9. All instances must satisfy both accuracy and relation.
Energy thresholds are converted before comparison. For a statistical estimate,
absolute error plus `z * stat_err` must meet the bar, where `z` is the two-sided
normal quantile at the gate's confidence (default 0.95, also used for task bars).
A stochastic estimate without a statistical error cannot pass.

Reports contain a program hash, gate-free template and target node, target, family,
seed, per-instance metrics and relation checks, maximum error, inferred relation,
`accuracy` and verdict. Accuracy records retain `kind`, `threshold`, `unit`, and
`source` (`program` or `task`), plus `confidence` for absolute-error bars, `relation`
when declared on the gate, and `task_hash` for task bars. Identity-mismatch reports
retain the same accuracy record. Gates are excluded from the template and program
hash: the same design at two bars has one program hash and separate reports.
Resource totals sum the design executor's node costs for
simulator shots, QPU shots and wall time, excluding the separate exact reference.
Every prepared circuit is built and passed through `compile_native`: reports retain
its instance/node, qubits, native two-qubit count and native depth. Aggregate qubits
and depth are maxima; aggregate two-qubit count is a sum. Native compilation time
is not part of executor wall time. Tool verification charges both numerical design
and exact-reference runs to the frozen task's shared ledger.

Reports are immutable canonical JSON payloads addressed by SHA-256 of their entire
content, excluding the derived id. Nested reads are detached copies. Equal payloads
have equal ids; measured wall times mean repeated runs need not have equal ids even
when sampled parameters and numerical metrics reproduce exactly. `DesignStore(path)`
persists records and unique registrations in SQLite; no path means in-memory SQLite.
No tool accepts an agent-authored report or passing verdict.

## Registration and reuse

`register_design(registry, store, report.id, name)` accepts only a recorded passing
report and a unique name. It publishes `design:<name>` through registry search/get,
with `status="experimental"`, inferred relation, report id, target, template and
the report's `accuracy`, persisted in SQLite. Legacy entries list `accuracy: null`.
`instantiate_design(store, name, model_params=...)` returns a detached program and
target node, with the family parameters replaced. The executor validates its node
metadata against the registered template before adding `design`, `design_status`
and `design_evidence` diagnostics, plus `design_accuracy`. Node metadata still holds
only the first three fields. Before any node executes, every design is checked
against every error gate on its node, including attached frozen-task gates. A
missing recorded bar, different measure or stricter threshold produces CF-E120:
verify at the stricter bar and register before use. Equal or looser thresholds
(after unit conversion) are allowed. Changing the algorithm invalidates the evidence;
changing the model parameters is the intended reuse operation.

`algorithm.verify` and `algorithm.register` are designer/orchestrator tools;
`algorithm.list` is available to every role. MCP uses underscore-separated names.
Model-facing schemas offer verification and registration only for supported
impurity-model ground-energy and spectral-function targets. Direct tool calls
retain their existing permissions and verification checks. An unknown registration
report id is a `ValueError` explaining that the id must come from `algorithm.verify`.
Verification obtains the target from the signed task record, checks allowed effects
and consumes its budget. It extracts the selected candidate's dependency graph so
a separate exact-reference graph in a task plan is not a second design family node.
Budget limits follow D-R1's overrun rules:
forecasts refuse infeasible work before execution; exhausted dimensions admit only
zero predictions, not positive or unknown amounts. With budget left, known predictions
must fit and unknown amounts are not counted. Measured overruns are persisted without
discarding completed results. D-R2 budgets track device resources only, not billing or job counts.
Runtime refusals propagate to dependents, fail their gates, and are reported alongside
overruns. Verification records a refused target as a failed instance with its reason.
Tools keep `designs.sqlite` in the plan directory and restore registry entries on
startup. Python callers can instantiate stored designs and merge them into task
plans with the usual exact reference and classical controls.

Experimental headlines always include the status and evidence id. The critic still
requires current-task gates and evidence and prefers a qualifying classical control
at no greater measured cost. A registration is not permission to bypass those checks.
