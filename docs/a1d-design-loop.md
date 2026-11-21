# A1d: larger-bath design and the diagnostic loop

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

## Contract

`chemflow.design.design_loop(program, target_node, *, target, registry, store,
instances, seed, max_rounds, moves=None, register_as=None, ledger=None, task=None)`
verifies each candidate with `verify_design`. The program's acceptance gate, or
the frozen task's matching gate for a gate-free plan, remains the only accuracy
bar. Relation requirements are enforced too. Redesign cannot change the model,
sector, orbital, spin, grid, broadening, or accuracy gate.

`max_rounds` counts **verified candidates**, including the seed. The result has
`outcome` (`passed` or `exhausted`), `trajectory`, and `design`. Every nonterminal
round applies exactly one named move. Passing, no applicable moves, and the
round limit terminate without applying another move. Registration happens only
on pass, only when requested, with experimental status and passing evidence.

Each entry records its round, report ID, pass verdict, maximum metric,
per-instance diagnosis, move and rationale, and resources. Resource records
include shots, compiled circuits, and ground/particle/hole subspace sizes,
full dimensions, and fractions for each instance. Diagnoses use recorded
ground-energy errors against the independent reference, coverage, finite-grid
sum-rule differences, and relation/accuracy failures. A grid-truncated integral
is compared with the exact integral on that same grid, **not** forced to one.

### Reproducibility versus measured time

The seed reuses identical random parameter instances across rounds. Each move
is tried at most once; caller move ordering does not override policy priority.
Stable evidence reports retain all scientific quantities and resources but
explicitly set `resources.wall_s` to `null`. IDs hash the complete canonical
payload. The trajectory's `execution_report_id` links to the original
`verify_design` report retaining measured wall time; trajectory resources also
retain timing. Evidence IDs and decisions reproduce without inventing
deterministic execution times or weakening content-addressed records.
Measured times and execution-report IDs naturally vary. A supplied ledger is
passed to every verification; timing-dependent refusals can vary with load.

## Move table

`MOVES` is the ordered tuple below. `apply_move` returns a detached transformed
program, or `None` if inapplicable, without mutating the seed or its gates.

| Move | Transformation | Bound / cost |
|---|---|---|
| `more_shots` | Double sample counts, retaining sampling seeds | Cap 40,000 per sampling node; once per loop |
| `enriched_ground` | Use an existing enriched N-sector space for ground Ritz; otherwise construct N → N+1 → N using singles-closed ladder extension | Larger ground solve; no new primitive |
| `doubles_closure` | Set ladder extensions to two layers of spin-preserving single excitations | Same- and mixed-spin doubles; O(D M⁴) construction |
| `more_ritz_roots` | Double reported root counts | Capped by the full sector dimension |
| `full_sector` | Replace sampled/extended spaces with `full_subspace`; prune unused dependencies | Exponential exact-sector fallback |

Full-sector fallback names contain `full`, so callers can exclude them with
`moves=[name for name in MOVES if "full" not in name]`. **Other moves can still
produce full coverage naturally**, especially in small sectors. Candidate dense
Ritz still caps sectors at 4,096 determinants; the independent reference's
eight-site limit does not imply all eight-site candidates are executable.

## Rule table

`MOVE_RULES` evaluates these rules in priority order. The first allowed, unused
move whose trigger matches and whose transformation is applicable wins.
Triggers in each row are alternatives. Identity mismatches and executor refusals
terminate rather than redesigning the declared problem.

| Priority | Recorded diagnosis | Move / rationale |
|---|---|---|
| 1 | Ground coverage < 1 | `more_shots`: cheapest sampling repair |
| 2 | Ground-energy discrepancy > 1e-8 Ha, or incomplete ground coverage | `enriched_ground`: repair the ground vector |
| 3 | Incomplete ground/response coverage, or sum-rule discrepancy > 1e-8 | `doubles_closure`: enlarge determinant support |
| 4 | Explicitly truncated spectral resolution | `more_ritz_roots`: restore missing roots |
| 5 | Accuracy or required relation fails | `full_sector`: last available exact-sector repair |

The diagnostic 1e-8 checks label failure causes; they are **not** acceptance
thresholds. Passing stops before any rule is used. Current A1a Ritz primitives
retain complete projected resolutions regardless of `n_roots`, so increasing
reported roots cannot repair their Lehmann spectra. The policy deliberately
skips that move for `complete_projected` responses.

## Independent reference

`domains.fock_reference` covers up to eight sites. It constructs N, N+1, and
N−1 matrices on determinant bit strings with its own creation/annihilation
algebra, without importing candidate `algorithms`, `dynamics`, or FCI builders.
Matrices are CSR sparse. Ground sectors larger than 512 use deterministically
seeded sparse lowest-eigenpair diagonalization; smaller ground sectors and
particle/hole sectors use dense Hermitian diagonalization. Adjacent sectors
retain complete spectral resolutions, not a truncated response approximation.

At eight sites and half filling, dimensions are 4,900 and 3,920 / 3,920, rather
than a 65,536-square full-Fock matrix. Spectral diagnostics expose `ground_energy`
in hartree. Tests compare five/seven sites with independent dense tensor-product
ED, six-site spectra with the reviewer's full-Fock oracle, and eight-site ground
energy with PySCF FCI. The six/eight-site reviewer reference tests together took
34.17 s on the development machine.

## Bath5 result — 2026-10-13

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  env -u DYLD_LIBRARY_PATH ../chemflow/.venv/bin/chemflow \
  bench run dynamics aim-design-bath5 --json benchmarks/results/a1d_bath5.json
```

Six sites, sector (3,3), 3 random AIM instances, seed 5. The program declares
dimensionless **L1 ≤ 0.1**, with 801 points on [-4,4] Ha and eta 0.1 Ha. The seed
uses sampled SQD (20,000 shots) and singles-closed ladder response spaces.

**Outcome: passed, 4 verified rounds, 5.10 s benchmark runtime** in the saved
`benchmarks/results/a1d_bath5.json` run (initial run: 5.87 s).
Moves below are applied *after* measuring that row.

| Round | Per-instance L1 errors | Maximum | Move |
|---|---|---|---|
| 1 | 0.569302, 0.851680, 0.220559 | 0.851680 | `more_shots` |
| 2 | 0.569302, 0.851680, 0.220559 | 0.851680 | `enriched_ground` |
| 3 | 0.268956, 0.257683, 0.148874 | 0.268956 | `doubles_closure` |
| 4 | 0.0211543, 0.00123634, 0.00141697 | 0.0211543 | none — pass |

Counts are ground / particle / hole; full dimensions are 400 / 300 / 300.

| Round | Instance 1 | Instance 2 | Instance 3 |
|---|---|---|---|
| 1–2 | 25 / 99 / 86 | 16 / 84 / 48 | 36 / 108 / 108 |
| 3 | 314 / 99 / 86 | 284 / 84 / 48 | 318 / 108 / 108 |
| 4 | 400 / 237 / 235 | 400 / 216 / 172 | 400 / 240 / 240 |

Final fractions (ground / particle / hole): 100% / 79% / 78.33%;
100% / 72% / 57.33%; 100% / 80% / 80%. Shots per round total 60,000,
120,000, 120,000, 120,000 across the three instances (420,000 total).

Doubling shots did not change sampled support. Ground enrichment helped but
did not meet the bar. Doubles closure passed without the named full-sector
fallback, **but reached the entire ground sector in all three instances** and
retained large response sectors. This is not evidence of quantum advantage or
a compact scalable design. The inferred relation remains `approximate` and
registration experimental. Repeatedly inspected instances are design evidence,
not independent held-out generalization tests. JSON includes the final executable
template, registration, trajectory, accuracy, runtime, and sector usage.
