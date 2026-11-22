# Error mitigation for QCL quantum workloads: design

Status (2026-11-22):
- **Implemented:**
  - `kernels.lucj(..., final_rotation=False)`, with SQD rotating the integrals;
  - `Device.with_options(dd="cpmg" | "cp")`;
  - QZ03 sampling with chain layouts on its grid;
  - offline readout inversion, post-selection and uniform-noise baselines in
    `qcl/apps/hardware_analysis.py`.
- **Phase 1 has run on MQ02 and QZ03** (60 jobs, 1,650 CNY). Results are in
  `docs/hardware-runs.md`:
  - omitting the final rotation is the one change that recovers signal at 12 qubits
    (in-sector signal from about 0 to 1.6–5.3 %);
  - post-selection is the largest offline gain;
  - the platform's automatic DD found almost no idle windows long enough to act.

## 1. What the MQ02 data says

`qcl/apps/qpu_error_budget.py` re-analyses the stored MQ02 runs of
`qcl/apps/qpu_sqd_experiment.py` (H2 and N2, 4–24 qubits, calibration of 2026-11-20)
without spending QPU time. For each run it computes the Hellinger fidelity `F` of the
hardware distribution to the ideal one, what readout inversion and particle-number
post-selection recover, and the fidelity predicted by independent error channels. Gate
durations are not in the calibration, so the schedule assumes 20 ns single-qubit gates
and 40 ns CZ (100 ns changes the predictions, not the measurements).

| run | qubits | CZ | duration | F measured | F readout-corrected | F post-selected | valid shots (uniform) | predicted: gates+readout | + T2 over the circuit |
|---|---|---|---|---|---|---|---|---|---|
| H2 | 4 | 11 | 0.5 µs | 0.79 | 0.82 | **0.99** | 0.80 (0.25) | 0.92 | 0.75 |
| N2 (4,4) | 8 | 40 | 1.8 µs | 0.34 | 0.37 | **0.83** | 0.41 (0.14) | 0.73 | 0.18 |
| N2 (6,6) | 12 | 174 | 3.8 µs | 0.0033 | 0.0033 | 0.034 | 0.10 (0.10) | 0.30 | 0.0028 |
| N2 (10,8) | 16 | 270 | 4.9 µs | 0.0005 | 0.0004 | 0.006 | 0.07 (0.05) | 0.18 | 5e-5 |
| N2 (10,10) | 20 | 508 | 7.7 µs | <1e-4 | <1e-4 | <1e-4 | 0.06 (0.06) | 0.035 | 2e-9 |
| N2 (12,12) | 24 | 690 | 8.9 µs | <1e-4 | <1e-4 | <1e-4 | 0.05 (0.05) | 0.010 | 3e-13 |

1. **Decoherence over the circuit duration dominates.** The gate-and-readout model
   overestimates the 12-qubit fidelity a hundredfold; multiplying by `exp(-t/T2)` per
   qubit over its active window (median T2 = 9.9 µs, T2* = 3.9 µs; about half of each
   window is idle) brings the prediction to the measured order of magnitude. The
   free-evolution `T2*` model undershoots the measurements, so some dephasing is already
   refocused; the model is a bracket, not a fit.
2. **Readout costs 4–30 %** of the fidelity (median readout fidelity 0.992 for |0>,
   0.981 for |1>). Tensored inversion recovers only that share, and nothing once the
   rest of the signal is gone.
3. **Particle-number post-selection is the strongest cheap lever where signal exists**:
   0.34 → 0.83 at 8 qubits, ten times at 12 and 16 qubits.
4. **From 12 qubits on, almost no signal is left.** The valid fraction is that of
   uniform random bitstrings at 12, 20 and 24 qubits and barely above it at 16, and
   post-selection leaves F below 0.04. No post-processing recovers a signal that is not
   in the samples; the circuits must get shorter.
5. **Crosstalk cannot be separated** from these runs. It needs its own characterization
   (Section 5).

The circuit itself has slack. `kernels.lucj` applies an orbital rotation, the
Jastrow layer, and the inverse rotation; the final rotation only changes the measurement
basis, which SQD and estimators can absorb into the Hamiltonian classically. Dropping
it roughly halves CZ count and depth at every size:

| active space | qubits | CZ now (depth) | without final rotation | + folded α/β order |
|---|---|---|---|---|
| (4,4) | 8 | 40 (90) | 35 (65) | 26 (48) |
| (6,6) | 12 | 174 (205) | 91 (112) | 88 (122) |
| (10,8) | 16 | 270 (262) | 151 (158) | 136 (146) |
| (12,12) | 24 | 690 (453) | 360 (243) | 342 (236) |

The folded order puts α₀ next to β₀ on the linear chain. Its gain is small because
the opposite-spin Jastrow terms sit on orbitals 0, 4, 8, … and only the first pair
becomes adjacent; a line-native Jastrow pattern is a separate ansatz choice (§3).

## 2. Principles

Mitigation fits QCL only if it keeps the language's guarantees:

- **Declared, not hidden.** Mitigation is part of the program: it changes the plan
  hash, appears in `plan.explain()`, and its overhead is in `plan.estimate()`. A backend
  never mitigates silently. (Today `lqcloud` jobs run with `readout_correction=False`, and
  the `Device` option `mitigate="readout"` is accepted but not implemented.)
- **Every technique states its effect on the relation.** SQD energies remain variational
  upper bounds under any mitigation, because mitigation only changes which determinants
  enter the subspace. Mitigated expectation values are `approximate`, with the method,
  model assumptions and extrapolation uncertainty in the provenance.
- **Overhead is a budget item.** Extra shots, extra circuits and extra *jobs* (lqcloud
  bills 5 CNY per job plus 2.5 CNY per 1000 shots) are counted by the cost model and
  checked against the operator policy before submission.
- **Calibrations are inputs.** A mitigated result names the calibration snapshot (hash
  and age) it used; the policy can bound the age.
- **Emulate before spending.** Each technique must show a gain on the calibrated
  emulator before it runs on hardware; hardware results are reported next to the
  emulator's prediction.
- **The signal check is always on.** Every hardware sample set gets a test against
  uniform bitstrings; when it fails, the result carries a validity flag saying the SQD
  energy reflects the classical post-processing, not the device.

## 3. Techniques

| layer | technique | targets | applies to | overhead | effect on the result | plan |
|---|---|---|---|---|---|---|
| circuit | drop the final orbital rotation; rotate the active-space integrals instead | duration, gate count | SQD, estimators | none (one integral rotation on the CPU) | none: an exact change of orbital basis | Phase 0 |
| circuit | coherence-aware layout: score chains by T2*, T2, readout and CZ fidelity, not CZ and readout alone | all | all | none | none | Phase 0 |
| circuit | line-native LUCJ: folded α/β order, opposite-spin terms only between chain neighbours | SWAPs | SQD, estimators | none | a different ansatz (not mitigation); recorded as such | Phase 1 |
| schedule | dynamical decoupling (X–X echo, XY4) in idle windows longer than a few gate times | T2* dephasing, static ZZ crosstalk | all | none in shots; needs timing control on the device | none | Phase 1 (if lqcloud accepts delays) |
| schedule | crosstalk-aware scheduling: no simultaneous CZ on neighbouring couplers | crosstalk | all | longer depth (a trade-off the twin decides) | none | Phase 1 |
| readout | tensored inversion from calibration circuits run in the same job (all-0, all-1; neighbour pairs for correlated readout) | readout | distributions, estimators | 2 to ~2n calibration circuits | estimators: unbiased under the calibration model; distributions: `approximate` | Phase 0 (post-processing), Phase 1 (own calibration circuits) |
| readout | readout-aware configuration recovery: weight bit flips by the asymmetric p(0\|1), p(1\|0) | readout | SQD | none | SQD stays an upper bound | Phase 0 |
| post-processing | symmetry post-selection on (N_α, N_β) | errors that leave the sector | estimators; SQD raw subspace | discarded shots (reported) | biased under noise, exact without it: `approximate` | Phase 0 |
| gates | Pauli twirling of CZ layers | coherent errors, ZZ | estimators (prerequisite for ZNE) | K circuit instances: K jobs unless lqcloud batches circuits | none on the mean; larger variance | Phase 2 |
| estimators | zero-noise extrapolation by CZ folding (scales 1, 3, 5), exponential or Richardson fit | gate and idle noise | `estimate.quantum`, `vqe.sampled` | ×3 circuits and shots (scales 1, 3, 5), the folded ones longer | `approximate`, fit model and bootstrap error in the provenance | Phase 2 |
| estimators | probabilistic error cancellation | all | — | sampling cost grows as exp(4λ); unaffordable at these depths | — | not planned |
| code | error detection or correction | all | — | needs 2-D connectivity and mid-circuit measurement; MQ02 is a 24-qubit line with T2* ≈ 4 µs | — | not feasible on MQ02 |

ZNE is rejected at compile time for samples that feed `sqd.diagonalize`: it
extrapolates expectation values, and an SQD energy is a variational bound, not an
expectation value of the noisy state.

On error correction: QCL already separates logical kernels from devices, so a
logical-qubit device type can be added when a backend offers one. On MQ02 the useful
work is mitigation and shorter circuits.

## 4. Language surface

```python
mitigation = qcl.Mitigation(
    readout="tensored",      # none | tensored | correlated; calibration circuits in the job
    symmetry="particles",    # post-select (N_alpha, N_beta) where the consumer allows it
    dd="xy4",                # none | echo | xy4 (needs device timing control)
    twirl=0,                 # Pauli-twirled instances per circuit
    zne=None,                # qcl.ZNE(scales=(1, 3, 5), fit="exponential"); estimators only
    layout="coherence",      # coherence-aware chain selection
)
kernel = qcl.kernels.lucj(space, params=amplitudes, final_rotation=False)
bits = qcl.sample(kernel, shots=4000, seed=11, mitigation=mitigation).to(qpu)
energy = qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu)
```

- `final_rotation=False` makes the kernel carry its measurement-basis rotation;
  the compiler inserts the integral rotation in front of every consumer
  (`sqd.diagonalize`, `estimate.quantum`), so a program cannot pair rotated samples
  with an unrotated Hamiltonian.
- Compile-time checks: ZNE with an SQD consumer (`QCL-E104`); DD or twirling on a device
  without the capability (`QCL-E203`); overhead beyond `[mitigation] max_overhead`,
  `max_jobs`, or a calibration older than `calibration_max_age` in the operator policy
  (`QCL-E301`).
- `plan.estimate()` reports shots, circuits, jobs and cost with and without mitigation.
- Provenance gains a `mitigation` block: the settings, calibration hash and age, overhead,
  and per-stage diagnostics (valid fraction before and after, readout-only fidelity
  factor, uniform-test p-value, the emulator's predicted fidelity).
- `Device(..., mitigate=...)` is replaced by `qcl.Mitigation`: one place, in the program.

## 5. Calibrated emulator

The current noisy emulator (`aer_noise.py`) models readout, single-qubit depolarizing
error with T1/T2 during single-qubit gates, and CZ depolarizing error. It omits idle
decoherence (no scheduled delays), dephasing at T2*, decoherence during CZ, and
crosstalk, which is why it, like the gate-error estimate, is too optimistic. The upgrade:

- schedule the native circuit with gate durations and insert delays; apply T1/T2*
  relaxation on idle qubits and T1/T2 during gates;
- add static ZZ coupling between chain neighbours during idles, with strength per
  coupler;
- fit the unknown CZ duration and ZZ strengths to the six stored runs (measured `F`
  and valid fraction); accept the twin when it reproduces them within a factor of two
  from 4 to 12 qubits;
- rank mitigation configurations on the twin before any hardware run.

Crosstalk characterization needs dedicated circuits: simultaneous versus isolated
readout of neighbours (correlated readout), and Ramsey with a neighbour in |0> and |1>
(static ZZ). These are a few short circuits per coupler and enter Phase 1.

## 6. Phases and cost

**Phase 0, no QPU time.**
- Add `final_rotation=False` and the integral rotation.
- Add the coherence-aware layout.
- Add readout inversion, readout-aware recovery and symmetry post-selection as
  post-processing.
- Add the uniform-signal test with its validity flag.
- Build the calibrated emulator.
- Re-analyse the stored runs and predict, on the emulator, each technique's gain at
  8 and 12 qubits.

**Phase 1, an approved QPU budget.**
- Runs: H2 (4 qubits), N2 (4,4) (8 qubits) and N2 (6,6) (12 qubits), one geometry
  each.
- Variants:
  - the current circuit;
  - without the final rotation;
  - plus DD;
  - plus in-job readout calibration.
- That is 12 jobs × 4000 shots ≈ 12 × (5 + 10) = **180 CNY**. Crosstalk
  characterization circuits add roughly 40–60 CNY if every circuit is its own job.
- Success criteria:
  - the measured `F` and valid fraction improve as the emulator predicted;
  - at 12 qubits, the hardware samples pass the uniform-signal test;
  - the raw-subspace SQD error falls.

**Phase 2.** Twirling and ZNE for the estimator path (`vqe.sampled`, `estimate.quantum`)
on H2. Its cost depends on whether lqcloud can batch several circuits into one job.
