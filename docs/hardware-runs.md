# Application quantum parts on real QPUs, with mitigation and timing (2026-11-22)

The quantum parts of the three applications in `docs/application-runs.md`, run on
lqcloud hardware by `qcl/apps/hardware_applications.py`, analysed offline by
`qcl/apps/hardware_analysis.py` and timed by `qcl/apps/timing_comparison.py`.

**Runs:**
- 60 jobs, all successful: 48 on MQ02 (24-qubit chain) and 12 on QZ03 (35-qubit grid);
- calibrations of 2026-11-22;
- charged 1,650 CNY in total (MQ02 1,290, QZ03 360), each job exactly 2.50 CNY + 2.50 CNY per 1000 shots;
- the plan forecasts used the older ledger tariff, twice as high.

Each circuit is the CCSD-seeded LUCJ kernel of the active space. Every hardware sample
set is compared with:
- exact CASCI;
- 10⁶ ideal samples of the same circuit;
- as many uniformly random bitstrings as the device returned.

| Group | Systems | Qubits | Device | Jobs |
|---|---|---|---|---|
| Diels–Alder π spaces | ethylene, butadiene, transition state, cyclohexene | 4, 8, 12, 12 | MQ02 | 4 variants × 2 repeats × 8000 shots |
| T4L benzene in the protein field | bound, separated (QM/MM CASCI(6,6)) | 12 | MQ02 | 2 variants × 2 × 8000 |
| H12 chain | 1.8 and 3.2 bohr | 24 | MQ02 | 2 variants × 1 × 20 000 |
| CASCI(14,14) | DA transition state; T4L bound and separated | 28 | QZ03 | 2 variants × 1 × 20 000 |
| Readout calibration | all-0 and all-1 circuits on each layout width | 4, 8, 12, 24, 28 | both | 10 × 8000 |

The protein-embedded benzene reached the QPU through `qcl.qmmm.scf`. It returns the
embedded RHF as an ordinary SCF result, so `qcl.active_space`, `kernels.lucj`,
`qcl.sample` and `sqd.diagonalize` apply unchanged, with the MM energy in the
Hamiltonian's constant.

## 1. Mitigation variants

| Variant | Change |
|---|---|
| `base` | the circuit used so far |
| `open` | final orbital rotation omitted (`kernels.lucj(..., final_rotation=False)`): CZ 174 → 91 at 12 qubits, 690 → 360 at 24, 1020 → 510 at 28. SQD rotates the Hamiltonian to the sampled orbital basis, an exact change of basis |
| `dd` | platform dynamical decoupling (CPMG, 4 pulses) via `Device.with_options(dd="cpmg")` |
| `open_dd` | both |

Readout inversion (from the measured calibration circuits) and particle-number
post-selection were applied offline to every variant.

**Metrics:**
- **F**: Hellinger fidelity of the hardware distribution to the ideal one of the same
  circuit.
- **F₀**: the same fidelity for uniformly random bitstrings. It is the noise floor, and it
  differs between variants: the `open` circuit's ideal distribution is more spread out,
  so noise already scores higher against it.
- **Signal s**: the weight of the ideal state in a globally depolarized output, from the
  in-sector fraction v: v = s + (1 − s)·v₀, with v₀ the uniform in-sector fraction. Every
  variant conserves particle number, so s does not depend on the measured orbital basis.
  It is the cleanest comparison across variants.

Means over two repeats:

| system | qubits | variant | CZ | F | F₀ | signal s | F post-selected | F readout-inverted |
|---|---|---|---|---|---|---|---|---|
| ethylene | 4 | base | 11 | 0.618 | 0.106 | 0.564 | 0.918 | 0.638 |
| | | open | 9 | 0.750 | 0.192 | 0.674 | 0.993 | 0.782 |
| | | dd | 11 | 0.648 | 0.106 | 0.594 | 0.932 | 0.670 |
| | | open_dd | 9 | 0.761 | 0.192 | 0.688 | 0.993 | 0.793 |
| butadiene | 8 | base | 40 | 0.231 | 0.008 | 0.214 | 0.710 | 0.250 |
| | | open | 35 | 0.415 | 0.060 | 0.340 | 0.959 | 0.448 |
| | | dd | 40 | 0.197 | 0.008 | 0.180 | 0.664 | 0.212 |
| | | open_dd | 35 | 0.400 | 0.060 | 0.321 | 0.961 | 0.431 |
| DA transition state | 12 | base | 174 | 0.0049 | 0.0005 | 0.002 | 0.049 | 0.0052 |
| | | open | 91 | 0.071 | 0.010 | **0.053** | 0.489 | 0.074 |
| | | dd | 174 | 0.0043 | 0.0005 | −0.004 | 0.045 | 0.0045 |
| | | open_dd | 91 | 0.075 | 0.010 | 0.051 | 0.524 | 0.078 |
| cyclohexene | 12 | base | 174 | 0.0024 | 0.0003 | −0.001 | 0.025 | 0.0025 |
| | | open | 91 | 0.066 | 0.003 | 0.033 | 0.519 | 0.070 |
| | | dd | 174 | 0.0017 | 0.0003 | 0.004 | 0.016 | 0.0017 |
| | | open_dd | 91 | 0.079 | 0.003 | 0.045 | 0.569 | 0.084 |
| T4L benzene, bound | 12 | base | 174 | 0.0027 | 0.0004 | −0.002 | 0.028 | 0.0027 |
| | | open_dd | 91 | 0.079 | 0.008 | 0.052 | 0.545 | 0.082 |
| T4L benzene, separated | 12 | base | 174 | 0.0013 | 0.0004 | 0.003 | 0.013 | 0.0013 |
| | | open_dd | 91 | 0.054 | 0.008 | 0.016 | 0.477 | 0.055 |
| H12, 1.8 bohr | 24 | base / open_dd | 690 / 360 | 0 / 7·10⁻⁶ | 6·10⁻⁷ / 10⁻⁴ | ≈ 0 / ≈ 0 | | |
| H12, 3.2 bohr | 24 | base / open_dd | 690 / 360 | 0 / 7·10⁻⁶ | 5·10⁻⁶ / 3·10⁻⁴ | ≈ 0 / ≈ 0 | | |
| CASCI(14,14), 3 systems, QZ03 | 28 | base / open_dd | 1020 / 510 | ≤ 10⁻⁷ | ≈ 10⁻⁸ / 5·10⁻⁶ | ≈ 0 / ≈ 0 | | |

Findings:

1. **Omitting the final rotation is the one change that recovers signal.**
   - At 12 qubits the `base` circuits (174 CZ) are indistinguishable from noise:
     s = −0.2 % to 0.3 %, and the in-sector fraction equals that of random bitstrings.
   - The `open` circuits (91 CZ) have s = 1.6–5.3 %.
   - At 8 qubits s rises from 21 % to 34 %, at 4 qubits from 56 % to 67 %.
   - With post-selection, the 12-qubit fidelities go from 0.01–0.05 to 0.48–0.57.
   - The gain is larger than the 2× in CZ count suggests: the circuit also gets shorter
     in time, and decoherence over the circuit dominates (see `docs/error-mitigation.md`).
2. **Particle-number post-selection is the largest offline gain.** It multiplies F by
   1.3–3 at 4–8 qubits and by 7–10 at 12 qubits, wherever signal is left.
3. **Readout inversion helps a little.** F rises 3–8 % relative; it removes only the
   readout share of the error.
4. **Dynamical decoupling did not act.** The platform inserted pulses only in the 8-qubit
   jobs (one window each) and the 24-qubit jobs (three windows each). Everywhere else it
   reported `cz_subwindow_too_short`: the transpiled LUCJ circuits are densely packed,
   with 40 ns gaps against the 160 ns that 4-pulse CPMG needs. The `dd` rows are
   therefore within noise of `base`. DD needs scheduling that leaves idle windows, or
   the platform's manual `timeanchor`/`dd` gates.
5. **From 24 qubits on, no variant has a measurable signal.** H12 on MQ02 (360–690 CZ)
   and CASCI(14,14) on QZ03 (510–1020 CZ) give s ≈ 0 and fidelities at the noise floor.

## 2. Application results from the hardware samples

| quantity | hardware SQD | hardware, samples only | exact CASCI | uniform-bitstring control |
|---|---|---|---|---|
| DA barrier, π spaces (kcal/mol) | 57.815, every variant | 57.815 | 57.815 | exact |
| DA reaction energy (kcal/mol) | −14.388 | −14.388 | −14.388 | exact |
| T4L QM/MM interaction, CAS(6,6) (kcal/mol) | −17.102 | −17.102 | −17.102 | exact |
| H12 1.8 bohr, error (mEh) | 0.000 (base), 0.000 (open_dd) | 57.8 / 37.8 | — | 22.9 (samples only) |
| H12 3.2 bohr, error (mEh) | 0.000 / 2.12 | 183.5 / 148.4 | — | 104.6 (samples only) |
| CAS(14,14) DA transition state, error (mEh) | not run (see below) | 466.7 / 118.0 | — | 465.1 |
| CAS(14,14) T4L interaction (kcal/mol) | — | −538.3 / −63.6 | −16.958 | — |

These energies do not show a hardware contribution:

- **≤ 12 qubits.** Every sector has at most 400 determinants. A few thousand noisy shots
  contain all of them, so SQD (and even the samples-only subspace) is exact, and uniform
  random bitstrings are exact as well.
- **24 qubits.** The full SQD (recovery and extension) reaches the whole 853,776-determinant
  sector from any input, so it is exact. The samples-only subspaces are worse than those
  from random bitstrings at 1.8 bohr (57.8 and 37.8 mEh against 22.9).
- **28 qubits.**
  - The samples-only energies are noise, 34–865 mEh above exact, and so are the
    interaction energies formed from them.
  - Full SQD was not run: on noise it would fill most of the 11.8-million-determinant
    sector, which is a classical FCI.

The device contribution is visible only in the sample statistics of Section 1. That is
also where the next steps have to show progress: shorter circuits until the samples carry
signal at the sizes where the sector cannot be filled by post-processing.

## 3. Time: exact solution on our server against the quantum route

Measured on F5090. Exact FCI uses PySCF on 16 CPU threads and ChemFlow's CUDA FCI on one
RTX 5090.

The quantum route is:
1. the CCSD seed and circuit transpilation;
2. the QPU jobs of the `base` variant, measured from the ledger:
   - ≤ 12 qubits: 2 jobs × 8000 shots;
   - 24 and 28 qubits: 1 job × 20 000 shots;
   - the queue was empty;
3. SQD on those samples:
   - full SQD up to 24 qubits;
   - samples-only at 28 qubits.

The last column projects the QPU wall time at the shot counts of the simulator runs
(100 000–200 000 shots), using the MQ02 job-time model and today's per-job overhead.

| system | qubits | exact FCI CPU | exact FCI GPU | QPU device | QPU wall | SQD | seed + transpile | quantum route | projected at 10⁵–2·10⁵ shots |
|---|---|---|---|---|---|---|---|---|---|
| ethylene | 4 | < 0.01 s | | 9.5 s | 14.1 s | 0.4 s | 0.07 s | 14.6 s | 92 s |
| butadiene | 8 | < 0.01 s | | 9.7 s | 14.2 s | 0.5 s | 0.08 s | 14.8 s | 92 s |
| DA transition state | 12 | 0.02 s | | 10.0 s | 14.2 s | 0.7 s | 0.12 s | 15.0 s | 92 s |
| cyclohexene | 12 | 0.02 s | | 19.9 s | 25.0 s | 0.7 s | 0.09 s | 25.8 s | 93 s |
| T4L benzene, bound | 12 | 0.02 s | | 11.3 s | 14.4 s | 0.8 s | 0.08 s | 15.3 s | 45 s |
| T4L benzene, separated | 12 | 0.02 s | | 17.9 s | 24.5 s | 0.7 s | 0.17 s | 25.4 s | 49 s |
| H12, 1.8 bohr | 24 | 1.6 s | 1.4 s | 14.4 s | 17.7 s | 39.0 s | 0.27 s | 56.9 s | 87 s |
| H12, 3.2 bohr | 24 | 5.3 s | 4.3 s | 15.8 s | 17.6 s | 81.2 s | 0.52 s | 99.3 s | 126 s |
| CAS(14,14) DA transition state | 28 | 22.6 s | 19.1 s | 5.7 s | 7.1 s | 6.3 s* | 0.22 s | 13.7 s* | 95 s |
| CAS(14,14) T4L bound | 28 | 22.5 s | 19.1 s | 5.5 s | 7.1 s | 5.6 s* | 0.21 s | 13.0 s* | 95 s |
| CAS(14,14) T4L separated | 28 | 22.6 s | 19.1 s | 6.0 s | 7.2 s | 8.8 s* | 0.21 s | 16.2 s* | 96 s |

\* Samples-only SQD; its energies are 34–865 mEh off (Section 2). An accurate SQD from
such samples would cost about as much as the exact FCI.

- **Up to 24 qubits** the exact solution is 10³–10⁴ times faster than the quantum route.
  Each QPU job carries about 2–5 s of overhead, and the device time is set by shots, not
  by problem size.
- **At 28 qubits** the QPU job (QZ03: 7 s for 20 000 shots) is already shorter than exact
  FCI (19–23 s). The route is not faster to a correct answer, because the device returns
  noise.
- **Scaling.** Exact FCI cost follows the sector dimension, about 14× per two added
  orbitals at half filling. CAS(16,16), 32 qubits, would be several minutes on the CPU
  and does not fit this GPU code's memory. CAS(18,18) is out of reach. The QPU job time
  barely changes with size. A time advantage could therefore appear around 32–36
  qubits, but only for circuits whose samples still carry signal at that size. Today
  that stops at about 12 qubits (91 CZ, s ≈ 5 %). These are estimates from the
  measured scaling, not measurements.

## 4. Reproduction

```sh
. ~/.config/chemflow/lqcloud.env              # QPU credential
QCL_POLICY_FILE=qcl/apps/hardware-policy.toml CUDA_VISIBLE_DEVICES=2,3 \
    python qcl/apps/hardware_applications.py run --out hw \
    --da da/result.json --t4l t4l-hw           # --qpu sim: free rehearsal on simulators
python qcl/apps/hardware_applications.py analyse --out hw
cd qcl/apps && python timing_comparison.py --out ../../hw
```

A rerun reuses the recorded jobs from the QPU ledger and pays nothing for them. Raw
counts, job IDs, layouts and per-job charges are in the group JSON files and the ledger.
