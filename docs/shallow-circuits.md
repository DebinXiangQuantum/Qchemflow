# Shallower circuits, better samplers, structure-aware mitigation (2026-11-23)

Goal: the shallowest circuit that gives the same SQD result. This note covers what was
tried, what was measured on simulators and on MQ02, and what it implies.

Programs:
- `qcl/apps/shallow_circuit_study.py`: CZ cost against SQD quality, and classical samplers;
- `qcl/apps/ffsim_lucj_states.py` and `qcl/apps/ffsim_lucj_evaluate.py`: IBM's LUCJ
  construction scored with the same SQD;
- `qcl/apps/twin_mitigation.py`: hardware test of the Jastrow-off twin.

## 1. Corrections to earlier claims

1. **Omitting the final orbital rotation makes the samples classical.**
   - The open LUCJ circuit is J·U|HF⟩, with J diagonal. Diagonal phases do not change
     computational-basis probabilities, so its samples are exactly those of the Slater
     determinant U|HF⟩ (checked to 10⁻¹⁷ for any Jastrow strength).
   - Slater determinants are efficiently classically samplable.
   - The `open` hardware runs of 2026-11-22/23 therefore sampled a classical distribution.
     Their higher signal shows what a shorter circuit survives, not quantum progress.
   - The energy equivalence of the rotated Hamiltonian is still exact; only the claim
     that it preserves the circuit's quantum content was wrong.
2. **The comparison with IBM's toolchain compared different circuits.**
   - IBM's `no_final_rot` removes only the extra t1 rotation and keeps U†JU; our `open`
     removes U†. The like-for-like comparison is our full LUCJ against IBM's (Section 2).
3. **ChemFlow's CCSD seed is close to classical.**
   - The seed has no same-spin Jastrow terms and n/4 opposite-spin ones.
   - Each factor exp(iθ n_a n_b) = 1 + (e^{iθ} − 1) n_a n_b, so the LUCJ state is a sum
     of 2^{n/4} Slater determinants: 8 at 24 qubits.
   - It is classically simulable at every size the devices allow.

## 2. Circuit cost (exact reductions, implemented)

`compile_native` / `compile_spec` (compiler `chain-layout-search-v4-construction-search`):
- the first rotation acts on |HF⟩ and is built as an optimal Slater-determinant
  preparation, m(n − m) Givens rotations per spin instead of n(n − 1)/2
  (`circuits.slater_givens`, exact, checked against the sector state for closed and
  open shells);
- the compiler keeps whichever construction transpiles shorter;
- identity and folded α/β placements are each tried with five transpiler seeds.

CZ on a linear chain, N₂/cc-pVDZ LUCJ circuits with the full U†JU:

| qubits | before (11-22) | full LUCJ now | IBM (U†JU, ffsim + Qiskit) | ratio to IBM | line-native Jastrow |
|---|---|---|---|---|---|
| 8 | 40 | 26 | 54 | 0.48 | 26 |
| 12 | 174 | 148 | 150 | 0.99 | 42 |
| 16 | 254 | 204 | 158 | 1.29 | 58 |
| 20 | 508 | 430 | 336 | 1.28 | 74 |
| 24 | 690 | 558 | 346 | 1.61 | 90 |
| 28 | 1020 | 850 | — | — | 106 |

- **Above 12 qubits we still trail IBM by 1.3–1.6×.** The remaining cost is SWAP routing
  of the opposite-spin pairs α₄–β₄ and α₈–β₈ on the line.
- **Line-native Jastrow** (`kernels.lucj(..., jastrow="line")`) keeps only the
  chain-adjacent pair. The transpiler then cancels every Givens rotation outside that
  pair's light cone, exactly, and the overlap with the full state stays ≥ 0.994. But the
  state is then a sum of two Slater determinants: shallow because it is nearly trivial.

## 3. Which sampler gives the better SQD (noiseless)

N₂/cc-pVDZ CAS(10,10), 20 qubits. SQD error (mEh) from 2000 / 10 000 ideal samples
(mean of three seeds), samples-only subspace and with the single-excitation extension.
All samplers are scored on one Hamiltonian (ffsim's exported active-space integrals).

| sampler | CZ | eq., samples only | eq., + ext | 2.2 Å, samples only | 2.2 Å, + ext |
|---|---|---|---|---|---|
| ChemFlow CCSD seed (full LUCJ) | 430 | 59 / 40 | 1.4 / 0.5 | 220 / 190 | 36 / 22 |
| IBM LUCJ, line pairs, 1 layer | 336 | 68 / 49 | 4.6 / 3.5 | 107 / 67 | 2.2 / 2.0 |
| IBM, opposite-spin pairs only | 216 | 71 / 45 | 4.4 / 3.3 | 183 / 159 | 19 / 5.7 |
| IBM, 2 layers | 644 | 48 / 30 | 3.5 / 2.8 | 68 / 51 | 2.0 / 1.9 |
| IBM, all pairs (not line-native) | 1345 | 28 / 27 | 2.7 / 2.6 | **32 / 29** | 2.6 / 2.5 |
| classical CISD amplitudes | 0 | **11 / 5** | **0.14 / 0.03** | 75 / 72 | **0.46 / 0.42** |

With ChemFlow's own Hamiltonian (`shallow_circuit_study.py`), the classical free-fermion
sampler in the LUCJ frame gives 25 / 11 mEh at equilibrium and 159 / 123 mEh stretched;
with extension 0.05 / 0.02 and 1.5 / 1.1 mEh.

- **The ideal LUCJ output is almost pure Hartree–Fock.** P(HF) is 0.97–0.998 at
  equilibrium, and 0.97–0.9997 for the Diels–Alder π spaces. Samples of it cover few
  configurations.
- **A classical CISD sampler, with no circuit, beats every quantum variant once the
  extension is applied**, by 10–100×. Without extension it also wins at equilibrium.
- **Only at the stretched geometry** does a quantum state beat CISD on the samples-only
  subspace: IBM's all-pairs LUCJ, at 1345 CZ (29 against 72 mEh at 10 000 shots).
- **IBM's double-factorized seed with same-spin pairs** is 2–3× better than ChemFlow's
  seed at the stretched geometry (10× with extension), at fewer CZ (336 against 430).
  ChemFlow's seed is better at equilibrium with extension.
- **Truncating small Givens angles of the final rotation saves almost nothing.** The
  angles are large: at 20 qubits, 45 → 38 Givens and 430 → 395 CZ. It also moves the
  frame toward the classical free-fermion limit.

## 4. Structure-aware mitigation on MQ02

| technique | uses | measured effect |
|---|---|---|
| particle-number post-selection | N_α and N_β conserved | F × 1.3–3 (4–8 q), × 7–10 (12 q) |
| tensored readout inversion | calibration circuits | F + 3–8 % (relative) |
| α↔β symmetrization of counts | singlet, spin-balanced circuit | F + 0–1 % (4–8 q), + 2–14 % (12 q) |
| **Jastrow-off twin**: same circuit with J = 0 | U†U = I: the twin's ideal output is \|HF⟩, with the same gates (40 vs 42 CZ at 8 q, 96 vs 98 at 12 q) | with post-selection, F 0.80 → 0.90 (8 q) and 0.078 → 0.21 (12 q, 2.6×) |
| platform dynamical decoupling | idle windows | not applied: windows of 40 ns, 160 ns needed |

Twin experiment details (2026-11-23, MQ02, 8 jobs, 184 CNY):
- the line-native LUCJ for butadiene (8 q, 42 CZ) and the Diels–Alder transition state
  (12 q, 98 CZ), each with its twin, 2 repeats × 8192 shots;
- twin correction: p̂ = (P − T)/T(HF) + δ_HF, clipped.

The correction mostly restores the Hartree–Fock peak. Against the ideal **off-HF**
distribution, the part that carries the correlation SQD needs:

| off-HF fidelity | hardware | uniform bitstrings | twin (pure noise) |
|---|---|---|---|
| butadiene, 8 q | 0.35 | 0.31 | 0.33 |
| transition state, 12 q | 0.068 | 0.092 | 0.086 |

**Even at 8 qubits the device's correlated part is at the noise level.** Mitigation
recovers the known Hartree–Fock weight, not the correlation.

## 5. Conclusions

1. **The tension is structural.** Every change that made these circuits shallow enough
   to survive MQ02 also made them classically simulable:
   - the open circuit is a Slater determinant;
   - the ChemFlow seed is 2^{n/4} determinants;
   - line-native LUCJ is 2 determinants.

   The circuits whose samples beat a classical sampler need ~1300 CZ at 20 qubits.
   MQ02 loses all signal above about 100 CZ at 12 qubits.
2. **For SQD at this scale the best "circuit" is none.** A CISD (or free-fermion) sampler
   costs nothing and gives better energies than any ideal LUCJ variant here. The QPU
   adds no information today.
3. **Worth carrying forward:**
   - the exact compiler gains: Slater preparation, construction and placement search;
   - α↔β symmetrization and the twin calibration, for any LUCJ-type circuit;
   - IBM's double-factorized seed with same-spin pairs, if quantum SQD is pursued;
   - closing the routing gap to IBM above 12 qubits.
4. **Where shallow hardware circuits could matter:**
   - workloads whose circuits are native to the hardware and classically hard at modest
     depth, such as lattice-model dynamics;
   - native fermionic two-qubit gates (fSim/Givens as one pulse,
     `hybrid-pulse-chemistry/docs/WAVEFORM_PROGRESS_20261122.md`) that would make
     600–1300-CZ circuits a few hundred native gates.

   Both need demonstrated hardware fidelity before any claim.

Follow-up: `docs/dynamics-mitigation.md` tests Krylov sampling (no better for SQD) and
interacting spin-chain dynamics on MQ02 with free-fermion-twin training and Pauli twirling.
