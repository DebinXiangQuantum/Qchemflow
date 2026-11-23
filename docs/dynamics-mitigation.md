# Beyond classically simulable circuits: Krylov sampling and interacting dynamics (2026-11-23)

`docs/shallow-circuits.md` showed that every shallow LUCJ variant was classically
simulable. Two directions that are not:
1. use interacting time evolution as the SQD sampler (Krylov / SKQD-style);
2. use the quantum output directly, as real-time dynamics observables, with error
   mitigation trained on classically exact free-fermion twins.

## 1. Interacting-dynamics samples for SQD (simulation)

`qcl/apps/hubbard_krylov_budget.py`: 1D Hubbard chain, open boundaries, half filling.

- **Quantum sampler:** samples of e^{−iHt}|Néel⟩ at t = 0.5 … 4, exact evolution.
- **Classical competitors:**
  - the U = 0 twin dynamics;
  - the U = 0 ground state (a Slater determinant);
  - CISD in the RHF basis;
  - uniform bitstrings.
- **Ideal reference:** the exact ground state's own string selection.

At 8–10 sites, any sampler that spreads fills the sector through SQD's α×β product, so
everything, uniform noise included, is exact. Beyond about 24 qubits the limit is how
many strings the classical diagonalization can afford. The test is therefore: keep the K
most frequent strings per spin, diagonalize in that K² subspace, and compare.

L = 10, U/t = 4 (63,504 determinants), error in mEh (t = 1 Eh):

| K strings per spin | exact ground state's selection | **U = 0 ground state (classical)** | interacting Krylov (quantum) | U = 0 Krylov twin | CISD (RHF basis) |
|---|---|---|---|---|---|
| 50 | 880 | **878** | 1535 | 4552 | 788 |
| 100 | 258 | **285** | 537 | 4552 | 498 |
| 200 | 3.2 | **3.9** | 7.1 | 509 | 233 |

- The free-fermion ground state picks nearly the same strings as the exact ground state.
- The quantum Krylov samples pick worse ones.

SQD uses the samples only to choose single-spin strings, and those marginals are
dominated by one-body physics that mean-field states capture. This agrees with the
published critique of quantum-selected CI. **SQD is not a route to quantum value at this
scale, whatever circuit feeds it.**

## 2. Interacting dynamics on MQ02, with free-fermion-twin training

`qcl/apps/xxz_dynamics.py`, analysed by `qcl/apps/xxz_mitigation_analysis.py`.

**Circuit.**
- The open XXZ chain H = Σ J(XX + YY) + Δ ZZ, quenched from the Néel state, is native to
  MQ02's linear chain.
- One Trotter step (dt = 0.25) is a brick wall of bonds, three CZ each: 33 CZ at
  12 qubits, 69 at 24.
- At Δ = 1 (Heisenberg) the circuit is interacting dynamics, not a fermionic Gaussian
  circuit.

**Twins.**
- With Δ = 0 the same circuit is the XX model, which is free fermions after
  Jordan–Wigner; its observables are classically computable at any size.
- Every bond uses the same three-CNOT block (`circuits.xxz_bond`, Vatan–Williams), and
  compilation does no two-qubit resynthesis (`CircuitSpec.optimization_level=1`).
- Twins (J = 0.6, 1.0, 1.4) and target therefore run identical gates; the compiler checks
  that the CZ counts match.

**Mitigation**, following Montanaro & Stanisic (arXiv:2102.02120):
- **Post-selection:** total S^z is conserved, so shots outside the initial sector are
  discarded.
- **Pauli twirling** of every CZ (`CircuitSpec.twirl_seed`, 8 random frames × 500 shots).
- **Corrections learned from the twins:**
  - an affine map per observable class;
  - a global depolarizing fidelity f, defined by noisy − mixed = f·(exact − mixed),
    where "mixed" is the maximally mixed S^z = 0 state.

**Observables** (all from Z-basis samples): ⟨Z_i⟩, ⟨Z_i Z_{i+1}⟩, and the staggered
magnetization m_s = (1/N) Σ (−1)^{i+1} ⟨Z_i⟩.

Hardware runs (MQ02, 2026-11-23): 28 circuits without twirling (4000 shots each), then
the same 28 twirled. 252 jobs, 1,190 CNY.

Staggered magnetization (post-selected), against the ideal Trotter circuit:

| N | steps | CZ | exact | untwirled | twirled | twirled + depolarizing rescale (f) | twirled + affine |
|---|---|---|---|---|---|---|---|
| 12 | 1 | 33 | +0.333 | +0.310 | +0.348 | +0.361 (0.97) | +0.359 |
| 12 | 2 | 66 | −0.151 | **+0.184** | −0.025 | −0.061 (0.41) | −0.027 |
| 12 | 3 | 99 | +0.072 | +0.240 | +0.055 | +0.158 (0.35) | +0.064 |
| 12 | 4 | 132 | +0.073 | +0.208 | +0.084 | +0.247 (0.34) | +0.098 |
| 24 | 1 | 69 | +0.313 | +0.249 | +0.268 | **+0.307** (0.87) | +0.295 |
| 24 | 2 | 138 | −0.144 | **+0.045** | −0.075 | **−0.136** (0.55) | −0.072 |
| 24 | 3 | 207 | +0.090 | +0.166 | +0.038 | +0.159 (0.24) | +0.050 |

Neighbour correlations, RMS error over bonds:

| N | steps | twirled | twirled + affine |
|---|---|---|---|
| 12 | 1 | 0.059 | 0.041 |
| 12 | 2 | 0.083 | 0.098 |
| 12 | 3 | 0.180 | 0.138 |
| 12 | 4 | 0.183 | 0.135 |
| 24 | 1 | 0.084 | 0.055 |
| 24 | 2 | 0.101 | 0.066 |
| 24 | 3 | 0.194 | 0.136 |

Untwirled, the twin-trained scaling did not help:
- the twins show the same distortion as the target (at 2 steps, the J = 1 twin gives
  +0.067 against an exact −0.263);
- but no common scale relates noisy to exact values, which points to coherent errors.

Findings:
1. **Without twirling, the device gets the dynamics wrong from two steps on.** The
   staggered magnetization keeps the wrong sign. The errors are coherent, so a rescaling
   learned from twins cannot fix them.
2. **Twirling turns them into damping.** The sign and the oscillation come out right up
   to 132 CZ at 12 qubits and 207 CZ at 24 qubits, past the ~100-CZ limit seen for the
   SQD circuits.
3. **The depolarizing fidelity learned from the J = 1 twin** restores the amplitude at
   24 qubits for one and two steps (m_s 0.307 vs 0.313, −0.136 vs −0.144). Once
   f < 0.35 it amplifies noise and overshoots.
4. **The affine twin correction** lowers the neighbour-correlation error by 25–35 % in
   most settings.
5. **Site-resolved errors** after mitigation are 0.04–0.06 for one step and 0.13–0.18
   for three. That is qualitatively right dynamics, not yet quantitative at depth.

## 3. What this establishes, and what it does not

- **This circuit family is not classically trivial.** Interacting Trotter dynamics is not
  a sum of a few Slater determinants. Its mitigation scales: the twins stay exactly
  computable at any size, and post-selection and twirling cost no extra qubits.
- **These instances are still classically easy.** 24 spins at total time t ≤ 0.75 have
  little entanglement, and a matrix-product-state simulation reproduces them easily.
  Quantum value needs longer times or 2D lattices, i.e. depths of several hundred to
  thousands of CZ at the same or better fidelity.
- **Next steps:**
  - fit the noise model on more twin circuits (random couplings per bond);
  - map the Hubbard ladder on the QZ03 grid, about 6L CZ per step instead of the chain's
    routing cost;
  - give the twins the same gates in time as well (identical schedule).
