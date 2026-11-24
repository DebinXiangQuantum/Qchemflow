# Beyond classically simulable circuits: Krylov sampling and interacting dynamics (2026-11-23)

Section 3 maps the hardware experiment onto the spin dynamics of the stretched hydrogen
chain.

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

## 3. The chemistry: spin dynamics of the stretched hydrogen chain

`qcl/apps/hchain_spin_model.py` (STO-6G, as in `hydrogen_chain.py`; F5090, about 2 min).

**Why this is the same circuit.**
- The equally spaced H_N chain is the benchmark for correlated 1D solids used in
  `hydrogen_chain.py`.
- Stretched, it is a Mott insulator with antiferromagnetic correlations (Motta et al.,
  Phys. Rev. X 10, 031058 (2020)). One electron sits on each atom, and the low-energy
  states are covalent.
- Superexchange reduces the electronic Hamiltonian to the Heisenberg chain
  J(R) Σ S_i·S_{i+1}: 2N spin orbitals become N spins.
- The Δ = 1 circuit evolves exactly that model, since exp(−i t Σ(XX + YY + ZZ)) =
  exp(−i 4t Σ S·S). One circuit step of 0.25 is therefore 1/J of physical time:
  - the quench from the Néel state is the relaxation of a staggered (antiferromagnetic)
    spin polarization of the chain;
  - m_s is the staggered spin density, ⟨S^z_i⟩ per atom.

**How the mapping was checked.**
- J(R) is the H₂ singlet–triplet gap from full CI, with no fitting.
- The spin gaps E(S) − E(0) of H_N from full CI are compared with the Heisenberg chain
  using that J.
- The dynamics use exact full-CI time evolution of all electrons (Löwdin-orthogonalized
  atomic orbitals) from the Néel configuration.

| R (bohr / Å) | J (mEh) | 1 circuit step | H12 S=1 gap: FCI / Heisenberg (mEh) | Néel covalent weight (H8) | max \|m_s error\|, H8, 4 steps |
|---|---|---|---|---|---|
| 2.4 / 1.27 | 191.4 | 0.13 fs | 52.58 / 53.76 | — | — |
| 2.8 / 1.48 | 112.1 | 0.22 fs | 31.10 / 31.49 | — | — |
| 3.2 / 1.69 | 62.2 | 0.39 fs | 17.42 / 17.47 | 0.67 | 0.059 |
| 3.6 / 1.91 | 32.9 | 0.74 fs | 9.240 / 9.230 | 0.80 | 0.034 |
| 4.0 / 2.12 | 16.7 | 1.45 fs | 4.695 / 4.685 | 0.89 | 0.019 |
| 4.5 / 2.38 | 6.85 | 3.53 fs | 1.928 / 1.925 | 0.95 | 0.008 |
| 5.0 / 2.65 | 2.72 | 8.89 fs | 0.765 / 0.764 | 0.98 | 0.004 |

Notes on the table:
- The S = 2 gaps agree as well (H12 at 4.0 bohr: 18.96 against 18.88 mEh).
- **The dynamics error is measured on the dressed Néel state.** Hopping mixes ionic
  (doubly occupied) configurations into the bare Néel determinant. The dressed state is
  its projection onto the covalent manifold, which is what a staggered pinning field
  prepares.
- Its moment is reduced to 0.97 at 4.0 bohr. The error column compares
  m_s(t)/m_s(0) with the spin model.
- From the bare determinant the error is 0.12 at 4.0 bohr. The ionic part dephases
  within a fraction of a femtosecond.

So from about 3.6 bohr (1.9 Å) onward, the spin dynamics of the electrons in H_N are
the Heisenberg circuit's dynamics to a few percent.

**What the MQ02 data are, in chemical terms.** At R = 4.0 bohr, the 24-qubit runs
predict the staggered spin density of an H₂₄ chain over 1.45, 2.90 and 4.35 fs.
- The electronic problem is 24 electrons in 24 orbitals: 7.3 × 10¹² determinants,
  beyond full CI.
- The spin model needs 24 qubits instead of 48.

Two separate errors stand between the hardware and the physics: the Trotter step and the
device.

| H₂₄, R = 4.0 bohr | 1.45 fs | 2.90 fs | 4.35 fs |
|---|---|---|---|
| Heisenberg, converged (dt/20) | +0.301 | −0.182 | +0.028 |
| Trotter circuit as run (dt = 0.25), ideal | +0.313 | −0.144 | +0.090 |
| MQ02 twirled, post-selected | +0.268 | −0.075 | +0.038 |
| MQ02 + twin-learned rescale | +0.307 | −0.136 | +0.159 |

- **The step on hardware is coarse.** It is J·dt = 1 in physical units; the Trotter
  error alone reaches 0.06.
- **Finer steps fix it at a cost.** Trotter error of m_s against the converged dynamics
  at 24 qubits (ideal circuits):

  | time (steps of 0.25) | first order, as run | first order, dt/2 | first order, dt/3 | second order, dt |
  |---|---|---|---|---|
  | 1 | +0.012 (69 CZ) | +0.002 (138) | +0.001 (207) | +0.028 (105) |
  | 2 | +0.038 (138) | +0.008 (276) | +0.004 (414) | +0.018 (174) |
  | 3 | +0.062 (207) | +0.017 (414) | +0.009 (621) | −0.003 (243) |

  - Second order merges the inner half-layers exactly (the bonds in a layer commute),
    so it costs 69k + 36 CZ for k steps.
  - It is the cheapest fix at longer times, not at the first step.
  - Errors near 0.01 need about 1.2–2× the CZ that ran, beyond today's fidelity at
    24 qubits.

**Where this is relevant to chemistry, and the limits.**
- **Exchange-coupled molecules.** The same Hamiltonian, with per-bond J from
  broken-symmetry DFT or CASSCF, is the standard model of exchange-coupled
  polynuclear transition-metal complexes and molecular spin chains.
  - Bond-dependent J only changes the angles in the same three-CZ bond, and the Δ = 0
    twins stay free fermions.
  - A dimerized chain (alternating spacing) is the next concrete instance.
- **NMR spectroscopy.** Isotropic nuclear J coupling is also a Heisenberg interaction,
  and chemical shifts add Z fields. XX plus Z fields is still free fermions, so the
  twins carry over.
  - Spectra need X/Y readout and coupling graphs that are not chains (Seetharam et al.,
  Sci. Adv. 9, eadh2594 (2023), simulated a 4-spin NMR spectrum this way).
- **Not shown here.**
  - The spin model is valid only for stretched, Mott-like geometries; at bonding
    distances (below about 3 bohr) charge fluctuations matter and the full electronic
    problem is needed.
  - 1D chains at these times are easy for DMRG and MPS. Value starts with 2D exchange
    networks and long times.

## 4. What this establishes, and what it does not

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
