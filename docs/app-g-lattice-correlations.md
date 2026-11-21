# APP G: lattice observables, DMRG and conventional Tc

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

Load the `materials` pack (or `load_registry(pack_names())`). The operators are
searchable in the registry; `pairing_sqd`, `dmrg_block2`, and `allen_dynes_tc`
cards describe their controls and limitations. All new implementations execute
classically and report zero GPU time, QPU time/shots and simulator shots. The SQD
workflow separately accounts for the shots used to obtain its configurations.

## Correlations and state artifacts

`correlation_function(state=ground_energy)` returns real `n_sites × n_sites`
matrices named `spin_spin`, `density_density`, and `pair_s_wave`:

- `spin_spin[i][j] = <S_i · S_j>`, including both transverse spin-flip terms.
- `density_density[i][j] = <n_i n_j>` (not the connected density correlation).
- `pair_s_wave[i][j] = <Delta_i† Delta_j>`, with `Delta_i = c_i,down c_i,up`.

Supplying `site_pair=[i,j]` and `quantity` selects a dimensionless scalar
`Estimate`. The default quantity is `pair_s_wave`.

Exact sector diagonalization and PySCF FCI export `diagnostics.state_ref` as a
content-addressed artifact. SCI and SQD already export their Ritz vectors; all
ground-energy results now also expose the sector in diagnostics. Vectors use
`sector_states` order (combinations of alpha occupations, then beta occupations),
with alpha/up orbitals before beta/down orbitals in the Jordan–Wigner ordering.
PySCF CI-string order is explicitly converted, not assumed identical. Correlations
are computed with creation/annihilation parity signs, including intermediate
states outside the particle sector. The bra and ket remain in that sector.

The lattice HF control exports a site-basis Slater determinant from its occupied
RHF or UHF orbitals. This export is limited to 50,000 sector determinants; an HF
energy itself remains available above that limit. Unconverged/unstable HF is an
optional refused control, not a fabricated reference.

## Pairing task

```yaml
target:
  kind: pairing_correlation
  lattice: {model: hubbard, geometry: ring, n_sites: 4, t: 1, U: 4, unit: hartree}
  sector: {n_alpha: 2, n_beta: 2, n_orb: 4}
  target_spin: 0
  site_pair: [0, 2]
  quantity: pair_s_wave
gates:
  - {kind: abs_error_vs_exact, threshold: 0.000001, unit: dimensionless}
  - {kind: relation_at_least, relation: approximate}
```

The target forbids unknown fields. `quantity` can also be `spin_spin` or
`density_density`. The base plan uses exact sector FCI as a screening-only
reference and an optional HF control. Scripted proposals use sampled LUCJ
configurations, classical SQD diagonalization, then classical correlations.
Preparation and sampling seeds are fixed. Increasing shots can improve coverage
but is not guaranteed to meet a specified accuracy bar.

Identity checks freeze lattice parameters, particle sector, total spin, quantity
and site pair. The `hubbard_u` validity axis varies U without changing the
observable. There is no variational bound on a correlation matrix element:
the no-ordering-law fallback proves `approximate` for SQD/SCI/HF correlations,
even when a small sampled space happens to reproduce the exact answer.
The existing expectation-evaluation lemma gives equality for an exact source;
the certificate kernel checks the upstream exact-state proof explicitly.
Validity maps retain `sampled_valid` for screening references: exporting an
exact vector does not turn a reference into a certified deployment candidate.
Its independent state checks and relation certificate are still retained.

Eval suites accept `expected: {value: <number>, unit: dimensionless}`. Correctness
means exactly `abs(answer - value) <= threshold`, with answer and the task's
accuracy bar converted to the expected unit. `value_hartree` remains supported
for existing energy tasks; the two forms cannot be mixed.

## block2 DMRG

Refine a lattice `ground_energy` with `dmrg`, bind `dmrg_block2`, and set positive
integer `bond_dimension` and `sweeps` (defaults 64 and 10). `seed` defaults to 11.
Install the `dmrg` extra for block2 0.5.4.

The driver receives the lattice spatial integrals and constant ledger. With a
total-spin target it uses SU(2); otherwise it uses SZ with the requested particle
counts. One thread, a private temporary scratch directory and a fixed random seed
make runs reproducible. Scratch is cleaned after finalization. The reported
energy is recomputed as `<MPS|H|MPS>/<MPS|MPS>`, **not** an extrapolation of sweep
energies. Diagnostics include norm, final discarded weight, actual sweep count,
bond dimension and seed. Early convergence can finish before the sweep limit.

The implementation axiom states that this is the expectation in the final
normalized MPS of the requested sector. Together with Hermitian lattice
integrals and sector decomposition, `variational_principle` proves
`variational_upper_bound`. A small discarded weight does not certify a numerical
error bar. No dense vector or correlation evaluation is exported from DMRG.

## Allen–Dynes (McMillan)

`allen_dynes_tc(lambda_ep, omega_log_K, mu_star)` is a pure model function:

`Tc = omega_log_K / 1.2 * exp(-1.04 * (1 + lambda_ep) /
(lambda_ep - mu_star * (1 + 0.62 * lambda_ep)))`.

Its result contains `tc_K`, `unit: K`, and `model: Allen-Dynes (McMillan)`.
The inputs must come from the user or an external EPW calculation; ChemFlow does
not infer electron–phonon coupling from a Hubbard model. The intended regime is
lambda ≲ 1.5 and mu* ≈ 0.1–0.15. Positive lambda, frequency and pairing denominator,
nonnegative mu*, and finite numeric inputs are required. Inputs outside the
recommended regime are not silently clipped. Strong-coupling/shape factors f1
and f2 are omitted. The denotation explicitly defines a model quantity; there is
no ordering/equality claim relative to experimental Tc.

## Limitations and references

Dense sector-vector export and explicit algebra scale combinatorially. Only real
site-basis correlations and on-site s-wave pairs are supported; no d-wave or
unconventional Tc claim is made. Degenerate exact ground states can have
state-dependent correlations. Spin-contaminated HF correlations remain approximate.

- Szabo and Ostlund, *Modern Quantum Chemistry*, chapters 2–3: second
  quantization, determinant expectations and the variational principle.
- White, *Phys. Rev. Lett.* **69**, 2863 (1992): density matrix renormalization.
- Zhai et al., *J. Chem. Phys.* **159**, 234801 (2023): the block2 package.
- Allen and Dynes, *Phys. Rev. B* **12**, 905 (1975): modified McMillan expression.
- Poncé et al., *Comput. Phys. Commun.* **209**, 116–133 (2016): EPW inputs.

The installed block2 driver source and docstrings are the offline API reference.
