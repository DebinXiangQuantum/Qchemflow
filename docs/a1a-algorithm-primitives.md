# A1a: algorithm primitives and real-axis AIM responses

Load `molecular,algorithms,dynamics` with `load_registry` or MCP `--packs`.
Operators expose their meaning, typed ports, parameter descriptions and resource
scaling through `registry.search` and `registry.get`. `plan.base` dispatches a
frozen `spectral_function` task to an independently named exact reference graph;
an active-space argument is optional for this task.

## Mathematical contract

`impurity_model(U, eps_bath, t_bath, unit="hartree")` returns the same site-basis
Hamiltonian representation as `materials.lattice_model`. The impurity energy is
`-U/2`, bath energies are `eps_bath`, hybridisations are `t_bath`, the only nonzero
two-electron integral is `(00|00)=U`, and the constant ledger is zero. Integrals
are stored in hartree. Ground-state and preparation nodes select their explicit
particle sector without changing the Hamiltonian.

| Primitive | Meaning | Resource scaling |
| --- | --- | --- |
| `full_subspace(sector)` | Every orthonormal determinant with the specified alpha/beta populations | O(D M) enumeration, O(D) storage |
| `subspace_from_samples(samples, sector, recovery="configuration", seed=7)` | Single-pass SQD particle-count recovery and product-of-spin-strings closure; equal fillings share the union of spin strings | O(shots M + D) time/storage |
| `extend_subspace(subspace, sector, operator, orbital, spin, closure="none")` | Exact support of the ladder image; `singles` adds one layer of all same-spin single excitations, not an implicit product closure | O(D M²) time |
| `rayleigh_ritz(hamiltonian, subspace, n_roots=1)` | Eigenpairs of the Hermitian projected Hamiltonian in the orthonormal determinant basis | O(D M⁴ + D³) time, O(D²) storage |
| `spectral_function` refined `lehmann` | Retarded Lehmann sum from ground, particle and hole spectra | O(grid size × total response dimension) beyond the solves |

Here M is the number of spatial orbitals and D the relevant subspace dimension.
Dense enumeration/diagonalization is limited to 4096 determinants. Configuration
recovery supports at most 62 spin orbitals. These CPU primitives submit no QPU
jobs or shots. Runtime costs record measured wall and CPU time; sampling costs
remain on the actual sampling node, not duplicated on downstream nodes.

`n_roots` selects the leading energies and `vectors_ref` exposed by a `Spectrum`.
Because this reference implementation performs a dense eigensolve, it also
stores the **complete projected resolution** in `resolution_energies_ref` and
`resolution_vectors_ref`. Lehmann uses that resolution, including roots beyond
the leading-root view. It does not truncate the response at `n_roots` or silently
solve outside the selected determinant subspace. This distinction matters:
full-sector basis coverage alone would not justify an exact eight-pole response.

All ladders use alpha modes first, then beta modes, with sign
`(-1)**popcount(state & ((1 << mode) - 1))`. The retarded hole denominator, like
the particle denominator, uses **+i eta**. The demonstration script outside this
project uses a negative imaginary broadening for its hole branch; that is not
the positive retarded spectral density required by the independent oracle.

## Relations, diagnostics and identity

The checker conservatively propagates full-sector certificates from
`full_subspace`; the executor also verifies actual coverage at runtime. The HUM
min-max rule labels Ritz eigenvalue k an upper bound on sector eigenvalue k for
any valid subspace. Full coverage upgrades it to `equivalent`, even if sampling
happened to find every determinant. Response equivalence requires full coverage
in all three sectors and complete projected resolutions. Otherwise it is
`approximate`, regardless of how close its curve happens to be. Existing scalar
chains and linear combinations continue to use `compose` and `combine_linear`.

`SpectralEstimate.omega` is in `unit`; its density has inverse energy units,
also stated in `diagnostics.density_unit`. `sum_rule` is the unrenormalized
trapezoidal integral on the supplied grid; Lorentzian tails outside the grid
are not artificially restored. `coverage` reports the three basis fractions.
Provenance contains upstream sample/subspace/spectrum lineage and artifact
hashes. Agent `run` returns spectral estimates directly; existing scalar
SQLite headline/report storage is unchanged.

A frozen task identifies the model parameters, sector, orbital, spin, omega
grid and eta. Changing any of them produces CF-E120. An L1 gate requires an
equivalent reference of the same identity and grid. Its integral and threshold
are dimensionless; the task gate's energy `unit` identifies the spectral axis.
No scalar-energy subtraction or spectral renormalization is used for this gate.

SQD's existing feedback loop, batching, carryover and selected-CI solvers retain
their behavior. Its recovery and product-closure operations now share helpers
with the primitives. The composition test uses identical samples, one batch,
one recovery round, and no external singles augmentation. Site-basis CCSD
initialization uses a converged site mean-field reference and pulls the heuristic
orbital-rotation generator back to the site basis; molecular initialization is
unchanged. Initialization does not invoke exact diagonalization.

## An assembled sampled algorithm

`examples/aim_spectral_primitives.py` is an unregistered graph assembled from
the primitives. It keeps the sampled ground subspace, then enriches its response
spaces by a particle → ground → particle/hole ladder walk with single-excitation
closure. For U=2, baths [-1.2, 0, 1.2], hybridisations [0.5, 0.6, 0.5], 20,000
shots and seed 7, its coverage is `(16/36, 24/24, 24/24)` and its finite-grid L1
error is approximately **0.04880513**, below 0.05. Its response correctly remains
`approximate`, with a variational sampled ground root. The full-sector graph
agrees with the independent reviewer Fock-space oracle to floating-point accuracy.

### One-layer versus enriched composition

The one-layer graph (N-sector samples, one ladder step with single-excitation closure)
covers `(16/36, 20/24, 20/24)` and reaches L1 ≈ 0.081; no non-full subspace of that form
gets below ≈ 0.052. The enriched walk above covers the response sectors completely and
reaches ≈ 0.049. Both stay `approximate` because the ground subspace is sampled. The
reviewer contract (`tests/test_review_a1a.py`) checks the one-layer graph against 0.1 and
requires the enriched composition to beat it — the framework measuring whether a new
composition of primitives is a better algorithm.
