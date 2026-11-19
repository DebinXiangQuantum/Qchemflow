# APP C: bonds, reactivity, rankings and conformers

The `reactivity` domain pack is included by `load_registry(pack_names())`. Install the
`reactivity` extra for RDKit, FreeSASA, tblite and ASE. Execution is local and classical;
there are no downloads, QPU jobs or sampling shots. Resource models explicitly report
zero GPU/QPU/shot costs and unknown CPU/wall costs before execution.

## Operators and results

Standalone operators return TypedDict-annotated JSON-compatible dictionaries; operator specs declare their
input/output types and descriptions. Indices are zero-based in the supplied geometry.

| Operator | Inputs/parameters | Result |
| --- | --- | --- |
| `enumerate_bonds` | `molecule`; optional `elements="C-H"` | `bonds`: atom pairs, element pairs and canonical `symmetry_class` strings |
| `homolysis` | `molecule`; `atoms=[i,j]` | `fragments`: two molecule dictionaries, including atoms, charge, spin and unit |
| `site_reactivity` | `molecule`, `model`; `level` | `sites`, `top_site`, `evidence="empirical"` |
| `boltzmann_weights` | `energies` in hartree; `temperature_K=298.15` | normalized `weights`, temperature |
| `conformer_search` | `molecule_smiles`; `level`, `count=20`, `rmsd=0.3` Å | energy-sorted `conformers`, method and seed |
| `app_component_energy` | molecule/model dictionaries; method | standard ChemFlow `Estimate` with Hamiltonian/sector witnesses |
| `bond_dissociation`, `ranking` | signed `target`, task `bar_hartree`, component energy ports | standard energy-collection `Estimate`, with vector fields in diagnostics and recorded headlines |

The energy-collection Estimate's scalar slot is the first component for compatibility
with existing storage. **Accuracy and validity use the maximum error across the entire
vector**, never just that scalar. Headlines expose `values_hartree`, `ranking`,
`certified_pairs` and bounds as applicable; evaluation scores these structured fields.

RDKit reads an XYZ block in angstrom and calls `rdDetermineBonds.DetermineBonds` with
the declared molecular charge. Symmetry classes are the sorted pair of
`Chem.CanonicalRankAtoms(mol, breakTies=False)` ranks. Bond perception is a model of
connectivity, not a quantum observable. A BDE uses the lexicographically first bond in
each selected class; classes are ordered by their representative atom indices.

Homolysis keeps the parent coordinates, assigns neutral fragments and the minimum
spin `2S` compatible with each electron-count parity. It rejects charged parents,
ring cleavages and cases that do not produce exactly two fragments.
These restrictions avoid silently inventing fragment charge or spin partitions.

## Signed targets

```yaml
kind: bond_dissociation
molecule: {atoms: "H 0 0 0; H 0 0 0.74"}
electronic_model: {basis: sto-3g}
bonds: all                  # or C-H, O-H, or [[0, 1], ...]
fragments: rigid
```

```yaml
kind: ranking
items:
  - name: short
    molecule: {atoms: "H 0 0 0; H 0 0 0.74"}
  - name: long
    molecule: {atoms: "H 0 0 0; H 0 0 1.2"}
electronic_model: {basis: sto-3g}
quantity: energy
```

Both schemas forbid extra fields. Ranking requires unique names, the same elemental
formula and the same total charge. It compares electronic energies, not free energies
or concentrations; differing spin states are explicit molecular inputs.

The base plan computes every molecule with HF, MP2, CCSD(T) and an FCI screening
reference. Open-shell components use ROHF orbitals; PySCF's unrestricted post-HF
implementations handle the open-shell correlation calculations. One-electron
components have zero correlation energy and bypass empty-excitation MP2/CC kernels.
FCI is capped at **250,000 alpha/beta determinants per component**, checked before SCF.
It never silently substitutes a cheaper reference. Full orbital spaces and all-electron
nonrelativistic Hamiltonians are required; ECPs, frozen cores and DFT are refused.

For each bond class,

`BDE = E(fragment 1) + E(fragment 2) - E(parent)`.

The FCI reference includes **both** fragments. Task identity checks cover every
component, model, ordered item/bond selection, error bar and FCI reference method.

## Relations and validity

Every implementation has a V1a denotation and axiom. Component FCI/HF relations use
the existing exact-diagonalization/variational lemmas with exported Hamiltonian and
sector witnesses. MP2 and CCSD(T) use the no-ordering-law fallback. In particular,
subtracting variational energies does **not** make BDEs variational bounds.

The `energy_collection_exact` lemma applies ordinary linearity componentwise and
the definition of sorting: every upstream component must have a checked equivalent
relation, and the vector must match the declared component mapping. The independent
kernel verifies these premises; otherwise the result is approximate. This adds no
claim of experimental accuracy or basis-set completeness.

For ranking, a pair is included only if its gap is strictly greater than the sum of
its two error bars. The task's absolute-error bar is the conservative default; an
already proved exact component can use zero. These pairs are **conditional on those
error bounds**, not newly proved physical accuracy bounds. Ties retain input order
and are never certified by a zero gap. Failing accuracy gates remain failures even
when the approximate order happens to be right.

V1c supports the `geometry_scale` family, scaling each item's geometry about its first
atom and rebuilding rigid fragments consistently. The metric is the maximum component
error. Sampling/interpolation is empirical validity evidence, not an interval proof.
Scaling that changes connectivity, defeats bond perception or exceeds the reference
cap is reported as unevaluable rather than silently retaining the original graph.

The evaluation harness accepts category `bond` with `expected.values_hartree`, and
category `ranking` with `expected.ranking`. Vector lengths must match and every BDE
must pass the task's error bar. Ranking scoring requires exact name order.

Pack cards live in `cards/data/reactivity/`; `load_cards(registry)` includes cards for
loaded operators. The no-argument loader retains the core molecular catalog. Agent
card tools and registry search discover the pack's recipes and operators.

## Empirical site reactivity

For **each C–H hydrogen**, not merely each symmetry class, compute a rigid homolytic
BDE at `level.method` (`hf`, `mp2`, `ccsd_t`, or capped `fci`). Optional `functional`
and `dispersion` may be `null`; non-null unsupported corrections are rejected.

FreeSASA uses RDKit periodic-table van der Waals radii and a 1.4 Å probe. The isolated
hydrogen area is `4*pi*(r_H + 1.4)^2`; `sasa_fraction` divides the hydrogen's per-atom
SASA by that area. Sites contain `atom`, `bde_hartree`, `bde_kcal`, `sasa` (Å²),
`sasa_fraction`, and

`score = sasa_fraction / (1 + exp((bde_kcal - 90) / 3))`.

The implementation uses a stable logistic evaluation. `top_site` is the highest-score
hydrogen, or null when there are no C–H sites. **This is empirical/heuristic evidence**:
the 90 kcal/mol center and 3 kcal/mol scale are stipulated model parameters, not a fitted
calibration or a validated metabolism predictor. Enzyme binding, kinetics, electronic
excited states, solvent, tunneling and accessibility in a protein are not modeled.

## Conformer and population models

ETKDGv3 uses seed 42 and one embedding thread. Every embedding is optimized with
either ASE BFGS plus tblite GFN2-xTB (`level.method: gfn2`) or RDKit MMFF94 (`mmff`).
Failed optimization is an error, not a reported converged conformer. Deduplication
uses RDKit `GetBestRMS` on hydrogen-stripped molecules and keeps the lowest-energy
representative. Output atom strings retain explicit hydrogens; energies are hartree.
Search currently requires a connected, closed-shell SMILES with at least one heavy atom.
GFN2 and MMFF energies are model-defined potentials and must not be mixed with each
other or interpreted as FCI references. Finite searches do not guarantee the global
minimum, exhaustive coverage, degeneracy counts or identical results across package
versions/platforms.

Boltzmann weights subtract the minimum energy before exponentiation, with CODATA
Hartree energy and exact SI Boltzmann constant. They are populations of the supplied
equal-degeneracy states at positive temperature, not vibrational/rotational Gibbs
populations or a completeness claim about a conformer search.

## References

- RDKit documentation: `MolFromXYZBlock`, `rdDetermineBonds.DetermineBonds`,
  `CanonicalRankAtoms`, ETKDGv3 and `rdMolAlign.GetBestRMS` (installed APIs).
- Riniker and Landrum, *J. Chem. Inf. Model.* **55**, 2562–2574 (2015), ETKDG.
- Mitternacht, *F1000Research* **5**, 189 (2016), FreeSASA.
- Bannwarth, Ehlert and Grimme, *J. Chem. Theory Comput.* **15**, 1652–1671 (2019), GFN2-xTB.
- Halgren, *J. Comput. Chem.* **17**, 490–519 (1996), MMFF94.
- Larsen et al., *J. Phys.: Condens. Matter* **29**, 273002 (2017), ASE.
- Szabo and Ostlund, *Modern Quantum Chemistry*, HF, correlation and FCI.
- McQuarrie, *Statistical Mechanics*, canonical populations.

Implementation and validation are offline using the supplied installed packages.
