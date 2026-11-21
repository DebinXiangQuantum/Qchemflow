# APP B: geometry, scans and reaction paths

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

The `paths` domain pack provides classical geometry transformations, constrained
optimization, PES and dimer scans, climbing-image NEB, bidirectional IRC and Eyring
rates. `load_registry(pack_names())` loads it; `load_registry(["paths"])` also loads
its molecular and counterpoise dependencies. Install the `paths` optional extra
for geomeTRIC, RDKit, ASE, tblite and Sella. Execution does not require a network.

## Operators and units

All atom indices are zero-based. Geometry output is in angstrom. Energies are in
Hartree, angles in degrees, rates in s^-1 and Wigner input frequencies in cm^-1.
Outputs are dictionaries; energy curves are validated `EnergyCurve` dictionaries.

| Operator | Inputs and parameters | Output |
| --- | --- | --- |
| `geometry_transform` | `molecule`; `operation`, `atoms`, `value`, optional `fragment` | Molecule dictionary |
| `constrained_optimize` | `molecule`, `model`; `level`, `constraints` | Molecule fields, `molecule`, `energy_hartree` |
| `pes_scan` | Ordered `point0`, … electronic-energy inputs; signed `target`, `method` | `values_hartree`, `coordinates`, relation, provenance |
| `interaction_scan` | Ordered interaction-energy inputs; signed `target`, `method` | Same curve schema |
| `neb_path` | `reactant`, `product`, `model`; `level`, `images=7`, `interpolation="idpp"`, `optimizer="fire"`, `fmax=0.02`, `maxsteps=1000` | `images`, `energies_hartree`, `climbing_image`, `converged` |
| `irc` | `ts`, `model`, optional paired `reactant`, `product`; `level`, `fmax=0.02`, `maxsteps=500` | `endpoints`, `connects` |
| `connected_ts` | TS `molecule`, `model`, `reactant`, `product`; HF `level`, `kind="ts"` | Refined molecule, only after one-imaginary-mode and connecting-IRC checks |
| `rate_constant` | `barrier`, `temperature_K`; optional `tunnelling="wigner"`, `frequency_cm1` | `k_per_s`, `kappa`, temperature, model label |

`geometry_transform` supports `set_bond`, `set_angle`, `set_dihedral` and
`translate_fragment`. RDKit `DetermineConnectivity` supplies the graph, not a
handwritten distance cutoff. Without an explicit fragment, cutting the defining
bond determines the moved component. A ring cut that does not separate the graph
is rejected. The other side is fixed, and each side moves rigidly. Translation
moves an explicit fragment relative to its complement along their centroid axis.
Undefined axes and coincident centroids are rejected rather than guessed.

`level` is `{method: hf|dft|gfn2, functional: string|null,
dispersion: string|null}`. DFT requires an explicit functional. HF/DFT use PySCF
and `pyscf.geomopt.geometric_solver.optimize(..., constraints=file)`; the temporary
file contains geomeTRIC `$set` constraints with one-based indices. GFN2 uses
`tblite.ase.TBLite`, ASE `FixInternals` and BFGS. Additional dispersion corrections
are currently rejected, not silently dropped. Constraints have exactly the fields
`{type: bond|angle|dihedral, atoms: [...], value: number}`.

The optimization result also exposes top-level `atoms`, `charge`, `spin` and
`unit` when supplied, so the entire output can feed a downstream `molecule` port.
The energy is evaluated again at the returned geometry.

## Signed target schemas

Both targets forbid extra top-level fields. Coordinates must be finite, with
positive bond lengths/separations and angles strictly between 0 and 180 degrees.
Scans currently evaluate neutral, closed-shell systems in all-electron
`nonrel_bo` models, using the existing interaction component code.

```yaml
kind: pes_scan
molecule: {atoms: "H 0 0 0; H 0 0 0.74"}
electronic_model: {basis: sto-3g}
coordinate: {type: bond, atoms: [0, 1], values: [0.6, 0.74, 1.2, 2.0]}
mode: rigid
relax_level: null
```

For `mode: relaxed`, supply a `relax_level` instead of null. Each point is first
transformed rigidly, then constrained-optimized once at that level. All evaluation
methods share that resulting geometry. The FCI reference is therefore exact only
for the finite-basis electronic problem at that geometry, not for the optimizer
or a globally relaxed surface.

```yaml
kind: interaction_scan
complex: {atoms: "H 0 0 0; H 0 0 0.74; H 3 0 0; H 3 0 0.74"}
electronic_model: {basis: sto-3g}
fragments: [[0, 1], [2, 3]]
counterpoise: true
separations: [2.5, 3.0, 4.0]
```

Fragments must partition the complex into exactly two nonempty sets. The second
fragment is moved rigidly. At each separation, `component_specs` builds the
complex and monomer/ghost-basis systems; `component_energy` evaluates them;
`scan_interaction` computes complex minus both fragments. These are interaction
energies, not binding free energies.

The pack's `base_plan(record, active=None)` constructs all geometry and scalar
energy nodes explicitly, followed by the ordered curve collector. It includes
HF, MP2, CCSD and CCSD(T) controls and a screening-only FCI reference. Signed
identity checks compare the full dependency graph, including constraints,
coordinate order, counterpoise settings, electronic model and reference method.
The scripted agent selects a gate-passing control rather than returning the
reference as its production answer. Cards and registry search expose the operators.

### Verified reaction barriers

`reaction_barrier` extends the existing `reaction_energy` target schema with a
required atom-mapped `product` molecule. Its `species` must contain exactly one
`role: ts, coefficient: 1` species and one `role: minimum, coefficient: -1`
reactant supermolecule, each with the existing `active` and `sector` declarations.
Reactant, TS and product must have identical ordered elements, charge and spin.
The operator remains `reaction_energy`, so scalar `expected.value_hartree`
scoring, classical controls and the FCI screening reference work unchanged.

The pack reuses the reaction plan but replaces TS optimization with `connected_ts`.
This requires an HF Sella saddle, an HF IRC connecting the declared endpoints,
exactly one significant imaginary frequency and a maximum gradient below
1e-4 Hartree/Bohr. All electronic evaluation methods share this verified geometry.
The reference certifies the active-space energy difference at these fixed
geometries, not the barrier on an exact optimized potential surface. Add
`thermo: {quantity: gibbs, temperature_K: 298.15, pressure_Pa: 101325}` for the
existing RRHO ideal-gas correction and minimum optimization. Product energies do
not enter the forward barrier; product connectivity is a required verification.
The signed identity check rejects removing or replacing that verification.

## Curves, gates and V1

The eval harness accepts category `path` and `expected.values_hartree`. A curve
is correct only when its length matches and **every** point is within the declared
Hartree error threshold. Runtime gates use the maximum absolute pointwise error,
requiring the same coordinates and target identity and a proved reference.
Reports retain the complete energy list; the legacy scalar result index stores
the point count with unit `points`, never a misleading scalar energy.

Each binding has a V1a denotation and implementation axiom. Geometry, optimization,
NEB, IRC and Eyring have conservative no-ordering-law relations. Model-defined
path and rate quantities are explicitly identified as such. Interaction sums use
the existing linear-combination lemmas: subtracting approximate variational
energies does not produce a variational interaction-energy bound.

The only added lemma, `pointwise_equality`, is extensional equality of finite
tuples. Its kernel checks actual point values, coordinates and identity, plus a
proved `equivalent` relation for **every** dependency. It asserts neither
interpolation accuracy nor geometry optimality. Other curves use
`no_ordering_law`, even when all component energies have upper bounds.

V1c axes are `internal_coordinate` for PES scans and `separation` for dimer scans.
The task point is the first supplied coordinate. Instantiation shifts the entire
grid by the requested first-coordinate displacement, preserving grid spacings;
rigid transforms, relaxed constraints and collector metadata shift together.
Invalid family geometries are unevaluable rather than treated as valid samples.
Validity intervals retain V1c's sampled/interpolated, non-universal interpretation.

## NEB, IRC and rates

`images` counts interior images: seven means nine total points. Endpoint atom
ordering, charge and spin must agree. Endpoints remain fixed. The ASE NEB uses
`climb=True` and improved tangents. Collinear initial paths receive a deterministic
transverse bend to break symmetry and avoid interpolating an atom directly
through another; IDPP can further relax the initial band. Failure to converge or
absence of an interior maximum raises an error.

HF/DFT use a small ASE calculator converting PySCF Hartree/Bohr gradients into
eV/angstrom forces. GFN2 uses the ASE tblite calculator in a local persistent
worker. Sella IRC runs in a separate local foreground process because the supplied
macOS wheels load incompatible OpenMP runtimes when Sella and tblite coexist.
The implementation does not enable the unsafe duplicate-OpenMP workaround.
Workers use the same interpreter, no network, and a temporary JAX cache.

IRC refines the supplied saddle with Sella, follows both IRC directions, and
compares RDKit bond sets against both declared endpoints in either orientation.
`connects` is null if no endpoints are declared, false for a connectivity mismatch,
and true only for a matching pair with the same ordered elements. This is an
atom-mapped connectivity test, not graph-isomorphism or stereochemical matching.

Eyring uses `k = kappa * k_B*T/h * exp(-barrier*E_h/(k_B*T))`, with exact SI
`k_B=1.380649e-23 J/K`, `h=6.62607015e-34 J s`, and the CODATA 2018 Hartree
conversion `E_h=4.3597447222071e-18 J`. Wigner uses
`kappa=1+(h*c*100*abs(frequency_cm1)/(k_B*T))^2/24`. The supplied barrier should be
a Gibbs activation free energy for an Eyring free-energy rate. Passing an
electronic barrier does not automatically include entropy or zero-point effects.

## Limitations and references

- All searches are local; neither NEB nor constrained minimization proves global
  optimality. Endpoint preparation and consistent atom mapping remain the caller's
  responsibility. SCF and optimizer failures are not converted to successful data.
- Curve references scale exponentially with full-space FCI. No large-system
  reference or quantum speedup is claimed.
- Interaction scans are rigid dimers; general relaxed intermolecular-coordinate
  scans and extra D3/D4 corrections are not exposed by this pack.
- `reaction_barrier` currently verifies stationary geometries and IRC at HF;
  its species must be atom-mapped supermolecules. Standalone NEB/IRC support HF,
  DFT and GFN2, but a DFT/GFN2 stationary-level barrier target is not claimed.
- References: RDKit `rdDetermineBonds`; Wang and Song, *J. Chem. Phys.* **144**,
  214108 (2016), geomeTRIC; ASE NEB/IDPP and optimizer documentation; Henkelman,
  Uberuaga and Jónsson, *J. Chem. Phys.* **113**, 9901 (2000), climbing-image NEB;
  Smidstrup et al., *J. Chem. Phys.* **140**, 214106 (2014), IDPP; Sella `IRC` API;
  Bannwarth, Ehlert and Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019),
  GFN2-xTB; Eyring, *J. Chem. Phys.* **3**, 107 (1935); Wigner, *Z. Phys. Chem. B*
  **19**, 203 (1932); CODATA 2018 constants. Package API details were checked
  against the installed sources, without network access.
