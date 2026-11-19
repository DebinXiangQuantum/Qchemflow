# APP A: molecular electronic properties

## Registry and execution

Load the `properties` pack (or `load_registry(pack_names())`). It registers the
`ground_energy` refinement `dft` / binding `dft_pyscf`, and the typed operators
`solvation_free_energy`, `frontier_orbitals`, `charge_energy_difference`, and
`eom_charge_energy`. All bindings have the classical execution role and known
zero QPU time/shots, simulator shots, and GPU time. CPU and wall time are measured,
not predicted as zero. Cards live in `src/chemflow/cards/data`; default card
validation is scoped to refinements available in the supplied registry.

DFT consumes the **MolecularData molecule from its Hamiltonian input**, never a
reconstructed active-space Hamiltonian. Any explicitly selected active space is
refused with CF-E080, even if its size happens to equal the full basis. Closed-shell
calculations use RKS; open-shell calculations use UKS. `functional` is a PySCF XC
string. `dispersion` is null, `d3bj`, or `d4`. D3 uses
`dftd3.interface.RationalDampingParam(method=functional)` (BJ rational damping);
D4 uses `dftd4.interface.DampingParam(method=functional)` and the molecular charge.
Both use nuclear charges and **bohr** coordinates from PySCF. The functional must
have parameters in the dispersion package when dispersion is requested.

Install the `dft` extra for dftd3/dftd4 and the existing `xtb` extra for tblite.
No executable download or network access occurs during calculation.

## Signed target schemas

All target and nested level/solvent models forbid extra fields. Common fields are
`molecule` (atoms, optional charge, integer twice-spin, coordinate unit) and
`electronic_model` (basis and all-electron nonrelativistic model). These targets
use the full orbital space; they do not accept an active-space field.

```yaml
kind: solvation_free_energy
molecule: {atoms: 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'}
electronic_model: {basis: 6-31g*}
level: {method: dft, functional: b3lyp, dispersion: null}
solvent: {model: pcm, name: null, eps: 78.3553}
```

`level.method` is `hf`, `dft`, or `gfn2`; a DFT functional is required, and
functional/dispersion fields must be null for other methods. PCM requires positive
finite `eps` and HF or DFT. It is **IEF-PCM electrostatic solvation only**, not a
complete experimental solvation free energy. SMD requires a solvent name and
HF/DFT; ALPB and GBSA require a solvent name and GFN2. tblite supplies its named
solvent parameters. The result is E(solvent) − E(gas), in hartree, at the same
geometry. Geometry-only dispersion cancels in this difference. SMD support is
detected by actually constructing/running PySCF SMD: an unavailable compiled
module produces CF-E080 with PCM and ALPB alternatives in the saved CLI summary.

```yaml
kind: frontier_orbitals
molecule: {atoms: 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'}
electronic_model: {basis: 6-31g}
level: {method: hf, functional: null, dispersion: null}
quantity: gap
spin: alpha
```

`quantity` is `homo`, `lumo`, or `gap`; `spin` is `alpha` or `beta`. Results come
from converged HF/DFT orbital energies (RHF/RKS for closed shells, UHF/UKS for open
shells), selecting occupied/virtual orbitals by their occupations. A selected
channel must have both. The orbital gap is **not** a physical fundamental gap.
Additive dispersion does not alter these fixed-geometry SCF orbital eigenvalues.

```yaml
kind: ionization_energy  # or electron_affinity
molecule: {atoms: 'Li 0 0 0; H 0 0 1.6'}
electronic_model: {basis: sto-3g}
final_spin: 0.5
```

`final_spin` is nonnegative integer/half-integer S; the final molecule uses
twice-spin 2S and charge q+1 for IP or q−1 for EA. IP = E(N−1) − E(N),
EA = E(N) − E(N+1), including nuclear constants. The initial spin is specified
by the molecule. These are vertical Born–Oppenheimer energies, not adiabatic or
finite-temperature quantities. FCI in both specified spin sectors is the
screening reference. Each sector must have at most **100,000 determinants**,
computed as C(norb,nalpha) C(norb,nbeta) before integral/FCI construction.
Excessive targets are refused with the dimension, cap, and smaller-basis advice.

Controls compose the same sector energies with HF and CCSD(T), using ROHF for
open-shell HF and its UHF conversion for coupled cluster. The third control uses
`pyscf.cc.eom_rccsd.EOMIP/EOMEA`; EOMEA's attachment excitation energy is negated
to obtain EA. Restricted EOM requires a closed-shell initial state and final
doublet; otherwise that optional control is refused, not substituted silently.
Negative EA (unbound within the chosen finite basis/model) is retained.

## Relations, validity, and orchestration

DFT and EOM have implementation axioms with no ordering law, hence proved
`approximate` relations. Delta-HF and delta-CCSD(T) likewise have no ordering law.
Delta-FCI uses the existing sector-decomposition, exact-diagonalization, and
signed-linear-combination lemmas to prove `equivalent`; no new physical inequality
is asserted.

Solvation and frontier targets are explicitly **model-defined**. Their denotations
name that fact and their runtime witnesses contain the declared model. The new
`declared_model_evaluation` lemma is definitional equality, conditioned on a
matching model witness; it is not experimental calibration. Certificates are
replayed by the independent kernel and depend on the implementation axioms.

Model-defined base plans intentionally have no controls: the screening reference
is the declared calculation and is itself an eligible answer. The critic selects
it when gates pass, and the report lists its classical deployment and model label.
The eval harness accepts category `property`, scores `expected.value_hartree`, and
retains the screening-reference score independently of the selected control.
Safe task identifiers may contain underscores as well as hyphens.

All four targets expose the `geometry_scale` validity axis. Derived plans scale
both embedded molecular specifications and ordinary molecule nodes, preserving
the original signed task. These sampled validity maps concern numerical agreement
with the declared screening reference, not physical validation of a density
functional or solvent model. Convergence failures remain failures, not estimates.

## References and limitations

- PySCF 2.14 APIs: `dft.RKS/UKS`, `solvent.pcm.PCM` (IEF-PCM),
  `solvent.smd.SMD`, `cc.eom_rccsd.EOMIP/EOMEA`, `fci` and `scf`.
- dftd3 Python API: `DispersionModel`, `RationalDampingParam`; Grimme et al.,
  J. Comput. Chem. 32, 1456–1465 (2011), BJ damping.
- dftd4 Python API: `DispersionModel`, `DampingParam`; Caldeweyher et al.,
  J. Chem. Phys. 150, 154122 (2019), charge-dependent D4 dispersion.
- tblite Python `Calculator.add('alpb-solvation'/'gbsa-solvation', name)`;
  GFN2-xTB and ALPB/GBSA parameters are supplied by the installed library.
- Existing proof references: Courant–Fischer, Szabo–Ostlund, and linearity over R.

No ECP, frozen-core model, periodic system, solvent geometry relaxation, empirical
accuracy guarantee, or large-system FCI is supplied here. Small bases can be poor
models for electron attachment even when the numerical FCI answer is converged.
SMD remains unavailable when the installed PySCF build lacks its compiled module.
PySCF/tblite calls run in the existing classical adapter process, as do the
pre-existing molecular solvers; no new external-engine service is introduced.
