# APP E: protein–ligand preparation and model-defined binding

The `binding` domain pack is loaded by `load_registry(pack_names())`. Its operators
are classical, with explicit zero GPU time, simulator shots, QPU shots and QPU time.
CPU time and wall time remain unknown until measured. These are not quantum speedups.

## Installation and isolation

Install the pip-installable pieces with `pip install 'chemflow[binding]'`. The
separate Python 3.11 environment owns PDBFixer, OpenFF, openmmforcefields and AmberTools:

```sh
conda create -p ../chemflow-tools-env -c conda-forge python=3.11 \
  openmm=8.6.1 openff-toolkit openmmforcefields openmmtools pymbar pdbfixer \
  mdtraj rdkit ambertools propka crest=3.0.2 xtb=6.7.1
export CHEMFLOW_TOOLS_PYTHON="$PWD/../chemflow-tools-env/bin/python"
```

The default interpreter is the sibling `chemflow-tools-env/bin/python`. Worker
requests and replies are JSON. Scientific logging goes to stderr, which is captured
in CF-E080 diagnostics; calls time out after 1800 seconds. The environment's `bin`
is prepended to PATH for `antechamber` and `sqm`. Neither OpenFF nor
openmmforcefields is imported into ChemFlow's main process. No downloads are
performed by the client. Install force-field data together with the environment.

**Offline environment limitation:** the supplied environment lacks PROPKA. When
PROPKA is installed in the side environment, its predicted pKa values select
HIS/ASP/GLU/LYS states at the requested pH. Otherwise standard pKa values are used
and explicitly recorded as `standard_pKa_fallback`, not represented as PROPKA
results. Install PROPKA in that environment for structure-specific predictions.

## Operators

| Operator | Inputs / parameters | Result |
|---|---|---|
| `protein_structure` | `pdb_text`, `ph=7`, `keep=[]`, `remove_water=True`, optional `chains` | PDB artifact, atom/residue counts, charge, protonation records |
| `ligand_structure` | `source`: PDB text/residue/optional SMILES, SMILES, SDF, or XYZ/charge | SDF artifact, canonical SMILES, formal charge and atom count |
| `complex` | prepared protein and ligand | PDB artifact retaining the ligand pose and residue name |
| `pocket` | complex, `cutoff_A=4` | residues `{id, name, chain, number}` within the cutoff of any ligand atom |
| `ligand_parameters` | ligand, `forcefield="gaff-2.11"` or `"openff-2.2.0"`, `charge_method="am1bcc"` | OpenMM ffxml artifact and partial charges |
| `mmgbsa` | complex, parameters, `frames=0`, `minimize=False`, `seed=0` | `dG_kcal`, three energy components, `stderr`, frames, Estimate provenance |
| `fragment_interaction` | complex, ligand, residue IDs, `level`, `basis`, `counterpoise=True` | per-residue and total Hartree energies, atoms, charges and approximation statement |

PDBFixer fills only internal missing-residue gaps, then missing heavy atoms and
hydrogens. Waters and unlisted heteroatoms are removed. Nonstandard retained
residues must have supported force-field templates; unsupported chemistry fails
rather than acquiring invented parameters. Hydrogen placement is seeded. PDB
ligands use RDKit `AssignBondOrdersFromTemplate` when a SMILES template is given;
explicit hydrogens receive coordinates. SMILES-only sources use seeded ETKDG, not
docking. Ambiguous multiple PDB residues are rejected.

Text artifacts have content-addressed filenames in the local temporary artifact
directory. `read_text_artifact(ref)` reads their PDB/SDF/ffxml text. Copy these files
to permanent storage when archiving a run; clearing the OS temporary directory
invalidates those paths.

## MM/GBSA definition and target

Amber14 (`amber14-all.xml`) and `implicit/gbn2.xml`, NoCutoff, no constraints and
OpenMM's Reference platform define the calculation. Ligands use GAFF 2.11 or
OpenFF 2.2.0 with AM1-BCC partial charges, whose sum is checked against formal charge.
The complex must contain exactly one BNZ, UNL or LIG ligand residue; the input
residue name is preserved. All three subsystem energies use the same coordinates:

`dG = E(complex) - E(receptor alone) - E(ligand alone)`.

Before any evaluation, the added hydrogens within 8 Å of the ligand are relaxed
with every other atom fixed (vacuum Amber14 + ligand force field, one CPU thread,
L-BFGS to 0.01 kJ/mol/nm), and the relaxed complex is evaluated at PDB precision
and recorded as `diagnostics.evaluated_pdb_ref`. Hydrogen positions are not
observed: PDBFixer and RDKit place methyl and hydroxyl hydrogens differently
across versions, and one unrelaxed 1.6 Å H–H contact at the 181L interface
(Val87) turned the single-point ΔG from about −15 to +12.9 kcal/mol. Two
independent implementations of this relaxation agree to about 0.05 kcal/mol, so
the apps-suite gate for this task is 0.1 kcal/mol.

With zero frames this is a single point, optionally after up to 500 minimizer
iterations. Positive frames use seeded 300 K Langevin dynamics (1/ps friction,
1 fs steps, ten steps per frame). Seed zero maps to one to avoid OpenMM's
auto-random seed behavior. The result is the frame mean and naive sample standard
error, without correcting time correlation. A single sampled frame has unavailable
standard error (`null`); a deterministic single point reports zero sampling error.
These short trajectories are diagnostics, not converged free-energy estimates.

```yaml
kind: binding_free_energy
receptor: {pdb_text: "...", chains: [A]}
ligand:
  source: {pdb_text: "...", residue: BNZ, smiles: c1ccccc1}
protocol: mmgbsa
forcefield: gaff-2.11
frames: 0
minimize: false
ph: 7.0
seed: 0
```

Only `mmgbsa` is accepted as a target protocol in this batch; docking, combined
fragment corrections and FEP are not falsely advertised as implemented targets.
The base plan supplies a control and a separate screening reference, both evaluating
the declared model. The reference is **not an experimental oracle**. Identity checks
compare the entire dependency graph, including receptor, ligand, parameters, pH,
sampling and seed. The agent's report states **model_change to experimental binding
free energy**. Eval category `binding` scores `expected: {value: ..., unit: kcal/mol}`;
the unit-aware expected-value format also works for other energy categories.

The V1a `model_evaluation` rule is definitional: equality is to the declared finite
protocol output, not to experiment or an infinite-sampling ensemble average. Its
numerical binding remains trusted, and its certificate does not claim a Lean proof
of OpenMM. Fragment interaction energies use the conservative no-ordering-law rule.
`binding_ph` is a V1c validity axis on [0,14]; both candidate and reference are
reprepared at each sampled pH. Agreement only validates model evaluation, not
physical accuracy. Protonation transitions can be discontinuous; interpolated
validity regions are not interval proofs.

## Pockets, fragment QM and docking

Each selected side chain starts at Cβ. The Cα–Cβ cut receives a hydrogen 1.09 Å
from Cβ toward Cα. GLY and PRO are skipped. Protonation-state formal charges and
electron-count spin parity are applied separately to ligand, side chain and pair.
HF, DFT (specified functional, default PBE) and MP2 are supported in PySCF;
dispersion corrections are rejected rather than silently ignored. Counterpoise
monomers retain the other fragment's ghost basis. Outputs include ligand atoms
`[(element, [x,y,z])]` and uncapped residue atoms
`{id: [(name, element, [x,y,z])]}`, all in Å, for independent reconstruction.

The sum assumes **pairwise additivity**. It excludes backbone interactions,
many-body polarization, solvent and configurational entropy; do not add it to
MM/GBSA without a separately defined double-counting correction. A metal-containing
or strongly correlated residue calls for an explicitly constructed ligand/residue
active region and an **SQD candidate (QPU-deployable)**. This operator does not
silently route a chemically inadequate single-reference calculation onto a QPU.

Vina remains Linux-only and unchanged. To evaluate a docked pose, export its ligand
coordinates to SDF or a ligand PDB residue, restore bond orders with the SMILES
template, prepare the receptor, assemble `complex(protein, ligand)`, parameterize
the ligand, and run `mmgbsa`. Keep both coordinates in the same frame. A Vina score
is neither an electronic energy nor the MM/GBSA result.

## References and limitations

- Eastman et al., OpenMM 7, PLoS Comput Biol 13 (2017), e1005659.
- Maier et al., ff14SB, J Chem Theory Comput 11 (2015), 3696–3713.
- Nguyen, Roe and Simmerling, GB-Neck2, J Chem Theory Comput 9 (2013), 2020–2034.
- Søndergaard et al., PROPKA3, J Chem Theory Comput 7 (2011), 2284–2295.
- Wang et al., GAFF, J Comput Chem 25 (2004), 1157–1174; GAFF2 via AmberTools.
- Jakalian et al., AM1-BCC, J Comput Chem 23 (2002), 1623–1641.
- Boys and Bernardi, counterpoise, Mol Phys 19 (1970), 553–566.
- Genheden and Ryde, MM/PBSA and MM/GBSA review, Expert Opin Drug Discov 10
  (2015), 449–461, doi:10.1517/17460441.2015.1032936.

The bundled 181L reviewer system independently checks the prepared single-point
MM/GBSA energy and two neutral capped-fragment counterpoise energies. It does not
establish experimental accuracy, protonation accuracy, or MD convergence. No FEP,
entropy calculation, metal parameterization or automatic SQD region selection is
provided in this batch.
