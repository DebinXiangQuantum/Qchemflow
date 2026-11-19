# M6b domain packs

Load `load_registry(["molecular", "qmmm", "drug", "synthesis"])`, or pass
`--packs molecular,qmmm,drug,synthesis` to `chemflow-mcp`. Imports are lazy;
no package installation or network access occurs in these adapters. Registry
search/get exposes each operator's scientific meaning and parameters.

## QM/MM

`mm_system(pdb, forcefields, nonbonded="NoCutoff")` delegates system creation
to OpenMM. `pdb` is text or a filename. Serialized System XML and the source
PDB topology are content-addressed artifacts, not opaque process-local objects.
System construction is uncached so changes to source/force-field files are
noticed even when reusing an Executor; downstream energies remain cacheable
by the resulting artifact identity and explicit snapshot.

`qmmm_energy(system, positions_nm, qm_atoms, basis, method="rhf")` reuses
`chemflow.plugins.openmm.QMMMCalculator`. It returns `qmmm_total_energy` in
hartree, with system identity, snapshot hash, QM region, electrostatic embedding,
and model identity. Snapshots are explicit coordinates in nm. CF-E022 rejects
empty, duplicate, invalid, and bond-cutting selections. No link atoms, quantum
forces, periodic electrostatics, or unsupported force-field terms are invented.

For CASCI, set `method="casci", cas=[n_orbitals, n_electrons]`. Default exact
FCI is equivalent **within that QM/MM-CASCI model**, not exact molecular physics.
RHF is a variational upper bound within its QM/MM model. Optional named
refinements/bindings are `qmmm_rhf`, `qmmm_casci`, `qmmm_gpu_fci`, and
`qmmm_sqd_statevector`, `qmmm_sqd_aer`, `qmmm_sqd_gpu`, `qmmm_sqd_qpu`.
Use the same name in `builder.refine` and `builder.bind`; quantum solver options
are passed through `solver_options` to the existing PySCF plugin. SQD bindings
have random effects and are variational upper bounds of the QM/MM-CASCI model
for every sampler, both statically and at runtime. GPU FCI is statically
equivalent; at runtime it is equivalent when converged and a variational upper
bound otherwise. Device, job IDs, and resources retain the plugin's provenance.
GPU/QPU bindings require their real backends; there is no CPU fallback.

## Drug discovery

`ligand_library(ligands={name: smiles})` records RDKit-sanitized canonical
isomeric SMILES and InChIKey. `filter_ligands(library, rules=[...])` supports
`lipinski`, `pains`, and `brenk`, separately or together. Lipinski flags any
MW > 500, logP > 5, HBD > 5, or HBA > 10; it is a strict cutoff screen,
not a claim that one violation disqualifies a drug. PAINS/Brenk use RDKit's
catalogs directly. Output includes passed names and all flagged reasons.
Alerts are not experimental assay outcomes or efficacy predictions.

`descriptors(library)` returns MW, logP, HBD, HBA, and TPSA.
`conformers(library, seed=0, count=10)` uses single-threaded ETKDGv3 and
MMFF94, recording coordinates (angstrom), energies (kcal/mol), seed, and
convergence status. Random effects prevent cached stochastic executions.

Task kind `interaction_energy` takes `complex={"atoms": ...}`, a complete
partition `fragments=[[...], [...]]`, `electronic_model={"basis": ...}`, and
`counterpoise=True|False`. The current reference path supports neutral
closed-shell fragments and an all-electron nonrelativistic model. It refuses
unsupported model controls rather than silently ignoring them. No solvent,
entropy, binding free energy, or geometry relaxation is implied.

`plan.base` builds HF, MP2, CCSD, CCSD(T), and full-space exact FCI paths.
Counterpoise fragments carry `ghost-` atoms on every other fragment. Complex
minus fragments is combined with `combine_linear`: subtracting HF upper
bounds is **not** an upper bound on the interaction energy. The FCI path is
equivalent; full-space FCI is only practical for small complexes. Frozen-task
identity checks reject changed fragment partitions, bases, geometries, or CP
protocols. Ordinary non-counterpoise calculations omit the other atoms' bases.

`dock(library, receptor_pdbqt, center, box, seed)` accepts one ligand per node,
prepares it with meeko, and calls AutoDock Vina. Receptor PDBQT may be a file
or text; center and box dimensions are in angstrom. Output is a
`docking_score`, never an electronic energy. CF-E002 refuses mixing these
kinds, even though both may have energy-like units. Missing Vina/meeko causes
CF-E080. The installed macOS environment exercises this refusal path.

## Synthesis

`search_routes(target, stock="zinc", iterations=100, time_limit_s=120)` runs
`$CHEMFLOW_AIZYNTH_HOME/env/bin/aizynthcli`; default home is
`/Volumes/macsoftware/caches/chemflow/aizynthfinder`. The isolated installation
must be version 4.4.1 and provide `data/config.yml` and its referenced data.
AiZynthFinder is never imported into ChemFlow's NumPy environment.
Failed subprocesses report CF-E080 with the exit status, signal name for negative
statuses, and the last 2,000 characters of both stderr and stdout. Empty streams
are marked `<empty>` so even a silent process failure has a diagnostic reason.

The adapter copies configuration into a temporary directory, preserving policy
order, overrides search limits, selects the requested stock, and runs with a
hard subprocess timeout of search time plus 60 seconds for model startup/output.
It does not modify the installed data. Version 4.4.1's `.json.gz` route output
is read as plain JSON. RouteCandidates records solved status, leaf SMILES,
per-leaf stock membership, scores, reaction count, raw trees, and the stock
file SHA-256. Provenance device is `aizynthfinder:4.4.1`; effect is `random`.
The CLI has no seed option, so an explicit seed is refused rather than ignored.
Missing installations, missing data, failed processes, and timeouts are CF-E080.
No synthetic energies or barriers are attached to routes.

## Independent benchmarks

Run `chemflow bench run PACK NAME --json report.json`:

| Pack | Name | Oracle |
|---|---|---|
| qmmm | water_dimer_qmmm | Independent PySCF point-charge RHF + analytic TIP3P O–O LJ |
| drug | h2_dimer_cp | Explicit ghost-H direct PySCF FCI/HF, (H2)2/6-31G at 3 Å |
| drug | ligand_funnel | Drug controls vs literature rhodanine/quinone PAINS labels |
| synthesis | paracetamol_routes | Acetic anhydride + 4-aminophenol, both in ZINC stock |

The interaction oracles are +0.354128 kcal/mol (FCI) and +0.498135 kcal/mol
(HF). `tests/fixtures/m6b/paracetamol.json.gz` is an unchanged copy of the
reviewer's isolated-environment smoke output (15 routes), used for offline
parser regression tests. Live synthesis tests skip only if the installation
is missing. Optional ensemble averaging and reaction-barrier subtasks are
not implemented in this milestone.
