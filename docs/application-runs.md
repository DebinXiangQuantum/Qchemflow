# Application runs on F5090 (2026-11-22)

Three applications, each a QCL program in `qcl/apps/`, run on the F5090 server:
2 RTX 5090 GPUs (the other two belong to another user) and up to 16 CPU threads per
process. Quantum circuits were sampled on GPU-hosted statevector simulators; no QPU time
was used. Every quantum result is checked against an exact classical reference computed in
the same plan.

| application | program | system size | quantum part | plan wall time |
|---|---|---|---|---|
| Drug design: T4 lysozyme L99A with benzene | `t4l_quantum_binding.py` | 162 residues, 2616 atoms (whole protein in the MM region) | QM/MM CASCI with SQD: 12 and 28 qubits | 258 s |
| Correlated materials: hydrogen chain equation of state | `hydrogen_chain.py` | H12, 8 spacings | SQD from LUCJ samples, 24 qubits | 6209 s |
| Synthesis: Diels-Alder barrier | `diels_alder.py` | C6H10 transition state, 6-31G* | SQD at 4–12 and 28 qubits | 2325 s |

## 1. T4 lysozyme L99A with benzene (PDB 181L)

```sh
CUDA_VISIBLE_DEVICES=2,3 python qcl/apps/t4l_quantum_binding.py --out t4l
```

Preparation (PDBFixer, PROPKA protonation at pH 7, GAFF2/AM1-BCC benzene, relaxation of
added hydrogens) took 14 s: 162 residues, 2616 atoms, net charge +9. The plan then ran
24 nodes in 258 s:

- MM/GBSA (Amber14/GBn2, whole protein) on the CPU;
- counterpoise MP2/6-31G* for benzene with each of the 12 pocket side chains, on the
  CPU;
- 8 Vina docking runs on the CPU;
- 14 QM/MM energies, on the CPU and both GPUs.

**Binding estimates.** Experiment: ΔG = −5.2 kcal/mol for benzene (Morton & Matthews 1995,
as tabulated in the literature).

| quantity | value | comment |
|---|---|---|
| MM/GBSA single point | −15.4 kcal/mol | overbinds by ~10 kcal/mol: no entropy, no sampling, as expected for single-point MM/GBSA |
| QM/MM interaction energy, RHF/6-31G* | −17.17 kcal/mol | vertical interaction in vacuum (electrostatics, polarization, LJ), not a free energy |
| pocket MP2/6-31G* sum (12 side chains) | +4.2 kcal/mol | pairwise, counterpoise-corrected, unrelaxed hydrogens |

Per-residue MP2 interactions (kcal/mol). Attractive: Met102 −0.93, Phe153 −0.86,
Tyr88 −0.40, Val103 −0.30, Ile78 −0.28, Leu91 −0.26, Leu121 −0.19. Repulsive: Val87
+3.31, Val111 +2.82, Ala99 +1.30. The repulsive contacts are the tightest
side chains around the ring; with hydrogens placed by PDBFixer and not relaxed, and a
small basis that underestimates dispersion, the pairwise sum is not a binding energy.

**Docking (Vina, same box).** Scores in kcal/mol, with the experimental ΔG for comparison:

| ligand | Vina | experiment |
|---|---|---|
| indole | −6.76 | −4.9 |
| toluene | −6.28 | −5.5 |
| n-butylbenzene | −6.25 | −6.7 |
| ethylbenzene | −6.12 | −5.8 |
| n-propylbenzene | −6.10 | — |
| phenol | −5.64 | — |
| p-xylene | −5.58 | −4.7 |
| benzene | −5.48 | −5.2 |

Vina ranks indole first although it binds weakest; its Spearman correlation with experiment
over the six measured ligands is 0.14 (0.6 without indole). Docking is a screen, not
an affinity.

**QM/MM with quantum solvers.** The QM region is benzene (6-31G*, 102 basis functions);
the whole protein is the electrostatic environment. Each energy is computed for the bound
pose and with benzene moved 100 Å out of the protein; the difference is the interaction
energy at that level.

| level | qubits | E(bound) / Eh | interaction / kcal/mol | SQD determinants (bound / separated) |
|---|---|---|---|---|
| RHF | — | −236.018662 | −17.167 | |
| CASCI(6,6), exact FCI | 12 | −236.053113 | −17.102 | |
| CASCI(6,6), SQD from samples only | 12 | −236.047207 | −22.559 | 64 / 49 of 400 |
| CASCI(6,6), SQD + single-excitation extension | 12 | −236.053112 | −17.102 | 361 / 361 |
| CASCI(14,14), exact FCI on GPU | 28 | −236.060879 | −16.958 | |
| CASCI(14,14), SQD from samples only | 28 | −236.050614 | −17.138 | 1024 / 1089 of 11.8 M |
| CASCI(14,14), SQD + extension | 28 | −236.060784 | −16.981 | 277 729 / 263 169 |

- The CCSD-seeded LUCJ distribution is very concentrated. At 100 000 shots the sampled
  subspace holds 49–64 of the 400 determinants of CAS(6,6), and about 1000 of the 11.8
  million of CAS(14,14). Samples alone give energies 6–15 mEh above exact, and the 12-qubit
  interaction energy is off by 5.5 kcal/mol, because the errors do not cancel between the
  bound and separated geometries.
- With the single-excitation extension (now `qcl.qmmm.energy(..., extend=True)`), the
  28-qubit SQD uses 2.3 % of the sector. It lies 0.095 mEh above exact, and its
  interaction energy (−16.98 kcal/mol) is within 0.02 kcal/mol of exact.
- Correlation changes the interaction by only 0.2 kcal/mol: the QM/MM interaction of a
  rigid, closed-shell benzene is dominated by electrostatics, polarization and LJ. That
  is a statement about this system, not about the solver.

## 2. Hydrogen chain H12: equation of state of a correlated 1D solid

```sh
QCL_POLICY_FILE=qcl/apps/applications-policy.toml CUDA_VISIBLE_DEVICES=2,3 \
    python qcl/apps/hydrogen_chain.py --atoms 12 --json hchain.json
```

The equally spaced hydrogen chain (Motta et al., PRX 7, 031059, 2017) is a standard
benchmark for correlated materials. It goes from a weakly correlated, metal-like chain at
short spacing to a Mott-like insulator when stretched, where single-reference methods
fail.

- **Model:** H12, STO-6G, all 12 orbitals active (24 qubits), 8 spacings from 1.4 to
  3.6 bohr. Each spacing runs RHF, CCSD(T), DMRG (M = 512), EWF embedding, GPU FCI, and
  SQD.
- **SQD input:** 100,000 samples of the CCSD-seeded LUCJ circuit on a GPU statevector
  simulator.
- **Run:** plan wall time 6209 s. DMRG took 3616 s and EWF 2367 s. The 16 SQD
  diagonalizations took 109 s, the 8 GPU FCI references 110 s and sampling 3 s.

Error against exact FCI, in mEh per chain (12 atoms):

| R / bohr | FCI E/N / Eh | RHF | CCSD(T) | DMRG M=512 | SQD | SQD, samples only | SQD determinants (of 853,776) |
|---|---|---|---|---|---|---|---|
| 1.4 | −0.518594 | 128.8 | 0.28 | 0.04 | 1.7 | 69.9 | 69,169 |
| 1.6 | −0.537495 | 154.2 | 0.37 | 0.10 | 2.1 | 71.7 | 67,600 |
| 1.8 | −0.542019 | 184.9 | 0.36 | 0.25 | 3.0 | 49.6 | 85,849 |
| 2.0 | −0.538819 | 222.6 | 0.05 | 0.58 | 3.7 | 53.6 | 127,449 |
| 2.4 | −0.522854 | 324.5 | −4.17 | 2.5 | 10.5 | 80.5 | 198,025 |
| 2.8 | −0.505108 | 467.9 | no convergence | 8.4 | 29.0 | 151.0 | 235,225 |
| 3.2 | −0.491098 | 649.9 | no convergence | 19.3 | 61.9 | 254.9 | 275,625 |
| 3.6 | −0.481904 | 854.5 | no convergence | 27.7 | 69.6 | 358.3 | 391,876 |

Equilibrium spacing from a parabola through the three lowest points:

| method | R / bohr | E/N / Eh |
|---|---|---|
| FCI | 1.817 | −0.542047 |
| SQD | 1.815 | −0.541793 |
| DMRG | 1.817 | −0.542024 |
| RHF | 1.747 | −0.526896 |

- **Near equilibrium,** SQD on 24 qubits is a variational bound 1.7–3.7 mEh above
  exact, using 8–15 % of the sector. It reproduces the equilibrium spacing to
  0.002 bohr.
- **Stretched chain:** CCSD fails here. At 2.4 bohr CCSD(T) falls 4.2 mEh *below* the
  exact energy, and beyond that CCSD does not converge. The CCSD-seeded circuit is then
  a poor sampler, and SQD errors grow to 30–70 mEh. The strongly correlated regime is
  exactly where better state preparation is needed: optimized LUCJ parameters, or seeds
  from a multireference or DMRG state rather than CCSD.
- **Samples alone** (no configuration recovery, no extension) are 50–360 mEh above
  exact. Most of the SQD accuracy here comes from the classical post-processing; the
  same lesson as the hardware runs in the QCL paper.
- **DMRG also degrades** when stretched (28 mEh at 3.6 bohr). It runs in the delocalized
  canonical RHF orbitals; a localized, chain-ordered orbital basis would make it nearly
  exact at this size. QCL has no orbital-localization step in front of `dmrg.solve`
  yet.
- **EWF with bath natural orbitals** (`bath="bno"`, η = 10⁻⁶) reproduces FCI to
  0.004 mEh. But every cluster here contains all 12 orbitals, so at this size it is
  twelve full FCI calculations, not a cheaper embedding. The DMET bath (`bath="dmet"`)
  is cheap and 97 mEh off at 1.8 bohr.

## 3. Diels–Alder cycloaddition: butadiene + ethylene → cyclohexene

```sh
QCL_POLICY_FILE=qcl/apps/applications-policy.toml CUDA_VISIBLE_DEVICES=2,3 \
    python qcl/apps/diels_alder.py --json diels_alder.json
```

One plan, 2325 s wall time. Geometries are HF/6-31G*:
- a Sella transition-state search, whose IRC connects the reactants to cyclohexene
  (1191 s, the longest step);
- optimizations of cyclohexene, s-trans butadiene and ethylene (62 s).

Single-point energies at those geometries:
- RHF, MP2 and frozen-core CCSD(T) (56 s);
- π-space CASCI and SQD, plus a 28-qubit CASCI(14,14) at the transition state and
  product, each solved three ways: GPU FCI (322 s), DMRG (M = 500, 508 s) and SQD
  (sampling 13 s, diagonalization 9 s).

The transition state is synchronous: both forming C–C bonds are 2.202 Å.

**Barrier and reaction energy.** Electronic energies, kcal/mol, relative to s-trans
butadiene plus ethylene.

| method | barrier | reaction energy |
|---|---|---|
| RHF | 44.85 | −42.85 |
| MP2 | 16.32 | −53.53 |
| CCSD(T), frozen core | **24.37** | **−47.81** |
| CASCI(π), exact FCI | 57.81 | −14.39 |
| CASCI(π), SQD | 57.81 | −14.39 |
| CASCI(π), SQD from samples only | 61.11 | −7.70 |
| experiment | ≈ 25.8 (ΔH‡ from Eₐ = 27.5 kcal/mol, 760–921 K) | ≈ −40 (ΔH, from gas-phase heats of formation) |

The π spaces are (6e,6o) at the transition state and in cyclohexene, (4e,4o) for
butadiene and (2e,2o) for ethylene, so the barrier is size-consistent.

- **The classical ladder behaves as expected.** RHF overestimates the barrier and MP2
  underestimates it. CCSD(T)/6-31G*//HF/6-31G* is within about 1.5 kcal/mol of the
  experimental activation enthalpy. That comparison is indicative only: these are
  electronic energies, with no zero-point or thermal corrections, in a small basis.
- **The π-only CASCI is the wrong model for this barrier.**
  - It puts the barrier 13 kcal/mol *above* RHF, and makes the reaction 28 kcal/mol
    less exothermic than RHF does.
  - It has static correlation only, and the canonical orbitals nearest the Fermi level
    have different character at different geometries. In cyclohexene the six "π"
    orbitals are no longer the two new σ bonds and the remaining π bond.
  - A quantitative barrier needs dynamic correlation on top of the active space (for
    example NEVPT2) and active orbitals chosen by character. QCL has neither primitive
    yet.
- **The quantum solver reproduces the model it is given.**

  | space | qubits | SQD − exact / mEh | determinants used |
  |---|---|---|---|
  | π spaces | 4–12 | ≤ 0.001 | full sector |
  | CASCI(14,14), transition state | 28 | 0.28 | 198,025 of 11.8 million (1.7 %) |
  | CASCI(14,14), product | 28 | 0.17 | 104,329 (0.9 %) |
  | samples alone, π spaces | 4–12 | up to 12 | |
  | samples alone, CASCI(14,14) | 28 | 18–19 | 256–625 |
  | DMRG (M = 500), CASCI(14,14) | 28 | 0.003 and 0.001 | |

  Samples alone shift the π barrier by 3.3 kcal/mol.

## Summary

| | T4L–benzene | H12 chain | Diels–Alder |
|---|---|---|---|
| classical scale | 2616-atom protein (MM), benzene 6-31G* (QM) | 12 atoms, full space | C6H10, 6-31G*, 110 basis functions |
| largest quantum space | 28 qubits (CASCI(14,14)) | 24 qubits (full space) | 28 qubits (CASCI(14,14)) |
| SQD vs exact (best setting) | 0.095 mEh; interaction within 0.02 kcal/mol | 1.7–3.7 mEh near equilibrium; 30–70 mEh stretched | 0.17–0.28 mEh |
| fraction of the sector SQD used | 2.3 % | 8–46 % | 0.9–1.7 % |
| samples alone | 6–15 mEh; interaction off by 5.5 kcal/mol at 12 qubits | 50–360 mEh | 18–19 mEh |
| main limitation found | QM/MM interaction of benzene is insensitive to correlation; MM/GBSA and docking are rough | the CCSD-seeded circuit is a poor sampler exactly where correlation is strong | static-correlation-only active spaces cannot give the barrier; CCSD(T) can |

What these runs show:
- QCL programs of realistic size run end to end on one server: protein preparation,
  MM, QM/MM, docking, transition-state search with an IRC check, coupled cluster, DMRG,
  GPU FCI, and sampled quantum circuits, each with its relation and provenance.
- At 24–28 qubits, sampled LUCJ circuits plus SQD reproduce exact active-space
  energies to sub-mEh accuracy, but only with configuration recovery and the
  single-excitation extension.
- The circuit's own samples are far too concentrated to span the subspace, as on the
  MQ02 hardware runs. Noiseless simulators were used throughout, so these runs say
  nothing about hardware noise; that is the subject of `docs/error-mitigation.md`.

Code changes made while running them:
- `qcl.qmmm.energy(..., extend=True)` adds the single-excitation extension to QM/MM
  SQD.
- QM/MM provenance records the SQD determinant count, shots and sampler.
- The QCL worker now writes its result on a private copy of stdout, so a library
  printing to file descriptor 1 can no longer corrupt the result. A 40-minute
  Diels–Alder plan was lost this way before the fix.

