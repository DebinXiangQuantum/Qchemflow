# QCL development package

QCL is a Python-embedded language implemented alongside ChemFlow. The current
executable slice supports `scf.rhf`, `scf.hf`, `scf.rohf`, CPU/GPU
`scf.uhf`, `dft.rks` and `dft.uks` energies, restricted orbital projections,
CPU/GPU RHF-based `mp2`, `active_space`, CPU/GPU
`fci.solve`, CPU `sci.solve`, HF/LUCJ `sample` on CPU/GPU-backed statevector simulators or
the existing lqcloud MQ02 adapter, and CPU `sqd.diagonalize` of those samples.
It also supports frontier orbital and charge-sector projections, finite Hubbard
cluster ED and correlations on CPU/GPU, static and bounded dynamic `map` with
uncertainty-aware `rank` and `compare`, sequential static `scan`, bounded
`iterate`, deferred `require` and `cond`, unit-checked `mean`, RDKit conformer
workflows, rigid bond/dimer energy
differences, rigid PES and PCM solvation scans, a connected reaction barrier
with optional RRHO/Eyring thermochemistry, empirical C-H site scoring,
MM/GBSA binding through ChemFlow's OpenMM side environment, ESOL/GSE solubility, and
Allen-Dynes temperature estimates.
Operations need explicit `.to(device)`
placement. Unsupported implementations fail during compilation.
`qcl.Budget(cpu_core_s=...)` limits process CPU time for a plan, including
CPU work in a GPU plan. `Plan.cpu_core_s` reports measured worker CPU time
after execution or failure. Signed Agent runs share their CPU allowance with
any independent reference calculation and record consumption in the ledger.
`workflow.compile(..., deterministic=True)` fixes CPU worker thread counts and
random seeds and gives the plan a distinct hash. Seeded local CPU QPU simulators
can replay sampled bitstrings. GPU numerical kernels and GPU-backed simulators
currently reject this flag because bitwise reproducibility is unverified.

The Rust IR, checker, cluster parser and Python extension are built with
`maturin`. From the repository root, with Python 3.11 and Rust installed:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[quantum,reactivity,paths]' pytest ruff maturin
PYO3_PYTHON="$PWD/.venv/bin/python" .venv/bin/maturin develop --manifest-path qcl/Cargo.toml
OMP_NUM_THREADS=1 .venv/bin/python -m pytest qcl/tests tests/test_qcl_agent.py -q
```

Run `apps/h2_ground.py` for CPU FCI. The `apps/frontier_orbitals.py`,
`apps/hubbard_pairing.py`, `apps/stability_ranking.py`, and
`apps/aqueous_solubility.py` examples exercise other CPU workflows. The
`apps/conformer_ranking.py`, `apps/bond_dissociation.py`,
`apps/dimer_binding_scan.py`, `apps/pes_scan.py`,
`apps/relaxed_dihedral_scan.py`, `apps/solvation_scan.py`,
`apps/metabolic_sites.py`, `apps/protein_ligand_binding.py`, and
`apps/reaction_barrier.py` exercise collections, geometry edits and reused
integrals. `apps/open_shell.py` demonstrates UKS spin-resolved orbital
projections; `apps/h2_mp2.py` and `apps/h2_sci.py` demonstrate correlated
methods. `apps/h2_gfn2.py`, `apps/h2_gfn2_optimize.py`, and
`apps/h2_gfn2_solvation.py` exercise the CPU tblite engine.
PCM/RHF runs on CPU or GPU; GPU PCM does not accept a precomputed ERI override.
`apps/h2_pcm.py` chooses the first available GPU or a CPU and computes
`E(PCM) - E(gas)`; the CPU path reuses integrals.
PCM/RHF PES callbacks also support analytic and finite-difference gradients
and analytic Hessians on CPU or GPU. Frequency analysis retains the solvent
identity when it consumes a PCM Hessian on a CPU lane.
`qcl.SMD(name)` uses PySCF or GPU4PySCF SMD for RHF energy and the same
derivatives. CPU SMD requires a PySCF build with its SMD module enabled;
the current macOS development environment lacks that module, while A100 and
F5090 provide it. Signed HF/SMD solvation tasks use the same Agent checks as PCM.
RKS/UKS also accept `solvent=qcl.PCM(name)` or `qcl.SMD(name)` on CPU and
GPU. Closed-shell RKS PES callbacks support solvated analytic gradients and
Hessians. `apps/h2_dft_solvation.py` computes signed PBE/PCM solvation energy;
the QCL Agent checks its XC functional, solvent and subtraction order for
closed-shell RKS and open-shell UKS tasks.
Local sampling uses
the included resource policy:

```sh
QCL_POLICY_FILE=qcl/apps/local-policy.toml .venv/bin/python qcl/apps/h2_sampling.py
QCL_POLICY_FILE=qcl/apps/local-policy.toml .venv/bin/python qcl/apps/h2_sqd.py
QCL_POLICY_FILE=qcl/apps/local-policy.toml .venv/bin/python qcl/apps/h2_lucj_sqd.py
```

GPU FCI, HF and KS DFT require ChemFlow's `gpu` extra and a CUDA device.
GPU HF/DFT use GPU4PySCF; GPU SCF results can be copied to a CPU lane.
GPU HF/PCM and DFT/PCM-SMD are registered; GPU SCF with a precomputed ERI
override is not.
On a host with two visible GPUs, one plan may execute SCF operations on both
devices and explicitly copy Tensor or SCF values between them. Execution is
sequential within the worker.
`scf.rhf(molecule, guess=previous_rhf)` seeds a new geometry from a placed RHF
density with the same basis and electron sector. `apps/h2_warm_start.py` shows
three sequential bond lengths on CPU or GPU. Cross-device guesses require an
explicit SCF result copy under `strict_transfers=True`.
`apps/h2_pbe.py` runs PBE/RKS on CPU or GPU. UHF/UKS orbital projections use
`.orbital_energy("homo" | "lumo", spin="alpha" | "beta")`; GPU projections
rebuild canonical Fock eigenvalues from the converged density on the CPU.
DFT orbitals need an explicit correlated-space policy before they can feed
`active_space`.
`qcl.mp2(rhf_result).to(device).energy` uses PySCF on CPU or GPU4PySCF on
GPU and rejects non-RHF orbitals at compile time.
`qcl.sci.solve(H, sector, select_cutoff=...).to(cpu).energy` reuses ChemFlow's
spin-complete Selected CI adapter; signed Agent tasks can compare it against an
FCI error gate.
`qcl.fci.solve(H, sector, engine="native").to(cpu)` selects a Rust CPU CI
solver for sectors up to 8192 determinants. The default remains the PySCF
adapter, and oversized native sectors fail during compilation. H2, Li, H4,
H6 and H8 closed/open-shell and shifted-sector tests match PySCF within
`1e-10` hartree; `apps/h2_native_ci.py` runs the native path. The solver uses
dense Jacobi diagonalization through 256 determinants and a sorted sparse
matrix with reorthogonalized Lanczos above that. The H8 native solve was slower
than PySCF on the local test host (0.83 s versus 0.044 s for the FCI node), so
the native engine remains opt-in. `qcl.rdm1(ci_result).to(cpu)` and
`qcl.rdm2(ci_result).to(cpu)` return spin-summed active-MO dimensionless
Tensors for PySCF or native CPU FCI and PySCF Selected CI. The native densities
use the Rust determinant ordering and match PySCF for H2/H4, with 1-RDM also
checked for H6/H8 and shifted electron sectors. QCL requests a PySCF CI vector
residual below `1e-9` before computing either density; a stricter retry handles
cases where energy alone converges first. GPU FCI results can feed a CPU density
operation after an explicit `.to(cpu)` copy, or feed a CuPy density operation
placed on the same GPU. GPU density evaluation also requires a measured CI
residual at most `1e-9`, since energy convergence alone can leave H4 2-RDM
entries inaccurate. The GPU FCI Davidson driver currently keeps its CI vector
on the host between CuPy sigma calls in the legacy ChemFlow backend. QCL's GPU
FCI path now uses CuPy's GPU Lanczos eigensolver, with GPU dense diagonalization
for small sectors. Its CI vector stays on the selected GPU for coefficient,
σ and RDM nodes in the same worker; a measured residual is checked before
returning. Returned Tensors still cross the one-shot worker's host transport
boundary. Custom CUDA C++ CI kernels and cross-run resident buffers remain open.
`apps/h2_density.py` returns FCI energy and both density matrices in one plan.
`state.coefficients` projects a placed FCI result into a dimensionless CI
Tensor in PySCF determinant order. `qcl.sigma(H, state.coefficients).to(cpu_or_gpu)`
applies the full active-space Hamiltonian, including its scalar core energy,
and returns a hartree Tensor. CPU execution uses ChemFlow's NumPy kernel by
default; `engine="native"` selects the Rust CI sparse matrix product for
sectors within its documented determinant limit. GPU execution uses ChemFlow's
CuPy kernel. PySCF-order coefficients
are converted at the Rust boundary and checked against PySCF random-vector
products and eigenvector residuals in `apps/h2_sigma.py`. Coefficients
copied from a GPU FCI result to CPU remain on the CPU lane. To apply the
Hamiltonian to an arbitrary vector, use
`qcl.ci_vector(array, active.sector).to(device)`; the array must have PySCF
alpha/beta determinant-string shape and finite real entries. The constructor
is a host operation, and its `.to()` is an explicit CPU/GPU copy. A single
device executes each σ operation; determinant-block device lists remain open.
The native Rust implementation rebuilds its sparse matrix for each σ call; caching and
native CUDA CI kernels remain future runtime work.
`qcl.engines.tblite.gfn2.energy(mol).to(cpu)` uses the semiempirical GFN2-xTB
Hamiltonian, reported as a distinct method family from HF/DFT.
`qcl.engines.tblite.gfn2.optimize(mol).to(cpu)` returns a geometry that can
feed subsequent QCL operations. It uses ChemFlow's isolated tblite/ASE worker
and accepts the same optional ALPB/GBSA solvent descriptor as `energy`.
`qcl.engines.tblite.gfn2.energy(mol, solvent=qcl.ALPB("water")).to(cpu)`
and `qcl.GBSA(name)` select named implicit solvents. A signed solvation task
checks the solvent model, name, molecule and `E(solvent)-E(gas)` ordering.
`apps/h2_integrals.py` demonstrates CPU `int1e` overlap/kinetic/nuclear AO
`Tensor` values, `qcl.eigh(A, B=None)` and `qcl.contract(spec, *tensors)`.
`qcl.int3c(mol, auxbasis).to(cpu)` returns full `(nao, nao, naux)`
density-fitting integrals through PySCF, including the auxiliary basis in
provenance. `apps/h2_density_fitting.py` shows its contraction. The integral
primitive has a CPU implementation; its Tensor can be explicitly copied to a
GPU for CuPy contractions, including either card on a two-GPU node.
`qcl.cholesky(mol, tol).to(cpu)` returns pivoted Cholesky factors with shape
`(rank, nao, nao)` and records the measured maximum ERI reconstruction error.
Contracting `Pij,Pkl->ijkl` reconstructs the four-center ERI. This CPU
reference implementation forms a dense ERI before factorization, so it is
suited to small basis sets; a direct, screened integral implementation and
device-list splitting remain open. Factor tensors can be copied explicitly
to a GPU for CuPy contractions.
`qcl.transform(integrals, coefficients)` transforms two-index AO integrals,
dipole and quadrupole matrices, and four-center ERIs into a molecular-orbital
basis on CPU or GPU. The coefficients and integrals must be placed explicitly;
GPU ERIs currently come from an explicit CPU-to-GPU copy. The implementation
uses opt_einsum with NumPy or CuPy and is checked against PySCF AO-to-MO results.
`apps/h2_integrals.py` runs both the one-electron and ERI transformations.
GPU tensor contractions accept explicit CPU-to-GPU Tensor copies and return a
CuPy-backed Tensor with CUDA DLPack support. The current one-shot worker uses a
GPU-host-GPU round trip for returned Tensor values; `Plan.explain()` records
both transport legs. Standard and generalized `eigh`, `orthogonalize(S)` and
`solve(A, b)` have CPU and CuPy implementations. Persistent resident buffers
remain a separate runtime task. `svd(A)` returns typed `u`, `s` and `vh`
Tensors, with an economy decomposition on either device kind.
`Cluster.local()` discovers every CUDA device visible to the process as
`gpu0`, `gpu1`, and so on, and reports its physical memory in decimal GB when
CUDA exposes that property. Explicit placement on the second visible GPU is
tested with CuPy tensor eigensolvers and GPU4PySCF gradients.
`qcl.gradient(qcl.PES(callback, gradient=qcl.Analytic()), molecule).to(device)`
returns a Cartesian `hartree/bohr` Tensor for closed-shell HF or RKS with a
named XC functional. A `qcl.FiniteDifference(step=...)` policy uses central
differences with a Bohr step on CPU or GPU. `apps/h2_gradient.py` demonstrates
HF and PBE/RKS; open-shell and other methods are rejected at compilation.
`qcl.hessian(pes, molecule).to(device)` returns a closed-shell HF or RKS Hessian
Tensor in `hartree/bohr^2` on CPU or GPU. Frequency analysis runs on CPU; a GPU Hessian
must be copied with `.to(cpu)` before `qcl.frequencies(hessian).to(cpu)` when
`strict_transfers=True`. `apps/h2_hessian.py` demonstrates both placements.
GPU4PySCF currently returns its Hessian array to host memory; QCL uploads it
to the requested GPU, and a returned GPU Tensor also crosses the one-shot
worker's host serialization boundary. CPU and GPU PBE Hessians agree within
`1e-7 hartree/bohr^2` on the H2/STO-3G reference.
When a plan uses one GPU and several CUDA devices are visible, its worker sees
only the selected card. The result and profile retain the original `gpuN`
identity. Multi-GPU operations in one plan run sequentially in the current
one-shot worker; concurrent per-device scheduling remains open.
`qcl.periodic.scf(system, kmesh=...).to(cpu)` reuses the verified PySCF PBC
adapter for pseudopotential KRHF cells. `qcl.dynamics.spectral_exact(model,
sector, orbital=..., spin=..., omega=..., eta=...).to(cpu)` computes a complete
Lehmann spectral density. The signed Agent compares its L1 error with the
independent occupation-basis reference. Both currently have CPU implementations.
Returned CPU tensors expose DLPack
through `numpy.from_dlpack`. This reference path serializes arrays across the
current one-shot worker boundary; persistent GPU-resident tensors remain open.
`Plan.profile()` includes wall duration and measured process CPU time per
executed node. A plan-level or signed `cpu_core_s` budget limits the worker;
the Agent ledger records worker CPU time across repeated runs.
`Plan.error_budget()` reports the output Estimate's measured statistical
variance after a run and deviations observed by `Plan.validate()` against
explicit reference plans. Systematic and total upper bounds remain `None`:
one reduced-instance comparison does not establish an error bound for other
systems or geometries. Calling `error_budget()` does not run a plan.
The signed QCL Agent adds its evaluated absolute-error gate as an observed
reference deviation in `qcl-error-budget.json`, including the reference plan
hash and task record hash; it does not promote that observation to a bound.
CPU `int1e` also provides origin-based dipole and quadrupole position matrices
with explicit Bohr and Bohr-squared units.
MQ02
execution requires the `qpu` extra, `LQCLOUD_API_KEY`, an explicitly metered
QPU device, and an operator QCL policy. The policy requires a shot limit for
every QPU; money and approval settings apply only when configured for a
metered cloud device. The built-in MQ02 estimate uses a historical ChemFlow
price record, reported as `tariff_source` in `Plan.estimate()`.
`workflow.compile(..., budget=qcl.Budget(qpu_shots=..., qpu_s=...,
wall_s=...))` can lower operator resource limits. `qpu_cost` is available
only for metered cloud QPUs. A program cannot raise an operator limit;
`Plan.estimate()` reports the program budget and `Plan.run()` checks it
before QPU submission.
The CCSD-to-LUCJ initializer currently runs on CPU; the LUCJ circuit is
accepted by both the local statevector simulator and the MQ02 adapter.

ChemFlow Agent can run an agent-authored QCL program against a signed task:

```sh
chemflow agent run --task task.json --public-key public.pem \
  --program qcl/apps/agent_h2.py --out qcl-run
```

For supported signed tasks, the Agent can construct a QCL workflow directly:

```sh
chemflow agent run --task task.json --public-key public.pem \
  --engine qcl --out qcl-run
```

For a fixed-active-space HF ground task with a signed numerical gate,
`--engine qcl --screen` runs HF, PySCF FCI and (within its determinant limit)
native Rust CI candidates under one cumulative resource allowance. It writes
`qcl-screen.json` with each gate result and measured cost, selects the least
costly gate-passing candidate, and records the selected result and ledger.
`QclAgent.select()` exposes the same reviewed-plan screen to Python clients.
Repeated `--candidate-program path.py` arguments screen authored workflows for
any supported signed numeric target; each file supplies
`build(task_spec, cluster)`. The screen reserves aggregate QPU shots and device
time before execution, and a configured signed cloud cost limit accumulates
forecast charges across reruns. Mixed billing currencies are rejected.
The screen reuses an identical exact reference within one signed Agent session;
candidate ranking uses measured candidate compute time, while the ledger records
all wall and CPU time. Reviews bind both the graph hash and the chosen result
output, including the independent reference output.

`--device gpu0` selects registered GPU implementations for signed ground
energy, Hubbard or HF/DFT PCM-SMD solvation tasks. `--device sim0` constructs CPU active-space preparation,
QPU sampling and CPU SQD for a fixed-space HF ground-energy task with signed
shot budget and simulator effect. A metered cloud QPU also needs the operator
policy and a signed network effect; no cloud job is submitted by these examples.

The built-in workflow covers molecular ground energy, frontier orbitals,
ionization energy/electron affinity, Hubbard energy and pair correlations,
ESOL/GSE solubility, electronic reaction energy, ranking, rigid bond and
dimer scans, rigid PES, HF/DFT PCM-SMD and GFN2/ALPB-GBSA solvation, metabolic site
scoring, MM/GBSA binding, connected electronic or RRHO reaction barriers,
periodic KRHF and exact impurity spectral densities.
MM/GBSA and reaction barrier built-in plans have signed compile checks; their
real-input execution remains dependent on the corresponding engine setup.
The legacy Agent remains the default for its broader candidate-selection
contract; QCL execution is selected explicitly with `--engine qcl`.

`build(task_spec, cluster)` in the program returns `(workflow, args, kwargs)`.
The agent writes `qcl-review.json`, `journal.jsonl` and `summary.json`, then
`qcl-result.json`, `qcl-profile.json`, `report.md` and a cumulative resource
ledger after execution. It checks the signed question and budget
again at run time. `--dry-run` writes the review and summary without executing.
The existing `results.get/query` interface reads QCL runs from the same
`results.sqlite`; QCL rows carry `kind: "qcl"` and can be filtered by signed
record hash, plan hash or target kind.
`chemflow agent qcl-run` remains a compatibility alias. For signed molecular
ground-energy tasks with a fixed active space, `abs_error_vs_exact` gates run a
CPU FCI reference before the candidate and record the measured error and
pass/fail decision. Spectral tasks use an independent occupation-basis L1 gate;
other unsupported gate kinds are rejected on the QCL path.
This path currently supports signed molecular ground-energy, frontier orbital,
IP/EA, lattice energy, pairing correlation, ranking, bond dissociation,
counterpoise interaction scans, rigid PES, HF/PCM solvation, empirical metabolic
site scores, MM/GBSA binding, connected electronic or RRHO reaction barriers,
periodic KRHF, exact impurity spectra, and
electronic reaction energies with signed stoichiometry, counterpoise interaction
energies with signed fragments, and aqueous-solubility tasks. Reaction and
interaction energies support FCI error gates. The existing ChemFlow Agent path
remains available for its existing report and candidate-selection workflows.
Signed molecular ground-energy tasks can require a specific RKS or UKS
functional. Open-shell tasks can require UHF. The Agent checks the functional
and calculation lineage before execution; unsigned DFT substitutions are
rejected.
QPU sampling in a signed Agent task also requires `simulator` or `network` in
the task's allowed effects and a signed shot budget.
`qcl.vqe.sampled` reuses ChemFlow's finite-shot LUCJ/UCCSD VQE on a CPU-backed
QPU simulator or metered MQ02. The caller declares the Pauli measurement-group
count for a conservative shot and cloud-cost forecast; execution checks the
count against the Hamiltonian. `qcl.qubit_hamiltonian(...).to(cpu)` produces a
typed Jordan-Wigner `QubitOperator`; FCI rejects that type. For the local H2
example:

```sh
QCL_POLICY_FILE=qcl/apps/vqe-policy.toml python qcl/apps/h2_vqe.py
```

`@qcl.quantum_kernel` traces gate-level `Qubits[n]` and `Params[m]` functions.
The HEA example in `apps/hea_sampling.py` samples on a local CPU- or
GPU-backed QPU simulator with `QCL_POLICY_FILE=qcl/apps/local-policy.toml`.
The same circuit lowers through ChemFlow's MQ02 adapter when placed on an
explicitly metered cloud QPU. `qcl.measure` and kernel-local `qcl.cond` execute
through Qiskit Aer on CPU-backed simulators; unsupported targets report
`QCL-E206` during compilation.
`qcl.estimate(qubit_op, kernel, n_groups=, shots_per_group=)` measures QWC
Pauli groups on a CPU-backed QPU simulator or MQ02. It checks the declared
group count against the Hamiltonian before submitting any cloud jobs. The
`apps/h2_pauli_estimate.py` example uses the local policy in
`apps/vqe-policy.toml`. Custom unitary kernels can also estimate on a
GPU-backed QPU simulator; HF/LUCJ Pauli estimation currently uses the CPU
simulator or MQ02.
Passing multiple QPUs to `.to([qpu0, qpu1])` assigns disjoint Pauli groups to
each device and combines their estimates and variances. Shot and cloud-charge
forecasts count each group once; signed Agent reviews accept this plan shape.
`compile(..., deterministic=True)` rejects cloud QPU measurements that cannot
be replayed from a seed. `CircuitCut` and `Teleport` are typed partition
descriptors: cutting cannot return raw samples (`QCL-E104`), and Teleport
checks cluster quantum links (`QCL-E208`). Partitioned execution is not yet
registered and reports `QCL-E203` when a link is present.

The full design contract is `docs/spec/qcl-design.md` at the repository root.
The remaining phases include the rest of the primitive and application
catalogue, dynamic loops and drivers, multi-device scheduling, native kernels,
and broader agent support. Bounded dynamic `map` traces a branch for each
possible item, then skips absent items at execution; it does not yet schedule
branches concurrently across lanes. `scan` currently requires a static finite
sequence and executes points in order. `cond` executes only its selected
branch; plan resource estimates conservatively include both branches.
`iterate` requires a positive `max_steps`, stops on a computed predicate and
fails when the bound is reached without convergence; its plan estimate covers
the full bound.
Estimate multiplication, division and negation by finite scalars preserve units
and covariance when the same sampled result appears more than once in an
expression. `apps/reaction_energy.py` demonstrates a signed-compatible
stoichiometric combination.
PES-guided constrained optimization accepts closed-shell analytic or
finite-difference HF gradients on one CPU lane. Finite differences use
central Cartesian displacements in angstroms and record their step size.
Open-shell reference matching and multi-device GPU gradients remain open.
`qcl.irc(pes, transition).to(cpu)` propagates an analytic HF path through the
existing Sella adapter. `path.connects(reactant, product)` is a deferred host
predicate suitable for `qcl.require`; the path can also be returned as a
`ReactionPath`. IRC currently supports one CPU lane and closed-shell HF.
