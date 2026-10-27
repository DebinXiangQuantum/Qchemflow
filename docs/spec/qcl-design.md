# QCL Design Specification

**A Python-embedded language for quantum chemistry on heterogeneous CPU, GPU and QPU clusters**

| | |
|---|---|
| Status | Design v1.0, ready for implementation |
| Date | 2026-10-22 |
| Name | `qcl` is a placeholder name |
| Languages | Python (frontend), Rust (compiler and runtime), CUDA C++ (GPU kernels), C ABI (interop) |

---

## Contents

0. Instructions for implementing agents
1. Goals and non-goals
2. Architecture overview
3. Programming model (normative)
4. Types and result semantics (normative)
5. Primitive catalogue
6. Devices and clusters
7. Compilation, plans and diagnostics
8. Agent interface
9. Application catalogue and worked examples
10. Implementation architecture (Rust)
11. Primitive implementation plan
12. Repository layout
13. Build phases and acceptance criteria
14. Testing and validation strategy
15. Engineering risks
16. Decision log
17. Open questions
18. References
- Appendix A. Design rationale

---

## 0. Instructions for implementing agents

This document is the source of truth for building QCL. Read it fully before writing code.

- **Normative vs. rationale.** Sections 3, 4, 6 and 7 are normative. Rules use MUST, MUST NOT and SHOULD. Text marked *Rationale* and Appendix A explain why; they are not requirements.
- **Don't relitigate decisions.** Section 16 lists decisions already made and why. If an implementation constraint forces a change, update this document and the decision log in the same change.
- **Build in phase order** (§13). A phase is done only when its acceptance criteria pass in CI.
- **Correctness before speed.** Every primitive MUST have a CPU reference implementation before any GPU or QPU implementation is merged.
- **No silent fallbacks.** If something can't run where the program placed it, that is a compile error with a helpful diagnostic, never a quiet switch to another device or method.
- **Wrap before writing.** Use the third-party libraries listed in §11. Write new numerical kernels only where §11 says so, or where profiling shows a wrapped library is the bottleneck.
- **Test every user-visible behavior,** including every placement rule (§3.3) and every diagnostic code (§7.3).

---

## 1. Goals and non-goals

### Goals

1. One programming model for quantum chemistry workflows across CPUs, GPUs and QPUs, including multi-GPU nodes and distributed QPU clusters.
2. Python-embedded: humans and AI agents write ordinary Python.
3. Full control at every level, for humans and agents alike: algorithm, kernels (including gate-level quantum circuits) and deployment.
4. Explicit, checkable deployment: every computation states where it runs, and infeasible programs fail at compile time, before any QPU time or money is spent.
5. Honest results: every result carries uncertainty, units, provenance and its validity domain.
6. Reuse existing scientific software rather than rewriting it.
7. Cover the 13-application catalogue (§9), which also serves as the permanent regression suite.

### Non-goals (v1)

- A new syntax or standalone parser.
- Reimplementing mature engines (OpenMM, RDKit, tblite, Sella) or PySCF-level SCF.
- Automatic placement as the default. Placement is always written in the program (decision D3).
- Claims of experimental accuracy. A gate against a model-defined reference measures reproducibility of that model, not agreement with experiment.

---

## 2. Architecture overview

```
Python frontend     user and agent programs, tracing, eager mode       python/qcl
       │
Chemistry IR        workflows, patterns, typed values                  qcl-ir
       │
Primitive IR        primitives, each with several implementations      qcl-ir, qcl-check
       │
Plan                placement checks, transfers, cost, budgets         qcl-sched
       │
Execution           async executor, buffers, streams, QPU jobs         qcl-runtime
       │
Backends            CPU | CUDA GPU | QPU and simulators | engines      qcl-cpu, qcl-cuda, qcl-qpu, workers/
       │
Cluster topology    devices, classical links, quantum links            qcl-cluster
```

**Lifecycle of a program**

1. The user calls `workflow.compile(*args)`. The Python frontend traces the function into an IR graph held in Rust.
2. The checker validates types, units, sectors, model compatibility and placement rules (§3, §4).
3. The planner checks feasibility against the cluster (memory, qubits, connectivity, links), inserts transfers, estimates cost and checks budgets (§7).
4. `plan.run()` executes each primitive through an implementation on the device the program named.
5. Every result carries provenance: what was computed, how, where, with which seeds and library versions.

---

## 3. Programming model (normative)

### 3.1 Workflows

- A workflow is a Python function decorated with `@qcl.workflow`. Its body is traced, not executed, when compiled.
- `plan = wf.compile(*args, **kwargs)` returns a `Plan` (§7.2); `plan.run()` executes it.
- Calling a workflow inside another workflow is an ordinary function call; it is inlined into the caller's graph. Workflow calls take no `.to()`; the operations inside them are placed individually.
- Workflows receive devices as ordinary arguments (rule P5).
- **Eager mode.** Inside `with qcl.eager():`, each compute operation executes as soon as `.to()` is applied, and values are concrete. Semantics are otherwise identical. Eager mode is for notebooks and debugging.
- **Control flow.** In traced mode, Python control flow runs at trace time and MAY depend only on static metadata (for example `mol.spin` or `len(bonds)`). Branching or looping on computed values MUST use patterns (`qcl.cond`, `qcl.iterate`, `qcl.scan`, `qcl.optimize`). In eager mode, Python control flow works on concrete values.

### 3.2 Three kinds of calls

| Kind | Examples | Placement |
|---|---|---|
| Compute operation | `qcl.scf.rhf`, `qcl.int2e`, `qcl.fci.solve`, `qcl.sample`, `qcl.engines.openmm.sample`, `qcl.thermo.rrho` | MUST be placed with `.to()` |
| Pattern | Parallel: `qcl.map`. Drivers: `qcl.scan`, `qcl.optimize`, `qcl.iterate`, `qcl.irc`, `qcl.propagate`, `qcl.engines.sella.saddle` | `.to()` supplies the lanes for callbacks (P7, P8) |
| Host operation | Molecule edits, attribute access, arithmetic on results, small reductions, construction of operators, circuits and descriptors, checks | Never placed; runs on the host process |

Host operations are part of the traced graph and recorded in provenance. On concrete static inputs they execute during tracing. When they read computed values, they become symbolic host nodes and execute on the host after their inputs are available. The complete list:

- **Molecule and geometry edits:** `cleave`, `set_bond`, `set_angle`, `set_dihedral`, `set_separation`, `ghost`, `with_charge`, `bonds(...)`.
- **Attribute projections:** `.energy`, `.hamiltonian`, `.sector`, `.homo`, `.lumo`, `.n_imaginary`, `.connects(...)`, and similar fields of composite results.
- **Arithmetic and small reductions** on `Estimate`s and scalars: `+ - * /`, `qcl.mean`, `qcl.rank`, `qcl.compare`.
- **Construction:** `qcl.kernels.*(...)`, `qcl.models.*(...)`, `qcl.lattice.*(...)`, `qcl.ops.*(...)`, `qcl.PES(...)`, `qcl.Active(...)`, `qcl.Sector(...)`, solvent, force-field and constraint descriptors.
- **Checks and branching:** `qcl.require`, `qcl.cond`.

### 3.3 Placement with `.to()`: the only mechanism

- **P1 (meaning).** `v.to(x)` declares that value `v` lives on `x`. If the operation producing `v` is not yet placed, it runs on `x`. If it is already placed elsewhere, `.to(x)` returns a new value that is a copy on `x`, and the original stays where it was.
- **P2 (mandatory).** Every compute operation and every pattern MUST be placed exactly once. Further `.to()` calls on its value make copies. An unplaced value is error `QCL-E201`.
- **P3 (targets).** `x` is a device or a list of devices. With a list, the operation splits along its natural parallel axis (§3.4). Composite operations split their parallel stages across the list and run their serial stages on the first device. Where a list is expected, a single device means a one-element list.
- **P4 (device options).** Options such as precision, transpile level and error mitigation belong to device handles: `gpus[0].with_options(precision="fp64")`. Operations never take device options. An option that is invalid for the device kind is error `QCL-E202`.
- **P5 (devices are arguments).** Devices come from the cluster (§6) and SHOULD be passed into workflows as arguments, so the same workflow runs on a laptop or a supercomputer by changing only the call.
- **P6 (host values).** Host operations produce values that live on the host. `.to(x)` on such a value copies it to an addressable CPU or GPU, for example to move a constructed operator onto a GPU. A QPU has no addressable value memory: `.to(qpu)` places a QPU-capable compute operation, and arbitrary host or tensor copies to a QPU are rejected.
- **P7 (parallel patterns).** `qcl.map(f, items).to(lanes)` distributes calls over the elements of `lanes`. Each call receives its element, called a *lane*, as its last argument: `f(item, lane)`. A lane is a device or a tuple of devices, such as `(gpu, qpu)`. Computed `items` remain a symbolic collection during tracing. The callback is traced for each lane, and items are assigned dynamically as lanes become free. Callbacks MAY use the lane for placement and device options but MUST NOT change the scientific question or result based on lane identity; results MUST NOT depend on the assignment.
- **P8 (driver patterns).** Drivers run their control logic on the host. `.to(x)` supplies the lane passed to every callback invocation. `x` may itself be a list, which enables parallel work inside each call.
- **P9 (implementation availability).** Placing an operation on a device requires a registered implementation for that device kind. Otherwise it is compile error `QCL-E203`, which lists the device kinds that do have implementations. There is no silent fallback.
- **P10 (transfers).** When an operation reads a value that lives elsewhere, the planner inserts a transfer, and `plan.explain()` lists it. With `compile(..., strict_transfers=True)`, implicit device-to-device transfers become error `QCL-E205`, so every movement must be an explicit `.to()` copy. Host operations reading small results are always allowed.
- **P11 (`.to` is reserved).** `.to()` means placement only. Unit conversion uses `.in_units("kcal/mol")`. No other API may be named `to`.

### 3.4 Natural parallel axes

| Operation | Axis split across a device list |
|---|---|
| `int2e`, `int3c` (density fitting), `cholesky` | Shell pairs, or the auxiliary index |
| Fock and J/K builds | Shell-pair blocks |
| `contract`, `transform` | Chosen by the contraction planner |
| `fci.solve`, `sci.solve`, `ed.ground_state`, `sqd.diagonalize` | Determinant-string blocks of the σ-vector |
| `map` | Items |
| `hessian`, finite-difference gradients | Displacements |
| `sample` | Shots, proportional to device throughput unless `shots=[...]` gives the split |
| `estimate` | Measurement groups |
| `partition=qcl.CircuitCut(...)` or `partition=qcl.Teleport()` | Circuit fragments or modules |
| Operations without an axis (for example `eigh`, `openmm.sample`) | A list of more than one device is error `QCL-E204` |

Each primitive's documentation MUST state its axis.

### 3.5 Potential energy surfaces and derivatives

`qcl.PES(fn, gradient=...)` is a host construction. It wraps an energy function `fn(geometry, lane)` that returns a result with `.energy`. Drivers and derivative operations evaluate it as follows:

- `gradient=qcl.Analytic()`: `fn` is called with the whole lane, and the analytic gradient is computed where the result lives.
- `gradient=qcl.FiniteDifference(step=...)`: the 2 × 3N displaced energies (central differences) are distributed over the lane's elements, and `fn` is called with one device each.
- `gradient=qcl.ParameterShift()`: for objectives over circuit parameters that are evaluated with `estimate`.

### 3.6 Quantum kernels

```python
@qcl.quantum_kernel
def hea(q: qcl.Qubits[8], theta: qcl.Params[16]):
    for i in range(8):
        qcl.ry(q[i], theta[i])
    for i in range(7):
        qcl.cz(q[i], q[i + 1])
    for i in range(8):
        qcl.ry(q[i], theta[8 + i])
```

- Kernels are device-independent. Each kernel is lowered separately for every target QPU (native gates, connectivity, transpile options) and checked at compile time for qubit count and parameter shapes.
- Kernels support mid-circuit measurement (`qcl.measure`) and classical feedback (`qcl.cond` inside a kernel), subject to the target's capabilities. Unsupported features are error `QCL-E206`.
- Library kernels such as `qcl.kernels.lucj(act)` (built with ffsim) are host constructions.

---

## 4. Types and result semantics (normative)

### 4.1 Core types

| Type | Meaning |
|---|---|
| `Molecule` | Atoms, coordinates, charge, spin, basis. May contain ghost atoms (basis functions without nucleus or electrons) and external point charges |
| `Active` | Active-space request: `Active(electrons=, orbitals=)`, or `Active(max_orbitals=)` with electrons inferred per species |
| `Sector` | Particle number, S_z, optional point-group irrep. `sector.add_electrons(n)` gives neighboring charge sectors |
| `Tensor` | Device-resident array with layout and optional symmetry blocks; exchangeable via DLPack |
| `FermionOperator` | Second-quantized operator. The common currency between molecular Hamiltonians, lattice models and qubit mappings |
| `QubitOperator` | Pauli-sum operator. Produced only by an explicit mapping (`jordan_wigner`, `bravyi_kitaev`, `parity`) |
| `Kernel` | Parametrized, gate-level quantum circuit |
| `Bitstrings` | Raw measured samples, tagged by the device that produced them |
| `Estimate` | Value, statistical variance, optional systematic error components, units, provenance |
| `Geometry`, `Trajectory`, `Path` | Coordinates, MD frames, IRC paths |
| `Frequencies` | Vibrational frequencies and normal modes; `.n_imaginary` |
| `SCFResult`, `ActiveSpace`, `GroundState` | Composite results with projections such as `.energy`, `.hamiltonian`, `.homo` |

`FermionOperator` and `QubitOperator` are distinct types. An operation expecting one MUST reject the other with `QCL-E101`. For example, `sqd.diagonalize` takes the fermionic Hamiltonian.

### 4.2 Estimates and uncertainty

- Every numerical result is an `Estimate`. Deterministic results have zero statistical variance.
- Arithmetic propagates variance. Operands that share samples carry covariance through shared provenance IDs; otherwise they are treated as independent.
- `qcl.rank` and `qcl.compare` report an ordering only for pairs whose difference exceeds the combined uncertainty. Other pairs are reported as unresolved.

### 4.3 Units

- Units are part of the type. Defaults: hartree for electronic energies, kcal/mol for MM/GBSA (inherited from the application catalogue).
- Adding or subtracting mismatched units is error `QCL-E102`. Conversion is explicit: `x.in_units("kcal/mol")`.

### 4.4 Provenance

Every result records:

- **The question:** molecule, property, sector, geometry mode (rigid or relaxed).
- **The method:** method family, basis, active space, solvation model, thresholds.
- **The implementations:** implementation ID, library and version, deterministic flag.
- **The deployment:** devices, transfers and their links.
- **Reproducibility data:** all seeds, timestamps.

Provenance is serializable and attached to every value returned by `plan.run()`.

### 4.5 Validity domains and model compatibility

- Each primitive declares validity flags that are carried into its results, for example "PCM: electrostatic only", "MM/GBSA: no configurational entropy", "orbital gap is not the fundamental gap", or "finite cluster".
- Subtracting or comparing two quantities of the same kind (for example two electronic energies) checks model compatibility: method family, basis, geometry mode and active-space policy. A mismatch is error `QCL-E103`. Different active-space *sizes* under the same policy (for example fragments under `max_orbitals`) are allowed. Solvation model is not part of the compatibility key, because solvation energies are defined as differences across solvent models.
- Adding a correction of a different kind (for example an RRHO thermal correction to an electronic energy) is allowed and recorded in provenance.

### 4.6 Seeds and determinism

- Every stochastic operation (conformer embedding, sampling, subsampling, configuration recovery, MD) takes `seed=`. If it is omitted, a seed is derived deterministically from the workflow seed and the operation's position in the graph, and recorded.
- Implementations declare whether they are bitwise deterministic. `compile(..., deterministic=True)` restricts selection to deterministic implementations, or fails with `QCL-E207`.

### 4.7 Schedule invariance

With the same device options, changing only `.to()` targets MUST NOT change results beyond floating-point reordering tolerance (documented per primitive) and statistical noise. Precision options are accuracy-trading and recorded in provenance. Property tests enforce this invariant (§14). It is what makes deployment experiments safe.

Implementations registered under one primitive and fidelity contract MUST meet its documented tolerance. A method or approximation that cannot meet that contract is a distinct operation or an explicit accuracy-trading option, not a silent consequence of placement.

---

## 5. Primitive catalogue

Every entry below is a compute operation unless marked *host* or *driver*.

### 5.1 Integrals

- `qcl.int1e(mol, kind)`: overlap, kinetic, nuclear attraction, multipoles, electrostatic potential at points.
- `qcl.int2e(mol)`: four-center electron repulsion integrals, with screening.
- `qcl.int3c(mol, auxbasis)`: density-fitting three-center integrals. `qcl.cholesky(mol, tol)`.
- All integral operations support ghost atoms and external point charges.

### 5.2 Tensor algebra

- `qcl.contract(spec, *tensors)`: einsum-style contraction with contraction-path planning.
- `qcl.transform(t, C)`: AO→MO transformation.
- `qcl.decompose(t, method, rank=)`, where method is `qcl.DF()`, `qcl.Cholesky()` or `qcl.THC()`.

### 5.3 Dense linear algebra

- `qcl.eigh(A, B=None)`, `qcl.orthogonalize(S)`, `qcl.solve(A, b)`, `qcl.svd(A)`, `qcl.diis(history)`.

### 5.4 Mean-field and correlated composites

- `qcl.scf.hf(mol)` (RHF or ROHF, chosen from `mol.spin`), `qcl.scf.rhf`, `qcl.scf.rohf`, `qcl.scf.uhf`, `qcl.dft.rks(mol, xc=)`, `qcl.dft.uks`.
  - Optional data inputs: `eri=` (precomputed integrals), `guess=` (orbitals from an earlier result), `solvent=` (`qcl.PCM(name)` or `qcl.SMD(name)`).
- `qcl.ccsd.amplitudes(act)`, `qcl.mp2(mf)`.
- Grids and exchange-correlation: `qcl.grids(mol, level)`, `qcl.xc(grid, density, functional)`.

### 5.5 Many-body / CI space

- `qcl.active_space(mf, active)` returns an `ActiveSpace` with `.hamiltonian` (a `FermionOperator`) and `.sector`. Requesting more orbitals than the species has is error `QCL-E105`.
- Solvers: `qcl.fci.solve(H, sector)`, `qcl.sci.solve(H, sector, ...)`, `qcl.ed.ground_state(H, sector)` for lattice models.
- Building blocks: `qcl.sigma(H, c)` (the H·c product), `qcl.davidson`, `qcl.lanczos`, `qcl.rdm1`, `qcl.rdm2`, `qcl.expect(state, op)`.
- `qcl.sqd.diagonalize(H, bits, sector)`: configuration recovery, subsampling and projected diagonalization.

### 5.6 Operators and models

- *Host:* `FermionOperator` algebra, `qcl.models.hubbard(lattice, t, U)`, `qcl.lattice.square(nx, ny)`, `qcl.lattice.chain(n)`, observables such as `qcl.ops.onsite_pair(i, j)`, `qcl.ops.spin_spin(i, j)`, `qcl.ops.number(i)`.
- `qcl.qubit_hamiltonian(act_or_H, mapping=)`: fermion-to-qubit mapping. It is a compute operation, because large Hamiltonians make it nontrivial.

### 5.7 Quantum execution

- `qcl.sample(kernel, params=, shots=, seed=, partition=)` returns `Bitstrings`.
- `qcl.estimate(qubit_op, kernel, params=, partition=)` returns an `Estimate`.
- `qcl.time_evolve(H, state, t, method=qcl.Trotter(...))`, `qcl.phase_estimate(...)`.
- Partitioning a circuit across QPUs:
  - `qcl.CircuitCut(max_cuts=)` reconstructs expectation values, so it is allowed with `estimate` only. Using it with `sample` is error `QCL-E104`.
  - `qcl.Teleport()` requires a quantum link between the QPUs, otherwise it is error `QCL-E208`.
- Error mitigation is a QPU device option and applies to `estimate` only. `sample` always returns raw bitstrings; SQD handles noise through configuration recovery.

### 5.8 Derivatives and geometry drivers

- *Host:* `qcl.PES(fn, gradient=)` (§3.5); constraint descriptors `qcl.Bond`, `qcl.Angle`, `qcl.Dihedral`.
- `qcl.gradient(pes, geom)`, `qcl.hessian(pes, geom)`, `qcl.frequencies(hessian)`.
- *Drivers:* `qcl.optimize(pes, geom, fixed=[...])`, `qcl.engines.sella.saddle(pes, guess)`, `qcl.irc(pes, ts)`.

### 5.9 Patterns

- *Parallel:* `qcl.map(f, items)`, with `f(item, lane)`.
- *Drivers:*
  - `qcl.scan(f, init, over)`: `f(carry, x, lane)` returns `(carry, out)`. The scan returns the list of `(carry, out)` pairs.
  - `qcl.iterate(step, init, until)`: data-dependent loops.
  - `qcl.optimize`: geometries or circuit parameters.
  - `qcl.propagate(rhs, y0, t, integrator=)`: time integration with `qcl.Verlet()`, `qcl.RK45()` or `qcl.BDF()` for stiff systems. MD, IRC and kinetics use this pattern.
- *Host:* `qcl.require`, `qcl.cond`, `qcl.rank`, `qcl.mean`, `qcl.compare`.

### 5.10 Engines (opaque compute operations)

| Engine | Operations | Devices |
|---|---|---|
| RDKit | `embed` (ETKDG), `mmff_optimize`, `dedupe` (RMSD), `descriptors` | CPU |
| FreeSASA | `sasa` | CPU |
| tblite | `gfn2.energy`, `gfn2.optimize`, solvent descriptor `qcl.ALPB(name)` | CPU |
| Sella | `saddle` (driver); also an implementation of `qcl.irc` | Host, with lanes for callbacks |
| OpenMM | `prepare`, `minimize`, `sample`, `energy` | CPU, GPU |

An engine that isn't installed is error `QCL-E401`. This corresponds to the catalogue's existing `CF-E080`.

### 5.11 Post-processing (compute operations, usually placed on a CPU)

- `qcl.thermo.rrho(geom, freqs, T, P=)`, `qcl.thermo.eyring(dG, T)`.
- `qcl.superconductivity.allen_dynes(lambda_, omega_log, mu_star)`, only for user- or EPW-supplied electron-phonon inputs.
- `qcl.solubility.esol(mol)`, `qcl.solubility.gse(mol, melting_point_C)`.
- `qcl.metabolism.site_scores(bdes, sasa)`.

*Rationale:* each of these is microseconds of arithmetic, less than the cost of a single GPU kernel launch. They remain ordinary placed operations, so large batches or GPU-resident inputs can still be placed on a GPU.

---

## 6. Devices and clusters (normative)

### 6.1 Cluster file

```toml
# cluster.toml (all numbers are example values)

[[cpu]]
id = "cpu0"
threads = 64

[[gpu]]
id = "gpu0"
memory_gb = 80

[[gpu]]
id = "gpu1"
memory_gb = 80

[[qpu]]
id = "qpu0"
qubits = 156
site = "local"
queue = "dedicated"
native_gates = ["rz", "sx", "cz"]
calibration = "vendor-api"        # source of per-qubit and per-gate fidelities

[[qpu]]
id = "qpu1"
qubits = 156
site = "local"
queue = "dedicated"
native_gates = ["rz", "sx", "cz"]
calibration = "vendor-api"

[[qpu]]
id = "qpu-cloud"
qubits = 100
site = "remote"
queue = "shared"
billing = "metered"                 # optional; remote alone does not imply billing

[[qpu]]
id = "sim0"
simulator = { method = "statevector", host = "gpu1" }

[[link]]
between = ["gpu0", "gpu1"]
kind = "nvlink"
bandwidth_GBps = 450
latency_us = 2

[[link]]
between = ["gpu0", "qpu0"]
kind = "nvqlink"
bandwidth_GBps = 50
latency_us = 4

[[link]]
between = ["cpu0", "qpu-cloud"]
kind = "network"
bandwidth_GBps = 1
latency_us = 50000

[[quantum_link]]
between = ["qpu0", "qpu1"]
bell_pairs_per_s = 1000
fidelity = 0.95
```

### 6.2 Device handles

```python
cluster = qcl.Cluster.load("cluster.toml")     # or qcl.Cluster.local()
cpus, gpus, qpus = cluster.cpus, cluster.gpus, cluster.qpus

g0 = gpus[0].with_options(precision="fp64")
q0 = qpus[0].with_options(transpile=3, mitigate="readout")   # mitigation affects estimate only
```

- Indexing and slicing follow the order of the cluster file. `cluster["gpu0"]` selects by ID.
- Options per device kind:
  - CPU: `threads`.
  - GPU: `precision` (default `"fp64"`), `memory_limit_gb`.
  - QPU: `transpile`, `mitigate`, `max_shots`.
- `qcl.Cluster.local()` detects local CPUs and GPUs. It adds one statevector simulator per GPU, or one CPU simulator if there is no GPU, as QPU devices.
- Simulators are QPU devices backed by a classical device. The planner accounts for their memory on that device.

### 6.3 QPU semantics

- **No addressable memory.** QPU operations take circuits and parameters and return classical results (`Bitstrings` or `Estimate`) through the QPU's link.
- **Stochastic results.** Shots are a budgeted resource, like memory.
- **Latency.** Local, tightly coupled QPUs and queued cloud QPUs differ by orders of magnitude. The planner warns with `QCL-W301` when a QPU with a queue sits inside a driver loop.

### 6.4 Distributed QPU clusters

There are three distinct ways to use several QPUs. Each has different physics, so the program chooses explicitly:

```python
H_qubit = qcl.qubit_hamiltonian(act, mapping="jordan_wigner").to(cpu)
k = qcl.kernels.lucj(act)

# 1. The circuit fits on every QPU: split the work
bits = qcl.sample(k, params=t2, shots=400_000).to(qpus[0:4])              # split by throughput
bits = qcl.sample(k, params=t2, shots=[300_000, 100_000]).to(qpus[0:2])   # explicit split
E    = qcl.estimate(H_qubit, k, params=t2).to(qpus[0:4])                  # measurement groups split

# 2. Too wide for one QPU, classical links only: circuit cutting (estimate only)
E = qcl.estimate(H_qubit, big, params=theta,
                 partition=qcl.CircuitCut(max_cuts=3)).to(qpus[0:2])

# 3. Too wide for one QPU, with a quantum link: teleported gates
bits = qcl.sample(big, params=theta, shots=100_000,
                  partition=qcl.Teleport()).to(qpus[0:2])
```

Rules:

- **Per-device lowering.** The kernel is compiled separately for each QPU's native gates and connectivity. Qubit counts are checked after partitioning (`QCL-E209`).
- **Cutting overhead.** The sampling overhead of circuit cutting grows exponentially with the number of cuts. `plan.estimate()` reports the multiplied shot count.
- **Teleportation cost.** The plan reports how many Bell pairs teleportation consumes, compared with the link's rate and fidelity.
- **Merging results across devices.** Each QPU has its own noise, so results are combined honestly:
  - Bitstrings from several QPUs MAY be merged for SQD, because sampled configurations only define the subspace, and the energy remains a variational bound. Each bitstring keeps its device tag.
  - Expectation values MUST be estimated per device and combined by inverse-variance weighting, keeping per-device provenance.

---

## 7. Compilation, plans and diagnostics (normative)

### 7.1 Pipeline

1. Trace the workflow into the IR.
2. Check types, units, sectors and model compatibility (§4).
3. Check placement rules P1–P11 (§3.3).
4. Check feasibility: memory, qubit counts, connectivity, native gates, links.
5. Insert transfers.
6. Estimate costs.
7. Check budgets (§8.2).
8. Produce the `Plan`.

All errors from steps 2–7 MUST be collected and reported together, not one per run.

### 7.2 Plan API

| Method | Returns |
|---|---|
| `plan.explain(format="text")` | Per operation: device, implementation, transfers (size and link), warnings. `format="json"` for machine use |
| `plan.estimate()` | Predicted wall time, peak memory per device, bytes moved, shots and, for metered cloud QPUs, monetary cost, without running anything |
| `plan.validate(reference=, on=)` | Runs the plan and a reference implementation (for example FCI) on a reduced instance, and reports deviations |
| `plan.run()` | Results with provenance. Enforces budgets |
| `plan.profile()` | Measured timings per operation and device after a run. Feeds cost-model calibration |
| `plan.error_budget()` | Statistical and systematic error components (Phase 6, §8.4) |

### 7.3 Diagnostics

Every diagnostic has a stable code, a source location, the offending expression and a fix hint:

```
error[QCL-E201]: value is never placed
  --> apps/bond_dissociation.py:9:9
   |
 9 |     H = qcl.qubit_hamiltonian(act, mapping="jordan_wigner")
   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ add .to(<device>)
   = help: implementations exist for: cpu, gpu
```

| Code | Meaning |
|---|---|
| `QCL-E101` | Wrong argument type (for example a `QubitOperator` where a `FermionOperator` is required) |
| `QCL-E102` | Unit mismatch |
| `QCL-E103` | Model-incompatible subtraction or comparison |
| `QCL-E104` | Partition strategy not allowed for this operation |
| `QCL-E105` | Active space infeasible for the species |
| `QCL-E201` | Unplaced compute value or pattern |
| `QCL-E202` | Device option invalid for the device kind |
| `QCL-E203` | No implementation for this device kind |
| `QCL-E204` | Device list given to an operation without a parallel axis |
| `QCL-E205` | Implicit transfer under `strict_transfers=True` |
| `QCL-E206` | Kernel feature unsupported by the target QPU |
| `QCL-E207` | No deterministic implementation available |
| `QCL-E208` | `Teleport` without a quantum link |
| `QCL-E209` | Insufficient memory or qubits on the target |
| `QCL-E210` | Computed value used as a Python branch condition while tracing |
| `QCL-E301` | Estimated resource use exceeds the remaining budget, or metered cloud QPU cost exceeds an operator-set monetary limit |
| `QCL-E302` | Operator-configured approval required for a metered cloud QPU job |
| `QCL-E401` | Engine or external tool not installed |
| `QCL-E501` | Device failure at runtime |
| `QCL-E502` | QPU job failed |
| `QCL-E503` | Worker process crashed |
| `QCL-E504` | Convergence failure |
| `QCL-W101` | Validity note (for example "orbital gap is not the fundamental gap") |
| `QCL-W201` | Large implicit transfer |
| `QCL-W301` | Queued QPU inside a driver loop |
| `QCL-W302` | Non-deterministic implementation selected |

---

## 8. Agent interface

### 8.1 How agents use QCL

Agents write QCL programs directly, with the same freedom as humans at every level: algorithm, kernels and deployment. No separate tool protocol is needed. Compile diagnostics, `plan.explain()`, `plan.estimate()`, `plan.validate()` and `plan.profile()` form the agent's feedback loop, and `format="json"` gives machine-readable output.

The recommended loop: write → compile and fix diagnostics → estimate → validate on a reduced instance → run → compare → keep the Pareto front of cost against error.

### 8.2 Budgets (operator-controlled)

Budgets are the only limit that is not a physical fact. The operator sets them in a policy file that the runtime loads:

```toml
# qcl-policy.toml, written by the operator, not by programs
[budget]
qpu_shots = 5_000_000
wall_time = "4h"
allowed_qpus = ["qpu0", "qpu1", "sim0"]
```

For a metered cloud QPU, the operator MAY additionally configure a monetary
limit and an approval threshold for its billing account:

```toml
[cloud_qpu]
qpu_cost = 200.0                 # in the declared account currency
currency = "CNY"
approval_above = 50.0
```

- Programs MAY lower limits, by passing `budget=qcl.Budget(...)` to `compile`, but MUST NOT raise them.
- Every QPU is subject to resource budgets such as shots, device time and wall time. Local and non-billable QPUs require no monetary forecast or approval. `site = "remote"` does not imply billing; metered billing MUST be declared explicitly by the device adapter or cluster configuration.
- Monetary limits and approval thresholds apply only to explicitly metered cloud QPUs and only when the operator configured them. `plan.run()` checks `plan.estimate()` against the remaining applicable budget (`QCL-E301`) and requests approval above an applicable threshold (`QCL-E302`). If a configured monetary limit cannot be checked because the cloud tariff or cost forecast is unavailable, submission is refused before the job starts.
- A cloud price estimate records its account currency, tariff version and validity time. When approval is configured, the operator's approval is bound to the plan hash, billing account and maximum approved cost; a different plan or account requires a new approval.

### 8.3 Optimization space

| Kind | Knobs | Effect on results |
|---|---|---|
| Accuracy-preserving | `.to()` targets, device lists, explicit copies, reusing integrals (`eri=`) and orbitals (`guess=`), batching, shot allocation across measurement groups, choice among exact implementations | None beyond schedule-invariance tolerance (§4.7) |
| Accuracy-trading | Basis, active space, method level, DF/THC rank, DFT grid, convergence thresholds, shot count, SQD subspace size, conformer count, scan points, solvation model, GPU precision | Changes the answer; recorded in provenance |

### 8.4 Error budget (Phase 6)

The design target is a well-posed optimization: minimize cost subject to total error ≤ ε, for example chemical accuracy (about 1.6 mEh). `plan.error_budget()` reports:

- **Statistical error** from shots.
- **Systematic error** from truncations (basis, active space, DF rank, thresholds), estimated by convergence sequences, extrapolation, or comparison with a reference on a reduced instance.

### 8.5 Pitfalls the design guards against

- **Goodhart's law.** Results state which reference they were validated against. Matching a model-defined reference measures reproducibility of that model, not experimental accuracy.
- **Small-system overfitting.** Settings tuned on small systems can fail at scale. `plan.validate` SHOULD be run at two or more system sizes before trusting a method choice.
- **Noise-chasing.** `qcl.compare` reports a difference only when it exceeds the combined uncertainty.

### 8.6 Learnability

The language is new, so agents can only learn it from its documentation. Therefore:

- Ship this specification, the API reference and the §9 example library inside the package.
- Keep the API regular: one placement mechanism, consistent argument names, and every operation documents its axis, implementations and validity flags.
- Treat diagnostic quality as a feature. Error messages are how agents learn.

---

## 9. Application catalogue and worked examples

Catalogue policy carries over: energies are in hartree except MM/GBSA (kcal/mol), STO-3G examples are inexpensive software checks rather than production chemistry, and a gate against a model-defined reference measures reproducibility of that model.

| Application | Patterns | Main operations and engines | QPU path | Example |
|---|---|---|---|---|
| `ground_state_energy` | Single point | Integrals, SCF, active space, FCI/SCI, SQD, sampled VQE | SQD sampling; sampled VQE | §9.1 |
| `stability_ranking` | `map` + `rank` | Ground-state energy per isomer | Via the ground-state kernel | — |
| `conformer_ranking` | Engines + `map` + `rank` | RDKit ETKDG and MMFF (or tblite GFN2), RMSD dedupe, electronic energies | SQD sampling | §9.2 |
| `pes_scan` | `map` (rigid) or `scan` + `optimize` (relaxed) | Internal-coordinate edits, gradients, FCI | Via the ground-state kernel | §9.4 |
| `metabolic_sites` | `map` + post-processing | ROHF/RHF C–H BDEs, FreeSASA, site scoring | — | — |
| `bond_dissociation` | `map` | Bond cleavage, fragment FCI | Via the ground-state kernel | §9.3 |
| `solvation_scan` | `map` | PCM SCF with integral reuse, or tblite GFN2/ALPB | — | §9.6 |
| `frontier_orbitals` | Single point | SCF eigenvalues, N±1 charge-sector FCI | Charge-sector SQD | §9.8 |
| `aqueous_solubility` | Post-processing | RDKit descriptors, ESOL/GSE | — | — |
| `reaction_barrier` | `optimize`, `saddle`, `hessian`, `irc` | Sella, Hessian, IRC, CAS FCI, RRHO, Eyring | Via the ground-state kernel | §9.5 |
| `dimer_binding_scan` | `map` | Ghost atoms (counterpoise), FCI | Via the ground-state kernel | §9.9 |
| `protein_ligand_binding` | Engines + `map` + `mean` | OpenMM prepare, sample, energy (Amber14, GAFF2, GBn2) | — | §9.7 |
| `lattice_superconductivity` | Single point | Hubbard model, exact diagonalization, correlation functions; Allen–Dynes for user inputs | `estimate` of correlations (future) | §9.10 |

"Via the ground-state kernel" means the application calls a correlated-energy function, so a QPU-capable implementation of it (such as `sqd_energy` below) lights up that application without changes.

### 9.1 Shared setup and helpers

```python
import qcl

cluster = qcl.Cluster.load("cluster.toml")
cpus, gpus, qpus = cluster.cpus, cluster.gpus, cluster.qpus

cpu = cpus[0]
g0  = gpus[0].with_options(precision="fp64")
g1  = gpus[1].with_options(precision="fp64")
q0  = qpus[0].with_options(transpile=3)
q1  = qpus[1].with_options(transpile=3)


@qcl.workflow
def fci_energy(mol, active, dev):
    mf  = qcl.scf.hf(mol).to(dev)                     # RHF or ROHF, chosen from mol.spin
    act = qcl.active_space(mf, active).to(dev)
    return qcl.fci.solve(act.hamiltonian, sector=act.sector).to(dev).energy


@qcl.workflow
def sqd_energy(mol, active, gpu, qpu, shots=100_000):
    mf   = qcl.scf.rhf(mol).to(gpu)
    act  = qcl.active_space(mf, active).to(gpu)
    t2   = qcl.ccsd.amplitudes(act).to(gpu)           # parameters for the LUCJ circuit
    bits = qcl.sample(qcl.kernels.lucj(act), params=t2, shots=shots).to(qpu)
    return qcl.sqd.diagonalize(act.hamiltonian, bits, sector=act.sector).to(gpu).energy
```

`dev` may be a single device or a list (rule P3). Passing `qpu=qpus[0:4]` to `sqd_energy` splits its shots across four QPUs without changing the function.

### 9.2 Conformer ranking (CPU → GPU → QPU)

```python
@qcl.workflow
def conformer_ranking(smiles, cpu, workers, lanes, n=20, rmsd=0.5, seed=42,
                      active=qcl.Active(electrons=10, orbitals=8)):
    confs = qcl.engines.rdkit.embed(smiles, n=n, seed=seed).to(cpu)          # ETKDG
    confs = qcl.map(lambda c, w: qcl.engines.rdkit.mmff_optimize(c).to(w),
                    confs).to(workers)                                       # MMFF
    confs = qcl.engines.rdkit.dedupe(confs, rmsd=rmsd).to(cpu)               # may leave one
    E = qcl.map(lambda c, lane: sqd_energy(c, active, gpu=lane[0], qpu=lane[1]),
                confs).to(lanes)
    return qcl.rank(E)                    # reports only pairs resolved beyond uncertainty

plan = conformer_ranking.compile("CCOC(=O)C", cpu=cpu, workers=cpus[0:8],
                                 lanes=[(g0, q0), (g1, q1)])
plan.explain()
result = plan.run()
```

The executor pipelines this automatically from data dependencies: while a QPU samples conformer 2, a GPU diagonalizes conformer 1.

### 9.3 Bond dissociation energies

```python
@qcl.workflow
def bond_dissociation(mol, bonds, lanes, active=qcl.Active(max_orbitals=6)):
    E0 = fci_energy(mol, active, dev=lanes[0])
    def bde(bond, dev):
        a, b = mol.cleave(bond)        # host edit: two doublet radicals at the parent geometry
        return fci_energy(a, active, dev) + fci_energy(b, active, dev) - E0
    return qcl.map(bde, bonds).to(lanes)

bond_dissociation.compile(mol, mol.bonds(elements=("C", "H")), lanes=[g0, g1]).run()
```

The active space uses `max_orbitals` because an H-atom fragment has fewer than six orbitals in most basis sets; a fixed size would be `QCL-E105`. These are rigid homolytic BDEs, not relaxed or thermal ones.

### 9.4 Potential energy surface scans

```python
@qcl.workflow
def relaxed_dihedral_scan(mol, atoms, angles, devs,
                          active=qcl.Active(electrons=6, orbitals=6)):
    pes = qcl.PES(lambda m, dev: qcl.scf.hf(m).to(dev),
                  gradient=qcl.FiniteDifference())
    def point(prev, angle, lane):                      # sequential driver: lane = all devs
        guess = prev.set_dihedral(*atoms, angle)       # warm start from the previous point
        geom  = qcl.optimize(pes, guess, fixed=[qcl.Dihedral(*atoms)]).to(lane)
        return geom, fci_energy(geom, active, dev=lane[0])
    return qcl.scan(point, init=mol, over=angles).to(devs)


@qcl.workflow
def rigid_dihedral_scan(mol, atoms, angles, lanes,
                        active=qcl.Active(electrons=6, orbitals=6)):
    def point(angle, dev):
        return fci_energy(mol.set_dihedral(*atoms, angle), active, dev)
    return qcl.map(point, angles).to(lanes)

relaxed_dihedral_scan.compile(mol, (0, 1, 2, 3), range(0, 360, 15), devs=[g0, g1]).run()
```

In the relaxed scan, the 6N displaced energies of each finite-difference gradient are spread over both GPUs, but the points run in sequence because each starts from the previous geometry. The rigid scan parallelizes across points instead. Choosing between them is a real design decision.

### 9.5 Reaction barrier

```python
@qcl.workflow
def reaction_barrier(reactant, ts_guess, product, cpu, devs, T=298.15,
                     active=qcl.Active(electrons=4, orbitals=4)):
    pes = qcl.PES(lambda m, lane: qcl.scf.hf(m).to(lane), gradient=qcl.Analytic())

    reactant = qcl.optimize(pes, reactant).to(devs)
    ts       = qcl.engines.sella.saddle(pes, ts_guess).to(devs)   # driver on host, PES on devs

    f_ts, f_r = qcl.map(
        lambda m, dev: qcl.frequencies(qcl.hessian(pes, m).to(dev)).to(cpu),
        [ts, reactant]).to(devs)
    qcl.require(f_ts.n_imaginary == 1)

    path = qcl.irc(pes, ts).to(devs)                               # an ODE: propagate pattern
    qcl.require(path.connects(reactant, product))

    E_ts, E_r = qcl.map(lambda m, dev: fci_energy(m, active, dev),
                        [ts, reactant]).to(devs)
    G_ts = E_ts + qcl.thermo.rrho(ts, f_ts, T).to(cpu)
    G_r  = E_r  + qcl.thermo.rrho(reactant, f_r, T).to(cpu)
    dG   = G_ts - G_r
    return dG, qcl.thermo.eyring(dG, T).to(cpu)

reaction_barrier.compile(r, ts0, p, cpu=cpu, devs=[g0, g1]).run()
```

The CAS(4,4) default matches the catalogue's HCN example. No tunnelling or dynamical recrossing correction is applied.

### 9.6 Solvation scan

```python
@qcl.workflow
def solvation_scan(mol, solvents, devs):
    eri  = qcl.int2e(mol).to(devs[0])
    eris = {devs[0]: eri, **{d: eri.to(d) for d in devs[1:]}}   # one explicit copy per extra device
    E_gas = qcl.scf.rhf(mol, eri=eri).to(devs[0]).energy
    def solvated(solvent, dev):
        E = qcl.scf.rhf(mol, eri=eris[dev], solvent=qcl.PCM(solvent)).to(dev).energy
        return E - E_gas
    return qcl.map(solvated, solvents).to(devs)

solvation_scan.compile(mol, ["water", "methanol", "toluene"], devs=[g0, g1],
                       strict_transfers=True).run()
```

The integrals are computed once and reused for every solvent. Because every copy is an explicit `.to()`, the plan compiles under `strict_transfers=True`. PCM here is electrostatic only, and all solvents use the same geometry.

### 9.7 Protein–ligand binding (MM/GBSA)

```python
@qcl.workflow
def protein_ligand_binding(receptor, ligand, cpu, gpu, frames=10, seed=42):
    system = qcl.engines.openmm.prepare(
        receptor, ligand,
        forcefield=qcl.mm.Amber14(ligand=qcl.mm.GAFF2()),
        implicit=qcl.mm.GBn2()).to(cpu)
    traj = qcl.engines.openmm.sample(system, frames=frames, minimize=True,
                                     seed=seed).to(gpu)
    def mmgbsa(frame, dev):
        E = lambda x: qcl.engines.openmm.energy(x).to(dev)
        return E(frame.complex) - E(frame.receptor) - E(frame.ligand)
    return qcl.mean(qcl.map(mmgbsa, traj).to(gpu))

protein_ligand_binding.compile(qcl.PDB("receptor.pdb"), qcl.SDF("ligand.sdf"),
                               cpu=cpu, gpu=g0).run()
```

The result is in kcal/mol, depends on preparation and protonation, and has no configurational entropy term. A SMILES ligand alone is not docking.

### 9.8 Frontier orbitals, IP and EA

```python
@qcl.workflow
def frontier_orbitals(mol, dev, active=qcl.Active(electrons=6, orbitals=6)):
    mf  = qcl.scf.rhf(mol).to(dev)
    act = qcl.active_space(mf, active).to(dev)
    E = {dn: qcl.fci.solve(act.hamiltonian,
                           sector=act.sector.add_electrons(dn)).to(dev).energy
         for dn in (-1, 0, +1)}
    ip = E[-1] - E[0]
    ea = E[0] - E[+1]
    return mf.homo, mf.lumo, ip, ea, ip - ea      # the fundamental gap is IP − EA
```

All three charge sectors use the same orbitals. The orbital gap carries validity note `QCL-W101`.

### 9.9 Dimer binding scan with counterpoise correction

```python
@qcl.workflow
def dimer_binding_scan(dimer, frag_a, frag_b, separations, lanes,
                       active=qcl.Active(max_orbitals=8)):
    def point(r, dev):
        d    = dimer.set_separation(frag_a, frag_b, r)    # rigid monomers, centroid distance
        E_ab = fci_energy(d, active, dev)
        E_a  = fci_energy(d.ghost(frag_b), active, dev)   # monomer A in the full dimer basis
        E_b  = fci_energy(d.ghost(frag_a), active, dev)
        return E_ab - E_a - E_b                           # Boys–Bernardi counterpoise
    return qcl.map(point, separations).to(lanes)
```

This is a rigid interaction curve, not a binding free energy.

### 9.10 Hubbard lattice pairing

```python
@qcl.workflow
def hubbard_pairing(lattice, U, sector, i, j, dev):
    H    = qcl.models.hubbard(lattice, t=1.0, U=U)             # host: builds a FermionOperator
    gs   = qcl.ed.ground_state(H, sector=sector).to(dev)       # sparse Lanczos
    pair = qcl.expect(gs, qcl.ops.onsite_pair(i, j)).to(dev)
    spin = qcl.expect(gs, qcl.ops.spin_spin(i, j)).to(dev)
    return gs.energy, pair, spin

hubbard_pairing.compile(qcl.lattice.square(2, 4), U=4.0, sector=qcl.Sector(n=8, sz=0),
                        i=0, j=5, dev=g0).run()
```

Finite clusters only; unconventional Tc is not predicted.

### 9.11 Portability

The same workflows run on a laptop by changing only the call. `qcl.Cluster.local()` includes a GPU statevector simulator as a QPU device:

```python
local = qcl.Cluster.local()
compute = local.gpus[0] if local.gpus else local.cpus[0]
conformer_ranking.compile("CCOC(=O)C", cpu=local.cpus[0], workers=local.cpus,
                          lanes=[(compute, local.qpus[0])]).run()
```

An agent sweeps deployments by changing the device arguments, and changes the algorithm by editing the function body.

---

## 10. Implementation architecture (Rust)

### 10.1 Language choices

| Layer | Language | Reason |
|---|---|---|
| Frontend | Python | Users, agents and the chemistry ecosystem live there |
| Compiler, runtime, scheduler | Rust | The core is a concurrent async runtime managing device buffers, CUDA streams, in-flight QPU jobs and worker processes. Rust rules out use-after-free and data races at compile time, and cargo + maturin make packaging simple. Precedent: Qiskit moved its core to Rust and exposes it through a C API |
| GPU kernels | CUDA C++ (HIP later for AMD) | Rust GPU kernel authoring is still maturing. Kernels are built with nvcc from `build.rs` (or NVRTC at runtime) and launched through `cudarc` |
| Interop boundary | C ABI (header generated by cbindgen) | Stable across compilers; callable from Fortran, C++, Julia and Python |
| Multi-node | MPI (`rsmpi` bindings) for CPUs, NCCL for GPUs | Standard on HPC systems |
| Existing Fortran/C/C++ libraries | Link, don't rewrite | libcint, libxc, tblite, SUNDIALS are already optimized |

### 10.2 Crates

| Crate | Responsibility |
|---|---|
| `qcl-ir` | Graph, operations, core types (`FermionOperator`, `QubitOperator`, `Estimate`, units), provenance |
| `qcl-check` | Type, unit, sector and compatibility checks; placement rules P1–P11; feasibility against the cluster |
| `qcl-sched` | Transfer insertion, cost estimation, budget checks, pipelining |
| `qcl-runtime` | Async executor (tokio), buffers, CUDA streams and events, worker management |
| `qcl-cluster` | Cluster file parsing (serde + toml), device registry, policy file |
| `qcl-cpu` | CPU implementations (faer or BLAS/LAPACK, rayon) |
| `qcl-cuda` | GPU implementations via `cudarc` (cuBLAS, cuSOLVER, cuTENSOR, NCCL, NVRTC) and custom CUDA kernels |
| `qcl-qpu` | QPU device abstraction, job adapters, simulators, shot splitting, result merging |
| `qcl-abi` | Stable C ABI and plugin registration |
| `qcl-py` | PyO3 bindings, packaged with maturin |

### 10.3 Core traits

```rust
pub enum DeviceKind { Host, Cpu, Gpu, Qpu }

pub struct Cost {
    pub seconds: f64,
    pub peak_bytes: Vec<(DeviceId, u64)>,
    pub bytes_moved: u64,
    pub shots: u64,
    pub qpu_cost: Option<f64>,             // None when a metered tariff is unavailable
}

pub trait Primitive: Send + Sync {
    fn id(&self) -> &'static str;                    // e.g. "scf.rhf"
    fn signature(&self) -> &Signature;               // typed inputs and outputs, units
    fn parallel_axis(&self) -> Option<Axis>;         // §3.4
    fn validity(&self) -> &[ValidityFlag];           // §4.5
    fn impls(&self) -> &[Arc<dyn Impl>];
}

pub trait Impl: Send + Sync {
    fn id(&self) -> &'static str;                    // e.g. "scf.rhf/gpu4pyscf", recorded in provenance
    fn device_kind(&self) -> DeviceKind;
    fn fidelity(&self) -> Fidelity;                  // Exact | Approx { .. } | Sampled { .. }
    fn deterministic(&self) -> bool;
    fn check(&self, args: &ArgMeta, devs: &[DeviceRef]) -> Result<(), Diagnostic>;
    fn cost(&self, args: &ArgMeta, devs: &[DeviceRef]) -> Cost;
    fn launch(&self, ctx: LaunchCtx, args: Vec<Buffer>)
        -> BoxFuture<'static, Result<Vec<Buffer>, ExecError>>;
}

pub enum Buffer {
    Host(Arc<HostBuf>),
    Cuda(Arc<CudaBuf>),      // DLPack export/import; CUDA IPC across processes
    Worker(WorkerRef),       // resident inside a Python worker process
    QpuJob(JobHandle),       // resolves to Bitstrings or Estimate when the job finishes
}
```

- `check` turns an infeasible `.to()` into a compile error. `cost` powers `plan.estimate()`.
- If several implementations exist for a device kind, the highest registered priority wins. An operation may pin one with `impl="pyscf"`. The chosen ID is recorded in provenance.

### 10.4 Integrating third-party code

1. **Linked natively, in the same process.** C, C++, CUDA and Fortran libraries with C APIs: libcint, libxc, FreeSASA, tblite, SUNDIALS, cuQuantum, and the CUDA math libraries through `cudarc`. Use this route for hot primitives.
2. **Python engines in worker processes.** PySCF, gpu4pyscf, RDKit, Sella, ffsim, qiskit-addon-sqd, qiskit-addon-cutting, the Qiskit transpiler, and OpenMM (its Python API at first; native C++ linking later).
   - One long-lived process per (library group, device). RPC over a local socket with MessagePack.
   - CPU arrays move through shared memory, GPU arrays through CUDA IPC handles (zero-copy).
   - Workers are stateless apart from content-addressed caches, and each has a GPU memory cap.
   - A crashed worker surfaces as `QCL-E503` and is restarted. Only idempotent operations are retried.
   - *Rationale:* calling Python inside the Rust process would let the GIL serialize the scheduler, and each library brings its own GPU memory pool (CuPy for gpu4pyscf). Separate processes isolate both problems.
3. **Remote and device services.** QPUs through vendor SDKs or REST, using OpenQASM 3 or QIR as the interchange format. CUDA-Q provides GPU simulators and NVQLink-coupled QPUs. QPU results are `QpuJob` buffers that the executor awaits while other work continues.

### 10.5 Memory and transfers

- One pooled, stream-ordered allocator per GPU inside the Rust process.
- The transfer path is chosen from the cluster topology: NVLink peer copies, PCIe through pinned host memory, NCCL for collectives and multi-node, the network for remote devices, and NVQLink through CUDA-Q's real-time interfaces.
- DLPack for zero-copy exchange with in-process libraries; CUDA IPC with worker processes.
- CUDA streams and events express dependencies, so the executor overlaps transfers with compute.

### 10.6 Python frontend

- PyO3 bindings, built with maturin into a wheel.
- Tracing: Python proxy objects record each call into a Rust graph builder, capturing source locations for diagnostics.
- `qcl.map` callbacks are traced once per lane, with the lane as a concrete device value. This is why a callback can, for example, index a dictionary by its device.
- The IR has a versioned serialization format for caching, reproducibility and agent inspection (format: §17).

### 10.7 C ABI

- `qcl.h` is generated by cbindgen, uses opaque handles and carries an explicit ABI version.
- `qcl_register_impl(...)` lets external C, C++ or Fortran code add implementations or new device kinds.
- A later phase adds C entry points for building and running plans, so HPC codes can use QCL without Python.

---

## 11. Primitive implementation plan

| Family | CPU | GPU | QPU | Long term |
|---|---|---|---|---|
| Integrals (1e/2e, DF, ghost atoms, point charges) | libcint (linked) | gpu4pyscf kernels | — | Keep wrapped |
| Tensor contractions, AO→MO transforms | faer or BLAS | cuTENSOR via `cudarc` | — | Write the contraction-path planner (small) |
| Dense linear algebra | faer or LAPACK | cuSOLVER via `cudarc` | — | Keep wrapped |
| SCF variants, gradients, Hessians, PCM/SMD | PySCF | gpu4pyscf | — | Keep wrapped |
| XC functionals on grids | libxc | gpu4pyscf-libxc-cuda | — | Keep wrapped |
| CI space (σ-vector, Davidson, Lanczos, RDMs, sectors) | PySCF `fci` | gpu4pyscf in-core FCI | — | **Write in Rust + CUDA (Phase 4)** |
| Fermion/qubit operators and mappings | Own Rust types | — | — | Write (simple and central) |
| Circuits (LUCJ, transpilation, cutting) | ffsim, Qiskit, qiskit-addon-cutting | CUDA-Q simulators | CUDA-Q, vendor SDKs | Keep wrapped |
| SQD post-processing | qiskit-addon-sqd + own CI engine | Own CI engine | — | Wrap configuration recovery; own the diagonalization |
| `Estimate`, shot statistics, cross-device merging | Own Rust | — | — | Write (it is the type system) |
| Engines | RDKit, FreeSASA, tblite, Sella | OpenMM | — | Stay opaque |
| Integrators (MD, IRC, kinetics) | Own Verlet/RK; SUNDIALS for stiff systems | Own CUDA later | CUDA-Q time evolution | Mostly write (small) |
| Post-processing (RRHO, Eyring, ESOL, scoring) | Own Rust | — | — | Write (tens of lines each) |

Notes:

- **gpu4pyscf coverage.** It provides SCF with analytical gradients and Hessians for HF and DFT, PCM models with gradients and Hessians, and unrestricted HF/DFT. MP2/DF-MP2 and CCSD are marked experimental. Release notes also list an in-core FCI solver and a separate CUDA build of libxc. Verify ROHF support before Phase 2; if it is missing, open-shell SCF on GPU is `QCL-E203` and programs place it on a CPU.
- **SQD stack.** qiskit-addon-sqd provides configuration recovery, subsampling and counts utilities under Apache 2.0. ffsim builds the LUCJ circuit from CCSD t-amplitudes.
- **The native CI engine** is the one major numerical component written in-house. Nine catalogue applications, the SQD diagonalization and Hubbard exact diagonalization all reduce to the same operation: a sparse H·c product in a determinant subspace with sector constraints, inside a Davidson or Lanczos loop.
  - Design: alpha/beta string factorization of the σ-vector; Davidson and Lanczos drivers; 1- and 2-RDMs; sector and point-group constraints; CUDA C++ kernels for the σ-vector; CPU implementation in Rust as the reference.
  - It is well-defined enough to be tested exhaustively against PySCF.

---

## 12. Repository layout

```
qcl/
├── Cargo.toml                  # Rust workspace
├── pyproject.toml              # maturin build for the Python package
├── crates/
│   ├── qcl-ir/  qcl-check/  qcl-sched/  qcl-runtime/  qcl-cluster/
│   ├── qcl-cpu/  qcl-cuda/  qcl-qpu/
│   ├── qcl-abi/                # C ABI, cbindgen config
│   └── qcl-py/                 # PyO3 bindings
├── kernels/cuda/               # CUDA C++ kernels (CI engine, fused kernels), built by build.rs
├── python/qcl/                 # Python frontend (thin; delegates to qcl-py)
├── workers/                    # Python worker processes, one package per library group
│   ├── pyscf/  gpu4pyscf/  rdkit/  sella/  openmm/  quantum/   # quantum: ffsim, qiskit addons
├── apps/                       # the 13 catalogue applications written in QCL
├── tests/
│   ├── unit/  integration/  reference/  property/  regression/  perf/
│   └── tolerances.toml         # per-primitive numerical tolerances
├── docs/
│   ├── spec/                   # this document
│   ├── api/                    # generated API reference
│   └── examples/               # example library for humans and agents
└── containers/                 # pinned CUDA/toolchain images and lockfiles
```

---

## 13. Build phases and acceptance criteria

### Phase 0: Specification and skeleton

- Cargo workspace, maturin build, CI.
- IR types, Python tracing, checker implementing P1–P11 and the `E1xx`/`E2xx` diagnostics, with stub implementations.
- **Acceptance:** every example in §9 traces and type-checks; each placement rule and each `E1xx`/`E2xx` code has a negative test that produces the documented diagnostic.

### Phase 1: CPU correctness

- Worker infrastructure; libcint linked natively; PySCF (SCF, FCI), RDKit, FreeSASA, tblite, Sella and OpenMM (CPU platform) implementations.
- `Estimate`, units, provenance, validity flags, seeds, budgets, `plan.explain()` and `plan.estimate()`.
- **Acceptance:** all 13 catalogue applications run on CPU and match their references within `tests/tolerances.toml`; two runs in deterministic mode are bitwise identical; every result has complete provenance.

### Phase 2: GPU, single node

- `cudarc` backend with cuBLAS, cuSOLVER and cuTENSOR implementations; gpu4pyscf worker; CUDA IPC; NCCL; device lists split along parallel axes.
- **Acceptance:** GPU results match CPU results within tolerance (≤ 1e-8 Eh for SCF energies unless a primitive documents otherwise); schedule-invariance property tests pass over randomized placements; `plan.estimate()` is within a factor of 2 of measured time on the benchmark set.

### Phase 3: QPU

- QPU device abstraction; CUDA-Q simulators registered as QPU devices; ffsim LUCJ circuits; qiskit-addon-sqd; `sample` and `estimate`; shot splitting; per-device result merging; circuit cutting; `Teleport` checks; cloud QPU adapters; resource budgets on every QPU and optional monetary/approval policy for metered cloud QPUs.
- **Acceptance:** on simulators, SQD reproduces FCI within tolerance for the small active spaces in the catalogue examples; resource budgets apply to every QPU, and monetary/approval tests apply only to explicitly metered cloud QPUs; the `E104`, `E206`, `E208` and `E209` checks have tests.

### Phase 4: Native CI engine

- Rust CPU reference and CUDA σ-vector kernels; Davidson, Lanczos, RDMs and sectors; used by `fci.solve`, `sqd.diagonalize` and `ed.ground_state`.
- **Acceptance:** matches PySCF FCI to 1e-10 Eh on the test set; faster than the wrapped implementation at the sizes listed in `tests/perf/`.

### Phase 5: Multi-node

- MPI and NCCL across nodes; remote device sites in the cluster file; failure handling and job recovery for long runs; C entry points for building and running plans.
- **Acceptance:** catalogue applications run across two or more nodes with results identical to single-node runs within tolerance; injected node failures produce `E5xx` diagnostics, and recovery tests pass.

### Phase 6: Agent features and performance

- Cost models calibrated from `plan.profile()` logs; `plan.error_budget()`; JSON outputs everywhere; the example library in `docs/examples/`; profile-driven replacement of wrapped kernels.
- **Acceptance:** `plan.estimate()` within 25% of measured time on the benchmark set; `error_budget()` brackets the reference error on the validation set.

---

## 14. Testing and validation strategy

- **Unit tests** for every primitive implementation against its CPU reference implementation.
- **Reference tests** against independent results: PySCF FCI, the catalogue's literature references, and independent recomputations (for example RDKit descriptor fits and OpenMM Reference-platform energies).
- **Property tests:** schedule invariance (§4.7), unit algebra, variance propagation, model-compatibility rules.
- **Regression suite:** the 13 catalogue applications with stored reference outputs, run in CI on every change.
- **Determinism tests:** bitwise reproducibility in deterministic mode; seed recording for stochastic operations.
- **Diagnostic tests:** one negative test per diagnostic code.
- **Fault injection:** worker crashes, QPU job failures, device loss, and timeouts.
- **Performance benchmarks** in `tests/perf/`, tracked over time.

---

## 15. Engineering risks

| Risk | Mitigation |
|---|---|
| Several libraries competing for GPU memory (CuPy, cuQuantum, OpenMM, the runtime) | One allocator inside the Rust process; each Python engine in its own worker with a memory cap |
| CUDA version misalignment across `cudarc`, CuPy, gpu4pyscf, cuQuantum and CUDA-Q | Ship pinned container images with lockfiles; test the matrix in CI |
| GPU reductions with atomics are not bitwise reproducible | `deterministic()` flag on every implementation; deterministic mode for validation runs |
| Worker RPC overhead for fine-grained operations | Keep worker operations coarse; move hot primitives to native implementations |
| Experimental upstream APIs (gpu4pyscf MP2/CCSD; the Qiskit C API) and QPU vendor API churn | Isolate each behind an `Impl`; pin versions; contract tests per adapter |
| Licensing of redistributed binaries, especially NVIDIA libraries | License audit before shipping containers |

---

## 16. Decision log

| ID | Decision | Rationale |
|---|---|---|
| D1 | Embedded Python DSL; no new syntax | Users, agents and libraries live in Python; a new syntax would not be adopted |
| D2 | `.to()` is the only placement mechanism | One construct at every level; familiar from PyTorch; earlier designs with `at=`, `on=` and per-stage keywords were rejected as inconsistent |
| D3 | Every compute value must be placed; no default scheduler | Deployment is fully visible in the code; mistakes are compile errors |
| D4 | Agents get full freedom at every level, through the language itself; no separate tool protocol | The language plus the plan API is a complete interface |
| D5 | Operator-set resource budgets apply to every device; monetary caps and approval thresholds are optional for explicitly metered cloud QPUs only | Local QPUs consume scarce resources without necessarily incurring a cloud charge |
| D6 | Rust core, CUDA C++ kernels, C ABI boundary, Python via PyO3 | Safety for a concurrent runtime, simple packaging, stable interop. A C++ core would be preferred only if building on MLIR or with a C++-only team |
| D7 | No MLIR in v1 | Circuit lowering is delegated to CUDA-Q and Qiskit; a chemistry-level IR is simpler as Rust data structures |
| D8 | Engines (OpenMM, RDKit, tblite, Sella, FreeSASA) are opaque | Decomposing them gives no benefit; the planner only needs inputs, outputs, device and cost |
| D9 | Wrap first; write only the CI engine early | It accelerates 9 applications plus SQD and lattice ED; other rewrites are profile-driven |
| D10 | Molecule and geometry edits are host operations, not primitives | They are instant data edits; recorded for provenance, never scheduled |
| D11 | Post-processing is ordinary placed operations, typically on CPU | A GPU launch costs more than the arithmetic; batched or GPU-resident cases can still use a GPU |
| D12 | QPU results are typed (`Bitstrings`, `Estimate`); mitigation applies to `estimate` only; circuit cutting is `estimate`-only | Prevents silently invalid statistics |
| D13 | Quantum ODE solvers, if added, return only `Estimate`s of observables | Reading out a full solution vector erases the quantum speedup |
| D14 | Unit conversion is `.in_units()` | `.to()` is reserved for placement |
| D15 | Host expressions over computed values are deferred host IR nodes | Tracing cannot know runtime energies, paths or ranks |
| D16 | `.to(qpu)` places only QPU-capable operations; arbitrary value copies to QPUs are invalid | QPUs have no addressable value memory |
| D17 | Implementations sharing a primitive must satisfy one fidelity contract | Placement changes alone must preserve the scientific result within tolerance |
| D18 | Local examples must support a CPU-only host | `Cluster.local()` does not guarantee a GPU |
| D19 | Dynamic pattern collections remain symbolic; lane identity cannot change the scientific question | Work assignment may vary without changing results |

---

## 17. Open questions

1. Final project name.
2. IR serialization format and versioning policy (for example MessagePack or FlatBuffers).
3. An optional placement-suggestion tool that *writes* `.to()` calls for review, without weakening explicit placement.
4. ROHF and other open-shell coverage on GPU in gpu4pyscf.
5. Timing of the AMD/HIP backend.
6. Covariance tracking for `Estimate`s that share inputs beyond shared samples.
7. Methods for estimating systematic errors in `plan.error_budget()`.
8. Initial QPU vendor list, and when quantum-link hardware is available to exercise `Teleport` beyond simulation.
9. Cost and safety policy for eager mode on real QPUs.

---

## 18. References

- NVQLink overview (Quantum Computing Report): https://quantumcomputingreport.com/?p=37637
- CUDA-Q 0.5 release blog (NVIDIA): https://developer.nvidia.com/blog/cuda-quantum-0-5-delivers-new-features-for-quantum-classical-computing/
- gpu4pyscf: https://github.com/pyscf/gpu4pyscf
- gpu4pyscf v0.7.9 release notes: https://github.com/pyscf/gpu4pyscf/releases/tag/v0.7.9
- cudarc: https://github.com/chelsea0x3b/cudarc and https://docs.rs/cudarc
- qiskit-addon-sqd: https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-sqd
- IBM tutorial, SQD for a chemistry Hamiltonian: https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization
- Qiskit C API: https://quantum.cloud.ibm.com/docs/api/qiskit-c
- Qiskit beyond Python (IBM blog): https://www.ibm.com/quantum/blog/qiskit-beyond-python
- Other libraries referenced: PySCF, libcint, libxc, OpenMM, RDKit, FreeSASA, tblite, Sella, ffsim, Qiskit, qiskit-addon-cutting, CUDA-Q, cuQuantum, SUNDIALS, PyO3, maturin, cbindgen, faer, tokio, rsmpi, NCCL.

---

## Appendix A. Design rationale

### A.1 Why a QPU is not just another GPU

A QPU has no memory space: you cannot copy a tensor onto it, only send a circuit with parameters and receive classical bitstrings. Its outputs are stochastic, so shots are a budgeted resource and results need error bars. Its latency ranges from microseconds (tightly coupled, for example over NVQLink) to hours (queued cloud access), which decides whether variational loops are viable at all. These facts drive the `Bitstrings` and `Estimate` types, shot budgets and the `QCL-W301` warning.

### A.2 Why explicit placement

Implicit placement (inheritance, hidden scheduler decisions, string-keyed schedules separated from the code) hides exactly the information this language exists to expose, and invites errors such as typos in stage names that fail silently. With `.to()` on every compute value, the deployment can be read directly from the code, every mistake is a compile error, and schedule invariance makes deployment experiments safe. The cost is longer programs, which matters little when agents write much of the code.

### A.3 Why post-processing usually runs on a CPU

RRHO partition functions, Eyring rates, Allen–Dynes, ESOL/GSE and site scoring are microseconds of arithmetic per molecule, less than one GPU kernel launch. The expensive parts are their inputs (Hessians, electron-phonon couplings, BDEs), which are separate primitives. GPUs only pay off for very large batches or when the inputs already live on the GPU, and explicit placement allows that.

### A.4 Differential equations

Standard quantum chemistry converts the Schrödinger equation into linear algebra through basis sets, which is why most primitives are linear algebra. Differential equations still appear explicitly: IRC paths and MD are ODEs, PCM and GB solvation descend from the Poisson and Poisson–Boltzmann equations, and electron dynamics and kinetic networks are time-dependent problems. These share the `propagate` pattern. QPUs have a strong case for quantum dynamics (time evolution under a Hamiltonian, which also underlies phase estimation) and, today, no practical advantage for classical ODEs such as MD or kinetics (decision D13).

### A.5 Why no separate agent protocol

An agent with access to the language and the plan API (diagnostics, explain, estimate, validate, profile) already has everything a tool protocol would provide, with less machinery and no second interface to keep consistent.

### A.6 Why Rust

The core is not number-crunching; it is a concurrent system that moves buffers between devices, tracks in-flight QPU jobs and supervises worker processes. Memory-safety and data-race bugs concentrate in exactly that kind of code. Rust prevents them at compile time, packages cleanly as Python wheels, and exposes a C ABI as easily as C++ does.
