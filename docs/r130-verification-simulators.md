# R130: simulators are for verification only

Date: 2026-10-14

## Deployment contract

A screened result is not a hardware execution. Deployable hybrid algorithms must
consist of classical computation and quantum steps with realizable circuits and
known QPU implementations. Exact classical solvers (FCI, SCI, projected Ritz,
response solvers) remain classical algorithms; diagonalizing a classical matrix
is not a claim that a QPU exposes its state amplitudes.

`ImplementationSpec.execution_role` is `classical`, `qpu`, `emulator`, or
`verification_only`. CPU/GPU numeric implementations default to `classical` and
QPU implementations to `qpu`. Registration rejects simulator or fault-tolerant
projection implementations without an explicit emulator/verification role.
An emulator must name its own target op in `emulates`, and that op must exist in
`chemflow.backends.QPU_CATALOG`. Classification does not certify a numerical
result or replace the existing relation, context, budget, and evidence checks.

## Checker purposes and CF-E140

- `check(..., purpose="screen")` is the backward-compatible default. All
  simulators remain available for verification, exact references, V1a
  certificates, V1b falsification, V1c validity maps, `algorithm.verify`, and
  `diagnose`.
- `purpose="deploy"` walks the dependency closure of every non-reference target.
  It rejects verification-only steps, missing deployment implementations, and
  known device-limit violations with CF-E140. A shared dependency does not become
  exempt merely because it is also used by a reference. Reference-only graphs
  are exempt; separately constructed verification graphs retain screen purpose.
- `purpose="production"` additionally rejects every emulator in that closure.
  Binding a sampler to a QPU is insufficient if its preparation still simulates
  a statevector. Use `prepare_circuit` to construct the circuit classically.
- Unknown purposes are errors rather than silently falling back to screening.

### Contextual preparation

A standalone `prepare_state` exposes amplitudes and is conservatively registered
as `verification_only`, not as a fictitious QPU preparation op. The checker treats
it as an **emulation artifact** only if it prepares an implemented HF/LUCJ circuit,
all its consumers are QPU-backed sampling ops through their `prep` port, and the
state is not itself a non-reference target. Even an otherwise unused amplitude
consumer invalidates deployment. This reconciles the op-level catalog (only
`sample_configurations` today) with context-dependent preparation safety.
Its deployment mapping points to the same QPU as sampling; it is not a second job.
UCCSD preparation has no hardware circuit implementation here and is rejected.
`prepare_circuit` performs only classical circuit/parameter construction, never
`hf_vector` or `lucj_state`, and carries no statevector artifact.

## What can be deployed today

- Classical methods, subject to their ordinary capability/context checks.
- SQD: classical integral generation and circuit parameter selection, circuit
  preparation and computational-basis sampling on **lqcloud MQ02**, then classical
  configuration recovery and selected-space diagonalization.
- The catalog records `sample_configurations → lqcloud_mq02`, **24 qubits** and
  **1–50,000 shots per sampling call**, from the bundled MQ02 calibration and
  pricing snapshots. These are declared limits, not a promise of current device
  availability, compilation success, connectivity, or calibration quality.
- Screening statevector, Aer-noise, and GPU samplers draw computational-basis
  samples of the prepared circuit; none exposes exact energies as a QPU result.
  The existing hardware-circuit/statevector equivalence tests verify HF/LUCJ
  semantics, including bit ordering. Noise adds the declared calibration model.
- For production IR, bind `prepare_state` to `prepare_circuit` and
  `sample_configurations` to the registered `lqcloud_mq02` adapter. Registration,
  hardware authorization, resource budgets, and backend checks still apply.
- The PySCF SQD plugin declares its sampling role (`emulator` or `qpu`), accepts
  the same purpose names, and refuses simulator sampling at production purpose.
  Its deploy/production lqcloud path constructs a circuit without materializing
  a statevector; screen purpose retains the verification artifact.
- Monolithic QM/MM simulator solvers are verification-only, not sampling
  emulators. The QPU QM/MM wrapper is classified `qpu`, but currently has no
  op-level deployment catalog entry and is conservatively refused by deployment
  planning; decompose it into catalog-backed sampling before advertising a plan.
- EWF embedding is classical, including its `solver="sqd"` option: this option
  explicitly uses a uniform **classical** configuration sampler, not a simulated
  quantum circuit. GPU FCI likewise remains classical numerical linear algebra.

## What would make VQE deployable

Implement a QPU Pauli estimator with actual realizable basis rotations, finite-shot
measurement, uncertainty/resource accounting, and a registered hardware sampling
contract. Then compose circuit preparation, that estimator, and classical
optimization. Neither exact nor shots-mode `vqe_statevector` qualifies today:
both still depend on simulator-only estimation. ADAPT also requires realizable
measurement-based gradients/operator selection. `pauli_shot_estimator` has **no
QPU counterpart yet**; an ideal Hamiltonian-eigenbasis projective sampler is not
a substitute.

## Agent and report behavior

`plan.candidates` retains the `vqe` enum to return an explanatory verification-only
error, but only offers SCI/SQD deployment templates. Both `design.submit` and
orchestrator candidate checks use deploy purpose; production `run` checks before
execution or spending. Scripted designers omit VQE/ADAPT. Cards state
“verification-only (statevector); no deployable implementation”, and the designer
prompt requires QPU-realizable or classical candidates.

Candidate summaries and recorded headlines contain `deployable`, `deployment`
(node to QPU implementation), `deployment_reason`, and `classical_nodes`.
A device-limit violation marks a candidate non-deployable with the reason.
The critic cannot escalate an explicitly non-deployable candidate. Decision
reports identify QPU and classical steps and explicitly distinguish emulator
screening from production. These mappings are plans, not automatic submission
or automatic rebinding to hardware.

## Complete implementation role table

This table includes every registry key in all packs, including aliases. Optional
backend and plugin entries follow. A dash means no `emulates` declaration.

| Implementation / registry key | Target op | Backend kind | Execution role | Emulates |
| --- | --- | --- | --- | --- |
| `active_space` | `active_space` | `cpu_numeric` | `classical` | — |
| `adapt_statevector` | `ground_energy` | `quantum_simulator` | `verification_only` | — |
| `ccsd` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `ccsd_t` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `compare` | `compare` | `cpu_numeric` | `classical` | — |
| `component_energy` | `component_energy` | `cpu_numeric` | `classical` | — |
| `conformers` | `conformers` | `cpu_numeric` | `classical` | — |
| `descriptors` | `descriptors` | `cpu_numeric` | `classical` | — |
| `dock` | `dock` | `cpu_numeric` | `classical` | — |
| `electronic_model` | `electronic_model` | `cpu_numeric` | `classical` | — |
| `encode` | `encode` | `cpu_numeric` | `classical` | — |
| `energy_difference` | `energy_difference` | `cpu_numeric` | `classical` | — |
| `ewf_embedding` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `expectation` | `expectation` | `quantum_simulator` | `verification_only` | — |
| `extend_subspace` | `extend_subspace` | `cpu_numeric` | `classical` | — |
| `fci_pyscf` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `filter_ligands` | `filter_ligands` | `cpu_numeric` | `classical` | — |
| `frequencies` | `frequencies` | `cpu_numeric` | `classical` | — |
| `full_subspace` | `full_subspace` | `cpu_numeric` | `classical` | — |
| `geometry` | `geometry` | `cpu_numeric` | `classical` | — |
| `geometry_optimize` | `geometry_optimize` | `cpu_numeric` | `classical` | — |
| `ground_energy` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `hamiltonian` | `hamiltonian` | `cpu_numeric` | `classical` | — |
| `hf` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `ideal_energy_projective_sampler` | `sample` | `quantum_simulator` | `verification_only` | — |
| `impurity_model` | `impurity_model` | `cpu_numeric` | `classical` | — |
| `integrals` | `integrals` | `cpu_numeric` | `classical` | — |
| `interaction_energy` | `interaction_energy` | `cpu_numeric` | `classical` | — |
| `lattice_hf` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `lattice_model` | `lattice_model` | `cpu_numeric` | `classical` | — |
| `lehmann_from_spectra` | `spectral_function` | `cpu_numeric` | `classical` | — |
| `ligand_library` | `ligand_library` | `cpu_numeric` | `classical` | — |
| `ligands_from_names` | `ligands_from_names` | `cpu_numeric` | `classical` | — |
| `map` | `map` | `cpu_numeric` | `classical` | — |
| `mm_system` | `mm_system` | `cpu_numeric` | `classical` | — |
| `molecule` | `molecule` | `cpu_numeric` | `classical` | — |
| `mp2` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `pauli_shot_estimator` | `sample` | `quantum_simulator` | `verification_only` | — |
| `periodic_scf` | `periodic_scf` | `cpu_numeric` | `classical` | — |
| `periodic_system` | `periodic_system` | `cpu_numeric` | `classical` | — |
| `potential_energy` | `potential_energy` | `cpu_numeric` | `classical` | — |
| `prepare` | `prepare` | `quantum_simulator` | `verification_only` | — |
| `prepare_circuit` | `prepare_state` | `cpu_numeric` | `classical` | — |
| `prepare_state` | `prepare_state` | `quantum_simulator` | `verification_only` | — |
| `qmmm_casci` | `qmmm_energy` | `cpu_numeric` | `classical` | — |
| `qmmm_energy` | `qmmm_energy` | `cpu_numeric` | `classical` | — |
| `qmmm_gpu_fci` | `qmmm_energy` | `gpu_numeric` | `classical` | — |
| `qmmm_rhf` | `qmmm_energy` | `cpu_numeric` | `classical` | — |
| `qmmm_sqd_aer` | `qmmm_energy` | `quantum_simulator` | `verification_only` | — |
| `qmmm_sqd_gpu` | `qmmm_energy` | `quantum_simulator` | `verification_only` | — |
| `qmmm_sqd_qpu` | `qmmm_energy` | `qpu` | `qpu` | — |
| `qmmm_sqd_statevector` | `qmmm_energy` | `quantum_simulator` | `verification_only` | — |
| `rayleigh_ritz` | `rayleigh_ritz` | `cpu_numeric` | `classical` | — |
| `reaction_ccsd` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `reaction_ccsd_t` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `reaction_energy` | `reaction_energy` | `cpu_numeric` | `classical` | — |
| `reaction_hf` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `reaction_mp2` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `sample` | `sample` | `quantum_simulator` | `verification_only` | — |
| `sample_configurations` | `sample_configurations` | `quantum_simulator` | `emulator` | `sample_configurations` |
| `scf` | `scf` | `cpu_numeric` | `classical` | — |
| `sci_pyscf` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `search_routes` | `search_routes` | `cpu_numeric` | `classical` | — |
| `sector` | `sector` | `cpu_numeric` | `classical` | — |
| `sector_diag_numpy` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `spectral_exact` | `spectral_function` | `cpu_numeric` | `classical` | — |
| `sqd_sector` | `ground_energy` | `cpu_numeric` | `classical` | — |
| `subspace_from_samples` | `subspace_from_samples` | `cpu_numeric` | `classical` | — |
| `thermochemistry` | `thermochemistry` | `cpu_numeric` | `classical` | — |
| `vqe_statevector` | `ground_energy` | `quantum_simulator` | `verification_only` | — |

### Optional backends, plugins, and projections

| Implementation | Backend / role | Emulates | Deployment note |
| --- | --- | --- | --- |
| `aer_noisy_mq02` (or registered alias) | quantum_simulator / emulator | sample_configurations | MQ02 circuit sampling with calibration noise; verification screening only |
| `statevector_cupy` | quantum_simulator / emulator | sample_configurations | GPU circuit-state sampling; verification screening only |
| `fci_cupy` | gpu_numeric / classical | — | Classical FCI eigensolver |
| `lqcloud_mq02` | qpu / qpu | — | MQ02 sampling; limits in catalog |
| PySCF `SQDSolver`, statevector / aer_noisy / gpu samplers | quantum_simulator / emulator (sampling stage) | sample_configurations | Classical SQD postprocessing; production purpose refused |
| PySCF `SQDSolver`, lqcloud sampler | qpu / qpu (sampling stage) | — | Circuit-only preparation, QPU sampling, classical SQD postprocessing |
| PySCF `GPUFCISolver` | gpu_numeric / classical | — | No quantum simulation |
| fault_tolerant_projection | no implementations registered | — | Future registrations must explicitly declare emulator or verification_only; emulator requires a catalog-backed op |

`sample` aliases `pauli_shot_estimator`; `ground_energy` aliases the classical
sector diagonalization implementation. `prepare_state` has the contextual
exception described above, without changing the registry's conservative role.

## Tests changed

- `tests/test_r114_protocol.py`: nullable candidate-parameter equivalence now
  covers deployable SCI/SQD templates, rather than requesting forbidden VQE.
- `tests/test_g5_answers.py`: fake-designer selection uses low-shot SQD instead of
  VQE; the mixed passing/failing selection test explicitly injects a low-shot SQD
  candidate to preserve its original assertion after scripted VQE removal.
- `tests/test_v1c_domains.py`: VQE validity coverage remains intact, constructing
  a verification graph directly instead of calling the deployment-template API.
- Added `tests/test_r130_deployment.py`: role registration/defaults, device limits,
  contextual preparation, non-sampling consumers, production QPU bindings,
  simulator-free circuit preparation, PySCF production refusal, and rejection of
  QPU-labelled samples from an explicitly registered emulator.
- `tests/test_r121_fallback.py`: checker test double accepts and verifies the new
  deploy purpose while preserving warning/error severity assertions.
- `tests/test_r1_surfaces.py`: production now binds circuit-only preparation;
  scale is no longer merely informational, and the fake-QPU execution assertions
  remain intact.
- `tests/test_suite_full_parallel.py`: scripted parallel selection now verifies
  deployable passing candidates, not a failing VQE candidate. Mixed positive and
  negative gate outcomes remain covered explicitly in `test_g5_answers.py`.

### Incompatible historical reviewer assertion

`tests/test_review_m5a.py::test_samples_source_must_match_the_binding[quantum_simulator-qpu]`
registers a quantum simulator without an execution role outside its expected
exception block. R130 explicitly requires registration to reject this, so the
old test fails earlier than its expected runtime CF-E060. Both requirements
cannot pass simultaneously without changing that fixture. Its intended
provenance invariant remains covered by the explicit-emulator regression above.
The reviewer file is left unchanged; no skip, xfail, name-based exception, or
weakened registration rule is introduced.

No `tests/test_review_*.py` files were modified. No new dependencies or live
hardware/LLM calls are needed for this change.
