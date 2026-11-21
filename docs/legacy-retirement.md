# Retiring the earlier ChemFlow engine

On 2026-11-21 ChemFlow's earlier execution engine was removed; QCL is the only engine.
Signed tasks compile into QCL screens (`docs/spec/qcl-design.md` §8.1a, D23–D25), and the
numerical kernels that engine wrapped are plain functions called by QCL's worker
implementations.

## What was removed

| Removed | Replaced by |
| --- | --- |
| `chemflow.ir` (IR programs, builder, renderer) | QCL workflows and compiled plans (`qcl.workflow`, `plan.explain()`) |
| `chemflow.registry`, `chemflow.ops.molecular`, domain-pack `register()` functions | `qcl-ir/primitives.toml` and `@implements` handlers in `qcl._impls` |
| `chemflow.check` (static checker, deployment table) | `qcl-check` diagnostics; deployability in the screen report |
| `chemflow.runtime.executor`, `runtime.task`, `runtime.evidence` | the QCL worker; `chemflow.runtime` keeps `ArtifactStore`, `BudgetLedger`, `ExecutionResult` |
| `chemflow.verify` (relation certificates, formal kernel, falsification) | compiler-inferred relations (`plan.relations()`) with per-output certificates; the Lean sources stay in `formal/` |
| `chemflow.cards` (method cards) | the screen's candidate templates (`chemflow.agent.screen.CARDS`) and `chemflow.empirical` model cards |
| `chemflow.design` (algorithm design loop) | agent-authored QCL candidate programs (`--screen --candidate-program`) |
| `chemflow.plan_view` (`chemflow plan show`) | `plan.explain()`, `plan.schedule()` and the run's `qcl-screen.json`/`qcl-profile.json` |
| `chemflow.agent.orchestrator`, `tools`, `families`, `report`, `costs`, `replay` | `chemflow.agent.qcl_run` (driver), `screen` (builder), `qcl_tools` (typed role tools), `validity`, `report_lint` |
| `--engine auto|qcl|legacy`, `CHEMFLOW_ENGINE` | removed: every policy runs QCL screens |
| MCP profiles `planner`, `modeler`, `designer`, `critic`, `analyst`, `full`; `--packs` | the `qcl` profile (`task.get`, `methods.list`, `diagnose.model`, `screen.review`, `screen.run`, `results.query`) |
| `chemflow bench` on the legacy executor | `chemflow.bench`: QCL screens and primitives against the same independent oracles |

The worker start-up that made small QCL tasks slower than the in-process executor is
gone for CPU plans: they fork from a pre-warmed worker server (`qcl._zygote`).

## Where the legacy operations went

| Legacy operation or refinement | QCL |
| --- | --- |
| `scf`, `hamiltonian`, `active_space`, `sector` | `qcl.scf.*`, `qcl.active_space(...).hamiltonian/.sector` |
| `ground_energy` with `hf`, `mp2`, `ccsd`, `ccsd_t` | `qcl.classical.solve(H, sector, method=)` |
| `fci`, `sci`, `sqd`, `vqe_sampled`, `dft` | `qcl.fci.solve`, `qcl.sci.solve`, `qcl.sqd.diagonalize`, `qcl.vqe.sampled`, `qcl.dft.rks/uks` |
| `ewf` | `qcl.embedding.ewf(H, sector, bath=, solver=)` |
| `encode`, `prepare`, `expectation`, `sample`, `sample_configurations` | `qcl.qubit_hamiltonian`, `qcl.kernels.*`, `qcl.estimate`, `qcl.sample` |
| `energy_difference`, `compare`, `map` | estimate arithmetic, `qcl.compare`, `qcl.map` |
| `reaction_energy`, `geometry_optimize`, `frequencies`, `thermochemistry` | composite screens; `qcl.optimize`, `qcl.frequencies`, `qcl.thermo.rrho` |
| `constrained_optimize`, `irc`, `connected_ts`, `rate_constant` | `qcl.optimize(fixed=)`, `qcl.irc`, `qcl.engines.sella.saddle`, `qcl.thermo.eyring` |
| `bond_enumeration`, `homolysis`, `site_reactivity`, `conformer_search` | `Molecule.bonds()`, bond-dissociation screens, `qcl.metabolism.sites`, `qcl.engines.rdkit.*` |
| solvation, frontier orbitals, IP/EA | `qcl.dft`/`qcl.scf` with `qcl.PCM`/`qcl.SMD`, `scf.homo/lumo`, charge-ladder screens |
| `logs_esol`, `logs_gse`, `metabolic_sites`, `mmgbsa` | `qcl.solubility.esol/gse`, `qcl.metabolism.sites`, `qcl.binding.mmgbsa` |
| `ligand_library`, `filter_ligands`, `descriptors` | `qcl.drug.filter`, `qcl.drug.descriptors` |
| `mm_system` + `qmmm_energy` (RHF, CASCI/FCI) | `qcl.qmmm.energy` |
| `search_routes` | `qcl.synthesis.routes` |
| `lattice_model`, `periodic_scf`, pairing correlations, `allen_dynes_tc` | `qcl.models.hubbard`, `qcl.periodic.scf`, lattice screens, `qcl.superconductivity.allen_dynes` |
| `impurity_model`, `spectral_exact` | `qcl.dynamics.spectral_exact` |

These kernels remain plain functions without a QCL primitive yet: DMRG
(`chemflow.impls.dmrg`), statevector and ADAPT-VQE (`chemflow.impls.vqe`, `impls.adapt`),
NEB paths (`chemflow.domains.paths.neb_path`), MACE potentials
(`chemflow.domains.baselines.potential_energy`), EOM-CCSD IP/EA
(`chemflow.domains.properties.eom_charge_energy`), docking and name resolution
(`chemflow.domains.drug.dock`, `ligands_from_names`), QM fragment binding
(`chemflow.domains.binding.fragment_interaction`), the sampled-subspace Lehmann
primitives of A1a (`chemflow.domains.algorithms`, `dynamics.lehmann_from_spectra`) and the
QM/MM quantum solvers (`chemflow.plugins.pyscf.SQDSolver`, `GPUFCISolver`). Their tests
call them directly.

## Retired tests

Tests that exercised the removed engine were retired, including reviewer contracts
marked "implementers must not edit", with the project owner's approval. Within a file
only the tests that depended on the removed engine were taken out; tests of kernels,
signing, the evaluation harness and the application catalogue were kept unchanged, and
the contracts that still describe the product were ported to QCL.

133 published test files changed: 62 were retired whole and
71 kept their engine-independent tests. 466 test functions were
retired, 174 of them in 42 reviewer-contract files. Local-only test files
followed the same rule.

| Area | Files | Tests retired | QCL coverage |
| --- | --- | --- | --- |
| Algorithm primitives and the design loop | 8 | 60 | Kept kernel tests in `tests/test_a1a_*.py`; the design loop is retired (QCL candidate programs replace it) |
| Application packs A–H | 15 | 68 | The `apps` evaluation suite on QCL (`tests/test_qcl_engine.py`) and the QCL app tests in `qcl/tests/` |
| Orchestrator, legacy tools and the legacy LLM designer protocol (M3–M4, R106–R144) | 45 | 130 | `tests/test_qcl_engine.py` (screens, critic, refusals, LLM designer: submit, decline, invalid and unaffordable candidates), `tests/test_llm_policy_qcl.py`, `tests/test_qcl_mcp.py` |
| Resources, budgets, backends, plan view and deployment (M5, P3, FULL, R99, R130) | 7 | 28 | `qcl/tests/test_budget.py`, `test_schedule.py`, `test_quantum.py`; deployability in `tests/test_qcl_engine.py` |
| Thermochemistry, baselines, names and evaluation (G1–G5) | 6 | 28 | `qcl/tests/test_thermochemistry.py`, `test_gfn2.py`, `tests/test_kernels.py` (xTB and MACE in one process); the evaluation suites on QCL |
| Legacy IR, static checker, gates and budgets | 14 | 39 | QCL compiler and relation checks (`qcl/tests/test_primitives.py`, `test_workflow.py`, `test_relations.py`, `test_budget.py`); kernels in `tests/test_kernels.py` |
| VQE, ADAPT, SCI and SQD through the executor | 19 | 34 | `tests/test_kernels.py` and the kept kernel tests; `qcl/tests/test_sci.py`, `test_quantum.py`, `test_lucj.py` |
| Domain packs (QM/MM, drug, synthesis, reactions, lattices) | 3 | 26 | `tests/test_qcl_domains.py` (`qmmm.energy`, `embedding.ewf`, `drug.*`, `synthesis.routes`, benchmarks) |
| MCP server, plugins and hooks (P1–P2) | 7 | 9 | `tests/test_qcl_mcp.py`, `test_p1_mcp.py`, `test_p2_bridge.py`, `test_p2_build.py`, `test_p1_hooks.py` (ported to the `qcl` profile) |
| `--engine` selection | 1 | 1 | `tests/test_qcl_engine.py::test_the_engine_option_is_gone` |
| Relation certificates, falsification, validity and formal lemmas | 8 | 43 | `qcl/tests/test_relations.py`, certificates and validity maps in `tests/test_qcl_engine.py`; Lean sources stay in `formal/` |
| **Total** | **133** | **466** | |

New QCL coverage: `tests/test_qcl_domains.py` (QM/MM, EWF, ligand screens, synthesis and
the benchmarks), `tests/test_kernels.py` (spin-targeted FCI, SQD/SCI spin completion,
ADAPT-VQE, open-shell controls, finite shots), `tests/test_llm_policy_qcl.py` (prompts,
argument decoding, permissions, terminal tools and turn reminders on the QCL tools) and
`qcl/tests/test_zygote.py` (the warm worker), alongside `tests/test_qcl_engine.py` and
`tests/test_qcl_mcp.py`.

Per file (published test files only):

| File | Reviewer contract | Tests retired | Tests kept | Status | Area |
| --- | --- | --- | --- | --- | --- |
| `tests/test_a1a_primitives.py` |  | 2 | 6 | trimmed | A1 |
| `tests/test_a1b_design.py` |  | 11 | 0 | retired | A1 |
| `tests/test_a1c_accuracy.py` |  | 13 | 0 | retired | A1 |
| `tests/test_a1d_loop.py` |  | 8 | 1 | trimmed | A1 |
| `tests/test_app_a_properties.py` |  | 15 | 1 | trimmed | APP |
| `tests/test_app_b_barrier.py` |  | 1 | 2 | trimmed | APP |
| `tests/test_app_b_paths.py` |  | 2 | 4 | trimmed | APP |
| `tests/test_app_b_scans.py` |  | 4 | 1 | trimmed | APP |
| `tests/test_app_c_reactivity.py` |  | 7 | 6 | trimmed | APP |
| `tests/test_app_d_solubility.py` |  | 6 | 5 | trimmed | APP |
| `tests/test_app_e_binding.py` |  | 5 | 6 | trimmed | APP |
| `tests/test_app_f_measurement.py` |  | 4 | 2 | trimmed | APP |
| `tests/test_app_g_lattice.py` |  | 8 | 2 | trimmed | APP |
| `tests/test_app_h_catalogue.py` |  | 1 | 5 | trimmed | APP |
| `tests/test_app_h_interfaces.py` |  | 2 | 5 | trimmed | APP |
| `tests/test_critic_answer.py` |  | 1 | 5 | trimmed | Agent |
| `tests/test_full1_budgets.py` |  | 5 | 0 | retired | Runtime |
| `tests/test_full1_parameters_shots_noise.py` |  | 2 | 1 | trimmed | Runtime |
| `tests/test_full1_reports.py` |  | 4 | 0 | retired | Runtime |
| `tests/test_full2_interfaces.py` |  | 6 | 1 | trimmed | Runtime |
| `tests/test_g1_thermochemistry.py` |  | 5 | 5 | trimmed | G |
| `tests/test_g3_names.py` |  | 5 | 6 | trimmed | G |
| `tests/test_g5_task_accuracy.py` |  | 1 | 6 | trimmed | G |
| `tests/test_glm_policy.py` |  | 1 | 4 | trimmed | Agent |
| `tests/test_kernel.py` |  | 5 | 3 | trimmed | M0–M1 |
| `tests/test_m02_budget.py` |  | 2 | 0 | retired | M0–M1 |
| `tests/test_m02_gates.py` |  | 2 | 0 | retired | M0–M1 |
| `tests/test_m02_model.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m02_relations.py` |  | 1 | 1 | trimmed | M0–M1 |
| `tests/test_m02_spin_diagnostics.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m03_active_diagnostics.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m03_cpu_budget.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m03_exact_reference.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m03_stochastic_gate.py` |  | 1 | 0 | retired | M0–M1 |
| `tests/test_m2a1_adapt_spin.py` |  | 1 | 0 | retired | M2 |
| `tests/test_m2a_adapt.py` |  | 1 | 0 | retired | M2 |
| `tests/test_m2a_cards.py` |  | 2 | 0 | retired | M2 |
| `tests/test_m2a_review_fixes.py` |  | 2 | 0 | retired | M2 |
| `tests/test_m2a_shots_relations.py` |  | 2 | 0 | retired | M2 |
| `tests/test_m2a_uccsd_vqe.py` |  | 1 | 1 | trimmed | M2 |
| `tests/test_m2b1_evidence.py` |  | 1 | 0 | retired | M2 |
| `tests/test_m2b1_pipeline.py` |  | 1 | 0 | retired | M2 |
| `tests/test_m2b4_consistency.py` |  | 3 | 0 | retired | M2 |
| `tests/test_m2b4_fci_roots.py` |  | 1 | 0 | retired | M2 |
| `tests/test_m2b_cards.py` |  | 2 | 0 | retired | M2 |
| `tests/test_m2b_sampling.py` |  | 1 | 1 | trimmed | M2 |
| `tests/test_m2b_sci.py` |  | 1 | 1 | trimmed | M2 |
| `tests/test_m2b_sqd.py` |  | 2 | 4 | trimmed | M2 |
| `tests/test_m3a_example.py` |  | 1 | 0 | retired | Agent |
| `tests/test_m3a_fixes.py` |  | 1 | 3 | trimmed | Agent |
| `tests/test_m3a_r42.py` |  | 1 | 0 | retired | Agent |
| `tests/test_m3a_report.py` |  | 1 | 0 | retired | Agent |
| `tests/test_m3a_task.py` |  | 2 | 1 | trimmed | Agent |
| `tests/test_m3a_tools.py` |  | 2 | 0 | retired | Agent |
| `tests/test_m3b_costs.py` |  | 2 | 0 | retired | Agent |
| `tests/test_m3b_llm.py` |  | 3 | 0 | retired | Agent |
| `tests/test_m3b_provenance.py` |  | 1 | 0 | retired | Agent |
| `tests/test_m4a_orchestrator.py` |  | 2 | 0 | retired | Agent |
| `tests/test_m4a_schemas.py` |  | 1 | 0 | retired | Agent |
| `tests/test_m6a_domains.py` |  | 11 | 2 | trimmed | M6 |
| `tests/test_m6b_synthesis.py` |  | 3 | 1 | trimmed | M6 |
| `tests/test_p1_1_mcp_schemas.py` |  | 1 | 0 | retired | P |
| `tests/test_p1_hooks.py` |  | 1 | 3 | trimmed | P |
| `tests/test_p1_mcp.py` |  | 2 | 0 | rewritten for QCL | P |
| `tests/test_p1_plugin.py` |  | 1 | 0 | retired | P |
| `tests/test_p2_conformance.py` |  | 1 | 3 | trimmed | P |
| `tests/test_physics.py` |  | 2 | 2 | trimmed | M0–M1 |
| `tests/test_qcl_engine.py` |  | 1 | 9 | trimmed | `--engine` |
| `tests/test_r106_loop.py` |  | 6 | 0 | retired | Agent |
| `tests/test_r106_orchestrator.py` |  | 1 | 0 | retired | Agent |
| `tests/test_r107_open_shell.py` |  | 4 | 0 | retired | Agent |
| `tests/test_r109_lattice_scf.py` |  | 2 | 1 | trimmed | Agent |
| `tests/test_r110_tools.py` |  | 2 | 0 | retired | Agent |
| `tests/test_r113_failures.py` |  | 1 | 5 | trimmed | Agent |
| `tests/test_r114_protocol.py` |  | 9 | 1 | trimmed | Agent |
| `tests/test_r119_modeler.py` |  | 3 | 2 | trimmed | Agent |
| `tests/test_r120_submission.py` |  | 1 | 4 | trimmed | Agent |
| `tests/test_r121_fallback.py` |  | 3 | 3 | trimmed | Agent |
| `tests/test_r125_decisions.py` |  | 5 | 0 | retired | Agent |
| `tests/test_r126_regressions.py` |  | 6 | 2 | trimmed | Agent |
| `tests/test_r132_verification.py` |  | 3 | 1 | trimmed | Agent |
| `tests/test_r135_families.py` |  | 7 | 0 | retired | Agent |
| `tests/test_r135_parameters.py` |  | 5 | 0 | retired | Agent |
| `tests/test_r137_candidates.py` |  | 2 | 2 | trimmed | Agent |
| `tests/test_r99_budget.py` |  | 3 | 3 | trimmed | Runtime |
| `tests/test_review_a1a.py` | yes | 7 | 0 | retired | A1 |
| `tests/test_review_a1b.py` | yes | 7 | 0 | retired | A1 |
| `tests/test_review_a1c.py` | yes | 6 | 0 | retired | A1 |
| `tests/test_review_a1d.py` | yes | 6 | 0 | retired | A1 |
| `tests/test_review_app_c.py` | yes | 5 | 2 | trimmed | APP |
| `tests/test_review_app_d.py` | yes | 2 | 2 | trimmed | APP |
| `tests/test_review_app_f.py` | yes | 5 | 0 | retired | APP |
| `tests/test_review_app_h.py` |  | 1 | 4 | trimmed | APP |
| `tests/test_review_full_1.py` | yes | 5 | 1 | trimmed | Runtime |
| `tests/test_review_g1.py` |  | 4 | 0 | retired | G |
| `tests/test_review_g2.py` | yes | 8 | 0 | retired | G |
| `tests/test_review_g3.py` | yes | 5 | 0 | retired | G |
| `tests/test_review_m0.py` | yes | 9 | 1 | trimmed | M0–M1 |
| `tests/test_review_m0_2.py` | yes | 7 | 2 | trimmed | M0–M1 |
| `tests/test_review_m0_3.py` | yes | 5 | 0 | retired | M0–M1 |
| `tests/test_review_m2a.py` | yes | 2 | 1 | trimmed | M2 |
| `tests/test_review_m2b.py` | yes | 4 | 2 | trimmed | M2 |
| `tests/test_review_m2b_2.py` | yes | 3 | 0 | retired | M2 |
| `tests/test_review_m2b_3.py` | yes | 1 | 0 | retired | M2 |
| `tests/test_review_m2b_4.py` | yes | 3 | 0 | retired | M2 |
| `tests/test_review_m3a.py` | yes | 4 | 0 | retired | Agent |
| `tests/test_review_m4a.py` | yes | 3 | 2 | trimmed | Agent |
| `tests/test_review_m5b.py` | yes | 3 | 2 | trimmed | Runtime |
| `tests/test_review_m6a.py` | yes | 12 | 2 | trimmed | M6 |
| `tests/test_review_p1.py` | yes | 2 | 2 | trimmed | P |
| `tests/test_review_p2.py` | yes | 1 | 4 | trimmed | P |
| `tests/test_review_r106.py` | yes | 2 | 0 | retired | Agent |
| `tests/test_review_r107_111.py` | yes | 3 | 1 | trimmed | Agent |
| `tests/test_review_r112.py` | yes | 2 | 1 | trimmed | Agent |
| `tests/test_review_r114_118.py` | yes | 10 | 2 | trimmed | Agent |
| `tests/test_review_r119.py` | yes | 3 | 0 | retired | Agent |
| `tests/test_review_r120.py` | yes | 2 | 0 | retired | Agent |
| `tests/test_review_r121_122.py` | yes | 3 | 1 | trimmed | Agent |
| `tests/test_review_r123_124.py` | yes | 2 | 0 | retired | Agent |
| `tests/test_review_r125.py` | yes | 1 | 1 | trimmed | Agent |
| `tests/test_review_r126_128.py` | yes | 4 | 4 | trimmed | Agent |
| `tests/test_review_r130.py` | yes | 4 | 2 | trimmed | Agent |
| `tests/test_review_r132_134.py` | yes | 3 | 0 | retired | Agent |
| `tests/test_review_r135_136.py` | yes | 3 | 0 | retired | Agent |
| `tests/test_review_r137_139.py` | yes | 2 | 0 | retired | Agent |
| `tests/test_review_v1a.py` | yes | 5 | 2 | trimmed | V1 |
| `tests/test_review_v1b.py` | yes | 3 | 1 | trimmed | V1 |
| `tests/test_review_v1c.py` | yes | 4 | 2 | trimmed | V1 |
| `tests/test_review_v1d.py` | yes | 3 | 2 | trimmed | V1 |
| `tests/test_v1a_certificates.py` |  | 17 | 1 | trimmed | V1 |
| `tests/test_v1b_falsification.py` |  | 3 | 0 | retired | V1 |
| `tests/test_v1c_domains.py` |  | 5 | 2 | trimmed | V1 |
| `tests/test_v1d_formal_integrity.py` |  | 3 | 0 | retired | V1 |
