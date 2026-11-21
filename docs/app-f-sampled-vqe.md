# APP F: Pauli measurement and deployable sampled VQE

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

The existing molecular `ground_energy` operator and molecular target schema gain
the `vqe_sampled` refinement; no new target kind is introduced. The opt-in `quantum`
domain pack (included by `load_registry(pack_names())`) registers `vqe_shots_ideal`,
`vqe_aer_mq02`, and `vqe_lqcloud`, preserving the legacy molecular-only registry.
The first two are
ground-energy emulators; only the last is production hardware. `plan.candidates`
accepts `vqe_sampled` with ideal screening and still refuses statevector `vqe`.

Pauli words use qubit zero first; count strings use qubit zero last. Sorted greedy
QWC groups exclude the identity. X rotates with H and Y with S-dagger then H.
Energy uncertainty includes within-group covariance; independent group variances
add. Tensor readout correction uses the dual inverse confusion matrices, including
the asymmetric offset, without clipping quasi-probabilities. `allocate_shots`
uses deterministic largest remainders with a one-shot minimum per group.

UCCSD uses the existing ordered excitation pool's Jordan–Wigner exponentials;
LUCJ reuses the M5a circuit builder. Both feed native compilation for noisy/hardware
execution. COBYLA and seeded SPSA reserve an evaluation for independent final
measurement. `max_evaluations` includes that final measurement. Each evaluation
submits one job per group. Backend injection through `register_sampled_vqe` keeps
tests offline; LqcloudBackend owns prediction, durable records, retries and
reconciliation. Evaluation indices distinguish repeated circuits.

Forecast shots are `max_evaluations * n_groups * shots_per_group`; unknown groups
remain unknown rather than reporting zero. Supply `n_groups` for preflight budgets.
MQ02 permits at most 24 qubits and 1–50,000 shots per group/job.

Each binding has a V1 implementation axiom denoting the sampled estimator of
the optimized circuit expectation, proved by the existing `stochastic_estimate`
lemma. This is neither a ground-state accuracy guarantee nor a variational bound.
Finite sampling, optimization error, ansatz restrictions, device noise and
calibration error are separate limitations. One-shot groups cannot identify
variance; their plug-in standard error is zero, not a confidence guarantee.
Readout correction is available in the Pauli API, not enabled implicitly in VQE.
Ideal simulation scales exponentially; native UCCSD circuits may be too deep for
useful noisy results even when width and shot limits pass.

References: Jordan and Wigner (1928), fermion-to-spin transformation; Peruzzo et al.
(2014), Nature Communications 5, 4213, variational eigensolver; Kandala et al.
(2017), Nature 549, 242–246, noisy variational optimization; the repository's M5a
MQ02 calibration and backend contracts. No network evidence or live QPU run was used.
