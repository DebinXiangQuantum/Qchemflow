"""Finite-shot circuit VQE; hardware submissions retain backend reconciliation records."""

from functools import partial
from math import comb

import numpy as np
from scipy.optimize import minimize

from chemflow.diagnostics import AdapterError
from chemflow.impls.jw import pauli_terms
from chemflow.impls.lucj import LucjParams
from chemflow.quantum.pauli import estimate, qwc_groups
from chemflow.registry import ImplementationSpec, RefinementRule
from chemflow.relations import TargetRelation
from chemflow.runtime.result import ExecutionResult
from chemflow.types import CircuitSpec, StatePrep


def resources(max_evaluations=60, shots_per_group=1024, n_groups=None,
              sampler="ideal", backend=None, **kwargs):
    shots = None if n_groups is None else max_evaluations * n_groups * shots_per_group
    seconds = 0
    if sampler == "lqcloud":
        if backend is None:
            from chemflow.backends.lqcloud import LqcloudBackend

            backend = LqcloudBackend()
        prediction = backend.prediction(shots=shots_per_group)
        seconds = (None if n_groups is None or prediction.get("qpu_s") is None else
                   max_evaluations * n_groups * prediction["qpu_s"])
    return {"qpu_shots": shots if sampler == "lqcloud" else 0,
            "sim_shots": 0 if sampler == "lqcloud" else shots,
            "qpu_s": seconds, "gpu_s": 0}


def sampled_vqe(store, hamiltonian, sector, ansatz="uccsd", optimizer="cobyla",
                max_evaluations=60, shots_per_group=1024, seed=0, sampler=None,
                bound_sampler="ideal", backend=None, **kwargs):
    from chemflow.backends.circuits import build_circuit, compile_native

    if sampler is not None and sampler != bound_sampler:
        raise ValueError("Sampler conflicts with binding")
    if ansatz not in ("uccsd", "lucj") or optimizer not in ("cobyla", "spsa"):
        raise ValueError("Unsupported ansatz or optimizer")
    if (type(max_evaluations) is not int or max_evaluations < 2 or
            type(shots_per_group) is not int or not 1 <= shots_per_group <= 50000):
        raise ValueError("Invalid evaluation or per-job shot budget")
    physical = hamiltonian.hamiltonian
    selected = hamiltonian.sector
    if any(sector.get(key) != getattr(selected, key) for key in ("n_orb", "n_alpha", "n_beta")):
        raise ValueError("Sector differs from Hamiltonian")
    if 2 * selected.n_orb > 24:
        raise ValueError("MQ02 supports at most 24 qubits")
    terms = pauli_terms(physical, store)
    groups = qwc_groups(terms)
    if kwargs.get("n_groups", len(groups)) != len(groups):
        raise AdapterError("CF-E080", None, "n_groups differs from Hamiltonian grouping",
                           "Use the deterministic QWC group count", nothing_consumed=True)
    alpha, beta = selected.n_alpha, selected.n_beta
    virtual_alpha, virtual_beta = selected.n_orb-alpha, selected.n_orb-beta
    size = (alpha*virtual_alpha + beta*virtual_beta +
            comb(alpha, 2)*comb(virtual_alpha, 2) + comb(beta, 2)*comb(virtual_beta, 2) +
            alpha*beta*virtual_alpha*virtual_beta if ansatz == "uccsd" else
            LucjParams.zeros(selected.n_orb).size)
    rng = np.random.default_rng(seed)
    parameters = np.zeros(size) if ansatz == "uccsd" else rng.normal(0, .12, size)
    evaluations, job_ids = 0, []
    cost = {"qpu_shots": 0, "sim_shots": 0, "qpu_s": 0, "gpu_s": 0}
    if bound_sampler == "lqcloud" and backend is None:
        from chemflow.backends.lqcloud import LqcloudBackend

        backend = LqcloudBackend()
        backend.identifier = "vqe_lqcloud"
    if bound_sampler == "aer_mq02":
        from qiskit_aer import AerSimulator

        from chemflow.backends.aer_noise import noise_model
        from chemflow.backends.calibration import best_layout, load_calibration

        calibration = load_calibration()
        simulator = AerSimulator(noise_model=noise_model(
            calibration, best_layout(calibration, 2 * selected.n_orb)))

    def measure(values):
        nonlocal evaluations
        counts = []
        for group in groups:
            spec = CircuitSpec(ansatz=ansatz, n_orb=selected.n_orb,
                n_alpha=selected.n_alpha, n_beta=selected.n_beta,
                layers=(tuple(float(value) for value in values),),
                measurement_basis=group["basis"])
            sample_seed = int(rng.integers(0, 2**31))
            if bound_sampler == "lqcloud":
                prep = StatePrep(state_ref="", encoding_id=physical.id,
                    hamiltonian_id=physical.id, sector=selected, circuit=spec,
                    recipe="vqe_sampled")
                try:
                    output = backend.sample(store, prep, shots_per_group, seed=sample_seed,
                                            repetition=evaluations)
                except AdapterError as exc:
                    incurred = dict(exc.incurred_cost)
                    for dimension in ("qpu_shots", "qpu_s"):
                        previous = cost[dimension]
                        current = incurred.get(dimension, 0 if exc.nothing_consumed else None)
                        incurred[dimension] = (None if previous is None or current is None
                                               else previous + current)
                    raise AdapterError("CF-E080", None, str(exc),
                        "Reconcile the recorded group job before retrying",
                        nothing_consumed=exc.nothing_consumed and not job_ids,
                        incurred_cost=incurred, job_ids=(*job_ids, *exc.job_ids)) from exc
                job_ids.extend(output.job_ids)
                for dimension in ("qpu_shots", "qpu_s"):
                    amount = output.cost.get(dimension)
                    cost[dimension] = (None if amount is None or cost[dimension] is None
                                       else cost[dimension] + amount)
                measured = {format(int(key), f"0{2*selected.n_orb}b"): value
                            for key, value in output.value.counts.items()}
            elif bound_sampler == "aer_mq02":
                measured = simulator.run(compile_native(build_circuit(spec)),
                    shots=shots_per_group, seed_simulator=sample_seed).result().get_counts()
            else:
                from qiskit.quantum_info import Statevector

                state = Statevector.from_instruction(build_circuit(spec))
                state.seed(sample_seed)
                measured = state.sample_counts(shots_per_group)
            counts.append(measured)
            if bound_sampler != "lqcloud":
                cost["sim_shots"] += shots_per_group
        evaluations += 1
        return estimate(terms, groups, counts)

    def objective(values):
        return measure(values)["energy"]

    if size and max_evaluations > 2:
        if optimizer == "cobyla":
            class EvaluationLimit(Exception):
                pass

            best = [float("inf"), parameters.copy()]

            def limited(values):
                if evaluations >= max_evaluations - 1:
                    raise EvaluationLimit
                energy = objective(values)
                if energy < best[0]:
                    best[:] = [energy, values.copy()]
                return energy

            try:
                result = minimize(limited, parameters, method="COBYLA",
                    options={"maxiter": max(size+2, max_evaluations-1),
                             "rhobeg": .15, "tol": .002})
                parameters = result.x
            except EvaluationLimit:
                parameters = best[1]
        else:
            iteration = 0
            while evaluations + 2 <= max_evaluations - 1:
                delta = rng.choice([-1., 1.], size=size)
                radius = .1 / (iteration + 1)**.101
                gradient = (objective(parameters + radius*delta) -
                            objective(parameters - radius*delta)) / (2*radius) * delta
                parameters -= .1 / (iteration + 1)**.602 * gradient
                iteration += 1
    final = measure(parameters)
    diagnostics = {**final, "evaluations": evaluations, "groups": len(groups),
                   "parameters": parameters.tolist(), "sampler": bound_sampler,
                   "evaluation": "shots"}
    return ExecutionResult((final["energy"], diagnostics, {"stat_err": final["stat_err"]}),
        cost=cost, job_ids=tuple(job_ids), device=bound_sampler)


def register_sampled_vqe(registry, backend=None):
    registry.register(RefinementRule(target="ground_energy", template_id="vqe_sampled",
                                     relation=TargetRelation.stochastic_estimate))
    for binding, sampler in (("vqe_shots_ideal", "ideal"), ("vqe_aer_mq02", "aer_mq02"),
                             ("vqe_lqcloud", "lqcloud")):
        hardware = sampler == "lqcloud"
        registry.register(ImplementationSpec(id=binding, version="1", target="ground_energy",
            implements="vqe_sampled", backend_kind="qpu" if hardware else "quantum_simulator",
            execution_role="qpu" if hardware else "emulator",
            emulates=None if hardware else "ground_energy", effects="random",
            adapter=partial(sampled_vqe, bound_sampler=sampler, backend=backend),
            params=frozenset({"ansatz", "optimizer", "max_evaluations", "shots_per_group",
                              "seed", "sampler", "s", "total_spin", "n_groups"}),
            resource_model=partial(resources, sampler=sampler, backend=backend)))
