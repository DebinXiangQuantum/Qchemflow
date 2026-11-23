"""Qubit operators, QPU/simulator sampling and estimation, VQE and SQD."""

from qcl._impls import implements
from qcl._impls._common import (
    Any,
    _custom_statevector,
    _gpu_index,
)


@implements("qubit_hamiltonian")
def _op_qubit_hamiltonian(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    from chemflow.impls.jw import pauli_terms
    from chemflow.types import QubitEncoding

    device = node["device"]
    if device["kind"] != "cpu" or node["params"]["mapping"] != "jordan_wigner":
        raise ValueError("Jordan-Wigner mapping currently requires a CPU")
    data = inputs[0]["data"]
    encoding = QubitEncoding(n_qubits=2 * data.sector.n_orb,
                             hamiltonian_id=data.hamiltonian.id)
    values[identifier] = {**inputs[0],
                          "terms": pauli_terms(data.hamiltonian, store),
                          "n_qubits": encoding.n_qubits,
                          "encoding_id": encoding.id,
                          "provenance": {
                              "question": inputs[0]["question"],
                              "method": {"family": "Jordan-Wigner",
                                         "mapping": "jordan_wigner",
                                         "basis": data.model.basis},
                              "implementation": "qubit_hamiltonian/chemflow-jw",
                              "device": device["id"],
                              "versions": inputs[0]["versions"],
                          }}


# lqcloud backends with a circuit sampling path; MQ02 is a chain, QZ03 a grid.
SAMPLING_BACKENDS = ("MQ02", "QZ03")


@implements("sample")
def _op_sample(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    request = ctx.request
    qpu_backend_factory = ctx.qpu_backend_factory
    from chemflow.ids import content_hash
    from chemflow.impls.prepare import prepare_sector_state
    from chemflow.types import CircuitSpec, Samples, StatePrep

    device = node["device"]
    if device["kind"] != "qpu":
        raise ValueError("sample requires a QPU or QPU simulator")
    custom = inputs[0]["ansatz"] == "custom"
    data = None if custom else inputs[0]["data"]
    versions = dict(inputs[0]["versions"])
    if custom:
        import qiskit

        versions["qiskit"] = qiskit.__version__
    shots = node["params"]["shots"]
    seed = node["params"]["seed"]
    simulator_host = device["simulator_host"]
    qpu_diagnostics = None
    if custom:
        circuit = CircuitSpec(ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                              n_qubits=inputs[0]["n_qubits"],
                              n_clbits=inputs[0]["n_clbits"],
                              gates=tuple(inputs[0]["gates"]),
                              layers=(tuple(inputs[0]["params"]),))
        identity = content_hash(circuit.model_dump(mode="json"))
        prep = StatePrep(state_ref=identity, encoding_id=identity,
                         hamiltonian_id=identity, recipe="custom_circuit",
                         circuit=circuit)
    else:
        prep = prepare_sector_state(data, data.sector.model_dump(), store,
                                    ansatz=inputs[0]["ansatz"],
                                    params=inputs[0].get("params"),
                                    materialize=simulator_host is not None,
                                    final_rotation=inputs[0].get("final_rotation", True))
    if custom and simulator_host is not None and simulator_host.startswith("gpu"):
        import cupy as cp
        import numpy as np

        versions["cupy"] = cp.__version__
        with cp.cuda.Device(_gpu_index(simulator_host)):
            state = _custom_statevector(circuit, cp)
            probabilities = cp.asnumpy(cp.abs(state) ** 2)
            cp.cuda.Device().synchronize()
        drawn = np.random.default_rng(seed).choice(len(probabilities), size=shots,
                                                   p=probabilities)
        keys, frequencies = np.unique(drawn, return_counts=True)
        counts = {str(key): int(count) for key, count in zip(keys, frequencies)}
        samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                          basis="computational", source="quantum_simulator",
                          seed=seed, device=device["id"])
        implementation = "sample/qcl-cupy-statevector"
        resources = {"sim_shots": shots}
    elif custom and simulator_host is not None and circuit.n_clbits:
        import qiskit_aer
        from qiskit_aer import AerSimulator

        from chemflow.backends.circuits import build_circuit

        program = build_circuit(circuit)
        program.measure(range(circuit.n_qubits),
                        range(circuit.n_clbits,
                              circuit.n_clbits + circuit.n_qubits))
        measured = AerSimulator(method="statevector").run(
            program, shots=shots, seed_simulator=seed).result().get_counts()
        counts: dict[str, int] = {}
        for bits, count in measured.items():
            key = str(int(bits.replace(" ", ""), 2) >> circuit.n_clbits)
            counts[key] = counts.get(key, 0) + count
        samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                          basis="computational", source="quantum_simulator",
                          seed=seed, device=device["id"])
        implementation = "sample/qiskit-aer-dynamic"
        versions["qiskit-aer"] = qiskit_aer.__version__
        resources = {"sim_shots": shots}
    elif custom and simulator_host is not None:
        from qiskit.quantum_info import Statevector

        from chemflow.backends.circuits import build_circuit

        state = Statevector.from_instruction(build_circuit(prep.circuit))
        state.seed(seed)
        counts = {str(int(bits, 2)): int(count)
                  for bits, count in state.sample_counts(shots).items()}
        samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                          basis="computational", source="quantum_simulator",
                          seed=seed, device=device["id"])
        implementation = "sample/qiskit-statevector"
        resources = {"sim_shots": shots}
    elif simulator_host is not None and simulator_host.startswith("gpu"):
        from chemflow.backends.gpu import gpu_sample

        computed = gpu_sample(store, prep, shots, seed)
        samples = computed.value
        implementation = "sample/chemflow-cupy-statevector"
        resources = computed.cost
    elif simulator_host is not None:
        from chemflow.impls.statevector import sample_configurations

        samples = sample_configurations(prep, shots, seed, store=store)
        implementation = "sample/chemflow-statevector"
        resources = {"sim_shots": shots}
    elif device["provider"] == "lqcloud" and device["backend"] in SAMPLING_BACKENDS:
        from chemflow.backends.lqcloud import LqcloudBackend, dynamic_decoupling

        policy = request.get("cloud_qpu_policy")
        billing_ceiling = (policy["max_cost"] / policy["forecast_cost"]
                           if policy and policy["forecast_cost"] > 0 else None)
        backend = (qpu_backend_factory(device) if qpu_backend_factory else
                   LqcloudBackend(backend=device["backend"],
                                  max_billing_multiplier=billing_ceiling))
        options = {}
        decoupling = dynamic_decoupling((device.get("options") or {}).get("dd", "none"))
        if decoupling is not None:
            options["dynamic_decoupling"] = decoupling
        if (device.get("options") or {}).get("compile", "search") == "plain":
            options["compile_search"] = False
        if "repetition" in node["params"]:
            options["repetition"] = node["params"]["repetition"]
        computed = backend.sample(store, prep, shots, seed, **options)
        samples = computed.value
        implementation = f"sample/chemflow-lqcloud-{device['backend'].lower()}"
        resources = computed.cost
        billing = computed.diagnostics.get("billing")
        qpu_diagnostics = {key: value for key, value in computed.diagnostics.items()
                           if key != "billing"}
    else:
        raise ValueError("No QPU implementation is registered for this device")
    expected_source = "quantum_simulator" if simulator_host else "qpu"
    if (samples.source != expected_source or samples.shots != shots or
            sum(samples.counts.values()) != shots):
        raise ValueError("Sampling adapter returned inconsistent source or shot counts")
    provenance = {
        "question": {**inputs[0]["question"], "property": "bitstrings"},
        "method": {"family": f"{inputs[0]['ansatz'].upper()} sampling",
                   "basis": data.model.basis if data is not None else "computational",
                   **({"active_electrons": data.active.n_elec,
                       "active_orbitals": data.active.n_orb}
                      if data is not None else
                      {"n_qubits": inputs[0]["n_qubits"]})},
        "implementation": implementation, "device": device["id"],
        "versions": versions, "resources": resources,
        "job_ids": samples.job_ids,
    }
    if simulator_host is None and billing is not None:
        provenance["billing"] = billing
    if simulator_host is None and qpu_diagnostics:
        provenance["qpu_diagnostics"] = qpu_diagnostics
    # Samples of an LUCJ circuit without its final rotation are in the rotated orbital
    # basis; consumers must rotate the Hamiltonian (sqd.diagonalize does).
    rotation = (list(prep.circuit.layers[0]) if prep.circuit is not None and
                not prep.circuit.final_rotation else None)
    if rotation is not None:
        provenance["orbital_basis"] = "LUCJ-rotated (final orbital rotation omitted)"
    values[identifier] = {
        "orbital_rotation": rotation,
        "counts": samples.counts, "shots": samples.shots,
        "source": samples.source, "device": device["id"],
        "encoding_id": samples.encoding_id, "hamiltonian_id": prep.hamiltonian_id,
        "basis": samples.basis, "seed": seed,
        "job_ids": samples.job_ids,
        "parts": [{"device": device["id"], "source": samples.source,
                   "shots": samples.shots, "counts": samples.counts,
                   "job_ids": samples.job_ids}],
        "provenance": provenance,
    }


@implements("estimate.quantum")
def _op_estimate_quantum(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    request = ctx.request
    qpu_backend_factory = ctx.qpu_backend_factory
    from qiskit.quantum_info import Statevector

    from chemflow.backends.circuits import build_circuit
    from chemflow.diagnostics import AdapterError
    from chemflow.impls.prepare import prepare_sector_state
    from chemflow.quantum.pauli import estimate as pauli_estimate
    from chemflow.quantum.pauli import qwc_groups
    from chemflow.types import CircuitSpec, StatePrep

    device = node["device"]
    operator, kernel = inputs
    data = operator["data"]
    groups = qwc_groups(operator["terms"])
    params = node["params"]
    if len(groups) != params["n_groups"]:
        raise ValueError("n_groups differs from Hamiltonian grouping")
    group_indices = params.get("group_indices", list(range(len(groups))))
    selected_groups = [(index, groups[index]) for index in group_indices]
    if kernel["ansatz"] != "custom" and kernel["data"] is not data:
        raise ValueError("Pauli operator and kernel refer to different active spaces")
    simulator = device["simulator_host"] is not None
    backend = None
    if not simulator:
        from chemflow.backends.lqcloud import LqcloudBackend

        policy = request.get("cloud_qpu_policy")
        ceiling = (policy["max_cost"] / policy["forecast_cost"]
                   if policy and policy["forecast_cost"] > 0 else None)
        backend = (qpu_backend_factory(device) if qpu_backend_factory else
                   LqcloudBackend(backend="MQ02",
                                  max_billing_multiplier=ceiling))
    measured = []
    job_ids = []
    resources: dict[str, Any] = {"sim_shots": 0, "qpu_shots": 0, "qpu_s": 0.0}
    versions = dict(operator["versions"])
    for index, group in selected_groups:
        if kernel["ansatz"] == "custom":
            circuit = CircuitSpec(
                ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                n_qubits=kernel["n_qubits"], n_clbits=kernel["n_clbits"],
                gates=tuple(kernel["gates"]),
                layers=(tuple(kernel["params"]),),
                measurement_basis=group["basis"])
            prep = StatePrep(state_ref="", encoding_id=operator["encoding_id"],
                             hamiltonian_id=data.hamiltonian.id,
                             recipe="custom_pauli_estimate", circuit=circuit)
        else:
            prep = prepare_sector_state(data, data.sector.model_dump(), store,
                                        ansatz=kernel["ansatz"],
                                        params=kernel.get("params"),
                                        materialize=False)
            prep = prep.model_copy(update={"circuit": prep.circuit.model_copy(
                update={"measurement_basis": group["basis"]})})
        shot_seed = params["seed"] + index
        if simulator and device["simulator_host"].startswith("gpu"):
            import cupy as cp
            import numpy as np

            versions["cupy"] = cp.__version__
            with cp.cuda.Device(_gpu_index(device["simulator_host"])):
                state = _custom_statevector(circuit, cp)
                probabilities = cp.asnumpy(cp.abs(state) ** 2)
                cp.cuda.Device().synchronize()
            drawn = np.random.default_rng(shot_seed).choice(
                len(probabilities), size=params["shots_per_group"], p=probabilities)
            keys, frequencies = np.unique(drawn, return_counts=True)
            measured.append({format(int(key), f"0{operator['n_qubits']}b"): int(count)
                             for key, count in zip(keys, frequencies)})
            resources["sim_shots"] += params["shots_per_group"]
        elif simulator:
            state = Statevector.from_instruction(build_circuit(prep.circuit))
            state.seed(shot_seed)
            measured.append(dict(state.sample_counts(params["shots_per_group"])))
            resources["sim_shots"] += params["shots_per_group"]
        else:
            try:
                computed = backend.sample(store, prep, params["shots_per_group"],
                                          shot_seed, repetition=index)
            except AdapterError as error:
                incurred = dict(error.incurred_cost)
                for dimension in ("qpu_shots", "qpu_s"):
                    previous = resources[dimension]
                    current = incurred.get(dimension,
                                           0 if error.nothing_consumed else None)
                    incurred[dimension] = (None if previous is None or
                                           current is None else previous + current)
                raise AdapterError(
                    "CF-E080", None, str(error),
                    "Reconcile completed Pauli groups before retrying",
                    nothing_consumed=error.nothing_consumed and not job_ids,
                    incurred_cost=incurred,
                    job_ids=(*job_ids, *error.job_ids)) from error
            samples = computed.value
            if (samples.source != "qpu" or
                    samples.shots != params["shots_per_group"] or
                    sum(samples.counts.values()) != samples.shots):
                raise ValueError("Pauli sampler returned inconsistent shot counts")
            measured.append({format(int(key), f"0{operator['n_qubits']}b"): count
                             for key, count in samples.counts.items()})
            job_ids.extend(samples.job_ids)
            resources["qpu_shots"] += samples.shots
            seconds = computed.cost.get("qpu_s")
            resources["qpu_s"] = (None if seconds is None or
                                  resources["qpu_s"] is None else
                                  resources["qpu_s"] + seconds)
    measured_words = {word for _, group in selected_groups
                      for word in group["terms"]}
    selected_terms = {
        word: coefficient for word, coefficient in operator["terms"].items()
        if word in measured_words or
        (0 in group_indices and not set(word) - {"I"})}
    result = pauli_estimate(selected_terms,
                            [group for _, group in selected_groups], measured)
    values[identifier] = {
        "value": result["energy"], "unit": "hartree",
        "variance": result["stat_err"] ** 2,
        "provenance": {
            "question": operator["question"],
            "method": {"family": "Pauli expectation",
                       "ansatz": kernel["ansatz"],
                       "basis": data.model.basis,
                       "active_electrons": data.active.n_elec,
                       "active_orbitals": data.active.n_orb,
                       "n_groups": len(groups),
                       "group_indices": group_indices},
            "implementation": ("estimate.quantum/qcl-cupy-statevector"
                               if simulator and device["simulator_host"].startswith("gpu")
                               else "estimate.quantum/qiskit-statevector" if simulator
                               else "estimate.quantum/chemflow-lqcloud-mq02"),
            "device": device["id"], "versions": versions,
            "sample_source": "quantum_simulator" if simulator else "qpu",
            "sample_shots": len(group_indices) * params["shots_per_group"],
            "resources": resources, "job_ids": job_ids,
            "per_group_stat_err": result["per_group_stat_err"],
        },
    }


@implements("estimates.merge")
def _op_estimates_merge(ctx, node, identifier, operation, inputs):
    values = ctx.values
    first = inputs[0]
    parts = [item["provenance"] for item in inputs]
    groups = node["params"]["n_groups"]
    assigned = [index for part in parts
                for index in part["method"]["group_indices"]]
    if (sorted(assigned) != list(range(groups)) or
            any(item["unit"] != first["unit"] or
                item["provenance"]["question"] != first["provenance"]["question"] or
                {key: value for key, value in item["provenance"]["method"].items()
                 if key != "group_indices"} !=
                {key: value for key, value in parts[0]["method"].items()
                 if key != "group_indices"}
                for item in inputs[1:])):
        raise ValueError("Cannot merge incompatible Pauli estimates")
    sources = {part["sample_source"] for part in parts}
    resources = {}
    for dimension in ("sim_shots", "qpu_shots", "qpu_s"):
        amounts = [part["resources"][dimension] for part in parts]
        resources[dimension] = (None if any(amount is None for amount in amounts)
                                else sum(amounts))
    values[identifier] = {
        "value": sum(item["value"] for item in inputs),
        "unit": first["unit"],
        "variance": sum(item["variance"] for item in inputs),
        "provenance": {
            "question": parts[0]["question"],
            "method": {key: value for key, value in parts[0]["method"].items()
                       if key != "group_indices"},
            "implementation": "estimates.merge/qcl-host",
            "device": "multi", "versions": parts[0]["versions"],
            "sample_source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "sample_shots": sum(part["sample_shots"] for part in parts),
            "resources": resources,
            "job_ids": [job for part in parts for job in part["job_ids"]],
            "per_group_stat_err": [error for part in parts
                                   for error in part["per_group_stat_err"]],
            "parts": parts,
        },
    }


@implements("vqe.sampled")
def _op_vqe_sampled(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    request = ctx.request
    qpu_backend_factory = ctx.qpu_backend_factory
    from chemflow.backends.lqcloud import LqcloudBackend
    from chemflow.quantum.vqe import sampled_vqe

    device = node["device"]
    if device["kind"] != "qpu":
        raise ValueError("Sampled VQE requires a QPU device")
    if inputs[0]["data"] is not inputs[1]["data"]:
        raise ValueError("VQE operator and sector refer to different active spaces")
    data = inputs[0]["data"]
    params = node["params"]
    simulator = device["simulator_host"] is not None
    backend = None
    if not simulator:
        policy = request.get("cloud_qpu_policy")
        ceiling = (policy["max_cost"] / policy["forecast_cost"]
                   if policy and policy["forecast_cost"] > 0 else None)
        backend = (qpu_backend_factory(device) if qpu_backend_factory else
                   LqcloudBackend(backend="MQ02", max_billing_multiplier=ceiling))
    computed = sampled_vqe(
        store, data, data.sector.model_dump(),
        ansatz=params["ansatz"], optimizer=params["optimizer"],
        max_evaluations=params["max_evaluations"],
        shots_per_group=params["shots_per_group"],
        n_groups=params["n_groups"], seed=params["seed"],
        bound_sampler="ideal" if simulator else "lqcloud", backend=backend)
    energy, diagnostics, uncertainty = computed.value
    actual_shots = (diagnostics["evaluations"] * diagnostics["groups"] *
                    params["shots_per_group"])
    values[identifier] = {
        "value": float(energy), "unit": "hartree",
        "variance": float(uncertainty["stat_err"]) ** 2,
        "provenance": {
            "question": inputs[0]["question"],
            "method": {"family": "sampled VQE", "ansatz": params["ansatz"],
                       "optimizer": params["optimizer"],
                       "basis": data.model.basis,
                       "active_electrons": data.active.n_elec,
                       "active_orbitals": data.active.n_orb},
            "implementation": "vqe.sampled/chemflow-sampled-vqe",
            "device": device["id"], "versions": inputs[0]["versions"],
            "sample_source": "quantum_simulator" if simulator else "qpu",
            "sample_shots": actual_shots,
            "sample_job_ids": computed.job_ids,
            "evaluations": diagnostics["evaluations"],
            "measurement_groups": diagnostics["groups"],
            "parameters": diagnostics["parameters"],
            "resources": computed.cost,
            "seed": params["seed"],
            "validity_flags": ["Finite-shot and ansatz errors are not fully bounded"],
        },
    }


@implements("samples.merge")
def _op_samples_merge(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from collections import Counter

    first = inputs[0]
    if any((item["encoding_id"], item["hamiltonian_id"], item["basis"],
            item["source"], item.get("orbital_rotation")) != (
               first["encoding_id"], first["hamiltonian_id"], first["basis"],
               first["source"], first.get("orbital_rotation"))
           for item in inputs[1:]):
        raise ValueError("Cannot merge incompatible QPU samples")
    counts = Counter()
    for item in inputs:
        counts.update(item["counts"])
    parts = [part for item in inputs for part in item["parts"]]
    values[identifier] = {
        "counts": dict(counts), "shots": sum(item["shots"] for item in inputs),
        "source": first["source"], "device": "multi",
        "encoding_id": first["encoding_id"],
        "hamiltonian_id": first["hamiltonian_id"],
        "basis": first["basis"], "seed": first["seed"],
        "orbital_rotation": first.get("orbital_rotation"),
        "job_ids": [job for item in inputs for job in item["job_ids"]],
        "parts": parts,
        "provenance": {"question": first["provenance"]["question"],
                       "method": first["provenance"]["method"],
                       "implementation": "samples.merge/qcl-host",
                       "device": "multi", "parts": [item["provenance"]
                                                    for item in inputs]},
    }


def _rotated_space(data, store, vector):
    """The active space with its integrals in the orbitals an open LUCJ circuit measures."""
    import copy
    import dataclasses

    import numpy as np

    from chemflow.impls.lucj import LucjParams, rotated_integrals

    n_orb = data.sector.n_orb
    h1, h2 = rotated_integrals(store.get(data.hamiltonian.h1_ref),
                               store.get(data.hamiltonian.h2_ref),
                               LucjParams.from_vector(np.asarray(vector), n_orb).kparams)
    hamiltonian = data.hamiltonian.model_copy(update={"h1_ref": store.put(h1),
                                                      "h2_ref": store.put(h2)})
    if dataclasses.is_dataclass(data):
        return dataclasses.replace(data, hamiltonian=hamiltonian)
    rotated = copy.copy(data)
    rotated.hamiltonian = hamiltonian
    return rotated


@implements("sqd.diagonalize")
def _op_sqd_diagonalize(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    import numpy as np

    from chemflow.impls.sqd import sqd_sector
    from chemflow.types import Samples

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("SQD currently requires a CPU")
    if inputs[0]["data"] is not inputs[2]["data"]:
        raise ValueError("SQD operator and sector refer to different active spaces")
    data = inputs[0]["data"]
    bits = inputs[1]
    if bits["hamiltonian_id"] != data.hamiltonian.id:
        raise ValueError("SQD samples refer to a different Hamiltonian")
    samples = Samples(counts=bits["counts"], shots=bits["shots"],
                      encoding_id=bits["encoding_id"], basis=bits["basis"],
                      source=bits["source"], seed=bits["seed"],
                      device=bits["device"], job_ids=bits["job_ids"])
    recovery = node["params"].get("recovery", True)
    extend = node["params"].get("extend", True)
    if bits.get("orbital_rotation") is not None:
        data = _rotated_space(data, store, bits["orbital_rotation"])
    energy, diagnostics = sqd_sector(data, data.sector, store, samples,
                                     seed=node["params"]["seed"], recovery=recovery,
                                     with_ext=extend,
                                     n_batches=node["params"].get("n_batches", 10),
                                     max_iter=node["params"].get("max_iter", 5))
    batches = diagnostics["batch_energies"]
    variance = (float(np.var(batches, ddof=1) / len(batches))
                if len(batches) > 1 else 0.0)
    values[identifier] = {
        "value": float(energy), "unit": "hartree", "variance": variance,
        "provenance": {
            "question": inputs[0]["question"],
            "method": {"family": "SQD", "basis": data.model.basis,
                       "active_electrons": data.active.n_elec,
                       "active_orbitals": data.active.n_orb},
            "implementation": "sqd.diagonalize/chemflow-pyscf",
            "device": device["id"], "versions": inputs[0]["versions"],
            "sample_source": bits["source"], "sample_device": bits["device"],
            "sample_job_ids": bits["job_ids"], "sample_shots": bits["shots"],
            "sample_parts": bits["parts"],
            "seed": node["params"]["seed"],
            "subspace_dimension": diagnostics["subspace_dimension"],
            "sector_dimension": diagnostics["sector_dimension"],
            "valid_fraction": diagnostics["valid_fraction"],
            "recovery": recovery, "extend": extend,
            **({"orbital_basis": "LUCJ-rotated: integrals rotated to the sampled basis"}
               if bits.get("orbital_rotation") is not None else {}),
            "batch_spread": diagnostics["batch_spread"],
            "validity_flags": ["SQD subspace bias is not bounded by batch variance"],
            **({"sample_billing": bits["provenance"]["billing"]}
               if "billing" in bits["provenance"] else {}),
        },
    }
