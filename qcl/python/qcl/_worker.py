"""QCL graph worker backed by ChemFlow's scientific adapters.

Operation implementations live in ``qcl._impls`` and are registered by name.
"""

import base64
import json
import math
import sys
import time
from collections.abc import Callable
from typing import Any

from qcl._impls import IMPLEMENTATIONS, Context
from qcl._impls._common import (  # noqa: F401  (re-exported for callers and tests)
    _MISSING,
    _canonical_unrestricted_energies,
    _custom_statevector,
    _finite_difference_optimize,
    _gpu_index,
    _gpu_kernel,
    _linear_uncertainty,
    _molecule_identity,
    _native_ci_order,
    lattice_observable_value,
)


def execute(request: dict[str, Any],
            qpu_backend_factory: Callable[[dict[str, Any]], Any] | None = None
            ) -> dict[str, Any]:

    from chemflow.runtime.artifacts import ArtifactStore

    values: dict[int, Any] = {}
    ci_vectors: dict[int, Any] = {}
    by_id = {node["id"]: node for node in request["graph"]["nodes"]}
    density_sources = set()
    for density_node in request["graph"]["nodes"]:
        if density_node["op"] in ("rdm1", "rdm2", "fci.coefficients"):
            source_node = by_id[density_node["inputs"][0]]
            while source_node["op"] == "copy":
                source_node = by_id[source_node["inputs"][0]]
            density_sources.add(source_node["id"])
    profile: list[dict[str, Any]] = []
    store = ArtifactStore()
    context = Context(values, ci_vectors, by_id, density_sources, profile, store,
                      request, qpu_backend_factory)
    for node in request["graph"]["nodes"]:
        identifier = node["id"]
        operation = node["op"]
        if any(values[guard["condition"]] is not guard["expected"]
               for guard in node.get("guards", ())):
            values[identifier] = _MISSING
            continue
        inputs = [values[item] for item in node["inputs"]]
        if operation not in ("map", "cond") and any(item is _MISSING for item in inputs):
            values[identifier] = _MISSING
            continue
        started = (time.perf_counter() if node["kind"] == "compute" or
                   node["kind"] == "copy" and node["device"]["kind"] == "gpu" else None)
        started_cpu = time.process_time() if started is not None else None
        handler = IMPLEMENTATIONS.get(operation)
        if handler is None:
            raise ValueError(f"Unsupported QCL operation: {operation}")
        handler(context, node, identifier, operation, inputs)
        if (isinstance(values[identifier], dict) and
                "variance" in values[identifier] and
                "_uncertainty_components" not in values[identifier]):
            values[identifier]["_uncertainty_components"] = {
                str(identifier): (1.0, values[identifier]["variance"])}
        if started is not None:
            gpu_host = (node["device"]["id"] if node["device"]["kind"] == "gpu" else
                        node["device"].get("simulator_host")
                        if node["device"]["kind"] == "qpu" else None)
            if gpu_host is not None and gpu_host.startswith("gpu"):
                import cupy as cp

                with cp.cuda.Device(_gpu_index(gpu_host)) as gpu_device:
                    gpu_device.synchronize()
            profile.append({"id": identifier, "op": operation,
                            "device": node["device"]["id"],
                            "duration_s": time.perf_counter() - started,
                            "cpu_duration_s": time.process_time() - started_cpu})
    def result_for(output):
        def tensor_payload(value):
            import numpy as np

            source = value["array"]
            original = (source.get() if hasattr(source, "get") else
                        np.asarray(source, dtype=np.float64))
            array = np.ascontiguousarray(original)
            return {"data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
                    "shape": list(original.shape), "dtype": str(array.dtype),
                    "device": value["device"], "unit": value["unit"],
                    "layout": value["layout"], "provenance": value["provenance"]}

        def public_value(identifier):
            value = values[identifier]
            if request["graph"]["nodes"][identifier]["value_type"] in ("tensor", "hessian"):
                return tensor_payload(value)
            if request["graph"]["nodes"][identifier]["value_type"] == "eigensystem":
                return {"values": tensor_payload(value["values"]),
                        "vectors": tensor_payload(value["vectors"]),
                        "provenance": value["provenance"]}
            if request["graph"]["nodes"][identifier]["value_type"] == "svd_result":
                return {**{part: tensor_payload(value[part])
                           for part in ("u", "s", "vh")},
                        "provenance": value["provenance"]}
            if request["graph"]["nodes"][identifier]["value_type"] == "qubit_operator":
                return {key: value[key] for key in
                        ("terms", "n_qubits", "encoding_id", "provenance")}
            return value

        if type(output) is int:
            return public_value(output)
        if output["type"] == "node":
            return public_value(output["id"])
        if output["type"] in ("tuple", "list"):
            return [result_for(item) for item in output["items"]]
        if output["type"] == "dict":
            return {key: result_for(item) for key, item in output["items"].items()}
        return output["value"]

    return {"result": result_for(request["output"]), "profile": profile}


def main() -> None:
    from chemflow.backends.cuda_libs import (
        import_gpu4pyscf_single_device,
        prefer_wheel_cuda_libraries,
    )

    prefer_wheel_cuda_libraries()
    request = json.load(sys.stdin)
    # Placement is per device: keep GPU4PySCF from spreading one SCF over every GPU.
    if any((node.get("device") or {}).get("kind") == "gpu"
           for node in request["graph"]["nodes"]):
        import_gpu4pyscf_single_device()
    if request.get("deterministic"):
        import random

        import numpy as np

        random.seed(0)
        np.random.seed(0)
    cpu_core_s = request.get("cpu_core_s")
    if cpu_core_s is not None:
        import os
        import resource
        import signal

        def exceeded(_signum, _frame):
            # Exit instead of raising: an exception raised from a signal handler
            # surfaces at an arbitrary line, where library code (for example an
            # import) can catch or replace it and lose the marker the parent
            # process uses to classify the failure.
            os.write(2, b"QCL-E503: CPU core-time limit exceeded\n")
            os._exit(128 + signal.SIGXCPU)

        signal.signal(signal.SIGPROF, exceeded)
        signal.setitimer(signal.ITIMER_PROF, cpu_core_s)
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        ceiling = math.ceil(time.process_time() + cpu_core_s) + 1
        soft = ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_CPU,
                           (soft, hard))
    json.dump(execute(request), sys.stdout)


if __name__ == "__main__":
    main()
