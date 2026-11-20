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
    plan = request.get("schedule")
    lanes = isinstance(plan, dict)
    processes = sorted((plan.get("processes") or {}).items()) if lanes else []
    shared = _shared_store_path() if any(enabled for _, enabled in processes) else None
    store = ArtifactStore(shared)
    context = Context(values, ci_vectors, by_id, density_sources, profile, store,
                      request, qpu_backend_factory)
    lane_threads = plan.get("threads", {}) if lanes else {}
    origin = time.perf_counter()
    # Lanes share the coordinator process, so its CPU time would mix concurrent
    # nodes; nodes run in a lane process are timed with that process's clock.
    clock = time.thread_time if lanes else time.process_time

    def run_node(node: dict[str, Any]) -> None:
        _run_node(context, node, lane_threads, origin, clock)

    if lanes:
        limits = _blas_limits(min(lane_threads.values())) if lane_threads else None
        runtime = None
        try:
            if shared is not None:
                from qcl._runtime import Runtime

                runtime = Runtime(request, plan, context, shared, origin, run_node)
            _run_scheduled(by_id, run_node, plan, values, runtime)
            if runtime is not None:
                runtime.collect(sorted(_output_ids(request["output"])))
        finally:
            if runtime is not None:
                runtime.shutdown()
            if limits is not None:
                limits.unregister()
            if shared is not None:
                import shutil

                shutil.rmtree(shared, ignore_errors=True)
        profile.sort(key=lambda entry: entry["id"])
    else:
        for node in request["graph"]["nodes"]:
            run_node(node)

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


def _run_node(context: Context, node: dict[str, Any], lane_threads: dict[str, int],
              origin: float, clock: Callable[[], float]) -> None:
    values = context.values
    identifier = node["id"]
    operation = node["op"]
    if any(values[guard["condition"]] is not guard["expected"]
           for guard in node.get("guards", ())):
        values[identifier] = _MISSING
        return
    inputs = [values[item] for item in node["inputs"]]
    if operation not in ("map", "cond") and any(item is _MISSING for item in inputs):
        values[identifier] = _MISSING
        return
    started = (time.perf_counter() if node["kind"] == "compute" or
               node["kind"] == "copy" and node["device"]["kind"] == "gpu" else None)
    started_cpu = clock() if started is not None else None
    handler = IMPLEMENTATIONS.get(operation)
    if handler is None:
        raise ValueError(f"Unsupported QCL operation: {operation}")
    threads = lane_threads.get((node.get("device") or {}).get("id"))
    if threads is not None:
        # OpenMP thread counts are per calling thread, so concurrent CPU
        # nodes each get their share of the device's cores.
        import pyscf

        pyscf.lib.num_threads(threads)
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
        finished = time.perf_counter()
        context.profile.append({"id": identifier, "op": operation,
                                "device": node["device"]["id"],
                                "duration_s": finished - started,
                                "cpu_duration_s": clock() - started_cpu,
                                "start_s": started - origin, "end_s": finished - origin})


# ---- lane processes -------------------------------------------------------------
# CPU compute nodes of a compiled schedule run in a pool of worker processes per CPU
# lane, so Python-heavy CPU work (SQD, CCSD set-up) never holds the GIL that GPU
# launches and QPU polling in the coordinator need. Inputs travel by (cloud)pickle;
# arrays travel through a content-addressed store in shared memory.

def _shared_store_path() -> str:
    import os
    import tempfile

    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    return tempfile.mkdtemp(prefix="qcl-store-", dir=base)


def _output_ids(output: Any) -> set[int]:
    if type(output) is int:
        return {output}
    if output["type"] == "node":
        return {output["id"]}
    if output["type"] in ("tuple", "list"):
        return set().union(*(_output_ids(item) for item in output["items"]))
    if output["type"] == "dict":
        return set().union(*(_output_ids(item) for item in output["items"].values()))
    return set()


def _lane(node: dict[str, Any]) -> str:
    device = node.get("device")
    return "host" if device is None else device["id"]


def _dependencies(node: dict[str, Any]) -> list[int]:
    return [*node["inputs"], *(guard["condition"] for guard in node.get("guards", ()))]


def _blas_limits(threads: int) -> Any:
    """Limit BLAS threads process-wide while CPU nodes share the cores."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=max(1, threads), user_api="blas")


def _run_scheduled(by_id: dict[int, dict[str, Any]],
                   run_node: Callable[[dict[str, Any]], None],
                   plan: dict[str, Any], values: dict[int, Any],
                   runtime: Any = None) -> None:
    """Execute a compiled schedule (``qcl-sched``) on per-device lanes.

    Each lane has its own thread pool with the schedule's slot count. A node is
    ready when all of its inputs and guards have values; each lane dispatches
    its ready nodes highest priority (upward rank) first, up to its slots.
    Eliminated nodes never run, and an alias takes its canonical node's value
    as soon as that finishes. After a failure no further node is started, and
    the failure with the lowest node id is raised once the running nodes end.
    """
    import heapq
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    aliases = {int(alias): int(canonical) for alias, canonical in plan["aliases"].items()}
    followers: dict[int, list[int]] = {}
    for alias, canonical in aliases.items():
        followers.setdefault(canonical, []).append(alias)
    priority = {int(key): value for key, value in plan["priority"].items()}
    lanes = {int(key): value for key, value in plan["lane"].items()}
    slots = {lane: max(1, int(plan["slots"].get(lane, 1))) for lane in set(lanes.values())}
    waiting = {identifier: {aliases.get(item, item) for item in _dependencies(by_id[identifier])}
               for identifier in plan["execute"]}
    dependents: dict[int, list[int]] = {}
    for identifier, needs in waiting.items():
        for need in needs:
            dependents.setdefault(need, []).append(identifier)
    executors = {lane: ThreadPoolExecutor(max_workers=count, thread_name_prefix=f"qcl-{lane}")
                 for lane, count in slots.items()}
    ready: dict[str, list[tuple[float, int]]] = {lane: [] for lane in slots}
    busy = dict.fromkeys(slots, 0)
    running: dict[Any, int] = {}
    failures: dict[int, BaseException] = {}

    def push(identifier: int) -> None:
        heapq.heappush(ready[lanes[identifier]], (-priority.get(identifier, 0.0), identifier))

    in_process: dict[Any, str] = {}
    import os

    trace = os.environ.get("QCL_TRACE")   # a file to append dispatch events to
    started_at = time.perf_counter()

    def note(event: str, identifier: int) -> None:
        if trace:
            node = by_id[identifier]
            with open(trace, "a", encoding="utf-8") as stream:
                stream.write(f"{time.perf_counter() - started_at:8.2f} {event:6s} "
                             f"{identifier:4d} {node['op']} {lanes[identifier]}\n")

    def dispatch() -> None:
        for lane, heap in ready.items():
            while heap and busy[lane] < slots[lane]:
                _, identifier = heapq.heappop(heap)
                busy[lane] += 1
                if runtime is None:
                    future = executors[lane].submit(run_node, by_id[identifier])
                else:
                    future, target = runtime.submit(by_id[identifier], executors[lane])
                    if target is not None:
                        in_process[future] = target
                running[future] = identifier
                note("start", identifier)

    try:
        for identifier in plan["execute"]:
            if not waiting[identifier]:
                push(identifier)
        dispatch()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=running.__getitem__):
                identifier = running.pop(future)
                busy[lanes[identifier]] -= 1
                note("done", identifier)
                error = future.exception()
                if error is not None:
                    failures[identifier] = error
                    continue
                if future in in_process:
                    runtime.merge(in_process.pop(future), future.result())
                for alias in followers.get(identifier, ()):
                    if identifier in values:
                        values[alias] = values[identifier]
                for dependent in dependents.get(identifier, ()):
                    waiting[dependent].discard(identifier)
                    if not waiting[dependent]:
                        push(dependent)
            if not failures:
                dispatch()
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
    if failures:
        raise failures[min(failures)]


def _run_lanes(nodes: list[dict[str, Any]],
               run_node: Callable[[dict[str, Any]], None]) -> None:
    """One slot per device, program order among ready nodes (no optimization)."""
    _run_scheduled({node["id"]: node for node in nodes}, run_node, {
        "execute": [node["id"] for node in nodes], "aliases": {},
        "priority": {str(node["id"]): -node["id"] for node in nodes},
        "lane": {str(node["id"]): _lane(node) for node in nodes}, "slots": {}}, {})


def main() -> None:
    from chemflow.backends.cuda_libs import (
        import_gpu4pyscf_single_device,
        prefer_wheel_cuda_libraries,
    )

    prefer_wheel_cuda_libraries()
    import os

    if os.environ.get("QCL_TRACE"):
        import faulthandler
        import signal

        # Stays open for the process's life: SIGUSR1 dumps every thread's stack.
        stacks = open(f"{os.environ['QCL_TRACE']}.coordinator", "w")  # noqa: SIM115
        faulthandler.register(signal.SIGUSR1, all_threads=True, file=stacks)
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
