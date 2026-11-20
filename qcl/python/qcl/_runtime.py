"""Process-backed execution of compiled QCL schedules.

A compiled schedule (``qcl-sched``) runs on device lanes. Python-level work in one
process serializes on the interpreter lock, so lanes that do real work run in their
own processes:

* a GPU lane is one persistent process that keeps its values resident on the GPU;
* a CPU lane is a pool of processes, one per slot, that receive their inputs and
  return their results;
* a GPU-hosted QPU simulator lane is one such process;
* cloud-QPU lanes (which mostly wait for the network) and cheap host operations run
  in threads of the coordinator.

Values travel between processes only when a node elsewhere needs them, serialized
with cloudpickle; arrays go through a content-addressed artifact store in shared
memory. A value holding GPU arrays stays in its GPU process, and any node reading it
(a projection, or a copy to the host) runs in that process, so GPU data leave the
device only through an explicit copy.

Object identity is preserved across processes. Handlers rely on it (an operator and
a sector must come from the *same* active space), so every non-trivial object gets a
token: a process that already holds the object receives a reference instead of a
duplicate, and results that contain objects the coordinator sent resolve back to the
coordinator's originals.
"""

from __future__ import annotations

import io
import itertools
import json
import pickle
import sys
import time
import types
from collections.abc import Callable
from typing import Any

from qcl._impls import Context

# ---- identity-preserving serialization ---------------------------------------------

_INTERN: dict[str, Any] = {}      # token -> object held by this process
_TOKENS: dict[int, str] = {}      # id(object) -> token
_PEER: set[str] = set()           # tokens the coordinator (or, there, a lane) also holds
_TAG = ["c"]
_COUNTER = itertools.count()      # never reset: a token names one object for good
_PLAIN = (dict, list, tuple, set, frozenset, str, bytes, bytearray, int, float, complex,
          bool, type(None), type, types.FunctionType, types.BuiltinFunctionType,
          types.MethodType, types.ModuleType, io.IOBase)


class _DeviceValue(Exception):
    """Raised while pickling a value that holds GPU arrays."""


def _shareable(obj: Any) -> bool:
    if isinstance(obj, _PLAIN) or not hasattr(obj, "__dict__"):
        return False
    return type(obj).__module__.split(".")[0] not in ("numpy", "cupy")


def _token(obj: Any) -> str:
    token = _TOKENS.get(id(obj))
    if token is None:
        token = f"{_TAG[0]}:{next(_COUNTER)}"
        _TOKENS[id(obj)] = token
        _INTERN[token] = obj
    return token


def _register(token: str, obj: Any) -> Any:
    known = _INTERN.get(token)
    if known is not None:
        return known
    _INTERN[token] = obj
    _TOKENS[id(obj)] = token
    _PEER.add(token)
    return obj


def _lookup(token: str) -> Any:
    return _INTERN[token]


def _restore_stream(name: str) -> Any:
    return getattr(sys, name, None)


def _host_array(array: Any) -> Any:
    return array


def _is_device_array(obj: Any) -> bool:
    kind = type(obj)
    return kind.__module__.split(".")[0] == "cupy" and kind.__name__ == "ndarray"


def dumps(value: Any, *, known: Callable[[str], bool] = lambda token: False,
          device: str = "keep", sent: set[str] | None = None) -> bytes:
    """Serialize ``value``. Objects the receiver already holds (``known``) travel as
    references. GPU arrays: ``keep`` pickles them, ``forbid`` raises, ``host``
    copies them to numpy. Tokens of objects sent in full are added to ``sent``."""
    try:
        import cloudpickle

        base = cloudpickle.Pickler
    except ImportError:
        base = pickle.Pickler

    class Pickler(base):
        def __init__(self, stream):
            super().__init__(stream, protocol=5)
            self._full: set[int] = set()

        def reducer_override(self, obj: Any) -> Any:
            # PySCF objects keep a reference to sys.stdout; restore it on load.
            if isinstance(obj, io.IOBase):
                return _restore_stream, ("stderr" if obj is sys.stderr else "stdout",)
            if device != "keep" and _is_device_array(obj):
                if device == "forbid":
                    raise _DeviceValue
                return _host_array, (obj.get(),)
            if _shareable(obj) and id(obj) not in self._full:
                token = _token(obj)
                if known(token):
                    return _lookup, (token,)
                self._full.add(id(obj))
                if sent is not None:
                    sent.add(token)
                return _register, (token, obj)
            parent = getattr(super(), "reducer_override", None)
            return parent(obj) if parent is not None else NotImplemented

    buffer = io.BytesIO()
    Pickler(buffer).dump(value)
    return buffer.getvalue()


# ---- inside a lane process ----------------------------------------------------------

_STATE: dict[str, Any] = {}


def _init(request_json: str, store_path: str, density_sources: list[int], origin: float,
          gpu: bool, stateful: bool) -> None:
    import os

    from chemflow.backends.cuda_libs import (
        import_gpu4pyscf_single_device,
        prefer_wheel_cuda_libraries,
    )
    from chemflow.runtime.artifacts import ArtifactStore

    _TAG[0] = f"p{os.getpid()}"
    if os.environ.get("QCL_TRACE"):
        import faulthandler
        import signal

        # Stays open for the process's life: SIGUSR1 dumps every thread's stack.
        stacks = open(f"{os.environ['QCL_TRACE']}.{os.getpid()}", "w")  # noqa: SIM115
        faulthandler.register(signal.SIGUSR1, all_threads=True, file=stacks)
    prefer_wheel_cuda_libraries()
    if gpu:
        import_gpu4pyscf_single_device()
    request = json.loads(request_json)
    _STATE.update(request=request, origin=origin, density_sources=set(density_sources),
                  by_id={node["id"]: node for node in request["graph"]["nodes"]},
                  store=ArtifactStore(store_path), values={}, ci_vectors={},
                  stateful=stateful)
    import pyscf  # noqa: F401  (import while idle, not inside the first node)


def _warm() -> None:
    return None


def _task(identifier: int, payload: bytes) -> bytes:
    from qcl._worker import _run_node

    stateful = _STATE["stateful"]
    if not stateful:  # a pool process may run any task: hold nothing between tasks
        _INTERN.clear()
        _TOKENS.clear()
        _PEER.clear()
        _STATE["values"], _STATE["ci_vectors"] = {}, {}
    values, ci_vectors = _STATE["values"], _STATE["ci_vectors"]
    given = pickle.loads(payload)
    for key, (value, vector) in given["values"].items():
        values[key] = value
        if vector is not None:
            ci_vectors[key] = vector
    for alias, canonical in given["aliases"].items():
        values[alias] = values[canonical]
        if canonical in ci_vectors:
            ci_vectors[alias] = ci_vectors[canonical]
    before = set(values)
    profile: list[dict[str, Any]] = []
    context = Context(values, ci_vectors, _STATE["by_id"], _STATE["density_sources"],
                      profile, _STATE["store"], _STATE["request"], None)
    _run_node(context, _STATE["by_id"][identifier], {}, _STATE["origin"], time.process_time)
    produced = {}
    for key in [key for key in values if key not in before]:
        sent: set[str] = set()
        try:
            produced[key] = dumps((values[key], ci_vectors.get(key)), known=_PEER.__contains__,
                                  device="forbid" if stateful else "keep", sent=sent)
            _PEER.update(sent)   # only what was actually sent is held by the coordinator
        except Exception:
            if not stateful:
                raise
            # GPU arrays, CUDA handles and similar: the value stays resident here.
            produced[key] = None
    return pickle.dumps({"values": produced, "profile": profile}, protocol=5)


def _export(identifiers: list[int]) -> dict[int, bytes]:
    values, vectors = _STATE["values"], _STATE["ci_vectors"]
    exported = {}
    for key in identifiers:
        sent: set[str] = set()
        exported[key] = dumps((values[key], vectors.get(key)), known=_PEER.__contains__,
                              device="host", sent=sent)
        _PEER.update(sent)
    return exported


# ---- in the coordinator ---------------------------------------------------------------

class Runtime:
    """Routes the nodes of a compiled schedule to lane processes or threads."""

    def __init__(self, request: dict[str, Any], plan: dict[str, Any],
                 context: Context, store_path: str, origin: float,
                 run_here: Callable[[dict[str, Any]], None]):
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor

        _TAG[0] = "c"
        _INTERN.clear()
        _TOKENS.clear()
        _PEER.clear()
        self.context = context
        self.run_here_node = run_here
        self.aliases = {int(alias): int(canonical)
                        for alias, canonical in plan["aliases"].items()}
        self.resident: dict[int, str] = {}
        self.pools: dict[str, Any] = {}
        self.stateful: set[str] = set()
        self.holds: dict[str, set[str]] = {}   # tokens each stateful lane process holds
        devices = {node["device"]["id"]: node["device"] for node in request["graph"]["nodes"]
                   if node.get("device")}
        threads = plan.get("threads", {})
        names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS")
        saved = {name: os.environ.get(name) for name in names}
        request_json = json.dumps(request)
        try:
            for lane, enabled in sorted((plan.get("processes") or {}).items()):
                device = devices.get(lane)
                if not enabled or device is None:
                    continue
                kind = device["kind"]
                host = device.get("simulator_host") or ""
                if kind == "qpu" and not host:
                    continue  # cloud QPU: waiting on the network, keep it in a thread
                gpu = kind == "gpu" or host.startswith("gpu")
                stateful = kind == "gpu"
                count = 1 if stateful else max(1, int(plan["slots"].get(lane, 1)))
                os.environ.update(dict.fromkeys(
                    names, str(max(1, int(threads.get(lane, 16 if gpu else 8))))))
                pool = ProcessPoolExecutor(
                    max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init,
                    initargs=(request_json, store_path, sorted(context.density_sources),
                              origin, gpu, stateful))
                for _ in range(count):  # start now: imports overlap the first nodes
                    pool.submit(_warm)
                self.pools[lane] = pool
                if stateful:
                    self.stateful.add(lane)
                    self.holds[lane] = set()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def shutdown(self) -> None:
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=True)

    def _canonical(self, key: int) -> int:
        return self.aliases.get(key, key)

    def _loads(self, blob: bytes, source: str) -> Any:
        before = set(_PEER)
        value = pickle.loads(blob)
        if source in self.holds:  # objects that came from a lane process are held there
            self.holds[source].update(_PEER - before)
        return value

    def _materialize(self, keys: list[int]) -> None:
        """Make ``keys`` available as objects in the coordinator."""
        values, vectors = self.context.values, self.context.ci_vectors
        for key in keys:
            if key in values:
                continue
            canonical = self._canonical(key)
            if canonical not in values and canonical in self.resident:
                owner = self.resident[canonical]
                blob = self.pools[owner].submit(_export, [canonical]).result()[canonical]
                value, vector = self._loads(blob, owner)
                values[canonical] = value
                if vector is not None:
                    vectors[canonical] = vector
            values[key] = values[canonical]
            if canonical in vectors:
                vectors[key] = vectors[canonical]

    def _target(self, node: dict[str, Any], needed: list[int]) -> str | None:
        owners = {self.resident[self._canonical(key)] for key in needed
                  if self._canonical(key) in self.resident and
                  self._canonical(key) not in self.context.values}
        if len(owners) > 1:
            raise RuntimeError(f"QCL-E503: node {node['id']} ({node['op']}) reads GPU values "
                               f"held by {sorted(owners)}; copy them to one device first")
        if owners:
            return owners.pop()
        device = node.get("device") or {}
        lane = device.get("id")
        if node["kind"] == "compute" and lane in self.pools:
            return lane
        if node["kind"] == "copy" and device.get("kind") == "gpu" and lane in self.pools:
            return lane
        return None

    def submit(self, node: dict[str, Any], threads: Any) -> tuple[Any, str | None]:
        """Start ``node``: in its lane process, in the process holding its GPU inputs,
        or in a coordinator thread. Returns the future and the process lane (or None)."""
        needed = [*node["inputs"], *(guard["condition"] for guard in node.get("guards", ()))]
        target = self._target(node, needed)
        if target is None:
            def run() -> None:
                self._materialize(needed)
                self.run_here_node(node)
            return threads.submit(run), None
        there = {self._canonical(key) for key in needed
                 if self.resident.get(self._canonical(key)) == target}
        self._materialize([key for key in needed if self._canonical(key) not in there])
        values, vectors = self.context.values, self.context.ci_vectors
        holds = self.holds.get(target)
        sent: set[str] = set()
        payload = dumps({"values": {self._canonical(key): (values[self._canonical(key)],
                                                           vectors.get(self._canonical(key)))
                                    for key in needed if self._canonical(key) not in there},
                         "aliases": {key: self._canonical(key) for key in needed
                                     if key != self._canonical(key)}},
                        known=(holds.__contains__ if holds is not None else
                               (lambda token: False)), sent=sent)
        if holds is not None:
            holds.update(sent)
        return self.pools[target].submit(_task, node["id"], payload), target

    def merge(self, target: str, result: bytes) -> None:
        update = pickle.loads(result)
        values, vectors = self.context.values, self.context.ci_vectors
        pinned = set()
        for key, blob in update["values"].items():
            if blob is None:
                self.resident[key] = target
                node = self.context.by_id[key]
                if (node.get("device") or {}).get("kind") != "gpu":
                    pinned.add(key)
                continue
            value, vector = self._loads(blob, target)
            values[key] = value
            if vector is not None:
                vectors[key] = vector
        for entry in update["profile"]:
            if entry["id"] in pinned:
                # A host value that cannot leave the GPU process pins every consumer
                # to that process; record it rather than hide it.
                entry["warning"] = ("QCL-W202: value holds device data; its consumers "
                                    f"run in the {target} process")
        self.context.profile.extend(update["profile"])

    def collect(self, keys: list[int]) -> None:
        self._materialize(keys)
