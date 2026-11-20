"""Predicted node costs for the scheduler, refined by measured profiles.

A node's cost signature is its operation, device kind and the problem sizes it
works on (the nearest active space, the nearest molecule and the shot count).
After every run the measured durations are folded into a small profile cache
(an exponential moving average per signature), so a plan that has run before is
scheduled with measured costs; a new signature falls back to a size-based
heuristic. Costs only order the work: results never depend on them.

The cache lives in ``$QCL_PROFILE_CACHE`` (default ``~/.cache/qcl/profile.json``);
set it to ``off`` to disable reading and writing.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_ALPHA = 0.5   # weight of the newest measurement in the moving average
# Most CPU nodes (PySCF) scale poorly beyond a few dozen OpenMP threads, and PySCF's
# selected CI crashes with several hundred; one node never gets more than this.
MAX_NODE_THREADS = 32
_PROJECTIONS = ("host", "projection")


def cache_path() -> Path | None:
    setting = os.environ.get("QCL_PROFILE_CACHE")
    if setting == "off":
        return None
    return Path(setting) if setting else Path.home() / ".cache" / "qcl" / "profile.json"


def _load() -> dict[str, dict[str, float]]:
    path = cache_path()
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sizes(nodes: list[dict[str, Any]], node: dict[str, Any]) -> dict[str, Any]:
    """Nearest active space and molecule above ``node`` (breadth first)."""
    sizes: dict[str, Any] = {}
    frontier, seen = [node["id"]], set()
    while frontier and not ("orbitals" in sizes and "atoms" in sizes):
        following = []
        for identifier in frontier:
            if identifier in seen:
                continue
            seen.add(identifier)
            current = nodes[identifier]
            params = current.get("params") or {}
            if current["op"] == "active_space" and "orbitals" not in sizes:
                sizes["electrons"] = params.get("electrons")
                sizes["orbitals"] = params.get("orbitals")
            if current["op"] == "molecule" and "atoms" not in sizes:
                atoms = str(params.get("atoms", ""))
                sizes["atoms"] = len([part for part in atoms.split(";") if part.strip()])
                sizes["basis"] = params.get("basis")
            following.extend(current["inputs"])
        frontier = following
    return sizes


def _remote(device: dict[str, Any] | None) -> bool:
    return bool(device) and (device.get("site") == "remote" or
                             device.get("billing") == "metered")


def signature(nodes: list[dict[str, Any]], node: dict[str, Any]) -> str:
    device = node.get("device") or {}
    params = node.get("params") or {}
    sizes = _sizes(nodes, node)
    parts = [node["op"], device.get("kind", "host"),
             "remote" if _remote(device) else "local",
             f"sim={device.get('simulator_host') or ''}",
             f"e={sizes.get('electrons')}", f"o={sizes.get('orbitals')}",
             f"atoms={sizes.get('atoms')}", f"basis={sizes.get('basis')}",
             f"shots={params.get('shots')}",
             f"recovery={params.get('recovery', True)}", f"extend={params.get('extend', True)}"]
    return "|".join(str(part) for part in parts)


def heuristic(nodes: list[dict[str, Any]], node: dict[str, Any]) -> float:
    """Rough seconds from problem sizes; used until a signature has been measured."""
    if node["kind"] in _PROJECTIONS:
        return 0.0
    device = node.get("device") or {}
    params = node.get("params") or {}
    sizes = _sizes(nodes, node)
    orbitals = sizes.get("orbitals") or 0
    electrons = sizes.get("electrons") or 0
    determinants = (math.comb(orbitals, electrons // 2) ** 2) if orbitals else 1
    atoms = sizes.get("atoms") or 2
    operation = node["op"]
    if node["kind"] == "copy":
        return 0.01
    if operation.startswith("scf.") or operation == "mean_field":
        return 0.2 * atoms ** 2
    if operation == "active_space":
        return 0.1 * atoms
    if operation == "ccsd.amplitudes":
        return 0.05 + 1e-4 * orbitals ** 4
    if operation in ("fci.solve", "ed.ground_state", "sci.solve"):
        return 0.05 + 2e-6 * determinants * orbitals ** 2
    if operation == "sqd.diagonalize":
        rounds = 10 if params.get("recovery", True) else 1
        return 0.2 + 2e-7 * rounds * determinants * orbitals ** 2
    if operation in ("sample", "vqe.sampled", "estimate.quantum"):
        shots = params.get("shots") or 1000
        if _remote(device):
            return 8.0 + 4e-4 * shots
        return 0.05 + 2e-7 * determinants * 10 + 1e-6 * shots
    return 0.5


def predict(nodes: list[dict[str, Any]]) -> dict[int, float]:
    cache = _load()
    costs = {}
    for node in nodes:
        entry = cache.get(signature(nodes, node))
        costs[node["id"]] = (float(entry["seconds"]) if isinstance(entry, dict) and
                             "seconds" in entry else heuristic(nodes, node))
    return costs


def record(nodes: list[dict[str, Any]], profile: list[dict[str, Any]]) -> None:
    """Fold measured node durations into the profile cache (best effort)."""
    path = cache_path()
    if path is None or not profile:
        return
    cache = _load()
    by_id = {node["id"]: node for node in nodes}
    for entry in profile:
        node = by_id.get(entry["id"])
        if node is None or entry.get("duration_s") is None:
            continue
        key = signature(nodes, node)
        previous = cache.get(key)
        measured = float(entry["duration_s"])
        if isinstance(previous, dict) and "seconds" in previous:
            cache[key] = {"seconds": (1 - _ALPHA) * float(previous["seconds"]) + _ALPHA * measured,
                          "count": int(previous.get("count", 1)) + 1}
        else:
            cache[key] = {"seconds": measured, "count": 1}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                         encoding="utf-8", suffix=".tmp") as stream:
            json.dump(cache, stream, indent=1, sort_keys=True)
        os.replace(stream.name, path)
    except OSError:
        pass


def default_slots(nodes: list[dict[str, Any]], executed: set[int] | None = None
                  ) -> dict[str, int]:
    """Concurrency per lane: how many of its nodes can usefully run at once.

    A lane's width is the largest number of its nodes at one dependency depth.
    CPU lanes split their threads (at least 8 per node), cloud QPUs keep up to
    four jobs in flight, local simulators up to four samples, and GPUs run one
    node at a time.
    A device's ``slots`` option overrides the default.
    """
    depth: dict[int, int] = {}
    width: dict[str, dict[int, int]] = {}
    devices: dict[str, dict[str, Any]] = {}
    for node in nodes:
        inputs = [*node["inputs"], *(guard["condition"] for guard in node.get("guards", ()))]
        depth[node["id"]] = 1 + max((depth[item] for item in inputs), default=0)
        device = node.get("device")
        if device is None or node["kind"] in _PROJECTIONS:
            continue
        if executed is not None and node["id"] not in executed:
            continue
        devices[device["id"]] = device
        levels = width.setdefault(device["id"], {})
        levels[depth[node["id"]]] = levels.get(depth[node["id"]], 0) + 1
    slots = {}
    for lane, device in devices.items():
        lane_width = max(width[lane].values(), default=1)
        requested = (device.get("options") or {}).get("slots")
        if type(requested) is int and requested >= 1:
            slots[lane] = requested
        elif device["kind"] == "cpu":
            threads = device.get("threads") or os.cpu_count() or 1
            slots[lane] = max(1, min(lane_width, threads // 8))
        elif device["kind"] == "qpu" and (_remote(device) or device.get("simulator_host")):
            # Cloud jobs wait in flight together; simulator samples are mostly
            # host-side state preparation, which separate processes overlap.
            slots[lane] = max(1, min(lane_width, 4))
        else:
            slots[lane] = 1
    slots["host"] = 8   # attribute access and small host operations: never a bottleneck
    return slots


def lane_threads(nodes: list[dict[str, Any]], slots: dict[str, int]) -> dict[str, int]:
    """CPU threads for each node of a CPU lane: the device's threads over its slots,
    at most ``MAX_NODE_THREADS``."""
    threads = {}
    for node in nodes:
        device = node.get("device")
        if device and device["kind"] == "cpu" and device["id"] in slots:
            total = device.get("threads") or os.cpu_count() or 1
            threads[device["id"]] = max(1, min(MAX_NODE_THREADS, total // slots[device["id"]]))
    return threads
