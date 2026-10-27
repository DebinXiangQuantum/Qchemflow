"""Isolated AiZynthFinder CLI route search; no chemistry or MCTS reimplementation."""

import hashlib
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from chemflow.diagnostics import AdapterError
from chemflow.types import Provenance

from ._adapters import register_adapter

DEFAULT_HOME = "/Volumes/macsoftware/caches/chemflow/aizynthfinder"


class RouteCandidates(BaseModel, frozen=True):
    solved: bool
    routes: list[dict]
    inventory: dict[str, str]
    provenance: Provenance


def parse_routes(payload):
    routes = []
    for tree in payload:
        leaves, in_stock, steps = [], [], 0
        pending = [tree]
        while pending:
            node = pending.pop()
            children = node.get("children", [])
            if node["type"] == "reaction":
                steps += 1
            elif not children:
                leaves.append(node["smiles"])
                in_stock.append(bool(node.get("in_stock", False)))
            pending.extend(reversed(children))
        routes.append({"leaves": leaves, "in_stock": in_stock, "step_count": steps,
                       "scores": tree.get("scores", {}),
                       "solved": bool(leaves) and all(in_stock), "tree": tree})
    return routes


def search_routes(store, target, stock="zinc", iterations=100, time_limit_s=120, seed=None):
    import yaml

    home = Path(os.environ.get("CHEMFLOW_AIZYNTH_HOME", DEFAULT_HOME))
    executable, config_path = home / "env/bin/aizynthcli", home / "data/config.yml"
    try:
        if not executable.is_file() or not config_path.is_file():
            raise ValueError("AiZynthFinder isolated environment/data missing")
        metadata = list((home / "env/lib").glob(
            "python*/site-packages/aizynthfinder-4.4.1.dist-info/METADATA"))
        if not metadata or "Version: 4.4.1\n" not in metadata[0].read_text():
            raise ValueError("The isolated adapter requires AiZynthFinder 4.4.1")
        if type(iterations) is not int or iterations < 1 or not 0 < time_limit_s < float("inf"):
            raise ValueError("Route search requires positive iterations and finite time limit")
        if seed is not None:
            raise ValueError("AiZynthFinder 4.4.1 CLI does not expose a random seed")
        config = yaml.safe_load(config_path.read_text())
        stock_path = Path(config["stock"][stock])
        if not stock_path.is_absolute():
            stock_path = config_path.parent / stock_path
        with stock_path.open("rb") as source:
            inventory_hash = hashlib.file_digest(source, "sha256").hexdigest()
        config["stock"] = {stock: str(stock_path)}
        for section in ("expansion", "filter"):
            for name, paths in config.get(section, {}).items():
                resolved = []
                for path in paths if isinstance(paths, list) else [paths]:
                    path = Path(path)
                    path = path if path.is_absolute() else config_path.parent / path
                    if not path.is_file():
                        raise ValueError(f"Missing AiZynthFinder data: {path}")
                    resolved.append(str(path))
                config[section][name] = resolved if isinstance(paths, list) else resolved[0]
        config.setdefault("search", {}).update(iteration_limit=iterations, time_limit=time_limit_s)
        with tempfile.TemporaryDirectory(prefix="chemflow-aizynth-") as directory:
            temporary = Path(directory)
            configuration, output = temporary / "config.yml", temporary / "routes.json.gz"
            configuration.write_text(yaml.safe_dump(config, sort_keys=False))
            completed = subprocess.run([str(executable), "--config", str(configuration),
                "--smiles", target, "--stocks", stock, "--output", str(output)],
                check=False, cwd=directory, capture_output=True, text=True, timeout=time_limit_s + 60,
                env={**os.environ, "PYTHONNOUSERSITE": "1"})
            if completed.returncode:
                status = f"exit status {completed.returncode}"
                if completed.returncode < 0:
                    try:
                        status += f" ({signal.Signals(-completed.returncode).name})"
                    except ValueError:
                        status += f" (signal {-completed.returncode})"
                raise ValueError(f"AiZynthFinder failed: {status}; "
                                 f"stderr tail: {completed.stderr[-2000:] or '<empty>'}; "
                                 f"stdout tail: {completed.stdout[-2000:] or '<empty>'}")
            routes = parse_routes(json.loads(output.read_text()))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError,
            subprocess.TimeoutExpired) as exc:
        raise AdapterError("CF-E080", None, str(exc),
                           "Provide the isolated AiZynthFinder 4.4.1 environment and data") from exc
    return RouteCandidates(solved=any(route["solved"] for route in routes), routes=routes,
        inventory={"stock": stock, "sha256": inventory_hash},
        provenance=Provenance(impl_id="search_routes", impl_version="1", backend_kind="cpu_numeric",
            device="aizynthfinder:4.4.1", software_versions={"aizynthfinder": "4.4.1"}))


def register(registry):
    register_adapter(registry, "search_routes", search_routes, {}, "route_candidates",
        ("target", "stock", "iterations", "time_limit_s", "seed"),
        "Isolated AiZynthFinder MCTS routes, stock membership and inventory SHA-256; no energies",
        effects="random")
