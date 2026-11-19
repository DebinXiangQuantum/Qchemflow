import json
import os
import subprocess
from pathlib import Path

import pytest

from chemflow import build
from chemflow.diagnostics import CheckError
from chemflow.domains import load_registry
from chemflow.domains.synthesis import DEFAULT_HOME, parse_routes
from chemflow.runtime import Executor

FIXTURE = Path(__file__).parent / "fixtures/m6b/paracetamol.json.gz"


def program():
    with build.program("synthesis") as builder:
        node = builder.node("search_routes", target="CC(=O)Nc1ccc(O)cc1", stock="zinc",
                            iterations=100, time_limit_s=120)
    return builder.to_ir(), node.id


def test_recorded_plain_json_routes():
    routes = parse_routes(json.loads(FIXTURE.read_text()))
    assert len(routes) == 15
    assert routes[0]["solved"]
    assert routes[0]["leaves"] == ["CC(=O)OC(C)=O", "Nc1ccc(O)cc1"]
    assert routes[0]["in_stock"] == [True, True]
    assert routes[0]["step_count"] == 1
    assert routes[0]["scores"]["state score"] > 0.99


def test_missing_environment_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMFLOW_AIZYNTH_HOME", str(tmp_path))
    plan, _ = program()
    with pytest.raises(CheckError, match="CF-E080"):
        Executor(load_registry(["synthesis"])).run(plan)


def test_subprocess_adapter_recording_and_timeout(tmp_path, monkeypatch):
    import yaml

    home = tmp_path / "home"
    (home / "env/bin").mkdir(parents=True)
    (home / "env/bin/aizynthcli").touch()
    metadata = home / "env/lib/python3.11/site-packages/aizynthfinder-4.4.1.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Version: 4.4.1\n")
    (home / "data").mkdir()
    (home / "data/stock.hdf5").write_bytes(b"recorded-stock")
    (home / "data/config.yml").write_text("stock:\n  zinc: stock.hdf5\n")
    monkeypatch.setenv("CHEMFLOW_AIZYNTH_HOME", str(home))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        configuration = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        assert configuration["search"] == {"iteration_limit": 100, "time_limit": 120}
        assert kwargs["timeout"] == 180 and kwargs["check"] is False
        Path(command[command.index("--output") + 1]).write_bytes(FIXTURE.read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    plan, node = program()
    executor = Executor(load_registry(["synthesis"]))
    result = executor.run(plan)[node]
    executor.run(plan)
    assert len(calls) == 2
    assert result.solved and len(result.inventory["sha256"]) == 64
    assert result.provenance.device == "aizynthfinder:4.4.1"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(CheckError, match="CF-E080"):
        executor.run(plan)


@pytest.mark.skipif(not (Path(os.environ.get("CHEMFLOW_AIZYNTH_HOME", DEFAULT_HOME)) /
                        "env/bin/aizynthcli").is_file(), reason="isolated AiZynthFinder missing")
def test_live_paracetamol():
    plan, node = program()
    result = Executor(load_registry(["synthesis"])).run(plan)[node]
    assert result.solved
    assert {"CC(=O)OC(C)=O", "Nc1ccc(O)cc1"} <= set(result.routes[0]["leaves"])
