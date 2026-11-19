import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import pytest
from test_m3a_task import spec

from chemflow import build
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.check import check
from chemflow.diagnostics import AdapterError
from chemflow.domains import drug, load_registry
from chemflow.runtime import Executor

ASPIRIN = {"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
           "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
WATER = {"CID": 962, "SMILES": "O", "InChIKey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N"}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMFLOW_NAME_CACHE", str(tmp_path / "names.json"))

    def unavailable(*args, **kwargs):
        raise URLError("offline test: transport not injected")

    monkeypatch.setattr(drug, "urlopen", unavailable)


def program(names):
    with build.program("names") as builder:
        library = builder.node("ligands_from_names", names=names)
    return builder.to_ir(), library.id


@pytest.mark.parametrize("field", ["SMILES", "IsomericSMILES", "CanonicalSMILES"])
def test_transport_quotes_names_and_parses_recorded_json(monkeypatch, field):
    response = {"PropertyTable": {"Properties": [{"CID": 2244,
                field: "CC(=O)OC1=CC=CC=C1C(=O)O",
                "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}]}}
    calls = []

    def transport(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(drug, "urlopen", transport)
    assert drug.fetch_pubchem("aspirin /?#β") == ASPIRIN
    expected = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                "aspirin%20%2F%3F%23%CE%B2/property/SMILES,InChIKey/JSON")
    assert calls == [(expected, 15)]


def test_unreachable_transport_and_named_diagnostic():
    with pytest.raises(OSError):
        drug.fetch_pubchem("aspirin")
    plan, _ = program(["aspirin"])
    with pytest.raises(AdapterError, match="CF-E080.*aspirin"):
        Executor(load_registry(["drug"])).run(plan)


def test_multiple_names_exact_keys_and_original_timestamps(monkeypatch, tmp_path):
    records = {"aspirin": ASPIRIN, "Aspirin": ASPIRIN, "water": WATER}
    calls = []

    def fetch(name):
        calls.append(name)
        return dict(records[name])

    monkeypatch.setattr(drug, "fetch_pubchem", fetch)
    plan, library = program(["aspirin", "Aspirin", "water", "aspirin"])
    executor = Executor(load_registry(["drug"]))
    assert not check(plan, executor.registry)
    first = executor.run(plan)[library].ligands
    second = executor.run(plan)[library].ligands
    assert calls == ["aspirin", "Aspirin", "water"]
    assert set(first) == set(records)
    for name in records:
        assert first[name]["cached"] is False and second[name]["cached"] is True
        assert second[name]["retrieved_at"] == first[name]["retrieved_at"]
        assert datetime.fromisoformat(first[name]["retrieved_at"]).utcoffset().total_seconds() == 0
    assert set(json.loads((tmp_path / "names.json").read_text())) == set(records)
    assert executor.cache_hits == 0 and not executor.cache


@pytest.mark.parametrize("contents", ["{broken", "[]", '{"aspirin": null}',
                                    '{"aspirin": {"CID": 2244}}'])
def test_corrupt_cache_is_a_miss(monkeypatch, tmp_path, contents):
    cache = tmp_path / "names.json"
    cache.write_text(contents)
    with pytest.raises(AdapterError, match="CF-E080.*aspirin"):
        drug.ligands_from_names(None, ["aspirin"])
    assert cache.read_text() == contents
    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: dict(ASPIRIN))
    assert drug.ligands_from_names(None, ["aspirin"]).ligands["aspirin"]["cached"] is False
    assert json.loads(cache.read_text())["aspirin"]["CID"] == 2244


def test_atomic_replacement_preserves_old_file_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: dict(ASPIRIN))
    cache = tmp_path / "names.json"
    cache.write_text("{}")
    replace = drug.os.replace
    calls = []

    def interrupted(source, destination):
        assert Path(source).parent == destination.parent
        assert cache.read_text() == "{}"
        assert json.loads(Path(source).read_text())["aspirin"]["CID"] == 2244
        calls.append(source)
        raise OSError("interrupted replacement")

    monkeypatch.setattr(drug.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        drug.ligands_from_names(None, ["aspirin"])
    assert calls and cache.read_text() == "{}"
    assert list(tmp_path.iterdir()) == [cache]
    monkeypatch.setattr(drug.os, "replace", replace)
    drug.ligands_from_names(None, ["aspirin"])
    assert list(tmp_path.iterdir()) == [cache]
    assert json.loads(cache.read_text())["aspirin"]["CID"] == 2244


@pytest.mark.parametrize("bad_record", [None, {**ASPIRIN, "SMILES": "not a smiles"},
                                      {**ASPIRIN, "InChIKey": WATER["InChIKey"]}])
def test_partial_failure_has_no_library_or_cache(monkeypatch, tmp_path, bad_record):
    def fetch(name):
        if name == "aspirin":
            return dict(ASPIRIN)
        if bad_record is None:
            raise ConnectionError("not found")
        return bad_record

    monkeypatch.setattr(drug, "fetch_pubchem", fetch)
    plan, _ = program(["aspirin", "unresolvable"])
    executor = Executor(load_registry(["drug"]))
    with pytest.raises(AdapterError, match=("CF-E080" if bad_record is None else "CF-E060")):
        executor.run(plan)
    assert not (tmp_path / "names.json").exists()
    assert not executor.cache


def test_inconsistent_cached_record_is_refused(tmp_path):
    cache = tmp_path / "names.json"
    cache.write_text(json.dumps({"aspirin": {**ASPIRIN, "InChIKey": WATER["InChIKey"],
                                            "retrieved_at": "2026-10-13T00:00:00+00:00"}}))
    original = cache.read_bytes()
    with pytest.raises(AdapterError, match="CF-E060.*aspirin"):
        drug.ligands_from_names(None, ["aspirin"])
    assert cache.read_bytes() == original


@pytest.mark.parametrize("allowed", [False, True])
def test_agent_tools_require_network_before_any_execution(monkeypatch, allowed):
    registry = load_registry(["molecular", "drug"])
    store = TaskStore("g3-test")
    task = spec().model_copy(update={"allowed_effects": ("pure", "network") if allowed
                                   else ("pure",)})
    record = store.freeze(task, "human")
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store)
    base, _ = Orchestrator._base_plan(record, record.spec.target.active)
    plan_id = tools.plans.put(base)
    plan_id = tools.call("designer", "plan.add_node", {"plan_id": plan_id, "target": False,
        "node": {"id": "names", "op": "chemflow.ligands_from_names", "inputs": [],
                 "params": [{"key": "names", "value_json": '["aspirin"]'}],
                 "effects": "network", "binding": None, "refinement": None}})["data"]["plan_id"]
    assert tools.call("critic", "check", {"plan_id": plan_id})["data"] == []
    calls = []
    for implementation in registry.implementations.values():
        adapter = implementation.adapter

        def tracked(*args, _adapter=adapter, **kwargs):
            calls.append(_adapter.__name__)
            return _adapter(*args, **kwargs)

        implementation.adapter = tracked
    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: dict(ASPIRIN))
    if allowed:
        tools.call("designer", "run", {"plan_id": plan_id})
        assert "ligands_from_names" in calls
    else:
        with pytest.raises(AdapterError, match="CF-E120.*network"):
            tools.call("designer", "run", {"plan_id": plan_id})
        assert calls == []


def test_network_resource_forecast_and_no_executor_memoisation():
    implementation = load_registry(["drug"]).implementations["ligands_from_names"]
    assert implementation.effects == "network" and implementation.cacheable is False
    assert implementation.backend_kind == "cpu_numeric"
    assert implementation.resource_model(names=["aspirin"]) == {
        "wall_s": None, "cpu_core_s": 0, "gpu_s": 0, "sim_shots": 0,
        "qpu_s": 0, "qpu_shots": 0}


def test_replay_in_new_process_uses_only_name_cache(monkeypatch):
    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: dict(ASPIRIN))
    original = drug.ligands_from_names(None, ["aspirin"]).ligands["aspirin"]
    script = '''
import json
from chemflow import build
from chemflow.domains import drug, load_registry
from chemflow.runtime import Executor

def forbidden(*args, **kwargs):
    raise AssertionError("No network allowed in replay")

drug.fetch_pubchem = forbidden
drug.urlopen = forbidden
with build.program("replay") as builder:
    library = builder.node("ligands_from_names", names=["aspirin"])
executor = Executor(load_registry(["drug"]))
entry = executor.run(builder.to_ir())[library.id].ligands["aspirin"]
assert executor.cache_hits == 0 and not executor.cache
print(json.dumps(entry))
'''
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("DYLD_LIBRARY_PATH", None)
    output = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, check=True)
    assert json.loads(output.stdout) == {**original, "cached": True}


@pytest.mark.parametrize("payload", [b"{broken", b"{}",
                                    b'{"PropertyTable": {"Properties": []}}'])
def test_malformed_pubchem_response_is_not_cached(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(drug, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    with pytest.raises(AdapterError, match="CF-E060.*aspirin"):
        drug.ligands_from_names(None, ["aspirin"])
    assert not (tmp_path / "names.json").exists()
