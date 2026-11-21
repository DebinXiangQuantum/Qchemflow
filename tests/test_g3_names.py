import io
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

from chemflow.diagnostics import AdapterError
from chemflow.domains import drug

ASPIRIN = {"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
           "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
WATER = {"CID": 962, "SMILES": "O", "InChIKey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N"}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMFLOW_NAME_CACHE", str(tmp_path / "names.json"))

    def unavailable(*args, **kwargs):
        raise URLError("offline test: transport not injected")

    monkeypatch.setattr(drug, "urlopen", unavailable)


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


def test_inconsistent_cached_record_is_refused(tmp_path):
    cache = tmp_path / "names.json"
    cache.write_text(json.dumps({"aspirin": {**ASPIRIN, "InChIKey": WATER["InChIKey"],
                                            "retrieved_at": "2026-10-13T00:00:00+00:00"}}))
    original = cache.read_bytes()
    with pytest.raises(AdapterError, match="CF-E060.*aspirin"):
        drug.ligands_from_names(None, ["aspirin"])
    assert cache.read_bytes() == original


def test_replay_in_new_process_uses_only_name_cache(monkeypatch):
    monkeypatch.setattr(drug, "fetch_pubchem", lambda name: dict(ASPIRIN))
    original = drug.ligands_from_names(None, ["aspirin"]).ligands["aspirin"]
    script = '''
import json
from chemflow.domains import drug

def forbidden(*args, **kwargs):
    raise AssertionError("No network allowed in replay")

drug.fetch_pubchem = forbidden
drug.urlopen = forbidden
print(json.dumps(drug.ligands_from_names(None, ["aspirin"]).ligands["aspirin"]))
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
