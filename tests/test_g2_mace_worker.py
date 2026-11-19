import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chemflow.diagnostics import AdapterError
from chemflow.domains import _mace_client
from chemflow.domains.baselines import DEFAULT_MACE_MODEL, _backend

WATER = {"atoms": "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"}
POSITIONS = np.array([[0., 0., .117], [0., .757, -.469], [0., -.757, -.469]])


@pytest.fixture
def fake_worker(monkeypatch):
    _mace_client.shutdown()
    original = subprocess.Popen
    processes = []

    def install(script):
        def start(command, **kwargs):
            assert command[:4] == [sys.executable, "-u", "-m", "chemflow.domains._mace_worker"]
            process = original([sys.executable, "-u", "-c", script], **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr(_mace_client.subprocess, "Popen", start)
        return processes

    yield install
    _mace_client.shutdown()
    assert all(process.poll() is not None for process in processes)


def test_reuse_model_change_and_shutdown(fake_worker):
    processes = fake_worker("""
import json, sys
print('{"ready": true}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    assert request['charge'] == request['spin'] == 0
    assert request['symbols'] == ['O', 'H', 'H']
    print(json.dumps({'energy': -0.5, 'gradient': [[0, 0, 0]] * 3}), flush=True)
""")
    for _ in range(3):
        energy, gradient = _mace_client.evaluate("model", "hash-a", ["O", "H", "H"], POSITIONS)
        assert energy == -.5 and gradient.shape == (3, 3)
    assert len(processes) == 1
    _mace_client.evaluate("model", "hash-b", ["O", "H", "H"], POSITIONS)
    assert len(processes) == 2 and processes[0].poll() == 0
    _mace_client.shutdown()
    _mace_client.shutdown()
    assert processes[1].poll() == 0


@pytest.mark.parametrize("body", [
    "sys.exit(7)",
    "print('not JSON', flush=True)",
    "print('{\"ready\": true}', flush=True); sys.stdin.readline(); sys.exit(8)",
    "print('{\"ready\": true}', flush=True); sys.stdin.readline(); print('{}', flush=True)",
])
def test_failures_include_stderr_and_reap_process(fake_worker, body):
    processes = fake_worker("import sys\nprint('worker diagnostic marker', file=sys.stderr, flush=True)\n"
                            + body)
    with pytest.raises(AdapterError, match="CF-E080.*stderr tail:.*worker diagnostic marker"):
        _mace_client.evaluate("model", "hash", ["O", "H", "H"], POSITIONS)
    assert processes[0].poll() is not None


def test_hung_worker_is_bounded_and_reaped(fake_worker, monkeypatch):
    processes = fake_worker("import time; time.sleep(60)")
    monkeypatch.setattr(_mace_client, "TIMEOUT_S", .1)
    with pytest.raises(AdapterError, match="CF-E080.*timed out"):
        _mace_client.evaluate("model", "hash", ["O", "H", "H"], POSITIONS)
    assert processes[0].poll() is not None


def test_corrupt_model_reports_worker_stderr(tmp_path):
    pytest.importorskip("mace")
    path = tmp_path / "bad.model"
    path.write_bytes(b"not a torch model")
    with pytest.raises(AdapterError, match="CF-E080") as error:
        _mace_client.evaluate(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                              ["O", "H", "H"], POSITIONS)
    assert "stderr tail:" in str(error.value) and "Traceback" in str(error.value)


def test_worker_reused_across_backend_instances():
    pytest.importorskip("mace")
    path = Path(os.environ.get("CHEMFLOW_MACE_MODEL", DEFAULT_MACE_MODEL))
    if not path.is_file():
        pytest.skip("MACE weights unavailable")
    first_atoms, first, _, _ = _backend(WATER, "mace-mp-0-small")
    first(first_atoms.positions)
    worker = _mace_client._worker
    second_atoms, second, _, _ = _backend(WATER, "mace-mp-0-small")
    second(second_atoms.positions)
    assert _mace_client._worker is worker


def test_clean_parent_imports_and_exit_cleanup():
    pytest.importorskip("mace")
    path = Path(os.environ.get("CHEMFLOW_MACE_MODEL", DEFAULT_MACE_MODEL))
    if not path.is_file():
        pytest.skip("MACE weights unavailable")
    script = f"""
import json, sys
from pyscf import gto, scf
scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)).kernel()
from chemflow.domains.baselines import _backend
from chemflow.domains import _mace_client
for method in ('gfn2-xtb', 'mace-mp-0-small', 'gfn2-xtb', 'mace-mp-0-small'):
    atoms, evaluate, _, _ = _backend({WATER!r}, method)
    evaluate(atoms.positions)
assert not any(name.split('.')[0] in ('torch', 'mace') for name in sys.modules)
print(json.dumps({{'pid': _mace_client._worker.process.pid}}))
"""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("DYLD_", "KMP_"))}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, env=environment, timeout=120, check=True)
    worker_pid = json.loads(result.stdout)["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)
