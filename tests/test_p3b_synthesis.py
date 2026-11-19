import subprocess

import pytest

from chemflow.diagnostics import AdapterError
from chemflow.domains import synthesis


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    (tmp_path / "env/bin").mkdir(parents=True)
    (tmp_path / "env/bin/aizynthcli").touch()
    metadata = tmp_path / "env/lib/python3.11/site-packages/aizynthfinder-4.4.1.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Version: 4.4.1\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/stock.hdf5").write_bytes(b"stock")
    (tmp_path / "data/config.yml").write_text("stock:\n  zinc: stock.hdf5\n")
    monkeypatch.setenv("CHEMFLOW_AIZYNTH_HOME", str(tmp_path))


@pytest.mark.parametrize("status,signal_name", [(1, None), (-9, "SIGKILL"),
                                               (-15, "SIGTERM"), (-999, "signal 999")])
@pytest.mark.parametrize("stderr,stdout", [("", ""), ("", "Loading expansion policy...\n"),
                                          ("error details", "progress")])
def test_failure_status_and_output_tails(isolated_home, monkeypatch, status, signal_name,
                                        stderr, stdout):
    monkeypatch.setattr(synthesis.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, status, stdout, stderr))
    with pytest.raises(AdapterError) as info:
        synthesis.search_routes(None, "CC(=O)Nc1ccc(O)cc1", iterations=1, time_limit_s=5)
    message = str(info.value)
    assert "CF-E080" in message
    assert f"exit status {status}" in message
    if signal_name:
        assert signal_name in message
    assert f"stderr tail: {stderr or '<empty>'}" in message
    assert f"stdout tail: {stdout or '<empty>'}" in message


def test_failure_output_is_bounded(isolated_home, monkeypatch):
    monkeypatch.setattr(synthesis.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 2,
                                                    "stdout prefix" + "o" * 2500,
                                                    "stderr prefix" + "e" * 2500))
    with pytest.raises(AdapterError) as info:
        synthesis.search_routes(None, "CCO")
    message = str(info.value)
    assert "prefix" not in message
    assert "o" * 2000 in message and "e" * 2000 in message
    assert "o" * 2001 not in message and "e" * 2001 not in message
