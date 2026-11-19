import json

from test_m3a_task import spec

from chemflow.agent.task import TaskRecord
from chemflow.cli import main
from chemflow.signing import TaskVerifier


def test_human_freeze(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHEMFLOW_KEY_PASSPHRASE", "human-key")
    directory = tmp_path / "keys"
    monkeypatch.setattr("sys.argv", ["chemflow", "keys", "init", "--dir", str(directory)])
    main()
    capsys.readouterr()
    source = tmp_path / "spec.yaml"
    source.write_text(spec().model_dump_json())
    task = tmp_path / "task.json"
    monkeypatch.setattr("sys.argv", ["chemflow", "task", "draft", "--from", str(source)])
    main()
    assert json.loads(capsys.readouterr().out)["question"] == spec().question
    monkeypatch.setattr("sys.argv", ["chemflow", "--key-dir", str(directory), "task", "freeze", "--from", str(source),
                                     "--approver", "human", "--out", str(task)])
    main()
    verifier = TaskVerifier((directory / "public.pem").read_bytes())
    verifier.verify(TaskRecord.model_validate_json(task.read_text()))
