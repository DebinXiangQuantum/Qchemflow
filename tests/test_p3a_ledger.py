import json

from chemflow.cli import main


def test_ledger_lists_resolved_states_and_per_backend_resources(tmp_path, monkeypatch, capsys):
    for index, state in enumerate(("prepared", "submitted", "abandoned", "rejected",
                                    "failed", "completed")):
        record = {"key": str(index), "backend": "MQ02", "state": state, "shots": 2048,
                  "predicted_cost": {"qpu_s": 1.95}, "cost": 15.24, "currency": "CNY"}
        if state == "completed":
            record["status_info"] = {"execution_time": 5.1, "shots": 2048}
        (tmp_path / f"{index}.json").write_text(json.dumps(record))
    (tmp_path / "other.json").write_text(json.dumps({
        "key": "other", "backend": "other", "state": "completed", "shots": 128,
        "status_info": {"execution_time": 3}}))
    monkeypatch.setattr("sys.argv", ["chemflow", "qpu", "ledger", "--state-dir", str(tmp_path)])
    main()
    text = capsys.readouterr().out
    assert "total MQ02: qpu_shots=4096 qpu_s=7.05; unresolved 2" in text
    assert "total other: qpu_shots=128 qpu_s=3; unresolved 0" in text
    for state in ("prepared", "submitted", "abandoned", "rejected", "failed", "completed"):
        assert state in text
    assert not any(word in text for word in ("money", "currency", "CNY", "qpu_jobs"))
