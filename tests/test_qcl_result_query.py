from chemflow.agent.memory import ResultsDB


def test_existing_result_queries_can_find_qcl_runs():
    database = ResultsDB()
    identifier = database.add_qcl_run(
        record_hash="signed-task", plan_hash="qcl-plan",
        target_kind="molecular_ground_energy", decision="gate_passed",
        result={"value": -1.13, "unit": "hartree", "provenance": {}},
        profile=[{"op": "scf.rhf", "device": "cpu0"}],
        spent={"wall_s": 1.2})

    row = database.get(identifier)
    assert row["kind"] == "qcl"
    assert row["result"]["value"] == -1.13
    assert row["profile"][0]["op"] == "scf.rhf"
    assert database.query(record_hash="signed-task") == [row]
    assert database.query(plan_hash="qcl-plan") == [row]
    assert database.query(target_kind="molecular_ground_energy") == [row]
    assert database.query(operator="ground_energy") == []
    assert database.query() == [row]
