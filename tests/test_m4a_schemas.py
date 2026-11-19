from test_m3b_llm import policy


def test_per_tool_strict_schemas_and_neutral_messages(tmp_path):
    agent, client, _ = policy(tmp_path, "designer", [
        {"tool_calls": [{"id": "use-1", "name": "plan.new", "arguments": {
            "identifier": "trial"}}, {"id": "use-2", "name": "run", "arguments": {
            "scale": "production"}}], "content": ""},
        {"tool_calls": [], "content": "done"}])
    schemas = {item["name"]: item["parameters"] for item in agent.tools.schemas("designer")}
    assert set(schemas["plan.new"]["properties"]) == {"identifier"}
    assert set(schemas["run"]["properties"]) == {"plan_id", "scale"}
    for schema in schemas.values():
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert all("description" in value for value in schema["properties"].values())
    agent.decide({"diagnostics": {"x": 1}})
    assert client.messages[0][0]["role"] == "system"
    assert "diagnostics" not in client.messages[0][0]["content"]
    assert "diagnostics" in client.messages[0][1]["content"]
    results = [item for item in client.messages[1] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in results] == ["use-1", "use-2"]
    assert [item["is_error"] for item in results] == [False, True]
