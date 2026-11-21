import json
from pathlib import Path

from chemflow.domains.synthesis import parse_routes

FIXTURE = Path(__file__).parent / "fixtures/m6b/paracetamol.json.gz"


def test_recorded_plain_json_routes():
    routes = parse_routes(json.loads(FIXTURE.read_text()))
    assert len(routes) == 15
    assert routes[0]["solved"]
    assert routes[0]["leaves"] == ["CC(=O)OC(C)=O", "Nc1ccc(O)cc1"]
    assert routes[0]["in_stock"] == [True, True]
    assert routes[0]["step_count"] == 1
    assert routes[0]["scores"]["state score"] > 0.99
