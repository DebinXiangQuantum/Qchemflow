from chemflow.agent.critic import review


def test_classical_dominance_and_quantum_escalation():
    quantum = {"id": "q", "gate_passed": True, "relation": "stochastic_estimate",
               "evidence": True, "cost": {"wall_s": 10, "sim_shots": 256}}
    classical = {"id": "c", "gate_passed": True, "relation": "approximate",
                 "evidence": True, "cost": {"wall_s": 1, "sim_shots": 0}}
    assert review([quantum], [classical]).decision == "stop"
    assert "use classical" in review([quantum], [classical]).reason
    assert review([quantum], []).decision == "escalate"
