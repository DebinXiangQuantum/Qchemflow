import json
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from chemflow.backends.aer_noise import noise_model
from chemflow.backends.calibration import best_layout, load_calibration
from chemflow.backends.circuits import build_circuit, compile_native, memory_to_index
from chemflow.backends.lqcloud import LqcloudBackend, load_pricing
from chemflow.cli import main
from chemflow.impls.fci import sector_states
from chemflow.impls.lucj import LucjParams, lucj_state
from chemflow.types import CircuitSpec, Sector


@pytest.mark.parametrize("sector", [(1, 1, 2), (2, 1, 3), (2, 2, 4)])
def test_hardware_circuit_matches_sector_state(sector):
    alpha, beta, orbitals = sector
    rng = np.random.default_rng(13)
    target = Sector(n_alpha=alpha, n_beta=beta, n_orb=orbitals)
    for ansatz in ("hf", "lucj"):
        layers = [LucjParams.random(orbitals, rng, scale=.3) for _ in range(2)]
        spec = CircuitSpec(ansatz=ansatz, n_orb=orbitals, n_alpha=alpha, n_beta=beta,
                           layers=tuple(tuple(layer.to_vector()) for layer in layers)
                           if ansatz == "lucj" else ())
        expected = np.zeros(1 << (2 * orbitals), dtype=complex)
        if ansatz == "hf":
            expected[(1 << alpha) - 1 | ((1 << beta) - 1) << orbitals] = 1
        else:
            expected[sector_states(target)] = lucj_state(target, layers)
        circuit = build_circuit(spec)
        assert abs(np.vdot(expected, Statevector(circuit).data)) ** 2 == pytest.approx(1, abs=1e-9)
        native = compile_native(circuit)
        assert set(native.count_ops()) <= {"rz", "sx", "x", "cz", "measure", "barrier"}
        compiled = Statevector(native.remove_final_measurements(inplace=False))
        layout = native.layout.final_index_layout()
        permuted = np.zeros_like(expected)
        for logical_index, amplitude in enumerate(expected):
            physical_index = sum(((logical_index >> logical) & 1) << physical
                                 for logical, physical in enumerate(layout))
            permuted[physical_index] = amplitude
        assert abs(np.vdot(permuted, compiled.data)) ** 2 == pytest.approx(1, abs=1e-9)


def test_bit_order_noise_snapshot_and_pricing():
    assert memory_to_index("10 10", q0_left=True) == 5
    assert memory_to_index("10 10", q0_left=False) == 10
    with pytest.raises(ValueError):
        memory_to_index("01x0", q0_left=False)
    snapshot = load_calibration()
    assert best_layout(snapshot, 4) == [19, 20, 21, 22]
    assert set(noise_model(snapshot, [19, 20, 21, 22]).noise_qubits) == {0, 1, 2, 3}
    snapshot["name"] = "mutated"
    assert load_calibration()["name"] == "MQ02"
    pricing = load_pricing()
    assert pricing.currency == "CNY" and pricing.per_task == 5


def test_ledger_and_human_reconcile(tmp_path, monkeypatch, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    key = "a" * 64
    (state_dir / f"{key}.json").write_text(json.dumps({
        "key": key, "state": "prepared", "shots": 2048, "currency": "CNY",
        "predicted_cost": {"qpu_s": 1.95, "qpu_shots": 2048}, "created": "2026-10-01"}))
    monkeypatch.setattr(sys, "argv", ["chemflow", "qpu", "ledger", "--state-dir", str(state_dir)])
    main()
    assert "unresolved 1" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["chemflow", "qpu", "reconcile", key,
                                    "--job-id", "found-job", "--state-dir", str(state_dir)])
    main()
    assert json.loads((state_dir / f"{key}.json").read_text())["job_id"] == "found-job"
    LqcloudBackend(state_dir=state_dir).reconcile(key, abandon=True)
    assert not (state_dir / f"{key}.json").exists()
    assert len(list((state_dir / "archive").glob("*.json"))) == 1
