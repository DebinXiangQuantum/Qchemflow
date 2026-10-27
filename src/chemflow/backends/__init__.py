"""Optional hardware and calibrated simulation implementations."""

QPU_CATALOG = {
    "ground_energy": {
        "vqe_lqcloud": {"qubits": 24, "min_shots": 1, "max_shots": 50000},
    },
    "sample_configurations": {
        "lqcloud_mq02": {"qubits": 24, "min_shots": 1, "max_shots": 50000},
    },
}
