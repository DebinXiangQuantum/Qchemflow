"""A rerun whose circuit angles differ only by floating-point noise reuses the paid job."""

import pytest

from chemflow.backends.lqcloud import LqcloudBackend
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import CircuitSpec, Sector, StatePrep


class Job:
    def __init__(self, job_id, shots):
        self.job_id, self.shots, self._params_resubmit = job_id, shots, object()

    def result(self, timeout=None, wait=2, verbose=True):
        from types import SimpleNamespace

        return SimpleNamespace(success=True, data={"memory": ["1010"] * self.shots})

    def status_info(self):
        return {"status": "completed", "cost": 5.0, "currency": "CNY",
                "execution_time": 1.0, "error_type": None, "shots": self.shots}


class Cloud:
    def __init__(self):
        from chemflow.backends.calibration import load_calibration

        self.config = {**load_calibration("MQ02"), "status": "active",
                       "billing_multiplier": 1.0}
        self.submissions = []

    def provider_factory(self, key):
        cloud = self

        class Device:
            config = cloud.config

            def run(self, program, shots=1024, **kwargs):
                cloud.submissions.append(shots)
                return Job(f"job-{len(cloud.submissions)}", shots)

        class Provider:
            def get_backend(self, name):
                return Device()

        return Provider()

    def job_factory(self, job_id, provider, backend_name):
        return Job(job_id, 100)


def prep(angles, state="state-a"):
    sector = Sector(n_alpha=1, n_beta=1, n_orb=2)
    circuit = CircuitSpec(ansatz="lucj", n_orb=2, n_alpha=1, n_beta=1, layers=(tuple(angles),))
    return StatePrep(state_ref=state, encoding_id=f"encoding-{state}",
                     hamiltonian_id=f"hamiltonian-{state}", recipe="lucj", sector=sector,
                     circuit=circuit)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from chemflow.backends import lqcloud

    monkeypatch.setenv("LQCLOUD_API_KEY", "test-key")
    # Job reuse is under test, not program conversion (which needs the lqcloud SDK).
    monkeypatch.setattr(lqcloud, "to_lqcloud", lambda circuit: "program")
    cloud = Cloud()
    qpu = LqcloudBackend(state_dir=tmp_path / "qpu", provider_factory=cloud.provider_factory,
                         job_factory=cloud.job_factory)
    return qpu, cloud


def test_noise_level_angle_changes_reuse_the_recorded_job(backend):
    qpu, cloud = backend
    store = ArtifactStore()
    first = qpu.sample(store, prep([0.10, 0.20, 0.30]), 100)
    # Another run: SCF/CCSD noise moves the angles by 1e-10 and every content hash.
    again = qpu.sample(store, prep([0.10 + 1e-10, 0.20, 0.30 - 1e-10], state="state-b"), 100)
    assert cloud.submissions == [100]
    assert again.job_ids == first.job_ids == ("job-1",)
    assert again.cost == {"qpu_shots": 0, "qpu_s": 0}
    assert again.value.encoding_id == "encoding-state-b"


def test_a_different_circuit_or_shot_count_is_a_new_job(backend):
    qpu, cloud = backend
    store = ArtifactStore()
    qpu.sample(store, prep([0.10, 0.20, 0.30]), 100)
    qpu.sample(store, prep([0.10, 0.20, 0.30 + 1e-3]), 100)
    qpu.sample(store, prep([0.10, 0.20, 0.30]), 101)
    qpu.sample(store, prep([0.10, 0.20, 0.30]), 100, repetition=1)
    assert cloud.submissions == [100, 100, 101, 100]
