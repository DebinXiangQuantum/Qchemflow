"""The dataflow compiler: elimination, merging, priorities, slots and measured costs."""

import json

import pytest

import qcl
from qcl import _costmodel

CPU_SIMULATOR = qcl.Device("sim0", "qpu", qubits=20, simulator_host="cpu0",
                           simulator_method="statevector")
H4 = "H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 10000\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    monkeypatch.setenv("QCL_PROFILE_CACHE", str(tmp_path / "profile.json"))


@qcl.workflow
def redundant(cpu, qpu):
    molecule = qcl.Molecule(H4, "sto-3g")
    first = qcl.scf.hf(molecule).to(cpu)
    again = qcl.scf.hf(molecule).to(cpu)          # equal to `first`: merged
    qcl.scf.hf(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")).to(cpu)  # unused: eliminated
    space = qcl.active_space(first, qcl.Active(4, 4)).to(cpu)
    kernel = qcl.kernels.lucj(space, params=qcl.ccsd.amplitudes(space).to(cpu))
    bits = [qcl.sample(kernel, shots=200, seed=seed).to(qpu) for seed in (1, 2, 3)]
    energies = [qcl.sqd.diagonalize(space.hamiltonian, item, space.sector).to(cpu)
                for item in bits]
    exact = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
    return {"again": again.energy, "first": first.energy, "exact": exact,
            "sqd": energies}


def compiled():
    cpu = qcl.Device("cpu0", "cpu", threads=24)
    return redundant.compile(cpu, CPU_SIMULATOR)


def test_schedule_eliminates_merges_and_ranks():
    plan = compiled()
    nodes = plan.graph_json()["nodes"]
    schedule = plan.schedule()
    hf = [node["id"] for node in nodes if node["op"] == "scf.hf"]
    assert schedule["aliases"] == {str(hf[1]): hf[0]} or hf[1] in map(int, schedule["aliases"])
    assert hf[2] in schedule["eliminated"]
    samples = [node["id"] for node in nodes if node["op"] == "sample"]
    assert set(samples) <= set(schedule["execute"])      # QPU jobs are never merged
    predicted = schedule["predicted"]
    assert 0 < predicted["critical_path_s"] <= predicted["makespan_s"] <= predicted["sequential_s"]
    assert predicted["critical_path"][0] in schedule["execute"]
    # Three independent SQDs on a 24-thread CPU: three slots of eight threads.
    assert schedule["slots"]["cpu0"] == 3 and schedule["threads"]["cpu0"] == 8
    ranks = schedule["priority"]
    first_sqd = next(node["id"] for node in nodes if node["op"] == "sqd.diagonalize")
    assert ranks[str(hf[0])] > ranks[str(first_sqd)]


def test_compiled_run_matches_sequential_and_skips_eliminated_nodes():
    plan = compiled()
    sequential = plan.run()
    sequential_profile = plan.profile()
    overlapped = plan.run(overlap=True)
    assert overlapped["first"].value == overlapped["again"].value
    for key in ("first", "again", "exact"):
        assert overlapped[key].value == pytest.approx(sequential[key].value, abs=1e-10)
    for left, right in zip(overlapped["sqd"], sequential["sqd"], strict=True):
        assert left.value == pytest.approx(right.value, abs=1e-10)
    executed = {entry["id"] for entry in plan.profile()}
    schedule = plan.schedule()
    assert executed.isdisjoint(schedule["eliminated"])
    assert executed.isdisjoint(map(int, schedule["aliases"]))
    assert len(executed) < len(sequential_profile)


def test_measured_costs_feed_the_next_schedule(tmp_path):
    plan = compiled()
    before = plan.schedule()["costs"]
    plan.run(overlap=True, processes=True)
    cache = json.loads((tmp_path / "profile.json").read_text())
    assert any(key.startswith("sqd.diagonalize|cpu") for key in cache)
    after = plan.schedule()["costs"]
    measured = {entry["id"]: entry["duration_s"] for entry in plan.profile()}
    sqd = next(node["id"] for node in plan.graph_json()["nodes"]
               if node["op"] == "sqd.diagonalize")
    assert after[str(sqd)] != before[str(sqd)]
    assert after[str(sqd)] == pytest.approx(
        _costmodel.predict(plan.graph_json()["nodes"])[sqd])
    assert measured[sqd] > 0


def test_slots_are_validated_and_overridable():
    with pytest.raises(qcl.CompileError, match="QCL-E202"):
        qcl.Device("cpu0", "cpu").with_options(slots=0)
    plan = compiled()
    assert plan.schedule(slots={"cpu0": 1})["slots"]["cpu0"] == 1
    assert plan.schedule(slots={"cpu0": 1})["threads"]["cpu0"] == 24
    big = redundant.compile(qcl.Device("cpu0", "cpu", threads=768), CPU_SIMULATOR)
    assert big.schedule(slots={"cpu0": 2})["threads"]["cpu0"] == _costmodel.MAX_NODE_THREADS
    with pytest.raises(ValueError, match="slots"):
        plan.schedule(slots={"cpu0": "many"})
    wide = redundant.compile(qcl.Device("cpu0", "cpu", threads=24).with_options(slots=2),
                             CPU_SIMULATOR)
    assert wide.schedule()["slots"]["cpu0"] == 2


def test_cloud_qpu_keeps_several_jobs_in_flight():
    cloud = qcl.Device("mq02", "qpu", qubits=24, site="remote", billing="metered",
                       provider="lqcloud", backend="MQ02")
    plan = redundant.compile(qcl.Device("cpu0", "cpu", threads=24), cloud)
    schedule = plan.schedule()
    assert schedule["slots"]["mq02"] == 3
    one_at_a_time = plan.schedule(slots={"mq02": 1})["predicted"]["makespan_s"]
    assert schedule["predicted"]["makespan_s"] < one_at_a_time


def test_cpu_nodes_in_lane_processes_give_the_same_results():
    plan = compiled()
    threads = plan.run(overlap=True, processes=False)
    processes = plan.run(overlap=True, processes=True)
    assert plan.schedule(processes=True)["processes"] == {"cpu0": True, "sim0": True}
    for left, right in zip(processes["sqd"], threads["sqd"], strict=True):
        assert left.value == pytest.approx(right.value, abs=1e-10)
        assert left.provenance["subspace_dimension"] == right.provenance["subspace_dimension"]
    assert processes["exact"].value == pytest.approx(threads["exact"].value, abs=1e-10)
    lane_cpu = [entry["cpu_duration_s"] for entry in plan.profile()
                if entry["op"] == "sqd.diagonalize"]
    assert all(value >= 0 for value in lane_cpu)
    with pytest.raises(TypeError, match="processes"):
        plan.schedule(processes="yes")


def test_only_isolated_measurements_feed_the_cost_model(tmp_path):
    plan = compiled()
    plan.run(overlap=True, processes=False)
    cache = tmp_path / "profile.json"
    learned = json.loads(cache.read_text()) if cache.exists() else {}
    assert not any(key.startswith("sqd.diagonalize|") for key in learned)
    plan.run(overlap=True, processes=True)
    learned = json.loads(cache.read_text())
    assert any(key.startswith("sqd.diagonalize|") for key in learned)
