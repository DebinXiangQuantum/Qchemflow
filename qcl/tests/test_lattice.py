import math

import pytest
from qcl._worker import lattice_observable_value

import qcl
from chemflow.domains.lattice_correlations import correlation_matrices
from chemflow.types import Sector as ChemflowSector


@qcl.workflow
def hubbard_pairing(lattice, U, sector, i, j, dev):
    operator = qcl.models.hubbard(lattice, t=1.0, U=U)
    ground = qcl.ed.ground_state(operator, sector=sector).to(dev)
    pair = qcl.expect(ground, qcl.ops.onsite_pair(i, j)).to(dev)
    spin = qcl.expect(ground, qcl.ops.spin_spin(i, j)).to(dev)
    return ground.energy, pair, spin


def test_hubbard_dimer_energy_and_correlations():
    cpu = qcl.Cluster.local().cpus[0]
    plan = hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                   qcl.Sector(n=2, sz=0), 0, 1, cpu)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "ed.ground_state", "expect", "expect",
    ]
    energy, pair, spin = plan.run()
    assert energy.value == pytest.approx(2 - math.sqrt(8), abs=1e-10)
    assert energy.unit == "hartree"
    assert pair.unit == spin.unit == "dimensionless"
    assert spin.value == pytest.approx(-0.75 * (0.5 + 1 / math.sqrt(8)), abs=1e-10)
    assert pair.value == pytest.approx((1 - 1 / math.sqrt(2)) / 4, abs=1e-10)
    assert energy.provenance["method"]["family"] == "Hubbard ED"


def test_square_lattice_shape_and_sector_validation():
    assert qcl.lattice.square(2, 4).n_sites == 8
    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError) as error:
        hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                qcl.Sector(n=3, sz=1), 0, 1, cpu)
    assert error.value.diagnostics[0]["code"] == "QCL-E105"
    with pytest.raises(qcl.CompileError) as error:
        hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                qcl.Sector(n=2, sz=0), 0, 2, cpu)
    assert error.value.diagnostics[0]["code"] == "QCL-E105"


def test_hubbard_device_is_checked_before_execution():
    with pytest.raises(qcl.CompileError) as error:
        hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                qcl.Sector(n=2, sz=0), 0, 1,
                                qcl.Cluster.local().qpus[0])
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_number_observable_is_particle_density_not_density_squared():
    @qcl.workflow
    def density(dev):
        operator = qcl.models.hubbard(qcl.lattice.chain(2), t=1, U=4)
        ground = qcl.ed.ground_state(operator, sector=qcl.Sector(n=2, sz=0)).to(dev)
        return qcl.expect(ground, qcl.ops.number(0)).to(dev)

    result = density.compile(qcl.Cluster.local().cpus[0]).run()
    assert result.value == pytest.approx(1.0, abs=1e-10)


def test_selected_lattice_observables_match_chemflow_matrices():
    import numpy as np

    sector = ChemflowSector(n_alpha=1, n_beta=1, n_orb=3)
    vector = np.arange(1, 10, dtype=float)
    vector /= np.linalg.norm(vector)
    matrices = correlation_matrices(vector, sector)
    for kind in ("pair_s_wave", "spin_spin", "density_density"):
        for left, right in ((0, 1), (1, 1), (2, 0)):
            value = lattice_observable_value(vector, sector, kind, left, right, np)
            assert value == pytest.approx(matrices[kind][left][right], abs=1e-12)
    number = lattice_observable_value(vector, sector, "number", 1, 1, np)
    assert number == pytest.approx(matrices["density_density"][1][1] -
                                   2 * matrices["pair_s_wave"][1][1], abs=1e-12)


def test_hubbard_workflow_compiles_for_gpu_device():
    gpu = qcl.Device("gpu0", "gpu", memory_gb=16)
    plan = hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                   qcl.Sector(n=2, sz=0), 0, 1, gpu)
    assert [item["device"] for item in plan.explain(format="json")["operations"]] == [
        "gpu0", "gpu0", "gpu0",
    ]


def test_gpu_ed_path_matches_cpu_with_numpy_device_double(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np
    from qcl._worker import execute

    class DeviceDouble:
        def __init__(self, _index=0):
            self.id = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def synchronize(self):
            pass

    class CupyDouble:
        __name__ = "numpy"
        cuda = SimpleNamespace(Device=DeviceDouble)

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setitem(sys.modules, "cupy", CupyDouble())
    gpu = qcl.Device("gpu0", "gpu", memory_gb=16)
    plan = hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                   qcl.Sector(n=2, sz=0), 0, 1, gpu)
    output = {"type": "tuple", "items": [
        {"type": "node", "id": value.node_id} for value in plan._output]}
    result = execute({"graph": plan.graph_json(), "output": output})["result"]
    reference = hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                        qcl.Sector(n=2, sz=0), 0, 1,
                                        qcl.Device("cpu0", "cpu")).run()
    assert [item["value"] for item in result] == pytest.approx(
        [item.value for item in reference], abs=1e-9)
    assert result[0]["provenance"]["implementation"] == "ed.ground_state/chemflow-cupy"


def test_hubbard_gpu_memory_shortage_is_a_compile_error():
    tiny_gpu = qcl.Device("gpu0", "gpu", memory_gb=1e-9)
    with pytest.raises(qcl.CompileError) as error:
        hubbard_pairing.compile(qcl.lattice.chain(2), 4.0,
                                qcl.Sector(n=2, sz=0), 0, 1, tiny_gpu)
    assert any(item["code"] == "QCL-E209" for item in error.value.diagnostics)
