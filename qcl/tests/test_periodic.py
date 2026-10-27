import pytest

import qcl

SYSTEM = {
    "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
    "atoms": "C 0 0 0; C 0.8917 0.8917 0.8917",
    "basis": "gth-szv", "pseudo": "gth-pade", "unit": "angstrom",
}


@qcl.workflow
def periodic_energy(system, device, kmesh=(1, 1, 1)):
    return qcl.periodic.scf(system, kmesh=kmesh).to(device)


def test_periodic_energy_matches_direct_pyscf():
    from pyscf.pbc import gto, scf

    plan = periodic_energy.compile(SYSTEM, qcl.Device("cpu0", "cpu"))
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "periodic.scf/chemflow-pyscf-pbc")
    result = plan.run(timeout_s=180)
    cell = gto.M(a=SYSTEM["lattice"], atom=SYSTEM["atoms"],
                 basis=SYSTEM["basis"], pseudo=SYSTEM["pseudo"],
                 unit=SYSTEM["unit"], verbose=0)
    mean_field = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv="ewald")
    mean_field.conv_tol = 1e-10
    assert result.value == pytest.approx(mean_field.kernel(), abs=1e-8)
    assert result.unit == "hartree"
    assert result.provenance["method"]["kmesh"] == [1, 1, 1]


def test_periodic_rejects_gpu_and_invalid_mesh():
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        periodic_energy.compile(SYSTEM, qcl.Device("gpu0", "gpu"))
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        periodic_energy.compile(SYSTEM, qcl.Device("cpu0", "cpu"), (1, 0, 1))
