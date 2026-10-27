import numpy as np
import pytest
from pyscf import dft, gto, scf

import qcl

MOLECULE = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


def hf_surface(geometry, lane):
    return qcl.scf.hf(geometry).to(lane)


@qcl.workflow
def hf_gradient(molecule, device):
    surface = qcl.PES(hf_surface, gradient=qcl.Analytic())
    return qcl.gradient(surface, molecule).to(device)


@qcl.workflow
def finite_difference_gradient(molecule, device, step=1e-3):
    surface = qcl.PES(hf_surface, gradient=qcl.FiniteDifference(step=step))
    return qcl.gradient(surface, molecule).to(device)


@qcl.workflow
def pbe_gradient(molecule, device):
    surface = qcl.PES(lambda geometry, lane: qcl.dft.rks(geometry, xc="pbe").to(lane),
                      gradient=qcl.Analytic())
    return qcl.gradient(surface, molecule).to(device)


@qcl.workflow
def pbe_gradient_fd(molecule, device):
    surface = qcl.PES(lambda geometry, lane: qcl.dft.rks(geometry, xc="pbe").to(lane),
                      gradient=qcl.FiniteDifference(step=1e-3))
    return qcl.gradient(surface, molecule).to(device)


def test_cpu_gradient_matches_pyscf_and_energy_difference():
    result = hf_gradient.compile(MOLECULE, CPU).run()
    system = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    mean_field = scf.RHF(system).run(conv_tol=1e-10)
    np.testing.assert_allclose(result.data, mean_field.nuc_grad_method().kernel(),
                               atol=1e-9)
    distance = system.atom_coords(unit="Bohr")[1, 2]
    step = 1e-4

    def energy(separation):
        moved = gto.M(atom=[("H", (0, 0, 0)), ("H", (0, 0, separation))],
                      basis="sto-3g", unit="Bohr", verbose=0)
        return float(scf.RHF(moved).run(conv_tol=1e-10).e_tot)

    difference = (energy(distance + step) - energy(distance - step)) / (2 * step)
    assert result.shape == (2, 3)
    assert result.unit == "hartree/bohr"
    assert result.data[1, 2] == pytest.approx(difference, abs=1e-7)
    assert result.provenance["implementation"] == "gradient/pyscf"


def test_finite_difference_gradient_matches_analytic_cpu_reference():
    analytic = hf_gradient.compile(MOLECULE, CPU).run()
    plan = finite_difference_gradient.compile(MOLECULE, CPU)
    assert "gradient/fd-pyscf" in plan.explain()
    finite = plan.run()
    np.testing.assert_allclose(finite.data, analytic.data, atol=1e-5)
    assert finite.provenance["method"]["step_bohr"] == 1e-3
    assert finite.provenance["implementation"] == "gradient/fd-pyscf"


def test_pbe_gradient_matches_pyscf_and_records_functional():
    plan = pbe_gradient.compile(MOLECULE, CPU)
    assert "gradient/dft-pyscf" in plan.explain()
    result = plan.run()
    system = gto.M(atom=MOLECULE.atoms, basis=MOLECULE.basis, verbose=0)
    reference = dft.RKS(system)
    reference.xc = "pbe"
    reference.kernel()
    np.testing.assert_allclose(result.data, reference.nuc_grad_method().kernel(), atol=1e-8)
    assert result.provenance["method"]["xc"] == "pbe"


def test_pbe_finite_difference_gradient_matches_analytic():
    plan = pbe_gradient_fd.compile(MOLECULE, CPU)
    assert "gradient/fd-dft-pyscf" in plan.explain()
    finite = plan.run()
    analytic = pbe_gradient.compile(MOLECULE, CPU).run()
    np.testing.assert_allclose(finite.data, analytic.data, atol=1e-5)
    assert finite.provenance["method"]["xc"] == "pbe"


def test_gradient_rejects_unsupported_pes_at_compile_time():
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        finite_difference_gradient.compile(MOLECULE, CPU, step=0)

    @qcl.workflow
    def wrong_model(molecule, device):
        surface = qcl.PES(lambda geometry, lane:
                          qcl.scf.uhf(geometry).to(lane))
        return qcl.gradient(surface, molecule).to(device)

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        wrong_model.compile(MOLECULE, CPU)


def test_gpu_gradient_matches_cpu_and_copies_explicitly():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")
    gpu = cluster.gpus[0]
    plan = hf_gradient.compile(MOLECULE, gpu, strict_transfers=True)
    result = plan.run()
    reference = hf_gradient.compile(MOLECULE, cluster.cpus[0]).run()
    assert isinstance(result.data, cupy.ndarray)
    assert result.__dlpack_device__()[0] == 2
    assert result.provenance["implementation"] == "gradient/gpu4pyscf"
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference.data), atol=1e-7)

    finite = finite_difference_gradient.compile(MOLECULE, gpu).run()
    cupy.testing.assert_allclose(finite.data, cupy.asarray(reference.data), atol=1e-5)
    assert finite.provenance["implementation"] == "gradient/fd-gpu4pyscf"

    @qcl.workflow
    def copied(molecule, gpu, cpu):
        surface = qcl.PES(hf_surface)
        return qcl.gradient(surface, molecule).to(gpu).to(cpu)

    copied_result = copied.compile(MOLECULE, gpu, cluster.cpus[0],
                                   strict_transfers=True).run()
    assert isinstance(copied_result.data, np.ndarray)
    np.testing.assert_allclose(copied_result.data, reference.data, atol=1e-7)


def test_gpu_gradient_uses_second_visible_device():
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) < 2:
        pytest.skip("Two CUDA GPUs are required")

    result = hf_gradient.compile(MOLECULE, cluster.gpus[1],
                                 strict_transfers=True).run()
    reference = hf_gradient.compile(MOLECULE, cluster.cpus[0]).run()
    assert result.device == "gpu1"
    assert result.__dlpack_device__() == (2, 1)
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference.data), atol=1e-7)


@pytest.mark.parametrize("ordinal", [0, 1])
def test_pbe_gradient_on_gpu_matches_cpu(ordinal):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("The requested CUDA GPU is unavailable")
    plan = pbe_gradient.compile(MOLECULE, cluster.gpus[ordinal])
    result = plan.run()
    reference = pbe_gradient.compile(MOLECULE, cluster.cpus[0]).run()
    assert result.provenance["implementation"] == "gradient/dft-gpu4pyscf"
    assert result.device == f"gpu{ordinal}"
    assert result.__dlpack_device__() == (2, ordinal)
    assert plan.profile()[-1]["device"] == f"gpu{ordinal}"
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference.data), atol=1e-7)
