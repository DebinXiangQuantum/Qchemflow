import math

import numpy as np
import pytest
from pyscf import dft, gto, scf

import qcl
from chemflow.domains.paths import rate_constant


@qcl.workflow
def thermal_correction(molecule, cpu):
    minimum = qcl.optimize(molecule).to(cpu)
    modes = qcl.frequencies(minimum).to(cpu)
    return qcl.thermo.rrho(minimum, modes, temperature_K=298.15).to(cpu)


@qcl.workflow
def eyring_from_hf_difference(reactant, transition, cpu):
    initial = qcl.scf.rhf(reactant).to(cpu).energy
    high = qcl.scf.rhf(transition).to(cpu).energy
    difference = high - initial
    return difference, qcl.thermo.eyring(difference, 298.15).to(cpu)


def test_rrho_on_optimized_diatomic_has_finite_provenance():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    result = thermal_correction.compile(molecule, qcl.Cluster.local().cpus[0]).run()
    assert result.unit == "hartree"
    assert math.isfinite(result.value)
    assert result.provenance["method"]["family"] == "RRHO"
    assert result.provenance["method"]["n_imaginary"] == 0


def test_eyring_rate_matches_reference_formula():
    reactant = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    transition = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    barrier, rate = eyring_from_hf_difference.compile(
        reactant, transition, qcl.Cluster.local().cpus[0]).run()
    assert rate.unit == "1/s"
    assert rate.value == pytest.approx(
        rate_constant(None, barrier.value, 298.15)["k_per_s"], rel=1e-10)


def test_rrho_requires_matching_geometry_and_cpu():
    @qcl.workflow
    def wrong(molecule, other, cpu):
        optimized = qcl.optimize(molecule).to(cpu)
        modes = qcl.frequencies(optimized).to(cpu)
        unrelated = qcl.optimize(other).to(cpu)
        return qcl.thermo.rrho(unrelated, modes, temperature_K=298.15).to(cpu)

    first = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    second = qcl.Molecule("H 0 0 0; H 0 0 1.0", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        wrong.compile(first, second, qcl.Cluster.local().cpus[0])
    assert any(item["code"] == "QCL-E103" for item in error.value.diagnostics)


def test_pes_hessian_frequencies_match_direct_hf_modes():
    @qcl.workflow
    def from_hessian(molecule, cpu):
        pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                      gradient=qcl.Analytic())
        matrix = qcl.hessian(pes, molecule).to(cpu)
        return qcl.frequencies(matrix).to(cpu)

    @qcl.workflow
    def direct(molecule, cpu):
        return qcl.frequencies(molecule).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    modes = from_hessian.compile(molecule, cpu).run()
    baseline = direct.compile(molecule, cpu).run()
    assert modes.n_imaginary == baseline.n_imaginary
    assert modes.freq_wavenumber == pytest.approx(baseline.freq_wavenumber, abs=1e-5)


def test_hessian_returns_unit_bearing_tensor_matching_pyscf():
    @qcl.workflow
    def matrix(molecule, device):
        surface = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                          gradient=qcl.Analytic())
        return qcl.hessian(surface, molecule).to(device)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = matrix.compile(molecule, qcl.Cluster.local().cpus[0]).run()
    reference_molecule = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    reference = scf.RHF(reference_molecule).run(conv_tol=1e-10).Hessian().kernel()
    assert result.shape == (2, 2, 3, 3)
    assert result.unit == "hartree/bohr^2"
    np.testing.assert_allclose(result.data, reference, atol=1e-9)
    np.testing.assert_allclose(np.from_dlpack(result), reference, atol=1e-9)


def test_pbe_hessian_matches_pyscf_and_records_functional():
    @qcl.workflow
    def matrix(molecule, device):
        surface = qcl.PES(lambda geometry, lane: qcl.dft.rks(geometry, xc="pbe").to(lane),
                          gradient=qcl.Analytic())
        return qcl.hessian(surface, molecule).to(device)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = matrix.compile(molecule, qcl.Cluster.local().cpus[0])
    assert "hessian/dft-pyscf" in plan.explain()
    result = plan.run()
    reference_molecule = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    reference = dft.RKS(reference_molecule)
    reference.xc = "pbe"
    reference.kernel()
    np.testing.assert_allclose(result.data, reference.Hessian().kernel(), atol=1e-8)
    assert result.provenance["method"]["xc"] == "pbe"


def test_hessian_frequency_analysis_requires_explicit_gpu_to_cpu_copy():
    @qcl.workflow
    def frequencies(molecule, gpu, cpu, copy):
        surface = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane))
        matrix = qcl.hessian(surface, molecule).to(gpu)
        return qcl.frequencies(matrix.to(cpu) if copy else matrix).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    gpu = qcl.Device("gpu0", "gpu")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(qcl.CompileError, match="QCL-E205"):
        frequencies.compile(molecule, gpu, cpu, False, strict_transfers=True)
    plan = frequencies.compile(molecule, gpu, cpu, True, strict_transfers=True)
    assert any(item["kind"] == "explicit" and item["to"] == "cpu0"
               for item in plan.explain(format="json")["transfers"])


@pytest.mark.parametrize("xc", [None, "pbe"])
@pytest.mark.parametrize("ordinal", [0, 1])
def test_gpu_hessian_requires_explicit_cpu_copy_for_frequency_analysis(ordinal, xc):
    cupy = pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("The requested CUDA GPU is unavailable")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")

    @qcl.workflow
    def hessian(molecule, gpu):
        callback = (lambda geometry, lane: qcl.dft.rks(geometry, xc=xc).to(lane)
                    if xc else qcl.scf.hf(geometry).to(lane))
        surface = qcl.PES(callback)
        return qcl.hessian(surface, molecule).to(gpu)

    result = hessian.compile(molecule, cluster.gpus[ordinal]).run()
    reference_molecule = gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)
    calculation = dft.RKS(reference_molecule) if xc else scf.RHF(reference_molecule)
    if xc:
        calculation.xc = xc
    reference = calculation.run(conv_tol=1e-10).Hessian().kernel()
    assert result.__dlpack_device__() == (2, ordinal)
    assert result.provenance["implementation"] == (
        "hessian/dft-gpu4pyscf" if xc else "hessian/gpu4pyscf")
    cupy.testing.assert_allclose(result.data, cupy.asarray(reference),
                                 rtol=0, atol=1e-7 if xc else 1e-9)

    @qcl.workflow
    def frequencies(molecule, gpu, cpu):
        callback = (lambda geometry, lane: qcl.dft.rks(geometry, xc=xc).to(lane)
                    if xc else qcl.scf.hf(geometry).to(lane))
        surface = qcl.PES(callback)
        matrix = qcl.hessian(surface, molecule).to(gpu).to(cpu)
        return qcl.frequencies(matrix).to(cpu)

    modes = frequencies.compile(molecule, cluster.gpus[ordinal], cluster.cpus[0],
                                strict_transfers=True).run()
    assert modes.n_imaginary == 0
