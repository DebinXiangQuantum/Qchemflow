"""Reviewer regression tests for M5b (written by Claude; implementers must not edit).

The GPU FCI kernel is array-module generic: its numpy path is checked against PySCF on any
machine; the CUDA bindings are checked on the A100 and must refuse to register without CUDA
(a GPU label is never put on a CPU computation).
"""
from __future__ import annotations

import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, mcscf, scf

from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def _integrals(atoms: str, basis: str, ncas: int, nelecas: int, spin: int = 0):
    molecule = gto.M(atom=atoms, basis=basis, spin=spin, verbose=0)
    reference = (scf.ROHF if spin else scf.RHF)(molecule).run()
    cas = mcscf.CASCI(reference, ncas, nelecas)
    h1, ecore = cas.get_h1eff()
    return h1, ao2mo.restore(1, cas.get_h2eff(), ncas), ecore


def _pyscf_spin_root(h1, eri, ecore, norb, nelec, total_spin):
    energies, vectors = fci.direct_spin1.FCI().kernel(h1, eri, norb, nelec, nroots=12,
                                                       ecore=ecore, conv_tol=1e-12)
    for energy, vector in zip(energies, vectors):
        if abs(fci.spin_op.spin_square(vector, norb, nelec)[0] -
               total_spin * (total_spin + 1)) < 1e-5:
            return energy
    raise AssertionError("reference root not found")


@pytest.mark.parametrize("nelec", [(3, 3), (4, 2), (2, 4), (3, 2)])
def test_numpy_kernel_reproduces_pyscf_sigma_and_diagonal(nelec) -> None:
    from chemflow.backends.gpu import FciKernel

    h1, eri, _ = _integrals("N 0 0 0; N 0 0 1.1", "sto-3g", 6, 6)
    kernel = FciKernel(h1, eri, 6, nelec)  # numpy array module by default
    count = fci.cistring.num_strings(6, nelec[0]) * fci.cistring.num_strings(6, nelec[1])
    vector = np.random.default_rng(3).standard_normal(count)
    reference = fci.direct_spin1.contract_2e(
        fci.direct_spin1.absorb_h1e(h1, eri, 6, nelec, 0.5), vector, 6, nelec).ravel()
    for rows_per_block in (None, 1, 5):
        np.testing.assert_allclose(kernel.sigma(vector, rows_per_block=rows_per_block),
                                   reference, atol=1e-10)
    np.testing.assert_allclose(kernel.hdiag(), fci.direct_spin1.make_hdiag(h1, eri, 6, nelec),
                               atol=1e-10)


@pytest.mark.parametrize("total_spin", [0, 1])
def test_numpy_solver_finds_the_requested_spin_root_without_penalty(total_spin) -> None:
    from chemflow.backends.gpu import FciKernel, fci_ground_state

    # O2: triplet ground state, so the singlet needs a root search in the Ms=0 sector.
    h1, eri, ecore = _integrals("O 0 0 0; O 0 0 1.21", "sto-3g", 6, 8, spin=2)
    energy, s2, diagnostics = fci_ground_state(FciKernel(h1, eri, 6, (4, 4)), ecore,
                                               total_spin=total_spin)
    reference = _pyscf_spin_root(h1, eri, ecore, 6, (4, 4), total_spin)
    assert energy == pytest.approx(reference, abs=1e-8)
    assert s2 == pytest.approx(total_spin * (total_spin + 1), abs=1e-5)
    assert diagnostics["n_roots_examined"] >= 1 and diagnostics["n_sigma"] >= 1


def _cuda() -> bool:
    from chemflow.backends.gpu import cuda_available

    return cuda_available()


def test_gpu_bindings_refuse_to_register_without_cuda() -> None:
    from chemflow.backends.gpu import register_gpu

    if _cuda():
        pytest.skip("CUDA present; covered by the A100 tests")
    with pytest.raises(RuntimeError):
        register_gpu(molecular_registry())


@pytest.mark.parametrize("atoms,total_spin", [("N 0 0 0; N 0 0 1.1", 0),
                                              ("O 0 0 0; O 0 0 1.21", 0),
                                              ("O 0 0 0; O 0 0 1.21", 1)])
def test_gpu_fci_matches_cpu_fci_with_an_honest_label(atoms, total_spin) -> None:
    if not _cuda():
        pytest.skip("needs CUDA (run on the A100)")
    from chemflow.backends.gpu import register_gpu

    registry = molecular_registry()
    register_gpu(registry)
    with build.program("gpu-fci") as builder:
        model = builder.electronic_model(basis="cc-pvdz")
        hamiltonian = builder.hamiltonian(builder.molecule(atoms), model,
                                          active={"n_elec": 8, "n_orb": 8})
        sector = builder.sector(n_alpha=4, n_beta=4, n_orb=8)  # Ms=0; spin by root search
        nodes = {}
        for binding in ("fci_pyscf", "fci_cupy"):
            node = builder.ground_energy(hamiltonian, sector, s=total_spin)
            builder.refine(node, "fci")
            builder.bind(node, binding)
            nodes[binding] = node
    result = Executor(registry).run(builder.to_ir())
    cpu, gpu = (result[nodes[name].id] for name in ("fci_pyscf", "fci_cupy"))
    assert gpu.value.value == pytest.approx(cpu.value.value, abs=1e-8)
    assert gpu.relation == cpu.relation == "equivalent"
    assert gpu.provenance.backend_kind == "gpu_numeric"
    assert gpu.provenance.device.startswith("cuda:")
    assert result["costs"][nodes["fci_cupy"].id].gpu_s > 0
    assert gpu.diagnostics["s2"] == pytest.approx(total_spin * (total_spin + 1), abs=1e-5)


def test_gpu_sampler_reproduces_the_state_distribution() -> None:
    if not _cuda():
        pytest.skip("needs CUDA (run on the A100)")
    from chemflow.backends.gpu import register_gpu
    from chemflow.impls.fci import sector_states

    registry = molecular_registry()
    register_gpu(registry)
    with build.program("gpu-sample") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("Li 0 0 0; H 0 0 1.6"), model,
                                          active={"n_elec": 2, "n_orb": 3})
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=3)
        prep = builder.node("prepare_state", hamiltonian=hamiltonian, sector=sector,
                            ansatz="lucj", init="random", seed=4)
        samples = builder.node("sample_configurations", prep=prep, shots=40000, seed=9)
        builder.bind(samples, "statevector_cupy")
    executor = Executor(registry)
    result = executor.run(builder.to_ir())
    drawn = result[samples.id]
    assert drawn.source == "quantum_simulator" and drawn.device.startswith("cuda:")
    state = executor.store.get(result[prep.id].state_ref)
    exact = {str(bits): float(abs(amplitude) ** 2)
             for bits, amplitude in zip(sector_states(result[prep.id].sector), state)}
    observed = {key: count / drawn.shots for key, count in drawn.counts.items()}
    distance = 0.5 * sum(abs(exact.get(key, 0) - observed.get(key, 0))
                         for key in set(exact) | set(observed))
    assert distance < 0.02
