
import pytest

from chemflow.domains.materials import lattice_arrays, site_reference


def test_strong_ring_reference_converges_and_checks_stability():
    h1, eri = lattice_arrays(geometry="ring", n_sites=4, U=8)
    reference = site_reference(h1, eri, 2, 2)
    assert reference.converged
    assert reference.chemflow_diagnostics["stability"]["internal"]
    settings = reference.chemflow_diagnostics["scf_settings"]
    assert settings["damping"] > 0 and settings["level_shift"] > 0
    assert reference.e_tot == pytest.approx(-0.9688706635, abs=1e-8)


def test_a_failed_diis_extrapolation_falls_back_to_damped_scf(monkeypatch):
    from pyscf.lib import diis

    original = diis.DIIS.update
    failures = []

    def fail_once(self, *args, **kwargs):
        if not failures:
            failures.append(True)
            raise AttributeError("module 'numpy.linalg' has no attribute 'linalg'")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(diis.DIIS, "update", fail_once)
    h1, eri = lattice_arrays(geometry="ring", n_sites=4, U=8)
    reference = site_reference(h1, eri, 2, 2)
    assert failures and reference.converged
    assert reference.chemflow_diagnostics["scf_settings"]["diis"] is False
    assert reference.e_tot == pytest.approx(-0.9688706635, abs=1e-8)
