
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
