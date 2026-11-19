from types import SimpleNamespace

import pytest

from chemflow.domains import load_registry
from chemflow.domains.materials import (
    base_plan,
    lattice_arrays,
    lattice_hf,
    lattice_model,
    site_reference,
)
from chemflow.runtime import Executor
from chemflow.runtime.artifacts import ArtifactStore


def test_strong_ring_reference_converges_and_checks_stability():
    h1, eri = lattice_arrays(geometry="ring", n_sites=4, U=8)
    reference = site_reference(h1, eri, 2, 2)
    assert reference.converged
    assert reference.chemflow_diagnostics["stability"]["internal"]
    settings = reference.chemflow_diagnostics["scf_settings"]
    assert settings["damping"] > 0 and settings["level_shift"] > 0
    assert reference.e_tot == pytest.approx(-0.9688706635, abs=1e-8)


def test_lattice_contamination_weakens_spin_claim():
    store = ArtifactStore()
    data = lattice_model(store, geometry="ring", n_sites=4, U=8)
    _, diagnostics, metadata = lattice_hf(store, data, data.sector.model_dump(), s=0)
    assert diagnostics["s2"] > 1e-4
    assert metadata["weakening"]["relation"] == "approximate"


def test_failed_control_in_base_plan_does_not_block_exact(monkeypatch):
    target = SimpleNamespace(kind="lattice_ground_energy", lattice={"n_sites": 2, "U": 8},
                             sector={"n_orb": 2, "n_alpha": 1, "n_beta": 1}, target_spin=0)
    plan, controls = base_plan(SimpleNamespace(spec=SimpleNamespace(target=target)))

    def fail(*args):
        raise ValueError("SCF did not converge")

    monkeypatch.setattr("chemflow.domains.materials.site_reference", fail)
    result = Executor(load_registry(["materials"])).run(plan)
    assert "SCF did not converge" in result["refused"][controls[0]]
    assert result[plan.references[0]].diagnostics["s2"] == pytest.approx(0)
