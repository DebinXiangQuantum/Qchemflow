import numpy as np
import pytest

from chemflow import build
from chemflow.diagnostics import AdapterError
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.ops.molecular import _ground_adapter, ground_energy, molecular_registry
from chemflow.runtime import Executor
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ElectronicModel


@pytest.mark.parametrize("method", ["hf", "mp2", "ccsd", "ccsd_t"])
def test_high_spin_controls_report_spin(method):
    store = ArtifactStore()
    data = build_molecule("Li 0 0 0", ElectronicModel(basis="sto-3g"), store, spin=1)
    energy, diagnostics, metadata = _ground_adapter(method)(
        store, data, data.sector.model_dump(), s=0.5)
    assert np.isfinite(energy)
    assert diagnostics["reference"] == "ROHF"
    assert np.isfinite(diagnostics["s2"])
    if abs(diagnostics["s2"] - 0.75) > 1e-4:
        assert metadata["weakening"]["relation"] == "approximate"


def test_low_spin_control_refused_but_exact_continues():
    with build.program("low-spin") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                  builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        control = builder.ground_energy(ham, sector, s=1)
        builder.refine(control, "hf")
        builder.bind(control, "hf")
        builder.ir.nodes[-1].metadata["optional_control"] = True
        exact = builder.ground_energy(ham, sector, s=1)
        builder.refine(exact, "fci")
        builder.bind(exact, "fci_pyscf")
        builder.ir.references = [exact.id]
    computed = Executor(molecular_registry()).run(builder.to_ir())
    assert "CF-E020" in computed["refused"][control.id]
    assert computed[exact.id].diagnostics["s2"] == pytest.approx(2)
    assert computed["costs"][control.id].wall_s >= 0


def test_low_spin_direct_call_still_raises():
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 0.74", ElectronicModel(basis="sto-3g"), store)
    with pytest.raises(AdapterError, match="CF-E020"):
        ground_energy(data, data.sector.model_dump(), store, "mp2", s=1)


def test_contaminated_hf_relation_is_weakened(monkeypatch):
    def contaminated(data, method, store, diagnostics):
        diagnostics["s2"] = 0.76
        return -1.

    monkeypatch.setattr("chemflow.ops.molecular.classical_energy", contaminated)
    store = ArtifactStore()
    data = build_molecule("Li 0 0 0", ElectronicModel(basis="sto-3g"), store, spin=1)
    _, _, metadata = _ground_adapter("hf")(store, data, data.sector.model_dump(), s=0.5)
    assert metadata["weakening"]["relation"] == "approximate"
