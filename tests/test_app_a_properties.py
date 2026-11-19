from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemflow import build
from chemflow.agent.families import default_axes, instantiate
from chemflow.agent.task import FrontierTarget, IonizationTarget, SolvationTarget
from chemflow.check import check
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.domains import load_registry, pack_names
from chemflow.domains.properties import base_plan, identity
from chemflow.runtime import Executor
from chemflow.verify import check_certificate, prove

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def record(target):
    return SimpleNamespace(spec=SimpleNamespace(target=target))


def frontier(**changes):
    return FrontierTarget.model_validate({"kind": "frontier_orbitals",
        "molecule": {"atoms": WATER}, "electronic_model": {"basis": "6-31g"},
        "level": {"method": "hf", "functional": None, "dispersion": None},
        "quantity": "gap", "spin": "alpha", **changes})


def ionization(**changes):
    return IonizationTarget.model_validate({"kind": "ionization_energy",
        "molecule": {"atoms": "Li 0 0 0; H 0 0 1.6"},
        "electronic_model": {"basis": "sto-3g"}, "final_spin": 0.5, **changes})


def test_reference_only_model_has_replayable_certificate_and_declared_model():
    from chemflow.agent.report import result_headline

    registry = load_registry(pack_names())
    target = frontier()
    plan, controls = base_plan(record(target))
    assert controls == [] and plan.targets == plan.references
    assert not check(plan, registry)
    computed = Executor(registry).run(plan)
    node = plan.targets[0]
    headline = result_headline(node, computed, {}, node,
        SimpleNamespace(target=target, gates=(), validity=()), plan, registry)
    assert headline["classical_nodes"] == [node]
    assert "model-defined" in headline["quantity_label"]
    assert computed[node].diagnostics["declared_model"] == plan.nodes[0].params
    certificate = prove(plan, node, registry=registry, computed=computed)
    assert certificate["relation"] == "equivalent"
    assert check_certificate(certificate, plan, registry=registry, computed=computed).valid
    computed[node].diagnostics["declared_model"]["quantity"] = "homo"
    assert not check_certificate(certificate, plan, registry=registry, computed=computed).valid


@pytest.mark.parametrize("changes", [{"level": {"method": "dft"}},
    {"level": {"method": "hf", "dispersion": "d4"}}, {"spin": "gamma"},
    {"quantity": "ip"}, {"unexpected": True}, {"level": {"method": "gfn2"}}])
def test_frontier_rejects_unsupported_schema(changes):
    with pytest.raises(ValidationError):
        frontier(**changes)


def test_r136_nullable_level_and_invalid_value():
    registry = load_registry(pack_names())
    plan, _ = base_plan(record(frontier()))
    assert not check(plan, registry)
    plan.nodes[0].params["level"]["functional"] = []
    with pytest.raises(CheckError, match="CF-E080"):
        Executor(registry).run(plan)


def test_model_identity_cannot_change_quantity_or_model():
    target = frontier()
    plan, _ = base_plan(record(target))
    plan.nodes[0].params["quantity"] = "homo"
    assert identity(plan, record(target), {})[0].code == "CF-E120"


def test_fci_cap_refuses_before_integral_construction(monkeypatch):
    from chemflow.domains import properties

    monkeypatch.setattr(properties, "DETERMINANT_CAP", 10)
    with pytest.raises(AdapterError, match=r"FCI.*determinants.*cap 10"):
        base_plan(record(ionization()))


@pytest.mark.parametrize("kind", ["ionization_energy", "electron_affinity"])
def test_charge_controls_and_fci_reference_relations(kind):
    from pyscf import cc, gto, scf

    registry = load_registry(pack_names())
    target = ionization(kind=kind)
    plan, controls = base_plan(record(target))
    assert len(controls) == 3
    assert not identity(plan, record(target), {})
    computed = Executor(registry).run(plan)
    assert computed[plan.references[0]].relation == "equivalent"
    for identifier in [*controls, *plan.references]:
        assert computed[identifier].diagnostics["relation_status"] == "proved"
    assert all(computed[identifier].relation == "approximate" for identifier in controls)
    independent = []
    charge = 1 if kind == "ionization_energy" else -1
    for sector_charge, spin in ((0, 0), (charge, 1)):
        mol = gto.M(atom=target.molecule["atoms"], basis="sto-3g", charge=sector_charge,
                    spin=spin, verbose=0)
        mean_field = (scf.ROHF(mol) if spin else scf.RHF(mol)).run()
        solver = cc.CCSD(mean_field.to_uhf() if spin else mean_field).run()
        independent.append(solver.e_tot + solver.ccsd_t())
    assert computed[controls[1]].value.value == pytest.approx(
        charge * (independent[1] - independent[0]), abs=1e-7)
    assert computed[controls[2]].context.keys == computed[plan.references[0]].context.keys
    charged = next(node for node in plan.nodes if node.op == "molecule" and
                   node.params.get("charge"))
    charged.params["charge"] *= -1
    assert identity(plan, record(target), {})


def test_charged_spin_identity_is_enforced():
    target = ionization()
    plan, _ = base_plan(record(target))
    energy = next(node for node in plan.nodes if node.op == "ground_energy")
    energy.params["s"] = 1
    assert identity(plan, record(target), {})


def test_geometry_validity_changes_model_defined_geometry():
    target = frontier()
    plan, _ = base_plan(record(target))
    assert default_axes(target)[0].axis == "geometry_scale"
    scaled = instantiate(plan, "geometry_scale", 1.2)
    assert scaled.nodes[0].params["molecule"] != plan.nodes[0].params["molecule"]
    assert plan.nodes[0].params["molecule"] == target.molecule


def test_open_shell_beta_frontier_matches_uhf():
    from pyscf import gto, scf

    target = frontier(molecule={"atoms": "O 0 0 0; H 0 0 0.97", "spin": 1}, spin="beta")
    plan, _ = base_plan(record(target))
    computed = Executor(load_registry(pack_names())).run(plan)
    reference = scf.UHF(gto.M(atom=target.molecule["atoms"], spin=1, basis="6-31g",
                              verbose=0)).run()
    occupied = reference.mo_energy[1][reference.mo_occ[1] > 0]
    virtual = reference.mo_energy[1][reference.mo_occ[1] == 0]
    assert computed[plan.targets[0]].value.value == pytest.approx(
        min(virtual) - max(occupied), abs=1e-7)


def test_gbsa_model_matches_tblite():
    from pyscf import gto
    from tblite.interface import Calculator

    target = SolvationTarget(kind="solvation_free_energy", molecule={"atoms": WATER},
        electronic_model={"basis": "sto-3g"}, level={"method": "gfn2"},
        solvent={"model": "gbsa", "name": "water"})
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    energies = []
    for medium in (None, "water"):
        calculator = Calculator("GFN2-xTB", np.asarray(mol.atom_charges()), mol.atom_coords())
        calculator.set("verbosity", 0)
        if medium:
            calculator.add("gbsa-solvation", medium)
        energies.append(calculator.singlepoint().get("energy"))
    plan, _ = base_plan(record(target))
    computed = Executor(load_registry(pack_names())).run(plan)
    assert computed[plan.targets[0]].value.value == pytest.approx(energies[1] - energies[0], abs=1e-8)


def test_invalid_dft_dispersion_refused_statically():
    with build.program("invalid-dispersion") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 .74"),
                                  builder.electronic_model(basis="sto-3g"))
        energy = builder.ground_energy(ham, builder.sector(n_alpha=1, n_beta=1, n_orb=2))
        builder.refine(energy, "dft", functional="b3lyp", dispersion="d2")
        builder.bind(energy, "dft_pyscf")
    with pytest.raises(CheckError, match="CF-E080"):
        Executor(load_registry(pack_names())).run(builder.to_ir())


def test_property_cards_validate_in_loaded_pack():
    from pathlib import Path

    from chemflow.cards import validate_cards

    report = validate_cards(load_registry(pack_names()), Path(__file__).resolve().parents[1])
    assert report.ok
    assert {"dft", "frontier_orbitals", "solvation_free_energy", "charge_energy_difference",
            "eom_charge_energy"} <= {card.id for card in report.cards}


def test_property_cards_require_their_pack_and_valid_templates():
    from pathlib import Path

    from chemflow.cards import load_cards, validate_cards

    root = Path(__file__).resolve().parents[1]
    identifiers = {"dft", "frontier_orbitals", "solvation_free_energy",
                   "charge_energy_difference", "eom_charge_energy"}
    assert {card.id for card in load_cards() if card.pack == "properties"} == identifiers
    report = validate_cards(load_registry(["molecular", "materials", "solubility"]), root)
    assert identifiers.isdisjoint(card.id for card in report.cards)
    registry = load_registry(["properties"])
    assert identifiers <= {card.id for card in validate_cards(registry, root).cards}
    del registry.refinements["dft"]
    with pytest.raises(ValueError, match="Unknown template or target: dft"):
        validate_cards(registry, root)


def test_default_fci_cap_refuses_large_full_basis():
    target = ionization(molecule={"atoms": WATER}, electronic_model={"basis": "6-31g"})
    with pytest.raises(AdapterError, match=r"FCI.*determinants.*cap 100000"):
        base_plan(record(target))


def test_optional_restricted_eom_does_not_block_open_shell_reference():
    target = ionization(molecule={"atoms": "Li 0 0 0", "spin": 1}, final_spin=0)
    plan, controls = base_plan(record(target))
    computed = Executor(load_registry(pack_names())).run(plan)
    assert "closed-shell" in computed["refused"][controls[2]]
    assert computed[plan.references[0]].relation == "equivalent"


def test_smd_feature_detection_also_handles_construction_failure(monkeypatch):
    from pyscf import dft, gto

    from chemflow.domains.properties import scf_model

    class MissingSMD:
        def SMD(self):
            raise RuntimeError("SMD module is not available")

    monkeypatch.setattr(dft, "RKS", lambda molecule: MissingSMD())
    molecule = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
    with pytest.raises(AdapterError, match="CF-E080.*SMD") as caught:
        scf_model(molecule, {"method": "dft", "functional": "b3lyp"},
                  {"model": "smd", "name": "water"})
    assert "PCM" in caught.value.diagnostics[0].fix_hints[0]
    assert "ALPB" in caught.value.diagnostics[0].fix_hints[0]
