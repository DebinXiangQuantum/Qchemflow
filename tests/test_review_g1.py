"""Reviewer regression tests for G1, thermochemistry (written by Claude; implementers must not
edit).

From the ChemGraph comparison (user priority 2026-10-13): ChemFlow computes harmonic frequencies
but reports electronic energies only. G1 adds ideal-gas rigid-rotor harmonic-oscillator (RRHO)
thermochemistry, i.e. ZPE, U(T), H(T), S(T, p), G(T, p), and reaction enthalpies and Gibbs
energies at a temperature. Masses are most-abundant isotopes.

Contract: reactions-pack operator `thermochemistry` (inputs molecule, model, frequencies; params
temperature_K, pressure_Pa, optional symmetry_number override) returning a mapping with zpe,
u_corr, h_corr, g_corr (hartree), s (hartree/K), temperature_K, pressure_Pa, symmetry_number,
spin_multiplicity, linear, thermal_model == "rrho_ideal_gas". It refuses a geometry that is not
a stationary point of the declared kind. Reaction tasks take an optional
`thermo = {quantity: enthalpy | gibbs, temperature_K, pressure_Pa}`. The reference then
combines E_el + correction per species, the fci reaction node gets inputs species<i> and
thermo<i>, and the result kind is reaction_enthalpy or reaction_gibbs_energy.

The oracle below is the reviewer's own RRHO implementation (it does not call pyscf.thermo).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import constants as sc

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.runtime import Executor

MASS = {"H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957}
HARTREE = sc.physical_constants["Hartree energy"][0]
T, P = 298.15, 101325.0
H2O = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
NH3 = "N 0 0 0.112; H 0 0.939 -0.262; H 0.813 -0.469 -0.262; H -0.813 -0.469 -0.262"
HCN = "H 0 0 -1.065; C 0 0 0; N 0 0 1.156"
HNC = "H 0 0 -0.995; N 0 0 0; C 0 0 1.170"


def _atoms(text):
    entries = [item.split() for item in text.split(";") if item.strip()]
    return [item[0] for item in entries], np.array([[float(x) for x in item[1:4]]
                                                    for item in entries])


def rrho(atoms_text, frequencies_cm, sigma, multiplicity=1, temperature=T, pressure=P):
    """Ideal-gas RRHO corrections per molecule, in hartree (s in hartree/K)."""
    symbols, coordinates = _atoms(atoms_text)
    masses = np.array([MASS[symbol] for symbol in symbols]) * sc.atomic_mass  # kg
    positions = coordinates * 1e-10  # m
    centre = masses @ positions / masses.sum()
    relative = positions - centre
    inertia = sum(mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
                  for mass, r in zip(masses, relative, strict=True))
    moments = np.sort(np.linalg.eigvalsh(inertia))
    linear = moments[0] < 1e-3 * moments[-1]
    kt = sc.k * temperature
    total = masses.sum()
    q_trans = (2 * np.pi * total * kt / sc.h ** 2) ** 1.5 * kt / pressure
    s_trans = sc.k * (np.log(q_trans) + 2.5)
    u_trans = 1.5 * kt
    thetas = sc.h ** 2 / (8 * np.pi ** 2 * moments[moments > 1e-3 * moments[-1]] * sc.k)
    if linear:
        q_rot = temperature / (sigma * thetas[0])
        s_rot, u_rot = sc.k * (np.log(q_rot) + 1), kt
    else:
        q_rot = np.sqrt(np.pi) / sigma * np.sqrt(temperature ** 3 / np.prod(thetas))
        s_rot, u_rot = sc.k * (np.log(q_rot) + 1.5), 1.5 * kt
    nu = np.array([value for value in frequencies_cm if value > 0]) * 100 * sc.c  # Hz
    theta_v = sc.h * nu / sc.k
    zpe = 0.5 * sc.h * nu.sum()
    x = theta_v / temperature
    u_vib = zpe + np.sum(sc.k * theta_v / np.expm1(x))
    s_vib = sc.k * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x)))
    s_el = sc.k * np.log(multiplicity)
    u_corr = u_trans + u_rot + u_vib
    h_corr = u_corr + kt
    s = s_trans + s_rot + s_vib + s_el
    return {"zpe": zpe / HARTREE, "u_corr": u_corr / HARTREE, "h_corr": h_corr / HARTREE,
            "s": s / HARTREE, "g_corr": (h_corr - temperature * s) / HARTREE,
            "linear": bool(linear)}


def _registry():
    from chemflow.domains import load_registry

    return load_registry(["molecular", "reactions"])


def _thermo_program(atoms, optimize=True, **params):
    with build.program("thermo") as builder:
        molecule = builder.molecule(atoms)
        model = builder.electronic_model(basis="sto-3g")
        level = {"method": "rhf"}
        geometry = (builder.node("geometry_optimize", molecule=molecule, model=model,
                                 kind="minimum", level=level) if optimize else molecule)
        frequencies = builder.node("frequencies", molecule=geometry, model=model, level=level)
        thermo = builder.node("thermochemistry", molecule=geometry, model=model,
                              frequencies=frequencies, temperature_K=T, pressure_Pa=P, **params)
    return builder.to_ir(), geometry.id, frequencies.id, thermo.id


@pytest.mark.parametrize("atoms,sigma", [(H2O, 2), (NH3, 3), (HCN, 1)])
def test_rrho_thermochemistry_matches_the_reviewers_independent_formulas(atoms, sigma) -> None:
    program, geometry, frequencies, thermo = _thermo_program(atoms)
    computed = Executor(_registry()).run(program)
    result = computed[thermo]
    optimized = computed[geometry]["atoms"]
    oracle = rrho(optimized, computed[frequencies]["freq_wavenumber"], sigma)
    assert result["thermal_model"] == "rrho_ideal_gas"
    assert result["symmetry_number"] == sigma and result["linear"] == oracle["linear"]
    assert result["temperature_K"] == T and result["pressure_Pa"] == P
    for key in ("zpe", "u_corr", "h_corr", "g_corr"):
        assert result[key] == pytest.approx(oracle[key], abs=1e-6), key
    assert result["s"] == pytest.approx(oracle["s"], rel=1e-5)
    assert result["g_corr"] < result["h_corr"] and result["zpe"] > 0


def test_thermochemistry_refuses_a_geometry_that_is_not_stationary() -> None:
    program, _, _, _ = _thermo_program("O 0 0 0; H 0 0.9 -0.3; H 0 -0.7 -0.6", optimize=False)
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run(program)
    assert "CF-E020" in str(info.value) and "stationary" in str(info.value).lower()


def _isomerization(thermo):
    species = []
    for name, atoms, coefficient in (("HCN", HCN, -1), ("HNC", HNC, 1)):
        species.append({"name": name, "molecule": {"atoms": atoms}, "coefficient": coefficient,
                        "active": {"n_elec": 4, "n_orb": 4},
                        "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "target_spin": 0})
    spec = {"question": "HCN -> HNC", "target": {
        "kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"}, "species": species,
        **({"thermo": thermo} if thermo else {})},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "budget": {"wall_s": 1800}, "allowed_effects": ["pure"], "decision": "d"}
    store = TaskStore("reviewer")
    return store.freeze(TaskSpec.model_validate(spec), "reviewer"), store


def test_reaction_gibbs_energy_combines_electronic_energies_and_rrho_corrections() -> None:
    from chemflow.domains import get_pack

    record, store = _isomerization({"quantity": "gibbs", "temperature_K": T,
                                    "pressure_Pa": P})
    plan, _ = get_pack("reactions").base_plan(record)
    computed = Executor(_registry()).run_plan(plan, record, store)
    reference = next(node for node in plan.nodes if node.id == plan.references[0])
    coefficients = reference.params["coefficients"]
    expected = 0.0
    for port, coefficient in coefficients.items():
        index = port.removeprefix("species")
        energy = computed[reference.inputs[port]].value.to("hartree").value
        thermo_node = next(node for node in plan.nodes
                           if node.id == reference.inputs[f"thermo{index}"])
        atoms = computed[thermo_node.inputs["molecule"]]["atoms"]
        frequencies = computed[thermo_node.inputs["frequencies"]]["freq_wavenumber"]
        expected += coefficient * (energy + rrho(atoms, frequencies, 1)["g_corr"])
    value = computed[reference.id]
    assert value.value.to("hartree").value == pytest.approx(expected, abs=1e-6)
    assert value.kind == "reaction_gibbs_energy"
    assert value.relation == "equivalent"  # exact within the declared RRHO + FCI model
    assert all(gate.passed for gate in computed["gates"] if gate.node == reference.id)


def test_thermal_quantities_at_different_conditions_are_not_mixed() -> None:
    from chemflow.check import check

    with build.program("mixed") as builder:
        model = builder.electronic_model(basis="sto-3g")
        level = {"method": "rhf"}
        made = {}
        for name, atoms, temperature in (("a", HCN, 298.15), ("b", HNC, 500.0)):
            molecule = builder.node("geometry_optimize", molecule=builder.molecule(atoms),
                                    model=model, kind="minimum", level=level)
            frequencies = builder.node("frequencies", molecule=molecule, model=model,
                                       level=level)
            made[name] = builder.node("thermochemistry", molecule=molecule, model=model,
                                      frequencies=frequencies, temperature_K=temperature,
                                      pressure_Pa=P)
        builder.node("reaction_energy", thermo0=made["a"], thermo1=made["b"],
                     coefficients={"thermo0": -1, "thermo1": 1})
    issues = check(builder.to_ir(), _registry())
    assert any(item.code in ("CF-E011", "CF-E010") and any(
        word in item.message.lower() for word in ("temperature", "pressure", "condition"))
        for item in issues), [item.message for item in issues]
