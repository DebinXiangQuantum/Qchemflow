"""Denotations and concrete witnesses shared by proof construction and checking.

No relation is inferred here. Artifact predicates belong to the trusted kernel base.
"""

from fractions import Fraction
from math import comb
from types import SimpleNamespace

import numpy as np

from chemflow.check.static import parameter_diagnostics
from chemflow.domains.materials import LatticeData
from chemflow.ids import content_hash
from chemflow.ir.schema import Program
from chemflow.types import Hamiltonian, Sector, Spectrum, Subspace
from chemflow.verify.lemmas import AXIOMS

DENOTATIONS = {
    "empirical": "declared empirical model evaluated on its specified molecular descriptors",
    "declared_model": "declared solvent, frontier orbital or empirical site-score model evaluation",
    "model": "output of the explicitly declared finite model protocol, not experimental energy",
    "exact": "lambda_0(H restricted to the requested sector)",
    "ritz": "ordered eigenvalues of P_V H P_V",
    "trial": "Rayleigh quotient of the sector-engine trial state",
    "hf": "physical energy of an HF determinant",
    "linear": "sum of signed coefficients times input energies",
    "energy_collection": "finite vector of signed component energies, with energy sorting",
    "response": "retarded Lehmann sum of the supplied projected resolutions",
    "response_exact": "retarded Lehmann sum of full sector resolutions",
    "sample": "finite-sample observable estimate",
    "expectation": "expectation of the specified observable in the specified state",
    "thermal": "RRHO ideal-gas thermal correction at the specified conditions",
    "opaque": "named approximation with no ordering law",
    "data": "typed input or intermediate object",
    "curve": "ordered finite tuple of electronic energies at the declared scan coordinates",
}


def denotation(node, registry):
    if parameter_diagnostics(Program(name="denotation", nodes=[node]), registry,
                             require_complete=False):
        return None
    implementation = registry.implementations.get(node.binding or node.op)
    if implementation is None or implementation.target != node.op:
        return None
    contract = AXIOMS.get(f"binding:{implementation.id}")
    if node.op == implementation.id and node.op in ("component_energy", "app_component_energy"):
        contract = AXIOMS.get(f"binding:{node.op}:{node.params.get('method')}", contract)
    if (contract is None or contract["op"] != node.op or
            str(implementation.version) != contract["version"] or
            node.params.get("model_change")):
        return None
    return contract


def dependency_closure(program, node_id):
    nodes = {node.id: node for node in program.nodes}
    ordered, visiting, visited = [], set(), set()

    def visit(identifier):
        if identifier in visiting:
            raise ValueError("Cyclic dependency")
        if identifier in visited:
            return
        visiting.add(identifier)
        for port in sorted(nodes[identifier].inputs):
            visit(nodes[identifier].inputs[port])
        visiting.remove(identifier)
        visited.add(identifier)
        ordered.append(nodes[identifier])

    visit(node_id)
    return ordered


def _sector(value):
    if isinstance(value, Sector):
        return value
    if isinstance(value, dict):
        return Sector.model_validate({key: item for key, item in value.items() if key != "s"})
    return getattr(value, "sector", None)


def _basis_valid(basis, sector):
    if sector is None or not basis or len(set(basis)) != len(basis):
        return False
    half = (1 << sector.n_orb) - 1
    return all(type(state) is int and 0 <= state < 1 << (2 * sector.n_orb) and
               (state & half).bit_count() == sector.n_alpha and
               (state >> sector.n_orb).bit_count() == sector.n_beta for state in basis)


def _full(spectrum, store):
    if not isinstance(spectrum, Spectrum) or store is None:
        return False
    basis, sector = spectrum.subspace.determinants, spectrum.sector
    dimension = comb(sector.n_orb, sector.n_alpha) * comb(sector.n_orb, sector.n_beta)
    if not _basis_valid(basis, sector) or len(basis) != dimension:
        return False
    energies = store.get(spectrum.resolution_energies_ref, verify=True)
    vectors = store.get(spectrum.resolution_vectors_ref, verify=True)
    leading = store.get(spectrum.vectors_ref, verify=True)
    return (energies.shape == (dimension,) and vectors.shape == (dimension, dimension) and
            leading.shape == (dimension, len(spectrum.energies)) and
            np.isfinite(energies).all() and np.isfinite(vectors).all() and
            np.array_equal(energies[:len(spectrum.energies)], spectrum.energies) and
            np.array_equal(vectors[:, :len(spectrum.energies)], leading) and
            np.allclose(vectors.conj().T @ vectors, np.eye(dimension), atol=1e-10, rtol=0))


def _facts(node, registry, computed):
    contract = denotation(node, registry)
    value = computed.get(node.id)
    inputs = {port: computed.get(source) for port, source in node.inputs.items()}
    store = computed.get("_artifacts")
    diagnostics = getattr(value, "diagnostics", {})
    result = {"known_denotation": contract is not None and value is not None and
              getattr(value, "relation", None) != "model_change" and
              all(getattr(item, "relation", None) != "model_change" for item in inputs.values()),
              "binding": contract["binding"] if contract else None,
              "parameters_hash": content_hash(node.params)}
    if contract is None:
        return result
    provenance = getattr(value, "provenance", None)
    if provenance is not None:
        result["known_denotation"] = result["known_denotation"] and (
            provenance.impl_id == contract["binding"] and provenance.impl_version == contract["version"])
    kind = contract["kind"]
    if kind == "empirical":
        from chemflow.empirical import evaluate, model_card

        model = node.params.get("model", node.op.removeprefix("logs_"))
        expected = evaluate(model, node.params["molecule_smiles"],
                            node.params.get("melting_point_C"))
        actual = value if isinstance(value, dict) else value.diagnostics
        result["empirical_evaluation_matches"] = all(
            actual.get(key) == item for key, item in expected.items())
        if not isinstance(value, dict):
            result["empirical_evaluation_matches"] &= (
                value.value.to(expected["unit"]).value == expected["value"] and
                value.kind == "aqueous_solubility" and
                actual.get("experimental_relation") == "model_change" and
                value.target == content_hash((model, node.params["molecule_smiles"],
                                              node.params.get("melting_point_C"))) and
                actual.get("model_card") == model_card(model))
    if kind == "declared_model":
        result["declared_model_matches"] = (result["known_denotation"] and
            diagnostics.get("declared_model") == node.params)
    if kind in ("exact", "ritz", "trial", "hf", "response_exact"):
        data = inputs.get("hamiltonian")
        if node.op in ("component_energy", "app_component_energy") and diagnostics.get(
                "hamiltonian_witness"):
            data = SimpleNamespace(
                hamiltonian=Hamiltonian.model_validate(diagnostics["hamiltonian_witness"]),
                sector=Sector.model_validate(diagnostics["sector_witness"]))
            inputs["sector"] = data.sector
        hamiltonian = getattr(data, "hamiltonian", None)
        sector = _sector(inputs.get("sector"))
        subspace = inputs.get("subspace")
        if isinstance(subspace, Subspace):
            sector = subspace.sector
        matching = (hamiltonian is not None and sector is not None and
                    hamiltonian.n_orb == sector.n_orb)
        if (kind != "ritz" or node.op == "ground_energy") and not isinstance(data, LatticeData):
            original = getattr(data, "sector", None)
            matching = matching and original is not None and all(
                getattr(original, key) == getattr(sector, key)
                for key in ("n_alpha", "n_beta", "n_orb"))
        conserving = (hamiltonian is not None and
                      hamiltonian.convention == "chemist_spatial_pqrs")
        hermitian = False
        if conserving and store is not None:
            one = store.get(hamiltonian.h1_ref, verify=True)
            two = store.get(hamiltonian.h2_ref, verify=True)
            size = hamiltonian.n_orb
            hermitian = (one.shape == (size, size) and two.shape == (size,) * 4 and
                         np.isfinite(one).all() and np.isfinite(two).all() and
                         np.allclose(one, one.conj().T, atol=1e-10, rtol=0) and
                         np.allclose(two, two.transpose(1, 0, 3, 2).conj(), atol=1e-10, rtol=0))
            result["integrals"] = [hamiltonian.h1_ref, hamiltonian.h2_ref]
            result["hamiltonian_id"] = hamiltonian.id
        result["sector"] = sector.model_dump(mode="json") if sector else None
        target_spin = node.params.get("s", node.params.get("total_spin"))
        if target_spin is None and isinstance(inputs.get("sector"), dict):
            target_spin = inputs["sector"].get("s", inputs["sector"].get("total_spin"))
        if target_spin is None and sector is not None:
            target_spin = sector.total_spin
        spin = target_spin is None
        if target_spin is not None:
            spin = ("s2" in diagnostics and abs(diagnostics["s2"] -
                    target_spin * (target_spin + 1)) <= 1e-4 and
                    not diagnostics.get("spin_completion_reason"))
        result.update(hermitian=bool(hermitian), number_conserving=conserving,
                      sector_matches=bool(matching), spin_supported=bool(spin))
        basis = tuple(subspace.determinants) if isinstance(subspace, Subspace) else ()
        if diagnostics.get("determinants_ref") and store is not None:
            strings = store.get(diagnostics["determinants_ref"], verify=True)
            if strings.ndim == 2 and strings.shape[1] == 2 and sector is not None:
                pairs = strings.tolist()
                if all(type(mask) is int and 0 <= mask < 1 << sector.n_orb
                       for pair in pairs for mask in pair):
                    basis = tuple(alpha | (beta << sector.n_orb) for alpha, beta in pairs)
            elif strings.ndim == 1:
                basis = tuple(strings.tolist())
            result["determinants_ref"] = diagnostics["determinants_ref"]
        result["basis_in_sector"] = _basis_valid(basis, sector)
        result["basis_size"] = len(basis)
        result["excited_roots"] = isinstance(value, Spectrum) and len(basis) > 1
        result["basis_hash"] = content_hash(basis)
        result["sector_dimension"] = (comb(sector.n_orb, sector.n_alpha) *
                                      comb(sector.n_orb, sector.n_beta) if sector else 0)
        result["full_basis"] = (result["basis_in_sector"] and len(basis) ==
                                result["sector_dimension"] and target_spin is None)
        trial = kind == "hf" and bool(matching)
        if contract["binding"] == "dmrg_block2":
            trial = (bool(matching) and diagnostics.get("engine") == "block2" and
                     diagnostics.get("normalized_mps_expectation") is True and
                     np.isfinite(diagnostics.get("mps_norm", np.nan)) and
                     diagnostics.get("mps_norm", 0) > 0)
        if kind == "trial" and store is not None and diagnostics.get("state_ref"):
            state = store.get(diagnostics["state_ref"], verify=True)
            trial = (state.shape == (result["sector_dimension"],) and
                     np.isfinite(state).all() and
                     abs(float(np.vdot(state, state).real) - 1) <= 1e-10 and
                     diagnostics.get("engine") == "SectorEngine")
            if contract["binding"] == "vqe_statevector":
                trial = trial and node.params.get("ansatz", "uccsd") in ("uccsd", "lucj")
            result["state_ref"] = diagnostics["state_ref"]
        result["trial_in_sector"] = bool(trial)
        result["exact_evaluation"] = (node.params.get("measurement", "exact") == "exact" and
                                      (kind != "trial" or
                                       diagnostics.get("evaluation") == "exact") and
                                      getattr(getattr(value, "uncertainty", None), "stat_err", None) is None)
    shots = (node.params.get("shots_per_group", 1024) if node.refinement == "vqe_sampled"
             else node.params.get("shots"))
    result["sampled_evaluation"] = kind == "sample" and type(shots) is int and shots > 0
    if kind == "thermal":
        result["thermal_contract"] = (isinstance(value, dict) and
            value.get("thermal_model") == "rrho_ideal_gas" and
            value.get("temperature_K") == node.params.get("temperature_K", 298.15) and
            value.get("pressure_Pa") == node.params.get("pressure_Pa", 101325.0) and
            all(np.isfinite(value.get(key, float("nan"))) for key in ("h_corr", "g_corr")))
    if kind == "model":
        result["model_contract"] = bool(
            getattr(value, "kind", None) == "binding_free_energy" and
            getattr(value, "frames", None) == node.params.get("frames", 0) and
            diagnostics.get("experimental_relation") == "model_change" and
            np.isfinite(value.value.to("kcal/mol").value) and
            abs(value.value.to("kcal/mol").value - value.dG_kcal) < 1e-10 and
            value.provenance.seed == node.params.get("seed", 0) and
            abs(value.dG_kcal - (value.components["complex"] - value.components["receptor"] -
                                value.components["ligand"])) < 1e-6)
    if kind == "linear":
        coefficients = ({"left": 1, "right": -1} if node.op in ("energy_difference", "compare")
                        else dict(node.params.get("coefficients", {})))
        if node.op == "reaction_energy" and node.params.get("quantity") in ("enthalpy", "gibbs"):
            coefficients.update({f"thermo{port.removeprefix('species')}": coefficient
                                 for port, coefficient in list(coefficients.items())})
        result["linear_inputs"] = bool(coefficients) and set(coefficients) == set(inputs)
        result["coefficients"] = {port: str(Fraction(str(amount)))
                                  for port, amount in coefficients.items()}
    if kind == "energy_collection":
        from chemflow.domains.reactivity import collection_values, components

        target = node.params["target"]
        count = len(components(target))
        valid = set(inputs) == {f"component{index}" for index in range(count)}
        if valid:
            values = collection_values(target, [inputs[f"component{index}"].value.to(
                "hartree").value for index in range(count)])
            valid = (diagnostics.get("values_hartree") == values and
                     value.value.to("hartree").value == values[0])
            if target["kind"] == "ranking":
                valid = valid and diagnostics.get("ranking") == [target["items"][index]["name"]
                    for index in sorted(range(count), key=lambda index: values[index])]
        result["collection_inputs"] = bool(valid)
    if kind == "curve":
        from chemflow.domains.scans import point_matches_target
        from chemflow.types import EnergyCurve, Estimate

        target = node.params.get("target", {})
        coordinates = (target.get("coordinate", {}).get("values") if node.op == "pes_scan"
                       else target.get("separations"))
        result["curve_inputs"] = (
            isinstance(value, EnergyCurve) and coordinates == value.coordinates and
            value.context.keys.get("hamiltonian_id") == content_hash(target) and
            set(inputs) == {f"point{index}" for index in range(len(value.values_hartree))} and
            all(isinstance(inputs[f"point{index}"], Estimate) and
                point_matches_target(inputs[f"point{index}"], target, coordinates[index]) and
                inputs[f"point{index}"].value.to("hartree").value == energy
                for index, energy in enumerate(value.values_hartree)))
    if kind in ("response", "response_exact"):
        spectra = ([inputs.get(port) for port in ("ground", "particle", "hole")]
                   if kind == "response" else [])
        if kind == "response_exact" and store is not None:
            witnesses = diagnostics.get("resolution_witnesses", [])
            for witness in witnesses:
                if witness is None:
                    spectra.append(None)
                else:
                    witness = {**witness, "subspace": {
                        "sector": witness["sector"], "determinants":
                        store.get(witness["determinants_ref"], verify=True).tolist()}}
                    spectra.append(Spectrum.model_validate(witness))
        complete = len(spectra) == 3 and isinstance(spectra[0], Spectrum)
        if complete:
            ground = spectra[0]
            for index, spectrum in enumerate(spectra):
                if spectrum is None:
                    count = getattr(ground.sector, "n_" + node.params["spin"])
                    complete = complete and ((index == 1 and count == ground.sector.n_orb) or
                                             (index == 2 and count == 0))
                    continue
                complete = complete and _full(spectrum, store) and (
                    spectrum.hamiltonian_id == ground.hamiltonian_id)
                if index:
                    spin, orbital = node.params["spin"], node.params["orbital"]
                    complete = complete and spin in ("alpha", "beta") and (
                        type(orbital) is int and 0 <= orbital < ground.sector.n_orb)
                    for axis in ("alpha", "beta"):
                        expected = getattr(ground.sector, "n_" + axis) + (
                            (1 if index == 1 else -1) if axis == spin else 0)
                        complete = complete and getattr(spectrum.sector, "n_" + axis) == expected
                    complete = complete and spectrum.sector.n_orb == ground.sector.n_orb
        if kind == "response_exact":
            complete = (complete and result["hermitian"] and result["sector_matches"] and
                        spectra[0].hamiltonian_id == hamiltonian.id and
                        spectra[0].sector == sector)
        result["complete_response"] = bool(complete)
    return result


def facts(node, registry, computed):
    try:
        return _facts(node, registry, computed)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, OSError) as exc:
        contract = denotation(node, registry)
        return {"known_denotation": contract is not None and node.id in computed and
                getattr(computed.get(node.id), "relation", None) != "model_change",
                "binding": contract["binding"] if contract else None,
                "parameters_hash": content_hash(node.params),
                "unavailable_witness": type(exc).__name__}
