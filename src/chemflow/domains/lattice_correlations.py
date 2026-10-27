"""Classical site-basis observables and the conventional Allen–Dynes model."""

import math
from itertools import combinations

import numpy as np

from chemflow.diagnostics import Diagnostic
from chemflow.impls.fci import _annihilate, _create, sector_states
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.relations import TargetRelation
from chemflow.types import BackendKind, Sector

QUANTITIES = ("spin_spin", "density_density", "pair_s_wave")


def resources(**kwargs):
    return {"gpu_s": 0, "sim_shots": 0, "qpu_s": 0, "qpu_shots": 0}


def correlation_matrices(vector, sector):
    basis = sector_states(sector)
    vector = np.asarray(vector)
    if (vector.shape != (len(basis),) or not np.isfinite(vector).all() or
            not np.isclose(np.vdot(vector, vector), 1, atol=1e-10, rtol=0)):
        raise ValueError("Correlations require a normalized finite sector vector")
    index = {bits: position for position, bits in enumerate(basis)}

    def expectation(operators):
        value = 0j
        for column, bits in enumerate(basis):
            sign = 1
            for orbital, creation in reversed(operators):
                result = (_create if creation else _annihilate)(bits, orbital)
                if result is None:
                    break
                bits, parity = result
                sign *= parity
            else:
                if bits in index:
                    value += vector[index[bits]].conjugate() * sign * vector[column]
        if abs(value.imag) > 1e-10:
            raise ValueError("Complex correlations are outside the real-matrix contract")
        return float(value.real)

    size = sector.n_orb
    matrices = {key: np.zeros((size, size)) for key in QUANTITIES}
    for left in range(size):
        for right in range(size):
            for first, first_sign in ((left, 0.5), (left + size, -0.5)):
                for second, second_sign in ((right, 0.5), (right + size, -0.5)):
                    density = expectation([(first, True), (first, False),
                                           (second, True), (second, False)])
                    matrices["density_density"][left, right] += density
                    matrices["spin_spin"][left, right] += first_sign * second_sign * density
            matrices["spin_spin"][left, right] += 0.5 * (
                expectation([(left, True), (left + size, False),
                             (right + size, True), (right, False)]) +
                expectation([(left + size, True), (left, False),
                             (right, True), (right + size, False)]))
            matrices["pair_s_wave"][left, right] = expectation([
                (left, True), (left + size, True), (right + size, False), (right, False)])
    return {key: value.tolist() for key, value in matrices.items()}


def correlation_function(store, state, site_pair=None, quantity="pair_s_wave"):
    validate_selection(site_pair, quantity)
    sector = Sector.model_validate(state.diagnostics["sector"])
    if site_pair is not None and max(site_pair) >= sector.n_orb:
        raise ValueError("Correlation site is outside the lattice")
    reference = state.diagnostics.get("state_ref")
    if not reference:
        raise ValueError("The ground-state implementation has not exported a sector vector")
    matrices = correlation_matrices(store.get(reference, verify=True), sector)
    return matrices if site_pair is None else matrices[quantity][site_pair[0]][site_pair[1]]


def validate_selection(site_pair, quantity):
    if quantity not in QUANTITIES:
        raise ValueError("Unknown lattice correlation quantity")
    if site_pair is not None and (not isinstance(site_pair, (list, tuple)) or
            len(site_pair) != 2 or any(type(site) is not int or site < 0 for site in site_pair)):
        raise ValueError("site_pair must contain two nonnegative integer site indices")


def hf_state(data, store):
    dimension = (math.comb(data.sector.n_orb, data.sector.n_alpha) *
                 math.comb(data.sector.n_orb, data.sector.n_beta))
    if dimension > 50000:
        return None
    mean_field = data.mean_field(store)
    coefficients = np.asarray(mean_field.mo_coeff)
    if coefficients.ndim == 2:
        coefficients = np.stack([coefficients, coefficients])
    amplitudes = []
    for spin, count in enumerate((data.sector.n_alpha, data.sector.n_beta)):
        amplitudes.append([np.linalg.det(coefficients[spin][list(occupied), :count])
                           for occupied in combinations(range(data.sector.n_orb), count)])
    return store.put(np.outer(*amplitudes).ravel())


def allen_dynes_tc(store=None, *, lambda_ep, omega_log_K, mu_star):
    values = (lambda_ep, omega_log_K, mu_star)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Allen–Dynes inputs must be finite numbers")
    denominator = lambda_ep - mu_star * (1 + 0.62 * lambda_ep)
    if lambda_ep <= 0 or omega_log_K <= 0 or mu_star < 0 or denominator <= 0:
        raise ValueError("Require positive lambda, frequency and pairing denominator; mu* >= 0")
    return {"tc_K": omega_log_K / 1.2 * math.exp(-1.04 * (1 + lambda_ep) / denominator),
            "model": "Allen-Dynes (McMillan)", "unit": "K",
            "input_source": "user or EPW",
            "recommended_domain": "lambda <= 1.5; mu_star approximately 0.1–0.15"}


def context_check(node, inputs, context, nodes):
    if node.op == "correlation_function":
        context.update(inputs.get("state", {}))
        context.update(type="estimate" if node.params.get("site_pair") is not None
                       else "correlations", kind="pairing_correlation", unit="dimensionless",
                       quantity=node.params.get("quantity", "pair_s_wave"),
                       site_pair=str(node.params.get("site_pair")))
    return []


def static_check(program, registry):
    issues = []
    nodes = {node.id: node for node in program.nodes}
    for node in program.nodes:
        try:
            if node.op == "allen_dynes_tc":
                allen_dynes_tc(**node.params)
            elif node.op == "correlation_function":
                validate_selection(node.params.get("site_pair"),
                                   node.params.get("quantity", "pair_s_wave"))
                state = nodes.get(node.inputs.get("state"))
                if state is not None:
                    hamiltonian = nodes.get(state.inputs.get("hamiltonian"))
                    sector = nodes.get(state.inputs.get("sector"))
                    if state.op != "ground_energy" or hamiltonian is None or (
                            hamiltonian.op != "lattice_model"):
                        raise ValueError("Correlations require a site-basis lattice ground state")
                    pair = node.params.get("site_pair")
                    if pair is not None and sector is not None and max(pair) >= sector.params.get(
                            "n_orb", 0):
                        raise ValueError("Correlation site is outside the lattice")
            elif node.binding == "dmrg_block2":
                for key, default in (("bond_dimension", 64), ("sweeps", 10)):
                    value = node.params.get(key, default)
                    if type(value) is not int or value < 1:
                        raise ValueError(f"{key} must be a positive integer")
                if type(node.params.get("seed", 11)) is not int:
                    raise ValueError("seed must be an integer")
        except (ValueError, TypeError) as exc:
            issues.append(Diagnostic(code="CF-E080", node=node.id, message=str(exc),
                                     fix_hints=["Supply valid lattice/model parameters"]))
    return issues


def register(registry):
    from chemflow.impls.dmrg import dmrg_block2

    for name, inputs, output, adapter, parameters, description in (
        ("correlation_function", {"state": "estimate"}, "correlations", correlation_function,
         {"site_pair", "quantity"}, ("Spin, density and s-wave pairing correlations of a lattice "
         "sector vector; optionally select one dimensionless matrix element.")),
        ("allen_dynes_tc", {}, "tc_model", allen_dynes_tc,
         {"lambda_ep", "omega_log_K", "mu_star"}, ("Allen-Dynes (McMillan) conventional Tc "
         "model in kelvin, from user or EPW inputs; not a Hubbard Tc prediction.")),
    ):
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
                                       output=output, semantics=description))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
                          backend_kind=BackendKind.cpu_numeric, execution_role="classical",
                          params=frozenset(parameters), resource_model=resources))
    registry.register(RefinementRule(target="ground_energy", template_id="dmrg",
                                     relation=TargetRelation.variational_upper_bound))
    registry.register(RefinementRule(target="correlation_function", template_id="pairing_sqd",
                                     relation=TargetRelation.approximate))
    registry.register(RefinementRule(target="allen_dynes_tc", template_id="allen_dynes_tc",
                                     relation=TargetRelation.approximate))
    registry.register(ImplementationSpec(id="dmrg_block2", version="1", target="ground_energy",
                      implements="dmrg", adapter=dmrg_block2, backend_kind=BackendKind.cpu_numeric,
                      execution_role="classical", resource_model=resources,
                      params=frozenset({"s", "total_spin", "seed", "bond_dimension", "sweeps"})))
    registry.context_checks.append(context_check)
    registry.checks.append(static_check)
