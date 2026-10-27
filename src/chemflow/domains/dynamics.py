"""Real-axis retarded spectral functions for site-basis impurity models.

H = -U/2 (n_0α+n_0β) + U n_0α n_0β + Σ ε_b n_bσ
    + Σ t_b (a†_0σ a_bσ + a†_bσ a_0σ), with zero constant.
Lehmann evaluates A(ω)=-Im G(ω+iη)/π using squared fermionic ladder
matrix elements. Both particle and hole denominators have +iη. Costs are
O(grid_size * (D_particle + D_hole)) beyond projected diagonalizations.
The exact reference enumerates full sectors (exponential time/storage).

Relations are executor/checker rules, not binding assertions: a response is
equivalent only for complete ground/particle/hole resolutions with full
sector coverage; otherwise approximate. The finite-grid sum rule is an
unrenormalized trapezoidal integral, not an assertion of unit spectral weight.
The L1 gate integrates absolute density differences on an identical grid;
its threshold is dimensionless (the gate unit identifies the energy axis).
"""

import numpy as np

from chemflow import build
from chemflow.diagnostics import Diagnostic
from chemflow.domains.algorithms import (
    as_sector,
    full_subspace,
    rayleigh_ritz,
    resources,
    transition_sector,
)
from chemflow.domains.algorithms import ladder as determinant_ladder
from chemflow.domains.materials import LatticeData
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.relations import TargetRelation
from chemflow.types import (
    BackendKind,
    ConstantLedger,
    Hamiltonian,
    OrbitalSpace,
    Sector,
    SpectralEstimate,
)
from chemflow.units import Q


def random_instance(family, rng, *, n_bath=3):
    """Draw AIM parameters in hartree: U∈[1,4], ε∈[-1.5,1.5], t∈[0.3,0.8]."""
    if family != "impurity_model":
        return NotImplemented
    if type(n_bath) is not int or n_bath < 0:
        raise ValueError("n_bath must be a nonnegative integer")
    return {"U": float(rng.uniform(1, 4)),
            "eps_bath": sorted(rng.uniform(-1.5, 1.5, n_bath).tolist()),
            "t_bath": rng.uniform(0.3, 0.8, n_bath).tolist(), "unit": "hartree"}


def impurity_arrays(U, eps_bath, t_bath, unit="hartree"):
    if (not isinstance(eps_bath, (list, tuple)) or
            not isinstance(t_bath, (list, tuple)) or len(eps_bath) != len(t_bath)):
        raise ValueError("Bath energies and hybridisations must have equal lengths")
    scale = Q(1, unit).to("hartree").value
    values = np.asarray([U, *eps_bath, *t_bath], dtype=float) * scale
    if not np.isfinite(values).all():
        raise ValueError("Impurity parameters must be finite")
    size = 1 + len(eps_bath)
    h1 = np.diag([-values[0] / 2, *values[1:size]])
    h1[0, 1:] = h1[1:, 0] = values[size:]
    eri = np.zeros((size,) * 4)
    eri[0, 0, 0, 0] = values[0]
    return h1, eri


def impurity_model(store, **params):
    h1, eri = impurity_arrays(**params)
    size = len(h1)
    sector = Sector(n_alpha=(size + 1) // 2, n_beta=size // 2, n_orb=size)
    space = OrbitalSpace(coefficients_ref=store.put(np.eye(size)), n_orb=size,
                         n_elec=size, ordering="sites")
    ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), n_orb=size,
                      space=content_hash(space), ledger=ConstantLedger())
    identity = content_hash(params)
    return LatticeData(mol=None, mf=None, hamiltonian=ham, space=space, sector=sector,
                       model_id=identity, geometry_id=identity, lattice_id=identity)


def grid(omega, eta):
    if (set(omega) != {"start", "stop", "n", "unit"} or type(omega["n"]) is not int or
            omega["n"] < 2 or not np.isfinite([omega["start"], omega["stop"], eta]).all() or
            omega["stop"] <= omega["start"] or eta <= 0):
        raise ValueError("Require a finite increasing omega grid and eta > 0")
    scale = Q(1, omega["unit"]).to("hartree").value
    return np.linspace(omega["start"], omega["stop"], omega["n"]), scale


def ladder(state, mode, create):
    """Fermionic transition amplitude; the destination flips the selected bit."""
    transition = determinant_ladder(state, mode, create)
    return transition[1] if transition is not None else None


def lehmann_from_spectra(store, ground, particle, hole, orbital, spin, omega, eta):
    frequencies, scale = grid(omega, eta)
    if type(orbital) is not int or not 0 <= orbital < ground.sector.n_orb:
        raise ValueError("Spectral orbital outside sector")
    if spin not in ("alpha", "beta"):
        raise ValueError("Spectral spin must be alpha or beta")
    ground_vector = store.get(ground.vectors_ref)[:, 0]
    green = np.zeros(len(frequencies), dtype=complex)
    coverage = {"ground": ground.coverage}
    for label, excited, operator, sign in (("particle", particle, "create", 1),
                                            ("hole", hole, "annihilate", -1)):
        if excited is None:
            coverage[label] = 1.0
            continue
        if excited.hamiltonian_id != ground.hamiltonian_id:
            raise ValueError("Lehmann spectra require the same Hamiltonian")
        mode = transition_sector(ground.sector, excited.sector, operator, orbital, spin)
        positions = {state: index for index, state in enumerate(excited.subspace.determinants)}
        image = np.zeros(len(positions), dtype=complex)
        for state, coefficient in zip(ground.subspace.determinants, ground_vector):
            applied = ladder(state, mode, operator == "create")
            destination = state ^ (1 << mode)
            if applied is not None and destination in positions:
                image[positions[destination]] += applied * coefficient
        vectors = store.get(excited.resolution_vectors_ref)
        weights = np.abs(vectors.conj().T @ image) ** 2
        energies = store.get(excited.resolution_energies_ref)
        poles = sign * (energies - ground.energies[0]) / scale
        green += (weights[None, :] / (frequencies[:, None] - poles[None, :] +
                                      1j * eta)).sum(axis=1)
        coverage[label] = excited.coverage
    spectral = -green.imag / np.pi
    return SpectralEstimate(omega=tuple(frequencies), spectral=tuple(spectral),
                            unit=omega["unit"], diagnostics={
                                "sum_rule": float(np.trapezoid(spectral, frequencies)),
                                "ground_energy": float(ground.energies[0]),
                                "coverage": coverage, "density_unit": f"1/{omega['unit']}",
                                "resolution": "complete_projected"})


def spectral_exact(store, hamiltonian, sector, orbital, spin, omega, eta):
    sector = as_sector(sector)
    ground = rayleigh_ritz(store, hamiltonian, full_subspace(store, sector))
    spectra = []
    for delta in (1, -1):
        counts = {key: getattr(sector, key) for key in ("n_alpha", "n_beta", "n_orb")}
        if spin not in ("alpha", "beta"):
            raise ValueError("Spectral spin must be alpha or beta")
        counts[f"n_{spin}"] += delta
        if not 0 <= counts[f"n_{spin}"] <= sector.n_orb:
            spectra.append(None)
        else:
            spectra.append(rayleigh_ritz(store, hamiltonian, full_subspace(store, counts)))
    result = lehmann_from_spectra(store, ground, *spectra, orbital, spin, omega, eta)
    witnesses = []
    for spectrum in (ground, *spectra):
        witnesses.append(None if spectrum is None else {
            **spectrum.model_dump(mode="json", exclude={"subspace"}),
            "sector": spectrum.sector.model_dump(mode="json"),
            "determinants_ref": store.put(np.asarray(spectrum.subspace.determinants, dtype=np.int64))})
    return result.model_copy(update={"diagnostics": {
        **result.diagnostics, "resolution_witnesses": witnesses}})


def context_check(node, inputs, context, nodes):
    if node.op == "impurity_model":
        identity = content_hash(node.params)
        context.update(hamiltonian_id=identity, model_id=identity, model_protocol_id=identity)
    elif node.op == "spectral_function":
        context.update(inputs.get("hamiltonian", inputs.get("ground", {})))
        context.update(type="spectral_estimate")
        if "sector" in inputs:
            context["sector_id"] = inputs["sector"].get("sector_id")
            context["sector_params"] = nodes[node.inputs["sector"]].params
        context["spectral_identity"] = content_hash({
            "hamiltonian": context.get("hamiltonian_id"), "sector": context.get("sector_id"),
            **node.params})
        try:
            source = context.get("sector_params")
            if source is not None:
                sector = as_sector(source)
                orbital = node.params["orbital"]
                if type(orbital) is not int or not 0 <= orbital < sector.n_orb:
                    raise ValueError("Spectral orbital outside sector")
                for port, operator in (("particle", "create"), ("hole", "annihilate")):
                    if port in inputs:
                        if inputs[port].get("hamiltonian_id") != context.get("hamiltonian_id"):
                            raise ValueError("Lehmann spectra require the same Hamiltonian")
                        target = inputs[port].get("sector_params")
                        if target is not None:
                            transition_sector(sector, as_sector(target), operator,
                                              orbital, node.params["spin"])
        except (ValueError, KeyError, TypeError) as exc:
            return [Diagnostic(code="CF-E020", node=node.id, message=str(exc),
                               fix_hints=["Match Hamiltonians and N, N+1, N-1 sectors"])]
    return []


def static_check(program, registry):
    issues = []
    for node in program.nodes:
        try:
            if node.op == "impurity_model":
                impurity_arrays(**node.params)
            elif node.op == "spectral_function":
                if node.refinement not in ("exact_lehmann", "lehmann"):
                    raise ValueError("Select exact_lehmann or lehmann refinement")
                grid(node.params["omega"], node.params["eta"])
                if node.params["spin"] not in ("alpha", "beta"):
                    raise ValueError("Unknown spin")
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(Diagnostic(code="CF-E080", node=node.id, message=str(exc),
                                     fix_hints=["Supply valid model and spectral parameters"]))
    return issues


def base_plan(record, active=None):
    target = record.spec.target
    if active is not None and active != {
            "n_orb": target.sector["n_orb"],
            "n_elec": target.sector["n_alpha"] + target.sector["n_beta"]}:
        raise ValueError("Active space must match the impurity sector")
    with build.program("spectral-reference") as builder:
        ham = builder.node("impurity_model", **target.model["impurity_model"])
        sector = builder.sector(**target.sector)
        response = builder.node("spectral_function", hamiltonian=ham, sector=sector,
                                orbital=target.orbital, spin=target.spin,
                                omega=target.omega, eta=target.eta)
        builder.refine(response, "exact_lehmann")
        builder.bind(response, "spectral_exact")
        builder.ir.targets = [response.id]
        builder.ir.references = [response.id]
    plan = builder.to_ir()
    mapping = {node.id: f"spectral_reference_{node.id}" for node in plan.nodes}
    for node in plan.nodes:
        node.id = mapping[node.id]
        node.inputs = {port: mapping[source] for port, source in node.inputs.items()}
    plan.edges = [edge.model_copy(update={"source": mapping[edge.source],
                                         "target": mapping[edge.target]}) for edge in plan.edges]
    plan.targets = [mapping[identifier] for identifier in plan.targets]
    plan.references = [mapping[identifier] for identifier in plan.references]
    return plan, []


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    target = record.spec.target
    expected = content_hash({"hamiltonian": content_hash(target.model["impurity_model"]),
                             "sector": content_hash(target.sector), "orbital": target.orbital,
                             "spin": target.spin, "omega": target.omega, "eta": target.eta})
    nodes = {node.id: node for node in plan.nodes}
    return [Diagnostic(code="CF-E120", node=identifier, message="Spectral target identity mismatch",
                       fix_hints=plan_hints(plan, base_plan(record)[0], identifier))
            for identifier in set(plan.targets + plan.references)
            if identifier not in nodes or nodes[identifier].op != "spectral_function" or
            contexts.get(identifier, {}).get("spectral_identity") != expected]


def register(registry):
    registry.register(OperatorSpec(id="chemflow.impurity_model", version=1, inputs={},
                      output="hamiltonian", semantics="Site-basis AIM: epsilon_d=-U/2, "
                      "hybridisations t_bath, (00|00)=U, zero constant; O(M^4) integral storage.",
                      parameters={"U": "finite on-site repulsion", "eps_bath": "bath energies",
                                  "t_bath": "hybridisations, same length as eps_bath",
                                  "unit": "energy unit, default hartree"}))
    registry.register(ImplementationSpec(id="impurity_model", version="1", target="impurity_model",
                      adapter=impurity_model, backend_kind=BackendKind.cpu_numeric,
                      params=frozenset({"U", "eps_bath", "t_bath", "unit"}),
                      resource_model=resources))
    registry.register(OperatorSpec(id="chemflow.spectral_function", version=1, inputs={},
                      output="spectral_estimate", semantics="Retarded Lehmann density -Im G/pi; "
                      "eta and omega in the grid energy unit, density in its inverse. "
                      "O(grid_size * projected_dimension) response evaluation.",
                      parameters={"orbital": "zero-based spatial orbital", "spin": "alpha | beta",
                                  "omega": "{start, stop, n >= 2, unit}: increasing energy grid",
                                  "eta": "positive broadening in the omega energy unit"}))
    for template, binding, inputs, adapter in (
        ("exact_lehmann", "spectral_exact", {"hamiltonian": "hamiltonian", "sector": "sector"},
         spectral_exact),
        ("lehmann", "lehmann_from_spectra",
         {"ground": "spectrum", "particle": "spectrum", "hole": "spectrum"}, lehmann_from_spectra),
    ):
        registry.register(RefinementRule(target="spectral_function", template_id=template,
                          relation=TargetRelation.approximate, inputs=inputs))
        registry.register(ImplementationSpec(id=binding, version="1", target="spectral_function",
                          implements=template, adapter=adapter, backend_kind=BackendKind.cpu_numeric,
                          params=frozenset({"orbital", "spin", "omega", "eta"}),
                          resource_model=resources))
    registry.context_checks.append(context_check)
    registry.checks.append(static_check)
