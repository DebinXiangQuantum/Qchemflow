"""Stoichiometric electronic energies and stationary-point verification."""

from collections import Counter

import numpy as np

from chemflow import build
from chemflow.diagnostics import AdapterError, Diagnostic
from chemflow.domains.thermochemistry import THERMAL_KEYS, thermochemistry
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec
from chemflow.types import BackendKind


def balanced(terms):
    from pyscf.gto.mole import _std_symbol, format_atom

    counts = Counter()
    charge = 0.0
    for coefficient, molecule in terms:
        for symbol, _ in format_atom(molecule["atoms"]):
            counts[_std_symbol(symbol)] += coefficient
        charge += coefficient * molecule.get("charge", 0)
    return abs(charge) < 1e-10 and all(abs(value) < 1e-10 for value in counts.values())


def _issue(code, node, message):
    return Diagnostic(code=code, node=node, message=message,
                      fix_hints=["Match the declared reaction and verify its geometries"])


def _geometry(nodes, source):
    node = nodes.get(source)
    if node is None:
        return None
    if node.op in ("molecule", "geometry_optimize", "connected_ts"):
        return node
    for port in ("hamiltonian", "molecule", "scf"):
        if port in node.inputs:
            return _geometry(nodes, node.inputs[port])
    return None


def _input_molecule(nodes, geometry):
    while geometry is not None and geometry.op in ("geometry_optimize", "connected_ts"):
        geometry = nodes.get(geometry.inputs.get("molecule"))
    return geometry


def static_check(program, registry):
    issues = []
    nodes = {node.id: node for node in program.nodes}
    for node in program.nodes:
        if node.op != "reaction_energy":
            continue
        coefficients = node.params.get("coefficients", {})
        roles = node.params.get("roles", {})
        thermal_ports = {f"thermo{port.removeprefix('species')}" for port in coefficients}
        expected_ports = set(coefficients) | (thermal_ports if node.params.get("quantity") else set())
        if (expected_ports != set(node.inputs) or not coefficients or
                set(roles) - set(coefficients) or
                any(role not in ("minimum", "ts") for role in roles.values())):
            issues.append(_issue("CF-E080", node.id, "Invalid reaction ports or roles"))
            continue
        terms = []
        for port in coefficients:
            source = node.inputs[port]
            molecule = _input_molecule(nodes, _geometry(nodes, source))
            if molecule is None:
                issues.append(_issue("CF-E012", node.id, "Species lacks molecular identity"))
                break
            terms.append((coefficients[port], molecule.params))
        else:
            try:
                valid = balanced(terms)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                issues.append(_issue("CF-E012", node.id,
                                     f"Invalid reaction atoms or charge parameter: {exc}"))
                continue
            if not valid:
                issues.append(_issue("CF-E012", node.id, "Unbalanced reaction"))
    return issues


def context_check(node, inputs, context, nodes):
    issues = []
    if node.op == "hamiltonian" and any(
            node.id in {nodes[source].inputs.get("hamiltonian")
                        for source in reaction.inputs.values() if source in nodes}
            for reaction in nodes.values() if reaction.op == "reaction_energy"):
        active = node.params.get("active", {})
        context["model_protocol_id"] = content_hash((context.get("model_id"),
            active.get("method", "manual"), active.get("orbital_source", "canonical_hf")))
        context["reaction_protocol"] = True
    elif node.op in ("geometry_optimize", "connected_ts"):
        context.update(inputs.get("molecule", {}))
        context["geometry_id"] = content_hash((inputs, node.params))
    elif node.op in ("frequencies", "thermochemistry"):
        context.update(inputs.get("molecule", {}))
        context["type"] = node.op
        context["model_id"] = inputs.get("model", {}).get("model_id")
        if node.op == "frequencies":
            molecule = _input_molecule(nodes, _geometry(nodes, node.inputs.get("molecule")))
            default = "uhf" if molecule and molecule.params.get("spin", 0) else "rhf"
            context["frequency_level"] = node.params.get("level", {}).get("method", default).lower()
        else:
            frequency = inputs.get("frequencies", {})
            for key in ("geometry_id", "model_id"):
                if frequency.get(key) != context.get(key):
                    issues.append(_issue("CF-E011", node.id, f"Thermochemistry {key} differs"))
            context.update(thermal_model="rrho_ideal_gas",
                           temperature_K=node.params.get("temperature_K", 298.15),
                           pressure_Pa=node.params.get("pressure_Pa", 101325.0),
                           frequency_level=frequency.get("frequency_level"))
    elif node.op == "reaction_energy":
        thermal = {port: item for port, item in inputs.items() if port.startswith("thermo")}
        for key in THERMAL_KEYS:
            if len({item.get(key) for item in thermal.values()}) > 1:
                issues.append(_issue("CF-E011", node.id, f"Thermal condition {key} differs"))
        all_inputs = inputs
        inputs = {port: item for port, item in inputs.items() if not port.startswith("thermo")}
        if any(item.get("type") != "estimate" for item in inputs.values()):
            issues.append(_issue("CF-E002", node.id, "Reaction ports require estimates"))
        models = {item.get("model_id") for item in inputs.values()}
        protocols = {item.get("model_protocol_id") for item in inputs.values()}
        policies = set()
        for port in inputs:
            source = node.inputs[port]
            energy = nodes[source]
            ham = nodes.get(energy.inputs.get("hamiltonian"))
            active = ham.params.get("active", {}) if ham else {}
            policies.add((active.get("method", "manual"),
                          active.get("orbital_source", "canonical_hf")))
        if (len(models) != 1 or None in models or len(policies) != 1 or
                len(protocols) != 1 or None in protocols):
            issues.append(_issue("CF-E011", node.id, "Reaction model protocols differ"))
        protocol = content_hash((sorted(models, key=str), sorted(policies)))
        identity = content_hash({port: (inputs[port], node.params.get("coefficients", {}).get(port),
                                       node.params.get("roles", {}).get(port, "minimum"))
                                 for port in inputs})
        context.update(kind="reaction_energy", unit="hartree", reaction_id=identity,
                       hamiltonian_id=identity, model_protocol_id=protocol)
        if node.params.get("quantity"):
            quantity = node.params["quantity"]
            if quantity not in ("enthalpy", "gibbs"):
                issues.append(_issue("CF-E002", node.id, "Unknown thermal quantity"))
            for port, item in inputs.items():
                correction = thermal.get(f"thermo{port.removeprefix('species')}", {})
                if correction.get("type") != "thermochemistry" or any(
                        correction.get(key) != item.get(key) for key in ("geometry_id", "model_id")):
                    issues.append(_issue("CF-E011", node.id, "Thermal species geometry/model differs"))
            context.update({key: next(iter(thermal.values()), {}).get(key) for key in THERMAL_KEYS})
            identity = content_hash((identity, all_inputs, quantity))
            context.update(kind="reaction_enthalpy" if quantity == "enthalpy" else
                           "reaction_gibbs_energy", reaction_id=identity, hamiltonian_id=identity)
    return issues


def verify_frequencies(program, results):
    nodes = {node.id: node for node in program.nodes}
    for reaction in program.nodes:
        if reaction.op != "reaction_energy":
            continue
        for port, source in reaction.inputs.items():
            if port not in reaction.params.get("coefficients", {}):
                continue
            geometry = _geometry(nodes, source)
            if geometry is None:
                continue
            role = reaction.params.get("roles", {}).get(port, "minimum")
            energy = nodes[source]
            ham = nodes.get(energy.inputs.get("hamiltonian"))
            model = nodes.get(ham.inputs.get("model")) if ham else None
            frequency_nodes = [node for node in program.nodes if node.op == "frequencies"
                               and node.inputs.get("molecule") == geometry.id]
            checks = []
            for node in frequency_nodes:
                frequency_model = nodes.get(node.inputs.get("model"))
                if (model is None or frequency_model is None or
                        frequency_model.params != model.params):
                    raise AdapterError("CF-E021", reaction.id, "Frequency model differs",
                                       "Verify this geometry with the declared electronic model")
                checks.append(results[node.id])
            valid = (bool(checks) if role == "ts" else True) and all(
                item["n_imaginary"] == (1 if role == "ts" else 0) and
                (role != "ts" or item["max_gradient"] < 1e-4) for item in checks)
            if not valid:
                raise AdapterError("CF-E021", reaction.id, "Unverified stationary geometry",
                                   "Compute frequencies on this geometry; optimise the TS")


def _mean_field(molecule, model, level):
    from pyscf import gto, scf

    if model.hamiltonian_class != "nonrel_bo" or model.ecp or model.dft_functional:
        raise ValueError("Geometry and Hessian adapters require the nonrelativistic HF model")
    method = level.get("method", "uhf" if molecule.get("spin", 0) else "rhf").lower()
    if method not in ("rhf", "uhf"):
        raise ValueError("Geometry and Hessian levels must be rhf or uhf")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                basis=model.basis, verbose=0)
    mf = (scf.UHF if method == "uhf" else scf.RHF)(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise ValueError("Geometry reference SCF failed to converge")
    return mf


def geometry_optimize(store, molecule, model, kind="minimum", level=None):
    from chemflow.domains import baselines

    if kind not in ("minimum", "ts"):
        raise ValueError("Geometry kind must be minimum or ts")
    method = (level or {}).get("method", "").lower()
    if method in baselines.METHODS:
        return baselines.geometry_optimize(molecule, method, kind)
    from pyscf.geomopt.geometric_solver import optimize

    mf = _mean_field(molecule, model, level or {})
    optimized = optimize(mf, transition=kind == "ts", maxsteps=100,
                         convergence_grms=1e-6, convergence_gmax=3e-6,
                         convergence_drms=1e-5, convergence_dmax=3e-5)
    atoms = "; ".join(f"{optimized.atom_symbol(index)} " + " ".join(
        f"{coordinate:.15g}" for coordinate in position)
        for index, position in enumerate(optimized.atom_coords(unit="Angstrom")))
    return {**molecule, "atoms": atoms, "unit": "angstrom", "stationary_kind": kind}


def frequencies(store, molecule, model, level=None):
    from pyscf.hessian.thermo import harmonic_analysis

    from chemflow.domains import baselines

    method = (level or {}).get("method", "uhf" if molecule.get("spin", 0) else "rhf").lower()
    if method in baselines.METHODS:
        return baselines.frequencies(molecule, method)
    mf = _mean_field(molecule, model, level or {})
    modes = harmonic_analysis(mf.mol, mf.Hessian().kernel())["freq_wavenumber"]
    return {"freq_wavenumber": [float(value.real) if abs(value.imag) < 1e-8 else
                                -float(abs(value.imag)) for value in modes],
            "n_imaginary": int(np.count_nonzero(np.abs(np.imag(modes)) > 50)),
            "max_gradient": float(np.max(np.abs(mf.nuc_grad_method().kernel()))),
            "frequency_level": method}


def reaction_energy(store, coefficients, roles=None, quantity=None, **inputs):
    species = {port: inputs[port] for port in coefficients}
    if len({item.context.keys.get("model_protocol_id") for item in species.values()}) != 1:
        raise AdapterError("CF-E011", None, "Reaction models differ", "Use one model protocol")
    if quantity:
        thermal = [inputs[f"thermo{port.removeprefix('species')}"] for port in coefficients]
        if any(len({item.get(key) for item in thermal}) != 1 for key in THERMAL_KEYS):
            raise AdapterError("CF-E011", None, "Reaction thermal conditions or frequency levels differ",
                               "Use one thermal protocol and frequency level")
    correction = "h_corr" if quantity == "enthalpy" else "g_corr"
    return sum(coefficients[port] * (item.value.to("hartree").value +
               (inputs[f"thermo{port.removeprefix('species')}"][correction] if quantity else 0))
               for port, item in species.items())


def _control_adapter(method):
    def adapter(store, hamiltonian, sector, s=None, total_spin=None):
        from chemflow.ops.molecular import ground_energy

        if tuple(sector[key] for key in ("n_alpha", "n_beta", "n_orb")) != (
                hamiltonian.sector.n_alpha, hamiltonian.sector.n_beta,
                hamiltonian.sector.n_orb):
            raise AdapterError("CF-E020", None, "Reaction control sector differs",
                               "Match the declared active-space sector")
        if sector["n_alpha"] == sector["n_beta"]:
            return ground_energy(hamiltonian, sector, store, method, s=s, total_spin=total_spin)
        from pyscf import cc, gto, mp, scf

        constant = 0.
        if (hamiltonian.space.frozen_orbitals or
                hamiltonian.space.n_orb != hamiltonian.mol.nao_nr()):
            mol = gto.M(verbose=0)
            mol.nelectron = sector["n_alpha"] + sector["n_beta"]
            mol.spin = sector["n_alpha"] - sector["n_beta"]
            mf = scf.UHF(mol)
            h1 = store.get(hamiltonian.hamiltonian.h1_ref)
            mf.get_hcore = lambda *args: h1
            mf.get_ovlp = lambda *args: np.eye(len(h1))
            mf._eri = store.get(hamiltonian.hamiltonian.h2_ref)
            constant = hamiltonian.hamiltonian.ledger.total().value
            initial_density = np.array([np.diag([1.] * sector[key] +
                                        [0.] * (len(h1) - sector[key]))
                                        for key in ("n_alpha", "n_beta")])
        else:
            mf = scf.UHF(hamiltonian.mol)
            initial_density = None
        mf.conv_tol = 1e-12
        mf.kernel(dm0=initial_density)
        if not mf.converged:
            raise ValueError("Unrestricted control failed to converge")
        if method == "hf" or mf.mol.nelectron == 1:
            return float(mf.e_tot + constant), {"s2": float(mf.spin_square()[0])}, {
                "weakening": {"relation": "approximate", "reason": "Unrestricted spin control"}}
        solver = mp.UMP2(mf) if method == "mp2" else cc.UCCSD(mf)
        solver.kernel()
        if method != "mp2" and not solver.converged:
            raise ValueError("Unrestricted coupled-cluster control failed to converge")
        energy = solver.e_tot + (solver.ccsd_t() if method == "ccsd_t" else 0)
        return float(energy + constant), {}
    return adapter


def register(registry):
    for name, inputs, output, adapter, params in (
        ("reaction_energy", {}, "estimate", reaction_energy, {"coefficients", "roles", "quantity"}),
        ("thermochemistry", {"molecule": "molecule", "model": "model",
                            "frequencies": "frequencies"}, "thermochemistry", thermochemistry,
         {"temperature_K", "pressure_Pa", "symmetry_number"}),
        ("geometry_optimize", {"molecule": "molecule", "model": "model"}, "molecule",
         geometry_optimize, {"kind", "level"}),
        ("frequencies", {"molecule": "molecule", "model": "model"}, "frequencies",
         frequencies, {"level"}),
    ):
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
                                       output=output, semantics=name))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
                          backend_kind=BackendKind.cpu_numeric, params=frozenset(params)))
    for method in ("hf", "mp2", "ccsd", "ccsd_t"):
        registry.register(ImplementationSpec(id=f"reaction_{method}", version="1",
                          target="ground_energy", implements=method,
                          adapter=_control_adapter(method), backend_kind=BackendKind.cpu_numeric,
                          params=frozenset({"s", "total_spin"})))
    registry.checks.append(static_check)
    registry.context_checks.append(context_check)
    registry.runtime_checks.append(verify_frequencies)


def base_plan(record, active=None):
    target = record.spec.target
    expected = {"n_orb": sum(item.active["n_orb"] for item in target.species),
                "n_elec": sum(item.active["n_elec"] for item in target.species)}
    if active is not None and active != expected:
        raise ValueError("Reaction active argument must equal the sum of declared species spaces")
    with build.program("reaction-controls") as builder:
        model = builder.electronic_model(**target.electronic_model)
        methods = {method: {} for method in ("hf", "mp2", "ccsd", "ccsd_t", "fci")}
        coefficients, roles = {}, {}
        for index, species in enumerate(target.species):
            port = f"species{index}"
            coefficients[port], roles[port] = species.coefficient, species.role
            molecule = builder.molecule(**species.molecule)
            if species.role == "ts" or target.thermo:
                level = {"method": "uhf" if species.molecule.get("spin", 0) else "rhf"}
                molecule = builder.node("geometry_optimize", molecule=molecule, model=model,
                                        kind=species.role, level=level)
                modes = builder.node("frequencies", molecule=molecule, model=model, level=level)
                if target.thermo:
                    thermal = builder.node("thermochemistry", molecule=molecule, model=model,
                        frequencies=modes, temperature_K=target.thermo.temperature_K,
                        pressure_Pa=target.thermo.pressure_Pa)
                    for inputs in methods.values():
                        inputs[f"thermo{index}"] = thermal
            ham = builder.hamiltonian(molecule, model, active=species.active)
            sector = builder.sector(**species.sector)
            for method, inputs in methods.items():
                node = builder.ground_energy(ham, sector, **(
                    {"s": species.target_spin} if species.target_spin is not None else {}))
                builder.refine(node, method)
                builder.bind(node, "fci_pyscf" if method == "fci" else f"reaction_{method}")
                inputs[port] = node
        controls = []
        for method, inputs in methods.items():
            node = builder.node("reaction_energy", **inputs, coefficients=coefficients, roles=roles,
                                **({"quantity": target.thermo.quantity} if target.thermo else {}))
            if method == "fci":
                builder.ir.references = [node.id]
            else:
                controls.append(node.id)
        builder.ir.targets = [*controls, *builder.ir.references]
        return builder.to_ir(), controls


def identity(plan, record, contexts):
    from chemflow.domains.baselines import METHODS
    from chemflow.verify.identity_diff import plan_hints

    target = record.spec.target
    nodes = {node.id: node for node in plan.nodes}
    issues = []
    for identifier in plan.targets:
        node = nodes.get(identifier)
        valid = node is not None and node.op == "reaction_energy"
        if valid:
            valid = len(node.inputs) == len(target.species) * (2 if target.thermo else 1)
            valid = valid and node.params.get("quantity") == (
                target.thermo.quantity if target.thermo else None)
            for index, species in enumerate(target.species):
                port = f"species{index}"
                energy = nodes.get(node.inputs.get(port))
                ham = nodes.get(energy.inputs.get("hamiltonian")) if energy else None
                sector = nodes.get(energy.inputs.get("sector")) if energy else None
                geometry = _geometry(nodes, energy.id) if energy else None
                molecule = _input_molecule(nodes, geometry)
                model = nodes.get(ham.inputs.get("model")) if ham else None
                if geometry is not None and geometry.op == "geometry_optimize":
                    optimization_model = nodes.get(geometry.inputs.get("model"))
                    levels = {"uhf" if species.molecule.get("spin", 0) else "rhf"}
                    if species.role == "minimum" and target.thermo is not None:
                        levels.update(METHODS)
                    valid = valid and ((species.role == "ts" or target.thermo is not None) and
                        geometry.params in [{"kind": species.role, "level": {"method": method}}
                                            for method in levels] and
                        optimization_model is not None and
                        optimization_model.params == target.electronic_model)
                if target.thermo:
                    thermal = nodes.get(node.inputs.get(f"thermo{index}"))
                    modes = nodes.get(thermal.inputs.get("frequencies")) if thermal else None
                    valid = valid and (geometry is not None and geometry.op == "geometry_optimize"
                        and thermal is not None and thermal.op == "thermochemistry"
                        and thermal.inputs.get("molecule") == geometry.id
                        and thermal.inputs.get("model") == ham.inputs.get("model")
                        and thermal.params == {"temperature_K": target.thermo.temperature_K,
                                               "pressure_Pa": target.thermo.pressure_Pa}
                        and modes is not None and modes.op == "frequencies"
                        and modes.inputs == {"molecule": geometry.id, "model": ham.inputs["model"]}
                        and modes.params == {"level": geometry.params["level"]})
                valid = valid and (energy is not None and energy.op == "ground_energy" and
                    ham is not None and ham.params.get("active") == species.active and
                    sector is not None and sector.params == species.sector and
                    contexts[energy.id].get("target_spin") == species.target_spin and
                    molecule is not None and {"charge": 0, "spin": 0, "unit": "angstrom",
                                              **molecule.params} ==
                    {"charge": 0, "spin": 0, "unit": "angstrom", **species.molecule} and
                    model is not None and model.params == target.electronic_model and
                    node.params.get("coefficients", {}).get(port) == species.coefficient and
                    node.params.get("roles", {}).get(port, "minimum") == species.role)
        if not valid:
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                message="Reaction target identity mismatch",
                fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
    return issues
