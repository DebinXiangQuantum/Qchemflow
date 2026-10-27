from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

from chemflow.diagnostics import Diagnostic
from chemflow.ids import content_hash
from chemflow.ir.schema import Program
from chemflow.registry import Registry
from chemflow.relations import (
    TargetRelation,
    combine_difference,
    combine_linear,
    compose,
    response_relation,
    ritz_relation,
    satisfies,
)
from chemflow.types import ActiveSpaceSpec, ElectronicModel
from chemflow.types.model import model_protocol_id
from chemflow.units import Q


def _plain_json(value: object) -> bool:
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json(item) for key, item in value.items())
    return False

if TYPE_CHECKING:
    from chemflow.runtime.task import TaskRecord


def parameter_diagnostics(program, registry, *, require_complete=True):
    issues = []
    for node in program.nodes:
        failures = []
        impl = registry.implementations.get(node.binding or node.op)
        if impl is not None and require_complete:
            spec = registry.operators.get(f"chemflow.{node.op}@1")
            refinement = registry.refinements.get(node.refinement or impl.implements or "")
            ports = {*(spec.inputs if spec else {}),
                     *(refinement.inputs if refinement else {}), *node.inputs}
            signature = inspect.signature(impl.adapter)
            required = {name for name, parameter in signature.parameters.items()
                        if parameter.default is inspect.Parameter.empty and parameter.kind in
                        (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                        and name not in {"store", *ports}}
            failures.extend(f"Missing required parameter {name}" for name in
                            sorted(required - node.params.keys()))
        required = {"molecule": {"atoms"}, "sector": {"n_alpha", "n_beta", "n_orb"},
                    "electronic_model": {"basis"},
                    "impurity_model": {"U", "eps_bath", "t_bath"}}.get(node.op, set())
        if require_complete:
            failures.extend(f"Missing required parameter {name}" for name in
                            sorted(required - node.params.keys()))
        for name, value in node.params.items():
            expected = None
            if name in {"n_alpha", "n_beta", "n_orb", "n_elec", "shots", "seed", "charge"}:
                expected = (int,)
            elif name in {"s", "total_spin"} and value is not None:
                expected = (int, float)
            elif name == "spin" and node.op in {"sector", "molecule", "qmmm_energy"}:
                expected = (int,)
            elif name in {"atoms", "basis", "unit"} and node.op in {
                    "molecule", "electronic_model"}:
                expected = (str,)
            elif name in {"active", "solver_options"} and value is not None:
                expected = (dict,)
            elif name in {"method", "measurement", "ansatz", "operator", "closure", "quantity",
                          "solver", "model", "geometry", "unit", "kind", "spin"}:
                expected = (str,)
            elif name in {"orbital", "n_roots"}:
                expected = (int,)
            elif name in {"temperature_K", "pressure_Pa", "U", "t", "eta"}:
                expected = (int, float)
            elif name in {"coefficients", "roles", "level", "molecule", "electronic_model"} or (
                    name == "target" and node.op == "interaction_energy"):
                expected = (dict,)
            elif name == "model_change":
                expected = (bool,)
            if expected and (type(value) not in expected or
                             type(value) is float and not math.isfinite(value)):
                failures.append(f"Invalid type for parameter {name}")
            if name == "coefficients" and isinstance(value, dict):
                failures.extend(f"Invalid numeric parameter coefficients.{port}"
                                for port, amount in value.items()
                                if type(amount) not in (int, float) or
                                not math.isfinite(amount))
            if name in {"roles", "level"} and isinstance(value, dict):
                failures.extend(f"Invalid string parameter {name}.{key}"
                                for key, item in value.items() if not isinstance(item, str)
                                and not (name == "level" and item is None))
        if node.op == "metabolic_sites":
            from chemflow.agent.task import MetabolicSitesTarget

            try:
                MetabolicSitesTarget.model_validate({"kind": "metabolic_sites", **node.params})
            except (ValueError, TypeError) as exc:
                failures.append(str(exc))
        if node.op in {"logs_esol", "logs_gse", "empirical_estimate"}:
            from chemflow.empirical import molecule

            try:
                molecule(node.params.get("molecule_smiles"))
                model = node.params.get("model") if node.op == "empirical_estimate" else (
                    node.op.removeprefix("logs_"))
                if model not in ("esol", "gse"):
                    raise ValueError("Unknown empirical model")
                melting = node.params.get("melting_point_C")
                if (model == "gse" or melting is not None) and (
                        type(melting) not in (int, float) or not math.isfinite(melting)):
                    raise ValueError("melting_point_C must be finite")
            except (ValueError, TypeError) as exc:
                failures.append(str(exc))
        if node.binding == "dft_pyscf":
            if not isinstance(node.params.get("functional"), str) or not node.params["functional"]:
                failures.append("DFT requires a functional string")
            if node.params.get("dispersion") not in (None, "d3bj", "d4"):
                failures.append("Dispersion must be None, d3bj or d4")
        if node.op in ("solvation_free_energy", "frontier_orbitals"):
            from chemflow.agent.task import FrontierTarget, SolvationTarget

            schema = SolvationTarget if node.op == "solvation_free_energy" else FrontierTarget
            try:
                schema.model_validate({"kind": node.op, **node.params})
            except ValueError as exc:
                failures.append(str(exc))
        if node.op in ("charge_energy_difference", "eom_charge_energy"):
            from chemflow.agent.task import IonizationTarget

            try:
                target = IonizationTarget.model_validate(node.params["target"] if
                    node.op == "charge_energy_difference" else node.params)
                if node.op == "charge_energy_difference":
                    sign = 1 if target.kind == "ionization_energy" else -1
                    if node.params.get("coefficients") != {"initial": -sign, "final": sign}:
                        failures.append("Charge-energy coefficients disagree with IP/EA definition")
            except (ValueError, KeyError) as exc:
                failures.append(str(exc))
        if node.op in {"protein_structure", "ligand_structure", "complex", "pocket",
                       "ligand_parameters", "mmgbsa", "fragment_interaction"}:
            from chemflow.domains.binding import validate_parameters

            failures.extend(validate_parameters(node.op, node.params))
        if impl is not None and impl.implements == "vqe_sampled":
            for name, default, minimum, maximum in (("max_evaluations", 60, 2, math.inf),
                    ("shots_per_group", 1024, 1, 50000), ("n_groups", 1, 0, math.inf)):
                value = node.params.get(name, default)
                if type(value) is not int or not minimum <= value <= maximum:
                    failures.append(f"Invalid sampled VQE parameter {name}")
            for name, default, choices in (("ansatz", "uccsd", ("uccsd", "lucj")),
                    ("optimizer", "cobyla", ("cobyla", "spsa"))):
                if node.params.get(name, default) not in choices:
                    failures.append(f"Invalid sampled VQE parameter {name}")
            sampler = {"vqe_lqcloud": "lqcloud", "vqe_aer_mq02": "aer_mq02",
                       "vqe_shots_ideal": "ideal"}.get(impl.id)
            if node.params.get("sampler", sampler) != sampler:
                failures.append("Sampler conflicts with binding")
        if node.op == "component_energy" and node.params.get("method") not in (
                "hf", "fci", "mp2", "ccsd", "ccsd_t"):
            failures.append("Invalid parameter method: choose hf, fci, mp2, ccsd or ccsd_t")
        if node.op in {"app_component_energy", "bond_dissociation", "ranking",
                       "site_reactivity", "boltzmann_weights", "conformer_search",
                       "enumerate_bonds", "homolysis"}:
            from chemflow.domains.reactivity import parameter_errors

            failures.extend(parameter_errors(node))
        if node.op in ("geometry_transform", "constrained_optimize", "neb_path", "irc",
                       "rate_constant", "connected_ts"):
            from chemflow.domains.paths import validate_parameters

            try:
                validate_parameters(node.op, node.params)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                failures.append(str(exc))
        if node.op in ("pes_scan", "interaction_scan", "scan_fragment", "scan_interaction"):
            from chemflow.domains.scans import validate_parameters as validate_scan_parameters

            try:
                validate_scan_parameters(node.op, node.params)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError,
                    AssertionError, RuntimeError, OverflowError) as exc:
                failures.append(str(exc))
        if not _plain_json(node.params):
            failures.append("IR parameters must be finite plain JSON")
        issues.extend(Diagnostic(code="CF-E020" if node.op == "sector" else "CF-E080",
                                 node=node.id, message=message) for message in failures)
    return issues


def infer_context(program: Program, registry: Registry) -> tuple[dict[str, dict], list[Diagnostic]]:
    contexts: dict[str, dict] = {}
    issues = parameter_diagnostics(program, registry)
    if issues:
        return {node.id: {} for node in program.nodes}, issues
    nodes = {node.id: node for node in program.nodes}

    def error(code: str, node: str, message: str, hint: str) -> None:
        issues.append(Diagnostic(code=code, node=node, message=message, fix_hints=[hint]))

    def parse_active(node_id: str, raw: object) -> ActiveSpaceSpec | None:
        try:
            active = ActiveSpaceSpec.model_validate(raw)
        except ValueError as exc:
            error("CF-E080", node_id, f"Invalid active-space specification: {exc}",
                  "Supply n_elec and n_orb as integers")
            return None
        if (active.n_orb <= 0 or active.n_elec < 0 or
            active.n_elec > 2 * active.n_orb):
            error("CF-E020", node_id, "Inconsistent active-space electron/orbital counts",
                  "Use 0 <= n_elec <= 2 * n_orb and n_orb > 0")
            return None
        return active

    remaining = list(program.nodes)
    while remaining:
        ready = [node for node in remaining if all(source in contexts for source in
                                                    node.inputs.values())]
        if not ready:
            for node in remaining:
                error("CF-E080", node.id, "Missing source or cyclic graph", "Connect valid inputs")
            break
        for node in ready:
            remaining.remove(node)
            spec = registry.operators.get(f"chemflow.{node.op}@1")
            if spec is None:
                error("CF-E080", node.id, "Unsupported operator", "Register operator")
                contexts[node.id] = {}
                continue
            refinement = registry.refinements.get(node.refinement or "")
            required = {**spec.inputs, **(refinement.inputs if refinement else {})}
            for port, expected in required.items():
                source = nodes.get(node.inputs.get(port, ""))
                if source is None:
                    error("CF-E080", node.id, f"Missing input {port}", "Connect required port")
                elif contexts[source.id].get("type") != expected:
                    error("CF-E002", node.id, f"{port} requires {expected}", "Connect typed input")
            inputs = {port: contexts[source] for port, source in node.inputs.items()}
            for rule in [*spec.context_rules, *(refinement.context_rules if refinement else [])]:
                code, left, right, key = rule.split(":")
                if left in inputs and right in inputs and inputs[left].get(key) != inputs[right].get(key):
                    if code == "CF-E011" and node.params.get("model_change"):
                        continue
                    error(code, node.id, f"{key} mismatch between {left} and {right}",
                          "Align source contexts or declare model_change")
            context: dict = {"type": spec.output}
            if node.op == "molecule":
                source = {key: node.params.get(key, default) for key, default in
                          (("atoms", None), ("unit", "angstrom"), ("charge", 0), ("spin", 0))}
                species = [atom.strip().split()[0] for atom in
                           source["atoms"].replace("\n", ";").split(";") if atom.strip()]
                context.update(geometry_id=content_hash(source), species_id=content_hash(
                    {"species": species, "charge": source["charge"], "spin": source["spin"]}))
                if source["unit"] not in ("angstrom", "bohr"):
                    error("CF-E001", node.id, "Unsupported coordinate unit", "Use angstrom or bohr")
            elif node.op == "electronic_model":
                try:
                    context["model_id"] = content_hash(ElectronicModel(**node.params))
                except ValueError as exc:
                    error("CF-E080", node.id, str(exc), "Supply a valid electronic model")
            elif node.op in ("scf", "hamiltonian"):
                molecule = inputs.get("molecule", {})
                model = inputs.get("model", {})
                context.update(geometry_id=molecule.get("geometry_id"),
                               model_id=model.get("model_id"))
                if context["model_id"] is not None:
                    active = (parse_active(node.id, node.params["active"])
                              if "active" in node.params else None)
                    context["model_protocol_id"] = model_protocol_id(context["model_id"], active)
                context["hamiltonian_id"] = content_hash((context["geometry_id"],
                                                          context["model_id"],
                                                          node.params.get("active")))
            elif node.op in ("integrals", "active_space"):
                context.update(inputs.get("scf", {}))
                context["type"] = spec.output
                if node.op == "active_space":
                    context["model_protocol_id"] = model_protocol_id(
                        context.get("model_id"), parse_active(node.id, node.params))
                    context["hamiltonian_id"] = content_hash((context.get("hamiltonian_id"),
                                                              node.params))
            elif node.op == "sector":
                alpha, beta = node.params.get("n_alpha"), node.params.get("n_beta")
                count, projection = node.params.get("n_elec"), node.params.get("spin")
                spin = node.params.get("s", node.params.get("total_spin"))
                orbital = node.params.get("n_orb")
                if (alpha is None or beta is None or orbital is None or
                    min(alpha, beta) < 0 or max(alpha, beta) > orbital or
                    (count is not None and (count != alpha + beta or
                                            (count - (alpha - beta)) % 2)) or
                    (projection is not None and projection != alpha - beta) or
                    (spin is not None and (2 * spin < abs(alpha - beta) or
                                           (2 * spin - (alpha - beta)) % 2))):
                    error("CF-E020", node.id, "Infeasible particle/spin sector",
                          "Match electrons, spin projection and orbital count")
                context["sector_id"] = content_hash(node.params)
            elif node.op == "encode":
                context.update(inputs.get("hamiltonian", {}))
                context["type"] = spec.output
                context["encoding_id"] = content_hash((context.get("hamiltonian_id"), "JW"))
                ham = nodes.get(node.inputs.get("hamiltonian", ""))
                active = ham.params.get("active", {}) if ham else {}
                if isinstance(active, dict) and isinstance(active.get("n_orb"), int) and active["n_orb"] > 7:
                    error("CF-E080", node.id, "Dense encoding exceeds 14 qubits",
                          "Use a smaller active space or sector method")
            elif node.op in ("prepare", "prepare_state"):
                context.update(inputs.get("hamiltonian", {}))
                context["type"] = spec.output
                context["sector_id"] = inputs.get("sector", {}).get("sector_id")
                context["encoding_id"] = inputs.get("encoding", {}).get("encoding_id")
            elif node.op == "sample_configurations":
                context.update(inputs.get("prep", {}))
                context["type"] = spec.output
            elif node.op in ("ground_energy", "expectation", "sample"):
                source = inputs.get("hamiltonian", inputs.get("operator", {}))
                context.update(source)
                context.update(type=spec.output, kind="total_electronic_energy", unit="hartree")
                if node.op == "ground_energy":
                    context["sector_id"] = inputs.get("sector", {}).get("sector_id")
                    sector_node = nodes.get(node.inputs.get("sector"))
                    sector_params = sector_node.params if sector_node is not None else {}
                    context["target_spin"] = node.params.get(
                        "s", node.params.get("total_spin", sector_params.get(
                            "s", sector_params.get("total_spin"))))
            elif node.op in ("energy_difference", "compare"):
                left, right = inputs.get("left", {}), inputs.get("right", {})
                if left.get("kind") != right.get("kind"):
                    error("CF-E002", node.id, "Quantity kinds differ", "Compare like quantities")
                if left.get("unit") != right.get("unit"):
                    error("CF-E001", node.id, "Units differ", "Convert units explicitly")
                context.update(left)
                context.update(type=spec.output, kind="energy_difference")
            for plugin in registry.context_checks:
                try:
                    issues.extend(plugin(node, inputs, context, nodes))
                except (TypeError, ValueError, KeyError, AttributeError) as exc:
                    error("CF-E080", node.id, str(exc), "Supply valid operator parameters")
            contexts[node.id] = context
    return contexts, issues


def infer_relations(program: Program, registry: Registry) -> dict[str, TargetRelation]:
    relations: dict[str, TargetRelation] = {}
    invalid = {issue.node for issue in parameter_diagnostics(program, registry,
                                                            require_complete=False)}
    full_spaces: set[str] = set()
    full_spectra: set[str] = set()
    remaining = list(program.nodes)
    while remaining:
        ready = [node for node in remaining if all(source in relations for source in
                                                    node.inputs.values())]
        if not ready:
            break
        for node in ready:
            remaining.remove(node)
            if node.id in invalid or any(source in invalid for source in node.inputs.values()):
                invalid.add(node.id)
                relations[node.id] = TargetRelation.model_change
                continue
            refinement = registry.refinements.get(node.refinement or
                                                  ("fci" if node.op == "ground_energy" else ""))
            relation = refinement.relation if refinement else TargetRelation.equivalent
            if node.op == "full_subspace":
                full_spaces.add(node.id)
            if node.op == "rayleigh_ritz":
                full = node.inputs.get("subspace") in full_spaces
                relation = ritz_relation(full)
                if full:
                    full_spectra.add(node.id)
            if node.op == "spectral_function":
                relation = response_relation(
                    node.refinement == "exact_lehmann" or
                    all(node.inputs.get(port) in full_spectra
                        for port in ("ground", "particle", "hole")))
            if node.op == "sample":
                relation = compose(relation, TargetRelation.stochastic_estimate)
            if node.op == "periodic_scf":
                relation = TargetRelation.approximate
            if node.op == "correlation_function":
                relation = (TargetRelation.equivalent if relations.get(node.inputs.get("state"))
                            == TargetRelation.equivalent else TargetRelation.approximate)
            if node.op == "allen_dynes_tc":
                relation = TargetRelation.approximate
            if node.op == "ground_energy" and node.params.get("measurement") == "shots":
                relation = compose(relation, TargetRelation.stochastic_estimate)
            if node.op in ("energy_difference", "compare") and {"left", "right"} <= node.inputs.keys():
                relation = combine_difference(relations[node.inputs["left"]],
                                              relations[node.inputs["right"]])
                if node.params.get("model_change"):
                    relation = TargetRelation.model_change
            if node.op == "qmmm_energy":
                from chemflow.domains.qmmm import relation as qmmm_relation

                relation = TargetRelation(qmmm_relation(node))
            if node.op in ("component_energy", "app_component_energy"):
                relation = {"hf": TargetRelation.variational_upper_bound,
                            "fci": TargetRelation.equivalent}.get(
                                node.params.get("method"), TargetRelation.approximate)
            if node.op in ("bond_dissociation", "ranking"):
                relation = (TargetRelation.equivalent if node.inputs and all(
                    relations[source] == TargetRelation.equivalent for source in node.inputs.values())
                    else TargetRelation.approximate)
            if node.op in ("enumerate_bonds", "homolysis", "site_reactivity",
                           "boltzmann_weights", "conformer_search"):
                relation = TargetRelation.approximate
            if node.op in ("dock", "fragment_interaction"):
                relation = TargetRelation.approximate
            if node.op == "potential_energy":
                relation = TargetRelation.model_change
            if node.op == "eom_charge_energy":
                relation = TargetRelation.approximate
            if node.op in ("reaction_energy", "interaction_energy", "charge_energy_difference",
                           "scan_interaction"):
                relation = combine_linear([(node.params.get("coefficients", {}).get(port, 0),
                                            relations[source])
                                           for port, source in node.inputs.items()])
            if node.op in ("pes_scan", "interaction_scan"):
                relation = (TargetRelation.equivalent if node.inputs and all(
                    relations[source] == TargetRelation.equivalent for source in node.inputs.values())
                    else TargetRelation.approximate)
            relations[node.id] = relation
    return relations


def check(program: Program, registry: Registry, task: TaskRecord | None = None,
          purpose: str = "screen") -> list[Diagnostic]:
    if purpose not in ("screen", "deploy", "production"):
        raise ValueError(f"Unknown checker purpose: {purpose}")
    issues: list[Diagnostic] = []
    if purpose != "screen":
        from chemflow.check.deployment import deployment

        issues.extend(deployment(program, registry, purpose=purpose)[1])

    def error(code: str, node: str | None, message: str, hint: str) -> None:
        issues.append(Diagnostic(code=code, node=node, message=message, fix_hints=[hint]))

    if program.gates and task is None:
        error("CF-E110", None, "Gates require a frozen task record", "Freeze the task")
    if task is not None:
        for index, gate in enumerate(program.gates):
            if task.gate_hashes.get(str(index)) != content_hash(gate.payload()):
                error("CF-E110", gate.node, "Gate changed after task freeze", "Freeze a new task")
        if len(task.gate_hashes) != len(program.gates):
            error("CF-E110", None, "Gate set changed after task freeze", "Freeze a new task")
        if task.program_hash != program.hash:
            error("CF-E110", None, "Program changed after task freeze", "Freeze a new task")
    contexts, inferred = infer_context(program, registry)
    issues.extend(inferred)
    invalid_parameters = {issue.node for issue in parameter_diagnostics(program, registry)}
    nodes = {node.id: node for node in program.nodes}
    for node in program.nodes:
        if not _plain_json(node.params):
            error("CF-E080", node.id, "IR parameters must be plain JSON",
                  "Pass domain objects through a typed input port")
        impl = registry.implementations.get(node.binding or node.op)
        if impl is None or impl.target != node.op:
            error("CF-E080", node.id, "Unsupported implementation binding", "Bind registered impl")
            continue
        unexpected = set(node.params) - (impl.params or frozenset())
        if unexpected:
            error("CF-E080", node.id, f"Unsupported implementation parameters: {sorted(unexpected)}",
                  "Use only parameters declared by the bound implementation")
        template = node.refinement or ("fci" if node.op == "ground_energy" else None)
        if template in ("fci_pyscf", "sector_diag_numpy"):
            template = "fci"
        if node.refinement and template not in registry.refinements:
            error("CF-E080", node.id, "Unknown refinement", "Register template")
        elif template and impl.implements != template:
            error("CF-E050", node.id, "Binding does not implement refinement",
                  "Bind an implementation of the selected template")
        if node.params.get("evidence"):
            error("CF-E040", node.id, "IR cannot assert evidence", "Let runtime issue evidence")
        if node.op == "ground_energy":
            sector = nodes.get(node.inputs.get("sector", ""))
            hamiltonian = nodes.get(node.inputs.get("hamiltonian", ""))
            if sector and node.id not in invalid_parameters and sector.id not in invalid_parameters:
                requested = node.params.get("s", node.params.get("total_spin"))
                bound = sector.params.get("s", sector.params.get("total_spin"))
                alpha, beta = sector.params.get("n_alpha", 0), sector.params.get("n_beta", 0)
                if (requested is not None and bound is not None and requested != bound or
                    requested is not None and 2 * requested < abs(alpha - beta)):
                    error("CF-E020", node.id, "Spin request incompatible with sector",
                          "Choose a feasible spin")
                active = hamiltonian.params.get("active", {}) if hamiltonian else {}
                if (isinstance(active, dict) and active and
                    (active.get("n_orb") != sector.params.get("n_orb") or
                                active.get("n_elec") != alpha + beta)):
                    error("CF-E020", node.id, "Sector differs from active Hamiltonian",
                          "Match active electrons and orbitals")
        if (impl.effects not in ("pure", "random", "network") or
                node.effects not in ("pure", "random", "network")):
            error("CF-E080", node.id, "Unknown determinism effect",
                  "Use pure, random or network effects")
    relations = infer_relations(program, registry)
    for gate in program.gates:
        node = nodes.get(gate.node)
        target_unit = contexts.get(gate.node, {}).get("unit", "hartree")
        if gate.abs_error_le is not None and (
                (gate.abs_error_le.dimension != Q(1, target_unit).dimension
                 if "unit" in contexts.get(gate.node, {}) else
                 gate.abs_error_le.dimension not in ("energy", "dimensionless")) or
                not isinstance(gate.abs_error_le.value, (int, float)) or
                not math.isfinite(gate.abs_error_le.value) or gate.abs_error_le.value < 0):
            error("CF-E001", gate.node, "Absolute-error threshold must match the target unit",
                  "Use a finite nonnegative target-compatible threshold")
        if gate.l1_error_le is not None and (
                gate.l1_error_le.dimension != "dimensionless" or
                not isinstance(gate.l1_error_le.value, (int, float)) or
                not math.isfinite(gate.l1_error_le.value) or gate.l1_error_le.value < 0):
            error("CF-E001", gate.node, "L1 integral threshold must be nonnegative dimensionless",
                  "Use a finite dimensionless threshold")
        if gate.l1_error_le is not None and gate.reference is not None:
            if (contexts.get(gate.node, {}).get("type") != "spectral_estimate" or
                    contexts.get(gate.node, {}).get("spectral_identity") !=
                    contexts.get(gate.reference, {}).get("spectral_identity")):
                from chemflow.verify.identity_diff import identity_hints, signature_tree

                issues.append(Diagnostic(code="CF-E120", node=gate.node,
                    message="Spectral gate target identities differ", fix_hints=identity_hints(
                        signature_tree(program, gate.node),
                        [signature_tree(program, gate.reference)])))
            if relations.get(gate.reference) != TargetRelation.equivalent:
                error("CF-E050", gate.node, "L1 gate requires an exact reference",
                      "Use an equivalent spectral reference")
        if gate.abs_error_le is not None and gate.reference is not None:
            candidate = contexts.get(gate.node, {})
            reference = contexts.get(gate.reference, {})
            if (node is None or nodes.get(gate.reference) is None or
                not (node.op == nodes[gate.reference].op or
                     {node.op, nodes[gate.reference].op} in (
                         {"sample", "expectation"},
                         {"charge_energy_difference", "eom_charge_energy"})) or
                any(candidate.get(key) != reference.get(key) for key in (
                    "hamiltonian_id", "model_protocol_id", "sector_id", "target_spin",
                    "encoding_id", "quantity", "site_pair"))):
                error("CF-E011", gate.node, "Absolute-error targets differ",
                      "Use a reference with the same operator, model, sector and spin target")
        if (gate.abs_error_le is not None and gate.reference is not None and
            relations.get(gate.reference) != TargetRelation.equivalent):
            error("CF-E050", gate.node, "Absolute-error gate requires an exact reference",
                  "Use an equivalent reference")
        if (node and gate.relation is not None and
            (gate.relation not in TargetRelation._value2member_map_ or
             not satisfies(relations.get(node.id), TargetRelation(gate.relation)))):
            error("CF-E050", node.id, "Gate requires a stronger relation",
                  "Choose a refinement satisfying the relation")
    for limit in program.budget.values():
        if limit < 0:
            error("CF-E070", None, "Negative budget", "Use nonnegative limit")
    if program.loops:
        error("CF-E080", None, "Bounded loops not executable", "Unroll explicitly")
    if not invalid_parameters:
        for plugin in registry.checks:
            issues.extend(plugin(program, registry))
    return issues
