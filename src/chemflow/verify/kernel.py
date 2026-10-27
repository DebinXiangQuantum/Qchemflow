"""Small certificate checker. This module never imports or invokes the prover."""

from dataclasses import dataclass
from fractions import Fraction

from chemflow.ids import canonical_json
from chemflow.verify.formal import checked_lemmas
from chemflow.verify.lemmas import AXIOMS, LEMMAS
from chemflow.verify.semantics import denotation, dependency_closure, facts


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reasons: tuple[str, ...]
    machine_checked: tuple[str, ...] = ()
    trusted: tuple[str, ...] = ()


RULES = {
    "empirical_model_evaluation": ({"empirical"}, "equivalent"),
    "declared_model_evaluation": ({"declared_model"}, "equivalent"),
    "energy_collection_exact": ({"energy_collection"}, "equivalent"),
    "pointwise_equality": ({"curve"}, "equivalent"),
    "model_evaluation": ({"model"}, "equivalent"),
    "rrho_evaluation": ({"thermal"}, "equivalent"),
    "expectation_evaluation": ({"expectation"}, "equivalent"),
    "exact_diagonalization": ({"exact"}, "equivalent"),
    "sector_decomposition": ({"exact", "ritz", "trial", "hf", "response_exact"}, "sector_invariant"),
    "rayleigh_ritz": ({"ritz"}, "variational_upper_bound"),
    "rayleigh_ritz_excited": ({"ritz"}, "excited_ritz_bounds"),
    "rayleigh_ritz_full": ({"ritz"}, "equivalent"),
    "variational_principle": ({"trial", "hf"}, "variational_upper_bound"),
    "positive_linear_combination": ({"linear"}, "variational_upper_bound"),
    "linear_combination_exact": ({"linear"}, "equivalent"),
    "stochastic_linear_combination": ({"linear"}, "stochastic_estimate"),
    "lehmann_completeness": ({"response", "response_exact"}, "equivalent"),
    "stochastic_estimate": ({"sample"}, "stochastic_estimate"),
    "no_ordering_law": ({"exact", "ritz", "trial", "hf", "linear", "response",
                         "response_exact", "sample", "expectation", "opaque", "thermal",
                         "empirical", "declared_model", "energy_collection", "curve", "model"}, "approximate"),
}


def _check(certificate, program, registry, computed):
    if set(certificate) != {"version", "node", "relation", "program_hash", "steps"}:
        raise ValueError("Invalid certificate fields")
    if type(certificate["version"]) is not int or certificate["version"] != 1:
        raise ValueError("Unknown certificate version")
    if certificate["program_hash"] != program.hash:
        raise ValueError("Program hash mismatch")
    closure = dependency_closure(program, certificate["node"])
    nodes = {node.id: node for node in closure}
    witnesses = {node.id: facts(node, registry, computed) for node in closure}
    accepted, latest = {}, {}
    for step in certificate["steps"]:
        if set(step) != {"id", "lemma", "node", "uses", "side_conditions", "conclusion"}:
            raise ValueError("Invalid step fields")
        node = nodes[step["node"]]
        contract = denotation(node, registry)
        if contract is None or not witnesses[node.id]["known_denotation"]:
            raise ValueError("Unknown or unexecuted denotation")
        if step["id"] in accepted:
            raise ValueError("Duplicate step id")
        if canonical_json(step["side_conditions"]) != canonical_json(witnesses[node.id]):
            raise ValueError("Side-condition witness differs from recomputation")
        lemma = step["lemma"]
        dependencies = [latest[source] for source in dict.fromkeys(
                        node.inputs[port] for port in sorted(node.inputs))
                        if source in latest]
        uses = dependencies.copy()
        if lemma in AXIOMS:
            if lemma != contract["name"]:
                raise ValueError("Binding axiom does not apply to this node")
            suffix, relation = "axiom", "denotation"
        else:
            if lemma not in LEMMAS or lemma not in RULES:
                raise ValueError("Unknown lemma or axiom")
            kinds, relation = RULES[lemma]
            if contract["kind"] not in kinds:
                raise ValueError("Lemma does not apply to this denotation")
            if not all(witnesses[node.id].get(name) is True
                       for name in LEMMAS[lemma]["side_conditions"]):
                raise ValueError("False or missing side condition")
            uses.append(f"{node.id}:axiom")
            if lemma == "expectation_evaluation" and node.op == "correlation_function":
                source = node.inputs.get("state")
                source_step = accepted.get(latest.get(source), {})
                if source_step.get("conclusion", {}).get("relation") != "equivalent":
                    raise ValueError("Ground-state correlations require an exact source state")
            suffix = {"sector_decomposition": "sector", "rayleigh_ritz_excited":
                      "excited"}.get(lemma, "result")
            sector_step = f"{node.id}:sector"
            if (lemma in ("exact_diagonalization", "rayleigh_ritz", "rayleigh_ritz_full",
                          "rayleigh_ritz_excited",
                          "variational_principle") and sector_step not in accepted):
                raise ValueError("Missing sector decomposition")
            if suffix in ("result", "excited") and sector_step in accepted:
                uses.append(sector_step)
            if lemma in ("rayleigh_ritz", "rayleigh_ritz_full") and witnesses[node.id].get(
                    "excited_roots"):
                excited_step = f"{node.id}:excited"
                if excited_step not in accepted:
                    raise ValueError("Missing excited-root citation")
                uses.append(excited_step)
            if lemma in ("linear_combination_exact", "positive_linear_combination",
                         "stochastic_linear_combination"):
                for port, coefficient in witnesses[node.id]["coefficients"].items():
                    coefficient = Fraction(coefficient)
                    if not coefficient:
                        continue
                    source = node.inputs[port]
                    source_step = accepted.get(latest.get(source), {})
                    upstream = source_step.get("conclusion", {}).get("relation")
                    if upstream == "equivalent":
                        continue
                    if lemma == "stochastic_linear_combination" and upstream == "stochastic_estimate":
                        continue
                    if (lemma == "positive_linear_combination" and coefficient > 0 and
                            upstream == "variational_upper_bound"):
                        continue
                    raise ValueError("Linear combination lacks the required input relations")
            if lemma == "energy_collection_exact" and any(
                    accepted.get(latest.get(source), {}).get("conclusion", {}).get("relation")
                    != "equivalent" for source in node.inputs.values()):
                raise ValueError("Energy collection requires exact component energies")
            if (lemma == "lehmann_completeness" and contract["kind"] == "response" and
                    any(accepted.get(latest.get(source), {}).get("conclusion", {}).get(
                        "relation") != "equivalent" for source in node.inputs.values())):
                raise ValueError("Lehmann requires proved complete input spectra")
            if lemma == "pointwise_equality" and any(
                    accepted.get(latest.get(source), {}).get("conclusion", {}).get("relation")
                    != "equivalent" for source in node.inputs.values()):
                raise ValueError("Pointwise equality requires proved exact input energies")
        if step["id"] != f"{node.id}:{suffix}":
            raise ValueError("Invalid step identity")
        if step["uses"] != uses or any(identifier not in accepted for identifier in uses):
            raise ValueError("Invalid or forward step references")
        if step["conclusion"] != {"node": node.id, "relation": relation}:
            raise ValueError("Conclusion does not follow from the rule")
        accepted[step["id"]] = step
        latest[node.id] = step["id"]
    if not certificate["steps"] or certificate["steps"][-1]["conclusion"] != {
            "node": certificate["node"], "relation": certificate["relation"]}:
        raise ValueError("Missing final conclusion")
    if certificate["relation"] not in ("equivalent", "variational_upper_bound",
                                       "stochastic_estimate", "approximate"):
        raise ValueError("Not a proved target relation")


def check_certificate(certificate, program, *, registry, computed):
    try:
        _check(certificate, program, registry, computed)
        used = {step["lemma"] for step in certificate["steps"]}
        checked = checked_lemmas(used)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        return Verdict(False, (str(exc),))
    return Verdict(True, (), tuple(sorted(checked)), tuple(sorted(used - checked)))
