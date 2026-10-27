"""Deterministic bottom-up proof search; untrusted by the checking kernel."""

from fractions import Fraction

from chemflow.verify.lemmas import LEMMAS
from chemflow.verify.semantics import denotation, dependency_closure, facts


def prove(program, node_id, *, registry, computed):
    steps, latest, relations = [], {}, {}

    def add(node, lemma, suffix, relation, witness, uses):
        identifier = f"{node.id}:{suffix}"
        steps.append({"id": identifier, "lemma": lemma, "node": node.id,
                      "uses": uses, "side_conditions": witness,
                      "conclusion": {"node": node.id, "relation": relation}})
        latest[node.id] = identifier
        return identifier

    for node in dependency_closure(program, node_id):
        contract = denotation(node, registry)
        if contract is None or node.id not in computed:
            continue
        try:
            witness = facts(node, registry, computed)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError):
            continue
        dependencies = [latest[source] for source in dict.fromkeys(
                        node.inputs[port] for port in sorted(node.inputs))
                        if source in latest]
        axiom = add(node, contract["name"], "axiom", "denotation",
                    witness, dependencies)
        kind = contract["kind"]
        if kind == "data":
            continue
        uses = [*dependencies, axiom]

        def holds(lemma, witness=witness):
            return all(witness.get(name) is True for name in LEMMAS[lemma]["side_conditions"])

        sector = kind in ("exact", "ritz", "trial", "hf", "response_exact") and holds(
            "sector_decomposition")
        if sector:
            uses = [*uses, add(node, "sector_decomposition", "sector", "sector_invariant",
                               witness, uses)]
        lemma, relation = "no_ordering_law", "approximate"
        if kind == "empirical" and holds("empirical_model_evaluation"):
            lemma, relation = "empirical_model_evaluation", "equivalent"
        elif kind == "declared_model" and holds("declared_model_evaluation"):
            lemma, relation = "declared_model_evaluation", "equivalent"
        elif kind == "model" and holds("model_evaluation"):
            lemma, relation = "model_evaluation", "equivalent"
        elif kind == "thermal" and holds("rrho_evaluation"):
            lemma, relation = "rrho_evaluation", "equivalent"
        elif (kind == "energy_collection" and holds("energy_collection_exact") and
              all(relations.get(source) == "equivalent" for source in node.inputs.values())):
            lemma, relation = "energy_collection_exact", "equivalent"
        elif kind == "curve" and holds("pointwise_equality") and all(
                relations.get(source) == "equivalent" for source in node.inputs.values()):
            lemma, relation = "pointwise_equality", "equivalent"
        elif kind == "expectation" and (node.op != "correlation_function" or
                relations.get(node.inputs.get("state")) == "equivalent"):
            lemma, relation = "expectation_evaluation", "equivalent"
        elif kind == "exact" and sector and holds("exact_diagonalization"):
            lemma, relation = "exact_diagonalization", "equivalent"
        elif kind == "ritz" and sector and holds("rayleigh_ritz"):
            if witness.get("excited_roots"):
                uses = [*uses, add(node, "rayleigh_ritz_excited", "excited",
                                   "excited_ritz_bounds", witness, uses)]
            lemma, relation = "rayleigh_ritz", "variational_upper_bound"
            if node.op == "rayleigh_ritz" and holds("rayleigh_ritz_full"):
                lemma, relation = "rayleigh_ritz_full", "equivalent"
        elif kind in ("trial", "hf") and sector and holds("variational_principle"):
            lemma, relation = "variational_principle", "variational_upper_bound"
        elif kind == "sample" and holds("stochastic_estimate"):
            lemma, relation = "stochastic_estimate", "stochastic_estimate"
        elif kind == "linear" and holds("linear_combination_exact"):
            terms = [(Fraction(coefficient), relations.get(node.inputs[port]))
                     for port, coefficient in witness["coefficients"].items()
                     if Fraction(coefficient)]
            if all(upstream == "equivalent" for _, upstream in terms):
                lemma, relation = "linear_combination_exact", "equivalent"
            elif all(upstream == "equivalent" or
                     (coefficient > 0 and upstream == "variational_upper_bound")
                     for coefficient, upstream in terms):
                lemma, relation = "positive_linear_combination", "variational_upper_bound"
            elif all(upstream in ("equivalent", "stochastic_estimate") for _, upstream in terms):
                lemma, relation = "stochastic_linear_combination", "stochastic_estimate"
        elif (kind in ("response", "response_exact") and holds("lehmann_completeness") and
              (kind == "response_exact" or all(relations.get(source) == "equivalent"
                                               for source in node.inputs.values()))):
            lemma, relation = "lehmann_completeness", "equivalent"
        add(node, lemma, "result", relation, witness, uses)
        relations[node.id] = relation
    return {"version": 1, "node": node_id, "relation": relations.get(node_id, "declared"),
            "program_hash": program.hash, "steps": steps}
