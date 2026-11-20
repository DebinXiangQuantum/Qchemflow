//! Target relations: how a value relates to the exact answer of the question it states.
//!
//! Every estimate-producing primitive declares a relation in `primitives.toml`, either
//! fixed (`fci.solve` is `equivalent`, `scf.rhf` a `variational_upper_bound`) or as a
//! rule that derives it from its inputs (`estimate.sub` composes two relations the way a
//! difference does). `infer` propagates them through a graph, so a signed
//! `relation_at_least` gate is a static property of a compiled plan rather than a claim
//! a result makes about itself.

use serde::{Deserialize, Serialize};

use crate::primitives::primitive;
use crate::Graph;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Equivalent,
    VariationalUpperBound,
    StochasticEstimate,
    Approximate,
    ModelChange,
}

/// Fixed relation names accepted in `primitives.toml`.
pub const RELATIONS: [&str; 5] = [
    "equivalent",
    "variational_upper_bound",
    "stochastic_estimate",
    "approximate",
    "model_change",
];

/// Rules that derive a relation from a node's inputs or parameters:
/// - `input`: the relation of input 0 (projections, unit changes, copies);
/// - `sum`: a positively weighted sum of the inputs;
/// - `difference`: input 0 minus input 1;
/// - `scale`: input 0 times `params.factor` (a negative factor reverses a bound);
/// - `decreasing`: a monotonically decreasing function of input 0;
/// - `meet`: the weakest relation all inputs share (orderings, comparisons, branches);
/// - `param`: looked up in `relations` by the value of `params[relation_param]`.
pub const RULES: [&str; 7] = [
    "input",
    "sum",
    "difference",
    "scale",
    "decreasing",
    "meet",
    "param",
];

impl Relation {
    pub fn parse(name: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(name.to_string())).ok()
    }

    pub fn name(self) -> &'static str {
        match self {
            Relation::Equivalent => "equivalent",
            Relation::VariationalUpperBound => "variational_upper_bound",
            Relation::StochasticEstimate => "stochastic_estimate",
            Relation::Approximate => "approximate",
            Relation::ModelChange => "model_change",
        }
    }

    /// Whether this relation meets a required minimum (as `chemflow.relations.satisfies`).
    pub fn satisfies(self, required: Relation) -> bool {
        self == Relation::Equivalent
            || self == required
            || (required == Relation::Approximate && self != Relation::ModelChange)
    }

    /// `self + other`: exact sums stay exact and sums of upper bounds stay upper bounds.
    pub fn sum(self, other: Relation) -> Relation {
        use Relation::*;
        match (self, other) {
            (ModelChange, _) | (_, ModelChange) => ModelChange,
            (Equivalent, other) | (other, Equivalent) => other,
            (VariationalUpperBound, VariationalUpperBound) => VariationalUpperBound,
            (StochasticEstimate, StochasticEstimate) => StochasticEstimate,
            _ => Approximate,
        }
    }

    /// `-self`: negation turns an upper bound into an (unrepresented) lower bound.
    pub fn negated(self) -> Relation {
        match self {
            Relation::VariationalUpperBound => Relation::Approximate,
            other => other,
        }
    }

    /// `self - other` (as `chemflow.relations.combine_difference`).
    pub fn difference(self, other: Relation) -> Relation {
        self.sum(other.negated())
    }

    /// The weakest relation both values share.
    pub fn meet(self, other: Relation) -> Relation {
        use Relation::*;
        match (self, other) {
            (ModelChange, _) | (_, ModelChange) => ModelChange,
            (left, right) if left == right => left,
            (Equivalent, StochasticEstimate) | (StochasticEstimate, Equivalent) => {
                StochasticEstimate
            }
            _ => Approximate,
        }
    }
}

/// Is `name` a valid `relation` entry of a primitive?
pub fn valid_entry(name: &str) -> bool {
    RELATIONS.contains(&name) || RULES.contains(&name)
}

/// The relation of every node (`None` where a value has no stated question or depends on
/// a value without one). Nodes are in topological order, as `Graph::add` builds them.
pub fn infer(graph: &Graph) -> Vec<Option<Relation>> {
    let mut found: Vec<Option<Relation>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let input = |index: usize| -> Option<Relation> {
            node.inputs
                .get(index)
                .and_then(|id| found.get(*id).copied().flatten())
        };
        let all = || -> Option<Vec<Relation>> {
            node.inputs
                .iter()
                .map(|id| found.get(*id).copied().flatten())
                .collect()
        };
        let relation = primitive(&node.op).and_then(|spec| {
            let entry = spec.relation.as_deref()?;
            if let Some(fixed) = Relation::parse(entry) {
                return Some(fixed);
            }
            match entry {
                "input" => input(0),
                "sum" => all()?.into_iter().reduce(Relation::sum),
                "difference" => Some(input(0)?.difference(input(1)?)),
                "scale" => {
                    let factor = node.params.get("factor").and_then(|value| value.as_f64())?;
                    let source = input(0)?;
                    Some(if factor < 0.0 { source.negated() } else { source })
                }
                "decreasing" => Some(input(0)?.negated()),
                "meet" => {
                    // Branch conditions and other unrelated inputs carry no relation.
                    let known: Vec<Relation> = node
                        .inputs
                        .iter()
                        .filter_map(|id| found.get(*id).copied().flatten())
                        .collect();
                    known.into_iter().reduce(Relation::meet)
                }
                "param" => {
                    let key = spec.relation_param.as_deref()?;
                    let value = node.params.get(key)?.as_str()?;
                    spec.relations.get(value).and_then(|name| Relation::parse(name))
                }
                _ => None,
            }
        });
        found.push(relation);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::Relation::*;
    use super::*;

    #[test]
    fn composition_matches_the_chemflow_relation_table() {
        assert_eq!(Equivalent.difference(Equivalent), Equivalent);
        assert_eq!(VariationalUpperBound.difference(Equivalent), VariationalUpperBound);
        assert_eq!(Equivalent.difference(VariationalUpperBound), Approximate);
        assert_eq!(VariationalUpperBound.difference(VariationalUpperBound), Approximate);
        assert_eq!(StochasticEstimate.difference(Equivalent), StochasticEstimate);
        assert_eq!(StochasticEstimate.difference(VariationalUpperBound), Approximate);
        assert_eq!(ModelChange.difference(Equivalent), ModelChange);
        assert_eq!(VariationalUpperBound.sum(VariationalUpperBound), VariationalUpperBound);
        assert_eq!(Equivalent.meet(VariationalUpperBound), Approximate);
    }

    #[test]
    fn satisfies_orders_relations() {
        assert!(Equivalent.satisfies(VariationalUpperBound));
        assert!(VariationalUpperBound.satisfies(Approximate));
        assert!(!Approximate.satisfies(VariationalUpperBound));
        assert!(!ModelChange.satisfies(Approximate));
    }

    #[test]
    fn every_declared_relation_is_valid() {
        for (op, spec) in crate::primitives::primitives() {
            if let Some(entry) = &spec.relation {
                assert!(valid_entry(entry), "{op}: {entry}");
            }
        }
    }
}
