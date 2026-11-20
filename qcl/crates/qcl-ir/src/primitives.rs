//! The primitive table: one declaration per IR operation (see `primitives.toml`).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Graph, NodeKind, ValueType};

/// Source of the table, embedded at compile time.
pub const PRIMITIVES_TOML: &str = include_str!("../primitives.toml");

/// Device kinds an implementation can be registered for.
pub const DEVICE_KINDS: [&str; 3] = ["cpu", "gpu", "qpu"];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub kind: NodeKind,
    /// A `ValueType` name, or `"input"` when the op returns one of its inputs' types.
    pub output: String,
    /// Which input supplies an `"input"` output type.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub output_from: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// How the op's value relates to the exact answer of its question: a fixed relation
    /// or a rule over its inputs (see `relations.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
    /// For `relation = "param"`: the parameter whose value selects from `relations`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation_param: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relations: BTreeMap<String, String>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

impl Primitive {
    /// The declared output type; `None` means "an input's type" (see `output_from`).
    pub fn output_type(&self) -> Option<ValueType> {
        if self.output == "input" {
            None
        } else {
            serde_json::from_value(Value::String(self.output.clone())).ok()
        }
    }

    /// The type produced for a node whose inputs have `inputs` types, in order.
    pub fn output_for(&self, inputs: &[ValueType]) -> Option<ValueType> {
        self.output_type()
            .or_else(|| inputs.get(self.output_from).cloned())
    }

    pub fn supports(&self, device_kind: &str) -> bool {
        self.devices.iter().any(|kind| kind == device_kind)
    }

    /// QCL-E203 message and hint for placing this op on an unsupported device kind.
    pub fn unavailable(&self, op: &str, device_kind: &str) -> (String, String) {
        let message = self
            .message
            .clone()
            .unwrap_or_else(|| format!("{op} has no {device_kind} implementation"));
        let hint = self
            .hint
            .clone()
            .unwrap_or_else(|| format!("Available device kinds: {}", self.devices.join(", ")));
        (message, hint)
    }
}

/// Parse and validate a primitive table.
pub fn parse(source: &str) -> Result<BTreeMap<String, Primitive>, String> {
    let table: BTreeMap<String, Primitive> =
        toml::from_str(source).map_err(|error| error.to_string())?;
    for (op, primitive) in &table {
        if primitive.output != "input" && primitive.output_type().is_none() {
            return Err(format!("{op}: unknown output type {:?}", primitive.output));
        }
        if primitive.output != "input" && primitive.output_from != 0 {
            return Err(format!(
                "{op}: output_from applies only to output = \"input\""
            ));
        }
        if let Some(entry) = &primitive.relation {
            if !crate::relations::valid_entry(entry) {
                return Err(format!("{op}: unknown relation {entry:?}"));
            }
            let by_param = entry == "param";
            if by_param != primitive.relation_param.is_some()
                || by_param == primitive.relations.is_empty()
            {
                return Err(format!(
                    "{op}: relation = \"param\" needs relation_param and relations"
                ));
            }
        } else if primitive.relation_param.is_some() || !primitive.relations.is_empty() {
            return Err(format!(
                "{op}: relation_param and relations need relation = \"param\""
            ));
        }
        if let Some(name) = primitive
            .relations
            .values()
            .find(|name| !crate::relations::RELATIONS.contains(&name.as_str()))
        {
            return Err(format!("{op}: unknown relation {name:?}"));
        }
        if let Some(kind) = primitive
            .devices
            .iter()
            .find(|kind| !DEVICE_KINDS.contains(&kind.as_str()))
        {
            return Err(format!("{op}: unknown device kind {kind:?}"));
        }
        match primitive.kind {
            NodeKind::Compute if primitive.devices.is_empty() => {
                return Err(format!("{op}: a compute op needs at least one device kind"));
            }
            NodeKind::Host | NodeKind::Projection | NodeKind::Copy
                if !primitive.devices.is_empty() =>
            {
                return Err(format!("{op}: only compute ops declare devices"));
            }
            _ => {}
        }
    }
    Ok(table)
}

/// The embedded primitive table. Panics only if `primitives.toml` is malformed, which
/// the crate's tests rule out.
pub fn primitives() -> &'static BTreeMap<String, Primitive> {
    static TABLE: OnceLock<BTreeMap<String, Primitive>> = OnceLock::new();
    TABLE.get_or_init(|| parse(PRIMITIVES_TOML).expect("valid primitives.toml"))
}

pub fn primitive(op: &str) -> Option<&'static Primitive> {
    primitives().get(op)
}

/// Nodes whose op, kind or output type disagree with the table (empty when consistent).
pub fn mismatches(graph: &Graph) -> Vec<String> {
    let mut found = Vec::new();
    for node in &graph.nodes {
        let Some(spec) = primitive(&node.op) else {
            found.push(format!(
                "node {}: {} is not in primitives.toml",
                node.id, node.op
            ));
            continue;
        };
        if spec.kind != node.kind {
            found.push(format!(
                "node {}: {} is {:?}, table says {:?}",
                node.id, node.op, node.kind, spec.kind
            ));
        }
        let input_types: Vec<ValueType> = node
            .inputs
            .iter()
            .filter_map(|id| graph.nodes.get(*id))
            .map(|input| input.value_type.clone())
            .collect();
        let expected = spec.output_for(&input_types);
        if expected
            .as_ref()
            .is_some_and(|value_type| *value_type != node.value_type)
        {
            found.push(format!(
                "node {}: {} returns {:?}, table says {:?}",
                node.id, node.op, node.value_type, expected
            ));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table_is_valid() {
        let table = parse(PRIMITIVES_TOML).unwrap();
        assert!(table.len() > 100);
        assert_eq!(table["scf.rhf"].output_type(), Some(ValueType::ScfResult));
        assert_eq!(table["copy"].output_type(), None);
        assert_eq!(
            table["cond"].output_for(&[ValueType::Boolean, ValueType::Estimate]),
            Some(ValueType::Estimate)
        );
        assert!(table["sample"].supports("qpu") && !table["sample"].supports("cpu"));
    }

    #[test]
    fn rejects_unknown_types_devices_and_misplaced_devices() {
        let bad_output = "[x]\nkind = \"host\"\noutput = \"nonsense\"\n";
        assert!(
            parse(bad_output)
                .unwrap_err()
                .contains("unknown output type")
        );
        let bad_device = "[x]\nkind = \"compute\"\noutput = \"tensor\"\ndevices = [\"tpu\"]\n";
        assert!(
            parse(bad_device)
                .unwrap_err()
                .contains("unknown device kind")
        );
        let host_device = "[x]\nkind = \"host\"\noutput = \"tensor\"\ndevices = [\"cpu\"]\n";
        assert!(parse(host_device).unwrap_err().contains("only compute ops"));
        let no_device = "[x]\nkind = \"compute\"\noutput = \"tensor\"\n";
        assert!(
            parse(no_device)
                .unwrap_err()
                .contains("at least one device")
        );
        let bad_from = "[x]\nkind = \"host\"\noutput = \"tensor\"\noutput_from = 1\n";
        assert!(parse(bad_from).unwrap_err().contains("output_from"));
    }

    #[test]
    fn default_and_custom_unavailable_wording() {
        let table = primitives();
        assert_eq!(
            table["eigh"].unavailable("eigh", "qpu"),
            (
                "eigh has no qpu implementation".to_string(),
                "Available device kinds: cpu, gpu".to_string()
            )
        );
        assert_eq!(
            table["sample"].unavailable("sample", "cpu").0,
            "sample requires a QPU or QPU simulator"
        );
    }

    #[test]
    fn graph_nodes_are_checked_against_the_table() {
        let mut graph = Graph::new();
        let molecule = graph.add(
            "molecule",
            NodeKind::Host,
            ValueType::Molecule,
            vec![],
            Value::Null,
            "t:1".into(),
        );
        graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            Value::Null,
            "t:2".into(),
        );
        assert!(mismatches(&graph).is_empty());
        graph.add(
            "scf.rhf",
            NodeKind::Host,
            ValueType::Tensor,
            vec![molecule],
            Value::Null,
            "t:3".into(),
        );
        graph.add(
            "no.such.op",
            NodeKind::Host,
            ValueType::Tensor,
            vec![],
            Value::Null,
            "t:4".into(),
        );
        let found = mismatches(&graph);
        assert_eq!(found.len(), 3, "{found:?}");
    }
}
