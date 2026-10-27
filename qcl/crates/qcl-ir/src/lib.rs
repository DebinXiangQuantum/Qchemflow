use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Host,
    Compute,
    Projection,
    Copy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    Molecule,
    Tensor,
    Eigensystem,
    SvdResult,
    SpectralEstimate,
    ScfResult,
    Mp2Result,
    ActiveSpace,
    FermionOperator,
    QubitOperator,
    Sector,
    FciResult,
    SciResult,
    GroundState,
    Estimate,
    Kernel,
    Bitstrings,
    Collection,
    Ranking,
    Eri,
    SiteScores,
    Frequencies,
    LucjParams,
    Integer,
    Boolean,
    Comparison,
    Hessian,
    ReactionPath,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceSpan {
    pub file: String,
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

impl SourceSpan {
    pub fn parse(source: &str) -> Option<Self> {
        let (head, end_column) = source.rsplit_once(':')?;
        let (head, end_line) = head.rsplit_once('-')?;
        let (head, start_column) = head.rsplit_once(':')?;
        let (file, start_line) = head.rsplit_once(':')?;
        Some(Self {
            file: file.into(),
            start_line: start_line.parse().ok()?,
            start_column: start_column.parse().ok()?,
            end_line: end_line.parse().ok()?,
            end_column: end_column.parse().ok()?,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Device {
    pub id: String,
    pub kind: String,
    pub billing: Option<String>,
    pub billing_account: Option<String>,
    pub threads: Option<u32>,
    pub memory_gb: Option<f64>,
    pub qubits: Option<u32>,
    pub site: Option<String>,
    pub simulator_host: Option<String>,
    pub simulator_method: Option<String>,
    pub provider: Option<String>,
    pub backend: Option<String>,
    #[serde(default)]
    pub quantum_links: Vec<[String; 2]>,
    #[serde(default)]
    pub options: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Node {
    pub id: usize,
    pub op: String,
    pub kind: NodeKind,
    pub value_type: ValueType,
    pub inputs: Vec<usize>,
    pub params: Value,
    pub device: Option<Device>,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_span: Option<SourceSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guards: Vec<Guard>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Guard {
    pub condition: usize,
    pub expected: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Graph {
    pub version: u32,
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            version: 1,
            nodes: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        op: &str,
        kind: NodeKind,
        value_type: ValueType,
        inputs: Vec<usize>,
        params: Value,
        source: String,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: op.to_owned(),
            kind,
            value_type,
            inputs,
            params,
            device: None,
            source_span: SourceSpan::parse(&source),
            source,
            guards: Vec::new(),
        });
        id
    }

    pub fn place(&mut self, id: usize, device: Device, source: String) -> Result<usize, String> {
        let node = self.nodes.get_mut(id).ok_or("Unknown value")?;
        if node.kind == NodeKind::Compute && node.device.is_none() {
            node.device = Some(device);
            return Ok(id);
        }
        let value_type = node.value_type.clone();
        let copy = self.add(
            "copy",
            NodeKind::Copy,
            value_type,
            vec![id],
            Value::Null,
            source,
        );
        self.nodes[copy].device = Some(device);
        Ok(copy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_placement_is_execution_and_later_placement_is_copy() {
        let mut graph = Graph::new();
        let op = graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![],
            Value::Null,
            "test:1".into(),
        );
        let first = graph
            .place(
                op,
                Device {
                    id: "cpu0".into(),
                    kind: "cpu".into(),
                    ..Device::default()
                },
                "test:2".into(),
            )
            .unwrap();
        let second = graph
            .place(
                op,
                Device {
                    id: "cpu1".into(),
                    kind: "cpu".into(),
                    ..Device::default()
                },
                "test:3".into(),
            )
            .unwrap();
        assert_eq!(first, op);
        assert_eq!(graph.nodes[second].kind, NodeKind::Copy);
        assert_eq!(graph.nodes[second].inputs, vec![op]);
    }

    #[test]
    fn parses_full_span_with_colons_in_file_path() {
        let span = SourceSpan::parse("C:/work/example.py:12:9-13:21").unwrap();
        assert_eq!(span.file, "C:/work/example.py");
        assert_eq!((span.start_line, span.start_column), (12, 9));
        assert_eq!((span.end_line, span.end_column), (13, 21));
    }
}
