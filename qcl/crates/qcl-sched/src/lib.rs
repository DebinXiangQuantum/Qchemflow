//! Dataflow optimization and scheduling for QCL plans.
//!
//! The planner keeps the program's placement: every node runs on the device the
//! program named. What it chooses is *when* and *how many at once*:
//!
//! 1. Dead-node elimination. Only nodes that reach an output (or a `require`
//!    check) run; an unused QPU job is never submitted.
//! 2. Common-subexpression elimination. A node equal to an earlier one (same
//!    operation, parameters, device, inputs and guards) becomes an alias of it,
//!    so a value copied twice to the same GPU moves once. QPU jobs are never
//!    merged: each requested job runs.
//! 3. Priorities. Each node's upward rank is its predicted cost plus the most
//!    expensive path to an output; ready nodes run highest rank first, so the
//!    critical path is never kept waiting behind work that has slack.
//! 4. Capacity. Each device lane runs up to `slots` nodes at once.
//!
//! The same list-scheduling rule is simulated on the predicted costs to give
//! the predicted makespan reported by `Plan.schedule()`.

use qcl_ir::{Graph, Node};
use serde::Serialize;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// Operations whose duplicates are not merged: each one is a separate QPU job.
pub const UNMERGEABLE: &[&str] = &["sample", "vqe.sampled", "estimate.quantum"];
/// Operations that run even when no output reads them.
pub const ALWAYS_LIVE: &[&str] = &["require"];

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Prediction {
    /// Sum of all executed node costs: the time of a one-at-a-time run.
    pub sequential_s: f64,
    /// Predicted time of the schedule on the given lanes and slots.
    pub makespan_s: f64,
    /// Length of the most expensive dependency chain: no schedule beats it.
    pub critical_path_s: f64,
    pub critical_path: Vec<usize>,
    pub start_s: BTreeMap<usize, f64>,
    pub lane_busy_s: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Schedule {
    /// Nodes to execute, in program order.
    pub execute: Vec<usize>,
    /// Nodes no output depends on.
    pub eliminated: Vec<usize>,
    /// Duplicate node -> the earlier node whose value it reuses.
    pub aliases: BTreeMap<usize, usize>,
    /// Upward rank in predicted seconds; higher runs first.
    pub priority: BTreeMap<usize, f64>,
    /// Device lane of every executed node ("host" for unplaced nodes).
    pub lane: BTreeMap<usize, String>,
    /// Concurrent slots per lane.
    pub slots: BTreeMap<String, usize>,
    pub predicted: Prediction,
}

pub fn lane(node: &Node) -> String {
    node.device
        .as_ref()
        .map_or_else(|| "host".to_string(), |device| device.id.clone())
}

fn dependencies(node: &Node) -> impl Iterator<Item = usize> + '_ {
    node.inputs
        .iter()
        .copied()
        .chain(node.guards.iter().map(|guard| guard.condition))
}

/// Nodes that an output or an always-live check depends on.
pub fn live_nodes(graph: &Graph, outputs: &[usize]) -> Vec<bool> {
    let mut live = vec![false; graph.nodes.len()];
    let mut stack: Vec<usize> = outputs
        .iter()
        .copied()
        .chain(
            graph
                .nodes
                .iter()
                .filter(|node| ALWAYS_LIVE.contains(&node.op.as_str()))
                .map(|node| node.id),
        )
        .collect();
    while let Some(id) = stack.pop() {
        if id >= live.len() || live[id] {
            continue;
        }
        live[id] = true;
        stack.extend(dependencies(&graph.nodes[id]));
    }
    live
}

/// Map each duplicate live node to the first equal node. Inputs are compared
/// after their own duplicates are resolved, so equal chains collapse entirely.
pub fn common_nodes(graph: &Graph, live: &[bool]) -> BTreeMap<usize, usize> {
    let mut canonical: Vec<usize> = (0..graph.nodes.len()).collect();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut aliases = BTreeMap::new();
    for node in &graph.nodes {
        if !live[node.id] || UNMERGEABLE.contains(&node.op.as_str()) {
            continue;
        }
        let key = json!({
            "op": node.op,
            "kind": node.kind,
            "params": node.params,
            "device": node.device,
            "inputs": node.inputs.iter().map(|id| canonical[*id]).collect::<Vec<_>>(),
            "guards": node.guards.iter()
                .map(|guard| (canonical[guard.condition], guard.expected))
                .collect::<Vec<_>>(),
        })
        .to_string();
        match seen.get(&key) {
            Some(&first) => {
                canonical[node.id] = first;
                aliases.insert(node.id, first);
            }
            None => {
                seen.insert(key, node.id);
            }
        }
    }
    aliases
}

#[derive(PartialEq)]
struct Ready {
    priority: f64,
    id: usize,
}

impl Eq for Ready {}

impl Ord for Ready {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Ready {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Optimize and schedule `graph`. `costs` gives predicted seconds per node
/// (missing nodes cost zero); `slots` gives the concurrency of each lane
/// (missing lanes get one slot).
pub fn schedule(
    graph: &Graph,
    outputs: &[usize],
    costs: &HashMap<usize, f64>,
    slots: &HashMap<String, usize>,
) -> Schedule {
    let live = live_nodes(graph, outputs);
    let aliases = common_nodes(graph, &live);
    let resolve = |id: usize| *aliases.get(&id).unwrap_or(&id);
    let execute: Vec<usize> = graph
        .nodes
        .iter()
        .filter(|node| live[node.id] && !aliases.contains_key(&node.id))
        .map(|node| node.id)
        .collect();
    let eliminated = graph
        .nodes
        .iter()
        .filter(|node| !live[node.id])
        .map(|node| node.id)
        .collect();
    let cost = |id: usize| costs.get(&id).copied().unwrap_or(0.0).max(0.0);

    let mut needs: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for &id in &execute {
        let mut inputs: Vec<usize> = dependencies(&graph.nodes[id]).map(resolve).collect();
        inputs.sort_unstable();
        inputs.dedup();
        for &input in &inputs {
            successors.entry(input).or_default().push(id);
        }
        needs.insert(id, inputs);
    }
    // Node ids are a topological order (inputs always precede their users).
    let mut priority: BTreeMap<usize, f64> = BTreeMap::new();
    for &id in execute.iter().rev() {
        let downstream = successors
            .get(&id)
            .into_iter()
            .flatten()
            .map(|next| priority[next])
            .fold(0.0, f64::max);
        priority.insert(id, cost(id) + downstream);
    }
    let lanes: BTreeMap<usize, String> = execute
        .iter()
        .map(|&id| (id, lane(&graph.nodes[id])))
        .collect();
    let mut lane_slots: BTreeMap<String, usize> = BTreeMap::new();
    for name in lanes.values() {
        lane_slots.insert(name.clone(), slots.get(name).copied().unwrap_or(1).max(1));
    }

    let mut critical_path = Vec::new();
    let mut current = execute
        .iter()
        .copied()
        .filter(|id| needs[id].is_empty())
        .max_by(|a, b| priority[a].total_cmp(&priority[b]).then_with(|| b.cmp(a)));
    while let Some(id) = current {
        critical_path.push(id);
        current = successors
            .get(&id)
            .into_iter()
            .flatten()
            .copied()
            .max_by(|a, b| priority[a].total_cmp(&priority[b]).then_with(|| b.cmp(a)));
    }
    let critical_path_s = critical_path.first().map_or(0.0, |id| priority[id]);
    let sequential_s = execute.iter().map(|&id| cost(id)).sum();

    // Event-driven simulation of the runtime's dispatch rule.
    let mut remaining: HashMap<usize, usize> =
        execute.iter().map(|&id| (id, needs[&id].len())).collect();
    let mut ready: BTreeMap<String, BinaryHeap<Ready>> = BTreeMap::new();
    for &id in &execute {
        if needs[&id].is_empty() {
            ready.entry(lanes[&id].clone()).or_default().push(Ready {
                priority: priority[&id],
                id,
            });
        }
    }
    let mut busy: BTreeMap<String, usize> = BTreeMap::new();
    let mut lane_busy_s: BTreeMap<String, f64> =
        lane_slots.keys().map(|name| (name.clone(), 0.0)).collect();
    let mut running: Vec<(f64, usize)> = Vec::new();
    let mut start_s = BTreeMap::new();
    let mut now = 0.0f64;
    let mut makespan_s = 0.0f64;
    loop {
        for (name, heap) in ready.iter_mut() {
            while busy.get(name).copied().unwrap_or(0) < lane_slots[name] {
                let Some(next) = heap.pop() else { break };
                *busy.entry(name.clone()).or_default() += 1;
                *lane_busy_s.get_mut(name).expect("lane") += cost(next.id);
                start_s.insert(next.id, now);
                running.push((now + cost(next.id), next.id));
            }
        }
        if running.is_empty() {
            break;
        }
        let index = (0..running.len())
            .min_by(|&a, &b| {
                running[a]
                    .0
                    .total_cmp(&running[b].0)
                    .then_with(|| running[a].1.cmp(&running[b].1))
            })
            .expect("running");
        let (finish, id) = running.swap_remove(index);
        now = finish;
        makespan_s = makespan_s.max(finish);
        *busy.get_mut(&lanes[&id]).expect("lane") -= 1;
        for &next in successors.get(&id).into_iter().flatten() {
            let count = remaining.get_mut(&next).expect("node");
            *count -= 1;
            if *count == 0 {
                ready.entry(lanes[&next].clone()).or_default().push(Ready {
                    priority: priority[&next],
                    id: next,
                });
            }
        }
    }
    Schedule {
        execute,
        eliminated,
        aliases,
        priority,
        lane: lanes,
        slots: lane_slots,
        predicted: Prediction {
            sequential_s,
            makespan_s,
            critical_path_s,
            critical_path,
            start_s,
            lane_busy_s,
        },
    }
}

/// JSON entry point: `outputs` is a list of node ids, `costs` maps node ids
/// (as strings) to seconds, `slots` maps lane names to slot counts.
pub fn schedule_json(
    graph: &str,
    outputs: &str,
    costs: &str,
    slots: &str,
) -> Result<String, String> {
    let graph: Graph = serde_json::from_str(graph).map_err(|error| error.to_string())?;
    let outputs: Vec<usize> = serde_json::from_str(outputs).map_err(|error| error.to_string())?;
    let costs: HashMap<String, f64> =
        serde_json::from_str(costs).map_err(|error| error.to_string())?;
    let slots: HashMap<String, usize> =
        serde_json::from_str(slots).map_err(|error| error.to_string())?;
    if let Some(bad) = outputs.iter().find(|&&id| id >= graph.nodes.len()) {
        return Err(format!("Output node {bad} is not in the graph"));
    }
    let costs = costs
        .into_iter()
        .map(|(id, seconds)| id.parse::<usize>().map(|id| (id, seconds)))
        .collect::<Result<HashMap<_, _>, _>>()
        .map_err(|error| error.to_string())?;
    let value: Value = serde_json::to_value(schedule(&graph, &outputs, &costs, &slots))
        .map_err(|error| error.to_string())?;
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcl_ir::{Device, Guard, NodeKind, ValueType};

    fn device(id: &str) -> Option<Device> {
        serde_json::from_value(json!({"id": id, "kind": "cpu"})).ok()
    }

    fn node(id: usize, op: &str, inputs: &[usize], lane: Option<&str>) -> Node {
        Node {
            id,
            op: op.to_string(),
            kind: if lane.is_some() {
                NodeKind::Compute
            } else {
                NodeKind::Host
            },
            value_type: ValueType::Estimate,
            inputs: inputs.to_vec(),
            params: json!({}),
            device: lane.and_then(device),
            source: String::new(),
            source_span: None,
            guards: Vec::new(),
        }
    }

    fn graph(nodes: Vec<Node>) -> Graph {
        Graph { version: 1, nodes }
    }

    #[test]
    fn dead_nodes_are_not_executed_but_checks_are() {
        let mut check = node(3, "require", &[2], None);
        check.guards.push(Guard {
            condition: 2,
            expected: true,
        });
        let g = graph(vec![
            node(0, "molecule", &[], None),
            node(1, "scf.rhf", &[0], Some("cpu0")),
            node(2, "scf.rhf", &[0], Some("gpu0")),
            check,
            node(4, "sample", &[0], Some("qpu0")),
        ]);
        let plan = schedule(&g, &[1], &HashMap::new(), &HashMap::new());
        assert_eq!(plan.execute, vec![0, 1, 2, 3]);
        assert_eq!(plan.eliminated, vec![4]);
    }

    #[test]
    fn equal_chains_collapse_but_qpu_jobs_do_not() {
        let mut first = node(1, "copy", &[0], Some("gpu0"));
        first.params = json!({"to": "gpu0"});
        let mut second = first.clone();
        second.id = 2;
        let g = graph(vec![
            node(0, "scf.rhf", &[], Some("cpu0")),
            first,
            second,
            node(3, "fci.solve", &[1], Some("gpu0")),
            node(4, "fci.solve", &[2], Some("gpu0")),
            node(5, "sample", &[0], Some("qpu0")),
            node(6, "sample", &[0], Some("qpu0")),
        ]);
        let plan = schedule(&g, &[3, 4, 5, 6], &HashMap::new(), &HashMap::new());
        assert_eq!(plan.aliases, BTreeMap::from([(2, 1), (4, 3)]));
        assert_eq!(plan.execute, vec![0, 1, 3, 5, 6]);
    }

    #[test]
    fn critical_path_runs_first_and_lanes_overlap() {
        // 0 -> 1 (long) -> 3 ; 0 -> 2 (short, same lane as 1)
        let g = graph(vec![
            node(0, "scf.rhf", &[], Some("gpu0")),
            node(1, "ccsd", &[0], Some("cpu0")),
            node(2, "sqd", &[0], Some("cpu0")),
            node(3, "fci", &[1], Some("gpu0")),
        ]);
        let costs = HashMap::from([(0, 1.0), (1, 5.0), (2, 2.0), (3, 4.0)]);
        let plan = schedule(&g, &[2, 3], &costs, &HashMap::new());
        assert_eq!(plan.predicted.critical_path, vec![0, 1, 3]);
        assert_eq!(plan.predicted.critical_path_s, 10.0);
        assert!(plan.priority[&1] > plan.priority[&2]);
        assert_eq!(plan.predicted.start_s[&1], 1.0);
        assert_eq!(plan.predicted.start_s[&2], 6.0);
        assert_eq!(plan.predicted.makespan_s, 10.0);
        assert_eq!(plan.predicted.sequential_s, 12.0);
        let wide = schedule(&g, &[2, 3], &costs, &HashMap::from([("cpu0".into(), 2)]));
        assert_eq!(wide.predicted.start_s[&2], 1.0);
        assert_eq!(wide.predicted.makespan_s, 10.0);
    }

    #[test]
    fn slots_shorten_a_wide_lane() {
        let mut nodes = vec![node(0, "scf.rhf", &[], Some("gpu0"))];
        for id in 1..=4 {
            let mut sqd = node(id, "sqd", &[0], Some("cpu0"));
            sqd.params = json!({"seed": id});
            nodes.push(sqd);
        }
        let g = graph(nodes);
        let costs: HashMap<usize, f64> = (1..=4).map(|id| (id, 3.0)).collect();
        let one = schedule(&g, &[1, 2, 3, 4], &costs, &HashMap::new());
        let four = schedule(
            &g,
            &[1, 2, 3, 4],
            &costs,
            &HashMap::from([("cpu0".into(), 4)]),
        );
        assert_eq!(one.predicted.makespan_s, 12.0);
        assert_eq!(four.predicted.makespan_s, 3.0);
        assert_eq!(four.predicted.lane_busy_s["cpu0"], 12.0);
    }

    #[test]
    fn json_entry_point_round_trips_and_rejects_unknown_outputs() {
        let g = graph(vec![node(0, "scf.rhf", &[], Some("cpu0"))]);
        let text = serde_json::to_string(&g).unwrap();
        let result = schedule_json(&text, "[0]", r#"{"0": 2.5}"#, "{}").unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["predicted"]["makespan_s"], 2.5);
        assert!(schedule_json(&text, "[3]", "{}", "{}").is_err());
    }
}
