use qcl_ir::{Graph, Node, NodeKind, ValueType};
use serde::Serialize;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub source: String,
    pub message: String,
    pub hint: String,
}

fn combinations(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    (1..=k.min(n - k)).fold(1.0, |value, index| {
        value * (n + 1 - index) as f64 / index as f64
    })
}

fn original_node(graph: &Graph, id: usize) -> Option<&Node> {
    let mut node = graph.nodes.get(id)?;
    while node.kind == NodeKind::Copy {
        node = graph.nodes.get(*node.inputs.first()?)?;
    }
    Some(node)
}

fn molecule_spin(graph: &Graph, id: usize) -> Option<i64> {
    let node = graph.nodes.get(id)?;
    if node.value_type != ValueType::Molecule {
        return None;
    }
    node.params
        .get("spin")
        .and_then(|value| value.as_i64())
        .or_else(|| {
            node.inputs
                .first()
                .and_then(|input| molecule_spin(graph, *input))
        })
}

fn active_orbitals(graph: &Graph, id: usize) -> Option<u64> {
    let node = original_node(graph, id)?;
    if node.op == "active_space" {
        return node.params.get("orbitals")?.as_u64();
    }
    node.inputs
        .first()
        .and_then(|input| active_orbitals(graph, *input))
}

fn kernel_qubits(graph: &Graph, id: usize) -> Option<u64> {
    let node = original_node(graph, id)?;
    if node.op == "kernel.custom" {
        return node.params.get("n_qubits")?.as_u64();
    }
    active_orbitals(graph, id)?.checked_mul(2)
}

pub fn check(graph: &Graph) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for node in &graph.nodes {
        if matches!(node.op.as_str(), "scf.rhf" | "dft.rks") {
            let spin = node.inputs.first().and_then(|id| molecule_spin(graph, *id));
            if spin != Some(0) {
                diagnostics.push(Diagnostic {
                    code: "QCL-E101",
                    source: node.source.clone(),
                    message: format!("{} requires a closed-shell molecule", node.op),
                    hint: "Use a zero-spin molecule or an open-shell SCF operation".into(),
                });
            }
        }
        if matches!(node.op.as_str(), "scf.rohf" | "scf.uhf" | "dft.uks")
            && node.inputs.first().and_then(|id| molecule_spin(graph, *id)) == Some(0)
        {
            diagnostics.push(Diagnostic {
                code: "QCL-E101",
                source: node.source.clone(),
                message: format!("{} requires an open-shell molecule", node.op),
                hint: "Use scf.rhf or scf.hf for a zero-spin molecule".into(),
            });
        }
        if node.kind == NodeKind::Compute && node.device.is_none() {
            diagnostics.push(Diagnostic {
                code: "QCL-E201",
                source: node.source.clone(),
                message: format!("{} is never placed", node.op),
                hint: "Add .to(<device>) to the compute operation".into(),
            });
        }
        if let Some(device) = &node.device {
            let scf_operation = matches!(
                node.op.as_str(),
                "scf.rhf" | "scf.hf" | "scf.rohf" | "scf.uhf" | "dft.rks" | "dft.uks"
            );
            let gpu_scf_supported = device.kind == "gpu"
                && (matches!(
                    node.op.as_str(),
                    "scf.hf" | "scf.rohf" | "scf.uhf" | "dft.uks"
                ) || node.inputs.first().and_then(|id| molecule_spin(graph, *id)) == Some(0))
                && node
                    .params
                    .get("reuse_eri")
                    .and_then(|value| value.as_bool())
                    != Some(true);
            let local_gpu = device.id.strip_prefix("gpu").is_some_and(|ordinal| {
                !ordinal.is_empty() && ordinal.bytes().all(|b| b.is_ascii_digit())
            });
            if matches!(node.kind, NodeKind::Compute | NodeKind::Copy)
                && ((device.kind == "cpu" && device.id != "cpu0")
                    || (device.kind == "gpu" && !local_gpu))
            {
                diagnostics.push(Diagnostic {
                    code: "QCL-E203",
                    source: node.source.clone(),
                    message: format!("No local worker is registered for {}", device.id),
                    hint: "Use cpu0 or a local gpuN ordinal".into(),
                });
            }
            if (matches!(
                node.op.as_str(),
                "active_space"
                    | "qubit_hamiltonian"
                    | "sqd.diagonalize"
                    | "int2e"
                    | "int1e"
                    | "int3c"
                    | "cholesky"
                    | "periodic.scf"
                    | "dynamics.spectral_exact"
                    | "solubility.esol"
                    | "solubility.gse"
                    | "superconductivity.allen_dynes"
                    | "metabolism.sites"
                    | "binding.mmgbsa"
                    | "geometry.optimize"
                    | "sella.saddle"
                    | "geometry.constrained_optimize"
                    | "geometry.fd_optimize"
                    | "irc"
                    | "frequencies"
                    | "frequencies.from_hessian"
                    | "thermo.rrho"
                    | "thermo.eyring"
                    | "ccsd.amplitudes"
                    | "sci.solve"
                    | "gfn2.energy"
                    | "gfn2.optimize"
                    | "rdkit.embed"
                    | "rdkit.mmff_optimize"
                    | "rdkit.dedupe"
            ) && device.kind != "cpu")
                || (matches!(
                    node.op.as_str(),
                    "contract"
                        | "eigh"
                        | "orthogonalize"
                        | "solve"
                        | "svd"
                        | "transform"
                        | "gradient"
                        | "hessian"
                ) && !matches!(device.kind.as_str(), "cpu" | "gpu"))
                || (scf_operation && device.kind != "cpu" && !gpu_scf_supported)
                || (matches!(
                    node.op.as_str(),
                    "fci.solve" | "ed.ground_state" | "expect" | "mp2" | "rdm1" | "rdm2" | "sigma"
                ) && !matches!(device.kind.as_str(), "cpu" | "gpu"))
                || (node.op == "fci.solve"
                    && node.params.get("engine").and_then(|value| value.as_str()) == Some("native")
                    && device.kind != "cpu")
                || (node.op == "sigma"
                    && node.params.get("engine").and_then(|value| value.as_str()) == Some("native")
                    && device.kind != "cpu")
            {
                diagnostics.push(Diagnostic {
                    code: "QCL-E203",
                    source: node.source.clone(),
                    message: format!("{} has no {} implementation", node.op, device.kind),
                    hint: if matches!(
                        node.op.as_str(),
                        "fci.solve"
                            | "ed.ground_state"
                            | "expect"
                            | "mp2"
                            | "rdm1"
                            | "rdm2"
                            | "sigma"
                            | "contract"
                            | "eigh"
                            | "orthogonalize"
                            | "solve"
                            | "svd"
                            | "transform"
                            | "gradient"
                            | "hessian"
                    ) {
                        if matches!(node.op.as_str(), "fci.solve" | "sigma")
                            && node.params.get("engine").and_then(|value| value.as_str())
                                == Some("native")
                        {
                            "Native CI is available on CPU only"
                        } else {
                            "Available device kinds: cpu, gpu"
                        }
                    } else if scf_operation {
                        "HF/KS without an ERI override supports cpu and gpu"
                    } else {
                        "Available device kinds: cpu"
                    }
                    .into(),
                });
            }
            if node.kind == NodeKind::Copy && device.kind == "qpu" {
                diagnostics.push(Diagnostic {
                    code: "QCL-E203",
                    source: node.source.clone(),
                    message: "A QPU has no addressable value memory".into(),
                    hint: "Place a QPU-capable operation instead of copying a value".into(),
                });
            }
            if matches!(node.op.as_str(), "rdm1" | "rdm2") && device.kind == "gpu" {
                let mut input = node.inputs.first().and_then(|id| graph.nodes.get(*id));
                while input.is_some_and(|source| source.op == "copy") {
                    input = input
                        .and_then(|source| source.inputs.first())
                        .and_then(|id| graph.nodes.get(*id));
                }
                if !input.is_some_and(|source| {
                    source.op == "fci.solve"
                        && source
                            .device
                            .as_ref()
                            .is_some_and(|placed| placed.id == device.id)
                }) {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E203",
                        source: node.source.clone(),
                        message: format!("{} needs an FCI state on the same GPU", node.op),
                        hint: "Place FCI and the density operation on the same GPU".into(),
                    });
                }
            }
            if node.kind == NodeKind::Copy
                && device.kind == "gpu"
                && !matches!(
                    node.value_type,
                    ValueType::FermionOperator
                        | ValueType::Sector
                        | ValueType::GroundState
                        | ValueType::Tensor
                        | ValueType::Hessian
                        | ValueType::Eri
                        | ValueType::ScfResult
                )
            {
                diagnostics.push(Diagnostic {
                    code: "QCL-E203",
                    source: node.source.clone(),
                    message: "No GPU value-copy implementation is registered for this type".into(),
                    hint: "Only FermionOperator, Sector, GroundState, Tensor, Hessian, ERI and SCF result copies are supported"
                        .into(),
                });
            }
            if node.op == "ed.ground_state" && device.kind == "gpu" {
                let sites = node
                    .inputs
                    .first()
                    .and_then(|id| graph.nodes.get(*id))
                    .and_then(|operator| operator.params.get("shape"))
                    .and_then(|shape| shape.as_array())
                    .and_then(|shape| {
                        shape
                            .iter()
                            .try_fold(1_u64, |size, item| size.checked_mul(item.as_u64()?))
                    });
                let sector = node.inputs.get(1).and_then(|id| graph.nodes.get(*id));
                let alpha = sector
                    .and_then(|item| item.params.get("n_alpha"))
                    .and_then(|value| value.as_u64());
                let beta = sector
                    .and_then(|item| item.params.get("n_beta"))
                    .and_then(|value| value.as_u64());
                if let (Some(sites), Some(alpha), Some(beta), Some(memory_gb)) =
                    (sites, alpha, beta, device.memory_gb)
                {
                    let dimension = combinations(sites, alpha) * combinations(sites, beta);
                    let minimum_bytes = 8.0 * ((sites as f64).powi(4) + 4.0 * dimension);
                    if minimum_bytes > memory_gb * 1e9 {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E209",
                            source: node.source.clone(),
                            message: format!(
                                "ed.ground_state needs at least {minimum_bytes:.0} GPU bytes"
                            ),
                            hint: "Choose a larger GPU or a smaller lattice sector".into(),
                        });
                    }
                }
            }
            if node.op == "sample" && device.kind != "qpu" {
                diagnostics.push(Diagnostic {
                    code: "QCL-E203",
                    source: node.source.clone(),
                    message: "sample requires a QPU or QPU simulator".into(),
                    hint: "Available device kinds: qpu".into(),
                });
            }
            if node.op == "estimate.quantum" {
                if device.kind != "qpu" {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E203",
                        source: node.source.clone(),
                        message: "Pauli estimate requires a QPU or QPU simulator".into(),
                        hint: "Place the estimate on a QPU device".into(),
                    });
                } else {
                    let kernel_node = node.inputs.get(1).and_then(|id| original_node(graph, *id));
                    let custom_unitary = kernel_node.is_some_and(|kernel| {
                        kernel.op == "kernel.custom"
                            && kernel
                                .params
                                .get("n_clbits")
                                .and_then(|value| value.as_u64())
                                == Some(0)
                    });
                    let supported_simulator = device.simulator_host.as_deref() == Some("cpu0")
                        && device.simulator_method.as_deref() == Some("statevector")
                        && !matches!(device.site.as_deref(), Some(site) if site != "local")
                        && device.billing.as_deref().unwrap_or("none") == "none";
                    let supported_gpu_simulator = custom_unitary
                        && device.simulator_host.as_deref().is_some_and(|host| {
                            host.strip_prefix("gpu").is_some_and(|ordinal| {
                                !ordinal.is_empty()
                                    && ordinal.bytes().all(|byte| byte.is_ascii_digit())
                            })
                        })
                        && device.simulator_method.as_deref() == Some("statevector")
                        && !matches!(device.site.as_deref(), Some(site) if site != "local")
                        && device.billing.as_deref().unwrap_or("none") == "none";
                    let supported_cloud = device.provider.as_deref() == Some("lqcloud")
                        && device.backend.as_deref() == Some("MQ02")
                        && device.billing.as_deref() == Some("metered")
                        && device.site.as_deref() == Some("remote");
                    if !supported_simulator && !supported_gpu_simulator && !supported_cloud {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E203",
                            source: node.source.clone(),
                            message: format!(
                                "Pauli estimate has no implementation for {}",
                                device.id
                            ),
                            hint: "Use a compatible QPU simulator or metered lqcloud MQ02".into(),
                        });
                    }
                    let operator_qubits = node
                        .inputs
                        .first()
                        .and_then(|id| active_orbitals(graph, *id))
                        .and_then(|orbitals| orbitals.checked_mul(2));
                    let kernel_qubits = node.inputs.get(1).and_then(|id| kernel_qubits(graph, *id));
                    if let (Some(left), Some(right)) = (operator_qubits, kernel_qubits)
                        && left != right
                    {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E104",
                            source: node.source.clone(),
                            message: format!("Operator needs {left} qubits but kernel has {right}"),
                            hint: "Use a kernel with the operator's qubit count".into(),
                        });
                    }
                    if let (Some(required), Some(capacity)) = (operator_qubits, device.qubits)
                        && required > u64::from(capacity)
                    {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E209",
                            source: node.source.clone(),
                            message: format!(
                                "Pauli estimate needs {required} qubits; {} has {capacity}",
                                device.id
                            ),
                            hint: "Choose a larger QPU or a smaller active space".into(),
                        });
                    }
                    if device.qubits.is_none() {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E209",
                            source: node.source.clone(),
                            message: format!("{} has no declared qubit capacity", device.id),
                            hint: "Declare the QPU's physical or logical qubit capacity".into(),
                        });
                    }
                    let dynamic_kernel = node
                        .inputs
                        .get(1)
                        .and_then(|id| original_node(graph, *id))
                        .is_some_and(|kernel| {
                            kernel.op == "kernel.custom"
                                && kernel
                                    .params
                                    .get("n_clbits")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    > 0
                        });
                    if dynamic_kernel {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E206",
                            source: node.source.clone(),
                            message: "Pauli estimation does not support mid-circuit feedback"
                                .into(),
                            hint: "Use a unitary kernel for Pauli estimation".into(),
                        });
                    }
                }
            }
            if node.op == "sample" && device.kind == "qpu" {
                let supported_simulator = device.simulator_method.as_deref() == Some("statevector")
                    && device.simulator_host.as_deref().is_some_and(|host| {
                        host == "cpu0"
                            || host.strip_prefix("gpu").is_some_and(|ordinal| {
                                !ordinal.is_empty() && ordinal.bytes().all(|b| b.is_ascii_digit())
                            })
                    })
                    && !matches!(device.site.as_deref(), Some(site) if site != "local")
                    && device.billing.as_deref().unwrap_or("none") == "none";
                let supported_cloud = device.provider.as_deref() == Some("lqcloud")
                    && device.backend.as_deref() == Some("MQ02")
                    && device.billing.as_deref() == Some("metered")
                    && device.site.as_deref() == Some("remote");
                if !supported_simulator && !supported_cloud {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E203",
                        source: node.source.clone(),
                        message: format!("sample has no implementation for {}", device.id),
                        hint: "Use a statevector simulator or metered lqcloud MQ02".into(),
                    });
                }
                let required_qubits = node.inputs.first().and_then(|id| kernel_qubits(graph, *id));
                if let (Some(required_qubits), Some(qubits)) = (required_qubits, device.qubits)
                    && required_qubits > u64::from(qubits)
                {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E209",
                        source: node.source.clone(),
                        message: format!(
                            "sample requires {} qubits; {} has {}",
                            required_qubits, device.id, qubits
                        ),
                        hint: "Choose a larger QPU or a smaller active space".into(),
                    });
                }
                if device.qubits.is_none() {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E209",
                        source: node.source.clone(),
                        message: format!("{} has no declared qubit capacity", device.id),
                        hint: "Declare the QPU's physical or logical qubit capacity".into(),
                    });
                }
                let dynamic_kernel = node
                    .inputs
                    .first()
                    .and_then(|id| original_node(graph, *id))
                    .is_some_and(|kernel| {
                        kernel.op == "kernel.custom"
                            && kernel
                                .params
                                .get("n_clbits")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                > 0
                    });
                if dynamic_kernel && device.simulator_host.as_deref() != Some("cpu0") {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E206",
                        source: node.source.clone(),
                        message: "Mid-circuit feedback is unavailable on this QPU implementation"
                            .into(),
                        hint: "Use a CPU-backed QPU simulator with dynamic-circuit support".into(),
                    });
                }
            }
            if node.op == "vqe.sampled" {
                if device.kind != "qpu" {
                    diagnostics.push(Diagnostic {
                        code: "QCL-E203",
                        source: node.source.clone(),
                        message: "Sampled VQE requires a QPU or QPU simulator".into(),
                        hint: "Place the VQE estimate on a QPU device".into(),
                    });
                } else {
                    let supported_simulator = device.simulator_method.as_deref()
                        == Some("statevector")
                        && device.simulator_host.as_deref() == Some("cpu0")
                        && !matches!(device.site.as_deref(), Some(site) if site != "local")
                        && device.billing.as_deref().unwrap_or("none") == "none";
                    let supported_cloud = device.provider.as_deref() == Some("lqcloud")
                        && device.backend.as_deref() == Some("MQ02")
                        && device.billing.as_deref() == Some("metered")
                        && device.site.as_deref() == Some("remote");
                    if !supported_simulator && !supported_cloud {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E203",
                            source: node.source.clone(),
                            message: format!("Sampled VQE has no implementation for {}", device.id),
                            hint: "Use a CPU-backed statevector simulator or metered MQ02".into(),
                        });
                    }
                    let orbitals = node
                        .inputs
                        .first()
                        .and_then(|id| active_orbitals(graph, *id));
                    if let (Some(orbitals), Some(qubits)) = (orbitals, device.qubits)
                        && 2 * orbitals > u64::from(qubits)
                    {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E209",
                            source: node.source.clone(),
                            message: format!(
                                "Sampled VQE requires {} qubits; {} has {}",
                                2 * orbitals,
                                device.id,
                                qubits
                            ),
                            hint: "Choose a larger QPU or a smaller active space".into(),
                        });
                    }
                    if device.qubits.is_none() {
                        diagnostics.push(Diagnostic {
                            code: "QCL-E209",
                            source: node.source.clone(),
                            message: format!("{} has no declared qubit capacity", device.id),
                            hint: "Declare the QPU's physical or logical qubit capacity".into(),
                        });
                    }
                }
            }
        }
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcl_ir::{Device, NodeKind, ValueType};

    #[test]
    fn checks_unplaced_and_unsupported_device() {
        let mut graph = Graph::new();
        let molecule = graph.add(
            "molecule",
            NodeKind::Host,
            ValueType::Molecule,
            vec![],
            serde_json::json!({"spin": 0}),
            "molecule.py:3".into(),
        );
        let first = graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            serde_json::Value::Null,
            "first.py:4".into(),
        );
        let second = graph.add(
            "scf.rhf",
            NodeKind::Compute,
            ValueType::ScfResult,
            vec![molecule],
            serde_json::Value::Null,
            "second.py:5".into(),
        );
        graph
            .place(
                second,
                Device {
                    id: "sim0".into(),
                    kind: "qpu".into(),
                    ..Device::default()
                },
                "test".into(),
            )
            .unwrap();
        let diagnostics = check(&graph);
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[0].code, "QCL-E201");
        assert_eq!(diagnostics[0].source, "first.py:4");
        assert_eq!(diagnostics[1].code, "QCL-E203");
        assert_eq!(first, 1);
    }
}
