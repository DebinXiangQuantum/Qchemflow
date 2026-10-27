"""Deployment eligibility, independent of verification and reference execution."""

from chemflow.backends import QPU_CATALOG
from chemflow.diagnostics import Diagnostic


def deployment(program, registry, targets=None, purpose="deploy"):
    nodes = {node.id: node for node in program.nodes}
    roots = set(program.targets if targets is None else targets) - set(program.references)
    closure = set()
    pending = list(roots)
    while pending:
        identifier = pending.pop()
        if identifier in closure or identifier not in nodes:
            continue
        closure.add(identifier)
        pending.extend(nodes[identifier].inputs.values())
    issues, mapping, classical = [], {}, []

    def reject(node, message):
        issues.append(Diagnostic(code="CF-E140", node=node.id, message=message))

    for identifier in sorted(closure):
        node = nodes[identifier]
        implementation = registry.implementations.get(node.binding or node.op)
        if implementation is None:
            reject(node, "No registered deployment implementation")
            continue
        role = implementation.execution_role
        if node.op == "prepare_state":
            consumers = [item for item in program.nodes if identifier in item.inputs.values()]
            sampling_only = consumers and identifier not in roots and all(
                item.op in QPU_CATALOG and item.inputs.get("prep") == identifier
                for item in consumers)
            if sampling_only and node.params.get("ansatz", "hf") in ("hf", "lucj"):
                if implementation.id == "prepare_state":
                    role = "emulator"
            else:
                reject(node, "Prepared state is verification-only: require a realizable HF/LUCJ "
                       "circuit consumed only by QPU-backed sampling, never amplitudes")
                continue
        if role == "verification_only":
            reject(node, f"{implementation.id} is verification-only; no deployable "
                   "implementation (no QPU Pauli estimator yet for VQE/Pauli estimation)")
        elif role == "emulator" and purpose == "production":
            reject(node, f"{implementation.id} is an emulator; production requires QPU "
                   "bindings and circuit preparation without a statevector")
        elif role in ("emulator", "qpu"):
            op = "sample_configurations" if node.op == "prepare_state" else node.op
            devices = QPU_CATALOG.get(op, {})
            choices = devices if role == "emulator" else {
                key: value for key, value in devices.items() if key == implementation.id}
            ancestors, todo = set(), [identifier]
            while todo:
                ancestor = todo.pop()
                if ancestor in ancestors or ancestor not in nodes:
                    continue
                ancestors.add(ancestor)
                todo.extend(nodes[ancestor].inputs.values())
            widths = [2 * nodes[key].params["n_orb"] for key in ancestors
                      if nodes[key].op == "sector" and
                      type(nodes[key].params.get("n_orb")) is int]
            shots = node.params.get("shots_per_group", 1024) if implementation.implements == (
                "vqe_sampled") else node.params.get("shots")
            failures = []
            for device, limits in choices.items():
                if widths and max(widths) > limits.get("qubits", float("inf")):
                    failures.append(f"{device}: {max(widths)} qubits exceeds {limits['qubits']}")
                elif type(shots) is int and not limits.get("min_shots", 1) <= shots <= limits.get(
                        "max_shots", float("inf")):
                    failures.append(f"{device}: shots={shots} outside device limits")
                else:
                    mapping[identifier] = device
                    break
            else:
                reject(node, "; ".join(failures) or "No known QPU implementation for this step")
        elif role == "classical":
            classical.append(identifier)
        else:
            reject(node, "Implementation has no execution role")
    return {"deployable": not issues, "deployment": mapping,
            "deployment_reason": "; ".join(issue.message for issue in issues),
            "classical_nodes": classical}, issues
