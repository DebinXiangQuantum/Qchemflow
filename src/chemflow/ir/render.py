from chemflow.ir.schema import Program


def render(program: Program) -> str:
    lines = [f"task {program.name}"]
    lines.extend(f"  scan {scan.name} in {scan.values.value} {scan.values.unit}"
                 for scan in program.scans)
    for node in program.nodes:
        args = ", ".join([*(f"{key}={value}" for key, value in node.inputs.items()),
                          *(f"{key}={value}" for key, value in node.params.items())])
        lines.append(f"  {node.id} = {node.op}({args})")
        if node.refinement:
            lines.append(f"  refine {node.id} -> {node.refinement}")
        if node.binding:
            lines.append(f"  bind {node.id} -> {node.binding}")
    lines.extend(f"  accept {gate.node} relation={gate.relation}"
                 for gate in program.gates)
    lines.append(f"  budget {program.budget}")
    return "\n".join(lines)
