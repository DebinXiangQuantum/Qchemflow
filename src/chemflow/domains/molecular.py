from chemflow import build
from chemflow.diagnostics import AdapterError
from chemflow.ops.molecular import molecular_registry


def register(registry):
    source = molecular_registry()
    registry.operators.update(source.operators)
    registry.refinements.update(source.refinements)
    registry.implementations.update(source.implementations)


def identity(plan, record, contexts):
    try:
        nodes = {node.id: node for node in plan.nodes}
        target = record.spec.target
        molecule = next((node for node in plan.nodes if node.op == "molecule"), None)
        model = next((node for node in plan.nodes if node.op == "electronic_model"), None)
        if molecule is None or model is None:
            raise AdapterError("CF-E120", None, "Missing target system", "Match the frozen task")
        expected_molecule = {"charge": 0, "spin": 0, "unit": "angstrom",
                             **target.molecule}
        actual_molecule = {"charge": 0, "spin": 0, "unit": "angstrom",
                           **molecule.params}
        if actual_molecule != expected_molecule or model.params != target.electronic_model:
            raise AdapterError("CF-E120", None, "Target system mismatch", "Match the frozen task")
        for node_id in plan.targets:
            node = nodes.get(node_id)
            context = contexts.get(node_id, {})
            sector = nodes.get(node.inputs.get("sector", "")) if node else None
            hamiltonian = nodes.get(node.inputs.get("hamiltonian", "")) if node else None
            active = hamiltonian.params.get("active") if hamiltonian else None
            if (node is None or node.op != target.operator or
                hamiltonian is None or hamiltonian.op != "hamiltonian" or
                hamiltonian.inputs.get("molecule") != molecule.id or
                hamiltonian.inputs.get("model") != model.id or
                (target.active != "auto" and active != target.active) or
                (target.active == "auto" and not isinstance(active, dict)) or
                sector is None or (sector.params != target.sector if target.active != "auto"
                                   else (sector.params.get("n_orb") != active.get("n_orb") or
                                         sector.params.get("n_alpha", 0) +
                                         sector.params.get("n_beta", 0) != active.get("n_elec") or
                                         any(sector.params.get(key) != value for key, value in
                                             target.sector.items()))) or
                context.get("target_spin") != target.target_spin):
                raise AdapterError("CF-E120", node_id, "Plan target identity mismatch",
                                   "Match operator, model, active space, sector and spin")
    except AdapterError as exc:
        from chemflow.verify.identity_diff import identity_hints, plan_hints

        if exc.diagnostics[0].node is None:
            hints = identity_hints({"molecule": {"charge": 0, "spin": 0, "unit": "angstrom",
                                               **molecule.params} if molecule else None,
                                    "model": model.params if model else None},
                                   [{"molecule": {"charge": 0, "spin": 0, "unit": "angstrom",
                                                  **target.molecule},
                                     "model": target.electronic_model}])
        elif target.active == "auto" and (not isinstance(active, dict) or
                                         not {"n_orb", "n_elec"} <= active.keys()):
            hints = identity_hints({"active": active}, [{"active": {"n_orb": "required",
                                                                     "n_elec": "required"}}])
        else:
            hints = plan_hints(plan, base_plan(record, active=active)[0], node_id)
        return [issue.model_copy(update={"fix_hints": hints}) for issue in exc.diagnostics]
    return []


def base_plan(record, active=None):
    active = record.spec.target.active if active is None else active
    if active == "auto":
        raise ValueError("Resolve the molecular active space before building the plan")
    target = record.spec.target
    with build.program("agent-screen") as builder:
        molecule = builder.molecule(**target.molecule)
        model = builder.electronic_model(**target.electronic_model)
        hamiltonian = builder.hamiltonian(molecule, model, active=active)
        sector_params = (target.sector if target.active != "auto" else
                         {**target.sector, "n_orb": active["n_orb"],
                          "n_alpha": (active["n_elec"] +
                                      int(target.molecule.get("spin", 0))) // 2,
                          "n_beta": (active["n_elec"] -
                                     int(target.molecule.get("spin", 0))) // 2})
        sector = builder.sector(**sector_params)
        controls = []
        for template, binding in (("hf", "hf"), ("mp2", "mp2"),
                                  ("ccsd", "ccsd"), ("ccsd_t", "ccsd_t"),
                                  ("fci", "fci_pyscf")):
            node = builder.ground_energy(hamiltonian, sector,
                **({"s": target.target_spin} if target.target_spin is not None else {}))
            builder.refine(node, template)
            builder.bind(node, binding)
            if template == "fci":
                builder.ir.references = [node.id]
            else:
                controls.append(node.id)
                next(item for item in builder.ir.nodes if item.id == node.id).metadata[
                    "optional_control"] = True
        builder.ir.targets = builder.ir.references[:]
        return builder.to_ir(), controls
