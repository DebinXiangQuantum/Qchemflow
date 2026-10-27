"""Full-space ground-energy expansions of molecular component references."""

from chemflow.ir.schema import Node


def molecule_spec(nodes, identifier):
    node = nodes[identifier]
    if node.op == "molecule":
        return {"charge": 0, "spin": 0, "unit": "angstrom", **node.params}
    if node.op == "scan_fragment":
        from chemflow.domains.drug import component_specs

        return component_specs(molecule_spec(nodes, node.inputs["molecule"]),
            node.params["fragments"], node.params["electronic_model"],
            node.params["counterpoise"])[node.params["index"]]
    if node.op in ("geometry_transform", "constrained_optimize"):
        return molecule_spec(nodes, node.inputs["molecule"])
    raise ValueError(f"Cannot determine molecular basis for {node.op}")


def ground_reference(nodes, reference):
    from pyscf import gto

    prefix = f"{reference.id}_full"
    additions = []
    if "molecule" in reference.inputs:
        molecule_id = reference.inputs["molecule"]
        molecule = molecule_spec(nodes, molecule_id)
    else:
        molecule = {"charge": 0, "spin": 0, "unit": "angstrom", **reference.params["molecule"]}
        molecule_id = f"{prefix}_molecule"
        additions.append(Node(id=molecule_id, op="molecule", params=molecule))
    model = Node(id=f"{prefix}_model", op="electronic_model",
                 params=dict(reference.params["electronic_model"]))
    basis = gto.M(atom=molecule["atoms"], basis=model.params["basis"],
                  charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                  unit=molecule.get("unit", "angstrom"), verbose=0)
    hamiltonian = Node(id=f"{prefix}_hamiltonian", op="hamiltonian",
                       inputs={"molecule": molecule_id, "model": model.id})
    sector = Node(id=f"{prefix}_sector", op="sector",
                  params={"n_alpha": basis.nelec[0], "n_beta": basis.nelec[1],
                          "n_orb": basis.nao_nr()})
    additions.extend([model, hamiltonian, sector])
    energy = Node(id=reference.id, op="ground_energy",
                  inputs={"hamiltonian": hamiltonian.id, "sector": sector.id})
    return additions, energy
