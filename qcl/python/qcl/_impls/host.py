"""Host operations: molecule edits, checks, control-flow results and constructions."""

from qcl._impls import implements
from qcl._impls._common import (
    _MISSING,
    _molecule_identity,
)


@implements("molecule")
def _op_molecule(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = node["params"]


@implements("molecule.edit")
def _op_molecule_edit(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.domains.paths import geometry_transform

    molecule = inputs[0]
    identity = _molecule_identity(molecule)
    params = node["params"]
    result = geometry_transform(None, identity, f"set_{params['kind']}",
                                params["atoms"], params["value"])
    values[identifier] = {**_molecule_identity({**result, "basis": identity["basis"]}),
                          **({"conformer_provenance": molecule["conformer_provenance"]}
                             if "conformer_provenance" in molecule else {}),
                          "provenance": {
                              "question": {"property": "geometry_edit",
                                           "starting_molecule": identity},
                              "method": {"coordinate": params["kind"],
                                         "atoms": params["atoms"],
                                         "value": params["value"]},
                              "implementation": "molecule.edit/chemflow-host",
                              "device": "host"}}


@implements("frequencies.n_imaginary")
def _op_frequencies_n_imaginary(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[0]["n_imaginary"]


@implements("integer.compare")
def _op_integer_compare(ctx, node, identifier, operation, inputs):
    values = ctx.values
    equal = inputs[0] == node["params"]["expected"]
    values[identifier] = equal if node["params"]["equal"] else not equal


@implements("boolean.constant")
def _op_boolean_constant(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = node["params"]["value"]


@implements("require")
def _op_require(ctx, node, identifier, operation, inputs):
    values = ctx.values
    if not inputs[0]:
        raise ValueError(f"QCL requirement failed at {node['source']}")
    values[identifier] = True


@implements("cond")
def _op_cond(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[1] if inputs[0] else inputs[2]


@implements("eigh.values", "eigh.vectors", "svd.u", "svd.s", "svd.vh")
def _op_eigh_values(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[0][operation.split(".", 1)[1]]


@implements("optimize")
def _op_optimize(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[0]


@implements("collection.index")
def _op_collection_index(ctx, node, identifier, operation, inputs):
    values = ctx.values
    collection = inputs[0]
    index = node["params"]["index"]
    values[identifier] = collection[index] if index < len(collection) else _MISSING


@implements("models.hubbard")
def _op_models_hubbard(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    store = ctx.store
    from chemflow.domains.materials import lattice_model

    params = node["params"]
    lattice_params = {"geometry": params["geometry"], "t": params["t"],
                      "U": params["U"]}
    if params["geometry"] == "square":
        lattice_params["shape"] = params["shape"]
    else:
        lattice_params["n_sites"] = params["shape"][0]
    data = lattice_model(store, **lattice_params)
    values[identifier] = {"data": data, "lattice": params,
                          "question": {"property": "lattice_ground_state",
                                       "lattice": params},
                          "versions": {"pyscf": pyscf.__version__}}


@implements("lattice.sector")
def _op_lattice_sector(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.types import Sector

    params = node["params"]
    values[identifier] = {"sector": Sector(n_alpha=params["n_alpha"],
                                            n_beta=params["n_beta"],
                                            n_orb=inputs[0]["data"].sector.n_orb)}


@implements("active.hamiltonian", "active.sector")
def _op_active_hamiltonian(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[0]


@implements("kernel.hf")
def _op_kernel_hf(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = {**inputs[0], "ansatz": "hf"}


@implements("kernel.lucj")
def _op_kernel_lucj(ctx, node, identifier, operation, inputs):
    values = ctx.values
    vector = inputs[1]["vector"] if len(inputs) == 2 else node["params"]["params"]
    if node["params"].get("jastrow") == "line" and vector is not None:
        import numpy as np

        from chemflow.impls.lucj import LucjParams

        layer = LucjParams.from_vector(np.asarray(vector), inputs[0]["data"].sector.n_orb)
        layer.j_opp[1:] = 0.0   # keep only the chain-adjacent opposite-spin pair
        vector = layer.to_vector().tolist()
    values[identifier] = {**inputs[0], "ansatz": "lucj", "params": vector,
                          "final_rotation": node["params"].get("final_rotation", True)}


@implements("kernel.custom")
def _op_kernel_custom(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = {"ansatz": "custom", **node["params"],
                          "question": {"property": "bitstrings",
                                       "kernel": "custom"},
                          "versions": {}}


@implements("map")
def _op_map(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = ([item for item in inputs[1:] if item is not _MISSING]
                          if node["params"]["dynamic"] else inputs)


@implements("scan")
def _op_scan(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = [(inputs[index], inputs[index + 1])
                          for index in range(1, len(inputs), 2)]


@implements("iterate")
def _op_iterate(ctx, node, identifier, operation, inputs):
    values = ctx.values
    values[identifier] = inputs[0]
