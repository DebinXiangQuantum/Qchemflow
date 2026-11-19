"""Explicit value copies between devices."""

from qcl._impls import implements
from qcl._impls._common import (
    _canonical_unrestricted_energies,
    _gpu_index,
    _gpu_kernel,
)


@implements("copy")
def _op_copy(ctx, node, identifier, operation, inputs):
    values = ctx.values
    ci_vectors = ctx.ci_vectors
    store = ctx.store
    device = node["device"]
    if device["kind"] == "cpu":
        if node["inputs"][0] in ci_vectors:
            vector = ci_vectors[node["inputs"][0]]
            ci_vectors[identifier] = (vector.get() if hasattr(vector, "get")
                                      else vector.copy())
        if node["value_type"] == "eri":
            array = inputs[0]["eri"]
            if hasattr(array, "get"):
                array = array.get()
            values[identifier] = {**inputs[0], "eri": array.copy(),
                                  "device": device["id"]}
        elif node["value_type"] in ("tensor", "hessian"):
            array = inputs[0]["array"]
            if hasattr(array, "get"):
                array = array.get()
            values[identifier] = {**inputs[0], "array": array.copy(),
                                  "device": device["id"]}
        elif node["value_type"] in ("eigensystem", "svd_result"):
            parts = (("values", "vectors") if node["value_type"] == "eigensystem"
                     else ("u", "s", "vh"))
            values[identifier] = {
                **inputs[0],
                **{part: {**inputs[0][part],
                          "array": (inputs[0][part]["array"].get()
                                    if hasattr(inputs[0][part]["array"], "get") else
                                    inputs[0][part]["array"].copy()),
                          "device": device["id"]}
                   for part in parts},
            }
        elif node["value_type"] in ("fci_result", "sci_result"):
            values[identifier] = {**inputs[0], "device": device["id"]}
        elif inputs[0].get("gpu_scf"):
            from types import SimpleNamespace

            source = inputs[0]["data"]
            mean_field = source.mf.to_cpu()
            if inputs[0]["method"]["family"] in ("UHF", "UKS"):
                mean_field.mo_energy = _canonical_unrestricted_energies(mean_field)
            values[identifier] = {
                **inputs[0], "gpu_scf": False, "device": device["id"],
                "data": SimpleNamespace(mf=mean_field, mol=source.mol,
                                        model=source.model),
            }
        else:
            values[identifier] = inputs[0]
    elif device["kind"] == "gpu" and node["value_type"] in (
            "tensor", "hessian", "eri"):
        import cupy as cp

        with cp.cuda.Device(_gpu_index(device["id"])):
            key = "eri" if node["value_type"] == "eri" else "array"
            values[identifier] = {**inputs[0],
                                  key: cp.asarray(inputs[0][key]),
                                  "device": device["id"]}
    elif device["kind"] == "gpu" and node["value_type"] == "scf_result":
        from types import SimpleNamespace

        import cupy as cp

        source = inputs[0]["data"]
        with cp.cuda.Device(_gpu_index(device["id"])):
            mean_field = source.mf.to_gpu()
        values[identifier] = {
            **inputs[0], "gpu_scf": True, "device": device["id"],
            "data": SimpleNamespace(mf=mean_field, mol=source.mol,
                                    model=source.model),
        }
    elif device["kind"] == "gpu" and node["value_type"] == "fermion_operator":
        values[identifier] = {**inputs[0], "gpu_kernel":
                              _gpu_kernel(inputs[0]["data"], store, device["id"])}
    elif device["kind"] == "gpu" and node["value_type"] == "sector":
        import cupy as cp

        sector = inputs[0].get("sector", inputs[0]["data"].sector)
        with cp.cuda.Device(_gpu_index(device["id"])):
            resident = cp.asarray([sector.n_alpha, sector.n_beta, sector.n_orb])
        values[identifier] = {**inputs[0], "gpu_sector": resident}
    elif device["kind"] == "gpu" and node["value_type"] == "ground_state":
        import cupy as cp

        with cp.cuda.Device(_gpu_index(device["id"])):
            values[identifier] = {**inputs[0],
                                  "vector_gpu": cp.asarray(inputs[0]["vector"])}
    else:
        raise ValueError(f"Unsupported value copy to {device['kind']}")
