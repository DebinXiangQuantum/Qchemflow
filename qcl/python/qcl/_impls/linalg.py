"""Dense linear algebra and tensor contractions."""

from qcl._impls import implements
from qcl._impls._common import _gpu_index
from qcl._tensor_units import product_unit


@implements("eigh")
def _op_eigh(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np
    from scipy.linalg import eigh as scipy_eigh

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        matrix = cp.asarray(inputs[0]["array"])
        metric = cp.asarray(inputs[1]["array"]) if len(inputs) > 1 else None
        xp = cp
    elif device["kind"] == "cpu":
        matrix = np.asarray(inputs[0]["array"])
        metric = np.asarray(inputs[1]["array"]) if len(inputs) > 1 else None
        xp = np
    else:
        raise ValueError("Dense eigensystems require a CPU or GPU")
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            metric is not None and metric.shape != matrix.shape or
            not bool(xp.allclose(matrix, matrix.T, atol=1e-10)) or
            metric is not None and not bool(xp.allclose(
                metric, metric.T, atol=1e-10))):
        raise ValueError("eigh requires symmetric square matrices of the same shape")
    if gpu:
        if metric is None:
            eigenvalues, eigenvectors = cp.linalg.eigh(matrix)
        else:
            factor = cp.linalg.cholesky(metric)
            inverse = cp.linalg.solve(factor, cp.eye(len(factor)))
            transformed = inverse @ matrix @ inverse.T
            eigenvalues, rotated = cp.linalg.eigh(transformed)
            eigenvectors = inverse.T @ rotated
    else:
        eigenvalues, eigenvectors = scipy_eigh(matrix, metric)
    provenance = {
        "question": {"property": "generalized_eigensystem"
                     if metric is not None else "eigensystem"},
        "method": {"family": "dense Hermitian eigensolver",
                   "generalized": metric is not None},
        "implementation": "eigh/cupy-cusolver" if gpu else "eigh/scipy",
        "device": device["id"],
        "versions": {"cupy": cp.__version__} if gpu else
                    {"scipy": __import__("scipy").__version__},
    }
    values[identifier] = {
        "values": {"array": eigenvalues, "device": device["id"],
                   "unit": inputs[0]["unit"], "layout": "C",
                   "provenance": provenance.copy()},
        "vectors": {"array": eigenvectors, "device": device["id"],
                    "unit": "dimensionless", "layout": "C",
                    "provenance": provenance.copy()},
        "provenance": provenance,
    }


@implements("orthogonalize")
def _op_orthogonalize(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np
    from scipy.linalg import eigh as scipy_eigh

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        xp = cp
        matrix = cp.asarray(inputs[0]["array"])
    elif device["kind"] == "cpu":
        xp = np
        matrix = np.asarray(inputs[0]["array"])
    else:
        raise ValueError("Orthogonalization requires a CPU or GPU")
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            not bool(xp.allclose(matrix, matrix.T, atol=1e-10))):
        raise ValueError("orthogonalize requires a symmetric square matrix")
    eigenvalues, eigenvectors = (cp.linalg.eigh(matrix) if gpu else
                                  scipy_eigh(matrix))
    if bool(xp.any(eigenvalues <= 1e-12)):
        raise ValueError("orthogonalize requires a positive-definite overlap matrix")
    orthogonalizer = (eigenvectors / xp.sqrt(eigenvalues)) @ eigenvectors.T
    values[identifier] = {
        "array": orthogonalizer, "device": device["id"],
        "unit": "dimensionless", "layout": "C",
        "provenance": {
            "question": {"property": "symmetric_orthogonalization"},
            "method": {"family": "Lowdin symmetric inverse square root"},
            "implementation": "orthogonalize/cupy" if gpu else
                              "orthogonalize/scipy",
            "device": device["id"],
            "versions": {"cupy": cp.__version__} if gpu else
                        {"scipy": __import__("scipy").__version__},
        },
    }


@implements("solve")
def _op_solve(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        xp = cp
        matrix, rhs = (cp.asarray(item["array"]) for item in inputs)
    elif device["kind"] == "cpu":
        from scipy.linalg import solve as scipy_solve

        xp = np
        matrix, rhs = (np.asarray(item["array"]) for item in inputs)
    else:
        raise ValueError("Linear solve requires a CPU or GPU")
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            rhs.ndim not in (1, 2) or rhs.shape[0] != matrix.shape[0]):
        raise ValueError("solve requires a square matrix and matching right-hand side")
    answer = xp.linalg.solve(matrix, rhs) if gpu else scipy_solve(matrix, rhs)
    values[identifier] = {
        "array": answer, "device": device["id"],
        "unit": node["params"]["unit"], "layout": "C",
        "provenance": {
            "question": {"property": "linear_system"},
            "method": {"family": "dense linear solve"},
            "implementation": "solve/cupy" if gpu else "solve/scipy",
            "device": device["id"],
            "versions": {"cupy": cp.__version__} if gpu else
                        {"scipy": __import__("scipy").__version__},
        },
    }


@implements("svd")
def _op_svd(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        matrix = cp.asarray(inputs[0]["array"])
    elif device["kind"] == "cpu":
        from scipy.linalg import svd as scipy_svd

        matrix = np.asarray(inputs[0]["array"])
    else:
        raise ValueError("SVD requires a CPU or GPU")
    if matrix.ndim != 2:
        raise ValueError("svd requires a rank-two Tensor")
    u, singular, vh = (cp.linalg.svd(matrix, full_matrices=False) if gpu else
                       scipy_svd(matrix, full_matrices=False))
    provenance = {
        "question": {"property": "singular_value_decomposition"},
        "method": {"family": "economy SVD"},
        "implementation": "svd/cupy" if gpu else "svd/scipy",
        "device": device["id"],
        "versions": {"cupy": cp.__version__} if gpu else
                    {"scipy": __import__("scipy").__version__},
    }
    values[identifier] = {
        part: {"array": array, "device": device["id"], "unit": unit,
               "layout": "C", "provenance": provenance.copy()}
        for part, array, unit in (("u", u, "dimensionless"),
                                  ("s", singular, inputs[0]["unit"]),
                                  ("vh", vh, "dimensionless"))
    }
    values[identifier]["provenance"] = provenance


@implements("transform")
def _op_transform(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np
    import opt_einsum

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        xp = cp
    elif device["kind"] == "cpu":
        xp = np
    else:
        raise ValueError("AO-to-MO transformation requires a CPU or GPU")
    source, coefficients = inputs
    integrals = xp.asarray(source.get("array", source.get("eri")))
    coefficients = xp.asarray(coefficients["array"])
    if (coefficients.ndim != 2 or integrals.ndim not in (2, 3, 4) or
            any(size != coefficients.shape[0] for size in integrals.shape[-2:]) or
            "eri" in source and any(
                size != coefficients.shape[0] for size in integrals.shape)):
        raise ValueError("transform requires AO integrals and matching coefficient rows")
    spec = {2: "pq,pi,qj->ij", 3: "xpq,pi,qj->xij",
            4: "pqrs,pi,qj,rk,sl->ijkl" if "eri" in source else
               "xypq,pi,qj->xyij"}[integrals.ndim]
    operands = [integrals] + [coefficients] * (4 if "eri" in source else 2)
    path, _ = opt_einsum.contract_path(spec, *operands, optimize="greedy")
    result = opt_einsum.contract(spec, *operands, optimize=path,
                                 backend="cupy" if gpu else "numpy")
    values[identifier] = {
        "array": xp.asarray(result), "device": device["id"],
        "unit": "hartree" if "eri" in source else source["unit"],
        "layout": "C", "provenance": {
            "question": {"property": "ao_to_mo_transform", "rank": integrals.ndim},
            "method": {"family": "einsum", "path": [list(step) for step in path]},
            "implementation": "transform/opt-einsum-cupy" if gpu else
                              "transform/opt-einsum",
            "device": device["id"],
            "versions": {"opt_einsum": opt_einsum.__version__},
        },
    }


@implements("contract")
def _op_contract(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np
    import opt_einsum

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
        arrays = [cp.asarray(item["array"]) for item in inputs]
    elif device["kind"] == "cpu":
        arrays = [np.asarray(item["array"]) for item in inputs]
    else:
        raise ValueError("Tensor contraction requires a CPU or GPU")
    spec = node["params"]["spec"]
    path, _ = opt_einsum.contract_path(spec, *arrays, optimize="greedy")
    result = opt_einsum.contract(spec, *arrays, optimize=path,
                                 backend="cupy" if gpu else "numpy")
    values[identifier] = {
        "array": cp.asarray(result) if gpu else np.asarray(result),
        "device": device["id"],
        "unit": product_unit(item["unit"] for item in inputs),
        "layout": "C", "provenance": {
            "question": {"property": "tensor_contraction", "spec": spec},
            "method": {"family": "einsum", "path": [list(step) for step in path]},
            "implementation": "contract/opt-einsum-cupy" if gpu else
                              "contract/opt-einsum", "device": device["id"],
            "versions": {"opt_einsum": opt_einsum.__version__},
        },
    }
