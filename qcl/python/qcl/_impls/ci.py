"""CI-space solvers, sigma products, densities, lattice ED and spectral functions."""

from qcl._impls import implements
from qcl._impls._common import (
    _gpu_index,
    _gpu_kernel,
    _native_ci_order,
    lattice_observable_value,
)


@implements("dynamics.spectral_exact")
def _op_dynamics_spectral_exact(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    store = ctx.store
    from chemflow.domains.dynamics import impurity_model, spectral_exact

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Lehmann spectral resolution currently requires a CPU")
    params = node["params"]
    data = impurity_model(store, **params["model"])
    result = spectral_exact(store, data, params["sector"], params["orbital"],
                            params["spin"], params["omega"], params["eta"])
    values[identifier] = {
        "omega": list(result.omega), "spectral": list(result.spectral),
        "unit": result.unit, "relation": "equivalent",
        "diagnostics": result.diagnostics,
        "provenance": {
            "question": {"property": "spectral_function", **params},
            "method": {"family": "exact Lehmann", "resolution": "complete"},
            "implementation": "dynamics.spectral_exact/chemflow-lehmann",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
        },
    }


@implements("ed.ground_state")
def _op_ed_ground_state(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    import numpy as np
    from scipy.sparse.linalg import eigsh

    from chemflow.impls.fci import ci_string_order, matrix, sector_states

    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError("Lattice ED requires a CPU or GPU")
    data = inputs[0]["data"]
    sector = inputs[1]["sector"]
    basis = sector_states(sector)
    if device["kind"] == "gpu":
        import cupy as cp

        from chemflow.backends.gpu import FciKernel, fci_ground_state

        with cp.cuda.Device(_gpu_index(device["id"])):
            kernel = FciKernel(store.get(data.hamiltonian.h1_ref),
                               store.get(data.hamiltonian.h2_ref), sector.n_orb,
                               (sector.n_alpha, sector.n_beta), xp=cp)
            energy, _spin, diagnostics, ci_vector = fci_ground_state(
                kernel, data.hamiltonian.ledger.total().value,
                return_wavefunction=True)
            alpha_order, beta_order = ci_string_order(sector)
            vector = np.asarray(ci_vector)[np.ix_(alpha_order, beta_order)].ravel()
            vector_gpu = cp.asarray(vector)
            cp.cuda.Device().synchronize()
        implementation = "ed.ground_state/chemflow-cupy"
    else:
        operator = matrix(data.hamiltonian, store, basis)
        if len(basis) <= 32:
            eigvals, eigvecs = np.linalg.eigh(operator.toarray())
            energy, vector = float(eigvals[0]), eigvecs[:, 0]
        else:
            eigvals, eigvecs = eigsh(operator, k=1, which="SA", tol=1e-11)
            energy, vector = float(eigvals[0]), eigvecs[:, 0]
        implementation = "ed.ground_state/chemflow-sparse"
    values[identifier] = {
        **inputs[0], "sector": sector, "energy": energy, "vector": vector,
        "device": device["id"],
        "method": {"family": "Hubbard ED", "geometry": inputs[0]["lattice"]["geometry"],
                   "n_sites": sector.n_orb, "n_alpha": sector.n_alpha,
                   "n_beta": sector.n_beta, "t": inputs[0]["lattice"]["t"],
                   "U": inputs[0]["lattice"]["U"]},
        "implementation": implementation,
        **({"vector_gpu": vector_gpu, "diagnostics": diagnostics}
           if device["kind"] == "gpu" else {}),
    }


@implements("expect")
def _op_expect(ctx, node, identifier, operation, inputs):
    values = ctx.values
    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError("Lattice expectation requires a CPU or GPU")
    state = inputs[0]
    observable = node["params"]["observable"]
    left, right = node["params"]["left"], node["params"]["right"]
    if device["kind"] == "gpu":
        import cupy as cp

        ordinal = _gpu_index(device["id"])
        with cp.cuda.Device(ordinal):
            vector = state.get("vector_gpu")
            if getattr(getattr(vector, "device", None), "id", None) != ordinal:
                vector = cp.asarray(state["vector"])
            value = lattice_observable_value(vector, state["sector"],
                                             observable, left, right, cp)
            cp.cuda.Device().synchronize()
        implementation = "expect/chemflow-cupy"
    else:
        import numpy as np

        value = lattice_observable_value(state["vector"], state["sector"],
                                         observable, left, right, np)
        implementation = "expect/chemflow-lattice-correlations"
    values[identifier] = {
        "value": value,
        "unit": "dimensionless", "variance": 0.0,
        "provenance": {
            "question": {**state["question"], "property": observable,
                         "sites": [left, right]},
            "method": state["method"],
            "implementation": implementation,
            "device": device["id"], "versions": state["versions"],
            "validity_flags": ["Finite lattice correlation; no Tc inference"],
        },
    }


@implements("ci.vector")
def _op_ci_vector(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np
    from pyscf import fci

    sector = inputs[0].get("sector", inputs[0]["data"].sector)
    shape = tuple(fci.cistring.num_strings(sector.n_orb, count)
                  for count in (sector.n_alpha, sector.n_beta))
    array = np.asarray(node["params"]["values"], dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("CI coefficient array differs from its sector")
    values[identifier] = {
        "array": array, "device": "host", "unit": "dimensionless",
        "layout": "C", "sector": {"n_alpha": sector.n_alpha,
                                 "n_beta": sector.n_beta},
        "provenance": {
            "question": inputs[0]["question"],
            "method": {"family": "CI coefficients",
                       "active_orbitals": sector.n_orb,
                       "sector": {"n_alpha": sector.n_alpha,
                                  "n_beta": sector.n_beta},
                       "ordering": "pyscf_cistring"},
            "implementation": "ci.vector/qcl-host", "device": "host",
            "versions": inputs[0]["versions"],
        },
    }


@implements("fci.solve")
def _op_fci_solve(ctx, node, identifier, operation, inputs):
    from chemflow.impls.pyscf_impls import fci_pyscf

    values = ctx.values
    ci_vectors = ctx.ci_vectors
    density_sources = ctx.density_sources
    store = ctx.store
    from dataclasses import replace

    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported fci.solve device: {device['kind']}")
    if inputs[0]["data"] is not inputs[1]["data"]:
        raise ValueError("FCI operator and sector refer to different active spaces")
    data = inputs[0]["data"]
    sector = inputs[1].get("sector", data.sector)
    if sector != data.sector:
        data = replace(data, sector=sector)
    if node["params"].get("engine") == "native":
        from qcl._qcl import CI_VERSION, ci_ground_state

        h1 = store.get(data.hamiltonian.h1_ref)
        eri = store.get(data.hamiltonian.h2_ref)
        energy, vector = ci_ground_state(
            sector.n_orb, sector.n_alpha, sector.n_beta,
            h1.ravel().tolist(), eri.ravel().tolist(),
            data.hamiltonian.ledger.total().value)
        ci_vectors[identifier] = vector
        implementation = "fci.solve/qcl-rust"
        versions = {**inputs[0]["versions"], "qcl-ci": CI_VERSION}
    elif device["kind"] == "gpu":
        import cupy as cp

        from chemflow.backends.gpu import fci_ground_state

        with cp.cuda.Device(_gpu_index(device["id"])):
            kernel = (inputs[0].get("gpu_kernel") if sector == inputs[0]["data"].sector
                      else None) or _gpu_kernel(data, store, device["id"])
            energy, _spin_squared, _diagnostics, vector = fci_ground_state(
                kernel, data.hamiltonian.ledger.total().value,
                return_wavefunction=True,
                residual_tol=1e-9 if identifier in density_sources else None,
                gpu_resident=True)
            cp.cuda.Device().synchronize()
        ci_vectors[identifier] = vector
        implementation = "fci.solve/chemflow-cupy"
        versions = {**inputs[0]["versions"], "cupy": cp.__version__}
    else:
        energy, _spin_squared, vector = fci_pyscf(
            data, store, return_wavefunction=True,
            residual_tol=1e-9 if identifier in density_sources else None)
        ci_vectors[identifier] = vector
        implementation = "fci.solve/chemflow-pyscf"
        versions = inputs[0]["versions"]
    values[identifier] = {
        **inputs[0], "energy": energy, "device": device["id"],
        "method": {"family": "FCI", "basis": data.model.basis,
                   "active_electrons": data.active.n_elec,
                   "active_orbitals": data.active.n_orb,
                   "sector": {"n_alpha": sector.n_alpha,
                              "n_beta": sector.n_beta},
                   **({"solver": _diagnostics["solver"]}
                      if device["kind"] == "gpu" else {})},
        "implementation": implementation, "versions": versions,
    }


@implements("rdm1", "rdm2")
def _op_rdm1(ctx, node, identifier, operation, inputs):
    values = ctx.values
    ci_vectors = ctx.ci_vectors
    import numpy as np
    from pyscf.fci import direct_spin1

    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError(f"FCI {operation} requires a CPU or GPU")
    state = inputs[0]
    vector = ci_vectors.get(node["inputs"][0])
    if vector is None:
        raise ValueError(f"FCI wavefunction is unavailable for {operation}")
    sector = state["method"]["sector"]
    orbitals = state["method"]["active_orbitals"]
    if device["kind"] == "gpu":
        import cupy as cp

        from chemflow.backends.gpu import fci_density_matrix

        with cp.cuda.Device(_gpu_index(device["id"])):
            matrix = fci_density_matrix(
                vector, orbitals, (sector["n_alpha"], sector["n_beta"]),
                1 if operation == "rdm1" else 2, cp)
        implementation = f"{operation}/chemflow-cupy"
    elif state["implementation"] == "fci.solve/qcl-rust":
        from qcl._qcl import ci_rdm1, ci_rdm2

        kernel = ci_rdm1 if operation == "rdm1" else ci_rdm2
        matrix = np.asarray(kernel(
            orbitals, sector["n_alpha"], sector["n_beta"], vector))
        matrix = matrix.reshape((orbitals,) * (2 if operation == "rdm1" else 4))
        implementation = f"{operation}/qcl-rust"
    else:
        electrons = (sector["n_alpha"], sector["n_beta"])
        matrix = (direct_spin1.make_rdm1(vector, orbitals, electrons)
                  if operation == "rdm1" else
                  direct_spin1.make_rdm12(vector, orbitals, electrons)[1])
        implementation = f"{operation}/chemflow-pyscf"
    values[identifier] = {
        "array": matrix, "device": device["id"],
        "unit": "dimensionless", "layout": "C",
        "provenance": {
            "question": state["question"],
            "method": {**state["method"], "property": f"spin_summed_{operation}",
                       "orbital_basis": "active_mo"},
            "implementation": implementation, "device": device["id"],
            "versions": state["versions"],
        },
    }


@implements("fci.coefficients")
def _op_fci_coefficients(ctx, node, identifier, operation, inputs):
    values = ctx.values
    ci_vectors = ctx.ci_vectors
    import numpy as np
    from pyscf import fci

    state = inputs[0]
    vector = ci_vectors.get(node["inputs"][0])
    if vector is None:
        raise ValueError("FCI wavefunction is unavailable for coefficients")
    sector = state["method"]["sector"]
    norb = state["method"]["active_orbitals"]
    nelec = (sector["n_alpha"], sector["n_beta"])
    shape = tuple(fci.cistring.num_strings(norb, count) for count in nelec)
    matrix = vector.reshape(shape) if hasattr(vector, "get") else np.asarray(vector).reshape(shape)
    if state["implementation"] == "fci.solve/qcl-rust":
        order = [_native_ci_order(norb, count) for count in nelec]
        matrix = matrix[np.ix_(*order)]
    if state["device"].startswith("gpu"):
        import cupy as cp

        with cp.cuda.Device(_gpu_index(state["device"])):
            matrix = cp.asarray(matrix)
    values[identifier] = {
        "array": matrix, "device": state["device"],
        "unit": "dimensionless", "layout": "C",
        "sector": sector, "provenance": {
            "question": state["question"],
            "method": {**state["method"], "property": "ci_coefficients",
                       "ordering": "pyscf_cistring"},
            "implementation": "fci.coefficients/chemflow",
            "device": state["device"], "versions": state["versions"],
        },
    }


@implements("sigma")
def _op_sigma(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    device = node["device"]
    operator, coefficients = inputs
    sector = coefficients["sector"]
    norb = coefficients["provenance"]["method"]["active_orbitals"]
    nelec = (sector["n_alpha"], sector["n_beta"])
    data = operator["data"]
    if data.active.n_orb != norb:
        raise ValueError("Sigma operator and CI coefficients have different orbitals")
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        from chemflow.backends.gpu import FciKernel

        with cp.cuda.Device(_gpu_index(device["id"])):
            kernel = FciKernel(store.get(data.hamiltonian.h1_ref),
                               store.get(data.hamiltonian.h2_ref), norb, nelec,
                               xp=cp)
            vector = cp.asarray(coefficients["array"])
            product = kernel.sigma(vector).reshape(vector.shape)
            product += data.hamiltonian.ledger.total().value * vector
        implementation = "sigma/chemflow-cupy"
        versions = {**operator["versions"], "cupy": cp.__version__}
    elif node["params"].get("engine") == "native":
        import numpy as np

        from qcl._qcl import CI_VERSION, ci_sigma

        vector = np.asarray(coefficients["array"])
        order = [_native_ci_order(norb, count) for count in nelec]
        native = vector[np.ix_(*(np.argsort(indices) for indices in order))]
        h1 = store.get(data.hamiltonian.h1_ref)
        eri = store.get(data.hamiltonian.h2_ref)
        native_product = np.asarray(ci_sigma(
            norb, *nelec, h1.ravel().tolist(), eri.ravel().tolist(),
            data.hamiltonian.ledger.total().value,
            native.ravel().tolist())).reshape(vector.shape)
        product = native_product[np.ix_(*order)]
        implementation = "sigma/qcl-rust"
        versions = {**operator["versions"], "qcl-ci": CI_VERSION}
    else:
        import numpy as np

        from chemflow.backends.gpu import FciKernel

        vector = np.asarray(coefficients["array"])
        kernel = FciKernel(store.get(data.hamiltonian.h1_ref),
                           store.get(data.hamiltonian.h2_ref), norb, nelec)
        product = kernel.sigma(vector).reshape(vector.shape)
        product += data.hamiltonian.ledger.total().value * vector
        implementation = "sigma/chemflow-numpy"
        versions = operator["versions"]
    values[identifier] = {
        "array": product, "device": device["id"],
        "unit": "hartree", "layout": "C",
        "sector": sector, "provenance": {
            "question": operator["question"],
            "method": {**coefficients["provenance"]["method"],
                       "property": "hamiltonian_action"},
            "implementation": implementation,
            "device": device["id"], "versions": versions,
        },
    }


@implements("sci.solve")
def _op_sci_solve(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    ci_vectors = ctx.ci_vectors
    store = ctx.store
    from dataclasses import replace

    import numpy as np

    from chemflow.impls.fci import ci_string_order
    from chemflow.impls.sci import sci_pyscf

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"Unsupported SCI device: {device['kind']}")
    if inputs[0]["data"] is not inputs[1]["data"]:
        raise ValueError("SCI operator and sector refer to different active spaces")
    data = inputs[0]["data"]
    sector = inputs[1].get("sector", data.sector)
    if sector != data.sector:
        data = replace(data, sector=sector)
    settings = node["params"]
    energy, diagnostics = sci_pyscf(
        data, store, select_cutoff=settings["select_cutoff"],
        total_spin=settings["total_spin"])
    alpha_order, beta_order = ci_string_order(sector)
    amplitudes = np.asarray(store.get(diagnostics["state_ref"])).reshape(
        len(alpha_order), len(beta_order))
    vector = np.empty_like(amplitudes)
    vector[np.ix_(alpha_order, beta_order)] = amplitudes
    ci_vectors[identifier] = vector
    values[identifier] = {
        **inputs[0], "energy": energy, "device": device["id"],
        "method": {"family": "SCI", "basis": data.model.basis,
                   "active_electrons": data.active.n_elec,
                   "active_orbitals": data.active.n_orb,
                   "sector": {"n_alpha": sector.n_alpha,
                              "n_beta": sector.n_beta},
                   "select_cutoff": settings["select_cutoff"],
                   "total_spin": settings["total_spin"],
                   "determinant_count": diagnostics["determinant_count"],
                   "spin_squared": diagnostics["s2"]},
        "implementation": "sci.solve/chemflow-pyscf",
        "versions": {"pyscf": pyscf.__version__},
        "validity_flags": ["Variational upper bound within the signed Hamiltonian"],
    }
