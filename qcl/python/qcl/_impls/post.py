"""Post-processing: thermochemistry, rate constants and Tc estimates."""

from qcl._impls import implements


@implements("thermo.rrho")
def _op_thermo_rrho(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    from chemflow.domains.thermochemistry import thermochemistry

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("RRHO currently requires a CPU")
    molecule, modes = inputs
    model = ElectronicModel(basis=molecule["basis"],
                            reference="ROHF" if molecule["spin"] else "RHF")
    params = node["params"]
    result = thermochemistry(store, molecule, model, modes,
                             temperature_K=params["temperature_K"],
                             pressure_Pa=params["pressure_Pa"])
    correction = "g_corr" if params["quantity"] == "gibbs" else "h_corr"
    values[identifier] = {
        "value": result[correction], "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": "rrho_correction", "molecule": molecule},
            "method": {"family": "RRHO", "quantity": params["quantity"],
                       "temperature_K": params["temperature_K"],
                       "pressure_Pa": params["pressure_Pa"],
                       "n_imaginary": result["n_imaginary"]},
            "implementation": "thermo.rrho/chemflow-thermo",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
        },
    }


@implements("thermo.eyring")
def _op_thermo_eyring(ctx, node, identifier, operation, inputs):
    values = ctx.values
    store = ctx.store
    from chemflow.domains.paths import rate_constant

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Eyring currently requires a CPU")
    temperature = node["params"]["temperature_K"]
    result = rate_constant(store, inputs[0]["value"], temperature)
    values[identifier] = {
        "value": result["k_per_s"], "unit": "1/s", "variance": 0.0,
        "provenance": {
            "question": {"property": "eyring_rate"},
            "method": {"family": "Eyring", "temperature_K": temperature,
                       "tunnelling": None},
            "implementation": "thermo.eyring/chemflow-thermo",
            "device": device["id"], "versions": {},
            "barrier": inputs[0]["value"],
            "validity_flags": ["No tunnelling or dynamical recrossing correction"],
        },
    }


@implements("superconductivity.allen_dynes")
def _op_superconductivity_allen_dynes(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.domains.lattice_correlations import allen_dynes_tc

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Allen-Dynes currently requires a CPU")
    params = node["params"]
    result = allen_dynes_tc(lambda_ep=params["lambda_ep"],
                            omega_log_K=params["omega_log_K"],
                            mu_star=params["mu_star"])
    values[identifier] = {
        "value": result["tc_K"], "unit": "K", "variance": 0.0,
        "provenance": {
            "question": {"property": "conventional_critical_temperature"},
            "method": {"family": "Allen-Dynes", "lambda_ep": params["lambda_ep"],
                       "omega_log_K": params["omega_log_K"],
                       "mu_star": params["mu_star"]},
            "implementation": operation + "/chemflow",
            "device": device["id"], "versions": {},
            "validity_flags": ["Requires independently supplied electron-phonon inputs",
                               "Not a Hubbard Tc prediction"],
        },
    }
