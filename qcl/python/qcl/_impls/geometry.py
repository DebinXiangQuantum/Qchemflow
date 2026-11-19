"""Geometry optimisation, derivatives, frequencies, saddles and IRC paths."""

import sys

from qcl._impls import implements
from qcl._impls._common import (
    _finite_difference_optimize,
    _gpu4pyscf_version,
    _gpu_index,
    _molecule_identity,
)


@implements("irc")
def _op_irc(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    from chemflow.domains.paths import irc

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("IRC currently requires a CPU")
    molecule = _molecule_identity(inputs[0])
    model = ElectronicModel(basis=molecule["basis"], reference="RHF")
    result = irc(store, molecule, model, {"method": "hf"})
    values[identifier] = {**result, "provenance": {
        "question": {"property": "intrinsic_reaction_coordinate",
                     "transition_state": molecule},
        "method": {"family": "Sella IRC/HF", "basis": molecule["basis"]},
        "implementation": "irc/chemflow-sella-irc",
        "device": device["id"], "versions": {"pyscf": pyscf.__version__},
    }}


@implements("path.connects")
def _op_path_connects(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.domains.paths import bond_graph, coordinates

    path, reactant, product = inputs
    observed = [bond_graph(item) for item in path["endpoints"]]
    expected = [bond_graph(item) for item in (reactant, product)]
    symbols = [coordinates(item)[0] for item in
               (path["ts"], reactant, product)]
    values[identifier] = (symbols[0] == symbols[1] == symbols[2] and
                          (observed == expected or observed == expected[::-1]))


@implements("geometry.optimize")
def _op_geometry_optimize(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    import contextlib

    from chemflow.domains.reactions import geometry_optimize

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Geometry optimization currently requires a CPU")
    molecule = inputs[0]
    model = ElectronicModel(basis=molecule["basis"],
                            reference="ROHF" if molecule["spin"] else "RHF")
    with contextlib.redirect_stdout(sys.stderr):
        result = geometry_optimize(store, molecule, model, kind="minimum",
                                   level={"method": "uhf" if molecule["spin"] else "rhf"})
    values[identifier] = {**result, "basis": molecule["basis"],
                          "provenance": {
                              "question": {"property": "optimized_geometry",
                                           "starting_molecule": molecule},
                              "method": {"family": "HF/geomeTRIC", "kind": "minimum"},
                              "implementation": "geometry.optimize/chemflow-geometric",
                              "device": device["id"], "versions": {"pyscf": pyscf.__version__},
                          }}


@implements("geometry.constrained_optimize")
def _op_geometry_constrained_optimize(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    import contextlib

    from ase import Atoms

    from chemflow.domains.paths import constrained_optimize, coordinates

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Constrained optimization currently requires a CPU")
    molecule = _molecule_identity(inputs[0])
    model = ElectronicModel(basis=molecule["basis"],
                            reference="ROHF" if molecule["spin"] else "RHF")
    symbols, positions = coordinates(molecule)
    geometry = Atoms(symbols, positions=positions)
    getters = {"bond": geometry.get_distance, "angle": geometry.get_angle,
               "dihedral": geometry.get_dihedral}
    constraints = [{**item, "value": float(getters[item["type"]](*item["atoms"]))}
                   for item in node["params"]["fixed"]]
    with contextlib.redirect_stdout(sys.stderr):
        result = constrained_optimize(store, molecule, model, {"method": "hf"},
                                      constraints)
    values[identifier] = {**_molecule_identity({**result, "basis": molecule["basis"]}),
                          **({"conformer_provenance": inputs[0]["conformer_provenance"]}
                             if "conformer_provenance" in inputs[0] else {}),
                          "provenance": {
                              "question": {"property": "constrained_geometry",
                                           "starting_molecule": molecule},
                              "method": {"family": "HF/geomeTRIC",
                                         "constraints": constraints},
                              "implementation": (
                                  "geometry.constrained_optimize/chemflow-geometric"),
                              "device": device["id"],
                              "versions": {"pyscf": pyscf.__version__},
                          }}


@implements("geometry.fd_optimize")
def _op_geometry_fd_optimize(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Finite-difference optimization currently requires a CPU")
    molecule = _molecule_identity(inputs[0])
    params = node["params"]
    result = _finite_difference_optimize(molecule, params["fixed"], params["step"])
    values[identifier] = {**_molecule_identity(result),
                          "provenance": {
                              "question": {"property": "optimized_geometry",
                                           "starting_molecule": molecule},
                              "method": {"family": "HF/ASE-BFGS",
                                         "gradient": "finite_difference",
                                         "step_angstrom": params["step"],
                                         "constraints": result["constraints"]},
                              "implementation": (
                                  "geometry.fd_optimize/ase-pyscf-finite-difference"),
                              "device": device["id"],
                              "versions": {"pyscf": pyscf.__version__},
                          }}


@implements("sella.saddle")
def _op_sella_saddle(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    from chemflow.domains.paths import connected_ts

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Sella/IRC currently requires a CPU")
    guess, reactant, product = inputs
    model = ElectronicModel(basis=guess["basis"],
                            reference="ROHF" if guess["spin"] else "RHF")
    result = connected_ts(store, guess, model, reactant, product,
                          {"method": "uhf" if guess["spin"] else "rhf"})
    values[identifier] = {**result, "basis": guess["basis"],
                          "provenance": {
                              "question": {"property": "connected_transition_state",
                                           "reactant": reactant, "product": product},
                              "method": {"family": "Sella IRC/HF", "connected": True,
                                         "n_imaginary": 1},
                              "implementation": "sella.saddle/chemflow-sella-irc",
                              "device": device["id"], "versions": {"pyscf": pyscf.__version__},
                          }}


@implements("frequencies")
def _op_frequencies(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    from chemflow.domains.reactions import frequencies as compute_frequencies

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Harmonic frequencies currently require a CPU")
    molecule = inputs[0]
    model = ElectronicModel(basis=molecule["basis"],
                            reference="ROHF" if molecule["spin"] else "RHF")
    method = "uhf" if molecule["spin"] else "rhf"
    result = compute_frequencies(store, molecule, model, {"method": method})
    values[identifier] = {**result, "provenance": {
        "question": {"property": "harmonic_frequencies", "molecule": molecule},
        "method": {"family": "HF Hessian", "basis": molecule["basis"]},
        "implementation": "frequencies/chemflow-pyscf-hessian",
        "device": device["id"], "versions": {"pyscf": pyscf.__version__},
    }}


@implements("gradient")
def _op_gradient(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values

    import numpy as np
    from pyscf import dft, gto, scf

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
    elif device["kind"] != "cpu":
        raise ValueError("HF/RKS gradient requires a CPU or GPU")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)

    xc = node["params"].get("xc")
    solvent = node["params"].get("solvent")

    def run_mean_field(geometry, *, use_gpu=gpu, functional=xc,
                       pcm_solvent=solvent):
        calculation = (dft.RKS(geometry) if functional is not None
                       else scf.RHF(geometry))
        if functional is not None:
            calculation.xc = functional
        calculation.conv_tol = 1e-10
        if use_gpu:
            calculation = calculation.to_gpu()
        if pcm_solvent:
            if pcm_solvent.get("model") == "smd":
                calculation = calculation.SMD()
                calculation.with_solvent.solvent = pcm_solvent["name"]
            else:
                calculation = calculation.PCM()
                calculation.with_solvent.method = "IEF-PCM"
                calculation.with_solvent.eps = pcm_solvent["eps"]
        calculation.kernel()
        if not calculation.converged:
            raise ValueError("Gradient reference mean field did not converge")
        return calculation

    step = node["params"]["step"]
    if step is None:
        array = run_mean_field(system).nuc_grad_method().kernel()
    else:
        positions = system.atom_coords(unit="Bohr")
        array = np.empty_like(positions)
        for atom in range(system.natm):
            for axis in range(3):
                energies = []
                for sign in (1, -1):
                    displaced = positions.copy()
                    displaced[atom, axis] += sign * step
                    geometry = system.copy()
                    geometry.set_geom_(displaced, unit="Bohr")
                    energies.append(float(run_mean_field(geometry).e_tot))
                array[atom, axis] = (energies[0] - energies[1]) / (2 * step)
    values[identifier] = {
        "array": cp.asarray(array) if gpu else array,
        "device": device["id"], "unit": "hartree/bohr", "layout": "C",
        "provenance": {
            "question": {"property": "electronic_gradient", "molecule": molecule},
            "method": {"family": ("finite-difference" if step else "analytic") +
                                 (" RKS gradient" if xc else " RHF gradient"),
                       "basis": molecule["basis"],
                       **({"xc": xc} if xc else {}),
                       **({"solvent": solvent["name"],
                           **({"solvent_model": "SMD"}
                              if solvent.get("model") == "smd" else
                              {"pcm_eps": solvent["eps"]})} if solvent else {}),
                       **({"step_bohr": step} if step else {})},
            "implementation": ("gradient/" + ("fd-" if step else "") +
                               ("dft-" if xc else "") +
                               (solvent.get("model", "pcm") + "-" if solvent else "") +
                               ("gpu4pyscf" if gpu else "pyscf")),
            "device": device["id"],
            "versions": {"pyscf": pyscf.__version__,
                         **({"gpu4pyscf": _gpu4pyscf_version(),
                             "cupy": cp.__version__} if gpu else {})},
        },
    }


@implements("hessian")
def _op_hessian(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    from unittest.mock import patch

    import numpy as np
    from pyscf import dft, gto, scf

    device = node["device"]
    gpu = device["kind"] == "gpu"
    if gpu:
        import cupy as cp

        cp.cuda.Device(_gpu_index(device["id"])).use()
    elif device["kind"] != "cpu":
        raise ValueError("HF/RKS Hessian requires a CPU or GPU")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    xc = node["params"].get("xc")
    solvent = node["params"].get("solvent")
    calculation = dft.RKS(system) if xc else scf.RHF(system)
    if xc:
        calculation.xc = xc
    calculation.conv_tol = 1e-10
    if gpu:
        calculation = calculation.to_gpu()
    if solvent:
        if solvent.get("model") == "smd":
            calculation = calculation.SMD()
            calculation.with_solvent.solvent = solvent["name"]
        else:
            calculation = calculation.PCM()
            calculation.with_solvent.method = "IEF-PCM"
            calculation.with_solvent.eps = solvent["eps"]
    calculation.kernel()
    if not calculation.converged:
        raise ValueError("Hessian reference mean field did not converge")
    if gpu:
        from gpu4pyscf.lib import multi_gpu

        # Hessian's default dispatch spans all visible GPUs, ignoring this node's placement.
        with patch.object(multi_gpu, "num_devices", 1):
            hessian_array = cp.asarray(calculation.Hessian().kernel())
    else:
        hessian_array = calculation.Hessian().kernel()
    values[identifier] = {
        "mol": system, "array": hessian_array,
        "max_gradient": float(np.max(np.abs(calculation.nuc_grad_method().kernel()))),
        "molecule": molecule, "device": device["id"],
        "unit": "hartree/bohr^2", "layout": "C",
        "provenance": {
            "question": {"property": "electronic_hessian", "molecule": molecule},
            "method": {"family": "analytic RKS Hessian" if xc else
                                 "analytic RHF Hessian",
                       "basis": molecule["basis"],
                       **({"xc": xc} if xc else {}),
                       **({"solvent": solvent["name"],
                           **({"solvent_model": "SMD"}
                              if solvent.get("model") == "smd" else
                              {"pcm_eps": solvent["eps"]})} if solvent else {})},
            "implementation": ("hessian/" + ("dft-" if xc else "") +
                               (solvent.get("model", "pcm") + "-" if solvent else "") +
                               ("gpu4pyscf" if gpu else "pyscf")),
            "device": device["id"],
            "versions": {"pyscf": pyscf.__version__,
                         **({"gpu4pyscf": _gpu4pyscf_version(),
                             "cupy": cp.__version__} if gpu else {})},
        },
    }


@implements("frequencies.from_hessian")
def _op_frequencies_from_hessian(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    import numpy as np
    from pyscf.hessian.thermo import harmonic_analysis

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Frequency analysis currently requires a CPU")
    matrix = inputs[0]
    hessian_array = matrix["array"]
    if hasattr(hessian_array, "get"):
        hessian_array = hessian_array.get()
    modes = harmonic_analysis(matrix["mol"], hessian_array)["freq_wavenumber"]
    hessian_method = matrix["provenance"]["method"]
    values[identifier] = {
        "freq_wavenumber": [float(value.real) if abs(value.imag) < 1e-8 else
                            -float(abs(value.imag)) for value in modes],
        "n_imaginary": int(np.count_nonzero(np.abs(np.imag(modes)) > 50)),
        "max_gradient": matrix["max_gradient"],
        "provenance": {
            "question": {"property": "harmonic_frequencies",
                         "molecule": matrix["molecule"]},
            "method": {"family": ("RKS Hessian" if hessian_method.get("xc")
                                  else "HF Hessian"),
                       "basis": matrix["molecule"]["basis"],
                       **{key: hessian_method[key]
                          for key in ("xc", "solvent", "pcm_eps", "solvent_model")
                          if key in hessian_method}},
            "implementation": "frequencies.from_hessian/pyscf",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
        },
    }
