"""DMRG, statevector and ADAPT VQE, EOM-CCSD, NEB, MACE, docking, ligand names,
fragment QM binding and subspace Lehmann spectra."""

from dataclasses import replace

from qcl._impls import implements
from qcl._impls._common import _molecule_identity


def _cpu(node, what):
    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"{what} currently requires a CPU")
    return device


def _space(ctx, inputs):
    """The (data, sector) of an operator/sector pair, as classical.solve reads them."""
    if inputs[1].get("data", inputs[0]["data"]) is not inputs[0]["data"]:
        raise ValueError("Operator and sector refer to different spaces")
    data = inputs[0]["data"]
    sector = inputs[1].get("sector", data.sector)
    if sector != data.sector:
        data = replace(data, sector=sector)
    return data, sector


def _space_method(data, sector, family, **extra):
    return {"family": family,
            **({"basis": data.model.basis, "active_electrons": data.active.n_elec,
                "active_orbitals": data.active.n_orb}
               if getattr(data, "model", None) is not None and data.active is not None else
               {"basis": data.model.basis} if getattr(data, "model", None) is not None else
               {"sites": sector.n_orb}),
            "sector": {"n_alpha": sector.n_alpha, "n_beta": sector.n_beta}, **extra}


def _version(package):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


@implements("dmrg.solve")
def _op_dmrg_solve(ctx, node, identifier, operation, inputs):
    from chemflow.impls.dmrg import dmrg_block2

    device = _cpu(node, "DMRG")
    data, sector = _space(ctx, inputs)
    params = node["params"]
    energy, diagnostics = dmrg_block2(
        ctx.store, data, sector.model_dump(), total_spin=params["total_spin"],
        bond_dimension=params["bond_dimension"], sweeps=params["sweeps"], seed=params["seed"])
    ctx.values[identifier] = {
        **inputs[0], "energy": float(energy), "device": device["id"],
        "method": _space_method(data, sector, "DMRG", bond_dimension=params["bond_dimension"],
                                sweeps=diagnostics["sweeps"],
                                discarded_weight=diagnostics["discarded_weight"]),
        "implementation": "dmrg.solve/block2",
        "versions": {"block2": _version("block2")},
        "validity_flags": ["Normalized MPS expectation: a variational upper bound",
                           f"Final discarded weight {diagnostics['discarded_weight']:.2e}"],
    }


def _sector_engine(ctx, data, sector):
    from chemflow.impls.sector import SectorEngine

    if hasattr(data, "mean_field") and data.mf is None:
        data.mean_field(ctx.store)
    return SectorEngine(data.hamiltonian, sector, ctx.store)


@implements("vqe.statevector")
def _op_vqe_statevector(ctx, node, identifier, operation, inputs):
    from chemflow.impls.vqe import vqe

    device = _cpu(node, "Statevector VQE")
    data, sector = _space(ctx, inputs)
    params = node["params"]
    result = vqe(_sector_engine(ctx, data, sector), sector, ansatz=params["ansatz"],
                 optimizer=params["optimizer"], maxiter=params["maxiter"],
                 spin_adapted=params["spin_adapted"], layers=params["layers"],
                 seed=params["seed"], hamiltonian=data.hamiltonian, store=ctx.store,
                 n_starts=params["n_starts"], init=params["init"])
    ctx.values[identifier] = {
        **inputs[0], "energy": float(result.energy), "device": device["id"],
        "method": _space_method(data, sector, f"VQE({params['ansatz'].upper()})",
                                s2=float(result.s2), n_energy_evals=result.n_energy_evals,
                                converged=bool(result.converged),
                                start_energies=[float(value) for value in
                                                result.start_energies or []]),
        "implementation": "vqe.statevector/chemflow-sector",
        "versions": {"scipy": _version("scipy")},
        "validity_flags": ["Exact statevector emulation: verification only, not deployable",
                           "Variational upper bound within the signed Hamiltonian"],
    }


@implements("vqe.adapt")
def _op_vqe_adapt(ctx, node, identifier, operation, inputs):
    from chemflow.impls.adapt import adapt_vqe

    device = _cpu(node, "ADAPT-VQE")
    data, sector = _space(ctx, inputs)
    params = node["params"]
    result = adapt_vqe(_sector_engine(ctx, data, sector), sector,
                       threshold=params["threshold"], max_operators=params["max_operators"],
                       maxiter=params["maxiter"], spin_adapted=params["spin_adapted"])
    ctx.values[identifier] = {
        **inputs[0], "energy": float(result.energy), "device": device["id"],
        "method": _space_method(data, sector, "ADAPT-VQE", s2=float(result.s2),
                                n_operators=result.n_operators,
                                selected_operators=result.selected_operators,
                                n_gradient_evaluations=result.n_gradient_evaluations),
        "implementation": "vqe.adapt/chemflow-sector",
        "versions": {"scipy": _version("scipy")},
        "validity_flags": ["Exact statevector emulation: verification only, not deployable",
                           "Variational upper bound within the signed Hamiltonian"],
    }


@implements("eom.ip", "eom.ea")
def _op_eom(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.domains.properties import eom_charge_energy

    device = _cpu(node, "EOM-CCSD")
    molecule = _molecule_identity(inputs[0])
    kind = "ionization_energy" if operation == "eom.ip" else "electron_affinity"
    estimate = eom_charge_energy(ctx.store, molecule, {"basis": molecule["basis"]}, kind, 0.5)
    ctx.values[identifier] = {
        "value": float(estimate.value.to("hartree").value), "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": kind, "molecule": molecule, "final_spin": 0.5},
            "method": {"family": "EOM-IP-CCSD" if operation == "eom.ip" else "EOM-EA-CCSD",
                       "basis": molecule["basis"], "reference": "RHF"},
            "implementation": f"{operation}/pyscf-eom-ccsd", "device": device["id"],
            "versions": {"pyscf": pyscf.__version__},
            "validity_flags": ["Vertical, at the given geometry",
                               "EOM-CCSD approximation; not an experimental value"],
        },
    }


@implements("neb")
def _op_neb(ctx, node, identifier, operation, inputs):
    from chemflow.domains.paths import neb_path
    from chemflow.types.model import ElectronicModel

    device = _cpu(node, "NEB")
    reactant, product = (_molecule_identity(item) for item in inputs)
    params = node["params"]
    level = {"method": params["method"]}
    if params["functional"] is not None:
        level["functional"] = params["functional"]
    result = neb_path(ctx.store, reactant, product, ElectronicModel(basis=reactant["basis"]),
                      level, images=params["images"], interpolation=params["interpolation"],
                      optimizer=params["optimizer"], fmax=params["fmax"],
                      maxsteps=params["maxsteps"])
    images = [_molecule_identity({**image, "basis": reactant["basis"]})
              for image in result["images"]]
    climbing = result["climbing_image"]
    ctx.values[identifier] = {
        "ts": images[climbing], "endpoints": [images[0], images[-1]],
        "images": images, "energies_hartree": result["energies_hartree"],
        "climbing_image": climbing,
        "provenance": {
            "question": {"property": "minimum_energy_path", "reactant": reactant,
                         "product": product},
            "method": {"family": "climbing-image NEB", "level": level, **{
                key: params[key] for key in ("images", "interpolation", "optimizer", "fmax")}},
            "implementation": "neb/ase-climbing-image", "device": device["id"],
            "versions": {"ase": _version("ase")},
            "images": images, "energies_hartree": result["energies_hartree"],
            "climbing_image": climbing,
            "barrier_hartree": result["energies_hartree"][climbing] -
            result["energies_hartree"][0],
            "validity_flags": ["Discrete band: the climbing image approximates the saddle"],
        },
    }


MACE_FLAG = ("Machine-learned potential: a model change with respect to electronic-structure "
             "energies")


def _mace_versions():
    return {"mace-torch": _version("mace-torch"), "torch": _version("torch")}


@implements("mace.energy")
def _op_mace_energy(ctx, node, identifier, operation, inputs):
    from chemflow.domains.baselines import potential_energy

    device = _cpu(node, "MACE")
    molecule = _molecule_identity(inputs[0])
    computed = potential_energy(ctx.store, molecule, "mace-mp-0-small")
    ctx.values[identifier] = {
        "value": float(computed.value), "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": "potential_energy", "atoms": molecule["atoms"],
                         "unit": molecule["unit"]},
            "method": {"family": "MACE-MP-0 small",
                       "model_sha256": computed.diagnostics.get("model_sha256")},
            "implementation": "mace.energy/mace-mp-0-small", "device": device["id"],
            "versions": _mace_versions(),
            "validity_flags": [MACE_FLAG],
        },
    }


@implements("mace.optimize")
def _op_mace_optimize(ctx, node, identifier, operation, inputs):
    from chemflow.domains.baselines import geometry_optimize

    device = _cpu(node, "MACE")
    molecule = _molecule_identity(inputs[0])
    geometry = geometry_optimize(molecule, "mace-mp-0-small", "minimum")
    ctx.values[identifier] = {
        **_molecule_identity({**geometry, "basis": molecule["basis"]}),
        "provenance": {
            "question": {"property": "optimized_geometry", "starting_molecule": molecule},
            "method": {"family": "MACE-MP-0 small", "kind": "minimum"},
            "implementation": "mace.optimize/ase-bfgs", "device": device["id"],
            "versions": _mace_versions(),
        },
    }


@implements("drug.dock")
def _op_drug_dock(ctx, node, identifier, operation, inputs):
    from chemflow.domains.drug import dock, ligand_library

    device = _cpu(node, "Docking")
    params = node["params"]
    library = ligand_library(None, {"ligand": params["smiles"]})
    score = dock(ctx.store, library, params["receptor_pdbqt"], params["center"],
                 params["box"], seed=params["seed"])
    ctx.values[identifier] = {
        "value": float(score.value.value), "unit": "kcal/mol", "variance": 0.0,
        "provenance": {
            "question": {"property": "docking_score", "ligand": library.ligands["ligand"],
                         "center": params["center"], "box": params["box"]},
            "method": {"family": "AutoDock Vina", "scoring": "vina", "seed": params["seed"]},
            "implementation": "drug.dock/autodock-vina", "device": device["id"],
            "versions": {"vina": _version("vina"), "meeko": _version("meeko")},
            "validity_flags": ["Empirical scoring function of a seeded pose search",
                               "Not a binding free energy"],
        },
    }


@implements("drug.ligands")
def _op_drug_ligands(ctx, node, identifier, operation, inputs):
    from chemflow.domains import drug

    device = _cpu(node, "Ligand name resolution")
    params = node["params"]
    fetch = drug.fetch_pubchem
    if not params["network"]:
        def offline(name):
            raise OSError("network access was not declared (drug.ligands(..., network=True))")

        drug.fetch_pubchem = offline
    try:
        library = drug.ligands_from_names(ctx.store, list(params["names"]))
    finally:
        drug.fetch_pubchem = fetch
    ctx.values[identifier] = {
        "data": {"ligands": library.ligands},
        "provenance": {
            "question": {"property": "ligand_structures", "names": params["names"]},
            "method": {"family": "PubChem name resolution", "network": params["network"]},
            "implementation": "drug.ligands/pubchem-cache", "device": device["id"],
            "versions": {},
            "validity_flags": ["Structures as recorded by PubChem, checked by InChIKey"],
        },
    }


@implements("binding.fragments")
def _op_binding_fragments(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.domains import binding

    device = _cpu(node, "Fragment QM binding")
    params = node["params"]
    receptor = params["receptor"]
    ligand = binding.ligand_structure(params["ligand"]["source"])
    protein = binding.protein_structure(receptor["pdb_text"], ph=params["ph"],
                                        chains=receptor.get("chains"))
    combined = binding.complex(protein, ligand)
    residues = (params["residues"] if params["residues"] is not None else
                [item["id"] for item in binding.pocket(combined, params["cutoff_A"])["residues"]])
    level = {"method": params["method"]}
    if params["functional"] is not None:
        level["functional"] = params["functional"]
    result = binding.fragment_interaction(combined, ligand, residues, level=level,
                                          basis=params["basis"],
                                          counterpoise=params["counterpoise"])
    ctx.values[identifier] = {
        "value": float(result["total"]), "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": "fragment_interaction_energy", "receptor": receptor,
                         "ligand": params["ligand"], "residues": residues},
            "method": {"family": f"Fragment QM ({params['method'].upper()})",
                       "basis": params["basis"], "counterpoise": params["counterpoise"],
                       "cutoff_A": params["cutoff_A"], "ph": params["ph"]},
            "implementation": "binding.fragments/chemflow-pyscf", "device": device["id"],
            "versions": {"pyscf": pyscf.__version__},
            "per_residue": result["per_residue"], "formal_charges": result["formal_charges"],
            "ligand_atoms": result["ligand_atoms"], "residue_atoms": result["residue_atoms"],
            "validity_flags": [result["approximation"]],
        },
    }


@implements("dynamics.spectral_subspace")
def _op_dynamics_spectral_subspace(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.domains.algorithms import (
        as_sector,
        extend_subspace,
        full_subspace,
        rayleigh_ritz,
        subspace_from_samples,
    )
    from chemflow.domains.dynamics import impurity_model, lehmann_from_spectra
    from chemflow.impls.prepare import prepare_sector_state
    from chemflow.impls.statevector import sample_configurations

    device = _cpu(node, "Subspace Lehmann spectra")
    params = node["params"]
    store = ctx.store
    sector = as_sector(params["sector"])
    data = impurity_model(store, **params["model"])
    if data.sector != sector:
        data = replace(data, sector=sector)
    sampled = params["subspace"] == "sampled"
    if sampled:
        prep = prepare_sector_state(data, sector.model_dump(), store, "lucj", "ccsd")
        samples = sample_configurations(prep, params["shots"], params["seed"], store=store)
        ground_space = subspace_from_samples(store, samples, sector)
    else:
        ground_space = full_subspace(store, sector)
    ground = rayleigh_ritz(store, data, ground_space, n_roots=1)
    spectra, sizes = {}, {"ground": len(ground_space.determinants)}
    for role, delta, operator in (("particle", 1, "create"), ("hole", -1, "annihilate")):
        counts = sector.model_dump(include={"n_alpha", "n_beta", "n_orb"})
        counts[f"n_{params['spin']}"] += delta
        if not 0 <= counts[f"n_{params['spin']}"] <= sector.n_orb:
            spectra[role] = None
            continue
        excited = as_sector(counts)
        space = (extend_subspace(store, ground_space, excited, operator, params["orbital"],
                                 params["spin"], closure=params["closure"])
                 if sampled else full_subspace(store, excited))
        sizes[role] = len(space.determinants)
        spectra[role] = rayleigh_ritz(store, data, space, n_roots=1) if space.determinants else None
    result = lehmann_from_spectra(store, ground, spectra["particle"], spectra["hole"],
                                  params["orbital"], params["spin"], params["omega"],
                                  params["eta"])
    ctx.values[identifier] = {
        "omega": list(result.omega), "spectral": list(result.spectral), "unit": result.unit,
        "relation": "approximate" if sampled else "equivalent",
        "diagnostics": {**result.diagnostics, "subspace_sizes": sizes},
        "provenance": {
            "question": {"property": "spectral_function", **{
                key: params[key] for key in ("model", "sector", "orbital", "spin", "omega",
                                             "eta")}},
            "method": {"family": "subspace Lehmann", "subspace": params["subspace"],
                       **({key: params[key] for key in ("shots", "seed", "closure")}
                          if sampled else {})},
            "implementation": "dynamics.spectral_subspace/chemflow-lehmann",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
        },
    }
