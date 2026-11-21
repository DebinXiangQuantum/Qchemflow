"""QM/MM energies, embedded wave-function theory, ligand screens and retrosynthesis."""

from qcl._impls import implements


def _cpu(node, what):
    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"{what} currently requires a CPU")
    return device


@implements("qmmm.energy")
def _op_qmmm_energy(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.domains.qmmm import mm_system, qmmm_energy

    device = _cpu(node, "QM/MM")
    params = node["params"]
    system = mm_system(ctx.store, params["pdb"], params["forcefields"], params["nonbonded"])
    cas = tuple(params["cas"]) if params["cas"] is not None else None
    computed = qmmm_energy(ctx.store, system, params["positions_nm"], params["qm_atoms"],
                           params["basis"], method=params["method"], cas=cas,
                           charge=params["charge"], spin=params["spin"])
    estimate = computed.value
    diagnostics = estimate.diagnostics
    method = "RHF" if params["method"] == "rhf" else f"CASCI{list(cas)}/FCI"
    ctx.values[identifier] = {
        "value": float(estimate.value.to("hartree").value), "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": "qmmm_total_energy", "system_id": system.id,
                         "snapshot_hash": estimate.context.keys["snapshot_hash"],
                         "qm_atoms": params["qm_atoms"], "embedding": "electrostatic"},
            "method": {"family": f"QM/MM-{method}", "basis": params["basis"],
                       "forcefields": params["forcefields"],
                       "nonbonded": params["nonbonded"],
                       "charge": params["charge"], "spin": params["spin"]},
            "implementation": "qmmm.energy/chemflow-openmm-pyscf",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
            "components": {key: diagnostics[key] for key in ("qm", "mm", "total")
                           if key in diagnostics},
            "validity_flags": ["Electrostatic embedding; no link atoms or QM forces",
                               "Exact only within the stated QM/MM model"],
        },
    }


@implements("embedding.ewf")
def _op_embedding_ewf(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.domains.embedding import ewf_embedding

    device = _cpu(node, "EWF embedding")
    if inputs[1].get("data", inputs[0]["data"]) is not inputs[0]["data"]:
        raise ValueError("EWF operator and sector refer to different active spaces")
    data = inputs[0]["data"]
    sector = inputs[1].get("sector", data.sector)
    params = node["params"]
    computed = ewf_embedding(ctx.store, data, sector.model_dump(), bath=params["bath"],
                             bath_tol=params["bath_tol"], bno_eta=params["bno_eta"],
                             solver=params["solver"])
    energy, diagnostics = computed.value
    exact = params["scheme"] == "full/fci"
    ctx.values[identifier] = {
        "value": float(energy), "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"property": "ground_energy", "molecule": inputs[0].get("molecule"),
                         "sector": {"n_alpha": sector.n_alpha, "n_beta": sector.n_beta,
                                    "n_orb": sector.n_orb}},
            "method": {"family": f"EWF({params['bath']}/{params['solver'].upper()})",
                       "basis": data.model.basis if data.model is not None else None,
                       "fragments": "atoms", "bath_tol": params["bath_tol"],
                       "bno_eta": params["bno_eta"]},
            "implementation": "embedding.ewf/chemflow-pyscf",
            "device": device["id"], "versions": {"pyscf": pyscf.__version__},
            "diagnostics": diagnostics,
            "validity_flags": (["Full bath with FCI fragments reproduces full FCI"] if exact
                               else ["Truncated bath or approximate fragment solver"]),
        },
    }


def _library(params):
    from chemflow.domains.drug import ligand_library

    return ligand_library(None, params["ligands"])


def _rdkit_version():
    import rdkit

    return {"rdkit": rdkit.__version__}


@implements("drug.descriptors")
def _op_drug_descriptors(ctx, node, identifier, operation, inputs):
    from chemflow.domains.drug import descriptors

    device = _cpu(node, "Ligand descriptors")
    library = _library(node["params"])
    ctx.values[identifier] = {
        "data": {"ligands": library.ligands, "descriptors": descriptors(None, library)},
        "provenance": {
            "question": {"property": "ligand_descriptors",
                         "ligands": sorted(node["params"]["ligands"])},
            "method": {"family": "RDKit descriptors (MW, logP, HBD, HBA, TPSA)"},
            "implementation": "drug.descriptors/chemflow-rdkit",
            "device": device["id"], "versions": _rdkit_version(),
            "validity_flags": ["Computed descriptors, not measured properties"],
        },
    }


@implements("drug.filter")
def _op_drug_filter(ctx, node, identifier, operation, inputs):
    from chemflow.domains.drug import filter_ligands

    device = _cpu(node, "Ligand filtering")
    params = node["params"]
    library = _library(params)
    result = filter_ligands(None, library, params["rules"])
    ctx.values[identifier] = {
        "data": {"passed": list(result.passed), "flagged": dict(result.flagged),
                 "ligands": library.ligands},
        "provenance": {
            "question": {"property": "ligand_alerts", "ligands": sorted(params["ligands"]),
                         "rules": params["rules"]},
            "method": {"family": "RDKit " + "/".join(params["rules"]) + " screen"},
            "implementation": "drug.filter/chemflow-rdkit",
            "device": device["id"], "versions": _rdkit_version(),
            "validity_flags": ["Rule-based alerts; not assay outcomes or efficacy"],
        },
    }


@implements("synthesis.routes")
def _op_synthesis_routes(ctx, node, identifier, operation, inputs):
    from chemflow.domains.synthesis import search_routes

    device = _cpu(node, "Retrosynthesis")
    params = node["params"]
    result = search_routes(ctx.store, params["target"], stock=params["stock"],
                           iterations=params["iterations"],
                           time_limit_s=params["time_limit_s"])
    ctx.values[identifier] = {
        "data": {"solved": result.solved, "routes": result.routes,
                 "inventory": result.inventory},
        "provenance": {
            "question": {"property": "synthesis_routes", "target": params["target"],
                         "stock": params["stock"]},
            "method": {"family": "AiZynthFinder MCTS", "iterations": params["iterations"],
                       "time_limit_s": params["time_limit_s"]},
            "implementation": "synthesis.routes/aizynthfinder-4.4.1",
            "device": device["id"], "versions": dict(result.provenance.software_versions),
            "validity_flags": ["Template-based proposals; routes are not validated"],
        },
    }
