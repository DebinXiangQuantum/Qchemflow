"""Full-space DFT, declared solvent/orbital models, and vertical IP/EA."""

from math import comb

import numpy as np

from chemflow import build
from chemflow.diagnostics import AdapterError, Diagnostic
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, RefinementRule
from chemflow.relations import TargetRelation, combine_linear
from chemflow.types import BackendKind

from ._adapters import estimate, register_adapter

DETERMINANT_CAP = 100_000


def refuse(message):
    raise AdapterError("CF-E080", None, message, "Use a supported full-space model or smaller basis")


def molecule_for(molecule, electronic_model):
    from pyscf import gto

    if (set(electronic_model) - {"basis", "reference", "hamiltonian_class"} or
            electronic_model.get("hamiltonian_class", "nonrel_bo") != "nonrel_bo"):
        refuse("Properties require an all-electron nonrelativistic full-space model")
    return gto.M(atom=molecule["atoms"], basis=electronic_model["basis"],
                 charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                 unit=molecule.get("unit", "angstrom"), verbose=0)


def dispersion_energy(mol, functional, dispersion):
    if dispersion is None:
        return 0.0
    numbers, positions = np.asarray(mol.atom_charges()), mol.atom_coords()
    if dispersion == "d3bj":
        from dftd3.interface import DispersionModel, RationalDampingParam

        model = DispersionModel(numbers, positions)
        damping = RationalDampingParam(method=functional)
    elif dispersion == "d4":
        from dftd4.interface import DampingParam, DispersionModel

        model = DispersionModel(numbers, positions, charge=float(mol.charge))
        damping = DampingParam(method=functional)
    else:
        refuse("Unsupported dispersion; choose d3bj or d4")
    return float(model.get_dispersion(damping, grad=False)["energy"])


def scf_model(mol, level, solvent=None):
    from pyscf import dft, scf

    if level["method"] == "dft":
        mean_field = dft.UKS(mol) if mol.spin else dft.RKS(mol)
        mean_field.xc = level["functional"]
    elif level["method"] == "hf":
        mean_field = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    else:
        refuse("SCF orbitals require HF or DFT")
    try:
        if solvent:
            if solvent["model"] == "pcm":
                mean_field = mean_field.PCM()
                mean_field.with_solvent.method = "IEF-PCM"
                mean_field.with_solvent.eps = solvent["eps"]
            elif solvent["model"] == "smd":
                mean_field = mean_field.SMD()
                mean_field.with_solvent.solvent = solvent["name"]
            else:
                refuse("HF/DFT solvation supports PCM or SMD")
        mean_field.kernel()
    except RuntimeError as exc:
        if solvent and solvent["model"] == "smd":
            raise AdapterError("CF-E080", None, f"SMD unavailable in this PySCF build: {exc}",
                               "Use PCM (HF/DFT) or ALPB (GFN2) instead") from exc
        raise
    if not mean_field.converged:
        refuse("SCF did not converge")
    return mean_field


def dft_energy(store, hamiltonian, sector, functional, dispersion=None, s=None):
    if getattr(hamiltonian, "active", None) is not None:
        refuse("DFT does not support active-space Hamiltonians")
    if not hasattr(hamiltonian, "mol"):
        refuse("DFT requires a molecular Hamiltonian")
    if any(sector[key] != getattr(hamiltonian.sector, key)
           for key in ("n_alpha", "n_beta", "n_orb")):
        refuse("DFT sector differs from the molecular Hamiltonian")
    if s is not None and 2 * s != hamiltonian.mol.spin:
        refuse("DFT cannot constrain the requested total spin")
    mean_field = scf_model(hamiltonian.mol, {"method": "dft", "functional": functional})
    correction = dispersion_energy(hamiltonian.mol, functional, dispersion)
    return float(mean_field.e_tot + correction), {"functional": functional,
        "dispersion": dispersion, "dispersion_hartree": correction,
        "model_defined": True}


def _result(value, kind, params, diagnostics=None, relation="equivalent"):
    if not np.isfinite(value):
        refuse("Property calculation returned a nonfinite value")
    return estimate(float(value), kind, relation, {"hamiltonian_id": content_hash(params)},
                    kind, diagnostics)


def solvation_free_energy(store, molecule, electronic_model, level, solvent):
    mol = molecule_for(molecule, electronic_model)
    if level["method"] == "gfn2":
        from tblite.interface import Calculator

        energies = []
        for medium in (None, solvent):
            calculator = Calculator("GFN2-xTB", np.asarray(mol.atom_charges()), mol.atom_coords(),
                                    charge=mol.charge, uhf=mol.spin)
            calculator.set("verbosity", 0)
            if medium:
                calculator.add(f"{medium['model']}-solvation", medium["name"])
            energies.append(float(calculator.singlepoint().get("energy")))
    else:
        correction = dispersion_energy(mol, level.get("functional"), level.get("dispersion"))
        energies = [scf_model(mol, level, medium).e_tot + correction
                    for medium in (None, solvent)]
    params = {"molecule": molecule, "electronic_model": electronic_model,
              "level": level, "solvent": solvent}
    return _result(energies[1] - energies[0], "solvation_free_energy", params,
                   {"declared_model": params, "electrostatic_only": solvent["model"] == "pcm",
                    "gas_hartree": float(energies[0]), "solvent_hartree": float(energies[1])})


def frontier_orbitals(store, molecule, electronic_model, level, quantity, spin="alpha"):
    mol = molecule_for(molecule, electronic_model)
    mean_field = scf_model(mol, level)
    energies, occupations = np.asarray(mean_field.mo_energy), np.asarray(mean_field.mo_occ)
    if energies.ndim == 2:
        channel = 0 if spin == "alpha" else 1
        energies, occupations = energies[channel], occupations[channel]
    occupied, virtual = energies[occupations > 0], energies[occupations == 0]
    if not len(occupied) or not len(virtual):
        refuse("Frontier orbitals require occupied and virtual orbitals in the selected spin")
    homo, lumo = float(max(occupied)), float(min(virtual))
    params = {"molecule": molecule, "electronic_model": electronic_model, "level": level,
              "quantity": quantity, "spin": spin}
    return _result({"homo": homo, "lumo": lumo, "gap": lumo - homo}[quantity],
                   "frontier_orbitals", params,
                   {"declared_model": params, "homo": homo, "lumo": lumo})


def charge_energy_difference(store, initial, final, target, coefficients):
    values = {"initial": initial, "final": final}
    relation = combine_linear([(coefficients[key], value.relation)
                               for key, value in values.items()])
    return _result(sum(coefficients[key] * value.value.to("hartree").value
                       for key, value in values.items()), target["kind"],
                   target, relation=relation)


def eom_charge_energy(store, molecule, electronic_model, kind, final_spin):
    from pyscf import cc, scf
    from pyscf.cc import eom_rccsd

    mol = molecule_for(molecule, electronic_model)
    if mol.spin or final_spin != 0.5:
        refuse("Restricted EOM-IP/EA-CCSD requires a closed-shell initial state and final S=1/2")
    reference = scf.RHF(mol).run()
    solver = cc.CCSD(reference).run()
    if not reference.converged or not solver.converged:
        refuse("EOM reference did not converge")
    response = eom_rccsd.EOMIP(solver) if kind == "ionization_energy" else eom_rccsd.EOMEA(solver)
    energy = response.kernel(nroots=1)[0]
    if not np.all(response.converged):
        refuse("EOM-CCSD did not converge")
    return _result(energy if kind == "ionization_energy" else -energy, kind,
                   {"molecule": molecule, "electronic_model": electronic_model, "kind": kind,
                    "final_spin": final_spin, "operator": "charge_energy_difference"},
                   relation="approximate")


def charge_sectors(target):
    initial = dict(target.molecule)
    final = {**initial, "charge": initial.get("charge", 0) +
             (1 if target.kind == "ionization_energy" else -1),
             "spin": int(2 * target.final_spin)}
    return initial, final


def base_plan(record, active=None):
    if active is not None:
        refuse("Property targets require the full orbital space")
    target = record.spec.target
    params = target.model_dump(exclude={"kind", "operator"})
    with build.program("property-screen") as builder:
        if target.kind in ("solvation_free_energy", "frontier_orbitals"):
            reference = builder.node(target.operator, **params)
            controls = []
        else:
            sectors = []
            for molecule in charge_sectors(target):
                mol = molecule_for(molecule, target.electronic_model)
                dimension = comb(mol.nao_nr(), mol.nelec[0]) * comb(mol.nao_nr(), mol.nelec[1])
                if dimension > DETERMINANT_CAP:
                    refuse(f"FCI sector has {dimension} determinants, exceeding cap "
                           f"{DETERMINANT_CAP}; reduce the basis or molecule")
                ham = builder.hamiltonian(builder.molecule(**molecule),
                                          builder.electronic_model(**target.electronic_model))
                sector = builder.sector(n_orb=mol.nao_nr(), n_alpha=mol.nelec[0],
                                        n_beta=mol.nelec[1])
                sectors.append((ham, sector, mol.spin / 2))
            controls = []
            for method in ("hf", "ccsd_t", "fci"):
                energies = []
                for ham, sector, spin in sectors:
                    energy = builder.ground_energy(ham, sector, s=spin)
                    builder.refine(energy, method)
                    builder.bind(energy, "fci_pyscf" if method == "fci" else method)
                    energies.append(energy)
                sign = 1 if target.kind == "ionization_energy" else -1
                result = builder.node("charge_energy_difference", initial=energies[0],
                    final=energies[1], target=target.model_dump(),
                    coefficients={"initial": -sign, "final": sign})
                if method == "fci":
                    reference = result
                else:
                    controls.append(result.id)
            eom = builder.node("eom_charge_energy", **params, kind=target.kind)
            next(node for node in builder.ir.nodes if node.id == eom.id).metadata[
                "optional_control"] = True
            controls.append(eom.id)
        builder.ir.references = [reference.id]
        builder.ir.targets = [*controls, reference.id]
        return builder.to_ir(), controls


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    target = record.spec.target
    nodes = {node.id: node for node in plan.nodes}
    issues = []
    for identifier in set(plan.targets + plan.references):
        node = nodes.get(identifier)
        valid = node is not None
        if target.kind in ("solvation_free_energy", "frontier_orbitals"):
            valid = valid and node.op == target.operator and node.params == target.model_dump(
                exclude={"kind", "operator"})
        elif valid and node.op == "eom_charge_energy":
            valid = node.params == target.model_dump(exclude={"operator"})
        elif valid and node.op == "charge_energy_difference":
            sign = 1 if target.kind == "ionization_energy" else -1
            valid = node.params == {"target": target.model_dump(),
                                    "coefficients": {"initial": -sign, "final": sign}}
            valid = valid and set(node.inputs) == {"initial", "final"}
            for port, molecule in zip(("initial", "final"), charge_sectors(target), strict=True):
                energy = nodes.get(node.inputs.get(port))
                ham = nodes.get(energy.inputs.get("hamiltonian")) if energy else None
                source = nodes.get(ham.inputs.get("molecule")) if ham else None
                model = nodes.get(ham.inputs.get("model")) if ham else None
                defaults = {"charge": 0, "spin": 0, "unit": "angstrom"}
                sector = nodes.get(energy.inputs.get("sector")) if energy else None
                mol = molecule_for(molecule, target.electronic_model)
                valid = valid and (energy is not None and energy.op == "ground_energy" and
                    energy.params.get("s") == mol.spin / 2 and sector is not None and
                    sector.op == "sector" and sector.params == {"n_orb": mol.nao_nr(),
                        "n_alpha": mol.nelec[0], "n_beta": mol.nelec[1]} and
                    ham is not None and ham.op == "hamiltonian" and not ham.params.get("active")
                    and source is not None and {**defaults, **source.params} ==
                    {**defaults, **molecule} and model is not None and
                    model.params == target.electronic_model)
        else:
            valid = False
        if not valid:
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                                      message="Property target identity mismatch",
                                      fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
    return issues


def context_check(node, inputs, context, nodes):
    if node.op in ("solvation_free_energy", "frontier_orbitals", "charge_energy_difference",
                   "eom_charge_energy"):
        kind = node.params.get("target", {}).get("kind", node.params.get("kind", node.op))
        params = node.params
        if node.op == "charge_energy_difference":
            params = node.params["target"]
        elif node.op == "eom_charge_energy":
            params = {**params, "operator": "charge_energy_difference"}
        context.update(kind=kind, unit="hartree", hamiltonian_id=content_hash(params))
    return []


def resources(**kwargs):
    return {"qpu_shots": 0, "qpu_s": 0, "gpu_s": 0, "sim_shots": 0}


def register(registry):
    registry.register(RefinementRule(target="ground_energy", template_id="dft",
                                     relation=TargetRelation.approximate))
    registry.register(ImplementationSpec(id="dft_pyscf", version="1", target="ground_energy",
        implements="dft", adapter=dft_energy, backend_kind=BackendKind.cpu_numeric,
        execution_role="classical", params=frozenset({"functional", "dispersion", "s"}),
        resource_model=resources))
    for name, adapter, inputs, parameters, description in (
        ("solvation_free_energy", solvation_free_energy, {},
         ("molecule", "electronic_model", "level", "solvent"),
         "Declared solvent model E(solvent)-E(gas); IEF-PCM is electrostatic only; GFN2 ALPB/GBSA"),
        ("frontier_orbitals", frontier_orbitals, {},
         ("molecule", "electronic_model", "level", "quantity", "spin"),
         "Model-defined converged HF/DFT spin-channel HOMO, LUMO or orbital gap, not physical IP/EA"),
        ("charge_energy_difference", charge_energy_difference,
         {"initial": "estimate", "final": "estimate"}, ("target", "coefficients"),
         "Vertical IP=E(N-1)-E(N), EA=E(N)-E(N+1); signed physical total energies"),
        ("eom_charge_energy", eom_charge_energy, {},
         ("molecule", "electronic_model", "kind", "final_spin"),
         "Restricted EOM-IP/EA-CCSD vertical charged excitation; no ordering law"),
    ):
        register_adapter(registry, name, adapter, inputs, "estimate", parameters, description)
        registry.implementations[name].execution_role = "classical"
        registry.implementations[name].resource_model = resources
        spec = registry.operators[f"chemflow.{name}@1"]
        parameter_types = {"molecule": "molecule specification", "electronic_model": "model",
            "level": "PropertyLevel", "solvent": "SolventModel", "quantity": "homo|lumo|gap",
            "spin": "alpha|beta", "target": "IonizationTarget", "coefficients": "dict[str,float]",
            "kind": "ionization_energy|electron_affinity", "final_spin": "nonnegative half-integer"}
        registry.operators[f"chemflow.{name}@1"] = spec.model_copy(update={
            "parameters": {key: parameter_types[key] for key in parameters}})
        registry.register(RefinementRule(target=name, template_id=name,
            relation=TargetRelation.equivalent if name in ("solvation_free_energy",
                "frontier_orbitals") else TargetRelation.approximate))
    registry.context_checks.append(context_check)
