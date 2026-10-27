from typing import Any

import numpy as np

from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash
from chemflow.impls.adapt import adapt_vqe
from chemflow.impls.ansatz import uccsd_pool, uccsd_state
from chemflow.impls.fci import sector_spin_eigenstate
from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals, lucj_state
from chemflow.impls.pyscf_impls import active_from_scf, build_molecule, classical_energy, fci_pyscf
from chemflow.impls.sci import sci_pyscf
from chemflow.impls.sector import SectorEngine, hf_vector
from chemflow.impls.sqd import sqd_sector
from chemflow.impls.statevector import (
    encode,
    expectation,
    ideal_energy_projective_sampler,
    pauli_measurement_groups,
    pauli_shot_estimator,
    prepare_hf,
    sample_configurations,
)
from chemflow.impls.vqe import vqe
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule, Registry
from chemflow.relations import TargetRelation
from chemflow.runtime.result import ExecutionResult
from chemflow.types import ActiveSpaceSpec, BackendKind, CircuitSpec, ElectronicModel, StatePrep


def molecular_registry() -> Registry:
    registry = Registry()
    signatures = {
        "molecule": ({}, "molecule", []),
        "geometry": ({}, "geometry", []),
        "electronic_model": ({}, "model", []),
        "scf": ({"molecule": "molecule", "model": "model"}, "hamiltonian", []),
        "integrals": ({"scf": "hamiltonian"}, "hamiltonian", []),
        "active_space": ({"scf": "hamiltonian"}, "hamiltonian", []),
        "hamiltonian": ({"molecule": "molecule", "model": "model"}, "hamiltonian", []),
        "sector": ({}, "sector", []),
        "encode": ({"hamiltonian": "hamiltonian"}, "encoding", []),
        "prepare": ({"hamiltonian": "hamiltonian", "encoding": "encoding"}, "prep",
                    ["CF-E090:hamiltonian:encoding:hamiltonian_id"]),
        "prepare_state": ({"hamiltonian": "hamiltonian", "sector": "sector"}, "prep", []),
        "expectation": ({"operator": "encoding", "prep": "prep"}, "estimate",
                        ["CF-E010:operator:prep:hamiltonian_id",
                         "CF-E090:operator:prep:encoding_id"]),
        "sample": ({"operator": "encoding", "prep": "prep"}, "estimate",
                   ["CF-E010:operator:prep:hamiltonian_id",
                    "CF-E090:operator:prep:encoding_id"]),
        "sample_configurations": ({"prep": "prep"}, "samples", []),
        "ground_energy": ({"hamiltonian": "hamiltonian", "sector": "sector"},
                          "estimate", []),
        "energy_difference": ({"left": "estimate", "right": "estimate"}, "estimate",
                              ["CF-E011:left:right:model_protocol_id"]),
        "compare": ({"left": "estimate", "right": "estimate"}, "estimate",
                    ["CF-E011:left:right:model_protocol_id"]),
        "map": ({}, "list", []),
    }
    for name, (inputs, output, rules) in signatures.items():
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
                                       output=output, semantics=f"{name}({inputs}) -> {output}",
                                       context_rules=rules))

    parameters = {
        "molecule": {"atoms", "charge", "spin", "unit", "params"},
        "geometry": {"coordinates", "unit", "atom_order", "conformer_id"},
        "electronic_model": set(ElectronicModel.model_fields),
        "sector": {"n_alpha", "n_beta", "n_orb", "n_elec", "s", "total_spin"},
        "active_space": {"n_elec", "n_orb"}, "hamiltonian": {"active"},
        "prepare_state": {"ansatz", "init", "params", "seed"},
        "sample": {"shots", "seed"},
        "sample_configurations": {"shots", "seed", "readout_flip_p"},
        "ground_energy": {"s", "total_spin"},
        "vqe_statevector": {"s", "total_spin", "ansatz", "optimizer", "maxiter",
                            "spin_adapted", "layers", "initial", "measurement", "shots",
                            "seed", "n_starts", "init"},
        "adapt_statevector": {"s", "total_spin", "threshold", "max_operators", "maxiter",
                              "spin_adapted"},
        "sci_pyscf": {"s", "total_spin", "select_cutoff", "max_product_dimension",
                      "spin_completion"},
        "sqd_sector": {"s", "total_spin", "seed", "n_batches", "batch_size", "max_iter",
                       "carryover_thresh", "energy_tol", "occ_tol", "with_ext", "ext_keep",
                       "max_product_dimension", "spin_completion"},
        "energy_difference": {"model_change", "label"}, "compare": {"model_change", "label"},
        "map": {"items"},
    }

    def register(name: str, adapter: Any, backend: BackendKind = BackendKind.cpu_numeric,
                 identifier: str | None = None, implements: str | None = None,
                 effects: str = "pure") -> None:
        registry.register(ImplementationSpec(id=identifier or name, version="1", target=name,
                                             implements=implements, backend_kind=backend,
                                             execution_role=("emulator" if name ==
                                                 "sample_configurations" else
                                                 "verification_only" if backend ==
                                                 BackendKind.quantum_simulator else "classical"),
                                             emulates=(name if name ==
                                                       "sample_configurations" else None),
                                             adapter=adapter, effects=effects,
                                             params=frozenset(parameters.get(identifier,
                                                               parameters.get(name, set()))),
                                             resource_model=lambda **kwargs: {
                                                 "qpu_shots": 0, "qpu_s": 0,
                                                 "gpu_s": 0, "sim_shots": 0}))

    register("molecule", lambda store, atoms, charge=0, spin=0, unit="angstrom", **kwargs:
             {"atoms": atoms, "charge": charge, "spin": spin, "unit": unit})
    register("geometry", lambda store, **kwargs: kwargs)
    register("electronic_model", lambda store, **kwargs: ElectronicModel(**kwargs))
    register("sector", lambda store, **kwargs: kwargs)
    register("scf", lambda store, molecule, model, **kwargs:
             build_molecule(molecule["atoms"], model, store, molecule["unit"],
                            molecule["charge"], molecule["spin"]))
    register("integrals", lambda store, scf, **kwargs: scf.hamiltonian)
    register("active_space", lambda store, scf, n_elec, n_orb, **kwargs:
             active_from_scf(scf, n_elec, n_orb, store))
    register("hamiltonian", lambda store, molecule, model, active=None, **kwargs:
             build_molecule(molecule["atoms"], model, store, molecule.get("unit", "angstrom"),
                            molecule.get("charge", 0), molecule.get("spin", 0),
                            ActiveSpaceSpec(**active) if active else None))
    register("encode", lambda store, hamiltonian, **kwargs:
             encode(hamiltonian.hamiltonian, store))
    register("prepare", lambda store, hamiltonian, encoding, **kwargs:
             prepare_hf(hamiltonian.hamiltonian, encoding[0], store,
                        hamiltonian.sector.n_alpha, hamiltonian.sector.n_beta),
             BackendKind.quantum_simulator)
    register("prepare_state", lambda store, hamiltonian, sector, **kwargs:
             prepare_sector_state(hamiltonian, sector, store, **kwargs),
             BackendKind.quantum_simulator)
    register("prepare_state", lambda store, hamiltonian, sector, **kwargs:
             prepare_sector_state(hamiltonian, sector, store, materialize=False, **kwargs),
             identifier="prepare_circuit")
    register("expectation", lambda store, operator, prep, **kwargs:
             expectation(operator[1] if isinstance(operator, tuple) else operator, prep, store),
             BackendKind.quantum_simulator)
    for identifier, sampler in (("pauli_shot_estimator", pauli_shot_estimator),
                                ("ideal_energy_projective_sampler",
                                 ideal_energy_projective_sampler)):
        register("sample", _sample_adapter(sampler), BackendKind.quantum_simulator,
                 identifier=identifier, effects="random")
        registry.implementations[identifier].resource_model = _sample_resources(
            identifier == "pauli_shot_estimator")
    registry.implementations["sample"] = registry.implementations["pauli_shot_estimator"]
    register("sample_configurations", lambda store, prep, shots, seed,
             readout_flip_p=0.0, **kwargs: sample_configurations(
                 prep, shots, seed, readout_flip_p, store=store),
             BackendKind.quantum_simulator, effects="random")
    registry.implementations["sample_configurations"].resource_model = lambda shots=None, **kwargs: {
        "qpu_shots": 0, "qpu_s": 0, "gpu_s": 0, "sim_shots": shots}
    register("ground_energy", lambda store, hamiltonian, sector, **kwargs:
             ground_energy(hamiltonian, sector, store, "sector_diag_numpy", **kwargs),
             identifier="sector_diag_numpy", implements="fci")
    registry.implementations["ground_energy"] = registry.implementations["sector_diag_numpy"]
    registry.register(RefinementRule(target="ground_energy", template_id="fci",
                                     relation=TargetRelation.equivalent))
    for method in ("fci_pyscf", "hf", "mp2", "ccsd", "ccsd_t"):
        template = "fci" if method == "fci_pyscf" else method
        register("ground_energy", _ground_adapter(method),
                 identifier=method, implements=template)
        relation = (TargetRelation.variational_upper_bound if method == "hf" else
                    TargetRelation.approximate if method != "fci_pyscf" else
                    TargetRelation.equivalent)
        registry.register(RefinementRule(target="ground_energy", template_id=method,
                                         relation=relation))
    registry.register(RefinementRule(target="ground_energy", template_id="vqe",
                                     relation=TargetRelation.variational_upper_bound))
    registry.register(RefinementRule(target="ground_energy", template_id="vqe_sampled",
                                     relation=TargetRelation.stochastic_estimate))
    register("ground_energy", lambda store, hamiltonian, sector, **kwargs:
             variational_energy(hamiltonian, sector, store, **kwargs),
             BackendKind.quantum_simulator, identifier="vqe_statevector", implements="vqe")
    registry.register(RefinementRule(target="ground_energy", template_id="adapt_vqe",
                                     relation=TargetRelation.variational_upper_bound))
    register("ground_energy", lambda store, hamiltonian, sector, **kwargs:
             adaptive_energy(hamiltonian, sector, store, **kwargs),
             BackendKind.quantum_simulator, identifier="adapt_statevector",
             implements="adapt_vqe")
    registry.register(RefinementRule(target="ground_energy", template_id="sqd",
                                     relation=TargetRelation.variational_upper_bound,
                                     inputs={"samples": "samples"},
                                     context_rules=["CF-E010:hamiltonian:samples:hamiltonian_id"]))
    register("ground_energy", lambda store, hamiltonian, sector, samples, **kwargs:
             sqd_energy(hamiltonian, sector, store, samples, **kwargs),
             identifier="sqd_sector", implements="sqd")
    registry.register(RefinementRule(target="ground_energy", template_id="sci",
                                     relation=TargetRelation.variational_upper_bound))
    register("ground_energy", lambda store, hamiltonian, sector, **kwargs:
             sci_energy(hamiltonian, sector, store, **kwargs),
             identifier="sci_pyscf", implements="sci")
    register("energy_difference", lambda store, left, right, **kwargs:
             difference(left, right, kwargs.get("model_change", False)))
    register("compare", lambda store, left, right, **kwargs:
             difference(left, right, kwargs.get("model_change", False)))
    register("map", lambda store, items, **kwargs: list(items))
    return registry


def _ground_adapter(method):
    def adapter(store, hamiltonian, sector, **kwargs):
        energy, diagnostics = ground_energy(hamiltonian, sector, store, method, **kwargs)
        return energy, diagnostics, {"weakening": _spin_target_weakening(
            sector, kwargs, diagnostics["s2"]) if "s2" in diagnostics else None}
    return adapter


def _sample_adapter(sampler):
    def adapter(store, operator, prep, shots, seed):
        samples, energy, error = sampler(
            operator[1] if isinstance(operator, tuple) else operator, prep, store, shots, seed)
        return ExecutionResult((energy, error), cost={"sim_shots": samples.shots})
    return adapter


def _sample_resources(pauli):
    def resources(shots=None, operator=None, **kwargs):
        observable = operator[1] if isinstance(operator, tuple) else operator
        groups = (pauli_measurement_groups(observable)
                  if pauli and observable is not None else None if pauli else 1)
        return {"sim_shots": shots * groups if shots is not None and groups is not None else None,
                "qpu_shots": 0,
                "qpu_s": 0, "gpu_s": 0}
    return resources


def prepare_sector_state(data: Any, sector: dict, store: Any, ansatz: str = "hf",
                         init: str = "zero", params: list[float] | None = None,
                         seed: int = 0, *, materialize: bool = True) -> StatePrep:
    if ansatz not in ("hf", "uccsd", "lucj") or init not in ("zero", "ccsd", "random"):
        raise ValueError("Unsupported ansatz or initialization")
    if (sector["n_alpha"], sector["n_beta"], sector["n_orb"]) != (
            data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb):
        raise ValueError("Sector differs from Hamiltonian")
    if ansatz == "hf":
        state = hf_vector(data.sector) if materialize else None
        circuit = CircuitSpec(ansatz="hf", n_orb=data.sector.n_orb,
                              n_alpha=data.sector.n_alpha, n_beta=data.sector.n_beta)
    elif ansatz == "uccsd":
        if not materialize:
            raise ValueError("UCCSD preparation has no hardware circuit implementation")
        circuit = None
        _, generators = uccsd_pool(data.sector)
        values = np.asarray(params if params is not None else np.zeros(len(generators)), dtype=float)
        if init == "random" and params is None:
            values = np.random.default_rng(seed).normal(0, 0.12, len(generators))
        state = uccsd_state(data.sector, values, generators)
    else:
        values = LucjParams.zeros(data.sector.n_orb)
        if init == "ccsd":
            if hasattr(data, "mean_field"):
                data.mean_field(store)
            values = ccsd_seed_from_integrals(
                data.hamiltonian, data.sector, store,
                site_reference=data.mf if getattr(getattr(data, "space", None), "ordering", None)
                == "sites" else None) or values
        elif init == "random":
            values = LucjParams.random(data.sector.n_orb, np.random.default_rng(seed))
        if params is not None:
            values = LucjParams.from_vector(np.asarray(params), data.sector.n_orb)
        state = lucj_state(data.sector, [values]) if materialize else None
        circuit = CircuitSpec(ansatz="lucj", n_orb=data.sector.n_orb,
                              n_alpha=data.sector.n_alpha, n_beta=data.sector.n_beta,
                              layers=(tuple(float(item) for item in values.to_vector()),))
    return StatePrep(state_ref=store.put(state) if materialize else "", encoding_id="sector_state",
                     hamiltonian_id=data.hamiltonian.id, sector=data.sector,
                     recipe=f"{ansatz}:{init}:{content_hash(params)}", circuit=circuit)


def difference(left: Any, right: Any, model_change: bool = False) -> float:
    if left.kind != right.kind:
        raise AdapterError("CF-E002", None, "Quantity-kind mismatch", "Compare like kinds")
    if left.value.unit != right.value.unit:
        raise AdapterError("CF-E001", None, "Unit mismatch", "Convert explicitly")
    if (left.context.keys.get("model_protocol_id") !=
        right.context.keys.get("model_protocol_id") and not model_change):
        raise AdapterError("CF-E011", None, "Incompatible models", "Declare model_change")
    return left.value.value - right.value.value


def ground_energy(data: Any, sector: dict[str, Any], store: Any, method: str,
                  s: int | None = None, total_spin: int | None = None,
                  **kwargs: Any) -> tuple[float, dict[str, float]]:
    if ((sector.get("n_alpha"), sector.get("n_beta"), sector.get("n_orb")) !=
        (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb)):
        raise AdapterError("CF-E020", None, "Sector differs from Hamiltonian",
                           "Match particle counts")
    target = s if s is not None else total_spin
    target = sector.get("s", sector.get("total_spin", target)) if target is None else target
    if method == "sector_diag_numpy":
        energy, spin_squared, vector = sector_spin_eigenstate(
            data.hamiltonian, data.sector, store, target, return_wavefunction=True)
        return energy, {"s2": spin_squared, "state_ref": store.put(vector)}
    if method == "fci_pyscf":
        from chemflow.impls.sci import state_artifact

        energy, spin_squared, roots, vector = fci_pyscf(
            data, store, target, return_roots=True, return_wavefunction=True)
        return energy, {"s2": spin_squared, "n_roots_examined": roots,
                        "state_ref": state_artifact(vector, data.sector, store)}
    if target is not None and data.sector.n_alpha - data.sector.n_beta != 2 * target:
        raise AdapterError("CF-E020", None, "Approximate solver cannot constrain spin",
                           "Use a high-spin sector (M_S = S) or choose FCI")
    if hasattr(data, "mean_field"):
        data.mean_field(store)
    diagnostics = {}
    energy = classical_energy(data, method, store, diagnostics)
    return energy, diagnostics


def _spin_target_weakening(sector: dict[str, Any], options: dict[str, Any],
                           spin_squared: float) -> dict[str, Any] | None:
    target = options.get("s", options.get("total_spin", sector.get("s", sector.get("total_spin"))))
    if target is not None and abs(spin_squared - target * (target + 1)) > 1e-4:
        return {"relation": TargetRelation.approximate,
                "reason": "Spin expectation differs from requested S(S+1)"}
    return None


def _completion_weakening(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    if diagnostics.get("spin_completion_reason"):
        return {"relation": TargetRelation.approximate,
                "reason": diagnostics["spin_completion_reason"]}
    return None


def variational_energy(data: Any, sector: dict[str, Any], store: Any,
                       **kwargs: Any) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if ((sector.get("n_alpha"), sector.get("n_beta"), sector.get("n_orb")) !=
        (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb)):
        raise AdapterError("CF-E020", None, "Sector differs from Hamiltonian",
                           "Match particle counts")
    result = vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector,
                 hamiltonian=data.hamiltonian, store=store, **kwargs)
    diagnostics = {"s2": result.s2, "n_iter": result.n_iter,
                   "engine": "SectorEngine", "state_ref": store.put(result.state),
                   "evaluation": kwargs.get("measurement", "exact"),
                   "n_energy_evals": result.n_energy_evals,
                   "converged": float(result.converged),
                   "start_energies": result.start_energies}
    weakening = _spin_target_weakening(sector, kwargs, result.s2)
    return result.energy, diagnostics, {"weakening": weakening, "stat_err": result.stat_err,
                                        "sim_shots": result.sim_shots}


def adaptive_energy(data: Any, sector: dict[str, Any], store: Any,
                    **kwargs: Any) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if ((sector.get("n_alpha"), sector.get("n_beta"), sector.get("n_orb")) !=
        (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb)):
        raise AdapterError("CF-E020", None, "Sector differs from Hamiltonian",
                           "Match particle counts")
    result = adapt_vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector,
                       **kwargs)
    return (result.energy, {"s2": result.s2,
                            "engine": "SectorEngine", "state_ref": store.put(result.state),
                            "evaluation": "exact",
                            "n_gradient_evaluations": result.n_gradient_evaluations,
                            "n_operators": result.n_operators,
                            "selected_operators": result.selected_operators},
            {"weakening": _spin_target_weakening(sector, kwargs, result.s2)})


def sqd_energy(data: Any, sector: dict[str, Any], store: Any, samples: Any,
               **kwargs: Any) -> tuple[float, dict[str, Any], dict[str, Any]]:
    from chemflow.impls.sci import state_artifact

    spin = kwargs.pop("s", kwargs.pop("total_spin", sector.get("s", sector.get("total_spin"))))
    energy, diagnostics, state = sqd_sector(data, data.sector, store, samples,
                                     total_spin=spin, return_wavefunction=True, **kwargs)
    diagnostics["state_ref"] = state_artifact(state, data.sector, store)
    return energy, diagnostics, {"weakening": _completion_weakening(diagnostics) or
            _spin_target_weakening(sector, {"s": spin}, diagnostics["s2"])}


def sci_energy(data: Any, sector: dict[str, Any], store: Any,
               **kwargs: Any) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if ((sector.get("n_alpha"), sector.get("n_beta"), sector.get("n_orb")) !=
        (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb)):
        raise AdapterError("CF-E020", None, "Sector differs from Hamiltonian",
                           "Match particle counts")
    spin = kwargs.pop("s", kwargs.pop("total_spin", sector.get("s", sector.get("total_spin"))))
    energy, diagnostics = sci_pyscf(data, store, total_spin=spin, **kwargs)
    return energy, diagnostics, {"weakening": _completion_weakening(diagnostics) or
            _spin_target_weakening(sector, {"s": spin}, diagnostics["s2"])}
