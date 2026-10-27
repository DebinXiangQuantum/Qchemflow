"""Signed-task bridge for ChemFlow agents that author QCL workflows."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Protocol

from chemflow.agent.task import TaskRecord
from chemflow.diagnostics import AdapterError
from chemflow.types.model import ElectronicModel


class TaskVerifier(Protocol):
    def verify(self, record: TaskRecord) -> None: ...


@dataclass(frozen=True)
class QclReview:
    plan: Any
    plan_hash: str
    explanation: dict[str, Any]
    estimate: dict[str, Any]
    record_hash: str
    reference_plan: Any | None = None
    reference_hash: str | None = None
    output_hash: str | None = None
    reference_output_hash: str | None = None


@dataclass(frozen=True)
class QclCandidateOutcome:
    name: str
    plan_hash: str
    passed: bool
    gates: tuple[dict[str, Any], ...]
    wall_s: float
    compute_s: float
    cpu_core_s: float
    gpu_s: float
    qpu_shots: int
    qpu_cost: float | None
    value: float | None
    unit: str | None


@dataclass(frozen=True)
class QclSelection:
    decision: str
    candidate: str | None
    result: Any | None
    outcomes: tuple[QclCandidateOutcome, ...]


def _gpu_node_ids(plan: Any) -> set[int]:
    return {node["id"] for node in plan.graph_json()["nodes"]
            if node["device"] is not None and
            (node["device"]["kind"] == "gpu" or
             node["device"]["kind"] == "qpu" and
             (node["device"].get("simulator_host") or "").startswith("gpu"))}


def _output_signature(plan: Any) -> str:
    from qcl._frontend import _Value

    def shape(value: Any) -> Any:
        if isinstance(value, _Value):
            return {"node": value.node_id, "kind": value.kind}
        if isinstance(value, (tuple, list)):
            return [shape(item) for item in value]
        if isinstance(value, dict):
            return {key: shape(item) for key, item in value.items()}
        if is_dataclass(value):
            return {"type": type(value).__name__, "value": asdict(value)}
        return value

    encoded = json.dumps(shape(plan._output), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reserve_cloud_cost(ledger: Any, signed_budget: dict[str, float],
                        estimate: dict[str, Any]) -> None:
    if "qpu_cost" not in signed_budget:
        return
    from chemflow.runtime.budget import BudgetExceeded

    cost = estimate["qpu_cost"]
    spent = ledger.spent.get("qpu_cost", 0)
    limit = signed_budget["qpu_cost"]
    if (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0 or
            cost > 0 and estimate["currency"] is None or
            type(spent) not in (int, float) or not math.isfinite(spent) or
            spent + cost > limit):
        raise BudgetExceeded("CF-E070: cumulative cloud QPU cost budget exceeded")
    ledger.spent["qpu_cost"] = spent + cost


def _check_screen_currency(reviews: Mapping[str, QclReview]) -> None:
    currencies = {review.estimate["currency"] for review in reviews.values()
                  if review.estimate["currency"] is not None}
    if len(currencies) > 1:
        raise AdapterError("CF-E070", None,
                           "QCL screen cannot compare cloud costs in different currencies",
                           "Screen one billing currency at a time")


def _agent_error_budget(review: QclReview, result: Any) -> dict[str, Any]:
    from chemflow.units import Q

    budget = review.plan.error_budget()
    gates = getattr(result, "provenance", {}).get("signed_gates", [])
    observations = [
        {"kind": gate["kind"], "reference_plan_hash": gate["reference_plan_hash"],
         "task_record_hash": review.record_hash, "absolute_error": gate["error"],
         "unit": gate["unit"], "threshold": gate["threshold"],
         "passed": gate["passed"]}
        for gate in gates
        if gate["kind"] == "abs_error_vs_exact" and
        gate.get("reference_plan_hash") == review.reference_hash
    ]
    budget["systematic"]["signed_gate_observations"] = observations
    observed = [Q(item["absolute_error"], item["unit"]).to(budget["unit"]).value
                for item in observations]
    previous = budget["systematic"]["max_observed_absolute_error"]
    if previous is not None:
        observed.append(previous)
    budget["systematic"]["max_observed_absolute_error"] = (
        max(observed) if observed else None)
    return budget


class QclAgent:
    """Compile agent-authored QCL against a frozen molecular task."""

    def __init__(self, record: TaskRecord, verifier: TaskVerifier):
        verifier.verify(record)
        self.record = record
        self.verifier = verifier
        self._reviews: dict[str, QclReview] = {}
        self._reference_results: dict[str, Any] = {}

    def compile(self, workflow: Any, *args: Any, **kwargs: Any) -> QclReview:
        self.verifier.verify(self.record)
        self._check_gates()
        if self.record.spec.target.kind not in (
                "molecular_ground_energy", "frontier_orbitals",
                "lattice_ground_energy", "pairing_correlation",
                "aqueous_solubility", "ionization_energy", "electron_affinity",
                "ranking", "bond_dissociation", "interaction_scan", "pes_scan",
                "solvation_free_energy", "metabolic_sites", "binding_free_energy",
                "reaction_barrier", "reaction_energy", "interaction_energy",
                "periodic_energy", "spectral_function"):
            raise NotImplementedError("QCL agent does not support this signed target yet")
        plan = workflow.compile(*args, **kwargs)
        self._check_question(plan)
        self._check_effects(plan)
        estimate = plan.estimate()
        self._check_budget(estimate, plan)
        reference = (self._compile_reference() if self.record.spec.gates and
                     self.record.spec.target.kind != "spectral_function" else None)
        review = QclReview(
            plan, plan.hash, plan.explain(format="json"), estimate,
            self.record.hash, reference,
            reference.hash if reference is not None else None,
            output_hash=_output_signature(plan),
            reference_output_hash=(_output_signature(reference)
                                   if reference is not None else None))
        self._reviews[review.plan_hash] = review
        return review

    def run(self, review: QclReview, *, timeout_s: float | None = None,
            cpu_core_s: float | None = None) -> Any:
        self.verifier.verify(self.record)
        if (self._reviews.get(review.plan_hash) is not review or
                review.record_hash != self.record.hash or
                review.plan_hash != review.plan.hash or
                review.output_hash != _output_signature(review.plan) or
                (review.reference_plan is not None and
                 (review.reference_hash != review.reference_plan.hash or
                  review.reference_output_hash !=
                  _output_signature(review.reference_plan)))):
            raise AdapterError("CF-E120", None, "QCL plan changed after review",
                               "Compile and review the current plan")
        self._check_question(review.plan)
        self._check_effects(review.plan)
        self._check_budget(review.plan.estimate(), review.plan)
        timeout = self.record.spec.budget.get("wall_s", 300)
        uses_gpu = bool(_gpu_node_ids(review.plan))
        if uses_gpu and "gpu_s" in self.record.spec.budget:
            timeout = min(timeout, self.record.spec.budget["gpu_s"])
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise AdapterError("CF-E070", None, "QCL wall-time budget is exhausted",
                                   "Use a task with remaining wall time")
            timeout = min(timeout, timeout_s)
        deadline = time.monotonic() + timeout
        cpu_remaining = self.record.spec.budget.get("cpu_core_s")
        if cpu_core_s is not None:
            if not math.isfinite(cpu_core_s) or cpu_core_s <= 0:
                raise AdapterError("CF-E070", None, "QCL CPU core-time budget is exhausted",
                                   "Use a task with remaining CPU core time")
            cpu_remaining = (min(cpu_remaining, cpu_core_s)
                             if cpu_remaining is not None else cpu_core_s)
        reference = None
        if review.reference_plan is not None:
            reference = self._reference_results.get(review.reference_hash)
            if reference is None:
                reference = review.reference_plan.run(
                    timeout_s=max(0.001, deadline - time.monotonic()),
                    cpu_core_s=cpu_remaining)
                self._reference_results[review.reference_hash] = reference
                if cpu_remaining is not None:
                    cpu_remaining -= review.reference_plan.cpu_core_s or 0
                    if cpu_remaining <= 0:
                        raise RuntimeError("QCL-E503: worker exceeded CPU core-time limit")
        if deadline <= time.monotonic():
            raise TimeoutError("Signed QCL wall-time budget expired before candidate execution")
        result = review.plan.run(timeout_s=deadline - time.monotonic(),
                                 cpu_core_s=cpu_remaining)
        gate_results = self._evaluate_gates(result, reference, review)
        if time.monotonic() > deadline:
            raise TimeoutError("Signed QCL wall-time budget expired during gate evaluation")

        def stamp(value: Any) -> Any:
            if isinstance(value, list):
                return [stamp(item) for item in value]
            if isinstance(value, tuple):
                return tuple(stamp(item) for item in value)
            if isinstance(value, dict):
                return {key: stamp(item) for key, item in value.items()}
            if is_dataclass(value) and hasattr(value, "provenance"):
                changes = {"provenance": {**value.provenance,
                                          "task_record_hash": self.record.hash,
                                          "plan_hash": review.plan_hash,
                                          "transfers": review.explanation["transfers"]}}
                if gate_results:
                    changes["provenance"]["signed_gates"] = gate_results
                if hasattr(value, "values"):
                    changes["values"] = stamp(value.values)
                return replace(value, **changes)
            return value

        return stamp(result)

    def select(self, reviews: Mapping[str, QclReview], *,
               limits: Mapping[str, float] | None = None) -> QclSelection:
        """Run reviewed plans against one signed numeric gate and shared limits."""
        self.verifier.verify(self.record)
        if not reviews or not self.record.spec.gates:
            raise AdapterError("CF-E120", None,
                               "QCL candidate screening requires signed numeric gates",
                               "Sign an absolute-error gate and provide reviewed plans")
        for name, review in reviews.items():
            if not name or self._reviews.get(review.plan_hash) is not review:
                raise AdapterError("CF-E120", None, "QCL candidate was not reviewed",
                                   "Compile each named candidate against this signed task")
        _check_screen_currency(reviews)

        signed_budget = self.record.spec.budget
        if limits is not None and any(
                key not in signed_budget or type(value) not in (int, float) or
                not math.isfinite(value) or value > signed_budget[key] or value < 0
                for key, value in limits.items()):
            raise AdapterError("CF-E070", None, "QCL screen limits exceed signed budget",
                               "Use the remaining signed resource budget")
        budget = {**signed_budget, **(limits or {})}
        for dimension, field in (("qpu_shots", "shots"), ("qpu_s", "qpu_s"),
                                 ("qpu_cost", "qpu_cost")):
            if dimension not in budget and dimension == "qpu_cost":
                continue
            forecast = [review.estimate[field] for review in reviews.values()]
            if any(value is None for value in forecast) or (
                    sum(forecast) > budget.get(dimension, 0)):
                raise AdapterError("CF-E070", None,
                                   f"QCL candidate screen exceeds {dimension} budget",
                                   "Reduce the screen or use a larger signed budget")

        started = time.monotonic()
        cpu_spent = 0.0
        gpu_spent = 0.0
        outcomes: list[QclCandidateOutcome] = []
        results: dict[str, Any] = {}
        for name, review in reviews.items():
            wall_remaining = budget.get("wall_s", 300) - (time.monotonic() - started)
            cpu_remaining = (budget["cpu_core_s"] - cpu_spent
                             if "cpu_core_s" in budget else None)
            uses_gpu = bool(_gpu_node_ids(review.plan))
            gpu_remaining = (budget["gpu_s"] - gpu_spent
                             if uses_gpu and "gpu_s" in budget else None)
            if (wall_remaining <= 0 or cpu_remaining is not None and cpu_remaining <= 0 or
                    gpu_remaining is not None and gpu_remaining <= 0):
                raise AdapterError("CF-E070", None,
                                   "QCL candidate screen exhausted its signed budget",
                                   "Reduce the number of candidates")
            candidate_started = time.monotonic()
            result = self.run(review, timeout_s=min(wall_remaining, gpu_remaining)
                              if gpu_remaining is not None else wall_remaining,
                              cpu_core_s=cpu_remaining)
            elapsed = time.monotonic() - candidate_started
            cpu_seconds = (review.plan.cpu_core_s or 0) + (
                review.reference_plan.cpu_core_s or 0
                if review.reference_plan is not None else 0)
            gpu_seconds = sum(item["duration_s"] for item in review.plan.profile()
                              if item["id"] in _gpu_node_ids(review.plan))
            compute_seconds = sum(item["duration_s"] for item in review.plan.profile())
            cpu_spent += cpu_seconds
            gpu_spent += gpu_seconds
            if (cpu_remaining is not None and cpu_spent > budget["cpu_core_s"] or
                    gpu_remaining is not None and gpu_spent > budget["gpu_s"]):
                raise AdapterError("CF-E070", None,
                                   "QCL candidate screen exceeded its signed budget",
                                   "Reduce the number of candidates")
            gates = tuple(result.provenance.get("signed_gates", ()))
            if not gates:
                raise AdapterError("CF-E120", None,
                                   "QCL candidate has no evaluated signed gates",
                                   "Return a supported signed numerical result")
            outcomes.append(QclCandidateOutcome(
                name, review.plan_hash, all(gate["passed"] for gate in gates),
                gates, elapsed, compute_seconds, cpu_seconds, gpu_seconds,
                review.estimate["shots"], review.estimate["qpu_cost"],
                getattr(result, "value", None), getattr(result, "unit", None)))
            results[name] = result

        passing = [item for item in outcomes if item.passed]
        if not passing:
            return QclSelection("revise", None, None, tuple(outcomes))
        chosen = min(passing, key=lambda item: (
            item.qpu_cost if item.qpu_cost is not None else math.inf,
            item.qpu_shots, item.compute_s, item.cpu_core_s, item.name))
        return QclSelection("stop", chosen.name, results[chosen.name], tuple(outcomes))

    def _check_gates(self) -> None:
        gates = self.record.spec.gates
        if not gates:
            return
        target = self.record.spec.target
        if target.kind == "spectral_function":
            from chemflow.domains.fock_reference import MAX_ORBITALS

            size = len(target.model["impurity_model"]["eps_bath"]) + 1
            if (size > MAX_ORBITALS or
                    any(gate.kind != "l1_error_vs_exact" or
                        gate.unit != "dimensionless" or
                        gate.threshold is None or not math.isfinite(gate.threshold) or
                        gate.threshold < 0 for gate in gates)):
                raise AdapterError("CF-E120", None,
                                   "Independent spectral gate is unsupported",
                                   "Use an L1 gate within the exact reference orbital limit")
            return
        fixed_space = (isinstance(target.active, dict)
                       if target.kind == "molecular_ground_energy" else
                       target.kind == "reaction_energy" and
                       all(isinstance(species.active, dict)
                           for species in target.species) or
                       target.kind == "interaction_energy")
        if (not fixed_space or
                any(gate.kind != "abs_error_vs_exact" for gate in gates)):
            raise AdapterError("CF-E120", None, "QCL cannot evaluate these signed gates",
                               "Use an absolute-error gate with fixed active spaces")
        from chemflow.units import Q

        for gate in gates:
            try:
                if (gate.threshold is None or not math.isfinite(gate.threshold) or
                        gate.threshold < 0):
                    raise ValueError("invalid threshold")
                Q(1, "hartree").to(gate.unit)
            except (TypeError, ValueError) as error:
                raise AdapterError("CF-E120", None, "Signed energy gate has invalid units or threshold",
                                   "Use a finite nonnegative threshold in energy units") from error

    def _compile_reference(self) -> Any:
        import qcl

        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)

        @qcl.workflow
        def exact_energy(mol: Any, active: Any, cpu: Any) -> Any:
            mean_field = qcl.scf.hf(mol).to(cpu)
            space = qcl.active_space(mean_field, active).to(cpu)
            return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy

        if target.kind == "reaction_energy":
            @qcl.workflow
            def exact_reaction(species: Any, cpu: Any) -> Any:
                result = None
                for mol, active, coefficient in species:
                    term = coefficient * exact_energy(mol, active, cpu)
                    result = term if result is None else result + term
                return result

            species = [
                (qcl.Molecule(item.molecule["atoms"], model.basis,
                              item.molecule.get("charge", 0),
                              item.molecule.get("spin", 0),
                              item.molecule.get("unit", "angstrom")),
                 qcl.Active(item.active["n_elec"], item.active["n_orb"]),
                 item.coefficient)
                for item in target.species]
            return exact_reaction.compile(species, qcl.Cluster.local().cpus[0])

        if target.kind == "interaction_energy":
            from chemflow.domains.drug import component_specs

            components = component_specs(target.complex, target.fragments,
                                         target.electronic_model, target.counterpoise)
            molecules = [qcl.Molecule(item["atoms"], model.basis,
                                      unit=item.get("unit", "angstrom"))
                         for item in components]

            @qcl.workflow
            def exact_interaction(items: Any, cpu: Any) -> Any:
                energies = [exact_energy(mol, qcl.Active(max_orbitals=999), cpu)
                            for mol in items]
                return energies[0] - energies[1] - energies[2]

            return exact_interaction.compile(molecules, qcl.Cluster.local().cpus[0])

        molecule = target.molecule
        return exact_energy.compile(
            qcl.Molecule(molecule["atoms"], model.basis,
                         molecule.get("charge", 0), molecule.get("spin", 0),
                         molecule.get("unit", "angstrom")),
            qcl.Active(target.active["n_elec"], target.active["n_orb"]),
            qcl.Cluster.local().cpus[0])

    def _evaluate_gates(self, result: Any, reference: Any,
                        review: QclReview) -> list[dict[str, Any]]:
        if not self.record.spec.gates:
            return []
        import qcl

        if self.record.spec.target.kind == "spectral_function":
            import numpy as np

            from chemflow.domains.fock_reference import reference as spectral_reference

            target = self.record.spec.target
            exact = spectral_reference(target.model_dump(mode="json"),
                                       target.model["impurity_model"])
            if (not isinstance(result, qcl.SpectralEstimate) or
                    result.unit != exact.unit or result.omega != exact.omega):
                raise AdapterError("CF-E120", None,
                                   "Spectral grid differs from the signed task",
                                   "Return density on the signed frequency grid")
            error = float(np.trapezoid(np.abs(np.asarray(result.spectral) -
                                               np.asarray(exact.spectral)), result.omega))
            return [{"kind": gate.kind, "threshold": gate.threshold, "unit": gate.unit,
                     "error": error, "passed": error <= gate.threshold}
                    for gate in self.record.spec.gates]

        if not isinstance(result, qcl.Estimate) or not isinstance(reference, qcl.Estimate):
            raise AdapterError("CF-E120", None, "Signed energy gate requires energy estimates",
                               "Return an energy estimate from the signed molecule")
        return [{"kind": gate.kind, "threshold": gate.threshold, "unit": gate.unit,
                 "error": error, "passed": error <= gate.threshold,
                 "reference_plan_hash": review.reference_hash}
                for gate in self.record.spec.gates
                for error in [abs(result.in_units(gate.unit).value -
                                  reference.in_units(gate.unit).value)]]

    def _check_question(self, plan: Any) -> None:
        target = self.record.spec.target
        if target.kind == "spectral_function":
            nodes = plan.graph_json()["nodes"]
            output = plan._output
            compute = [node for node in nodes if node["kind"] == "compute"]
            required = {"model": target.model["impurity_model"],
                        "sector": target.sector, "orbital": target.orbital,
                        "spin": target.spin, "omega": target.omega, "eta": target.eta}
            if (plan.output_type != "spectral_estimate" or
                    not hasattr(output, "node_id") or len(compute) != 1 or
                    compute[0]["id"] != output.node_id or
                    compute[0]["op"] != "dynamics.spectral_exact" or
                    compute[0]["params"] != required):
                raise AdapterError("CF-E120", None,
                                   "QCL spectrum differs from the signed task",
                                   "Use the signed model, sector, spin and frequency grid")
            return
        if target.kind == "periodic_energy":
            nodes = plan.graph_json()["nodes"]
            output = plan._output
            compute = [node for node in nodes if node["kind"] == "compute"]
            required = {"system": {**target.system, "unit": target.system.get("unit", "angstrom")},
                        "kmesh": list(target.kmesh), "method": target.method,
                        "exxdiv": target.exxdiv}
            if (plan.output_type != "estimate" or
                    not hasattr(output, "node_id") or len(compute) != 1 or
                    compute[0]["id"] != output.node_id or
                    compute[0]["op"] != "periodic.scf" or
                    compute[0]["params"] != required):
                raise AdapterError("CF-E120", None,
                                   "QCL periodic plan differs from the signed task",
                                   "Use the signed cell, kmesh, method and exxdiv")
            return
        if target.kind == "reaction_energy":
            self._check_reaction_question(plan)
            return
        if target.kind == "frontier_orbitals":
            self._check_frontier_question(plan)
            return
        if target.kind == "solvation_free_energy":
            self._check_solvation_question(plan)
            return
        if target.kind == "metabolic_sites":
            self._check_metabolic_question(plan)
            return
        if target.kind == "binding_free_energy":
            self._check_binding_question(plan)
            return
        if target.kind == "reaction_barrier":
            self._check_barrier_question(plan)
            return
        if target.kind in ("lattice_ground_energy", "pairing_correlation"):
            self._check_lattice_question(plan)
            return
        if target.kind == "aqueous_solubility":
            self._check_solubility_question(plan)
            return
        if target.kind in ("ionization_energy", "electron_affinity"):
            self._check_charge_question(plan)
            return
        if target.kind == "ranking":
            self._check_ranking_question(plan)
            return
        if target.kind in ("bond_dissociation", "interaction_scan", "interaction_energy",
                           "pes_scan"):
            self._check_energy_collection(plan)
            return
        graph = plan.graph_json()
        molecules = [node["params"] for node in graph["nodes"] if node["op"] == "molecule"]
        expected = target.molecule
        model = ElectronicModel.model_validate(target.electronic_model)
        spin = expected.get("spin", 0)
        references = ("ROHF", "UHF") if spin else ("RHF",)
        if (model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
                model.frozen_core != 0 or model.reference not in references or
                model.dft_functional is not None and spin and model.reference != "UHF"):
            raise AdapterError("CF-E120", None, "Signed electronic model is unsupported by QCL",
                               "Use a matching HF model or the existing ChemFlow Agent")
        if (len(molecules) != 1 or
                any(molecules[0].get(key) != value for key, value in {
                    "atoms": expected.get("atoms"), "basis": model.basis,
                    "charge": expected.get("charge", 0),
                    "spin": expected.get("spin", 0),
                    "unit": expected.get("unit", "angstrom"),
                }.items()) or plan.output_type != "estimate"):
            raise AdapterError("CF-E120", None, "QCL plan differs from the signed question",
                               "Use the frozen molecule, model and an energy result")
        nodes = {node["id"]: node for node in graph["nodes"]}
        output = nodes[plan._output.node_id]
        if output["op"] not in ("scf.energy", "fci.energy", "sci.energy", "sqd.diagonalize",
                                "vqe.sampled", "estimate.quantum", "estimates.merge"):
            raise AdapterError("CF-E120", None, "QCL output is not a ground-energy calculation",
                               "Return an SCF, FCI, SQD, VQE or Pauli energy")
        dft_nodes = [node for node in nodes.values()
                     if node["op"] in ("dft.rks", "dft.uks")]
        if model.dft_functional is None:
            if dft_nodes:
                raise AdapterError("CF-E120", None, "QCL DFT method is not signed",
                                   "Use the signed HF electronic model")
            if output["op"] == "scf.energy":
                source = nodes[output["inputs"][0]] if output["inputs"] else None
                while source is not None and source["op"] == "copy":
                    source = nodes[source["inputs"][0]]
                computes = [node for node in nodes.values() if node["kind"] == "compute"]
                expected_scf = {"RHF": ("scf.rhf", "scf.hf"),
                                "ROHF": ("scf.rohf", "scf.hf"),
                                "UHF": ("scf.uhf",)}[model.reference]
                if (source is None or source["op"] not in expected_scf or
                        len(computes) != 1):
                    raise AdapterError("CF-E120", None,
                                       "QCL SCF calculation differs from signed model",
                                       "Return the signed reference's SCF energy")
            elif model.reference == "UHF":
                raise AdapterError("CF-E120", None,
                                   "QCL UHF calculation differs from signed model",
                                   "Return the signed UHF energy")
        else:
            source = nodes[output["inputs"][0]] if output["inputs"] else None
            while source is not None and source["op"] == "copy":
                source = nodes[source["inputs"][0]]
            computes = [node for node in nodes.values() if node["kind"] == "compute"]
            expected_dft = "dft.uks" if spin else "dft.rks"
            if (output["op"] != "scf.energy" or len(dft_nodes) != 1 or
                    dft_nodes[0]["op"] != expected_dft or
                    source is not dft_nodes[0] or len(computes) != 1 or
                    dft_nodes[0]["params"]["xc"].casefold() !=
                    model.dft_functional.casefold()):
                raise AdapterError("CF-E120", None, "QCL DFT calculation differs from signed model",
                                   "Return the signed KS functional's energy")
        pauli_nodes = ([output] if output["op"] == "estimate.quantum" else
                       [nodes[index] for index in output["inputs"]]
                       if output["op"] == "estimates.merge" else [])
        if output["op"] == "estimates.merge" and (
                not pauli_nodes or any(node["op"] != "estimate.quantum"
                                       for node in pauli_nodes)):
            raise AdapterError("CF-E120", None, "QCL Pauli merge is invalid",
                               "Merge only Pauli estimates from the signed Hamiltonian")
        for pauli_node in pauli_nodes:
            def original(identifier: int) -> dict[str, Any]:
                node = nodes[identifier]
                while node["op"] == "copy":
                    node = nodes[node["inputs"][0]]
                return node

            mapped = original(pauli_node["inputs"][0])
            fermion = original(mapped["inputs"][0]) if mapped["inputs"] else {}
            space = original(fermion["inputs"][0]) if fermion.get("inputs") else {}
            if (mapped["op"] != "qubit_hamiltonian" or
                    mapped["params"].get("mapping") != "jordan_wigner" or
                    fermion.get("op") != "active.hamiltonian" or
                    space.get("op") != "active_space"):
                raise AdapterError("CF-E120", None,
                                   "QCL Pauli estimate uses a different Hamiltonian",
                                   "Map the signed active-space Hamiltonian")
        if isinstance(target.active, dict):
            if (target.active.get("method", "manual") != "manual" or
                    target.active.get("orbital_source", "canonical_hf") != "canonical_hf"):
                raise AdapterError("CF-E120", None, "Signed orbital policy is unsupported by QCL",
                                   "Use a canonical HF active space")
            spaces = [node["params"] for node in graph["nodes"]
                      if node["op"] == "active_space"]
            if spaces and any(space.get("electrons") != target.active.get("n_elec") or
                              space.get("orbitals") != target.active.get("n_orb")
                              for space in spaces):
                raise AdapterError("CF-E120", None, "QCL active space differs from signed task",
                                   "Use the frozen active-space counts")
            sector = target.sector
            electrons = target.active["n_elec"]
            if (sector.get("n_orb") != target.active["n_orb"] or
                    sector.get("n_alpha") != (electrons + spin) // 2 or
                    sector.get("n_beta") != (electrons - spin) // 2):
                raise AdapterError("CF-E120", None, "Signed sector differs from QCL active space",
                                   "Use a matching signed sector and active space")

    def _check_reaction_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if (model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
                model.frozen_core != 0 or model.dft_functional is not None or
                model.reference != "RHF" or target.thermo is not None):
            raise AdapterError("CF-E120", None, "Signed reaction model is unsupported by QCL",
                               "Use an all-electron HF electronic reaction model")
        if plan.output_type != "estimate":
            raise AdapterError("CF-E120", None, "QCL reaction needs one energy estimate",
                               "Return the signed linear combination of species energies")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}

        def lineage(identifier: int) -> set[int]:
            return {identifier} | {ancestor for input_id in nodes[identifier]["inputs"]
                                   for ancestor in lineage(input_id)}

        def unwrap(identifier: int) -> dict[str, Any]:
            node = nodes[identifier]
            while node["op"] == "copy":
                node = nodes[node["inputs"][0]]
            return node

        def molecule_key(data: dict[str, Any]) -> str:
            return json.dumps({"atoms": data.get("atoms"), "basis": data.get("basis"),
                               "charge": data.get("charge", 0),
                               "spin": data.get("spin", 0),
                               "unit": data.get("unit", "angstrom")}, sort_keys=True)

        expected_species = {}
        for species in target.species:
            data = {**species.molecule, "basis": model.basis}
            key = molecule_key(data)
            if key in expected_species:
                raise AdapterError("CF-E120", None, "QCL reaction species are ambiguous",
                                   "Use distinct molecule identities per signed species")
            expected_species[key] = species

        def terms(identifier: int, coefficient: float = 1.0) -> list[tuple[str, float]]:
            node = unwrap(identifier)
            if node["op"] == "estimate.scale":
                return terms(node["inputs"][0], coefficient * node["params"]["factor"])
            if node["op"] in ("estimate.add", "estimate.sub"):
                left, right = node["inputs"]
                return terms(left, coefficient) + terms(
                    right, coefficient if node["op"] == "estimate.add" else -coefficient)
            if node["op"] not in ("scf.energy", "fci.energy"):
                raise ValueError("reaction term is not a supported electronic energy")
            ancestors = lineage(node["id"])
            molecule_nodes = [nodes[index] for index in ancestors
                              if nodes[index]["op"] == "molecule"]
            if len(molecule_nodes) != 1:
                raise ValueError("reaction energy has ambiguous molecule lineage")
            key = molecule_key(molecule_nodes[0]["params"])
            species = expected_species.get(key)
            if species is None:
                raise ValueError("reaction species differs from signed task")
            scf_nodes = [nodes[index] for index in ancestors
                         if nodes[index]["op"] in ("scf.hf", "scf.rhf", "scf.rohf")]
            if len(scf_nodes) != 1 or unwrap(scf_nodes[0]["inputs"][0])["id"] != (
                    molecule_nodes[0]["id"]):
                raise ValueError("reaction SCF differs from signed species")
            if node["op"] == "fci.energy":
                spaces = [nodes[index] for index in ancestors
                          if nodes[index]["op"] == "active_space"]
                solvers = [nodes[index] for index in ancestors
                           if nodes[index]["op"] == "fci.solve"]
                if (len(spaces) != 1 or len(solvers) != 1 or
                        spaces[0]["params"] != {"electrons": species.active["n_elec"],
                                                "orbitals": species.active["n_orb"]} or
                        unwrap(solvers[0]["inputs"][1])["op"] != "active.sector" or
                        species.sector != {
                            "n_alpha": (species.active["n_elec"] +
                                        species.molecule.get("spin", 0)) // 2,
                            "n_beta": (species.active["n_elec"] -
                                       species.molecule.get("spin", 0)) // 2,
                            "n_orb": species.active["n_orb"]}):
                    raise ValueError("reaction active space or sector differs")
            return [(key, coefficient)]

        try:
            actual = Counter()
            for key, coefficient in terms(plan._output.node_id):
                actual[key] += coefficient
            expected = {key: species.coefficient for key, species in expected_species.items()}
            if (set(actual) != set(expected) or any(
                    not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12)
                    for key, value in expected.items())):
                raise ValueError("reaction stoichiometry differs")
            used = lineage(plan._output.node_id)
            if any(node["kind"] == "compute" and node["id"] not in used
                   for node in nodes.values()):
                raise ValueError("reaction has unrelated calculations")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise AdapterError("CF-E120", None, "QCL reaction differs from signed task",
                               "Match signed species, active spaces and coefficients") from error

    def _check_barrier_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        reactant = next(item for item in target.species if item.role == "minimum")
        transition = next(item for item in target.species if item.role == "ts")
        if (model.reference != "RHF" or model.hamiltonian_class != "nonrel_bo" or
                model.ecp is not None or model.frozen_core != 0 or
                model.dft_functional is not None or
                any(item.molecule.get("spin", 0) != 0 or item.target_spin is not None
                    for item in target.species)):
            raise AdapterError("CF-E120", None, "Signed barrier model is unsupported by QCL",
                               "Use a closed-shell all-electron barrier model")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}
        computes = Counter(node["op"] for node in nodes.values()
                           if node["kind"] == "compute")
        required = Counter({"geometry.optimize": 1, "sella.saddle": 1,
                            "scf.hf": 2, "active_space": 2, "fci.solve": 2})
        if target.thermo is not None:
            required.update({"frequencies": 2, "thermo.rrho": 2})
        if target.rate_constant:
            required.update({"thermo.eyring": 1})
        expected_output = "composite" if target.rate_constant else "estimate"
        if plan.output_type != expected_output or computes != required:
            raise AdapterError("CF-E120", None, "QCL barrier operations differ from task",
                               "Optimize and verify the TS, then compare two FCI energies")

        def unwrap(identifier: int) -> dict[str, Any]:
            node = nodes[identifier]
            while node["op"] == "copy":
                node = nodes[node["inputs"][0]]
            return node

        def expected_molecule(data: dict[str, Any]) -> dict[str, Any]:
            return {"atoms": data.get("atoms"), "basis": model.basis,
                    "charge": data.get("charge", 0), "spin": data.get("spin", 0),
                    "unit": data.get("unit", "angstrom")}

        def branch(identifier: int, species: Any, geometry_op: str) -> None:
            energy = unwrap(identifier)
            correction = None
            if target.thermo is not None:
                if energy["op"] != "estimate.add" or len(energy["inputs"]) != 2:
                    raise ValueError("thermal barrier requires an electronic plus RRHO energy")
                correction = unwrap(energy["inputs"][1])
                energy = unwrap(energy["inputs"][0])
            fci = unwrap(energy["inputs"][0])
            hamiltonian = unwrap(fci["inputs"][0])
            sector = unwrap(fci["inputs"][1])
            active = unwrap(hamiltonian["inputs"][0])
            mean_field = unwrap(active["inputs"][0])
            geometry = unwrap(mean_field["inputs"][0])
            if (energy["op"] != "fci.energy" or fci["op"] != "fci.solve" or
                    hamiltonian["op"] != "active.hamiltonian" or
                    sector["op"] != "active.sector" or
                    sector["inputs"] != [active["id"]] or
                    active["op"] != "active_space" or
                    active["params"] != {"electrons": species.active.get("n_elec"),
                                          "orbitals": species.active.get("n_orb")} or
                    species.sector != {"n_alpha": species.active.get("n_elec") // 2,
                                       "n_beta": species.active.get("n_elec") // 2,
                                       "n_orb": species.active.get("n_orb")} or
                    mean_field["op"] != "scf.hf" or geometry["op"] != geometry_op):
                raise ValueError("barrier energy lineage differs")
            if correction is not None:
                modes = unwrap(correction["inputs"][1])
                thermal = target.thermo
                if (correction["op"] != "thermo.rrho" or
                        correction["inputs"][0] != geometry["id"] or
                        correction["params"] != {
                            "quantity": thermal.quantity,
                            "temperature_K": thermal.temperature_K,
                            "pressure_Pa": thermal.pressure_Pa} or
                        modes["op"] != "frequencies" or
                        modes["inputs"] != [geometry["id"]]):
                    raise ValueError("RRHO geometry or conditions differ")
            if geometry_op == "geometry.optimize":
                initial = unwrap(geometry["inputs"][0])
                if (geometry["params"] != {"kind": "minimum", "spin": 0} or
                        initial["op"] != "molecule" or
                        initial["params"] != expected_molecule(species.molecule)):
                    raise ValueError("reactant optimization differs")
            else:
                inputs = [unwrap(item) for item in geometry["inputs"]]
                if (geometry["params"] != {"spin": 0, "verify_irc": True} or
                        [node["op"] for node in inputs] != ["molecule"] * 3 or
                        [node["params"] for node in inputs] != [
                            expected_molecule(species.molecule),
                            expected_molecule(reactant.molecule),
                            expected_molecule(target.product)]):
                    raise ValueError("connected TS differs")

        try:
            barrier_output = plan._output
            if target.rate_constant:
                if not isinstance(barrier_output, tuple) or len(barrier_output) != 2:
                    raise ValueError("Gibbs barrier must return barrier and rate")
                barrier_output, rate_output = barrier_output
                rate = unwrap(rate_output.node_id)
                if (rate["op"] != "thermo.eyring" or
                        rate["params"] != {"temperature_K": target.thermo.temperature_K} or
                        rate["inputs"] != [barrier_output.node_id]):
                    raise ValueError("Eyring rate does not use the signed Gibbs barrier")
            output = unwrap(barrier_output.node_id)
            if output["op"] != "estimate.sub" or len(output["inputs"]) != 2:
                raise ValueError("barrier difference differs")
            branch(output["inputs"][0], transition, "sella.saddle")
            branch(output["inputs"][1], reactant, "geometry.optimize")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise AdapterError("CF-E120", None, "QCL barrier differs from signed task",
                               "Return connected TS FCI energy minus reactant FCI energy") from error

    def _check_binding_question(self, plan: Any) -> None:
        target = self.record.spec.target
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        output = nodes[plan._output.node_id] if plan.output_type == "estimate" else None
        while output is not None and output["op"] == "copy":
            output = nodes[output["inputs"][0]]
        expected = target.model_dump(exclude={"kind", "operator", "protocol"})
        if (len(computes) != 1 or computes[0]["op"] != "binding.mmgbsa" or
                computes[0]["params"] != expected or output is not computes[0]):
            raise AdapterError("CF-E120", None, "QCL binding protocol differs from task",
                               "Use the signed receptor, ligand and MM/GBSA protocol")

    def _check_metabolic_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}
        molecules = [node for node in nodes.values() if node["op"] == "molecule"]
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        expected = {"atoms": target.molecule.get("atoms"), "basis": model.basis,
                    "charge": target.molecule.get("charge", 0),
                    "spin": target.molecule.get("spin", 0),
                    "unit": target.molecule.get("unit", "angstrom")}
        output = nodes[plan._output.node_id] if plan.output_type == "site_scores" else None
        while output is not None and output["op"] == "copy":
            output = nodes[output["inputs"][0]]
        if (len(molecules) != 1 or molecules[0]["params"] != expected or
                len(computes) != 1 or computes[0]["op"] != "metabolism.sites" or
                computes[0]["inputs"] != [molecules[0]["id"]] or
                computes[0]["params"] != {"model": model.model_dump(),
                                           "level": target.level.method} or
                output is not computes[0]):
            raise AdapterError("CF-E120", None, "QCL metabolic site model differs from task",
                               "Use the signed molecule, model and scoring level")

    def _check_solvation_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if target.level.method == "gfn2":
            self._check_gfn2_solvation_question(plan, target, model)
            return
        spin = target.molecule.get("spin", 0)
        if (target.level.method not in ("hf", "dft") or
                target.solvent.model not in ("pcm", "smd") or
                target.solvent.model == "smd" and target.solvent.eps is not None or
                model.reference != ("UHF" if spin else "RHF") or
                model.hamiltonian_class != "nonrel_bo" or
                model.ecp is not None or model.frozen_core != 0 or
                spin and target.level.method != "dft" or
                target.level.dispersion is not None or
                (target.level.method == "hf" and model.dft_functional is not None) or
                (target.level.method == "dft" and
                 (not target.level.functional or
                  model.dft_functional is not None and
                  model.dft_functional.casefold() != target.level.functional.casefold()))):
            raise AdapterError("CF-E120", None, "Signed solvation model is unsupported by QCL",
                               "Use closed-shell HF or DFT with PCM/SMD and no dispersion")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}

        def unwrap(identifier: int) -> dict[str, Any]:
            node = nodes[identifier]
            while node["op"] == "copy":
                node = nodes[node["inputs"][0]]
            return node

        expected = {"atoms": target.molecule.get("atoms"), "basis": model.basis,
                    "charge": target.molecule.get("charge", 0), "spin": spin,
                    "unit": target.molecule.get("unit", "angstrom")}
        molecules = [node for node in nodes.values() if node["op"] == "molecule"]
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        if (plan.output_type != "estimate" or
                any(node["params"] != expected for node in molecules)):
            raise AdapterError("CF-E120", None, "QCL solvation plan differs from signed task",
                               "Use gas and solvent energies on the frozen molecule")
        operations = [node["op"] for node in computes]
        operation = ("dft.uks" if spin else "dft.rks") if target.level.method == "dft" else "scf.rhf"
        if operation == "scf.rhf" and operations == ["int2e", operation, operation]:
            eri, gas, liquid = computes
            valid_inputs = (
                len(molecules) == 3 and
                unwrap(eri["inputs"][0])["params"] == expected and
                all(node["device"]["kind"] == "cpu" and
                    len(node["inputs"]) == 2 and
                    unwrap(node["inputs"][0])["params"] == expected and
                    unwrap(node["inputs"][1])["id"] == eri["id"] and
                    node["params"].get("reuse_eri")
                    for node in (gas, liquid)))
        elif operations == [operation, operation]:
            gas, liquid = computes
            valid_inputs = (
                len(molecules) == 2 and
                gas["device"]["kind"] == liquid["device"]["kind"] and
                gas["device"]["kind"] in (("cpu", "gpu") if operation.startswith("dft.")
                                            else ("gpu",)) and
                gas["device"]["id"] == liquid["device"]["id"] and
                all(len(node["inputs"]) == 1 and
                    unwrap(node["inputs"][0])["params"] == expected and
                    not node["params"].get("reuse_eri")
                    for node in (gas, liquid)))
        else:
            valid_inputs = False
        expected_solvent = ({"model": "smd", "name": target.solvent.name}
                            if target.solvent.model == "smd" else
                            {"name": target.solvent.name or "custom",
                             "eps": target.solvent.eps})
        if (not valid_inputs or
                operation.startswith("dft.") and any(
                    node["params"].get("xc", "").casefold() !=
                    target.level.functional.casefold() for node in (gas, liquid)) or
                gas["params"].get("solvent") is not None or
                liquid["params"].get("solvent") != expected_solvent):
            raise AdapterError("CF-E120", None, "QCL solvent or integrals differ from signed task",
                               "Use the signed PCM and shared molecular integrals")
        output = unwrap(plan._output.node_id)
        if (output["op"] != "estimate.sub" or len(output["inputs"]) != 2 or
                [unwrap(identifier)["op"] for identifier in output["inputs"]] !=
                ["scf.energy", "scf.energy"] or
                [unwrap(unwrap(identifier)["inputs"][0])["id"]
                 for identifier in output["inputs"]] != [liquid["id"], gas["id"]]):
            raise AdapterError("CF-E120", None, "QCL solvation difference has wrong sign",
                               "Return solvent energy minus gas-phase energy")

    @staticmethod
    def _check_gfn2_solvation_question(plan: Any, target: Any,
                                      model: ElectronicModel) -> None:
        if (target.solvent.model not in ("alpb", "gbsa") or
                target.solvent.eps is not None or
                model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
                model.frozen_core != 0 or model.dft_functional is not None):
            raise AdapterError("CF-E120", None, "Signed GFN2 solvation model is unsupported",
                               "Use a named ALPB or GBSA solvent without ECP")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}

        def unwrap(identifier: int) -> dict[str, Any]:
            node = nodes[identifier]
            while node["op"] == "copy":
                node = nodes[node["inputs"][0]]
            return node

        expected = {"atoms": target.molecule.get("atoms"), "basis": model.basis,
                    "charge": target.molecule.get("charge", 0),
                    "spin": target.molecule.get("spin", 0),
                    "unit": target.molecule.get("unit", "angstrom")}
        molecules = [node for node in nodes.values() if node["op"] == "molecule"]
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        output = unwrap(plan._output.node_id) if plan.output_type == "estimate" else {}
        if (len(molecules) != 2 or any(node["params"] != expected for node in molecules) or
                [node["op"] for node in computes] != ["gfn2.energy", "gfn2.energy"] or
                any(len(node["inputs"]) != 1 or
                    unwrap(node["inputs"][0])["params"] != expected for node in computes) or
                computes[0]["params"].get("solvent") is not None or
                computes[1]["params"].get("solvent") != {
                    "model": target.solvent.model, "name": target.solvent.name} or
                output.get("op") != "estimate.sub" or
                output.get("inputs") != [computes[1]["id"], computes[0]["id"]]):
            raise AdapterError("CF-E120", None, "QCL GFN2 solvation differs from signed task",
                               "Use signed solvent energy minus gas energy")

    def _check_frontier_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if (target.level.method != "hf" or target.spin != "alpha" or
                target.molecule.get("spin", 0) != 0 or
                model.reference != "RHF" or model.hamiltonian_class != "nonrel_bo" or
                model.ecp is not None or model.frozen_core != 0 or
                model.dft_functional is not None):
            raise AdapterError("CF-E120", None, "Signed frontier model is unsupported by QCL",
                               "Use a closed-shell HF frontier target")
        graph = plan.graph_json()
        nodes = {node["id"]: node for node in graph["nodes"]}
        molecules = [node for node in nodes.values() if node["op"] == "molecule"]
        expected = {"atoms": target.molecule.get("atoms"), "basis": model.basis,
                    "charge": target.molecule.get("charge", 0), "spin": 0,
                    "unit": target.molecule.get("unit", "angstrom")}
        if (len(molecules) != 1 or
                any(molecules[0]["params"].get(key) != value
                    for key, value in expected.items()) or
                plan.output_type != "estimate" or
                [node["op"] for node in nodes.values() if node["kind"] == "compute"]
                != ["scf.rhf"]):
            raise AdapterError("CF-E120", None, "QCL frontier plan differs from signed task",
                               "Use one matching RHF calculation and a frontier result")
        output = nodes[plan._output.node_id]
        while output["op"] == "copy":
            output = nodes[output["inputs"][0]]
        scf_id = next(node["id"] for node in nodes.values() if node["op"] == "scf.rhf")
        if target.quantity == "gap":
            valid = (output["op"] == "estimate.sub" and
                     [nodes[item]["op"] for item in output["inputs"]] ==
                     ["scf.lumo", "scf.homo"] and
                     all(nodes[item]["inputs"] == [scf_id]
                         for item in output["inputs"]))
        else:
            valid = (output["op"] == f"scf.{target.quantity}" and
                     output["inputs"] == [scf_id])
        if not valid:
            raise AdapterError("CF-E120", None, "QCL frontier quantity differs from signed task",
                               "Return the signed HOMO, LUMO or LUMO-minus-HOMO gap")

    def _check_lattice_question(self, plan: Any) -> None:
        target = self.record.spec.target
        lattice = target.lattice
        if (target.target_spin is not None or lattice.get("model", "hubbard") != "hubbard" or
                lattice.get("unit", "hartree") != "hartree"):
            raise AdapterError("CF-E120", None, "Signed lattice model is unsupported by QCL",
                               "Use a Hubbard lattice in hartree without a spin-state constraint")
        from chemflow.domains.materials import lattice_arrays

        try:
            expected_sites = len(lattice_arrays(**lattice)[0])
        except (TypeError, ValueError) as error:
            raise AdapterError("CF-E120", None, "Signed lattice model is invalid",
                               "Use a supported Hubbard lattice") from error
        shape = (lattice.get("shape") if lattice.get("geometry", "chain") == "square"
                 else [lattice.get("n_sites")])
        if (not isinstance(shape, (list, tuple)) or not shape or
                any(type(size) is not int or size < 1 for size in shape)):
            raise AdapterError("CF-E120", None, "Signed lattice shape is invalid",
                               "Use a positive chain, ring or square shape")
        expected_model = {"geometry": lattice.get("geometry", "chain"),
                          "shape": list(shape),
                          "t": lattice.get("t", 1.0), "U": lattice.get("U", 4.0)}
        graph = plan.graph_json()
        nodes = {node["id"]: node for node in graph["nodes"]}
        models = [node for node in nodes.values() if node["op"] == "models.hubbard"]
        sectors = [node for node in nodes.values() if node["op"] == "lattice.sector"]
        states = [node for node in nodes.values() if node["op"] == "ed.ground_state"]
        if (len(models) != 1 or models[0]["params"] != expected_model or
                len(sectors) != 1 or len(states) != 1 or
                sectors[0]["inputs"] != [models[0]["id"]] or
                sectors[0]["params"] != {key: target.sector[key]
                                          for key in ("n_alpha", "n_beta")}):
            raise AdapterError("CF-E120", None, "QCL lattice differs from signed task",
                               "Use the frozen Hubbard lattice and electron sector")
        if (states[0]["inputs"] != [models[0]["id"], sectors[0]["id"]] or
                target.sector.get("n_orb") != expected_sites or
                plan.output_type != "estimate"):
            raise AdapterError("CF-E120", None, "QCL lattice differs from signed task",
                               "Use the frozen Hubbard lattice and electron sector")
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        output = nodes[plan._output.node_id]
        if target.kind == "lattice_ground_energy":
            valid = (computes == states and output["op"] == "ed.energy" and
                     output["inputs"] == [states[0]["id"]])
        else:
            valid = (len(computes) == 2 and computes[0] == states[0] and
                     computes[1]["op"] == "expect" and output == computes[1] and
                     output["inputs"] == [states[0]["id"]] and
                     output["params"] == {"observable": target.quantity,
                                          "left": target.site_pair[0],
                                          "right": target.site_pair[1]})
        if not valid:
            raise AdapterError("CF-E120", None, "QCL lattice quantity differs from signed task",
                               "Return the signed lattice energy or site correlation")

    def _check_solubility_question(self, plan: Any) -> None:
        target = self.record.spec.target
        computes = [node for node in plan.graph_json()["nodes"]
                    if node["kind"] == "compute"]
        expected = {"smiles": target.molecule_smiles,
                    "melting_point_C": target.melting_point_C if target.model == "gse" else None}
        if (plan.output_type != "estimate" or len(computes) != 1 or
                computes[0]["op"] != f"solubility.{target.model}" or
                computes[0]["params"] != expected or
                computes[0]["id"] != plan._output.node_id):
            raise AdapterError("CF-E120", None, "QCL solubility plan differs from signed task",
                               "Use the frozen SMILES, model and melting point")

    def _check_charge_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if (model.reference != "RHF" or model.hamiltonian_class != "nonrel_bo" or
                model.ecp is not None or model.frozen_core != 0 or
                model.dft_functional is not None or target.molecule.get("spin", 0) != 0):
            raise AdapterError("CF-E120", None, "Signed charge model is unsupported by QCL",
                               "Use a closed-shell HF initial state")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}

        def only(op: str) -> dict[str, Any]:
            found = [node for node in nodes.values() if node["op"] == op]
            if len(found) != 1:
                raise AdapterError("CF-E120", None, "QCL charge plan is ambiguous",
                                   "Use one molecule, active space and charge shift")
            return found[0]

        molecule = only("molecule")
        expected = {"atoms": target.molecule.get("atoms"), "basis": model.basis,
                    "charge": target.molecule.get("charge", 0), "spin": 0,
                    "unit": target.molecule.get("unit", "angstrom")}
        scf = only("scf.hf")
        active = only("active_space")
        operators = [node for node in nodes.values() if node["op"] == "active.hamiltonian"]
        base_sectors = [node for node in nodes.values() if node["op"] == "active.sector"]
        shifted_sector = only("sector.shift")
        output = nodes[plan._output.node_id] if plan.output_type == "estimate" else {}
        computes = [node for node in nodes.values() if node["kind"] == "compute"]
        if (molecule["params"] != expected or scf["inputs"] != [molecule["id"]] or
                active["inputs"] != [scf["id"]] or
                not operators or not base_sectors or
                any(node["inputs"] != [active["id"]] for node in
                    [*operators, *base_sectors]) or
                shifted_sector["inputs"][0] not in
                {node["id"] for node in base_sectors} or
                [node["op"] for node in computes] !=
                ["scf.hf", "active_space", "fci.solve", "fci.solve"] or
                output.get("op") != "estimate.sub"):
            raise AdapterError("CF-E120", None, "QCL charge plan differs from signed task",
                               "Use one shared active space and two FCI charge sectors")
        shift = -1 if target.kind == "ionization_energy" else 1
        counts = shifted_sector["params"]
        if (counts["n_alpha"] + counts["n_beta"] != active["params"]["electrons"] + shift or
                abs(counts["n_alpha"] - counts["n_beta"]) / 2 != target.final_spin):
            raise AdapterError("CF-E120", None, "QCL final charge sector differs from signed task",
                               "Use the signed final spin and electron difference")
        energies = [node for node in nodes.values() if node["op"] == "fci.energy"]
        solvers = [node for node in nodes.values() if node["op"] == "fci.solve"]
        if len(energies) != 2 or len(solvers) != 2:
            raise AdapterError("CF-E120", None, "QCL charge energies are incomplete",
                               "Compute both FCI sector energies")
        energy_for_sector = {solver["inputs"][1]: next(
            (energy["id"] for energy in energies if energy["inputs"] == [solver["id"]]), None)
            for solver in solvers if solver["inputs"][0] in
            {node["id"] for node in operators}}
        base = next((energy_for_sector[node["id"]] for node in base_sectors
                     if node["id"] in energy_for_sector), None)
        final = energy_for_sector.get(shifted_sector["id"])
        expected_inputs = ([final, base] if shift < 0 else [base, final])
        if base is None or final is None or output["inputs"] != expected_inputs:
            raise AdapterError("CF-E120", None, "QCL charge difference has wrong direction",
                               "Return E(N-1)-E(N) for IP or E(N)-E(N+1) for EA")

    def _check_ranking_question(self, plan: Any) -> None:
        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if (model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
                model.frozen_core != 0 or model.dft_functional is not None):
            raise AdapterError("CF-E120", None, "Signed ranking model is unsupported by QCL",
                               "Use a matching HF molecular model")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}
        output = nodes[plan._output.node_id] if plan.output_type == "ranking" else {}
        if (output.get("op") != "rank" or
                len(output["inputs"]) != len(target.items) or
                len([node for node in nodes.values() if node["op"] == "molecule"]) !=
                len(target.items)):
            raise AdapterError("CF-E120", None, "QCL ranking differs from signed items",
                               "Rank one energy per signed molecule in the signed order")

        def ancestors(identifier: int) -> set[int]:
            result = {identifier}
            for input_id in nodes[identifier]["inputs"]:
                result.update(ancestors(input_id))
            return result

        used: set[int] = set()
        for energy_id, item in zip(output["inputs"], target.items, strict=True):
            energy = nodes[energy_id]
            if energy["op"] not in ("scf.energy", "fci.energy", "sqd.diagonalize"):
                raise AdapterError("CF-E120", None, "QCL ranking uses a non-energy score",
                                   "Rank molecular electronic energies")
            lineage = ancestors(energy_id)
            used.update(lineage)
            molecules = [nodes[index]["params"] for index in lineage
                         if nodes[index]["op"] == "molecule"]
            expected = {"atoms": item.molecule.get("atoms"), "basis": model.basis,
                        "charge": item.molecule.get("charge", 0),
                        "spin": item.molecule.get("spin", 0),
                        "unit": item.molecule.get("unit", "angstrom")}
            if len(molecules) != 1 or molecules[0] != expected:
                raise AdapterError("CF-E120", None, "QCL ranking item differs from signed task",
                                   "Use the frozen molecule and order for every item")
        if any(node["kind"] == "compute" and node["id"] not in used
               for node in nodes.values()):
            raise AdapterError("CF-E120", None, "QCL ranking has unrelated calculations",
                               "Return only energies used by the signed ranking")

    def _check_energy_collection(self, plan: Any) -> None:
        import qcl
        from chemflow.domains.reactivity import representatives

        target = self.record.spec.target
        model = ElectronicModel.model_validate(target.electronic_model)
        if (model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
                model.frozen_core != 0 or model.dft_functional is not None or
                model.reference != "RHF"):
            raise AdapterError("CF-E120", None, "Signed fragment model is unsupported by QCL",
                               "Use an all-electron HF basis model")
        if target.kind == "bond_dissociation":
            parent = qcl.Molecule(target.molecule["atoms"], model.basis,
                                  target.molecule.get("charge", 0),
                                  target.molecule.get("spin", 0),
                                  target.molecule.get("unit", "angstrom"))
            bonds = [tuple(item["atoms"]) for item in
                     representatives(target.molecule, target.bonds)]
            expected = []
            for bond in bonds:
                first, second = parent.cleave(bond)
                expected.append([(1, first), (1, second), (-1, parent)])
        elif target.kind == "interaction_energy":
            from chemflow.domains.drug import component_specs

            components = component_specs(target.complex, target.fragments,
                                         target.electronic_model, target.counterpoise)
            molecules = [qcl.Molecule(item["atoms"], model.basis,
                                      unit=item.get("unit", "angstrom"))
                         for item in components]
            expected = [[(1, molecules[0]), (-1, molecules[1]),
                         (-1, molecules[2])]]
        elif target.kind == "interaction_scan":
            if not target.counterpoise:
                raise AdapterError("CF-E120", None, "QCL scan requires counterpoise",
                                   "Use a signed counterpoise interaction scan")
            parent = qcl.Molecule(target.complex["atoms"], model.basis,
                                  target.complex.get("charge", 0),
                                  target.complex.get("spin", 0),
                                  target.complex.get("unit", "angstrom"))
            first, second = target.fragments
            expected = []
            for separation in target.separations:
                geometry = parent.set_separation(first, second, separation)
                expected.append([(1, geometry), (-1, geometry.ghost(second)),
                                 (-1, geometry.ghost(first))])
        else:
            if target.mode != "rigid":
                raise AdapterError("CF-E120", None, "QCL PES requires a rigid scan",
                                   "Use a signed rigid coordinate scan")
            parent = qcl.Molecule(target.molecule["atoms"], model.basis,
                                  target.molecule.get("charge", 0),
                                  target.molecule.get("spin", 0),
                                  target.molecule.get("unit", "angstrom"))
            edit = getattr(parent, f"set_{target.coordinate.type}")
            expected = [[(1, edit(*target.coordinate.atoms, value))]
                        for value in target.coordinate.values]
        outputs = ([plan._output] if target.kind == "interaction_energy" else plan._output)
        if (not isinstance(outputs, list) or len(outputs) != len(expected) or
                any(not isinstance(item, qcl._frontend._Value) or item.kind != "estimate"
                    for item in outputs)):
            raise AdapterError("CF-E120", None, "QCL collection differs from signed task",
                               "Return one energy difference per signed item")
        nodes = {node["id"]: node for node in plan.graph_json()["nodes"]}

        def ancestors(identifier: int) -> set[int]:
            found = {identifier}
            for input_id in nodes[identifier]["inputs"]:
                found.update(ancestors(input_id))
            return found

        def terms(identifier: int, sign: int = 1) -> list[tuple[int, dict[str, Any]]]:
            node = nodes[identifier]
            if node["op"] in ("estimate.add", "estimate.sub"):
                left, right = node["inputs"]
                return (terms(left, sign) + terms(
                    right, sign if node["op"] == "estimate.add" else -sign))
            if node["op"] == "copy":
                return terms(node["inputs"][0], sign)
            if node["op"] not in ("scf.energy", "fci.energy", "sqd.diagonalize"):
                raise AdapterError("CF-E120", None, "QCL fragment result is not an energy sum",
                                   "Use signed component electronic energies")
            molecules = [nodes[index]["params"] for index in ancestors(identifier)
                         if nodes[index]["op"] == "molecule"]
            if len(molecules) != 1:
                raise AdapterError("CF-E120", None, "QCL fragment energy is ambiguous",
                                   "Use one molecule per component energy")
            return [(sign, molecules[0])]

        def normalized(items: list[tuple[int, Any]]) -> dict[str, int]:
            counts: dict[str, int] = {}
            for sign, molecule in items:
                params = asdict(molecule) if is_dataclass(molecule) else molecule
                key = json.dumps(params, sort_keys=True)
                counts[key] = counts.get(key, 0) + sign
            return {key: value for key, value in counts.items() if value}

        used: set[int] = set()
        for output, required in zip(outputs, expected, strict=True):
            used.update(ancestors(output.node_id))
            if normalized(terms(output.node_id)) != normalized(required):
                raise AdapterError("CF-E120", None, "QCL fragment coefficients differ from task",
                                   "Use the signed fragments and energy-difference signs")
        if any(node["kind"] == "compute" and node["id"] not in used
               for node in nodes.values()):
            raise AdapterError("CF-E120", None, "QCL fragment plan has unrelated calculations",
                               "Return only signed fragment energies")

    def _check_budget(self, estimate: dict[str, Any], plan: Any | None = None) -> None:
        budget = self.record.spec.budget
        wall_s = budget.get("wall_s", 300)
        if not math.isfinite(wall_s) or wall_s <= 0:
            raise AdapterError("CF-E070", None, "QCL wall-time budget is not positive",
                               "Use a positive signed wall_s budget")
        if "cpu_core_s" in budget and (not math.isfinite(budget["cpu_core_s"]) or
                                       budget["cpu_core_s"] <= 0):
            raise AdapterError("CF-E070", None, "QCL CPU core-time budget is not positive",
                               "Use a positive signed cpu_core_s budget")
        uses_gpu = bool(_gpu_node_ids(plan)) if plan is not None else False
        if uses_gpu and "gpu_s" in budget and (
                not math.isfinite(budget["gpu_s"]) or budget["gpu_s"] <= 0):
            raise AdapterError("CF-E070", None, "QCL GPU budget is not positive",
                               "Use a positive signed gpu_s budget")
        for dimension, predicted in (("qpu_shots", estimate["shots"]),
                                     ("qpu_s", estimate["qpu_s"])):
            limit = budget.get(dimension, 0)
            if not math.isfinite(limit) or limit < 0 or (predicted is not None and
                                                         predicted > limit):
                raise AdapterError("CF-E070", None, f"QCL {dimension} exceeds task budget",
                                   "Reduce the workload or use the approved task budget")
            if predicted is None and estimate["shots"]:
                raise AdapterError("CF-E070", None, f"QCL {dimension} is unknown",
                                   "Use a calibrated device resource estimate")
        if "qpu_cost" in budget and (estimate["currency"] is not None or
                                     estimate["qpu_cost"] is None):
            limit = budget["qpu_cost"]
            cost = estimate["qpu_cost"]
            if not math.isfinite(limit) or limit < 0 or cost is None or cost > limit:
                raise AdapterError("CF-E070", None, "QCL cloud cost exceeds task budget",
                                   "Reduce the cloud workload or use the signed cost limit")

    def _check_effects(self, plan: Any) -> None:
        allowed = set(self.record.spec.allowed_effects)
        for node in plan.graph_json()["nodes"]:
            if node["op"] not in ("sample", "vqe.sampled", "estimate.quantum"):
                continue
            effect = "simulator" if node["device"]["simulator_host"] else "network"
            if effect not in allowed:
                raise AdapterError("CF-E120", None, f"QCL {effect} effect is not signed",
                                   "Use a task that explicitly permits this effect")


def run_cli(args: Any) -> None:
    import qcl
    from chemflow.agent.journal import Journal
    from chemflow.agent.memory import ResultsDB
    from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetExceeded, BudgetLedger
    from chemflow.signing import TaskVerifier as SignatureVerifier
    from chemflow.types import CostRecord

    record = TaskRecord.model_validate_json(args.task.read_text())
    verifier = SignatureVerifier(args.public_key.read_bytes())
    agent = QclAgent(record, verifier)
    cluster = qcl.Cluster.load(args.cluster) if args.cluster else qcl.Cluster.local()
    device_id = getattr(args, "device", None)
    if getattr(args, "candidate_program", None) and not getattr(args, "screen", False):
        raise ValueError("CF-E080: --candidate-program requires --screen")
    if getattr(args, "screen", False):
        _run_cli_screen(args, agent, cluster, device_id)
        return
    if args.program is None:
        workflow, positional, keyword = _builtin_program(record.spec, cluster, device_id=device_id)
    else:
        if device_id is not None:
            raise ValueError("--device applies to built-in QCL workflows only")
        workflow, positional, keyword = _load_qcl_program(args.program, record.spec,
                                                          cluster)
    review = agent.compile(workflow, *positional, **keyword)
    args.out.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.out / "journal.jsonl")
    (args.out / "qcl-review.json").write_text(json.dumps({
        "record_hash": review.record_hash, "plan_hash": review.plan_hash,
        "output_hash": review.output_hash,
        "reference_plan_hash": review.reference_hash,
        "reference_output_hash": review.reference_output_hash,
        "explanation": review.explanation, "estimate": review.estimate,
    }, indent=2, sort_keys=True) + "\n")
    journal.append(state="QCL_REVIEW", role="orchestrator",
                   observation={"record_hash": record.hash, "plan_hash": review.plan_hash},
                   decision="reviewed")
    summary = {"record_hash": record.hash, "plan_hash": review.plan_hash,
               "estimate": review.estimate,
               "paths": {"review": str(args.out / "qcl-review.json"),
                         "journal": str(journal.path),
                         "summary": str(args.out / "summary.json")}}
    if args.dry_run:
        (args.out / "summary.json").write_text(json.dumps(
            {**summary, "decision": "reviewed"}, indent=2, sort_keys=True) + "\n")
        print(review.plan_hash)
        return
    limits = {key: value for key, value in record.spec.budget.items()
              if key in RESOURCE_DIMENSIONS}
    ledger = BudgetLedger(limits, args.out / f"{record.content_hash}.ledger.json")
    ledger_lock = args.out / f"{record.content_hash}.ledger.lock"
    gpu_nodes = _gpu_node_ids(review.plan)
    uses_gpu = bool(gpu_nodes)
    reservation = CostRecord(qpu_shots=review.estimate["shots"],
                             qpu_s=review.estimate["qpu_s"])
    with ledger_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger.reload()
        try:
            ledger.reserve(reservation)
            remaining_wall = record.spec.budget.get("wall_s", 300) - ledger.spent.get("wall_s", 0)
            if remaining_wall <= 0:
                raise BudgetExceeded("CF-E070: wall_s budget exceeded")
            remaining_gpu = (record.spec.budget["gpu_s"] - ledger.spent.get("gpu_s", 0)
                             if uses_gpu and "gpu_s" in record.spec.budget else None)
            if remaining_gpu is not None and remaining_gpu <= 0:
                raise BudgetExceeded("CF-E070: gpu_s budget exceeded")
            remaining_cpu = (record.spec.budget["cpu_core_s"] -
                             ledger.spent.get("cpu_core_s", 0)
                             if "cpu_core_s" in record.spec.budget else None)
            if remaining_cpu is not None and remaining_cpu <= 0:
                raise BudgetExceeded("CF-E070: cpu_core_s budget exceeded")
            _reserve_cloud_cost(ledger, record.spec.budget, review.estimate)
        except BudgetExceeded as error:
            journal.append(state="QCL_REFUSED", role="orchestrator",
                           observation={"plan_hash": review.plan_hash},
                           decision="refused", reason=str(error))
            raise AdapterError("CF-E070", None, str(error),
                               "Use a task with enough remaining resource budget") from error
        ledger.record(reservation)
        started = time.monotonic()
        try:
            result = agent.run(review, timeout_s=min(remaining_wall, remaining_gpu)
                               if remaining_gpu is not None else remaining_wall,
                               cpu_core_s=remaining_cpu)
        except Exception as error:
            journal.append(state="QCL_FAILED", role="orchestrator",
                           observation={"plan_hash": review.plan_hash},
                           decision="failed", reason=type(error).__name__)
            elapsed = time.monotonic() - started
            cpu_seconds = (review.plan.cpu_core_s or 0) + (
                review.reference_plan.cpu_core_s or 0
                if review.reference_plan is not None else 0)
            ledger.record(CostRecord(wall_s=elapsed,
                                     gpu_s=elapsed if uses_gpu else 0,
                                     cpu_core_s=cpu_seconds))
            raise
        gpu_seconds = sum(item["duration_s"] for item in review.plan.profile()
                          if item["id"] in gpu_nodes)
        cpu_seconds = (review.plan.cpu_core_s or 0) + (
            review.reference_plan.cpu_core_s or 0
            if review.reference_plan is not None else 0)
        ledger.record(CostRecord(wall_s=time.monotonic() - started,
                                 gpu_s=gpu_seconds, cpu_core_s=cpu_seconds))
    def serializable(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        return value

    (args.out / "qcl-result.json").write_text(json.dumps(serializable(result),
                                                        indent=2, sort_keys=True) + "\n")
    gates = result.provenance.get("signed_gates", []) if hasattr(result, "provenance") else []
    decision = ("gate_passed" if all(gate["passed"] for gate in gates) else "gate_failed") if gates else "executed"
    profile = review.plan.profile()
    (args.out / "qcl-profile.json").write_text(json.dumps(profile, indent=2,
                                                         sort_keys=True) + "\n")
    error_budget_path = None
    if review.plan.output_type == "estimate":
        error_budget_path = args.out / "qcl-error-budget.json"
        error_budget_path.write_text(json.dumps(_agent_error_budget(review, result),
                                                indent=2, sort_keys=True) + "\n")
    value = (f"{result.value:.12g} {result.unit}" if isinstance(result, qcl.Estimate) else
             f"{len(result.omega)} frequencies, {result.unit}" if isinstance(
                 result, qcl.SpectralEstimate) else type(result).__name__)
    report = ["# QCL result", "", f"Task: {record.spec.question}",
              f"Decision: {decision}", f"Plan: {review.plan_hash}",
              f"Result: {value}", "", "## Signed gates", ""]
    report.extend(f"- {gate['kind']}: {'pass' if gate['passed'] else 'fail'} "
                  f"(error {gate['error']:.6g}, limit {gate['threshold']:.6g} {gate['unit']})"
                  for gate in gates)
    if not gates:
        report.append("- No numerical gate was signed.")
    report.extend(["", "## Execution", ""])
    report.extend(f"- {operation['op']} on {operation['device']} "
                  f"({operation['implementation']})"
                  for operation in review.explanation["operations"])
    (args.out / "report.md").write_text("\n".join(report) + "\n")
    journal.append(state="QCL_EXECUTED", role="orchestrator",
                   observation={"plan_hash": review.plan_hash, "signed_gates": gates},
                   decision=decision,
                   outputs=[{"path": str(args.out / "qcl-result.json")}])
    results_path = args.out / "results.sqlite"
    results = ResultsDB(results_path)
    result_id = results.add_qcl_run(
        record_hash=record.hash, plan_hash=review.plan_hash,
        target_kind=record.spec.target.kind, decision=decision,
        result=serializable(result), profile=profile, spent=ledger.spent)
    results.connection.close()
    (args.out / "summary.json").write_text(json.dumps({
        **summary, "decision": decision, "signed_gates": gates,
        "result_id": result_id,
        "spent": ledger.spent, "limits": record.spec.budget,
        "paths": {**summary["paths"], "result": str(args.out / "qcl-result.json"),
                  "profile": str(args.out / "qcl-profile.json"),
                  **({"error_budget": str(error_budget_path)}
                     if error_budget_path is not None else {}),
                  "report": str(args.out / "report.md"),
                  "results": str(results_path),
                  "ledger": str(ledger.path)},
    }, indent=2, sort_keys=True) + "\n")
    print(args.out / "qcl-result.json")


def _load_qcl_program(path: Any, task_spec: Any, cluster: Any
                      ) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("chemflow_qcl_program", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load QCL program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError("QCL program must define build(task_spec, cluster)")
    return build(task_spec, cluster)


def _run_cli_screen(args: Any, agent: QclAgent, cluster: Any,
                    device_id: str | None) -> None:
    import qcl
    from chemflow.agent.journal import Journal
    from chemflow.agent.memory import ResultsDB
    from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetExceeded, BudgetLedger
    from chemflow.types import CostRecord

    record = agent.record
    target = record.spec.target
    reviews = {}
    authored = ([args.program] if args.program is not None else []) + list(
        getattr(args, "candidate_program", None) or [])
    if authored:
        if device_id is not None:
            raise ValueError("CF-E080: --device applies to built-in QCL screens only")
        for path in authored:
            name = path.stem
            if name in reviews:
                raise ValueError("CF-E080: QCL candidate program names must be unique")
            workflow, positional, keyword = _load_qcl_program(path, record.spec, cluster)
            reviews[name] = agent.compile(workflow, *positional, **keyword)
    else:
        if target.kind != "molecular_ground_energy":
            raise ValueError("CF-E080: Built-in QCL screen supports molecular ground energy")
        if not cluster.cpus:
            raise ValueError("CF-E080: QCL screen requires a CPU device")
        device = cluster[device_id] if device_id is not None else cluster.cpus[0]
        model = ElectronicModel.model_validate(target.electronic_model)
        if (device.kind != "cpu" or not isinstance(target.active, dict) or
                not record.spec.gates or model.dft_functional is not None or
                model.reference not in ("RHF", "ROHF")):
            raise ValueError("CF-E080: QCL screen needs a fixed-space HF ground task, "
                             "numeric gate and CPU device")
        for name, solver in (("hf", "hf"), ("pyscf_fci", "pyscf"),
                             ("native_ci", "native")):
            workflow, positional, keyword = _builtin_program(
                record.spec, cluster, device_id=device.id, ground_solver=solver)
            try:
                reviews[name] = agent.compile(workflow, *positional, **keyword)
            except qcl.CompileError as error:
                if solver != "native" or "QCL-E209" not in str(error):
                    raise

    _check_screen_currency(reviews)

    args.out.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.out / "journal.jsonl")
    review_path = args.out / "qcl-review.json"
    review_path.write_text(json.dumps({
        "record_hash": record.hash,
        "candidates": {name: {"plan_hash": review.plan_hash,
                              "output_hash": review.output_hash,
                              "reference_plan_hash": review.reference_hash,
                              "reference_output_hash": review.reference_output_hash,
                              "explanation": review.explanation,
                              "estimate": review.estimate}
                       for name, review in reviews.items()},
    }, indent=2, sort_keys=True) + "\n")
    journal.append(state="QCL_REVIEW", role="orchestrator",
                   observation={"record_hash": record.hash,
                                "candidates": list(reviews)}, decision="reviewed")
    summary_path = args.out / "summary.json"
    if args.dry_run:
        summary_path.write_text(json.dumps({
            "record_hash": record.hash, "decision": "reviewed",
            "candidates": list(reviews), "paths": {"review": str(review_path),
                                                   "journal": str(journal.path)}},
            indent=2, sort_keys=True) + "\n")
        print(review_path)
        return

    limits = {key: value for key, value in record.spec.budget.items()
              if key in RESOURCE_DIMENSIONS}
    ledger = BudgetLedger(limits, args.out / f"{record.content_hash}.ledger.json")
    lock_path = args.out / f"{record.content_hash}.ledger.lock"
    qpu_seconds = [review.estimate["qpu_s"] for review in reviews.values()]
    if any(value is None for value in qpu_seconds):
        raise AdapterError("CF-E070", None, "QCL screen has unknown QPU time",
                           "Use a calibrated QPU device estimate")
    reservation = CostRecord(
        qpu_shots=sum(review.estimate["shots"] for review in reviews.values()),
        qpu_s=sum(qpu_seconds))
    uses_gpu = any(_gpu_node_ids(review.plan) for review in reviews.values())
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger.reload()
        remaining = {key: value - ledger.spent.get(key, 0)
                     for key, value in limits.items()}
        if "qpu_cost" in record.spec.budget:
            remaining["qpu_cost"] = (record.spec.budget["qpu_cost"] -
                                     ledger.spent.get("qpu_cost", 0))
        if (remaining.get("wall_s", 300) <= 0 or
                "cpu_core_s" in remaining and remaining["cpu_core_s"] <= 0 or
                uses_gpu and "gpu_s" in remaining and remaining["gpu_s"] <= 0):
            raise AdapterError("CF-E070", None, "QCL screen has no remaining runtime budget",
                               "Use a task with remaining wall, CPU and GPU time")
        try:
            ledger.reserve(reservation)
            _reserve_cloud_cost(ledger, record.spec.budget, {
                "qpu_cost": sum(review.estimate["qpu_cost"] for review in reviews.values()
                                if review.estimate["qpu_cost"] is not None),
                "currency": next((review.estimate["currency"]
                                  for review in reviews.values()
                                  if review.estimate["currency"] is not None), None),
            })
        except BudgetExceeded as error:
            raise AdapterError("CF-E070", None, str(error),
                               "Reduce the screen or use remaining QPU budget") from error
        ledger.record(reservation)
        started = time.monotonic()
        try:
            selection = agent.select(reviews, limits=remaining)
        except Exception:
            cpu_seconds = sum((review.plan.cpu_core_s or 0) +
                              (review.reference_plan.cpu_core_s or 0
                               if review.reference_plan is not None else 0)
                              for review in reviews.values())
            elapsed = time.monotonic() - started
            ledger.record(CostRecord(wall_s=elapsed, cpu_core_s=cpu_seconds,
                                     gpu_s=elapsed if uses_gpu else 0))
            journal.append(state="QCL_FAILED", role="orchestrator",
                           observation={"candidates": list(reviews)},
                           decision="failed")
            raise
        elapsed = time.monotonic() - started
        ledger.record(CostRecord(
            wall_s=elapsed,
            cpu_core_s=sum(item.cpu_core_s for item in selection.outcomes),
            gpu_s=sum(item.gpu_s for item in selection.outcomes)))

    screen = {item.name: asdict(item) for item in selection.outcomes}
    screen_path = args.out / "qcl-screen.json"
    screen_path.write_text(json.dumps(screen, indent=2, sort_keys=True) + "\n")
    report = ["# QCL candidate screen", "", f"Task: {record.spec.question}",
              f"Decision: {selection.decision}",
              f"Selected: {selection.candidate or 'none'}", "", "## Candidates", ""]
    report.extend(f"- {item.name}: {'pass' if item.passed else 'fail'}, "
                  f"{item.compute_s:.3f} compute-s, {item.wall_s:.3f} wall-s, "
                  f"{item.cpu_core_s:.3f} CPU core-s, "
                  f"plan {item.plan_hash}" for item in selection.outcomes)
    report_path = args.out / "report.md"
    report_path.write_text("\n".join(report) + "\n")
    journal.append(state="QCL_SCREENED", role="orchestrator",
                   observation={"candidate": selection.candidate,
                                "screen": str(screen_path)},
                   decision=selection.decision)
    paths = {"review": str(review_path), "journal": str(journal.path),
             "screen": str(screen_path), "report": str(report_path),
             "ledger": str(ledger.path), "summary": str(summary_path)}
    result_id = None
    chosen = selection.candidate
    if chosen is not None:
        result_path = args.out / "qcl-result.json"
        result = asdict(selection.result) if is_dataclass(selection.result) else selection.result
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        profile_path = args.out / "qcl-profile.json"
        profile = reviews[chosen].plan.profile()
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
        if reviews[chosen].plan.output_type == "estimate":
            error_budget_path = args.out / "qcl-error-budget.json"
            error_budget_path.write_text(json.dumps(
                _agent_error_budget(reviews[chosen], selection.result), indent=2,
                sort_keys=True) + "\n")
            paths["error_budget"] = str(error_budget_path)
        results_path = args.out / "results.sqlite"
        results = ResultsDB(results_path)
        result_id = results.add_qcl_run(
            record_hash=record.hash, plan_hash=reviews[chosen].plan_hash,
            target_kind=target.kind, decision=selection.decision,
            result=result, profile=profile, spent=ledger.spent)
        results.connection.close()
        paths.update(result=str(result_path), profile=str(profile_path),
                     results=str(results_path))
    summary_path.write_text(json.dumps({
        "record_hash": record.hash,
        "plan_hash": reviews[chosen].plan_hash if chosen is not None else None,
        "decision": selection.decision, "candidate": chosen,
        "result_id": result_id, "spent": ledger.spent,
        "limits": record.spec.budget, "paths": paths,
    }, indent=2, sort_keys=True) + "\n")
    print(f"{selection.decision} {report_path}")


def _builtin_program(task_spec: Any, cluster: Any, *, device_id: str | None = None,
                     ground_solver: str = "pyscf"
                     ) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    import qcl

    target = task_spec.target
    if not cluster.cpus:
        raise ValueError("CF-E080: Built-in QCL workflows require a CPU device")
    cpu = cluster[device_id] if device_id is not None else cluster.cpus[0]

    if target.kind == "spectral_function":
        @qcl.workflow
        def spectrum(model: Any, sector: Any, device: Any) -> Any:
            return qcl.dynamics.spectral_exact(
                model, sector, orbital=target.orbital, spin=target.spin,
                omega=target.omega, eta=target.eta).to(device)

        return spectrum, (target.model["impurity_model"], target.sector, cpu), {}

    if target.kind == "periodic_energy":
        @qcl.workflow
        def periodic_energy(system: Any, device: Any) -> Any:
            return qcl.periodic.scf(system, kmesh=target.kmesh, method=target.method,
                                    exxdiv=target.exxdiv).to(device)

        return periodic_energy, (target.system, cpu), {}

    if target.kind == "aqueous_solubility":
        @qcl.workflow
        def solubility(smiles: str, device: Any) -> Any:
            if target.model == "gse":
                return qcl.solubility.gse(smiles, target.melting_point_C).to(device)
            return qcl.solubility.esol(smiles).to(device)

        return solubility, (target.molecule_smiles, cpu), {}

    if target.kind == "binding_free_energy":
        @qcl.workflow
        def binding(receptor: Any, ligand: Any, device: Any) -> Any:
            return qcl.binding.mmgbsa(
                receptor, ligand, frames=target.frames, minimize=target.minimize,
                seed=target.seed, forcefield=target.forcefield, ph=target.ph).to(device)

        return binding, (target.receptor, target.ligand, cpu), {}

    if target.kind in ("metabolic_sites", "solvation_free_energy"):
        model = ElectronicModel.model_validate(target.electronic_model)
        molecule = qcl.Molecule(target.molecule["atoms"], model.basis,
                                target.molecule.get("charge", 0),
                                target.molecule.get("spin", 0),
                                target.molecule.get("unit", "angstrom"))
        if target.kind == "metabolic_sites":
            @qcl.workflow
            def sites(mol: Any, device: Any) -> Any:
                return qcl.metabolism.sites(mol, level=target.level.method).to(device)

            return sites, (molecule, cpu), {}

        if target.level.method == "gfn2":
            @qcl.workflow
            def gfn2_solvation(mol: Any, device: Any) -> Any:
                gas = qcl.engines.tblite.gfn2.energy(mol).to(device)
                solvent = (qcl.ALPB(target.solvent.name)
                           if target.solvent.model == "alpb" else
                           qcl.GBSA(target.solvent.name))
                liquid = qcl.engines.tblite.gfn2.energy(
                    mol, solvent=solvent).to(device)
                return liquid - gas

            return gfn2_solvation, (molecule, cpu), {}

        if (target.level.method not in ("hf", "dft") or
                target.solvent.model not in ("pcm", "smd")):
            raise ValueError("CF-E080: Built-in QCL solvation supports HF/DFT PCM-SMD or GFN2 ALPB-GBSA")
        solvent = (qcl.SMD(target.solvent.name) if target.solvent.model == "smd" else
                   qcl.PCM(target.solvent.name, eps=target.solvent.eps))

        @qcl.workflow
        def solvation(mol: Any, device: Any) -> Any:
            if target.level.method == "dft":
                method = qcl.dft.uks if mol.spin else qcl.dft.rks
                gas = method(mol, xc=target.level.functional).to(device).energy
                liquid = method(mol, xc=target.level.functional,
                                solvent=solvent).to(device).energy
                return liquid - gas
            if device.kind == "gpu":
                gas = qcl.scf.rhf(mol).to(device).energy
                liquid = qcl.scf.rhf(mol, solvent=solvent).to(device).energy
                return liquid - gas
            integrals = qcl.int2e(mol).to(device)
            gas = qcl.scf.rhf(mol, eri=integrals).to(device).energy
            liquid = qcl.scf.rhf(mol, eri=integrals,
                                 solvent=solvent).to(device).energy
            return liquid - gas

        return solvation, (molecule, cpu), {}

    if target.kind in ("lattice_ground_energy", "pairing_correlation"):
        lattice_spec = target.lattice
        geometry = lattice_spec.get("geometry", "chain")
        if geometry == "square":
            lattice = qcl.lattice.square(*lattice_spec["shape"])
        elif geometry == "ring":
            lattice = qcl.lattice.ring(lattice_spec["n_sites"])
        elif geometry == "chain":
            lattice = qcl.lattice.chain(lattice_spec["n_sites"])
        else:
            raise ValueError(f"CF-E080: Unsupported QCL lattice geometry: {geometry}")
        n_alpha, n_beta = target.sector["n_alpha"], target.sector["n_beta"]
        sector = qcl.Sector(n=n_alpha + n_beta, sz=(n_alpha - n_beta) / 2)

        @qcl.workflow
        def lattice_result(system: Any, particles: Any, device: Any) -> Any:
            operator = qcl.models.hubbard(system, t=lattice_spec.get("t", 1.0),
                                          U=lattice_spec.get("U", 4.0))
            ground_state = qcl.ed.ground_state(operator, sector=particles).to(device)
            if target.kind == "lattice_ground_energy":
                return ground_state.energy
            observables = {"pair_s_wave": qcl.ops.onsite_pair,
                           "spin_spin": qcl.ops.spin_spin,
                           "density_density": qcl.ops.density_density}
            observable = observables[target.quantity](*target.site_pair)
            return qcl.expect(ground_state, observable).to(device)

        return lattice_result, (lattice, sector, cpu), {}

    if target.kind == "reaction_energy":
        model = ElectronicModel.model_validate(target.electronic_model)
        species = tuple((qcl.Molecule(item.molecule["atoms"], model.basis,
                                      item.molecule.get("charge", 0),
                                      item.molecule.get("spin", 0),
                                      item.molecule.get("unit", "angstrom")),
                         qcl.Active(item.active["n_elec"], item.active["n_orb"]),
                         item.coefficient) for item in target.species)

        @qcl.workflow
        def reaction(items: Any, device: Any) -> Any:
            result = None
            for mol, active_space, coefficient in items:
                mean_field = qcl.scf.hf(mol).to(device)
                space = qcl.active_space(mean_field, active_space).to(device)
                energy = qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy
                term = coefficient * energy
                result = term if result is None else result + term
            return result

        return reaction, (species, cpu), {}

    if target.kind == "reaction_barrier":
        model = ElectronicModel.model_validate(target.electronic_model)
        reactant = next(item for item in target.species if item.role == "minimum")
        transition = next(item for item in target.species if item.role == "ts")

        def molecule(data: dict[str, Any]) -> Any:
            return qcl.Molecule(data["atoms"], model.basis, data.get("charge", 0),
                                data.get("spin", 0), data.get("unit", "angstrom"))

        @qcl.workflow
        def barrier(start: Any, guess: Any, product: Any, device: Any) -> Any:
            minimum = qcl.optimize(start).to(device)
            saddle = qcl.engines.sella.saddle(guess, start, product).to(device)

            def energy(geometry: Any, active: Any) -> Any:
                mean_field = qcl.scf.hf(geometry).to(device)
                space = qcl.active_space(mean_field, active).to(device)
                electronic = qcl.fci.solve(space.hamiltonian,
                                           space.sector).to(device).energy
                if target.thermo is None:
                    return electronic
                modes = qcl.frequencies(geometry).to(device)
                thermal = qcl.thermo.rrho(
                    geometry, modes, temperature_K=target.thermo.temperature_K,
                    pressure_Pa=target.thermo.pressure_Pa,
                    quantity=target.thermo.quantity).to(device)
                return electronic + thermal

            difference = (energy(saddle, qcl.Active(transition.active["n_elec"],
                                                    transition.active["n_orb"])) -
                          energy(minimum, qcl.Active(reactant.active["n_elec"],
                                                     reactant.active["n_orb"])))
            if target.rate_constant:
                return difference, qcl.thermo.eyring(
                    difference, target.thermo.temperature_K).to(device)
            return difference

        return barrier, (molecule(reactant.molecule), molecule(transition.molecule),
                         molecule(target.product), cpu), {}

    if target.kind in ("ranking", "bond_dissociation", "interaction_energy",
                       "interaction_scan", "pes_scan"):
        model = ElectronicModel.model_validate(target.electronic_model)

        def molecule(data: dict[str, Any]) -> Any:
            return qcl.Molecule(data["atoms"], model.basis, data.get("charge", 0),
                                data.get("spin", 0), data.get("unit", "angstrom"))

        @qcl.workflow
        def rank_items(items: Any, device: Any) -> Any:
            values = qcl.map(lambda item, lane: qcl.scf.hf(item).to(lane).energy,
                             items).to([device])
            return qcl.rank(values)

        if target.kind == "ranking":
            return rank_items, ([molecule(item.molecule) for item in target.items], cpu), {}

        @qcl.workflow
        def bond_energies(parent: Any, bonds: Any, device: Any) -> Any:
            parent_energy = qcl.scf.hf(parent).to(device).energy

            def point(bond: Any, lane: Any) -> Any:
                first, second = parent.cleave(bond)
                return (qcl.scf.hf(first).to(lane).energy +
                        qcl.scf.hf(second).to(lane).energy - parent_energy)

            return qcl.map(point, bonds).to([device])

        if target.kind == "bond_dissociation":
            from chemflow.domains.reactivity import representatives

            bonds = [tuple(item["atoms"]) for item in
                     representatives(target.molecule, target.bonds)]
            return bond_energies, (molecule(target.molecule), bonds, cpu), {}

        @qcl.workflow
        def interaction_components(items: Any, device: Any) -> Any:
            energies = [qcl.scf.hf(item).to(device).energy for item in items]
            return energies[0] - energies[1] - energies[2]

        if target.kind == "interaction_energy":
            from chemflow.domains.drug import component_specs

            components = component_specs(target.complex, target.fragments,
                                         target.electronic_model, target.counterpoise)
            return interaction_components, ([molecule(item) for item in components], cpu), {}

        @qcl.workflow
        def interaction_curve(parent: Any, first: Any, second: Any,
                              separations: Any, device: Any) -> Any:
            def point(distance: float, lane: Any) -> Any:
                geometry = parent.set_separation(first, second, distance)
                return (qcl.scf.hf(geometry).to(lane).energy -
                        qcl.scf.hf(geometry.ghost(second)).to(lane).energy -
                        qcl.scf.hf(geometry.ghost(first)).to(lane).energy)

            return qcl.map(point, separations).to([device])

        if target.kind == "interaction_scan":
            return interaction_curve, (molecule(target.complex), *target.fragments,
                                       target.separations, cpu), {}

        @qcl.workflow
        def rigid_scan(parent: Any, values: Any, device: Any) -> Any:
            edit = getattr(parent, f"set_{target.coordinate.type}")
            return qcl.map(lambda value, lane: qcl.scf.hf(
                edit(*target.coordinate.atoms, value)).to(lane).energy,
                values).to([device])

        return rigid_scan, (molecule(target.molecule), target.coordinate.values, cpu), {}

    if target.kind not in ("molecular_ground_energy", "frontier_orbitals",
                           "ionization_energy", "electron_affinity"):
        raise ValueError(f"CF-E080: No built-in QCL workflow for {target.kind}")
    model = ElectronicModel.model_validate(target.electronic_model)
    molecule = qcl.Molecule(target.molecule["atoms"], model.basis,
                            target.molecule.get("charge", 0),
                            target.molecule.get("spin", 0),
                            target.molecule.get("unit", "angstrom"))
    if target.kind == "frontier_orbitals":
        @qcl.workflow
        def frontier(mol: Any, device: Any) -> Any:
            mean_field = qcl.scf.rhf(mol).to(device)
            if target.quantity == "gap":
                return mean_field.lumo - mean_field.homo
            return getattr(mean_field, target.quantity)

        return frontier, (molecule, cpu), {}
    if target.kind in ("ionization_energy", "electron_affinity"):
        @qcl.workflow
        def charge_difference(mol: Any, device: Any) -> Any:
            mean_field = qcl.scf.hf(mol).to(device)
            space = qcl.active_space(mean_field, qcl.Active(max_orbitals=999)).to(device)
            base = qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy
            if target.kind == "ionization_energy":
                final = qcl.fci.solve(space.hamiltonian,
                                      space.sector.add_electrons(-1)).to(device).energy
                return final - base
            final = qcl.fci.solve(space.hamiltonian,
                                  space.sector.add_electrons(1)).to(device).energy
            return base - final

        return charge_difference, (molecule, cpu), {}
    if ground_solver not in ("hf", "pyscf", "native"):
        raise ValueError("CF-E080: Unknown built-in QCL ground solver")
    active = (qcl.Active(target.active["n_elec"], target.active["n_orb"])
              if ground_solver != "hf" and isinstance(target.active, dict) and task_spec.gates and
              model.dft_functional is None and model.reference in ("RHF", "ROHF") else None)
    if cpu.kind == "qpu" and (not isinstance(target.active, dict) or
                               model.dft_functional is not None or
                               model.reference not in ("RHF", "ROHF")):
        raise ValueError("CF-E080: QPU SQD needs a fixed HF active space")
    qpu_shots = task_spec.budget.get("qpu_shots", 0)
    if cpu.kind == "qpu" and (type(qpu_shots) not in (int, float) or
                               not math.isfinite(qpu_shots) or qpu_shots < 1 or
                               int(qpu_shots) != qpu_shots):
        raise ValueError("CF-E070: QPU SQD needs a signed positive shot budget")
    qpu_shots = int(qpu_shots) if cpu.kind == "qpu" else 0

    @qcl.workflow
    def ground(mol: Any, cpu: Any) -> Any:
        if cpu.kind == "qpu":
            host = cluster.cpus[0]
            mean_field = qcl.scf.hf(mol).to(host)
            space = qcl.active_space(
                mean_field, qcl.Active(target.active["n_elec"],
                                       target.active["n_orb"])).to(host)
            samples = qcl.sample(qcl.kernels.hf(space), shots=qpu_shots,
                                 seed=0).to(cpu)
            return qcl.sqd.diagonalize(space.hamiltonian, samples,
                                       space.sector).to(host)
        if model.dft_functional is not None:
            calculation = (qcl.dft.uks(mol, xc=model.dft_functional) if mol.spin else
                           qcl.dft.rks(mol, xc=model.dft_functional))
        elif model.reference == "UHF":
            calculation = qcl.scf.uhf(mol)
        elif model.reference == "ROHF":
            calculation = qcl.scf.rohf(mol)
        else:
            calculation = qcl.scf.rhf(mol)
        mean_field = calculation.to(cpu)
        if active is None:
            return mean_field.energy
        host = cluster.cpus[0]
        space = qcl.active_space(
            mean_field.to(host) if cpu.kind == "gpu" else mean_field,
            active).to(host)
        operator = space.hamiltonian.to(cpu) if cpu.kind == "gpu" else space.hamiltonian
        sector = space.sector.to(cpu) if cpu.kind == "gpu" else space.sector
        return qcl.fci.solve(operator, sector, engine=ground_solver).to(cpu).energy

    return ground, (molecule, cpu), {}
