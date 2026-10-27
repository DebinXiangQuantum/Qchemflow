"""Whole-molecule electrostatic QM/MM single points, delegated to the Q1 plugin."""

import io
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from chemflow.diagnostics import AdapterError, Diagnostic
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, RefinementRule
from chemflow.runtime import ExecutionResult

from ._adapters import estimate, register_adapter


class MMSystem(BaseModel, frozen=True):
    id: str
    system_ref: str
    topology_ref: str


def mm_system(store, pdb, forcefields, nonbonded="NoCutoff"):
    try:
        import openmm
        from openmm import app
    except ImportError as exc:
        raise AdapterError("CF-E080", None, "OpenMM is unavailable", "Install the mm extra") from exc
    text = pdb if "\n" in pdb else Path(pdb).read_text()
    structure = app.PDBFile(io.StringIO(text))
    if nonbonded not in ("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald", "LJPME"):
        raise ValueError("Unsupported OpenMM nonbonded method")
    system = app.ForceField(*forcefields).createSystem(
        structure.topology, nonbondedMethod=getattr(app, nonbonded))
    xml = openmm.XmlSerializer.serialize(system)
    system_ref = store.put(np.frombuffer(xml.encode(), dtype=np.uint8))
    topology_ref = store.put(np.frombuffer(text.encode(), dtype=np.uint8))
    return MMSystem(id=content_hash((system_ref, topology_ref)),
                    system_ref=system_ref, topology_ref=topology_ref)


def energy_keys(system_id, positions_nm, qm_atoms, basis, method="rhf", cas=None,
                charge=0, spin=0, **unused):
    keys = {"system_id": system_id, "snapshot_hash": content_hash(
                np.asarray(positions_nm, dtype=float).tolist()),
            "qm_region": content_hash(sorted(qm_atoms)), "embedding": "electrostatic",
            "model_id": content_hash((basis, charge, spin, cas if method == "casci" else None))}
    keys["hamiltonian_id"] = content_hash(keys)
    return keys


def _check_region(topology, qm_atoms):
    selected = set(qm_atoms)
    if (not selected or len(selected) != len(qm_atoms) or
            any(type(index) is not int or not 0 <= index < topology.getNumAtoms()
                for index in qm_atoms) or
            any((left.index in selected) != (right.index in selected)
                for left, right in topology.bonds())):
        raise AdapterError("CF-E022", None, "QM region must contain whole molecules",
                           "Select unique valid indices without cutting bonds")


def static_check(program, registry):
    nodes = {node.id: node for node in program.nodes}
    issues = []
    for node in program.nodes:
        if node.op != "qmmm_energy":
            continue
        source = nodes.get(node.inputs.get("system"))
        if source is None or source.op != "mm_system":
            continue
        try:
            from openmm import app

            pdb = source.params["pdb"]
            text = pdb if "\n" in pdb else Path(pdb).read_text()
            _check_region(app.PDBFile(io.StringIO(text)).topology, node.params["qm_atoms"])
        except AdapterError as exc:
            issues.extend(issue.model_copy(update={"node": node.id}) for issue in exc.diagnostics)
        except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(Diagnostic(code="CF-E080", node=node.id, message=str(exc)))
    return issues


def qmmm_energy(store, system, positions_nm, qm_atoms, basis, method="rhf", cas=None,
                solver="fci", solver_options=None, charge=0, spin=0, remaining_budget=None):
    import openmm
    from openmm import app

    from chemflow.plugins.openmm import QMMMCalculator

    if method not in ("rhf", "casci") or (method == "casci" and cas is None):
        raise ValueError("Use rhf or casci with cas=[n_orbitals, n_electrons]")
    if method == "rhf" and (cas is not None or solver != "fci" or solver_options):
        raise ValueError("Solver choices and active spaces require CASCI")
    topology = app.PDBFile(io.StringIO(store.get(system.topology_ref).tobytes().decode())).topology
    mm = openmm.XmlSerializer.deserialize(store.get(system.system_ref).tobytes().decode())
    _check_region(topology, qm_atoms)
    fcisolver = None
    if solver != "fci":
        from chemflow.plugins.pyscf import GPUFCISolver, SQDSolver

        if solver not in ("sqd", "gpu_fci"):
            raise ValueError("Supported CASCI solvers: fci, sqd, gpu_fci")
        options = dict(solver_options or {})
        if remaining_budget is not None:
            limits = dict(options.get("budget") or {})
            for dimension, amount in remaining_budget.items():
                limits[dimension] = min(limits.get(dimension, amount), amount)
            options["budget"] = limits
        fcisolver = (SQDSolver if solver == "sqd" else GPUFCISolver)(**options)
    elif solver_options:
        raise ValueError("Exact FCI does not accept plugin solver options")
    calculator = QMMMCalculator(topology, mm, qm_atoms, basis, charge=charge, spin=spin,
                                method="rhf", cas=cas, fcisolver=fcisolver)
    result = calculator.energy(positions_nm)
    relation = ("variational_upper_bound" if method == "rhf" or solver == "sqd" else
                result["provenance"]["qm_solver"].get("relation", "equivalent"))
    keys = energy_keys(system.id, positions_nm, qm_atoms, basis, method, cas, charge, spin)
    run = result["provenance"]["qm_solver"]
    backend = "cpu_numeric"
    if solver == "gpu_fci":
        backend = "gpu_numeric"
    elif solver == "sqd":
        backend = ("qpu" if (solver_options or {}).get("sampler") == "lqcloud"
                   else "quantum_simulator")
    value = estimate(result["total"], "qmmm_total_energy", relation, keys,
                     run.get("device", "openmm+pyscf"), diagnostics=result,
                     seed=(solver_options or {}).get("seed"))
    value = value.model_copy(update={"provenance": value.provenance.model_copy(
        update={"backend_kind": backend})})
    return ExecutionResult(value, device=value.provenance.device,
                           cost=run.get("resources", {}),
                           job_ids=tuple(run.get("job_ids", [])))


def _binding(method, solver, sampler=None):
    def run(store, **parameters):
        if parameters.get("method", method) != method or parameters.get("solver", solver) != solver:
            raise ValueError("QM/MM parameters conflict with the solver binding")
        parameters.update(method=method, solver=solver)
        if sampler is not None:
            options = dict(parameters.get("solver_options") or {})
            if options.get("sampler", sampler) != sampler:
                raise ValueError("QM/MM sampler conflicts with binding backend")
            options["sampler"] = sampler
            parameters["solver_options"] = options
        return qmmm_energy(store, **parameters)
    return run


def _resources(solver, sampler):
    def predict(solver_options=None, **parameters):
        if solver != "sqd":
            return {}
        options = solver_options or {}
        shots = options.get("shots", 4096)
        if sampler == "lqcloud":
            from chemflow.backends.lqcloud import LqcloudBackend

            return LqcloudBackend().prediction(shots=shots)
        return {"sim_shots": shots}
    return predict


def context_check(node, inputs, context, nodes):
    if node.op == "mm_system":
        context["system_id"] = content_hash(node.params)
    elif node.op == "qmmm_energy":
        parameters = dict(node.params)
        if node.binding in BINDINGS:
            method, solver, _, _ = BINDINGS[node.binding]
            parameters.setdefault("method", method)
            parameters.setdefault("solver", solver)
        elif parameters.get("solver", "fci") != "fci":
            return [Diagnostic(code="CF-E080", node=node.id,
                message="Quantum/GPU QM/MM solvers require their explicit backend binding")]
        try:
            context.update(energy_keys(inputs["system"]["system_id"], **parameters),
                           kind="qmmm_total_energy", unit="hartree")
        except (TypeError, ValueError, KeyError) as exc:
            return [Diagnostic(code="CF-E080", node=node.id, message=str(exc))]
    return []


BINDINGS = {
    "qmmm_rhf": ("rhf", "fci", None, "cpu_numeric"),
    "qmmm_casci": ("casci", "fci", None, "cpu_numeric"),
    "qmmm_sqd_statevector": ("casci", "sqd", "statevector", "quantum_simulator"),
    "qmmm_sqd_aer": ("casci", "sqd", "aer_noisy", "quantum_simulator"),
    "qmmm_sqd_gpu": ("casci", "sqd", "gpu", "quantum_simulator"),
    "qmmm_sqd_qpu": ("casci", "sqd", "lqcloud", "qpu"),
    "qmmm_gpu_fci": ("casci", "gpu_fci", None, "gpu_numeric"),
}


def relation(node):
    method, solver = node.params.get("method", "rhf"), node.params.get("solver", "fci")
    if node.binding in BINDINGS:
        method, solver, _, _ = BINDINGS[node.binding]
    return ("variational_upper_bound" if method == "rhf" or solver == "sqd" else
            "equivalent")


def register(registry):
    register_adapter(registry, "mm_system", mm_system, {}, "mm_system",
        ("pdb", "forcefields", "nonbonded"),
        "Artifact-backed OpenMM System and PDB Topology; force-field parameters are not rebuilt.",
        cacheable=False)
    register_adapter(registry, "qmmm_energy", qmmm_energy, {"system": "mm_system"}, "estimate",
        ("positions_nm", "qm_atoms", "basis", "method", "cas", "solver", "solver_options",
         "charge", "spin"),
        "Whole-molecule electrostatic QM/MM total energy; RHF upper bound or CASCI solver relation.")
    registry.context_checks.append(context_check)
    registry.checks.append(static_check)
    for name, (method, solver, sampler, backend) in BINDINGS.items():
        bound_relation = ("variational_upper_bound" if method == "rhf" or solver == "sqd"
                          else "equivalent")
        registry.register(RefinementRule(target="qmmm_energy", template_id=name,
                                          relation=bound_relation))
        registry.register(ImplementationSpec(id=name, version="1", target="qmmm_energy",
            implements=name, adapter=_binding(method, solver, sampler), backend_kind=backend,
            execution_role="verification_only" if backend == "quantum_simulator" else None,
            params=registry.implementations["qmmm_energy"].params,
            resource_model=_resources(solver, sampler),
            effects="random" if solver == "sqd" else "pure"))
