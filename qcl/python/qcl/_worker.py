"""QCL graph worker backed by ChemFlow's scientific adapters."""

import base64
import json
import math
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from qcl._tensor_units import product_unit

_MISSING = object()


def _native_ci_order(norb: int, electrons: int) -> Any:
    from itertools import combinations

    import numpy as np

    return np.argsort([sum(1 << orbital for orbital in occupied)
                       for occupied in combinations(range(norb), electrons)])


def _linear_uncertainty(*terms: tuple[float, dict[str, Any], int | str]
                        ) -> tuple[dict[str, tuple[float, float]], float]:
    components: dict[str, tuple[float, float]] = {}
    for coefficient, value, identifier in terms:
        sources = value.get("_uncertainty_components", {
            str(identifier): (1.0, value["variance"])})
        for source, (weight, variance) in sources.items():
            prior = components.get(source, (0.0, variance))[0]
            components[source] = (prior + coefficient * weight, variance)
    return components, sum(weight * weight * variance
                           for weight, variance in components.values())


def _molecule_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("atoms", "basis", "charge", "spin", "unit")}


def lattice_observable_value(vector: Any, sector: Any, observable: str,
                             left: int, right: int, xp: Any) -> float:
    from chemflow.impls.fci import _annihilate, _create, sector_states

    size = sector.n_orb
    if observable == "pair_s_wave":
        terms = [(1.0, [(left, True), (left + size, True),
                        (right + size, False), (right, False)])]
    elif observable == "number":
        terms = [(1.0, [(orbital, True), (orbital, False)])
                 for orbital in (left, left + size)]
    elif observable in ("density_density", "spin_spin"):
        terms = []
        for first, first_sign in ((left, 0.5), (left + size, -0.5)):
            for second, second_sign in ((right, 0.5), (right + size, -0.5)):
                weight = (first_sign * second_sign if observable == "spin_spin"
                          else 1.0)
                terms.append((weight, [(first, True), (first, False),
                                       (second, True), (second, False)]))
        if observable == "spin_spin":
            terms.extend([
                (0.5, [(left, True), (left + size, False),
                       (right + size, True), (right, False)]),
                (0.5, [(left + size, True), (left, False),
                       (right, True), (right + size, False)]),
            ])
    else:
        raise ValueError(f"Unknown lattice observable: {observable}")
    states = sector_states(sector)
    positions = {bits: index for index, bits in enumerate(states)}
    rows, columns, coefficients = [], [], []
    for column, initial in enumerate(states):
        for weight, operators in terms:
            bits, sign = initial, 1
            for orbital, creation in reversed(operators):
                result = (_create if creation else _annihilate)(bits, orbital)
                if result is None:
                    break
                bits, parity = result
                sign *= parity
            else:
                if bits in positions:
                    rows.append(positions[bits])
                    columns.append(column)
                    coefficients.append(weight * sign)
    amplitudes = xp.asarray(vector)
    value = xp.vdot(amplitudes[xp.asarray(rows, dtype=xp.int64)],
                     xp.asarray(coefficients) *
                     amplitudes[xp.asarray(columns, dtype=xp.int64)])
    if abs(float(value.imag)) > 1e-10:
        raise ValueError("Complex lattice expectation exceeds the real-valued contract")
    return float(value.real)


def _gpu_index(identifier: str) -> int:
    match = re.fullmatch(r"gpu(\d+)", identifier)
    if match is None:
        raise ValueError(f"GPU device id has no local CUDA ordinal: {identifier}")
    return int(match.group(1))


def _canonical_unrestricted_energies(mean_field: Any) -> Any:
    density = mean_field.make_rdm1()
    fock = mean_field.get_fock(dm=density)
    energies, _ = mean_field.eig(fock, mean_field.get_ovlp())
    return energies


def _gpu_kernel(data: Any, store: Any, device_id: str) -> Any:
    import cupy as cp

    from chemflow.backends.gpu import FciKernel

    with cp.cuda.Device(_gpu_index(device_id)):
        return FciKernel(store.get(data.hamiltonian.h1_ref),
                         store.get(data.hamiltonian.h2_ref), data.sector.n_orb,
                         (data.sector.n_alpha, data.sector.n_beta), xp=cp)


def _custom_statevector(circuit: Any, xp: Any) -> Any:
    n_qubits = circuit.n_qubits
    state = xp.zeros(1 << n_qubits, dtype=xp.complex128)
    state[0] = 1
    params = circuit.layers[0]
    for gate in circuit.gates:
        if gate["op"] == "ry":
            stride = 1 << gate["qubit"]
            pairs = state.reshape(-1, 2, stride)
            low, high = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
            angle = params[gate["parameter"]] / 2
            cosine, sine = math.cos(angle), math.sin(angle)
            pairs[:, 0, :] = cosine * low - sine * high
            pairs[:, 1, :] = sine * low + cosine * high
        elif gate["op"] == "cz":
            indices = xp.arange(len(state))
            mask = ((indices & (1 << gate["control"])) != 0) & (
                (indices & (1 << gate["target"])) != 0)
            state[mask] *= -1
        else:
            raise ValueError(f"Unsupported custom gate: {gate['op']}")
    for qubit, basis in enumerate(circuit.measurement_basis or ""):
        if basis not in "XY":
            continue
        pairs = state.reshape(-1, 2, 1 << qubit)
        if basis == "Y":
            pairs[:, 1, :] *= -1j
        low, high = pairs[:, 0, :].copy(), pairs[:, 1, :].copy()
        pairs[:, 0, :] = (low + high) / math.sqrt(2)
        pairs[:, 1, :] = (low - high) / math.sqrt(2)
    return state


def _finite_difference_optimize(molecule: dict[str, Any], fixed: list[dict[str, Any]],
                                step: float) -> dict[str, Any]:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixInternals
    from ase.optimize import BFGS
    from ase.units import Bohr, Hartree

    from chemflow.domains.paths import coordinates, mean_field, molecular_result, verify_constraints
    from chemflow.types.model import ElectronicModel

    model = ElectronicModel(basis=molecule["basis"], reference="RHF")
    symbols, positions = coordinates(molecule)
    atoms = Atoms(symbols, positions=positions)

    class FiniteDifferenceCalculator(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            current = self.atoms.positions.copy()

            def energy(position):
                geometry = molecular_result(molecule, symbols, position)
                return mean_field(geometry, model, {"method": "hf"}).e_tot

            center = energy(current)
            gradient = np.empty_like(current)
            for index in np.ndindex(current.shape):
                plus, minus = current.copy(), current.copy()
                plus[index] += step
                minus[index] -= step
                gradient[index] = (energy(plus) - energy(minus)) / (2 * step)
            self.results = {"energy": center * Hartree,
                            "forces": -gradient * Hartree}

    calculator = FiniteDifferenceCalculator()
    atoms.calc = calculator
    constraints = []
    if fixed:
        getters = {"bond": atoms.get_distance, "angle": atoms.get_angle,
                   "dihedral": atoms.get_dihedral}
        constraints = [{**item, "value": float(getters[item["type"]](*item["atoms"]))}
                       for item in fixed]
        options = {"bonds": [], "angles_deg": [], "dihedrals_deg": []}
        names = {"bond": "bonds", "angle": "angles_deg", "dihedral": "dihedrals_deg"}
        for item in constraints:
            options[names[item["type"]]].append([item["value"], item["atoms"]])
        atoms.set_constraint(FixInternals(**options))
        atoms.set_positions(atoms.positions)
    if ((len(symbols) != 2 or not constraints) and
            not BFGS(atoms, logfile=None).run(fmax=1e-3, steps=200)):
        raise ValueError("Finite-difference HF optimization did not converge")
    result = molecular_result(molecule, symbols, atoms.positions)
    if constraints:
        atoms.get_potential_energy()
        gradient_bohr = -calculator.results["forces"] / Hartree * Bohr
        verify_constraints(result, constraints, gradient_bohr)
    return {**result, "energy_hartree": float(atoms.get_potential_energy() / Hartree),
            "constraints": constraints}


def execute(request: dict[str, Any],
            qpu_backend_factory: Callable[[dict[str, Any]], Any] | None = None
            ) -> dict[str, Any]:
    import pyscf

    from chemflow.impls.pyscf_impls import active_from_scf, build_molecule, fci_pyscf
    from chemflow.runtime.artifacts import ArtifactStore
    from chemflow.types.model import ElectronicModel

    values: dict[int, Any] = {}
    ci_vectors: dict[int, Any] = {}
    by_id = {node["id"]: node for node in request["graph"]["nodes"]}
    density_sources = set()
    for density_node in request["graph"]["nodes"]:
        if density_node["op"] in ("rdm1", "rdm2", "fci.coefficients"):
            source_node = by_id[density_node["inputs"][0]]
            while source_node["op"] == "copy":
                source_node = by_id[source_node["inputs"][0]]
            density_sources.add(source_node["id"])
    profile: list[dict[str, Any]] = []
    store = ArtifactStore()
    for node in request["graph"]["nodes"]:
        identifier = node["id"]
        operation = node["op"]
        if any(values[guard["condition"]] is not guard["expected"]
               for guard in node.get("guards", ())):
            values[identifier] = _MISSING
            continue
        inputs = [values[item] for item in node["inputs"]]
        if operation not in ("map", "cond") and any(item is _MISSING for item in inputs):
            values[identifier] = _MISSING
            continue
        started = (time.perf_counter() if node["kind"] == "compute" or
                   node["kind"] == "copy" and node["device"]["kind"] == "gpu" else None)
        started_cpu = time.process_time() if started is not None else None
        if operation == "molecule":
            values[identifier] = node["params"]
        elif operation == "molecule.edit":
            from chemflow.domains.paths import geometry_transform

            molecule = inputs[0]
            identity = _molecule_identity(molecule)
            params = node["params"]
            result = geometry_transform(None, identity, f"set_{params['kind']}",
                                        params["atoms"], params["value"])
            values[identifier] = {**_molecule_identity({**result, "basis": identity["basis"]}),
                                  **({"conformer_provenance": molecule["conformer_provenance"]}
                                     if "conformer_provenance" in molecule else {}),
                                  "provenance": {
                                      "question": {"property": "geometry_edit",
                                                   "starting_molecule": identity},
                                      "method": {"coordinate": params["kind"],
                                                 "atoms": params["atoms"],
                                                 "value": params["value"]},
                                      "implementation": "molecule.edit/chemflow-host",
                                      "device": "host"}}
        elif operation == "frequencies.n_imaginary":
            values[identifier] = inputs[0]["n_imaginary"]
        elif operation == "integer.compare":
            equal = inputs[0] == node["params"]["expected"]
            values[identifier] = equal if node["params"]["equal"] else not equal
        elif operation == "boolean.constant":
            values[identifier] = node["params"]["value"]
        elif operation == "require":
            if not inputs[0]:
                raise ValueError(f"QCL requirement failed at {node['source']}")
            values[identifier] = True
        elif operation == "cond":
            values[identifier] = inputs[1] if inputs[0] else inputs[2]
        elif operation == "irc":
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
        elif operation == "path.connects":
            from chemflow.domains.paths import bond_graph, coordinates

            path, reactant, product = inputs
            observed = [bond_graph(item) for item in path["endpoints"]]
            expected = [bond_graph(item) for item in (reactant, product)]
            symbols = [coordinates(item)[0] for item in
                       (path["ts"], reactant, product)]
            values[identifier] = (symbols[0] == symbols[1] == symbols[2] and
                                  (observed == expected or observed == expected[::-1]))
        elif operation == "int2e":
            from pyscf import gto

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Molecular int2e currently requires a CPU")
            molecule = inputs[0]
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            values[identifier] = {"eri": system.intor("int2e"), "device": device["id"]}
        elif operation == "int1e":
            from pyscf import gto

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("One-electron integrals currently require a CPU")
            molecule = _molecule_identity(inputs[0])
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            kind = node["params"]["kind"]
            integrals = {"overlap": "int1e_ovlp", "kinetic": "int1e_kin",
                         "nuclear": "int1e_nuc", "dipole": "int1e_r",
                         "quadrupole": "int1e_rr"}
            array = system.intor(integrals[kind])
            if kind == "quadrupole":
                array = array.reshape(3, 3, system.nao_nr(), system.nao_nr())
            values[identifier] = {
                "array": array, "device": device["id"],
                "unit": {"overlap": "dimensionless", "kinetic": "hartree",
                         "nuclear": "hartree", "dipole": "bohr",
                         "quadrupole": "bohr^2"}[kind],
                "layout": "C", "provenance": {
                    "question": {"property": "one_electron_integral", "kind": kind,
                                 "molecule": molecule},
                    "method": {"family": "AO integral", "basis": molecule["basis"]},
                    "implementation": "int1e/chemflow-pyscf", "device": device["id"],
                    "versions": {"pyscf": pyscf.__version__},
                },
            }
        elif operation == "int3c":
            from pyscf import df, gto

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Three-center integrals currently require a CPU")
            molecule = _molecule_identity(inputs[0])
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            auxbasis = node["params"]["auxbasis"]
            auxiliary = df.make_auxmol(system, auxbasis)
            array = df.incore.aux_e2(system, auxiliary, aosym="s1")
            values[identifier] = {
                "array": array, "device": device["id"], "unit": "hartree",
                "layout": "C", "provenance": {
                    "question": {"property": "three_center_integral",
                                 "molecule": molecule},
                    "method": {"family": "AO density fitting integral",
                               "basis": molecule["basis"], "auxbasis": auxbasis},
                    "implementation": "int3c/chemflow-pyscf", "device": device["id"],
                    "versions": {"pyscf": pyscf.__version__},
                },
            }
        elif operation == "cholesky":
            import numpy as np
            from pyscf import gto, lib

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Cholesky ERI factors currently require a CPU")
            molecule = _molecule_identity(inputs[0])
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            nao = system.nao_nr()
            eri = system.intor("int2e").reshape(nao * nao, nao * nao)
            tolerance = node["params"]["tolerance"]
            upper, pivots, rank = lib.pivoted_cholesky(eri, tol=tolerance)
            factors = np.ascontiguousarray(
                upper[:rank, np.argsort(pivots)].reshape(rank, nao, nao))
            residual = float(np.max(np.abs(
                eri - factors.reshape(rank, nao * nao).T @
                factors.reshape(rank, nao * nao))))
            values[identifier] = {
                "array": factors, "device": device["id"],
                "unit": "hartree^1/2", "layout": "C", "provenance": {
                    "question": {"property": "cholesky_eri_factors",
                                 "molecule": molecule},
                    "method": {"family": "pivoted Cholesky ERI",
                               "basis": molecule["basis"],
                               "tolerance": tolerance,
                               "max_abs_reconstruction_error": residual},
                    "implementation": "cholesky/pyscf-pivoted",
                    "device": device["id"],
                    "versions": {"pyscf": pyscf.__version__},
                },
            }
        elif operation == "eigh":
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
        elif operation == "orthogonalize":
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
        elif operation == "solve":
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
        elif operation == "svd":
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
        elif operation == "transform":
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
        elif operation == "contract":
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
        elif operation == "periodic.scf":
            from chemflow.domains.materials import periodic_scf, periodic_system

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Periodic SCF currently requires a CPU")
            params = node["params"]
            system = periodic_system(store, **params["system"])
            energy = periodic_scf(store, system, method=params["method"],
                                  kmesh=params["kmesh"], exxdiv=params["exxdiv"])
            values[identifier] = {"value": energy, "unit": "hartree", "variance": 0.0,
                                  "provenance": {
                                      "question": {"property": "periodic_energy",
                                                   "system": params["system"]},
                                      "method": {"family": "periodic KRHF",
                                                 "kmesh": params["kmesh"],
                                                 "exxdiv": params["exxdiv"]},
                                      "implementation": "periodic.scf/chemflow-pyscf-pbc",
                                      "device": device["id"],
                                      "versions": {"pyscf": pyscf.__version__}}}
        elif operation == "dynamics.spectral_exact":
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
        elif operation in ("eigh.values", "eigh.vectors", "svd.u", "svd.s", "svd.vh"):
            values[identifier] = inputs[0][operation.split(".", 1)[1]]
        elif operation == "metabolism.sites":
            from chemflow.domains.reactivity import site_reactivity

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Metabolic site scoring currently requires a CPU")
            params = node["params"]
            result = site_reactivity(store, inputs[0], params["model"],
                                     {"method": params["level"]})
            values[identifier] = {**result, "provenance": {
                "question": {"property": "metabolic_sites", "molecule": inputs[0]},
                "method": {"family": "C-H BDE/FreeSASA site scoring",
                           "electronic_model": params["model"], "level": params["level"]},
                "implementation": "metabolism.sites/chemflow-reactivity",
                "device": device["id"], "versions": {"pyscf": pyscf.__version__},
                "validity_flags": ["Empirical site ranking only",
                                   "No enzyme or metabolic-rate prediction"],
            }}
        elif operation == "binding.mmgbsa":
            from chemflow.domains import binding

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("MM/GBSA currently requires a CPU")
            params = node["params"]
            receptor = params["receptor"]
            ligand = binding.ligand_structure(params["ligand"]["source"])
            protein = binding.protein_structure(
                receptor["pdb_text"], ph=params["ph"], chains=receptor.get("chains"))
            combined = binding.complex(protein, ligand)
            parameters = binding.ligand_parameters(ligand, forcefield=params["forcefield"])
            result = binding.mmgbsa(combined, parameters, frames=params["frames"],
                                    minimize=params["minimize"], seed=params["seed"])
            stderr = result.stderr
            values[identifier] = {
                "value": result.dG_kcal, "unit": "kcal/mol",
                "variance": stderr**2 if stderr is not None else 0.0,
                "provenance": {
                    "question": {"property": "binding_free_energy",
                                 "receptor": receptor, "ligand": params["ligand"]},
                    "method": {"family": "Amber14/GBn2 MM/GBSA",
                               "forcefield": params["forcefield"],
                               "frames": params["frames"], "minimize": params["minimize"],
                               "seed": params["seed"], "ph": params["ph"]},
                    "implementation": "binding.mmgbsa/chemflow-openmm",
                    "device": device["id"], "versions": {},
                    "components": result.components, "stderr": stderr,
                    "validity_flags": ["No configurational entropy correction",
                                       "Model-defined binding energy, not experimental affinity"],
                },
            }
        elif operation == "geometry.optimize":
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
        elif operation == "geometry.constrained_optimize":
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
        elif operation == "geometry.fd_optimize":
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
        elif operation == "optimize":
            values[identifier] = inputs[0]
        elif operation == "sella.saddle":
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
        elif operation == "frequencies":
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
        elif operation == "gradient":
            from importlib.metadata import version

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
                                 **({"gpu4pyscf": version("gpu4pyscf-cuda12x"),
                                     "cupy": cp.__version__} if gpu else {})},
                },
            }
        elif operation == "hessian":
            from importlib.metadata import version
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
                                 **({"gpu4pyscf": version("gpu4pyscf-cuda12x"),
                                     "cupy": cp.__version__} if gpu else {})},
                },
            }
        elif operation == "frequencies.from_hessian":
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
        elif operation == "thermo.rrho":
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
        elif operation == "thermo.eyring":
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
        elif operation == "rdkit.embed":
            from rdkit import Chem
            from rdkit.Chem import AllChem

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("RDKit embedding requires a CPU")
            params = node["params"]
            molecule = Chem.MolFromSmiles(params["smiles"])
            if (molecule is None or not molecule.GetNumHeavyAtoms() or
                    len(Chem.GetMolFrags(molecule)) != 1 or
                    any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())):
                raise ValueError("Embedding requires one closed-shell SMILES molecule")
            molecule = Chem.AddHs(molecule)
            options = AllChem.ETKDGv3()
            options.randomSeed = params["seed"]
            options.numThreads = 1
            conformer_ids = list(AllChem.EmbedMultipleConfs(
                molecule, numConfs=params["count"], params=options))
            if not conformer_ids:
                raise ValueError("ETKDG failed to embed conformers")

            def atoms_for(conf_id, molecule=molecule):
                coordinates = molecule.GetConformer(conf_id).GetPositions()
                return "; ".join(
                    atom.GetSymbol() + " " + " ".join(str(float(x)) for x in coordinates[index])
                    for index, atom in enumerate(molecule.GetAtoms()))

            values[identifier] = [{"atoms": atoms_for(conf_id), "basis": params["basis"],
                                   "charge": Chem.GetFormalCharge(molecule), "spin": 0,
                                   "unit": "angstrom", "rdkit_mol": molecule,
                                   "conf_id": conf_id,
                                   "conformer_provenance": {
                                       "smiles": params["smiles"],
                                       "embedding": "ETKDGv3",
                                       "embedding_seed": params["seed"],
                                       "conformer_id": conf_id}}
                                  for conf_id in conformer_ids]
        elif operation == "collection.index":
            collection = inputs[0]
            index = node["params"]["index"]
            values[identifier] = collection[index] if index < len(collection) else _MISSING
        elif operation == "rdkit.mmff_optimize":
            from rdkit.Chem import AllChem

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("MMFF optimization requires a CPU")
            conformer = inputs[0]
            molecule = conformer["rdkit_mol"]
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            if properties is None:
                raise ValueError("MMFF parameters unavailable")
            field = AllChem.MMFFGetMoleculeForceField(
                molecule, properties, confId=conformer["conf_id"])
            if field.Minimize(maxIts=1000) != 0:
                raise ValueError("MMFF optimization did not converge")
            coordinates = molecule.GetConformer(conformer["conf_id"]).GetPositions()
            atoms = "; ".join(
                atom.GetSymbol() + " " + " ".join(str(float(x)) for x in coordinates[index])
                for index, atom in enumerate(molecule.GetAtoms()))
            values[identifier] = {**conformer, "atoms": atoms,
                                  "energy_hartree": field.CalcEnergy() / 627.5094740631,
                                  "conformer_provenance": {
                                      **conformer["conformer_provenance"],
                                      "optimization": "MMFF94",
                                      "mmff_energy_hartree": (
                                          field.CalcEnergy() / 627.5094740631)}}
        elif operation == "rdkit.dedupe":
            from rdkit import Chem
            from rdkit.Chem import rdMolAlign

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("RDKit deduplication requires a CPU")
            conformers = sorted(inputs[0], key=lambda item: item["energy_hartree"])
            if not conformers:
                values[identifier] = []
            else:
                heavy = Chem.RemoveHs(conformers[0]["rdkit_mol"])
                retained = []
                for conformer in conformers:
                    if any(rdMolAlign.GetBestRMS(
                            Chem.Mol(heavy), heavy, prbId=conformer["conf_id"],
                            refId=previous["conf_id"]) < node["params"]["rmsd"]
                           for previous in retained):
                        continue
                    retained.append({**conformer, "conformer_provenance": {
                        **conformer["conformer_provenance"],
                        "rmsd_threshold": node["params"]["rmsd"]}})
                values[identifier] = retained
        elif operation == "models.hubbard":
            from chemflow.domains.materials import lattice_model

            params = node["params"]
            lattice_params = {"geometry": params["geometry"], "t": params["t"],
                              "U": params["U"]}
            if params["geometry"] == "square":
                lattice_params["shape"] = params["shape"]
            else:
                lattice_params["n_sites"] = params["shape"][0]
            data = lattice_model(store, **lattice_params)
            values[identifier] = {"data": data, "lattice": params,
                                  "question": {"property": "lattice_ground_state",
                                               "lattice": params},
                                  "versions": {"pyscf": pyscf.__version__}}
        elif operation == "lattice.sector":
            from chemflow.types import Sector

            params = node["params"]
            values[identifier] = {"sector": Sector(n_alpha=params["n_alpha"],
                                                    n_beta=params["n_beta"],
                                                    n_orb=inputs[0]["data"].sector.n_orb)}
        elif operation == "ed.ground_state":
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
        elif operation == "expect":
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
        elif operation in ("solubility.esol", "solubility.gse"):
            from chemflow.empirical import evaluate, model_card

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("Solubility models currently require a CPU")
            model = operation.removeprefix("solubility.")
            params = node["params"]
            evaluated = evaluate(model, params["smiles"], params["melting_point_C"])
            card = model_card(model)
            values[identifier] = {
                "value": evaluated["value"], "unit": evaluated["unit"], "variance": 0.0,
                "provenance": {
                    "question": {"property": "aqueous_solubility",
                                 "smiles": params["smiles"]},
                    "method": {"family": model.upper(),
                               "melting_point_C": params["melting_point_C"]},
                    "implementation": operation + "/chemflow-empirical",
                    "device": device["id"], "versions": card["software_versions"],
                    "model_card": {"dataset_sha256": card["dataset"]["sha256"],
                                   "train_rmse": card["train_rmse"],
                                   "cv_rmse": card["cv_rmse"]},
                    "validity_status": evaluated["validity_status"],
                    "max_tanimoto": evaluated["max_tanimoto"],
                },
            }
        elif operation == "superconductivity.allen_dynes":
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
        elif operation == "gfn2.energy":
            from importlib.metadata import version

            from pyscf import gto
            from tblite.interface import Calculator

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError(f"Unsupported GFN2 device: {device['kind']}")
            molecule = inputs[0]
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            charges = system.atom_charges()
            if any(charge == 0 for charge in charges):
                raise ValueError("GFN2 does not support ghost atoms")
            calculation = Calculator("GFN2-xTB", charges, system.atom_coords(),
                                     charge=molecule["charge"], uhf=molecule["spin"])
            calculation.set("verbosity", 0)
            solvent = node["params"].get("solvent")
            if solvent is not None:
                calculation.add(f"{solvent['model']}-solvation", solvent["name"])
            energy = float(calculation.singlepoint().get("energy"))
            values[identifier] = {
                "value": energy, "unit": "hartree", "variance": 0.0,
                "provenance": {
                    "question": {"atoms": molecule["atoms"],
                                 "charge": molecule["charge"],
                                 "spin": molecule["spin"], "unit": molecule["unit"],
                                 "property": "electronic_energy"},
                    "method": {"family": "GFN2-xTB",
                               **({"solvent": solvent} if solvent is not None else {})},
                    "implementation": "gfn2.energy/chemflow-tblite",
                    "device": device["id"],
                    "versions": {"tblite": version("tblite")},
                    "validity_flags": ["Semiempirical GFN2-xTB model"],
                },
            }
        elif operation == "gfn2.optimize":
            from importlib.metadata import version

            from ase import Atoms
            from ase.optimize import BFGS
            from ase.units import Hartree
            from pyscf import gto

            from chemflow.domains._path_calculator import isolated_tblite
            from chemflow.domains.paths import coordinates, molecular_result

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError(f"Unsupported GFN2 device: {device['kind']}")
            molecule = _molecule_identity(inputs[0])
            system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                           charge=molecule["charge"], spin=molecule["spin"],
                           unit=molecule["unit"], verbose=0)
            if any(charge == 0 for charge in system.atom_charges()):
                raise ValueError("GFN2 does not support ghost atoms")
            symbols, positions = coordinates(molecule)
            atoms = Atoms(symbols, positions=positions)
            solvent = node["params"].get("solvent")
            atoms.calc = isolated_tblite(molecule, solvent=solvent)
            try:
                if not BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500):
                    raise ValueError("GFN2 optimization did not converge")
                geometry = molecular_result(molecule, atoms.get_chemical_symbols(),
                                            atoms.positions)
                energy = float(atoms.get_potential_energy() / Hartree)
            finally:
                atoms.calc.close()
            values[identifier] = {**_molecule_identity({**geometry,
                                                        "basis": molecule["basis"]}),
                                  "provenance": {
                                      "question": {"property": "optimized_geometry",
                                                   "starting_molecule": molecule},
                                      "method": {"family": "GFN2-xTB", "kind": "minimum",
                                                 "energy_hartree": energy,
                                                 **({"solvent": solvent}
                                                    if solvent is not None else {})},
                                      "implementation": "gfn2.optimize/chemflow-tblite",
                                      "device": device["id"],
                                      "versions": {"tblite": version("tblite")},
                                      "validity_flags": ["Semiempirical GFN2-xTB model"],
                                  }}
        elif operation in ("scf.rhf", "scf.hf", "scf.rohf", "scf.uhf",
                           "dft.rks", "dft.uks"):
            device = node["device"]
            if device["kind"] not in ("cpu", "gpu"):
                raise ValueError(f"Unsupported {operation} device: {device['kind']}")
            molecule = inputs[0]
            reference = ("UKS" if operation == "dft.uks" else
                         "RKS" if operation == "dft.rks" else
                         "UHF" if operation == "scf.uhf" else
                         "ROHF" if operation == "scf.rohf" or
                         operation == "scf.hf" and molecule["spin"] else "RHF")
            solvent = node["params"].get("solvent")
            guess = inputs[-1] if node["params"].get("reuse_guess") else None
            guess_dm = guess["data"].mf.make_rdm1() if guess is not None else None
            gpu_scf = device["kind"] == "gpu"
            if operation in ("dft.rks", "dft.uks"):
                from types import SimpleNamespace

                from pyscf import dft, gto

                system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                               charge=molecule["charge"], spin=molecule["spin"],
                               unit=molecule["unit"], verbose=0)
                mean_field = (dft.UKS(system) if operation == "dft.uks"
                              else dft.RKS(system))
                mean_field.xc = node["params"]["xc"]
                mean_field.grids.level = 3
                mean_field.conv_tol = 1e-10
                if gpu_scf:
                    from importlib.metadata import version

                    import cupy as cp

                    with cp.cuda.Device(_gpu_index(device["id"])):
                        mean_field = mean_field.to_gpu()
                        if solvent:
                            if solvent.get("model") == "smd":
                                mean_field = mean_field.SMD()
                                mean_field.with_solvent.solvent = solvent["name"]
                            else:
                                mean_field = mean_field.PCM()
                                mean_field.with_solvent.method = "IEF-PCM"
                                mean_field.with_solvent.eps = solvent["eps"]
                        mean_field.kernel()
                    versions = {"pyscf": pyscf.__version__,
                                "gpu4pyscf": version("gpu4pyscf-cuda12x"),
                                "cupy": cp.__version__}
                    implementation = (f"{operation}/gpu4pyscf-" +
                                      solvent.get("model", "pcm") if solvent else
                                      f"{operation}/gpu4pyscf")
                else:
                    if solvent:
                        if solvent.get("model") == "smd":
                            mean_field = mean_field.SMD()
                            mean_field.with_solvent.solvent = solvent["name"]
                        else:
                            mean_field = mean_field.PCM()
                            mean_field.with_solvent.method = "IEF-PCM"
                            mean_field.with_solvent.eps = solvent["eps"]
                    mean_field.kernel()
                    versions = {"pyscf": pyscf.__version__}
                    implementation = (f"{operation}/chemflow-pyscf-" +
                                      solvent.get("model", "pcm") if solvent else
                                      f"{operation}/chemflow-pyscf")
                if not mean_field.converged:
                    raise ValueError(f"{reference} did not converge")
                data = SimpleNamespace(mf=mean_field, mol=system,
                                       model=ElectronicModel(
                                           basis=molecule["basis"],
                                           reference="UHF" if operation == "dft.uks" else "RHF",
                                           dft_functional=node["params"]["xc"]))
            elif gpu_scf or operation == "scf.uhf":
                from types import SimpleNamespace

                from pyscf import gto, scf

                system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                               charge=molecule["charge"], spin=molecule["spin"],
                               unit=molecule["unit"], verbose=0)
                mean_field = (scf.UHF(system) if operation == "scf.uhf" else
                              scf.ROHF(system) if reference == "ROHF" else
                              scf.RHF(system))
                mean_field.conv_tol = 1e-10
                if gpu_scf:
                    from importlib.metadata import version

                    import cupy as cp

                    with cp.cuda.Device(_gpu_index(device["id"])):
                        if operation == "scf.uhf":
                            from gpu4pyscf.scf import uhf as gpu_uhf

                            mean_field = gpu_uhf.UHF(system)
                            mean_field.conv_tol = 1e-10
                        elif reference == "ROHF":
                            from gpu4pyscf.scf import rohf as gpu_rohf

                            mean_field = gpu_rohf.ROHF(system)
                            mean_field.conv_tol = 1e-10
                        else:
                            mean_field = mean_field.to_gpu()
                        if solvent:
                            if solvent.get("model") == "smd":
                                mean_field = mean_field.SMD()
                                mean_field.with_solvent.solvent = solvent["name"]
                            else:
                                mean_field = mean_field.PCM()
                                mean_field.with_solvent.method = "IEF-PCM"
                                mean_field.with_solvent.eps = solvent["eps"]
                        mean_field.kernel(dm0=cp.asarray(guess_dm)
                                          if guess_dm is not None else None)
                    versions = {"pyscf": pyscf.__version__,
                                "gpu4pyscf": version("gpu4pyscf-cuda12x"),
                                "cupy": cp.__version__}
                    implementation = (f"{operation}/gpu4pyscf-" +
                                      solvent.get("model", "pcm") if solvent else
                                      f"{operation}/gpu4pyscf")
                else:
                    mean_field.kernel()
                    versions = {"pyscf": pyscf.__version__}
                    implementation = f"{operation}/chemflow-pyscf"
                if not mean_field.converged:
                    raise ValueError(f"{reference} did not converge")
                data = SimpleNamespace(mf=mean_field, mol=system,
                                       model=ElectronicModel(basis=molecule["basis"],
                                                             reference=reference))
            elif operation == "scf.rhf" and (
                    node["params"].get("reuse_eri") or
                    node["params"].get("reuse_guess") or solvent):
                from types import SimpleNamespace

                from pyscf import gto, scf

                system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                               charge=molecule["charge"], spin=molecule["spin"],
                               unit=molecule["unit"], verbose=0)
                mean_field = scf.RHF(system)
                if node["params"].get("reuse_eri"):
                    mean_field._eri = inputs[1]["eri"]
                if solvent:
                    if solvent.get("model") == "smd":
                        mean_field = mean_field.SMD()
                        mean_field.with_solvent.solvent = solvent["name"]
                    else:
                        mean_field = mean_field.PCM()
                        mean_field.with_solvent.method = "IEF-PCM"
                        mean_field.with_solvent.eps = solvent["eps"]
                mean_field.conv_tol = 1e-10
                if hasattr(guess_dm, "get"):
                    guess_dm = guess_dm.get()
                mean_field.kernel(dm0=guess_dm)
                if not mean_field.converged:
                    raise ValueError("RHF/PCM did not converge")
                data = SimpleNamespace(mf=mean_field, mol=system,
                                       model=ElectronicModel(basis=molecule["basis"],
                                                             reference="RHF"))
                implementation = ("scf.rhf/chemflow-pyscf-" +
                                  solvent.get("model", "pcm") if solvent else
                                  "scf.rhf/chemflow-pyscf-eri" if node["params"].get(
                                      "reuse_eri") else "scf.rhf/chemflow-pyscf")
                versions = {"pyscf": pyscf.__version__}
            else:
                data = build_molecule(
                    molecule["atoms"], ElectronicModel(basis=molecule["basis"],
                                                         reference=reference), store,
                    unit=molecule["unit"], charge=molecule["charge"], spin=molecule["spin"],
                )
                implementation = f"{operation}/chemflow-pyscf"
                versions = {"pyscf": pyscf.__version__}
            values[identifier] = {
                "data": data, "energy": float(data.mf.e_tot), "device": device["id"],
                "gpu_scf": gpu_scf,
                "question": {"atoms": molecule["atoms"], "charge": molecule["charge"],
                             "spin": molecule["spin"], "unit": molecule["unit"],
                             "property": "electronic_energy",
                             **({"conformer": molecule["conformer_provenance"]}
                                if "conformer_provenance" in molecule else {})},
                "method": {"family": reference, "basis": molecule["basis"],
                           **({"guess": "RHF density",
                               "guess_molecule": guess["question"]}
                              if guess is not None else {}),
                           **({"functional": node["params"]["xc"]}
                              if operation in ("dft.rks", "dft.uks") else {}),
                           **({"solvent": solvent["name"],
                               **({"solvent_model": "SMD"}
                                  if solvent.get("model") == "smd" else
                                  {"pcm_eps": solvent["eps"]})}
                              if solvent else {})},
                "implementation": implementation,
                "versions": versions,
            }
        elif operation == "mp2":
            device = node["device"]
            if device["kind"] not in ("cpu", "gpu"):
                raise ValueError(f"Unsupported MP2 device: {device['kind']}")
            source = inputs[0]
            mean_field = source["data"].mf
            if device["kind"] == "gpu":
                from importlib.metadata import version

                import cupy as cp
                from gpu4pyscf.mp import mp2 as gpu_mp2

                with cp.cuda.Device(_gpu_index(device["id"])):
                    if not source.get("gpu_scf"):
                        mean_field = mean_field.to_gpu()
                    calculation = gpu_mp2.MP2(mean_field)
                    calculation.kernel()
                implementation = "mp2/gpu4pyscf"
                versions = {"pyscf": pyscf.__version__,
                            "gpu4pyscf": version("gpu4pyscf-cuda12x"),
                            "cupy": cp.__version__}
            else:
                from pyscf import mp

                if source.get("gpu_scf"):
                    mean_field = mean_field.to_cpu()
                calculation = mp.MP2(mean_field)
                calculation.kernel()
                implementation = "mp2/chemflow-pyscf"
                versions = {"pyscf": pyscf.__version__}
            values[identifier] = {
                "energy": float(calculation.e_tot), "device": device["id"],
                "question": source["question"],
                "method": {"family": "MP2", "basis": source["method"]["basis"],
                           "reference": "RHF"},
                "implementation": implementation, "versions": versions,
            }
        elif operation == "active_space":
            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError(f"Unsupported active_space device: {device['kind']}")
            active = node["params"]
            source = inputs[0]
            if source.get("gpu_scf"):
                from types import SimpleNamespace

                data = SimpleNamespace(mf=source["data"].mf.to_cpu(),
                                       mol=source["data"].mol,
                                       model=source["data"].model)
            else:
                data = source["data"]
            values[identifier] = {
                **inputs[0],
                "data": active_from_scf(data, active["electrons"],
                                        active["orbitals"], store),
                "device": device["id"],
                "gpu_scf": False,
            }
        elif operation in ("active.hamiltonian", "active.sector"):
            values[identifier] = inputs[0]
        elif operation == "qubit_hamiltonian":
            from chemflow.impls.jw import pauli_terms
            from chemflow.types import QubitEncoding

            device = node["device"]
            if device["kind"] != "cpu" or node["params"]["mapping"] != "jordan_wigner":
                raise ValueError("Jordan-Wigner mapping currently requires a CPU")
            data = inputs[0]["data"]
            encoding = QubitEncoding(n_qubits=2 * data.sector.n_orb,
                                     hamiltonian_id=data.hamiltonian.id)
            values[identifier] = {**inputs[0],
                                  "terms": pauli_terms(data.hamiltonian, store),
                                  "n_qubits": encoding.n_qubits,
                                  "encoding_id": encoding.id,
                                  "provenance": {
                                      "question": inputs[0]["question"],
                                      "method": {"family": "Jordan-Wigner",
                                                 "mapping": "jordan_wigner",
                                                 "basis": data.model.basis},
                                      "implementation": "qubit_hamiltonian/chemflow-jw",
                                      "device": device["id"],
                                      "versions": inputs[0]["versions"],
                                  }}
        elif operation == "sector.shift":
            from chemflow.types import Sector

            original = inputs[0]
            values[identifier] = {
                **original,
                "sector": Sector(n_alpha=node["params"]["n_alpha"],
                                 n_beta=node["params"]["n_beta"],
                                 n_orb=original["data"].sector.n_orb),
            }
        elif operation == "ci.vector":
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
        elif operation == "kernel.hf":
            values[identifier] = {**inputs[0], "ansatz": "hf"}
        elif operation == "ccsd.amplitudes":
            from chemflow.impls.lucj import ccsd_seed_from_integrals

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("CCSD initialization currently requires a CPU")
            data = inputs[0]["data"]
            seed = ccsd_seed_from_integrals(data.hamiltonian, data.sector, store)
            if seed is None:
                raise ValueError("CCSD initialization is unavailable for this sector")
            values[identifier] = {"vector": seed.to_vector().tolist(),
                                  "provenance": {
                                      "question": inputs[0]["question"],
                                      "method": {"family": "CCSD-to-LUCJ initialization"},
                                      "implementation": "ccsd.amplitudes/chemflow-pyscf",
                                      "device": device["id"],
                                      "versions": {"pyscf": pyscf.__version__},
                                  }}
        elif operation == "kernel.lucj":
            vector = inputs[1]["vector"] if len(inputs) == 2 else node["params"]["params"]
            values[identifier] = {**inputs[0], "ansatz": "lucj", "params": vector}
        elif operation == "kernel.custom":
            values[identifier] = {"ansatz": "custom", **node["params"],
                                  "question": {"property": "bitstrings",
                                               "kernel": "custom"},
                                  "versions": {}}
        elif operation == "fci.solve":
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
        elif operation in ("rdm1", "rdm2"):
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
        elif operation == "fci.coefficients":
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
        elif operation == "sigma":
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
        elif operation == "sci.solve":
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
        elif operation in ("scf.homo", "scf.lumo", "scf.homo_alpha",
                           "scf.homo_beta", "scf.lumo_alpha", "scf.lumo_beta"):
            import numpy as np

            result = inputs[0]
            spin_channel = (0 if operation.endswith("_alpha") else
                            1 if operation.endswith("_beta") else None)
            if result.get("gpu_scf") and spin_channel is not None:
                if "_canonical_orbitals" not in result:
                    cpu_mean_field = result["data"].mf.to_cpu()
                    result["_canonical_orbitals"] = (
                        cpu_mean_field.mo_occ,
                        _canonical_unrestricted_energies(cpu_mean_field))
                occupations, energies = (np.asarray(values[spin_channel])
                                         for values in result["_canonical_orbitals"])
            elif result.get("gpu_scf"):
                import cupy as cp

                occupations = cp.asnumpy(result["data"].mf.mo_occ)
                energies = cp.asnumpy(result["data"].mf.mo_energy)
            else:
                occupation = result["data"].mf.mo_occ
                energy = result["data"].mf.mo_energy
                if spin_channel is not None:
                    occupation = occupation[spin_channel]
                    energy = energy[spin_channel]
                occupations = np.asarray(occupation)
                energies = np.asarray(energy)
            occupied = np.flatnonzero(occupations > 0)
            virtual = np.flatnonzero(occupations == 0)
            is_homo = operation.startswith("scf.homo")
            if not len(occupied) and is_homo or not len(virtual) and not is_homo:
                raise ValueError(f"{operation} is unavailable in this basis")
            index = occupied[-1] if is_homo else virtual[0]
            values[identifier] = {
                "value": float(energies[index]), "unit": "hartree", "variance": 0.0,
                "provenance": {"question": {**result["question"],
                                            "property": operation.removeprefix("scf.")},
                               "method": {**result["method"],
                                          **({"orbital_convention": "canonical_fock_from_density"}
                                             if spin_channel is not None else {})},
                               "implementation": (operation + "/pyscf-canonical-from-gpu"
                                                  if result.get("gpu_scf") and
                                                  spin_channel is not None else
                                                  operation + "/chemflow-pyscf"),
                               "device": result["device"], "versions": result["versions"],
                               "validity_flags": ["Orbital gap is not the fundamental gap"]},
            }
        elif operation in ("scf.energy", "mp2.energy", "fci.energy", "sci.energy",
                           "ed.energy"):
            result = inputs[0]
            values[identifier] = {
                "value": result["energy"], "unit": "hartree", "variance": 0.0,
                "provenance": {**{key: result[key] for key in
                                  ("question", "method", "implementation", "device", "versions")},
                               **({"validity_flags": result["validity_flags"]}
                                  if "validity_flags" in result else {})},
            }
        elif operation in ("estimate.add", "estimate.sub"):
            left, right = inputs
            sign = 1 if operation == "estimate.add" else -1
            components, variance = _linear_uncertainty(
                (1.0, left, node["inputs"][0]),
                (float(sign), right, node["inputs"][1]))
            values[identifier] = {
                "value": left["value"] + sign * right["value"],
                "unit": left["unit"],
                "variance": variance, "_uncertainty_components": components,
                "provenance": {"question": {"property": "estimate_expression"},
                               "method": {"family": "host_arithmetic"},
                               "implementation": operation + "/qcl-host",
                               "device": "host", "operands": [left["provenance"],
                                                              right["provenance"]]},
            }
        elif operation == "estimate.scale":
            original = inputs[0]
            factor = node["params"]["factor"]
            components, variance = _linear_uncertainty(
                (factor, original, node["inputs"][0]))
            values[identifier] = {
                "value": factor * original["value"], "unit": original["unit"],
                "variance": variance, "_uncertainty_components": components,
                "provenance": {"question": original["provenance"]["question"],
                               "method": {"family": "scalar_arithmetic",
                                          "factor": factor},
                               "implementation": "estimate.scale/qcl-host",
                               "device": "host", "operand": original["provenance"]},
            }
        elif operation == "estimate.mean":
            estimates = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], list) else inputs
            if not estimates:
                raise ValueError("Mean requires at least one estimate")
            units = {item["unit"] for item in estimates}
            if len(units) != 1:
                raise ValueError("Mean estimate units differ")
            count = len(estimates)
            source_ids = (node["inputs"] if len(node["inputs"]) == count else
                          [f"{identifier}:{index}" for index in range(count)])
            components, variance = _linear_uncertainty(*(
                (1 / count, item, source)
                for item, source in zip(estimates, source_ids, strict=True)))
            questions = [item["provenance"]["question"] for item in estimates]
            values[identifier] = {
                "value": sum(item["value"] for item in estimates) / count,
                "unit": estimates[0]["unit"], "variance": variance,
                "_uncertainty_components": components,
                "provenance": {"question": questions[0] if all(
                    item == questions[0] for item in questions) else
                    {"property": "mean_estimate", "inputs": questions},
                    "method": {"family": "host_mean", "count": count},
                    "implementation": "estimate.mean/qcl-host", "device": "host",
                    "input_implementations": [item["provenance"]["implementation"]
                                              for item in estimates]},
            }
        elif operation == "estimate.compare":
            import math

            left, right = inputs
            if left["unit"] != right["unit"]:
                raise ValueError("Comparison estimate units differ")
            _components, variance = _linear_uncertainty(
                (1.0, left, node["inputs"][0]),
                (-1.0, right, node["inputs"][1]))
            sigma = math.sqrt(max(0.0, variance))
            difference = left["value"] - right["value"]
            threshold = node["params"]["threshold_sigma"] * sigma
            relation = ("less" if difference < -threshold else
                        "greater" if difference > threshold else "unresolved")
            values[identifier] = {
                "relation": relation, "difference": difference,
                "combined_sigma": sigma, "unit": left["unit"],
                "provenance": {"question": {"property": "estimate_comparison"},
                               "method": {"family": "uncertainty_aware_comparison",
                                          "threshold_sigma": node["params"]["threshold_sigma"]},
                               "implementation": "estimate.compare/qcl-host",
                               "device": "host", "operands": [left["provenance"],
                                                               right["provenance"]]},
            }
        elif operation == "estimate.in_units":
            from chemflow.units import Q

            original = inputs[0]
            converted = Q(original["value"], original["unit"]).to(node["params"]["unit"])
            factor = Q(1, original["unit"]).to(node["params"]["unit"]).value
            components, variance = _linear_uncertainty(
                (factor, original, node["inputs"][0]))
            values[identifier] = {
                "value": converted.value, "unit": node["params"]["unit"],
                "variance": variance, "_uncertainty_components": components,
                "provenance": {**original["provenance"],
                               "unit_conversion": {"from": original["unit"],
                                                   "to": node["params"]["unit"]}},
            }
        elif operation == "sample":
            from chemflow.ids import content_hash
            from chemflow.ops.molecular import prepare_sector_state
            from chemflow.types import CircuitSpec, Samples, StatePrep

            device = node["device"]
            if device["kind"] != "qpu":
                raise ValueError("sample requires a QPU or QPU simulator")
            custom = inputs[0]["ansatz"] == "custom"
            data = None if custom else inputs[0]["data"]
            versions = dict(inputs[0]["versions"])
            if custom:
                import qiskit

                versions["qiskit"] = qiskit.__version__
            shots = node["params"]["shots"]
            seed = node["params"]["seed"]
            simulator_host = device["simulator_host"]
            if custom:
                circuit = CircuitSpec(ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                                      n_qubits=inputs[0]["n_qubits"],
                                      n_clbits=inputs[0]["n_clbits"],
                                      gates=tuple(inputs[0]["gates"]),
                                      layers=(tuple(inputs[0]["params"]),))
                identity = content_hash(circuit.model_dump(mode="json"))
                prep = StatePrep(state_ref=identity, encoding_id=identity,
                                 hamiltonian_id=identity, recipe="custom_circuit",
                                 circuit=circuit)
            else:
                prep = prepare_sector_state(data, data.sector.model_dump(), store,
                                            ansatz=inputs[0]["ansatz"],
                                            params=inputs[0].get("params"),
                                            materialize=simulator_host is not None)
            if custom and simulator_host is not None and simulator_host.startswith("gpu"):
                import cupy as cp
                import numpy as np

                versions["cupy"] = cp.__version__
                with cp.cuda.Device(_gpu_index(simulator_host)):
                    state = _custom_statevector(circuit, cp)
                    probabilities = cp.asnumpy(cp.abs(state) ** 2)
                    cp.cuda.Device().synchronize()
                drawn = np.random.default_rng(seed).choice(len(probabilities), size=shots,
                                                           p=probabilities)
                keys, frequencies = np.unique(drawn, return_counts=True)
                counts = {str(key): int(count) for key, count in zip(keys, frequencies)}
                samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                                  basis="computational", source="quantum_simulator",
                                  seed=seed, device=device["id"])
                implementation = "sample/qcl-cupy-statevector"
                resources = {"sim_shots": shots}
            elif custom and simulator_host is not None and circuit.n_clbits:
                import qiskit_aer
                from qiskit_aer import AerSimulator

                from chemflow.backends.circuits import build_circuit

                program = build_circuit(circuit)
                program.measure(range(circuit.n_qubits),
                                range(circuit.n_clbits,
                                      circuit.n_clbits + circuit.n_qubits))
                measured = AerSimulator(method="statevector").run(
                    program, shots=shots, seed_simulator=seed).result().get_counts()
                counts: dict[str, int] = {}
                for bits, count in measured.items():
                    key = str(int(bits.replace(" ", ""), 2) >> circuit.n_clbits)
                    counts[key] = counts.get(key, 0) + count
                samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                                  basis="computational", source="quantum_simulator",
                                  seed=seed, device=device["id"])
                implementation = "sample/qiskit-aer-dynamic"
                versions["qiskit-aer"] = qiskit_aer.__version__
                resources = {"sim_shots": shots}
            elif custom and simulator_host is not None:
                from qiskit.quantum_info import Statevector

                from chemflow.backends.circuits import build_circuit

                state = Statevector.from_instruction(build_circuit(prep.circuit))
                state.seed(seed)
                counts = {str(int(bits, 2)): int(count)
                          for bits, count in state.sample_counts(shots).items()}
                samples = Samples(counts=counts, shots=shots, encoding_id=prep.encoding_id,
                                  basis="computational", source="quantum_simulator",
                                  seed=seed, device=device["id"])
                implementation = "sample/qiskit-statevector"
                resources = {"sim_shots": shots}
            elif simulator_host is not None and simulator_host.startswith("gpu"):
                from chemflow.backends.gpu import catalogue

                computed = catalogue()["statevector_cupy"].adapter(store, prep, shots, seed)
                samples = computed.value
                implementation = "sample/chemflow-cupy-statevector"
                resources = computed.cost
            elif simulator_host is not None:
                from chemflow.impls.statevector import sample_configurations

                samples = sample_configurations(prep, shots, seed, store=store)
                implementation = "sample/chemflow-statevector"
                resources = {"sim_shots": shots}
            elif device["provider"] == "lqcloud" and device["backend"] == "MQ02":
                from chemflow.backends.lqcloud import LqcloudBackend

                policy = request.get("cloud_qpu_policy")
                billing_ceiling = (policy["max_cost"] / policy["forecast_cost"]
                                   if policy and policy["forecast_cost"] > 0 else None)
                backend = (qpu_backend_factory(device) if qpu_backend_factory else
                           LqcloudBackend(backend="MQ02",
                                          max_billing_multiplier=billing_ceiling))
                if "repetition" in node["params"]:
                    computed = backend.sample(store, prep, shots, seed,
                                              repetition=node["params"]["repetition"])
                else:
                    computed = backend.sample(store, prep, shots, seed)
                samples = computed.value
                implementation = "sample/chemflow-lqcloud-mq02"
                resources = computed.cost
                billing = computed.diagnostics.get("billing")
            else:
                raise ValueError("No QPU implementation is registered for this device")
            expected_source = "quantum_simulator" if simulator_host else "qpu"
            if (samples.source != expected_source or samples.shots != shots or
                    sum(samples.counts.values()) != shots):
                raise ValueError("Sampling adapter returned inconsistent source or shot counts")
            provenance = {
                "question": {**inputs[0]["question"], "property": "bitstrings"},
                "method": {"family": f"{inputs[0]['ansatz'].upper()} sampling",
                           "basis": data.model.basis if data is not None else "computational",
                           **({"active_electrons": data.active.n_elec,
                               "active_orbitals": data.active.n_orb}
                              if data is not None else
                              {"n_qubits": inputs[0]["n_qubits"]})},
                "implementation": implementation, "device": device["id"],
                "versions": versions, "resources": resources,
                "job_ids": samples.job_ids,
            }
            if simulator_host is None and billing is not None:
                provenance["billing"] = billing
            values[identifier] = {
                "counts": samples.counts, "shots": samples.shots,
                "source": samples.source, "device": device["id"],
                "encoding_id": samples.encoding_id, "hamiltonian_id": prep.hamiltonian_id,
                "basis": samples.basis, "seed": seed,
                "job_ids": samples.job_ids,
                "parts": [{"device": device["id"], "source": samples.source,
                           "shots": samples.shots, "counts": samples.counts,
                           "job_ids": samples.job_ids}],
                "provenance": provenance,
            }
        elif operation == "estimate.quantum":
            from qiskit.quantum_info import Statevector

            from chemflow.backends.circuits import build_circuit
            from chemflow.diagnostics import AdapterError
            from chemflow.ops.molecular import prepare_sector_state
            from chemflow.quantum.pauli import estimate as pauli_estimate
            from chemflow.quantum.pauli import qwc_groups
            from chemflow.types import CircuitSpec, StatePrep

            device = node["device"]
            operator, kernel = inputs
            data = operator["data"]
            groups = qwc_groups(operator["terms"])
            params = node["params"]
            if len(groups) != params["n_groups"]:
                raise ValueError("n_groups differs from Hamiltonian grouping")
            group_indices = params.get("group_indices", list(range(len(groups))))
            selected_groups = [(index, groups[index]) for index in group_indices]
            if kernel["ansatz"] != "custom" and kernel["data"] is not data:
                raise ValueError("Pauli operator and kernel refer to different active spaces")
            simulator = device["simulator_host"] is not None
            backend = None
            if not simulator:
                from chemflow.backends.lqcloud import LqcloudBackend

                policy = request.get("cloud_qpu_policy")
                ceiling = (policy["max_cost"] / policy["forecast_cost"]
                           if policy and policy["forecast_cost"] > 0 else None)
                backend = (qpu_backend_factory(device) if qpu_backend_factory else
                           LqcloudBackend(backend="MQ02",
                                          max_billing_multiplier=ceiling))
            measured = []
            job_ids = []
            resources: dict[str, Any] = {"sim_shots": 0, "qpu_shots": 0, "qpu_s": 0.0}
            versions = dict(operator["versions"])
            for index, group in selected_groups:
                if kernel["ansatz"] == "custom":
                    circuit = CircuitSpec(
                        ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                        n_qubits=kernel["n_qubits"], n_clbits=kernel["n_clbits"],
                        gates=tuple(kernel["gates"]),
                        layers=(tuple(kernel["params"]),),
                        measurement_basis=group["basis"])
                    prep = StatePrep(state_ref="", encoding_id=operator["encoding_id"],
                                     hamiltonian_id=data.hamiltonian.id,
                                     recipe="custom_pauli_estimate", circuit=circuit)
                else:
                    prep = prepare_sector_state(data, data.sector.model_dump(), store,
                                                ansatz=kernel["ansatz"],
                                                params=kernel.get("params"),
                                                materialize=False)
                    prep = prep.model_copy(update={"circuit": prep.circuit.model_copy(
                        update={"measurement_basis": group["basis"]})})
                shot_seed = params["seed"] + index
                if simulator and device["simulator_host"].startswith("gpu"):
                    import cupy as cp
                    import numpy as np

                    versions["cupy"] = cp.__version__
                    with cp.cuda.Device(_gpu_index(device["simulator_host"])):
                        state = _custom_statevector(circuit, cp)
                        probabilities = cp.asnumpy(cp.abs(state) ** 2)
                        cp.cuda.Device().synchronize()
                    drawn = np.random.default_rng(shot_seed).choice(
                        len(probabilities), size=params["shots_per_group"], p=probabilities)
                    keys, frequencies = np.unique(drawn, return_counts=True)
                    measured.append({format(int(key), f"0{operator['n_qubits']}b"): int(count)
                                     for key, count in zip(keys, frequencies)})
                    resources["sim_shots"] += params["shots_per_group"]
                elif simulator:
                    state = Statevector.from_instruction(build_circuit(prep.circuit))
                    state.seed(shot_seed)
                    measured.append(dict(state.sample_counts(params["shots_per_group"])))
                    resources["sim_shots"] += params["shots_per_group"]
                else:
                    try:
                        computed = backend.sample(store, prep, params["shots_per_group"],
                                                  shot_seed, repetition=index)
                    except AdapterError as error:
                        incurred = dict(error.incurred_cost)
                        for dimension in ("qpu_shots", "qpu_s"):
                            previous = resources[dimension]
                            current = incurred.get(dimension,
                                                   0 if error.nothing_consumed else None)
                            incurred[dimension] = (None if previous is None or
                                                   current is None else previous + current)
                        raise AdapterError(
                            "CF-E080", None, str(error),
                            "Reconcile completed Pauli groups before retrying",
                            nothing_consumed=error.nothing_consumed and not job_ids,
                            incurred_cost=incurred,
                            job_ids=(*job_ids, *error.job_ids)) from error
                    samples = computed.value
                    if (samples.source != "qpu" or
                            samples.shots != params["shots_per_group"] or
                            sum(samples.counts.values()) != samples.shots):
                        raise ValueError("Pauli sampler returned inconsistent shot counts")
                    measured.append({format(int(key), f"0{operator['n_qubits']}b"): count
                                     for key, count in samples.counts.items()})
                    job_ids.extend(samples.job_ids)
                    resources["qpu_shots"] += samples.shots
                    seconds = computed.cost.get("qpu_s")
                    resources["qpu_s"] = (None if seconds is None or
                                          resources["qpu_s"] is None else
                                          resources["qpu_s"] + seconds)
            measured_words = {word for _, group in selected_groups
                              for word in group["terms"]}
            selected_terms = {
                word: coefficient for word, coefficient in operator["terms"].items()
                if word in measured_words or
                (0 in group_indices and not set(word) - {"I"})}
            result = pauli_estimate(selected_terms,
                                    [group for _, group in selected_groups], measured)
            values[identifier] = {
                "value": result["energy"], "unit": "hartree",
                "variance": result["stat_err"] ** 2,
                "provenance": {
                    "question": operator["question"],
                    "method": {"family": "Pauli expectation",
                               "ansatz": kernel["ansatz"],
                               "basis": data.model.basis,
                               "active_electrons": data.active.n_elec,
                               "active_orbitals": data.active.n_orb,
                               "n_groups": len(groups),
                               "group_indices": group_indices},
                    "implementation": ("estimate.quantum/qcl-cupy-statevector"
                                       if simulator and device["simulator_host"].startswith("gpu")
                                       else "estimate.quantum/qiskit-statevector" if simulator
                                       else "estimate.quantum/chemflow-lqcloud-mq02"),
                    "device": device["id"], "versions": versions,
                    "sample_source": "quantum_simulator" if simulator else "qpu",
                    "sample_shots": len(group_indices) * params["shots_per_group"],
                    "resources": resources, "job_ids": job_ids,
                    "per_group_stat_err": result["per_group_stat_err"],
                },
            }
        elif operation == "estimates.merge":
            first = inputs[0]
            parts = [item["provenance"] for item in inputs]
            groups = node["params"]["n_groups"]
            assigned = [index for part in parts
                        for index in part["method"]["group_indices"]]
            if (sorted(assigned) != list(range(groups)) or
                    any(item["unit"] != first["unit"] or
                        item["provenance"]["question"] != first["provenance"]["question"] or
                        {key: value for key, value in item["provenance"]["method"].items()
                         if key != "group_indices"} !=
                        {key: value for key, value in parts[0]["method"].items()
                         if key != "group_indices"}
                        for item in inputs[1:])):
                raise ValueError("Cannot merge incompatible Pauli estimates")
            sources = {part["sample_source"] for part in parts}
            resources = {}
            for dimension in ("sim_shots", "qpu_shots", "qpu_s"):
                amounts = [part["resources"][dimension] for part in parts]
                resources[dimension] = (None if any(amount is None for amount in amounts)
                                        else sum(amounts))
            values[identifier] = {
                "value": sum(item["value"] for item in inputs),
                "unit": first["unit"],
                "variance": sum(item["variance"] for item in inputs),
                "provenance": {
                    "question": parts[0]["question"],
                    "method": {key: value for key, value in parts[0]["method"].items()
                               if key != "group_indices"},
                    "implementation": "estimates.merge/qcl-host",
                    "device": "multi", "versions": parts[0]["versions"],
                    "sample_source": next(iter(sources)) if len(sources) == 1 else "mixed",
                    "sample_shots": sum(part["sample_shots"] for part in parts),
                    "resources": resources,
                    "job_ids": [job for part in parts for job in part["job_ids"]],
                    "per_group_stat_err": [error for part in parts
                                           for error in part["per_group_stat_err"]],
                    "parts": parts,
                },
            }
        elif operation == "vqe.sampled":
            from chemflow.backends.lqcloud import LqcloudBackend
            from chemflow.quantum.vqe import sampled_vqe

            device = node["device"]
            if device["kind"] != "qpu":
                raise ValueError("Sampled VQE requires a QPU device")
            if inputs[0]["data"] is not inputs[1]["data"]:
                raise ValueError("VQE operator and sector refer to different active spaces")
            data = inputs[0]["data"]
            params = node["params"]
            simulator = device["simulator_host"] is not None
            backend = None
            if not simulator:
                policy = request.get("cloud_qpu_policy")
                ceiling = (policy["max_cost"] / policy["forecast_cost"]
                           if policy and policy["forecast_cost"] > 0 else None)
                backend = (qpu_backend_factory(device) if qpu_backend_factory else
                           LqcloudBackend(backend="MQ02", max_billing_multiplier=ceiling))
            computed = sampled_vqe(
                store, data, data.sector.model_dump(),
                ansatz=params["ansatz"], optimizer=params["optimizer"],
                max_evaluations=params["max_evaluations"],
                shots_per_group=params["shots_per_group"],
                n_groups=params["n_groups"], seed=params["seed"],
                bound_sampler="ideal" if simulator else "lqcloud", backend=backend)
            energy, diagnostics, uncertainty = computed.value
            actual_shots = (diagnostics["evaluations"] * diagnostics["groups"] *
                            params["shots_per_group"])
            values[identifier] = {
                "value": float(energy), "unit": "hartree",
                "variance": float(uncertainty["stat_err"]) ** 2,
                "provenance": {
                    "question": inputs[0]["question"],
                    "method": {"family": "sampled VQE", "ansatz": params["ansatz"],
                               "optimizer": params["optimizer"],
                               "basis": data.model.basis,
                               "active_electrons": data.active.n_elec,
                               "active_orbitals": data.active.n_orb},
                    "implementation": "vqe.sampled/chemflow-sampled-vqe",
                    "device": device["id"], "versions": inputs[0]["versions"],
                    "sample_source": "quantum_simulator" if simulator else "qpu",
                    "sample_shots": actual_shots,
                    "sample_job_ids": computed.job_ids,
                    "evaluations": diagnostics["evaluations"],
                    "measurement_groups": diagnostics["groups"],
                    "parameters": diagnostics["parameters"],
                    "resources": computed.cost,
                    "seed": params["seed"],
                    "validity_flags": ["Finite-shot and ansatz errors are not fully bounded"],
                },
            }
        elif operation == "samples.merge":
            from collections import Counter

            first = inputs[0]
            if any((item["encoding_id"], item["hamiltonian_id"], item["basis"],
                    item["source"]) != (first["encoding_id"], first["hamiltonian_id"],
                                       first["basis"], first["source"])
                   for item in inputs[1:]):
                raise ValueError("Cannot merge incompatible QPU samples")
            counts = Counter()
            for item in inputs:
                counts.update(item["counts"])
            parts = [part for item in inputs for part in item["parts"]]
            values[identifier] = {
                "counts": dict(counts), "shots": sum(item["shots"] for item in inputs),
                "source": first["source"], "device": "multi",
                "encoding_id": first["encoding_id"],
                "hamiltonian_id": first["hamiltonian_id"],
                "basis": first["basis"], "seed": first["seed"],
                "job_ids": [job for item in inputs for job in item["job_ids"]],
                "parts": parts,
                "provenance": {"question": first["provenance"]["question"],
                               "method": first["provenance"]["method"],
                               "implementation": "samples.merge/qcl-host",
                               "device": "multi", "parts": [item["provenance"]
                                                            for item in inputs]},
            }
        elif operation == "sqd.diagonalize":
            import numpy as np

            from chemflow.impls.sqd import sqd_sector
            from chemflow.types import Samples

            device = node["device"]
            if device["kind"] != "cpu":
                raise ValueError("SQD currently requires a CPU")
            if inputs[0]["data"] is not inputs[2]["data"]:
                raise ValueError("SQD operator and sector refer to different active spaces")
            data = inputs[0]["data"]
            bits = inputs[1]
            if bits["hamiltonian_id"] != data.hamiltonian.id:
                raise ValueError("SQD samples refer to a different Hamiltonian")
            samples = Samples(counts=bits["counts"], shots=bits["shots"],
                              encoding_id=bits["encoding_id"], basis=bits["basis"],
                              source=bits["source"], seed=bits["seed"],
                              device=bits["device"], job_ids=bits["job_ids"])
            energy, diagnostics = sqd_sector(data, data.sector, store, samples,
                                             seed=node["params"]["seed"])
            batches = diagnostics["batch_energies"]
            variance = (float(np.var(batches, ddof=1) / len(batches))
                        if len(batches) > 1 else 0.0)
            values[identifier] = {
                "value": float(energy), "unit": "hartree", "variance": variance,
                "provenance": {
                    "question": inputs[0]["question"],
                    "method": {"family": "SQD", "basis": data.model.basis,
                               "active_electrons": data.active.n_elec,
                               "active_orbitals": data.active.n_orb},
                    "implementation": "sqd.diagonalize/chemflow-pyscf",
                    "device": device["id"], "versions": inputs[0]["versions"],
                    "sample_source": bits["source"], "sample_device": bits["device"],
                    "sample_job_ids": bits["job_ids"], "sample_shots": bits["shots"],
                    "sample_parts": bits["parts"],
                    "seed": node["params"]["seed"],
                    "subspace_dimension": diagnostics["subspace_dimension"],
                    "batch_spread": diagnostics["batch_spread"],
                    "validity_flags": ["SQD subspace bias is not bounded by batch variance"],
                    **({"sample_billing": bits["provenance"]["billing"]}
                       if "billing" in bits["provenance"] else {}),
                },
            }
        elif operation == "map":
            values[identifier] = ([item for item in inputs[1:] if item is not _MISSING]
                                  if node["params"]["dynamic"] else inputs)
        elif operation == "scan":
            values[identifier] = [(inputs[index], inputs[index + 1])
                                  for index in range(1, len(inputs), 2)]
        elif operation == "iterate":
            values[identifier] = inputs[0]
        elif operation == "rank":
            import math

            if len(inputs) == 1 and isinstance(inputs[0], list):
                inputs = inputs[0]
            if not inputs:
                raise ValueError("Ranking requires at least one computed estimate")
            source_ids = (node["inputs"] if len(node["inputs"]) == len(inputs) else
                          [f"{identifier}:{index}" for index in range(len(inputs))])
            order = sorted(range(len(inputs)), key=lambda index: inputs[index]["value"])
            unresolved = [
                [left, right] for left in range(len(inputs))
                for right in range(left + 1, len(inputs))
                if abs(inputs[left]["value"] - inputs[right]["value"]) <=
                2 * math.sqrt(max(0.0, _linear_uncertainty(
                    (1.0, inputs[left], source_ids[left]),
                    (-1.0, inputs[right], source_ids[right]))[1]))
            ]
            values[identifier] = {
                "order": order, "unresolved": unresolved, "values": inputs,
                "provenance": {"question": {"property": "energy_ranking"},
                               "method": {"family": "uncertainty_aware_ranking",
                                          "threshold_sigma": 2},
                               "implementation": "rank/qcl-host", "device": "host",
                               "input_implementations": [item["provenance"]["implementation"]
                                                         for item in inputs]},
            }
        elif operation == "copy":
            device = node["device"]
            if device["kind"] == "cpu":
                if node["inputs"][0] in ci_vectors:
                    vector = ci_vectors[node["inputs"][0]]
                    ci_vectors[identifier] = (vector.get() if hasattr(vector, "get")
                                              else vector.copy())
                if node["value_type"] == "eri":
                    array = inputs[0]["eri"]
                    if hasattr(array, "get"):
                        array = array.get()
                    values[identifier] = {**inputs[0], "eri": array.copy(),
                                          "device": device["id"]}
                elif node["value_type"] in ("tensor", "hessian"):
                    array = inputs[0]["array"]
                    if hasattr(array, "get"):
                        array = array.get()
                    values[identifier] = {**inputs[0], "array": array.copy(),
                                          "device": device["id"]}
                elif node["value_type"] in ("eigensystem", "svd_result"):
                    parts = (("values", "vectors") if node["value_type"] == "eigensystem"
                             else ("u", "s", "vh"))
                    values[identifier] = {
                        **inputs[0],
                        **{part: {**inputs[0][part],
                                  "array": (inputs[0][part]["array"].get()
                                            if hasattr(inputs[0][part]["array"], "get") else
                                            inputs[0][part]["array"].copy()),
                                  "device": device["id"]}
                           for part in parts},
                    }
                elif node["value_type"] in ("fci_result", "sci_result"):
                    values[identifier] = {**inputs[0], "device": device["id"]}
                elif inputs[0].get("gpu_scf"):
                    from types import SimpleNamespace

                    source = inputs[0]["data"]
                    mean_field = source.mf.to_cpu()
                    if inputs[0]["method"]["family"] in ("UHF", "UKS"):
                        mean_field.mo_energy = _canonical_unrestricted_energies(mean_field)
                    values[identifier] = {
                        **inputs[0], "gpu_scf": False, "device": device["id"],
                        "data": SimpleNamespace(mf=mean_field, mol=source.mol,
                                                model=source.model),
                    }
                else:
                    values[identifier] = inputs[0]
            elif device["kind"] == "gpu" and node["value_type"] in (
                    "tensor", "hessian", "eri"):
                import cupy as cp

                with cp.cuda.Device(_gpu_index(device["id"])):
                    key = "eri" if node["value_type"] == "eri" else "array"
                    values[identifier] = {**inputs[0],
                                          key: cp.asarray(inputs[0][key]),
                                          "device": device["id"]}
            elif device["kind"] == "gpu" and node["value_type"] == "scf_result":
                from types import SimpleNamespace

                import cupy as cp

                source = inputs[0]["data"]
                with cp.cuda.Device(_gpu_index(device["id"])):
                    mean_field = source.mf.to_gpu()
                values[identifier] = {
                    **inputs[0], "gpu_scf": True, "device": device["id"],
                    "data": SimpleNamespace(mf=mean_field, mol=source.mol,
                                            model=source.model),
                }
            elif device["kind"] == "gpu" and node["value_type"] == "fermion_operator":
                values[identifier] = {**inputs[0], "gpu_kernel":
                                      _gpu_kernel(inputs[0]["data"], store, device["id"])}
            elif device["kind"] == "gpu" and node["value_type"] == "sector":
                import cupy as cp

                sector = inputs[0].get("sector", inputs[0]["data"].sector)
                with cp.cuda.Device(_gpu_index(device["id"])):
                    resident = cp.asarray([sector.n_alpha, sector.n_beta, sector.n_orb])
                values[identifier] = {**inputs[0], "gpu_sector": resident}
            elif device["kind"] == "gpu" and node["value_type"] == "ground_state":
                import cupy as cp

                with cp.cuda.Device(_gpu_index(device["id"])):
                    values[identifier] = {**inputs[0],
                                          "vector_gpu": cp.asarray(inputs[0]["vector"])}
            else:
                raise ValueError(f"Unsupported value copy to {device['kind']}")
        else:
            raise ValueError(f"Unsupported QCL operation: {operation}")
        if (isinstance(values[identifier], dict) and
                "variance" in values[identifier] and
                "_uncertainty_components" not in values[identifier]):
            values[identifier]["_uncertainty_components"] = {
                str(identifier): (1.0, values[identifier]["variance"])}
        if started is not None:
            gpu_host = (node["device"]["id"] if node["device"]["kind"] == "gpu" else
                        node["device"].get("simulator_host")
                        if node["device"]["kind"] == "qpu" else None)
            if gpu_host is not None and gpu_host.startswith("gpu"):
                import cupy as cp

                with cp.cuda.Device(_gpu_index(gpu_host)) as gpu_device:
                    gpu_device.synchronize()
            profile.append({"id": identifier, "op": operation,
                            "device": node["device"]["id"],
                            "duration_s": time.perf_counter() - started,
                            "cpu_duration_s": time.process_time() - started_cpu})
    def result_for(output):
        def tensor_payload(value):
            import numpy as np

            source = value["array"]
            original = (source.get() if hasattr(source, "get") else
                        np.asarray(source, dtype=np.float64))
            array = np.ascontiguousarray(original)
            return {"data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
                    "shape": list(original.shape), "dtype": str(array.dtype),
                    "device": value["device"], "unit": value["unit"],
                    "layout": value["layout"], "provenance": value["provenance"]}

        def public_value(identifier):
            value = values[identifier]
            if request["graph"]["nodes"][identifier]["value_type"] in ("tensor", "hessian"):
                return tensor_payload(value)
            if request["graph"]["nodes"][identifier]["value_type"] == "eigensystem":
                return {"values": tensor_payload(value["values"]),
                        "vectors": tensor_payload(value["vectors"]),
                        "provenance": value["provenance"]}
            if request["graph"]["nodes"][identifier]["value_type"] == "svd_result":
                return {**{part: tensor_payload(value[part])
                           for part in ("u", "s", "vh")},
                        "provenance": value["provenance"]}
            if request["graph"]["nodes"][identifier]["value_type"] == "qubit_operator":
                return {key: value[key] for key in
                        ("terms", "n_qubits", "encoding_id", "provenance")}
            return value

        if type(output) is int:
            return public_value(output)
        if output["type"] == "node":
            return public_value(output["id"])
        if output["type"] in ("tuple", "list"):
            return [result_for(item) for item in output["items"]]
        if output["type"] == "dict":
            return {key: result_for(item) for key, item in output["items"].items()}
        return output["value"]

    return {"result": result_for(request["output"]), "profile": profile}


def main() -> None:
    request = json.load(sys.stdin)
    if request.get("deterministic"):
        import random

        import numpy as np

        random.seed(0)
        np.random.seed(0)
    cpu_core_s = request.get("cpu_core_s")
    if cpu_core_s is not None:
        import resource
        import signal

        def exceeded(_signum, _frame):
            raise TimeoutError("QCL-E503: CPU core-time limit exceeded")

        signal.signal(signal.SIGPROF, exceeded)
        signal.setitimer(signal.ITIMER_PROF, cpu_core_s)
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        ceiling = math.ceil(time.process_time() + cpu_core_s) + 1
        soft = ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_CPU,
                           (soft, hard))
    json.dump(execute(request), sys.stdout)


if __name__ == "__main__":
    main()
