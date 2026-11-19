"""Helpers shared by QCL worker implementations (moved verbatim from _worker)."""

import math
import re
from typing import Any

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


def _gpu4pyscf_version() -> str:
    """Version of whichever GPU4PySCF build (CUDA 12 or 13 wheel) is installed."""
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("gpu4pyscf-cuda12x", "gpu4pyscf-cuda13x", "gpu4pyscf"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    import gpu4pyscf

    return gpu4pyscf.__version__
