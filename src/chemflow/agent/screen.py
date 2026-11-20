"""Signed tasks as QCL screens.

A screen is one QCL plan whose named outputs are the classical controls, the selected-CI
and sampled-quantum candidates, and the exact in-model reference for a signed target.
Compiling them together lets the QCL compiler share the mean-field and integral work
between methods (common-subexpression elimination) and run independent solvers side by
side (the lane scheduler), instead of executing a control plan, then a candidate plan.

Each output is graded against the reference output with the signed gates; relation gates
use the relation the compiler infers for the output (qcl-ir relation inference), so a
relation is a property of the compiled program, not a claim of its result. The critic is
the deterministic one in :mod:`chemflow.agent.critic`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from chemflow.diagnostics import AdapterError
from chemflow.types.model import ElectronicModel
from chemflow.units import Q

# Classical ladders, per target family, as the domain packs define them.
LADDER = ("hf", "mp2", "ccsd", "ccsd_t")
REACTIVITY_LADDER = ("hf", "mp2", "ccsd_t")
CHARGE_LADDER = ("hf", "ccsd_t")
LATTICE_LADDER = ("hf",)
# The scripted designer's candidate rounds (sci/sqd cards for a ground-energy operator).
CANDIDATE_ROUNDS = (
    (("sci", 1e-2), ("sci", 1e-3), ("sqd", 1024), ("sqd", 2048)),
    (("sci", 1e-4), ("sqd", 4096), ("sqd", 8192)),
)
SQD_SEED = 11
# Screening-scale SQD: two subsamples and two recovery rounds (the legacy candidate).
SQD_BATCHES = 2
SQD_ITERATIONS = 2
FCI_MAX_DETERMINANTS = 250_000
HARTREE_EV = 27.211386245988
# Targets defined by a named model (the signed level, solvent or empirical model): a
# plan that computes that model is equivalent to the target by construction.
MODEL_DEFINED = ("aqueous_solubility", "binding_free_energy", "metabolic_sites",
                 "solvation_free_energy", "frontier_orbitals", "spectral_function",
                 "periodic_energy")
COMPOSITE = ("reaction_energy", "reaction_barrier", "interaction_energy", "ranking",
             "bond_dissociation", "pes_scan", "interaction_scan", "ionization_energy",
             "electron_affinity")


@dataclass(frozen=True)
class Entry:
    """One named output of a screen."""

    name: str
    role: str  # "control" | "candidate" | "reference"
    method: str  # hf mp2 ccsd ccsd_t fci sci sqd ed model
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.params)


@dataclass
class Screen:
    """A compiled-ready screen: a QCL workflow, its arguments and its named outputs."""

    workflow: Any
    args: tuple[Any, ...]
    entries: tuple[Entry, ...]
    reference: str | None
    work: dict[str, int | None] = field(default_factory=dict)
    sim_shots: int = 0
    qubits: int = 0  # qubits of the widest sampled circuit (for QPU deployability)

    def names(self, role: str) -> list[str]:
        return [entry.name for entry in self.entries if entry.role == role]

    def entry(self, name: str) -> Entry:
        return next(entry for entry in self.entries if entry.name == name)


def entry_name(method: str, value: Any = None) -> str:
    return method if value is None else f"{method}-{value:g}"


# The method cards the scripted designer chooses from (as chemflow/cards/data).
CARDS = ({"id": "sci_pyscf", "template": "sci", "target": "chemflow.ground_energy@1"},
         {"id": "sqd_lucj", "template": "sqd", "target": "chemflow.ground_energy@1"},
         {"id": "pairing_sqd", "template": "pairing_sqd",
          "target": "chemflow.correlation_function@1"})


def candidates_for(target: Any, round_index: int, *, closed_shell: bool,
                   first_index: int = 0) -> tuple[Entry, ...]:
    """The scripted designer's candidates for a round (its sci/sqd cards).

    The designer is :class:`chemflow.agent.policies.ScriptedPolicy`; cards apply to
    ground-energy operators, and sampled candidates to closed-shell molecules only.
    """
    from chemflow.agent.policies import ScriptedPolicy

    if round_index >= len(CANDIDATE_ROUNDS):
        return ()
    attempted = [{"template": method, ("select_cutoff" if method == "sci" else "shots"): value}
                 for round_ in CANDIDATE_ROUNDS[:round_index] for method, value in round_]
    proposal = ScriptedPolicy("designer").decide({
        "cards": [dict(card) for card in CARDS],
        "target": getattr(target, "operator", "ground_energy"), "attempted": attempted})
    if target.kind == "molecular_ground_energy":
        allowed = ("sci", "sqd") if closed_shell else ("sci",)
    elif target.kind == "lattice_ground_energy":
        allowed = ("sci",)
    else:
        return ()
    entries = []
    for candidate in proposal.get("candidates", []):
        method = candidate.get("template")
        if method not in allowed:
            continue
        key = "select_cutoff" if method == "sci" else "shots"
        value = candidate.get(key)
        if value is None:
            continue
        entries.append(Entry(f"candidate{first_index + len(entries)}", "candidate", method,
                             ((key, value),)))
    return tuple(entries)


def _molecule(data: dict[str, Any], basis: str) -> Any:
    import qcl

    return qcl.Molecule(data["atoms"], basis, data.get("charge", 0), data.get("spin", 0),
                        data.get("unit", "angstrom"))


def _model(target: Any) -> ElectronicModel:
    model = ElectronicModel.model_validate(target.electronic_model)
    if (model.hamiltonian_class != "nonrel_bo" or model.ecp is not None or
            model.frozen_core != 0 or model.dft_functional is not None or
            model.reference not in ("RHF", "ROHF")):
        raise AdapterError("CF-E080", None, "QCL screens need a nonrelativistic HF model",
                           "Use an RHF/ROHF all-electron model without ECP or DFT")
    return model


def _determinants(n_orb: int, n_alpha: int, n_beta: int) -> int:
    return math.comb(n_orb, n_alpha) * math.comb(n_orb, n_beta)


def _work(method: str, n_orb: int, n_alpha: int, n_beta: int) -> int | None:
    """Reproducible asymptotic solver-work proxy (as chemflow.agent.report)."""
    powers = {"hf": 3, "mp2": 5, "ccsd": 6, "ccsd_t": 7}
    if method in powers:
        return n_orb ** powers[method]
    if method in ("sci", "sqd", "fci", "ed"):
        return _determinants(n_orb, n_alpha, n_beta) ** 2
    return None


def _full_space(data: dict[str, Any], basis: str) -> tuple[int, int, int]:
    from pyscf import gto

    mol = gto.M(atom=data["atoms"], basis=basis, charge=data.get("charge", 0),
                spin=data.get("spin", 0), unit=data.get("unit", "angstrom"), verbose=0)
    return mol.nao_nr(), mol.nelec[0], mol.nelec[1]


class _Energies:
    """Per-geometry mean fields and active spaces, shared by every method in a screen."""

    def __init__(self, cpu: Any, simulator: Any):
        self.cpu = cpu
        self.simulator = simulator
        self._spaces: dict[tuple[int, Any], Any] = {}

    def space(self, geometry: Any, active: Any) -> Any:
        import qcl

        key = (id(geometry), active)
        if key not in self._spaces:
            mean_field = qcl.scf.hf(geometry).to(self.cpu)
            self._spaces[key] = (geometry, qcl.active_space(mean_field, active).to(self.cpu))
        return self._spaces[key][1]

    def energy(self, geometry: Any, active: Any, entry: Entry) -> Any:
        import qcl

        space = self.space(geometry, active)
        options = entry.options
        if entry.method == "fci":
            return qcl.fci.solve(space.hamiltonian, space.sector).to(self.cpu).energy
        if entry.method in qcl.CLASSICAL_METHODS:
            return qcl.classical.solve(space.hamiltonian, space.sector,
                                       method=entry.method).to(self.cpu).energy
        if entry.method == "sci":
            return qcl.sci.solve(space.hamiltonian, space.sector,
                                 select_cutoff=options["select_cutoff"]).to(self.cpu).energy
        if entry.method == "sqd":
            amplitudes = qcl.ccsd.amplitudes(space).to(self.cpu)
            kernel = qcl.kernels.lucj(space, params=amplitudes)
            samples = qcl.sample(kernel, shots=options["shots"],
                                 seed=SQD_SEED).to(self.simulator)
            return qcl.sqd.diagonalize(space.hamiltonian, samples, space.sector,
                                       n_batches=SQD_BATCHES, max_iter=SQD_ITERATIONS
                                       ).to(self.cpu)
        raise ValueError(f"Unknown screen method {entry.method}")


def _ladder_entries(ladder: tuple[str, ...], candidates: tuple[Entry, ...] = (), *,
                    controls: bool = True, reference: str = "fci") -> tuple[Entry, ...]:
    """Round 0 carries the controls and the reference; later rounds only candidates."""
    entries = [Entry(method, "control", method) for method in ladder] if controls else []
    entries.extend(candidates)
    if reference and controls:
        entries.append(Entry(reference, "reference", reference))
    return tuple(entries)


CPU_SIMULATOR_ID = "sim-cpu"
# Below this many qubits a host statevector beats a GPU one (no CUDA start-up).
CPU_SIMULATOR_QUBITS = 18


def _devices(cluster: Any, device_id: str | None, qubits: int = 0) -> tuple[Any, Any]:
    import qcl

    if not cluster.cpus:
        raise AdapterError("CF-E080", None, "QCL screens require a CPU device",
                           "Use a cluster with at least one CPU")
    cpu = cluster[device_id] if device_id is not None else cluster.cpus[0]
    if qubits and qubits <= CPU_SIMULATOR_QUBITS:
        return cpu, qcl.Device(CPU_SIMULATOR_ID, "qpu", qubits=CPU_SIMULATOR_QUBITS,
                               simulator_host=cpu.id, simulator_method="statevector")
    simulators = [device for device in cluster.qpus if device.simulator_host]
    return cpu, simulators[0] if simulators else None


def build(task_spec: Any, cluster: Any, *, round_index: int = 0, device_id: str | None = None,
          controls: bool = True, active: dict[str, int] | None = None,
          only: set[str] | None = None, simulator: bool = True,
          proposed: tuple[Entry, ...] | None = None, first_index: int = 0) -> Screen:
    """The screen for a signed task: round 0 has controls, candidates and reference.

    ``only`` keeps the named outputs (to isolate a failing method); ``simulator=False``
    drops sampled candidates when the task does not permit simulator effects;
    ``proposed`` replaces the scripted designer's candidates (see ``candidate_entries``).
    """
    return _build(task_spec, cluster, round_index, device_id, controls, active,
                  only, simulator, proposed, first_index)


MAX_CANDIDATES = 8
SCI_CUTOFFS = (1e-8, 1e-1)
SQD_SHOTS = (1, 1_000_000)


def candidate_entries(task_spec: Any, candidates: list[dict[str, Any]], *,
                      active: dict[str, int] | None = None,
                      first_index: int = 0) -> tuple[Entry, ...]:
    """Validate designer-proposed candidates: ``sci`` (select_cutoff) or ``sqd`` (shots).

    Candidates apply to ground-energy targets and componentwise to composite energy
    targets; sampled (sqd) candidates need closed-shell species.
    """
    target = task_spec.target
    energy_targets = ("molecular_ground_energy", "lattice_ground_energy", *COMPOSITE)
    if target.kind not in energy_targets:
        raise ValueError(f"Candidates do not apply to {target.kind} targets")
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"Propose between 1 and {MAX_CANDIDATES} candidates")
    entries = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) - {"template", "select_cutoff", "shots"}:
            raise ValueError("A candidate is {template, select_cutoff} or {template, shots}")
        template = item.get("template")
        if template == "sci":
            cutoff = item.get("select_cutoff")
            if (type(cutoff) not in (int, float) or not math.isfinite(cutoff) or
                    not SCI_CUTOFFS[0] <= cutoff <= SCI_CUTOFFS[1] or
                    item.get("shots") is not None):
                raise ValueError(f"sci needs select_cutoff in {list(SCI_CUTOFFS)}")
            entries.append(Entry(f"candidate{first_index + len(entries)}", "candidate",
                                 "sci", (("select_cutoff", float(cutoff)),)))
        elif template == "sqd":
            shots = item.get("shots")
            if (type(shots) is not int or not SQD_SHOTS[0] <= shots <= SQD_SHOTS[1] or
                    item.get("select_cutoff") is not None):
                raise ValueError(f"sqd needs integer shots in {list(SQD_SHOTS)}")
            if target.kind == "lattice_ground_energy":
                raise ValueError("Sampled candidates are not available for lattice models")
            if not _closed_shell(task_spec, active):
                raise ValueError("Sampled candidates need closed-shell species")
            entries.append(Entry(f"candidate{first_index + len(entries)}", "candidate",
                                 "sqd", (("shots", shots),)))
        else:
            raise ValueError("Unknown candidate template; use sci or sqd")
    if len({entry.params for entry in entries}) != len(entries):
        raise ValueError("Candidates must be distinct")
    return tuple(entries)


def _closed_shell(task_spec: Any, active: dict[str, int] | None) -> bool:
    target = task_spec.target
    if target.kind == "molecular_ground_energy":
        return target.molecule.get("spin", 0) == 0
    molecules = []
    if target.kind in ("reaction_energy", "reaction_barrier"):
        molecules = [species.molecule for species in target.species]
    elif target.kind == "ranking":
        molecules = [item.molecule for item in target.items]
    elif target.kind in ("pes_scan", "bond_dissociation", "ionization_energy",
                         "electron_affinity"):
        return False if target.kind != "pes_scan" else target.molecule.get("spin", 0) == 0
    return all(molecule.get("spin", 0) == 0 for molecule in molecules)


def _keep(entries: tuple[Entry, ...], only: set[str] | None,
          simulator: bool) -> tuple[Entry, ...]:
    return tuple(entry for entry in entries
                 if (only is None or entry.name in only) and
                 (simulator or entry.method != "sqd"))


def _build(task_spec: Any, cluster: Any, round_index: int, device_id: str | None,
           controls: bool, active: dict[str, int] | None, only: set[str] | None,
           use_simulator: bool, proposed: tuple[Entry, ...] | None = None,
           first_index: int = 0) -> Screen:
    import qcl

    target = task_spec.target
    qubits = (2 * (active or target.active)["n_orb"]
              if target.kind == "molecular_ground_energy" and
              isinstance(active or target.active, dict) else 0)
    cpu, simulator = _devices(cluster, device_id, qubits)
    kind = target.kind

    if kind in MODEL_DEFINED or kind in ("lattice_ground_energy", "pairing_correlation"):
        return _model_screen(task_spec, cluster, cpu, simulator, round_index, controls, only,
                             proposed, first_index)

    model = _model(target)
    basis = model.basis
    full = qcl.Active(max_orbitals=999)

    if kind == "molecular_ground_energy":
        declared = active or target.active
        if declared == "auto":
            raise AdapterError("CF-E080", None, "Resolve the automatic active space first",
                               "Call resolve_active before building the screen")
        molecule = _molecule(target.molecule, basis)
        n_alpha = (declared["n_elec"] + molecule.spin) // 2
        n_beta = declared["n_elec"] - n_alpha
        closed = n_alpha == n_beta
        if _determinants(declared["n_orb"], n_alpha, n_beta) > FCI_MAX_DETERMINANTS:
            raise AdapterError("CF-E080", None, "Active space exceeds the exact-reference cap",
                               "Use a smaller active space")
        candidates = (proposed if proposed is not None else
                      candidates_for(target, round_index, closed_shell=closed,
                                     first_index=first_index))
        entries = _keep(_ladder_entries(LADDER, candidates, controls=controls), only,
                        use_simulator and simulator is not None)
        space = qcl.Active(declared["n_elec"], declared["n_orb"])

        @qcl.workflow
        def ground(mol: Any, device: Any, sim: Any) -> Any:
            energies = _Energies(device, sim)
            return {entry.name: energies.energy(mol, space, entry) for entry in entries}

        work = {entry.name: _work(entry.method, declared["n_orb"], n_alpha, n_beta)
                for entry in entries}
        shots = sum(entry.options.get("shots", 0) for entry in entries)
        return Screen(ground, (molecule, cpu, simulator), entries, "fci", work, shots,
                      2 * declared["n_orb"])

    species, combine, ladder = _composite(target, basis, full)
    entries = _keep(_ladder_entries(ladder, proposed or (), controls=controls), only,
                    use_simulator)

    @qcl.workflow
    def composite(device: Any, sim: Any) -> Any:
        energies = _Energies(device, sim)
        geometries = [item["geometry"](device) for item in species]
        outputs = {}
        for entry in entries:
            terms = []
            for item, geometry in zip(species, geometries, strict=True):
                energy = energies.energy(geometry, item["active"], entry)
                if item.get("thermal") is not None:
                    energy = energy + item["thermal"](geometry, device)
                terms.append(energy)
            outputs[entry.name] = combine(terms, device)
        return outputs

    work = {}
    for entry in entries:
        sizes = [_work(entry.method, *item["size"]) for item in species]
        work[entry.name] = None if None in sizes else sum(sizes)
    for item in species:
        if entry_fci_too_large(item["size"]):
            raise AdapterError("CF-E080", None, "A species exceeds the exact-reference cap",
                               "Use smaller species or basis sets")
    shots = sum(entry.options.get("shots", 0) for entry in entries) * len(species)
    if shots and simulator is None:
        raise AdapterError("CF-E080", None, "Sampled candidates need a simulator",
                           "Use a cluster with a QPU simulator")
    return Screen(composite, (cpu, simulator), entries, "fci", work, shots,
                  max(2 * item["size"][0] for item in species))


def entry_fci_too_large(size: tuple[int, int, int]) -> bool:
    return _determinants(*size) > FCI_MAX_DETERMINANTS


def _static(geometry: Any):
    return lambda device: geometry


def _thermal(target: Any):
    import qcl

    def correction(geometry: Any, device: Any) -> Any:
        modes = qcl.frequencies(geometry).to(device)
        return qcl.thermo.rrho(geometry, modes, temperature_K=target.thermo.temperature_K,
                               pressure_Pa=target.thermo.pressure_Pa,
                               quantity=target.thermo.quantity).to(device)

    return correction


def _composite(target: Any, basis: str, full: Any) -> tuple[list[dict], Any, tuple[str, ...]]:
    """Species (geometry builder, active space, size) and how their energies combine."""
    import qcl

    kind = target.kind

    def species_item(data: dict[str, Any], geometry: Any = None, active: Any = None,
                     thermal: Any = None) -> dict[str, Any]:
        molecule = _molecule(data, basis)
        if active is None:
            size = _full_space(data, basis)
            space = full
        else:
            spin = data.get("spin", 0)
            n_alpha = (active["n_elec"] + spin) // 2
            size = (active["n_orb"], n_alpha, active["n_elec"] - n_alpha)
            space = qcl.Active(active["n_elec"], active["n_orb"])
        return {"geometry": geometry or _static(molecule), "active": space, "size": size,
                "thermal": thermal}

    if kind == "reaction_energy":
        items = []
        for species in target.species:
            if target.thermo is not None:
                molecule = _molecule(species.molecule, basis)
                optimized = (lambda device, molecule=molecule:
                             qcl.optimize(molecule).to(device))
                items.append(species_item(species.molecule, optimized, species.active,
                                          _thermal(target)))
            else:
                items.append(species_item(species.molecule, active=species.active))
        coefficients = [species.coefficient for species in target.species]

        def reaction(terms: list[Any], device: Any) -> Any:
            total = None
            for coefficient, term in zip(coefficients, terms, strict=True):
                scaled = coefficient * term
                total = scaled if total is None else total + scaled
            return total

        return items, reaction, LADDER

    if kind == "reaction_barrier":
        reactant = next(item for item in target.species if item.role == "minimum")
        transition = next(item for item in target.species if item.role == "ts")
        start = _molecule(reactant.molecule, basis)
        guess = _molecule(transition.molecule, basis)
        product = _molecule(target.product, basis)
        thermal = _thermal(target) if target.thermo is not None else None
        minimum = ((lambda device: qcl.optimize(start).to(device))
                   if target.thermo is not None else _static(start))

        def saddle(device: Any) -> Any:
            return qcl.engines.sella.saddle(guess, start, product).to(device)

        items = [species_item(transition.molecule, saddle, transition.active, thermal),
                 species_item(reactant.molecule, minimum, reactant.active, thermal)]

        def barrier(terms: list[Any], device: Any) -> Any:
            difference = terms[0] - terms[1]
            if target.rate_constant:
                return difference, qcl.thermo.eyring(
                    difference, target.thermo.temperature_K).to(device)
            return difference

        return items, barrier, LADDER

    if kind == "interaction_energy":
        from chemflow.domains.drug import component_specs

        components = component_specs(target.complex, target.fragments,
                                     target.electronic_model, target.counterpoise)
        items = [species_item(component) for component in components]
        return items, (lambda terms, device: terms[0] - terms[1] - terms[2]), LADDER

    if kind in ("ionization_energy", "electron_affinity"):
        molecule = target.molecule
        shift = 1 if kind == "ionization_energy" else -1
        final = {**molecule, "charge": molecule.get("charge", 0) + shift,
                 "spin": round(2 * target.final_spin)}
        items = [species_item(molecule), species_item(final)]
        if kind == "ionization_energy":
            return items, (lambda terms, device: terms[1] - terms[0]), CHARGE_LADDER
        return items, (lambda terms, device: terms[0] - terms[1]), CHARGE_LADDER

    if kind == "ranking":
        items = [species_item(item.molecule) for item in target.items]
        return items, (lambda terms, device: qcl.rank(terms)), REACTIVITY_LADDER

    if kind == "bond_dissociation":
        from chemflow.domains.reactivity import homolysis, representatives

        bonds = representatives(target.molecule, target.bonds)
        items = [species_item(target.molecule)]
        for bond in bonds:
            items.extend(species_item(fragment) for fragment in
                         homolysis(None, target.molecule, bond["atoms"])["fragments"])

        def energies(terms: list[Any], device: Any) -> Any:
            return [terms[index] + terms[index + 1] - terms[0]
                    for index in range(1, len(terms), 2)]

        return items, energies, REACTIVITY_LADDER

    if kind == "pes_scan":
        parent = _molecule(target.molecule, basis)
        coordinate = target.coordinate
        edit = getattr(parent, f"set_{coordinate.type}")
        items = []
        for value in coordinate.values:
            moved = edit(*coordinate.atoms, value)
            moved_data = {**target.molecule, "atoms": moved.atoms, "unit": moved.unit}
            if target.mode == "relaxed":
                fixed = {"bond": qcl.Bond, "angle": qcl.Angle,
                         "dihedral": qcl.Dihedral}[coordinate.type](*coordinate.atoms)
                pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                              gradient=qcl.Analytic())

                def relaxed(device: Any, moved: Any = moved, fixed: Any = fixed,
                            pes: Any = pes) -> Any:
                    return qcl.optimize(pes, moved, fixed=[fixed]).to(device)

                items.append(species_item(moved_data, relaxed))
            else:
                items.append(species_item(moved_data))
        return items, (lambda terms, device: list(terms)), LADDER

    if kind == "interaction_scan":
        parent = _molecule(target.complex, basis)
        first, second = target.fragments
        items = []
        for distance in target.separations:
            geometry = parent.set_separation(first, second, distance)
            data = {**target.complex, "atoms": geometry.atoms, "unit": geometry.unit}
            parts = ([geometry, geometry.ghost(second), geometry.ghost(first)]
                     if target.counterpoise else
                     [geometry, _subset(geometry, first), _subset(geometry, second)])
            items.append(species_item(data))
            for part in parts[1:]:
                items.append(species_item({"atoms": part.atoms, "unit": part.unit,
                                           "charge": 0, "spin": 0}))

        def curve(terms: list[Any], device: Any) -> Any:
            return [terms[index] - terms[index + 1] - terms[index + 2]
                    for index in range(0, len(terms), 3)]

        return items, curve, LADDER

    raise AdapterError("CF-E080", None, f"No QCL screen for {kind}",
                       "Use a supported signed target")


def _subset(geometry: Any, atoms: list[int]) -> Any:
    import qcl

    entries = [item.strip() for item in geometry.atoms.split(";") if item.strip()]
    return qcl.Molecule("; ".join(entries[index] for index in atoms), geometry.basis,
                        0, 0, geometry.unit)


def _model_screen(task_spec: Any, cluster: Any, cpu: Any, simulator: Any, round_index: int,
                  controls: bool, only: set[str] | None = None,
                  proposed: tuple[Entry, ...] | None = None, first_index: int = 0) -> Screen:
    """Lattice targets and model-defined targets (one model output graded against itself)."""
    import qcl

    target = task_spec.target
    kind = target.kind
    if kind in ("lattice_ground_energy", "pairing_correlation"):
        spec = dict(target.lattice)
        scale = 1 / HARTREE_EV if spec.get("unit", "hartree") == "eV" else 1.0
        if spec.get("unit", "hartree") not in ("hartree", "eV") or spec.get(
                "model", "hubbard") != "hubbard":
            raise AdapterError("CF-E080", None, "Unsupported lattice model or unit",
                               "Use a Hubbard lattice in hartree or eV")
        geometry = spec.get("geometry", "chain")
        lattice = (qcl.lattice.square(*spec["shape"]) if geometry == "square" else
                   qcl.lattice.ring(spec["n_sites"]) if geometry == "ring" else
                   qcl.lattice.chain(spec["n_sites"]))
        n_alpha, n_beta = target.sector["n_alpha"], target.sector["n_beta"]
        sector = qcl.Sector(n=n_alpha + n_beta, sz=(n_alpha - n_beta) / 2)
        sites = target.sector["n_orb"]
        t, U = spec.get("t", 1.0) * scale, spec.get("U", 4.0) * scale
        if kind == "lattice_ground_energy":
            candidates = (proposed if proposed is not None else
                          candidates_for(target, round_index, closed_shell=True,
                                         first_index=first_index))
            entries = _ladder_entries(LATTICE_LADDER, candidates, controls=controls,
                                      reference="ed")
        else:
            entries = (Entry("ed", "reference", "ed"),) if controls else ()
        entries = _keep(entries, only, True)
        observables = {"pair_s_wave": qcl.ops.onsite_pair, "spin_spin": qcl.ops.spin_spin,
                       "density_density": qcl.ops.density_density}

        @qcl.workflow
        def lattice_screen(system: Any, particles: Any, device: Any) -> Any:
            operator = qcl.models.hubbard(system, t=t, U=U)
            subspace = qcl.lattice.particles(operator, particles)
            outputs = {}
            for entry in entries:
                if entry.method == "ed":
                    state = qcl.ed.ground_state(operator, sector=particles).to(device)
                    outputs[entry.name] = (
                        state.energy if kind == "lattice_ground_energy" else
                        qcl.expect(state, observables[target.quantity](
                            *target.site_pair)).to(device))
                elif entry.method == "sci":
                    outputs[entry.name] = qcl.sci.solve(
                        operator, subspace,
                        select_cutoff=entry.options["select_cutoff"]).to(device).energy
                else:
                    outputs[entry.name] = qcl.classical.solve(
                        operator, subspace, method=entry.method).to(device).energy
            return outputs

        work = {entry.name: _work(entry.method, sites, n_alpha, n_beta) for entry in entries}
        return Screen(lattice_screen, (lattice, sector, cpu), entries, "ed", work)

    from chemflow.agent.qcl import _builtin_program

    workflow, positional, keyword = _builtin_program(task_spec, cluster, device_id=cpu.id)
    if keyword:
        raise AdapterError("CF-E080", None, "Unexpected keyword arguments for a model screen",
                           "Report this as a QCL screen bug")
    entries = (Entry("model", "control", "model"),)
    return Screen(workflow, tuple(positional), entries, "model", {"model": None})


def active_error(active: Any, target: Any) -> str | None:
    """Why a chosen active space is infeasible for the signed molecule (None when fine)."""
    if (not isinstance(active, dict) or any(type(active.get(key)) is not int
                                           for key in ("n_elec", "n_orb"))):
        return "Modeler did not choose integer active-space counts with model.choose"
    if not 0 < active["n_orb"] <= 10:
        return "Active space exceeds the screening orbital cap or has no orbitals"
    spin = target.molecule.get("spin", 0)
    if (active["n_elec"] <= 0 or active["n_elec"] > 2 * active["n_orb"] or
            active["n_elec"] < spin or (active["n_elec"] - spin) % 2):
        return "Modeler chose an infeasible spin/electron active space"
    return None


def resolve_active(target: Any) -> tuple[dict[str, int], dict[str, Any] | None]:
    """Fixed spaces pass through; ``auto`` uses MP2 natural occupations (capped at 10)."""
    if target.kind != "molecular_ground_energy" or target.active != "auto":
        return (target.active if target.kind == "molecular_ground_energy" else None), None
    from chemflow.agent.diagnose import diagnose_model

    diagnostics = diagnose_model(target.molecule, target.electronic_model)
    return diagnostics["suggested_active"], diagnostics


# ---- grading -----------------------------------------------------------------------

def scalars(value: Any, unit: str = "hartree") -> list[float] | None:
    """Energies (or model values) of an output in ``unit``; None when not numeric."""
    import qcl

    if isinstance(value, qcl.Estimate):
        try:
            return [value.in_units(unit).value]
        except (TypeError, ValueError):
            return [value.value]
    if isinstance(value, qcl.Ranking):
        return [item.in_units(unit).value for item in value.values]
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, qcl.Estimate) for item in value):
        return [item.in_units(unit).value for item in value]
    if isinstance(value, tuple) and value and isinstance(value[0], qcl.Estimate):
        return scalars(value[0], unit)
    if isinstance(value, qcl.SiteScores):
        return [site["score"] for site in value.sites]
    return None


def error(value: Any, reference: Any, unit: str) -> float:
    """Signed-gate error of an output against the reference output (max over points)."""
    import numpy as np

    import qcl

    if isinstance(value, qcl.SpectralEstimate):
        if (not isinstance(reference, qcl.SpectralEstimate) or
                value.omega != reference.omega):
            return math.inf
        return float(np.trapezoid(np.abs(np.asarray(value.spectral) -
                                         np.asarray(reference.spectral)), value.omega))
    left, right = scalars(value, unit), scalars(reference, unit)
    if left is None or right is None or len(left) != len(right):
        return math.inf
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=math.inf)


def meet(relations: Any) -> str | None:
    """One relation for an output whose parts may each carry one."""
    from chemflow.relations import TargetRelation

    if relations is None or isinstance(relations, str):
        return relations
    flat = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for part in item:
                walk(part)
        elif isinstance(item, dict):
            for part in item.values():
                walk(part)
        else:
            flat.append(item)

    walk(relations)
    if not flat or None in flat:
        return None
    order = [TargetRelation(item) for item in flat]
    if TargetRelation.model_change in order:
        return "model_change"
    if all(item == order[0] for item in order):
        return order[0].value
    if set(order) <= {TargetRelation.equivalent, TargetRelation.stochastic_estimate}:
        return "stochastic_estimate"
    return "approximate"


@dataclass
class Graded:
    name: str
    entry: Entry
    value: Any
    relation: str | None
    error: float
    gates: list[dict[str, Any]]
    passed: bool
    failure: str | None
    certificate: str | None = None
    validity: dict[str, Any] | None = None


def grade(task_spec: Any, entry: Entry, value: Any, reference: Any,
          relation: str | None) -> Graded:
    from chemflow.relations import TargetRelation, satisfies

    gates = []
    failures = []
    measured = math.inf
    for gate in task_spec.gates:
        if gate.kind in ("abs_error_vs_exact", "l1_error_vs_exact"):
            unit = gate.unit if gate.kind == "abs_error_vs_exact" else "dimensionless"
            measured = error(value, reference, unit)
            passed = math.isfinite(measured) and measured <= gate.threshold
            gates.append({"kind": gate.kind, "threshold": gate.threshold, "unit": gate.unit,
                          "error": measured if math.isfinite(measured) else None,
                          "passed": passed})
            if not passed:
                failures.append("threshold exceeded")
        elif gate.kind == "relation_at_least":
            passed = relation is not None and satisfies(TargetRelation(relation),
                                                        TargetRelation(gate.relation))
            gates.append({"kind": gate.kind, "relation": gate.relation, "actual": relation,
                          "passed": passed})
            if not passed:
                failures.append("Relation does not match required relation")
    passed = bool(gates) and all(gate["passed"] for gate in gates)
    return Graded(entry.name, entry, value, relation, measured, gates, passed,
                  "; ".join(failures) or None)


def target_relation(task_spec: Any, inferred: Any) -> str | None:
    """Model-defined targets are equivalent to the trusted plan computing their model."""
    if task_spec.target.kind in MODEL_DEFINED:
        return "equivalent"
    return meet(inferred)


def unit_for(task_spec: Any) -> str:
    gates = [gate for gate in task_spec.gates if gate.kind == "abs_error_vs_exact"]
    if gates:
        try:
            Q(1, gates[0].unit).to("hartree")
            return "hartree"
        except (TypeError, ValueError):
            return gates[0].unit
    return "hartree"
