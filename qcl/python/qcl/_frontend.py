from __future__ import annotations

import inspect
import json
import linecache
import math
import os
import re
import resource
import signal
import subprocess
import sys
import tomllib
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import update_wrapper
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any, get_type_hints

from qcl._qcl import CI_MAX_DETERMINANTS, GraphBuilder, parse_cluster_json
from qcl._tensor_units import product_unit


@dataclass(frozen=True)
class Device:
    id: str
    kind: str
    billing: str = "none"
    billing_account: str | None = None
    threads: int | None = None
    memory_gb: float | None = None
    qubits: int | None = None
    site: str | None = None
    simulator_host: str | None = None
    simulator_method: str | None = None
    provider: str | None = None
    backend: str | None = None
    quantum_links: tuple[tuple[str, str], ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    def with_options(self, **options: Any) -> Device:
        allowed = {"cpu": {"threads"}, "gpu": {"precision", "memory_limit_gb"},
                   "qpu": {"transpile", "mitigate", "max_shots"}}[self.kind]
        invalid = set(options) - allowed
        if invalid:
            raise CompileError([{"code": "QCL-E202", "source": _source(),
                                 "message": f"Invalid {self.kind} option: {sorted(invalid)}",
                                 "hint": f"Allowed options: {sorted(allowed)}"}])
        value_error = (
            "threads" in options and (type(options["threads"]) is not int or
                                      options["threads"] < 1 or
                                      self.threads is not None and
                                      options["threads"] > self.threads)
            or "precision" in options and options["precision"] not in
            ("fp64", "fp32", "tf32")
            or "memory_limit_gb" in options and (
                type(options["memory_limit_gb"]) not in (int, float) or
                options["memory_limit_gb"] <= 0 or
                self.memory_gb is not None and options["memory_limit_gb"] > self.memory_gb)
            or "transpile" in options and (
                type(options["transpile"]) is not int or options["transpile"] < 0)
            or "max_shots" in options and (
                type(options["max_shots"]) is not int or options["max_shots"] < 1)
            or "mitigate" in options and options["mitigate"] not in ("none", "readout")
        )
        if value_error:
            raise CompileError([{"code": "QCL-E202", "source": _source(),
                                 "message": f"Invalid {self.kind} option value",
                                 "hint": "Use a supported value within the device capacity"}])
        return replace(self, options={**self.options, **options})


@dataclass(frozen=True)
class Cluster:
    cpus: tuple[Device, ...]
    gpus: tuple[Device, ...]
    qpus: tuple[Device, ...]
    links: tuple[dict[str, Any], ...] = ()
    quantum_links: tuple[dict[str, Any], ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> Cluster:
        config = json.loads(parse_cluster_json(Path(path).read_text()))
        quantum_links = tuple(tuple(link["between"])
                              for link in config["quantum_link"])
        return cls(
            cpus=tuple(Device(item["id"], "cpu", threads=item["threads"])
                       for item in config["cpu"]),
            gpus=tuple(Device(item["id"], "gpu", memory_gb=item["memory_gb"])
                       for item in config["gpu"]),
            qpus=tuple(Device(item["id"], "qpu", billing=item["billing"] or "none",
                               billing_account=item["billing_account"],
                               qubits=item["qubits"], site=item["site"],
                               simulator_host=item["simulator"]["host"]
                               if item["simulator"] else None,
                               simulator_method=item["simulator"]["method"]
                               if item["simulator"] else None,
                               provider=item["provider"], backend=item["backend"],
                               quantum_links=quantum_links)
                       for item in config["qpu"]),
            links=tuple(config["link"]), quantum_links=tuple(config["quantum_link"]),
        )

    def __getitem__(self, identifier: str) -> Device:
        for device in (*self.cpus, *self.gpus, *self.qpus):
            if device.id == identifier:
                return device
        raise KeyError(identifier)

    @classmethod
    def local(cls) -> Cluster:
        cpu = Device("cpu0", "cpu", threads=os.cpu_count())
        gpus: tuple[Device, ...] = ()
        try:
            import cupy

            detected = []
            for index in range(cupy.cuda.runtime.getDeviceCount()):
                try:
                    memory_gb = (cupy.cuda.runtime.getDeviceProperties(index)[
                        "totalGlobalMem"] / 1_000_000_000)
                except RuntimeError:
                    memory_gb = None
                detected.append(Device(f"gpu{index}", "gpu", memory_gb=memory_gb))
            gpus = tuple(detected)
        except (ImportError, RuntimeError):
            pass
        hosts = gpus or (cpu,)
        simulators = tuple(Device(f"sim{index}", "qpu", qubits=20,
                                  simulator_host=host.id,
                                  simulator_method="statevector")
                           for index, host in enumerate(hosts))
        return cls(cpus=(cpu,), gpus=gpus, qpus=simulators)


@dataclass(frozen=True)
class Molecule:
    atoms: str
    basis: str
    charge: int = 0
    spin: int = 0
    unit: str = "angstrom"

    def bonds(self, *, elements: tuple[str, str] | None = None) -> list[tuple[int, int]]:
        from chemflow.domains.reactivity import enumerate_bonds

        selector = "-".join(elements) if elements is not None else None
        result = enumerate_bonds(None, {"atoms": self.atoms, "charge": self.charge,
                                        "unit": self.unit}, selector)
        return [tuple(item["atoms"]) for item in result["bonds"]]

    def cleave(self, bond: tuple[int, int]) -> tuple[Molecule, Molecule]:
        from chemflow.domains.reactivity import homolysis

        result = homolysis(None, {"atoms": self.atoms, "charge": self.charge,
                                  "spin": self.spin, "unit": self.unit}, list(bond))
        return tuple(Molecule(fragment["atoms"], self.basis, fragment["charge"],
                              fragment["spin"], fragment["unit"])
                     for fragment in result["fragments"])

    def set_separation(self, frag_a: list[int] | tuple[int, ...],
                       frag_b: list[int] | tuple[int, ...], distance: float) -> Molecule:
        from chemflow.domains.paths import coordinates, geometry_transform

        symbols, _positions = coordinates({"atoms": self.atoms, "unit": self.unit})
        first, second = set(frag_a), set(frag_b)
        if (not first or not second or first & second or
                first | second != set(range(len(symbols))) or
                len(first) != len(frag_a) or len(second) != len(frag_b)):
            raise ValueError("Dimer fragments must partition the atom indices")
        transformed = geometry_transform(
            None, {"atoms": self.atoms, "unit": self.unit,
                   "charge": self.charge, "spin": self.spin},
            "translate_fragment", [], distance, fragment=list(frag_b))
        return Molecule(transformed["atoms"], self.basis, self.charge, self.spin,
                        transformed["unit"])

    def ghost(self, fragment: list[int] | tuple[int, ...]) -> Molecule:
        from pyscf.data.elements import charge

        from chemflow.domains.paths import coordinates

        symbols, positions = coordinates({"atoms": self.atoms, "unit": self.unit})
        selected = set(fragment)
        if (not selected or len(selected) == len(symbols) or
                len(selected) != len(fragment) or
                any(type(index) is not int or index < 0 or index >= len(symbols)
                    for index in fragment)):
            raise ValueError("Ghost fragment must be a proper atom-index subset")
        atoms = "; ".join(
            ("ghost-" if index in selected else "") + symbol + " " +
            " ".join(f"{value:.15g}" for value in positions[index])
            for index, symbol in enumerate(symbols))
        electrons = sum(charge(symbol) for index, symbol in enumerate(symbols)
                        if index not in selected)
        return Molecule(atoms, self.basis, charge=0, spin=electrons % 2)

    def _edit_coordinate(self, kind: str, atoms: tuple[int, ...], value: float) -> Molecule:
        from chemflow.domains.paths import geometry_transform

        transformed = geometry_transform(
            None, {"atoms": self.atoms, "unit": self.unit,
                   "charge": self.charge, "spin": self.spin},
            f"set_{kind}", list(atoms), value)
        return Molecule(transformed["atoms"], self.basis, self.charge, self.spin,
                        transformed["unit"])

    def set_bond(self, i: int, j: int, distance: float) -> Molecule:
        return self._edit_coordinate("bond", (i, j), distance)

    def set_angle(self, i: int, j: int, k: int, degrees: float) -> Molecule:
        return self._edit_coordinate("angle", (i, j, k), degrees)

    def set_dihedral(self, i: int, j: int, k: int, l: int, degrees: float) -> Molecule:
        return self._edit_coordinate("dihedral", (i, j, k, l), degrees)


@dataclass(frozen=True)
class Active:
    electrons: int | None = None
    orbitals: int | None = None
    max_orbitals: int | None = None


@dataclass(frozen=True)
class Analytic:
    pass


@dataclass(frozen=True)
class FiniteDifference:
    step: float = 1e-3


@dataclass(frozen=True)
class CircuitCut:
    max_cuts: int


@dataclass(frozen=True)
class Teleport:
    pass


@dataclass(frozen=True)
class Budget:
    qpu_shots: int | None = None
    qpu_s: float | None = None
    qpu_cost: float | None = None
    wall_s: float | None = None
    cpu_core_s: float | None = None

    def __post_init__(self) -> None:
        if self.qpu_shots is not None and (
                type(self.qpu_shots) is not int or self.qpu_shots < 0):
            raise ValueError("qpu_shots must be a nonnegative integer")
        for name in ("qpu_s", "qpu_cost", "wall_s", "cpu_core_s"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or
                                      not math.isfinite(value) or
                                      (value <= 0 if name in ("wall_s", "cpu_core_s")
                                       else value < 0)):
                raise ValueError(f"{name} must be finite and positive where required")


@dataclass(frozen=True)
class Bond:
    i: int
    j: int

    @property
    def atoms(self) -> tuple[int, ...]:
        return self.i, self.j


@dataclass(frozen=True)
class Angle:
    i: int
    j: int
    k: int

    @property
    def atoms(self) -> tuple[int, ...]:
        return self.i, self.j, self.k


@dataclass(frozen=True)
class Dihedral:
    i: int
    j: int
    k: int
    l: int

    @property
    def atoms(self) -> tuple[int, ...]:
        return self.i, self.j, self.k, self.l


@dataclass(frozen=True)
class PES:
    function: Callable[[Any, Any], Any]
    gradient: Analytic | FiniteDifference = Analytic()


@dataclass(frozen=True)
class PCM:
    name: str
    eps: float | None = None

    def __post_init__(self) -> None:
        from pyscf.solvent.smd import solvent_db

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PCM solvent name must be nonempty")
        eps = self.eps
        if eps is None:
            if self.name not in solvent_db:
                raise ValueError("Unknown PCM solvent name")
            eps = float(solvent_db[self.name][5])
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("PCM dielectric constant must be positive and finite")
        object.__setattr__(self, "eps", float(eps))


@dataclass(frozen=True)
class SMD:
    name: str

    def __post_init__(self) -> None:
        from pyscf.solvent.smd import solvent_db

        if not isinstance(self.name, str) or self.name not in solvent_db:
            raise ValueError("Unknown SMD solvent name")


@dataclass(frozen=True)
class _NamedTbliteSolvent:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tblite solvent name must be nonempty")


@dataclass(frozen=True)
class ALPB(_NamedTbliteSolvent):
    pass


@dataclass(frozen=True)
class GBSA(_NamedTbliteSolvent):
    pass


@dataclass(frozen=True)
class Lattice:
    geometry: str
    shape: tuple[int, ...]

    @property
    def n_sites(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Sector:
    n: int
    sz: float


@dataclass(frozen=True)
class Observable:
    kind: str
    sites: tuple[int, int]


@dataclass(frozen=True)
class Estimate:
    value: float
    unit: str
    variance: float
    provenance: dict[str, Any]

    def in_units(self, unit: str) -> Estimate:
        from chemflow.units import Q

        converted = Q(self.value, self.unit).to(unit)
        factor = converted.value / self.value if self.value else Q(1, self.unit).to(unit).value
        return Estimate(converted.value, unit, self.variance * factor * factor,
                        self.provenance)


@dataclass(frozen=True)
class QubitOperator:
    terms: dict[str, float]
    n_qubits: int
    encoding_id: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Bitstrings:
    counts: dict[str, int]
    shots: int
    source: str
    device: str
    job_ids: tuple[str, ...]
    provenance: dict[str, Any]
    parts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Ranking:
    order: tuple[int, ...]
    unresolved: tuple[tuple[int, int], ...]
    values: tuple[Estimate, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    relation: str
    difference: float
    combined_sigma: float
    unit: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SiteScores:
    sites: tuple[dict[str, Any], ...]
    top_site: int | None
    evidence: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class GeometryResult:
    molecule: Molecule
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Tensor:
    data: Any
    device: str
    unit: str
    layout: str
    provenance: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> str:
        return str(self.data.dtype)

    def __dlpack__(self, stream: Any = None) -> Any:
        return self.data.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.data.__dlpack_device__()


@dataclass(frozen=True)
class Eigensystem:
    values: Tensor
    vectors: Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SvdResult:
    u: Tensor
    s: Tensor
    vh: Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SpectralEstimate:
    omega: tuple[float, ...]
    spectral: tuple[float, ...]
    unit: str
    relation: str
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ReactionPath:
    transition: Molecule
    endpoints: tuple[Molecule, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FrequencyResult:
    freq_wavenumber: tuple[float, ...]
    n_imaginary: int
    max_gradient: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LucjParameters:
    vector: tuple[float, ...]
    provenance: dict[str, Any]


class CompileError(Exception):
    def __init__(self, diagnostics: list[dict[str, str]]):
        self.diagnostics = []
        for item in diagnostics:
            enriched = dict(item)
            match = re.fullmatch(r"(.*):(\d+):(\d+)-(\d+):(\d+)",
                                 item.get("source", ""))
            if match:
                file, start_line, start_column, end_line, end_column = match.groups()
                enriched["source_span"] = {
                    "file": file, "start_line": int(start_line),
                    "start_column": int(start_column),
                    "end_line": int(end_line), "end_column": int(end_column),
                }
                expression = linecache.getline(file, int(start_line)).strip()
                if expression:
                    enriched["expression"] = expression
            self.diagnostics.append(enriched)
        super().__init__("; ".join(f"{item['code']}: {item['message']}"
                                  for item in self.diagnostics))


_active_graph: ContextVar[GraphBuilder | None] = ContextVar("qcl_graph", default=None)
_active_patterns: ContextVar[list[Any] | None] = ContextVar("qcl_patterns", default=None)
_active_kernel_gates: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "qcl_kernel_gates", default=None)
_active_kernel_bits: ContextVar[list[Any] | None] = ContextVar(
    "qcl_kernel_bits", default=None)


def _source() -> str:
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None and frame.f_back.f_back is not None
    caller = frame.f_back.f_back
    position = inspect.getframeinfo(caller).positions
    column = position.col_offset + 1 if position and position.col_offset is not None else 1
    end_line = position.end_lineno if position and position.end_lineno else caller.f_lineno
    end_column = (position.end_col_offset + 1 if position and
                  position.end_col_offset is not None else column)
    return f"{caller.f_code.co_filename}:{caller.f_lineno}:{column}-{end_line}:{end_column}"


class _Value:
    def __init__(self, graph: GraphBuilder, node_id: int, kind: str,
                 molecule: Molecule | None = None, parent: int | None = None,
                 unit: str | None = None, family: str | None = None,
                 shot_split: tuple[int, ...] | None = None,
                 lattice: Lattice | None = None, max_items: int | None = None,
                 item_kind: str | None = None,
                 partition: CircuitCut | Teleport | None = None):
        self.graph = graph
        self.node_id = node_id
        self.kind = kind
        self.molecule = molecule
        self.parent = parent
        self.unit = unit
        self.family = family
        self.shot_split = shot_split
        self.lattice = lattice
        self.max_items = max_items
        self.item_kind = item_kind
        self.partition = partition

    def _edit_coordinate(self, kind: str, atoms: tuple[int, ...], value: float) -> _Value:
        if (self.kind != "molecule" or self.molecule is None or
                len(atoms) != {"bond": 2, "angle": 3, "dihedral": 4}[kind] or
                any(type(index) is not int or index < 0 for index in atoms) or
                type(value) not in (int, float) or not math.isfinite(value)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Invalid symbolic molecule geometry edit",
                                 "hint": "Use valid atom indices and a finite coordinate"}])
        node = self.graph.molecule_edit(self.node_id, kind, list(atoms), float(value), _source())
        return _Value(self.graph, node, "molecule", self.molecule)

    def set_bond(self, i: int, j: int, distance: float) -> _Value:
        return self._edit_coordinate("bond", (i, j), distance)

    def set_angle(self, i: int, j: int, k: int, degrees: float) -> _Value:
        return self._edit_coordinate("angle", (i, j, k), degrees)

    def set_dihedral(self, i: int, j: int, k: int, l: int, degrees: float) -> _Value:
        return self._edit_coordinate("dihedral", (i, j, k, l), degrees)

    def connects(self, reactant: Molecule, product: Molecule) -> _Value:
        if self.kind != "reaction_path" or self.molecule is None:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Connectivity requires an IRC reaction path",
                                 "hint": "Call qcl.irc(pes, transition).to(cpu)"}])
        if (not isinstance(reactant, Molecule) or not isinstance(product, Molecule) or
                any(item.charge != self.molecule.charge or
                    item.spin != self.molecule.spin or item.basis != self.molecule.basis
                    for item in (reactant, product))):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "IRC endpoints differ from the transition-state model",
                                 "hint": "Use endpoints with the same charge, spin and basis"}])
        reactant_id, _ = _molecule_node(self.graph, reactant)
        product_id, _ = _molecule_node(self.graph, product)
        node = self.graph.path_connects(self.node_id, reactant_id, product_id, _source())
        return _Value(self.graph, node, "boolean")

    def to(self, device: Device | list[Device] | tuple[Device, ...]) -> _Value:
        if self.partition is not None:
            if (not isinstance(device, (list, tuple)) or
                    any(not isinstance(item, Device) for item in device)):
                raise TypeError(".to() requires QCL devices")
            if len(device) < 2:
                raise CompileError([{"code": "QCL-E204", "source": _source(),
                                     "message": "Partition needs multiple QPUs",
                                     "hint": "Place the partitioned operation on a QPU list"}])
            if isinstance(self.partition, Teleport):
                links = {frozenset(link) for item in device
                         for link in item.quantum_links}
                if any(frozenset((left.id, right.id)) not in links
                       for left, right in pairwise(device)):
                    raise CompileError([{"code": "QCL-E208", "source": _source(),
                                         "message": "Teleport needs a quantum link between QPUs",
                                         "hint": "Declare quantum_link in the cluster topology"}])
            raise CompileError([{"code": "QCL-E203", "source": _source(),
                                 "message": "No partitioned QPU implementation is registered",
                                 "hint": "Use independent shots or measurement-group splitting"}])
        if isinstance(device, (list, tuple)):
            if not device or any(not isinstance(item, Device) for item in device):
                raise TypeError(".to() requires QCL devices")
            if len(device) == 1:
                if self.shot_split is not None and len(self.shot_split) != 1:
                    raise CompileError([{"code": "QCL-E204", "source": _source(),
                                         "message": "Shot split does not match device count",
                                         "hint": "Provide one shot count per QPU"}])
                device = device[0]
            else:
                nodes = json.loads(self.graph.graph_json())["nodes"]
                source = nodes[self.node_id]
                if (self.kind == "estimate" and source["op"] == "estimate.quantum" and
                        source["device"] is None):
                    if len({item.id for item in device}) != len(device) or (
                            len(device) > source["params"]["n_groups"]):
                        raise CompileError([{"code": "QCL-E204", "source": _source(),
                                             "message": "Pauli groups cannot split over these QPUs",
                                             "hint": "Use distinct QPUs with at least one group each"}])
                    node_id = self.graph.split_quantum_estimate(
                        self.node_id, json.dumps([asdict(item) for item in device]),
                        _source())
                    return _Value(self.graph, node_id, self.kind, self.molecule,
                                  self.parent, self.unit, self.family)
                if self.kind != "bitstrings" or len({item.id for item in device}) != len(device):
                    raise CompileError([{"code": "QCL-E204", "source": _source(),
                                         "message": "This value cannot split over device lanes",
                                         "hint": "Use distinct devices and a parallel operation"}])
                if self.shot_split is not None and len(self.shot_split) != len(device):
                    raise CompileError([{"code": "QCL-E204", "source": _source(),
                                         "message": "Shot split does not match device count",
                                         "hint": "Provide one shot count per QPU"}])
                total = json.loads(self.graph.graph_json())["nodes"][self.node_id]["params"]["shots"]
                counts = self.shot_split or tuple(divmod(total, len(device))[0] +
                                                   (index < total % len(device))
                                                   for index in range(len(device)))
                if any(count == 0 for count in counts):
                    raise CompileError([{"code": "QCL-E204", "source": _source(),
                                         "message": "Each QPU needs at least one shot",
                                         "hint": "Use fewer QPUs or more shots"}])
                node_id = self.graph.split_sample(
                    self.node_id, json.dumps([asdict(item) for item in device]),
                    list(counts), _source())
                return _Value(self.graph, node_id, self.kind, self.molecule, self.parent)
        if not isinstance(device, Device):
            raise TypeError(".to() requires a QCL device")
        if self.shot_split is not None and len(self.shot_split) != 1:
            raise CompileError([{"code": "QCL-E204", "source": _source(),
                                 "message": "Shot split does not match device count",
                                 "hint": "Pass a matching QPU device list"}])
        node_id = self.graph.place(self.node_id, json.dumps(asdict(device)), _source())
        return _Value(self.graph, node_id, self.kind, self.molecule, self.parent,
                      self.unit, self.family, lattice=self.lattice,
                      max_items=self.max_items, item_kind=self.item_kind)

    @property
    def energy(self) -> _Value:
        if self.kind not in ("scf_result", "mp2_result", "fci_result", "sci_result",
                             "ground_state"):
            raise TypeError("Only an SCF, MP2, CI or ground-state result has an energy projection")
        return _Value(self.graph, self.graph.energy(self.node_id, _source()),
                      "estimate", self.molecule, unit="hartree", family=self.family,
                      lattice=self.lattice)

    @property
    def coefficients(self) -> _Value:
        if self.kind != "fci_result":
            raise TypeError("Only an FCI result has CI coefficients")
        return _Value(self.graph, self.graph.ci_coefficients(self.node_id, _source()),
                      "tensor", self.molecule, parent=self.parent,
                      unit="dimensionless", family="CI coefficients")

    @property
    def values(self) -> _Value:
        if self.kind != "eigensystem":
            raise TypeError("Only an eigensystem has eigenvalues")
        return _Value(self.graph, self.graph.eigh_projection(
            self.node_id, "values", _source()), "tensor", self.molecule,
            unit=self.unit)

    @property
    def vectors(self) -> _Value:
        if self.kind != "eigensystem":
            raise TypeError("Only an eigensystem has eigenvectors")
        return _Value(self.graph, self.graph.eigh_projection(
            self.node_id, "vectors", _source()), "tensor", self.molecule,
            unit="dimensionless")

    def _svd_part(self, component: str, unit: str) -> _Value:
        if self.kind != "svd_result":
            raise TypeError("SVD components require an SVD result")
        return _Value(self.graph, self.graph.svd_projection(
            self.node_id, component, _source()), "tensor", self.molecule, unit=unit)

    @property
    def u(self) -> _Value:
        return self._svd_part("u", "dimensionless")

    @property
    def s(self) -> _Value:
        return self._svd_part("s", self.unit)

    @property
    def vh(self) -> _Value:
        return self._svd_part("vh", "dimensionless")

    def _orbital_energy(self, orbital: str, source: str) -> _Value:
        if self.kind != "scf_result":
            raise TypeError("Orbital energies require an SCF result")
        if self.family in ("UHF",) or (self.family or "").startswith("UKS:"):
            raise CompileError([{"code": "QCL-E101", "source": source,
                                 "message": "Unrestricted orbital energy needs a spin channel",
                                 "hint": "Use a spin-resolved orbital projection"}])
        return _Value(self.graph,
                      self.graph.orbital_energy(self.node_id, orbital, source),
                      "estimate", self.molecule, unit="hartree", family="SCF orbital")

    def orbital_energy(self, orbital: str, *, spin: str | None = None) -> _Value:
        source = _source()
        if orbital not in ("homo", "lumo") or spin not in (None, "alpha", "beta"):
            raise CompileError([{"code": "QCL-E101", "source": source,
                                 "message": "Unknown orbital or spin channel",
                                 "hint": "Use homo/lumo with alpha/beta for unrestricted SCF"}])
        unrestricted = self.family == "UHF" or (self.family or "").startswith("UKS:")
        if unrestricted and spin is None or not unrestricted and spin is not None:
            raise CompileError([{"code": "QCL-E101", "source": source,
                                 "message": "Orbital spin channel differs from SCF reference",
                                 "hint": "Set spin for UHF/UKS and omit it for restricted SCF"}])
        if spin is None:
            return self._orbital_energy(orbital, source)
        return _Value(self.graph,
                      self.graph.orbital_energy(self.node_id, f"{orbital}_{spin}", source),
                      "estimate", self.molecule, unit="hartree", family="SCF orbital")

    @property
    def homo(self) -> _Value:
        return self._orbital_energy("homo", _source())

    @property
    def lumo(self) -> _Value:
        return self._orbital_energy("lumo", _source())

    @property
    def n_imaginary(self) -> _Value:
        if self.kind != "frequencies":
            raise TypeError("Imaginary mode count requires frequencies")
        return _Value(self.graph, self.graph.imaginary_count(self.node_id, _source()),
                      "integer", self.molecule)

    def __eq__(self, other: object) -> _Value | bool:
        if self.kind == "integer" and type(other) is int:
            return _Value(self.graph, self.graph.integer_compare(self.node_id, other,
                                                                 True, _source()), "boolean")
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Python equality cannot compare computed QCL values",
                             "hint": "Use qcl.compare for Estimates or a typed projection"}])

    def __ne__(self, other: object) -> _Value | bool:
        if self.kind == "integer" and type(other) is int:
            return _Value(self.graph, self.graph.integer_compare(self.node_id, other,
                                                                 False, _source()), "boolean")
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Python inequality cannot compare computed QCL values",
                             "hint": "Use qcl.compare for Estimates or a typed projection"}])

    def in_units(self, unit: str) -> _Value:
        if self.kind != "estimate" or self.unit is None:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Unit conversion requires an Estimate",
                                 "hint": "Convert a numerical QCL result"}])
        from chemflow.units import Q

        try:
            Q(1, self.unit).to(unit)
        except (KeyError, ValueError) as error:
            raise CompileError([{"code": "QCL-E102", "source": _source(),
                                 "message": f"Cannot convert {self.unit} to {unit}",
                                 "hint": "Use a compatible energy unit"}]) from error
        return _Value(self.graph, self.graph.convert_unit(self.node_id, unit, _source()),
                      "estimate", self.molecule, unit=unit, family=self.family)

    def _binary(self, other: Any, operation: str, source: str) -> _Value:
        if (not isinstance(other, _Value) or self.kind != "estimate" or
                other.kind != "estimate" or self.graph is not other.graph):
            raise CompileError([{"code": "QCL-E101", "source": source,
                                 "message": "Estimate arithmetic requires two QCL Estimates",
                                 "hint": "Use compatible numerical workflow results"}])
        if self.unit != other.unit:
            raise CompileError([{"code": "QCL-E102", "source": source,
                                 "message": "Estimate units differ",
                                 "hint": "Convert units explicitly with .in_units()"}])
        mixed_rrho = (operation == "add" and
                      {self.family, other.family} == {"FCI", "RRHO correction"})
        mixed_hf = ({self.family, other.family} <= {"HF", "RHF", "ROHF"})
        if ((self.family != other.family and not mixed_rrho and not mixed_hf) or
                self.molecule is None or
                other.molecule is None or self.molecule.basis != other.molecule.basis):
            raise CompileError([{"code": "QCL-E103", "source": source,
                                 "message": "Estimate methods or basis sets are incompatible",
                                 "hint": "Use matching method families and basis sets"}])
        return _Value(self.graph,
                      self.graph.estimate_binary(operation, self.node_id,
                                                 other.node_id, source),
                      "estimate", self.molecule, unit=self.unit,
                      family=("FCI+RRHO" if mixed_rrho else "HF" if mixed_hf else
                              self.family))

    def __add__(self, other: Any) -> _Value:
        return self._binary(other, "add", _source())

    def __sub__(self, other: Any) -> _Value:
        return self._binary(other, "sub", _source())

    def _scale(self, factor: Any, source: str) -> _Value:
        if (self.kind != "estimate" or type(factor) not in (int, float) or
                not math.isfinite(factor)):
            raise CompileError([{"code": "QCL-E101", "source": source,
                                 "message": "Estimate scaling needs a finite scalar",
                                 "hint": "Multiply or divide an Estimate by a finite number"}])
        return _Value(self.graph,
                      self.graph.estimate_scale(self.node_id, float(factor), source),
                      "estimate", self.molecule, unit=self.unit, family=self.family)

    def __mul__(self, other: Any) -> _Value:
        return self._scale(other, _source())

    def __rmul__(self, other: Any) -> _Value:
        return self._scale(other, _source())

    def __truediv__(self, other: Any) -> _Value:
        if type(other) not in (int, float) or not math.isfinite(other) or other == 0:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Estimate division needs a finite nonzero scalar",
                                 "hint": "Divide by a finite number other than zero"}])
        return self._scale(1 / other, _source())

    def __neg__(self) -> _Value:
        return self._scale(-1, _source())

    @property
    def hamiltonian(self) -> _Value:
        if self.kind != "active_space":
            raise TypeError("Only an active space has a Hamiltonian")
        return _Value(self.graph, self.graph.hamiltonian(self.node_id, _source()),
                      "fermion_operator", self.molecule, parent=self.node_id)

    @property
    def sector(self) -> _Value:
        if self.kind != "active_space":
            raise TypeError("Only an active space has a sector")
        return _Value(self.graph, self.graph.sector(self.node_id, _source()),
                      "sector", self.molecule, parent=self.node_id)

    def add_electrons(self, count: int) -> _Value:
        if self.kind != "sector" or self.parent is None or self.molecule is None:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Electron shift requires an active-space Sector",
                                 "hint": "Use active.sector.add_electrons(n)"}])
        if type(count) is not int:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Electron shift must be an integer",
                                 "hint": "Pass an integer electron count"}])
        nodes = json.loads(self.graph.graph_json())["nodes"]
        active_node = nodes[self.parent]
        while active_node["op"] == "copy":
            active_node = nodes[active_node["inputs"][0]]
        active = active_node["params"]
        n_orb = active["orbitals"]
        original = nodes[self.node_id]
        while original["op"] == "copy":
            original = nodes[original["inputs"][0]]
        if original["op"] == "sector.shift":
            n_alpha = original["params"]["n_alpha"]
            n_beta = original["params"]["n_beta"]
        else:
            n_alpha = (active["electrons"] + self.molecule.spin) // 2
            n_beta = (active["electrons"] - self.molecule.spin) // 2
        for _ in range(abs(count)):
            if count > 0:
                if (n_alpha <= n_beta and n_alpha < n_orb) or n_beta == n_orb:
                    n_alpha += 1
                else:
                    n_beta += 1
            elif n_alpha > n_beta or n_beta == 0:
                n_alpha -= 1
            else:
                n_beta -= 1
        if min(n_alpha, n_beta) < 0 or max(n_alpha, n_beta) > n_orb:
            raise CompileError([{"code": "QCL-E105", "source": _source(),
                                 "message": "Shifted electron sector exceeds the active space",
                                 "hint": "Use a feasible charge sector or more orbitals"}])
        return _Value(self.graph,
                      self.graph.sector_shift(self.node_id, n_alpha, n_beta, _source()),
                      "sector", self.molecule, self.parent)

    def __bool__(self) -> bool:
        raise CompileError([{
            "code": "QCL-E210", "source": _source(),
            "message": "Computed value cannot control a Python branch",
            "hint": "Use qcl.cond or a driver pattern for data-dependent control flow",
        }])


class _SCF:
    def rhf(self, molecule: Molecule | _Value, *, eri: _Value | None = None,
            guess: _Value | None = None, solvent: PCM | SMD | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call scf.rhf inside a QCL workflow compilation")
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("scf.rhf requires a QCL Molecule")
        if eri is not None and (not isinstance(eri, _Value) or eri.kind != "eri" or
                                eri.graph is not graph or eri.molecule != metadata):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "SCF integrals refer to a different molecule",
                                 "hint": "Use int2e on the same molecule"}])
        if guess is not None and (
                not isinstance(guess, _Value) or guess.kind != "scf_result" or
                guess.graph is not graph or guess.family != "RHF" or
                guess.molecule is None or
                (guess.molecule.basis, guess.molecule.charge, guess.molecule.spin) !=
                (metadata.basis, metadata.charge, metadata.spin)):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "RHF guess has an incompatible reference or basis",
                                 "hint": "Use a placed RHF result with the same basis and electron sector"}])
        if solvent is not None and not isinstance(solvent, (PCM, SMD)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "SCF solvent requires PCM or SMD",
                                 "hint": "Use qcl.PCM(name) or qcl.SMD(name)"}])
        node = (graph.rhf_with_options(mol_id, eri.node_id if eri is not None else None,
                                       guess.node_id if guess is not None else None,
                                       type(solvent).__name__.lower() if solvent else None,
                                       solvent.name if solvent else None,
                                       solvent.eps if isinstance(solvent, PCM) else None,
                                       _source())
                if eri is not None or guess is not None or solvent is not None
                else graph.rhf(mol_id, _source()))
        return _Value(graph, node, "scf_result", metadata,
                      family="RHF")

    def hf(self, molecule: Molecule | _Value) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call scf.hf inside a QCL workflow compilation")
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("scf.hf requires a QCL Molecule")
        assert metadata is not None
        return _Value(graph, graph.hf(mol_id, _source()), "scf_result", metadata,
                      family="ROHF" if metadata.spin else "RHF")

    def rohf(self, molecule: Molecule | _Value) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call scf.rohf inside a QCL workflow compilation")
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("scf.rohf requires a QCL Molecule")
        return _Value(graph, graph.rohf(mol_id, _source()), "scf_result", metadata,
                      family="ROHF")

    def uhf(self, molecule: Molecule | _Value) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call scf.uhf inside a QCL workflow compilation")
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("scf.uhf requires a QCL Molecule")
        return _Value(graph, graph.uhf(mol_id, _source()), "scf_result", metadata,
                      family="UHF")


scf = _SCF()


class _DFT:
    @staticmethod
    def rks(molecule: Molecule | _Value, *, xc: str,
            solvent: PCM | SMD | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call dft.rks inside a QCL workflow compilation")
        if type(xc) is not str or not xc.strip():
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "RKS requires a named XC functional",
                                 "hint": "Pass a PySCF functional such as 'pbe'"}])
        from pyscf.dft import libxc

        try:
            libxc.parse_xc(xc)
        except (KeyError, ValueError) as error:
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "RKS XC functional is unknown",
                                 "hint": "Use a functional recognized by PySCF LibXC"}]) from error
        if solvent is not None and not isinstance(solvent, (PCM, SMD)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "RKS solvent requires PCM or SMD",
                                 "hint": "Use qcl.PCM(name) or qcl.SMD(name)"}])
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("dft.rks requires a QCL Molecule")
        return _Value(graph, graph.dft_rks(
            mol_id, xc, type(solvent).__name__.lower() if solvent else None,
            solvent.name if solvent else None,
            solvent.eps if isinstance(solvent, PCM) else None,
            _source()), "scf_result",
                      metadata, family=f"RKS:{xc.lower()}")

    @staticmethod
    def uks(molecule: Molecule | _Value, *, xc: str,
            solvent: PCM | SMD | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call dft.uks inside a QCL workflow compilation")
        if type(xc) is not str or not xc.strip():
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "UKS requires a named XC functional",
                                 "hint": "Pass a PySCF functional such as 'pbe'"}])
        from pyscf.dft import libxc

        try:
            libxc.parse_xc(xc)
        except (KeyError, ValueError) as error:
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "UKS XC functional is unknown",
                                 "hint": "Use a functional recognized by PySCF LibXC"}]) from error
        if solvent is not None and not isinstance(solvent, (PCM, SMD)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "UKS solvent requires PCM or SMD",
                                 "hint": "Use qcl.PCM(name) or qcl.SMD(name)"}])
        if isinstance(molecule, _Value) and molecule.kind == "molecule" and (
                molecule.graph is graph):
            mol_id, metadata = molecule.node_id, molecule.molecule
        elif isinstance(molecule, Molecule):
            mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                    molecule.spin, molecule.unit, _source())
            metadata = molecule
        else:
            raise TypeError("dft.uks requires a QCL Molecule")
        return _Value(graph, graph.dft_uks(
            mol_id, xc, type(solvent).__name__.lower() if solvent else None,
            solvent.name if solvent else None,
            solvent.eps if isinstance(solvent, PCM) else None,
            _source()), "scf_result",
                      metadata, family=f"UKS:{xc.lower()}")


dft = _DFT()


def mp2(mean_field: _Value) -> _Value:
    if not isinstance(mean_field, _Value) or mean_field.kind != "scf_result":
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "MP2 requires an SCF result",
                             "hint": "Pass a placed RHF result"}])
    if mean_field.family != "RHF":
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Current MP2 implementation requires RHF orbitals",
                             "hint": "Use scf.rhf or closed-shell scf.hf"}])
    return _Value(mean_field.graph, mean_field.graph.mp2(mean_field.node_id, _source()),
                  "mp2_result", mean_field.molecule, family="MP2")


def int2e(molecule: Molecule) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call int2e inside workflow compilation")
    if not isinstance(molecule, Molecule):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "int2e requires a static Molecule",
                             "hint": "Pass qcl.Molecule"}])
    mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                            molecule.spin, molecule.unit, _source())
    return _Value(graph, graph.int2e(mol_id, _source()), "eri", molecule)


def int1e(molecule: Molecule | _Value, kind: str) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call int1e inside workflow compilation")
    if kind not in ("overlap", "kinetic", "nuclear", "dipole", "quadrupole"):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Unsupported one-electron integral kind",
                             "hint": "Use overlap, kinetic, nuclear, dipole or quadrupole"}])
    mol_id, metadata = _molecule_node(graph, molecule)
    return _Value(graph, graph.int1e(mol_id, kind, _source()),
                  "tensor", metadata,
                  unit={"overlap": "dimensionless", "kinetic": "hartree",
                        "nuclear": "hartree", "dipole": "bohr",
                        "quadrupole": "bohr^2"}[kind])


def int3c(molecule: Molecule | _Value, auxbasis: str) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call int3c inside workflow compilation")
    if type(auxbasis) is not str or not auxbasis.strip():
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "int3c requires a nonempty auxiliary basis name",
                             "hint": "Pass a PySCF auxiliary basis such as 'weigend'"}])
    mol_id, metadata = _molecule_node(graph, molecule)
    return _Value(graph, graph.int3c(mol_id, auxbasis, _source()),
                  "tensor", metadata, unit="hartree")


def cholesky(molecule: Molecule | _Value, tol: float) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call cholesky inside workflow compilation")
    if type(tol) not in (int, float) or not math.isfinite(tol) or tol <= 0:
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Cholesky tolerance must be finite and positive",
                             "hint": "Pass a positive residual tolerance such as 1e-8"}])
    mol_id, metadata = _molecule_node(graph, molecule)
    return _Value(graph, graph.cholesky(mol_id, float(tol), _source()),
                  "tensor", metadata, unit="hartree^1/2")


def eigh(matrix: _Value, metric: _Value | None = None) -> _Value:
    source = _source()
    if (not isinstance(matrix, _Value) or matrix.kind != "tensor" or
            metric is not None and (not isinstance(metric, _Value) or
                                    metric.kind != "tensor" or
                                    metric.graph is not matrix.graph)):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "eigh requires placed Tensor inputs",
                             "hint": "Pass one or two QCL Tensors"}])
    if metric is not None and metric.unit != "dimensionless":
        raise CompileError([{"code": "QCL-E102", "source": source,
                             "message": "Generalized eigensystem metric must be dimensionless",
                             "hint": "Use an overlap Tensor as the metric"}])
    return _Value(matrix.graph, matrix.graph.eigh(
        matrix.node_id, metric.node_id if metric is not None else None, source),
        "eigensystem", matrix.molecule, unit=matrix.unit)


def orthogonalize(metric: _Value) -> _Value:
    source = _source()
    if not isinstance(metric, _Value) or metric.kind != "tensor":
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "orthogonalize requires an overlap Tensor",
                             "hint": "Pass a placed qcl.int1e(mol, 'overlap')"}])
    if metric.unit != "dimensionless":
        raise CompileError([{"code": "QCL-E102", "source": source,
                             "message": "Orthogonalization metric must be dimensionless",
                             "hint": "Use a dimensionless overlap Tensor"}])
    return _Value(metric.graph, metric.graph.orthogonalize(metric.node_id, source),
                  "tensor", metric.molecule, unit="dimensionless")


def solve(matrix: _Value, rhs: _Value) -> _Value:
    source = _source()
    if (not isinstance(matrix, _Value) or matrix.kind != "tensor" or
            not isinstance(rhs, _Value) or rhs.kind != "tensor" or
            matrix.graph is not rhs.graph):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "solve requires Tensors from one workflow",
                             "hint": "Pass a matrix and a right-hand-side Tensor"}])
    if matrix.molecule is not None and rhs.molecule is not None and (
            matrix.molecule != rhs.molecule):
        raise CompileError([{"code": "QCL-E103", "source": source,
                             "message": "Linear-system tensors use different molecules",
                             "hint": "Use inputs from the same molecular model"}])
    unit = (rhs.unit if matrix.unit == "dimensionless" else
            "dimensionless" if rhs.unit == matrix.unit else
            f"1/{matrix.unit}" if rhs.unit == "dimensionless" else
            f"{rhs.unit}/{matrix.unit}")
    return _Value(matrix.graph, matrix.graph.solve(
        matrix.node_id, rhs.node_id, unit, source), "tensor", matrix.molecule, unit=unit)


def svd(matrix: _Value) -> _Value:
    source = _source()
    if not isinstance(matrix, _Value) or matrix.kind != "tensor":
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "svd requires a Tensor matrix",
                             "hint": "Pass a placed QCL Tensor"}])
    return _Value(matrix.graph, matrix.graph.svd(matrix.node_id, source),
                  "svd_result", matrix.molecule, unit=matrix.unit)


def transform(integrals: _Value, coefficients: _Value) -> _Value:
    source = _source()
    if (not isinstance(integrals, _Value) or integrals.kind not in ("tensor", "eri") or
            not isinstance(coefficients, _Value) or coefficients.kind != "tensor" or
            integrals.graph is not coefficients.graph):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "transform requires integrals and coefficient Tensor",
                             "hint": "Pass AO integrals and an eigensystem's vectors"}])
    if integrals.molecule != coefficients.molecule:
        raise CompileError([{"code": "QCL-E103", "source": source,
                             "message": "AO integrals and coefficients use different molecules",
                             "hint": "Use orbitals from the same AO basis and geometry"}])
    if coefficients.unit != "dimensionless":
        raise CompileError([{"code": "QCL-E102", "source": source,
                             "message": "Orbital coefficients must be dimensionless",
                             "hint": "Pass an eigensystem's vectors, not its values"}])
    unit = "hartree" if integrals.kind == "eri" else integrals.unit
    return _Value(integrals.graph, integrals.graph.transform(
        integrals.node_id, coefficients.node_id, source),
        "tensor", integrals.molecule, unit=unit)


def contract(spec: str, *tensors: _Value) -> _Value:
    source = _source()
    if (not isinstance(spec, str) or not spec or len(tensors) < 1 or
            spec.split("->", 1)[0].count(",") + 1 != len(tensors) or
            any(not isinstance(tensor, _Value) or tensor.kind != "tensor" or
                tensor.graph is not tensors[0].graph for tensor in tensors)):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "Contraction requires a matching einsum spec and Tensors",
                             "hint": "Pass qcl.contract('ij,jk->ik', left, right)"}])
    unit = product_unit(tensor.unit for tensor in tensors)
    graph = tensors[0].graph
    return _Value(graph, graph.contract(spec, [tensor.node_id for tensor in tensors],
                                        source), "tensor", tensors[0].molecule, unit=unit)


class _Periodic:
    @staticmethod
    def scf(system: dict[str, Any], *, kmesh: tuple[int, int, int] = (1, 1, 1),
            method: str = "rhf", exxdiv: str | None = "ewald") -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call periodic.scf inside workflow compilation")
        if (not isinstance(system, dict) or
                not {"lattice", "atoms", "basis", "pseudo"} <= set(system) or
                set(system) - {"lattice", "atoms", "basis", "pseudo", "unit"} or
                not isinstance(kmesh, (list, tuple)) or len(kmesh) != 3 or
                any(type(size) is not int or size < 1 for size in kmesh) or
                method != "rhf" or exxdiv not in ("ewald", None)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Invalid periodic SCF system or options",
                                 "hint": "Use a pseudopotential system, positive kmesh and RHF"}])
        normalized = {**system, "unit": system.get("unit", "angstrom")}
        return _Value(graph, graph.periodic_scf(json.dumps(normalized, sort_keys=True),
                                                list(kmesh), method, exxdiv, _source()),
                      "estimate", unit="hartree")


periodic = _Periodic()


class _Dynamics:
    @staticmethod
    def spectral_exact(model: dict[str, Any], sector: dict[str, Any], *,
                       orbital: int, spin: str, omega: dict[str, Any],
                       eta: float) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call dynamics.spectral_exact inside workflow compilation")
        try:
            from chemflow.domains.algorithms import as_sector
            from chemflow.domains.dynamics import grid, impurity_arrays

            h1, _ = impurity_arrays(**model)
            particles = as_sector(sector)
            grid(omega, eta)
            if (particles.n_orb != len(h1) or type(orbital) is not int or
                    not 0 <= orbital < particles.n_orb or spin not in ("alpha", "beta")):
                raise ValueError("Spectral sector, orbital or spin differs from model")
        except (TypeError, KeyError, ValueError) as error:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": f"Invalid spectral input: {error}",
                                 "hint": "Match the impurity model, sector and frequency grid"}]) from error
        params = {"model": model, "sector": sector, "orbital": orbital,
                  "spin": spin, "omega": omega, "eta": eta}
        return _Value(graph, graph.spectral_exact(json.dumps(params, sort_keys=True),
                                                   _source()), "spectral_estimate")


dynamics = _Dynamics()


def active_space(mean_field: _Value, active: Active) -> _Value:
    if not isinstance(mean_field, _Value) or mean_field.kind != "scf_result":
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "active_space requires an SCF result",
                             "hint": "Pass a placed scf.hf or scf.rhf result"}])
    if not isinstance(active, Active):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "active_space requires an Active specification",
                             "hint": "Use qcl.Active(electrons=..., orbitals=...)"}])
    molecule = mean_field.molecule
    assert molecule is not None
    nodes = json.loads(mean_field.graph.graph_json())["nodes"]
    scf_node = nodes[mean_field.node_id]
    while scf_node["op"] == "copy":
        scf_node = nodes[scf_node["inputs"][0]]
    if scf_node["op"].startswith("dft."):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "DFT orbitals need an explicit correlated-space policy",
                             "hint": "Use an HF result for the current active_space operation"}])
    if scf_node["op"] == "scf.uhf":
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "UHF orbitals need a spin-resolved active-space policy",
                             "hint": "Use RHF or ROHF for the current active_space operation"}])
    if scf_node["params"].get("reuse_eri") or scf_node["params"].get("solvent"):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Active-space extraction requires a plain SCF result",
                             "hint": "Use scf.hf without PCM or an integral override"}])
    from pyscf import gto

    system = (gto.M(atom=molecule.atoms, basis=molecule.basis, charge=molecule.charge,
                    spin=molecule.spin, unit=molecule.unit, verbose=0)
              if molecule.atoms else None)
    if active.max_orbitals is not None:
        if (type(active.max_orbitals) is not int or active.max_orbitals < 1 or
                active.electrons is not None or active.orbitals is not None or system is None):
            raise CompileError([{"code": "QCL-E105", "source": _source(),
                                 "message": "Maximum-orbital policy is infeasible",
                                 "hint": "Use a positive maximum with a known molecule"}])
        electrons = system.nelectron
        orbitals = min(active.max_orbitals, system.nao_nr())
    else:
        electrons, orbitals = active.electrons, active.orbitals
    if type(electrons) is not int or type(orbitals) is not int:
        raise CompileError([{"code": "QCL-E105", "source": _source(),
                             "message": "Active space needs electron and orbital counts",
                             "hint": "Pass Active(electrons=..., orbitals=...) or max_orbitals"}])
    frozen = system.nelectron - electrons if system is not None else 0
    if (electrons < 1 or orbitals < 1 or frozen < 0 or
            frozen % 2 or system is not None and
            frozen // 2 + orbitals > system.nao_nr() or
            electrons > 2 * orbitals or
            (electrons - molecule.spin) % 2 or
            electrons < abs(molecule.spin)):
        raise CompileError([{"code": "QCL-E105", "source": _source(),
                             "message": "Active space is infeasible for this molecule and basis",
                             "hint": "Check electron count, spin, and available orbitals"}])
    return _Value(mean_field.graph,
                  mean_field.graph.active_space(mean_field.node_id, electrons,
                                                orbitals, _source()),
                  "active_space", molecule)


def qubit_hamiltonian(operator: _Value, *, mapping: str = "jordan_wigner") -> _Value:
    if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
            operator.molecule is None or operator.parent is None):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Qubit mapping requires a molecular FermionOperator",
                             "hint": "Pass active.hamiltonian"}])
    if mapping != "jordan_wigner":
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Qubit mapping is unavailable",
                             "hint": "Use mapping='jordan_wigner'"}])
    node = operator.graph.qubit_hamiltonian(operator.node_id, mapping, _source())
    return _Value(operator.graph, node, "qubit_operator", operator.molecule,
                  parent=operator.parent)


class _FCI:
    def solve(self, operator: _Value, sector: _Value, *,
              engine: str = "pyscf") -> _Value:
        if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
                not isinstance(sector, _Value) or sector.kind != "sector"):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "fci.solve requires a FermionOperator and Sector",
                                 "hint": "Use active.hamiltonian and active.sector"}])
        if operator.graph is not sector.graph or operator.parent != sector.parent:
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "FCI operator and sector refer to different spaces",
                                 "hint": "Use both projections from the same active space"}])
        if engine not in ("pyscf", "native"):
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "Unknown FCI engine",
                                 "hint": "Use engine='pyscf' or engine='native'"}])
        graph = operator.graph
        if engine == "native":
            nodes = json.loads(graph.graph_json())["nodes"]
            active = nodes[operator.parent]["params"]
            sector_node = nodes[sector.node_id]
            while sector_node["op"] == "copy":
                sector_node = nodes[sector_node["inputs"][0]]
            electrons = active["electrons"]
            spin = operator.molecule.spin
            alpha = (electrons + spin) // 2
            beta = electrons - alpha
            if sector_node["op"] == "sector.shift":
                alpha = sector_node["params"]["n_alpha"]
                beta = sector_node["params"]["n_beta"]
            orbitals = active["orbitals"]
            if (alpha < 0 or beta < 0 or alpha > orbitals or beta > orbitals or
                    math.comb(orbitals, alpha) * math.comb(orbitals, beta) >
                    CI_MAX_DETERMINANTS):
                raise CompileError([{"code": "QCL-E209", "source": _source(),
                                     "message": "Native CI sector exceeds its determinant limit",
                                     "hint": "Use a smaller active space or engine='pyscf'"}])
        return _Value(graph, graph.fci(operator.node_id, sector.node_id,
                                       engine, _source()),
                      "fci_result", operator.molecule, parent=operator.parent,
                      family="FCI")


fci = _FCI()


def ci_vector(values: Any, sector: _Value) -> _Value:
    source = _source()
    if not isinstance(sector, _Value) or sector.kind != "sector" or (
            sector.parent is None or sector.molecule is None):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "CI vector needs an active-space Sector",
                             "hint": "Pass active.sector or a shifted active-space sector"}])
    import numpy as np
    from pyscf import fci

    nodes = json.loads(sector.graph.graph_json())["nodes"]
    active = nodes[sector.parent]
    while active["op"] == "copy":
        active = nodes[active["inputs"][0]]
    orbitals = active["params"]["orbitals"]
    sector_node = nodes[sector.node_id]
    while sector_node["op"] == "copy":
        sector_node = nodes[sector_node["inputs"][0]]
    if sector_node["op"] == "sector.shift":
        nelec = (sector_node["params"]["n_alpha"],
                 sector_node["params"]["n_beta"])
    else:
        electrons = active["params"]["electrons"]
        alpha = (electrons + sector.molecule.spin) // 2
        nelec = (alpha, electrons - alpha)
    shape = tuple(fci.cistring.num_strings(orbitals, count) for count in nelec)
    try:
        array = np.asarray(values)
        valid = (array.dtype.kind in "fiu" and array.shape == shape and
                 np.isfinite(array).all())
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": f"CI coefficient array must be finite with shape {shape}",
                             "hint": "Use PySCF alpha/beta determinant-string order"}])
    node = sector.graph.ci_vector(sector.node_id, json.dumps(
        array.astype(np.float64).tolist(), allow_nan=False), source)
    return _Value(sector.graph, node, "tensor", sector.molecule,
                  parent=sector.parent, unit="dimensionless",
                  family="CI coefficients")


def sigma(operator: _Value, coefficients: _Value, *,
          engine: str = "pyscf") -> _Value:
    source = _source()
    if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
            not isinstance(coefficients, _Value) or coefficients.kind != "tensor" or
            coefficients.family != "CI coefficients"):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": "sigma needs a FermionOperator and CI coefficient Tensor",
                             "hint": "Pass active.hamiltonian and state.coefficients"}])
    if operator.graph is not coefficients.graph or operator.parent != coefficients.parent:
        raise CompileError([{"code": "QCL-E103", "source": source,
                             "message": "Sigma operator and CI coefficients use different spaces",
                             "hint": "Use the Hamiltonian that produced the FCI state"}])
    if engine not in ("pyscf", "native"):
        raise CompileError([{"code": "QCL-E104", "source": source,
                             "message": "Unknown sigma implementation",
                             "hint": "Use engine='pyscf' or engine='native'"}])
    if engine == "native":
        nodes = {node["id"]: node for node in json.loads(
            operator.graph.graph_json())["nodes"]}

        def original(identifier: int) -> dict[str, Any]:
            node = nodes[identifier]
            while node["op"] == "copy":
                node = nodes[node["inputs"][0]]
            return node

        coefficient_node = original(coefficients.node_id)
        if coefficient_node["op"] == "fci.coefficients":
            sector_node = original(original(coefficient_node["inputs"][0])["inputs"][1])
        else:
            sector_node = original(coefficient_node["inputs"][0])
        active = nodes[operator.parent]["params"]
        orbitals = active.get("orbitals")
        electrons = active["electrons"]
        alpha = (electrons + operator.molecule.spin) // 2
        beta = electrons - alpha
        if sector_node["op"] == "sector.shift":
            alpha, beta = (sector_node["params"][key]
                           for key in ("n_alpha", "n_beta"))
        if (orbitals is None or alpha < 0 or beta < 0 or
                max(alpha, beta) > orbitals or
                math.comb(orbitals, alpha) * math.comb(orbitals, beta) >
                CI_MAX_DETERMINANTS):
            raise CompileError([{"code": "QCL-E209", "source": source,
                                 "message": "Native sigma sector exceeds its determinant limit",
                                 "hint": "Use a fixed smaller active space or engine='pyscf'"}])
    return _Value(operator.graph, operator.graph.sigma(
        operator.node_id, coefficients.node_id, engine, source),
        "tensor", operator.molecule, parent=operator.parent, unit="hartree")


def _rdm(state: _Value, order: int, source: str) -> _Value:
    if not isinstance(state, _Value) or state.kind not in ("fci_result", "sci_result"):
        raise CompileError([{"code": "QCL-E101", "source": source,
                             "message": f"rdm{order} requires an FCI or SCI result",
                             "hint": "Pass a placed qcl.fci.solve or qcl.sci.solve result"}])
    node = (state.graph.rdm1 if order == 1 else state.graph.rdm2)(state.node_id, source)
    return _Value(state.graph, node,
                  "tensor", state.molecule, unit="dimensionless")


def rdm1(state: _Value) -> _Value:
    return _rdm(state, 1, _source())


def rdm2(state: _Value) -> _Value:
    return _rdm(state, 2, _source())


class _SCI:
    def solve(self, operator: _Value, sector: _Value, *,
              select_cutoff: float = 5e-4, total_spin: int | None = None) -> _Value:
        if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
                not isinstance(sector, _Value) or sector.kind != "sector"):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "sci.solve requires a FermionOperator and Sector",
                                 "hint": "Use active.hamiltonian and active.sector"}])
        if operator.graph is not sector.graph or operator.parent != sector.parent:
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "SCI operator and sector refer to different spaces",
                                 "hint": "Use both projections from the same active space"}])
        if (type(select_cutoff) not in (int, float) or
                not math.isfinite(select_cutoff) or select_cutoff < 0 or
                total_spin is not None and
                (type(total_spin) is not int or total_spin < 0)):
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "SCI cutoff or total spin is invalid",
                                 "hint": "Use a nonnegative finite cutoff and integer spin"}])
        graph = operator.graph
        node = graph.sci(operator.node_id, sector.node_id, float(select_cutoff),
                         total_spin, _source())
        return _Value(graph, node, "sci_result", operator.molecule, family="SCI")


sci = _SCI()


class _Lattice:
    @staticmethod
    def chain(size: int) -> Lattice:
        if type(size) is not int or size < 1:
            raise ValueError("Chain size must be a positive integer")
        return Lattice("chain", (size,))

    @staticmethod
    def ring(size: int) -> Lattice:
        if type(size) is not int or size < 1:
            raise ValueError("Ring size must be a positive integer")
        return Lattice("ring", (size,))

    @staticmethod
    def square(nx: int, ny: int) -> Lattice:
        if any(type(size) is not int or size < 1 for size in (nx, ny)):
            raise ValueError("Square shape must contain positive integers")
        return Lattice("square", (nx, ny))


lattice = _Lattice()


class _Models:
    @staticmethod
    def hubbard(lattice: Lattice, *, t: float, U: float) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call models.hubbard inside a QCL workflow compilation")
        if not isinstance(lattice, Lattice):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Hubbard requires a QCL Lattice",
                                 "hint": "Use qcl.lattice.chain, ring or square"}])
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in (t, U)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Hubbard couplings must be finite numbers",
                                 "hint": "Pass finite hopping and repulsion values"}])
        node = graph.hubbard(lattice.geometry, list(lattice.shape), float(t), float(U),
                             _source())
        return _Value(graph, node, "fermion_operator", lattice=lattice)


models = _Models()


class _Solubility:
    @staticmethod
    def _estimate(smiles: str, model: str, melting_point: float | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call qcl.solubility inside workflow compilation")
        if not isinstance(smiles, str) or not smiles.strip():
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Solubility requires a nonempty SMILES string",
                                 "hint": "Pass a molecular SMILES string"}])
        if model == "gse" and (type(melting_point) not in (int, float) or
                               not math.isfinite(melting_point)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "GSE requires a finite melting point",
                                 "hint": "Pass melting_point_C as a finite number"}])
        node = graph.solubility(model, smiles, melting_point, _source())
        return _Value(graph, node, "estimate", unit="log10(mol/L)",
                      family=model.upper())

    @classmethod
    def esol(cls, smiles: str) -> _Value:
        return cls._estimate(smiles, "esol")

    @classmethod
    def gse(cls, smiles: str, melting_point_C: float) -> _Value:
        return cls._estimate(smiles, "gse", melting_point_C)


solubility = _Solubility()


class _Metabolism:
    @staticmethod
    def sites(molecule: Molecule, *, model: dict[str, Any] | None = None,
              level: str = "hf") -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call qcl.metabolism inside workflow compilation")
        if not isinstance(molecule, Molecule):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Metabolic sites require a static Molecule",
                                 "hint": "Pass a QCL Molecule"}])
        from chemflow.domains.reactivity import validate_model
        from chemflow.types.model import ElectronicModel

        try:
            normalized = ElectronicModel.model_validate(
                model if model is not None else
                {"basis": molecule.basis, "reference": "ROHF" if molecule.spin else "RHF"})
            validate_model(normalized)
        except (TypeError, ValueError) as error:
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": str(error),
                                 "hint": "Use an all-electron nonrelativistic HF model"}]) from error
        if normalized.basis != molecule.basis or level not in ("hf", "mp2", "ccsd_t", "fci"):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "Metabolic model or level is incompatible",
                                 "hint": "Match the molecule basis and use HF, MP2, CCSD(T) or FCI"}])
        mol_id = graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                                molecule.spin, molecule.unit, _source())
        node = graph.metabolic_sites(mol_id, normalized.model_dump_json(), level, _source())
        return _Value(graph, node, "site_scores", molecule, family=level.upper())


metabolism = _Metabolism()


class _Binding:
    @staticmethod
    def mmgbsa(receptor: dict[str, Any], ligand: dict[str, Any], *,
               frames: int = 0, minimize: bool = False, seed: int = 0,
               forcefield: str = "gaff-2.11", ph: float = 7.0) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call qcl.binding inside workflow compilation")
        from chemflow.agent.task import BindingTarget

        try:
            target = BindingTarget.model_validate({
                "kind": "binding_free_energy", "receptor": receptor, "ligand": ligand,
                "frames": frames, "minimize": minimize, "seed": seed,
                "forcefield": forcefield, "ph": ph,
            })
        except ValueError as error:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": str(error),
                                 "hint": "Use a valid receptor, ligand and MM/GBSA protocol"}]) from error
        params = target.model_dump(exclude={"kind", "operator", "protocol"})
        node = graph.binding_mmgbsa(json.dumps(params), _source())
        return _Value(graph, node, "estimate", unit="kcal/mol", family="MM/GBSA")


binding = _Binding()


class _Superconductivity:
    @staticmethod
    def allen_dynes(lambda_: float, omega_log: float, mu_star: float) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call qcl.superconductivity inside workflow compilation")
        from chemflow.domains.lattice_correlations import allen_dynes_tc

        try:
            allen_dynes_tc(lambda_ep=lambda_, omega_log_K=omega_log, mu_star=mu_star)
        except (TypeError, ValueError) as error:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": str(error),
                                 "hint": "Use valid finite Allen-Dynes inputs"}]) from error
        node = graph.allen_dynes(float(lambda_), float(omega_log), float(mu_star), _source())
        return _Value(graph, node, "estimate", unit="K", family="Allen-Dynes")


superconductivity = _Superconductivity()


class _RDKit:
    @staticmethod
    def embed(smiles: str, *, n: int = 20, seed: int = 42,
              basis: str = "sto-3g") -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call RDKit engines inside workflow compilation")
        if (not isinstance(smiles, str) or not smiles.strip() or
                type(n) is not int or not 1 <= n <= 1000 or
                type(seed) is not int or not 0 <= seed < 2**31 or
                not isinstance(basis, str) or
                not basis.strip()):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "RDKit embedding inputs are invalid",
                                 "hint": "Pass SMILES, count 1..1000, seed and basis"}])
        node = graph.rdkit_embed(smiles, n, seed, basis, _source())
        return _Value(graph, node, "collection", Molecule("", basis),
                      max_items=n, item_kind="molecule")

    @staticmethod
    def mmff_optimize(conformer: _Value) -> _Value:
        if not isinstance(conformer, _Value) or conformer.kind != "molecule":
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "MMFF optimization requires a conformer",
                                 "hint": "Use an item from rdkit.embed"}])
        node = conformer.graph.rdkit_mmff_optimize(conformer.node_id, _source())
        return _Value(conformer.graph, node, "molecule", conformer.molecule)

    @staticmethod
    def dedupe(conformers: _Value, *, rmsd: float = 0.5) -> _Value:
        if (not isinstance(conformers, _Value) or conformers.kind != "collection" or
                conformers.item_kind != "molecule" or type(rmsd) not in (int, float) or
                not math.isfinite(rmsd) or rmsd <= 0):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Dedupe requires conformers and positive RMSD",
                                 "hint": "Pass an RDKit conformer collection"}])
        node = conformers.graph.rdkit_dedupe(conformers.node_id, float(rmsd), _source())
        return _Value(conformers.graph, node, "collection", conformers.molecule,
                      max_items=conformers.max_items, item_kind="molecule")


class _Engines:
    rdkit = _RDKit()
    sella: Any
    tblite: Any


engines = _Engines()


class _GFN2:
    @staticmethod
    def _available() -> None:
        from importlib.util import find_spec

        if find_spec("tblite") is None:
            raise CompileError([{"code": "QCL-E401", "source": _source(),
                                 "message": "tblite engine is unavailable",
                                 "hint": "Install the ChemFlow engine dependencies"}])

    @staticmethod
    def energy(molecule: Molecule | _Value, *,
               solvent: ALPB | GBSA | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call GFN2 inside QCL workflow compilation")
        _GFN2._available()
        if solvent is not None and not isinstance(solvent, (ALPB, GBSA)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "GFN2 requires an ALPB or GBSA solvent",
                                 "hint": "Use qcl.ALPB(name) or qcl.GBSA(name)"}])
        node_id, metadata = _molecule_node(graph, molecule)
        result = graph.gfn2_energy(node_id, type(solvent).__name__.lower()
                                   if solvent is not None else None,
                                   solvent.name if solvent is not None else None,
                                   _source())
        return _Value(graph, result, "estimate", metadata, unit="hartree",
                      family="GFN2-xTB")

    @staticmethod
    def optimize(molecule: Molecule | _Value, *,
                 solvent: ALPB | GBSA | None = None) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call GFN2 inside QCL workflow compilation")
        _GFN2._available()
        if solvent is not None and not isinstance(solvent, (ALPB, GBSA)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "GFN2 requires an ALPB or GBSA solvent",
                                 "hint": "Use qcl.ALPB(name) or qcl.GBSA(name)"}])
        node_id, metadata = _molecule_node(graph, molecule)
        return _Value(graph, graph.gfn2_optimize(
            node_id, type(solvent).__name__.lower() if solvent is not None else None,
            solvent.name if solvent is not None else None, _source()),
                      "molecule", metadata)


class _Tblite:
    gfn2 = _GFN2()


engines.tblite = _Tblite()


def _molecule_node(graph: GraphBuilder, molecule: Molecule | _Value) -> tuple[int, Molecule]:
    if isinstance(molecule, _Value) and molecule.kind == "molecule" and molecule.graph is graph:
        assert molecule.molecule is not None
        return molecule.node_id, molecule.molecule
    if isinstance(molecule, Molecule):
        return graph.molecule(molecule.atoms, molecule.basis, molecule.charge,
                              molecule.spin, molecule.unit, _source()), molecule
    raise CompileError([{"code": "QCL-E101", "source": _source(),
                         "message": "Geometry operation requires a QCL Molecule",
                         "hint": "Pass a molecule or placed geometry result"}])


def _validate_analytic_hf_pes(pes: PES, metadata: Molecule,
                              lane: Device | list[Device] | tuple[Device, ...],
                              device: Device, source: str, *,
                              allow_rks: bool = False) -> dict[str, Any]:
    if metadata.spin != 0:
        raise CompileError([{"code": "QCL-E103", "source": source,
                             "message": "Open-shell PES gradient reference matching is unavailable",
                             "hint": "Use a closed-shell HF or RKS PES"}])
    prototype_graph = GraphBuilder()
    token = _active_graph.set(prototype_graph)
    try:
        prototype_id = prototype_graph.molecule(
            metadata.atoms, metadata.basis, metadata.charge, metadata.spin,
            metadata.unit, source)
        prototype = _Value(prototype_graph, prototype_id, "molecule", metadata)
        result = pes.function(prototype, lane)
    finally:
        _active_graph.reset(token)
    nodes = json.loads(prototype_graph.graph_json())["nodes"]
    computes = [item for item in nodes if item["kind"] == "compute"]
    if (not isinstance(result, _Value) or result.graph is not prototype_graph or
            result.kind != "scf_result" or len(computes) != 1 or
            computes[0]["op"] not in (("scf.hf", "scf.rhf", "dft.rks") if allow_rks
                                      else ("scf.hf", "scf.rhf")) or
            computes[0]["id"] != result.node_id or
            computes[0]["inputs"] != [prototype_id] or
            computes[0]["device"] is None or
            computes[0]["device"]["id"] != device.id):
        raise CompileError([{"code": "QCL-E103", "source": source,
                             "message": "PES callback changes the geometry or method",
                             "hint": ("Return placed HF or RKS for the callback geometry"
                                      if allow_rks else
                                      "Return qcl.scf.hf(geometry).to(lane)")}])
    return computes[0]


class _PESOptimize:
    def __init__(self, pes: PES, geometry: Molecule | _Value,
                 fixed: list[Bond | Angle | Dihedral] | tuple[Bond | Angle | Dihedral, ...],
                 source: str):
        self.pes = pes
        self.geometry = geometry
        self.fixed = fixed
        self.source = source
        self.placed = False

    def to(self, lane: Device | list[Device] | tuple[Device, ...]) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Place PES optimization inside workflow compilation")
        targets = [lane] if isinstance(lane, Device) else lane
        if (not isinstance(targets, (list, tuple)) or len(targets) != 1 or
                not isinstance(targets[0], Device)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Analytic HF optimization needs one device lane",
                                 "hint": "Pass one CPU device"}])
        device = targets[0]
        if device.kind != "cpu" or not isinstance(self.pes.gradient,
                                                  (Analytic, FiniteDifference)):
            raise CompileError([{"code": "QCL-E203", "source": self.source,
                                 "message": "PES optimization supports HF on CPU",
                                 "hint": "Use qcl.Analytic() or qcl.FiniteDifference() on CPU"}])
        if isinstance(self.pes.gradient, FiniteDifference) and (
                type(self.pes.gradient.step) not in (int, float) or
                not math.isfinite(self.pes.gradient.step) or
                self.pes.gradient.step <= 0):
            raise CompileError([{"code": "QCL-E101", "source": self.source,
                                 "message": "Finite-difference step must be positive and finite",
                                 "hint": "Use a step in angstroms, such as 1e-3"}])
        mol_id, metadata = _molecule_node(graph, self.geometry)
        _validate_analytic_hf_pes(self.pes, metadata, lane, device, self.source)
        from chemflow.domains.paths import coordinates

        size = len(coordinates({"atoms": metadata.atoms, "unit": metadata.unit})[0])
        if (not isinstance(self.fixed, (list, tuple)) or any(
                not isinstance(item, (Bond, Angle, Dihedral)) or
                len(set(item.atoms)) != len(item.atoms) or
                any(type(index) is not int or index < 0 or index >= size
                    for index in item.atoms) for item in self.fixed)):
            raise CompileError([{"code": "QCL-E105", "source": self.source,
                                 "message": "Invalid fixed internal coordinate",
                                 "hint": "Use distinct in-range atom indices"}])
        fixed = [{"type": type(item).__name__.lower(), "atoms": list(item.atoms)}
                 for item in self.fixed]
        if isinstance(self.pes.gradient, FiniteDifference):
            node = graph.fd_optimize(mol_id, json.dumps(fixed),
                                     float(self.pes.gradient.step), self.source)
        else:
            node = (graph.constrained_optimize(mol_id, json.dumps(fixed), self.source)
                    if fixed else graph.geometry_optimize(mol_id, metadata.spin, self.source))
        placed = graph.place(node, json.dumps(asdict(device)), self.source)
        group = graph.optimize_group(placed, device.id, self.source)
        self.placed = True
        return _Value(graph, group, "molecule", metadata)


def optimize(molecule: Molecule | _Value | PES,
             geometry: Molecule | _Value | None = None, *,
             fixed: list[Bond | Angle | Dihedral] | tuple[Bond | Angle | Dihedral, ...] = (
             )) -> _Value | _PESOptimize:
    if isinstance(molecule, PES):
        if geometry is None:
            raise TypeError("PES optimization requires a starting geometry")
        pattern = _PESOptimize(molecule, geometry, fixed, _source())
        patterns = _active_patterns.get()
        if patterns is None:
            raise RuntimeError("Call qcl.optimize inside workflow compilation")
        patterns.append(pattern)
        return pattern
    if geometry is not None or fixed:
        raise TypeError("A fixed coordinate requires a PES")
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call optimize inside workflow compilation")
    mol_id, metadata = _molecule_node(graph, molecule)
    return _Value(graph, graph.geometry_optimize(mol_id, metadata.spin, _source()),
                  "molecule", metadata)


class _IRC:
    def __init__(self, pes: PES, transition: Molecule | _Value, source: str):
        self.pes = pes
        self.transition = transition
        self.source = source
        self.placed = False

    def to(self, lane: Device | list[Device] | tuple[Device, ...]) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Place IRC inside workflow compilation")
        targets = [lane] if isinstance(lane, Device) else lane
        if (not isinstance(targets, (list, tuple)) or len(targets) != 1 or
                not isinstance(targets[0], Device)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "IRC currently needs one CPU lane",
                                 "hint": "Pass one CPU device"}])
        device = targets[0]
        if device.kind != "cpu" or not isinstance(self.pes.gradient, Analytic):
            raise CompileError([{"code": "QCL-E203", "source": self.source,
                                 "message": "IRC currently supports analytic HF on CPU",
                                 "hint": "Use qcl.Analytic() and a CPU lane"}])
        transition_id, metadata = _molecule_node(graph, self.transition)
        _validate_analytic_hf_pes(self.pes, metadata, lane, device, self.source)
        node = graph.irc(transition_id, self.source)
        placed = graph.place(node, json.dumps(asdict(device)), self.source)
        self.placed = True
        return _Value(graph, placed, "reaction_path", metadata)


def irc(pes: PES, transition: Molecule | _Value) -> _IRC:
    if not isinstance(pes, PES):
        raise TypeError("IRC requires a QCL PES")
    patterns = _active_patterns.get()
    if patterns is None:
        raise RuntimeError("Call qcl.irc inside workflow compilation")
    pattern = _IRC(pes, transition, _source())
    patterns.append(pattern)
    return pattern


def hessian(pes: PES, molecule: Molecule | _Value) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call hessian inside workflow compilation")
    if not isinstance(pes, PES) or not isinstance(pes.gradient, Analytic):
        raise CompileError([{"code": "QCL-E203", "source": _source(),
                             "message": "Hessian currently supports analytic HF/RKS PES",
                             "hint": "Use a closed-shell HF or RKS PES with qcl.Analytic()"}])
    mol_id, metadata = _molecule_node(graph, molecule)
    method = _validate_analytic_hf_pes(pes, metadata, Device("cpu0", "cpu"),
                                       Device("cpu0", "cpu"), _source(), allow_rks=True)
    solvent = method["params"].get("solvent") or {}
    node = graph.hessian(mol_id, method["params"].get("xc"),
                         solvent.get("model", "pcm") if solvent else None,
                         solvent.get("name"), solvent.get("eps"), _source())
    return _Value(graph, node, "hessian", metadata, parent=mol_id,
                  unit="hartree/bohr^2")


def gradient(pes: PES, molecule: Molecule | _Value) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call gradient inside workflow compilation")
    if not isinstance(pes, PES) or not isinstance(pes.gradient,
                                                  (Analytic, FiniteDifference)):
        raise CompileError([{"code": "QCL-E203", "source": _source(),
                             "message": "Gradient currently supports closed-shell HF/RKS PES",
                             "hint": "Use qcl.PES with an HF or RKS callback"}])
    step = pes.gradient.step if isinstance(pes.gradient, FiniteDifference) else None
    if step is not None and (type(step) not in (int, float) or
                             not math.isfinite(step) or step <= 0):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Finite-difference gradient step must be positive",
                             "hint": "Use a finite positive step in bohr"}])
    mol_id, metadata = _molecule_node(graph, molecule)
    method = _validate_analytic_hf_pes(pes, metadata, Device("cpu0", "cpu"),
                                       Device("cpu0", "cpu"), _source(), allow_rks=True)
    solvent = method["params"].get("solvent") or {}
    node = graph.gradient(mol_id, step, method["params"].get("xc"),
                          solvent.get("model", "pcm") if solvent else None,
                          solvent.get("name"), solvent.get("eps"), _source())
    return _Value(graph, node, "tensor", metadata, unit="hartree/bohr")


class _Sella:
    @staticmethod
    def saddle(guess: Molecule | _Value, reactant: Molecule,
               product: Molecule) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call sella.saddle inside workflow compilation")
        guess_id, metadata = _molecule_node(graph, guess)
        if not isinstance(reactant, Molecule) or not isinstance(product, Molecule):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "IRC endpoints must be static Molecules",
                                 "hint": "Pass signed reactant and product geometries"}])
        from chemflow.domains.paths import coordinates

        try:
            symbols = [coordinates({"atoms": item.atoms, "unit": item.unit})[0]
                       for item in (metadata, reactant, product)]
        except ValueError as error:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": str(error),
                                 "hint": "Use valid atom-mapped geometries"}]) from error
        if (symbols[0] != symbols[1] or symbols[0] != symbols[2] or
                any((item.basis, item.charge, item.spin) !=
                    (metadata.basis, metadata.charge, metadata.spin)
                    for item in (reactant, product))):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "IRC endpoints differ from the transition guess",
                                 "hint": "Use the same ordered atoms, basis, charge and spin"}])
        reactant_id, _ = _molecule_node(graph, reactant)
        product_id, _ = _molecule_node(graph, product)
        node = graph.sella_saddle(guess_id, reactant_id, product_id,
                                  metadata.spin, _source())
        return _Value(graph, node, "molecule", metadata)


engines.sella = _Sella()


def frequencies(molecule: Molecule | _Value) -> _Value:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call frequencies inside workflow compilation")
    if isinstance(molecule, _Value) and molecule.kind == "hessian":
        if molecule.graph is not graph:
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "Hessian belongs to another graph",
                                 "hint": "Use one workflow graph"}])
        node = graph.frequencies_from_hessian(molecule.node_id, _source())
        return _Value(graph, node, "frequencies", molecule.molecule,
                      parent=molecule.parent)
    mol_id, metadata = _molecule_node(graph, molecule)
    node = graph.frequencies(mol_id, _source())
    return _Value(graph, node, "frequencies", metadata, parent=mol_id)


class _Thermo:
    @staticmethod
    def rrho(molecule: Molecule | _Value, modes: _Value, *,
             temperature_K: float = 298.15, pressure_Pa: float = 101325.0,
             quantity: str = "gibbs") -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call thermo.rrho inside workflow compilation")
        mol_id, metadata = _molecule_node(graph, molecule)
        if (not isinstance(modes, _Value) or modes.kind != "frequencies" or
                modes.graph is not graph or modes.parent != mol_id):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "RRHO frequencies refer to a different geometry",
                                 "hint": "Use frequencies of the same molecule value"}])
        if (quantity not in ("gibbs", "enthalpy") or
                any(type(value) not in (int, float) or
                    not math.isfinite(value) or value <= 0
                    for value in (temperature_K, pressure_Pa))):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "RRHO conditions are invalid",
                                 "hint": "Use Gibbs or enthalpy with positive finite T and P"}])
        node = graph.rrho(mol_id, modes.node_id, quantity, float(temperature_K),
                          float(pressure_Pa), _source())
        return _Value(graph, node, "estimate", metadata, unit="hartree",
                      family="RRHO correction")

    @staticmethod
    def eyring(barrier: _Value, temperature_K: float) -> _Value:
        if (not isinstance(barrier, _Value) or barrier.kind != "estimate" or
                barrier.unit != "hartree" or type(temperature_K) not in (int, float) or
                not math.isfinite(temperature_K) or temperature_K <= 0):
            raise CompileError([{"code": "QCL-E102", "source": _source(),
                                 "message": "Eyring requires a Hartree barrier and positive T",
                                 "hint": "Use a Gibbs barrier in Hartree"}])
        node = barrier.graph.eyring(barrier.node_id, float(temperature_K), _source())
        return _Value(barrier.graph, node, "estimate", barrier.molecule,
                      unit="1/s", family="Eyring")


thermo = _Thermo()


class _ED:
    @staticmethod
    def ground_state(operator: _Value, *, sector: Sector) -> _Value:
        if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
                operator.lattice is None or not isinstance(sector, Sector)):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Lattice ED requires a Hubbard operator and Sector",
                                 "hint": "Use qcl.models.hubbard and qcl.Sector"}])
        if (type(sector.n) is not int or type(sector.sz) not in (int, float) or
                not math.isfinite(sector.sz)):
            raise CompileError([{"code": "QCL-E105", "source": _source(),
                                 "message": "Electron sector is infeasible for this lattice",
                                 "hint": "Check electron count, spin and site count"}])
        n_alpha = (sector.n + 2 * sector.sz) / 2
        n_beta = (sector.n - 2 * sector.sz) / 2
        if (n_alpha != int(n_alpha) or
                n_beta != int(n_beta) or min(n_alpha, n_beta) < 0 or
                max(n_alpha, n_beta) > operator.lattice.n_sites):
            raise CompileError([{"code": "QCL-E105", "source": _source(),
                                 "message": "Electron sector is infeasible for this lattice",
                                 "hint": "Check electron count, spin and site count"}])
        graph = operator.graph
        sector_node = graph.lattice_sector(operator.node_id, int(n_alpha), int(n_beta),
                                           _source())
        state = graph.ed_ground_state(operator.node_id, sector_node, _source())
        return _Value(graph, state, "ground_state", parent=operator.node_id,
                      family="Hubbard ED", lattice=operator.lattice)


ed = _ED()


class _Ops:
    @staticmethod
    def onsite_pair(i: int, j: int) -> Observable:
        return Observable("pair_s_wave", (i, j))

    @staticmethod
    def spin_spin(i: int, j: int) -> Observable:
        return Observable("spin_spin", (i, j))

    @staticmethod
    def number(i: int) -> Observable:
        return Observable("number", (i, i))

    @staticmethod
    def density_density(i: int, j: int) -> Observable:
        return Observable("density_density", (i, j))


ops = _Ops()


def expect(state: _Value, observable: Observable) -> _Value:
    if (not isinstance(state, _Value) or state.kind != "ground_state" or
            state.lattice is None or not isinstance(observable, Observable)):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "expect requires a lattice ground state and observable",
                             "hint": "Use qcl.ed.ground_state and qcl.ops"}])
    if any(type(site) is not int or site < 0 or site >= state.lattice.n_sites
           for site in observable.sites):
        raise CompileError([{"code": "QCL-E105", "source": _source(),
                             "message": "Observable site is outside the lattice",
                             "hint": "Choose a site within the lattice shape"}])
    return _Value(state.graph,
                  state.graph.expect(state.node_id, observable.kind,
                                     *observable.sites, _source()),
                  "estimate", unit="dimensionless", family=state.family,
                  lattice=state.lattice)


class _Kernels:
    def hf(self, active: _Value) -> _Value:
        if not isinstance(active, _Value) or active.kind != "active_space":
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "HF kernel requires an active space",
                                 "hint": "Pass a placed active_space result"}])
        return _Value(active.graph, active.graph.hf_kernel(active.node_id, _source()),
                      "kernel", active.molecule, active.node_id)

    def lucj(self, active: _Value, *, params: _Value | list[float] | None = None) -> _Value:
        if not isinstance(active, _Value) or active.kind != "active_space":
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "LUCJ kernel requires an active space",
                                 "hint": "Pass a placed active_space result"}])
        graph = active.graph
        nodes = json.loads(graph.graph_json())["nodes"]
        active_node = nodes[active.node_id]
        while active_node["op"] == "copy":
            active_node = nodes[active_node["inputs"][0]]
        n_orb = active_node["params"]["orbitals"]
        size = n_orb * (n_orb - 1) // 2 + n_orb - 1 + (n_orb + 3) // 4
        params_node = None
        vector = None
        if isinstance(params, _Value):
            if (params.graph is not graph or params.kind != "lucj_params" or
                    params.parent != active.node_id):
                raise CompileError([{"code": "QCL-E103", "source": _source(),
                                     "message": "LUCJ parameters belong to another active space",
                                     "hint": "Use CCSD amplitudes from the same active space"}])
            params_node = params.node_id
        elif params is not None:
            if (not isinstance(params, (list, tuple)) or len(params) != size or
                    any(type(value) not in (int, float) or not math.isfinite(value)
                        for value in params)):
                raise CompileError([{"code": "QCL-E104", "source": _source(),
                                     "message": "LUCJ parameter shape is invalid",
                                     "hint": f"Pass {size} finite parameters"}])
            vector = [float(value) for value in params]
        node = graph.lucj_kernel(active.node_id, params_node,
                                 json.dumps(vector) if vector is not None else None,
                                 _source())
        return _Value(graph, node, "kernel", active.molecule, active.node_id)


kernels = _Kernels()


@dataclass(frozen=True)
class _KernelShape:
    kind: str
    size: int


class Qubits:
    def __class_getitem__(cls, size: int) -> _KernelShape:
        return _KernelShape("qubits", size)


class Params:
    def __class_getitem__(cls, size: int) -> _KernelShape:
        return _KernelShape("params", size)


@dataclass(frozen=True)
class _QubitRef:
    index: int


@dataclass(frozen=True)
class _ParamRef:
    index: int


@dataclass(frozen=True)
class _BitRef:
    index: int

    def __bool__(self) -> bool:
        raise CompileError([{"code": "QCL-E210", "source": _source(),
                             "message": "Measured bit cannot control a Python branch",
                             "hint": "Use qcl.cond inside a quantum_kernel"}])


@dataclass(frozen=True)
class QuantumKernel:
    name: str
    n_qubits: int
    n_params: int
    n_clbits: int
    gates: tuple[dict[str, Any], ...]


def quantum_kernel(function: Callable[..., Any]) -> QuantumKernel:
    annotations = get_type_hints(function)
    parameters = list(inspect.signature(function).parameters)
    shapes = [annotations.get(name) for name in parameters]
    if (len(shapes) != 2 or not isinstance(shapes[0], _KernelShape) or
            not isinstance(shapes[1], _KernelShape) or
            shapes[0].kind != "qubits" or shapes[1].kind != "params" or
            any(type(shape.size) is not int or shape.size < 1 for shape in shapes)):
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Quantum kernel requires Qubits[n] and Params[m]",
                             "hint": "Declare positive, static qubit and parameter shapes"}])
    gates: list[dict[str, Any]] = []
    bits: list[_BitRef] = []
    token = _active_kernel_gates.set(gates)
    bit_token = _active_kernel_bits.set(bits)
    try:
        function(tuple(_QubitRef(index) for index in range(shapes[0].size)),
                 tuple(_ParamRef(index) for index in range(shapes[1].size)))
    finally:
        _active_kernel_gates.reset(token)
        _active_kernel_bits.reset(bit_token)
    return QuantumKernel(function.__name__, shapes[0].size, shapes[1].size,
                         len(bits), tuple(gates))


def ry(qubit: _QubitRef, angle: _ParamRef) -> None:
    gates = _active_kernel_gates.get()
    if gates is None or not isinstance(qubit, _QubitRef) or not isinstance(angle, _ParamRef):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "ry requires a kernel qubit and parameter",
                             "hint": "Call ry inside a quantum_kernel"}])
    gates.append({"op": "ry", "qubit": qubit.index, "parameter": angle.index})


def cz(control: _QubitRef, target: _QubitRef) -> None:
    gates = _active_kernel_gates.get()
    if (gates is None or not isinstance(control, _QubitRef) or
            not isinstance(target, _QubitRef) or control == target):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "cz requires distinct kernel qubits",
                             "hint": "Call cz inside a quantum_kernel"}])
    gates.append({"op": "cz", "control": control.index, "target": target.index})


def x(qubit: _QubitRef) -> None:
    gates = _active_kernel_gates.get()
    if gates is None or not isinstance(qubit, _QubitRef):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "x requires a kernel qubit",
                             "hint": "Call x inside a quantum_kernel"}])
    gates.append({"op": "x", "qubit": qubit.index})


def measure(qubit: _QubitRef) -> _BitRef:
    gates, bits = _active_kernel_gates.get(), _active_kernel_bits.get()
    if gates is None or bits is None or not isinstance(qubit, _QubitRef):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "measure requires a kernel qubit",
                             "hint": "Call measure inside a quantum_kernel"}])
    bit = _BitRef(len(bits))
    bits.append(bit)
    gates.append({"op": "measure", "qubit": qubit.index, "bit": bit.index})
    return bit


class _CCSD:
    @staticmethod
    def amplitudes(active: _Value) -> _Value:
        if (not isinstance(active, _Value) or active.kind != "active_space" or
                active.molecule is None or active.molecule.spin != 0):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "CCSD seed requires a closed-shell active space",
                                 "hint": "Use a zero-spin molecular active space"}])
        node = active.graph.ccsd_amplitudes(active.node_id, _source())
        return _Value(active.graph, node, "lucj_params", active.molecule,
                      parent=active.node_id)


ccsd = _CCSD()


def _bind_kernel(kernel: _Value | QuantumKernel,
                 params: list[float] | tuple[float, ...] | None) -> _Value:
    if isinstance(kernel, QuantumKernel):
        if (not isinstance(params, (list, tuple)) or len(params) != kernel.n_params or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in params)):
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "Quantum kernel parameter shape is invalid",
                                 "hint": f"Pass {kernel.n_params} finite parameters"}])
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Call sample inside a QCL workflow compilation")
        node = graph.custom_kernel(kernel.n_qubits, kernel.n_clbits,
                                   json.dumps([float(x) for x in params]),
                                   json.dumps(kernel.gates), _source())
        kernel = _Value(graph, node, "kernel")
    elif params is not None:
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Library kernel parameters are bound during creation",
                             "hint": "Pass params to qcl.kernels.lucj"}])
    return kernel


def sample(kernel: _Value | QuantumKernel, *, shots: int | list[int], seed: int = 0,
           params: list[float] | tuple[float, ...] | None = None,
           partition: CircuitCut | Teleport | None = None) -> _Value:
    if isinstance(partition, CircuitCut):
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "CircuitCut reconstructs estimates, not raw bitstrings",
                             "hint": "Use qcl.estimate for circuit cutting"}])
    if partition is not None and not isinstance(partition, Teleport):
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Invalid sampling partition",
                             "hint": "Use qcl.Teleport() or omit partition"}])
    kernel = _bind_kernel(kernel, params)
    if not isinstance(kernel, _Value) or kernel.kind != "kernel":
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "sample requires a QCL Kernel",
                             "hint": "Use qcl.kernels.hf(active)"}])
    shot_split = tuple(shots) if isinstance(shots, list) else None
    if ((shot_split is None and (type(shots) is not int or shots < 1)) or
            (shot_split is not None and (not shot_split or
                                         any(type(item) is not int or item < 1
                                             for item in shot_split))) or
            type(seed) is not int or seed < 0):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "sample requires positive shots and a nonnegative seed",
                             "hint": "Pass integer shots >= 1 and seed >= 0"}])
    total = sum(shot_split) if shot_split is not None else shots
    return _Value(kernel.graph, kernel.graph.sample(kernel.node_id, total, seed, _source()),
                  "bitstrings", kernel.molecule, kernel.parent, shot_split=shot_split,
                  partition=partition)


def estimate(operator: _Value, kernel: _Value | QuantumKernel, *,
             params: list[float] | tuple[float, ...] | None = None,
             n_groups: int, shots_per_group: int = 1024, seed: int = 0,
             partition: CircuitCut | Teleport | None = None) -> _Value:
    if (partition is not None and not isinstance(partition, (CircuitCut, Teleport)) or
            isinstance(partition, CircuitCut) and
            (type(partition.max_cuts) is not int or partition.max_cuts < 1)):
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Invalid quantum partition",
                             "hint": "Use CircuitCut(max_cuts>=1) or Teleport()"}])
    if not isinstance(operator, _Value) or operator.kind != "qubit_operator":
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Pauli estimate requires a QubitOperator",
                             "hint": "Map a FermionOperator with qcl.qubit_hamiltonian"}])
    kernel = _bind_kernel(kernel, params)
    if not isinstance(kernel, _Value) or kernel.kind != "kernel":
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Pauli estimate requires a quantum kernel",
                             "hint": "Use qcl.kernels.hf or a custom quantum_kernel"}])
    if (operator.graph is not kernel.graph or
            operator.parent is not None and kernel.parent is not None and
            operator.parent != kernel.parent):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Operator and kernel refer to different spaces",
                             "hint": "Use one active space for the operator and kernel"}])
    if (type(n_groups) is not int or n_groups < 1 or
            type(shots_per_group) is not int or not 1 <= shots_per_group <= 50000 or
            type(seed) is not int or seed < 0):
        raise CompileError([{"code": "QCL-E104", "source": _source(),
                             "message": "Invalid Pauli measurement budget",
                             "hint": "Declare positive groups, 1..50000 shots per group and a seed"}])
    node = operator.graph.estimate_quantum(operator.node_id, kernel.node_id,
                                           n_groups, shots_per_group, seed, _source())
    return _Value(operator.graph, node, "estimate", operator.molecule,
                  unit="hartree", family="Pauli expectation", partition=partition)


class _VQE:
    @staticmethod
    def sampled(operator: _Value, sector: _Value, *, ansatz: str = "lucj",
                optimizer: str = "cobyla", max_evaluations: int = 60,
                shots_per_group: int = 1024, n_groups: int,
                seed: int = 0) -> _Value:
        if (not isinstance(operator, _Value) or
                operator.kind not in ("fermion_operator", "qubit_operator") or
                not isinstance(sector, _Value) or sector.kind != "sector"):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Sampled VQE requires a molecular operator and Sector",
                                 "hint": "Use matching active.hamiltonian and active.sector"}])
        if (operator.graph is not sector.graph or operator.parent != sector.parent):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "VQE operator and sector refer to different spaces",
                                 "hint": "Use both projections from one active space"}])
        if (ansatz not in ("lucj", "uccsd") or optimizer not in ("cobyla", "spsa") or
                type(max_evaluations) is not int or max_evaluations < 2 or
                type(shots_per_group) is not int or not 1 <= shots_per_group <= 50000 or
                type(n_groups) is not int or n_groups < 1 or
                type(seed) is not int or seed < 0):
            raise CompileError([{"code": "QCL-E104", "source": _source(),
                                 "message": "Invalid sampled VQE configuration",
                                 "hint": "Declare positive groups, shots, evaluations and a supported ansatz"}])
        node = operator.graph.vqe_sampled(operator.node_id, sector.node_id, ansatz,
                                          optimizer, max_evaluations,
                                          shots_per_group, n_groups, seed, _source())
        return _Value(operator.graph, node, "estimate", operator.molecule,
                      unit="hartree", family="VQE")


vqe = _VQE()


class _SQD:
    def diagonalize(self, operator: _Value, bitstrings: _Value, sector: _Value,
                    *, seed: int = 7) -> _Value:
        if (not isinstance(operator, _Value) or operator.kind != "fermion_operator" or
                not isinstance(bitstrings, _Value) or bitstrings.kind != "bitstrings" or
                not isinstance(sector, _Value) or sector.kind != "sector"):
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "SQD requires a FermionOperator, Bitstrings and Sector",
                                 "hint": "Use active.hamiltonian, sample(kernel) and active.sector"}])
        if (operator.graph is not bitstrings.graph or operator.graph is not sector.graph or
                operator.parent != bitstrings.parent or operator.parent != sector.parent):
            raise CompileError([{"code": "QCL-E103", "source": _source(),
                                 "message": "SQD inputs refer to different active spaces",
                                 "hint": "Construct the kernel and operators from one active space"}])
        if type(seed) is not int or seed < 0:
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "SQD seed must be a nonnegative integer",
                                 "hint": "Pass seed >= 0"}])
        return _Value(operator.graph, operator.graph.sqd(operator.node_id, bitstrings.node_id,
                                                         sector.node_id, seed, _source()),
                      "estimate", operator.molecule, unit="hartree", family="SQD")


sqd = _SQD()


class _Map:
    def __init__(self, function: Callable[[Any, Any], Any], items: Any, source: str):
        self.function = function
        self.items = items
        self.source = source
        self.placed = False

    def to(self, lanes: Device | list[Any] | tuple[Any, ...]) -> list[_Value] | _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Place qcl.map inside workflow compilation")
        targets = list(lanes) if isinstance(lanes, (list, tuple)) else [lanes]
        if (not targets or any(not isinstance(lane, Device) and
                               (not isinstance(lane, tuple) or not lane or
                                any(not isinstance(dev, Device) for dev in lane))
                               for lane in targets)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Map lanes must be QCL devices or device tuples",
                                 "hint": "Pass a device or a nonempty list of lanes"}])
        dynamic = isinstance(self.items, _Value) and self.items.kind == "collection"
        if not dynamic and not isinstance(self.items, (list, tuple, range)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Map requires a QCL or static collection",
                                 "hint": "Pass a collection, list, tuple or range"}])
        if dynamic and (self.items.graph is not graph or
                        self.items.item_kind != "molecule" or
                        self.items.max_items is None):
            raise CompileError([{"code": "QCL-E101", "source": self.source,
                                 "message": "This dynamic collection has no molecule item type",
                                 "hint": "Use a typed conformer collection"}])
        count = self.items.max_items if dynamic else len(self.items)
        if count and len(targets) > 1:
            signatures = []
            for lane in targets:
                prototype_graph = GraphBuilder()
                token = _active_graph.set(prototype_graph)
                try:
                    if dynamic:
                        metadata = self.items.molecule
                        assert metadata is not None
                        molecule_id = prototype_graph.molecule(
                            metadata.atoms, metadata.basis, metadata.charge,
                            metadata.spin, metadata.unit, self.source)
                        item = _Value(prototype_graph, molecule_id, "molecule", metadata)
                    else:
                        item = self.items[0]
                    prototype = self.function(item, lane)
                finally:
                    _active_graph.reset(token)
                if not isinstance(prototype, _Value) or prototype.graph is not prototype_graph:
                    raise CompileError([{"code": "QCL-E101", "source": self.source,
                                         "message": "Map callback must return one QCL value",
                                         "hint": "Return a placed compute result"}])
                signatures.append((prototype.kind, [
                    (node["op"], node["kind"], node["value_type"],
                     node["inputs"], node["params"])
                    for node in json.loads(prototype_graph.graph_json())["nodes"]
                ]))
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise CompileError([{"code": "QCL-E103", "source": self.source,
                                     "message": "Map callback changes the scientific graph by lane",
                                     "hint": "Use lanes only for placement and device options"}])
        if dynamic:
            metadata = self.items.molecule
            assert metadata is not None
            values = [self.function(
                _Value(graph, graph.collection_index(self.items.node_id, index,
                                                     metadata.spin, self.source),
                       "molecule", metadata), targets[index % len(targets)])
                for index in range(count)]
        else:
            values = [self.function(item, targets[index % len(targets)])
                      for index, item in enumerate(self.items)]
        if any(not isinstance(value, _Value) or value.graph is not graph for value in values):
            raise CompileError([{"code": "QCL-E101", "source": self.source,
                                 "message": "Map callback must return one QCL value",
                                 "hint": "Return a placed compute result from each callback"}])
        group = graph.map_group([value.node_id for value in values],
                                [lane.id if isinstance(lane, Device) else
                                 "+".join(device.id for device in lane) for lane in targets],
                                self.items.node_id if dynamic else None, self.source)
        self.placed = True
        if dynamic:
            first = values[0]
            return _Value(graph, group, "collection", first.molecule, unit=first.unit,
                          family=first.family, max_items=count, item_kind=first.kind)
        return values


def map(function: Callable[[Any, Any], Any], items: Any) -> _Map:
    patterns = _active_patterns.get()
    if patterns is None:
        raise RuntimeError("Call qcl.map inside workflow compilation")
    if not callable(function):
        raise TypeError("qcl.map requires a callback")
    pattern = _Map(function, items, _source())
    patterns.append(pattern)
    return pattern


class _Scan:
    def __init__(self, function: Callable[[Any, Any, Any], Any], init: Any,
                 over: Any, source: str):
        self.function = function
        self.init = init
        self.over = over
        self.source = source
        self.placed = False

    def to(self, lane: Device | list[Device] | tuple[Device, ...]) -> list[tuple[_Value, _Value]]:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Place qcl.scan inside workflow compilation")
        devices = [lane] if isinstance(lane, Device) else lane
        if (not isinstance(devices, (list, tuple)) or not devices or
                any(not isinstance(item, Device) for item in devices)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Scan lane must contain QCL devices",
                                 "hint": "Pass a device or nonempty device list"}])
        if not isinstance(self.over, (list, tuple, range)):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Scan currently requires a static finite sequence",
                                 "hint": "Pass a list, tuple or range"}])
        if isinstance(self.init, Molecule):
            mol = self.init
            carry = _Value(graph, graph.molecule(mol.atoms, mol.basis, mol.charge,
                                                mol.spin, mol.unit, self.source),
                           "molecule", mol)
        elif isinstance(self.init, _Value) and self.init.graph is graph:
            carry = self.init
        else:
            raise CompileError([{"code": "QCL-E101", "source": self.source,
                                 "message": "Scan carry must be a QCL value or molecule",
                                 "hint": "Use a typed workflow value as init"}])
        callback_lane = lane if isinstance(lane, Device) else list(lane)
        results: list[tuple[_Value, _Value]] = []
        ids = [carry.node_id]
        for item in self.over:
            pair = self.function(carry, item, callback_lane)
            if (not isinstance(pair, tuple) or len(pair) != 2 or
                    any(not isinstance(value, _Value) or value.graph is not graph
                        for value in pair) or pair[0].kind != carry.kind or
                    results and pair[1].kind != results[0][1].kind):
                raise CompileError([{"code": "QCL-E101", "source": self.source,
                                     "message": "Scan callback must return a typed (carry, out) pair",
                                     "hint": "Keep carry and output types consistent across steps"}])
            carry, output = pair
            results.append((carry, output))
            ids.extend((carry.node_id, output.node_id))
        graph.scan_group(ids, [item.id for item in devices], self.source)
        self.placed = True
        return results


def scan(function: Callable[[Any, Any, Any], Any], *, init: Any, over: Any) -> _Scan:
    patterns = _active_patterns.get()
    if patterns is None:
        raise RuntimeError("Call qcl.scan inside workflow compilation")
    if not callable(function):
        raise TypeError("qcl.scan requires a callback")
    pattern = _Scan(function, init, over, _source())
    patterns.append(pattern)
    return pattern


class _Iterate:
    def __init__(self, step: Callable[[Any, Any], Any], init: Any,
                 until: Callable[[Any, Any], Any], max_steps: int, source: str):
        self.step = step
        self.init = init
        self.until = until
        self.max_steps = max_steps
        self.source = source
        self.placed = False

    def to(self, lane: Device) -> _Value:
        graph = _active_graph.get()
        if graph is None:
            raise RuntimeError("Place qcl.iterate inside workflow compilation")
        if not isinstance(lane, Device):
            raise CompileError([{"code": "QCL-E204", "source": self.source,
                                 "message": "Iteration needs one QCL device lane",
                                 "hint": "Pass a device"}])
        if isinstance(self.init, Molecule):
            mol = self.init
            carry = _Value(graph, graph.molecule(mol.atoms, mol.basis, mol.charge,
                                                mol.spin, mol.unit, self.source),
                           "molecule", mol)
        elif isinstance(self.init, _Value) and self.init.graph is graph:
            carry = self.init
        else:
            raise CompileError([{"code": "QCL-E101", "source": self.source,
                                 "message": "Iteration carry must be a QCL value or molecule",
                                 "hint": "Use a typed workflow value as init"}])
        done = None
        for _ in range(self.max_steps):
            start = graph.node_count()
            candidate = self.step(carry, lane)
            if (not isinstance(candidate, _Value) or candidate.graph is not graph or
                    candidate.kind != carry.kind):
                raise CompileError([{"code": "QCL-E101", "source": self.source,
                                     "message": "Iteration step must preserve the carry type",
                                     "hint": "Return a QCL value of the init type"}])
            predicate = self.until(candidate, lane)
            if type(predicate) is bool:
                predicate = _Value(graph, graph.boolean_constant(predicate, self.source),
                                   "boolean")
            if (not isinstance(predicate, _Value) or predicate.graph is not graph or
                    predicate.kind != "boolean"):
                raise CompileError([{"code": "QCL-E101", "source": self.source,
                                     "message": "Iteration until must return a boolean",
                                     "hint": "Compare a computed result to a stopping value"}])
            if done is not None:
                graph.guard_nodes(start, graph.node_count(), done.node_id, False)
                selected = graph.conditional(done.node_id, carry.node_id,
                                             candidate.node_id, self.source)
                done_id = graph.conditional(done.node_id, done.node_id,
                                            predicate.node_id, self.source)
                carry = _Value(graph, selected, carry.kind, carry.molecule, carry.parent,
                               carry.unit, carry.family, lattice=carry.lattice,
                               max_items=carry.max_items, item_kind=carry.item_kind)
                done = _Value(graph, done_id, "boolean")
            else:
                carry, done = candidate, predicate
        graph.require_condition(done.node_id, self.source)
        result = graph.iterate_group(carry.node_id, done.node_id,
                                     self.max_steps, lane.id, self.source)
        self.placed = True
        return _Value(graph, result, carry.kind, carry.molecule, carry.parent,
                      carry.unit, carry.family, lattice=carry.lattice,
                      max_items=carry.max_items, item_kind=carry.item_kind)


def iterate(step: Callable[[Any, Any], Any], *, init: Any,
            until: Callable[[Any, Any], Any], max_steps: int = 32) -> _Iterate:
    patterns = _active_patterns.get()
    if patterns is None:
        raise RuntimeError("Call qcl.iterate inside workflow compilation")
    if not callable(step) or not callable(until):
        raise TypeError("qcl.iterate requires step and until callbacks")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    pattern = _Iterate(step, init, until, max_steps, _source())
    patterns.append(pattern)
    return pattern


def require(condition: bool | _Value) -> None:
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call qcl.require inside workflow compilation")
    if type(condition) is bool:
        if condition:
            return
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Static QCL requirement is false",
                             "hint": "Correct the input or remove the invalid workflow"}])
    if not isinstance(condition, _Value) or condition.kind != "boolean" or (
            condition.graph is not graph):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "QCL requirement needs a boolean condition",
                             "hint": "Compare a computed projection with an expected value"}])
    graph.require_condition(condition.node_id, _source())


def cond(condition: bool | _Value, when_true: Callable[[], _Value],
         when_false: Callable[[], _Value]) -> _Value:
    if isinstance(condition, _BitRef) and _active_kernel_gates.get() is not None:
        if not callable(when_true) or not callable(when_false):
            raise TypeError("qcl.cond requires two branch callbacks")
        outer = _active_kernel_gates.get()
        true_gates: list[dict[str, Any]] = []
        false_gates: list[dict[str, Any]] = []
        token = _active_kernel_gates.set(true_gates)
        try:
            when_true()
            _active_kernel_gates.set(false_gates)
            when_false()
        finally:
            _active_kernel_gates.reset(token)
        outer.append({"op": "if", "bit": condition.index,
                      "then": true_gates, "else": false_gates})
        return None
    graph = _active_graph.get()
    if graph is None:
        raise RuntimeError("Call qcl.cond inside workflow compilation")
    if not callable(when_true) or not callable(when_false):
        raise TypeError("qcl.cond requires two branch callbacks")
    if type(condition) is bool:
        return (when_true if condition else when_false)()
    if not isinstance(condition, _Value) or condition.kind != "boolean" or (
            condition.graph is not graph):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Conditional requires a computed boolean",
                             "hint": "Compare a computed projection to a static value"}])
    start = graph.node_count()
    true_value = when_true()
    graph.guard_nodes(start, graph.node_count(), condition.node_id, True)
    start = graph.node_count()
    false_value = when_false()
    graph.guard_nodes(start, graph.node_count(), condition.node_id, False)
    if (not isinstance(true_value, _Value) or not isinstance(false_value, _Value) or
            true_value.graph is not graph or false_value.graph is not graph or
            (true_value.kind, true_value.unit, true_value.family) !=
            (false_value.kind, false_value.unit, false_value.family) or
            (true_value.molecule is not None and false_value.molecule is not None and
             (true_value.molecule.basis, true_value.molecule.charge,
              true_value.molecule.spin, true_value.molecule.unit) !=
             (false_value.molecule.basis, false_value.molecule.charge,
              false_value.molecule.spin, false_value.molecule.unit))):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Conditional branches return incompatible values",
                             "hint": "Use the same value type, unit and method in both branches"}])
    node = graph.conditional(condition.node_id, true_value.node_id,
                             false_value.node_id, _source())
    return _Value(graph, node, true_value.kind, true_value.molecule, true_value.parent,
                  true_value.unit, true_value.family, lattice=true_value.lattice,
                  max_items=true_value.max_items, item_kind=true_value.item_kind)


def rank(values: list[_Value] | tuple[_Value, ...] | _Value) -> _Value:
    if isinstance(values, _Value):
        if values.kind != "collection" or values.item_kind != "estimate":
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Ranking requires an Estimate collection",
                                 "hint": "Map an energy calculation over the collection"}])
        return _Value(values.graph, values.graph.rank([values.node_id], _source()),
                      "ranking", family=values.family)
    if not isinstance(values, (list, tuple)) or not values or any(
            not isinstance(value, _Value) or value.kind != "estimate" for value in values):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Ranking requires a nonempty list of Estimates",
                             "hint": "Pass mapped energy estimates"}])
    first = values[0]
    if any(value.graph is not first.graph or value.unit != first.unit or
           value.family != first.family or value.molecule is None or
           first.molecule is None or value.molecule.basis != first.molecule.basis
           for value in values):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Ranking values use incompatible methods or models",
                             "hint": "Use one method, basis and energy unit"}])
    return _Value(first.graph,
                  first.graph.rank([value.node_id for value in values], _source()),
                  "ranking", family=first.family)


def mean(values: list[_Value] | tuple[_Value, ...] | _Value) -> _Value:
    if isinstance(values, _Value):
        if values.kind != "collection" or values.item_kind != "estimate":
            raise CompileError([{"code": "QCL-E101", "source": _source(),
                                 "message": "Mean requires an Estimate collection",
                                 "hint": "Map an estimate-producing callback"}])
        return _Value(values.graph, values.graph.mean([values.node_id], _source()),
                      "estimate", values.molecule, unit=values.unit, family=values.family)
    if not isinstance(values, (list, tuple)) or not values or any(
            not isinstance(value, _Value) or value.kind != "estimate" for value in values):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Mean requires a nonempty list of Estimates",
                             "hint": "Pass compatible QCL estimates"}])
    first = values[0]
    if any(value.graph is not first.graph or value.unit != first.unit for value in values):
        raise CompileError([{"code": "QCL-E102", "source": _source(),
                             "message": "Mean estimate units differ",
                             "hint": "Convert units explicitly with .in_units()"}])
    if any(value.family != first.family or value.molecule is None or
           first.molecule is None or value.molecule.basis != first.molecule.basis
           for value in values):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Mean estimates use incompatible methods or models",
                             "hint": "Use one method family and basis"}])
    return _Value(first.graph,
                  first.graph.mean([value.node_id for value in values], _source()),
                  "estimate", first.molecule, unit=first.unit, family=first.family)


def compare(left: _Value, right: _Value) -> _Value:
    if (not isinstance(left, _Value) or not isinstance(right, _Value) or
            left.kind != "estimate" or right.kind != "estimate" or
            left.graph is not right.graph):
        raise CompileError([{"code": "QCL-E101", "source": _source(),
                             "message": "Comparison requires two QCL Estimates",
                             "hint": "Compare compatible numerical workflow results"}])
    if left.unit != right.unit:
        raise CompileError([{"code": "QCL-E102", "source": _source(),
                             "message": "Comparison units differ",
                             "hint": "Convert units explicitly with .in_units()"}])
    if (left.family != right.family or left.molecule is None or
            right.molecule is None or left.molecule.basis != right.molecule.basis):
        raise CompileError([{"code": "QCL-E103", "source": _source(),
                             "message": "Comparison methods or models differ",
                             "hint": "Use one method family and basis"}])
    return _Value(left.graph, left.graph.compare(left.node_id, right.node_id, _source()),
                  "comparison", unit=left.unit, family=left.family)


def _output_values(value: Any):
    if isinstance(value, _Value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _output_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _output_values(item)


def _output_spec(value: Any) -> Any:
    if isinstance(value, _Value):
        if value.kind not in ("estimate", "bitstrings", "ranking", "comparison", "molecule",
                              "tensor", "hessian", "eigensystem", "svd_result",
                              "spectral_estimate",
                              "qubit_operator",
                              "reaction_path", "frequencies", "site_scores", "lucj_params"):
            raise TypeError("This QCL result cannot yet be returned from a worker")
        return {"type": "node", "id": value.node_id}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_output_spec(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_output_spec(item) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("QCL result dictionary keys must be strings")
        return {"type": "dict", "items": {key: _output_spec(item)
                                          for key, item in value.items()}}
    return {"type": "literal", "value": value}


def _map_gpu_label(value: Any, source_id: str, target_id: str) -> Any:
    if isinstance(value, dict):
        return {key: _map_gpu_label(item, source_id, target_id)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_map_gpu_label(item, source_id, target_id) for item in value]
    return target_id if value == source_id else value


def _isolated_gpu_worker(graph: dict[str, Any]
                         ) -> tuple[dict[str, Any], dict[str, str] | None, str | None]:
    gpu_ids = {device["id"] for node in graph["nodes"]
               if (device := node.get("device")) and device["kind"] == "gpu"}
    gpu_ids.update(device["simulator_host"] for node in graph["nodes"]
                   if (device := node.get("device")) and
                   (device.get("simulator_host") or "").startswith("gpu"))
    if len(gpu_ids) != 1:
        return graph, None, None
    try:
        import cupy

        visible_count = cupy.cuda.runtime.getDeviceCount()
    except (ImportError, AttributeError, RuntimeError):
        return graph, None, None
    gpu_id = next(iter(gpu_ids))
    if re.fullmatch(r"gpu\d+", gpu_id) is None:
        raise RuntimeError(f"QCL-E503: {gpu_id} is not a local CUDA ordinal")
    ordinal = int(gpu_id[3:])
    if ordinal >= visible_count:
        raise RuntimeError(f"QCL-E503: {gpu_id} is not a visible CUDA device")
    if visible_count == 1:
        return graph, None, None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = ([item.strip() for item in visible.split(",")]
              if visible is not None else [str(index) for index in range(visible_count)])
    if ordinal >= len(tokens) or not tokens[ordinal]:
        raise RuntimeError(f"QCL-E503: {gpu_id} is not a visible CUDA device")
    worker_graph = _map_gpu_label(graph, gpu_id, "gpu0")
    return worker_graph, {**os.environ, "CUDA_VISIBLE_DEVICES": tokens[ordinal]}, gpu_id


def _restore_worker_gpu(value: Any, gpu_id: str) -> Any:
    return _map_gpu_label(value, "gpu0", gpu_id)


class Plan:
    def __init__(self, graph: GraphBuilder, output: _Value | Any,
                 budget: Budget | None = None, *, deterministic: bool = False):
        self._graph = graph
        self._output = output
        self._budget = budget
        self._deterministic = deterministic
        self._profile: list[dict[str, Any]] | None = None
        self._last_result: Any = None
        self._validations: list[dict[str, Any]] = []
        self.cpu_core_s: float | None = None

    def _budget_payload(self) -> dict[str, Any] | None:
        if self._budget is None:
            return None
        payload = asdict(self._budget)
        if payload["cpu_core_s"] is None:
            del payload["cpu_core_s"]
        return payload

    def graph_json(self) -> dict[str, Any]:
        return json.loads(self._graph.graph_json())

    @property
    def output_type(self) -> str:
        if isinstance(self._output, _Value):
            return self._output.kind
        return ("composite" if next(_output_values(self._output), None) is not None
                else "static")

    def _decode_output(self, result: Any, output: Any, timestamp: str) -> Any:
        if isinstance(output, _Value):
            result["provenance"]["timestamp"] = timestamp
            if output.kind == "bitstrings":
                return Bitstrings(result["counts"], result["shots"], result["source"],
                                  result["device"], tuple(result["job_ids"]),
                                  result["provenance"], tuple(result.get("parts", ())))
            if output.kind == "ranking":
                for item in result["values"]:
                    item["provenance"]["timestamp"] = timestamp
                return Ranking(tuple(result["order"]),
                               tuple(tuple(pair) for pair in result["unresolved"]),
                               tuple(Estimate(item["value"], item["unit"], item["variance"],
                                              item["provenance"]) for item in result["values"]),
                               result["provenance"])
            if output.kind == "comparison":
                return Comparison(result["relation"], result["difference"],
                                  result["combined_sigma"], result["unit"],
                                  result["provenance"])
            if output.kind == "site_scores":
                return SiteScores(tuple(result["sites"]), result["top_site"],
                                  result["evidence"], result["provenance"])
            if output.kind == "spectral_estimate":
                return SpectralEstimate(tuple(result["omega"]), tuple(result["spectral"]),
                                        result["unit"], result["relation"],
                                        result["diagnostics"], result["provenance"])
            if output.kind == "molecule":
                return GeometryResult(
                    Molecule(result["atoms"], result["basis"], result["charge"],
                             result["spin"], result["unit"]), result["provenance"])
            if output.kind in ("tensor", "hessian", "eigensystem", "svd_result"):
                import base64

                import numpy as np

                def decode_tensor(payload: dict[str, Any]) -> Tensor:
                    payload["provenance"]["timestamp"] = timestamp
                    data = np.frombuffer(base64.b64decode(payload["data_b64"]),
                                         dtype=np.dtype(payload["dtype"])).copy()
                    data = data.reshape(payload["shape"])
                    if payload["device"].startswith("gpu"):
                        import cupy as cp

                        with cp.cuda.Device(int(payload["device"][3:])):
                            data = cp.asarray(data)
                        payload["provenance"]["result_transport"] = "gpu-host-gpu"
                    return Tensor(data, payload["device"],
                                  payload["unit"], payload["layout"],
                                  payload["provenance"])

                if output.kind in ("tensor", "hessian"):
                    return decode_tensor(result)
                if output.kind == "svd_result":
                    return SvdResult(*(decode_tensor(result[key]) for key in ("u", "s", "vh")),
                                     result["provenance"])
                return Eigensystem(decode_tensor(result["values"]),
                                   decode_tensor(result["vectors"]),
                                   result["provenance"])
            if output.kind == "qubit_operator":
                return QubitOperator(result["terms"], result["n_qubits"],
                                     result["encoding_id"], result["provenance"])
            if output.kind == "reaction_path":
                def molecule(data: dict[str, Any]) -> Molecule:
                    return Molecule(data["atoms"], data["basis"], data["charge"],
                                    data["spin"], data["unit"])

                return ReactionPath(molecule(result["ts"]),
                                    tuple(molecule(item) for item in result["endpoints"]),
                                    result["provenance"])
            if output.kind == "frequencies":
                return FrequencyResult(tuple(result["freq_wavenumber"]),
                                       result["n_imaginary"], result["max_gradient"],
                                       result["provenance"])
            if output.kind == "lucj_params":
                return LucjParameters(tuple(result["vector"]), result["provenance"])
            return Estimate(result["value"], result["unit"], result["variance"],
                            result["provenance"])
        if isinstance(output, tuple):
            return tuple(self._decode_output(item, original, timestamp)
                         for item, original in zip(result, output, strict=True))
        if isinstance(output, list):
            return [self._decode_output(item, original, timestamp)
                    for item, original in zip(result, output, strict=True)]
        if isinstance(output, dict):
            return {key: self._decode_output(result[key], original, timestamp)
                    for key, original in output.items()}
        return result

    def explain(self, format: str = "text") -> str | dict[str, Any]:
        nodes = json.loads(self._graph.graph_json())["nodes"]
        by_id = {node["id"]: node for node in nodes}

        def sample_implementation(node: dict[str, Any]) -> str:
            kernel = by_id[node["inputs"][0]]
            while kernel["op"] == "copy":
                kernel = by_id[kernel["inputs"][0]]
            device = node["device"]
            if kernel["op"] == "kernel.custom" and device["simulator_host"]:
                if kernel["params"]["n_clbits"]:
                    return "sample/qiskit-aer-dynamic"
                if device["simulator_host"].startswith("gpu"):
                    return "sample/qcl-cupy-statevector"
                return "sample/qiskit-statevector"
            if device["simulator_host"]:
                return ("sample/chemflow-cupy-statevector"
                        if device["simulator_host"].startswith("gpu") else
                        "sample/chemflow-statevector")
            return "sample/chemflow-lqcloud-mq02"

        def gradient_implementation(node: dict[str, Any]) -> str:
            method = ("fd-" if node["params"]["step"] is not None else "")
            method += "dft-" if node["params"].get("xc") is not None else ""
            solvent = node["params"].get("solvent")
            method += solvent.get("model", "pcm") + "-" if solvent else ""
            device = "gpu4pyscf" if node["device"]["kind"] == "gpu" else "pyscf"
            return f"gradient/{method}{device}"

        def rdm_implementation(node: dict[str, Any]) -> str:
            if node["device"]["kind"] == "gpu":
                return f"{node['op']}/chemflow-cupy"
            state = by_id[node["inputs"][0]]
            while state["op"] == "copy":
                state = by_id[state["inputs"][0]]
            return (f"{node['op']}/qcl-rust"
                    if state["params"].get("engine") == "native" else
                    f"{node['op']}/chemflow-pyscf")

        def resident(identifier: int) -> str | None:
            node = by_id[identifier]
            if node["device"] is not None:
                return node["device"]["id"]
            if node["kind"] == "projection":
                return resident(node["inputs"][0])
            return None

        operations = []
        transfers = []
        for node in nodes:
            if node["kind"] == "compute":
                device = node["device"]
                if device is None:
                    continue
                operations.append({"id": node["id"], "op": node["op"],
                                   "device": device["id"],
                                   **({"guards": node["guards"]} if node.get("guards") else {}),
                                   "implementation": (sample_implementation(node)
                                                      if node["op"] == "sample" else
                                                      "fci.solve/qcl-rust"
                                                      if node["op"] == "fci.solve" and
                                                      node["params"].get("engine") == "native" else
                                                      "fci.solve/chemflow-cupy"
                                                      if node["op"] == "fci.solve" and
                                                      device["kind"] == "gpu" else
                                                      rdm_implementation(node)
                                                      if node["op"] in ("rdm1", "rdm2") else
                                                      ("sigma/chemflow-cupy" if device["kind"] == "gpu"
                                                       else "sigma/qcl-rust" if node["params"].get(
                                                           "engine") == "native" else
                                                       "sigma/chemflow-numpy")
                                                      if node["op"] == "sigma" else
                                                      "vqe.sampled/chemflow-sampled-vqe"
                                                      if node["op"] == "vqe.sampled" else
                                                      ("estimate.quantum/qcl-cupy-statevector"
                                                       if (device["simulator_host"] or "").startswith("gpu") else
                                                       "estimate.quantum/qiskit-statevector"
                                                       if device["simulator_host"] else
                                                       "estimate.quantum/chemflow-lqcloud-mq02")
                                                      if node["op"] == "estimate.quantum" else
                                                      "qubit_hamiltonian/chemflow-jw"
                                                      if node["op"] == "qubit_hamiltonian" else
                                                      "ed.ground_state/chemflow-cupy"
                                                      if node["op"] == "ed.ground_state" and
                                                      device["kind"] == "gpu" else
                                                      "ed.ground_state/chemflow-sparse"
                                                      if node["op"] == "ed.ground_state" else
                                                      "expect/chemflow-cupy"
                                                      if node["op"] == "expect" and
                                                      device["kind"] == "gpu" else
                                                      "expect/chemflow-lattice-correlations"
                                                      if node["op"] == "expect" else
                                                      f"{node['op']}/chemflow-empirical"
                                                      if node["op"].startswith("solubility.") else
                                                      "superconductivity.allen_dynes/chemflow"
                                                      if node["op"] == "superconductivity.allen_dynes" else
                                                      "int2e/chemflow-pyscf"
                                                      if node["op"] == "int2e" else
                                                      "int1e/chemflow-pyscf"
                                                      if node["op"] == "int1e" else
                                                      "int3c/chemflow-pyscf"
                                                      if node["op"] == "int3c" else
                                                      "cholesky/pyscf-pivoted"
                                                      if node["op"] == "cholesky" else
                                                      ("eigh/cupy-cusolver" if device["kind"] == "gpu"
                                                       else "eigh/scipy")
                                                      if node["op"] == "eigh" else
                                                      (f"{node['op']}/cupy" if device["kind"] == "gpu"
                                                       else f"{node['op']}/scipy")
                                                      if node["op"] in ("orthogonalize", "solve") else
                                                      ("svd/cupy" if device["kind"] == "gpu"
                                                       else "svd/scipy")
                                                      if node["op"] == "svd" else
                                                      ("transform/opt-einsum-cupy"
                                                       if device["kind"] == "gpu" else
                                                       "transform/opt-einsum")
                                                      if node["op"] == "transform" else
                                                      ("contract/opt-einsum-cupy"
                                                       if device["kind"] == "gpu" else
                                                       "contract/opt-einsum")
                                                      if node["op"] == "contract" else
                                                      "periodic.scf/chemflow-pyscf-pbc"
                                                      if node["op"] == "periodic.scf" else
                                                      "dynamics.spectral_exact/chemflow-lehmann"
                                                      if node["op"] == "dynamics.spectral_exact" else
                                                      "metabolism.sites/chemflow-reactivity"
                                                      if node["op"] == "metabolism.sites" else
                                                      "binding.mmgbsa/chemflow-openmm"
                                                      if node["op"] == "binding.mmgbsa" else
                                                      "geometry.optimize/chemflow-geometric"
                                                      if node["op"] == "geometry.optimize" else
                                                      "geometry.constrained_optimize/chemflow-geometric"
                                                      if node["op"] == "geometry.constrained_optimize" else
                                                      "geometry.fd_optimize/ase-pyscf-finite-difference"
                                                      if node["op"] == "geometry.fd_optimize" else
                                                      "irc/chemflow-sella-irc"
                                                      if node["op"] == "irc" else
                                                      "sella.saddle/chemflow-sella-irc"
                                                      if node["op"] == "sella.saddle" else
                                                      "frequencies/chemflow-pyscf-hessian"
                                                      if node["op"] == "frequencies" else
                                                      ("hessian/" +
                                                       ("dft-" if node["params"].get("xc") else "") +
                                                       (node["params"]["solvent"].get("model", "pcm") + "-"
                                                        if node["params"].get("solvent") else "") +
                                                       ("gpu4pyscf" if device["kind"] == "gpu"
                                                        else "pyscf"))
                                                      if node["op"] == "hessian" else
                                                      gradient_implementation(node)
                                                      if node["op"] == "gradient" else
                                                      "frequencies.from_hessian/pyscf"
                                                      if node["op"] == "frequencies.from_hessian" else
                                                      "ccsd.amplitudes/chemflow-pyscf"
                                                      if node["op"] == "ccsd.amplitudes" else
                                                      "sci.solve/chemflow-pyscf"
                                                      if node["op"] == "sci.solve" else
                                                      "gfn2.energy/chemflow-tblite"
                                                      if node["op"] == "gfn2.energy" else
                                                      "gfn2.optimize/chemflow-tblite"
                                                      if node["op"] == "gfn2.optimize" else
                                                      ("mp2/gpu4pyscf" if device["kind"] == "gpu"
                                                       else "mp2/chemflow-pyscf")
                                                      if node["op"] == "mp2" else
                                                      f"{node['op']}/chemflow-thermo"
                                                      if node["op"].startswith("thermo.") else
                                                      (f"{node['op']}/gpu4pyscf-"
                                                       f"{node['params']['solvent'].get('model', 'pcm')}"
                                                       if node["params"].get("solvent") else
                                                       f"{node['op']}/gpu4pyscf")
                                                      if node["op"] in ("scf.hf", "scf.rhf",
                                                                             "scf.rohf", "scf.uhf",
                                                                             "dft.rks", "dft.uks") and
                                                      device["kind"] == "gpu" else
                                                      ("scf.rhf/chemflow-pyscf-" +
                                                       node["params"]["solvent"].get("model", "pcm"))
                                                      if node["op"] == "scf.rhf" and
                                                      node["params"].get("solvent") else
                                                      "scf.rhf/chemflow-pyscf-eri"
                                                      if node["op"] == "scf.rhf" and
                                                      node["params"].get("reuse_eri") else
                                                      (f"{node['op']}/chemflow-pyscf-" +
                                                       node["params"]["solvent"].get("model", "pcm"))
                                                      if node["op"] in ("dft.rks", "dft.uks") and
                                                      node["params"].get("solvent") else
                                                      f"{node['op']}/chemflow-rdkit"
                                                      if node["op"].startswith("rdkit.") else
                                                      f"{node['op']}/chemflow-pyscf"),
                                   "source": node["source"]})
                for input_id in node["inputs"]:
                    # These inputs construct a classical QPU job, not addressable QPU memory.
                    if device["kind"] == "qpu" and node["op"] in (
                            "sample", "vqe.sampled", "estimate.quantum"):
                        continue
                    source_device = resident(input_id)
                    if source_device is not None and source_device != device["id"]:
                        transfers.append({"kind": "implicit", "from": input_id,
                                          "source_device": source_device,
                                          "to": device["id"], "bytes": None,
                                          "source": node["source"]})
            elif node["kind"] == "copy":
                transfers.append({"kind": "explicit", "from": node["inputs"][0],
                                  "source_device": resident(node["inputs"][0]),
                                  "to": node["device"]["id"], "bytes": None,
                                  "source": node["source"]})
        for output in _output_values(self._output):
            if output.kind not in ("tensor", "hessian", "eigensystem", "svd_result"):
                continue
            output_device = resident(output.node_id)
            if output_device is None or not output_device.startswith("gpu"):
                continue
            transfers.extend((
                {"kind": "worker_transport", "from": output.node_id,
                 "source_device": output_device, "to": "host", "bytes": None,
                 "source": "result serialization"},
                {"kind": "worker_restore", "from": output.node_id,
                 "source_device": "host", "to": output_device, "bytes": None,
                 "source": "result decoding"},
            ))
        patterns = [{"id": node["id"], "op": node["op"],
                     "lanes": node["params"]["lanes"],
                     "items": (len(node["inputs"]) - node["params"]["dynamic"]
                               if node["op"] == "map" else
                               (len(node["inputs"]) - 1) // 2 if node["op"] == "scan" else
                               node["params"]["steps"] if node["op"] == "iterate" else 1),
                     "dynamic": (node["params"]["dynamic"] if node["op"] == "map" else
                                 node["op"] == "iterate"),
                     "source": node["source"]}
                    for node in nodes if node["op"] in ("map", "scan", "optimize", "iterate")]
        conditions = [{"id": node["id"], "condition": node["inputs"][0],
                       "true_value": node["inputs"][1],
                       "false_value": node["inputs"][2], "source": node["source"]}
                      for node in nodes if node["op"] == "cond"]
        explanation = {"operations": operations, "transfers": transfers,
                       "patterns": patterns, "conditions": conditions}
        if self._deterministic:
            explanation["deterministic"] = True
        if format == "json":
            return explanation
        if format == "text":
            return "\n".join(f"{item['op']} -> {item['device']} "
                             f"({item['implementation']})" for item in operations)
        raise ValueError("format must be 'text' or 'json'")

    def estimate(self) -> dict[str, Any]:
        nodes = json.loads(self._graph.graph_json())["nodes"]
        explanation = self.explain(format="json")
        devices = {item["device"] for item in explanation["operations"]}
        transfers = explanation["transfers"]
        unknown = ["wall_time_s", *(f"peak_memory_bytes.{device}"
                                     for device in sorted(devices))]
        if transfers:
            unknown.append("bytes_moved")
        jobs = [node for node in nodes if node["op"] in
                ("sample", "vqe.sampled", "estimate.quantum")]

        def jobs_and_shots(node: dict[str, Any]) -> tuple[int, int]:
            if node["op"] == "sample":
                return 1, node["params"]["shots"]
            groups = len(node["params"].get(
                "group_indices", range(node["params"]["n_groups"])))
            evaluations = (node["params"]["max_evaluations"]
                           if node["op"] == "vqe.sampled" else 1)
            return groups * evaluations, node["params"]["shots_per_group"]

        shots = sum(count * each for node in jobs
                    for count, each in [jobs_and_shots(node)])
        qpu_cost: float | None = 0.0
        currency = None
        tariff_source = None
        qpu_seconds: float | None = 0.0
        for node in jobs:
            device = node["device"]
            if device["simulator_host"] is not None:
                continue
            if device["provider"] == "lqcloud" and device["backend"] == "MQ02":
                from chemflow.backends.lqcloud import LqcloudBackend, load_pricing

                count, per_job_shots = jobs_and_shots(node)
                prediction = LqcloudBackend(backend="MQ02").prediction(
                    per_job_shots)
                if qpu_seconds is not None:
                    qpu_seconds += count * prediction["qpu_s"]
                if device["billing"] == "metered":
                    pricing = load_pricing("MQ02")
                    currency = pricing.currency
                    tariff_source = pricing.source
                    if qpu_cost is not None:
                        qpu_cost += count * (pricing.per_task +
                                             pricing.per_shot * per_job_shots)
            else:
                qpu_seconds = None
                if device["billing"] == "metered":
                    qpu_cost = None
                    unknown.append("qpu_cost")
        if qpu_seconds is None:
            unknown.append("qpu_s")
        return {
            "wall_time_s": None,
            "peak_memory_bytes": {device: None for device in sorted(devices)},
            "bytes_moved": None if transfers else 0,
            "shots": shots,
            "qpu_s": qpu_seconds,
            "qpu_cost": qpu_cost,
            "currency": currency,
            "tariff_source": tariff_source,
            "program_budget": self._budget_payload(),
            "unknown": unknown,
        }

    @property
    def hash(self) -> str:
        graph = json.loads(self._graph.graph_json())
        for node in graph["nodes"]:
            node.pop("source", None)
            node.pop("source_span", None)
        payload = ({"graph": graph, "budget": self._budget_payload()}
                   if self._budget else graph)
        if self._deterministic:
            payload = {"graph": graph, "budget": self._budget_payload(),
                       "deterministic": True}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def _operator_wall_timeout(self) -> float | None:
        path = os.environ.get("QCL_POLICY_FILE")
        if not path:
            return None
        try:
            policy = tomllib.loads(Path(path).read_text())
            limits = policy.get("budget", {})
            duration = limits.get("wall_time")
        except (OSError, tomllib.TOMLDecodeError, AttributeError) as error:
            raise CompileError([{"code": "QCL-E301", "source": path,
                                 "message": "Operator wall-time policy cannot be read",
                                 "hint": "Check QCL_POLICY_FILE and [budget] syntax"}]) from error
        if duration is None:
            return None
        match = (re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd])", duration)
                 if isinstance(duration, str) else None)
        if match is None:
            raise CompileError([{"code": "QCL-E301", "source": path,
                                 "message": "Operator wall_time is invalid",
                                 "hint": "Use a positive duration such as '4h'"}])
        seconds = float(match.group(1)) * {"s": 1, "m": 60, "h": 3600,
                                            "d": 86400}[match.group(2)]
        if not math.isfinite(seconds) or seconds <= 0:
            raise CompileError([{"code": "QCL-E301", "source": path,
                                 "message": "Operator wall_time is invalid",
                                 "hint": "Use a finite positive duration"}])
        if self._budget and self._budget.wall_s is not None and (
                self._budget.wall_s > seconds):
            raise CompileError([{"code": "QCL-E301", "source": path,
                                 "message": "Program cannot raise the operator wall-time limit",
                                 "hint": "Lower wall_s in qcl.Budget"}])
        return seconds

    def _check_qpu_policy(self) -> dict[str, Any] | None:
        nodes = json.loads(self._graph.graph_json())["nodes"]
        jobs = [node for node in nodes if node["op"] in
                ("sample", "vqe.sampled", "estimate.quantum")]
        if not jobs:
            return None
        policy_path = os.environ.get("QCL_POLICY_FILE")
        if not policy_path:
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU resource policy is missing",
                                 "hint": "Set QCL_POLICY_FILE to an operator policy"}])
        try:
            policy = tomllib.loads(Path(policy_path).read_text())
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU resource policy cannot be read",
                                 "hint": "Check QCL_POLICY_FILE and TOML syntax"}]) from error
        limits = policy.get("budget", {})
        if not isinstance(limits, dict):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU resource policy needs a [budget] table",
                                 "hint": "Set qpu_shots in [budget]"}])
        forecast = self.estimate()
        limit = limits.get("qpu_shots")
        if (self._budget and self._budget.qpu_shots is not None and
                (type(limit) is not int or self._budget.qpu_shots > limit)):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Program cannot raise the operator shot limit",
                                 "hint": "Lower qpu_shots in qcl.Budget"}])
        if self._budget and self._budget.qpu_shots is not None:
            limit = min(limit, self._budget.qpu_shots)
        if type(limit) is not int or limit < forecast["shots"]:
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU shot budget is absent or exceeded",
                                 "hint": "Set qpu_shots >= the planned shot count"}])
        allowed = limits.get("allowed_qpus")
        if allowed is not None and (not isinstance(allowed, list) or
                                    not all(isinstance(item, str) for item in allowed) or
                                    any(job["device"]["id"] not in allowed for job in jobs)):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU is not allowed by the operator policy",
                                 "hint": "Use an allowed_qpus device"}])
        seconds_limit = limits.get("qpu_s")
        if seconds_limit is not None and (type(seconds_limit) not in (int, float) or
                                         not math.isfinite(seconds_limit) or
                                         seconds_limit < 0):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Operator QPU time limit is invalid",
                                 "hint": "Set a nonnegative numeric qpu_s"}])
        if self._budget and self._budget.qpu_s is not None:
            if (seconds_limit is not None and
                    self._budget.qpu_s > seconds_limit):
                raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                     "message": "Program cannot raise the operator QPU time limit",
                                     "hint": "Lower qpu_s in qcl.Budget"}])
            seconds_limit = self._budget.qpu_s
        if seconds_limit is not None and (type(seconds_limit) not in (int, float) or
                                         not math.isfinite(seconds_limit) or
                                         seconds_limit < 0 or forecast["qpu_s"] is None or
                                         forecast["qpu_s"] > seconds_limit):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "QPU time is unknown or exceeds its budget",
                                 "hint": "Provide a calibrated forecast or increase qpu_s"}])
        metered = any(job["device"]["billing"] == "metered" for job in jobs)
        if self._budget and self._budget.qpu_cost is not None and not metered:
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Monetary budgets apply only to metered cloud QPUs",
                                 "hint": "Remove qpu_cost for local or non-billable QPUs"}])
        cloud = policy.get("cloud_qpu", {}) if metered else {}
        if not isinstance(cloud, dict):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Cloud QPU policy needs a [cloud_qpu] table",
                                 "hint": "Use a TOML table for cloud_qpu"}])
        operator_cost_limit = cloud.get("qpu_cost")
        cost_limit = operator_cost_limit
        if cost_limit is not None and (type(cost_limit) not in (int, float) or
                                       not math.isfinite(cost_limit) or cost_limit < 0):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Operator cloud cost limit is invalid",
                                 "hint": "Set a nonnegative numeric qpu_cost"}])
        if self._budget and self._budget.qpu_cost is not None:
            if cost_limit is not None and self._budget.qpu_cost > cost_limit:
                raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                     "message": "Program cannot raise the operator cloud cost limit",
                                     "hint": "Lower qpu_cost in qcl.Budget"}])
            cost_limit = self._budget.qpu_cost
        if cost_limit is not None and (type(cost_limit) not in (int, float) or
                                       not math.isfinite(cost_limit) or cost_limit < 0 or
                                       forecast["qpu_cost"] is None or
                                       forecast["qpu_cost"] > cost_limit or
                                       operator_cost_limit is not None and
                                       cloud.get("currency") != forecast["currency"]):
            raise CompileError([{"code": "QCL-E301", "source": jobs[0]["source"],
                                 "message": "Cloud QPU cost cannot fit the operator limit",
                                 "hint": "Check tariff, currency and qpu_cost"}])
        threshold = cloud.get("approval_above")
        if threshold is not None and (type(threshold) not in (int, float) or
                                      not math.isfinite(threshold) or threshold < 0 or
                                      forecast["qpu_cost"] is None or
                                      forecast["qpu_cost"] > threshold):
            approval = cloud.get("approval", {})
            if not isinstance(approval, dict):
                approval = {}
            accounts = {job["device"]["billing_account"] for job in jobs
                        if job["device"]["billing"] == "metered"}
            if (approval.get("plan_hash") != self.hash or
                    len(accounts) != 1 or None in accounts or
                    approval.get("billing_account") not in accounts or
                    approval.get("currency") != forecast["currency"] or
                    type(approval.get("max_cost")) not in (int, float) or
                    not math.isfinite(approval["max_cost"]) or
                    approval["max_cost"] < forecast["qpu_cost"]):
                raise CompileError([{"code": "QCL-E302", "source": jobs[0]["source"],
                                     "message": "Cloud QPU job needs plan-bound approval",
                                     "hint": "Approve this plan hash and maximum cost"}])
        if cost_limit is not None:
            return {"max_cost": cost_limit, "forecast_cost": forecast["qpu_cost"],
                    "currency": forecast["currency"]}
        return None

    def profile(self) -> list[dict[str, Any]]:
        if self._profile is None:
            raise RuntimeError("Run the plan before requesting a profile")
        return list(self._profile)

    def error_budget(self) -> dict[str, Any]:
        """Report measured uncertainty and reference deviations without claiming a bound."""
        if self.output_type != "estimate":
            raise TypeError("Error budgets require a plan returning an Estimate")
        result = self._last_result
        observations = [
            {"on": item["on"],
             "reference_plan_hash": item["reference"]["plan_hash"],
             "absolute_error": item["absolute_error"],
             "combined_sigma": item["combined_sigma"],
             "unit": item["candidate"]["unit"]}
            for item in self._validations
            if item["candidate"]["plan_hash"] == self.hash
        ]
        return {
            "plan_hash": self.hash,
            "unit": result.unit if result is not None else None,
            "statistical": {
                "variance": result.variance if result is not None else None,
                "std_dev": math.sqrt(result.variance) if result is not None else None,
            },
            "systematic": {
                "upper_bound": None,
                "max_observed_absolute_error": (max(item["absolute_error"]
                                                    for item in observations)
                                                if observations else None),
                "validation_observations": observations,
            },
            "total": {"upper_bound": None},
        }

    def validate(self, *, reference: Plan, on: str,
                 tolerance: float | None = None) -> dict[str, Any]:
        if not isinstance(reference, Plan) or self.output_type != "estimate" or (
                reference.output_type != "estimate"):
            raise TypeError("Validation requires two plans returning Estimates")
        if not isinstance(on, str) or not on:
            raise ValueError("on must name the validation instance")
        if tolerance is not None and (type(tolerance) not in (int, float) or
                                      not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative")
        candidate_molecules = [node["params"] for node in self.graph_json()["nodes"]
                               if node["op"] == "molecule"]
        reference_molecules = [node["params"] for node in reference.graph_json()["nodes"]
                               if node["op"] == "molecule"]
        if candidate_molecules != reference_molecules:
            raise CompileError([{"code": "QCL-E103", "source": on,
                                 "message": "Validation plans use different molecules",
                                 "hint": "Compile both plans for the same validation instance"}])
        candidate = self.run()
        baseline = reference.run()
        if candidate.provenance["question"] != baseline.provenance["question"]:
            raise CompileError([{"code": "QCL-E103", "source": on,
                                 "message": "Validation plans answer different questions",
                                 "hint": "Compile both plans for the same molecule and property"}])
        baseline = baseline.in_units(candidate.unit)
        absolute_error = abs(candidate.value - baseline.value)
        combined_sigma = math.sqrt(candidate.variance + baseline.variance)
        validation = {
            "on": on,
            "candidate": {"plan_hash": self.hash, "value": candidate.value,
                          "unit": candidate.unit, "variance": candidate.variance},
            "reference": {"plan_hash": reference.hash, "value": baseline.value,
                          "unit": baseline.unit, "variance": baseline.variance},
            "absolute_error": absolute_error,
            "combined_sigma": combined_sigma,
            "tolerance": tolerance,
            "within_tolerance": absolute_error <= tolerance if tolerance is not None else None,
        }
        self._validations.append(validation)
        return validation

    def run(self, *, timeout_s: float = 300,
            cpu_core_s: float | None = None) -> Estimate | Any:
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or
                timeout_s <= 0):
            raise ValueError("timeout_s must be positive")
        if cpu_core_s is not None and (type(cpu_core_s) not in (int, float) or
                                       not math.isfinite(cpu_core_s) or cpu_core_s <= 0):
            raise ValueError("cpu_core_s must be positive")
        self.cpu_core_s = None
        self._last_result = None
        self._profile = None
        if next(_output_values(self._output), None) is None:
            return self._output
        if self._budget and self._budget.wall_s is not None:
            timeout_s = min(timeout_s, self._budget.wall_s)
        if self._budget and self._budget.cpu_core_s is not None:
            cpu_core_s = (min(cpu_core_s, self._budget.cpu_core_s)
                          if cpu_core_s is not None else self._budget.cpu_core_s)
        operator_timeout = self._operator_wall_timeout()
        if operator_timeout is not None:
            timeout_s = min(timeout_s, operator_timeout)
        output = (self._output.node_id if isinstance(self._output, _Value)
                  else _output_spec(self._output))
        cloud_qpu_policy = self._check_qpu_policy()
        worker_graph, worker_env, gpu_id = _isolated_gpu_worker(
            json.loads(self._graph.graph_json()))
        if self._deterministic:
            worker_env = {**(worker_env or os.environ), **dict.fromkeys((
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")}
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            process = subprocess.run(
                [sys.executable, "-m", "qcl._worker"],
                input=json.dumps({"graph": worker_graph,
                                  "output": output,
                                  "cloud_qpu_policy": cloud_qpu_policy,
                                  "cpu_core_s": cpu_core_s,
                                  "deterministic": self._deterministic}),
                text=True, capture_output=True, check=False, timeout=timeout_s,
                env=worker_env,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("QCL-E503: worker exceeded the wall-time limit") from error
        finally:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.cpu_core_s = max(0.0, after.ru_utime + after.ru_stime -
                                  before.ru_utime - before.ru_stime)
        if process.returncode:
            if cpu_core_s is not None and (
                    "CPU core-time limit" in process.stderr or
                    process.returncode == -signal.SIGXCPU or
                    process.returncode == -signal.SIGKILL and
                    self.cpu_core_s >= cpu_core_s):
                raise RuntimeError("QCL-E503: worker exceeded CPU core-time limit")
            raise RuntimeError(f"QCL-E503: worker failed: {process.stderr.strip()}")
        if cpu_core_s is not None and self.cpu_core_s > cpu_core_s:
            raise RuntimeError("QCL-E503: worker exceeded CPU core-time limit")
        response = json.loads(process.stdout)
        if gpu_id is not None:
            response = _restore_worker_gpu(response, gpu_id)
        self._profile = response["profile"]
        result = self._decode_output(response["result"], self._output,
                                     datetime.now(UTC).isoformat())
        self._last_result = result
        return result


class Workflow:
    def __init__(self, function: Callable[..., Any]):
        self.function = function
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _active_graph.get() is None:
            raise RuntimeError("Call workflow.compile(...) outside another workflow")
        return self.function(*args, **kwargs)

    def compile(self, *args: Any, strict_transfers: bool = False,
                deterministic: bool = False, budget: Budget | None = None,
                **kwargs: Any) -> Plan:
        if type(deterministic) is not bool:
            raise TypeError("deterministic must be a boolean")
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError("budget must be a qcl.Budget")
        graph = GraphBuilder()
        token = _active_graph.set(graph)
        patterns: list[_Map | _Scan | _PESOptimize | _Iterate] = []
        pattern_token = _active_patterns.set(patterns)
        try:
            result = self.function(*args, **kwargs)
        finally:
            _active_graph.reset(token)
            _active_patterns.reset(pattern_token)
        diagnostics = json.loads(graph.diagnostics_json())
        if budget is not None and budget.qpu_cost is not None and not any(
                node["kind"] == "compute" and node["device"] is not None and
                node["device"]["billing"] == "metered"
                for node in json.loads(graph.graph_json())["nodes"]):
            diagnostics.append({"code": "QCL-E301", "source": _source(),
                                "message": "Monetary budgets require a metered cloud QPU",
                                "hint": "Remove qpu_cost for local or non-billable devices"})
        if deterministic:
            diagnostics.extend({
                "code": "QCL-E207", "source": node["source"],
                "message": "Cloud QPU measurements are not seed-replayable",
                "hint": "Use a seeded QPU simulator or omit deterministic=True",
            } for node in json.loads(graph.graph_json())["nodes"]
                if node["op"] in ("sample", "vqe.sampled", "estimate.quantum") and
                node["device"] is not None and
                node["device"]["simulator_host"] is None)
            diagnostics.extend({
                "code": "QCL-E207", "source": node["source"],
                "message": "GPU numerical implementation is not bitwise deterministic",
                "hint": "Use a CPU implementation or omit deterministic=True",
            } for node in json.loads(graph.graph_json())["nodes"]
                if node["kind"] == "compute" and node["device"] is not None and
                (node["device"]["kind"] == "gpu" or
                 (node["device"].get("simulator_host") or "").startswith("gpu")))
        diagnostics.extend({"code": "QCL-E201", "source": pattern.source,
                            "message": f"{type(pattern).__name__[1:].lower()} is never placed",
                            "hint": "Add .to(<lanes>) to the pattern"}
                           for pattern in patterns if not pattern.placed)
        plan = Plan(graph, result, budget, deterministic=deterministic)
        if strict_transfers:
            for transfer in plan.explain(format="json")["transfers"]:
                if transfer["kind"] == "implicit":
                    diagnostics.append({
                        "code": "QCL-E205", "source": transfer["source"],
                        "message": f"Implicit transfer to {transfer['to']} is forbidden",
                        "hint": "Add an explicit .to(<device>) copy to the input value",
                    })
        if diagnostics:
            raise CompileError(diagnostics)
        if any(value.graph is not graph for value in _output_values(result)):
            raise ValueError("Workflow returned a value from another graph")
        return plan


def workflow(function: Callable[..., Any]) -> Workflow:
    return Workflow(function)
