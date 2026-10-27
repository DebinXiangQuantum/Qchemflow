from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from chemflow.ids import canonical_json, content_hash


class MolecularTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["molecular_ground_energy"] = "molecular_ground_energy"
    operator: str = "ground_energy"
    molecule: dict
    electronic_model: dict
    active: dict | str = "auto"
    sector: dict
    target_spin: float | None = None

    @model_validator(mode="after")
    def auto_sector(self) -> MolecularTarget:
        if self.active == "auto" and any(key in self.sector for key in
                                         ("n_orb", "n_alpha", "n_beta", "n_elec")):
            raise ValueError("Auto active space cannot fix sector particle/orbital counts")
        return self


class ReactionSpecies(BaseModel, frozen=True, extra="forbid"):
    name: str
    molecule: dict
    coefficient: float = Field(allow_inf_nan=False)
    active: dict
    sector: dict
    target_spin: float | None = None
    role: Literal["minimum", "ts"] = "minimum"


class ReactionThermo(BaseModel, frozen=True, extra="forbid"):
    quantity: Literal["enthalpy", "gibbs"]
    temperature_K: float = Field(default=298.15, gt=0, allow_inf_nan=False)
    pressure_Pa: float = Field(default=101325.0, gt=0, allow_inf_nan=False)


class ReactionTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["reaction_energy"]
    operator: Literal["reaction_energy"] = "reaction_energy"
    electronic_model: dict
    species: tuple[ReactionSpecies, ...]
    thermo: ReactionThermo | None = None

    @model_validator(mode="after")
    def balanced(self) -> ReactionTarget:
        from chemflow.domains.reactions import balanced

        if not self.species or len({item.name for item in self.species}) != len(self.species):
            raise ValueError("Reaction requires uniquely named species")
        if not balanced([(item.coefficient, item.molecule) for item in self.species]):
            raise ValueError("CF-E012: Unbalanced reaction")
        return self


class ReactionBarrierTarget(ReactionTarget):
    kind: Literal["reaction_barrier"]
    product: dict
    rate_constant: bool = False

    @model_validator(mode="after")
    def connected_species(self):
        from chemflow.domains.paths import coordinates

        if self.rate_constant and (self.thermo is None or self.thermo.quantity != "gibbs"):
            raise ValueError("Eyring rates require a Gibbs barrier and declared temperature")
        transition = [species for species in self.species if species.role == "ts"]
        reactants = [species for species in self.species if species.role == "minimum"]
        if (len(transition) != 1 or len(reactants) != 1 or transition[0].coefficient != 1
                or reactants[0].coefficient != -1):
            raise ValueError("Barrier requires one +1 TS and one -1 reactant supermolecule")
        molecules = [transition[0].molecule, reactants[0].molecule, self.product]
        if any(coordinates(molecule)[0] != coordinates(molecules[0])[0] or
               any(molecule.get(key, 0) != molecules[0].get(key, 0) for key in ("charge", "spin"))
               for molecule in molecules[1:]):
            raise ValueError("Barrier endpoints and TS require the same ordered atoms, charge and spin")
        return self


class LatticeTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["lattice_ground_energy"]
    operator: Literal["ground_energy"] = "ground_energy"
    lattice: dict
    sector: dict
    target_spin: float | None = None


class PairingTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["pairing_correlation"]
    operator: Literal["correlation_function"] = "correlation_function"
    lattice: dict
    sector: dict
    target_spin: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    site_pair: tuple[int, int]
    quantity: Literal["pair_s_wave", "spin_spin", "density_density"] = "pair_s_wave"

    @model_validator(mode="before")
    @classmethod
    def strict_sites(cls, values):
        from chemflow.domains.lattice_correlations import validate_selection

        if isinstance(values, dict):
            if values.get("site_pair") is None:
                raise ValueError("A pairing target requires site_pair")
            validate_selection(values["site_pair"], values.get("quantity", "pair_s_wave"))
        return values

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.materials import lattice_arrays
        from chemflow.types import Sector

        size = len(lattice_arrays(**self.lattice)[0])
        sector = Sector.model_validate(self.sector)
        if sector.n_orb != size or max(self.site_pair) >= size:
            raise ValueError("Pairing target sector/sites differ from the lattice")
        return self


class PeriodicTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["periodic_energy"]
    operator: Literal["periodic_scf"] = "periodic_scf"
    system: dict
    kmesh: tuple[int, int, int] = (1, 1, 1)
    method: Literal["rhf"] = "rhf"
    exxdiv: str | None = "ewald"


class SpectralTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["spectral_function"]
    operator: Literal["spectral_function"] = "spectral_function"
    model: dict
    sector: dict
    orbital: int
    spin: Literal["alpha", "beta"]
    omega: dict
    eta: float

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.algorithms import as_sector
        from chemflow.domains.dynamics import grid, impurity_arrays

        if set(self.model) != {"impurity_model"}:
            raise ValueError("Spectral target requires an impurity_model")
        h1, _ = impurity_arrays(**self.model["impurity_model"])
        sector = as_sector(self.sector)
        if sector.n_orb != len(h1) or not 0 <= self.orbital < sector.n_orb:
            raise ValueError("Spectral sector/orbital differs from model")
        grid(self.omega, self.eta)
        return self


class InteractionTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["interaction_energy"]
    operator: Literal["interaction_energy"] = "interaction_energy"
    complex: dict
    fragments: list[list[int]]
    electronic_model: dict
    counterpoise: bool = True

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.drug import component_specs

        component_specs(**self.model_dump(exclude={"kind", "operator"}))
        return self


class AqueousSolubilityTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["aqueous_solubility"]
    operator: Literal["empirical_estimate"] = "empirical_estimate"
    molecule_smiles: str
    model: Literal["esol", "gse"] = "esol"
    melting_point_C: float | None = Field(default=None, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def valid(self):
        from chemflow.empirical import molecule

        molecule(self.molecule_smiles)
        if self.model == "gse" and self.melting_point_C is None:
            raise ValueError("GSE requires melting_point_C")
        return self


class PropertyLevel(BaseModel, frozen=True, extra="forbid"):
    method: Literal["hf", "dft", "gfn2"]
    functional: str | None = None
    dispersion: Literal["d3bj", "d4"] | None = None

    @model_validator(mode="after")
    def compatible(self):
        if self.method == "dft" and not self.functional:
            raise ValueError("DFT requires a functional")
        if self.method != "dft" and (self.functional or self.dispersion):
            raise ValueError("Functional and dispersion require DFT")
        return self


class SolventModel(BaseModel, frozen=True, extra="forbid"):
    model: Literal["pcm", "alpb", "gbsa", "smd"]
    name: str | None = None
    eps: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def specified(self):
        if self.model == "pcm" and self.eps is None:
            raise ValueError("PCM requires eps")
        if self.model != "pcm" and not self.name:
            raise ValueError("Named solvent required")
        return self


class SolvationTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["solvation_free_energy"]
    operator: Literal["solvation_free_energy"] = "solvation_free_energy"
    molecule: dict
    electronic_model: dict
    level: PropertyLevel
    solvent: SolventModel

    @model_validator(mode="after")
    def compatible(self):
        if (self.level.method == "gfn2") != (self.solvent.model in ("alpb", "gbsa")):
            raise ValueError("GFN2 requires ALPB/GBSA; HF/DFT require PCM/SMD")
        return self


class FrontierTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["frontier_orbitals"]
    operator: Literal["frontier_orbitals"] = "frontier_orbitals"
    molecule: dict
    electronic_model: dict
    level: PropertyLevel
    quantity: Literal["homo", "lumo", "gap"]
    spin: Literal["alpha", "beta"] = "alpha"

    @model_validator(mode="after")
    def scf_level(self):
        if self.level.method == "gfn2":
            raise ValueError("Frontier orbitals require HF or DFT")
        return self


class IonizationTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["ionization_energy", "electron_affinity"]
    operator: Literal["charge_energy_difference"] = "charge_energy_difference"
    molecule: dict
    electronic_model: dict
    final_spin: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def half_integer(self):
        if not (2 * self.final_spin).is_integer():
            raise ValueError("final_spin must be integer or half-integer")
        return self


class BondDissociationTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["bond_dissociation"]
    operator: Literal["bond_dissociation"] = "bond_dissociation"
    molecule: dict
    electronic_model: dict
    bonds: str | list[list[int]] = "all"
    fragments: Literal["rigid"] = "rigid"

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.reactivity import components, validate_model

        validate_model(self.electronic_model)
        components(self.model_dump())
        return self


class RankingItem(BaseModel, frozen=True, extra="forbid"):
    name: str = Field(min_length=1)
    molecule: dict


class RankingTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["ranking"]
    operator: Literal["ranking"] = "ranking"
    items: list[RankingItem] = Field(min_length=1)
    electronic_model: dict
    quantity: Literal["energy"] = "energy"

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.reactivity import components, validate_model

        validate_model(self.electronic_model)
        components(self.model_dump())
        return self


class ReactivityLevel(BaseModel, frozen=True, extra="forbid"):
    method: Literal["hf", "mp2", "ccsd_t", "fci"] = "hf"
    functional: None = None
    dispersion: None = None


class MetabolicSitesTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["metabolic_sites"]
    operator: Literal["metabolic_sites"] = "metabolic_sites"
    molecule: dict
    electronic_model: dict
    level: ReactivityLevel = ReactivityLevel()

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.reactivity import perceived, validate_model

        validate_model(self.electronic_model)
        perceived(self.molecule)
        return self


class PathLevel(BaseModel, frozen=True, extra="forbid"):
    method: Literal["hf", "dft", "gfn2"]
    functional: str | None = None
    dispersion: str | None = None

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.paths import validate_level

        validate_level(self.model_dump())
        return self


class ScanCoordinate(BaseModel, frozen=True, extra="forbid"):
    type: Literal["bond", "angle", "dihedral"]
    atoms: list[Annotated[int, Field(strict=True, ge=0)]]
    values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.paths import coordinate_check

        for value in self.values:
            coordinate_check(self.type, self.atoms, value)
        return self


class PESScanTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["pes_scan"]
    operator: Literal["pes_scan"] = "pes_scan"
    molecule: dict
    electronic_model: dict
    coordinate: ScanCoordinate
    mode: Literal["rigid", "relaxed"] = "rigid"
    relax_level: PathLevel | None = None

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.drug import component_specs
        from chemflow.domains.paths import coordinates

        symbols, _ = coordinates(self.molecule)
        component_specs(self.molecule, [list(range(len(symbols)))], self.electronic_model, False)
        if max(self.coordinate.atoms) >= len(symbols):
            raise ValueError("Scan atom index outside molecule")
        if (self.mode == "relaxed") != (self.relax_level is not None):
            raise ValueError("Only relaxed scans require a relax_level")
        return self


class InteractionScanTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["interaction_scan"]
    operator: Literal["interaction_scan"] = "interaction_scan"
    complex: dict
    electronic_model: dict
    fragments: list[list[Annotated[int, Field(strict=True, ge=0)]]]
    counterpoise: bool = True
    separations: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self):
        from chemflow.domains.drug import component_specs

        if len(self.fragments) != 2:
            raise ValueError("Dimer scans require exactly two fragments")
        component_specs(self.complex, self.fragments, self.electronic_model, self.counterpoise)
        return self


class BindingTarget(BaseModel, frozen=True, extra="forbid"):
    kind: Literal["binding_free_energy"]
    operator: Literal["mmgbsa"] = "mmgbsa"
    receptor: dict
    ligand: dict
    protocol: Literal["mmgbsa"] = "mmgbsa"
    frames: int = Field(default=0, ge=0, strict=True)
    minimize: bool = False
    forcefield: Literal["gaff-2.11", "openff-2.2.0"] = "gaff-2.11"
    ph: float = Field(default=7.0, ge=0, le=14, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def valid(self):
        if (set(self.receptor) - {"pdb_text", "chains"} or
                not isinstance(self.receptor.get("pdb_text"), str) or
                set(self.ligand) != {"source"} or not isinstance(self.ligand["source"], dict)):
            raise ValueError("Binding target requires receptor pdb_text and ligand source")
        return self


TaskTarget = Annotated[MolecularTarget | ReactionTarget | LatticeTarget | PeriodicTarget |
                       SpectralTarget | InteractionTarget | AqueousSolubilityTarget |
                       PairingTarget | SolvationTarget | FrontierTarget | IonizationTarget |
                       BondDissociationTarget | RankingTarget | MetabolicSitesTarget |
                       PESScanTarget | InteractionScanTarget | ReactionBarrierTarget | BindingTarget,
                       Field(discriminator="kind")]


class TaskGate(BaseModel, frozen=True, extra="forbid"):
    kind: str
    threshold: float | None = None
    unit: str | None = None
    relation: str | None = None

    @model_validator(mode="after")
    def valid(self) -> TaskGate:
        if self.kind in ("abs_error_vs_exact", "l1_error_vs_exact") and (
                self.threshold is None or self.unit is None):
            raise ValueError("Absolute-error gate requires threshold and unit")
        if self.kind == "relation_at_least" and self.relation is None:
            raise ValueError("Relation gate requires a relation")
        if self.kind not in ("abs_error_vs_exact", "l1_error_vs_exact", "relation_at_least"):
            raise ValueError("Unknown gate kind")
        if self.kind == "l1_error_vs_exact":
            import math

            if not math.isfinite(self.threshold) or self.threshold < 0:
                raise ValueError("L1 threshold must be finite and nonnegative")
            if self.unit != "dimensionless":
                raise ValueError("L1 integral gate requires dimensionless units")
        return self


class ValidityAxis(BaseModel, frozen=True, extra="forbid"):
    axis: Literal["geometry_scale", "hubbard_u", "impurity_u", "internal_coordinate", "separation",
                  "binding_ph"]
    range: tuple[Annotated[float, Field(strict=True)], Annotated[float, Field(strict=True)]]
    points: int = Field(ge=3, le=33, strict=True)
    resolution: float = Field(gt=0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        import math

        lo, hi = self.range
        if not all(math.isfinite(value) for value in self.range) or not lo < hi:
            raise ValueError("Validity range requires finite lo < hi")
        if self.resolution >= hi - lo:
            raise ValueError("Validity resolution must be smaller than the range")
        return self


def task_axis_point(target, axis):
    if axis == "binding_ph" and target.kind == "binding_free_energy":
        return target.ph
    if axis == "internal_coordinate" and target.kind == "pes_scan":
        return float(target.coordinate.values[0])
    if axis == "separation" and target.kind == "interaction_scan":
        return float(target.separations[0])
    if axis == "geometry_scale" and target.kind in ("molecular_ground_energy", "reaction_energy",
            "solvation_free_energy", "frontier_orbitals", "ionization_energy", "electron_affinity",
            "bond_dissociation", "ranking"):
        return 1.0
    if axis == "hubbard_u" and target.kind in ("lattice_ground_energy", "pairing_correlation"):
        return float(target.lattice.get("U", 4.0))
    if axis == "impurity_u" and target.kind == "spectral_function":
        return float(target.model["impurity_model"]["U"])
    raise ValueError("Validity axis is unsupported for the target kind")


class TaskSpec(BaseModel, frozen=True, extra="forbid"):
    question: str
    target: TaskTarget
    gates: tuple[TaskGate, ...]
    budget: dict[str, float] = Field(default_factory=dict)
    allowed_effects: tuple[str, ...] = ("pure",)
    decision: str
    validity: tuple[ValidityAxis, ...] = ()

    @model_validator(mode="after")
    def validity_domain(self):
        if len({item.axis for item in self.validity}) != len(self.validity):
            raise ValueError("Duplicate validity axis")
        for item in self.validity:
            point = task_axis_point(self.target, item.axis)
            if item.axis == "binding_ph" and not 0 <= item.range[0] < item.range[1] <= 14:
                raise ValueError("Binding pH validity must lie within [0, 14]")
            if not item.range[0] <= point <= item.range[1]:
                raise ValueError("Validity range must contain the task point")
        return self

    @model_validator(mode="before")
    @classmethod
    def legacy_target(cls, value):
        if isinstance(value, dict) and ("budget_currency" in value or
                set(value.get("budget", {})) & {"money", "qpu_jobs"}):
            raise ValueError("D-R2: money, qpu_jobs and budget_currency are not task resources")
        if isinstance(value, dict) and isinstance(value.get("target"), dict):
            value = {**value, "target": {"kind": "molecular_ground_energy", **value["target"]}}
        return value


class TaskRecord(BaseModel, frozen=True):
    spec: TaskSpec
    content_hash: str
    approver: str
    timestamp: datetime
    version: int
    predecessor_hash: str | None
    key_fingerprint: str | None = None
    signature: str

    @property
    def hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class TaskStore:
    def __init__(self, secret: bytes | str):
        self._secret = secret.encode() if isinstance(secret, str) else secret
        if not self._secret:
            raise ValueError("TaskStore secret must not be empty")

    def freeze(self, spec: TaskSpec, approver: str,
               predecessor: TaskRecord | None = None) -> TaskRecord:
        if not approver:
            raise ValueError("Approver identity is required")
        if predecessor is not None:
            self.verify(predecessor)
        payload = {"spec": spec.model_dump(mode="json"), "content_hash": content_hash(spec),
                   "approver": approver, "timestamp": datetime.now(UTC).isoformat(),
                   "version": predecessor.version + 1 if predecessor else 1,
                   "predecessor_hash": predecessor.hash if predecessor else None,
                   "key_fingerprint": None}
        unsigned = TaskRecord.model_validate({**payload, "signature": ""})
        canonical = canonical_json(unsigned.model_dump(mode="json", exclude={"signature"}))
        signature = hmac.new(self._secret, canonical.encode(), hashlib.sha256)
        return unsigned.model_copy(update={"signature": signature.hexdigest()})

    def verify(self, record: TaskRecord) -> None:
        from chemflow.diagnostics import AdapterError

        try:
            payload = record.model_dump(mode="json", exclude={"signature"})
            digest = hmac.new(self._secret, canonical_json(payload).encode(), hashlib.sha256)
            valid = (record.content_hash == content_hash(record.spec) and
                     hmac.compare_digest(record.signature, digest.hexdigest()))
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise AdapterError("CF-E120", None, "Task record integrity failure",
                               "Obtain a new approved task record")
