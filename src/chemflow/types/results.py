from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from chemflow.context import Context
from chemflow.relations import TargetRelation
from chemflow.units import Quantity


class QuantityKind(StrEnum):
    aqueous_solubility = "aqueous_solubility"
    metabolic_sites = "metabolic_sites"
    solvation_free_energy = "solvation_free_energy"
    frontier_orbitals = "frontier_orbitals"
    ionization_energy = "ionization_energy"
    electron_affinity = "electron_affinity"
    potential_energy = "potential_energy"
    qmmm_total_energy = "qmmm_total_energy"
    total_electronic_energy = "total_electronic_energy"
    active_space_energy = "active_space_energy"
    correlation_energy = "correlation_energy"
    reaction_energy = "reaction_energy"
    reaction_enthalpy = "reaction_enthalpy"
    reaction_gibbs_energy = "reaction_gibbs_energy"
    lattice_energy = "lattice_energy"
    pairing_correlation = "pairing_correlation"
    energy_per_cell = "energy_per_cell"
    interaction_energy = "interaction_energy"
    binding_free_energy = "binding_free_energy"
    docking_score = "docking_score"
    energy_difference = "energy_difference"


class EvidenceStatus(StrEnum):
    certified_bound = "certified_bound"
    empirical_calibration = "empirical_calibration"
    diagnostic_only = "diagnostic_only"
    unknown = "unknown"


_CAPABILITY = object()
_runtime_capability: ContextVar[object | None] = ContextVar("evidence_runtime", default=None)


class Evidence(BaseModel, frozen=True):
    status: EvidenceStatus
    artifact_ref: str | None = None
    source: str = "implementation"

    @classmethod
    def _runtime(cls, status: EvidenceStatus, artifact_ref: str | None = None,
                 capability: object | None = None) -> "Evidence":
        if (status in (EvidenceStatus.certified_bound, EvidenceStatus.empirical_calibration)
            and capability is not _CAPABILITY):
            raise ValueError("CF-E040: privileged evidence requires executor capability")
        token = _runtime_capability.set(_CAPABILITY)
        try:
            return cls(status=status, artifact_ref=artifact_ref, source="runtime")
        finally:
            _runtime_capability.reset(token)

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> "Evidence":
        if (values.get("status") in (EvidenceStatus.certified_bound,
                                     EvidenceStatus.empirical_calibration) and
            _runtime_capability.get() is not _CAPABILITY):
            raise ValueError("CF-E040: privileged evidence requires runtime verification")
        return super().model_construct(_fields_set=_fields_set, **values)

    @model_validator(mode="after")
    def trustworthy(self) -> "Evidence":
        if (self.status in (EvidenceStatus.certified_bound, EvidenceStatus.empirical_calibration)
            and (not self.artifact_ref or _runtime_capability.get() is not _CAPABILITY)):
            raise ValueError("CF-E040: privileged evidence requires runtime verification")
        return self


class BackendKind(StrEnum):
    cpu_numeric = "cpu_numeric"
    gpu_numeric = "gpu_numeric"
    quantum_simulator = "quantum_simulator"
    qpu = "qpu"
    fault_tolerant_projection = "fault_tolerant_projection"


class UpstreamRef(BaseModel, frozen=True):
    port: str
    node: str
    impl_id: str
    backend_kind: BackendKind
    device: str
    job_ids: tuple[str, ...] = ()


class Provenance(BaseModel, frozen=True):
    impl_id: str
    impl_version: str
    backend_kind: BackendKind
    device: str
    seed: int | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)
    input_hashes: list[str] = Field(default_factory=list)
    upstream: tuple[UpstreamRef, ...] = ()
    relation_weakening_reason: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CostRecord(BaseModel, frozen=True):
    wall_s: float | None = None
    cpu_core_s: float | None = None
    gpu_s: float | None = None
    qpu_shots: int | None = None
    sim_shots: int | None = None
    qpu_s: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Uncertainty(BaseModel, frozen=True):
    stat_err: float | None = None
    covariance_ref: str | None = None
    bias_bound: float | None = None
    coverage: float | None = None


class Estimate(BaseModel, frozen=True):
    value: Quantity
    kind: QuantityKind
    target: str
    relation: TargetRelation
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    evidence: list[Evidence] = Field(default_factory=list)
    context: Context
    provenance: Provenance
    cost: CostRecord
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class GateVerdict(BaseModel, frozen=True):
    node: str
    passed: bool
    measured: Quantity | None = None
    threshold: Quantity | None = None
    reference: str | None = None
    stat_err: float | None = None
    z: float | None = None
    relation: TargetRelation
    reason: str
