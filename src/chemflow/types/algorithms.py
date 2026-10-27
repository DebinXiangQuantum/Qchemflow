"""Determinant subspaces and artifact-backed projected spectral resolutions."""

from math import comb
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from chemflow.relations import TargetRelation
from chemflow.types.quantum import Sector
from chemflow.types.results import CostRecord, Provenance


class Subspace(BaseModel, frozen=True):
    sector: Sector
    determinants: tuple[int, ...]
    provenance: Provenance | None = None
    cost: CostRecord = Field(default_factory=CostRecord)

    @model_validator(mode="after")
    def valid(self):
        if self.sector.total_spin is not None:
            raise ValueError("Determinants do not certify a total-spin eigenspace")
        if self.sector.n_orb < 1:
            raise ValueError("Subspace requires at least one spatial orbital")
        half = (1 << self.sector.n_orb) - 1
        if len(set(self.determinants)) != len(self.determinants):
            raise ValueError("Subspace determinants must be distinct")
        for state in self.determinants:
            if (state < 0 or state >= 1 << (2 * self.sector.n_orb) or
                    (state & half).bit_count() != self.sector.n_alpha or
                    (state >> self.sector.n_orb).bit_count() != self.sector.n_beta):
                raise ValueError("Subspace determinant outside its sector")
        return self

    @property
    def coverage(self):
        return len(self.determinants) / (comb(self.sector.n_orb, self.sector.n_alpha) *
                                         comb(self.sector.n_orb, self.sector.n_beta))


class Spectrum(BaseModel, frozen=True):
    energies: tuple[float, ...]
    unit: Literal["hartree"] = "hartree"
    vectors_ref: str
    resolution_energies_ref: str
    resolution_vectors_ref: str
    subspace: Subspace
    hamiltonian_id: str
    relation: TargetRelation = TargetRelation.approximate
    provenance: Provenance | None = None
    cost: CostRecord = Field(default_factory=CostRecord)
    diagnostics: dict = Field(default_factory=dict)

    @property
    def sector(self):
        return self.subspace.sector

    @property
    def coverage(self):
        return self.subspace.coverage

    @model_validator(mode="after")
    def ordered(self):
        import math

        if (not self.energies or len(self.energies) > len(self.subspace.determinants) or
                not all(math.isfinite(value) for value in self.energies) or
                list(self.energies) != sorted(self.energies)):
            raise ValueError("Spectrum requires finite ordered roots within its subspace")
        return self


class SpectralEstimate(BaseModel, frozen=True):
    omega: tuple[float, ...]
    spectral: tuple[float, ...]
    unit: str
    relation: TargetRelation = TargetRelation.approximate
    diagnostics: dict = Field(default_factory=dict)
    provenance: Provenance | None = None
    cost: CostRecord = Field(default_factory=CostRecord)

    @model_validator(mode="after")
    def valid(self):
        import math

        if (len(self.omega) < 2 or len(self.omega) != len(self.spectral) or
                not all(math.isfinite(value) for value in (*self.omega, *self.spectral)) or
                any(right <= left for left, right in zip(self.omega, self.omega[1:])) or
                any(value < 0 for value in self.spectral)):
            raise ValueError("Spectral estimate requires an increasing finite grid and positive density")
        return self
