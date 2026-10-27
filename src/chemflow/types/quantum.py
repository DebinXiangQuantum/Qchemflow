from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from chemflow.ids import content_hash
from chemflow.units import Quantity


class Sector(BaseModel, frozen=True):
    n_alpha: int
    n_beta: int
    n_orb: int
    n_elec: int | None = None
    total_spin: int | None = None

    @model_validator(mode="after")
    def feasible(self) -> "Sector":
        if min(self.n_alpha, self.n_beta) < 0 or max(self.n_alpha, self.n_beta) > self.n_orb:
            raise ValueError("Infeasible particle sector")
        if self.n_elec is not None and self.n_alpha + self.n_beta != self.n_elec:
            raise ValueError("Particle counts disagree with N")
        if self.total_spin == 0 and self.n_alpha != self.n_beta:
            raise ValueError("Singlet requires equal alpha and beta counts")
        return self


class ConstantTerm(BaseModel, frozen=True):
    name: str
    value: Quantity
    included: bool = True


class ConstantLedger(BaseModel, frozen=True):
    terms: tuple[ConstantTerm, ...] = ()

    @model_validator(mode="after")
    def unique(self) -> "ConstantLedger":
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("CF-E030: duplicate constant term")
        return self

    def add(self, name: str, value: Quantity) -> "ConstantLedger":
        return ConstantLedger(terms=(*self.terms, ConstantTerm(name=name, value=value)))

    def total(self) -> Quantity:
        return Quantity(value=sum(term.value.to("hartree").value for term in self.terms
                                  if term.included), unit="hartree")


class Hamiltonian(BaseModel, frozen=True):
    h1_ref: str
    h2_ref: str
    space: str
    n_orb: int
    ledger: ConstantLedger = Field(default_factory=ConstantLedger)
    convention: str = "chemist_spatial_pqrs"

    @property
    def id(self) -> str:
        return content_hash(self)


class QubitEncoding(BaseModel, frozen=True):
    method: str = "JW"
    qubit_order: str = "alpha_block_then_beta_block"
    n_qubits: int
    hamiltonian_id: str

    @property
    def id(self) -> str:
        return content_hash(self)


class QubitOperator(BaseModel, frozen=True):
    terms: dict[str, float]
    encoding_id: str
    matrix_ref: str


class CircuitSpec(BaseModel, frozen=True):
    ansatz: Literal["hf", "lucj", "uccsd", "custom"]
    n_orb: int
    n_alpha: int
    n_beta: int
    layers: tuple[tuple[float, ...], ...] = ()
    measurement_basis: str | None = None
    n_qubits: int | None = None
    n_clbits: int = 0
    gates: tuple[dict[str, Any], ...] = ()


class StatePrep(BaseModel, frozen=True):
    state_ref: str
    encoding_id: str
    hamiltonian_id: str
    recipe: str = "explicit_statevector"
    sector: Sector | None = None
    circuit: CircuitSpec | None = None


class Samples(BaseModel, frozen=True):
    counts: dict[str, int]
    shots: int
    encoding_id: str
    basis: str
    source: str
    seed: int
    device: str = "local"
    job_ids: tuple[str, ...] = ()
