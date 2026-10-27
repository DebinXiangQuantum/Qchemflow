from pydantic import BaseModel

from chemflow.ids import content_hash


class ElectronicModel(BaseModel, frozen=True):
    basis: str
    hamiltonian_class: str = "nonrel_bo"
    ecp: str | None = None
    frozen_core: int = 0
    reference: str = "RHF"
    dft_functional: str | None = None


class ActiveSpaceSpec(BaseModel, frozen=True):
    n_elec: int
    n_orb: int
    method: str = "manual"
    orbital_source: str = "canonical_hf"


def model_protocol_id(model_id: str, active: ActiveSpaceSpec | None) -> str:
    """Identify both the electronic model and the declared orbital partition."""
    return content_hash({"model_id": model_id,
                         "active": active.model_dump() if active is not None else None})


class OrbitalSpace(BaseModel, frozen=True):
    coefficients_ref: str
    n_orb: int
    n_elec: int
    frozen_orbitals: int = 0
    ordering: str = "canonical_hf"
    spin_treatment: str = "restricted"
