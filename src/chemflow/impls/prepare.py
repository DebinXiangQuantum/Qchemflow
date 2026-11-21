"""Sector-state preparation (HF, UCCSD, LUCJ) with the hardware circuit it corresponds to."""

from typing import Any

import numpy as np

from chemflow.ids import content_hash
from chemflow.impls.ansatz import uccsd_pool, uccsd_state
from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals, lucj_state
from chemflow.impls.sector import hf_vector
from chemflow.types import CircuitSpec, StatePrep


def prepare_sector_state(data: Any, sector: dict, store: Any, ansatz: str = "hf",
                         init: str = "zero", params: list[float] | None = None,
                         seed: int = 0, *, materialize: bool = True) -> StatePrep:
    if ansatz not in ("hf", "uccsd", "lucj") or init not in ("zero", "ccsd", "random"):
        raise ValueError("Unsupported ansatz or initialization")
    if (sector["n_alpha"], sector["n_beta"], sector["n_orb"]) != (
            data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb):
        raise ValueError("Sector differs from Hamiltonian")
    if ansatz == "hf":
        state = hf_vector(data.sector) if materialize else None
        circuit = CircuitSpec(ansatz="hf", n_orb=data.sector.n_orb,
                              n_alpha=data.sector.n_alpha, n_beta=data.sector.n_beta)
    elif ansatz == "uccsd":
        if not materialize:
            raise ValueError("UCCSD preparation has no hardware circuit implementation")
        circuit = None
        _, generators = uccsd_pool(data.sector)
        values = np.asarray(params if params is not None else np.zeros(len(generators)), dtype=float)
        if init == "random" and params is None:
            values = np.random.default_rng(seed).normal(0, 0.12, len(generators))
        state = uccsd_state(data.sector, values, generators)
    else:
        values = LucjParams.zeros(data.sector.n_orb)
        if init == "ccsd":
            if hasattr(data, "mean_field"):
                data.mean_field(store)
            values = ccsd_seed_from_integrals(
                data.hamiltonian, data.sector, store,
                site_reference=data.mf if getattr(getattr(data, "space", None), "ordering", None)
                == "sites" else None) or values
        elif init == "random":
            values = LucjParams.random(data.sector.n_orb, np.random.default_rng(seed))
        if params is not None:
            values = LucjParams.from_vector(np.asarray(params), data.sector.n_orb)
        state = lucj_state(data.sector, [values]) if materialize else None
        circuit = CircuitSpec(ansatz="lucj", n_orb=data.sector.n_orb,
                              n_alpha=data.sector.n_alpha, n_beta=data.sector.n_beta,
                              layers=(tuple(float(item) for item in values.to_vector()),))
    return StatePrep(state_ref=store.put(state) if materialize else "", encoding_id="sector_state",
                     hamiltonian_id=data.hamiltonian.id, sector=data.sector,
                     recipe=f"{ansatz}:{init}:{content_hash(params)}", circuit=circuit)
