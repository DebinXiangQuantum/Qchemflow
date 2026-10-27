from math import comb

import numpy as np
from pyscf import fci

from chemflow.impls.fci import ci_string_order, sector_states
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian, Sector


class SectorEngine:
    def __init__(self, hamiltonian: Hamiltonian, sector: Sector, store: ArtifactStore):
        self.sector = sector
        self.states = sector_states(sector)
        self.shape = (comb(sector.n_orb, sector.n_alpha), comb(sector.n_orb, sector.n_beta))
        self.alpha_order, self.beta_order = ci_string_order(sector)
        self.eri = fci.direct_spin1.absorb_h1e(store.get(hamiltonian.h1_ref),
                                               store.get(hamiltonian.h2_ref),
                                               sector.n_orb, (sector.n_alpha, sector.n_beta),
                                               0.5)
        self.constant = hamiltonian.ledger.total().value

    def apply(self, vector: np.ndarray) -> np.ndarray:
        ci = np.empty(self.shape, dtype=np.asarray(vector).dtype)
        ci[np.ix_(self.alpha_order, self.beta_order)] = np.asarray(vector).reshape(self.shape)
        result = fci.direct_spin1.contract_2e(self.eri, ci, self.sector.n_orb,
                                               (self.sector.n_alpha, self.sector.n_beta))
        return result[np.ix_(self.alpha_order, self.beta_order)].ravel() + self.constant * vector

    def energy(self, vector: np.ndarray) -> float:
        return float(np.real(np.vdot(vector, self.apply(vector))))


def hf_vector(sector: Sector) -> np.ndarray:
    vector = np.zeros(len(sector_states(sector)))
    bits = sum(1 << orbital for orbital in range(sector.n_alpha))
    bits |= sum(1 << (sector.n_orb + orbital) for orbital in range(sector.n_beta))
    vector[sector_states(sector).index(bits)] = 1
    return vector
