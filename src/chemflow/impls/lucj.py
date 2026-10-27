import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from chemflow.impls.ansatz import excitation
from chemflow.impls.fci import sector_states
from chemflow.impls.sector import hf_vector
from chemflow.types import Sector


@dataclass
class LucjParams:
    norb: int
    kparams: np.ndarray
    j_same: np.ndarray
    j_opp: np.ndarray

    @property
    def size(self) -> int:
        return len(self.kparams) + len(self.j_same) + len(self.j_opp)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.kparams, self.j_same, self.j_opp])

    @classmethod
    def from_vector(cls, vector: np.ndarray, norb: int) -> "LucjParams":
        count = norb * (norb - 1) // 2
        return cls(norb, np.asarray(vector[:count], dtype=float),
                   np.asarray(vector[count:count + norb - 1], dtype=float),
                   np.asarray(vector[count + norb - 1:], dtype=float))

    @classmethod
    def zeros(cls, norb: int) -> "LucjParams":
        return cls(norb, np.zeros(norb * (norb - 1) // 2),
                   np.zeros(max(norb - 1, 0)), np.zeros((norb + 3) // 4))

    @classmethod
    def random(cls, norb: int, rng: np.random.Generator, scale: float = 0.1) -> "LucjParams":
        params = cls.zeros(norb)
        params.kparams = scale * rng.standard_normal(params.kparams.size)
        params.j_same = scale * rng.standard_normal(params.j_same.size)
        params.j_opp = scale * rng.standard_normal(params.j_opp.size)
        return params


    @classmethod
    def ccsd_seed(cls, data, scale_k: float = 1.0, scale_j: float = 1.0
                  ) -> "LucjParams | None":
        if data.ccsd_t1 is None or data.ccsd_t2 is None:
            return None
        params = cls.zeros(data.norb)
        n_pair = data.n_alpha
        t1, t2 = data.ccsd_t1, data.ccsd_t2
        rotation = np.zeros((data.norb, data.norb))
        for occupied in range(n_pair):
            for virtual in range(n_pair, data.norb):
                diagonal = float(t2[occupied, occupied, virtual - n_pair, virtual - n_pair])
                rotation[occupied, virtual] = scale_k * (
                    float(t1[occupied, virtual - n_pair]) +
                    math.copysign(math.sqrt(abs(diagonal)), diagonal))
        params.kparams = rotation[np.triu_indices(data.norb, 1)]
        for orbital in range(0, data.norb, 4):
            if orbital < n_pair:
                value = float(np.trace(t2[orbital, orbital]))
            else:
                value = float(np.trace(t2[:, :, orbital - n_pair, orbital - n_pair]))
            params.j_opp[orbital // 4] = scale_j * math.copysign(math.sqrt(abs(value)), value)
        return params


def ccsd_seed_from_integrals(hamiltonian, sector: Sector, store, site_reference=None
                             ) -> LucjParams | None:
    """Heuristic CC amplitude seed, with orbital rotations pulled back for site bases.

    Site integrals are not canonical MO integrals: CC must use the converged
    reference, and the antisymmetric rotation generator must be expressed in
    the site basis. J angles remain heuristic CC correlation-strength seeds.
    """
    from types import SimpleNamespace

    from pyscf import cc, gto, scf

    if sector.n_alpha != sector.n_beta:
        return None
    if sector.n_alpha in (0, sector.n_orb):
        return LucjParams.zeros(sector.n_orb)
    h1, h2 = store.get(hamiltonian.h1_ref), store.get(hamiltonian.h2_ref)
    molecule = gto.M(verbose=0)
    molecule.nelectron = sector.n_alpha + sector.n_beta
    reference = scf.RHF(molecule)
    reference.get_hcore = lambda *args: h1
    reference.get_ovlp = lambda *args: np.eye(sector.n_orb)
    reference._eri = h2
    reference.mo_coeff = np.eye(sector.n_orb)
    reference.mo_occ = np.array([2] * sector.n_alpha + [0] * (sector.n_orb - sector.n_alpha))
    reference.mo_energy = np.diag(h1)
    if site_reference is not None:
        reference = site_reference
    solver = cc.CCSD(reference)
    solver.verbose = 0
    solver.kernel()
    values = LucjParams.ccsd_seed(SimpleNamespace(ccsd_t1=solver.t1, ccsd_t2=solver.t2,
                                                 norb=sector.n_orb, n_alpha=sector.n_alpha))
    if site_reference is not None:
        if not solver.converged:
            raise ValueError("Site-basis CCSD initialization did not converge")
        rotation = np.zeros((sector.n_orb, sector.n_orb))
        rotation[np.triu_indices(sector.n_orb, 1)] = values.kparams
        rotation -= rotation.T
        rotation = reference.mo_coeff @ rotation @ reference.mo_coeff.T
        values.kparams = rotation[np.triu_indices(sector.n_orb, 1)]
    return values


def lucj_state(sector: Sector, layers: list[LucjParams]) -> np.ndarray:
    n_orb = sector.n_orb
    states = sector_states(sector)
    state = hf_vector(sector).astype(complex)
    for layer in layers:
        rotation = sparse.csr_matrix((len(states), len(states)))
        for index, (left, right) in enumerate(combinations(range(n_orb), 2)):
            for offset in (0, n_orb):
                forward = excitation(sector, (right + offset,), (left + offset,))
                generator = layer.kparams[index] * (forward - forward.T)
                rotation += generator
        state = expm_multiply(-rotation, state)
        phases = np.zeros(len(states))
        for index, bits in enumerate(states):
            for orbital in range(n_orb - 1):
                for offset in (0, n_orb):
                    phases[index] += (layer.j_same[orbital] *
                                      ((bits >> (offset + orbital)) & 1) *
                                      ((bits >> (offset + orbital + 1)) & 1))
            for orbital in range(0, n_orb, 4):
                phases[index] += (layer.j_opp[orbital // 4] *
                                  ((bits >> orbital) & 1) * ((bits >> (n_orb + orbital)) & 1))
        state *= np.exp(1j * phases)
        state = expm_multiply(rotation, state)
    return state
