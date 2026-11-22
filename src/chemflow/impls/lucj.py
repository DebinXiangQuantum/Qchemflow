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
    # Integrals only: keep PySCF on in-core paths (its out-of-core paths need a real basis).
    molecule.incore_anyway = True
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


def _string_rotation(n_orb: int, n_elec: int, kparams: np.ndarray) -> np.ndarray:
    """exp(-K) for one spin, as a matrix on that spin's occupation strings.

    K = sum_(p<q) kappa_pq (a+_p a_q - a+_q a_p). A same-spin excitation changes the
    other spin's bits not at all and crosses each of them an even number of times,
    so the sector operator is exp(-K_alpha) (x) exp(-K_beta).
    """
    from scipy.linalg import expm

    from chemflow.impls.fci import _annihilate, _create

    strings = [sum(1 << orbital for orbital in occupied)
               for occupied in combinations(range(n_orb), n_elec)]
    position = {bits: index for index, bits in enumerate(strings)}
    generator = np.zeros((len(strings), len(strings)))
    for index, (left, right) in enumerate(combinations(range(n_orb), 2)):
        kappa = float(kparams[index])
        if kappa == 0.0:
            continue
        for column, bits in enumerate(strings):
            removed = _annihilate(bits, right)
            if removed is None:
                continue
            created = _create(removed[0], left)
            if created is None:
                continue
            generator[position[created[0]], column] += kappa * removed[1] * created[1]
    return expm(-(generator - generator.T))


def lucj_state(sector: Sector, layers: list[LucjParams], *,
               final_rotation: bool = True) -> np.ndarray:
    """LUCJ state in the sector basis of ``sector_states`` (alpha strings outer).

    Each layer is exp(K) exp(i J) exp(-K) applied to the Hartree-Fock state; the
    orbital rotations act on the alpha and beta strings separately (two small dense
    matrices on the state reshaped to alpha x beta) and the Jastrow phases are
    diagonal in the occupation basis. ``final_rotation=False`` stops before the last
    layer's exp(K): the same state expressed in the orbitals rotated by exp(-K), whose
    Hamiltonian is ``rotated_hamiltonian``.
    """
    if not final_rotation and len(layers) != 1:
        raise ValueError("Omitting the final rotation needs one LUCJ layer")
    n_orb = sector.n_orb
    alpha_count = math.comb(n_orb, sector.n_alpha)
    beta_count = math.comb(n_orb, sector.n_beta)
    bits = np.asarray(sector_states(sector), dtype=np.int64)
    occupied = ((bits[:, None] >> np.arange(2 * n_orb)) & 1).astype(float)
    state = hf_vector(sector).astype(complex).reshape(alpha_count, beta_count)
    for layer in layers:
        alpha = _string_rotation(n_orb, sector.n_alpha, layer.kparams)
        beta = (alpha if sector.n_beta == sector.n_alpha else
                _string_rotation(n_orb, sector.n_beta, layer.kparams))
        state = alpha @ state @ beta.T
        phases = np.zeros(len(bits))
        for orbital in range(n_orb - 1):
            for offset in (0, n_orb):
                phases += (layer.j_same[orbital] * occupied[:, offset + orbital] *
                           occupied[:, offset + orbital + 1])
        for orbital in range(0, n_orb, 4):
            phases += layer.j_opp[orbital // 4] * occupied[:, orbital] * occupied[:, n_orb + orbital]
        state = state * np.exp(1j * phases).reshape(alpha_count, beta_count)
        if final_rotation:
            state = alpha.T @ state @ beta
    return state.reshape(-1)


def orbital_rotation(n_orb: int, kparams) -> np.ndarray:
    """The one-electron matrix exp(-kappa) of the LUCJ rotation exp(-K), K from kparams."""
    from scipy.linalg import expm

    generator = np.zeros((n_orb, n_orb))
    generator[np.triu_indices(n_orb, 1)] = np.asarray(kparams, dtype=float)
    return expm(-(generator - generator.T))


def rotated_integrals(h1: np.ndarray, h2: np.ndarray, kparams
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Integrals of F H F^dagger, F = exp(-K): the Hamiltonian that samples of an LUCJ
    circuit without its final rotation (``final_rotation=False``) diagonalize.

    With U = exp(-kappa), h' = U h U^T and the same on every index of the chemist-order
    two-electron integrals; the energies are unchanged (a unitary change of orbitals).
    """
    rotation = orbital_rotation(len(h1), kparams).T
    h1 = rotation.T @ h1 @ rotation
    h2 = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2, rotation, rotation, rotation, rotation,
                   optimize=True)
    return h1, h2


def lucj_state_reference(sector: Sector, layers: list[LucjParams]) -> np.ndarray:
    """The direct sparse construction in the full sector (slow; kept for tests)."""
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
