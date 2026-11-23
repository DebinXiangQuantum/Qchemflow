"""SQD for the 1D Hubbard chain: interacting-dynamics (Krylov) samples against classical samplers.

At a fixed budget of K strings per spin (the most frequent ones), compares exact
interacting time evolution from the Neel state with the U = 0 twin dynamics, the U = 0
ground state, CISD in the RHF basis and the exact ground state's own selection.

    python qcl/apps/hubbard_krylov_budget.py 10 4 out.json   # sites, U/t
"""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, ci, fci, gto, scf

from chemflow.domains.materials import lattice_arrays
from chemflow.impls.fci import ci_string_order, sector_states
from chemflow.impls.sector import SectorEngine
from chemflow.impls.sqd import sqd_sector
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian, Samples, Sector
from chemflow.types.quantum import ConstantLedger

L, U = int(sys.argv[1]), float(sys.argv[2])
out = sys.argv[3]
h1, eri = lattice_arrays(geometry="chain", n_sites=L, t=1.0, U=U)
na = nb = L // 2
store = ArtifactStore()
sector = Sector(n_alpha=na, n_beta=nb, n_orb=L)
states = sector_states(sector)
index = {s: i for i, s in enumerate(states)}


def data_for(a, b):
    ham = Hamiltonian(h1_ref=store.put(a), h2_ref=store.put(b), space="x", n_orb=L, ledger=ConstantLedger())
    return SimpleNamespace(hamiltonian=ham, sector=sector)


site = data_for(h1, eri)
engine = SectorEngine(site.hamiltonian, sector, store)
solver = fci.direct_spin1.FCI(); solver.conv_tol = 1e-10
e0, _ = solver.kernel(h1, eri, L, (na, nb))
free_engine = SectorEngine(data_for(h1, 0 * eri).hamiltonian, sector, store)
neel = np.zeros(len(states), dtype=complex)
neel_bits = sum(1 << s for s in range(0, L, 2)) | (sum(1 << s for s in range(1, L, 2)) << L)
neel[index[neel_bits]] = 1.0


def evolve(vector, eng, t, step=0.05, order=12):
    v = vector.copy()
    for _ in range(round(t / step)):
        term, total = v.copy(), v.copy()
        for k in range(1, order + 1):
            term = eng.apply(term) * (-1j * step / k)
            total = total + term
        v = total / np.linalg.norm(total)
    return v


def sample(probs, shots, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(states), size=shots, p=probs / probs.sum())
    keys, counts = np.unique(draws, return_counts=True)
    return {int(states[k]): int(c) for k, c in zip(keys, counts)}


def sqd(counts, data, extend, seed=0):
    shots = sum(counts.values())
    s = Samples(counts={str(k): v for k, v in counts.items()}, shots=shots, encoding_id="x",
                basis="computational", source="quantum_simulator", seed=seed)
    energy, diag = sqd_sector(data, sector, store, s, recovery=False, with_ext=extend, seed=seed)
    return 1000 * (energy - e0), diag["subspace_dimension"]


times = [0.5 * k for k in range(1, 9)]
t0 = time.time()
krylov, free_krylov = [], []
v, w = neel.copy(), neel.copy()
last = 0.0
for t in times:
    v = evolve(v, engine, t - last); w = evolve(w, free_engine, t - last); last = t
    krylov.append(np.abs(v) ** 2); free_krylov.append(np.abs(w) ** 2)
print("evolution", round(time.time() - t0), "s", flush=True)
# U = 0 ground state in the site basis (a Slater determinant)
eps, orbs = np.linalg.eigh(h1)
from chemflow.impls.lucj import LucjParams  # noqa: F401

occ = orbs[:, :na]
alpha = [s for s in sorted({x & ((1 << L) - 1) for x in states})]
def det_amp(mask):
    rows = [i for i in range(L) if mask >> i & 1]
    return np.linalg.det(occ[rows, :])
amp = {m: det_amp(m) for m in alpha}
free_gs = np.array([abs(amp[x & ((1 << L) - 1)] * amp[x >> L]) ** 2 for x in states])
# CISD in the RHF (momentum) basis, with H rotated to that basis
mol = gto.M(verbose=0); mol.nelectron = 2 * na; mol.incore_anyway = True
mf = scf.RHF(mol); mf.get_hcore = lambda *a: h1; mf.get_ovlp = lambda *a: np.eye(L)
mf._eri = ao2mo.restore(8, eri, L); mf.kernel()
C = mf.mo_coeff
h1m = C.T @ h1 @ C; erim = ao2mo.restore(1, ao2mo.full(ao2mo.restore(8, eri, L), C), L)
mo = data_for(h1m, erim)
cisd = ci.CISD(mf); cisd.verbose = 0; cisd.kernel()
a_order, b_order = ci_string_order(sector)
cisd_probs = (np.abs(cisd.to_fcivec(cisd.ci, L, 2 * na)[np.ix_(a_order, b_order)]) ** 2).ravel()


from pyscf.fci import selected_ci


def string_weights(probs):
    """Marginal weight of each alpha and beta string (dict mask -> weight), alpha+beta pooled."""
    mask = (1 << L) - 1
    weights = {}
    for x, p in zip(states, probs):
        if p <= 0:
            continue
        for s in (x & mask, x >> L):
            weights[s] = weights.get(s, 0.0) + p
    return weights


def subspace_energy(strings, a, b):
    strs = np.asarray(sorted(strings), dtype=np.int64)
    solver = selected_ci.SelectedCI(); solver.verbose = 0
    energy, _ = solver.kernel_fixed_space(a, b, L, (na, nb), (strs, strs))
    return 1000 * (energy - e0)


def top_k(weights, k):
    return [s for s, _ in sorted(weights.items(), key=lambda item: -item[1])[:k]]


solver_exact = fci.direct_spin1.FCI(); solver_exact.conv_tol = 1e-10
_, vec = solver_exact.kernel(h1, eri, L, (na, nb))
exact_probs = (np.abs(vec[np.ix_(a_order, b_order)]) ** 2).ravel()
pooled = {"interacting Krylov (quantum)": (np.sum(krylov, axis=0) / len(krylov), "site"),
          "U=0 Krylov twin (classical)": (np.sum(free_krylov, axis=0) / len(free_krylov), "site"),
          "U=0 ground state (classical)": (free_gs, "site"),
          "CISD, RHF basis (classical)": (cisd_probs, "mo"),
          "exact ground state (ideal selection)": (exact_probs, "site")}
ints = {"site": (h1, eri), "mo": (h1m, erim)}
n_strings = len(alpha)
results = {}
for k in (25, 50, 100, 200):
    if k > n_strings:
        continue
    for name, (probs, frame) in pooled.items():
        row = {}
        row["exact_marginals"] = subspace_energy(top_k(string_weights(probs), k), *ints[frame])
        for shots in (4000, 16000):
            errs = []
            for seed in (0, 1):
                counts = sample(probs, shots, seed)
                emp = np.zeros(len(states)); 
                for key, c in counts.items():
                    emp[index[key]] = c
                errs.append(subspace_energy(top_k(string_weights(emp), k), *ints[frame]))
            row[str(shots)] = float(np.mean(errs))
        results.setdefault(str(k), {})[name] = row
        print(f"L={L} U={U} K={k:4d} ({k*k} dets of {len(states)}) {name:38s} "
              f"exact-marginal {row['exact_marginals']:9.3f}  4k {row['4000']:9.3f}  16k {row['16000']:9.3f} mEh", flush=True)
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"L": L, "U": U, "e0": e0, "sector": len(states), "strings": n_strings,
               "results": results}, handle, indent=1)
