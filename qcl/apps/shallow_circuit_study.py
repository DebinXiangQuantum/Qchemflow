"""Shallow LUCJ variants: CZ cost against SQD quality from ideal samples, and classical samplers.

    python qcl/apps/shallow_circuit_study.py n2_1010_eq out.json

phi = R_f J U |HF>, U|HF> prepared with the optimal Slater Givens network, R_f a (possibly
truncated) copy of U^dagger. phi = W psi with W = R_f U and psi the full LUCJ state, so SQD
uses H rotated to frame W (exact). R_f = I is the free-fermion (classical) limit.
"""
import json
import sys
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from chemflow.backends.circuits import _rotation, compile_native, slater_givens
from chemflow.impls.fci import sector_states
from chemflow.impls.lucj import (
    _string_rotation,
    ccsd_seed_from_integrals,
    lucj_state,
    orbital_rotation,
)
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import SectorEngine
from chemflow.impls.sqd import sqd_sector
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Samples
from chemflow.types.model import ActiveSpaceSpec, ElectronicModel


def givens_gate(c, s):
    gate = np.eye(4)
    gate[1:3, 1:3] = [[c, -s], [s, c]]
    return UnitaryGate(gate)


def string_op(C, n, k):
    """N-particle operator of one-body matrix C on the strings of sector_states order."""
    occ = [list(c) for c in combinations(range(n), k)]
    idx = np.array(occ)
    sub = C[idx[:, None, :, None], idx[None, :, None, :]]   # [y, x, k, k] = C[occ_y, occ_x]
    return np.linalg.det(sub)


def one_body_of_gates(n, gates):
    """Single-particle matrix of a one-spin Givens gate list (qubits 0..n-1)."""
    columns = []
    for j in range(n):
        qc = QuantumCircuit(n)
        qc.x(j)
        for top, bottom, c, s in gates:
            qc.append(givens_gate(c, s), [top, bottom])
        sv = Statevector.from_instruction(qc).data
        columns.append([sv[1 << i] for i in range(n)])
    return np.real(np.array(columns).T)


def forward_gates(M, n):
    """The Givens list _rotation emits for matrix M (one spin), in circuit order."""
    qc = QuantumCircuit(2 * n)
    _rotation(qc, M, n)
    out = []
    for inst in qc.data:
        top, bottom = (qc.find_bit(q).index for q in inst.qubits)
        if top >= n:
            continue
        mat = inst.operation.to_matrix()
        out.append((top, bottom, float(np.real(mat[1, 1])), float(np.real(mat[2, 1]))))
    return out


def inverse_gates(gates):
    return [(t, b, c, -s) for t, b, c, s in reversed(gates)]


def circuit_of(n, n_occ, M, layer, final_gates, with_jastrow=True):
    qc = QuantumCircuit(2 * n)
    for off in (0, n):
        for o in range(n_occ):
            qc.x(off + o)
    for top, bottom, c, s in slater_givens(M[:, :n_occ]):
        for off in (0, n):
            qc.append(givens_gate(c, s), [off + top, off + bottom])
    if with_jastrow:
        for o, a in enumerate(layer.j_same):
            if abs(a) > 1e-12:
                for off in (0, n):
                    qc.cp(float(a), off + o, off + o + 1)
        for i, a in enumerate(layer.j_opp):
            if abs(a) > 1e-12:
                qc.cp(float(a), 4 * i, n + 4 * i)
    for top, bottom, c, s in final_gates:
        for off in (0, n):
            qc.append(givens_gate(c, s), [off + top, off + bottom])
    return qc


def sqd_error(data, store, h1w, h2w, probs, states, shots, seed, exact, extend):
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(states), size=shots, p=probs / probs.sum())
    keys, counts = np.unique(draws, return_counts=True)
    samples = Samples(counts={str(states[k]): int(c) for k, c in zip(keys, counts)}, shots=shots,
                      encoding_id="x", basis="computational", source="quantum_simulator", seed=seed)
    ham = data.hamiltonian.model_copy(update={"h1_ref": store.put(h1w), "h2_ref": store.put(h2w)})
    rotated = type(data)(**{**data.__dict__, "hamiltonian": ham})
    energy, diag = sqd_sector(rotated, data.sector, store, samples, recovery=False, with_ext=extend, seed=seed)
    return 1000 * (energy - exact), diag["subspace_dimension"]


def rotate(h1, h2, C):
    """Integrals in the frame where states are S(C) psi: h' = C h C^T (calibrated below)."""
    return C @ h1 @ C.T, np.einsum("pqrs,ip,jq,kr,ls->ijkl", h2, C, C, C, C, optimize=True)


def main(system, atoms, basis, ne, no, shots_list=(2000, 10000), seeds=(0, 1, 2)):
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis=basis), store,
                          active=ActiveSpaceSpec(n_elec=ne, n_orb=no))
    sector = data.sector
    layer = ccsd_seed_from_integrals(data.hamiltonian, sector, store)
    M = orbital_rotation(no, layer.kparams)
    n_occ = ne // 2
    states = sector_states(sector)
    h1, h2 = store.get(data.hamiltonian.h1_ref), store.get(data.hamiltonian.h2_ref)
    engine = SectorEngine(data.hamiltonian, sector, store)
    from pyscf import fci
    exact = fci.direct_spin1.FCI().kernel(h1, h2, no, (n_occ, n_occ))[0] + data.hamiltonian.ledger.total().value
    psi = lucj_state(sector, [layer])
    # conventions: string op of orbital_rotation must match _string_rotation
    S_ref = _string_rotation(no, n_occ, layer.kparams)
    conv = "C" if np.allclose(string_op(M, no, n_occ), S_ref, atol=1e-10) else "CT"
    assert conv == "C" or np.allclose(string_op(M.T, no, n_occ), S_ref, atol=1e-10)
    fwd = forward_gates(M, no)
    C_fwd = one_body_of_gates(no, fwd)
    flip = np.allclose(C_fwd, M, atol=1e-10)
    assert flip or np.allclose(C_fwd, M.T, atol=1e-10), "forward network convention"
    final_full = inverse_gates(fwd)
    angles = np.array([abs(np.arctan2(s, c)) for _, _, c, s in final_full])
    rows = []
    for eps in (None, 0.0, 0.01, 0.03, 0.1, 0.2, 0.5):
        final = [] if eps is None else [g for g, a in zip(final_full, angles) if a >= eps]
        R = one_body_of_gates(no, final) if final else np.eye(no)
        C_R = R if flip else R.T                       # orbital-rotation convention
        W = C_R @ M                                    # phi = S(W) psi
        Sw = string_op(W if conv == "C" else W.T, no, n_occ)
        dim = Sw.shape[0]
        phi = (Sw @ psi.reshape(dim, dim) @ Sw.T).ravel()
        h1w, h2w = rotate(h1, h2, W if conv == "C" else W.T)
        check = SectorEngine(data.hamiltonian.model_copy(update={"h1_ref": store.put(h1w), "h2_ref": store.put(h2w)}),
                             sector, store).energy(phi) - engine.energy(psi)
        qc = circuit_of(no, n_occ, M, layer, final)
        cz = compile_native(qc).count_ops().get("cz", 0)
        probs = np.abs(phi) ** 2
        row = {"family": "free-fermion (R_f=I)" if eps is None else f"final rotation, |angle|>={eps}",
               "final_givens": len(final), "cz": cz, "frame_check": float(check)}
        for shots in shots_list:
            for extend in (False, True):
                errs = [sqd_error(data, store, h1w, h2w, probs, states, shots, s, exact, extend) for s in seeds]
                row[f"{shots}{'+ext' if extend else ''}"] = (float(np.mean([e for e, _ in errs])), int(np.mean([d for _, d in errs])))
        rows.append(row)
        print(system, json.dumps(row), flush=True)
    # classical CISD sampler in the canonical frame
    from pyscf import ao2mo, ci, gto, scf
    mol = gto.M(verbose=0); mol.nelectron = ne; mol.incore_anyway = True
    mf = scf.RHF(mol); mf.get_hcore = lambda *a: h1; mf.get_ovlp = lambda *a: np.eye(no); mf._eri = ao2mo.restore(8, h2, no)
    mf.kernel(dm0=np.diag([2.0] * n_occ + [0.0] * (no - n_occ)))
    cisd = ci.CISD(mf); cisd.verbose = 0; cisd.kernel()
    vec = cisd.to_fcivec(cisd.ci, no, ne)            # pyscf string order
    from chemflow.impls.fci import ci_string_order
    a_order, b_order = ci_string_order(sector)
    probs = (np.abs(vec[np.ix_(a_order, b_order)]) ** 2).ravel()
    row = {"family": "classical CISD sampler (canonical)", "cz": 0}
    for shots in shots_list:
        for extend in (False, True):
            errs = [sqd_error(data, store, h1, h2, probs, states, shots, s, exact, extend) for s in seeds]
            row[f"{shots}{'+ext' if extend else ''}"] = (float(np.mean([e for e, _ in errs])), int(np.mean([d for _, d in errs])))
    rows.append(row)
    print(system, json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    specs = {"n2_1010_eq": ("N 0 0 0; N 0 0 1.098", "cc-pvdz", 10, 10),
             "n2_1010_str": ("N 0 0 0; N 0 0 2.2", "cc-pvdz", 10, 10),
             "n2_66_eq": ("N 0 0 0; N 0 0 1.098", "cc-pvdz", 6, 6),
             "n2_1212_eq": ("N 0 0 0; N 0 0 1.098", "cc-pvdz", 12, 12)}
    rows = main(which, *specs[which])
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)
