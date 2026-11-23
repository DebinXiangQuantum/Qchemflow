"""ffsim LUCJ states (IBM tutorial construction) for N2 CAS(10,10), with their integrals.

Runs in an environment with ffsim 0.0.86 and Qiskit (not ChemFlow's): it writes the
ideal probabilities, the active-space integrals and the IBM-pipeline CZ counts that
qcl/apps/ffsim_lucj_evaluate.py scores with ChemFlow's SQD.

    python qcl/apps/ffsim_lucj_states.py out_dir
"""
import json
import sys

import ffsim
import numpy as np
from pyscf import cc, gto, scf

out = sys.argv[1]
meta = {}
for label, R in (("eq", 1.098), ("str", 2.2)):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    nelecas, ncas = 10, 10
    ncore = (mol.nelectron - nelecas) // 2
    frozen = list(range(ncore)) + list(range(ncore + ncas, mf.mo_coeff.shape[1]))
    ccsd = cc.CCSD(mf, frozen=frozen)
    ccsd.max_cycle = 200
    ccsd.run()
    norb, nelec = ncas, (5, 5)
    from pyscf import mcscf
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1, ecore = cas.get_h1eff()
    from pyscf import ao2mo
    eri = ao2mo.restore(1, cas.get_h2eff(), ncas)
    np.savez(f"{out}/{label}_integrals.npz", h1=h1, eri=eri, ecore=ecore)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 4)]
    variants = {
        "ibm_line": {"n_reps": 1, "interaction_pairs": (pairs_aa, pairs_ab)},
        "ibm_line_reps2": {"n_reps": 2, "interaction_pairs": (pairs_aa, pairs_ab)},
        "ibm_allpairs": {"n_reps": 1, "interaction_pairs": None},
        "ibm_ab_only": {"n_reps": 1, "interaction_pairs": ([], pairs_ab)},
    }
    for name, kw in variants.items():
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, t1=ccsd.t1, **kw)
        op = ffsim.UCJOpSpinBalanced(diag_coulomb_mats=op.diag_coulomb_mats,
                                     orbital_rotations=op.orbital_rotations,
                                     final_orbital_rotation=None)
        vec = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), op, norb=norb, nelec=nelec)
        probs = np.abs(vec.reshape(ffsim.dim(norb, 5), ffsim.dim(norb, 5))) ** 2
        np.save(f"{out}/{label}_{name}.npy", probs)
        if label == "eq":
            from qiskit import QuantumCircuit, QuantumRegister
            from qiskit.transpiler import CouplingMap
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            q = QuantumRegister(2 * norb); c = QuantumCircuit(q)
            c.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), q)
            c.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), q); c.measure_all()
            best = None
            for seed in range(5):
                pm = generate_preset_pass_manager(optimization_level=3, coupling_map=CouplingMap.from_line(2 * norb),
                                                  basis_gates=["cz", "rz", "sx", "x"], seed_transpiler=seed)
                pm.pre_init = ffsim.qiskit.PRE_INIT
                cz = pm.run(c).count_ops().get("cz", 0)
                best = cz if best is None else min(best, cz)
            meta[f"cz_{name}"] = int(best)
        meta[f"{label}_{name}"] = {"R": R, "converged": bool(ccsd.converged),
                                   "p_hf": float(probs[0, 0]), "cz": meta.get(f"cz_{name}")}
        print(label, name, meta[f"{label}_{name}"], flush=True)
with open(f"{out}/meta.json", "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=1)
