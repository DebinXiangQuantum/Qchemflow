"""SQD from ideal samples of ffsim LUCJ states, ChemFlow's LUCJ seed and a CISD sampler.

    python qcl/apps/ffsim_lucj_evaluate.py out_dir     # after ffsim_lucj_states.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, ci, fci, gto, scf

from chemflow.impls.fci import ci_string_order, sector_states
from chemflow.impls.lucj import ccsd_seed_from_integrals, lucj_state
from chemflow.impls.sqd import sqd_sector
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian, Samples, Sector
from chemflow.types.quantum import ConstantLedger
from chemflow.units import Q

folder = Path(sys.argv[1])
meta = json.loads((folder / "meta.json").read_text())
results = {}
for label in ("eq", "str"):
    ints = np.load(folder / f"{label}_integrals.npz")
    h1, eri, ecore = ints["h1"], ints["eri"], float(ints["ecore"])
    n, ne = 10, 10
    store = ArtifactStore()
    sector = Sector(n_alpha=5, n_beta=5, n_orb=n)
    ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), space="ffsim", n_orb=n,
                      ledger=ConstantLedger().add("core", Q(ecore, "hartree")))
    data = SimpleNamespace(hamiltonian=ham, sector=sector)
    exact = fci.direct_spin1.FCI().kernel(h1, eri, n, (5, 5))[0] + ecore
    states = sector_states(sector)
    a_order, b_order = ci_string_order(sector)

    def ours(pyscf_probs, a_order=a_order, b_order=b_order):
        return pyscf_probs[np.ix_(a_order, b_order)].ravel()

    samplers = {name.split("_", 1)[1]: ours(np.load(folder / f"{name}.npy"))
                for name in meta if name.startswith(label + "_")}
    seed = ccsd_seed_from_integrals(ham, sector, store)
    samplers["chemflow_seed"] = np.abs(lucj_state(sector, [seed])) ** 2
    mol = gto.M(verbose=0); mol.nelectron = ne; mol.incore_anyway = True
    mf = scf.RHF(mol); mf.get_hcore = lambda *a, h1=h1: h1; mf.get_ovlp = lambda *a, n=n: np.eye(n)
    mf._eri = ao2mo.restore(8, eri, n); mf.kernel(dm0=np.diag([2.0] * 5 + [0.0] * 5))
    cisd = ci.CISD(mf); cisd.verbose = 0; cisd.kernel()
    samplers["classical_cisd"] = ours(np.abs(cisd.to_fcivec(cisd.ci, n, ne)) ** 2)
    out = {}
    for name, probs in samplers.items():
        row = {"p_hf": float(probs[states.index(31 | (31 << n))])}
        for shots in (2000, 10000):
            for extend in (False, True):
                errors = []
                for s in (0, 1, 2):
                    rng = np.random.default_rng(s)
                    draws = rng.choice(len(states), size=shots, p=probs / probs.sum())
                    keys, counts = np.unique(draws, return_counts=True)
                    samples = Samples(counts={str(states[k]): int(c) for k, c in zip(keys, counts)},
                                      shots=shots, encoding_id="x", basis="computational",
                                      source="quantum_simulator", seed=s)
                    energy, diag = sqd_sector(data, sector, store, samples, recovery=False,
                                              with_ext=extend, seed=s)
                    errors.append((1000 * (energy - exact), diag["subspace_dimension"]))
                row[f"{shots}{'+ext' if extend else ''}"] = [round(float(np.mean([e for e, _ in errors])), 3),
                                                            int(np.mean([d for _, d in errors]))]
        out[name] = row
        print(label, name, json.dumps(row), flush=True)
    results[label] = out
(folder / "evaluation.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
