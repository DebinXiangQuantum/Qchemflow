"""Integrals-only PySCF objects stay in core however large the calling process is.

PySCF compares its memory budget with the process's current RSS. Once a long-running
process (an agent, a test session with GPU stacks loaded) passes the default 4000 MB,
CCSD switches to the out-of-core integral transformation, which needs a real basis and
fails for ChemFlow's integrals-only molecules.
"""

import pytest
from pyscf import gto, lib, mcscf, scf

from chemflow.plugins.pyscf import SQDSolver


def test_sqd_ccsd_seed_runs_in_a_process_past_the_memory_budget(monkeypatch):
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)).run()
    reference = mcscf.CASCI(mf, 2, 2).kernel()[0]
    monkeypatch.setattr(lib, "current_memory", lambda: (1.0e6, 0))  # 1 TB resident
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver = SQDSolver(sampler="statevector", shots=4096, seed=3)
    assert mc.kernel()[0] == pytest.approx(reference, abs=1e-8)
