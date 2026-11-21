
import numpy as np
import pytest

from chemflow.quantum.pauli import allocate_shots, estimate, qwc_groups


def test_covariance_and_allocation():
    terms = {"ZI": 1., "IZ": 1., "II": 3.}
    groups = qwc_groups(terms)
    result = estimate(terms, groups, [{"00": 50, "11": 50}])
    assert result["energy"] == 3
    assert result["stat_err"] == pytest.approx(np.sqrt(4/99))
    assert result["per_group_stat_err"] == [result["stat_err"]]
    assert allocate_shots(11, [1, 2, 3]) == [4, 4, 3]
    assert allocate_shots(10, [1, 2], [1, 3]) == [3, 7]


@pytest.mark.parametrize("ansatz", ["uccsd", "lucj"])
def test_circuit_matches_sector_ansatz(ansatz):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from chemflow.backends.circuits import build_circuit
    from chemflow.impls.ansatz import uccsd_pool, uccsd_state
    from chemflow.impls.fci import sector_states
    from chemflow.impls.lucj import LucjParams, lucj_state
    from chemflow.types import CircuitSpec, Sector

    sector = Sector(n_orb=3, n_alpha=1, n_beta=1)
    generators = uccsd_pool(sector)[1]
    size = len(generators) if ansatz == "uccsd" else LucjParams.zeros(3).size
    values = np.random.default_rng(9).normal(0, .2, size)
    expected = (uccsd_state(sector, values, generators) if ansatz == "uccsd" else
                lucj_state(sector, [LucjParams.from_vector(values, 3)]))
    circuit = build_circuit(CircuitSpec(ansatz=ansatz, n_orb=3, n_alpha=1, n_beta=1,
                                       layers=(tuple(values),)))
    actual = Statevector.from_instruction(circuit).data[sector_states(sector)]
    assert abs(np.vdot(expected, actual)) == pytest.approx(1, abs=1e-10)
