from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from chemflow.impls.ansatz import generalized_pool, uccsd_pool, uccsd_state
from chemflow.impls.fci import spin_square
from chemflow.impls.sector import SectorEngine, hf_vector
from chemflow.types import Sector


@dataclass
class ADAPTResult:
    energy: float
    params: np.ndarray
    s2: float
    n_gradient_evaluations: int
    n_operators: int
    selected_operators: list[str]
    state: np.ndarray


def adapt_vqe(engine: SectorEngine, sector: Sector, threshold: float = 1e-3,
              max_operators: int = 40, maxiter: int = 150,
              spin_adapted: bool = False, **kwargs) -> ADAPTResult:
    names, pool = uccsd_pool(sector, True) if spin_adapted else generalized_pool(sector)
    selected: list[int] = []
    parameters = np.zeros(0)
    gradients_evaluated = 0
    state = hf_vector(sector)
    for _ in range(max_operators):
        h_state = engine.apply(state)
        gradients = [2 * np.real(np.vdot(h_state, generator @ state)) for generator in pool]
        gradients_evaluated += len(pool)
        available = [index for index in range(len(pool)) if index not in selected]
        if not available:
            break
        index = max(available, key=lambda candidate: abs(gradients[candidate]))
        if abs(gradients[index]) < threshold:
            break
        selected.append(index)
        generators = [pool[item] for item in selected]

        def objective(values, selected_generators=generators):
            return engine.energy(uccsd_state(sector, values, selected_generators))

        result = minimize(objective, np.r_[parameters, 0.0], method="L-BFGS-B",
                          options={"maxiter": maxiter, "ftol": 1e-11})
        parameters = result.x
        state = uccsd_state(sector, parameters, generators)
    return ADAPTResult(engine.energy(state), parameters, spin_square(state, sector),
                       gradients_evaluated, len(selected), [names[item] for item in selected],
                       state)
