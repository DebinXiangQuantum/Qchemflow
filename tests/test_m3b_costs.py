import pytest

from chemflow.agent.costs import estimate_cost, fit_cost
from chemflow.agent.memory import ResultsDB


def test_power_law_and_unknown():
    db = ResultsDB()
    for size in (10, 20):
        db.add_cost("sci", "determinants", size, "wall_s", 3 * size ** 2)
    assert estimate_cost(db, "sci", 80) is None
    db.add_cost("sci", "determinants", 40, "wall_s", 3 * 40 ** 2)
    fit = estimate_cost(db, "sci", 80)
    assert fit["estimate"] == pytest.approx(3 * 80 ** 2)
    assert fit["exponent"] == pytest.approx(2)


def test_intervals_widen_when_extrapolating():
    observations = [(10, 101), (20, 370), (30, 950), (40, 1500)]
    at_center = fit_cost(observations, 25)
    far = fit_cost(observations, 1000)
    assert far["upper"] / far["lower"] > at_center["upper"] / at_center["lower"]
