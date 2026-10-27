"""Residual-based log–log cost intervals; unknown until three distinct sizes."""

from __future__ import annotations

import math

from chemflow.agent.memory import ResultsDB

FEATURES = {"sci": "determinants", "sqd": "subspace_dimension",
            "exact": "sector_dimension", "fci": "sector_dimension",
            "sample": "shots", "sampling": "shots"}


def fit_cost(observations: list[tuple[float, float]], size: float) -> dict | None:
    if size <= 0:
        raise ValueError("Target size must be positive")
    valid = [(math.log(x), math.log(y)) for x, y in observations if x > 0 and y > 0]
    if len(valid) < 3 or len({x for x, _ in valid}) < 3:
        return None
    count = len(valid)
    mean_x = sum(x for x, _ in valid) / count
    mean_y = sum(y for _, y in valid) / count
    spread = sum((x - mean_x) ** 2 for x, _ in valid)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in valid) / spread
    intercept = mean_y - slope * mean_x
    log_size = math.log(size)
    predicted = intercept + slope * log_size
    residual = math.sqrt(sum((y - intercept - slope * x) ** 2 for x, y in valid) /
                         (count - 2))
    half_width = 1.96 * residual * math.sqrt(1 + 1 / count +
                                             (log_size - mean_x) ** 2 / spread)
    return {"estimate": math.exp(predicted), "lower": math.exp(predicted - half_width),
            "upper": math.exp(predicted + half_width), "observations": count,
            "exponent": slope}


def estimate_cost(db: ResultsDB, template: str, target_size: float,
                  dimension: str = "wall_s") -> dict | None:
    feature = FEATURES.get(template)
    if feature is None:
        return None
    result = fit_cost(db.cost_observations(template, feature, dimension), target_size)
    return {**result, "feature": feature, "target_size": target_size} if result else None
