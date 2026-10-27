"""Unit labels shared by QCL tracing and tensor execution."""

from collections.abc import Iterable


def product_unit(units: Iterable[str]) -> str:
    factors = [unit for unit in units if unit != "dimensionless"]
    if factors == ["hartree^1/2", "hartree^1/2"]:
        return "hartree"
    return "*".join(factors) if factors else "dimensionless"
