"""Reports may only quote recorded numbers and may not claim what the screen did not show."""

from __future__ import annotations

import re

from chemflow.units import Q

ENERGY_NUMBER = re.compile(
    r"(?<![\w.])([+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s*"
    r"(hartree|mHa|Ha|Eh|eV|kcal/mol|kJ/mol)\b", re.IGNORECASE)
UNITS = {"ha": "hartree", "eh": "hartree", "hartree": "hartree", "ev": "eV",
         "kcal/mol": "kcal/mol", "kj/mol": "kJ/mol", "mha": "hartree"}
UNSUPPORTED = ("quantum advantage", "speedup", "certified")


class ReportLintError(ValueError):
    pass


def _energies(text: str, *, precision: bool = False) -> list:
    """Quoted energies in hartree; with ``precision``, (value, half last-digit step)."""
    found = []
    for number, unit in ENERGY_NUMBER.findall(text):
        scale = 0.001 if unit.lower() == "mha" else 1.0
        mantissa, _, exponent = number.lower().partition("e")
        decimals = len(mantissa.partition(".")[2])
        step = 0.5 * 10.0 ** (-decimals + (int(exponent) if exponent else 0)) * scale
        value = Q(float(number.replace("−", "-")) * scale, UNITS[unit.lower()])
        hartree = value.to("hartree").value
        factor = abs(Q(1.0, UNITS[unit.lower()]).to("hartree").value)
        found.append((hartree, step * factor) if precision else hartree)
    return found


def lint_numbers(text: str, headlines: list[dict]) -> None:
    """Every quoted energy matches a recorded output (to its printed precision)."""
    for claim in UNSUPPORTED:
        if re.search(r"\b" + claim + r"\b", text, re.IGNORECASE):
            raise ReportLintError(f"Unsupported claim: {claim}")
    recorded = []
    for headline in headlines:
        value = headline.get("value")
        if isinstance(value, str):
            recorded.extend(_energies(value))
        recorded.extend(headline.get("values_hartree") or [])
    for quoted, step in _energies(text, precision=True):
        tolerance = step + 1e-12 * max(1.0, abs(quoted))
        if not any(abs(quoted - value) <= tolerance for value in recorded):
            raise ReportLintError(f"Quoted energy {quoted:.10g} hartree is not a recorded result")
