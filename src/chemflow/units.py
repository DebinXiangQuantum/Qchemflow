from __future__ import annotations

from pydantic import BaseModel
from scipy import constants

# SI factors from scipy.constants (CODATA as shipped with installed SciPy).
HARTREE = constants.physical_constants["Hartree energy"][0]
BOHR = constants.physical_constants["Bohr radius"][0]
_UNITS = {
    "hartree": ("energy", HARTREE), "eV": ("energy", constants.electron_volt),
    "kcal/mol": ("energy", 4184 / constants.Avogadro),
    "kJ/mol": ("energy", 1000 / constants.Avogadro),
    "cm^-1": ("energy", constants.h * constants.c * 100),
    "angstrom": ("length", 1e-10), "bohr": ("length", BOHR),
    "nm": ("length", 1e-9), "fs": ("time", 1e-15),
    "ps": ("time", 1e-12), "K": ("temperature", 1),
    "dimensionless": ("dimensionless", 1),
    "log10(mol/L)": ("dimensionless", 1),
}


class Quantity(BaseModel, frozen=True):
    value: float | list[float]
    unit: str

    @property
    def dimension(self) -> str:
        return _UNITS[self.unit][0]

    def model_post_init(self, __context: object, /) -> None:
        if self.unit not in _UNITS:
            raise ValueError(f"Unknown unit: {self.unit}")

    def to(self, unit: str) -> Quantity:
        if unit not in _UNITS or self.dimension != _UNITS[unit][0]:
            raise ValueError("Incompatible units")
        factor = _UNITS[self.unit][1] / _UNITS[unit][1]
        value = ([item * factor for item in self.value] if isinstance(self.value, list)
                 else self.value * factor)
        return Quantity(value=value, unit=unit)

    def __add__(self, other: Quantity) -> Quantity:
        if self.unit != other.unit or isinstance(self.value, list) or isinstance(other.value, list):
            raise ValueError("Explicitly convert scalar quantities to the same unit")
        return Quantity(value=self.value + other.value, unit=self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        if self.unit != other.unit or isinstance(self.value, list) or isinstance(other.value, list):
            raise ValueError("Explicitly convert scalar quantities to the same unit")
        return Quantity(value=self.value - other.value, unit=self.unit)


def Q(value: float | list[float], unit: str) -> Quantity:
    return Quantity(value=value, unit=unit)
