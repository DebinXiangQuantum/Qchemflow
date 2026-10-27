from pydantic import BaseModel

from chemflow.units import Quantity


class Molecule(BaseModel, frozen=True):
    atoms: list[str]
    charge: int = 0
    spin: int = 0
    isotopes: dict[int, int] = {}
    stereo_policy: str = "unspecified"
    protonation_policy: str = "explicit"


class Geometry(BaseModel, frozen=True):
    coordinates: list[list[float]]
    unit: str
    atom_order: list[str]
    conformer_id: str = "0"

    def to(self, unit: str) -> "Geometry":
        return self.model_copy(update={"coordinates": [
            [Quantity(value=value, unit=self.unit).to(unit).value for value in row]
            for row in self.coordinates], "unit": unit})
