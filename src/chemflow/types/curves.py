"""JSON dictionary results with the same provenance protocol as scalar estimates."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from chemflow.context import Context
from chemflow.relations import TargetRelation
from chemflow.types.results import CostRecord, Provenance


class CurveSchema(BaseModel, extra="forbid"):
    values_hartree: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=1)
    coordinates: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=1)
    kind: Literal["energy_curve"] = "energy_curve"
    relation: TargetRelation = TargetRelation.approximate
    context: Context
    provenance: Provenance
    cost: CostRecord = Field(default_factory=CostRecord)
    diagnostics: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def same_size(self):
        if len(self.values_hartree) != len(self.coordinates):
            raise ValueError("Curve coordinates and energies must have the same length")
        return self


class EnergyCurve(dict):
    """A plain JSON-compatible mapping, validated on construction and copying."""

    def __init__(self, **data):
        super().__init__(CurveSchema.model_validate(data).model_dump(
            mode="json", fallback=lambda value: dict(value), warnings=False))

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(name)
        if name == "relation":
            return TargetRelation(self[name])
        models = {"context": Context, "provenance": Provenance, "cost": CostRecord}
        return models[name].model_validate(self[name]) if name in models else self[name]

    def model_dump(self, **kwargs):
        return dict(self)

    def model_copy(self, *, update=None, deep=False):
        return EnergyCurve(**{**self, **(update or {})})

    @classmethod
    def model_validate(cls, data):
        return cls(**data)


def curve_error(value, reference):
    if (not isinstance(value, EnergyCurve) or not isinstance(reference, EnergyCurve)
            or value.coordinates != reference.coordinates
            or value.context.keys != reference.context.keys):
        raise ValueError("Curve comparison requires identical coordinates and target identity")
    return max(abs(left - right) for left, right in
               zip(value.values_hartree, reference.values_hartree, strict=True))
