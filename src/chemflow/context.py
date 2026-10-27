from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, Field


class Context(BaseModel, frozen=True):
    keys: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: object, /) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))

    def model_dump(self, **kwargs: object) -> dict:
        return {"keys": dict(self.keys)}


def same(left: Context, right: Context, *keys: str) -> bool:
    return all(key in left.keys and key in right.keys and
               left.keys[key] == right.keys[key] for key in keys)


def inherit(parent: Context, **updates: str) -> Context:
    return Context(keys={**parent.keys, **updates})


def compatible(left: Mapping[str, str], right: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    return all(key in left and key in right and left[key] == right[key] for key in keys)
