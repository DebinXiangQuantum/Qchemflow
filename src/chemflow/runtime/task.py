from collections.abc import Mapping
from types import MappingProxyType

from chemflow.ids import content_hash
from chemflow.ir.schema import Program


class TaskRecord:
    __slots__ = ("_gate_hashes", "_program_hash")

    def __init__(self, program_hash: str, gate_hashes: Mapping[str, str]):
        object.__setattr__(self, "_program_hash", program_hash)
        object.__setattr__(self, "_gate_hashes", MappingProxyType(dict(gate_hashes)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TaskRecord is immutable")

    @property
    def program_hash(self) -> str:
        return self._program_hash

    @property
    def gate_hashes(self) -> Mapping[str, str]:
        return self._gate_hashes


def freeze_task(program: Program) -> TaskRecord:
    return TaskRecord(program.hash, {str(index): content_hash(gate.payload())
                                     for index, gate in enumerate(program.gates)})
