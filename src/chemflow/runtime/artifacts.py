import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np


def write_text_artifact(text: str, suffix: str = ".txt") -> str:
    directory = Path(tempfile.gettempdir()) / "chemflow-text-artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(text.encode()).hexdigest() + suffix)
    path.write_text(text)
    return str(path)


def read_text_artifact(ref) -> str:
    """Read a text artifact path, including the content-addressed preparation artifacts."""
    return Path(ref).read_text()


class ArtifactStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.memory: dict[str, bytes] = {}
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

    def put(self, array: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        reference = hashlib.sha256(data).hexdigest()
        if self.path:
            (self.path / reference).write_bytes(data)
        else:
            self.memory[reference] = data
        return reference

    def get(self, reference: str, *, verify: bool = False) -> np.ndarray:
        data = (self.path / reference).read_bytes() if self.path else self.memory[reference]
        if verify and hashlib.sha256(data).hexdigest() != reference:
            raise ValueError("Artifact content does not match its SHA-256 reference")
        return np.load(io.BytesIO(data), allow_pickle=False)

    def has(self, reference: str) -> bool:
        return (self.path / reference).exists() if self.path else reference in self.memory

    def persist(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if self.path is not None and self.path != destination:
            for artifact in self.path.iterdir():
                if artifact.is_file():
                    (destination / artifact.name).write_bytes(artifact.read_bytes())
        for reference, data in self.memory.items():
            (destination / reference).write_bytes(data)
        self.path = destination
        self.memory.clear()
