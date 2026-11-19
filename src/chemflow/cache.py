"""Where ChemFlow looks for large optional data it does not ship.

Model weights and isolated tool installations (MACE-MP-0, AiZynthFinder) live
outside the repository. Their default home is ``$CHEMFLOW_CACHE_DIR``, falling
back to ``$XDG_CACHE_HOME/chemflow`` and then ``~/.cache/chemflow``. Each item
can still be pointed elsewhere with its own variable, such as
``CHEMFLOW_MACE_MODEL`` or ``CHEMFLOW_AIZYNTH_HOME``.
"""

import os
from pathlib import Path


def cache_dir() -> Path:
    """Return the root directory for optional ChemFlow data."""
    explicit = os.environ.get("CHEMFLOW_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "chemflow"
