"""JSON boundary to the optional conda chemistry environment."""

import json
import os
import subprocess
from pathlib import Path

from chemflow.diagnostics import AdapterError

TIMEOUT_S = 1800
INSTALL_HINT = ("Install the binding side environment (conda create command in "
                "docs/app-e-binding.md) and set CHEMFLOW_TOOLS_PYTHON to its bin/python.")


def tools_python():
    return Path(os.environ.get("CHEMFLOW_TOOLS_PYTHON", Path(__file__).resolve().parents[4] /
                               "chemflow-tools-env/bin/python"))


def call_tools(operation, payload, timeout=TIMEOUT_S):
    python = tools_python()
    environment = {**os.environ, "PATH": f"{python.parent}:{os.environ.get('PATH', '')}",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    environment.pop("DYLD_LIBRARY_PATH", None)
    try:
        completed = subprocess.run(
            [str(python), str(Path(__file__).with_name("_tools_worker.py"))],
            input=json.dumps({"operation": operation, "payload": payload}, allow_nan=False),
            capture_output=True, text=True, timeout=timeout, env=environment, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        tail = getattr(exc, "stderr", None) or "<empty>"
        raise AdapterError("CF-E080", None,
                           f"Chemistry worker failed: {exc}; stderr: {tail}", INSTALL_HINT) from exc
    try:
        response = json.loads(completed.stdout)
        if completed.returncode or "error" in response:
            raise ValueError(response.get("error", f"exit {completed.returncode}"))
        return response["result"]
    except (ValueError, KeyError, TypeError) as exc:
        raise AdapterError("CF-E080", None,
                           f"Chemistry worker: {exc}; stderr: {completed.stderr[-8000:]}",
                           INSTALL_HINT) from exc
