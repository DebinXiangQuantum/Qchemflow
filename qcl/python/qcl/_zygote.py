"""A pre-warmed fork server for QCL plan workers.

Starting ``python -m qcl._worker`` costs about 0.65 s per plan, mostly imports (NumPy,
SciPy, PySCF and the ChemFlow kernels). The zygote imports them once and forks a fresh
worker for every plan, so each run still gets its own process, CPU limit, process group
and resource accounting, but not its own interpreter start-up.

The client (``Plan.run``) uses the zygote only for plans the cold worker would run with
the caller's own environment: no GPU placement (CUDA must not be initialized before a
fork) and no deterministic thread pinning (thread counts are read when the numerical
libraries load). Every request carries the caller's current environment; a change to a
variable that libraries read at import restarts the zygote. ``QCL_WARM_WORKER=0``
disables it. If the zygote is busy or fails, the caller falls back to the cold worker.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

# Read by native libraries when they load, so a forked worker cannot change them.
IMPORT_TIME = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
               "PYTHONPATH", "PYTHONHASHSEED", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
               "PYSCF_MAX_MEMORY", "PYSCF_TMPDIR", "QCL_TRACE")
PRELOAD = ("numpy", "scipy", "scipy.sparse", "scipy.linalg", "pyscf", "pyscf.gto",
           "pyscf.scf", "pyscf.fci", "pyscf.cc", "pyscf.mp", "pyscf.ao2mo", "pyscf.mcscf",
           "pyscf.lo", "qcl._worker", "chemflow.impls.pyscf_impls", "chemflow.impls.sci",
           "chemflow.impls.sqd", "chemflow.impls.fci", "chemflow.impls.statevector",
           "chemflow.impls.lucj", "chemflow.impls.prepare", "chemflow.domains.paths",
           "chemflow.domains.reactivity", "chemflow.domains.materials")


def _key(environment: dict[str, str]) -> tuple:
    return tuple(environment.get(name) for name in IMPORT_TIME)


# ---- client -------------------------------------------------------------------------

class _Client:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._key: tuple | None = None
        self._ready = False
        self._broken = False

    def _stale(self) -> bool:
        return (self._process is None or self._process.poll() is not None or
                self._key != _key(os.environ))

    def _spawn(self) -> None:
        self.close()
        self._process = subprocess.Popen(
            [sys.executable, "-m", "qcl._zygote"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._key = _key(os.environ)
        self._ready = False

    def _ensure(self) -> None:
        if self._stale():
            self._spawn()
        if not self._ready:
            ready = self._process.stdout.readline()
            if json.loads(ready or "{}").get("ready") is not True:
                raise RuntimeError("QCL zygote did not start")
            self._ready = True

    def prestart(self) -> None:
        """Start the server in the background (it imports while the caller compiles)."""
        if self._broken or os.environ.get("QCL_WARM_WORKER", "1") == "0":
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._stale():
                self._spawn()
        except OSError:
            self._broken = True
        finally:
            self._lock.release()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()

    def run(self, payload: str, timeout_s: float) -> dict[str, Any] | None:
        """Run one worker request; ``None`` when the caller should use the cold worker."""
        if self._broken or os.environ.get("QCL_WARM_WORKER", "1") == "0":
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            self._ensure()
            request = {"payload": payload, "timeout_s": timeout_s,
                       "environ": dict(os.environ)}
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("QCL zygote exited")
            return json.loads(line)
        except (OSError, ValueError, RuntimeError):
            self._broken = True
            self.close()
            return None
        finally:
            self._lock.release()


_CLIENT = _Client()
atexit.register(_CLIENT.close)


def run(payload: str, timeout_s: float) -> dict[str, Any] | None:
    return _CLIENT.run(payload, timeout_s)


def prestart() -> None:
    _CLIENT.prestart()


# ---- server -------------------------------------------------------------------------

def _child(payload: str, environ: dict[str, str], out_fd: int, err_fd: int) -> None:
    """In the forked child: become the worker for one request, then exit."""
    import io

    os.setpgid(0, 0)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)
    os.environ.clear()
    os.environ.update(environ)
    sys.stdin = io.StringIO(payload)
    code = 0
    try:
        from qcl import _worker

        _worker.main()
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else 1
    except BaseException:  # noqa: BLE001  (the worker's own uncaught-error contract)
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


def _serve(request: dict[str, Any]) -> dict[str, Any]:
    directory = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
    out_fd, out_path = tempfile.mkstemp(prefix="qcl-worker-out-", dir=directory)
    err_fd, err_path = tempfile.mkstemp(prefix="qcl-worker-err-", dir=directory)
    try:
        pid = os.fork()
        if pid == 0:
            _child(request["payload"], request["environ"], out_fd, err_fd)
        deadline = time.monotonic() + float(request["timeout_s"])
        timed_out = False
        while True:
            done, status, usage = os.wait4(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > deadline:
                timed_out = True
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(pid, 0)
                break
            time.sleep(0.002)
        with open(out_path, encoding="utf-8", errors="replace") as stream:
            stdout = stream.read()
        with open(err_path, encoding="utf-8", errors="replace") as stream:
            stderr = stream.read()
        return {"returncode": (-os.WTERMSIG(status) if os.WIFSIGNALED(status)
                               else os.WEXITSTATUS(status)),
                "stdout": stdout, "stderr": stderr, "timed_out": timed_out,
                "cpu_s": usage.ru_utime + usage.ru_stime}
    finally:
        for fd, path in ((out_fd, out_path), (err_fd, err_path)):
            os.close(fd)
            os.unlink(path)


def main() -> None:
    from importlib import import_module

    from chemflow.backends.cuda_libs import prefer_wheel_cuda_libraries

    prefer_wheel_cuda_libraries()
    for name in PRELOAD:
        try:
            import_module(name)
        except ImportError:
            pass
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if line.strip():
            sys.stdout.write(json.dumps(_serve(json.loads(line))) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
