"""Serialized, process-local ownership of an isolated MACE calculator."""

import atexit
import json
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

import numpy as np

from chemflow.diagnostics import AdapterError

TIMEOUT_S = 300
_lock = threading.Lock()
_worker = None


class MaceWorker:
    def __init__(self, model_path, model_sha256):
        self.key = (str(model_path), model_sha256)
        self.stderr_tail = bytearray()
        self.responses = Queue()
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "chemflow.domains._mace_worker", str(model_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise AdapterError("CF-E080", None, f"MACE worker could not start: {exc}; stderr tail: <empty>",
                               "Check the Python executable and MACE installation") from exc
        self.readers = [threading.Thread(target=target, daemon=True)
                        for target in (self._read_stdout, self._read_stderr)]
        for reader in self.readers:
            reader.start()
        if self._receive() != {"ready": True}:
            self._fail("invalid startup response")
        self.startup_s = time.perf_counter() - started

    def _read_stdout(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put(b"")

    def _read_stderr(self):
        while chunk := self.process.stderr.read1(4096):
            self.stderr_tail.extend(chunk)
            del self.stderr_tail[:-8000]

    def _receive(self):
        try:
            line = self.responses.get(timeout=TIMEOUT_S)
        except Empty:
            self._fail(f"response timed out after {TIMEOUT_S} seconds")
        if not line:
            self._fail("closed its output unexpectedly")
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            self._fail("returned invalid JSON")

    def _fail(self, message):
        self.close()
        tail = self.stderr_tail.decode("utf-8", errors="replace")
        raise AdapterError("CF-E080", None,
                           f"MACE worker {message} (exit {self.process.returncode}); "
                           f"stderr tail: {tail or '<empty>'}",
                           "Check the local weights and MACE worker installation")

    def evaluate(self, symbols, positions, charge, spin):
        request = json.dumps({"symbols": symbols, "positions": positions.tolist(),
                              "charge": charge, "spin": spin}, allow_nan=False)
        try:
            self.process.stdin.write((request + "\n").encode())
            self.process.stdin.flush()
        except (OSError, ValueError):
            self._fail("request could not be sent")
        response = self._receive()
        try:
            energy = float(response["energy"])
            gradient = np.asarray(response["gradient"], dtype=float)
            valid = (np.isfinite(energy) and gradient.shape == (len(symbols), 3)
                     and np.isfinite(gradient).all())
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self._fail("returned invalid energy or gradient")
        return energy, gradient

    def close(self):
        if not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for reader in self.readers:
            reader.join()
        self.process.stdout.close()
        self.process.stderr.close()


def evaluate(model_path, model_sha256, symbols, positions, charge=0, spin=0):
    global _worker
    with _lock:
        key = (str(model_path), model_sha256)
        if _worker is not None and _worker.key != key:
            _worker.close()
            _worker = None
        if _worker is None:
            _worker = MaceWorker(model_path, model_sha256)
        try:
            return _worker.evaluate(symbols, positions, charge, spin)
        except AdapterError:
            _worker = None
            raise


def shutdown():
    global _worker
    with _lock:
        if _worker is not None:
            _worker.close()
            _worker = None


atexit.register(shutdown)
