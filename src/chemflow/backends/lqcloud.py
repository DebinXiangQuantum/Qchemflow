"""Budgeted QPU execution with durable reserve-before-submit records."""

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

from chemflow.backends.calibration import best_layout
from chemflow.backends.circuits import build_circuit, compile_native, memory_to_index, to_lqcloud
from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ExecutionResult
from chemflow.types import BackendKind, Samples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pricing:
    currency: str
    per_task: float
    per_shot: float
    max_billing_multiplier: float
    min_shots: int
    max_shots: int
    source: str


def load_pricing(backend: str = "MQ02") -> Pricing:
    data = json.loads(files("chemflow.backends.data").joinpath("lqcloud_pricing.json").read_text())
    return Pricing(**data[backend])


def load_resources(backend: str = "MQ02") -> dict:
    data = json.loads(files("chemflow.backends.data").joinpath("lqcloud_resources.json").read_text())
    return data[backend]


def _predict_time(resources: dict, shots: int, dimension: str) -> float:
    table = resources["table"]
    for lower, upper in pairwise(table):
        if lower["shots"] <= shots <= upper["shots"]:
            fraction = (shots - lower["shots"]) / (upper["shots"] - lower["shots"])
            return lower[dimension] + fraction * (upper[dimension] - lower[dimension])
    anchor = table[0] if shots < table[0]["shots"] else table[-1]
    fit = resources["device_linear_fit"]
    fitted = fit["intercept_s"] + fit["seconds_per_shot"] * shots
    anchor_fit = fit["intercept_s"] + fit["seconds_per_shot"] * anchor["shots"]
    return fitted * anchor[dimension] / anchor_fit


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _default_provider(key: str):
    import lqcloud

    return lqcloud.LQCloudProvider(api_key=key, interactive=False, store=False)


def _default_job(job_id: str, provider, backend_name: str):
    from lqcloud.job.job import Job

    return Job(job_id, provider, backend_name=backend_name)


class LqcloudBackend:
    def __init__(self, *, backend: str = "MQ02", state_dir: Path | None = None,
                 provider_factory=None, job_factory=None, pricing: Pricing | None = None,
                 api_key_env: str = "LQCLOUD_API_KEY", timeout_s: int = 900,
                 max_billing_multiplier: float | None = None):
        self.backend = backend
        self.state_dir = Path(state_dir) if state_dir is not None else Path.home() / ".chemflow/qpu" / backend
        self.provider_factory = provider_factory or _default_provider
        self.job_factory = job_factory or _default_job
        self.pricing = pricing or load_pricing(backend)
        self.resources = load_resources(backend)
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_billing_multiplier = max_billing_multiplier
        self.identifier = "lqcloud_mq02"

    def prediction(self, shots: int | None = None, **kwargs) -> dict:
        def predict(dimension):
            return None if shots is None else _predict_time(self.resources, shots, dimension)

        return {"qpu_shots": shots, "qpu_s": predict("device_p90_s"),
                "wall_s": predict("wall_p90_s"), "diagnostics": {
                    "qpu_s_p99": predict("device_p99_s"),
                    "qpu_s_max_observed": predict("device_max_s")}}

    def record_path(self, prep, shots: int, repetition: int = 0) -> Path:
        key = content_hash({"backend": self.backend, "impl_id": self.identifier,
                            "prep": prep.model_dump(mode="json"), "shots": shots,
                            "repetition": repetition})
        return self.state_dir / f"{key}.json"

    def _archive(self, path: Path) -> None:
        archive = self.state_dir / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        path.rename(archive / f"{path.stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json")

    def reconcile(self, key: str, *, abandon: bool = False, job_id: str | None = None) -> dict:
        if abandon == bool(job_id) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Choose exactly one reconciliation action and a valid key")
        path = self.state_dir / f"{key}.json"
        record = json.loads(path.read_text())
        if abandon:
            record.update(state="abandoned", updated=_now())
            _write(path, record)
            self._archive(path)
            return record
        if record["state"] == "completed":
            raise ValueError("Cannot attach a completed record")
        record.update(job_id=job_id, state="submitted", updated=_now())
        _write(path, record)
        return record

    def sample(self, store, prep, shots: int, seed: int = 0, repetition: int = 0, **kwargs):
        if prep.circuit is None:
            raise AdapterError("CF-E080", None, "State has no hardware circuit",
                               "Prepare HF or LUCJ", nothing_consumed=True)
        if not self.pricing.min_shots <= shots <= self.pricing.max_shots:
            raise AdapterError("CF-E080", None, "Shots outside supported range",
                               "Choose supported shots",
                               nothing_consumed=True)
        path = self.record_path(prep, shots, repetition)
        key = path.stem
        if path.exists():
            record = json.loads(path.read_text())
            if record["state"] == "completed":
                return self._output(prep, shots, seed, record, charged=False)
            if record["state"] in ("rejected", "failed"):
                self._archive(path)
                record = None
            elif not record.get("job_id"):
                raise AdapterError("CF-E070", None, "Unresolved QPU submission",
                                   "Reconcile the reservation before retrying")
        else:
            record = None
        key_value = os.environ.get(self.api_key_env)
        if not key_value:
            raise AdapterError("CF-E080", None, "Missing QPU credential",
                               f"Set {self.api_key_env}", nothing_consumed=True)
        provider = self.provider_factory(key_value)
        if record is not None and record.get("job_id"):
            job = self.job_factory(record["job_id"], provider, backend_name=self.backend)
        else:
            live = provider.get_backend(self.backend)
            config = live.config
            if config.get("status") != "active":
                raise AdapterError("CF-E080", None, "Device is inactive",
                                   "Review live configuration", nothing_consumed=True)
            if self.max_billing_multiplier is not None:
                multiplier = config.get("billing_multiplier")
                if (type(multiplier) not in (int, float) or
                        not math.isfinite(multiplier) or multiplier < 0 or
                        multiplier > self.max_billing_multiplier):
                    raise AdapterError("CF-E080", None, "Live QPU price exceeds the QCL limit",
                                       "Review the current tariff and plan budget",
                                       nothing_consumed=True)
            native = compile_native(build_circuit(prep.circuit))
            layout = best_layout(config, native.num_qubits)
            program = to_lqcloud(native)
            record = {"key": key, "state": "prepared", "backend": self.backend,
                      "impl_id": self.identifier, "shots": shots, "layout": layout,
                      "circuit_hash": content_hash(str(native)),
                      "predicted_cost": self.prediction(shots),
                      "diagnostics": {name: config[name] for name in (
                          "qpu_idle_eta_seconds", "queue_position") if name in config},
                      "currency": self.pricing.currency, "created": _now(), "updated": _now()}
            self.state_dir.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x") as stream:
                    json.dump(record, stream, indent=2)
            except FileExistsError as exc:
                raise AdapterError("CF-E070", None, "Unresolved concurrent reservation",
                                   "Reconcile the QPU ledger reservation") from exc
            try:
                job = live.run(program, shots=shots, result_format="memory",
                               initial_layout=layout, readout_correction=False)
            except Exception as exc:
                if re.search(r"HTTP (400|401|402|403|413|422)\b", str(exc)):
                    record.update(state="rejected", updated=_now(), error=str(exc))
                    _write(path, record)
                raise AdapterError("CF-E080", None, "QPU submission failed",
                                   "Inspect and reconcile the reservation",
                                   nothing_consumed=record["state"] == "rejected") from exc
            record.update(job_id=job.job_id, state="submitted", updated=_now())
            queue_position = getattr(job, "_queue_position", None)
            if queue_position is not None:
                record["diagnostics"]["queue_position"] = queue_position
            _write(path, record)
        job._params_resubmit = None
        status = None
        try:
            result = job.result(timeout=self.timeout_s, wait=3, verbose=False)
            status = job.status_info()
            record.setdefault("diagnostics", {}).update({name: status[name] for name in (
                "qpu_idle_eta_seconds", "queue_position") if status.get(name) is not None})
            unsuccessful = not result.success
            if unsuccessful:
                record["unsuccessful_results"] = record.get("unsuccessful_results", 0) + 1
            if unsuccessful or status.get("error_type") or status.get("status", "").lower() in (
                    "failed", "error", "cancelled"):
                raise RuntimeError("QPU job failed")
            memory = list(result.data["memory"])
            counts = Counter(str(memory_to_index(bits, q0_left=True)) for bits in memory)
            reported_cost = status.get("cost")
            cost = reported_cost
            cost_source = "platform" if reported_cost is not None else "unavailable"
            record.update(state="completed", memory=memory, counts=dict(counts),
                          status_info=status, cost=cost, cost_source=cost_source,
                          currency=status.get("currency", self.pricing.currency),
                          charge_recorded=True, updated=_now())
            _write(path, record)
        except Exception as exc:
            if status is None:
                try:
                    status = job.status_info()
                except Exception:
                    logger.exception("QPU status unavailable for job %s", record["job_id"])
            record.update(error=str(exc), updated=_now())
            if status is not None:
                record["status_info"] = status
                if status.get("cost") is not None:
                    record["cost"] = status["cost"]
                    record["cost_source"] = "platform"
                    record["currency"] = status.get("currency", self.pricing.currency)
                if status.get("status", "").lower() in ("failed", "error", "cancelled") or (
                        status.get("error_type") is not None) or (
                        record.get("unsuccessful_results", 0) >= 2):
                    record["state"] = "failed"
                    _write(path, record)
                    self._archive(path)
                    raise AdapterError("CF-E080", None, "QPU job failed on the platform",
                                       "Review the archived failure before resubmitting",
                                       incurred_cost=self._failure_cost(record, shots),
                                       job_ids=(record["job_id"],)) from exc
            _write(path, record)
            raise AdapterError("CF-E080", None, "QPU result unavailable",
                               "Retry by re-attaching the recorded job ID",
                               incurred_cost=self._failure_cost(record, shots),
                               job_ids=(record["job_id"],)) from exc
        return self._output(prep, shots, seed, record, charged=True)

    def _failure_cost(self, record, shots):
        status = record.get("status_info", {})
        predicted = record["predicted_cost"]
        return {**predicted,
                "qpu_shots": status.get("shots") if status.get("shots") is not None else shots,
                "qpu_s": (status["execution_time"] if status.get("execution_time") is not None
                          else predicted["qpu_s"])}

    def _output(self, prep, shots: int, seed: int, record: dict, *, charged: bool):
        device = f"lqcloud:{self.backend}"
        job_id = record["job_id"]
        diagnostics = {key: record[key] for key in ("cost_source", "charge_above_prediction")
                       if key in record}
        if record.get("cost") is not None:
            diagnostics["billing"] = {"cost": record["cost"],
                                      "currency": record["currency"],
                                      "source": record.get("cost_source", "unavailable")}
        diagnostics.update(record.get("diagnostics", {}))
        status = record.get("status_info", {})
        usage = {"qpu_shots": status.get("shots") if status.get("shots") is not None else shots,
                 "qpu_s": status.get("execution_time")}
        cost = {dimension: usage[dimension] if charged else 0
                for dimension in ("qpu_shots", "qpu_s")}
        samples = Samples(counts=record["counts"], shots=shots, encoding_id=prep.encoding_id,
                          basis="computational", source="qpu", seed=seed, device=device,
                          job_ids=(job_id,))
        return ExecutionResult(samples, device=device, cost=cost, job_ids=(job_id,),
                               diagnostics=diagnostics)


def register_lqcloud(registry, backend: LqcloudBackend,
                     identifier: str = "lqcloud_mq02") -> None:
    backend.identifier = identifier
    registry.register(ImplementationSpec(id=identifier, version="1", target="sample_configurations",
        backend_kind=BackendKind.qpu, effects="random", adapter=backend.sample,
        params=frozenset({"shots", "seed", "repetition"}),
        resource_model=backend.prediction))
