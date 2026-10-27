"""PySCF solvers backed by ChemFlow, without an agent or execution plan."""

import json
import time
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, fci

from chemflow.impls.sqd import sqd_sector
from chemflow.impls.statevector import sample_configurations
from chemflow.ops.molecular import prepare_sector_state
from chemflow.registry import Registry
from chemflow.runtime import ArtifactStore, BudgetLedger, ExecutionResult
from chemflow.runtime.budget import RESOURCE_DIMENSIONS, resource_defaults
from chemflow.types import CostRecord, Hamiltonian, Sector


class SQDSolver(fci.selected_ci.SelectedCI):
    """Sample LUCJ and solve its selected determinant space.

    ``spin`` follows PySCF's convention, 2S. Hardware defaults to ``resample='once'``:
    subsequent orbital iterations reuse bitstrings, not CI coefficients. Use a new
    solver for a different particle sector. ``always`` requests fresh sampling.
    Resource totals are cumulative over this solver's lifetime.
    """

    def __init__(self, sampler="statevector", shots=4096, seed=0, ansatz="lucj",
                 init="ccsd", n_batches=10, max_iter=5, spin=None, backend=None,
                 calibration=None, budget=None, resample=None, purpose="screen"):
        super().__init__()
        if sampler not in ("statevector", "aer_noisy", "gpu", "lqcloud"):
            raise ValueError("Unknown sampler")
        if purpose not in ("screen", "deploy", "production"):
            raise ValueError("Unknown execution purpose")
        if purpose == "production" and sampler != "lqcloud":
            raise ValueError("CF-E140: simulator is for verification only; production needs QPU")
        self.execution_role = "qpu" if sampler == "lqcloud" else "emulator"
        self.emulates = "sample_configurations" if sampler != "lqcloud" else None
        self.purpose = purpose
        if not isinstance(shots, int) or shots < 1 or n_batches < 1 or max_iter < 1:
            raise ValueError("shots, n_batches and max_iter must be positive integers")
        if ansatz not in ("hf", "lucj") or init not in ("zero", "ccsd", "random"):
            raise ValueError("Unsupported ansatz or initialization")
        self.resample = resample or ("once" if sampler == "lqcloud" else "always")
        if self.resample not in ("once", "always"):
            raise ValueError("resample must be 'once' or 'always'")
        self.sampler, self.shots, self.seed = sampler, shots, seed
        self.ansatz, self.init = ansatz, init
        self.n_batches, self.max_iter, self.spin = n_batches, max_iter, spin
        self.backend, self.calibration = backend, calibration
        if sampler == "lqcloud" and backend is None:
            from chemflow.backends.lqcloud import LqcloudBackend

            self.backend = LqcloudBackend()
        limits = dict(budget or {})
        self._budget = BudgetLedger(limits)
        allowed = set(RESOURCE_DIMENSIONS)
        if set(limits) - allowed or any(not np.isfinite(value) or value < 0
                                        for value in limits.values()):
            raise ValueError("Invalid resource budget")
        self._resources = dict.fromkeys((*sorted(allowed), "sim_shots"), 0)
        self._samples = None
        self._sector = None
        self._pending = None
        self._last_prediction = {}
        self._repetition = 0
        self._job_ids = []
        self.last_run = {}
        self.converged = False
        self.verbose = 0

    def _sample(self, store, prep):
        if self.purpose == "production" and self.sampler != "lqcloud":
            raise ValueError("CF-E140: production requires QPU sampling")
        if self.sampler != "lqcloud":
            self._last_prediction = {"sim_shots": self.shots}
            if "wall_s" in self._budget.limits:
                self._budget.spent["wall_s"] = (self._resources["wall_s"] +
                                                time.perf_counter() - self._call_started)
            self._budget.reserve(CostRecord(**resource_defaults(
                {"sim_shots": self.shots, "wall_s": None}, "quantum_simulator")))
        if self.sampler == "statevector":
            return ExecutionResult(sample_configurations(prep, self.shots, self.seed, store=store),
                                   cost={"sim_shots": self.shots})
        if self.sampler == "lqcloud":
            if self._pending is None:
                predicted = resource_defaults(self.backend.prediction(shots=self.shots), "qpu")
                self._pending = (store, prep, predicted)
            pending_store, pending_prep, predicted = self._pending
            self._last_prediction = predicted
            record_path = self._hardware_record_path(pending_prep)
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            if record.get("state") not in ("submitted", "completed"):
                if "wall_s" in self._budget.limits:
                    self._budget.spent["wall_s"] = (self._resources["wall_s"] +
                                                    time.perf_counter() - self._call_started)
                self._budget.reserve(CostRecord(**predicted))
            started = time.perf_counter()
            archive = record_path.parent / "archive"
            prior_archives = set(archive.glob(f"{record_path.stem}-*.json"))
            try:
                result = self.backend.sample(pending_store, pending_prep, self.shots,
                                             seed=self.seed, repetition=self._repetition)
            except Exception:
                archived = sorted(set(archive.glob(f"{record_path.stem}-*.json")) - prior_archives)
                if not record_path.exists() and archived:
                    failed = json.loads(archived[-1].read_text())
                    if failed.get("state") == "failed":
                        self._pending = None
                        self._repetition += 1
                        self._job_ids.append(failed["job_id"])
                        cost = self.backend._failure_cost(failed, self.shots)
                        self._record_cost(cost)
                raise
            result.cost["wall_s"] = time.perf_counter() - started
            self._pending = None
            self._repetition += 1
            return result
        registry = Registry()
        if self.sampler == "aer_noisy":
            from chemflow.backends.aer_noise import register_noisy_aer
            from chemflow.backends.calibration import load_calibration

            register_noisy_aer(registry, self.calibration or load_calibration())
            identifier = "aer_noisy_mq02"
        else:
            from chemflow.backends.gpu import register_gpu

            register_gpu(registry)
            identifier = "statevector_cupy"
        result = registry.implementations[identifier].adapter(
            store, prep, self.shots, seed=self.seed)
        result.cost["sim_shots"] = self.shots
        return result

    def _hardware_record_path(self, prep):
        """Locate the backend's durable record without writing or reserving it."""
        return self.backend.record_path(prep, self.shots, self._repetition)

    def _record_cost(self, cost):
        for dimension, amount in cost.items():
            if dimension in self._resources:
                previous = self._resources[dimension]
                self._resources[dimension] = (previous + amount if previous is not None
                                              and amount is not None else None)
        budget_cost = dict(cost)
        for dimension in self._budget.limits:
            if budget_cost.get(dimension) is None:
                budget_cost[dimension] = self._last_prediction.get(dimension)
        self._budget_overruns.update(self._budget.record(CostRecord(**budget_cost)))

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        if self.purpose == "production" and (self.sampler != "lqcloud" or
                self._samples is not None and self._samples.source != "qpu"):
            raise ValueError("CF-E140: production requires QPU samples, including reused samples")
        started = time.perf_counter()
        self._call_started = started
        self._budget_overruns = set()
        wall_before = self._resources["wall_s"]
        self.converged = False
        self.last_run = {"sampler": self.sampler, "device": (
            self._samples.device if self._samples is not None else
            f"lqcloud:{self.backend.backend}" if self.sampler == "lqcloud" else "local"),
            "shots": self.shots, "seed": self.seed, "energy": None,
            "n_determinants": 0, "relation": "variational_upper_bound",
            "job_ids": list(self._job_ids), "samples_reused": False}
        try:
            self._budget.reserve(CostRecord(**{
                **dict.fromkeys(RESOURCE_DIMENSIONS, 0), "wall_s": None,
                "gpu_s": None if self.sampler == "gpu" else 0}))
            return self._kernel(h1e, eri, norb, nelec, ci0=ci0, ecore=ecore, **kwargs)
        except Exception as error:
            self.last_run["error"] = str(error)
            raise
        finally:
            self._resources["wall_s"] = wall_before + time.perf_counter() - started
            if "wall_s" in self._budget.limits:
                self._budget.spent["wall_s"] = self._resources["wall_s"]
                if self._resources["wall_s"] > self._budget.limits["wall_s"]:
                    self._budget_overruns.add("wall_s")
            self.last_run["budget_overruns"] = sorted(self._budget_overruns)
            self.last_run["resources"] = dict(self._resources)
            self.last_run["job_ids"] = list(self._job_ids)

    def _kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        electrons = fci.direct_spin1._unpack_nelec(nelec, self.spin)
        sector = Sector(n_orb=norb, n_alpha=electrons[0], n_beta=electrons[1])
        if self._sector is not None and sector != self._sector and (
                self.resample == "once" or self._pending is not None):
            raise ValueError("Cannot reuse samples across different particle sectors")
        self._sector = sector
        store = ArtifactStore()
        hamiltonian = Hamiltonian(h1_ref=store.put(np.asarray(h1e)),
                                  h2_ref=store.put(ao2mo.restore(1, eri, norb)),
                                  space="active", n_orb=norb)
        data = SimpleNamespace(hamiltonian=hamiltonian, sector=sector)
        reused = self.resample == "once" and self._samples is not None
        if not reused:
            prep = prepare_sector_state(data, sector.model_dump(), store,
                                        ansatz=self.ansatz, init=self.init, seed=self.seed,
                                        materialize=(self.sampler != "lqcloud" or
                                                     self.purpose == "screen"))
            result = self._sample(store, prep)
            reused = self.sampler == "lqcloud" and result.cost.get("qpu_shots") == 0
            self._samples = result.value
            self.last_run["device"] = result.device
            self._job_ids.extend(job for job in result.job_ids if job not in self._job_ids)
            self._record_cost(result.cost)
        energy, diagnostics, vector = sqd_sector(
            data, sector, store, self._samples, seed=self.seed, n_batches=self.n_batches,
            max_iter=self.max_iter, with_ext=False, energy_tol=self.conv_tol,
            total_spin=None if self.spin is None else abs(self.spin) / 2,
            return_wavefunction=True, wavefunction_format="selected_ci")
        energy += ecore
        self._strs = vector._strs
        self.ci, self.eci, self.converged = vector, energy, True
        self.last_run = {"sampler": self.sampler, "device": self._samples.device,
                         "shots": self.shots, "seed": self.seed,
                         "n_determinants": diagnostics["subspace_dimension"],
                         "energy": energy, "relation": (
                             "approximate" if diagnostics["spin_completion_reason"] else
                             "variational_upper_bound"), "job_ids": list(self._job_ids),
                         "resources": dict(self._resources), "samples_reused": reused}
        return energy, vector


class GPUFCISolver(fci.direct_spin1.FCISolver):
    """Exact FCI using ChemFlow's CUDA contraction; never silently falls back."""

    execution_role = "classical"
    emulates = None

    def __init__(self, budget=None, **kwargs):
        from chemflow.backends.gpu import cuda_available

        if not cuda_available():
            raise RuntimeError("CUDA unavailable")
        super().__init__()
        limits = dict(budget or {})
        self._budget = BudgetLedger(limits)
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise TypeError(f"Unknown FCI option {name}")
            setattr(self, name, value)
        if self.nroots != 1:
            raise ValueError("GPUFCISolver supports only nroots=1")
        self.last_run = {}

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        import cupy as cp

        from chemflow.backends.gpu import FciKernel, fci_ground_state

        self._budget.reserve(CostRecord(**resource_defaults(
            {"wall_s": None, "gpu_s": None}, "gpu_numeric")))
        self.converged = False
        electrons = fci.direct_spin1._unpack_nelec(nelec, self.spin)
        device = cp.cuda.Device()
        device.synchronize()
        started = time.perf_counter()
        kernel = FciKernel(h1e, eri, norb, electrons, xp=cp)
        energy, _, diagnostics, vector = fci_ground_state(
            kernel, ecore, None if self.spin is None else abs(self.spin) / 2,
            tol=self.conv_tol, return_wavefunction=True)
        device.synchronize()
        elapsed = time.perf_counter() - started
        name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        self.ci, self.eci = vector, energy
        self.converged = diagnostics["residual"] < max(1e-8, self.conv_tol * 10)
        self.last_run = {"device": f"cuda:{device.id}:{name}", "gpu_s": elapsed,
                         "resources": resource_defaults({"gpu_s": elapsed, "wall_s": elapsed},
                                                         "gpu_numeric"), "energy": energy,
                         "relation": "equivalent" if self.converged else "variational_upper_bound"}
        self.last_run["budget_overruns"] = self._budget.record(CostRecord(
            **self.last_run["resources"]))
        return energy, vector
