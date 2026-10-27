"""Array-module FCI contraction and optional CUDA implementations."""

import time

import numpy as np
from pyscf import ao2mo, fci, lib

from chemflow.diagnostics import AdapterError
from chemflow.impls.fci import sector_states
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ExecutionResult
from chemflow.types import BackendKind, Samples


class FciKernel:
    def __init__(self, h1, eri, norb, nelec, xp=np):
        self.xp, self.norb, self.nelec = xp, norb, nelec
        self.h1 = np.asarray(h1)
        self.eri = ao2mo.restore(1, eri, norb)
        absorbed = fci.direct_spin1.absorb_h1e(self.h1, self.eri, norb, nelec, 0.5)
        self.h2 = xp.asarray(ao2mo.restore(1, absorbed, norb).reshape(norb * norb, -1))
        self.na = fci.cistring.num_strings(norb, nelec[0])
        self.nb = fci.cistring.num_strings(norb, nelec[1])
        self.links = [self._tables(fci.cistring.gen_linkstr_index(range(norb), electrons), count)
                      for electrons, count in ((nelec[0], self.na), (nelec[1], self.nb))]
        self.rows_per_block = None

    def _tables(self, link, count):
        orbital, hole, target, sign = (link[:, :, index] for index in range(4))
        source = np.broadcast_to(np.arange(count)[:, None], orbital.shape)
        xp = self.xp
        return {
            "scatter": xp.asarray(((orbital * self.norb + hole) * count + target).ravel()),
            "source": xp.asarray(source.ravel()),
            "sign": xp.asarray(sign.ravel().astype(float)),
            "gather": xp.asarray((hole * self.norb + orbital) * count + target),
            "gsign": xp.asarray(sign.astype(float)),
        }

    def hdiag(self):
        return fci.direct_spin1.make_hdiag(self.h1, self.eri, self.norb, self.nelec)

    def sigma(self, vector, rows_per_block=None):
        xp, na, nb = self.xp, self.na, self.nb
        n2 = self.norb * self.norb
        coefficients = xp.asarray(vector).reshape(na, nb)
        result = xp.zeros((na, nb))
        alpha, beta = self.links
        if rows_per_block is None:
            free = xp.cuda.Device().mem_info[0] if xp.__name__ == "cupy" else 512 << 20
            per_row = n2 * nb * 8 * 4 + beta["gather"].shape[1] * nb * 8
            rows_per_block = max(1, min(na, int(0.6 * free // per_row)))
        if rows_per_block < 1:
            raise ValueError("rows_per_block must be positive")
        self.rows_per_block = rows_per_block
        alpha_source = alpha["source"].reshape(na, -1)
        alpha_destination = alpha["scatter"].reshape(na, -1)
        alpha_sign = alpha["sign"].reshape(na, -1)
        for start in range(0, na, rows_per_block):
            rows = slice(start, min(na, start + rows_per_block))
            count = rows.stop - rows.start
            intermediate = xp.zeros((n2, count, nb))
            targets = alpha["scatter"] % na
            inside = (targets >= start) & (targets < rows.stop)
            pairs = alpha["scatter"][inside] // na
            intermediate[pairs, targets[inside] - start] = (
                alpha["sign"][inside][:, None] * coefficients[alpha["source"][inside]])
            beta_intermediate = xp.zeros((n2 * nb, count))
            beta_intermediate[beta["scatter"]] = (
                beta["sign"][:, None] * coefficients[rows].T[beta["source"]])
            intermediate += beta_intermediate.reshape(n2, nb, count).transpose(0, 2, 1)
            contracted = (self.h2 @ intermediate.reshape(n2, -1)).reshape(n2, count, nb)
            beta_view = contracted.transpose(0, 2, 1).reshape(n2 * nb, count)
            result[rows] += xp.einsum("sl,slk->sk", beta["gsign"],
                                      beta_view[beta["gather"]]).T
            sources = alpha_source[rows].ravel() - start
            destinations = alpha_destination[rows]
            contributions = (alpha_sign[rows].ravel()[:, None] *
                             contracted[(destinations // na).ravel(), sources])
            if xp.__name__ == "cupy":
                xp.add.at(result, (destinations % na).ravel(), contributions)
            else:
                np.add.at(result, (destinations % na).ravel(), contributions)
        return result.ravel()


def fci_density_matrix(vector, norb, nelec, order, xp=np):
    if order not in (1, 2):
        raise ValueError("CI density order must be 1 or 2")
    sizes = [fci.cistring.num_strings(norb, count) for count in nelec]
    ci = xp.asarray(vector).reshape(sizes)
    excitations = xp.zeros((norb * norb, *sizes), dtype=ci.dtype)
    flat = excitations.ravel()
    for spin, size in enumerate(sizes):
        links = fci.cistring.gen_linkstr_index(range(norb), nelec[spin])
        source = np.repeat(np.arange(size), links.shape[1])
        target = links[:, :, 2].ravel()
        pair = (links[:, :, 0] * norb + links[:, :, 1]).ravel()
        signs = xp.asarray(links[:, :, 3].ravel(), dtype=ci.dtype)
        if spin == 0:
            positions = ((pair[:, None] * sizes[0] + target[:, None]) * sizes[1] +
                         np.arange(sizes[1])[None, :]).ravel()
            amplitudes = (signs[:, None] * ci[source, :]).ravel()
        else:
            positions = ((pair[:, None] * sizes[0] +
                          np.arange(sizes[0])[None, :]) * sizes[1] +
                         target[:, None]).ravel()
            amplitudes = (signs[:, None] * ci[:, source].T).ravel()
        xp.add.at(flat, xp.asarray(positions), amplitudes)
    transformed = excitations.reshape(norb, norb, -1)
    density1 = xp.einsum("i,pqi->pq", ci.ravel(), transformed)
    if order == 1:
        return density1
    density2 = xp.einsum("qpi,rsi->pqrs", transformed, transformed,
                         optimize=True)
    for q in range(norb):
        density2[:, q, q, :] -= density1
    return density2


def _gpu_resident_ground_state(kernel, ecore, total_spin, tol, max_roots,
                               return_wavefunction, residual_tol):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, eigsh

    dimension = kernel.na * kernel.nb
    calls = 0

    def apply(vector):
        nonlocal calls
        calls += 1
        return kernel.sigma(vector)

    def dense_roots():
        basis = cp.eye(dimension, dtype=cp.float64)
        matrix = cp.column_stack([apply(basis[:, index]) for index in range(dimension)])
        hermitian_error = float(cp.max(cp.abs(matrix - matrix.T)))
        if hermitian_error > 1e-8:
            raise AdapterError("CF-E020", None, "GPU CI Hamiltonian is not symmetric",
                               "Check the integral tensor and sector")
        return cp.linalg.eigh(matrix)

    examined = 0
    for count in (1, 3, 6, 12):
        roots = min(count, max_roots, dimension)
        if roots <= examined:
            continue
        if roots + 2 >= dimension:
            energies, vectors = dense_roots()
            energies, vectors = energies[:roots], vectors[:, :roots]
            solver = "cupy.linalg.eigh"
        else:
            operator = LinearOperator((dimension, dimension), matvec=apply,
                                      dtype=np.float64)
            initial = cp.asarray(np.sin(np.arange(1, dimension + 1, dtype=np.float64)))
            energies, vectors = eigsh(
                operator, k=roots, which="SA", v0=initial,
                tol=min(tol, residual_tol / 100) if residual_tol is not None else tol,
                maxiter=max(500, 10 * dimension))
            solver = "cupy.eigsh"
        for index in range(roots):
            energy = float(energies[index])
            vector = vectors[:, index]
            spin_squared = float(fci.spin_op.spin_square(
                cp.asnumpy(vector).reshape(kernel.na, kernel.nb),
                kernel.norb, kernel.nelec)[0])
            if total_spin is not None and abs(
                    spin_squared - total_spin * (total_spin + 1)) >= 1e-5:
                continue
            residual = float(cp.linalg.norm(apply(vector) - energy * vector))
            threshold = residual_tol if residual_tol is not None else max(1e-8, tol * 100)
            if not np.isfinite(residual) or residual > threshold:
                raise AdapterError("CF-E020", None,
                                   "GPU FCI wavefunction residual exceeds tolerance",
                                   "Use a smaller active space or CPU CI engine")
            result = (energy + ecore, spin_squared, {
                "s2": spin_squared, "n_roots_examined": roots,
                "n_sigma": calls, "residual": residual,
                "rows_per_block": kernel.rows_per_block, "solver": solver,
            })
            return ((*result, vector.reshape(kernel.na, kernel.nb))
                    if return_wavefunction else result)
        examined = roots
        if roots == dimension or roots == max_roots:
            break
    raise AdapterError("CF-E020", None, "Requested spin state unavailable",
                       "Select a feasible total spin or increase max_roots")


def fci_ground_state(kernel, ecore, total_spin=None, *, tol=1e-10, max_roots=12,
                     return_wavefunction=False, residual_tol=None,
                     gpu_resident=False):
    if residual_tol is not None and (not return_wavefunction or
                                     not np.isfinite(residual_tol) or residual_tol <= 0):
        raise ValueError("Residual tolerance requires a returned CI wavefunction")
    if gpu_resident:
        if kernel.xp.__name__ != "cupy":
            raise ValueError("GPU-resident FCI requires a CuPy kernel")
        return _gpu_resident_ground_state(
            kernel, ecore, total_spin, tol, max_roots,
            return_wavefunction, residual_tol)
    diagonal = kernel.hdiag()
    dimension = len(diagonal)
    calls = 0

    def apply_one(vector):
        nonlocal calls
        calls += 1
        product = kernel.sigma(vector)
        return kernel.xp.asnumpy(product) if kernel.xp.__name__ == "cupy" else product

    def apply(vectors):
        return [apply_one(vector) for vector in vectors]

    def precondition(residual, energy, *args):
        denominator = diagonal - energy
        return residual / np.where(abs(denominator) < 1e-4, 1e-4, denominator)

    examined = 0
    for count in (1, 3, 6, 12):
        roots = min(count, max_roots, dimension)
        if roots <= examined:
            continue
        starts = []
        indices = ([int(np.argmin(diagonal))] if roots == 1 else
                   np.argsort(diagonal, kind="stable")[:roots])
        for index in indices:
            start = np.zeros(dimension)
            start[index] = 1
            starts.append(start)
        _converged, energies, vectors = lib.davidson1(
            apply, starts, precondition, nroots=roots,
            tol=min(tol, 1e-12) if residual_tol is not None else tol,
            max_cycle=300 if residual_tol is not None else 200,
            max_space=max(40 if residual_tol is not None else 12, roots + 4),
            lindep=1e-18 if residual_tol is not None else 1e-14,
            tol_residual=residual_tol / 100 if residual_tol is not None else None,
            verbose=0)
        for energy, vector in zip(np.atleast_1d(energies), vectors):
            spin_squared = float(fci.spin_op.spin_square(
                vector.reshape(kernel.na, kernel.nb), kernel.norb, kernel.nelec)[0])
            if total_spin is None or abs(spin_squared - total_spin * (total_spin + 1)) < 1e-5:
                residual = float(np.linalg.norm(apply_one(vector) - energy * vector))
                if residual_tol is not None and residual > residual_tol:
                    raise AdapterError("CF-E020", None,
                                       "GPU FCI wavefunction residual exceeds tolerance",
                                       "Use a smaller active space or CPU CI engine")
                result = (float(energy + ecore), spin_squared, {
                    "s2": spin_squared, "n_roots_examined": roots, "n_sigma": calls,
                    "residual": residual, "rows_per_block": kernel.rows_per_block,
                })
                return ((*result, vector.reshape(kernel.na, kernel.nb))
                        if return_wavefunction else result)
        examined = roots
        if roots == dimension or roots == max_roots:
            break
    raise AdapterError("CF-E020", None, "Requested spin state unavailable",
                       "Select a feasible total spin or increase max_roots")


def cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() >= 1
    except (ImportError, RuntimeError, OSError):
        return False


def register_gpu(registry):
    if not cuda_available():
        raise RuntimeError("CF-E080: CUDA unavailable: GPU bindings cannot run on this device")
    for implementation in catalogue().values():
        registry.register(implementation)


def catalogue() -> dict[str, ImplementationSpec]:
    """Describe all GPU bindings without loading CUDA or importing CuPy."""

    def device_name():
        import cupy as cp

        device = cp.cuda.Device()
        name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
        return f"cuda:{device.id}:{name.decode() if isinstance(name, bytes) else name}"

    def ground(store, hamiltonian, sector, s=None, total_spin=None, **kwargs):
        import cupy as cp

        data = hamiltonian
        if ((sector.get("n_alpha"), sector.get("n_beta"), sector.get("n_orb")) !=
            (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb)):
            raise AdapterError("CF-E020", None, "Sector differs from Hamiltonian",
                               "Match particle counts")
        target = s if s is not None else total_spin
        if target is None:
            target = sector.get("s", sector.get("total_spin"))
        cp.cuda.Device().synchronize()
        started = time.perf_counter()
        kernel = FciKernel(store.get(data.hamiltonian.h1_ref),
                           store.get(data.hamiltonian.h2_ref), data.sector.n_orb,
                           (data.sector.n_alpha, data.sector.n_beta), xp=cp)
        energy, _, diagnostics = fci_ground_state(
            kernel, data.hamiltonian.ledger.total().value, target)
        cp.cuda.Device().synchronize()
        return ExecutionResult((energy, diagnostics), device=device_name(),
                               cost={"gpu_s": time.perf_counter() - started},
                               diagnostics=diagnostics)

    def sample(store, prep, shots, seed=0, **kwargs):
        import cupy as cp

        if shots < 1:
            raise ValueError("Positive shots required")
        cp.cuda.Device().synchronize()
        started = time.perf_counter()
        state = cp.asarray(store.get(prep.state_ref))
        probabilities = cp.abs(state) ** 2
        if not bool(cp.isclose(probabilities.sum(), 1)):
            raise ValueError("State must be normalized")
        rng = cp.random.RandomState(seed)
        drawn = rng.choice(len(state), size=shots, p=probabilities)
        if prep.sector is not None:
            drawn = cp.asarray(sector_states(prep.sector))[drawn]
        keys, counts = cp.unique(drawn, return_counts=True)
        samples = Samples(counts={str(key): int(count) for key, count in
                                  zip(cp.asnumpy(keys), cp.asnumpy(counts))}, shots=shots,
                          encoding_id=prep.encoding_id, basis="computational",
                          source="quantum_simulator", seed=seed, device=device_name())
        cp.cuda.Device().synchronize()
        return ExecutionResult(samples, device=samples.device,
                               cost={"gpu_s": time.perf_counter() - started, "sim_shots": shots})

    implementations = (ImplementationSpec(id="fci_cupy", version="1", target="ground_energy",
        implements="fci", backend_kind=BackendKind.gpu_numeric, adapter=ground,
        params=frozenset({"s", "total_spin"})),
        ImplementationSpec(id="statevector_cupy", version="1",
        target="sample_configurations", backend_kind=BackendKind.quantum_simulator,
        execution_role="emulator", emulates="sample_configurations",
        effects="random", adapter=sample, params=frozenset({"shots", "seed"}),
        resource_model=lambda shots=None, **kwargs:
        {"sim_shots": shots}))
    return {implementation.id: implementation for implementation in implementations}
