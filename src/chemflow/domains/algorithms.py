"""Algorithm primitives, independent of registered method recipes.

full_subspace enumerates D=C(M,Na)C(M,Nb) orthonormal determinants (O(D M)).
subspace_from_samples applies SQD configuration recovery, then the Cartesian
product of recovered spin strings; equal fillings share their string union.
Recovery costs O(shots M), product storage O(D_selected), with no invented shots.
extend_subspace applies a fermionic ladder operator to every basis determinant;
singles adds every spin-preserving one-body excitation (O(D_selected M^2)).
doubles adds a second singles layer (including mixed-spin doubles), O(D_selected M^4).
rayleigh_ritz diagonalizes P* H P: O(D_selected M^4 + D_selected^3) time,
O(D_selected^2) storage. n_roots selects reported leading roots; the complete
projected resolution is retained in artifacts for response calculations.
The dense reference implementation limits enumeration/diagonalization to 4096
determinants. Recovery is a single pass; SQD's occupancy-feedback loop remains
in sqd_sector, using the same recovery and product-closure helpers.

The checker/executor infer relations structurally: the HUM min-max principle
gives ordered variational upper bounds for any valid determinant subspace;
full coverage makes roots equivalent. Response equivalence requires complete
coverage and spectral resolutions in all three sectors, not merely accurate
ground energies. Other chains/sums use compose/combine_linear, never a binding's
claim about its own result. Sampling alone does not invalidate Ritz bounds.
"""

from math import comb

import numpy as np

from chemflow.diagnostics import Diagnostic
from chemflow.impls.fci import matrix, sector_states
from chemflow.impls.sqd import product_strings, recover_configurations
from chemflow.registry import ImplementationSpec, OperatorSpec
from chemflow.types import BackendKind, Sector, Spectrum, Subspace


def as_sector(sector):
    if isinstance(sector, dict) and sector.get("s") is not None:
        raise ValueError("Determinant primitives fix particle counts, not total spin")
    result = sector if isinstance(sector, Sector) else Sector.model_validate(sector)
    if result.total_spin is not None:
        raise ValueError("Determinant primitives fix particle counts, not total spin")
    return result


def full_subspace(store, sector):
    sector = as_sector(sector)
    if comb(sector.n_orb, sector.n_alpha) * comb(sector.n_orb, sector.n_beta) > 4096:
        raise ValueError("Full-sector enumeration exceeds the 4096 determinant reference limit")
    return Subspace(sector=sector, determinants=tuple(sorted(sector_states(sector))))


def subspace_from_samples(store, samples, sector, recovery="configuration", seed=7):
    sector = as_sector(sector)
    if sector.n_orb > 31:
        raise ValueError("Configuration recovery supports at most 62 spin orbitals")
    if recovery != "configuration" or samples.basis != "computational":
        raise ValueError("Configuration recovery requires computational samples")
    if not samples.counts or any(type(count) is not int or count < 0
                                 for count in samples.counts.values()):
        raise ValueError("Configuration counts must be nonnegative integers")
    if sum(samples.counts.values()) != samples.shots:
        raise ValueError("Configuration counts must sum to shots")
    values = np.concatenate([np.full(count, int(mask), dtype=np.int64)
                             for mask, count in samples.counts.items() if count > 0])
    if not len(values) or min(values) < 0 or max(values) >= 1 << (2 * sector.n_orb):
        raise ValueError("Invalid configuration counts")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    occupancy = ((values[:, None] >> np.arange(2 * sector.n_orb)) & 1).mean(axis=0)
    masks = recover_configurations(values, sector, occupancy, rng)
    alpha, beta = product_strings(masks, sector)
    return Subspace(sector=sector, determinants=tuple(sorted(
        left | (right << sector.n_orb) for left in alpha for right in beta)))


def ladder(state, mode, create):
    if bool(state & (1 << mode)) == create:
        return None
    return state ^ (1 << mode), (-1) ** (state & ((1 << mode) - 1)).bit_count()


def transition_sector(source, target, operator, orbital, spin):
    if (operator not in ("create", "annihilate") or spin not in ("alpha", "beta") or
            type(orbital) is not int or not 0 <= orbital < source.n_orb):
        raise ValueError("Invalid ladder operator, orbital, or spin")
    counts = {"n_alpha": source.n_alpha, "n_beta": source.n_beta, "n_orb": source.n_orb}
    counts[f"n_{spin}"] += 1 if operator == "create" else -1
    if any(getattr(target, key) != value for key, value in counts.items()):
        raise ValueError("Ladder target must be the corresponding N±1 sector")
    return orbital + (source.n_orb if spin == "beta" else 0)


def extend_subspace(store, subspace, sector, operator, orbital, spin, closure="none"):
    sector = as_sector(sector)
    mode = transition_sector(subspace.sector, sector, operator, orbital, spin)
    if closure not in ("none", "singles", "doubles"):
        raise ValueError("Unknown subspace closure")
    states = {result[0] for state in subspace.determinants
              if (result := ladder(state, mode, operator == "create")) is not None}
    for _ in range({"none": 0, "singles": 1, "doubles": 2}[closure]):
        for state in tuple(states):
            for offset in (0, sector.n_orb):
                for occupied in range(offset, offset + sector.n_orb):
                    for vacant in range(offset, offset + sector.n_orb):
                        if state & (1 << occupied) and not state & (1 << vacant):
                            states.add(state ^ (1 << occupied) ^ (1 << vacant))
    return Subspace(sector=sector, determinants=tuple(sorted(states)))


def rayleigh_ritz(store, hamiltonian, subspace, n_roots=1):
    if type(n_roots) is not int or n_roots < 1:
        raise ValueError("n_roots must be a positive integer")
    if subspace.sector.n_orb != hamiltonian.hamiltonian.n_orb:
        raise ValueError("Subspace orbital count differs from Hamiltonian")
    if not subspace.determinants:
        raise ValueError("Cannot diagonalize an empty subspace")
    if len(subspace.determinants) > 4096:
        raise ValueError("Dense Rayleigh–Ritz exceeds the 4096 determinant reference limit")
    projected = matrix(hamiltonian.hamiltonian, store, list(subspace.determinants)).toarray()
    if not np.allclose(projected, projected.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Projected Hamiltonian must be Hermitian")
    energies, vectors = np.linalg.eigh(projected)
    return Spectrum(energies=tuple(energies[:n_roots]), vectors_ref=store.put(vectors[:, :n_roots]),
                    resolution_energies_ref=store.put(energies),
                    resolution_vectors_ref=store.put(vectors), subspace=subspace,
                    hamiltonian_id=hamiltonian.hamiltonian.id)


def resources(**kwargs):
    return {"sim_shots": 0, "qpu_shots": 0}


def context_check(node, inputs, context, nodes):
    if node.op == "sector":
        context["sector_params"] = node.params
    elif node.op in ("full_subspace", "subspace_from_samples", "extend_subspace"):
        context.update(sector_id=inputs.get("sector", {}).get("sector_id"),
                       sector_params=inputs.get("sector", {}).get("sector_params"))
    elif node.op == "rayleigh_ritz":
        context.update(inputs.get("hamiltonian", {}))
        context.update(type="spectrum", sector_id=inputs.get("subspace", {}).get("sector_id"))
        context["sector_params"] = inputs.get("subspace", {}).get("sector_params")
    try:
        if node.op == "extend_subspace" and all(port in inputs for port in ("subspace", "sector")):
            source = inputs["subspace"].get("sector_params")
            target = inputs["sector"].get("sector_params")
            if source is not None and target is not None:
                transition_sector(as_sector(source), as_sector(target), node.params["operator"],
                                  node.params["orbital"], node.params["spin"])
            if node.params.get("closure", "none") not in ("none", "singles", "doubles"):
                raise ValueError("Unknown subspace closure")
        if node.op == "rayleigh_ritz":
            roots = node.params.get("n_roots", 1)
            if type(roots) is not int or roots < 1:
                raise ValueError("n_roots must be a positive integer")
    except (ValueError, KeyError, TypeError) as exc:
        return [Diagnostic(code="CF-E020", node=node.id, message=str(exc),
                           fix_hints=["Use valid sectors, ladder parameters and root counts"])]
    return []


def register(registry):
    for name, inputs, output, adapter, params, meaning in (
        ("full_subspace", {"sector": "sector"}, "subspace", full_subspace, set(),
         "All orthonormal determinants in a fixed particle sector; O(D M) enumeration."),
        ("subspace_from_samples", {"samples": "samples", "sector": "sector"}, "subspace",
         subspace_from_samples, {"recovery", "seed"},
         "SQD configuration recovery and product-of-spin-strings closure; O(shots M + D)."),
        ("extend_subspace", {"subspace": "subspace", "sector": "sector"}, "subspace",
         extend_subspace, {"operator", "orbital", "spin", "closure"},
         "Ladder image in N±1; singles O(D M²) or doubles O(D M⁴) closure."),
        ("rayleigh_ritz", {"hamiltonian": "hamiltonian", "subspace": "subspace"}, "spectrum",
         rayleigh_ritz, {"n_roots"},
         ("Eigenpairs of P*HP, HUM variational bounds; O(D³) solve and O(D²) storage. "
          "n_roots selects leading outputs; complete projected resolution retained for responses.")),
    ):
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
                                       output=output, semantics=meaning,
                                       parameters={key: {
                                           "recovery": "configuration (default): SQD bit recovery",
                                           "seed": "integer RNG seed, default 7",
                                           "operator": "create | annihilate",
                                           "orbital": "zero-based spatial orbital",
                                           "spin": "alpha | beta",
                                           "closure": "none (default) | singles | doubles (two layers)",
                                           "n_roots": "positive integer, default 1; leading roots",
                                       }[key] for key in params}))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
                          backend_kind=BackendKind.cpu_numeric, params=frozenset(params),
                          resource_model=resources))
    registry.context_checks.append(context_check)
