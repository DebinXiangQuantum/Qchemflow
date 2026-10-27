"""Frozen six-site AIM design experiment; no tuned per-instance parameters."""

from time import perf_counter

from chemflow import build
from chemflow.design import DesignStore, design_loop
from chemflow.domains import load_registry
from chemflow.ir import AcceptanceGate
from chemflow.units import Q


def seed_design():
    """A1a-style sampled ground space with singles-closed ladder response spaces."""
    model = {"U": 2.5, "eps_bath": [-1.1, -0.4, 0.0, 0.5, 1.2],
             "t_bath": [0.45, 0.6, 0.5, 0.55, 0.4], "unit": "hartree"}
    target = {"kind": "spectral_function", "model": {"impurity_model": model},
              "sector": {"n_alpha": 3, "n_beta": 3, "n_orb": 6},
              "orbital": 0, "spin": "alpha",
              "omega": {"start": -4.0, "stop": 4.0, "n": 801, "unit": "hartree"},
              "eta": 0.1}
    with build.program("aim-design-bath5") as builder:
        hamiltonian = builder.node("impurity_model", **model)
        sector = builder.sector(**target["sector"])
        prep = builder.node("prepare_state", hamiltonian=hamiltonian, sector=sector,
                            ansatz="lucj", init="ccsd")
        samples = builder.node("sample_configurations", prep=prep, shots=20000, seed=7)
        sampled = builder.node("subspace_from_samples", samples=samples, sector=sector)
        ground = builder.node("rayleigh_ritz", hamiltonian=hamiltonian,
                              subspace=sampled, n_roots=1)
        spectra = {}
        for role, delta, operator in (("particle", 1, "create"), ("hole", -1, "annihilate")):
            excited = builder.sector(n_alpha=3 + delta, n_beta=3, n_orb=6)
            space = builder.node("extend_subspace", subspace=sampled, sector=excited,
                                 operator=operator, orbital=0, spin="alpha", closure="singles")
            spectra[role] = builder.node("rayleigh_ritz", hamiltonian=hamiltonian,
                                        subspace=space, n_roots=8)
        response = builder.node("spectral_function", ground=ground, **spectra,
                                **{key: target[key] for key in ("orbital", "spin", "omega", "eta")})
        builder.refine(response, "lehmann")
        builder.bind(response, "lehmann_from_spectra")
    program = builder.to_ir()
    program.gates.append(AcceptanceGate(node=response.id, l1_error_le=Q(0.1, "dimensionless")))
    return program, response.id, target


def run():
    started = perf_counter()
    program, response, target = seed_design()
    registry = load_registry(["molecular", "algorithms", "dynamics"])
    store = DesignStore()
    result = design_loop(program, response, target=target, registry=registry, store=store,
                         instances=3, seed=5, max_rounds=6, register_as="aim_bath5")
    final = store.get(result.trajectory[-1]["report_id"])
    return {"name": "aim-design-bath5", "pack": "dynamics", "outcome": result.outcome,
            "passed": result.outcome == "passed", "n_sites": 6, "instances": 3, "seed": 5,
            "accuracy": final.accuracy, "trajectory": result.trajectory,
            "design": result.design, "final_design": final.template,
            "sector_usage": final.resources.get("subspaces", []),
            "runtime_s": perf_counter() - started}
