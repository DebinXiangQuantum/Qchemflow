import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from chemflow import build
from chemflow.agent.report import ReportLintError, lint_recorded_report
from chemflow.check import check
from chemflow.check.static import infer_context
from chemflow.design import DesignStore, verify_design
from chemflow.domains import load_registry
from chemflow.domains.fock_reference import reference
from chemflow.plugin_build import TARGETS
from chemflow.plugin_build import build as build_plugin
from chemflow.signing import generate_keypair
from chemflow.units import Q

MODEL = {"U": 2., "eps_bath": [-1.], "t_bath": [.5], "unit": "hartree"}
TARGET = {"kind": "spectral_function", "model": {"impurity_model": MODEL},
          "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
          "orbital": 0, "spin": "alpha",
          "omega": {"start": -4., "stop": 4., "n": 101, "unit": "hartree"}, "eta": .1}


def design():
    with build.program("small-design") as builder:
        ham = builder.node("impurity_model", **MODEL)
        sector = builder.sector(**TARGET["sector"])
        response = builder.node("spectral_function", hamiltonian=ham, sector=sector,
                                **{key: TARGET[key] for key in ("orbital", "spin", "omega", "eta")})
        builder.refine(response, "exact_lehmann")
        builder.bind(response, "spectral_exact")
        builder.accept(response, l1_error_le=Q(100, "dimensionless"))
    return builder.to_ir(), response.id


@pytest.mark.parametrize("field,value", [("orbital", 1), ("spin", "beta"), ("eta", .2)])
def test_identity_mismatch_fails_before_execution(monkeypatch, field, value):
    program, node = design()
    next(item for item in program.nodes if item.id == node).params[field] = value

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched designs must not execute")

    monkeypatch.setattr("chemflow.design.Executor.run", forbidden)
    report = verify_design(program, node, target=TARGET,
                           registry=load_registry(["algorithms", "dynamics"]), store=DesignStore(),
                           instances=1, seed=0)
    assert not report.passed and "identity" in report.reason


def test_independent_atomic_lehmann_reference():
    target = deepcopy(TARGET)
    target["sector"] = {"n_alpha": 1, "n_beta": 0, "n_orb": 1}
    model = {"U": 2., "eps_bath": [], "t_bath": [], "unit": "hartree"}
    result = reference(target, model)
    frequencies = np.asarray(result.omega)
    expected = .1 / (np.pi * ((frequencies + 1) ** 2 + .1 ** 2))
    assert np.allclose(result.spectral, expected, atol=1e-12)


def test_unused_impurity_node_cannot_disguise_a_different_model_family():
    with build.program("wrong-family") as builder:
        builder.node("impurity_model", **MODEL)
        ham = builder.node("lattice_model", n_sites=2, t=1., U=2.)
        sector = builder.sector(**TARGET["sector"])
        energy = builder.ground_energy(ham, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "fci_pyscf")
        builder.accept(energy, abs_error_le=Q(100, "hartree"))
    target = {"kind": "ground_energy", "model": TARGET["model"], "sector": TARGET["sector"]}
    with pytest.raises(ValueError, match="family node"):
        verify_design(builder.to_ir(), energy.id, target=target,
                      registry=load_registry(["materials", "dynamics", "algorithms"]),
                      store=DesignStore(), instances=1, seed=0)


def test_verification_refuses_large_fock_space_before_execution(monkeypatch):
    from chemflow.domains.fock_reference import MAX_ORBITALS

    program, node = design()
    target = deepcopy(TARGET)
    target["model"]["impurity_model"].update(eps_bath=[0.] * MAX_ORBITALS,
                                             t_bath=[.5] * MAX_ORBITALS)
    with pytest.raises(ValueError, match=f"at most {MAX_ORBITALS}"):
        verify_design(program, node, target=target,
                      registry=load_registry(["algorithms", "dynamics"]), store=DesignStore(),
                      instances=1, seed=0)


@pytest.mark.parametrize("unit", ["hartree", "Ha", "Eh", "mHa", "eV", "kcal/mol",
                                 "kJ/mol", "cm-1", "cm^-1"])
def test_report_energy_units_require_recorded_converted_values(unit):
    headline = {"result_id": "recorded", "node": "energy", "value": "-1.000000000 hartree",
                "relation": "equivalent", "evidence": "diagnostic_only", "gate_verdict": "PASS",
                "cost": {}, "backend_kind": "cpu_numeric"}
    database = SimpleNamespace(get=lambda identifier: {"record_hash": "task"},
                               headline=lambda identifier: {"headline": headline, "critic": None})
    canonical = {"Ha": "hartree", "Eh": "hartree", "mHa": "hartree",
                 "cm-1": "cm^-1"}.get(unit, unit)
    number = Q(-1., "hartree").to(canonical).value * (1000 if unit == "mHa" else 1)
    text = f"Energy: {number:.9f} {unit}."
    assert lint_recorded_report(text, [{"result_id": "recorded"}], database, "task")
    with pytest.raises(ReportLintError):
        lint_recorded_report(f"Energy: {number * 2:.9f} {unit}.",
                             [{"result_id": "recorded"}], database, "task")
    with pytest.raises(ReportLintError):
        lint_recorded_report(text, [], database, "task")


@pytest.mark.parametrize("op,params", [("molecule", {}), ("molecule", {"atoms": 4}),
    ("sector", {"n_alpha": "1", "n_beta": 1, "n_orb": 2}),
    ("sector", {"n_alpha": 1, "n_beta": 1, "n_orb": 2, "s": "bad"}),
    ("electronic_model", {"basis": []})])
def test_checker_reports_bad_parameters_before_context_arithmetic(op, params):
    with build.program("malformed") as builder:
        node = builder.node(op, **params)
    registry = load_registry(["molecular"])
    for issues in (check(builder.to_ir(), registry), infer_context(builder.to_ir(), registry)[1]):
        assert issues and all(issue.node == node.id and issue.code in ("CF-E080", "CF-E020")
                              for issue in issues)


@pytest.mark.parametrize("target", TARGETS)
def test_all_plugin_targets_forward_packs(tmp_path, target):
    _, public = generate_keypair(b"offline")
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    out = tmp_path / target
    build_plugin(target, out, key, packs="algorithms,dynamics")
    if target == "pi":
        assert "'--packs', 'algorithms,dynamics'" in (out / "extensions/chemflow.ts").read_text()
        return
    if target == "goose":
        servers = yaml.safe_load((out / "goose-config.yaml").read_text())["extensions"]
    elif target == "hermes":
        servers = yaml.safe_load((out / "hermes-config.yaml").read_text())["mcp_servers"]
    else:
        filename = {"codex": "plugins/chemflow/mcp.json", "agent-plugins": "mcp.json",
                    "claude-code": ".mcp.json"}[target]
        servers = json.loads((out / filename).read_text())["mcpServers"]
    assert servers
    for server in servers.values():
        args = server["args"]
        assert args[args.index("--packs") + 1] == "algorithms,dynamics"
