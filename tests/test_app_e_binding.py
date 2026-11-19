import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemflow import build
from chemflow.agent.families import default_axes, instantiate
from chemflow.agent.task import BindingTarget
from chemflow.cards import validate_cards
from chemflow.check.static import infer_relations, parameter_diagnostics
from chemflow.diagnostics import AdapterError
from chemflow.domains import binding, load_registry, pack_names
from chemflow.domains._tools_client import call_tools, tools_python
from chemflow.domains._tools_worker import predicted_pkas, protonation_variants
from chemflow.eval.runner import Expected
from chemflow.ir.schema import Node, Program
from chemflow.runtime import Executor
from chemflow.runtime.artifacts import read_text_artifact, write_text_artifact
from chemflow.verify.lemmas import AXIOMS


def target():
    return BindingTarget(kind="binding_free_energy", receptor={"pdb_text": "PDB", "chains": ["A"]},
                         ligand={"source": {"smiles": "C"}})


def test_registered_contracts_and_cards():
    registry = load_registry(pack_names())
    operators = ("protein_structure", "ligand_structure", "complex", "pocket",
                 "ligand_parameters", "mmgbsa", "fragment_interaction")
    for name in operators:
        assert registry.get(f"chemflow.{name}@1").semantics
        implementation = registry.implementations[name]
        assert implementation.execution_role == "classical"
        assert implementation.resource_model()["qpu_shots"] == 0
        assert f"binding:{name}" in AXIOMS
    assert registry.search("binding_free_energy")
    assert validate_cards(registry, Path(__file__).parents[1]).ok


def test_text_artifact_and_ligand():
    text = "protein coordinates\n"
    assert read_text_artifact(write_text_artifact(text, ".pdb")) == text
    ligand = binding.ligand_structure({"smiles": "[NH4+]"})
    assert ligand["n_atoms"] == 5
    assert ligand["formal_charge"] == 1
    assert "V2000" in read_text_artifact(ligand["sdf_ref"])


@pytest.mark.parametrize("operation,params", [
    ("mmgbsa", {"frames": -1}), ("mmgbsa", {"frames": True}),
    ("mmgbsa", {"minimize": "yes"}), ("mmgbsa", {"seed": -1}),
    ("protein_structure", {"ph": 20}), ("protein_structure", {"chains": "A"}),
    ("pocket", {"cutoff_A": 0}), ("ligand_structure", {"source": {}}),
    ("ligand_parameters", {"forcefield": "unknown"}),
    ("ligand_parameters", {"charge_method": "gasteiger"}),
    ("fragment_interaction", {"level": {"method": "invalid"}}),
])
def test_parameter_validation(operation, params):
    program = Program(name="invalid", nodes=[Node(id="node", op=operation, params=params)])
    assert parameter_diagnostics(program, load_registry(["binding"]), require_complete=False)


def test_nullable_level_is_valid():
    from chemflow.verify.kernel import check_certificate
    from chemflow.verify.prover import prove

    program = Program(name="level", nodes=[Node(id="node", op="fragment_interaction", params={
        "residues": [], "level": {"method": "hf", "functional": None, "dispersion": None}})])
    registry = load_registry(["binding"])
    assert not parameter_diagnostics(program, registry, require_complete=False)
    assert infer_relations(program, registry)["node"] == "approximate"
    computed = {"node": {"total": 0, "per_residue": {}}}
    proof = prove(program, "node", registry=registry, computed=computed)
    assert proof["relation"] == "approximate"
    assert check_certificate(proof, program, registry=registry, computed=computed).valid


@pytest.mark.parametrize("name,protonated,deprotonated", [
    ("HIS", "HIP", "HIE"), ("ASP", "ASH", "ASP"), ("GLU", "GLH", "GLU"),
    ("LYS", "LYS", "LYN"),
])
def test_predicted_pka_controls_variants(name, protonated, deprotonated):
    atom = SimpleNamespace(chain_id="A", res_num=99)
    prediction = SimpleNamespace(conformations={"AVR": SimpleNamespace(groups=[
        SimpleNamespace(label=f"{name} 99 A", atom=atom, pka_value=8.0),
        SimpleNamespace(label="N+ 99 A", atom=atom, pka_value=9.0)])})
    assert predicted_pkas(prediction) == {"A:99": 8.0}
    residue = SimpleNamespace(name=name, id="99", chain=SimpleNamespace(id="A"))
    for pka, expected in ((8.0, protonated), (6.0, deprotonated)):
        records, variants = protonation_variants([residue], 7.0, {"A:99": pka})
        assert variants == [expected]
        assert records["A:99"]["pKa"] == pka
        assert records["A:99"]["source"] == "PROPKA"
    records, _ = protonation_variants([residue], 7.0, {})
    assert records["A:99"]["source"] == "standard_pKa_fallback"


def test_charged_fragment_parity_and_cap(monkeypatch):
    from pyscf import scf

    ligand = {"formal_charge": 0}
    ligand_residue = SimpleNamespace(name="LIG", id="1", chain=SimpleNamespace(id="L"))
    residue = SimpleNamespace(name="ASP", id="99", chain=SimpleNamespace(id="A"))
    atoms = [("CA", "C", [-1.5, 0, 0]), ("CB", "C", [0, 0, 0]),
             ("CG", "C", [1.4, 0, 0]), ("OD1", "O", [2.1, 1, 0]),
             ("OD2", "O", [2.1, -1, 0]), ("HB2", "H", [0, .8, .8]),
             ("HB3", "H", [0, -.8, -.8])]
    groups = [(residue, atoms), (ligand_residue,
              [("H1", "H", [0, 0, 4]), ("H2", "H", [0, 0, 4.74])])]
    monkeypatch.setattr(binding, "_residue_atoms", lambda _: groups)
    calls = []

    def meanfield(molecule):
        calls.append((molecule.charge, molecule.spin, molecule.atom))
        return SimpleNamespace(converged=True, kernel=lambda: 0.0)

    monkeypatch.setattr(scf, "RHF", meanfield)
    monkeypatch.setattr(scf, "ROHF", meanfield)
    complex_ = {"ligand": ligand, "ligand_residue": "L:1",
                "protonation": {"A:99": {"state": "ASP"}}}
    result = binding.fragment_interaction(complex_, ligand, ["A:99"])
    assert result["formal_charges"] == {"A:99": -1}
    assert [(charge, spin) for charge, spin, _ in calls] == [(-1, 0), (0, 0), (-1, 0)]
    assert calls[0][2][-1][1] == pytest.approx([-1.09, 0, 0])


def test_target_identity_and_ph_family():
    record = SimpleNamespace(spec=SimpleNamespace(target=target()))
    plan, controls = binding.base_plan(record)
    assert controls and plan.references and not set(controls) & set(plan.references)
    assert not binding.identity(plan, record, {})
    axis = default_axes(target())[0]
    assert axis.axis == "binding_ph"
    changed = instantiate(plan, axis.axis, 8.0)
    assert binding.identity(changed, record, {})
    changed = plan.model_copy(deep=True)
    next(node for node in changed.nodes if node.op == "ligand_structure").params["source"] = {
        "smiles": "CC"}
    assert binding.identity(changed, record, {})
    with pytest.raises(ValidationError):
        BindingTarget.model_validate({**target().model_dump(), "extra": True})


def test_expected_value_and_units():
    assert Expected(value=-4.0, unit="kcal/mol").value == -4
    assert Expected(value_hartree=-1).value_hartree == -1
    for invalid in ({"value": 2}, {"unit": "kcal/mol"},
                    {"value": 1, "unit": "kcal/mol", "value_hartree": 1}):
        with pytest.raises(ValidationError):
            Expected(**invalid)


def test_missing_side_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMFLOW_TOOLS_PYTHON", str(tmp_path / "missing"))
    with pytest.raises(AdapterError, match="CF-E080") as caught:
        call_tools("protein_structure", {})
    assert "app-e-binding.md" in caught.value.diagnostics[0].fix_hints[0]


def test_worker_timeout_and_stderr(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["env"]["PATH"].startswith(str(tools_python().parent))
        raise subprocess.TimeoutExpired(args[0], 0.01, stderr="worker stalled")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(AdapterError, match="worker stalled"):
        call_tools("protein_structure", {}, timeout=0.01)


@pytest.mark.skipif(not tools_python().exists(), reason="chemflow-tools-env not installed")
def test_openff_seeded_frames():
    from chemflow.verify.integration import certify_results
    from chemflow.verify.kernel import check_certificate

    pdb = (Path(__file__).parent / "data/181L.pdb").read_text()
    with build.program("sampled-binding") as builder:
        protein = builder.node("protein_structure", pdb_text=pdb)
        ligand = builder.node("ligand_structure", source={"pdb_text": pdb, "residue": "BNZ",
                                                         "smiles": "c1ccccc1"})
        assembly = builder.node("complex", protein=protein, ligand=ligand)
        parameters = builder.node("ligand_parameters", ligand=ligand, forcefield="openff-2.2.0")
        result = builder.node("mmgbsa", complex=assembly, parameters=parameters, frames=2, seed=7)
    registry = load_registry(["binding"])
    computed = Executor(registry).run(builder.to_ir())
    certify_results(builder.to_ir(), registry, computed)
    estimate = computed[result.id]
    assert estimate.frames == 2 and estimate.stderr >= 0
    assert estimate.dG_kcal == pytest.approx(estimate.components["complex"] -
        estimate.components["receptor"] - estimate.components["ligand"])
    assert estimate.diagnostics["experimental_relation"] == "model_change"
    assert estimate.diagnostics["relation_status"] == "proved"
    assert abs(sum(computed[parameters.id]["charges"])) < 1e-6
    computed[result.id] = estimate.model_copy(update={"dG_kcal": estimate.dG_kcal + 1})
    verdict = check_certificate(computed["relation_certificates"][result.id], builder.to_ir(),
                                registry=registry, computed=computed)
    assert not verdict.valid
