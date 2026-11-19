import copy
import json
import shutil

import pytest
from test_v1a_certificates import spectral_plan

from chemflow.runtime import Executor
from chemflow.verify import LEMMAS, check_certificate, formal, prove


@pytest.fixture
def audit_copy(tmp_path, monkeypatch):
    destination = tmp_path / "formal"
    destination.mkdir()
    shutil.copytree(formal.FORMAL / "ChemflowFormal", destination / "ChemflowFormal")
    for name in (*formal.project_hashes(formal.FORMAL), "manifest.json"):
        shutil.copyfile(formal.FORMAL / name, destination / name)
    monkeypatch.setattr(formal, "FORMAL", destination)
    return destination


@pytest.mark.parametrize("damage", ["source", "import", "missing", "name", "file", "hash",
                                    "statement", "axioms", "duplicate", "python"])
def test_stale_audit_never_reports_machine_checked(audit_copy, monkeypatch, damage):
    assert formal.checked_lemmas({"rayleigh_ritz"}) == {"rayleigh_ritz"}
    path = audit_copy / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = next(item for item in manifest["theorems"] if item["lemma"] == "rayleigh_ritz")
    if damage in ("source", "import"):
        source = audit_copy / (entry["file"] if damage == "source" else "ChemflowFormal.lean")
        source.write_text(source.read_text() + "\n")
    elif damage == "missing":
        path.unlink()
    elif damage == "python":
        monkeypatch.setitem(LEMMAS["rayleigh_ritz"], "statement", "An unaudited stronger claim")
    else:
        if damage == "duplicate":
            manifest["theorems"].append(copy.deepcopy(entry))
        else:
            field, value = {
                "name": ("name", "ChemflowFormal.not_the_theorem"),
                "file": ("file", "../outside.lean"),
                "hash": ("source_sha256", "0" * 64),
                "statement": ("statement", "theorem rayleigh_ritz : True"),
                "axioms": ("axioms", ["unapprovedAxiom"]),
            }[damage]
            entry[field] = value
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Formal audit"):
        formal.checked_lemmas({"rayleigh_ritz"})


def test_excited_spectra_cite_minmax_and_kernel_requires_it():
    program, registry, spectra, _ = spectral_plan()
    computed = Executor(registry).run(program)
    certificate = prove(program, spectra[0].id, registry=registry, computed=computed)
    verdict = check_certificate(certificate, program, registry=registry, computed=computed)
    assert verdict.valid, verdict.reasons
    assert "rayleigh_ritz_full" in verdict.machine_checked
    assert "rayleigh_ritz_excited" in verdict.trusted
    damaged = copy.deepcopy(certificate)
    omitted = {step["id"] for step in damaged["steps"]
               if step["lemma"] == "rayleigh_ritz_excited"}
    damaged["steps"] = [step for step in damaged["steps"] if step["id"] not in omitted]
    for step in damaged["steps"]:
        step["uses"] = [identifier for identifier in step["uses"] if identifier not in omitted]
    verdict = check_certificate(damaged, program, registry=registry, computed=computed)
    assert not verdict.valid
    assert "Missing excited-root citation" in verdict.reasons[0]


def test_kernel_fails_closed_on_missing_manifest(audit_copy):
    program, registry, spectra, _ = spectral_plan()
    computed = Executor(registry).run(program)
    certificate = prove(program, spectra[0].id, registry=registry, computed=computed)
    (audit_copy / "manifest.json").unlink()
    verdict = check_certificate(certificate, program, registry=registry, computed=computed)
    assert not verdict.valid
    assert verdict.machine_checked == ()
    assert "Formal audit" in verdict.reasons[0]
