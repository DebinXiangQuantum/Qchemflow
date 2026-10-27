import pytest

import qcl
from chemflow.domains.reactivity import site_reactivity

METHANE = ("C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; "
           "H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276")


@qcl.workflow
def sites(molecule, cpu, model=None, level="hf"):
    return qcl.metabolism.sites(molecule, model=model, level=level).to(cpu)


def test_metabolic_sites_keep_bde_sasa_and_score_evidence():
    molecule = qcl.Molecule(METHANE, "sto-3g")
    plan = sites.compile(molecule, qcl.Cluster.local().cpus[0])
    result = plan.run()
    reference = site_reactivity(None, {"atoms": METHANE}, {"basis": "sto-3g"},
                                {"method": "hf"})
    assert result.top_site == reference["top_site"]
    assert len(result.sites) == 4
    assert [item["score"] for item in result.sites] == pytest.approx(
        [item["score"] for item in reference["sites"]], abs=1e-9)
    assert result.provenance["validity_flags"]


def test_metabolic_sites_reject_unsupported_level_and_gpu():
    molecule = qcl.Molecule(METHANE, "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError):
        sites.compile(molecule, cpu, level="invalid")
    with pytest.raises(qcl.CompileError) as error:
        sites.compile(molecule, qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)
