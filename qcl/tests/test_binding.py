from pathlib import Path

import pytest

import qcl

PDB = Path(__file__).resolve().parents[2] / "tests" / "data" / "181L.pdb"
TOOLS = Path(__file__).resolve().parents[3] / "chemflow-tools-env" / "bin" / "python"


@qcl.workflow
def binding_energy(receptor, ligand, cpu, **options):
    return qcl.binding.mmgbsa(receptor, ligand, **options).to(cpu)


def inputs():
    pdb = PDB.read_text()
    return ({"pdb_text": pdb, "chains": ["A"]},
            {"source": {"pdb_text": pdb, "residue": "BNZ", "smiles": "c1ccccc1"}})


@pytest.mark.skipif(not TOOLS.exists(), reason="binding side environment unavailable")
def test_binding_qcl_runs_reference_protocol():
    receptor, ligand = inputs()
    plan = binding_energy.compile(receptor, ligand, qcl.Cluster.local().cpus[0],
                                  frames=0, minimize=False, seed=0)
    result = plan.run(timeout_s=1800)
    assert result.unit == "kcal/mol"
    assert result.value < 0
    assert result.provenance["components"]["complex"]
    assert result.provenance["validity_flags"]


def test_binding_rejects_invalid_options_and_gpu():
    receptor, ligand = inputs()
    with pytest.raises(qcl.CompileError):
        binding_energy.compile(receptor, ligand, qcl.Cluster.local().cpus[0], frames=-1)
    with pytest.raises(qcl.CompileError) as error:
        binding_energy.compile(receptor, ligand, qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)
