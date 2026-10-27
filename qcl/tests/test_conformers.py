import pytest
from pyscf import gto, scf

import qcl


@qcl.workflow
def conformer_ranking(smiles, cpu, n=3):
    conformers = qcl.engines.rdkit.embed(smiles, n=n, seed=42).to(cpu)
    optimized = qcl.map(lambda conformer, lane:
                        qcl.engines.rdkit.mmff_optimize(conformer).to(lane),
                        conformers).to([cpu])
    unique = qcl.engines.rdkit.dedupe(optimized, rmsd=0.5).to(cpu)
    energies = qcl.map(lambda conformer, lane:
                       qcl.scf.rhf(conformer).to(lane).energy,
                       unique).to([cpu])
    return qcl.rank(energies)


def test_dynamic_conformer_map_runs_only_retained_items():
    cpu = qcl.Cluster.local().cpus[0]
    plan = conformer_ranking.compile("CCO", cpu, n=3)
    result = plan.run()
    assert len(result.values) == 1
    assert sorted(result.order) == list(range(len(result.values)))
    assert len(plan.explain(format="json")["patterns"]) == 2
    for estimate in result.values:
        molecule = estimate.provenance["question"]
        assert molecule["conformer"]["smiles"] == "CCO"
        assert molecule["conformer"]["embedding_seed"] == 42
        assert molecule["conformer"]["optimization"] == "MMFF94"
        assert molecule["conformer"]["rmsd_threshold"] == 0.5
        reference = scf.RHF(gto.M(atom=molecule["atoms"], basis="sto-3g",
                                  verbose=0)).run()
        assert estimate.value == pytest.approx(reference.e_tot, abs=1e-9)


def test_dynamic_map_rejects_non_cpu_rdkit_placement():
    with pytest.raises(qcl.CompileError) as error:
        conformer_ranking.compile("CCO", qcl.Cluster.local().qpus[0], n=2)
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_symbolic_geometry_edit_retains_conformer_origin():
    @qcl.workflow
    def edited(smiles, cpu):
        conformers = qcl.engines.rdkit.embed(smiles, n=1, seed=13).to(cpu)
        energies = qcl.map(lambda conformer, lane:
                           qcl.scf.rhf(conformer.set_bond(0, 1, 0.96)).to(lane).energy,
                           conformers).to(cpu)
        return qcl.rank(energies)

    result = edited.compile("O", qcl.Cluster.local().cpus[0]).run()
    assert result.values[0].provenance["question"]["conformer"]["smiles"] == "O"
