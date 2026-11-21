"""QCL primitives that replace the legacy domain packs, and the QCL benchmarks.

QM/MM energies (qmmm.energy), embedded wave-function theory (embedding.ewf), ligand
descriptors and alerts (drug.descriptors, drug.filter) and retrosynthesis
(synthesis.routes) compile like every other QCL operation: the compiler states their
relations and the worker runs the ChemFlow kernels.
"""

import pytest

qcl = pytest.importorskip("qcl")

from chemflow import bench

H4 = "H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7"
LIBRARY = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
           "paracetamol": "CC(=O)Nc1ccc(O)cc1", "benzylidene_rhodanine": "O=C1NC(=S)SC1=Cc1ccccc1",
           "benzoquinone": "O=C1C=CC(=O)C=C1"}


def _cpu():
    return qcl.Cluster.local().cpus[0]


def _run(workflow, *args, **options):
    plan = workflow.compile(*args)
    return plan.run(**options), plan


def test_the_primitives_declare_their_relations():
    table = qcl.primitives()
    for op in ("embedding.ewf", "drug.descriptors", "drug.filter", "synthesis.routes"):
        assert table[op]["kind"] == "compute" and table[op]["devices"] == ["cpu"], op
    # GPU FCI and GPU-sampled SQD solve the QM/MM CASCI region on a GPU.
    assert table["qmmm.energy"]["devices"] == ["cpu", "gpu"]
    assert table["qmmm.energy"]["output"] == "estimate"
    assert table["drug.filter"]["output"] == "record"


def _ewf_workflow(atoms, active, baths, solver="fci"):
    @qcl.workflow
    def ewf(mol, cpu):
        space = qcl.active_space(qcl.scf.hf(mol).to(cpu), active).to(cpu)
        outputs = {bath: qcl.embedding.ewf(space.hamiltonian, space.sector, bath=bath,
                                           solver=solver).to(cpu) for bath in baths}
        outputs["fci"] = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        return outputs

    return ewf.compile(qcl.Molecule(atoms, "sto-3g"), _cpu())


def test_ewf_full_bath_with_fci_fragments_is_exact_and_truncated_baths_are_not():
    plan = _ewf_workflow(H4, qcl.Active(4, 4), ("full", "dmet", "bno"))
    assert plan.relations() == {"full": "equivalent", "dmet": "approximate",
                                "bno": "approximate", "fci": "equivalent"}
    energies = plan.run()
    assert energies["full"].value == pytest.approx(energies["fci"].value, abs=1e-8)
    assert energies["full"].provenance["implementation"] == "embedding.ewf/chemflow-pyscf"
    assert energies["full"].provenance["diagnostics"]["n_fragments"] == 4
    assert abs(energies["dmet"].value - energies["fci"].value) < 0.05


def test_ewf_with_a_ccsd_solver_is_approximate_even_with_the_full_bath():
    plan = _ewf_workflow(H4, qcl.Active(4, 4), ("full",), solver="ccsd")
    assert plan.relations()["full"] == "approximate"


def test_ewf_needs_the_full_orbital_space():
    plan = _ewf_workflow(H4, qcl.Active(2, 2), ("full",))
    with pytest.raises(RuntimeError, match="CF-E020"):
        plan.run()


def test_ewf_rejects_unknown_baths_at_compile_time():
    with pytest.raises(qcl.CompileError) as caught:
        _ewf_workflow(H4, qcl.Active(4, 4), ("iao",))
    assert caught.value.diagnostics[0]["code"] == "QCL-E104"


def test_qmmm_water_dimer_matches_an_independent_embedded_pyscf_oracle():
    pytest.importorskip("openmm")
    report = bench.run_benchmark("qmmm", "water_dimer_qmmm")
    assert report["passed"], report
    assert report["results"][0]["relation"] == "variational_upper_bound"


def _water_pdb():
    lines = []
    for index, (element, xpos, ypos, zpos) in enumerate(bench.WATER):
        atom = ("O", "H1", "H2")[index % 3]
        lines.append(f"HETATM{index + 1:5d} {atom:<4} HOH A{index // 3 + 1:4d}    "
                     f"{xpos:8.3f}{ypos:8.3f}{zpos:8.3f}  1.00  0.00          {element:>2}")
    return "\n".join(lines) + "\nEND\n", [[x / 10, y / 10, z / 10] for _, x, y, z in bench.WATER]


def test_qmmm_casci_is_equivalent_within_its_model_and_below_rhf():
    pytest.importorskip("openmm")
    pdb, positions = _water_pdb()

    @qcl.workflow
    def water(cpu):
        return {method: qcl.qmmm.energy(pdb, positions, [0, 1, 2], method=method,
                                        cas=(4, 4) if method == "casci" else None).to(cpu)
                for method in ("rhf", "casci")}

    energies, plan = _run(water, _cpu())
    assert plan.relations() == {"rhf": "variational_upper_bound", "casci": "equivalent"}
    assert energies["casci"].value < energies["rhf"].value
    assert energies["rhf"].provenance["question"]["embedding"] == "electrostatic"


@pytest.mark.parametrize("options,message", [
    ({"method": "mp2"}, "Unknown QM/MM method"),
    ({"method": "casci"}, "CASCI needs cas"),
    ({"method": "rhf", "cas": (2, 2)}, "CASCI needs cas"),
    ({"nonbonded": "Spherical"}, "nonbonded"),
])
def test_qmmm_options_are_checked_at_compile_time(options, message):
    pdb, positions = _water_pdb()

    @qcl.workflow
    def water(cpu):
        return qcl.qmmm.energy(pdb, positions, [0, 1, 2], **options).to(cpu)

    with pytest.raises(qcl.CompileError, match=message):
        water.compile(_cpu())


def test_qmmm_region_must_name_unique_atoms():
    pdb, positions = _water_pdb()

    @qcl.workflow
    def water(cpu):
        return qcl.qmmm.energy(pdb, positions, [0, 0, 7]).to(cpu)

    with pytest.raises(qcl.CompileError, match="unique atom indices"):
        water.compile(_cpu())


def _screen(rules, ligands):
    @qcl.workflow
    def funnel(cpu):
        return qcl.drug.filter(ligands, rules=rules).to(cpu)

    return funnel.compile(_cpu()).run()


def test_ligand_funnel_flags_the_literature_interference_compounds():
    pytest.importorskip("rdkit")
    report = bench.run_benchmark("drug", "ligand_funnel")
    assert report["passed"], report


@pytest.mark.parametrize("rules,ligands,passed,flagged,prefix", [
    (("lipinski",), {"aspirin": LIBRARY["aspirin"], "large": "C" * 50}, ["aspirin"],
     {"large"}, "lipinski:"),
    (("pains",), LIBRARY, ["aspirin", "ibuprofen", "paracetamol"],
     {"benzylidene_rhodanine", "benzoquinone"}, "pains:"),
    (("brenk",), {"ethanol": "CCO", "quinone": LIBRARY["benzoquinone"]}, ["ethanol"],
     {"quinone"}, "brenk:"),
])
def test_each_ligand_rule_applies_on_its_own(rules, ligands, passed, flagged, prefix):
    pytest.importorskip("rdkit")
    result = _screen(rules, ligands)
    assert sorted(result.data["passed"]) == passed
    assert set(result.data["flagged"]) == flagged
    assert all(reason.startswith(prefix) for reasons in result.data["flagged"].values()
               for reason in reasons)
    assert all(len(entry["inchikey"]) == 27 for entry in result.data["ligands"].values())
    assert result.provenance["validity_flags"] == [
        "Rule-based alerts; not assay outcomes or efficacy"]


def test_ligand_descriptors():
    pytest.importorskip("rdkit")

    @qcl.workflow
    def descriptors(cpu):
        return qcl.drug.descriptors({"butanol": "CCCCO"}).to(cpu)

    result = descriptors.compile(_cpu()).run()
    assert result.data["descriptors"]["butanol"]["mw"] == pytest.approx(74.123, abs=0.001)


@pytest.mark.parametrize("ligands,rules", [({}, ("pains",)), ({"x": "CCO"}, ("toxicity",))])
def test_ligand_screens_reject_invalid_inputs_at_compile_time(ligands, rules):
    @qcl.workflow
    def funnel(cpu):
        return qcl.drug.filter(ligands, rules=rules).to(cpu)

    with pytest.raises(qcl.CompileError):
        funnel.compile(_cpu())


def test_retrosynthesis_without_its_environment_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMFLOW_AIZYNTH_HOME", str(tmp_path))

    @qcl.workflow
    def routes(cpu):
        return qcl.synthesis.routes("CC(=O)Nc1ccc(O)cc1", time_limit_s=5).to(cpu)

    with pytest.raises(RuntimeError, match="CF-E080"):
        routes.compile(_cpu()).run()


def test_retrosynthesis_options_are_checked_at_compile_time():
    @qcl.workflow
    def routes(cpu):
        return qcl.synthesis.routes("CCO", iterations=0).to(cpu)

    with pytest.raises(qcl.CompileError):
        routes.compile(_cpu())


def test_benchmarks_are_listed_and_unknown_ones_refused():
    assert ("materials", "h6_ewf_full_bath") in bench.list_benchmarks()
    assert len(bench.list_benchmarks()) == 10
    with pytest.raises(ValueError, match="Unknown benchmark"):
        bench.run_benchmark("dynamics", "aim-design-bath5")


@pytest.mark.parametrize("pack,name", [("materials", "hubbard_dimer"),
                                       ("materials", "hubbard_ring6"),
                                       ("reactions", "h2_atomization"),
                                       ("materials", "h6_ewf_full_bath")])
def test_screen_benchmarks_match_their_oracles(pack, name):
    report = bench.run_benchmark(pack, name)
    assert report["passed"], report
