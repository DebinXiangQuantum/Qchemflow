"""Relation inference: a plan's outputs carry statically derived target relations."""

import pytest

import qcl

H2 = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
H2_STRETCHED = qcl.Molecule("H 0 0 0; H 0 0 1.5", "sto-3g")


def _cpu():
    return qcl.Cluster.local().cpus[0]


@qcl.workflow
def ladder(molecule, stretched, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
    outputs = {method: qcl.classical.solve(space.hamiltonian, space.sector,
                                           method=method).to(cpu).energy
               for method in qcl.CLASSICAL_METHODS}
    exact = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
    outputs["fci"] = exact
    outputs["sci"] = qcl.sci.solve(space.hamiltonian, space.sector,
                                   select_cutoff=1e-3).to(cpu).energy
    stretched_field = qcl.scf.hf(stretched).to(cpu)
    stretched_space = qcl.active_space(stretched_field, qcl.Active(2, 2)).to(cpu)
    stretched_exact = qcl.fci.solve(stretched_space.hamiltonian,
                                    stretched_space.sector).to(cpu).energy
    outputs["exact_difference"] = stretched_exact - exact
    outputs["hf_difference"] = stretched_field.energy - mean_field.energy
    outputs["hf_sum"] = stretched_field.energy + mean_field.energy
    outputs["negated_hf"] = -1 * mean_field.energy
    outputs["pair"] = [exact, outputs["sci"]]
    return outputs


def test_method_relations_follow_the_primitive_table():
    relations = ladder.compile(H2, H2_STRETCHED, _cpu()).relations()
    assert relations["hf"] == "variational_upper_bound"
    assert relations["mp2"] == relations["ccsd"] == relations["ccsd_t"] == "approximate"
    assert relations["fci"] == "equivalent"
    assert relations["sci"] == "variational_upper_bound"


def test_arithmetic_composes_relations():
    relations = ladder.compile(H2, H2_STRETCHED, _cpu()).relations()
    assert relations["exact_difference"] == "equivalent"
    # The difference of two upper bounds bounds nothing; their sum is still a bound.
    assert relations["hf_difference"] == "approximate"
    assert relations["hf_sum"] == "variational_upper_bound"
    # A negative factor turns an upper bound into an unrepresented lower bound.
    assert relations["negated_hf"] == "approximate"
    assert relations["pair"] == ["equivalent", "variational_upper_bound"]


def test_classical_ladder_matches_pyscf_on_the_active_space():
    from pyscf import cc, fci, gto, mp, scf

    values = ladder.compile(H2, H2_STRETCHED, _cpu()).run()
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    reference = scf.RHF(mol).run()
    assert values["hf"].value == pytest.approx(reference.e_tot, abs=1e-8)
    assert values["mp2"].value == pytest.approx(mp.MP2(reference).run().e_tot, abs=1e-8)
    coupled = cc.CCSD(reference).run()
    assert values["ccsd"].value == pytest.approx(coupled.e_tot, abs=1e-7)
    assert values["ccsd_t"].value == pytest.approx(coupled.e_tot + coupled.ccsd_t(),
                                                   abs=1e-7)
    assert values["fci"].value == pytest.approx(fci.FCI(reference).kernel()[0], abs=1e-8)
    assert values["ccsd"].provenance["implementation"] == "classical.solve/pyscf-ccsd"


def test_classical_solve_rejects_unknown_methods():
    @qcl.workflow
    def unknown(molecule, cpu):
        space = qcl.active_space(qcl.scf.hf(molecule).to(cpu), qcl.Active(2, 2)).to(cpu)
        return qcl.classical.solve(space.hamiltonian, space.sector, method="cisd").to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        unknown.compile(H2, _cpu())


def test_lattice_classical_and_selected_ci_solvers():
    @qcl.workflow
    def lattice(system, sector, cpu):
        operator = qcl.models.hubbard(system, t=1.0, U=4.0)
        particles = qcl.lattice.particles(operator, sector)
        return {"hf": qcl.classical.solve(operator, particles, method="hf").to(cpu).energy,
                "sci": qcl.sci.solve(operator, particles,
                                     select_cutoff=1e-6).to(cpu).energy,
                "ed": qcl.ed.ground_state(operator, sector=sector).to(cpu).energy}

    plan = lattice.compile(qcl.lattice.ring(4), qcl.Sector(n=4, sz=0), _cpu())
    assert plan.relations() == {"hf": "variational_upper_bound",
                                "sci": "variational_upper_bound", "ed": "equivalent"}
    values = plan.run()
    assert values["sci"].value == pytest.approx(values["ed"].value, abs=1e-8)
    assert values["hf"].value > values["ed"].value


def test_thermal_corrections_add_to_any_electronic_family():
    @qcl.workflow
    def corrected(molecule, cpu):
        geometry = qcl.optimize(molecule).to(cpu)
        space = qcl.active_space(qcl.scf.hf(geometry).to(cpu), qcl.Active(2, 2)).to(cpu)
        electronic = qcl.classical.solve(space.hamiltonian, space.sector,
                                         method="ccsd").to(cpu).energy
        modes = qcl.frequencies(geometry).to(cpu)
        thermal = qcl.thermo.rrho(geometry, modes, temperature_K=298.15,
                                  pressure_Pa=101325.0, quantity="enthalpy").to(cpu)
        return electronic + thermal

    plan = corrected.compile(H2, _cpu())
    assert plan.relations() == "approximate"
