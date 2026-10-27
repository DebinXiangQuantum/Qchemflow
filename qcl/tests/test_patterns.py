import pytest
from pyscf import gto, scf

import qcl


@qcl.workflow
def stability_ranking(molecules, lanes):
    energies = qcl.map(lambda mol, dev: qcl.scf.rhf(mol).to(dev).energy,
                       molecules).to(lanes)
    return qcl.rank(energies)


def test_static_map_and_rank_match_independent_energies():
    molecules = [qcl.Molecule(f"H 0 0 0; H 0 0 {distance}", "sto-3g")
                 for distance in (1.5, 0.74)]
    cpu = qcl.Cluster.local().cpus[0]
    plan = stability_ranking.compile(molecules, [cpu])
    result = plan.run()
    reference = [scf.RHF(gto.M(atom=mol.atoms, basis=mol.basis,
                              verbose=0)).run().e_tot for mol in molecules]
    assert result.order == (1, 0)
    assert [value.value for value in result.values] == pytest.approx(reference, abs=1e-10)
    assert result.unresolved == ()
    assert plan.explain(format="json")["patterns"][0]["op"] == "map"


def test_unplaced_map_reports_compile_error():
    @qcl.workflow
    def unplaced(molecules):
        qcl.map(lambda mol, dev: qcl.scf.rhf(mol).to(dev).energy, molecules)
        return "unused"

    with pytest.raises(qcl.CompileError) as error:
        unplaced.compile([qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")])
    assert error.value.diagnostics[0]["code"] == "QCL-E201"


def test_rank_rejects_mixed_basis():
    @qcl.workflow
    def incompatible(molecules, cpu):
        values = qcl.map(lambda mol, dev: qcl.scf.rhf(mol).to(dev).energy,
                         molecules).to(cpu)
        return qcl.rank(values)

    with pytest.raises(qcl.CompileError) as error:
        incompatible.compile([qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Molecule("H 0 0 0; H 0 0 0.74", "3-21g")],
                             qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_map_rejects_lane_dependent_scientific_question():
    @qcl.workflow
    def lane_dependent(cpu_a, cpu_b):
        def point(_item, lane):
            distance = 0.74 if lane.options["threads"] == 1 else 1.5
            molecule = qcl.Molecule(f"H 0 0 0; H 0 0 {distance}", "sto-3g")
            return qcl.scf.rhf(molecule).to(lane).energy

        return qcl.map(point, [0, 1]).to([cpu_a, cpu_b])

    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError) as error:
        lane_dependent.compile(cpu.with_options(threads=1),
                               cpu.with_options(threads=2))
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_scan_carries_symbolic_geometry_between_points():
    @qcl.workflow
    def sequential(molecule, cpu):
        def point(previous, distance, lane):
            geometry = previous.set_bond(0, 1, distance)
            energy = qcl.scf.rhf(geometry).to(lane).energy
            return geometry, energy

        return qcl.scan(point, init=molecule, over=[0.8, 1.2]).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = sequential.compile(molecule, cpu)
    result = plan.run()
    reference = [scf.RHF(gto.M(atom=f"H 0 0 0; H 0 0 {distance}",
                              basis="sto-3g", verbose=0)).run().e_tot
                 for distance in (0.8, 1.2)]
    assert [pair[1].value for pair in result] == pytest.approx(reference, abs=1e-9)
    assert plan.explain(format="json")["patterns"][0]["op"] == "scan"
    assert all(isinstance(pair[0], qcl.GeometryResult) for pair in result)


def test_unplaced_scan_reports_compile_error():
    @qcl.workflow
    def unplaced(molecule, cpu):
        qcl.scan(lambda previous, _item, _lane: (previous, previous),
                 init=molecule, over=[1])
        return qcl.scf.rhf(molecule).to(cpu).energy

    with pytest.raises(qcl.CompileError) as error:
        unplaced.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                         qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E201"


def test_deferred_require_checks_computed_frequencies():
    @qcl.workflow
    def guarded(molecule, cpu, expected):
        modes = qcl.frequencies(molecule).to(cpu)
        qcl.require(modes.n_imaginary == expected)
        return qcl.scf.rhf(molecule).to(cpu).energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    assert guarded.compile(molecule, cpu, 0).run().value < 0
    with pytest.raises(RuntimeError, match="QCL requirement failed"):
        guarded.compile(molecule, cpu, 1).run()


def test_require_rejects_false_static_condition():
    @qcl.workflow
    def guarded(molecule, cpu):
        qcl.require(False)
        return qcl.scf.rhf(molecule).to(cpu).energy

    with pytest.raises(qcl.CompileError) as error:
        guarded.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E101"


def test_cond_runs_only_selected_computed_branch():
    @qcl.workflow
    def conditional(probe, first, second, cpu, expected):
        modes = qcl.frequencies(probe).to(cpu)
        return qcl.cond(
            modes.n_imaginary == expected,
            lambda: qcl.scf.rhf(first).to(cpu).energy,
            lambda: qcl.scf.rhf(second).to(cpu).energy,
        )

    probe = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    first = qcl.Molecule("H 0 0 0; H 0 0 0.8", "sto-3g")
    second = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    first_plan = conditional.compile(probe, first, second, cpu, 0)
    second_plan = conditional.compile(probe, first, second, cpu, 1)
    results = (first_plan.run(), second_plan.run())
    reference = [scf.RHF(gto.M(atom=mol.atoms, basis=mol.basis,
                              verbose=0)).run().e_tot for mol in (first, second)]
    assert [value.value for value in results] == pytest.approx(reference, abs=1e-9)
    for plan in (first_plan, second_plan):
        assert [entry["op"] for entry in plan.profile()].count("scf.rhf") == 1


def test_cond_rejects_incompatible_branch_result_types():
    @qcl.workflow
    def incompatible(molecule, cpu):
        modes = qcl.frequencies(molecule).to(cpu)
        return qcl.cond(modes.n_imaginary == 0,
                        lambda: qcl.scf.rhf(molecule).to(cpu).energy,
                        lambda: qcl.frequencies(molecule).to(cpu))

    with pytest.raises(qcl.CompileError) as error:
        incompatible.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                             qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_mean_reduces_mapped_estimates_with_units():
    @qcl.workflow
    def average(molecules, cpu):
        energies = qcl.map(lambda mol, lane: qcl.scf.rhf(mol).to(lane).energy,
                           molecules).to(cpu)
        return qcl.mean(energies)

    molecules = [qcl.Molecule(f"H 0 0 0; H 0 0 {distance}", "sto-3g")
                 for distance in (0.8, 1.2)]
    result = average.compile(molecules, qcl.Cluster.local().cpus[0]).run()
    reference = [scf.RHF(gto.M(atom=mol.atoms, basis=mol.basis,
                              verbose=0)).run().e_tot for mol in molecules]
    assert result.value == pytest.approx(sum(reference) / 2, abs=1e-9)
    assert result.unit == "hartree"
    assert result.variance == 0


def test_mean_rejects_mixed_units():
    @qcl.workflow
    def mixed(molecule, cpu):
        energy = qcl.scf.rhf(molecule).to(cpu).energy
        return qcl.mean([energy, energy.in_units("eV")])

    with pytest.raises(qcl.CompileError) as error:
        mixed.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E102"


def test_cond_skips_unselected_cloud_qpu_submission():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    @qcl.workflow
    def choose(molecule, cpu, first_qpu, second_qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        kernel = qcl.kernels.hf(space)
        modes = qcl.frequencies(molecule).to(cpu)
        return qcl.cond(
            modes.n_imaginary == 0,
            lambda: qcl.sample(kernel, shots=16).to(first_qpu),
            lambda: qcl.sample(kernel, shots=16).to(second_qpu),
        )

    submissions = []

    class FakeBackend:
        def __init__(self, device):
            self.device = device

        def sample(self, _store, prep, shots, seed):
            submissions.append(self.device["id"])
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device=self.device["id"],
                              job_ids=("fake-job",))
            return ExecutionResult(samples, device=self.device["id"],
                                   cost={"qpu_shots": shots}, job_ids=("fake-job",))

    qpus = [qcl.Device(f"mq02-{index}", "qpu", qubits=4, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
            for index in range(2)]
    plan = choose.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Cluster.local().cpus[0], *qpus)
    conditional_samples = [item for item in plan.explain(format="json")["operations"]
                           if item["op"] == "sample"]
    assert [item["guards"][0]["expected"] for item in conditional_samples] == [True, False]
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=FakeBackend)["result"]
    assert result["shots"] == 16
    assert submissions == ["mq02-0"]


def test_compare_reports_order_and_unresolved_tie():
    @qcl.workflow
    def comparison(first, second, cpu):
        left = qcl.scf.rhf(first).to(cpu).energy
        right = qcl.scf.rhf(second).to(cpu).energy
        return qcl.compare(left, right), qcl.compare(left, left)

    first = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    second = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
    ordered, tied = comparison.compile(first, second, qcl.Cluster.local().cpus[0]).run()
    assert ordered.relation == "less"
    assert ordered.difference < 0
    assert ordered.unit == "hartree"
    assert tied.relation == "unresolved"


def test_python_equality_between_computed_values_is_not_static():
    @qcl.workflow
    def invalid(molecule, cpu):
        first = qcl.scf.rhf(molecule).to(cpu).energy
        second = qcl.scf.rhf(molecule).to(cpu).energy
        return qcl.cond(first == second, lambda: first, lambda: second)

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E101"


def test_iterate_stops_after_computed_condition_and_enforces_bound():
    @qcl.workflow
    def converge(molecule, cpu, expected):
        def step(previous, _lane):
            return previous.set_bond(0, 1, 0.8)

        def until(current, lane):
            return qcl.frequencies(current).to(lane).n_imaginary == expected

        return qcl.iterate(step, init=molecule, until=until, max_steps=3).to(cpu)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = converge.compile(molecule, cpu, 0)
    result = plan.run()
    assert isinstance(result, qcl.GeometryResult)
    assert [item["op"] for item in plan.profile()].count("frequencies") == 1
    with pytest.raises(RuntimeError, match="QCL requirement failed"):
        converge.compile(molecule, cpu, 1).run()


def test_iterate_requires_placement_and_positive_bound():
    @qcl.workflow
    def unplaced(molecule):
        qcl.iterate(lambda value, _lane: value,
                    init=molecule, until=lambda _value, _lane: True, max_steps=2)
        return molecule

    with pytest.raises(qcl.CompileError) as error:
        unplaced.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"))
    assert error.value.diagnostics[0]["code"] == "QCL-E201"
    with pytest.raises(ValueError):
        @qcl.workflow
        def invalid(molecule, cpu):
            return qcl.iterate(lambda value, _lane: value, init=molecule,
                               until=lambda _value, _lane: True,
                               max_steps=0).to(cpu)

        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                        qcl.Cluster.local().cpus[0])
