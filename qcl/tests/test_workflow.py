import builtins
import json
import subprocess
from hashlib import sha256

import pytest
from pyscf import fci, gto, scf

import qcl


@qcl.workflow
def rhf_energy(molecule, device):
    return qcl.scf.rhf(molecule).to(device).energy


def test_rhf_workflow_compiles_and_runs_against_pyscf():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    cpu = qcl.Cluster.local().cpus[0]

    plan = rhf_energy.compile(molecule, cpu)
    explanation = plan.explain(format="json")
    assert explanation["operations"][0]["op"] == "scf.rhf"
    assert explanation["operations"][0]["device"] == cpu.id
    assert explanation["operations"][0]["implementation"] == "scf.rhf/chemflow-pyscf"

    result = plan.run()
    reference = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis, verbose=0)).run()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-10)
    assert result.unit == "hartree"
    assert result.variance == 0
    assert result.provenance["implementation"] == "scf.rhf/chemflow-pyscf"
    assert result.provenance["device"] == cpu.id


def test_cpu_plan_estimate_does_not_import_cloud_qpu_backend(monkeypatch):
    plan = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Device("cpu0", "cpu"))
    original_import = builtins.__import__

    def import_without_cloud(name, *args, **kwargs):
        if name == "chemflow.backends.lqcloud":
            raise ModuleNotFoundError("QPU SDK is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cloud)
    assert plan.estimate()["shots"] == 0
    assert plan.estimate()["qpu_cost"] == 0


def test_standalone_plan_enforces_cpu_core_time_budget():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(ValueError, match="cpu_core_s"):
        qcl.Budget(cpu_core_s=0)
    plan = rhf_energy.compile(molecule, cpu, budget=qcl.Budget(cpu_core_s=100))
    plan.run()
    assert 0 < plan.cpu_core_s < 100
    limited = rhf_energy.compile(molecule, cpu, budget=qcl.Budget(cpu_core_s=1e-6))
    with pytest.raises(RuntimeError, match="CPU core-time"):
        limited.run()
    assert limited.cpu_core_s > 0


def test_unset_cpu_budget_preserves_existing_plan_hash():
    plan = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Device("cpu0", "cpu"), budget=qcl.Budget(wall_s=30))
    graph = plan.graph_json()
    for node in graph["nodes"]:
        node.pop("source", None)
        node.pop("source_span", None)
    legacy_budget = {"qpu_shots": None, "qpu_s": None, "qpu_cost": None, "wall_s": 30}
    encoded = json.dumps({"graph": graph, "budget": legacy_budget},
                         sort_keys=True, separators=(",", ":")).encode()
    assert plan.hash == sha256(encoded).hexdigest()


def test_cpu_budget_does_not_mislabel_unrelated_worker_crash(monkeypatch):
    plan = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Device("cpu0", "cpu"), budget=qcl.Budget(cpu_core_s=100))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, -11, "", "segmentation fault"))
    with pytest.raises(RuntimeError, match="worker failed: segmentation fault"):
        plan.run()


def test_unplaced_compute_is_a_compile_error():
    @qcl.workflow
    def unplaced(molecule):
        qcl.scf.rhf(molecule)
        return molecule

    with pytest.raises(qcl.CompileError) as error:
        unplaced.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"))
    assert any(item["code"] == "QCL-E201" for item in error.value.diagnostics)


def test_diagnostic_and_ir_include_full_source_span():
    @qcl.workflow
    def unplaced(molecule):
        return qcl.scf.rhf(molecule)

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        unplaced.compile(molecule)
    diagnostic = next(item for item in error.value.diagnostics
                      if item["code"] == "QCL-E201")
    span = diagnostic["source_span"]
    assert span["file"].endswith("test_workflow.py")
    assert span["start_line"] == span["end_line"]
    assert span["end_column"] > span["start_column"]
    assert "qcl.scf.rhf(molecule)" in diagnostic["expression"]
    plan = rhf_energy.compile(molecule, qcl.Cluster.local().cpus[0])
    scf_node = next(node for node in plan.graph_json()["nodes"]
                    if node["op"] == "scf.rhf")
    assert scf_node["source_span"]["file"].endswith("test_workflow.py")
    assert scf_node["source_span"]["end_column"] > 0


def test_plan_hash_ignores_source_location():
    @qcl.workflow
    def first(molecule, cpu):
        return qcl.scf.rhf(molecule).to(cpu).energy

    @qcl.workflow
    def second(molecule, cpu):
        return qcl.scf.rhf(molecule).to(cpu).energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    assert first.compile(molecule, cpu).hash == second.compile(molecule, cpu).hash


def test_wrong_device_never_falls_back_to_cpu():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    qpu = qcl.Cluster.local().qpus[0]

    with pytest.raises(qcl.CompileError) as error:
        rhf_energy.compile(molecule, qpu)
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_rhf_preserves_coordinate_units():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr")
    result = rhf_energy.compile(molecule, qcl.Cluster.local().cpus[0]).run()
    reference = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                             unit="Bohr", verbose=0)).run()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-10)


def test_rhf_rejects_open_shell_at_compile_time():
    molecule = qcl.Molecule("H 0 0 0", basis="sto-3g", spin=1)
    with pytest.raises(qcl.CompileError) as error:
        rhf_energy.compile(molecule, qcl.Cluster.local().cpus[0])
    assert any(item["code"] == "QCL-E101" for item in error.value.diagnostics)


def test_nested_workflow_is_inlined():
    @qcl.workflow
    def nested(molecule, device):
        return rhf_energy(molecule, device)

    plan = nested.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                          qcl.Cluster.local().cpus[0])
    assert len(plan.explain(format="json")["operations"]) == 1


def test_computed_python_branch_reports_source():
    @qcl.workflow
    def invalid(molecule, device):
        energy = qcl.scf.rhf(molecule).to(device).energy
        if energy:
            return energy
        return energy

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                        qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E210"
    assert "test_workflow.py" in error.value.diagnostics[0]["source"]


@qcl.workflow
def fci_energy(molecule, active, device):
    mean_field = qcl.scf.hf(molecule).to(device)
    space = qcl.active_space(mean_field, active).to(device)
    return qcl.fci.solve(space.hamiltonian, sector=space.sector).to(device).energy


def test_fci_pipeline_reuses_chemflow_scientific_adapters():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    plan = fci_energy.compile(molecule, qcl.Active(electrons=2, orbitals=2), cpu)
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == [
        "scf.hf", "active_space", "fci.solve",
    ]

    result = plan.run()
    reference_mf = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                                verbose=0)).run()
    reference_energy = fci.FCI(reference_mf).kernel()[0]
    assert result.value == pytest.approx(reference_energy, abs=1e-9)
    assert result.unit == "hartree"
    assert result.provenance["method"]["family"] == "FCI"


def test_active_space_too_large_reports_compile_error():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        fci_energy.compile(molecule, qcl.Active(electrons=2, orbitals=3),
                           qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E105"


def test_fci_rejects_mean_field_result_as_operator():
    @qcl.workflow
    def invalid(molecule, device):
        mean_field = qcl.scf.hf(molecule).to(device)
        return qcl.fci.solve(mean_field, sector=mean_field).to(device).energy

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                        qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E101"


def test_hf_selects_rohf_for_open_shell_species():
    molecule = qcl.Molecule("H 0 0 0", basis="sto-3g", spin=1)
    result = fci_energy.compile(molecule, qcl.Active(electrons=1, orbitals=1),
                                qcl.Cluster.local().cpus[0]).run()
    reference = scf.ROHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                              spin=1, verbose=0)).run()
    assert result.value == pytest.approx(reference.e_tot, abs=1e-10)
    assert result.provenance["method"]["family"] == "FCI"


def test_fci_rejects_mismatched_active_space_projections():
    @qcl.workflow
    def mismatched(molecule, device):
        mean_field = qcl.scf.hf(molecule).to(device)
        first = qcl.active_space(mean_field, qcl.Active(2, 2)).to(device)
        second = qcl.active_space(mean_field, qcl.Active(2, 2)).to(device)
        return qcl.fci.solve(first.hamiltonian, second.sector).to(device).energy

    with pytest.raises(qcl.CompileError) as error:
        mismatched.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                           qcl.Cluster.local().cpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_static_python_branch_is_traced_normally():
    @qcl.workflow
    def static(molecule, device):
        if molecule.spin == 0:
            return qcl.scf.rhf(molecule).to(device).energy
        return qcl.scf.hf(molecule).to(device).energy

    plan = static.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                          qcl.Cluster.local().cpus[0])
    assert [item["op"] for item in plan.explain(format="json")["operations"]] == ["scf.rhf"]


def test_plan_estimate_marks_unmodelled_costs_unknown():
    plan = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                              qcl.Cluster.local().cpus[0])
    estimate = plan.estimate()
    assert estimate["wall_time_s"] is None
    assert estimate["peak_memory_bytes"]["cpu0"] is None
    assert estimate["bytes_moved"] == 0
    assert estimate["shots"] == 0
    assert estimate["qpu_cost"] == 0.0
    assert "wall_time_s" in estimate["unknown"]


def test_plan_profile_uses_measured_worker_timings():
    plan = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"),
                              qcl.Cluster.local().cpus[0])
    with pytest.raises(RuntimeError, match="Run"):
        plan.profile()
    plan.run()
    profile = plan.profile()
    assert len(profile) == 1
    assert profile[0]["op"] == "scf.rhf"
    assert profile[0]["device"] == "cpu0"
    assert profile[0]["duration_s"] > 0
    assert profile[0]["cpu_duration_s"] >= 0


def test_plan_validate_compares_compiled_reduced_instance_to_reference():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    candidate = rhf_energy.compile(molecule, cpu)
    reference = fci_energy.compile(molecule, qcl.Active(2, 2), cpu)
    validation = candidate.validate(reference=reference, on="H2/sto-3g",
                                    tolerance=0.001)
    assert validation["on"] == "H2/sto-3g"
    assert validation["candidate"]["unit"] == "hartree"
    assert validation["absolute_error"] > 0.001
    assert validation["within_tolerance"] is False
    assert validation["candidate"]["plan_hash"] == candidate.hash
    assert validation["reference"]["plan_hash"] == reference.hash


def test_plan_error_budget_reports_observations_without_a_false_bound():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    candidate = rhf_energy.compile(molecule, cpu)
    reference = fci_energy.compile(molecule, qcl.Active(2, 2), cpu)
    before = candidate.error_budget()
    assert before["statistical"]["variance"] is None
    assert before["systematic"]["upper_bound"] is None
    validation = candidate.validate(reference=reference, on="H2/sto-3g")
    budget = candidate.error_budget()
    assert budget["plan_hash"] == candidate.hash
    assert budget["unit"] == "hartree"
    assert budget["statistical"] == {"variance": 0.0, "std_dev": 0.0}
    assert budget["systematic"]["max_observed_absolute_error"] == pytest.approx(
        validation["absolute_error"])
    assert budget["systematic"]["validation_observations"][0]["reference_plan_hash"] == (
        reference.hash)
    assert budget["total"]["upper_bound"] is None


def test_plan_validate_rejects_different_molecules_before_run(monkeypatch):
    cpu = qcl.Device("cpu0", "cpu")
    left = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), cpu)
    right = rhf_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.80", "sto-3g"), cpu)
    monkeypatch.setattr(left, "run", lambda: pytest.fail("candidate should not run"))
    with pytest.raises(qcl.CompileError) as error:
        left.validate(reference=right, on="different geometries")
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_estimate_arithmetic_units_and_structured_results():
    @qcl.workflow
    def energies(molecule, device):
        energy = qcl.scf.rhf(molecule).to(device).energy
        return {"hartree": energy, "difference": energy - energy,
                "converted": [energy.in_units("eV")]}

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = energies.compile(molecule, qcl.Device("cpu0", "cpu")).run()
    assert result["difference"].value == pytest.approx(0.0)
    assert result["difference"].variance == 0.0
    assert result["hartree"].unit == "hartree"
    assert result["converted"][0].unit == "eV"
    assert result["converted"][0].value == pytest.approx(
        result["hartree"].value * 27.211386245988, rel=1e-8)


def test_estimate_arithmetic_rejects_unit_and_model_mismatch():
    @qcl.workflow
    def wrong_units(molecule, device):
        energy = qcl.scf.rhf(molecule).to(device).energy
        return energy - energy.in_units("eV")

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        wrong_units.compile(molecule, qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E102"

    @qcl.workflow
    def wrong_model(first, second, device):
        left = qcl.scf.rhf(first).to(device).energy
        right = qcl.scf.rhf(second).to(device).energy
        return left - right

    other = qcl.Molecule("H 0 0 0; H 0 0 0.80", "6-31g")
    with pytest.raises(qcl.CompileError) as error:
        wrong_model.compile(molecule, other, qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E103"


def test_frontier_orbitals_and_charge_sectors_match_independent_pyscf():
    import numpy as np
    from pyscf import ao2mo, mcscf

    @qcl.workflow
    def frontier(molecule, device):
        mean_field = qcl.scf.rhf(molecule).to(device)
        active = qcl.active_space(mean_field, qcl.Active(2, 2)).to(device)
        energies = {delta: qcl.fci.solve(
            active.hamiltonian, active.sector.add_electrons(delta)
        ).to(device).energy for delta in (-1, 0, 1)}
        return (mean_field.homo, mean_field.lumo,
                energies[-1] - energies[0], energies[0] - energies[1])

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = frontier.compile(molecule, qcl.Device("cpu0", "cpu")).run()
    reference_mf = scf.RHF(gto.M(atom=molecule.atoms, basis=molecule.basis,
                                verbose=0)).run()
    cas = mcscf.CASCI(reference_mf, 2, 2)
    h1, ecore = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), 2)
    energies = {count: fci.direct_spin1.FCI().kernel(h1, eri, 2, sector)[0] + ecore
                for count, sector in ((-1, (1, 0)), (0, (1, 1)), (1, (2, 1)))}
    occupied = np.flatnonzero(reference_mf.mo_occ > 0)
    assert result[0].value == pytest.approx(reference_mf.mo_energy[occupied[-1]], abs=1e-10)
    assert result[1].value == pytest.approx(reference_mf.mo_energy[occupied[-1] + 1], abs=1e-10)
    assert result[2].value == pytest.approx(energies[-1] - energies[0], abs=1e-9)
    assert result[3].value == pytest.approx(energies[0] - energies[1], abs=1e-9)
    assert all(item.unit == "hartree" for item in result)


def test_charge_sector_capacity_is_checked_while_compiling():
    @qcl.workflow
    def impossible(molecule, device):
        mean_field = qcl.scf.rhf(molecule).to(device)
        active = qcl.active_space(mean_field, qcl.Active(2, 2)).to(device)
        return qcl.fci.solve(active.hamiltonian,
                             active.sector.add_electrons(3)).to(device).energy

    with pytest.raises(qcl.CompileError) as error:
        impossible.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E105"


def test_shifted_sector_can_follow_an_explicit_device_copy():
    @qcl.workflow
    def copied_sector(molecule, cpu, gpu):
        mean_field = qcl.scf.rhf(molecule).to(cpu)
        active = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        operator = active.hamiltonian.to(gpu)
        sector = active.sector.to(gpu).add_electrons(1)
        return qcl.fci.solve(operator, sector).to(gpu).energy

    plan = copied_sector.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                                 qcl.Device("cpu0", "cpu"), qcl.Device("gpu0", "gpu"))
    assert plan.graph_json()["nodes"][-2]["op"] == "fci.solve"


def test_strict_transfers_requires_explicit_gpu_copy():
    cpu0 = qcl.Device("cpu0", "cpu")
    gpu0 = qcl.Device("gpu0", "gpu")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    active = qcl.Active(electrons=2, orbitals=2)

    @qcl.workflow
    def implicit(molecule, active, first, second):
        mean_field = qcl.scf.hf(molecule).to(first)
        space = qcl.active_space(mean_field, active).to(first)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(second).energy

    plan = implicit.compile(molecule, active, cpu0, gpu0)
    assert any(item["kind"] == "implicit" for item in
               plan.explain(format="json")["transfers"])
    with pytest.raises(qcl.CompileError) as error:
        implicit.compile(molecule, active, cpu0, gpu0, strict_transfers=True)
    assert error.value.diagnostics[0]["code"] == "QCL-E205"

    @qcl.workflow
    def explicit(molecule, active, first, second):
        mean_field = qcl.scf.hf(molecule).to(first)
        space = qcl.active_space(mean_field, active).to(first)
        return qcl.fci.solve(space.hamiltonian.to(second),
                             space.sector.to(second)).to(second).energy

    copied = explicit.compile(molecule, active, cpu0, gpu0, strict_transfers=True)
    assert any(item["kind"] == "explicit" for item in
               copied.explain(format="json")["transfers"])


def test_nonlocal_cpu_handle_is_rejected_before_execution():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        rhf_energy.compile(molecule, qcl.Device("cpu1", "cpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)

    @qcl.workflow
    def copied(molecule, cpu, remote):
        return qcl.scf.rhf(molecule).to(cpu).to(remote).energy

    with pytest.raises(qcl.CompileError) as error:
        copied.compile(molecule, qcl.Device("cpu0", "cpu"), qcl.Device("cpu1", "cpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


@qcl.workflow
def gpu_fci_energy(molecule, active, cpu, gpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = space.hamiltonian.to(gpu)
    sector = space.sector.to(gpu)
    return qcl.fci.solve(operator, sector).to(gpu).energy


def test_gpu_fci_has_explicit_inputs_and_no_cpu_fallback():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    plan = gpu_fci_energy.compile(molecule, qcl.Active(2, 2),
                                  qcl.Device("cpu0", "cpu"), qcl.Device("gpu0", "gpu"),
                                  strict_transfers=True)
    explanation = plan.explain(format="json")
    assert [item["device"] for item in explanation["operations"]] == [
        "cpu0", "cpu0", "gpu0",
    ]
    assert len([item for item in explanation["transfers"]
                if item["kind"] == "explicit" and item["to"] == "gpu0"]) == 2

    @qcl.workflow
    def unsupported(molecule, gpu):
        mean_field = qcl.scf.hf(molecule).to(gpu)
        return qcl.active_space(mean_field, qcl.Active(2, 2)).to(gpu).hamiltonian

    with pytest.raises(qcl.CompileError) as error:
        unsupported.compile(molecule, qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_gpu_fci_agrees_with_cpu_when_cuda_is_available():
    from chemflow.backends.gpu import cuda_available

    if not cuda_available():
        pytest.skip("CUDA device and CuPy are not available")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
    active = qcl.Active(2, 2)
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device("gpu0", "gpu")
    reference = fci_energy.compile(molecule, active, cpu).run()
    actual = gpu_fci_energy.compile(molecule, active, cpu, gpu,
                                    strict_transfers=True).run()
    assert actual.value == pytest.approx(reference.value, abs=1e-8)
    assert actual.provenance["device"] == "gpu0"
    assert actual.provenance["implementation"] == "fci.solve/chemflow-cupy"
