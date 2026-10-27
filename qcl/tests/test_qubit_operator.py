import pytest

import qcl


@qcl.workflow
def mapped_h2(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)


def _inputs():
    return qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), qcl.Active(2, 2), qcl.Cluster.local().cpus[0]


def test_jordan_wigner_operator_is_typed_and_has_four_qubits():
    result = mapped_h2.compile(*_inputs()).run()
    assert isinstance(result, qcl.QubitOperator)
    assert result.n_qubits == 4
    assert result.terms
    assert all(len(word) == 4 for word in result.terms)
    assert result.provenance["implementation"] == "qubit_hamiltonian/chemflow-jw"


def test_fci_rejects_qubit_operator_and_mapping_needs_placement():
    @qcl.workflow
    def wrong(molecule, active, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        qubit = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
        return qcl.fci.solve(qubit, space.sector).to(cpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        wrong.compile(*_inputs())

    @qcl.workflow
    def unplaced(molecule, active, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        qcl.qubit_hamiltonian(space.hamiltonian)
        return mean_field.energy

    with pytest.raises(qcl.CompileError, match="QCL-E201"):
        unplaced.compile(*_inputs())

    @qcl.workflow
    def unsupported_mapping(molecule, active, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.qubit_hamiltonian(space.hamiltonian,
                                     mapping="bravyi_kitaev").to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        unsupported_mapping.compile(*_inputs())


def test_vqe_accepts_explicit_qubit_operator(tmp_path, monkeypatch):
    @qcl.workflow
    def mapped_vqe(molecule, active, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        qubit = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
        return qcl.vqe.sampled(qubit, space.sector,
                               max_evaluations=3, shots_per_group=128,
                               n_groups=5, seed=7).to(qpu)

    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1920\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    plan = mapped_vqe.compile(*_inputs(), qcl.Cluster.local().qpus[0])
    assert plan.estimate()["shots"] == 1920
    result = plan.run(timeout_s=120)
    assert result.value < -1.0
