"""Paths the hardware experiments need: open-rotation LUCJ, embedded QM/MM active spaces,
grid layouts and dynamical decoupling, each against an independent reference."""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

qcl = pytest.importorskip("qcl")

from chemflow.backends.calibration import best_layout
from chemflow.backends.circuits import build_circuit, compile_native
from chemflow.backends.lqcloud import LqcloudBackend, dynamic_decoupling
from chemflow.impls.fci import sector_states
from chemflow.impls.lucj import LucjParams, lucj_state, rotated_integrals
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import SectorEngine
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import CircuitSpec, Sector, StatePrep
from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qcl" / "apps"))

LIH = "Li 0 0 0; H 0 0 1.6"


def _cpu():
    return qcl.Cluster.local().cpus[0]


def _simulator():
    return qcl.Device("sim", "qpu", qubits=16, simulator_host="cpu0",
                      simulator_method="statevector")


def test_open_lucj_state_is_the_rotated_basis_image_of_the_full_state():
    from qiskit.quantum_info import Statevector

    store = ArtifactStore()
    data = build_molecule(LIH, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=5))
    params = LucjParams.random(5, np.random.default_rng(3), scale=0.4)
    full = lucj_state(data.sector, [params])
    open_ = lucj_state(data.sector, [params], final_rotation=False)
    h1, h2 = rotated_integrals(store.get(data.hamiltonian.h1_ref),
                               store.get(data.hamiltonian.h2_ref), params.kparams)
    rotated = data.hamiltonian.model_copy(update={"h1_ref": store.put(h1),
                                                  "h2_ref": store.put(h2)})
    assert SectorEngine(rotated, data.sector, store).energy(open_) == pytest.approx(
        SectorEngine(data.hamiltonian, data.sector, store).energy(full), abs=1e-10)
    states = sector_states(data.sector)
    for final, reference in ((True, full), (False, open_)):
        spec = CircuitSpec(ansatz="lucj", n_orb=5, n_alpha=2, n_beta=2,
                           layers=(tuple(params.to_vector()),), final_rotation=final)
        amplitudes = Statevector.from_instruction(build_circuit(spec)).data[states]
        assert abs(np.vdot(reference, amplitudes)) == pytest.approx(1, abs=1e-10)


def test_dropping_the_final_rotation_cuts_two_qubit_gates():
    params = LucjParams.random(6, np.random.default_rng(1), scale=0.3).to_vector()

    def cz(final):
        spec = CircuitSpec(ansatz="lucj", n_orb=6, n_alpha=3, n_beta=3,
                           layers=(tuple(params),), final_rotation=final)
        return compile_native(build_circuit(spec)).count_ops().get("cz", 0)

    assert cz(False) < 0.7 * cz(True)


def test_layout_search_shortens_lucj_and_keeps_the_logical_bit_order():
    from hardware_analysis import sparse_hellinger
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    params = LucjParams.random(4, np.random.default_rng(2), scale=0.3).to_vector()
    spec = CircuitSpec(ansatz="lucj", n_orb=4, n_alpha=2, n_beta=2,
                       layers=(tuple(params),), final_rotation=False)
    circuit = build_circuit(spec)
    searched = compile_native(circuit)
    plain = compile_native(circuit, search=False)
    assert searched.count_ops()["cz"] < plain.count_ops()["cz"]
    ideal = Statevector.from_instruction(circuit).probabilities_dict()
    exact = {str(int(bits, 2)): probability for bits, probability in ideal.items()
             if probability > 1e-12}
    counts = AerSimulator().run(searched, shots=40000, seed_simulator=3).result().get_counts()
    measured = {str(int(bits.replace(" ", ""), 2)): count for bits, count in counts.items()}
    assert sparse_hellinger(measured, exact) > 0.995


def test_a_new_compiler_is_a_new_job(tmp_path, monkeypatch):
    from chemflow.backends import lqcloud

    backend = LqcloudBackend(state_dir=tmp_path)
    before = backend.record_path(_prep(), 100)
    assert backend.record_path(_prep(), 100, compiler="chain-identity-seed91") != before
    monkeypatch.setattr(lqcloud, "COMPILER_VERSION", "another")
    assert backend.record_path(_prep(), 100) != before


def test_sqd_of_open_lucj_samples_reaches_casci(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 10000\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))

    @qcl.workflow
    def energies(cpu, qpu):
        space = qcl.active_space(qcl.scf.hf(qcl.Molecule(LIH, "sto-3g")).to(cpu),
                                 qcl.Active(4, 5)).to(cpu)
        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        results = {"casci": qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy}
        for name, final in (("full", True), ("open", False)):
            bits = qcl.sample(qcl.kernels.lucj(space, params=amplitudes,
                                               final_rotation=final),
                              shots=4000, seed=5).to(qpu).to(cpu)
            results[name] = qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu)
        return results

    result = energies.compile(_cpu(), _simulator()).run()
    assert result["open"].value == pytest.approx(result["casci"].value, abs=1e-6)
    assert result["open"].value >= result["casci"].value - 1e-9
    assert "LUCJ-rotated" in result["open"].provenance["orbital_basis"]
    assert "orbital_basis" not in result["full"].provenance


def test_final_rotation_must_be_a_boolean():
    @qcl.workflow
    def kernel(cpu):
        space = qcl.active_space(qcl.scf.hf(qcl.Molecule(LIH, "sto-3g")).to(cpu),
                                 qcl.Active(2, 2)).to(cpu)
        return qcl.kernels.lucj(space, final_rotation="no")

    with pytest.raises(qcl.CompileError, match="final_rotation"):
        kernel.compile(_cpu())


def _water_dimer():
    from chemflow import bench

    lines = []
    for index, (element, xpos, ypos, zpos) in enumerate(bench.WATER):
        atom = ("O", "H1", "H2")[index % 3]
        lines.append(f"HETATM{index + 1:5d} {atom:<4} HOH A{index // 3 + 1:4d}    "
                     f"{xpos:8.3f}{ypos:8.3f}{zpos:8.3f}  1.00  0.00          {element:>2}")
    return "\n".join(lines) + "\nEND\n", [[x / 10, y / 10, z / 10] for _, x, y, z in bench.WATER]


def test_embedded_active_space_reproduces_qmmm_casci():
    pytest.importorskip("openmm")
    pdb, positions = _water_dimer()

    @qcl.workflow
    def water(cpu):
        mean_field = qcl.qmmm.scf(pdb, positions, [0, 1, 2]).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(4, 4)).to(cpu)
        return {"rhf_total": mean_field.energy,
                "rhf": qcl.qmmm.energy(pdb, positions, [0, 1, 2]).to(cpu),
                "fci": qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy,
                "casci": qcl.qmmm.energy(pdb, positions, [0, 1, 2], method="casci",
                                         cas=(4, 4)).to(cpu)}

    result = water.compile(_cpu()).run()
    assert result["rhf_total"].value == pytest.approx(result["rhf"].value, abs=1e-8)
    assert result["fci"].value == pytest.approx(result["casci"].value, abs=1e-8)


def test_grid_layouts_are_connected_chains_and_chains_keep_their_windows():
    rows, cols = 4, 5
    index = {(row, col): row * cols + col for row in range(rows) for col in range(cols)}
    edges = [[index[row, col], index[row, col + 1]] for row in range(rows)
             for col in range(cols - 1)]
    edges += [[index[row, col], index[row + 1, col]] for row in range(rows - 1)
              for col in range(cols)]
    metrics = [{"id": qubit, "measure_f0": 0.99, "measure_f1": 0.98} for qubit in range(20)]
    couplers = [{"qubits": edge, "cz_fidelity": 0.99} for edge in edges]
    grid = {"qubits": 20, "topology": {"coupling_map": edges},
            "properties": {"qubit_metrics": metrics, "coupler_metrics": couplers}}
    layout = best_layout(grid, 16)
    coupled = {tuple(sorted(edge)) for edge in edges}
    assert len(set(layout)) == 16
    assert all(tuple(sorted(pair)) in coupled for pair in pairwise(layout))
    chain = {"qubits": 6, "topology": {"coupling_map": [[i, i + 1] for i in range(5)]},
             "properties": {"qubit_metrics": metrics[:6], "coupler_metrics": [
                 {"qubits": [i, i + 1], "cz_fidelity": 0.9 if i == 0 else 0.99}
                 for i in range(5)]}}
    assert best_layout(chain, 4) == [1, 2, 3, 4]


def _prep(final_rotation=True):
    sector = Sector(n_alpha=1, n_beta=1, n_orb=2)
    circuit = CircuitSpec(ansatz="lucj", n_orb=2, n_alpha=1, n_beta=1,
                          layers=((0.1, 0.2, 0.3),), final_rotation=final_rotation)
    return StatePrep(state_ref="s", encoding_id="e", hamiltonian_id="h", recipe="lucj",
                     sector=sector, circuit=circuit)


def test_decoupling_and_open_rotation_are_distinct_jobs(tmp_path):
    backend = LqcloudBackend(state_dir=tmp_path)
    cpmg = dynamic_decoupling("cpmg")
    assert dynamic_decoupling("none") is None and cpmg["dd_type"] == "CPMG"
    paths = {backend.record_path(_prep(), 100), backend.record_path(_prep(), 100, 0, cpmg),
             backend.record_path(_prep(False), 100)}
    assert len(paths) == 3
    structures = {backend._structure(_prep(), 100, 0), backend._structure(_prep(), 100, 0, cpmg),
                  backend._structure(_prep(False), 100, 0)}
    assert len(structures) == 3
    with pytest.raises(ValueError):
        dynamic_decoupling("xy8")


def test_qpu_devices_accept_only_known_decoupling_modes():
    qpu = qcl.Device("mq02", "qpu", qubits=24, provider="lqcloud", backend="MQ02")
    assert qpu.with_options(dd="cpmg").options["dd"] == "cpmg"
    assert qpu.with_options(compile="plain").options["compile"] == "plain"
    with pytest.raises(qcl.CompileError):
        qpu.with_options(dd="xy8")
    with pytest.raises(qcl.CompileError):
        qpu.with_options(compile="fast")
