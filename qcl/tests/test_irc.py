import pytest
from qcl._worker import execute

import qcl


@qcl.workflow
def checked_path(transition, reactant, product, cpu):
    pes = qcl.PES(lambda molecule, lane: qcl.scf.hf(molecule).to(lane),
                  gradient=qcl.Analytic())
    path = qcl.irc(pes, transition).to(cpu)
    qcl.require(path.connects(reactant, product))
    return qcl.scf.hf(transition).to(cpu).energy


def _molecules():
    transition = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    reactant = qcl.Molecule("H 0 0 0; H 0 0 0.80", "sto-3g")
    product = qcl.Molecule("H 0 0 0; H 0 0 4.0", "sto-3g")
    return transition, reactant, product


def test_irc_compiles_as_placed_path_and_deferred_connectivity():
    plan = checked_path.compile(*_molecules(), qcl.Cluster.local().cpus[0])
    ops = [node["op"] for node in plan.graph_json()["nodes"]]
    assert "irc" in ops
    assert "path.connects" in ops
    assert "require" in ops


@pytest.mark.parametrize("connected", [True, False])
def test_irc_connectivity_is_checked_at_runtime(monkeypatch, connected):
    from chemflow.domains import paths

    transition, reactant, product = _molecules()
    endpoints = [reactant, product if connected else reactant]
    monkeypatch.setattr(paths, "irc", lambda *_args, **_kwargs: {
        "ts": {"atoms": transition.atoms, "unit": transition.unit},
        "endpoints": [{"atoms": item.atoms, "charge": item.charge,
                       "spin": item.spin, "unit": item.unit} for item in endpoints],
        "connects": None,
    })
    plan = checked_path.compile(transition, reactant, product,
                                qcl.Cluster.local().cpus[0])
    request = {"graph": plan.graph_json(), "output": plan._output.node_id}
    if connected:
        assert execute(request)["result"]["value"] < 0
    else:
        with pytest.raises(ValueError, match="requirement failed"):
            execute(request)


def test_irc_rejects_pes_callback_that_changes_the_geometry():
    @qcl.workflow
    def wrong(transition, cpu):
        other = qcl.Molecule("H 0 0 0; H 0 0 1.2", "sto-3g")
        pes = qcl.PES(lambda _geometry, lane: qcl.scf.hf(other).to(lane))
        return qcl.irc(pes, transition).to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        wrong.compile(_molecules()[0], qcl.Cluster.local().cpus[0])


def test_irc_requires_cpu_placement():
    @qcl.workflow
    def unplaced(transition, cpu):
        pes = qcl.PES(lambda molecule, lane: qcl.scf.hf(molecule).to(lane))
        qcl.irc(pes, transition)
        return qcl.scf.hf(transition).to(cpu).energy

    transition = _molecules()[0]
    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError, match="QCL-E201"):
        unplaced.compile(transition, cpu)

    @qcl.workflow
    def wrong_device(transition, gpu):
        pes = qcl.PES(lambda molecule, lane: qcl.scf.hf(molecule).to(lane))
        return qcl.irc(pes, transition).to(gpu)

    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        wrong_device.compile(transition, qcl.Device("gpu0", "gpu"))


def test_irc_path_has_typed_output():
    @qcl.workflow
    def path_workflow(transition, cpu):
        pes = qcl.PES(lambda molecule, lane: qcl.scf.hf(molecule).to(lane))
        return qcl.irc(pes, transition).to(cpu)

    transition = _molecules()[0]
    plan = path_workflow.compile(transition, qcl.Cluster.local().cpus[0])
    raw = {"ts": {"atoms": transition.atoms, "basis": transition.basis,
                  "charge": 0, "spin": 0, "unit": "angstrom"},
           "endpoints": [], "provenance": {"implementation": "irc/test"}}
    result = plan._decode_output(raw, plan._output, "now")
    assert isinstance(result, qcl.ReactionPath)
    assert result.transition.atoms == transition.atoms
    assert result.provenance["timestamp"] == "now"
