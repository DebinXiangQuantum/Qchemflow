import math

import pytest

import qcl
from chemflow.empirical import evaluate


@qcl.workflow
def solubility(smiles, cpu, melting_point=None):
    if melting_point is None:
        return qcl.solubility.esol(smiles).to(cpu)
    return qcl.solubility.gse(smiles, melting_point).to(cpu)


def test_esol_and_gse_reuse_chemflow_empirical_models():
    cpu = qcl.Cluster.local().cpus[0]
    esol = solubility.compile("CCO", cpu).run()
    gse = solubility.compile("CCO", cpu, 30.0).run()
    assert esol.value == pytest.approx(evaluate("esol", "CCO")["value"], abs=1e-12)
    assert gse.value == pytest.approx(evaluate("gse", "CCO", 30.0)["value"], abs=1e-12)
    assert esol.unit == gse.unit == "log10(mol/L)"
    assert esol.provenance["method"]["family"] == "ESOL"
    assert gse.provenance["validity_status"] in ("inside_domain", "outside_domain")


def test_gse_requires_finite_melting_point_and_cpu():
    cpu = qcl.Cluster.local().cpus[0]
    with pytest.raises(qcl.CompileError) as error:
        solubility.compile("CCO", cpu, math.nan)
    assert error.value.diagnostics[0]["code"] == "QCL-E101"
    with pytest.raises(qcl.CompileError) as error:
        solubility.compile("CCO", qcl.Cluster.local().qpus[0])
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


@qcl.workflow
def critical_temperature(cpu):
    return qcl.superconductivity.allen_dynes(1.0, 100.0, 0.1).to(cpu)


def test_allen_dynes_is_a_conventional_model_not_hubbard_tc():
    result = critical_temperature.compile(qcl.Cluster.local().cpus[0]).run()
    denominator = 1.0 - 0.1 * (1 + 0.62)
    expected = 100 / 1.2 * math.exp(-1.04 * 2 / denominator)
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.unit == "K"
    assert result.provenance["method"]["family"] == "Allen-Dynes"
