"""One-, two- and three-centre integrals and Cholesky factors."""

from qcl._impls import implements
from qcl._impls._common import (
    _molecule_identity,
)


@implements("int2e")
def _op_int2e(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from pyscf import gto

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Molecular int2e currently requires a CPU")
    molecule = inputs[0]
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    values[identifier] = {"eri": system.intor("int2e"), "device": device["id"]}


@implements("int1e")
def _op_int1e(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    from pyscf import gto

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("One-electron integrals currently require a CPU")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    kind = node["params"]["kind"]
    integrals = {"overlap": "int1e_ovlp", "kinetic": "int1e_kin",
                 "nuclear": "int1e_nuc", "dipole": "int1e_r",
                 "quadrupole": "int1e_rr"}
    array = system.intor(integrals[kind])
    if kind == "quadrupole":
        array = array.reshape(3, 3, system.nao_nr(), system.nao_nr())
    values[identifier] = {
        "array": array, "device": device["id"],
        "unit": {"overlap": "dimensionless", "kinetic": "hartree",
                 "nuclear": "hartree", "dipole": "bohr",
                 "quadrupole": "bohr^2"}[kind],
        "layout": "C", "provenance": {
            "question": {"property": "one_electron_integral", "kind": kind,
                         "molecule": molecule},
            "method": {"family": "AO integral", "basis": molecule["basis"]},
            "implementation": "int1e/chemflow-pyscf", "device": device["id"],
            "versions": {"pyscf": pyscf.__version__},
        },
    }


@implements("int3c")
def _op_int3c(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    from pyscf import df, gto

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Three-center integrals currently require a CPU")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    auxbasis = node["params"]["auxbasis"]
    auxiliary = df.make_auxmol(system, auxbasis)
    array = df.incore.aux_e2(system, auxiliary, aosym="s1")
    values[identifier] = {
        "array": array, "device": device["id"], "unit": "hartree",
        "layout": "C", "provenance": {
            "question": {"property": "three_center_integral",
                         "molecule": molecule},
            "method": {"family": "AO density fitting integral",
                       "basis": molecule["basis"], "auxbasis": auxbasis},
            "implementation": "int3c/chemflow-pyscf", "device": device["id"],
            "versions": {"pyscf": pyscf.__version__},
        },
    }


@implements("cholesky")
def _op_cholesky(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    import numpy as np
    from pyscf import gto, lib

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Cholesky ERI factors currently require a CPU")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    nao = system.nao_nr()
    eri = system.intor("int2e").reshape(nao * nao, nao * nao)
    tolerance = node["params"]["tolerance"]
    upper, pivots, rank = lib.pivoted_cholesky(eri, tol=tolerance)
    factors = np.ascontiguousarray(
        upper[:rank, np.argsort(pivots)].reshape(rank, nao, nao))
    residual = float(np.max(np.abs(
        eri - factors.reshape(rank, nao * nao).T @
        factors.reshape(rank, nao * nao))))
    values[identifier] = {
        "array": factors, "device": device["id"],
        "unit": "hartree^1/2", "layout": "C", "provenance": {
            "question": {"property": "cholesky_eri_factors",
                         "molecule": molecule},
            "method": {"family": "pivoted Cholesky ERI",
                       "basis": molecule["basis"],
                       "tolerance": tolerance,
                       "max_abs_reconstruction_error": residual},
            "implementation": "cholesky/pyscf-pivoted",
            "device": device["id"],
            "versions": {"pyscf": pyscf.__version__},
        },
    }
