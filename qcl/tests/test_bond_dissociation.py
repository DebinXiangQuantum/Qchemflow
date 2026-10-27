import pytest
from pyscf import fci, gto, scf

import qcl


@qcl.workflow
def bond_dissociation(molecule, bonds, cpu):
    active = qcl.Active(max_orbitals=2)

    def energy(species, device):
        mean_field = qcl.scf.hf(species).to(device)
        space = qcl.active_space(mean_field, active).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    parent = energy(molecule, cpu)

    def bde(bond, lane):
        first, second = molecule.cleave(bond)
        return energy(first, lane) + energy(second, lane) - parent

    return qcl.map(bde, bonds).to([cpu])


def test_hydrogen_bond_dissociation_matches_independent_fci():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    bonds = molecule.bonds(elements=("H", "H"))
    assert bonds == [(0, 1)]
    result = bond_dissociation.compile(molecule, bonds,
                                       qcl.Cluster.local().cpus[0]).run()
    parent_mol = gto.M(atom=molecule.atoms, basis="sto-3g", verbose=0)
    parent_mf = scf.RHF(parent_mol).run()
    atom_mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    atom_mf = scf.ROHF(atom_mol).run()
    expected = 2 * fci.FCI(atom_mf).kernel()[0] - fci.FCI(parent_mf).kernel()[0]
    assert len(result) == 1
    assert result[0].value == pytest.approx(expected, abs=1e-9)
    assert result[0].unit == "hartree"


def test_bond_dissociation_rejects_nonbond_and_infeasible_policy():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(ValueError):
        molecule.cleave((0, 2))
    with pytest.raises(qcl.CompileError) as error:
        bond_dissociation.compile(molecule, molecule.bonds(),
                                  qcl.Cluster.local().qpus[0])
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)
