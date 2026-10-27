import pytest
from pyscf import gto, scf

import qcl


@qcl.workflow
def rigid_dihedral_scan(molecule, atoms, angles, cpu):
    def point(angle, lane):
        geometry = molecule.set_dihedral(*atoms, angle)
        return qcl.scf.hf(geometry).to(lane).energy

    return qcl.map(point, angles).to([cpu])


def test_rigid_scan_preserves_input_and_matches_pyscf():
    molecule = qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g")
    original_atoms = molecule.atoms
    cpu = qcl.Cluster.local().cpus[0]
    angles = [60.0, 120.0]
    result = rigid_dihedral_scan.compile(molecule, (0, 1, 2, 3), angles, cpu).run()
    assert molecule.atoms == original_atoms
    assert len(result) == 2
    for angle, estimate in zip(angles, result, strict=True):
        geometry = molecule.set_dihedral(0, 1, 2, 3, angle)
        reference = scf.RHF(gto.M(atom=geometry.atoms, basis=geometry.basis,
                                  verbose=0)).run()
        assert estimate.value == pytest.approx(reference.e_tot, abs=1e-9)


def test_internal_coordinate_edit_validates_atom_indices():
    molecule = qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g")
    with pytest.raises(ValueError):
        molecule.set_dihedral(0, 1, 2, 4, 60.0)
