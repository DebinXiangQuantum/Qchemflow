import pytest
from pyscf import fci, gto, scf

import qcl


@qcl.workflow
def dimer_binding_scan(dimer, frag_a, frag_b, separations, cpu):
    active = qcl.Active(max_orbitals=4)

    def energy(molecule, device):
        mean_field = qcl.scf.hf(molecule).to(device)
        space = qcl.active_space(mean_field, active).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    def point(distance, lane):
        geometry = dimer.set_separation(frag_a, frag_b, distance)
        complex_energy = energy(geometry, lane)
        first = energy(geometry.ghost(frag_b), lane)
        second = energy(geometry.ghost(frag_a), lane)
        return complex_energy - first - second

    return qcl.map(point, separations).to([cpu])


def test_counterpoise_dimer_scan_matches_independent_fci():
    dimer = qcl.Molecule("H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", "sto-3g")
    cpu = qcl.Cluster.local().cpus[0]
    result = dimer_binding_scan.compile(dimer, [0, 1], [2, 3], [2.0, 3.0], cpu).run()
    assert len(result) == 2

    def exact(molecule):
        system = gto.M(atom=molecule.atoms, basis=molecule.basis,
                       charge=molecule.charge, spin=molecule.spin, verbose=0)
        mean_field = scf.RHF(system).run()
        return fci.FCI(mean_field).kernel()[0]

    for distance, estimate in zip((2.0, 3.0), result, strict=True):
        geometry = dimer.set_separation([0, 1], [2, 3], distance)
        expected = (exact(geometry) - exact(geometry.ghost([2, 3])) -
                    exact(geometry.ghost([0, 1])))
        assert estimate.value == pytest.approx(expected, abs=1e-9)
        assert estimate.unit == "hartree"


def test_dimer_scan_rejects_overlapping_fragments():
    dimer = qcl.Molecule("H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", "sto-3g")
    with pytest.raises(ValueError):
        dimer.set_separation([0, 1], [1, 2], 3.0)
