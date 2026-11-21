"""Site-basis Hubbard Hamiltonians and primitive-cell periodic references."""

from dataclasses import dataclass, replace

import numpy as np

from chemflow.ids import content_hash
from chemflow.impls.pyscf_impls import MolecularData
from chemflow.types import ConstantLedger, Hamiltonian, OrbitalSpace, Sector
from chemflow.units import Q


def _kernel(mf, dm0):
    """Run an SCF; without DIIS if PySCF's DIIS extrapolation fails.

    A singular DIIS matrix makes PySCF 2.14 raise from its own error handler under NumPy
    2.4 (it names ``numpy.linalg.linalg``). Damping and the level shift still apply.
    """
    try:
        mf.kernel(dm0=dm0)
        return False
    except (np.linalg.LinAlgError, AttributeError) as error:
        if isinstance(error, AttributeError) and "linalg" not in str(error):
            raise
        mf.diis = False
        mf.kernel(dm0=dm0)
        return True


def site_reference(h1, eri, n_alpha, n_beta):
    from pyscf import gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = n_alpha + n_beta
    mol.spin = n_alpha - n_beta
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(len(h1))
    mf._eri = eri
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.damp = 0.5
    mf.level_shift = 0.5
    without_diis = _kernel(mf, np.eye(len(h1)) * mol.nelectron / len(h1))
    attempts = ["restricted"]
    stable = False
    if mf.converged:
        _, _, stable, _ = mf.stability(return_status=True)
    if not mf.converged or not stable:
        attempts.append("unrestricted_staggered")
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(len(h1))
        mf._eri = eri
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.damp = 0.5
        mf.level_shift = 0.5
        ordering = np.argsort(np.arange(len(h1)) % 2, kind="stable")
        density = np.zeros((2, len(h1), len(h1)))
        density[0, ordering[:n_alpha], ordering[:n_alpha]] = 1
        density[1, ordering[::-1][:n_beta], ordering[::-1][:n_beta]] = 1
        without_diis |= _kernel(mf, density)
        if mf.converged:
            orbitals, _, stable, _ = mf.stability(return_status=True)
            for _ in range(3):
                if stable:
                    break
                without_diis |= _kernel(mf, mf.make_rdm1(orbitals, mf.mo_occ))
                orbitals, _, stable, _ = mf.stability(return_status=True)
    if not mf.converged:
        raise ValueError("Lattice mean-field reference did not converge")
    if not stable:
        raise ValueError("Lattice mean-field reference is internally unstable")
    mf.chemflow_diagnostics = {"scf_settings": {
        "damping": mf.damp, "level_shift": mf.level_shift, "conv_tol": mf.conv_tol,
        "max_cycle": mf.max_cycle, "fractional_occupation": False, "attempts": attempts,
        "diis": not without_diis},
        "stability": {"internal": bool(stable)}, "reference": mf.__class__.__name__}
    return mf


@dataclass
class LatticeData(MolecularData):
    lattice_id: str = ""

    def for_sector(self, sector, store):
        requested = Sector(**{key: sector[key] for key in ("n_alpha", "n_beta", "n_orb")})
        if requested.n_orb != self.hamiltonian.n_orb:
            raise ValueError("Lattice sector orbital count must equal n_sites")
        if requested == self.sector:
            return self
        return replace(self, sector=requested, mf=None, mol=None)

    def mean_field(self, store):
        if self.mf is None:
            self.mf = site_reference(store.get(self.hamiltonian.h1_ref),
                                     store.get(self.hamiltonian.h2_ref),
                                     self.sector.n_alpha, self.sector.n_beta)
            self.mol = self.mf.mol
        return self.mf


def lattice_arrays(model="hubbard", geometry="chain", n_sites=None, t=1., U=4.,
                   unit="hartree", shape=None):
    if model != "hubbard" or geometry not in ("chain", "ring", "square"):
        raise ValueError("Supported lattices: Hubbard chain, ring, square")
    if geometry == "square":
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
                any(type(size) is not int or size < 1 for size in shape)):
            raise ValueError("Square lattice requires positive integer shape [nx, ny]")
        size = shape[0] * shape[1]
        if n_sites is not None and n_sites != size:
            raise ValueError("n_sites differs from shape product")
        n_sites = size
    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    hopping, repulsion = Q(t, unit).to("hartree").value, Q(U, unit).to("hartree").value
    if not np.isfinite([hopping, repulsion]).all():
        raise ValueError("Lattice couplings must be finite")
    bonds = set()
    if geometry == "square":
        for row in range(shape[0]):
            for column in range(shape[1]):
                site = row * shape[1] + column
                if row + 1 < shape[0]:
                    bonds.add((site, site + shape[1]))
                if column + 1 < shape[1]:
                    bonds.add((site, site + 1))
    else:
        bonds.update((site, site + 1) for site in range(n_sites - 1))
        if geometry == "ring" and n_sites > 2:
            bonds.add((0, n_sites - 1))
    h1 = np.zeros((n_sites, n_sites))
    eri = np.zeros((n_sites,) * 4)
    for left, right in bonds:
        h1[left, right] = h1[right, left] = -hopping
    for site in range(n_sites):
        eri[site, site, site, site] = repulsion
    return h1, eri


def lattice_model(store, **params):
    h1, eri = lattice_arrays(**params)
    size = len(h1)
    sector = Sector(n_alpha=(size + 1) // 2, n_beta=size // 2, n_orb=size)
    space = OrbitalSpace(coefficients_ref=store.put(np.eye(size)), n_orb=size,
                         n_elec=size, ordering="sites")
    ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), n_orb=size,
                      space=content_hash(space), ledger=ConstantLedger())
    identity = content_hash(params)
    return LatticeData(mol=None, mf=None, hamiltonian=ham, space=space, sector=sector,
                       model_id=identity, geometry_id=identity, lattice_id=identity)


def periodic_system(store, lattice, atoms, basis, pseudo=None, unit="angstrom"):
    vectors = np.asarray(lattice, dtype=float)
    if (vectors.shape != (3, 3) or not np.isfinite(vectors).all() or
            abs(np.linalg.det(vectors)) < 1e-9):
        raise ValueError("Periodic lattice must be finite and nonsingular 3x3 vectors")
    if unit not in ("angstrom", "bohr"):
        raise ValueError("Periodic lattice unit must be angstrom or bohr")
    if not pseudo:
        raise ValueError("Periodic reference requires an explicit pseudopotential")
    return {"lattice": lattice, "atoms": atoms, "basis": basis, "pseudo": pseudo, "unit": unit}


def periodic_scf(store, system, method="rhf", kmesh=None, supercell=None, exxdiv="ewald"):
    from pyscf.pbc import gto, scf, tools

    if method != "rhf" or (kmesh is not None and supercell is not None):
        raise ValueError("Use rhf and either kmesh or supercell, not both")
    mesh = kmesh if kmesh is not None else supercell if supercell is not None else [1, 1, 1]
    if len(mesh) != 3 or any(type(size) is not int or size < 1 for size in mesh):
        raise ValueError("Periodic mesh must contain three positive integers")
    cell = gto.Cell()
    cell.a = system["lattice"]
    cell.atom, cell.unit = system["atoms"], system["unit"]
    cell.basis, cell.pseudo = system["basis"], system["pseudo"]
    cell.verbose = 0
    cell.build()
    divisor = 1
    if supercell is not None:
        cell = tools.super_cell(cell, supercell)
        divisor = int(np.prod(supercell))
        mf = scf.RHF(cell, exxdiv=exxdiv)
    else:
        mf = scf.KRHF(cell, cell.make_kpts(mesh), exxdiv=exxdiv)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise ValueError("Periodic SCF did not converge")
    return float(mf.e_tot / divisor)
