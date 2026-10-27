from dataclasses import dataclass
from math import comb

import numpy as np
from pyscf import ao2mo, cc, fci, gto, mcscf, mp, scf

from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import (
    ActiveSpaceSpec,
    ConstantLedger,
    ElectronicModel,
    Hamiltonian,
    OrbitalSpace,
    Sector,
)
from chemflow.types.model import model_protocol_id
from chemflow.units import Q


@dataclass
class MolecularData:
    mol: gto.Mole
    mf: scf.hf.SCF
    hamiltonian: Hamiltonian
    sector: Sector
    space: OrbitalSpace
    model_id: str
    geometry_id: str
    model: ElectronicModel | None = None
    active: ActiveSpaceSpec | None = None

    @property
    def model_protocol_id(self) -> str:
        return model_protocol_id(self.model_id, self.active)


def build_molecule(atoms: str, model: ElectronicModel, store: ArtifactStore,
                   unit: str = "angstrom", charge: int = 0, spin: int = 0,
                   active: ActiveSpaceSpec | None = None) -> MolecularData:
    if model.hamiltonian_class != "nonrel_bo" or model.ecp or model.dft_functional:
        raise AdapterError("CF-E080", None, "Unsupported electronic model",
                           "Choose nonrelativistic Born–Oppenheimer model")
    if model.reference not in ("RHF", "ROHF"):
        raise AdapterError("CF-E080", None, "Unsupported reference", "Use RHF or ROHF")
    if unit not in ("angstrom", "bohr"):
        raise AdapterError("CF-E001", None, "Unsupported coordinate unit",
                           "Convert to angstrom or bohr")
    if active and (active.method != "manual" or active.orbital_source != "canonical_hf"):
        raise AdapterError("CF-E080", None, "Unsupported active-space selection",
                           "Use manual canonical HF orbitals")
    mol = gto.M(atom=atoms, basis=model.basis, unit="Angstrom" if unit == "angstrom" else
                "Bohr" if unit == "bohr" else unit, charge=charge, spin=spin, verbose=0)
    mf = scf.ROHF(mol) if spin or model.reference == "ROHF" else scf.RHF(mol)
    mf.kernel()
    return _from_scf(mf, model, store, active, atoms, unit, charge, spin)


def _from_scf(mf: scf.hf.SCF, model: ElectronicModel, store: ArtifactStore,
              active: ActiveSpaceSpec | None, atoms: str, unit: str,
              charge: int, spin: int) -> MolecularData:
    mol = mf.mol
    if not mf.converged:
        raise AdapterError("CF-E080", None, "SCF did not converge",
                           "Adjust the reference or geometry")
    n_orb = active.n_orb if active else mol.nao_nr()
    n_elec = active.n_elec if active else mol.nelectron
    n_frozen = (mol.nelectron - n_elec) // 2
    if n_frozen < 0 or 2 * n_frozen + n_elec != mol.nelectron:
        raise AdapterError("CF-E020", None, "Invalid active electron count",
                           "Match total and frozen electron counts")
    if n_frozen + n_orb > mol.nao_nr():
        raise AdapterError("CF-E020", None, "Active orbital count exceeds basis",
                           "Use fewer active orbitals")
    if model.frozen_core and model.frozen_core != n_frozen:
        raise AdapterError("CF-E020", None, "Frozen-core partition differs from model",
                           "Align active electrons and frozen core")
    cas = mcscf.CASCI(mf, n_orb, n_elec)
    cas.mo_coeff = mf.mo_coeff
    cas.ncore = n_frozen
    h1, ecore = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), n_orb)
    space = OrbitalSpace(coefficients_ref=store.put(np.asarray(mf.mo_coeff)),
                         n_orb=n_orb, n_elec=n_elec, frozen_orbitals=n_frozen)
    ledger = ConstantLedger().add("nuclear_repulsion", Q(mol.energy_nuc(), "hartree"))
    if n_frozen:
        ledger = ledger.add("frozen_core", Q(ecore - mol.energy_nuc(), "hartree"))
    hamiltonian = Hamiltonian(h1_ref=store.put(np.asarray(h1)),
                              h2_ref=store.put(np.asarray(eri)), space=content_hash(space),
                              n_orb=n_orb, ledger=ledger)
    if (n_elec + spin) % 2 or (n_elec - spin) % 2:
        raise AdapterError("CF-E020", None, "Particle/spin parity mismatch",
                           "Choose compatible electron and spin counts")
    sector = Sector(n_alpha=(n_elec + spin) // 2, n_beta=(n_elec - spin) // 2,
                    n_orb=n_orb)
    return MolecularData(mol, mf, hamiltonian, sector, space, content_hash(model),
                         content_hash({"atoms": atoms, "unit": unit, "charge": charge,
                                       "spin": spin}), model, active)


def active_from_scf(data: MolecularData, n_elec: int, n_orb: int,
                    store: ArtifactStore) -> MolecularData:
    if data.model is None:
        raise AdapterError("CF-E080", None, "SCF model unavailable",
                           "Use an SCF result with model provenance")
    return _from_scf(data.mf, data.model, store,
                     ActiveSpaceSpec(n_elec=n_elec, n_orb=n_orb),
                     data.mol.atom, data.mol.unit, data.mol.charge, data.mol.spin)


def fci_pyscf(data: MolecularData, store: ArtifactStore,
              total_spin: int | None = None, *, return_roots: bool = False,
              return_wavefunction: bool = False,
              residual_tol: float | None = None) -> tuple:
    if residual_tol is not None and (not return_wavefunction or total_spin is not None or
                                     not np.isfinite(residual_tol) or residual_tol <= 0):
        raise ValueError("Wavefunction residual tolerance requires one FCI root without a spin target")
    eri = store.get(data.hamiltonian.h2_ref)
    h1 = store.get(data.hamiltonian.h1_ref)
    solver = fci.direct_spin1.FCI()
    electrons = (data.sector.n_alpha, data.sector.n_beta)
    dimension = comb(data.sector.n_orb, electrons[0]) * comb(data.sector.n_orb, electrons[1])
    roots = 1 if total_spin is None else min(4, dimension)
    while True:
        solver.nroots = roots
        _, vectors = solver.kernel(h1, eri, data.hamiltonian.n_orb, electrons)
        candidates = vectors if isinstance(vectors, list) else [vectors]
        if total_spin is None:
            vector = candidates[0]
            break
        matched = [candidate for candidate in candidates
                   if abs(float(fci.spin_op.spin_square(candidate, data.hamiltonian.n_orb,
                                                       electrons)[0]) -
                          total_spin * (total_spin + 1)) < 1e-8]
        if matched:
            vector = matched[0]
            break
        if roots == dimension:
            raise AdapterError("CF-E020", None, "Requested spin state unavailable",
                               "Select a feasible total spin")
        roots = min(2 * roots, dimension)
    if residual_tol is not None:
        effective_h2 = fci.direct_spin1.absorb_h1e(
            h1, eri, data.sector.n_orb, electrons, 0.5)

        def residual(ci):
            energy = fci.direct_spin1.energy(h1, eri, ci,
                                             data.sector.n_orb, electrons)
            applied = fci.direct_spin1.contract_2e(
                effective_h2, ci, data.sector.n_orb, electrons)
            return float(np.linalg.norm(applied - energy * ci))

        if residual(vector) > residual_tol:
            solver.conv_tol = 1e-12
            # PySCF supports this setting but omits it from its warning allowlist.
            solver._keys = solver._keys | {"conv_tol_residual"}
            solver.conv_tol_residual = min(1e-11, residual_tol / 100)
            solver.max_cycle = 300
            solver.max_space = 40
            solver.lindep = 1e-18
            _, vector = solver.kernel(h1, eri, data.hamiltonian.n_orb,
                                      electrons, ci0=vector)
            if residual(vector) > residual_tol:
                raise AdapterError("CF-E020", None,
                                   "FCI wavefunction residual exceeds tolerance",
                                   "Use a smaller active space or another CI engine")
    s2 = float(fci.spin_op.spin_square(vector, data.hamiltonian.n_orb, electrons)[0])
    physical_energy = float(fci.direct_spin1.energy(h1, eri, vector,
                                                     data.hamiltonian.n_orb, electrons))
    result = (physical_energy + data.hamiltonian.ledger.total().value, s2)
    if return_roots:
        result = (*result, roots)
    return (*result, vector) if return_wavefunction else result


def classical_energy(data: MolecularData, method: str, store: ArtifactStore,
                     diagnostics: dict | None = None) -> float:
    diagnostics = diagnostics if diagnostics is not None else {}
    if method == "fci_pyscf":
        return fci_pyscf(data, store)[0]
    if method not in ("hf", "mp2", "ccsd", "ccsd_t"):
        raise AdapterError("CF-E080", None, f"Unsupported solver {method}",
                           "Bind a registered implementation")
    reference = data.mf
    constant = 0.
    if data.space.ordering == "sites":
        constant = data.hamiltonian.ledger.total().value
    elif data.space.frozen_orbitals or data.space.n_orb != data.mol.nao_nr():
        h1 = store.get(data.hamiltonian.h1_ref)
        eri = store.get(data.hamiltonian.h2_ref)
        molecule = gto.M(verbose=0)
        molecule.nelectron = data.sector.n_alpha + data.sector.n_beta
        molecule.spin = data.sector.n_alpha - data.sector.n_beta
        reference = scf.ROHF(molecule) if molecule.spin else scf.RHF(molecule)
        reference.get_hcore = lambda *args: h1
        reference.get_ovlp = lambda *args: np.eye(data.sector.n_orb)
        reference._eri = eri
        occupations = ((np.arange(data.sector.n_orb) < data.sector.n_alpha).astype(float) +
                       (np.arange(data.sector.n_orb) < data.sector.n_beta))
        reference.kernel(dm0=np.diag(occupations))
        if not reference.converged:
            raise AdapterError("CF-E080", None, "Active-space HF did not converge",
                               "Adjust reference inputs")
        constant = data.hamiltonian.ledger.total().value
    diagnostics.update(reference_s2=float(reference.spin_square()[0]),
                       reference=reference.__class__.__name__)
    diagnostics.update(getattr(reference, "chemflow_diagnostics", {}))
    hf_energy = float(reference.e_tot + constant)
    if method == "hf":
        diagnostics["s2"] = diagnostics["reference_s2"]
        return hf_energy
    sector = data.sector
    if all(count in (0, sector.n_orb) for count in (sector.n_alpha, sector.n_beta)):
        diagnostics["s2"] = diagnostics["reference_s2"]
        diagnostics["note"] = "No virtual excitations; correlation energy is exactly zero"
        return hf_energy
    if sector.n_alpha != sector.n_beta:
        reference = reference.to_uhf()
    solver = mp.MP2(reference) if method == "mp2" else cc.CCSD(reference)
    solver.kernel()
    if method != "mp2" and not solver.converged:
        raise AdapterError("CF-E080", None, "CCSD did not converge", "Adjust solver inputs")
    if isinstance(reference, scf.uhf.UHF):
        density_alpha, density_beta = solver.make_rdm1()
        density_aa, density_ab, density_bb = solver.make_rdm2()
        diagnostics["s2"] = float(fci.spin_op.spin_square_general(
            density_alpha, density_beta, density_aa, density_ab, density_bb,
            solver.mo_coeff, reference.get_ovlp())[0])
        diagnostics["s2_method"] = "mp2" if method == "mp2" else "ccsd"
    return float(solver.e_tot + constant + (solver.ccsd_t() if method == "ccsd_t" else 0))
