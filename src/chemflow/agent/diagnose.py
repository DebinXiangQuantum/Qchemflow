"""Model diagnostics: an MP2 natural-occupation active-space suggestion."""

from __future__ import annotations


def diagnose_model(system: dict, model: dict) -> dict:
    """RHF/MP2 diagnostics and an active space of at most ten fractional orbitals."""
    from pyscf import gto, mp, scf

    molecule = gto.M(atom=system["atoms"], basis=model["basis"], verbose=0,
                     charge=system.get("charge", 0), spin=system.get("spin", 0),
                     unit=system.get("unit", "angstrom"))
    reference = scf.RHF(molecule).run()
    occupied = molecule.nelectron // 2
    gap = float(reference.mo_energy[occupied] - reference.mo_energy[occupied - 1])
    perturbation = mp.MP2(reference).run()
    occupations = sorted((float(value) for value in
                          __import__("numpy").linalg.eigvalsh(perturbation.make_rdm1())),
                         reverse=True)
    fractional = [index for index, value in enumerate(occupations) if 0.02 <= value <= 1.98]
    selected = sorted(fractional, key=lambda index: abs(occupations[index] - 1))[:10]
    if len(selected) < 2:
        raise ValueError("Auto active space needs at least two fractional MP2 orbitals")
    orbital_count = len(selected)
    electron_count = round(sum(occupations[index] for index in selected))
    if electron_count % 2 != molecule.nelectron % 2:
        electron_count += 1 if electron_count < sum(occupations[index] for index in selected) else -1
    if not 0 < electron_count <= 2 * orbital_count or (molecule.nelectron - electron_count) < 0:
        raise ValueError("Auto active space has infeasible electron count")
    return {"rhf_energy": float(reference.e_tot), "homo_lumo_gap": gap,
            "mp2_noons": occupations, "selected_orbitals": selected,
            "orbital_provenance": ("size from MP2 natural occupations; "
                                   "orbitals: canonical HF frontier window"),
            "total_electrons": molecule.nelectron, "suggested_active": {
                "n_elec": electron_count, "n_orb": orbital_count}}

