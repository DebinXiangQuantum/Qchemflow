"""Full-space DFT, declared solvent/orbital models, and vertical IP/EA."""


import numpy as np

from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash

from ._adapters import estimate


def refuse(message):
    raise AdapterError("CF-E080", None, message, "Use a supported full-space model or smaller basis")


def molecule_for(molecule, electronic_model):
    from pyscf import gto

    if (set(electronic_model) - {"basis", "reference", "hamiltonian_class"} or
            electronic_model.get("hamiltonian_class", "nonrel_bo") != "nonrel_bo"):
        refuse("Properties require an all-electron nonrelativistic full-space model")
    return gto.M(atom=molecule["atoms"], basis=electronic_model["basis"],
                 charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                 unit=molecule.get("unit", "angstrom"), verbose=0)


def _result(value, kind, params, diagnostics=None, relation="equivalent"):
    if not np.isfinite(value):
        refuse("Property calculation returned a nonfinite value")
    return estimate(float(value), kind, relation, {"hamiltonian_id": content_hash(params)},
                    kind, diagnostics)


def eom_charge_energy(store, molecule, electronic_model, kind, final_spin):
    from pyscf import cc, scf
    from pyscf.cc import eom_rccsd

    mol = molecule_for(molecule, electronic_model)
    if mol.spin or final_spin != 0.5:
        refuse("Restricted EOM-IP/EA-CCSD requires a closed-shell initial state and final S=1/2")
    reference = scf.RHF(mol).run()
    solver = cc.CCSD(reference).run()
    if not reference.converged or not solver.converged:
        refuse("EOM reference did not converge")
    response = eom_rccsd.EOMIP(solver) if kind == "ionization_energy" else eom_rccsd.EOMEA(solver)
    energy = response.kernel(nroots=1)[0]
    if not np.all(response.converged):
        refuse("EOM-CCSD did not converge")
    return _result(energy if kind == "ionization_energy" else -energy, kind,
                   {"molecule": molecule, "electronic_model": electronic_model, "kind": kind,
                    "final_spin": final_spin, "operator": "charge_energy_difference"},
                   relation="approximate")
