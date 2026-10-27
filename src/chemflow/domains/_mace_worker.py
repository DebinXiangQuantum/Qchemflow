"""Private JSON-lines MACE server; never imported by the numerical parent."""

import json
import os
import sys


def main():
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        import numpy as np
        from ase import Atoms
        from mace.calculators import mace_mp
        from scipy import constants as sc

        calculator = mace_mp(model=sys.argv[1], default_dtype="float64", device="cpu")
        hartree_ev = sc.physical_constants["Hartree energy in eV"][0]
        bohr = sc.physical_constants["Bohr radius"][0] * 1e10
        protocol.write(json.dumps({"ready": True}) + "\n")
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("charge", 0) or request.get("spin", 0):
                raise ValueError("MACE-MP-0 does not model molecular charge or spin states")
            positions = np.asarray(request["positions"], dtype=np.float64)
            if positions.shape != (len(request["symbols"]), 3) or not np.isfinite(positions).all():
                raise ValueError("Expected finite positions with shape (number of atoms, 3)")
            atoms = Atoms(request["symbols"], positions=positions)
            atoms.calc = calculator
            response = {"energy": float(atoms.get_potential_energy() / hartree_ev),
                        "gradient": (-atoms.get_forces() * bohr / hartree_ev).tolist()}
            protocol.write(json.dumps(response, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
