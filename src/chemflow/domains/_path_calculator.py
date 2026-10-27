"""Isolate tblite's OpenMP runtime from Sella's on platforms with separate runtimes."""

import json
import subprocess
import sys
import weakref


def _close(worker):
    worker.stdin.close()
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    worker.stdout.close()


def isolated_tblite(molecule, solvent=None):
    from ase.calculators.calculator import Calculator, all_changes

    class TBLiteProcess(Calculator):
        implemented_properties = ("energy", "forces")

        def __init__(self):
            super().__init__()
            self.worker = None

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if self.worker is None:
                self.worker = subprocess.Popen(
                    [sys.executable, "-m", "chemflow.domains._path_calculator"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
                self.cleanup = weakref.finalize(self, _close, self.worker)
            request = {"symbols": self.atoms.get_chemical_symbols(),
                       "positions": self.atoms.positions.tolist(),
                       "charge": molecule.get("charge", 0), "spin": molecule.get("spin", 0),
                       "solvent": solvent}
            self.worker.stdin.write(json.dumps(request) + "\n")
            self.worker.stdin.flush()
            response = self.worker.stdout.readline()
            if not response:
                raise RuntimeError("tblite calculator worker exited unexpectedly")
            result = json.loads(response)
            if "error" in result:
                raise ValueError(result["error"])
            import numpy as np

            self.results = {"energy": result["energy"], "forces": np.array(result["forces"])}

        def close(self):
            if self.worker is not None:
                self.cleanup()
                self.worker = None

    return TBLiteProcess()


def main():
    from ase import Atoms
    from tblite.ase import TBLite

    for line in sys.stdin:
        try:
            request = json.loads(line)
            atoms = Atoms(request["symbols"], positions=request["positions"])
            atoms.calc = TBLite(method="GFN2-xTB", charge=request["charge"],
                                multiplicity=request["spin"] + 1, verbosity=0,
                                solvation=(request["solvent"]["model"],
                                           request["solvent"]["name"])
                                if request.get("solvent") else None)
            result = {"energy": float(atoms.get_potential_energy()),
                      "forces": atoms.get_forces().tolist()}
        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
