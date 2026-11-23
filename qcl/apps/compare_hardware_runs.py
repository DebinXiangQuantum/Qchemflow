"""Compare two analysed hardware runs variant by variant (CZ, signal, fidelities).

    python qcl/apps/compare_hardware_runs.py hw:open hw-opt:open
"""

import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("cz", "signal", "F", "F_uniform", "F_post", "F_readout_post", "device_s", "wall_s")


def summary(folder: str, variant: str) -> dict:
    report = json.loads((Path(folder) / "analysis.json").read_text())
    rows = {}
    for group, systems in report["systems"].items():
        for name, entry in systems.items():
            result = entry["variants"].get(variant)
            if result is None:
                continue
            parts = result["parts"]
            rows[f"{group}/{name}"] = {
                "qubits": entry["qubits"],
                **{key: float(np.mean([part[key] for part in parts]))
                   for key in KEYS if all(part.get(key) is not None for part in parts)},
                "raw_error_meh": result["raw_error_meh"]}
    return rows


def main():
    (first, first_variant), (second, second_variant) = (item.split(":") for item in sys.argv[1:3])
    left, right = summary(first, first_variant), summary(second, second_variant)
    print(f"{'system':28s} {'q':>3} | {'CZ':>9} | {'signal':>15} | {'F':>15} | "
          f"{'F post-selected':>15} | {'raw SQD mEh':>15}")
    for name in sorted(set(left) & set(right), key=lambda key: (left[key]["qubits"], key)):
        a, b = left[name], right[name]

        def pair(key, form, a=a, b=b):
            return (f"{a[key]:{form}} → {b[key]:{form}}" if key in a and key in b
                    else f"{'':15}")

        print(f"{name:28s} {a['qubits']:3d} | {pair('cz', '.0f'):>9} | "
              f"{pair('signal', '.4f'):>15} | {pair('F', '.4f'):>15} | "
              f"{pair('F_post', '.4f'):>15} | {pair('raw_error_meh', '.1f'):>15}")
    output = {"left": {"folder": first, "variant": first_variant, "rows": left},
              "right": {"folder": second, "variant": second_variant, "rows": right}}
    (Path(second) / "comparison.json").write_text(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
