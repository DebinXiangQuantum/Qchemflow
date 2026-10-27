"""Sequential H2 RHF points seeded from the preceding converged density."""

import qcl


@qcl.workflow
def warm_start_scan(molecules, device):
    previous = None
    energies = []
    for molecule in molecules:
        previous = qcl.scf.rhf(molecule, guess=previous).to(device)
        energies.append(previous.energy)
    return energies


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    device = cluster.gpus[0] if cluster.gpus else cluster.cpus[0]
    molecules = [qcl.Molecule(f"H 0 0 0; H 0 0 {distance}", "sto-3g")
                 for distance in (0.73, 0.74, 0.75)]
    results = warm_start_scan.compile(molecules, device, strict_transfers=True).run()
    for molecule, result in zip(molecules, results, strict=True):
        print(molecule.atoms, result.value, result.unit)
