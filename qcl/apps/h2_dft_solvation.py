"""H2 PBE/PCM solvation energy on the first available GPU or CPU."""

import qcl


@qcl.workflow
def solvation_energy(molecule, device, solvent):
    gas = qcl.dft.rks(molecule, xc="pbe").to(device).energy
    liquid = qcl.dft.rks(molecule, xc="pbe", solvent=solvent).to(device).energy
    return liquid - gas


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    device = cluster.gpus[0] if cluster.gpus else cluster.cpus[0]
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = solvation_energy.compile(
        molecule, device, qcl.PCM("water"), strict_transfers=True).run()
    print(result.value, result.unit, result.provenance["operands"][0]["device"])
