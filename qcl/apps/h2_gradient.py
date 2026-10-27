"""Analytic HF and PBE nuclear gradients on an explicitly selected device."""

import qcl


@qcl.workflow
def gradient(molecule, device):
    surface = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane))
    return qcl.gradient(surface, molecule).to(device)


@qcl.workflow
def pbe_gradient(molecule, device):
    surface = qcl.PES(lambda geometry, lane: qcl.dft.rks(geometry, xc="pbe").to(lane),
                      gradient=qcl.Analytic())
    return qcl.gradient(surface, molecule).to(device)


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = gradient.compile(molecule, qcl.Cluster.local().cpus[0]).run()
    print(result.data)
    print(result.unit)
    print(pbe_gradient.compile(molecule, qcl.Cluster.local().cpus[0]).run().data)
