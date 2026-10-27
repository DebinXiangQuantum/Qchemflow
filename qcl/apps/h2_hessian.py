"""HF Hessian tensor and harmonic frequencies with explicit device placement."""

import qcl


@qcl.workflow
def hessian_and_modes(molecule, compute_device, cpu, xc=None):
    callback = (lambda geometry, lane: qcl.dft.rks(geometry, xc=xc).to(lane)
                if xc else qcl.scf.hf(geometry).to(lane))
    surface = qcl.PES(callback, gradient=qcl.Analytic())
    matrix = qcl.hessian(surface, molecule).to(compute_device)
    on_cpu = matrix.to(cpu) if compute_device.kind == "gpu" else matrix
    return matrix, qcl.frequencies(on_cpu).to(cpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    compute_device = cluster.gpus[0] if cluster.gpus else cluster.cpus[0]
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    matrix, modes = hessian_and_modes.compile(
        molecule, compute_device, cluster.cpus[0], strict_transfers=True).run()
    print(matrix.shape, matrix.unit, matrix.device)
    print(modes.freq_wavenumber)
    pbe_matrix, pbe_modes = hessian_and_modes.compile(
        molecule, compute_device, cluster.cpus[0], xc="pbe",
        strict_transfers=True).run()
    print(pbe_matrix.provenance["method"], pbe_modes.freq_wavenumber)
