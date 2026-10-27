"""Exact retarded spectral density for a two-site impurity model."""

import qcl


@qcl.workflow
def spectrum(model, sector, cpu):
    return qcl.dynamics.spectral_exact(
        model, sector, orbital=0, spin="alpha",
        omega={"start": -4.0, "stop": 4.0, "n": 201, "unit": "hartree"},
        eta=0.1).to(cpu)


if __name__ == "__main__":
    result = spectrum.compile(
        {"U": 2.0, "eps_bath": [0.2], "t_bath": [0.5], "unit": "hartree"},
        {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
        qcl.Cluster.local().cpus[0]).run()
    print(f"{len(result.omega)} points; sum rule {result.diagnostics['sum_rule']:.6f}")
