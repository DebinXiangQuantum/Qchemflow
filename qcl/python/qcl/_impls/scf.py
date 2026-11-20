"""Mean-field methods, MP2, orbital projections, active spaces and CCSD amplitudes."""

from qcl._impls import implements
from qcl._impls._common import (
    _canonical_unrestricted_energies,
    _gpu4pyscf_version,
    _gpu_index,
)


@implements("periodic.scf")
def _op_periodic_scf(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    store = ctx.store
    from chemflow.domains.materials import periodic_scf, periodic_system

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Periodic SCF currently requires a CPU")
    params = node["params"]
    system = periodic_system(store, **params["system"])
    energy = periodic_scf(store, system, method=params["method"],
                          kmesh=params["kmesh"], exxdiv=params["exxdiv"])
    values[identifier] = {"value": energy, "unit": "hartree", "variance": 0.0,
                          "provenance": {
                              "question": {"property": "periodic_energy",
                                           "system": params["system"]},
                              "method": {"family": "periodic KRHF",
                                         "kmesh": params["kmesh"],
                                         "exxdiv": params["exxdiv"]},
                              "implementation": "periodic.scf/chemflow-pyscf-pbc",
                              "device": device["id"],
                              "versions": {"pyscf": pyscf.__version__}}}


@implements("scf.rhf", "scf.hf", "scf.rohf", "scf.uhf", "dft.rks", "dft.uks")
def _op_scf_rhf(ctx, node, identifier, operation, inputs):
    import pyscf

    from chemflow.impls.pyscf_impls import build_molecule
    from chemflow.types.model import ElectronicModel

    values = ctx.values
    store = ctx.store
    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported {operation} device: {device['kind']}")
    molecule = inputs[0]
    reference = ("UKS" if operation == "dft.uks" else
                 "RKS" if operation == "dft.rks" else
                 "UHF" if operation == "scf.uhf" else
                 "ROHF" if operation == "scf.rohf" or
                 operation == "scf.hf" and molecule["spin"] else "RHF")
    solvent = node["params"].get("solvent")
    guess = inputs[-1] if node["params"].get("reuse_guess") else None
    guess_dm = guess["data"].mf.make_rdm1() if guess is not None else None
    gpu_scf = device["kind"] == "gpu"
    if operation in ("dft.rks", "dft.uks"):
        from types import SimpleNamespace

        from pyscf import dft, gto

        system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                       charge=molecule["charge"], spin=molecule["spin"],
                       unit=molecule["unit"], verbose=0)
        mean_field = (dft.UKS(system) if operation == "dft.uks"
                      else dft.RKS(system))
        mean_field.xc = node["params"]["xc"]
        mean_field.grids.level = 3
        mean_field.conv_tol = 1e-10
        if gpu_scf:

            import cupy as cp

            with cp.cuda.Device(_gpu_index(device["id"])):
                mean_field = mean_field.to_gpu()
                if solvent:
                    if solvent.get("model") == "smd":
                        mean_field = mean_field.SMD()
                        mean_field.with_solvent.solvent = solvent["name"]
                    else:
                        mean_field = mean_field.PCM()
                        mean_field.with_solvent.method = "IEF-PCM"
                        mean_field.with_solvent.eps = solvent["eps"]
                mean_field.kernel()
            versions = {"pyscf": pyscf.__version__,
                        "gpu4pyscf": _gpu4pyscf_version(),
                        "cupy": cp.__version__}
            implementation = (f"{operation}/gpu4pyscf-" +
                              solvent.get("model", "pcm") if solvent else
                              f"{operation}/gpu4pyscf")
        else:
            if solvent:
                if solvent.get("model") == "smd":
                    mean_field = mean_field.SMD()
                    mean_field.with_solvent.solvent = solvent["name"]
                else:
                    mean_field = mean_field.PCM()
                    mean_field.with_solvent.method = "IEF-PCM"
                    mean_field.with_solvent.eps = solvent["eps"]
            mean_field.kernel()
            versions = {"pyscf": pyscf.__version__}
            implementation = (f"{operation}/chemflow-pyscf-" +
                              solvent.get("model", "pcm") if solvent else
                              f"{operation}/chemflow-pyscf")
        if not mean_field.converged:
            raise ValueError(f"{reference} did not converge")
        data = SimpleNamespace(mf=mean_field, mol=system,
                               model=ElectronicModel(
                                   basis=molecule["basis"],
                                   reference="UHF" if operation == "dft.uks" else "RHF",
                                   dft_functional=node["params"]["xc"]))
    elif gpu_scf or operation == "scf.uhf":
        from types import SimpleNamespace

        from pyscf import gto, scf

        system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                       charge=molecule["charge"], spin=molecule["spin"],
                       unit=molecule["unit"], verbose=0)
        mean_field = (scf.UHF(system) if operation == "scf.uhf" else
                      scf.ROHF(system) if reference == "ROHF" else
                      scf.RHF(system))
        mean_field.conv_tol = 1e-10
        if gpu_scf:

            import cupy as cp

            with cp.cuda.Device(_gpu_index(device["id"])):
                if operation == "scf.uhf":
                    from gpu4pyscf.scf import uhf as gpu_uhf

                    mean_field = gpu_uhf.UHF(system)
                    mean_field.conv_tol = 1e-10
                elif reference == "ROHF":
                    from gpu4pyscf.scf import rohf as gpu_rohf

                    mean_field = gpu_rohf.ROHF(system)
                    mean_field.conv_tol = 1e-10
                else:
                    mean_field = mean_field.to_gpu()
                if solvent:
                    if solvent.get("model") == "smd":
                        mean_field = mean_field.SMD()
                        mean_field.with_solvent.solvent = solvent["name"]
                    else:
                        mean_field = mean_field.PCM()
                        mean_field.with_solvent.method = "IEF-PCM"
                        mean_field.with_solvent.eps = solvent["eps"]
                mean_field.kernel(dm0=cp.asarray(guess_dm)
                                  if guess_dm is not None else None)
            versions = {"pyscf": pyscf.__version__,
                        "gpu4pyscf": _gpu4pyscf_version(),
                        "cupy": cp.__version__}
            implementation = (f"{operation}/gpu4pyscf-" +
                              solvent.get("model", "pcm") if solvent else
                              f"{operation}/gpu4pyscf")
        else:
            mean_field.kernel()
            versions = {"pyscf": pyscf.__version__}
            implementation = f"{operation}/chemflow-pyscf"
        if not mean_field.converged:
            raise ValueError(f"{reference} did not converge")
        data = SimpleNamespace(mf=mean_field, mol=system,
                               model=ElectronicModel(basis=molecule["basis"],
                                                     reference=reference))
    elif operation == "scf.rhf" and (
            node["params"].get("reuse_eri") or
            node["params"].get("reuse_guess") or solvent):
        from types import SimpleNamespace

        from pyscf import gto, scf

        system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                       charge=molecule["charge"], spin=molecule["spin"],
                       unit=molecule["unit"], verbose=0)
        mean_field = scf.RHF(system)
        if node["params"].get("reuse_eri"):
            mean_field._eri = inputs[1]["eri"]
        if solvent:
            if solvent.get("model") == "smd":
                mean_field = mean_field.SMD()
                mean_field.with_solvent.solvent = solvent["name"]
            else:
                mean_field = mean_field.PCM()
                mean_field.with_solvent.method = "IEF-PCM"
                mean_field.with_solvent.eps = solvent["eps"]
        mean_field.conv_tol = 1e-10
        if hasattr(guess_dm, "get"):
            guess_dm = guess_dm.get()
        mean_field.kernel(dm0=guess_dm)
        if not mean_field.converged:
            raise ValueError("RHF/PCM did not converge")
        data = SimpleNamespace(mf=mean_field, mol=system,
                               model=ElectronicModel(basis=molecule["basis"],
                                                     reference="RHF"))
        implementation = ("scf.rhf/chemflow-pyscf-" +
                          solvent.get("model", "pcm") if solvent else
                          "scf.rhf/chemflow-pyscf-eri" if node["params"].get(
                              "reuse_eri") else "scf.rhf/chemflow-pyscf")
        versions = {"pyscf": pyscf.__version__}
    else:
        data = build_molecule(
            molecule["atoms"], ElectronicModel(basis=molecule["basis"],
                                                 reference=reference), store,
            unit=molecule["unit"], charge=molecule["charge"], spin=molecule["spin"],
        )
        implementation = f"{operation}/chemflow-pyscf"
        versions = {"pyscf": pyscf.__version__}
    values[identifier] = {
        "data": data, "energy": float(data.mf.e_tot), "device": device["id"],
        "gpu_scf": gpu_scf,
        "question": {"atoms": molecule["atoms"], "charge": molecule["charge"],
                     "spin": molecule["spin"], "unit": molecule["unit"],
                     "property": "electronic_energy",
                     **({"conformer": molecule["conformer_provenance"]}
                        if "conformer_provenance" in molecule else {})},
        "method": {"family": reference, "basis": molecule["basis"],
                   **({"guess": "RHF density",
                       "guess_molecule": guess["question"]}
                      if guess is not None else {}),
                   **({"functional": node["params"]["xc"]}
                      if operation in ("dft.rks", "dft.uks") else {}),
                   **({"solvent": solvent["name"],
                       **({"solvent_model": "SMD"}
                          if solvent.get("model") == "smd" else
                          {"pcm_eps": solvent["eps"]})}
                      if solvent else {})},
        "implementation": implementation,
        "versions": versions,
    }


@implements("mp2")
def _op_mp2(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    device = node["device"]
    if device["kind"] not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported MP2 device: {device['kind']}")
    source = inputs[0]
    mean_field = source["data"].mf
    if device["kind"] == "gpu":

        import cupy as cp
        from gpu4pyscf.mp import mp2 as gpu_mp2

        with cp.cuda.Device(_gpu_index(device["id"])):
            if not source.get("gpu_scf"):
                mean_field = mean_field.to_gpu()
            calculation = gpu_mp2.MP2(mean_field)
            calculation.kernel()
        implementation = "mp2/gpu4pyscf"
        versions = {"pyscf": pyscf.__version__,
                    "gpu4pyscf": _gpu4pyscf_version(),
                    "cupy": cp.__version__}
    else:
        from pyscf import mp

        if source.get("gpu_scf"):
            mean_field = mean_field.to_cpu()
        calculation = mp.MP2(mean_field)
        calculation.kernel()
        implementation = "mp2/chemflow-pyscf"
        versions = {"pyscf": pyscf.__version__}
    values[identifier] = {
        "energy": float(calculation.e_tot), "device": device["id"],
        "question": source["question"],
        "method": {"family": "MP2", "basis": source["method"]["basis"],
                   "reference": "RHF"},
        "implementation": implementation, "versions": versions,
    }


@implements("active_space")
def _op_active_space(ctx, node, identifier, operation, inputs):
    from chemflow.impls.pyscf_impls import active_from_scf

    values = ctx.values
    store = ctx.store
    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"Unsupported active_space device: {device['kind']}")
    active = node["params"]
    source = inputs[0]
    if source.get("gpu_scf"):
        from types import SimpleNamespace

        data = SimpleNamespace(mf=source["data"].mf.to_cpu(),
                               mol=source["data"].mol,
                               model=source["data"].model)
    else:
        data = source["data"]
    values[identifier] = {
        **inputs[0],
        "data": active_from_scf(data, active["electrons"],
                                active["orbitals"], store),
        "device": device["id"],
        "gpu_scf": False,
    }


@implements("sector.shift")
def _op_sector_shift(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.types import Sector

    original = inputs[0]
    values[identifier] = {
        **original,
        "sector": Sector(n_alpha=node["params"]["n_alpha"],
                         n_beta=node["params"]["n_beta"],
                         n_orb=original["data"].sector.n_orb),
    }


@implements("ccsd.amplitudes")
def _op_ccsd_amplitudes(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    store = ctx.store
    from chemflow.impls.lucj import ccsd_seed_from_integrals

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("CCSD initialization currently requires a CPU")
    data = inputs[0]["data"]
    seed = ccsd_seed_from_integrals(data.hamiltonian, data.sector, store)
    if seed is None:
        raise ValueError("CCSD initialization is unavailable for this sector")
    values[identifier] = {"vector": seed.to_vector().tolist(),
                          "provenance": {
                              "question": inputs[0]["question"],
                              "method": {"family": "CCSD-to-LUCJ initialization"},
                              "implementation": "ccsd.amplitudes/chemflow-pyscf",
                              "device": device["id"],
                              "versions": {"pyscf": pyscf.__version__},
                          }}


@implements(
    "scf.homo",
    "scf.lumo",
    "scf.homo_alpha",
    "scf.homo_beta",
    "scf.lumo_alpha",
    "scf.lumo_beta",
)
def _op_scf_homo(ctx, node, identifier, operation, inputs):
    values = ctx.values
    import numpy as np

    result = inputs[0]
    spin_channel = (0 if operation.endswith("_alpha") else
                    1 if operation.endswith("_beta") else None)
    if result.get("gpu_scf") and spin_channel is not None:
        if "_canonical_orbitals" not in result:
            cpu_mean_field = result["data"].mf.to_cpu()
            result["_canonical_orbitals"] = (
                cpu_mean_field.mo_occ,
                _canonical_unrestricted_energies(cpu_mean_field))
        occupations, energies = (np.asarray(values[spin_channel])
                                 for values in result["_canonical_orbitals"])
    elif result.get("gpu_scf"):
        import cupy as cp

        occupations = cp.asnumpy(result["data"].mf.mo_occ)
        energies = cp.asnumpy(result["data"].mf.mo_energy)
    else:
        occupation = result["data"].mf.mo_occ
        energy = result["data"].mf.mo_energy
        if spin_channel is not None:
            occupation = occupation[spin_channel]
            energy = energy[spin_channel]
        occupations = np.asarray(occupation)
        energies = np.asarray(energy)
    occupied = np.flatnonzero(occupations > 0)
    virtual = np.flatnonzero(occupations == 0)
    is_homo = operation.startswith("scf.homo")
    if not len(occupied) and is_homo or not len(virtual) and not is_homo:
        raise ValueError(f"{operation} is unavailable in this basis")
    index = occupied[-1] if is_homo else virtual[0]
    values[identifier] = {
        "value": float(energies[index]), "unit": "hartree", "variance": 0.0,
        "provenance": {"question": {**result["question"],
                                    "property": operation.removeprefix("scf.")},
                       "method": {**result["method"],
                                  **({"orbital_convention": "canonical_fock_from_density"}
                                     if spin_channel is not None else {})},
                       "implementation": (operation + "/pyscf-canonical-from-gpu"
                                          if result.get("gpu_scf") and
                                          spin_channel is not None else
                                          operation + "/chemflow-pyscf"),
                       "device": result["device"], "versions": result["versions"],
                       "validity_flags": ["Orbital gap is not the fundamental gap"]},
    }


@implements("scf.energy", "mp2.energy", "fci.energy", "sci.energy", "classical.energy",
            "ed.energy")
def _op_scf_energy(ctx, node, identifier, operation, inputs):
    values = ctx.values
    result = inputs[0]
    values[identifier] = {
        "value": result["energy"], "unit": "hartree", "variance": 0.0,
        "provenance": {**{key: result[key] for key in
                          ("question", "method", "implementation", "device", "versions")},
                       **({"validity_flags": result["validity_flags"]}
                          if "validity_flags" in result else {})},
    }
