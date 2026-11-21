# G2: fast classical baselines

> **Status (2026-11-21).** This guide records a milestone built on ChemFlow's earlier
> engine (IR programs, registry operators, static checker and orchestrator), which has
> been removed. The science and the validation below still hold; signed tasks now run as
> QCL screens, and the operations named here map to QCL primitives or plain kernels as
> listed in [legacy-retirement.md](legacy-retirement.md).

Load `load_registry(["molecular", "reactions", "baselines"])` to combine
electronic structure, geometry/thermochemistry operations, and `potential_energy`.
The optional extras are `xtb` (tblite, ASE) and `mlp` (mace-torch, torch).
There is no automatic download, installation, or method fallback. Missing packages
or weights raise **CF-E080**.

## Models and scope

**GFN2-xTB** is a parametrised, self-consistent extended tight-binding method,
designed for geometries, vibrational frequencies, and noncovalent interactions
across a broad range of molecular chemistry (elements through radon). It is not
an ab initio electronic-structure calculation. Molecular charge and spin
(`spin = N_alpha - N_beta`, the number of unpaired electrons in this adapter)
are passed to native tblite as `charge` and `uhf`. Parameter coverage does not
guarantee accuracy for every oxidation state, radical, or bonding environment.

**MACE-MP-0 small** is a materials-trained, PBE-level machine-learned interatomic
potential, trained on Materials Project relaxation trajectories (MPtrj), rather
than an organic-molecule benchmark. Its energies and forces approximate its
training reference; molecular applications can be out of distribution. The
foundation model uses the **MIT licence**. MACE-OFF was not used because of its
different, Academic Software Licence (ASL). These distinctions are also described
in the installed `mace.calculators.foundations_models.mace_mp` and `mace_off`
documentation. This implementation uses float64, CPU execution, nonperiodic
molecules, and no added dispersion correction. Charged or open-shell MACE
requests are refused: this model has no charge/spin-state input.

MACE loads only the local file named by `CHEMFLOW_MACE_MODEL`, defaulting to
`$CHEMFLOW_CACHE_DIR/mace/mace-mp-0-small.model` (by default under `~/.cache/chemflow`). Supply trusted
weights only; model loading uses PyTorch serialization. The energy diagnostics
include the method and SHA-256 of the actual file. An override must still point
to MACE-MP-0 small weights; it is not a selector for arbitrary foundation models.

## Scientific contract

Both methods are **model changes** relative to every electronic-structure
target, not equivalent solvers or variational bounds. `potential_energy` returns
an `Estimate` with kind `potential_energy`, unit hartree, relation `model_change`,
and model protocol equal to the method name. Mixing these estimates with ab
initio energies in reactions is refused with **CF-E011**. Absolute energies from
different potentials are not comparable, especially their differing reference
energy offsets. Provenance records the actual installed package version, e.g.
`tblite:0.7.0` or `mace-torch:0.3.16`.

```python
from chemflow import build
from chemflow.domains import load_registry
from chemflow.runtime import Executor

with build.program("baseline-screen") as builder:
    molecule = builder.molecule("O 0 0 0.117; H 0 .757 -.469; H 0 -.757 -.469")
    model = builder.electronic_model(basis="sto-3g")
    level = {"method": "gfn2-xtb"}
    geometry = builder.node("geometry_optimize", molecule=molecule, model=model,
                            kind="minimum", level=level)
    energy = builder.node("potential_energy", molecule=geometry, method=level["method"])
    modes = builder.node("frequencies", molecule=geometry, model=model, level=level)
    thermal = builder.node("thermochemistry", molecule=geometry, model=model,
                           frequencies=modes, temperature_K=298.15, pressure_Pa=101325.)

results = Executor(load_registry(["molecular", "reactions", "baselines"])).run(builder.to_ir())
```

Baseline geometry/force evaluation ignores the electronic-model basis. The model
input remains required by the shared geometry and thermal operator interfaces.
ASE BFGS searches minima, stopping at maximum atomic force norm ≤ 1e-5 Ha/bohr
(stricter than the required 1e-4). Nonconvergence and transition-state searches
are explicitly refused; BFGS does not locate saddles.

xTB energies and gradients use native tblite in bohr/hartree, not tblite's ASE
energy conversion. Only the optimizer boundary converts gradients to ASE forces.
MACE's eV/angstrom outputs are converted using SciPy physical constants.
Frequencies use central differences of analytic gradients, step **0.005 bohr**,
a symmetrized Hessian, and G1 most-abundant-isotope masses. Mass-weighted rigid
translations and rotations are projected out, including the linear and monatomic
cases. Signed frequencies are in cm⁻¹; `max_gradient` is in Ha/bohr and
`n_imaginary` counts modes below −50 cm⁻¹, matching the RHF reporting convention.
RRHO still refuses any negative minimum mode or a nonstationary geometry.

`frequency_level` is recorded in frequency and thermochemistry results and
contexts (e.g. `gfn2-xtb`, `rhf`); reactions refuse mixed levels with CF-E011.
Thermal reaction tasks may use the same baseline geometry/frequency level for
every species while retaining their declared electronic-energy model. They may
also insert xTB pre-optimization before the final RHF optimization and frequencies.
Nonthermal fixed-geometry task identities are not relaxed by this feature.

## When to use

- Pre-optimize poor initial geometries, then optimize again at the target level.
- Screen conformers using relative energies within one method and molecular state;
  recompute finalists with the intended electronic-structure model.
- Obtain cheap frequencies and RRHO corrections for exploratory thermochemistry,
  keeping the frequency level consistent across every reaction species. Such
  corrections carry the baseline's modelling error; they are not ab initio Hessians.

Resource predictions retain unknown (`None`) CPU-core seconds and wall seconds
until calibrated. The CPU-numeric defaults give known zero GPU seconds,
simulator shots, QPU seconds and QPU shots. Measured execution times are still
recorded by the executor. Baseline energy caching is disabled so changes in local
weights or package availability are not hidden by a previous result.

## Local validation

Run `.venv/bin/python -m pytest -q -p no:cacheprovider` and
`.venv/bin/ruff check --no-cache`. Set `PYTHONDONTWRITEBYTECODE=1`, and point
`MPLCONFIGDIR` and `XDG_CACHE_HOME` outside the repository. No dynamic-library or
OpenMP runtime overrides are required.
Single-threaded validation uses `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.

## MACE process isolation (R104)

ChemFlow never imports torch or MACE into its numerical parent process. It starts
`sys.executable -u -m chemflow.domains._mace_worker` lazily and retains one worker
for energies, BFGS steps, and finite-difference displacements. The worker loads
the float64 model once, serving serialized JSON-lines requests containing symbols,
positions in angstrom, and charge/spin. Responses contain energy in hartree and
gradient in hartree/bohr. Package output is redirected to stderr, leaving stdout
exclusively for the protocol.

The parent checks the local model path and SHA-256. Changing either replaces the
worker, rather than reusing stale weights. Workers close at interpreter exit;
failed or unresponsive workers are reaped and reported as CF-E080 with their
stderr tail (a 300-second response timeout bounds hangs). There is no fallback.
This isolates macOS OpenMP runtimes without changing the process environment.
Independent MACE test oracles must likewise run in separate processes, rather
than importing torch alongside tblite/PySCF in the pytest process.
