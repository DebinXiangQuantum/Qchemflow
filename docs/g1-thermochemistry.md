# G1: ideal-gas thermochemistry

The reactions pack supplies `thermochemistry(molecule, model, frequencies)` with
`temperature_K=298.15`, `pressure_Pa=101325`, and an optional positive integer
`symmetry_number` override. The thermal model is `rrho_ideal_gas`.

## Model and units

- Translation uses the ideal-gas Sackur–Tetrode expression at the supplied pressure.
- Rotation uses principal moments about the mass centre. A rotor is linear when
  its smallest moment is less than 0.001 times its largest moment; otherwise all
  three moments enter the nonlinear rigid-rotor partition function. Atoms have
  no rotational contribution. PySCF point-group analysis supplies the rotational
  symmetry number unless overridden.
- Masses are most-abundant-isotope **masses**, not average atomic weights or
  integer mass numbers. The PySCF common-isotope mass table is used with
  higher-precision H, C, N and O values. Constants come from `scipy.constants`.
- Positive real harmonic frequencies contribute vibrational energy and entropy.
  Negative frequencies represent imaginary modes and are excluded, with their
  values recorded in `excluded_imaginary_modes` and their count in `n_imaginary`.
- Electronic entropy is k ln(2S+1), using the molecule's declared spin.

The mapping contains `zpe`, `u_corr` (including ZPE), `h_corr = u_corr + kT`,
and `g_corr = h_corr - T*s`, all in hartree per molecule; `s` is in hartree/K.
It also records temperature, pressure, symmetry number, spin multiplicity,
linear-rotor status and thermal model. Add an electronic energy to a correction
to obtain the corresponding total U, H or G; ZPE is not added a second time.

A maximum gradient above 1e-4 Ha/bohr is refused with CF-E020. Minima may have
no imaginary modes; a geometry declared as a TS may have at most one for this
operator. Reaction TS verification additionally requires exactly one significant
imaginary mode under the existing stationary-point verification policy.
Frequency inputs must describe the same geometry and electronic model.

## Reaction tasks

Add the following to a `reaction_energy` target:

```json
"thermo": {"quantity": "gibbs", "temperature_K": 298.15, "pressure_Pa": 101325}
```

`quantity` accepts `enthalpy` or `gibbs`. Both temperature and pressure are
optional and default as above. Every species is optimized as its declared
minimum or TS at RHF (closed shell) or UHF (open shell), with the target basis.
Frequencies and thermal corrections use that geometry level. Electronic
controls and the active-space FCI reference then use the optimized geometry.
Every reaction sums stoichiometric coefficients times electronic energy plus
the appropriate per-species correction. Omitting `thermo` preserves the existing
electronic-only plan.

The output kinds are `reaction_enthalpy` and `reaction_gibbs_energy`. Context
includes `thermal_model`, `temperature_K`, `pressure_Pa` and `frequency_level`
(a hash of the frequency model and method). Mixing these conditions is refused
with CF-E011. Signed-plan identity checks also prevent changing a frozen task's
conditions or geometry/frequency protocol. Reports label results as, for example,
ΔH(298.15 K) or ΔG(298.15 K, 1 atm).

The FCI reference retains `equivalent` **within the declared electronic-space
and RRHO model**, not exact physical thermodynamics. There is no anharmonicity,
no hindered-rotor treatment, and no low-frequency quasi-RRHO correction. Very
soft modes can dominate entropy. There is no solvent, solution standard-state
conversion, conformer ensemble, or thermally populated excited electronic state.
