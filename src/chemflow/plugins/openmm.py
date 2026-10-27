"""Whole-molecule, nonperiodic electrostatic embedding for snapshot energies.

No quantum forces are provided: QPU cost and latency preclude calling these
solvers at every MD step. Sample trajectories with OpenMM, then evaluate frames.
"""

from copy import deepcopy
from itertools import combinations

import numpy as np
import openmm
from openmm import unit
from pyscf import dft, gto, mcscf, qmmm, scf

KJ_PER_HARTREE = 2625.4996394799


class QMMMCalculator:
    """Add embedded QM energy to a copied System's residual MM energy.

    ``cas=(n_orbitals, n_electrons)`` follows PySCF CASCI. Positions are OpenMM
    Quantities or arrays in nm. Only standard bonded forces and NoCutoff
    NonbondedForce are supported; unsupported force fields fail explicitly.
    """

    def __init__(self, topology, system, qm_atoms, basis, charge=0, spin=0,
                 method="rhf", cas=None, fcisolver=None, xc=None):
        self.topology = topology
        self.atoms = list(topology.atoms())
        chosen = list(qm_atoms)
        if (not chosen or len(set(chosen)) != len(chosen) or
                any(not isinstance(index, (int, np.integer)) or
                    index < 0 or index >= len(self.atoms) for index in chosen)):
            raise ValueError("qm_atoms must be unique valid atom indices")
        self.qm_atoms = sorted(chosen)
        selection = set(chosen)
        if any((left.index in selection) != (right.index in selection)
               for left, right in topology.bonds()):
            raise ValueError("A bond crosses the QM/MM boundary; select whole molecules")
        if system.getNumParticles() != len(self.atoms):
            raise ValueError("Topology and System particle counts differ")
        if any(system.isVirtualSite(index) for index in range(system.getNumParticles())):
            raise ValueError("Virtual sites are not supported")
        self.method = method.lower()
        if self.method not in ("rhf", "hf", "uhf", "rohf", "dft", "rks", "uks"):
            raise ValueError("Unsupported QM method")
        if fcisolver is not None and cas is None:
            raise ValueError("fcisolver requires a CAS space")
        if cas is not None and self.method in ("uhf", "uks"):
            raise ValueError("CASCI requires a restricted reference")
        if self.method in ("dft", "rks", "uks") and not xc:
            raise ValueError("DFT requires an explicit xc functional")
        self.basis, self.charge, self.spin = basis, charge, spin
        self.cas, self.fcisolver, self.xc = cas, fcisolver, xc
        self.system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
        nonbonded = [force for force in self.system.getForces()
                     if isinstance(force, openmm.NonbondedForce)]
        if len(nonbonded) != 1:
            raise ValueError("Exactly one NonbondedForce is required")
        force = nonbonded[0]
        if force.getNonbondedMethod() != openmm.NonbondedForce.NoCutoff:
            raise ValueError("Electrostatic embedding currently requires NoCutoff")
        if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
            raise ValueError("Nonbonded parameter offsets are not supported")
        self.charges = np.array([force.getParticleParameters(index)[0].value_in_unit(
            unit.elementary_charge) for index in range(len(self.atoms))])
        for index in self.qm_atoms:
            _, sigma, epsilon = force.getParticleParameters(index)
            force.setParticleParameters(index, 0, sigma, epsilon)
        for index in range(force.getNumExceptions()):
            left, right, _, sigma, epsilon = force.getExceptionParameters(index)
            if left in selection or right in selection:
                force.setExceptionParameters(index, left, right, 0, sigma,
                                             0 if left in selection and right in selection
                                             else epsilon)
        for left, right in combinations(self.qm_atoms, 2):
            force.addException(left, right, 0, 1, 0, replace=True)
        for force in self.system.getForces():
            if isinstance(force, (openmm.NonbondedForce, openmm.CMMotionRemover)):
                continue
            if isinstance(force, openmm.HarmonicBondForce):
                count, get, put, width = (force.getNumBonds, force.getBondParameters,
                                          force.setBondParameters, 2)
            elif isinstance(force, openmm.HarmonicAngleForce):
                count, get, put, width = (force.getNumAngles, force.getAngleParameters,
                                          force.setAngleParameters, 3)
            elif isinstance(force, (openmm.PeriodicTorsionForce, openmm.RBTorsionForce)):
                count, get, put, width = (force.getNumTorsions, force.getTorsionParameters,
                                          force.setTorsionParameters, 4)
            else:
                raise TypeError("Unsupported MM force in the supplied System")
            for index in range(count()):
                parameters = list(get(index))
                membership = [atom in selection for atom in parameters[:width]]
                if any(membership) and not all(membership):
                    raise ValueError("A bonded term crosses the QM/MM boundary")
                if all(membership):
                    if isinstance(force, openmm.RBTorsionForce):
                        parameters[width:] = [value * 0 for value in parameters[width:]]
                    else:
                        parameters[-1] *= 0
                    put(index, *parameters)
        self.mm_atoms = [index for index in range(len(self.atoms)) if index not in selection]

    def energy(self, positions):
        coordinates = np.asarray(positions.value_in_unit(unit.nanometer)
                                 if unit.is_quantity(positions) else positions, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.isfinite(coordinates).all():
            raise ValueError("Positions must be finite (n_atoms, 3) coordinates")
        molecule = gto.M(atom=[(self.atoms[index].element.symbol, coordinates[index] * 10)
                               for index in self.qm_atoms], unit="angstrom", basis=self.basis,
                         charge=self.charge, spin=self.spin, verbose=0)
        factory = {"rhf": scf.RHF, "hf": scf.HF, "uhf": scf.UHF, "rohf": scf.ROHF,
                   "dft": dft.RKS, "rks": dft.RKS, "uks": dft.UKS}[self.method]
        reference = factory(molecule)
        if self.xc is not None:
            reference.xc = self.xc
        if self.mm_atoms:
            reference = qmmm.mm_charge(reference, coordinates[self.mm_atoms] * 10,
                                        self.charges[self.mm_atoms], unit="angstrom")
        reference.run(conv_tol=1e-11)
        if not reference.converged:
            raise RuntimeError("Embedded SCF did not converge")
        qm_energy = reference.e_tot
        solver_run = {"method": self.method, "converged": True}
        if self.cas is not None:
            calculation = mcscf.CASCI(reference, *self.cas)
            if self.fcisolver is not None:
                calculation.fcisolver = self.fcisolver
            qm_energy = calculation.kernel()[0]
            solver_run = deepcopy(getattr(calculation.fcisolver, "last_run",
                                           {"method": "fci", "device": "local"}))
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(self.system, integrator,
                                 openmm.Platform.getPlatformByName("Reference"))
        try:
            context.setPositions(coordinates * unit.nanometer)
            mm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole) / KJ_PER_HARTREE
        finally:
            del context
            del integrator
        return {"total": float(qm_energy + mm_energy), "qm": float(qm_energy),
                "mm": float(mm_energy), "unit": "hartree",
                "provenance": {"qm_solver": solver_run, "embedding": "electrostatic",
                               "qm_atoms": list(self.qm_atoms), "basis": self.basis,
                               "charge": self.charge, "spin": self.spin}}

    def energies(self, trajectory):
        """Evaluate snapshots, without integrating quantum forces."""
        return [self.energy(positions) for positions in trajectory]
