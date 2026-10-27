"""Signed-task QCL program for ChemFlow Agent's qcl-run command."""

import qcl


@qcl.workflow
def ground_energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy


def build(task_spec, cluster):
    target = task_spec.target
    molecule = qcl.Molecule(target.molecule["atoms"], target.electronic_model["basis"],
                            charge=target.molecule.get("charge", 0),
                            spin=target.molecule.get("spin", 0),
                            unit=target.molecule.get("unit", "angstrom"))
    active = qcl.Active(target.active["n_elec"], target.active["n_orb"])
    return ground_energy, (molecule, active, cluster.cpus[0]), {}
