"""Isolated chemistry worker; stdout is exclusively the JSON response."""

import contextlib
import io
import json
import random
import sys
import tempfile
import traceback
from pathlib import Path


def serialize_pdb(topology, positions):
    from openmm import app

    stream = io.StringIO()
    app.PDBFile.writeFile(topology, positions, stream, keepIds=True)
    return stream.getvalue()


def protonation_variants(residues, ph, pkas):
    records, variants = {}, []
    defaults = {"HIS": 6.5, "ASP": 3.8, "GLU": 4.5, "LYS": 10.5}
    states = {"HIS": ("HIP", "HIE"), "ASP": ("ASH", "ASP"),
              "GLU": ("GLH", "GLU"), "LYS": ("LYS", "LYN")}
    for residue in residues:
        identifier = f"{residue.chain.id}:{residue.id}"
        if residue.name in defaults:
            pka = pkas.get(identifier, defaults[residue.name])
            state = states[residue.name][int(ph >= pka)]
            records[identifier] = {"name": residue.name, "state": state, "pKa": pka,
                                   "source": "PROPKA" if identifier in pkas else
                                   "standard_pKa_fallback"}
            variants.append(state)
        else:
            variants.append(None)
    return records, variants


def predicted_pkas(prediction):
    return {f"{group.atom.chain_id}:{group.atom.res_num}": float(group.pka_value)
            for group in prediction.conformations["AVR"].groups
            if group.label[:3].strip() in ("HIS", "ASP", "GLU", "LYS")}


def protein_structure(pdb_text, ph=7.0, keep=None, remove_water=True, chains=None):
    import numpy as np
    from openmm import Platform, app, unit
    from pdbfixer import PDBFixer

    class OfflinePDBFixer(PDBFixer):
        def _downloadCCDDefinition(self, residueName):
            return None

    random.seed(0)
    np.random.seed(0)
    fixer = OfflinePDBFixer(pdbfile=io.StringIO(pdb_text),
                            platform=Platform.getPlatformByName("Reference"))
    modeller = app.Modeller(fixer.topology, fixer.positions)
    standard = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    modeller.delete([residue for residue in modeller.topology.residues()
                     if (chains and residue.chain.id not in chains) or
                     (remove_water and residue.name in ("HOH", "WAT")) or
                     (residue.name not in standard and residue.name not in (keep or []) and
                      (remove_water or residue.name not in ("HOH", "WAT")))])
    fixer.topology, fixer.positions = modeller.topology, modeller.positions
    fixer.findMissingResidues()
    chain_list = list(fixer.topology.chains())
    fixer.missingResidues = {key: names for key, names in fixer.missingResidues.items()
                             if 0 < key[1] < len(list(chain_list[key[0]].residues()))}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    heavy = serialize_pdb(fixer.topology, fixer.positions)
    pkas, source = {}, "standard_pKa_fallback"
    try:
        import propka.run
    except ImportError:
        print("PROPKA unavailable: using explicit standard-pKa fallback", file=sys.stderr)
    else:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protein.pdb"
            path.write_text(heavy)
            prediction = propka.run.single(str(path), write_pka=False)
            pkas = predicted_pkas(prediction)
        source = "PROPKA"
    fixer.addMissingHydrogens(ph)
    residues = list(fixer.topology.residues())
    protonation, variants = protonation_variants(residues, ph, pkas)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([atom for atom in modeller.topology.atoms() if atom.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    selected = modeller.addHydrogens(forcefield, pH=ph, variants=variants,
                                     platform=Platform.getPlatformByName("Reference"))
    for residue, state in zip(modeller.topology.residues(), selected, strict=True):
        identifier = f"{residue.chain.id}:{residue.id}"
        if identifier not in protonation and residue.name in ("ARG", "CYS", "TYR"):
            protonation[identifier] = {"name": residue.name, "state": state or residue.name,
                                       "source": "OpenMM hydrogen topology"}
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                    constraints=None)
    nonbonded = next(force for force in system.getForces()
                     if force.__class__.__name__ == "NonbondedForce")
    charge = sum(nonbonded.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge)
                 for index in range(system.getNumParticles()))
    return {"pdb_text": serialize_pdb(modeller.topology, modeller.positions),
            "n_atoms": modeller.topology.getNumAtoms(), "n_residues": len(residues),
            "protonation": protonation, "formal_charge": round(charge),
            "protonation_method": source}


def ligand_parameters(sdf, forcefield="gaff-2.11", charge_method="am1bcc"):
    from openff.toolkit import Molecule
    from openff.units import unit
    from openmm import app
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ligand.sdf"
        path.write_text(sdf)
        molecule = Molecule.from_file(str(path))
        molecule.assign_partial_charges(charge_method)
        generator_type = GAFFTemplateGenerator if forcefield == "gaff-2.11" else \
            SMIRNOFFTemplateGenerator
        generator = generator_type(molecules=[molecule], forcefield=forcefield)
        ffxml = generator.generate_residue_template(molecule)
        app.ForceField(io.StringIO(ffxml))
        return {"ffxml": ffxml, "charges": molecule.partial_charges.m_as(unit.elementary_charge).tolist(),
                "forcefield": forcefield, "charge_method": charge_method}


def mmgbsa(pdb, ffxml, frames=0, minimize=False, seed=0):
    import numpy as np
    from openmm import Context, LangevinMiddleIntegrator, Platform, VerletIntegrator, app, unit

    structure = app.PDBFile(io.StringIO(pdb))
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml", io.StringIO(ffxml))
    platform = Platform.getPlatformByName("Reference")
    ligand = [residue for residue in structure.topology.residues()
              if residue.name in ("BNZ", "UNL", "LIG")]
    if len(ligand) != 1:
        raise ValueError("MM/GBSA requires exactly one BNZ/UNL/LIG residue")
    receptor = app.Modeller(structure.topology, structure.positions)
    receptor.delete(ligand)
    alone = app.Modeller(structure.topology, structure.positions)
    alone.delete([residue for residue in structure.topology.residues() if residue not in ligand])
    ligand_indices = [atom.index for residue in ligand for atom in residue.atoms()]
    receptor_indices = [atom.index for atom in structure.topology.atoms()
                        if atom.residue not in ligand]
    contexts, integrators = [], []
    for topology in (structure.topology, receptor.topology, alone.topology):
        system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
        integrator = VerletIntegrator(0.001)
        integrators.append(integrator)
        contexts.append(Context(system, integrator, platform))
    positions = structure.positions
    simulation = None
    if minimize or frames:
        system = forcefield.createSystem(structure.topology, nonbondedMethod=app.NoCutoff,
                                         constraints=None)
        integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                              0.001 * unit.picoseconds)
        integrator.setRandomNumberSeed(seed or 1)
        simulation = app.Simulation(structure.topology, system, integrator, platform)
        simulation.context.setPositions(positions)
        if minimize:
            simulation.minimizeEnergy(maxIterations=500)
        if frames:
            simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, seed or 1)
        positions = simulation.context.getState(getPositions=True).getPositions()
    samples = []
    for _ in range(max(1, frames)):
        if frames:
            simulation.step(10)
            positions = simulation.context.getState(getPositions=True).getPositions()
        energies = []
        for context, indices in zip(contexts, (None, receptor_indices, ligand_indices), strict=True):
            selected = positions if indices is None else [positions[index] for index in indices]
            context.setPositions(selected)
            energies.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilocalorie_per_mole))
        samples.append(energies)
    means = np.mean(samples, axis=0)
    values = np.asarray(samples) @ np.array([1, -1, -1])
    return {"dG_kcal": float(np.mean(values)), "components": dict(zip(
                ("complex", "receptor", "ligand"), means.tolist(), strict=True)),
            "stderr": (float(np.std(values, ddof=1) / np.sqrt(frames)) if frames > 1 else
                       None if frames == 1 else 0.0),
            "frames": frames}


def main():
    request = json.load(sys.stdin)
    operations = {"protein_structure": protein_structure, "ligand_parameters": ligand_parameters,
                  "mmgbsa": mmgbsa}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = operations[request["operation"]](**request["payload"])
        print(json.dumps({"result": result}, allow_nan=False))
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, ImportError) as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
