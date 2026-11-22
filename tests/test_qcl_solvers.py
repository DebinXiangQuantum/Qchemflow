"""QCL primitives for the kernels that had none, each against an independent reference.

DMRG, statevector and ADAPT VQE, EOM-CCSD, climbing-image NEB, MACE, AutoDock Vina,
PubChem names, fragment QM binding, subspace Lehmann spectra and the QM/MM CASCI solvers.
"""

import json
from pathlib import Path

import numpy as np
import pytest

qcl = pytest.importorskip("qcl")

from chemflow import bench

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "tests" / "data" / "181L.pdb"
TOOLS = ROOT.parent / "chemflow-tools-env" / "bin" / "python"
H2 = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
H4 = qcl.Molecule("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0", basis="sto-3g")
WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
AIM = {"U": 2.0, "eps_bath": [-1.2, 0.0, 1.2], "t_bath": [0.5, 0.6, 0.5], "unit": "hartree"}
AIM_SECTOR = {"n_alpha": 2, "n_beta": 2, "n_orb": 4}
OMEGA = {"start": -4.0, "stop": 4.0, "n": 801, "unit": "hartree"}


def _cpu():
    return qcl.Cluster.local().cpus[0]


def _space(molecule, active, cpu):
    return qcl.active_space(qcl.scf.hf(molecule).to(cpu), active).to(cpu)


def _hubbard_ring():
    operator = qcl.models.hubbard(qcl.lattice.ring(6), t=1.0, U=4.0)
    sector = qcl.Sector(n=6, sz=0)
    return operator, sector, qcl.lattice.particles(operator, sector)


# ---- DMRG -------------------------------------------------------------------------------

def test_dmrg_reproduces_exact_lattice_and_active_space_energies():
    pytest.importorskip("pyblock2")

    @qcl.workflow
    def ground(cpu):
        operator, sector, particles = _hubbard_ring()
        space = _space(H4, qcl.Active(4, 4), cpu)
        return {"ring_dmrg": qcl.dmrg.solve(operator, particles,
                                            bond_dimension=100).to(cpu).energy,
                "ring_ed": qcl.ed.ground_state(operator, sector=sector).to(cpu).energy,
                "h4_dmrg": qcl.dmrg.solve(space.hamiltonian, space.sector).to(cpu).energy,
                "h4_fci": qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy}

    plan = ground.compile(_cpu())
    assert plan.relations()["ring_dmrg"] == plan.relations()["h4_dmrg"] == (
        "variational_upper_bound")
    energies = plan.run()
    assert energies["ring_ed"].value == pytest.approx(bench._hubbard_oracle(6, 1., 4., True),
                                                      abs=1e-9)
    for name in ("ring", "h4"):
        exact = energies[f"{name}_ed" if name == "ring" else f"{name}_fci"].value
        assert exact - 1e-9 <= energies[f"{name}_dmrg"].value < exact + 1e-7
    assert energies["h4_dmrg"].provenance["implementation"] == "dmrg.solve/block2"


def test_dmrg_checks_its_inputs_at_compile_time():
    pytest.importorskip("pyblock2")

    @qcl.workflow
    def ground(cpu):
        operator, _, particles = _hubbard_ring()
        return qcl.dmrg.solve(operator, particles, bond_dimension=0).to(cpu).energy

    with pytest.raises(qcl.CompileError, match="bond dimension"):
        ground.compile(_cpu())


# ---- statevector VQE and ADAPT-VQE --------------------------------------------------------

def test_statevector_vqe_and_adapt_are_upper_bounds_close_to_fci():
    @qcl.workflow
    def ground(cpu):
        h2 = _space(H2, qcl.Active(2, 2), cpu)
        lih = _space(qcl.Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g"), qcl.Active(2, 3), cpu)
        h4 = _space(H4, qcl.Active(4, 4), cpu)
        outputs = {}
        for name, space in (("h2", h2), ("lih", lih), ("h4", h4)):
            outputs[f"{name}_fci"] = qcl.fci.solve(space.hamiltonian,
                                                   space.sector).to(cpu).energy
        for name, space in (("h2", h2), ("lih", lih)):
            outputs[f"{name}_vqe"] = qcl.vqe.statevector(space.hamiltonian,
                                                         space.sector).to(cpu).energy
        outputs["h4_adapt"] = qcl.vqe.adapt(h4.hamiltonian, h4.sector).to(cpu).energy
        return outputs

    plan = ground.compile(_cpu())
    relations = plan.relations()
    assert {relations[name] for name in ("h2_vqe", "lih_vqe", "h4_adapt")} == {
        "variational_upper_bound"}
    energies = plan.run()
    for name in ("h2", "lih"):
        exact = energies[f"{name}_fci"].value
        assert exact - 1e-9 <= energies[f"{name}_vqe"].value < exact + 1e-6
    exact = energies["h4_fci"].value
    assert exact - 1e-9 <= energies["h4_adapt"].value < exact + 1.6e-3
    method = energies["h4_adapt"].provenance["method"]
    assert 0 < method["n_operators"] <= 40 and len(method["selected_operators"]) == (
        method["n_operators"])
    assert any("not deployable" in flag
               for flag in energies["h2_vqe"].provenance["validity_flags"])


def test_vqe_rejects_unknown_ansatz():
    @qcl.workflow
    def ground(cpu):
        space = _space(H2, qcl.Active(2, 2), cpu)
        return qcl.vqe.statevector(space.hamiltonian, space.sector,
                                   ansatz="hea").to(cpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        ground.compile(_cpu())


# ---- EOM-CCSD ---------------------------------------------------------------------------

def test_eom_ip_and_ea_match_pyscf():
    from pyscf import cc, gto, scf
    from pyscf.cc import eom_rccsd

    @qcl.workflow
    def charges(cpu):
        water = qcl.Molecule(WATER, basis="6-31g")
        return {"ip": qcl.eom.ip(water).to(cpu), "ea": qcl.eom.ea(water).to(cpu)}

    plan = charges.compile(_cpu())
    assert plan.relations() == {"ip": "approximate", "ea": "approximate"}
    values = plan.run()
    solver = cc.CCSD(scf.RHF(gto.M(atom=WATER, basis="6-31g", verbose=0)).run()).run()
    assert values["ip"].value == pytest.approx(eom_rccsd.EOMIP(solver).kernel(nroots=1)[0],
                                               abs=1e-7)
    assert values["ea"].value == pytest.approx(-eom_rccsd.EOMEA(solver).kernel(nroots=1)[0],
                                               abs=1e-7)
    assert 0.3 < values["ip"].value < 0.6


def test_eom_needs_a_closed_shell_molecule():
    @qcl.workflow
    def charges(cpu):
        return qcl.eom.ip(qcl.Molecule("O 0 0 0; H 0 0 0.97", "sto-3g", spin=1)).to(cpu)

    with pytest.raises(qcl.CompileError) as caught:
        charges.compile(_cpu())
    assert caught.value.diagnostics[0]["code"] == "QCL-E105"


# ---- NEB --------------------------------------------------------------------------------

def test_neb_finds_the_hcn_to_hnc_barrier_with_gfn2():
    pytest.importorskip("tblite")
    pytest.importorskip("ase")

    @qcl.workflow
    def path(cpu):
        return qcl.neb(qcl.Molecule("C 0 0 0; N 0 0 1.16; H 0 0 -1.07", "sto-3g"),
                       qcl.Molecule("C 0 0 0; N 0 0 1.17; H 0 0 2.17", "sto-3g"),
                       method="gfn2", images=7).to(cpu)

    band = path.compile(_cpu()).run()
    energies = band.provenance["energies_hartree"]
    climbing = band.provenance["climbing_image"]
    assert len(energies) == 9 and energies[climbing] == max(energies)
    assert band.provenance["barrier_hartree"] > 0.02
    assert band.transition.atoms == band.provenance["images"][climbing]["atoms"]
    assert len(band.endpoints) == 2


def test_neb_endpoints_must_share_their_model():
    @qcl.workflow
    def path(cpu):
        return qcl.neb(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                       qcl.Molecule("H 0 0 0; H 0 0 0.80", "6-31g")).to(cpu)

    with pytest.raises(qcl.CompileError, match="basis, charge and spin"):
        path.compile(_cpu())


# ---- MACE -------------------------------------------------------------------------------

def _mace_weights():
    import os

    from chemflow.domains.baselines import DEFAULT_MACE_MODEL

    path = Path(os.environ.get("CHEMFLOW_MACE_MODEL", DEFAULT_MACE_MODEL)).expanduser()
    if not path.is_file():
        pytest.skip("MACE weights unavailable")
    return path


def test_mace_energy_matches_the_native_calculator_and_optimization_lowers_it():
    pytest.importorskip("mace")
    weights = _mace_weights()
    from ase import Atoms
    from ase.units import Hartree
    from mace.calculators import mace_mp

    @qcl.workflow
    def potential(cpu):
        water = qcl.Molecule(WATER, basis="sto-3g")
        optimized = qcl.engines.mace.optimize(water).to(cpu)
        return {"start": qcl.engines.mace.energy(water).to(cpu),
                "optimized": qcl.engines.mace.energy(optimized).to(cpu)}

    plan = potential.compile(_cpu())
    assert plan.relations()["start"] == "equivalent"
    values = plan.run()
    atoms = Atoms("OHH", positions=[[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    atoms.calc = mace_mp(model=str(weights), default_dtype="float64", device="cpu")
    assert values["start"].value == pytest.approx(atoms.get_potential_energy() / Hartree,
                                                  abs=1e-8)
    assert values["optimized"].value <= values["start"].value + 1e-10


def test_mace_refuses_charged_molecules_at_compile_time():
    pytest.importorskip("mace")

    @qcl.workflow
    def potential(cpu):
        return qcl.engines.mace.energy(qcl.Molecule("O 0 0 0; H 0 0 0.97", "sto-3g",
                                                    charge=-1)).to(cpu)

    with pytest.raises(qcl.CompileError, match="charge or spin"):
        potential.compile(_cpu())


# ---- docking and ligand names --------------------------------------------------------------

AD_TYPES = {"C": "C", "N": "N", "O": "OA", "S": "SA"}


def _receptor_and_pocket():
    """A heavy-atom PDBQT of T4 lysozyme L99A chain A and the bound benzene's centroid."""
    lines, ligand = [], []
    for line in PDB.read_text().splitlines():
        if line.startswith("HETATM") and line[17:20] == "BNZ":
            ligand.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        element = line[76:78].strip()
        if line.startswith("ATOM") and line[21] == "A" and element in AD_TYPES:
            lines.append(f"{line[:54]}{1.0:6.2f}{0.0:6.2f}    {0.0:+6.3f} "
                         f"{AD_TYPES[element]:<2}")
    return "\n".join(lines) + "\n", np.mean(ligand, axis=0).tolist()


def test_vina_docks_benzene_into_the_l99a_cavity_reproducibly():
    for module in ("vina", "meeko", "rdkit"):
        pytest.importorskip(module)
    receptor, center = _receptor_and_pocket()

    @qcl.workflow
    def score(cpu):
        return qcl.drug.dock("c1ccccc1", receptor, center=center, box=(16, 16, 16),
                             seed=1).to(cpu)

    plan = score.compile(_cpu())
    assert plan.relations() == "approximate"
    first, second = plan.run(), plan.run()
    assert first.unit == "kcal/mol" and -8.0 < first.value < -2.0
    assert second.value == first.value


ASPIRIN = {"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
           "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
           "retrieved_at": "2026-10-13T00:00:00+00:00"}


def test_ligand_names_resolve_from_the_cache_and_refuse_undeclared_network(
        tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    cache = tmp_path / "names.json"
    cache.write_text(json.dumps({"aspirin": ASPIRIN}))
    monkeypatch.setenv("CHEMFLOW_NAME_CACHE", str(cache))

    @qcl.workflow
    def names(cpu, items):
        return qcl.drug.ligands(items).to(cpu)

    result = names.compile(_cpu(), ["aspirin"]).run()
    entry = result.data["ligands"]["aspirin"]
    assert entry["cached"] and entry["cid"] == 2244 and entry["inchikey"] == ASPIRIN["InChIKey"]
    with pytest.raises(RuntimeError, match="network access was not declared"):
        names.compile(_cpu(), ["ibuprofen"]).run()


# ---- fragment QM binding -------------------------------------------------------------------

def _capped_side_chain(atoms):
    """The reviewer's capping rule: cut Cα–Cβ, cap with H at 1.09 Å along Cβ→Cα."""
    names = {name: np.asarray(xyz) for name, _, xyz in atoms}
    backbone = {"N", "CA", "C", "O", "H", "HA", "OXT", "H2", "H3"}
    side = [(element, list(xyz)) for name, element, xyz in atoms if name not in backbone]
    direction = names["CA"] - names["CB"]
    side.append(("H", (names["CB"] + 1.09 * direction / np.linalg.norm(direction)).tolist()))
    return side


def test_fragment_interactions_match_an_independent_counterpoise_calculation():
    pytest.importorskip("openmm")
    if not TOOLS.exists():
        pytest.skip("binding side environment unavailable")
    from pyscf import gto, scf

    pdb = PDB.read_text()

    @qcl.workflow
    def fragments(cpu):
        return qcl.binding.fragments(
            {"pdb_text": pdb, "chains": ["A"]},
            {"source": {"pdb_text": pdb, "residue": "BNZ", "smiles": "c1ccccc1"}},
            residues=["A:99"]).to(cpu)

    plan = fragments.compile(_cpu())
    assert plan.relations() == "approximate"
    result = plan.run(timeout_s=1800)
    provenance = result.provenance
    ligand = [(element, xyz) for element, xyz in provenance["ligand_atoms"]]
    side = _capped_side_chain(provenance["residue_atoms"]["A:99"])

    def energy(ghost_ligand, ghost_side):
        atoms = [("ghost-" + element if ghost else element, xyz)
                 for fragment, ghost in ((ligand, ghost_ligand), (side, ghost_side))
                 for element, xyz in fragment]
        return scf.RHF(gto.M(atom=atoms, basis="sto-3g", verbose=0)).run().e_tot

    oracle = energy(False, False) - energy(False, True) - energy(True, False)
    assert provenance["per_residue"]["A:99"] == pytest.approx(oracle, abs=1e-7)
    assert result.value == pytest.approx(oracle, abs=1e-7)


def test_fragment_settings_are_checked_at_compile_time():
    @qcl.workflow
    def fragments(cpu):
        return qcl.binding.fragments({"pdb_text": "ATOM\n"}, {"source": {"smiles": "C"}},
                                     method="dft").to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        fragments.compile(_cpu())


# ---- subspace Lehmann spectra -----------------------------------------------------------------

def _l1(left, right):
    omega = np.linspace(OMEGA["start"], OMEGA["stop"], OMEGA["n"])
    return float(np.trapezoid(np.abs(np.asarray(left) - np.asarray(right)), omega))


def test_full_subspaces_are_exact_and_sampled_ones_are_close_but_approximate():
    @qcl.workflow
    def spectra(cpu):
        options = {"orbital": 0, "spin": "alpha", "omega": OMEGA, "eta": 0.1}
        return {"exact": qcl.dynamics.spectral_exact(AIM, AIM_SECTOR, **options).to(cpu),
                "full": qcl.dynamics.spectral_subspace(AIM, AIM_SECTOR, subspace="full",
                                                       **options).to(cpu),
                "sampled": qcl.dynamics.spectral_subspace(AIM, AIM_SECTOR, shots=20000,
                                                          **options).to(cpu)}

    plan = spectra.compile(_cpu())
    assert plan.relations() == {"exact": "equivalent", "full": "equivalent",
                                "sampled": "approximate"}
    values = plan.run()
    assert _l1(values["full"].spectral, values["exact"].spectral) < 1e-8
    # The A1a reviewer measured L1 = 0.081 for this seeded, singles-closed design.
    assert 1e-4 < _l1(values["sampled"].spectral, values["exact"].spectral) < 0.1


# ---- QM/MM CASCI solvers ------------------------------------------------------------------------

def _water_dimer():
    lines = []
    for index, (element, xpos, ypos, zpos) in enumerate(bench.WATER):
        atom = ("O", "H1", "H2")[index % 3]
        lines.append(f"HETATM{index + 1:5d} {atom:<4} HOH A{index // 3 + 1:4d}    "
                     f"{xpos:8.3f}{ypos:8.3f}{zpos:8.3f}  1.00  0.00          {element:>2}")
    return "\n".join(lines) + "\nEND\n", [[x / 10, y / 10, z / 10] for _, x, y, z in bench.WATER]


def test_qmmm_sqd_is_a_bound_on_the_casci_energy():
    pytest.importorskip("openmm")
    pdb, positions = _water_dimer()

    @qcl.workflow
    def water(cpu):
        options = {"method": "casci", "cas": (4, 4)}
        return {"fci": qcl.qmmm.energy(pdb, positions, [0, 1, 2], **options).to(cpu),
                "sqd": qcl.qmmm.energy(pdb, positions, [0, 1, 2], solver="sqd", shots=2000,
                                       seed=3, **options).to(cpu),
                "extended": qcl.qmmm.energy(pdb, positions, [0, 1, 2], solver="sqd",
                                            shots=2000, seed=3, extend=True,
                                            **options).to(cpu)}

    plan = water.compile(_cpu())
    assert plan.relations() == {"fci": "equivalent", "sqd": "variational_upper_bound",
                                "extended": "variational_upper_bound"}
    energies = plan.run()
    # SQD from 2000 LUCJ samples is a bound within a few mHa of the CASCI energy; the
    # single-excitation extension enlarges the subspace, so it is a tighter bound.
    fci, sqd, extended = (energies[key].value for key in ("fci", "sqd", "extended"))
    assert fci - 1e-9 <= extended <= sqd + 1e-9 and sqd < fci + 5e-3
    solver = energies["extended"].provenance["qm_solver"]
    assert solver["n_determinants"] >= energies["sqd"].provenance["qm_solver"]["n_determinants"]
    assert energies["sqd"].provenance["method"]["family"] == "QM/MM-CASCI[4, 4]/SQD(statevector)"


def test_qmmm_gpu_fci_matches_cpu_fci():
    pytest.importorskip("openmm")
    from chemflow.backends.gpu import cuda_available

    if not cuda_available():
        pytest.skip("no CUDA device")
    pdb, positions = _water_dimer()

    @qcl.workflow
    def water(cpu, gpu):
        options = {"method": "casci", "cas": (4, 4)}
        return {"cpu": qcl.qmmm.energy(pdb, positions, [0, 1, 2], **options).to(cpu),
                "gpu": qcl.qmmm.energy(pdb, positions, [0, 1, 2], solver="gpu_fci",
                                       **options).to(gpu)}

    plan = water.compile(_cpu(), qcl.Device("gpu0", "gpu"))
    assert plan.relations() == {"cpu": "equivalent", "gpu": "equivalent"}
    energies = plan.run()
    assert energies["gpu"].value == pytest.approx(energies["cpu"].value, abs=1e-8)


@pytest.mark.parametrize("solver,sampler,device,code", [
    ("gpu_fci", "statevector", "cpu", "QCL-E203"),
    ("sqd", "gpu", "cpu", "QCL-E203"),
    ("sqd", "statevector", "gpu", "QCL-E203"),
    ("sqd", "lqcloud", "cpu", "QCL-E101"),
])
def test_qmmm_solver_placement_is_checked_at_compile_time(solver, sampler, device, code):
    pdb, positions = _water_dimer()
    target = _cpu() if device == "cpu" else qcl.Device("gpu0", "gpu")

    @qcl.workflow
    def water(place):
        return qcl.qmmm.energy(pdb, positions, [0, 1, 2], method="casci", cas=(4, 4),
                               solver=solver, sampler=sampler).to(place)

    with pytest.raises(qcl.CompileError) as caught:
        water.compile(target)
    assert caught.value.diagnostics[0]["code"] == code
