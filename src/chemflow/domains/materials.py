"""Site-basis Hubbard Hamiltonians and primitive-cell periodic references."""

from dataclasses import dataclass, replace

import numpy as np

from chemflow import build
from chemflow.diagnostics import Diagnostic
from chemflow.ids import content_hash
from chemflow.impls.pyscf_impls import MolecularData
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.relations import TargetRelation
from chemflow.types import BackendKind, ConstantLedger, Hamiltonian, OrbitalSpace, Sector
from chemflow.units import Q


def site_reference(h1, eri, n_alpha, n_beta):
    from pyscf import gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = n_alpha + n_beta
    mol.spin = n_alpha - n_beta
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(len(h1))
    mf._eri = eri
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.damp = 0.5
    mf.level_shift = 0.5
    mf.kernel(dm0=np.eye(len(h1)) * mol.nelectron / len(h1))
    attempts = ["restricted"]
    stable = False
    if mf.converged:
        _, _, stable, _ = mf.stability(return_status=True)
    if not mf.converged or not stable:
        attempts.append("unrestricted_staggered")
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(len(h1))
        mf._eri = eri
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.damp = 0.5
        mf.level_shift = 0.5
        ordering = np.argsort(np.arange(len(h1)) % 2, kind="stable")
        density = np.zeros((2, len(h1), len(h1)))
        density[0, ordering[:n_alpha], ordering[:n_alpha]] = 1
        density[1, ordering[::-1][:n_beta], ordering[::-1][:n_beta]] = 1
        mf.kernel(dm0=density)
        if mf.converged:
            orbitals, _, stable, _ = mf.stability(return_status=True)
            for _ in range(3):
                if stable:
                    break
                mf.kernel(dm0=mf.make_rdm1(orbitals, mf.mo_occ))
                orbitals, _, stable, _ = mf.stability(return_status=True)
    if not mf.converged:
        raise ValueError("Lattice mean-field reference did not converge")
    if not stable:
        raise ValueError("Lattice mean-field reference is internally unstable")
    mf.chemflow_diagnostics = {"scf_settings": {
        "damping": mf.damp, "level_shift": mf.level_shift, "conv_tol": mf.conv_tol,
        "max_cycle": mf.max_cycle, "fractional_occupation": False, "attempts": attempts},
        "stability": {"internal": bool(stable)}, "reference": mf.__class__.__name__}
    return mf


@dataclass
class LatticeData(MolecularData):
    lattice_id: str = ""

    def for_sector(self, sector, store):
        requested = Sector(**{key: sector[key] for key in ("n_alpha", "n_beta", "n_orb")})
        if requested.n_orb != self.hamiltonian.n_orb:
            raise ValueError("Lattice sector orbital count must equal n_sites")
        if requested == self.sector:
            return self
        return replace(self, sector=requested, mf=None, mol=None)

    def mean_field(self, store):
        if self.mf is None:
            self.mf = site_reference(store.get(self.hamiltonian.h1_ref),
                                     store.get(self.hamiltonian.h2_ref),
                                     self.sector.n_alpha, self.sector.n_beta)
            self.mol = self.mf.mol
        return self.mf


def lattice_arrays(model="hubbard", geometry="chain", n_sites=None, t=1., U=4.,
                   unit="hartree", shape=None):
    if model != "hubbard" or geometry not in ("chain", "ring", "square"):
        raise ValueError("Supported lattices: Hubbard chain, ring, square")
    if geometry == "square":
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
                any(type(size) is not int or size < 1 for size in shape)):
            raise ValueError("Square lattice requires positive integer shape [nx, ny]")
        size = shape[0] * shape[1]
        if n_sites is not None and n_sites != size:
            raise ValueError("n_sites differs from shape product")
        n_sites = size
    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    hopping, repulsion = Q(t, unit).to("hartree").value, Q(U, unit).to("hartree").value
    if not np.isfinite([hopping, repulsion]).all():
        raise ValueError("Lattice couplings must be finite")
    bonds = set()
    if geometry == "square":
        for row in range(shape[0]):
            for column in range(shape[1]):
                site = row * shape[1] + column
                if row + 1 < shape[0]:
                    bonds.add((site, site + shape[1]))
                if column + 1 < shape[1]:
                    bonds.add((site, site + 1))
    else:
        bonds.update((site, site + 1) for site in range(n_sites - 1))
        if geometry == "ring" and n_sites > 2:
            bonds.add((0, n_sites - 1))
    h1 = np.zeros((n_sites, n_sites))
    eri = np.zeros((n_sites,) * 4)
    for left, right in bonds:
        h1[left, right] = h1[right, left] = -hopping
    for site in range(n_sites):
        eri[site, site, site, site] = repulsion
    return h1, eri


def lattice_model(store, **params):
    h1, eri = lattice_arrays(**params)
    size = len(h1)
    sector = Sector(n_alpha=(size + 1) // 2, n_beta=size // 2, n_orb=size)
    space = OrbitalSpace(coefficients_ref=store.put(np.eye(size)), n_orb=size,
                         n_elec=size, ordering="sites")
    ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), n_orb=size,
                      space=content_hash(space), ledger=ConstantLedger())
    identity = content_hash(params)
    return LatticeData(mol=None, mf=None, hamiltonian=ham, space=space, sector=sector,
                       model_id=identity, geometry_id=identity, lattice_id=identity)


def periodic_system(store, lattice, atoms, basis, pseudo=None, unit="angstrom"):
    vectors = np.asarray(lattice, dtype=float)
    if (vectors.shape != (3, 3) or not np.isfinite(vectors).all() or
            abs(np.linalg.det(vectors)) < 1e-9):
        raise ValueError("Periodic lattice must be finite and nonsingular 3x3 vectors")
    if unit not in ("angstrom", "bohr"):
        raise ValueError("Periodic lattice unit must be angstrom or bohr")
    if not pseudo:
        raise ValueError("Periodic reference requires an explicit pseudopotential")
    return {"lattice": lattice, "atoms": atoms, "basis": basis, "pseudo": pseudo, "unit": unit}


def periodic_scf(store, system, method="rhf", kmesh=None, supercell=None, exxdiv="ewald"):
    from pyscf.pbc import gto, scf, tools

    if method != "rhf" or (kmesh is not None and supercell is not None):
        raise ValueError("Use rhf and either kmesh or supercell, not both")
    mesh = kmesh if kmesh is not None else supercell if supercell is not None else [1, 1, 1]
    if len(mesh) != 3 or any(type(size) is not int or size < 1 for size in mesh):
        raise ValueError("Periodic mesh must contain three positive integers")
    cell = gto.Cell()
    cell.a = system["lattice"]
    cell.atom, cell.unit = system["atoms"], system["unit"]
    cell.basis, cell.pseudo = system["basis"], system["pseudo"]
    cell.verbose = 0
    cell.build()
    divisor = 1
    if supercell is not None:
        cell = tools.super_cell(cell, supercell)
        divisor = int(np.prod(supercell))
        mf = scf.RHF(cell, exxdiv=exxdiv)
    else:
        mf = scf.KRHF(cell, cell.make_kpts(mesh), exxdiv=exxdiv)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise ValueError("Periodic SCF did not converge")
    return float(mf.e_tot / divisor)


def context_check(node, inputs, context, nodes):
    if node.op == "lattice_model":
        identity = content_hash(node.params)
        context.update(lattice_id=identity, hamiltonian_id=identity, model_id=identity,
                       model_protocol_id=identity)
    elif node.op == "ground_energy" and "lattice_id" in context:
        context["kind"] = "lattice_energy"
    elif node.op == "periodic_system":
        context["cell_id"] = content_hash(node.params)
    elif node.op == "periodic_scf":
        context.update(cell_id=inputs.get("system", {}).get("cell_id"),
                       kind="energy_per_cell", unit="hartree")
    return []


def static_check(program, registry):
    issues = []
    nodes = {node.id: node for node in program.nodes}
    for node in program.nodes:
        try:
            if node.op == "lattice_model":
                lattice_arrays(**node.params)
            elif node.op == "periodic_system":
                periodic_system(None, **node.params)
            elif node.op == "ground_energy":
                ham = nodes.get(node.inputs.get("hamiltonian"))
                sector = nodes.get(node.inputs.get("sector"))
                if ham and ham.op == "lattice_model" and sector:
                    size = ham.params.get("n_sites", int(np.prod(ham.params.get("shape", []))))
                    if sector.params.get("n_orb") != size:
                        raise ValueError("Lattice sector n_orb differs from n_sites")
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(Diagnostic(code="CF-E080", node=node.id, message=str(exc),
                                     fix_hints=["Supply a valid lattice or periodic system"]))
    return issues


def register(registry):
    from chemflow.domains.embedding import ewf_embedding
    from chemflow.domains.lattice_correlations import register as register_correlations

    register_correlations(registry)

    for name, inputs, output, adapter, params in (
        ("lattice_model", {}, "hamiltonian", lattice_model,
         {"model", "geometry", "n_sites", "t", "U", "unit", "shape"}),
        ("periodic_system", {}, "periodic_system", periodic_system,
         {"lattice", "unit", "atoms", "basis", "pseudo"}),
        ("periodic_scf", {"system": "periodic_system"}, "estimate", periodic_scf,
         {"method", "kmesh", "supercell", "exxdiv"}),
    ):
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
                                       output=output, semantics=name))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
                          backend_kind=BackendKind.cpu_numeric, params=frozenset(params)))
    registry.register(RefinementRule(target="ground_energy", template_id="ewf",
                                     relation=TargetRelation.approximate))
    registry.register(ImplementationSpec(id="lattice_hf", version="1", target="ground_energy",
                      implements="hf", adapter=lattice_hf, backend_kind=BackendKind.cpu_numeric,
                      params=frozenset({"s", "total_spin"})))
    registry.register(ImplementationSpec(id="ewf_embedding", version="1", target="ground_energy",
                      implements="ewf", adapter=ewf_embedding, backend_kind=BackendKind.cpu_numeric,
                      resource_model=_embedding_resources,
                      params=frozenset({"s", "total_spin", "fragments", "bath", "bath_tol",
                                        "bno_eta", "solver", "shots", "seed"})))
    registry.context_checks.append(context_check)
    registry.checks.append(static_check)


def _embedding_resources(hamiltonian=None, solver="fci", shots=4096, **kwargs):
    fragments = hamiltonian.mol.natm if hamiltonian is not None else None
    return {"sim_shots": (None if fragments is None else fragments * shots)
            if solver == "sqd" else 0}


def lattice_hf(store, hamiltonian, sector, s=None, total_spin=None):
    from chemflow.diagnostics import AdapterError
    from chemflow.ops.molecular import _ground_adapter

    if not isinstance(hamiltonian, LatticeData):
        raise AdapterError("CF-E002", None, "Lattice HF requires a site Hamiltonian",
                           "Use the molecular HF binding for molecules")
    options = {"s": s} if s is not None else (
        {"total_spin": total_spin} if total_spin is not None else {})
    result = _ground_adapter("hf")(store, hamiltonian, sector, **options)
    from chemflow.domains.lattice_correlations import hf_state

    reference = hf_state(hamiltonian, store)
    if reference is not None:
        result[1]["state_ref"] = reference
    return result


def base_plan(record, active=None):
    target = record.spec.target
    if target.kind == "periodic_energy":
        if active is not None:
            raise ValueError("Periodic references do not take a molecular active space")
        with build.program("periodic-controls") as builder:
            system = builder.node("periodic_system", **target.system)
            energies = [builder.node("periodic_scf", system=system, method=target.method,
                                     exxdiv=target.exxdiv, **{mode: list(target.kmesh)})
                        for mode in ("kmesh", "supercell")]
            builder.ir.targets = [node.id for node in energies]
            return builder.to_ir(), builder.ir.targets[:]
    size = target.lattice.get("n_sites", int(np.prod(target.lattice.get("shape", []))))
    expected = {"n_orb": size, "n_elec": target.sector["n_alpha"] + target.sector["n_beta"]}
    if active is not None and active != expected:
        raise ValueError("Active space must match lattice sites and sector filling")
    with build.program("lattice-controls") as builder:
        ham = builder.node("lattice_model", **target.lattice)
        sector = builder.sector(**target.sector)
        controls = []
        for method in ("hf", "fci"):
            energy = builder.ground_energy(ham, sector, **(
                {"s": target.target_spin} if target.target_spin is not None else {}))
            builder.refine(energy, method)
            builder.bind(energy, "fci_pyscf" if method == "fci" else "lattice_hf")
            result = (builder.node("correlation_function", state=energy,
                                  site_pair=list(target.site_pair), quantity=target.quantity)
                      if target.kind == "pairing_correlation" else energy)
            if method == "fci":
                builder.ir.references = [result.id]
            else:
                controls.append(result.id)
                next(item for item in builder.ir.nodes if item.id == energy.id).metadata[
                    "optional_control"] = True
                next(item for item in builder.ir.nodes if item.id == result.id).metadata[
                    "optional_control"] = True
        builder.ir.targets = [*controls, *builder.ir.references]
        return builder.to_ir(), controls


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    nodes = {node.id: node for node in plan.nodes}
    target = record.spec.target
    issues = []
    for identifier in plan.targets:
        node = nodes.get(identifier)
        if target.kind == "pairing_correlation":
            if (node is None or node.op != "correlation_function" or
                    node.params.get("site_pair") != list(target.site_pair) or
                    node.params.get("quantity", "pair_s_wave") != target.quantity):
                issues.append(Diagnostic(code="CF-E120", node=identifier,
                              message="Correlation observable identity mismatch",
                              fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
                continue
            node = nodes.get(node.inputs.get("state"))
        if target.kind == "periodic_energy":
            system = nodes.get(node.inputs.get("system")) if node else None
            if (node is None or node.op != "periodic_scf" or system is None or
                    system.op != "periodic_system" or system.params != target.system or
                    node.params.get("method", "rhf") != target.method or
                    node.params.get("exxdiv", "ewald") != target.exxdiv or
                    tuple(node.params.get("kmesh", node.params.get("supercell", []))) !=
                    target.kmesh):
                issues.append(Diagnostic(code="CF-E120", node=identifier,
                              message="Periodic target identity mismatch",
                              fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
            continue
        ham = nodes.get(node.inputs.get("hamiltonian")) if node else None
        sector = nodes.get(node.inputs.get("sector")) if node else None
        operator = "ground_energy" if target.kind == "pairing_correlation" else target.operator
        if (node is None or node.op != operator or ham is None or
                ham.op != "lattice_model" or ham.params != target.lattice or
                sector is None or sector.params != target.sector or
                contexts[identifier].get("target_spin") != target.target_spin):
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                          message="Lattice target identity mismatch",
                          fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
    return issues
