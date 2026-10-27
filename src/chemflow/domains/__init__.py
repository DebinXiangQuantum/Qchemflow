"""Opt-in scientific domain packs and frozen-target ownership."""

from dataclasses import dataclass
from importlib import import_module

from chemflow.registry import Registry


@dataclass(frozen=True)
class DomainPack:
    name: str
    target_kinds: tuple[str, ...]
    benchmarks: tuple[str, ...] = ()

    def register(self, registry):
        self.module.register(registry)
        registry.loaded_packs.add(self.name)

    @property
    def module(self):
        return import_module(f"chemflow.domains.{self.name}")

    def identity(self, plan, record, contexts):
        return self.module.identity(plan, record, contexts)

    def base_plan(self, record, active=None):
        return self.module.base_plan(record, active)

    def random_instance(self, family, rng, **size):
        sampler = getattr(self.module, "random_instance", None)
        return sampler(family, rng, **size) if sampler else NotImplemented


_PACKS = {
    "metabolism": DomainPack("metabolism", ("metabolic_sites",)),
    "solubility": DomainPack("solubility", ("aqueous_solubility",)),
    "properties": DomainPack("properties", ("solvation_free_energy", "frontier_orbitals",
                                             "ionization_energy", "electron_affinity")),
    "baselines": DomainPack("baselines", ()),
    "qmmm": DomainPack("qmmm", (), ("water_dimer_qmmm",)),
    "drug": DomainPack("drug", ("interaction_energy",), ("ligand_funnel", "h2_dimer_cp")),
    "binding": DomainPack("binding", ("binding_free_energy",)),
    "synthesis": DomainPack("synthesis", (), ("paracetamol_routes",)),
    "algorithms": DomainPack("algorithms", ()),
    "quantum": DomainPack("quantum", ()),
    "dynamics": DomainPack("dynamics", ("spectral_function",), ("aim-design-bath5",)),
    "molecular": DomainPack("molecular", ("molecular_ground_energy",)),
    "reactivity": DomainPack("reactivity", ("bond_dissociation", "ranking")),
    "reactions": DomainPack("reactions", ("reaction_energy",),
                            ("h2_atomization", "h3_exchange_barrier")),
    "paths": DomainPack("paths", ("pes_scan", "interaction_scan", "reaction_barrier")),
    "materials": DomainPack("materials", ("lattice_ground_energy", "periodic_energy",
                                         "pairing_correlation"),
                            ("hubbard_dimer", "hubbard_ring6", "h6_ewf_full_bath",
                             "diamond_kpoint_supercell")),
}


def pack_names():
    return list(_PACKS)


def get_pack(name):
    if name not in _PACKS:
        raise ValueError(f"Unknown domain pack: {name}")
    return _PACKS[name]


def pack_for_target(kind):
    return next(pack for pack in _PACKS.values() if kind in pack.target_kinds)


def load_registry(names):
    packs = [get_pack(name) for name in names]
    registry = Registry()
    if packs:
        get_pack("molecular").register(registry)
    for pack in packs:
        if pack.name != "molecular":
            pack.register(registry)
    return registry
