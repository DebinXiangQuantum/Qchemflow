import json
import re
from enum import StrEnum
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from chemflow.registry import Registry
from chemflow.relations import TargetRelation


class Outcome(StrEnum):
    passed = "passed"
    failed = "failed"
    inconclusive = "inconclusive"


class Status(StrEnum):
    experimental = "experimental"
    validated = "validated"
    deprecated = "deprecated"


class CardEvidence(BaseModel):
    path: str
    summary: str
    outcome: Outcome
    date: str
    sha256: str
    quote: str | None = None


class EvidenceCheck(BaseModel):
    card: str
    path: str
    status: str
    message: str


class CardValidationReport(BaseModel):
    cards: list["AlgorithmCard"]
    evidence: list[EvidenceCheck]

    @property
    def ok(self) -> bool:
        return all(entry.status != "error" for entry in self.evidence)


class AlgorithmCard(BaseModel):
    id: str
    target: str
    template: str
    pack: str | None = None
    relation: TargetRelation
    subgraph: list[str]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    backends: dict[str, Any] = Field(default_factory=dict)
    known_controls: list[str] = Field(default_factory=list)
    evidence: list[CardEvidence] = Field(default_factory=list)
    cost_model: dict[str, Any] | None = None
    failure_modes: list[str] = Field(default_factory=list)
    status: Status = Status.experimental


def load_cards(registry: Registry | None = None) -> list[AlgorithmCard]:
    directory = files("chemflow.cards").joinpath("data")
    paths = [path for path in directory.iterdir() if path.name.endswith(".yaml")]
    if registry is not None:
        for pack in directory.iterdir():
            if pack.is_dir():
                paths.extend(path for path in pack.iterdir() if path.name.endswith(".yaml"))
    cards = [AlgorithmCard.model_validate(yaml.safe_load(path.read_text()))
             for path in sorted(paths, key=lambda item: item.name)]
    return [card for card in cards if registry is None or card.target in registry.operators]


def evidence_applicability(path: Path, root: Path):
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    try:
        if path.suffix != ".json":
            linked = [evidence_applicability(path.parent / link, root) for link in
                      re.findall(r"\]\(([^)]+\.json)\)", path.read_text())]
            scopes = [scope for scope in linked if scope]
            return scopes[0] if len(scopes) == 1 else {"records": scopes} if scopes else None
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    fields = {"system", "molecule", "molecule_definition", "n_orb", "n_elec", "norb",
              "nelec", "qubits", "n_qubits", "bar", "accuracy_bar", "threshold", "unit"}
    records = []

    def collect(value):
        if isinstance(value, dict):
            scope = {key: item for key, item in value.items() if key in fields}
            if scope and set(scope) != {"unit"} and scope not in records:
                records.append(scope)
            for key, item in value.items():
                if key not in fields:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    return {"records": records} if records else None


def _different_target(scope, target):
    molecule = target.get("molecule", {})
    active = target.get("active", {})
    active = active if isinstance(active, dict) else {}
    sector = target.get("sector", {})
    sizes = {**sector, **active}
    if "n_alpha" in sector and "n_beta" in sector:
        sizes.setdefault("n_elec", sector["n_alpha"] + sector["n_beta"])

    def differs(record):
        compared = False
        for key in ("system", "qubits", "n_qubits"):
            if key in record and key in target:
                compared = True
                if record[key] != target[key]:
                    return True
        definition = record.get("molecule_definition", record.get("molecule", {}))
        if isinstance(definition, dict) and isinstance(molecule, dict):
            for key, alias in (("atoms", "atom"), ("smiles", "smiles")):
                actual = definition.get(key, definition.get(alias))
                if actual is not None and key in molecule:
                    compared = True
                    if actual != molecule[key]:
                        return True
        elif isinstance(definition, str) and isinstance(molecule, str):
            compared = True
            if definition != molecule:
                return True
        for key, alias in (("n_orb", "norb"), ("n_elec", "nelec")):
            actual = record.get(key, record.get(alias))
            if actual is None and isinstance(definition, dict):
                actual = definition.get(key, definition.get(alias))
            if actual is not None and key in sizes:
                compared = True
                if actual != sizes[key]:
                    return True
        return False if compared else None

    records = scope.get("records", [scope])
    comparisons = [_different_target(item, target) if "records" in item else differs(item)
                   for item in records]
    known = [value for value in comparisons if value is not None]
    return bool(known) and all(known)


def card_data(card: AlgorithmCard, spec, root: Path | None = None):
    root = root or Path(__file__).resolve().parents[3]
    data = card.model_dump(mode="json")
    for evidence in data["evidence"]:
        scope = evidence_applicability(root / evidence["path"], root)
        evidence["applies_to"] = scope
        if scope and _different_target(scope, spec.target.model_dump(mode="json")):
            evidence["applies_to_current_task"] = False
    return data


def validate_cards(registry: Registry, repo_root: Path,
                   cards: list[AlgorithmCard] | None = None) -> CardValidationReport:
    loaded = load_cards(registry) if cards is None else cards
    if cards is None:
        loaded = [card for card in loaded if card.target in registry.operators and
                  (getattr(card, "pack", None) is None or
                   card.pack in getattr(registry, "loaded_packs", {card.pack}))]
    root = Path(repo_root).resolve()
    checks = []
    for card in loaded:
        rule = registry.refinements.get(card.template)
        if (rule is None or card.target not in registry.operators or
                rule.target != card.target.removeprefix("chemflow.").split("@")[0]):
            raise ValueError(f"Unknown template or target: {card.id}")
        if card.relation != rule.relation:
            raise ValueError(f"Relation mismatch: {card.id}")
        steps = {"gradient_screening", "cluster_optimization", "perturbative_triples",
                 "sector_diagonalization", "recover_configurations", "subspace_diagonalize",
                 "extend", "minimize", "prepare_uccsd", "prepare_lucj", "sci_selection",
                 "perturbation"}
        for step in card.subgraph:
            if step not in steps and f"chemflow.{step}@1" not in registry.operators:
                raise ValueError(f"Unknown subgraph step: {step}")
        for evidence in card.evidence:
            path = (root / evidence.path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Evidence path outside repository: {evidence.path}")
            if not path.is_file():
                status, message = "unavailable", "Evidence file not present locally"
            elif sha256(path.read_bytes()).hexdigest() != evidence.sha256:
                status, message = "error", "Evidence hash mismatch"
            elif evidence.quote is not None and evidence.quote not in path.read_text():
                status, message = "error", "Evidence quote not found"
            else:
                status, message = "ok", "Evidence hash and quote verified"
            checks.append(EvidenceCheck(card=card.id, path=evidence.path,
                                        status=status, message=message))
    return CardValidationReport(cards=loaded, evidence=checks)
