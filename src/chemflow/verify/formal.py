"""Fail-closed integrity checks for the offline Lean audit, without running Lean at runtime."""

import hashlib
import json
import re
from pathlib import Path

from chemflow.ids import canonical_json
from chemflow.verify.lemmas import LEMMAS

FORMAL = Path(__file__).resolve().parents[3] / "formal"
STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lemma_hash(lemma):
    return hashlib.sha256(canonical_json(lemma).encode()).hexdigest()


def source_hashes(formal):
    return {path.relative_to(formal).as_posix(): sha256(path)
            for path in sorted((formal / "ChemflowFormal").rglob("*.lean"))}


def project_hashes(formal):
    return {name: sha256(formal / name) for name in
            ("ChemflowFormal.lean", "lean-toolchain", "lakefile.toml", "lake-manifest.json")}


def mathlib_revision(formal):
    packages = json.loads((formal / "lake-manifest.json").read_text())["packages"]
    return next(package["rev"] for package in packages if package["name"] == "mathlib")


def theorem_statement(source, name):
    short_name = name.removeprefix("ChemflowFormal.")
    match = re.search(r"(?m)^theorem " + re.escape(short_name) +
                      r"\b.*?(?= :=(?: by\b|\n))", source, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing Lean theorem declaration: {name}")
    return match.group()


def checked_lemmas(used):
    required = {name for name in used if name in LEMMAS and
                LEMMAS[name]["lean_status"] == "machine_checked"}
    if not required:
        return set()
    try:
        manifest = json.loads((FORMAL / "manifest.json").read_text())
        if (manifest["schema_version"] != 1 or manifest["sources"] != source_hashes(FORMAL) or
                manifest["project"] != project_hashes(FORMAL) or
                manifest["lean_toolchain"] != (FORMAL / "lean-toolchain").read_text().strip() or
                manifest["mathlib_revision"] != mathlib_revision(FORMAL)):
            raise ValueError("Formal manifest does not match current Lean sources/toolchain")
        entries = manifest["theorems"]
        theorems = {entry["name"]: entry for entry in entries}
        if len(theorems) != len(entries):
            raise ValueError("Duplicate theorem in formal manifest")
        for name in required:
            lemma = LEMMAS[name]
            entry = theorems[lemma["lean"]]
            filename = entry["file"]
            if filename not in manifest["sources"]:
                raise ValueError(f"Unaudited theorem source: {name}")
            statement = theorem_statement((FORMAL / filename).read_text(), lemma["lean"])
            if (entry["lemma"] != name or entry["lemma_sha256"] != lemma_hash(lemma) or
                    entry["source_sha256"] != manifest["sources"][filename] or
                    entry["statement"] != statement or
                    not set(entry["axioms"]) <= STANDARD_AXIOMS):
                raise ValueError(f"Formal theorem/lemma mismatch: {name}")
    except (OSError, KeyError, TypeError, AttributeError, StopIteration, ValueError) as exc:
        raise ValueError(f"Formal audit unavailable or stale: {exc}") from exc
    return required
