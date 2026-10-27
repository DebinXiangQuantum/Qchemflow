"""Application catalogue CLI, deliberately independent of signing keys."""

import json
from pathlib import Path

from . import catalog, draft


def configure(commands):
    parser = commands.add_parser("app")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    describe = actions.add_parser("describe")
    describe.add_argument("app_id")
    drafting = actions.add_parser("draft")
    drafting.add_argument("app_id")
    drafting.add_argument("--example", action="store_true")
    drafting.add_argument("--input", action="append", default=[])
    drafting.add_argument("--param", action="append", default=[])
    drafting.add_argument("--out", type=Path, required=True)


def run(args):
    if args.action == "list":
        for entry in catalog():
            print(f"{entry['id']}\t{entry['name_zh']}\t{entry['name_en']}")
        return
    if args.action == "describe":
        entry = next((entry for entry in catalog() if entry["id"] == args.app_id), None)
        if entry is None:
            raise ValueError(f"Unknown application: {args.app_id}")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return
    inputs, parameters = {}, {}
    for assignment in args.param:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise ValueError("Parameters must be key=value")
        try:
            parameters[key] = json.loads(value)
        except json.JSONDecodeError:
            parameters[key] = value
    for filename in args.input:
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".json":
            inputs.update(json.loads(path.read_text()))
        elif suffix == ".pdb":
            inputs["receptor"] = {"pdb_text": path.read_text()}
        elif suffix in (".smi", ".smiles"):
            text = path.read_text().strip()
            if args.app_id == "protein_ligand_binding":
                inputs["ligand"] = {"smiles": text}
            else:
                inputs["smiles"] = text
        elif args.app_id == "protein_ligand_binding" and suffix == ".sdf":
            inputs["ligand"] = {"sdf": path.read_text()}
        elif args.app_id == "stability_ranking":
            inputs.setdefault("items", []).append({"name": path.stem, "molecule": str(path)})
        else:
            inputs["molecule"] = str(path)
    tasks = draft(args.app_id, inputs, parameters, args.example)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.out / f"{args.app_id}-{index:02d}.json" for index in range(1, len(tasks) + 1)]
    if any(path.exists() for path in paths):
        raise ValueError("Refusing to overwrite existing drafts; choose a new output directory")
    for path, task in zip(paths, tasks, strict=True):
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
        print(path)
