import re
from pathlib import Path

from test_m3a_task import spec

from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor

PLUGIN = Path(__file__).resolve().parents[1] / "plugin"


def test_single_source_contracts():
    registry = molecular_registry()
    store = TaskStore("test")
    tools = make_tools(registry, Executor(registry), store.freeze(spec(), "human"), store)
    known = {name.replace(".", "_") for name in tools._tools}
    skills = list((PLUGIN / "skills").glob("*/SKILL.md"))
    assert len(skills) == 5
    for skill in skills:
        text = skill.read_text()
        frontmatter = text.split("---", 2)[1]
        name = re.search(r"^name: (.+)$", frontmatter, re.MULTILINE).group(1)
        description = re.search(r"^description: (.+)$", frontmatter, re.MULTILINE).group(1)
        assert name == skill.parent.name and len(name) <= 64
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        assert len(description) <= 1024
        assert set(re.findall(r"`([a-z_]+)`", text)) <= known
    for role in ("planner", "modeler", "designer", "critic", "analyst"):
        text = (PLUGIN / "agents" / f"{role}.md").read_text()
        allowed = set(re.findall(r"`([a-z_]+)`", text))
        assert allowed == {name.replace(".", "_") for name, (_, roles, _, _) in
                           tools._tools.items() if role in roles}
        offered = {item["name"].replace(".", "_") for item in tools.schemas(role)}
        assert offered == allowed - {"algorithm_verify", "algorithm_register", "algorithm_validity"}
    assert (PLUGIN / "rules" / "chemflow-rules.md").exists()
