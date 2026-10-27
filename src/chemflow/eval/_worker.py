"""Private stdin/stdout protocol for one isolated evaluation task process."""

import json
import sys
from pathlib import Path

from chemflow.agent.llm import DEFAULT_MAX_TURNS
from chemflow.eval.runner import EvaluationTask, _run_task


def main():
    payload = json.load(sys.stdin)
    row = _run_task(EvaluationTask.model_validate(payload["task"]),
                    Path(payload["directory"]), Path(payload["public_path"]),
                    payload["policy"], payload["model"], payload["roles"],
                    max_turns=payload.get("max_turns", DEFAULT_MAX_TURNS))
    json.dump(row, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
