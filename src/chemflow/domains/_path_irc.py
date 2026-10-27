"""Private foreground worker isolating Sella from host OpenMP libraries."""

import json
import sys

from chemflow.domains.paths import _irc
from chemflow.types import ElectronicModel


def main():
    request = json.load(sys.stdin)
    request["model"] = ElectronicModel.model_validate(request["model"])
    print(json.dumps(_irc(**request)))


if __name__ == "__main__":
    main()
