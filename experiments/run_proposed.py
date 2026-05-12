from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.proposed_algorithm.final_output import final_output_to_dict
from src.proposed_algorithm.pipeline import run_proposed_algorithm


def main() -> None:
    text = """
    The customer submits a claim by sending relevant documentation.
    The notification department checks the documents and creates a claim record based on the submitted claim.
    The system sends the claim record to the handling department by email.
    """

    result = run_proposed_algorithm(text)

    print(json.dumps(
        final_output_to_dict(result),
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
