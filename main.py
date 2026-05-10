from __future__ import annotations

import json

from action_units import action_units_to_dict, extract_action_units
from preprocessing import build_stanza_pipeline, preprocess_text


def main() -> None:
    text = """
    The customer submits a claim by sending relevant documentation.
    The notification department checks the documents and creates a claim record based on the submitted claim.
    The system sends the claim record to the handling department by email.
    """

    nlp = build_stanza_pipeline()

    step1_result = preprocess_text(text, nlp)
    step2_result = extract_action_units(step1_result)

    print(json.dumps(
        action_units_to_dict(step2_result),
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()