from __future__ import annotations

from .action_units import extract_action_units
from .arguments import build_action_frames
from .final_output import FinalOutput, build_final_output
from .preprocessing import build_stanza_pipeline, preprocess_text
from .semantic_roles import infer_semantic_roles
from .thesis_roles import assign_thesis_roles


def run_proposed_algorithm(raw_text: str, nlp=None) -> FinalOutput:
    if nlp is None:
        nlp = build_stanza_pipeline()

    step1_result = preprocess_text(raw_text, nlp)
    step2_result = extract_action_units(step1_result)
    step3_result = build_action_frames(step1_result, step2_result)
    step4_result = infer_semantic_roles(step3_result)
    step5_result = assign_thesis_roles(step4_result)
    step6_result = build_final_output(step5_result)

    return step6_result
