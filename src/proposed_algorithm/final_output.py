from __future__ import annotations

from collections import Counter

from .models import FinalOutput, FinalRoleRecord, ThesisRoleInfo


# ============================================================
# STEP 6: final traceable output
# ============================================================


def _is_domain_object_record(role: ThesisRoleInfo) -> bool:
    """
    Keep only records that belong to domain-object role assignment.

    Actors/Agents ignored by Step 5 are excluded from the final role output.
    """
    return role.status != "ignored"


def _build_trace(role: ThesisRoleInfo) -> str:
    """
    Build a compact trace explaining how the final role was assigned.
    """
    parts: list[str] = []

    parts.append(f"predicate={role.predicate_text}")
    parts.append(f"object={role.argument_text}")

    if role.syntactic_relation:
        parts.append(f"syntax={role.syntactic_relation}")

    if role.preposition:
        parts.append(f"prep={role.preposition}")

    if role.context_marker:
        parts.append(f"context_marker={role.context_marker}")

    if role.semantic_role:
        parts.append(f"semantic_role={role.semantic_role}")

    if role.thesis_role:
        parts.append(f"thesis_role={role.thesis_role}")

    if role.rule_id:
        parts.append(f"rule={role.rule_id}")

    return " | ".join(parts)


def _to_final_record(role: ThesisRoleInfo) -> FinalRoleRecord:
    return FinalRoleRecord(
        action_unit_id=role.action_unit_id,
        sentence_id=role.sentence_id,
        predicate_text=role.predicate_text,
        predicate_lemma=role.predicate_lemma,

        object_text=role.argument_text,
        object_head_token_id=role.argument_head_token_id,

        syntactic_relation=role.syntactic_relation,
        preposition=role.preposition,
        context_marker=role.context_marker,

        semantic_role=role.semantic_role,
        semantic_source=role.semantic_source,

        thesis_role=role.thesis_role,
        status=role.status,
        rule_id=role.rule_id,

        trace=_build_trace(role),
    )


def build_final_output(thesis_roles: list[ThesisRoleInfo]) -> FinalOutput:
    """
    Step 6 of the algorithm.

    Input:
        ThesisRoleInfo records from Step 5.

    Output:
        FinalOutput containing:
        - final domain-object role records
        - role distribution summary
        - unresolved records
    """
    domain_roles = [
        role for role in thesis_roles
        if _is_domain_object_record(role)
    ]

    records = [
        _to_final_record(role)
        for role in domain_roles
    ]

    summary_counter = Counter()

    for record in records:
        if record.thesis_role is not None:
            summary_counter[record.thesis_role] += 1
        else:
            summary_counter["Unresolved"] += 1

    unresolved_records = [
        record for record in records
        if record.status == "unresolved" or record.thesis_role is None
    ]

    return FinalOutput(
        records=records,
        summary=dict(summary_counter),
        unresolved_records=unresolved_records,
    )


def final_output_to_dict(output: FinalOutput) -> dict:
    return {
        "summary": output.summary,
        "records": [
            {
                "action_unit_id": record.action_unit_id,
                "sentence_id": record.sentence_id,
                "predicate_text": record.predicate_text,
                "predicate_lemma": record.predicate_lemma,

                "object_text": record.object_text,
                "object_head_token_id": record.object_head_token_id,

                "syntactic_relation": record.syntactic_relation,
                "preposition": record.preposition,
                "context_marker": record.context_marker,

                "semantic_role": record.semantic_role,
                "semantic_source": record.semantic_source,

                "thesis_role": record.thesis_role,
                "status": record.status,
                "rule_id": record.rule_id,

                "trace": record.trace,
            }
            for record in output.records
        ],
        "unresolved_records": [
            {
                "action_unit_id": record.action_unit_id,
                "sentence_id": record.sentence_id,
                "predicate_text": record.predicate_text,
                "object_text": record.object_text,
                "semantic_role": record.semantic_role,
                "semantic_source": record.semantic_source,
                "thesis_role": record.thesis_role,
                "status": record.status,
                "rule_id": record.rule_id,
                "trace": record.trace,
            }
            for record in output.unresolved_records
        ],
    }
