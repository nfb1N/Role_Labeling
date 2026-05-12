from __future__ import annotations

from collections import defaultdict

from .models import SemanticRoleInfo, ThesisRoleInfo


# ============================================================
# STEP 5: thesis-specific role assignment
# ============================================================

DIRECT_ROLE_MAPPING: dict[str, tuple[str, str, str]] = {
    # Result / output
    "Product": (
        "Result / Output object",
        "T1_RESULT_OUTPUT",
        "Semantic role Product maps to Result / Output object.",
    ),
    "Result": (
        "Result / Output object",
        "T1_RESULT_OUTPUT",
        "Semantic role Result maps to Result / Output object.",
    ),

    # Source / input
    "Source": (
        "Source / Input object",
        "T2_SOURCE_INPUT",
        "Semantic role Source maps to Source / Input object.",
    ),
    "Material": (
        "Source / Input object",
        "T2_SOURCE_INPUT",
        "Semantic role Material maps to Source / Input object.",
    ),

    # Support / instrument
    "Instrument": (
        "Support / Instrument object",
        "T3_SUPPORT_INSTRUMENT",
        "Semantic role Instrument maps to Support / Instrument object.",
    ),
    "Means": (
        "Support / Instrument object",
        "T3_SUPPORT_INSTRUMENT",
        "Semantic role Means maps to Support / Instrument object.",
    ),

    # Recipient-linked element
    "Recipient": (
        "Recipient-linked element",
        "T4_RECIPIENT_LINKED",
        "Semantic role Recipient maps to Recipient-linked element.",
    ),
    "Destination": (
        "Recipient-linked element",
        "T4_RECIPIENT_LINKED",
        "Semantic role Destination maps to Recipient-linked element.",
    ),
    "Beneficiary": (
        "Recipient-linked element",
        "T4_RECIPIENT_LINKED",
        "Semantic role Beneficiary maps to Recipient-linked element.",
    ),
    "Goal": (
        "Recipient-linked element",
        "T4_RECIPIENT_LINKED",
        "Semantic role Goal maps to Recipient-linked element.",
    ),

    # Default affected object
    "Patient": (
        "Target / Affected object",
        "T7_PATIENT_TARGET",
        "Semantic role Patient maps to Target / Affected object.",
    ),
}


TRANSFER_CONTEXT_ROLES = {
    "Recipient",
    "Destination",
    "Beneficiary",
    "Goal",
}


def _make_thesis_role(
    role: SemanticRoleInfo,
    thesis_role: str | None,
    status: str,
    rule_id: str | None,
    justification: str,
) -> ThesisRoleInfo:
    """
    Create a ThesisRoleInfo object while copying all shared fields
    from the SemanticRoleInfo record.
    """
    return ThesisRoleInfo(
        action_unit_id=role.action_unit_id,
        sentence_id=role.sentence_id,
        predicate_text=role.predicate_text,
        predicate_lemma=role.predicate_lemma,

        argument_text=role.argument_text,
        argument_head_token_id=role.argument_head_token_id,
        syntactic_relation=role.syntactic_relation,
        preposition=role.preposition,
        context_marker=role.context_marker,
        argument_type=role.argument_type,

        semantic_role=role.semantic_role,
        semantic_source=role.semantic_source,

        thesis_role=thesis_role,
        status=status,
        rule_id=rule_id,
        justification=justification,
    )


def _group_by_action_unit(
    semantic_roles: list[SemanticRoleInfo],
) -> dict[str, list[SemanticRoleInfo]]:
    grouped: dict[str, list[SemanticRoleInfo]] = defaultdict(list)

    for role in semantic_roles:
        grouped[role.action_unit_id].append(role)

    return grouped


def _is_actor_or_agent(role: SemanticRoleInfo) -> bool:
    """
    Actors are process participants, not domain objects in this role layer.
    """
    return role.argument_type == "actor" or role.semantic_role == "Agent"


def _action_unit_has_transfer_context(
    roles_in_action_unit: list[SemanticRoleInfo],
) -> bool:
    """
    Theme becomes Transferred object only if the same action unit contains
    recipient/destination-like context.
    """
    return any(
        role.semantic_role in TRANSFER_CONTEXT_ROLES
        for role in roles_in_action_unit
    )


def _assign_theme_role(
    role: SemanticRoleInfo,
    roles_in_action_unit: list[SemanticRoleInfo],
) -> ThesisRoleInfo:
    """
    Theme is context-dependent.

    Theme + Recipient/Destination-like role
        -> Transferred object

    Theme without transfer context
        -> Target / Affected object
    """
    if _action_unit_has_transfer_context(roles_in_action_unit):
        return _make_thesis_role(
            role=role,
            thesis_role="Transferred object",
            status="resolved",
            rule_id="T5_THEME_WITH_TRANSFER_CONTEXT",
            justification=(
                "Semantic role Theme maps to Transferred object because "
                "the same action unit contains a recipient/destination-like semantic role."
            ),
        )

    return _make_thesis_role(
        role=role,
        thesis_role="Target / Affected object",
        status="resolved",
        rule_id="T6_THEME_DEFAULT_TARGET",
        justification=(
            "Semantic role Theme maps to Target / Affected object because "
            "no recipient/destination-like transfer context was detected in the same action unit."
        ),
    )


def _assign_role_for_argument(
    role: SemanticRoleInfo,
    roles_in_action_unit: list[SemanticRoleInfo],
) -> ThesisRoleInfo:
    """
    Assign one thesis-specific role for one semantic-role record.
    """
    if _is_actor_or_agent(role):
        return _make_thesis_role(
            role=role,
            thesis_role=None,
            status="ignored",
            rule_id="T0_IGNORE_ACTOR",
            justification=(
                "Actor/Agent argument is treated as a process participant, "
                "not as a domain-object role."
            ),
        )

    if role.semantic_role is None:
        return _make_thesis_role(
            role=role,
            thesis_role=None,
            status="unresolved",
            rule_id=None,
            justification="No semantic role was available for thesis-role assignment.",
        )

    if role.semantic_role == "Theme":
        return _assign_theme_role(
            role=role,
            roles_in_action_unit=roles_in_action_unit,
        )

    mapped = DIRECT_ROLE_MAPPING.get(role.semantic_role)

    if mapped is not None:
        thesis_role, rule_id, justification = mapped

        return _make_thesis_role(
            role=role,
            thesis_role=thesis_role,
            status="resolved",
            rule_id=rule_id,
            justification=justification,
        )

    return _make_thesis_role(
        role=role,
        thesis_role=None,
        status="unresolved",
        rule_id=None,
        justification=(
            f"Semantic role {role.semantic_role!r} is not mapped to a "
            "thesis-specific domain-object role."
        ),
    )


def assign_thesis_roles(
    semantic_roles: list[SemanticRoleInfo],
) -> list[ThesisRoleInfo]:
    """
    Step 5 of the algorithm.

    Input:
        SemanticRoleInfo records from Step 4.

    Output:
        ThesisRoleInfo records with thesis-specific domain-object roles.

    This step does not use verb-specific dictionaries.
    It maps semantic roles and local action-unit context to thesis roles.
    """
    grouped = _group_by_action_unit(semantic_roles)

    results: list[ThesisRoleInfo] = []

    for _, roles_in_action_unit in grouped.items():
        for role in roles_in_action_unit:
            results.append(
                _assign_role_for_argument(
                    role=role,
                    roles_in_action_unit=roles_in_action_unit,
                )
            )

    return results


def thesis_roles_to_dict(thesis_roles: list[ThesisRoleInfo]) -> list[dict]:
    return [
        {
            "action_unit_id": role.action_unit_id,
            "sentence_id": role.sentence_id,
            "predicate_text": role.predicate_text,
            "predicate_lemma": role.predicate_lemma,

            "argument_text": role.argument_text,
            "argument_head_token_id": role.argument_head_token_id,
            "syntactic_relation": role.syntactic_relation,
            "preposition": role.preposition,
            "context_marker": role.context_marker,
            "argument_type": role.argument_type,

            "semantic_role": role.semantic_role,
            "semantic_source": role.semantic_source,

            "thesis_role": role.thesis_role,
            "status": role.status,
            "rule_id": role.rule_id,
            "justification": role.justification,
        }
        for role in thesis_roles
    ]
