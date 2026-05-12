from __future__ import annotations

from functools import lru_cache
from typing import Any

import nltk
from nltk.corpus import verbnet as vn

from .models import ActionFrame, ArgumentInfo, SemanticRoleInfo


def ensure_verbnet() -> None:
    try:
        vn.classids(lemma="send")
    except LookupError:
        nltk.download("verbnet")


@lru_cache(maxsize=4096)
def _verbnet_frames_for_lemma(lemma: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    for class_id in vn.classids(lemma=lemma):
        for frame in vn.frames(class_id):
            frames.append(
                {
                    "class_id": class_id,
                    "description": frame.get("description"),
                    "syntax": frame.get("syntax", []),
                    "semantics": frame.get("semantics", []),
                }
            )

    return frames


def _argument_slot_kind(argument: ArgumentInfo) -> str:
    if argument.argument_type == "actor":
        return "SUBJECT"

    if argument.syntactic_relation in {"obj", "iobj", "nsubj:pass"}:
        return "NP"

    if argument.syntactic_relation in {"obl", "nmod"}:
        return "PP"

    return "OTHER"


def _observed_slots(arguments: list[ArgumentInfo]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []

    for argument in arguments:
        kind = _argument_slot_kind(argument)

        if kind == "OTHER":
            continue

        slots.append(
            {
                "argument": argument,
                "kind": kind,
                "preposition": argument.preposition,
            }
        )

    return slots


def _modifier_value(item: dict[str, Any]) -> str | None:
    value = item.get("modifiers", {}).get("value")

    if value is None:
        return None

    if isinstance(value, list):
        return value[0] if value else None

    return str(value)


def _extract_verbnet_slots(frame: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    syntax = frame.get("syntax", [])
    seen_verb = False
    index = 0

    while index < len(syntax):
        item = syntax[index]
        pos_tag = item.get("pos_tag")

        if pos_tag == "VERB":
            seen_verb = True
            index += 1
            continue

        if pos_tag == "NP":
            slots.append(
                {
                    "kind": "NP" if seen_verb else "SUBJECT",
                    "preposition": None,
                    "semantic_role": _modifier_value(item),
                }
            )
            index += 1
            continue

        if pos_tag == "PREP" and seen_verb:
            preposition = _modifier_value(item)
            next_item = syntax[index + 1] if index + 1 < len(syntax) else {}

            if next_item.get("pos_tag") == "NP":
                slots.append(
                    {
                        "kind": "PP",
                        "preposition": preposition,
                        "semantic_role": _modifier_value(next_item),
                    }
                )
                index += 2
                continue

        index += 1

    return slots


def _slot_compatible(
    observed_slot: dict[str, Any],
    frame_slot: dict[str, Any],
) -> bool:
    if observed_slot["kind"] != frame_slot["kind"]:
        return False

    if observed_slot["kind"] != "PP":
        return True

    frame_preposition = frame_slot.get("preposition")

    if frame_preposition is None:
        return True

    return observed_slot.get("preposition") == frame_preposition


def _align_slots(
    observed: list[dict[str, Any]],
    frame_slots: list[dict[str, Any]],
) -> dict[int, str] | None:
    mapping: dict[int, str] = {}
    frame_start_index = 0

    for observed_index, observed_slot in enumerate(observed):
        for frame_index in range(frame_start_index, len(frame_slots)):
            frame_slot = frame_slots[frame_index]

            if not _slot_compatible(observed_slot, frame_slot):
                continue

            semantic_role = frame_slot.get("semantic_role")
            if semantic_role is not None:
                mapping[observed_index] = semantic_role

            frame_start_index = frame_index + 1
            break

    if not mapping:
        return None

    return mapping


def _best_verbnet_alignment(
    predicate_lemma: str,
    arguments: list[ArgumentInfo],
) -> tuple[dict[int, str], dict[str, Any] | None]:
    observed = _observed_slots(arguments)
    best_mapping: dict[int, str] = {}
    best_frame: dict[str, Any] | None = None

    for frame in _verbnet_frames_for_lemma(predicate_lemma):
        frame_slots = _extract_verbnet_slots(frame)
        mapping = _align_slots(observed, frame_slots)

        if mapping is None:
            continue

        if len(mapping) > len(best_mapping):
            best_mapping = mapping
            best_frame = frame

    return best_mapping, best_frame


def _structural_fallback_role(argument: ArgumentInfo) -> str | None:
    if argument.argument_type == "actor":
        return "Agent"

    if argument.syntactic_relation == "nsubj:pass":
        return "Patient"

    if argument.syntactic_relation == "obj":
        return "Patient"

    if argument.syntactic_relation == "iobj":
        return "Recipient"

    return None


def _normalize_or_repair_semantic_role(
    argument: ArgumentInfo,
    semantic_role: str | None,
    semantic_source: str,
) -> tuple[str | None, str, str | None]:
    if (
        argument.syntactic_relation == "obj"
        and semantic_role in {"Location"}
    ):
        return (
            "Patient",
            "semantic_sanity_fallback",
            "Direct object received a location-like VerbNet role; repaired to Patient.",
        )

    if (
        semantic_role is None
        and argument.syntactic_relation in {"obl", "nmod"}
        and argument.preposition in {"by", "via", "through", "with"}
    ):
        return (
            "Instrument",
            "semantic_structural_fallback",
            (
                "Instrument-like prepositional argument inferred from syntactic "
                "relation and preposition."
            ),
        )

    if (
        semantic_role is None
        and argument.context_marker == "base"
        and argument.preposition == "on"
    ):
        return (
            "Source",
            "semantic_structural_fallback",
            "Basis construction inferred as Source from context marker and preposition.",
        )

    return semantic_role, semantic_source, None


def _frame_description_text(frame: dict[str, Any] | None) -> str | None:
    if frame is None:
        return None

    description = frame.get("description")

    if description is None:
        return None

    if isinstance(description, dict):
        parts = [
            str(value)
            for key, value in description.items()
            if key in {"primary", "secondary"} and value
        ]
        return " ".join(parts) if parts else None

    return str(description)


def _semantic_role_info(
    argument: ArgumentInfo,
    semantic_role: str | None,
    semantic_source: str,
    status: str,
    frame: dict[str, Any] | None,
    justification: str,
) -> SemanticRoleInfo:
    return SemanticRoleInfo(
        action_unit_id=argument.action_unit_id,
        sentence_id=argument.sentence_id,
        predicate_text=argument.predicate_text,
        predicate_lemma=argument.predicate_lemma,
        argument_text=argument.argument_text,
        argument_head_token_id=argument.argument_head_token_id,
        syntactic_relation=argument.syntactic_relation,
        preposition=argument.preposition,
        context_marker=argument.context_marker,
        argument_type=argument.argument_type,
        semantic_role=semantic_role,
        semantic_source=semantic_source,
        status=status,
        frame_class_id=frame.get("class_id") if frame else None,
        frame_description=_frame_description_text(frame),
        justification=justification,
    )


def infer_semantic_roles_for_frame(frame: ActionFrame) -> list[SemanticRoleInfo]:
    arguments = frame.arguments
    observed = _observed_slots(arguments)
    verbnet_mapping, matched_frame = _best_verbnet_alignment(
        frame.action_unit.predicate_lemma,
        arguments,
    )
    roles: list[SemanticRoleInfo] = []

    for observed_index, observed_slot in enumerate(observed):
        argument = observed_slot["argument"]
        verbnet_role = verbnet_mapping.get(observed_index)

        if verbnet_role is not None:
            (
                normalized_role,
                normalized_source,
                repair_justification,
            ) = _normalize_or_repair_semantic_role(
                argument=argument,
                semantic_role=verbnet_role,
                semantic_source="verbnet",
            )
            roles.append(
                _semantic_role_info(
                    argument=argument,
                    semantic_role=normalized_role,
                    semantic_source=normalized_source,
                    status="resolved",
                    frame=matched_frame,
                    justification=repair_justification or (
                        "Semantic role inferred by aligning extracted arguments "
                        "with a VerbNet syntax frame."
                    ),
                )
            )
            continue

        fallback_role = _structural_fallback_role(argument)

        if fallback_role is not None:
            (
                normalized_role,
                normalized_source,
                repair_justification,
            ) = _normalize_or_repair_semantic_role(
                argument=argument,
                semantic_role=fallback_role,
                semantic_source="structural_fallback",
            )
            roles.append(
                _semantic_role_info(
                    argument=argument,
                    semantic_role=normalized_role,
                    semantic_source=normalized_source,
                    status="resolved",
                    frame=None,
                    justification=repair_justification or (
                        "No VerbNet semantic role was aligned; broad semantic role "
                        "inferred from syntactic relation."
                    ),
                )
            )
            continue

        (
            normalized_role,
            normalized_source,
            repair_justification,
        ) = _normalize_or_repair_semantic_role(
            argument=argument,
            semantic_role=None,
            semantic_source="unresolved",
        )

        if normalized_role is not None:
            roles.append(
                _semantic_role_info(
                    argument=argument,
                    semantic_role=normalized_role,
                    semantic_source=normalized_source,
                    status="resolved",
                    frame=None,
                    justification=repair_justification or (
                        "Conservative semantic role inferred from argument structure."
                    ),
                )
            )
            continue

        roles.append(
            _semantic_role_info(
                argument=argument,
                semantic_role=None,
                semantic_source="unresolved",
                status="unresolved",
                frame=None,
                justification=(
                    "No VerbNet role was aligned and no conservative structural "
                    "fallback was available."
                ),
            )
        )

    return roles


def infer_semantic_roles(frames: list[ActionFrame]) -> list[SemanticRoleInfo]:
    ensure_verbnet()

    roles: list[SemanticRoleInfo] = []

    for frame in frames:
        roles.extend(infer_semantic_roles_for_frame(frame))

    return roles


def semantic_roles_to_dict(roles: list[SemanticRoleInfo]) -> list[dict]:
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
            "status": role.status,
            "frame_class_id": role.frame_class_id,
            "frame_description": role.frame_description,
            "justification": role.justification,
        }
        for role in roles
    ]
