from __future__ import annotations

from .action_units import extract_action_units
from .models import (
    ActionFrame,
    ActionUnit,
    ArgumentInfo,
    PreprocessingResult,
    SentenceInfo,
    TokenInfo,
)


CORE_ARGUMENT_RELATIONS = {
    "nsubj",
    "nsubj:pass",
    "obj",
    "iobj",
    "obl",
}


NESTED_ARGUMENT_RELATIONS = {
    "obl",
    "nmod",
}


NOUN_PHRASE_CHILD_RELATIONS = {
    "det",
    "amod",
    "compound",
    "nummod",
    "flat",
    "fixed",
    "case",
}


def _token_by_id(sentence: SentenceInfo) -> dict[int, TokenInfo]:
    return {token.id: token for token in sentence.tokens}


def _children_by_head(sentence: SentenceInfo) -> dict[int, list[TokenInfo]]:
    children: dict[int, list[TokenInfo]] = {}

    for token in sentence.tokens:
        children.setdefault(token.head, []).append(token)

    return children


def _collect_subtree_ids(
    head_id: int,
    children: dict[int, list[TokenInfo]],
) -> set[int]:
    ids = {head_id}

    for child in children.get(head_id, []):
        ids.update(_collect_subtree_ids(child.id, children))

    return ids


def _find_sentence(
    preprocessing_result: PreprocessingResult,
    sentence_id: int,
) -> SentenceInfo:
    for sentence in preprocessing_result.sentences:
        if sentence.sentence_id == sentence_id:
            return sentence

    raise ValueError(f"Sentence with id={sentence_id} was not found.")


def _get_preposition(
    token: TokenInfo,
    children: dict[int, list[TokenInfo]],
) -> str | None:
    """
    Return real ADP preposition for an argument.

    Example:
        to the department -> to
        by email -> by
        based on the claim -> on

    Verb-like markers such as 'based' are not returned as prepositions.
    """
    case_children = [
        child for child in children.get(token.id, [])
        if child.deprel == "case" and child.upos == "ADP"
    ]

    if not case_children:
        return None

    case_children = sorted(case_children, key=lambda x: x.id)
    return case_children[0].lemma or case_children[0].text.lower()

def _get_context_marker(
    token: TokenInfo,
    children: dict[int, list[TokenInfo]],
) -> str | None:
    """
    Detect non-prepositional context markers around an argument.

    Example:
        based on the submitted claim

    In some dependency parses, 'based' can appear as a case-like marker
    or modifier. It should not be treated as the preposition itself.
    """
    marker_relations = {"case", "mark", "acl", "advcl"}

    markers = [
        child for child in children.get(token.id, [])
        if child.deprel in marker_relations and child.upos in {"VERB", "AUX"}
    ]

    if not markers:
        return None

    markers = sorted(markers, key=lambda x: x.id)
    return markers[0].lemma or markers[0].text.lower()

def _collect_phrase_token_ids(
    head: TokenInfo,
    children: dict[int, list[TokenInfo]],
) -> set[int]:
    """
    Collect a simple noun phrase around a head token.

    This is intentionally conservative.
    It collects local nominal modifiers but does not try to understand roles yet.
    """
    ids = {head.id}

    for child in children.get(head.id, []):
        if child.deprel in NOUN_PHRASE_CHILD_RELATIONS:
            ids.add(child.id)

        # Include passive/submitted modifiers like "submitted claim"
        if child.deprel in {"acl", "acl:relcl"} and child.upos in {"VERB", "AUX"}:
            ids.add(child.id)

            for grandchild in children.get(child.id, []):
                if grandchild.deprel in {"advmod", "compound:prt"}:
                    ids.add(grandchild.id)

    return ids


def _surface_from_ids(
    token_ids: set[int],
    tokens_by_id: dict[int, TokenInfo],
) -> str:
    tokens = [
        tokens_by_id[token_id]
        for token_id in sorted(token_ids)
        if token_id in tokens_by_id
    ]

    text = " ".join(token.text for token in tokens)
    return (
        text
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ;", ";")
        .replace(" :", ":")
    )


def _argument_type_from_relation(token: TokenInfo) -> str:
    if token.deprel == "nsubj":
        return "actor"

    if token.deprel == "nsubj:pass":
        return "candidate_object"

    if token.deprel in {"obj", "iobj", "obl"}:
        if token.upos in {"NOUN", "PROPN", "PRON"}:
            return "candidate_object"

    return "context_argument"


def _extract_direct_arguments(
    sentence: SentenceInfo,
    action_unit: ActionUnit,
) -> list[ArgumentInfo]:
    tokens_by_id = _token_by_id(sentence)
    children = _children_by_head(sentence)

    arguments: list[ArgumentInfo] = []

    for token in sentence.tokens:
        if token.head != action_unit.predicate_token_id:
            continue

        if token.deprel not in CORE_ARGUMENT_RELATIONS:
            continue

        if token.upos not in {"NOUN", "PROPN", "PRON"}:
            continue

        phrase_ids = _collect_phrase_token_ids(token, children)
        argument_text = _surface_from_ids(phrase_ids, tokens_by_id)

        arguments.append(
            ArgumentInfo(
                action_unit_id=action_unit.action_unit_id,
                sentence_id=action_unit.sentence_id,
                predicate_token_id=action_unit.predicate_token_id,
                predicate_text=action_unit.predicate_text,
                predicate_lemma=action_unit.predicate_lemma,

                argument_text=argument_text,
                argument_head_token_id=token.id,
                argument_head_lemma=token.lemma,
                argument_upos=token.upos,

                syntactic_relation=token.deprel,
                preposition=_get_preposition(token, children),
                context_marker=_get_context_marker(token, children),
                argument_type=_argument_type_from_relation(token),
                inherited=False,
                inherited_from_action_unit_id=None,
            )
        )

    return arguments


def _extract_nested_arguments(
    sentence: SentenceInfo,
    action_unit: ActionUnit,
    direct_arguments: list[ArgumentInfo],
    sentence_action_units: list[ActionUnit],
) -> list[ArgumentInfo]:
    tokens_by_id = _token_by_id(sentence)
    children = _children_by_head(sentence)
    predicate_subtree_ids = _collect_subtree_ids(
        action_unit.predicate_token_id,
        children,
    )
    excluded_ids: set[int] = set()

    for other_unit in sentence_action_units:
        if other_unit.action_unit_id == action_unit.action_unit_id:
            continue

        if other_unit.predicate_token_id not in predicate_subtree_ids:
            continue

        excluded_ids.update(
            _collect_subtree_ids(other_unit.predicate_token_id, children)
        )

    search_area_ids = predicate_subtree_ids - excluded_ids
    direct_argument_head_ids = {
        argument.argument_head_token_id
        for argument in direct_arguments
    }

    arguments: list[ArgumentInfo] = []

    for token_id in sorted(search_area_ids):
        if token_id == action_unit.predicate_token_id:
            continue

        if token_id in direct_argument_head_ids:
            continue

        token = tokens_by_id.get(token_id)
        if token is None:
            continue

        if token.upos not in {"NOUN", "PROPN", "PRON"}:
            continue

        if token.deprel not in NESTED_ARGUMENT_RELATIONS:
            continue

        preposition = _get_preposition(token, children)
        context_marker = _get_context_marker(token, children)

        if preposition is None and context_marker is None:
            continue

        phrase_ids = _collect_phrase_token_ids(token, children)
        argument_text = _surface_from_ids(phrase_ids, tokens_by_id)

        arguments.append(
            ArgumentInfo(
                action_unit_id=action_unit.action_unit_id,
                sentence_id=action_unit.sentence_id,
                predicate_token_id=action_unit.predicate_token_id,
                predicate_text=action_unit.predicate_text,
                predicate_lemma=action_unit.predicate_lemma,

                argument_text=argument_text,
                argument_head_token_id=token.id,
                argument_head_lemma=token.lemma,
                argument_upos=token.upos,

                syntactic_relation=token.deprel,
                preposition=preposition,
                context_marker=context_marker,
                argument_type="candidate_object",
                inherited=False,
                inherited_from_action_unit_id=None,
            )
        )

    return arguments


def _deduplicate_arguments(
    arguments: list[ArgumentInfo],
) -> list[ArgumentInfo]:
    deduplicated_arguments: list[ArgumentInfo] = []
    seen: set[tuple[str, int, str, str | None, str | None]] = set()

    for argument in arguments:
        key = (
            argument.action_unit_id,
            argument.argument_head_token_id,
            argument.syntactic_relation,
            argument.preposition,
            argument.context_marker,
        )

        if key in seen:
            continue

        seen.add(key)
        deduplicated_arguments.append(argument)

    return deduplicated_arguments


def _find_subject_to_inherit(
    frames_for_sentence: list[ActionFrame],
    action_unit: ActionUnit,
) -> ArgumentInfo | None:
    """
    Coordinated or subordinate predicates often omit their subject.

    Example:
        The department checks the documents and creates a claim record.
        Create has no explicit nsubj, so it inherits subject from checks.

    This function searches previous frames in the same sentence.
    """
    if action_unit.parent_token_id is None:
        return None

    for frame in reversed(frames_for_sentence):
        if frame.action_unit.predicate_token_id != action_unit.parent_token_id:
            continue

        for arg in frame.arguments:
            if arg.argument_type == "actor":
                return arg

    return None


def _inherit_actor_if_needed(
    action_unit: ActionUnit,
    direct_arguments: list[ArgumentInfo],
    frames_for_sentence: list[ActionFrame],
) -> list[ArgumentInfo]:
    has_actor = any(arg.argument_type == "actor" for arg in direct_arguments)

    if has_actor:
        return direct_arguments

    inherited_actor = _find_subject_to_inherit(frames_for_sentence, action_unit)

    if inherited_actor is None:
        return direct_arguments

    inherited_copy = ArgumentInfo(
        action_unit_id=action_unit.action_unit_id,
        sentence_id=action_unit.sentence_id,
        predicate_token_id=action_unit.predicate_token_id,
        predicate_text=action_unit.predicate_text,
        predicate_lemma=action_unit.predicate_lemma,

        argument_text=inherited_actor.argument_text,
        argument_head_token_id=inherited_actor.argument_head_token_id,
        argument_head_lemma=inherited_actor.argument_head_lemma,
        argument_upos=inherited_actor.argument_upos,

        syntactic_relation=inherited_actor.syntactic_relation,
        preposition=inherited_actor.preposition,
        context_marker=inherited_actor.context_marker,
        argument_type="actor",
        inherited=True,
        inherited_from_action_unit_id=inherited_actor.action_unit_id,
    )

    return [inherited_copy] + direct_arguments


def build_action_frames(
    preprocessing_result: PreprocessingResult,
    action_units: list[ActionUnit] | None = None,
) -> list[ActionFrame]:
    """
    Step 3 of the algorithm.

    Input:
        Step 1 preprocessing result.
        Step 2 action units.

    Output:
        Action frames:
            action unit + extracted arguments.
    """
    if action_units is None:
        action_units = extract_action_units(preprocessing_result)

    frames: list[ActionFrame] = []

    for sentence in preprocessing_result.sentences:
        sentence_action_units = [
            unit for unit in action_units
            if unit.sentence_id == sentence.sentence_id
        ]

        frames_for_sentence: list[ActionFrame] = []

        for action_unit in sentence_action_units:
            direct_arguments = _extract_direct_arguments(sentence, action_unit)

            nested_arguments = _extract_nested_arguments(
                sentence=sentence,
                action_unit=action_unit,
                direct_arguments=direct_arguments,
                sentence_action_units=sentence_action_units,
            )

            combined_arguments = _deduplicate_arguments(
                direct_arguments + nested_arguments
            )

            arguments = _inherit_actor_if_needed(
                action_unit=action_unit,
                direct_arguments=combined_arguments,
                frames_for_sentence=frames_for_sentence,
            )

            frame = ActionFrame(
                action_unit=action_unit,
                arguments=arguments,
            )

            frames_for_sentence.append(frame)
            frames.append(frame)

    return frames


def action_frames_to_dict(frames: list[ActionFrame]) -> list[dict]:
    return [
        {
            "action_unit": {
                "action_unit_id": frame.action_unit.action_unit_id,
                "sentence_id": frame.action_unit.sentence_id,
                "predicate_token_id": frame.action_unit.predicate_token_id,
                "predicate_text": frame.action_unit.predicate_text,
                "predicate_lemma": frame.action_unit.predicate_lemma,
                "predicate_type": frame.action_unit.predicate_type,
                "parent_token_id": frame.action_unit.parent_token_id,
                "parent_lemma": frame.action_unit.parent_lemma,
                "text": frame.action_unit.text,
            },
            "arguments": [
                {
                    "argument_text": arg.argument_text,
                    "argument_head_token_id": arg.argument_head_token_id,
                    "argument_head_lemma": arg.argument_head_lemma,
                    "argument_upos": arg.argument_upos,
                    "syntactic_relation": arg.syntactic_relation,
                    "preposition": arg.preposition,
                    "context_marker": arg.context_marker,
                    "argument_type": arg.argument_type,
                    "inherited": arg.inherited,
                    "inherited_from_action_unit_id": arg.inherited_from_action_unit_id,
                }
                for arg in frame.arguments
            ],
        }
        for frame in frames
    ]
