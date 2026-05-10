from __future__ import annotations

from models import ActionUnit, PreprocessingResult, SentenceInfo, TokenInfo


ACTION_DEPRELS = {
    "root",
    "conj",
    "advcl",
    "xcomp",
    "ccomp",
    "acl",
}


def _token_by_id(sentence: SentenceInfo) -> dict[int, TokenInfo]:
    return {token.id: token for token in sentence.tokens}


def _children_by_head(sentence: SentenceInfo) -> dict[int, list[TokenInfo]]:
    children: dict[int, list[TokenInfo]] = {}

    for token in sentence.tokens:
        children.setdefault(token.head, []).append(token)

    return children


def _is_action_predicate(token: TokenInfo) -> bool:
    """
    Decide whether a token can be treated as an action predicate.

    This is grammar-based, not keyword-based.
    We only check POS and dependency relation.
    """
    if token.upos not in {"VERB", "AUX"}:
        return False

    if token.deprel in ACTION_DEPRELS:
        return True

    return False


def _predicate_type(token: TokenInfo) -> str:
    """
    Convert dependency relation into readable action-unit type.
    """
    if token.deprel == "root":
        return "root_action"

    if token.deprel == "conj":
        return "coordinated_action"

    if token.deprel in {"advcl", "xcomp", "ccomp", "acl"}:
        return "subordinate_action"

    return "other_action"


def _has_object_or_process_argument(token: TokenInfo, children: dict[int, list[TokenInfo]]) -> bool:
    """
    Filter out weak verbs that do not participate in process-object relations.

    This still does not use verb lists.
    It only checks whether the predicate has syntactic dependents that can matter later:
    object, passive subject, oblique complement, open complement, etc.
    """
    relevant_relations = {
        "obj",
        "iobj",
        "obl",
        "nsubj",
        "nsubj:pass",
        "ccomp",
        "xcomp",
    }

    for child in children.get(token.id, []):
        if child.deprel in relevant_relations:
            return True

    return False


def extract_action_units_from_sentence(sentence: SentenceInfo) -> list[ActionUnit]:
    """
    Extract action units from one sentence.

    An action unit is a process-relevant predicate:
    - root action,
    - coordinated action,
    - subordinate action.
    """
    tokens_by_id = _token_by_id(sentence)
    children = _children_by_head(sentence)

    action_tokens: list[TokenInfo] = []

    for token in sentence.tokens:
        if not _is_action_predicate(token):
            continue

        if not _has_object_or_process_argument(token, children):
            continue

        action_tokens.append(token)

    action_units: list[ActionUnit] = []

    for index, token in enumerate(action_tokens, start=1):
        parent = tokens_by_id.get(token.head)

        action_units.append(
            ActionUnit(
                action_unit_id=f"S{sentence.sentence_id}_AU{index}",
                sentence_id=sentence.sentence_id,
                predicate_token_id=token.id,
                predicate_text=token.text,
                predicate_lemma=token.lemma,
                predicate_upos=token.upos,
                predicate_deprel=token.deprel,
                predicate_type=_predicate_type(token),
                parent_token_id=parent.id if parent else None,
                parent_lemma=parent.lemma if parent else None,
                text=sentence.text,
            )
        )

    return action_units


def extract_action_units(preprocessing_result: PreprocessingResult) -> list[ActionUnit]:
    """
    Step 2 of the algorithm.

    Input:
        PreprocessingResult from Step 1.

    Output:
        List of action units found in the process description.
    """
    action_units: list[ActionUnit] = []

    for sentence in preprocessing_result.sentences:
        action_units.extend(extract_action_units_from_sentence(sentence))

    return action_units


def action_units_to_dict(action_units: list[ActionUnit]) -> list[dict]:
    """
    Convert action units to JSON-friendly dictionaries.
    """
    return [
        {
            "action_unit_id": unit.action_unit_id,
            "sentence_id": unit.sentence_id,
            "predicate_token_id": unit.predicate_token_id,
            "predicate_text": unit.predicate_text,
            "predicate_lemma": unit.predicate_lemma,
            "predicate_upos": unit.predicate_upos,
            "predicate_deprel": unit.predicate_deprel,
            "predicate_type": unit.predicate_type,
            "parent_token_id": unit.parent_token_id,
            "parent_lemma": unit.parent_lemma,
            "text": unit.text,
        }
        for unit in action_units
    ]