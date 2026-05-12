from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class FinalRoleRecord:
    action_unit_id: str
    sentence_id: int
    predicate_text: str
    predicate_lemma: str

    object_text: str
    object_head_token_id: int

    syntactic_relation: str
    preposition: Optional[str]
    context_marker: Optional[str]

    semantic_role: Optional[str]
    semantic_source: str

    thesis_role: Optional[str]
    status: str
    rule_id: Optional[str]

    trace: str


@dataclass(frozen=True)
class FinalOutput:
    records: list[FinalRoleRecord]
    summary: dict[str, int]
    unresolved_records: list[FinalRoleRecord]

@dataclass(frozen=True)
class ThesisRoleInfo:
    action_unit_id: str
    sentence_id: int
    predicate_text: str
    predicate_lemma: str

    argument_text: str
    argument_head_token_id: int
    syntactic_relation: str
    preposition: Optional[str]
    context_marker: Optional[str]
    argument_type: str

    semantic_role: Optional[str]
    semantic_source: str

    thesis_role: Optional[str]
    status: str  # resolved | unresolved | ignored
    rule_id: Optional[str]
    justification: str

@dataclass(frozen=True)
class TokenInfo:
    id: int
    text: str
    lemma: str
    upos: str
    xpos: Optional[str]
    feats: Optional[str]
    head: int
    deprel: str


@dataclass(frozen=True)
class SentenceInfo:
    sentence_id: int
    text: str
    tokens: list[TokenInfo]


@dataclass(frozen=True)
class PreprocessingResult:
    raw_text: str
    normalized_text: str
    sentences: list[SentenceInfo]


@dataclass(frozen=True)
class ActionUnit:
    action_unit_id: str
    sentence_id: int
    predicate_token_id: int
    predicate_text: str
    predicate_lemma: str
    predicate_upos: str
    predicate_deprel: str
    predicate_type: str
    parent_token_id: Optional[int]
    parent_lemma: Optional[str]
    text: str


@dataclass(frozen=True)
class ArgumentInfo:
    action_unit_id: str
    sentence_id: int
    predicate_token_id: int
    predicate_text: str
    predicate_lemma: str

    argument_text: str
    argument_head_token_id: int
    argument_head_lemma: str
    argument_upos: str

    syntactic_relation: str
    preposition: Optional[str]
    context_marker: Optional[str]
    argument_type: str  # actor | candidate_object | context_argument
    inherited: bool
    inherited_from_action_unit_id: Optional[str]


@dataclass(frozen=True)
class ActionFrame:
    action_unit: ActionUnit
    arguments: list[ArgumentInfo]


@dataclass(frozen=True)
class SemanticRoleInfo:
    action_unit_id: str
    sentence_id: int
    predicate_text: str
    predicate_lemma: str

    argument_text: str
    argument_head_token_id: int
    syntactic_relation: str
    preposition: Optional[str]
    context_marker: Optional[str]
    argument_type: str

    semantic_role: Optional[str]
    semantic_source: str  # verbnet | structural_fallback | unresolved
    status: str           # resolved | unresolved
    frame_class_id: Optional[str]
    frame_description: Optional[str]
    justification: str
