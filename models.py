from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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