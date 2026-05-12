from __future__ import annotations

import re
from typing import Any

import stanza

from .models import PreprocessingResult, SentenceInfo, TokenInfo


def normalize_text(text: str) -> str:
    """
    Normalize raw process text before linguistic parsing.
    This function does not change meaning; it only removes formatting noise.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.replace("\u00A0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


def build_stanza_pipeline() -> stanza.Pipeline:
    """
    Build the Stanza pipeline used in Step 1.

    Step 1 only needs:
    - tokenization
    - multi-word token expansion
    - POS tagging
    - lemmatization
    - dependency parsing
    """
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        verbose=False,
    )


def preprocess_text(raw_text: str, nlp: stanza.Pipeline | None = None) -> PreprocessingResult:
    """
    Step 1 of the algorithm.

    Input:
        Raw textual process description.

    Output:
        Structured linguistic representation:
        sentences, tokens, lemmas, POS tags, dependency heads and relations.
    """
    normalized_text = normalize_text(raw_text)

    if not normalized_text:
        raise ValueError("Input text is empty after normalization.")

    if nlp is None:
        nlp = build_stanza_pipeline()

    doc = nlp(normalized_text)

    sentences: list[SentenceInfo] = []

    for sentence_index, sentence in enumerate(doc.sentences, start=1):
        tokens: list[TokenInfo] = []

        for word in sentence.words:
            tokens.append(
                TokenInfo(
                    id=word.id,
                    text=word.text,
                    lemma=word.lemma or word.text.lower(),
                    upos=word.upos,
                    xpos=word.xpos,
                    feats=word.feats,
                    head=word.head,
                    deprel=word.deprel,
                )
            )

        sentences.append(
            SentenceInfo(
                sentence_id=sentence_index,
                text=sentence.text,
                tokens=tokens,
            )
        )

    return PreprocessingResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        sentences=sentences,
    )


def preprocessing_result_to_dict(result: PreprocessingResult) -> dict[str, Any]:
    """
    Convert dataclass result to plain dict for JSON export/debugging.
    """
    return {
        "raw_text": result.raw_text,
        "normalized_text": result.normalized_text,
        "sentences": [
            {
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
                "tokens": [
                    {
                        "id": token.id,
                        "text": token.text,
                        "lemma": token.lemma,
                        "upos": token.upos,
                        "xpos": token.xpos,
                        "feats": token.feats,
                        "head": token.head,
                        "deprel": token.deprel,
                    }
                    for token in sentence.tokens
                ],
            }
            for sentence in result.sentences
        ],
    }
