from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/raw/pet/petv11_relations.json")
OUTPUT_PATH = Path("data/gold/candidate_role_cases_v1.csv")


SUPPORTED_RELATIONS = {
    "uses",
    "actor recipient",
}


CSV_FIELDS = [
    "case_id",
    "source_dataset",
    "document_id",
    "sentence_id",
    "process_text",
    "predicate",
    "object_text",
    "pet_relation_type",
    "selection_source",
    "gold_role",
    "gold_status",
    "evidence_span",
    "comment",
]


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"PET relations file not found: {path}. "
            "Run python3 src/evaluation/pet_loader.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_get_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return value if isinstance(value, list) else []


def _build_sentence_texts(row: dict[str, Any]) -> dict[int, str]:
    tokens = _safe_get_list(row, "tokens")
    sentence_ids = _safe_get_list(row, "sentence-IDs")

    sentence_tokens: dict[int, list[str]] = {}

    for token, sentence_id in zip(tokens, sentence_ids):
        sentence_tokens.setdefault(int(sentence_id), []).append(str(token))

    sentence_texts: dict[int, str] = {}

    for sentence_id, toks in sentence_tokens.items():
        text = " ".join(toks)
        text = (
            text.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" ;", ";")
            .replace(" :", ":")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace("( ", "(")
            .replace(" )", ")")
        )
        sentence_texts[sentence_id] = text

    return sentence_texts


def _global_index_for_sentence_token(
    row: dict[str, Any],
    sentence_id: int,
    word_id: int,
) -> int | None:
    token_ids = _safe_get_list(row, "tokens-IDs")
    sentence_ids = _safe_get_list(row, "sentence-IDs")

    for index, (token_id, sent_id) in enumerate(zip(token_ids, sentence_ids)):
        if int(sent_id) == int(sentence_id) and int(token_id) == int(word_id):
            return index

    return None


def _clean_surface(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace("( ", "(")
        .replace(" )", ")")
        .strip()
    )


def _token_at(row: dict[str, Any], sentence_id: int, word_id: int) -> str:
    tokens = _safe_get_list(row, "tokens")
    index = _global_index_for_sentence_token(row, sentence_id, word_id)

    if index is None:
        return ""

    return str(tokens[index])


def _span_at(row: dict[str, Any], sentence_id: int, word_id: int) -> str:
    """
    Expand a PET head token to the full BIO entity span.

    Example:
        target word points to "the" with B-Activity Data
        next token "dismissal" is I-Activity Data
        → "the dismissal"
    """
    tokens = _safe_get_list(row, "tokens")
    ner_tags = _safe_get_list(row, "ner_tags")
    sentence_ids = _safe_get_list(row, "sentence-IDs")

    index = _global_index_for_sentence_token(row, sentence_id, word_id)

    if index is None:
        return ""

    tag = str(ner_tags[index])

    if tag == "O":
        return str(tokens[index])

    if "-" not in tag:
        return str(tokens[index])

    prefix, label = tag.split("-", 1)

    start = index
    end = index

    # If relation points to I-token, move left to the B-token.
    while start > 0:
        prev_tag = str(ner_tags[start - 1])
        prev_sentence_id = int(sentence_ids[start - 1])

        if prev_sentence_id != int(sentence_id):
            break

        if prev_tag == f"I-{label}":
            start -= 1
            continue

        if prev_tag == f"B-{label}":
            start -= 1
            break

        break

    # Move right across I-tags of the same label.
    while end + 1 < len(tokens):
        next_tag = str(ner_tags[end + 1])
        next_sentence_id = int(sentence_ids[end + 1])

        if next_sentence_id != int(sentence_id):
            break

        if next_tag == f"I-{label}":
            end += 1
            continue

        break

    return _clean_surface([str(token) for token in tokens[start : end + 1]])


def _relations_dict_to_rows(relations: Any) -> list[dict[str, Any]]:
    if isinstance(relations, list):
        return relations

    if not isinstance(relations, dict):
        return []

    required_keys = [
        "source-head-sentence-ID",
        "source-head-word-ID",
        "relation-type",
        "target-head-sentence-ID",
        "target-head-word-ID",
    ]

    if not all(key in relations for key in required_keys):
        return []

    relation_count = len(relations["relation-type"])
    rows: list[dict[str, Any]] = []

    for index in range(relation_count):
        rows.append(
            {
                key: relations[key][index]
                for key in required_keys
            }
        )

    return rows


def _normalize_relation_type(relation: dict[str, Any]) -> str:
    return str(relation.get("relation-type", "")).strip().lower()


def _get_relation_field(relation: dict[str, Any], field_name: str) -> int | None:
    value = relation.get(field_name)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relation_to_candidate(
    row: dict[str, Any],
    relation: dict[str, Any],
    case_number: int,
    sentence_texts: dict[int, str],
) -> dict[str, str] | None:
    relation_type = _normalize_relation_type(relation)

    if relation_type not in SUPPORTED_RELATIONS:
        return None

    source_sentence_id = _get_relation_field(relation, "source-head-sentence-ID")
    source_word_id = _get_relation_field(relation, "source-head-word-ID")
    target_sentence_id = _get_relation_field(relation, "target-head-sentence-ID")
    target_word_id = _get_relation_field(relation, "target-head-word-ID")

    if (
        source_sentence_id is None
        or source_word_id is None
        or target_sentence_id is None
        or target_word_id is None
    ):
        return None

    predicate = _token_at(row, source_sentence_id, source_word_id)
    object_text = _span_at(row, target_sentence_id, target_word_id)

    if not predicate or not object_text:
        return None

    process_text = sentence_texts.get(source_sentence_id, "")
    document_id = str(row.get("document name", ""))

    selection_source = (
        "PET_Uses"
        if relation_type == "uses"
        else "PET_ActorRecipient"
    )

    return {
        "case_id": f"C{case_number:04d}",
        "source_dataset": "PETv11",
        "document_id": document_id,
        "sentence_id": str(source_sentence_id),
        "process_text": process_text,
        "predicate": predicate,
        "object_text": object_text,
        "pet_relation_type": relation_type,
        "selection_source": selection_source,
        "gold_role": "",
        "gold_status": "",
        "evidence_span": "",
        "comment": "",
    }


def build_candidate_cases(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    documents = _load_json(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, str]] = []
    relation_type_counts: dict[str, int] = {}
    selected_relation_counts: dict[str, int] = {}

    case_number = 1

    for document in documents:
        sentence_texts = _build_sentence_texts(document)
        relations = _relations_dict_to_rows(document.get("relations", {}))

        for relation in relations:
            relation_type = _normalize_relation_type(relation)
            relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1

            candidate = _relation_to_candidate(
                row=document,
                relation=relation,
                case_number=case_number,
                sentence_texts=sentence_texts,
            )

            if candidate is None:
                continue

            selected_relation_counts[relation_type] = selected_relation_counts.get(relation_type, 0) + 1
            candidates.append(candidate)
            case_number += 1

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"[OK] Saved {len(candidates)} candidate cases to {output_path}")

    print("\n[ALL RELATION TYPE COUNTS]")
    for relation_type, count in sorted(relation_type_counts.items()):
        print(f"{relation_type}: {count}")

    print("\n[SELECTED RELATION TYPE COUNTS]")
    for relation_type, count in sorted(selected_relation_counts.items()):
        print(f"{relation_type}: {count}")


if __name__ == "__main__":
    build_candidate_cases()