from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


PET_REPOSITORY = "patriziobellan/PETv11"
RELATIONS_CONFIG = "relations-extraction"
TOKEN_CONFIG = "token-classification"

RAW_PET_DIR = Path("data/raw/pet")

RELATIONS_OUTPUT_PATH = RAW_PET_DIR / "petv11_relations.json"
TOKEN_OUTPUT_PATH = RAW_PET_DIR / "petv11_token_classification.json"
INSPECTION_OUTPUT_PATH = RAW_PET_DIR / "petv11_inspection.json"


def _save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {output_path}")


def _dataset_to_rows(dataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for split_name, split_data in dataset.items():
        for row in split_data:
            item = dict(row)
            item["_split"] = split_name
            rows.append(item)

    return rows


def _inspect_rows(rows: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "dataset_name": dataset_name,
            "record_count": 0,
            "keys": [],
            "first_record_preview": None,
        }

    first_record = rows[0]

    return {
        "dataset_name": dataset_name,
        "record_count": len(rows),
        "keys": list(first_record.keys()),
        "first_record_preview": first_record,
    }


def download_petv11_relations() -> list[dict[str, Any]]:
    print("[INFO] Downloading PETv11 relations-extraction dataset...")

    dataset = load_dataset(
        PET_REPOSITORY,
        name=RELATIONS_CONFIG,
        trust_remote_code=True,
    )

    rows = _dataset_to_rows(dataset)
    _save_json(rows, RELATIONS_OUTPUT_PATH)

    print(f"[OK] PETv11 relations records: {len(rows)}")
    return rows


def download_petv11_token_classification() -> list[dict[str, Any]]:
    print("[INFO] Downloading PETv11 token-classification dataset...")

    dataset = load_dataset(
        PET_REPOSITORY,
        name=TOKEN_CONFIG,
        trust_remote_code=True,
    )

    rows = _dataset_to_rows(dataset)
    _save_json(rows, TOKEN_OUTPUT_PATH)

    print(f"[OK] PETv11 token-classification records: {len(rows)}")
    return rows


def download_and_inspect_petv11() -> None:
    relations_rows = download_petv11_relations()
    token_rows = download_petv11_token_classification()

    inspection = {
        "repository": PET_REPOSITORY,
        "relations_extraction": _inspect_rows(
            relations_rows,
            "relations-extraction",
        ),
        "token_classification": _inspect_rows(
            token_rows,
            "token-classification",
        ),
    }

    _save_json(inspection, INSPECTION_OUTPUT_PATH)

    print("\n[SUMMARY]")
    print(f"Repository: {PET_REPOSITORY}")
    print(f"Relations records: {len(relations_rows)}")
    print(f"Token-classification records: {len(token_rows)}")

    if relations_rows:
        print("\n[RELATIONS KEYS]")
        print(list(relations_rows[0].keys()))

    if token_rows:
        print("\n[TOKEN CLASSIFICATION KEYS]")
        print(list(token_rows[0].keys()))

    print(f"\n[INSPECTION FILE]")
    print(INSPECTION_OUTPUT_PATH)


if __name__ == "__main__":
    download_and_inspect_petv11()