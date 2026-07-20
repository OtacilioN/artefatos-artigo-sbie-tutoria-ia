#!/usr/bin/env python3
"""Valida os arquivos locais antes de iniciar o treinamento de várias horas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from swda_taxonomy import SWDA_CODES, validate_observed_labels


EXPECTED_SWDA = {
    "train": {
        "rows": 179_766,
        "sha256": "dc28113d1068b0579a7c7eaec85d02a0343bb340a51a420798e7e80913b46c05",
    },
    "validation": {
        "rows": 19_974,
        "sha256": "b2164a396318302bf3868e29b2f918b985e0125a4b36ef236073e00610a4308b",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--studychat-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_swda(path: Path, split: str) -> dict[str, object]:
    expected = EXPECTED_SWDA[split]
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            f"Hash inesperado para {path}: {actual_hash}; esperado {expected['sha256']}."
        )
    frame = pd.read_csv(path)
    if len(frame) != expected["rows"]:
        raise ValueError(f"{path}: esperadas {expected['rows']} linhas; encontradas {len(frame)}.")
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{path}: colunas text,label ausentes.")
    validate_observed_labels(set(frame["label"].dropna().astype(str)))
    return {"rows": len(frame), "sha256": actual_hash, "classes": len(SWDA_CODES)}


def validate_studychat(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = {
        "source_row_id",
        "prompt",
        "topic",
        "timestamp",
        "chatId",
        "userId",
        "interactionCount",
        "semester",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: colunas ausentes {missing}.")
    if len(frame) != 16_851:
        raise ValueError(f"{path}: esperadas 16.851 linhas; encontradas {len(frame)}.")
    if frame["source_row_id"].nunique() != len(frame):
        raise ValueError("source_row_id deve ser único para cada prompt.")
    topics = sorted(frame["topic"].astype(str).unique().tolist())
    if topics != [f"a{index}" for index in range(1, 8)]:
        raise ValueError(f"Tópicos StudyChat inesperados: {topics}.")
    return {
        "rows": len(frame),
        "sha256": sha256_file(path),
        "users": int(frame["userId"].nunique()),
        "chats": int(frame["chatId"].nunique()),
        "topics": topics,
        "blank_prompts": int(frame["prompt"].fillna("").astype(str).str.strip().eq("").sum()),
    }


def main() -> None:
    args = parse_args()
    summary = {
        "swda_train": validate_swda(args.train_csv, "train"),
        "swda_validation": validate_swda(args.validation_csv, "validation"),
        "studychat": validate_studychat(args.studychat_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
