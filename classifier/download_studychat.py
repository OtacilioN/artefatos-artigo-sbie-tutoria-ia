#!/usr/bin/env python3
"""Congela as colunas necessárias do StudyChat com revisão e hashes.

O repositório oficial é condicionado a autenticação. Para o teste de viabilidade,
o script aceita também um espelho e registra explicitamente essa proveniência.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REQUIRED_COLUMNS = [
    "prompt",
    "topic",
    "timestamp",
    "chatId",
    "userId",
    "interactionCount",
    "semester",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wmcnicho/StudyChat")
    parser.add_argument(
        "--revision",
        default="24d7987d9fbb30d9da12acc53455a10f1cdd2d7f",
    )
    parser.add_argument("--output", type=Path, default=Path("data/studychat.csv"))
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.dataset,
        split="train",
        revision=args.revision,
        token=args.token if args.token else True,
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.column_names))
    if missing:
        raise ValueError(f"Colunas ausentes no StudyChat: {missing}.")
    if len(dataset) != 16_851:
        raise ValueError(f"Esperadas 16.851 linhas; encontradas {len(dataset)}.")

    frame = dataset.select_columns(REQUIRED_COLUMNS).to_pandas()
    # A revisão congelada contém um único prompt formado apenas por espaço. A
    # implementação original o enviava ao tokenizer e o novo experimento o
    # preserva para manter as 16.851 linhas, registrando a ocorrência.
    frame["prompt"] = frame["prompt"].fillna("").astype(str)
    frame.insert(0, "source_row_id", range(len(frame)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "rows": len(frame),
        "columns": list(frame.columns),
        "unique_users": int(frame["userId"].nunique()),
        "unique_chats": int(frame["chatId"].nunique()),
        "topics": sorted(frame["topic"].astype(str).unique().tolist()),
        "blank_prompt_count": int(frame["prompt"].str.strip().eq("").sum()),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
