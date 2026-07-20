#!/usr/bin/env python3
"""Executa inferência SwDA em um StudyChat congelado usando modelo treinado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from swda_taxonomy import SWDA_CODE_TO_NAME, SWDA_CODES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--studychat-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS foi solicitado, mas não está disponível.")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.studychat_csv)
    if "prompt" not in frame or len(frame) != 16_851:
        raise ValueError("A entrada deve conter exatamente 16.851 prompts StudyChat.")
    frame["prompt"] = frame["prompt"].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    if model.config.num_labels != len(SWDA_CODES):
        raise ValueError(f"Esperadas 41 saídas; checkpoint possui {model.config.num_labels}.")

    device = select_device(args.device)
    model.to(device)
    model.eval()
    prompts = frame["prompt"].astype(str).tolist()
    loader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)
    predicted_codes: list[str] = []
    predicted_scores: list[float] = []

    with torch.inference_mode():
        for texts in loader:
            encoded = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            probabilities = model(**encoded).logits.softmax(dim=-1)
            scores, indices = probabilities.max(dim=-1)
            for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist()):
                code = str(model.config.id2label[int(index)])
                if code not in SWDA_CODE_TO_NAME:
                    raise ValueError(f"Rótulo inesperado no checkpoint: {code}.")
                predicted_codes.append(code)
                predicted_scores.append(float(score))

    if len(predicted_codes) != len(frame):
        raise RuntimeError(
            f"Cardinalidade inválida: {len(predicted_codes)} previsões para {len(frame)} linhas."
        )
    if not all(0.0 <= value <= 1.0 for value in predicted_scores):
        raise RuntimeError("O classificador produziu probabilidade fora de [0,1].")

    result = frame.copy()
    if "source_row_id" not in result:
        result.insert(0, "source_row_id", np.arange(len(result), dtype=int))
    result["da_code"] = predicted_codes
    result["da_name"] = [SWDA_CODE_TO_NAME[code] for code in predicted_codes]
    result["da_score"] = predicted_scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(args.output, orient="records", lines=True, force_ascii=False)

    manifest = {
        "model_dir": str(args.model_dir),
        "studychat_csv": str(args.studychat_csv),
        "studychat_sha256": sha256_file(args.studychat_csv),
        "rows": len(result),
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "blank_prompt_count": int(frame["prompt"].str.strip().eq("").sum()),
        "label_counts": result["da_code"].value_counts().sort_index().to_dict(),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
