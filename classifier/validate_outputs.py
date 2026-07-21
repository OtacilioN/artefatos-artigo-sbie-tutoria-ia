#!/usr/bin/env python3
"""Valida a execução completa do classificador e grava um relatório sanitizado."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swda_taxonomy import SWDA_CODE_TO_NAME, SWDA_CODES


OUTPUT_FILES = {
    "json": "studychat_supervised_da.json",
    "jsonl": "studychat_supervised_da_new.jsonl",
    "csv": "studychat_supervised_da_new.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full"))
    parser.add_argument("--expected-rows", type=int, default=16_851)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Arquivo obrigatório ausente ou vazio: {path}")


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    expected_rows: int,
) -> pd.DataFrame:
    required = {"source_row_id", "da_code", "da_label", "da_name", "da_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: colunas ausentes {missing}.")
    if len(frame) != expected_rows:
        raise ValueError(
            f"{source}: esperadas {expected_rows} linhas; encontradas {len(frame)}."
        )

    normalized = frame.copy()
    normalized["source_row_id"] = pd.to_numeric(
        normalized["source_row_id"], errors="raise"
    ).astype(int)
    normalized = normalized.sort_values("source_row_id", kind="stable").reset_index(
        drop=True
    )
    expected_ids = np.arange(expected_rows, dtype=int)
    if not np.array_equal(normalized["source_row_id"].to_numpy(), expected_ids):
        raise ValueError(
            f"{source}: source_row_id deve cobrir 0..{expected_rows - 1} sem lacunas."
        )

    codes = normalized["da_code"].astype(str)
    labels = normalized["da_label"].astype(str)
    if not codes.equals(labels):
        raise ValueError(f"{source}: da_code e da_label divergem.")
    unexpected = sorted(set(codes) - set(SWDA_CODES))
    if unexpected:
        raise ValueError(f"{source}: códigos fora da taxonomia SwDA: {unexpected}.")

    expected_names = codes.map(SWDA_CODE_TO_NAME)
    if not expected_names.equals(normalized["da_name"].astype(str)):
        raise ValueError(f"{source}: da_name não corresponde ao código SwDA.")

    scores = pd.to_numeric(normalized["da_score"], errors="raise").astype(float)
    if not np.isfinite(scores.to_numpy()).all() or not scores.between(0.0, 1.0).all():
        raise ValueError(
            f"{source}: da_score deve ser finito e estar no intervalo [0,1]."
        )
    normalized["da_score"] = scores
    return normalized


def compare_formats(reference: pd.DataFrame, candidate: pd.DataFrame, source: str) -> None:
    for column in ["source_row_id", "da_code", "da_label", "da_name"]:
        if not reference[column].equals(candidate[column]):
            raise ValueError(f"{source}: coluna {column} diverge do JSONL.")
    if not np.allclose(
        reference["da_score"].to_numpy(),
        candidate["da_score"].to_numpy(),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(f"{source}: probabilidades divergem do JSONL.")


def validate_output_dir(output_dir: Path, expected_rows: int = 16_851) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    state_path = output_dir / "execution_state.json"
    manifest_path = output_dir / "experiment_manifest.json"
    history_path = output_dir / "trainer_log_history.json"
    config_path = output_dir / "model" / "config.json"
    weights_path = output_dir / "model" / "model.safetensors"
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILES.items()}

    for path in [
        state_path,
        manifest_path,
        history_path,
        config_path,
        weights_path,
        *output_paths.values(),
    ]:
        require_file(path)

    state = load_json(state_path)
    manifest = load_json(manifest_path)
    history = load_json(history_path)
    config = load_json(config_path)
    if state.get("status") != "completed":
        raise ValueError(f"Status da execução não é completed: {state.get('status')!r}.")
    if not isinstance(history, list) or not history:
        raise ValueError("trainer_log_history.json deve conter um histórico não vazio.")
    if manifest.get("studychat_rows") != expected_rows:
        raise ValueError("O manifesto não registra a cardinalidade StudyChat esperada.")
    if manifest.get("taxonomy", {}).get("count") != len(SWDA_CODES):
        raise ValueError("O manifesto não registra as 41 classes SwDA.")
    id2label = config.get("id2label", {})
    if len(id2label) != len(SWDA_CODES):
        raise ValueError("O modelo final não possui exatamente 41 saídas.")
    if set(id2label.values()) != set(SWDA_CODES):
        raise ValueError("O mapeamento id2label do modelo diverge da taxonomia SwDA.")

    metrics = manifest.get("validation_metrics", {})
    accuracy = float(metrics.get("eval_accuracy", np.nan))
    macro_f1 = float(metrics.get("eval_macro_f1", np.nan))
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= macro_f1 <= 1.0):
        raise ValueError("Métricas de validação ausentes ou fora de [0,1].")

    jsonl = validate_prediction_frame(
        pd.read_json(output_paths["jsonl"], lines=True),
        source=OUTPUT_FILES["jsonl"],
        expected_rows=expected_rows,
    )
    csv = validate_prediction_frame(
        pd.read_csv(output_paths["csv"], keep_default_na=False),
        source=OUTPUT_FILES["csv"],
        expected_rows=expected_rows,
    )
    json_array = validate_prediction_frame(
        pd.read_json(output_paths["json"]),
        source=OUTPUT_FILES["json"],
        expected_rows=expected_rows,
    )
    compare_formats(jsonl, csv, OUTPUT_FILES["csv"])
    compare_formats(jsonl, json_array, OUTPUT_FILES["json"])

    hashes = {name: sha256_file(path) for name, path in output_paths.items()}
    recorded_hashes = manifest.get("output_sha256", {})
    if hashes != recorded_hashes:
        raise ValueError(
            f"Hashes das saídas divergem do manifesto: atuais={hashes}; "
            f"registrados={recorded_hashes}."
        )

    observed_counts = {
        str(code): int(count)
        for code, count in jsonl["da_code"].value_counts().sort_index().items()
    }
    recorded_counts = {
        str(code): int(count)
        for code, count in manifest.get("prediction_label_counts", {}).items()
    }
    if observed_counts != recorded_counts:
        raise ValueError("Contagens de rótulos divergem do manifesto.")

    report = {
        "status": "valid",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir_name": output_dir.name,
        "rows": len(jsonl),
        "unique_source_rows": int(jsonl["source_row_id"].nunique()),
        "model_num_labels": len(id2label),
        "predicted_class_count": int(jsonl["da_code"].nunique()),
        "validation_metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
        "elapsed_train_hours": float(manifest["elapsed_train_seconds"]) / 3600.0,
        "prediction_label_counts": observed_counts,
        "sha256": {
            **hashes,
            "experiment_manifest": sha256_file(manifest_path),
            "model_safetensors": sha256_file(weights_path),
        },
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    report = validate_output_dir(args.output_dir, args.expected_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Validação concluída: {args.output_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
