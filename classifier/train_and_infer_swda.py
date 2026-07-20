#!/usr/bin/env python3
"""Treina BERT no SwDA e classifica apenas as mensagens dos estudantes.

Este script implementa um novo experimento compatível com o método descrito no
artigo. Ele não pretende reconstruir o checkpoint desaparecido nem afirmar que
as novas previsões são idênticas às usadas originalmente.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from swda_taxonomy import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    SWDA_CODE_TO_NAME,
    SWDA_CODES,
    validate_observed_labels,
)


DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_SEED = 42
ACTIVE_EXECUTION_STATE: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--studychat-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-studychat-samples", type=int)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto (padrão), none ou o caminho de um checkpoint-*.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        help="Salva um checkpoint retomável a cada N passos; 0 desativa.",
    )
    parser.add_argument(
        "--first-checkpoint-step",
        type=int,
        default=100,
        help="Salva um checkpoint inicial antecipado; 0 desativa.",
    )
    parser.add_argument(
        "--mps-empty-cache-steps",
        type=int,
        default=50,
        help="Libera o cache MPS a cada N passos; 0 desativa.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--overwrite-completed-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    # MPS ainda possui operações sem implementação determinística. O modo
    # warn_only registra essas ocorrências sem inviabilizar o experimento.
    torch.use_deterministic_algorithms(True, warn_only=True)


def stratified_limit(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    if limit < len(SWDA_CODES):
        raise ValueError(f"O limite deve ser >= {len(SWDA_CODES)} para conter todas as classes.")

    fractions = df["label"].value_counts(normalize=True)
    allocations = (fractions * limit).astype(int).clip(lower=1)
    while int(allocations.sum()) < limit:
        label = (fractions * limit - allocations).idxmax()
        allocations[label] += 1
    while int(allocations.sum()) > limit:
        candidates = allocations[allocations > 1]
        label = (allocations[candidates.index] - fractions[candidates.index] * limit).idxmax()
        allocations[label] -= 1

    parts = [
        group.sample(n=min(int(allocations[label]), len(group)), random_state=seed)
        for label, group in df.groupby("label", sort=False)
    ]
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_swda(path: Path, limit: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} deve conter as colunas {sorted(required)}.")
    df = df.dropna(subset=["text", "label"])[["text", "label"]].copy()
    validate_observed_labels(set(df["label"].astype(str)))
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["labels"] = df["label"].map(LABEL_TO_ID).astype(int)
    return stratified_limit(df, limit, seed)


def load_studychat_prompts(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        raise ValueError(f"{path} não contém a coluna prompt.")
    if limit is not None:
        df = df.head(limit).copy()
    else:
        df = df.copy()
    df["prompt"] = df["prompt"].fillna("").astype(str)
    if len(df) == 0:
        raise ValueError("O StudyChat não contém mensagens para classificar.")
    if limit is None and len(df) != 16_851:
        raise ValueError(
            f"Esperadas 16.851 mensagens StudyChat; encontradas {len(df)}."
        )
    return df.reset_index(drop=True)


def to_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    include_labels: bool,
) -> Dataset:
    columns = {"text": df["text"].tolist()} if "text" in df else {"text": df["prompt"].tolist()}
    if include_labels:
        columns["labels"] = df["labels"].astype(int).tolist()
    dataset = Dataset.from_dict(columns)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    remove_columns = ["text"]
    return dataset.map(tokenize, batched=True, remove_columns=remove_columns)


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, references = eval_prediction
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(
            f1_score(
                references,
                predictions,
                labels=list(range(len(SWDA_CODES))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def runtime_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "created_at_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": {
            "train_csv": sha256_file(args.train_csv),
            "validation_csv": sha256_file(args.validation_csv),
            "studychat_csv": sha256_file(args.studychat_csv),
        },
        "taxonomy": {
            "count": len(SWDA_CODES),
            "codes_in_id_order": list(SWDA_CODES),
        },
    }


class PeriodicCheckpointCallback(TrainerCallback):
    """Solicita checkpoints intermediários sem alterar a avaliação por época."""

    def __init__(self, every_n_steps: int, first_step: int) -> None:
        self.every_n_steps = every_n_steps
        self.first_step = first_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        is_first = self.first_step > 0 and state.global_step == self.first_step
        is_periodic = (
            self.every_n_steps > 0
            and state.global_step > 0
            and state.global_step % self.every_n_steps == 0
        )
        if is_first or is_periodic:
            control.should_save = True
        return control


class MPSMemoryCleanupCallback(TrainerCallback):
    """Reduz o acúmulo de blocos livres no cache do MPS durante treinos longos."""

    def __init__(self, every_n_steps: int) -> None:
        self.every_n_steps = every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if (
            self.every_n_steps > 0
            and state.global_step > 0
            and state.global_step % self.every_n_steps == 0
            and torch.backends.mps.is_available()
        ):
            cleanup_mps_memory()
        return control


def cleanup_mps_memory() -> None:
    """Sincroniza o dispositivo e devolve blocos MPS não utilizados."""

    if not torch.backends.mps.is_available():
        return
    torch.mps.synchronize()
    gc.collect()
    torch.mps.empty_cache()


def resolve_checkpoint(value: str, checkpoint_dir: Path) -> Path | None:
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        if not checkpoint_dir.exists():
            return None
        latest = get_last_checkpoint(str(checkpoint_dir))
        return Path(latest) if latest else None
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_dir() or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError(f"Checkpoint inválido: {checkpoint}")
    return checkpoint


def write_execution_state(path: Path, status: str, **details: Any) -> None:
    payload = {
        "status": status,
        "updated_at_unix": time.time(),
        **details,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    global ACTIVE_EXECUTION_STATE
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = args.output_dir / "experiment_manifest.json"
    if completed_manifest.exists() and not args.overwrite_completed_run:
        raise RuntimeError(
            f"A execução já está concluída em {args.output_dir}. "
            "Use outro diretório ou --overwrite-completed-run."
        )

    checkpoint_dir = args.output_dir / "checkpoints"
    resume_checkpoint = resolve_checkpoint(args.resume_from_checkpoint, checkpoint_dir)
    execution_state = args.output_dir / "execution_state.json"
    ACTIVE_EXECUTION_STATE = execution_state
    write_execution_state(
        execution_state,
        "starting",
        resume_from=str(resume_checkpoint) if resume_checkpoint else None,
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    seed_everything(args.seed)

    train_df = load_swda(args.train_csv, args.max_train_samples, args.seed)
    validation_df = load_swda(args.validation_csv, args.max_validation_samples, args.seed)
    studychat_df = load_studychat_prompts(args.studychat_csv, args.max_studychat_samples)

    print(
        f"Dados: treino={len(train_df)}, validação={len(validation_df)}, "
        f"StudyChat={len(studychat_df)}, classes={len(SWDA_CODES)}."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        num_labels=len(SWDA_CODES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = to_tokenized_dataset(train_df, tokenizer, args.max_length, include_labels=True)
    validation_dataset = to_tokenized_dataset(validation_df, tokenizer, args.max_length, include_labels=True)
    studychat_dataset = to_tokenized_dataset(studychat_df, tokenizer, args.max_length, include_labels=False)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_eval_batch_size=args.inference_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            PeriodicCheckpointCallback(
                args.checkpoint_steps,
                args.first_checkpoint_step,
            ),
            MPSMemoryCleanupCallback(args.mps_empty_cache_steps),
        ],
    )

    started = time.perf_counter()
    write_execution_state(
        execution_state,
        "training",
        resume_from=str(resume_checkpoint) if resume_checkpoint else None,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        studychat_rows=len(studychat_df),
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    elapsed = time.perf_counter() - started
    cleanup_mps_memory()
    validation_metrics = trainer.evaluate()
    cleanup_mps_memory()

    # Persiste o melhor modelo antes da inferência. Assim, uma falha posterior
    # na classificação do StudyChat não obriga a repetir o treinamento.
    model_dir = args.output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # O otimizador não é mais necessário depois do treino. Liberá-lo antes da
    # inferência deixa mais memória unificada disponível para os prompts.
    trainer.optimizer = None
    trainer.lr_scheduler = None
    cleanup_mps_memory()
    prediction_output = trainer.predict(studychat_dataset)
    probabilities = torch.softmax(torch.from_numpy(prediction_output.predictions), dim=1).numpy()
    predicted_ids = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    if len(predicted_ids) != len(studychat_df):
        raise RuntimeError(
            f"Cardinalidade inválida: {len(predicted_ids)} previsões para {len(studychat_df)} mensagens."
        )

    result = studychat_df.copy()
    if "source_row_id" not in result.columns:
        result.insert(0, "source_row_id", np.arange(len(result), dtype=int))
    result["da_code"] = [ID_TO_LABEL[int(index)] for index in predicted_ids]
    result["da_label"] = result["da_code"]
    result["da_name"] = result["da_code"].map(SWDA_CODE_TO_NAME)
    result["da_score"] = scores.astype(float)
    output_jsonl = args.output_dir / "studychat_supervised_da_new.jsonl"
    output_csv = args.output_dir / "studychat_supervised_da_new.csv"
    output_json = args.output_dir / "studychat_supervised_da.json"
    result.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    result.to_csv(output_csv, index=False)
    result.to_json(output_json, orient="records", force_ascii=False, indent=2)

    manifest = runtime_manifest(args)
    manifest.update(
        {
            "elapsed_train_seconds": elapsed,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "studychat_rows": len(studychat_df),
            "blank_prompt_count": int(studychat_df["prompt"].str.strip().eq("").sum()),
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
            "prediction_label_counts": result["da_code"].value_counts().sort_index().to_dict(),
            "output_sha256": {
                "jsonl": sha256_file(output_jsonl),
                "csv": sha256_file(output_csv),
                "json": sha256_file(output_json),
            },
            "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
    )
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_execution_state(
        execution_state,
        "completed",
        elapsed_train_seconds=elapsed,
        model_dir=str(model_dir),
        output_jsonl=str(output_jsonl),
        output_csv=str(output_csv),
        output_json=str(output_json),
    )

    print(
        "Concluído: "
        f"accuracy={validation_metrics['eval_accuracy']:.6f}, "
        f"macro_f1={validation_metrics['eval_macro_f1']:.6f}, "
        f"tempo={elapsed:.1f}s, saídas={output_jsonl}, {output_csv} e {output_json}."
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_EXECUTION_STATE is not None:
            write_execution_state(
                ACTIVE_EXECUTION_STATE,
                "failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        raise
