#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
TRAIN_CSV="${TRAIN_CSV:-$SCRIPT_DIR/data/swda_train.csv}"
VALIDATION_CSV="${VALIDATION_CSV:-$SCRIPT_DIR/data/swda_val.csv}"
STUDYCHAT_CSV="${STUDYCHAT_CSV:-$SCRIPT_DIR/data/studychat.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/full}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"

if [ ! -x "$PYTHON" ]; then
  echo "Ambiente ausente: $PYTHON" >&2
  echo "Crie o ambiente conforme o README antes de executar." >&2
  exit 1
fi

for input_file in "$TRAIN_CSV" "$VALIDATION_CSV" "$STUDYCHAT_CSV"; do
  if [ ! -f "$input_file" ]; then
    echo "Entrada ausente: $input_file" >&2
    echo "Consulte classifier/README.md para preparar os dados." >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

cd "$SCRIPT_DIR"

"$PYTHON" validate_inputs.py \
  --train-csv "$TRAIN_CSV" \
  --validation-csv "$VALIDATION_CSV" \
  --studychat-csv "$STUDYCHAT_CSV"

exec "$PYTHON" -u train_and_infer_swda.py \
  --train-csv "$TRAIN_CSV" \
  --validation-csv "$VALIDATION_CSV" \
  --studychat-csv "$STUDYCHAT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --model-name bert-base-uncased \
  --model-revision 86b5e0934494bd15c9632b12f734a8a67f723594 \
  --max-length 128 \
  --pad-to-multiple-of 8 \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient-checkpointing \
  --inference-batch-size "$INFERENCE_BATCH_SIZE" \
  --epochs 3 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --seed 42 \
  --checkpoint-steps 500 \
  --first-checkpoint-step 100 \
  --mps-empty-cache-steps 50 \
  --save-total-limit 2 \
  --resume-from-checkpoint auto \
  "$@"
