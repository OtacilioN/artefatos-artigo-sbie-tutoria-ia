#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON=$(command -v python3.11)
elif [ -x /opt/homebrew/bin/python3.11 ]; then
  PYTHON=/opt/homebrew/bin/python3.11
else
  echo "Python 3.11 não encontrado." >&2
  echo "Instale com: brew install python@3.11" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python - <<'PY'
import platform
import torch
from transformers import AutoModel, AutoTokenizer

model = "bert-base-uncased"
revision = "86b5e0934494bd15c9632b12f734a8a67f723594"

print("architecture:", platform.machine())
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
if platform.machine() != "arm64":
    raise SystemExit("Este setup foi preparado para macOS arm64.")
if not torch.backends.mps.is_available():
    raise SystemExit("PyTorch MPS não está disponível neste ambiente.")

AutoTokenizer.from_pretrained(model, revision=revision)
AutoModel.from_pretrained(model, revision=revision)
print("BERT e tokenizer armazenados no cache local.")
PY

echo
echo "Ambiente pronto em $SCRIPT_DIR/.venv"
echo "Próximo passo: autenticar no Hugging Face e preparar classifier/data/."
