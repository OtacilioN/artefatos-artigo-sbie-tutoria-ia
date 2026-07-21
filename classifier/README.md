# Supervised SwDA classifier

This directory trains `bert-base-uncased` on 41 Switchboard Dialogue Act
(SwDA) classes and predicts exactly one dialogue act for each student `prompt`
in StudyChat. It is the clean, command-line replacement for the exploratory
training notebook.

Raw SwDA transcripts, raw StudyChat messages, model checkpoints and classified
files containing prompts are intentionally excluded from this public Git
repository. See [`../DATA_NOTICE.md`](../DATA_NOTICE.md) and
[`data/README.md`](data/README.md).

## Method contract

- Python 3.11;
- fixed `bert-base-uncased` revision
  `86b5e0934494bd15c9632b12f734a8a67f723594`;
- 41 SwDA classes;
- 179,766 training and 19,974 validation utterances;
- maximum length 128, learning rate `2e-5`, weight decay `0.01`;
- three epochs, seed 42 and effective training batch 16;
- macro-F1 selects the best epoch;
- inference uses only the 16,851 StudyChat student prompts.

## 1. Clone or update

For a new checkout:

```bash
git clone https://github.com/OtacilioN/artefatos-artigo-sbie-tutoria-ia.git
cd artefatos-artigo-sbie-tutoria-ia/classifier
```

For an existing checkout:

```bash
cd artefatos-artigo-sbie-tutoria-ia
git pull --ff-only origin main
cd classifier
```

## 2. Install on an Apple Silicon Mac

Install Python 3.11 with Homebrew if it is not already available:

```bash
brew install python@3.11
```

Create the isolated environment, install pinned dependencies, run tests,
verify MPS and prefetch BERT:

```bash
./setup_macos.sh
```

## 3. Obtain StudyChat

First accept the access conditions at:

<https://huggingface.co/datasets/wmcnicho/StudyChat>

Then authenticate and download the fixed revision:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
.venv/bin/python download_studychat.py
```

The generated `data/studychat.csv` is ignored by Git.

## 4. Copy the authorized SwDA inputs

Copy the exact `swda_train.csv` and `swda_val.csv` used by this experiment into
`classifier/data/`. Their expected hashes are recorded in
[`data/README.md`](data/README.md). For example, from the M4 over SSH:

```bash
scp USER@OLD_MAC:/absolute/path/swda_train.csv data/swda_train.csv
scp USER@OLD_MAC:/absolute/path/swda_val.csv data/swda_val.csv
```

Replace `USER`, `OLD_MAC` and the paths with the values from the older Mac.

Validate all three inputs:

```bash
.venv/bin/python validate_inputs.py \
  --train-csv data/swda_train.csv \
  --validation-csv data/swda_val.csv \
  --studychat-csv data/studychat.csv
```

## 5. Start the full classifier

The default profile targets an Apple Silicon Mac with 24 GB of unified memory:
physical batch 4, gradient accumulation 4, effective batch 16, evaluation
batch 16, padding quantized to a multiple of 8, MPS cache cleanup and resumable
checkpoints.

```bash
mkdir -p outputs/full
nohup /usr/bin/caffeinate -dimsu ./run_full_classifier.sh \
  >> outputs/full/run.log 2>&1 &
echo $! > outputs/full/run.pid
```

Monitor the run:

```bash
tail -f outputs/full/run.log
```

Press `Ctrl+C` to stop monitoring; `nohup` keeps training. Keep the Mac plugged
in and its lid open. If training is interrupted, execute the same launch command
again: `--resume-from-checkpoint auto` uses the newest complete checkpoint.

For a lower-memory Mac, keep the effective batch at 16:

```bash
BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 INFERENCE_BATCH_SIZE=4 \
  ./run_full_classifier.sh
```

Never set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` on a memory-constrained Mac.

## Outputs

After completion, `outputs/full/execution_state.json` has status `completed`.
The main outputs are:

- `model/`: best classifier and tokenizer;
- `studychat_supervised_da.json`: JSON array expected by the original notebooks;
- `studychat_supervised_da_new.csv`: tabular classified dataset;
- `studychat_supervised_da_new.jsonl`: streaming classified dataset;
- `experiment_manifest.json`: parameters, versions, metrics and checksums;
- `trainer_log_history.json`: training and validation history.

All outputs remain local because they contain raw prompts or large model files.

## Validate a completed run

After `execution_state.json` reports `completed`, run the packaged validator:

```bash
.venv/bin/python validate_outputs.py --output-dir outputs/full
```

It checks the three classified formats against each other, requires exactly
16,851 source rows, validates all SwDA codes, names and probabilities, confirms
the 41-output model configuration, reconciles label counts and SHA-256 hashes
with the experiment manifest, and verifies the validation metrics. A sanitized
summary is written to `outputs/full/validation_report.json` and printed to the
terminal.
