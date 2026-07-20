from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from swda_taxonomy import SWDA_CODES, validate_observed_labels
from transformers import TrainerControl, TrainerState

from train_and_infer_swda import (
    PeriodicCheckpointCallback,
    resolve_checkpoint,
    write_execution_state,
)


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_has_exactly_41_unique_codes(self) -> None:
        self.assertEqual(len(SWDA_CODES), 41)
        self.assertEqual(len(set(SWDA_CODES)), 41)
        self.assertNotIn("cd", SWDA_CODES)
        self.assertNotIn("ac", SWDA_CODES)
        self.assertNotIn("err", SWDA_CODES)

    def test_validation_rejects_unexpected_labels(self) -> None:
        with self.assertRaises(ValueError):
            validate_observed_labels(set(SWDA_CODES) | {"err"})


class CardinalityContractTests(unittest.TestCase):
    def test_one_prompt_means_one_input_row(self) -> None:
        rows = [
            {"prompt": "first", "response": "assistant one"},
            {"prompt": "second", "response": "assistant two"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "studychat.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
                writer.writeheader()
                writer.writerows(rows)
            with path.open(encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        prompts = [row["prompt"] for row in loaded]
        self.assertEqual(prompts, ["first", "second"])
        self.assertEqual(len(prompts), len(rows))


class LongRunRecoveryTests(unittest.TestCase):
    def test_checkpoint_schedule_has_an_early_save(self) -> None:
        callback = PeriodicCheckpointCallback(every_n_steps=500, first_step=100)
        state = TrainerState()
        control = TrainerControl()
        state.global_step = 100
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_save)

    def test_checkpoint_schedule_continues_periodically(self) -> None:
        callback = PeriodicCheckpointCallback(every_n_steps=500, first_step=100)
        state = TrainerState()
        control = TrainerControl()
        state.global_step = 500
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_save)

    def test_auto_resume_uses_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-500").mkdir()
            (root / "checkpoint-2000").mkdir()
            checkpoint = resolve_checkpoint("auto", root)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.name, "checkpoint-2000")

    def test_none_starts_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = resolve_checkpoint("none", Path(directory))
        self.assertIsNone(checkpoint)

    def test_execution_state_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution_state.json"
            write_execution_state(path, "training", train_rows=179_766)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "training")
        self.assertEqual(payload["train_rows"], 179_766)


if __name__ == "__main__":
    unittest.main()
