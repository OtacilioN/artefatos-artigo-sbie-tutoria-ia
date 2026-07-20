from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from swda_taxonomy import SWDA_CODES, validate_observed_labels


class ClassifierContractTests(unittest.TestCase):
    def test_taxonomy_has_exactly_41_unique_codes(self) -> None:
        self.assertEqual(len(SWDA_CODES), 41)
        self.assertEqual(len(set(SWDA_CODES)), 41)
        self.assertNotIn("cd", SWDA_CODES)
        self.assertNotIn("ac", SWDA_CODES)
        self.assertNotIn("err", SWDA_CODES)

    def test_taxonomy_rejects_non_swda_labels(self) -> None:
        with self.assertRaises(ValueError):
            validate_observed_labels(set(SWDA_CODES) | {"err"})

    def test_one_prompt_is_one_input_row(self) -> None:
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
                prompts = [row["prompt"] for row in csv.DictReader(handle)]
        self.assertEqual(prompts, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
