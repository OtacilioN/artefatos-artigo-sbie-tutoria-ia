from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from swda_taxonomy import SWDA_CODE_TO_NAME, SWDA_CODES
from validate_outputs import OUTPUT_FILES, sha256_file, validate_output_dir


class ValidateOutputsTests(unittest.TestCase):
    def test_valid_complete_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            rows = pd.DataFrame(
                {
                    "source_row_id": [0, 1, 2],
                    "prompt": ["one", "two", "three"],
                    "da_code": ["sd", "b", "qy"],
                    "da_label": ["sd", "b", "qy"],
                    "da_name": [SWDA_CODE_TO_NAME[code] for code in ["sd", "b", "qy"]],
                    "da_score": [0.91, 0.82, 0.73],
                }
            )
            paths = {name: root / filename for name, filename in OUTPUT_FILES.items()}
            rows.to_json(paths["json"], orient="records", force_ascii=False, indent=2)
            rows.to_json(paths["jsonl"], orient="records", lines=True, force_ascii=False)
            rows.to_csv(paths["csv"], index=False)
            (root / "execution_state.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (root / "trainer_log_history.json").write_text(
                json.dumps([{"epoch": 1.0}]), encoding="utf-8"
            )
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "id2label": {str(index): code for index, code in enumerate(SWDA_CODES)},
                    }
                ),
                encoding="utf-8",
            )
            (model / "model.safetensors").write_bytes(b"test model")
            manifest = {
                "studychat_rows": 3,
                "taxonomy": {"count": 41},
                "validation_metrics": {"eval_accuracy": 0.76, "eval_macro_f1": 0.43},
                "elapsed_train_seconds": 3600.0,
                "prediction_label_counts": {"b": 1, "qy": 1, "sd": 1},
                "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
            }
            (root / "experiment_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            report = validate_output_dir(root, expected_rows=3)

            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["model_num_labels"], 41)
            self.assertTrue((root / "validation_report.json").is_file())

    def test_rejects_incomplete_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_state.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_output_dir(root, expected_rows=3)


if __name__ == "__main__":
    unittest.main()
