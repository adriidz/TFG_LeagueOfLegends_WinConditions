from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "07_model_comparison.py"

spec = importlib.util.spec_from_file_location("model_comparison", SCRIPT_PATH)
model_comparison = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_comparison)


def main_row(model: str, n_eval: int = 10, test_dataset: str = "test.parquet") -> dict:
    return {
        "model": model,
        "trained_target": "raw",
        "evaluation_scale": "raw",
        "r2": 0.1,
        "spearman_corr": 0.2,
        "pearson_corr": 0.3,
        "mae": 0.4,
        "rmse": 0.5,
        "pred_std": 0.6,
        "within_010": 0.7,
        "within_020": 0.8,
        "n_eval": n_eval,
        "seed": 42,
        "test_dataset": test_dataset,
        "metrics_source": "recomputed_from_predictions",
    }


class ModelComparisonChecksTest(unittest.TestCase):
    def test_final_table_rejects_mismatched_n_eval(self) -> None:
        rows = [
            main_row("Global Mean", n_eval=10),
            main_row("Champion Mean", n_eval=11),
        ]

        with self.assertRaises(SystemExit):
            model_comparison.build_final_main_table(rows)

    def test_final_table_rejects_mismatched_test_dataset(self) -> None:
        rows = [
            main_row("Global Mean", test_dataset="test_a.parquet"),
            main_row("Champion Mean", test_dataset="test_b.parquet"),
        ]

        with self.assertRaises(SystemExit):
            model_comparison.build_final_main_table(rows)

    def test_manifest_check_rejects_learned_model_without_protocol(self) -> None:
        rows = [main_row("HistGBT")]
        audit_rows = [
            {
                "model": "HistGBT",
                "manifest_feature_protocol_id": "missing_manifest",
                "matches_main_feature_protocol": False,
            }
        ]

        with self.assertRaises(SystemExit):
            model_comparison.validate_main_rows_have_manifests(rows, audit_rows)


if __name__ == "__main__":
    unittest.main()
