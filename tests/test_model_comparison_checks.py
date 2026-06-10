from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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


class FakeEncoder:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(frame), len(frame.columns)), dtype=np.float32)


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

    def test_residual_support_effect_falls_back_to_global_mean(self) -> None:
        df = pd.DataFrame({"ally_utility_champion_id": [100, 999, None]})
        baseline = {
            "support_means": {"100": 0.4},
            "global_mean": 0.25,
        }

        pred = model_comparison.predict_residual_support_effect(df, baseline)

        np.testing.assert_allclose(pred, np.array([0.4, 0.25, 0.25], dtype=np.float32))

    def test_build_residual_eval_matrix_uses_residual_mappings(self) -> None:
        df = pd.DataFrame(
            {
                "ally_bottom_champion_id": [40, 41],
                "ally_utility_champion_id": [100, 100],
                "side": ["blue", "red"],
            }
        )
        preprocess = {
            "encoder": FakeEncoder(),
            "categorical_columns": ["ally_bottom_champion_id", "side"],
            "numeric_columns": [
                "resid_te_support_adc_synergy",
                "resid_te_support_adc_synergy_log_count",
            ],
            "interaction_mappings": {
                "support_adc_synergy": {
                    "columns": ["ally_utility_champion_id", "ally_bottom_champion_id"],
                    "mean_column": "resid_te_support_adc_synergy",
                    "count_column": "resid_te_support_adc_synergy_log_count",
                    "means": {"100|40": 0.12},
                    "counts": {"100|40": 3},
                    "prior": -0.01,
                }
            },
        }

        matrix = model_comparison.build_residual_eval_matrix(df, preprocess)

        self.assertEqual(matrix.shape, (2, 4))
        np.testing.assert_allclose(matrix[:, -2], np.array([0.12, -0.01], dtype=np.float32))
        np.testing.assert_allclose(matrix[:, -1], np.log1p(np.array([3.0, 0.0], dtype=np.float32)))

    def test_residual_context_rows_store_lift_metrics(self) -> None:
        df = pd.DataFrame({"support_roam_score": [0.2, 0.4, 0.7, 0.9]})
        support_pred = np.array([0.3, 0.3, 0.8, 0.8], dtype=np.float32)
        residual_pred = np.array([-0.05, 0.05, -0.05, 0.05], dtype=np.float32)

        rows = model_comparison.make_residual_context_rows(
            df,
            support_pred,
            residual_pred,
            n_train=100,
        )

        self.assertEqual([row["model"] for row in rows][0], "Smoothed Support Mean")
        self.assertEqual(rows[1]["evaluation_scale"], "residual")
        final_row = rows[2]
        self.assertEqual(final_row["model"], "Smoothed Support Mean + Residual Context GBT")
        self.assertIn("r2_lift_over_support_effect", final_row)
        self.assertIn("residual_r2", final_row)
        self.assertEqual(final_row["diagnostic_family"], "support_residual")


if __name__ == "__main__":
    unittest.main()
