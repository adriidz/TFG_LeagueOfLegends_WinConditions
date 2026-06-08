from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "25_residual_interaction_experiment.py"

spec = importlib.util.spec_from_file_location("residual_interactions", SCRIPT_PATH)
residual_interactions = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = residual_interactions
spec.loader.exec_module(residual_interactions)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "ally_top_champion_id": [10, 10, 11, 11],
            "ally_jungle_champion_id": [20, 20, 21, 21],
            "ally_middle_champion_id": [30, 31, 30, 31],
            "ally_bottom_champion_id": [40, 41, 40, 41],
            "ally_utility_champion_id": [100, 100, 200, 200],
            "enemy_top_champion_id": [50, 51, 50, 51],
            "enemy_jungle_champion_id": [60, 61, 60, 61],
            "enemy_middle_champion_id": [70, 71, 70, 71],
            "enemy_bottom_champion_id": [80, 81, 80, 81],
            "enemy_utility_champion_id": [300, 300, 400, 400],
            "side": ["blue", "red", "blue", "red"],
            "support_roam_score": [0.2, 0.4, 0.6, 0.8],
            "sample_weight": [1.0, 3.0, 1.0, 1.0],
        }
    )


class ResidualInteractionExperimentTest(unittest.TestCase):
    def test_weighted_mean_matches_manual_sum_over_weight_sum(self) -> None:
        values = pd.Series([1.0, 3.0, 5.0])
        weights = pd.Series([1.0, 2.0, 7.0])

        expected = (values * weights).sum() / weights.sum()

        self.assertAlmostEqual(residual_interactions.weighted_mean(values, weights), expected)

    def test_smoothed_support_baseline_uses_weights_and_global_prior(self) -> None:
        df = toy_frame()
        baseline = residual_interactions.build_support_baseline(
            df,
            residual_interactions.TARGET_COL,
            support_smoothing=2.0,
        )

        # Global weighted mean = (0.2*1 + 0.4*3 + 0.6 + 0.8) / 6 = 0.466666...
        prior = 2.8 / 6.0
        support_100 = ((0.2 * 1.0 + 0.4 * 3.0) + 2.0 * prior) / (4.0 + 2.0)

        self.assertAlmostEqual(baseline["global_mean"], prior)
        self.assertAlmostEqual(baseline["support_means"]["100"], support_100)

    def test_context_features_exclude_direct_ally_support(self) -> None:
        self.assertNotIn(
            residual_interactions.SUPPORT_COL,
            residual_interactions.CONTEXT_CATEGORICAL_COLUMNS,
        )
        self.assertIn(
            residual_interactions.SUPPORT_COL,
            residual_interactions.INTERACTION_SPECS["support_adc_synergy"],
        )

    def test_oof_support_baseline_is_finite_and_same_length(self) -> None:
        df = toy_frame()

        pred = residual_interactions.build_oof_support_baseline(
            df,
            residual_interactions.TARGET_COL,
            n_folds=2,
            support_smoothing=2.0,
            seed=42,
        )

        self.assertEqual(len(pred), len(df))
        self.assertTrue(np.isfinite(pred).all())

    def test_prepare_model_matrix_rejects_direct_support_feature(self) -> None:
        df = toy_frame()
        specs = residual_interactions.available_interactions(df)
        baseline = residual_interactions.build_oof_support_baseline(
            df,
            residual_interactions.TARGET_COL,
            n_folds=2,
            support_smoothing=2.0,
            seed=42,
        )
        residual = df[residual_interactions.TARGET_COL].to_numpy(dtype=np.float32) - baseline
        numeric_train, numeric_val, _ = residual_interactions.build_residual_interaction_features(
            df,
            df,
            residual,
            specs,
            smoothing=2.0,
            n_folds=2,
            seed=42,
        )

        _, _, _, _, feature_columns = residual_interactions.prepare_model_matrix(
            df,
            df,
            numeric_train,
            numeric_val,
        )

        self.assertNotIn(residual_interactions.SUPPORT_COL, feature_columns)
        self.assertIn("resid_te_support_adc_synergy", feature_columns)

    def test_interaction_summary_can_include_champion_names(self) -> None:
        df = toy_frame()
        y = df[residual_interactions.TARGET_COL].to_numpy(dtype=np.float32)
        baseline = np.full(len(df), 0.5, dtype=np.float32)
        residual_pred = np.array([0.2, 0.1, -0.2, -0.1], dtype=np.float32)

        summary = residual_interactions.summarize_interactions(
            df,
            y,
            baseline,
            residual_pred,
            {"support_adc_synergy": ["ally_utility_champion_id", "ally_bottom_champion_id"]},
            champion_names={100: "Bard", 40: "Ezreal"},
            min_count=1,
            top_n=10,
        )

        self.assertIn("key_label", summary.columns)
        self.assertTrue(summary["key_label"].str.contains("Bard").any())


if __name__ == "__main__":
    unittest.main()
