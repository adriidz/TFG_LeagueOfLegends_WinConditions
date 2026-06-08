from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "03c_train_gbt_interactions.py"

spec = importlib.util.spec_from_file_location("gbt_interactions", SCRIPT_PATH)
gbt_interactions = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gbt_interactions
spec.loader.exec_module(gbt_interactions)


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
            "ally_top_summoner1_id": [4, 4, 4, 4],
            "ally_top_summoner2_id": [12, 12, 12, 12],
            "side": ["blue", "red", "blue", "red"],
            "support_roam_score": [0.2, 0.4, 0.6, 0.8],
            "sample_weight": [1.0, 3.0, 1.0, 1.0],
        }
    )


class GBTInteractionsTest(unittest.TestCase):
    def test_main_feature_set_excludes_summoner_spells(self) -> None:
        cols = gbt_interactions.available_base_features(toy_frame(), feature_set="main")

        self.assertIn("ally_utility_champion_id", cols)
        self.assertIn("side", cols)
        self.assertNotIn("ally_top_summoner1_id", cols)

    def test_weighted_smoothed_encoding_uses_weight_sum(self) -> None:
        keys = pd.Series(["a", "a", "b"])
        y = np.array([0.2, 0.4, 0.8], dtype=np.float32)
        weights = np.array([1.0, 3.0, 1.0], dtype=np.float32)
        prior = float(np.sum(y * weights) / np.sum(weights))

        values, counts, weight_sums = gbt_interactions.fit_encoding_map(
            keys,
            y,
            weights,
            global_mean=prior,
            smoothing=2.0,
        )

        expected_a = ((0.2 * 1.0 + 0.4 * 3.0) + 2.0 * prior) / (4.0 + 2.0)
        self.assertAlmostEqual(values["a"], expected_a)
        self.assertEqual(counts["a"], 2)
        self.assertAlmostEqual(weight_sums["a"], 4.0)

    def test_interaction_features_store_reusable_mapping(self) -> None:
        df = toy_frame()
        specs = {
            "support_adc_synergy": [
                "ally_utility_champion_id",
                "ally_bottom_champion_id",
            ]
        }

        train_num, eval_num, mappings = gbt_interactions.build_interaction_features(
            df,
            df,
            df["support_roam_score"].to_numpy(dtype=np.float32),
            df["sample_weight"].to_numpy(dtype=np.float32),
            specs,
            smoothing=2.0,
            n_folds=2,
            seed=42,
        )

        self.assertEqual(len(train_num), len(df))
        self.assertEqual(len(eval_num), len(df))
        self.assertIn("te_support_adc_synergy", train_num.columns)
        self.assertIn("support_adc_synergy", mappings)
        self.assertIn("values", mappings["support_adc_synergy"])
        self.assertIn("weight_sums", mappings["support_adc_synergy"])


if __name__ == "__main__":
    unittest.main()
