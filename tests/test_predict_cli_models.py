from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "predict_cli.py"

spec = importlib.util.spec_from_file_location("predict_cli", SCRIPT_PATH)
predict_cli = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = predict_cli
spec.loader.exec_module(predict_cli)


class PredictCliFinalModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.champion_info = predict_cli.load_champion_info(
            predict_cli.DEFAULT_CHAMPION_CLASSES,
            predict_cli.DEFAULT_CHAMPION_ARCHETYPES,
        )
        cls.lookup = predict_cli.build_champion_lookup(cls.champion_info)
        cls.report_metrics = predict_cli.load_report_metrics(predict_cli.DEFAULT_METRICS_TABLE)
        cls.row, cls.assumptions = predict_cli.build_row_from_values(
            {
                "side": "blue",
                "ally_top": "Ornn",
                "ally_jungle": "Lee Sin",
                "ally_middle": "Ahri",
                "ally_bottom": "Jinx",
                "ally_utility": "Nautilus",
                "enemy_top": "Gwen",
                "enemy_jungle": "Viego",
                "enemy_middle": "Orianna",
                "enemy_bottom": "Kai'Sa",
                "enemy_utility": "Lulu",
            },
            predict_cli.CANONICAL_FEATURE_COLUMNS,
            cls.lookup,
            cls.champion_info,
            interactive=False,
        )
        cls.predictors = predict_cli.build_predictors(
            "all",
            predict_cli.DEFAULT_MODELS_ROOT,
            predict_cli.DEFAULT_TRAIN_PATH,
            cls.champion_info,
            cls.report_metrics,
            background_size=8,
            no_shap=True,
        )

    def test_all_final_models_are_available(self) -> None:
        self.assertEqual(set(self.predictors), set(predict_cli.MODEL_KEYS))

    def test_predictions_are_finite_and_clipped(self) -> None:
        for key, predictor in self.predictors.items():
            with self.subTest(model=key):
                score = predictor.predict_score(self.row)

                self.assertTrue(math.isfinite(score))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_model_all_payload_contains_expected_predictions(self) -> None:
        matchups = predict_cli.predict_all_models(
            self.predictors,
            self.row,
            self.assumptions,
            top_n=3,
            explain_enemy=False,
        )

        payload = predict_cli.result_to_dict(
            self.row,
            matchups,
            self.champion_info,
            primary_key="histgbt",
            predictors=self.predictors,
        )

        self.assertEqual(payload["model_key"], "histgbt")
        self.assertEqual(set(payload["model_predictions"]), set(predict_cli.MODEL_KEYS))
        self.assertNotIn("similar_champions", payload)

    def test_batch_extra_columns_are_ignored(self) -> None:
        base = {
            "side": "blue",
            "ally_top": "Ornn",
            "ally_jungle": "Lee Sin",
            "ally_middle": "Ahri",
            "ally_bottom": "Jinx",
            "ally_utility": "Nautilus",
            "enemy_top": "Gwen",
            "enemy_jungle": "Viego",
            "enemy_middle": "Orianna",
            "enemy_bottom": "Kai'Sa",
            "enemy_utility": "Lulu",
        }
        with_extra = {
            **base,
            "ally_utility_spell1": "Flash",
            "ally_utility_spell2": "Ignite",
            "notes": "ignored",
        }

        row_base, _ = predict_cli.build_row_from_values(
            base,
            predict_cli.CANONICAL_FEATURE_COLUMNS,
            self.lookup,
            self.champion_info,
            interactive=False,
        )
        row_extra, _ = predict_cli.build_row_from_values(
            with_extra,
            predict_cli.CANONICAL_FEATURE_COLUMNS,
            self.lookup,
            self.champion_info,
            interactive=False,
        )

        pd.testing.assert_frame_equal(row_base, row_extra)

    def test_weighted_baselines_are_reproducible_from_train(self) -> None:
        train = pd.read_parquet(predict_cli.DEFAULT_TRAIN_PATH)
        expected_global = predict_cli.weighted_mean(
            train["support_roam_score"],
            train["sample_weight"],
        )
        global_predictor = self.predictors["global_mean"]
        champion_predictor = self.predictors["champion_mean"]

        self.assertAlmostEqual(global_predictor.mean, expected_global)
        self.assertAlmostEqual(champion_predictor.global_mean, expected_global)

    def test_report_metrics_match_final_table_contract(self) -> None:
        self.assertEqual(set(self.report_metrics), set(predict_cli.REPORT_MODEL_NAMES.values()))
        for label, metrics in self.report_metrics.items():
            with self.subTest(model=label):
                self.assertEqual(metrics["n_eval"], "57468")

    def test_learned_models_use_final_feature_protocol(self) -> None:
        for key in ("histgbt", "mlp_onehot", "mlp_embed", "mlp_per_role"):
            model_root = predict_cli.DEFAULT_MODELS_ROOT / predict_cli.MODEL_DIRS[key]
            for run_dir in predict_cli.find_run_dirs(model_root, predict_cli.MODEL_FILES[key]):
                with self.subTest(model=key, run=run_dir.name):
                    config = predict_cli.read_json(run_dir / "model_config.json")
                    self.assertEqual(
                        config["feature_protocol_id"],
                        "draft_10_champions_side",
                    )
                    feature_columns = config.get("input_feature_columns", config.get("feature_columns"))
                    self.assertEqual(feature_columns, predict_cli.CANONICAL_FEATURE_COLUMNS)
                    if key == "histgbt":
                        self.assertIn("summoner_spells", config["excluded_feature_groups"])


if __name__ == "__main__":
    unittest.main()
