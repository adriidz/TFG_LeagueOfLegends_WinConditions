from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "05_empirical_ceiling.py"

spec = importlib.util.spec_from_file_location("empirical_ceiling", SCRIPT_PATH)
empirical_ceiling = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(empirical_ceiling)


class GroupMeanOOSTest(unittest.TestCase):
    def test_oos_r2_matches_manual_expected_value(self) -> None:
        train_groups = pd.Series(["a", "a", "b", "b"])
        train_values = pd.Series([1.0, 3.0, 5.0, 7.0])
        test_groups = pd.Series(["a", "b"])
        test_values = pd.Series([2.0, 10.0])

        result = empirical_ceiling.group_mean_oos_r2(
            train_groups,
            train_values,
            test_groups,
            test_values,
        )

        # Train means are a=2, b=6. Test residuals are 0 and 4, so
        # SS_res=16. Test mean is 6, so SS_tot=(2-6)^2+(10-6)^2=32.
        self.assertAlmostEqual(result["r2_group_mean_oos"], 0.5)
        self.assertEqual(result["n_unseen_test_groups"], 0)
        self.assertEqual(result["n_unseen_test_rows"], 0)

    def test_unseen_test_group_uses_train_global_mean_fallback(self) -> None:
        train_groups = pd.Series(["a", "a", "b", "b"])
        train_values = pd.Series([1.0, 3.0, 5.0, 7.0])
        test_groups = pd.Series(["a", "c"])
        test_values = pd.Series([2.0, 4.0])

        result = empirical_ceiling.group_mean_oos_r2(
            train_groups,
            train_values,
            test_groups,
            test_values,
        )

        # Train means: a=2, b=6. Train global fallback=(1+3+5+7)/4=4.
        # Predictions for test are [2, 4], exactly matching y_true.
        self.assertAlmostEqual(result["r2_group_mean_oos"], 1.0)
        self.assertAlmostEqual(result["train_global_mean"], 4.0)
        self.assertEqual(result["n_unseen_test_groups"], 1)
        self.assertEqual(result["n_unseen_test_rows"], 1)


if __name__ == "__main__":
    unittest.main()
