from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "final" / "scripts" / "17_embedding_analysis.py"

spec = importlib.util.spec_from_file_location("embedding_analysis", SCRIPT_PATH)
embedding_analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(embedding_analysis)


class EmbeddingNeighborAnalysisTest(unittest.TestCase):
    def test_support_only_nearest_neighbor_has_expected_scope_and_neighbor(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "champion_id": 1,
                    "name": "A",
                    "support_archetype": "engage",
                    "is_support": True,
                    "primary_class": "Support",
                },
                {
                    "champion_id": 2,
                    "name": "B",
                    "support_archetype": "engage",
                    "is_support": True,
                    "primary_class": "Support",
                },
                {
                    "champion_id": 3,
                    "name": "C",
                    "support_archetype": "mage",
                    "is_support": True,
                    "primary_class": "Mage",
                },
                {
                    "champion_id": 4,
                    "name": "D",
                    "support_archetype": "",
                    "is_support": False,
                    "primary_class": "Mage",
                },
            ]
        )
        x = np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.999, 0.001],
            ],
            dtype=np.float32,
        )
        x_norm = embedding_analysis.normalize(x, norm="l2")

        all_neighbors = embedding_analysis.build_nearest_neighbors(df, x_norm, k=1)
        support_neighbors = embedding_analysis.build_nearest_neighbors_restricted(df, x_norm, k=1)

        a_all = all_neighbors[all_neighbors["support_name"] == "A"].iloc[0]
        self.assertEqual(a_all["neighbor_scope"], "all_champions_exploratory")
        self.assertEqual(a_all["neighbor_name"], "D")
        self.assertFalse(bool(a_all["neighbor_is_support"]))

        a_support = support_neighbors[support_neighbors["support_name"] == "A"].iloc[0]
        self.assertEqual(a_support["neighbor_scope"], "support_only")
        self.assertEqual(a_support["neighbor_name"], "B")
        self.assertTrue(bool(a_support["same_support_archetype"]))

        metrics = embedding_analysis.compute_neighbor_metrics(
            all_neighbors,
            support_neighbors,
            top_k=1,
        )
        self.assertAlmostEqual(metrics["support_only_topk_same_support_archetype_rate"], 2.0 / 3.0)

    def test_missing_archetype_metadata_keeps_empty_neighbor_tables_stable(self) -> None:
        rows = [
            {"champion_id": 10, "vocab_index": 0, "name": "NoMetaA", "embedding": [1.0, 0.0]},
            {"champion_id": 11, "vocab_index": 1, "name": "NoMetaB", "embedding": [0.0, 1.0]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_path = Path(tmpdir) / "embeddings.json"
            embeddings_path.write_text(json.dumps(rows), encoding="utf-8")

            df, x = embedding_analysis.build_embedding_frame(embeddings_path, {}, {})
            x_norm = embedding_analysis.normalize(x, norm="l2")
            support_neighbors = embedding_analysis.build_nearest_neighbors_restricted(df, x_norm, k=1)

        self.assertEqual(len(df), 2)
        self.assertFalse(df["is_support"].any())
        self.assertEqual(set(df["primary_class"]), {"Unknown"})
        self.assertTrue(support_neighbors.empty)
        self.assertEqual(list(support_neighbors.columns), embedding_analysis.NEAREST_NEIGHBOR_COLUMNS)


if __name__ == "__main__":
    unittest.main()
