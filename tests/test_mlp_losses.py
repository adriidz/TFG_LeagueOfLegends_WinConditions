from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "final" / "scripts"))

from mlp_losses import weighted_mse_loss  # noqa: E402


class WeightedMSELossTest(unittest.TestCase):
    def test_unit_weights_match_mse(self) -> None:
        pred = torch.tensor([0.0, 2.0, 5.0])
        target = torch.tensor([1.0, 2.0, 1.0])
        weight = torch.ones_like(pred)

        expected = torch.mean((pred - target) ** 2)

        self.assertTrue(torch.allclose(weighted_mse_loss(pred, target, weight), expected))

    def test_unbalanced_weights_match_manual_sum_over_weight_sum(self) -> None:
        pred = torch.tensor([0.0, 2.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        weight = torch.tensor([1.0, 2.0, 7.0])

        expected = (weight * (pred - target) ** 2).sum() / weight.sum()

        self.assertTrue(torch.allclose(weighted_mse_loss(pred, target, weight), expected))

    def test_near_zero_weights_are_finite(self) -> None:
        pred = torch.tensor([0.0, 2.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        weight = torch.full_like(pred, 1e-12)

        expected = (weight * (pred - target) ** 2).sum() / weight.sum().clamp_min(1e-8)
        actual = weighted_mse_loss(pred, target, weight)

        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
