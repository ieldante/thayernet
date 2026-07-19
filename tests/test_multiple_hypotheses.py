from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models_multiple_hypotheses import (
    ThayerMultipleHypotheses,
    parameter_count,
    permutation_invariant_target_loss,
    prompt_swap_set_loss,
    source_sum,
    swap_decomposition,
    unordered_set_distance,
    warm_start_condition_c,
)


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"


class MultipleHypothesesTests(unittest.TestCase):
    def test_shape_parameter_ceiling_and_warm_start(self) -> None:
        model = ThayerMultipleHypotheses()
        inventory = warm_start_condition_c(model, CHECKPOINT)
        self.assertGreater(len(inventory), 20)
        self.assertLessEqual(parameter_count(model), 300_000)
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 60, 60), torch.zeros(2, 1, 60, 60))
        self.assertEqual(output.shape, (2, 2, 6, 60, 60))

    def test_full_decomposition_sum_and_swap(self) -> None:
        value = torch.arange(12.0).reshape(1, 2, 6, 1, 1)
        self.assertTrue(torch.equal(source_sum(value), value[:, :, :3] + value[:, :, 3:]))
        self.assertTrue(torch.equal(swap_decomposition(swap_decomposition(value)), value))

    def test_two_target_loss_is_slot_permutation_invariant(self) -> None:
        targets = torch.zeros(1, 2, 6, 2, 2)
        targets[:, 1] = 2.0
        count = torch.tensor([2])
        direct = permutation_invariant_target_loss(targets.clone(), targets, count)["loss"]
        swapped = permutation_invariant_target_loss(targets[:, [1, 0]].clone(), targets, count)["loss"]
        self.assertEqual(float(direct), 0.0)
        self.assertEqual(float(swapped), 0.0)

    def test_ordinary_loss_supervises_both_and_concentrates(self) -> None:
        targets = torch.zeros(1, 2, 6, 2, 2)
        hypotheses = targets.clone()
        self.assertEqual(float(permutation_invariant_target_loss(hypotheses, targets, torch.tensor([1]))["loss"]), 0.0)
        hypotheses[:, 1] = 1.0
        result = permutation_invariant_target_loss(hypotheses, targets, torch.tensor([1]))
        self.assertGreater(float(result["loss"]), 0.0)
        self.assertGreater(float(result["ordinary_concentration"]), 0.0)

    def test_unordered_pair_set_consistency(self) -> None:
        left = torch.zeros(1, 2, 6, 2, 2)
        left[:, 1] = 3.0
        right = left[:, [1, 0]].clone()
        self.assertEqual(float(unordered_set_distance(left, right)), 0.0)

    def test_prompt_swap_is_slot_permutation_invariant(self) -> None:
        prompt_a = torch.randn(2, 2, 6, 3, 3)
        prompt_b = swap_decomposition(prompt_a)[:, [1, 0]]
        self.assertAlmostEqual(float(prompt_swap_set_loss(prompt_a, prompt_b)), 0.0, places=7)

    def test_invalid_target_count_fails(self) -> None:
        value = torch.zeros(1, 2, 6, 2, 2)
        with self.assertRaises(ValueError):
            permutation_invariant_target_loss(value, value, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
