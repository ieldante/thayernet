from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from src.models_two_expert_decoder import (
    ThayerMixtureExperts,
    expert_parameter_distance,
    parameter_count,
    permutation_invariant_target_loss,
    prompt_swap_set_loss,
    set_training_phase,
    source_sum,
    swap_decomposition,
    warm_start_condition_c_encoder,
)


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"


class TwoExpertDecoderTests(unittest.TestCase):
    def test_shape_parameter_ceiling_and_encoder_only_warm_start(self) -> None:
        model = ThayerMixtureExperts()
        inventory = warm_start_condition_c_encoder(model, CHECKPOINT)
        self.assertEqual(parameter_count(model), 165612)
        self.assertLessEqual(parameter_count(model), 250000)
        self.assertTrue(inventory)
        self.assertTrue(all(row["load_rule"] == "exact_encoder_only" for row in inventory))
        output = model(torch.zeros(2, 3, 60, 60), torch.zeros(2, 1, 60, 60))
        self.assertEqual(output.shape, (2, 2, 6, 60, 60))

    def test_expert_parameters_are_disjoint_and_distinct(self) -> None:
        model = ThayerMixtureExperts()
        left = {parameter.data_ptr() for parameter in model.expert_1.parameters()}
        right = {parameter.data_ptr() for parameter in model.expert_2.parameters()}
        self.assertFalse(left & right)
        self.assertGreater(float(expert_parameter_distance(model)), 0.0)

    def test_expert_identity_is_not_an_input(self) -> None:
        parameters = list(inspect.signature(ThayerMixtureExperts.forward).parameters)
        self.assertEqual(parameters, ["self", "observed_blend", "prompt"])

    def test_full_decomposition_sum_and_swap(self) -> None:
        output = torch.arange(12.0).reshape(1, 2, 6, 1, 1)
        self.assertTrue(torch.equal(swap_decomposition(swap_decomposition(output)), output))
        self.assertTrue(torch.equal(source_sum(output), output[..., :3, :, :] + output[..., 3:, :, :]))

    def test_two_target_loss_is_expert_permutation_invariant(self) -> None:
        hypotheses = torch.randn(3, 2, 6, 4, 4)
        targets = torch.randn(3, 2, 6, 4, 4)
        count = torch.full((3,), 2)
        left = permutation_invariant_target_loss(hypotheses, targets, count)["loss"]
        right = permutation_invariant_target_loss(hypotheses[:, [1, 0]], targets, count)["loss"]
        self.assertAlmostEqual(float(left), float(right), places=6)

    def test_ordinary_loss_supervises_both_and_concentrates(self) -> None:
        hypotheses = torch.zeros(1, 2, 6, 2, 2)
        hypotheses[:, 1] = 1
        targets = torch.zeros_like(hypotheses)
        result = permutation_invariant_target_loss(hypotheses, targets, torch.ones(1, dtype=torch.long))
        self.assertGreater(float(result["ordinary_concentration"]), 0.0)

    def test_prompt_swap_is_set_permutation_invariant(self) -> None:
        prompt_a = torch.randn(2, 2, 6, 3, 3)
        prompt_b = swap_decomposition(prompt_a[:, [1, 0]])
        self.assertAlmostEqual(float(prompt_swap_set_loss(prompt_a, prompt_b)), 0.0, places=7)

    def test_training_phase_keeps_early_encoder_frozen(self) -> None:
        model = ThayerMixtureExperts()
        set_training_phase(model, 1)
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.expert_1.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.expert_2.parameters()))
        set_training_phase(model, 2)
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.enc1.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.enc2.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.bottleneck.parameters()))

    def test_expert_one_gradient_does_not_enter_expert_two(self) -> None:
        model = ThayerMixtureExperts()
        output = model(torch.randn(1, 3, 60, 60), torch.randn(1, 1, 60, 60))
        output[:, 0].mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.expert_1.parameters()))
        self.assertTrue(all(parameter.grad is None or not bool(torch.count_nonzero(parameter.grad)) for parameter in model.expert_2.parameters()))

    def test_invalid_target_count_fails(self) -> None:
        with self.assertRaises(ValueError):
            permutation_invariant_target_loss(torch.zeros(1, 2, 6, 2, 2), torch.zeros(1, 2, 6, 2, 2), torch.tensor([3]))


if __name__ == "__main__":
    unittest.main()
