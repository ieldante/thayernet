from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from src.models_probabilistic_unet import (
    LATENT_DIMENSION,
    ThayerProbabilisticUNet,
    decomposition_reconstruction_loss,
    free_bits_kl,
    gaussian_kl_per_dimension,
    reparameterize,
    set_training_phase,
    source_sum,
    split_decomposition,
    swap_decomposition,
    trainable_parameter_count,
    warm_start_condition_c,
)


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"


class ProbabilisticUNetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ThayerProbabilisticUNet()
        self.blend = torch.randn(2, 3, 60, 60)
        self.prompt = torch.randn(2, 1, 60, 60)
        self.a = torch.randn(2, 3, 60, 60)
        self.b = torch.randn(2, 3, 60, 60)

    def test_parameter_budget_and_shapes(self) -> None:
        self.assertLessEqual(trainable_parameter_count(self.model), 600_000)
        mean, log_variance = self.model.encode_prior(self.blend)
        self.assertEqual(mean.shape, (2, LATENT_DIMENSION))
        output = self.model.decode(self.blend, self.prompt, torch.zeros_like(mean))
        self.assertEqual(output.shape, (2, 6, 60, 60))

    def test_prior_posterior_api_separation(self) -> None:
        self.assertEqual(list(inspect.signature(self.model.encode_prior).parameters), ["observed_blend"])
        self.assertEqual(
            list(inspect.signature(self.model.encode_posterior).parameters),
            ["observed_blend", "source_a", "source_b"],
        )
        prior_a = self.model.encode_prior(self.blend)
        prior_b = self.model.encode_prior(self.blend.clone())
        self.assertTrue(torch.equal(prior_a[0], prior_b[0]))
        posterior_a = self.model.encode_posterior(self.blend, self.a, self.b)[0]
        posterior_b = self.model.encode_posterior(self.blend, self.a + 1, self.b)[0]
        self.assertFalse(torch.equal(posterior_a, posterior_b))

    def test_reparameterization_and_kl_free_bits(self) -> None:
        mean = torch.tensor([[1.0, -2.0]])
        log_variance = torch.zeros_like(mean)
        epsilon = torch.tensor([[0.5, -0.25]])
        self.assertTrue(torch.equal(reparameterize(mean, log_variance, epsilon=epsilon), mean + epsilon))
        kl = gaussian_kl_per_dimension(mean, log_variance, torch.zeros_like(mean), torch.zeros_like(mean))
        self.assertTrue(torch.allclose(kl, 0.5 * mean.square()))
        self.assertAlmostEqual(float(free_bits_kl(torch.zeros(2, 8), 0.05)), 0.4, places=6)

    def test_decomposition_swap_sum_and_losses(self) -> None:
        output = torch.cat((self.a, self.b), dim=1)
        requested, companion = split_decomposition(output)
        self.assertTrue(torch.equal(requested, self.a))
        self.assertTrue(torch.equal(companion, self.b))
        self.assertTrue(torch.equal(swap_decomposition(output), torch.cat((self.b, self.a), dim=1)))
        self.assertTrue(torch.equal(source_sum(output), self.a + self.b))
        losses = decomposition_reconstruction_loss(output, self.a, self.b)
        self.assertTrue(all(float(value) == 0.0 for value in losses.values()))

    def test_warm_start_and_phase_freezing(self) -> None:
        inventory = warm_start_condition_c(self.model, CHECKPOINT)
        self.assertGreater(len(inventory), 20)
        self.assertTrue(all(row["sha256"] for row in inventory))
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
        self.assertTrue(torch.equal(self.model.enc1.block[0].weight, state["enc1.block.0.weight"]))
        self.assertTrue(torch.equal(self.model.decomposition_head.weight[:3], state["reconstruction_head.weight"]))
        self.assertTrue(torch.equal(self.model.decomposition_head.weight[3:], state["reconstruction_head.weight"]))
        set_training_phase(self.model, 1)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.enc1.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.bottleneck.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.dec1.parameters()))
        set_training_phase(self.model, 2)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.bottleneck.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.enc2.parameters()))


if __name__ == "__main__":
    unittest.main()
