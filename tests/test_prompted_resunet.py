"""Focused contract tests for the prospective prompted ResUNet."""

from __future__ import annotations

import unittest

import torch

from src.models_prompted_resunet import PromptedResUNet, ResidualBlock, trainable_parameter_count


class PromptedResUNetTests(unittest.TestCase):
    def test_exact_parameter_count_and_ceiling(self) -> None:
        count = trainable_parameter_count()
        self.assertEqual(count, 199_219)
        self.assertLess(count, 350_000)

    def test_frozen_input_output_shape(self) -> None:
        model = PromptedResUNet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 4, 60, 60))
        self.assertEqual(tuple(output.shape), (2, 3, 60, 60))
        self.assertTrue(torch.isfinite(output).all())

    def test_all_encoder_decoder_transforms_are_residual(self) -> None:
        model = PromptedResUNet()
        blocks = (model.enc0, model.enc1, model.enc2, model.bottleneck, model.dec1, model.dec0)
        self.assertTrue(all(isinstance(block, ResidualBlock) for block in blocks))

    def test_prompt_channel_changes_fresh_output(self) -> None:
        torch.manual_seed(7)
        model = PromptedResUNet().eval()
        image = torch.zeros(1, 4, 60, 60)
        shifted = image.clone()
        shifted[:, 3, 20, 20] = 1.0
        with torch.no_grad():
            difference = torch.max(torch.abs(model(image) - model(shifted)))
        self.assertGreater(float(difference), 0.0)

    def test_rejects_noncontract_shape(self) -> None:
        model = PromptedResUNet()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 60, 60))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4, 64, 64))


if __name__ == "__main__":
    unittest.main()
