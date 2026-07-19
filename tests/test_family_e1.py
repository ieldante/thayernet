from __future__ import annotations

import torch

from src.family_e1 import (
    EXPECTED_PARAMETERS,
    FamilyE1UNet,
    conservation_error,
    source_objective,
    trainable_parameter_count,
)


def test_architecture_count_and_single_six_channel_head() -> None:
    model = FamilyE1UNet()
    assert trainable_parameter_count(model) == EXPECTED_PARAMETERS
    heads = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d) and module.out_channels == 6]
    assert heads == [model.source_head]


def test_relu_sources_and_signed_exact_closure() -> None:
    model = FamilyE1UNet()
    model.eval()
    model_input = torch.randn(2, 4, 60, 60)
    observed = torch.randn(2, 3, 60, 60)
    output = model(model_input, observed)
    assert torch.all(output.requested >= 0)
    assert torch.all(output.companion >= 0)
    assert torch.equal(output.residual_noise, observed - output.requested - output.companion)
    assert float(conservation_error(output, observed)) <= 0.015625


def test_truth_is_zero_objective_and_swap_is_penalized() -> None:
    requested = torch.zeros(1, 3, 60, 60)
    companion = torch.zeros_like(requested)
    requested[:, :, 20:25, 15:22] = 3.0
    companion[:, :, 36:41, 38:45] = 1.0
    truth = source_objective(requested, companion, requested, companion)
    swapped = source_objective(companion, requested, requested, companion)
    assert float(truth["total"]) == 0.0
    assert float(swapped["total"]) > 0.0


def test_zero_target_pixels_are_representable() -> None:
    assert torch.equal(torch.relu(torch.tensor([0.0, -1.0])), torch.zeros(2))
