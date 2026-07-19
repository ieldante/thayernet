from __future__ import annotations

import numpy as np
import torch

from src.scientific_alignment import (
    corrected_objective,
    scientific_components,
    scientific_surrogate,
    smoothmax,
)


SCALES = torch.tensor([11.0, 17.0, 23.0], dtype=torch.float64)
PSF = 4.066666666666666


def source() -> torch.Tensor:
    value = torch.zeros((3, 12, 12), dtype=torch.float64)
    value[0, 5, 5] = 2.0
    value[1, 5, 5] = 3.0
    value[2, 5, 5] = 4.0
    value[:, 5, 6] = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64)
    return value


def test_truth_surrogate_is_zero_and_stationary() -> None:
    truth = source()
    predicted = truth.clone().requires_grad_(True)
    loss = scientific_surrogate(predicted, truth, SCALES, PSF)
    loss.backward()
    assert abs(float(loss.detach())) < 1e-12
    assert float(predicted.grad.abs().max()) < 1e-10


def test_smoothmax_is_zero_anchored_and_near_maximum() -> None:
    zeros = torch.zeros((2, 7), dtype=torch.float64)
    assert torch.allclose(smoothmax(zeros), torch.zeros(2, dtype=torch.float64), atol=1e-14)
    one_violation = zeros.clone()
    one_violation[0, 2] = 1.0
    assert 0.98 < float(smoothmax(one_violation)[0]) <= 1.0


def test_flux_color_translation_components_respond() -> None:
    truth = source()
    flux_changed = truth.clone()
    flux_changed[0] *= 1.2
    translated = torch.roll(truth, shifts=1, dims=-1)
    flux_components = scientific_components(flux_changed, truth, SCALES, PSF)
    translated_components = scientific_components(translated, truth, SCALES, PSF)
    assert float(flux_components.flux_grz[0]) > 0
    assert float(flux_components.color_gr_rz[0]) > 0
    assert float(translated_components.centroid) > 0


def test_corrected_objective_prefers_exact_unordered_set() -> None:
    first = torch.cat((source(), 0.5 * source()), dim=0)
    second = torch.cat((torch.roll(source(), 2, -1), 0.5 * torch.roll(source(), 2, -1)), dim=0)
    targets = torch.stack((first, second)).view(1, 1, 2, 6, 12, 12).repeat(1, 2, 1, 1, 1, 1)
    counts = torch.full((1, 2), 2, dtype=torch.long)
    exact = targets.clone()
    swapped = targets[:, :, [1, 0]].clone()
    compromise = targets.mean(dim=2, keepdim=True).repeat(1, 1, 2, 1, 1, 1)
    exact_loss = corrected_objective(exact, targets, counts, SCALES, PSF)["total"]
    swapped_loss = corrected_objective(swapped, targets, counts, SCALES, PSF)["total"]
    compromise_loss = corrected_objective(compromise, targets, counts, SCALES, PSF)["total"]
    assert float(exact_loss) < 1e-12
    assert float(swapped_loss) < 1e-12
    assert float(compromise_loss) > float(exact_loss)


def test_ordinary_concentration_penalizes_divergence() -> None:
    first = torch.cat((source(), 0.5 * source()), dim=0)
    targets = first.view(1, 1, 1, 6, 12, 12).repeat(1, 2, 2, 1, 1, 1)
    counts = torch.ones((1, 2), dtype=torch.long)
    exact = targets.clone()
    divergent = exact.clone()
    divergent[:, :, 1, :3] = torch.roll(divergent[:, :, 1, :3], 2, -1)
    assert float(corrected_objective(exact, targets, counts, SCALES, PSF)["total"]) < 1e-12
    assert float(corrected_objective(divergent, targets, counts, SCALES, PSF)["ordinary_concentration"].mean()) > 0
