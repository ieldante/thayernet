import numpy as np
import torch

from src.observability_distillation import (
    SpatialObservabilityHead,
    average_precision,
    binary_auroc,
    classification_metrics,
    connected_component_labels,
    patch_grid,
    radial_patch_summary,
    sample_prompt_patch,
    spatial_observation_channels,
    target_bins,
)


def test_patch_sampling_and_summary_are_prompt_centered():
    image = torch.arange(60 * 60, dtype=torch.float32).reshape(1, 1, 60, 60)
    grid = patch_grid(torch.tensor([[30.0, 30.0]]), patch_size=9, radius_pixels=4.0)
    patch = sample_prompt_patch(image, grid)
    assert patch.shape == (1, 1, 9, 9)
    assert torch.isclose(patch[0, 0, 4, 4], image[0, 0, 30, 30])
    assert radial_patch_summary(patch).shape == (1, 6)


def test_spatial_channels_never_need_truth():
    blend = torch.ones(2, 3, 9, 9)
    candidate = 0.25 * blend
    result = spatial_observation_channels(blend, candidate)
    assert result.shape == (2, 21, 9, 9)
    assert torch.allclose(result[:, 6:9], 0.75 * blend)


def test_metrics_and_target_intersection():
    truth = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(truth, score) == 1.0
    assert average_precision(truth, score) == 1.0
    assert classification_metrics(truth, score)["brier"] < 0.05
    snr_bin, obstruction_bin, joint = target_bins(
        np.array([5.0, 20.0, 40.0]), np.array([0.2, 0.01, 0.0])
    )
    assert snr_bin.tolist() == [0, 1, 2]
    assert obstruction_bin.tolist() == [2, 1, 0]
    assert joint.tolist() == [1.0, 0.0, 0.0]


def test_connected_source_components_and_parameter_ceiling():
    labels = connected_component_labels(["a", "c", "b"], ["b", "d", "e"])
    assert labels[0] == labels[2]
    assert labels[1] != labels[0]
    model = SpatialObservabilityHead(133, combined_scalar_dim=700, shared=True)
    assert sum(p.numel() for p in model.parameters()) < 150_000
