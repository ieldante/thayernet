from __future__ import annotations

import numpy as np
import pytest
import torch

from src.family_e1 import EXPECTED_PARAMETERS, FamilyE1UNet, trainable_parameter_count
from src.family_e1p import (
    activation_pair_metrics,
    build_paired_prompt_examples,
    paired_prediction_metrics,
)


def test_paired_examples_duplicate_only_observation_and_swap_source_roles() -> None:
    blend = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    prompt_a = np.zeros((2, 1, 4, 4), dtype=np.float32)
    prompt_b = np.ones((2, 1, 4, 4), dtype=np.float32)
    source_a = blend + 100.0
    source_b = blend + 200.0
    paired = build_paired_prompt_examples(
        blend, prompt_a, prompt_b, source_a, source_b, band_scales=(1.0, 2.0, 4.0)
    )

    assert np.array_equal(paired["observed"][:2], paired["observed"][2:])
    assert np.array_equal(paired["model_input"][:2, :3], paired["model_input"][2:, :3])
    assert np.array_equal(paired["model_input"][:2, 3:], prompt_a)
    assert np.array_equal(paired["model_input"][2:, 3:], prompt_b)
    assert np.array_equal(paired["requested"][:2], source_a)
    assert np.array_equal(paired["requested"][2:], source_b)
    assert np.array_equal(paired["companion"][:2], source_b)
    assert np.array_equal(paired["companion"][2:], source_a)


def test_pair_metrics_separate_prompt_sensitive_and_prompt_ignored_predictions() -> None:
    requested = torch.zeros(2, 3, 4, 4)
    companion = torch.zeros_like(requested)
    requested[0, :, 0, 0] = 1.0
    requested[1, :, 3, 3] = 1.0
    companion[0] = requested[1]
    companion[1] = requested[0]

    sensitive = paired_prediction_metrics(
        requested, requested, companion, scene_count=1, band_scales=(1.0, 1.0, 1.0)
    )
    ignored_prediction = requested[:1].repeat(2, 1, 1, 1)
    ignored = paired_prediction_metrics(
        ignored_prediction,
        requested,
        companion,
        scene_count=1,
        band_scales=(1.0, 1.0, 1.0),
    )

    assert sensitive["prompt_identity"] == 1.0
    assert sensitive["prompt_swap"] == 1.0
    assert sensitive["requested_source_error"] == 0.0
    assert sensitive["companion_leakage"] == 0.0
    assert sensitive["cross_prompt_l1_response_ratio"] == 1.0
    assert ignored["prompt_identity"] == 0.5
    assert ignored["prompt_swap"] == 0.0
    assert ignored["cross_prompt_l1_difference"] == 0.0
    assert ignored["cross_prompt_cosine_similarity"] == pytest.approx(1.0, abs=1.0e-6)


def test_activation_pair_metrics_detect_indistinguishable_features() -> None:
    first = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    identical = torch.cat((first, first), dim=0)
    changed = torch.cat((first, first + 2.0), dim=0)

    same_metrics = activation_pair_metrics(identical, scene_count=1)
    changed_metrics = activation_pair_metrics(changed, scene_count=1)

    assert same_metrics["prompt_activation_norm"] == 0.0
    assert same_metrics["feature_modulation"] == 0.0
    assert same_metrics["mutual_information_proxy_nats"] == 0.0
    assert same_metrics["cross_correlation"] == pytest.approx(1.0, abs=1.0e-6)
    assert changed_metrics["prompt_activation_norm"] > 0.0
    assert changed_metrics["feature_modulation"] > 0.0
    assert changed_metrics["mutual_information_proxy_nats"] > 0.0


def test_instrumentation_does_not_define_or_modify_the_family_e1_model() -> None:
    model = FamilyE1UNet()
    assert trainable_parameter_count(model) == EXPECTED_PARAMETERS == 1_162_662
    assert tuple(model.enc0_first[0].weight.shape) == (24, 4, 3, 3)
    assert tuple(model.source_head.weight.shape) == (6, 24, 1, 1)
