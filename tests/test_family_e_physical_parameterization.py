from __future__ import annotations

import numpy as np
import pytest
import torch

from src.family_e import (
    conservation_error,
    expected_trainable_parameter_count,
    representability_summary,
    simplex_source_allocation,
)


def test_preregistered_architecture_count_is_static_and_below_ceiling() -> None:
    assert expected_trainable_parameter_count() == 1_162_737
    assert expected_trainable_parameter_count() < 3_000_000


def test_simplex_nonnegative_and_conserving_for_nonnegative_observed() -> None:
    observed = torch.rand(4, 3, 11, 13)
    logits = torch.randn(4, 9, 11, 13, requires_grad=True)
    output = simplex_source_allocation(logits, observed)
    assert torch.all(output.requested >= 0)
    assert torch.all(output.companion >= 0)
    assert torch.all(output.residual >= 0)
    assert float(conservation_error(output, observed).detach()) <= 2e-7
    (output.requested.square().mean() + output.companion.abs().mean()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_source_has_no_fixed_positive_floor_within_tolerance() -> None:
    observed = torch.ones(1, 3, 1, 1)
    logits = torch.zeros(1, 3, 3, 1, 1)
    logits[:, :, 0] = -100.0
    output = simplex_source_allocation(logits, observed)
    assert float(output.requested.max()) <= 1e-7


def test_signed_observed_cannot_have_all_nonnegative_allocations() -> None:
    observed = torch.tensor([[[[-1.0]], [[2.0]], [[3.0]]]])
    output = simplex_source_allocation(torch.zeros(1, 9, 1, 1), observed)
    assert torch.any(output.requested < 0)
    assert torch.any(output.companion < 0)
    assert torch.any(output.residual < 0)
    assert float(conservation_error(output, observed)) == pytest.approx(0.0, abs=2e-7)


def test_target_representability_detects_exceedance() -> None:
    observed = np.ones((1, 3, 2, 2), dtype=np.float32)
    targets = np.full((1, 2, 3, 2, 2), 0.6, dtype=np.float32)
    result = representability_summary(observed, targets)
    assert result["target_sum_exceedance_count"] == 12
    assert not result["representable"]


def test_target_representability_accepts_valid_simplex_targets() -> None:
    observed = np.ones((1, 3, 2, 2), dtype=np.float32)
    targets = np.full((1, 2, 3, 2, 2), 0.4, dtype=np.float32)
    result = representability_summary(observed, targets)
    assert result["target_sum_exceedance_count"] == 0
    assert result["representable"]
