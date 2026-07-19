from __future__ import annotations

import numpy as np
import torch

from src.family_e_signed_residual import (
    apply_witness_numpy,
    conservation_error,
    inverse_target_witness,
    signed_noise_residual_allocation,
)


SCALES = torch.tensor([611.9199829101562, 1805.8800048828125, 1854.199951171875])


def test_sources_nonnegative_and_signed_residual_closes_observation() -> None:
    observed = torch.randn(3, 3, 9, 7)
    logits = torch.randn(3, 6, 9, 7, requires_grad=True)
    output = signed_noise_residual_allocation(logits, observed, SCALES)
    assert torch.all(output.requested >= 0)
    assert torch.all(output.companion >= 0)
    assert float(conservation_error(output, observed).detach()) <= 2e-3
    (output.requested.mean() + output.companion.mean()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_and_negative_logits_have_no_positive_floor() -> None:
    observed = torch.randn(1, 3, 4, 4)
    for logits in (torch.zeros(1, 6, 4, 4), -torch.ones(1, 6, 4, 4)):
        output = signed_noise_residual_allocation(logits, observed, SCALES)
        assert torch.equal(output.requested, torch.zeros_like(output.requested))
        assert torch.equal(output.companion, torch.zeros_like(output.companion))
        assert torch.equal(output.residual_noise, observed)


def test_observation_change_only_changes_closure_residual() -> None:
    observed = torch.randn(2, 3, 5, 5)
    logits = torch.rand(2, 6, 5, 5)
    first = signed_noise_residual_allocation(logits, observed, SCALES)
    second = signed_noise_residual_allocation(logits, observed + 1.0, SCALES)
    assert torch.equal(first.requested, second.requested)
    assert torch.equal(first.companion, second.companion)
    assert torch.allclose(second.residual_noise - first.residual_noise, torch.ones_like(observed))


def test_truth_only_inverse_witness_round_trips_nonnegative_targets() -> None:
    rng = np.random.default_rng(20260715)
    requested = rng.uniform(0.0, 10000.0, size=(8, 3, 6, 6)).astype(np.float32)
    companion = rng.uniform(0.0, 10000.0, size=(8, 3, 6, 6)).astype(np.float32)
    req_logits, comp_logits = inverse_target_witness(
        requested, companion, SCALES.numpy()
    )
    mapped_req, mapped_comp = apply_witness_numpy(
        req_logits, comp_logits, SCALES.numpy()
    )
    tolerance = 1.0e-6 * max(float(requested.max()), float(companion.max()))
    assert float(np.max(np.abs(mapped_req - requested))) <= tolerance
    assert float(np.max(np.abs(mapped_comp - companion))) <= tolerance


def test_companion_isolated_from_requested_logit_change() -> None:
    observed = torch.randn(1, 3, 3, 3)
    logits = torch.rand(1, 6, 3, 3)
    first = signed_noise_residual_allocation(logits, observed, SCALES)
    changed = logits.clone()
    changed[:, :3] += 0.5
    second = signed_noise_residual_allocation(changed, observed, SCALES)
    assert not torch.equal(first.requested, second.requested)
    assert torch.equal(first.companion, second.companion)
