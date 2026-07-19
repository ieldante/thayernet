"""Family-E physical-allocation primitives.

This module deliberately contains no model class.  The preregistered campaign
requires target-representability to pass before any neural model is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class AllocationOutput:
    requested: Tensor
    companion: Tensor
    residual: Tensor
    fractions: Tensor


def simplex_source_allocation(logits: Tensor, observed: Tensor) -> AllocationOutput:
    """Apply the frozen requested/companion/residual simplex allocation.

    Parameters
    ----------
    logits
        Shape (N, 9, H, W) or (N, 3, 3, H, W), with axes band then
        allocation for the five-dimensional form.
    observed
        Raw zero-background observed tensor with shape (N, 3, H, W).
        It is never clipped, offset, or otherwise transformed.
    """
    if observed.ndim != 4 or observed.shape[1] != 3:
        raise ValueError("observed must have shape (N,3,H,W)")
    if logits.ndim == 4:
        if logits.shape[1] != 9:
            raise ValueError("four-dimensional logits must have 9 channels")
        logits = logits.reshape(
            logits.shape[0], 3, 3, logits.shape[-2], logits.shape[-1]
        )
    if logits.ndim != 5 or logits.shape[1:3] != (3, 3):
        raise ValueError("logits must have shape (N,3,3,H,W)")
    if logits.shape[0] != observed.shape[0] or logits.shape[-2:] != observed.shape[-2:]:
        raise ValueError("logits and observed batch/spatial shapes must match")
    fractions = torch.softmax(logits, dim=2)
    allocated = fractions * observed.unsqueeze(2)
    return AllocationOutput(
        requested=allocated[:, :, 0],
        companion=allocated[:, :, 1],
        residual=allocated[:, :, 2],
        fractions=fractions,
    )


def conservation_error(output: AllocationOutput, observed: Tensor) -> Tensor:
    reconstructed = output.requested + output.companion + output.residual
    return torch.max(torch.abs(reconstructed - observed))


def frozen_tolerance(array: np.ndarray, factor: float = 1e-6) -> float:
    values = np.asarray(array)
    if values.size == 0:
        return float(factor)
    return float(factor * max(1.0, float(np.max(np.abs(values)))))


def expected_trainable_parameter_count() -> int:
    """Return the preregistered count without constructing a torch module."""

    def conv(source: int, target: int, kernel: int = 3, bias: bool = False) -> int:
        return source * target * kernel * kernel + (target if bias else 0)

    def group_norm(channels: int) -> int:
        return 2 * channels

    total = 0
    # Encoder and downsampling path.
    total += conv(4, 24) + group_norm(24)
    total += conv(24, 24) + group_norm(24)
    total += conv(24, 48) + group_norm(48)
    total += 2 * (conv(48, 48) + group_norm(48))
    total += conv(48, 96) + group_norm(96)
    total += 2 * (conv(96, 96) + group_norm(96))
    total += conv(96, 128) + group_norm(128)
    total += 2 * (conv(128, 128) + group_norm(128))
    # Decoder 2, 1, and 0.
    total += conv(128, 96) + group_norm(96)
    total += conv(192, 96) + group_norm(96)
    total += conv(96, 96) + group_norm(96)
    total += conv(96, 48) + group_norm(48)
    total += conv(96, 48) + group_norm(48)
    total += conv(48, 48) + group_norm(48)
    total += conv(48, 24) + group_norm(24)
    total += conv(48, 24) + group_norm(24)
    total += conv(24, 24) + group_norm(24)
    total += conv(24, 9, kernel=1, bias=True)
    return total


def representability_summary(
    observed: np.ndarray,
    isolated: np.ndarray,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Summarize whether nonnegative simplex allocations can express targets."""
    obs = np.asarray(observed)
    targets = np.asarray(isolated)
    if obs.ndim != 4 or obs.shape[1] != 3:
        raise ValueError("observed must have shape (N,3,H,W)")
    if targets.shape != (obs.shape[0], 2, 3, obs.shape[2], obs.shape[3]):
        raise ValueError("isolated must have shape (N,2,3,H,W)")
    tol = frozen_tolerance(obs) if tolerance is None else float(tolerance)
    target_sum = targets.sum(axis=1, dtype=np.float64)
    excess = target_sum - obs.astype(np.float64)
    return {
        "tolerance": tol,
        "observed_minimum": float(np.min(obs)),
        "observed_negative_count": int(np.count_nonzero(obs < 0)),
        "target_minimum": float(np.min(targets)),
        "target_negative_count": int(np.count_nonzero(targets < -tol)),
        "target_sum_exceedance_count": int(np.count_nonzero(excess > tol)),
        "target_sum_maximum_exceedance": float(np.max(excess)),
        "representable": bool(
            np.all(np.isfinite(obs))
            and np.all(np.isfinite(targets))
            and np.min(obs) >= 0
            and np.min(targets) >= -tol
            and np.max(excess) <= tol
        ),
    }
