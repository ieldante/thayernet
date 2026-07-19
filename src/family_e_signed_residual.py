"""Frozen signed-noise-residual physical construction for Family-E1 preflight."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SignedNoiseResidualOutput:
    requested: Tensor
    companion: Tensor
    residual_noise: Tensor


def signed_noise_residual_allocation(
    logits: Tensor,
    observed: Tensor,
    band_scales: Tensor,
) -> SignedNoiseResidualOutput:
    """Map six normalized logits to nonnegative sources and signed closure.

    Channels 0:3 are requested g/r/z and 3:6 are companion g/r/z.
    ReLU is the in-forward physical source mapping.  The signed residual is
    derived algebraically and is not an astronomical source layer.
    """
    if logits.ndim != 4 or logits.shape[1] != 6:
        raise ValueError("logits must have shape (N,6,H,W)")
    if observed.ndim != 4 or observed.shape[1] != 3:
        raise ValueError("observed must have shape (N,3,H,W)")
    if logits.shape[0] != observed.shape[0] or logits.shape[-2:] != observed.shape[-2:]:
        raise ValueError("logits and observed batch/spatial shapes must match")
    scales = torch.as_tensor(
        band_scales, dtype=logits.dtype, device=logits.device
    ).reshape(1, 3, 1, 1)
    if tuple(scales.shape) != (1, 3, 1, 1):
        raise ValueError("band_scales must contain exactly g/r/z")
    if not bool(torch.all(scales > 0).detach().cpu()):
        raise ValueError("band_scales must be finite and positive")
    requested = torch.relu(logits[:, :3]) * scales
    companion = torch.relu(logits[:, 3:]) * scales
    residual = observed - requested - companion
    return SignedNoiseResidualOutput(requested, companion, residual)


def conservation_error(output: SignedNoiseResidualOutput, observed: Tensor) -> Tensor:
    reconstructed = output.requested + output.companion + output.residual_noise
    return torch.max(torch.abs(reconstructed - observed))


def inverse_target_witness(
    requested_target: np.ndarray,
    companion_target: np.ndarray,
    band_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return truth-only normalized logits for offline representability audit."""
    requested = np.asarray(requested_target, dtype=np.float32)
    companion = np.asarray(companion_target, dtype=np.float32)
    scales = np.asarray(band_scales, dtype=np.float32).reshape(1, 3, 1, 1)
    if requested.shape != companion.shape or requested.ndim != 4 or requested.shape[1] != 3:
        raise ValueError("targets must share shape (N,3,H,W)")
    return requested / scales, companion / scales


def apply_witness_numpy(
    requested_logits: np.ndarray,
    companion_logits: np.ndarray,
    band_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.asarray(band_scales, dtype=np.float32).reshape(1, 3, 1, 1)
    requested = np.maximum(np.asarray(requested_logits, dtype=np.float32), 0.0) * scales
    companion = np.maximum(np.asarray(companion_logits, dtype=np.float32), 0.0) * scales
    return requested, companion
