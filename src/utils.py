"""Image reconstruction metrics for deblending experiments."""

from __future__ import annotations

from math import log10
from typing import Sequence

import numpy as np
from skimage.metrics import structural_similarity as ssim


def mse(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean squared error between predicted and reference images."""
    return float(np.mean((img - ref) ** 2))


def mae(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute error between predicted and reference images."""
    return float(np.mean(np.abs(img - ref)))


def psnr(img: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels."""
    mse_val = mse(img, ref)
    if mse_val == 0:
        return float("inf")
    return 20 * log10(data_range) - 10 * log10(mse_val)


def ssim_metric(img: np.ndarray, ref: np.ndarray) -> float:
    """Structural similarity for RGB images."""
    return float(ssim(img, ref, channel_axis=2, data_range=1.0))


def foreground_iou(img: np.ndarray, ref: np.ndarray, threshold: float = 0.05) -> float:
    """Foreground-mask IoU after a simple grayscale threshold."""
    img_mask = img.mean(axis=-1) > threshold
    ref_mask = ref.mean(axis=-1) > threshold
    intersection = np.logical_and(img_mask, ref_mask).sum()
    union = np.logical_or(img_mask, ref_mask).sum()
    return 1.0 if union == 0 else float(intersection / union)


def _compute_single(
    img: np.ndarray,
    ref: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    if img.shape != ref.shape:
        raise ValueError("Predicted and reference images must have matching shapes.")

    results: dict[str, float] = {}
    for name in metrics:
        if name == "mse":
            results[name] = mse(img, ref)
        elif name == "mae":
            results[name] = mae(img, ref)
        elif name == "psnr":
            results[name] = psnr(img, ref)
        elif name == "ssim":
            results[name] = ssim_metric(img, ref)
        elif name == "iou":
            results[name] = foreground_iou(img, ref)
        else:
            raise ValueError(f"Unsupported metric: {name}")
    return results


def compute_metrics(
    img: np.ndarray | Sequence[np.ndarray],
    ref: np.ndarray | Sequence[np.ndarray],
    metrics: Sequence[str] = ("mse", "mae", "psnr", "ssim"),
) -> dict[str, float]:
    """Compute metrics for one image pair or average them over many pairs."""
    if isinstance(img, (list, tuple)) and isinstance(ref, (list, tuple)):
        if len(img) != len(ref):
            raise ValueError("Predicted and reference image counts must match.")
        if not img:
            raise ValueError("Cannot compute metrics over an empty image sequence.")

        metric_sums = {name: 0.0 for name in metrics}
        for pred_img, ref_img in zip(img, ref):
            values = _compute_single(pred_img, ref_img, metrics)
            for name, value in values.items():
                metric_sums[name] += value
        return {name: value / len(img) for name, value in metric_sums.items()}

    if isinstance(img, np.ndarray) and isinstance(ref, np.ndarray):
        return _compute_single(img, ref, metrics)

    raise TypeError("img and ref must both be arrays or both be sequences of arrays.")
