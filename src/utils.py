"""Image reconstruction metrics for deblending experiments."""

from __future__ import annotations

from math import log10
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity as ssim


def _floating_pair(img: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite floating arrays with identical shapes.

    Casting before subtraction prevents unsigned-integer wraparound when a
    caller accidentally supplies the original Galaxy10 ``uint8`` arrays.
    """

    img_array = np.asarray(img, dtype=np.float64)
    ref_array = np.asarray(ref, dtype=np.float64)
    if img_array.shape != ref_array.shape:
        raise ValueError("Predicted and reference images must have matching shapes.")
    if not np.isfinite(img_array).all() or not np.isfinite(ref_array).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return img_array, ref_array


def mse(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean squared error between predicted and reference images."""
    img_array, ref_array = _floating_pair(img, ref)
    return float(np.mean((img_array - ref_array) ** 2))


def mae(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute error between predicted and reference images."""
    img_array, ref_array = _floating_pair(img, ref)
    return float(np.mean(np.abs(img_array - ref_array)))


def psnr(img: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels."""
    if data_range <= 0:
        raise ValueError("data_range must be positive.")
    mse_val = mse(img, ref)
    if mse_val == 0:
        return float("inf")
    return 20 * log10(data_range) - 10 * log10(mse_val)


def ssim_metric(img: np.ndarray, ref: np.ndarray) -> float:
    """Structural similarity for RGB images."""
    img_array, ref_array = _floating_pair(img, ref)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("SSIM inputs must have shape (H, W, 3).")
    return float(ssim(img_array, ref_array, channel_axis=2, data_range=1.0))


def foreground_iou(img: np.ndarray, ref: np.ndarray, threshold: float = 0.05) -> float:
    """Foreground-mask IoU after a simple grayscale threshold."""
    img_mask = img.mean(axis=-1) > threshold
    ref_mask = ref.mean(axis=-1) > threshold
    intersection = np.logical_and(img_mask, ref_mask).sum()
    union = np.logical_or(img_mask, ref_mask).sum()
    return 1.0 if union == 0 else float(intersection / union)


def affected_region_mask(
    target: np.ndarray,
    blended: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    """Mask pixels where blending measurably changed the target image."""
    if target.shape != blended.shape:
        raise ValueError("target and blended images must have matching shapes.")
    if target.ndim != 3:
        raise ValueError("target and blended images must have shape (H, W, C).")

    target_array, blended_array = _floating_pair(target, blended)
    rgb_delta = np.abs(blended_array - target_array).mean(axis=-1)
    return rgb_delta > threshold


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error over a boolean spatial mask."""
    if pred.shape != target.shape:
        raise ValueError("pred and target images must have matching shapes.")
    if mask.shape != pred.shape[:2]:
        raise ValueError("mask must have shape (H, W).")
    if not np.any(mask):
        return float("nan")
    pred_array, target_array = _floating_pair(pred, target)
    return float(np.mean((pred_array[mask] - target_array[mask]) ** 2))


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error over a boolean spatial mask."""
    if pred.shape != target.shape:
        raise ValueError("pred and target images must have matching shapes.")
    if mask.shape != pred.shape[:2]:
        raise ValueError("mask must have shape (H, W).")
    if not np.any(mask):
        return float("nan")
    pred_array, target_array = _floating_pair(pred, target)
    return float(np.mean(np.abs(pred_array[mask] - target_array[mask])))


def evaluation_core_mask_p85_v1(
    target: np.ndarray,
    aperture_fraction: float = 0.18,
    core_percentile: float = 85.0,
) -> np.ndarray:
    """Historical evaluation core mask, explicitly versioned.

    This preserves the original evaluation convention (center at ``H/2,W/2``
    and the brightest 15% of the central aperture).  It is intentionally
    distinct from the v0.2 training-loss core mask.
    """

    # Preserve the historical float32 percentile/tie behavior exactly.  A
    # float64 promotion changes a small number of boundary pixels in real
    # Galaxy10 cutouts and breaks manifest replay parity.
    target_array = np.asarray(target, dtype=np.float32)
    if target_array.ndim != 3 or target_array.shape[-1] != 3:
        raise ValueError("target must have shape (H, W, 3).")
    if not np.isfinite(target_array).all():
        raise ValueError("target must contain only finite values.")
    gray = target_array.mean(axis=-1)
    height, width = gray.shape
    center_y, center_x = height / 2.0, width / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = aperture_fraction * min(height, width)
    aperture = np.hypot(y_grid - center_y, x_grid - center_x) <= radius
    threshold = float(np.percentile(gray[aperture], core_percentile))
    core = aperture & (gray >= threshold)
    return core if np.any(core) else aperture


def loss_core_mask_v02_numpy(
    target: np.ndarray,
    aperture_fraction: float = 0.18,
    brightness_fraction: float = 0.55,
) -> np.ndarray:
    """NumPy reference for the distinct v0.2 training-loss core mask."""

    target_array = np.asarray(target, dtype=np.float32)
    if target_array.ndim != 3 or target_array.shape[-1] != 3:
        raise ValueError("target must have shape (H, W, 3).")
    gray = target_array.mean(axis=-1)
    height, width = gray.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = aperture_fraction * min(height, width)
    aperture = np.hypot(y_grid - center_y, x_grid - center_x) <= radius
    aperture_max = float(np.max(gray[aperture]))
    core = aperture & (gray >= aperture_max * brightness_fraction)
    return core if np.any(core) else aperture


def halo_band_mask_manhattan_v1(
    affected: np.ndarray,
    dilation_iters: int = 5,
) -> np.ndarray:
    """Historical evaluation halo: cross-connectivity dilation minus affected."""

    affected_array = np.asarray(affected, dtype=bool)
    if affected_array.ndim != 2:
        raise ValueError("affected must have shape (H, W).")
    if dilation_iters < 0:
        raise ValueError("dilation_iters must be non-negative.")
    if dilation_iters == 0:
        return np.zeros_like(affected_array)
    return binary_dilation(affected_array, iterations=dilation_iters) & ~affected_array


def masked_mse_summary(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Return macro and affected-pixel-weighted MSE with explicit coverage."""

    if not (len(predictions) == len(targets) == len(masks)):
        raise ValueError("predictions, targets, and masks must have equal length.")
    per_sample: list[float] = []
    squared_error_sum = 0.0
    channel_value_count = 0
    empty_count = 0
    for prediction, target, mask in zip(predictions, targets, masks):
        mask_array = np.asarray(mask, dtype=bool)
        if not np.any(mask_array):
            empty_count += 1
            continue
        prediction_array, target_array = _floating_pair(prediction, target)
        values = (prediction_array[mask_array] - target_array[mask_array]) ** 2
        per_sample.append(float(values.mean()))
        squared_error_sum += float(values.sum())
        channel_value_count += int(values.size)
    return {
        "n_total": len(predictions),
        "n_valid": len(per_sample),
        "n_empty": empty_count,
        "macro_mse": float(np.mean(per_sample)) if per_sample else float("nan"),
        "micro_mse": (
            squared_error_sum / channel_value_count
            if channel_value_count
            else float("nan")
        ),
    }


def aligned_pair_outcomes(
    sample_ids: Sequence[str],
    candidate_values: Sequence[float],
    reference_sample_ids: Sequence[str],
    reference_values: Sequence[float],
) -> dict[str, float | int]:
    """Validate exact sample alignment and report win/loss/tie coverage."""

    if len(sample_ids) != len(candidate_values):
        raise ValueError("Candidate sample IDs and values must have equal length.")
    if len(reference_sample_ids) != len(reference_values):
        raise ValueError("Reference sample IDs and values must have equal length.")
    candidate_ids = list(sample_ids)
    reference_ids = list(reference_sample_ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate sample IDs must be unique.")
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference sample IDs must be unique.")
    if candidate_ids != reference_ids:
        raise ValueError("Candidate/reference sample IDs are not identically ordered.")
    candidate = np.asarray(candidate_values, dtype=float)
    reference = np.asarray(reference_values, dtype=float)
    valid = np.isfinite(candidate) & np.isfinite(reference)
    wins = int(np.sum(candidate[valid] < reference[valid]))
    losses = int(np.sum(candidate[valid] > reference[valid]))
    ties = int(np.sum(candidate[valid] == reference[valid]))
    valid_count = int(valid.sum())
    return {
        "n_total": len(candidate_ids),
        "n_valid_pairs": valid_count,
        "n_missing_pairs": int(len(candidate_ids) - valid_count),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / valid_count if valid_count else float("nan"),
    }


def compute_affected_region_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    blends: Sequence[np.ndarray],
    threshold: float = 0.02,
) -> dict[str, float]:
    """Average masked MSE/MAE where each blend differs from its target."""
    if not (len(predictions) == len(targets) == len(blends)):
        raise ValueError("predictions, targets, and blends must have equal length.")
    if len(predictions) == 0:
        raise ValueError("Cannot compute affected-region metrics over empty inputs.")

    mse_values: list[float] = []
    mae_values: list[float] = []
    mask_fractions: list[float] = []
    for pred, target, blended in zip(predictions, targets, blends):
        mask = affected_region_mask(target, blended, threshold=threshold)
        mse_values.append(masked_mse(pred, target, mask))
        mae_values.append(masked_mae(pred, target, mask))
        mask_fractions.append(float(mask.mean()))

    mse_array = np.asarray(mse_values, dtype=float)
    mae_array = np.asarray(mae_values, dtype=float)
    return {
        "masked_mse": (
            float(np.nanmean(mse_array))
            if not np.all(np.isnan(mse_array))
            else float("nan")
        ),
        "masked_mae": (
            float(np.nanmean(mae_array))
            if not np.all(np.isnan(mae_array))
            else float("nan")
        ),
        "mean_mask_fraction": float(np.mean(mask_fractions)),
    }


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
