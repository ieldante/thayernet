"""Losses and evaluation utilities for operational recoverability.

This module deliberately separates model-accessible predictions from oracle
evaluation quantities.  Ground-truth reconstruction errors may supervise or
evaluate recoverability, but they are never model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .prompt_semantics import QueryClass


@dataclass(frozen=True)
class ReliabilityContract:
    """Predeclared Phase-II empirical reconstruction contract."""

    name: str
    max_normalized_rmse: float
    max_relative_flux_error: float
    max_color_error_mag: float
    max_centroid_error_pixels: float
    hallucination_flux_fraction: float = 0.10
    catastrophic_normalized_rmse: float = 2.0


PHASE2_CONTRACTS = {
    "strict": ReliabilityContract("strict", 0.40, 0.15, 0.15, 1.0),
    "moderate": ReliabilityContract("moderate", 0.75, 0.30, 0.30, 2.0),
    "permissive": ReliabilityContract("permissive", 1.25, 0.50, 0.50, 3.0),
}


def phase2_contract_success(
    metrics: dict[str, float | int | bool],
    query_class: str | QueryClass,
    contract: ReliabilityContract,
) -> bool:
    """Apply a contract to oracle outcomes, never to model inputs."""

    query = QueryClass(query_class)
    if not bool(metrics.get("evaluation_valid", False)):
        return False
    if query is QueryClass.AMBIGUOUS_SOURCE:
        return False
    if query is QueryClass.NULL_SOURCE:
        return not bool(metrics.get("hallucination", True)) and not bool(
            metrics.get("catastrophic_failure", True)
        )
    values = (
        float(metrics.get("normalized_rmse", np.inf)),
        float(metrics.get("max_relative_flux_error", np.inf)),
        float(metrics.get("max_color_error_mag", np.inf)),
        float(metrics.get("centroid_error_pixels", np.inf)),
    )
    if not np.isfinite(values).all():
        return False
    return bool(
        values[0] <= contract.max_normalized_rmse
        and values[1] <= contract.max_relative_flux_error
        and values[2] <= contract.max_color_error_mag
        and values[3] <= contract.max_centroid_error_pixels
        and not bool(metrics.get("source_confusion", True))
        and not bool(metrics.get("catastrophic_failure", True))
    )


@dataclass(frozen=True)
class RecoverabilityThresholds:
    """Provisional scientific thresholds evaluated after inverse normalization.

    Flux-bearing metrics use physical coadd units; dimensionless ratios are
    computed from those physical arrays. These values are candidates, not a
    calibrated operating point.
    """

    threshold_version: str = "recoverability-thresholds-provisional-v0.4"
    state: str = "provisional_not_frozen"
    affected_region_definition: str = "generator-derived injected-source support"
    core_definition: str = "frozen nonempty requested-source core mask"
    flux_error_units: str = "dimensionless fractional error after inverse normalization"
    pixel_flux_floor_nanomaggies_per_pixel: float = 1.0e-12
    aperture_flux_floor_nanomaggies: float = 1.0e-12
    usable_band_min_snr: float = 5.0
    centroid_error_pixels: float = 2.0
    affected_region_nmse: float = 0.10
    core_false_subtraction_fraction: float = 0.20
    whole_source_flux_fraction_error: float = 0.25
    outside_affected_rmse_in_background_sigma: float = 0.25


def bounded_heteroscedastic_gaussian_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    min_log_variance: float = -10.0,
    max_log_variance: float = 4.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gaussian NLL with a finite admissible variance interval.

    Bounding log-variance prevents the degenerate strategy of continually
    increasing predicted variance.  The uncertainty remains a learned loss
    scale; this alone does not make it calibrated.
    """
    if prediction.shape != target.shape or log_variance.shape != target.shape:
        raise ValueError("prediction, target, and log_variance shapes must match")
    if not min_log_variance < max_log_variance:
        raise ValueError("min_log_variance must be smaller than max_log_variance")
    bounded = torch.clamp(log_variance, min_log_variance, max_log_variance)
    per_element = 0.5 * (bounded + (prediction - target).square() * torch.exp(-bounded))
    if reduction == "mean":
        return per_element.mean()
    if reduction == "sum":
        return per_element.sum()
    if reduction == "none":
        return per_element
    raise ValueError("reduction must be one of: mean, sum, none")


def recoverability_bce_loss(
    predicted_score: torch.Tensor, recoverable_label: torch.Tensor
) -> torch.Tensor:
    """Binary supervision for a score already constrained to ``[0, 1]``."""
    if predicted_score.shape != recoverable_label.shape:
        raise ValueError("predicted_score and recoverable_label shapes must match")
    return F.binary_cross_entropy(
        predicted_score.clamp(1e-6, 1.0 - 1e-6),
        recoverable_label.to(predicted_score),
    )


def selective_risk_curve(
    recoverability_scores: np.ndarray,
    reconstruction_losses: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute empirical risk versus retained coverage, highest score first."""
    scores = np.asarray(recoverability_scores, dtype=np.float64).reshape(-1)
    losses = np.asarray(reconstruction_losses, dtype=np.float64).reshape(-1)
    if scores.shape != losses.shape:
        raise ValueError("scores and losses must contain the same number of samples")
    if scores.size == 0:
        return {
            "coverage": np.array([0.0], dtype=np.float64),
            "risk": np.array([np.nan], dtype=np.float64),
            "threshold": np.array([np.nan], dtype=np.float64),
            "n_excluded_nonfinite": np.array([0], dtype=np.int64),
        }
    if not (np.isfinite(scores).all() and np.isfinite(losses).all()):
        raise ValueError(
            "selective-risk inputs must be finite; map invalid evaluations to "
            "the frozen catastrophic loss and invalid scores to a finite "
            "below-range sentinel before calling"
        )
    # Stable ordering makes tied-score behavior deterministic and replayable.
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_losses = losses[order]
    cumulative_loss = np.cumsum(ordered_losses)
    # A score threshold cannot select only part of an equal-score group. Emit
    # points only after complete tie groups so every risk/coverage pair is an
    # attainable operating point.
    group_ends = np.r_[
        np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]) + 1,
        ordered_scores.size,
    ]
    empty_threshold = np.nextafter(ordered_scores[0], np.inf)
    return {
        "coverage": np.r_[0.0, group_ends.astype(np.float64) / ordered_losses.size],
        "risk": np.r_[np.nan, cumulative_loss[group_ends - 1] / group_ends],
        "threshold": np.r_[empty_threshold, ordered_scores[group_ends - 1]],
        "n_excluded_nonfinite": np.array([0], dtype=np.int64),
    }


def operational_recoverable_label_v04(
    *,
    evaluation_valid: bool,
    usable_band_count: int,
    affected_region_nmse: float,
    max_usable_band_flux_fraction_error: float,
    centroid_error_pixels: float,
    max_applicable_false_subtraction_fraction: float | None,
    outside_affected_rmse_in_background_sigma: float = 0.0,
    thresholds: RecoverabilityThresholds | None = None,
) -> bool:
    """Apply the active provisional physical-flux failure aggregation."""
    policy = thresholds or RecoverabilityThresholds()
    if (
        isinstance(usable_band_count, (bool, np.bool_))
        or not isinstance(usable_band_count, (int, np.integer))
        or not 0 <= int(usable_band_count) <= 3
    ):
        raise ValueError("usable_band_count must be a non-boolean integer in [0, 3]")
    scalar_values = (
        affected_region_nmse,
        max_usable_band_flux_fraction_error,
        centroid_error_pixels,
        outside_affected_rmse_in_background_sigma,
    )
    if not evaluation_valid or usable_band_count < 1:
        return False
    threshold_values = (
        policy.affected_region_nmse,
        policy.whole_source_flux_fraction_error,
        policy.centroid_error_pixels,
        policy.core_false_subtraction_fraction,
        policy.pixel_flux_floor_nanomaggies_per_pixel,
        policy.aperture_flux_floor_nanomaggies,
        policy.usable_band_min_snr,
        policy.outside_affected_rmse_in_background_sigma,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in threshold_values):
        raise ValueError("recoverability thresholds must be finite and nonnegative")
    if (
        policy.pixel_flux_floor_nanomaggies_per_pixel <= 0.0
        or policy.aperture_flux_floor_nanomaggies <= 0.0
        or policy.usable_band_min_snr <= 0.0
    ):
        raise ValueError("flux floors and usable-band SNR must be strictly positive")
    if not all(np.isfinite(value) and value >= 0.0 for value in scalar_values):
        return False
    if max_applicable_false_subtraction_fraction is not None:
        if not (
            np.isfinite(max_applicable_false_subtraction_fraction)
            and max_applicable_false_subtraction_fraction >= 0.0
        ):
            return False
    false_subtraction_failed = (
        max_applicable_false_subtraction_fraction is not None
        and max_applicable_false_subtraction_fraction
        > policy.core_false_subtraction_fraction
    )
    return bool(
        affected_region_nmse <= policy.affected_region_nmse
        and max_usable_band_flux_fraction_error
        <= policy.whole_source_flux_fraction_error
        and centroid_error_pixels <= policy.centroid_error_pixels
        and outside_affected_rmse_in_background_sigma
        <= policy.outside_affected_rmse_in_background_sigma
        and not false_subtraction_failed
    )


def operational_recoverable_label_v03(
    *,
    evaluation_valid: bool,
    usable_band_count: int,
    affected_region_nmse: float,
    max_usable_band_flux_fraction_error: float,
    centroid_error_pixels: float,
    max_applicable_false_subtraction_fraction: float | None,
    outside_affected_rmse_in_background_sigma: float = 0.0,
    thresholds: RecoverabilityThresholds | None = None,
) -> bool:
    """Compatibility alias; active default threshold metadata is v0.4."""
    return operational_recoverable_label_v04(
        evaluation_valid=evaluation_valid,
        usable_band_count=usable_band_count,
        affected_region_nmse=affected_region_nmse,
        max_usable_band_flux_fraction_error=max_usable_band_flux_fraction_error,
        centroid_error_pixels=centroid_error_pixels,
        max_applicable_false_subtraction_fraction=(
            max_applicable_false_subtraction_fraction
        ),
        outside_affected_rmse_in_background_sigma=(
            outside_affected_rmse_in_background_sigma
        ),
        thresholds=thresholds,
    )


operational_recoverable_label_v02 = operational_recoverable_label_v03


def empirical_coverage_at_threshold(
    scores: np.ndarray, threshold: float
) -> tuple[int, int, float]:
    """Return selected count, finite count, and empirical coverage."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError(
            "coverage inputs must be finite; map invalid scores to the frozen "
            "finite below-range sentinel first"
        )
    total = int(values.size)
    selected = int(np.sum(values >= threshold))
    return selected, total, float(selected / total) if total else float("nan")
