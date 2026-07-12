"""Preregistered hierarchical safety semantics and metric-specific risks.

This module intentionally contains no trainable reconstruction code.  Oracle
images are accepted only by empirical-risk functions used to create targets or
evaluate a frozen policy; they are never deployable features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import torch


class QueryState(str, Enum):
    UNIQUE_VALID = "UNIQUE_VALID"
    NULL = "NULL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class HierarchicalQuerySemantics:
    version: str = "thayer-select-hierarchical-query-v1"
    matching_radius_pixels: float = 4.0
    psf_fwhm_pixels: float = 4.066666666666666
    ambiguity_margin_pixels: float = 1.0
    maximum_perturbation_pixels: float = 3.5
    edge_policy: str = "prompt center must be finite and inside the image"

    @property
    def matching_radius_psf(self) -> float:
        return self.matching_radius_pixels / self.psf_fwhm_pixels

    @property
    def ambiguity_margin_psf(self) -> float:
        return self.ambiguity_margin_pixels / self.psf_fwhm_pixels


@dataclass(frozen=True)
class HierarchicalAssociation:
    state: QueryState
    matched_index: int | None
    nearest_distance_pixels: float
    second_distance_pixels: float
    candidate_count: int


def associate_hierarchical_query(
    source_xy: np.ndarray,
    prompt_xy: np.ndarray,
    *,
    image_shape: tuple[int, int],
    semantics: HierarchicalQuerySemantics | None = None,
) -> HierarchicalAssociation:
    """Apply the frozen UNIQUE_VALID/NULL/AMBIGUOUS association rule."""

    policy = semantics or HierarchicalQuerySemantics()
    sources = np.asarray(source_xy, dtype=np.float64)
    prompt = np.asarray(prompt_xy, dtype=np.float64)
    if sources.ndim != 2 or sources.shape[1] != 2 or len(sources) < 1:
        raise ValueError("source_xy must have shape (N, 2), N >= 1")
    if prompt.shape != (2,) or not np.isfinite(prompt).all() or not np.isfinite(sources).all():
        raise ValueError("source and prompt coordinates must be finite")
    height, width = image_shape
    if not (0.0 <= prompt[0] <= width - 1 and 0.0 <= prompt[1] <= height - 1):
        raise ValueError("prompt coordinate lies outside the image")
    distances = np.linalg.norm(sources - prompt[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    nearest = float(distances[order[0]])
    second = float(distances[order[1]]) if len(order) > 1 else math.inf
    candidate_count = int(np.sum(distances <= policy.matching_radius_pixels))
    if candidate_count == 0:
        return HierarchicalAssociation(QueryState.NULL, None, nearest, second, 0)
    # A near tie is ambiguous even when the second source lies just outside the
    # hard matching radius; otherwise an infinitesimal radius-boundary change
    # would create an unjustified unique ownership decision.
    if nearest <= policy.matching_radius_pixels and second - nearest <= policy.ambiguity_margin_pixels:
        return HierarchicalAssociation(QueryState.AMBIGUOUS, None, nearest, second, candidate_count)
    return HierarchicalAssociation(QueryState.UNIQUE_VALID, int(order[0]), nearest, second, candidate_count)


@dataclass(frozen=True)
class RiskLimits:
    image: float
    flux: float
    centroid_pixels: float


RISK_LIMITS = {
    "strict": RiskLimits(0.40, 0.30, 1.0),
    "moderate": RiskLimits(0.75, 0.50, 2.0),
    "permissive": RiskLimits(1.25, 1.00, 3.0),
}


def positive_centroid(image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64)
    weights = np.maximum(values.sum(axis=0), 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return math.nan, math.nan
    yy, xx = np.mgrid[: values.shape[-2], : values.shape[-1]]
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def metric_specific_risks(
    prediction: np.ndarray,
    requested_truth: np.ndarray,
    alternate_truth: np.ndarray,
    *,
    flux_floor_by_band: np.ndarray,
    psf_fwhm_pixels: float = HierarchicalQuerySemantics().psf_fwhm_pixels,
) -> dict[str, float | bool | list[float]]:
    """Compute continuous valid-query risks in physical image units."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(requested_truth, dtype=np.float64)
    alternate = np.asarray(alternate_truth, dtype=np.float64)
    floors = np.asarray(flux_floor_by_band, dtype=np.float64)
    if pred.shape != truth.shape or alternate.shape != truth.shape or truth.ndim != 3:
        raise ValueError("prediction and truths must share shape (bands, height, width)")
    if floors.shape != (truth.shape[0],) or np.any(~np.isfinite(floors)) or np.any(floors <= 0):
        raise ValueError("flux floors must be finite, positive, and per-band")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all() and np.isfinite(alternate).all()):
        return {
            "image_risk": math.inf,
            "flux_risk_by_band": [math.inf] * truth.shape[0],
            "flux_risk_max": math.inf,
            "centroid_risk_pixels": math.inf,
            "centroid_risk_psf": math.inf,
            "confusion_risk": True,
        }
    truth_power = float(np.mean(truth**2))
    image_risk = math.inf if truth_power <= 0 else float(np.sqrt(np.mean((pred - truth) ** 2) / truth_power))
    pred_flux = pred.sum(axis=(-2, -1))
    truth_flux = truth.sum(axis=(-2, -1))
    flux_error = np.abs(pred_flux - truth_flux) / np.maximum(np.abs(truth_flux), floors)
    pred_centroid = positive_centroid(pred)
    truth_centroid = positive_centroid(truth)
    if np.isfinite((*pred_centroid, *truth_centroid)).all():
        centroid_pixels = float(np.linalg.norm(np.asarray(pred_centroid) - np.asarray(truth_centroid)))
    else:
        centroid_pixels = math.inf
    requested_mse = float(np.mean((pred - truth) ** 2))
    alternate_mse = float(np.mean((pred - alternate) ** 2))
    return {
        "image_risk": image_risk,
        "flux_risk_by_band": flux_error.astype(float).tolist(),
        "flux_risk_max": float(np.max(flux_error)),
        "centroid_risk_pixels": centroid_pixels,
        "centroid_risk_psf": centroid_pixels / psf_fwhm_pixels,
        "confusion_risk": bool(alternate_mse < requested_mse),
    }


def normalized_policy_violation(risks: dict[str, float | bool], limits: RiskLimits) -> float:
    if bool(risks.get("confusion_risk", True)):
        return math.inf
    return float(max(
        float(risks["image_risk"]) / limits.image,
        float(risks["flux_risk_max"]) / limits.flux,
        float(risks["centroid_risk_pixels"]) / limits.centroid_pixels,
    ))


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


def conformal_upper_offset(residuals: np.ndarray, miscoverage: float) -> float:
    """Finite-sample split-conformal upper residual quantile."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if not 0.0 < miscoverage < 1.0 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("invalid conformal residuals or miscoverage")
    rank = min(values.size, int(math.ceil((values.size + 1) * (1.0 - miscoverage))))
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_upper_bound(predicted_quantile: np.ndarray, offset: float) -> np.ndarray:
    values = np.asarray(predicted_quantile, dtype=np.float64)
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    return values + offset
