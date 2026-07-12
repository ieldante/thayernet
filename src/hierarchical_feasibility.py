"""Prospective failure-specific targets for hierarchical feasibility.

This module deliberately separates query semantics from reconstruction risks.
Undefined risks are represented as not applicable by the caller; they must
never be coerced to negative labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.hierarchical_safety import HierarchicalQuerySemantics, RiskLimits


SEMANTICS = HierarchicalQuerySemantics(
    version="thayer-select-hierarchical-feasibility-v1",
    matching_radius_pixels=4.0,
    psf_fwhm_pixels=4.066666666666666,
    ambiguity_margin_pixels=1.0,
    maximum_perturbation_pixels=3.5,
    edge_policy="finite prompt center inside the inclusive image boundary",
)


@dataclass(frozen=True)
class FeasibilityThresholds:
    name: str
    risks: RiskLimits
    null_absolute_flux_ratio: float
    null_energy_ratio: float


THRESHOLDS = {
    "strict": FeasibilityThresholds("strict", RiskLimits(0.40, 0.30, 1.0), 0.05, 0.05),
    "moderate": FeasibilityThresholds("moderate", RiskLimits(0.75, 0.50, 2.0), 0.10, 0.10),
    "permissive": FeasibilityThresholds("permissive", RiskLimits(1.25, 1.00, 3.0), 0.20, 0.20),
}


def threshold_record(name: str) -> dict:
    value = THRESHOLDS[name]
    return {"name": name, **asdict(value.risks), "null_absolute_flux_ratio": value.null_absolute_flux_ratio,
            "null_energy_ratio": value.null_energy_ratio}


def catastrophic_valid_failure(
    *, image_risk: float, flux_risk_max: float, centroid_risk_pixels: float, confusion: bool
) -> bool:
    """Severe valid-source failure, frozen before prospective generation.

    Catastrophe means source confusion, a non-finite risk, or at least a
    two-fold violation of one moderate primary risk limit.
    """

    values = np.asarray([image_risk, flux_risk_max, centroid_risk_pixels], dtype=float)
    if not np.isfinite(values).all():
        return True
    limits = THRESHOLDS["moderate"].risks
    return bool(confusion or image_risk >= 2.0 * limits.image or flux_risk_max >= 2.0 * limits.flux
                or centroid_risk_pixels >= 2.0 * limits.centroid_pixels)


def null_hallucination_outcomes(prediction: np.ndarray, blend: np.ndarray) -> dict[str, float | bool]:
    """Continuous and binary NULL output exposure outcomes."""

    pred = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(blend, dtype=np.float64)
    if pred.shape != observed.shape or pred.ndim != 3:
        raise ValueError("prediction and blend must share (bands, height, width)")
    if not np.isfinite(pred).all() or not np.isfinite(observed).all():
        return {"null_output_energy_ratio": math.inf, "null_absolute_flux_ratio": math.inf,
                "null_hallucination": True}
    energy_ratio = float(np.sqrt(np.mean(pred ** 2)) / max(np.sqrt(np.mean(observed ** 2)), 1e-30))
    pred_flux = np.sum(np.abs(pred), axis=(-2, -1))
    blend_flux = np.sum(np.abs(observed), axis=(-2, -1))
    absolute_flux_ratio = float(np.max(pred_flux / np.maximum(blend_flux, 1e-30)))
    primary = THRESHOLDS["moderate"]
    hallucination = absolute_flux_ratio >= primary.null_absolute_flux_ratio or energy_ratio >= primary.null_energy_ratio
    return {"null_output_energy_ratio": energy_ratio, "null_absolute_flux_ratio": absolute_flux_ratio,
            "null_hallucination": bool(hallucination)}


def ambiguous_forced_output(prediction: np.ndarray, isolated_sources: np.ndarray, blend: np.ndarray) -> dict[str, float | bool | int]:
    """Descriptive AMBIGUOUS exposure; never a fitted target.

    A deterministic source is considered exposed when the output has at least
    the primary NULL energy and flux exposure and is closer in MSE to one of
    the two isolated sources.  This does not assign requested-source truth.
    """

    sources = np.asarray(isolated_sources, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if sources.shape != (2,) + pred.shape:
        raise ValueError("isolated_sources must have shape (2, bands, height, width)")
    exposure = null_hallucination_outcomes(pred, blend)
    mse = np.mean((sources - pred[None]) ** 2, axis=(1, 2, 3))
    if not np.isfinite(mse).all():
        return {"ambiguous_forced_output": True, "exposed_source_index": -1,
                "exposed_source_mse_margin": math.inf}
    order = np.argsort(mse, kind="stable")
    margin = float(mse[order[1]] - mse[order[0]])
    forced = bool(exposure["null_hallucination"] and margin > 0.0)
    return {"ambiguous_forced_output": forced, "exposed_source_index": int(order[0]) if forced else -1,
            "exposed_source_mse_margin": margin}
