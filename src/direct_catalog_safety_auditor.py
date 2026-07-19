"""Direct pre-deblendability and post-reconstruction catalog-safety audit.

This module contains only deployable feature construction, the two fixed
auditor architectures, calibration/metric helpers, and frozen policy logic.
Truth-bearing arrays are accepted only by :func:`post_audit_supervision`, which
is used to construct supervision and evaluation labels.  They are never
accepted by either network or by :func:`deployable_scalar_features`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from torch import nn

from src.competing_hypotheses import scientific_distance


PRE_CLASSES = ("VALID", "NULL_OR_WRONG", "AMBIGUOUS_OR_UNSUPPORTED")
QUERY_TO_PRE = {
    "UNIQUE_VALID": "VALID",
    "NULL": "NULL_OR_WRONG",
    "AMBIGUOUS": "AMBIGUOUS_OR_UNSUPPORTED",
}
PRE_TO_INDEX = {name: index for index, name in enumerate(PRE_CLASSES)}

IMAGE_CHANNEL_CLIP = 20.0
MEAN_PSF_FWHM_PIXELS = (0.86 + 0.81 + 0.77) / (3.0 * 0.2)
SCIENTIFIC_IMAGE_LIMIT = 0.25
SCIENTIFIC_FLUX_LIMIT = 0.20
SCIENTIFIC_COLOR_LIMIT_MAG = 0.20
SCIENTIFIC_CENTROID_LIMIT_PSF = 0.50
FALSE_SUBTRACTION_LIMIT = 0.20
SOURCE_SUPPORT_FRACTION = 0.01

SCALAR_FEATURE_NAMES = (
    "reconstruction_signed_log_flux_g",
    "reconstruction_signed_log_flux_r",
    "reconstruction_signed_log_flux_z",
    "residual_log_l1_g",
    "residual_log_l1_r",
    "residual_log_l1_z",
    "residual_log_l2_g",
    "residual_log_l2_r",
    "residual_log_l2_z",
    "reconstruction_signed_log_peak_g",
    "reconstruction_signed_log_peak_r",
    "reconstruction_signed_log_peak_z",
    "reconstruction_sparsity_g",
    "reconstruction_sparsity_r",
    "reconstruction_sparsity_z",
    "observation_reconstruction_centroid_displacement",
    "prompt_reconstruction_centroid_displacement",
    "reconstruction_residual_log_abs_flux_ratio_g",
    "reconstruction_residual_log_abs_flux_ratio_r",
    "reconstruction_residual_log_abs_flux_ratio_z",
    "reconstruction_observation_log_abs_flux_ratio_g",
    "reconstruction_observation_log_abs_flux_ratio_r",
    "reconstruction_observation_log_abs_flux_ratio_z",
    "output_finite_indicator",
    "output_nonnegative_indicator",
)


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class _StrideBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class PreAuditQueryNetwork(nn.Module):
    """Frozen A1: four-channel compact three-class query network."""

    input_channels = 4
    output_classes = 3

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _StrideBlock(4, 16),
            _StrideBlock(16, 32),
            _StrideBlock(32, 64),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("PRE-AUDIT expects (N,4,H,W)")
        pooled = self.features(image).mean(dim=(-2, -1))
        return self.classifier(pooled)


class PostAuditSafetyNetwork(nn.Module):
    """Frozen A2: ten image channels plus the deployable scalar vector."""

    input_channels = 10
    scalar_dimension = len(SCALAR_FEATURE_NAMES)

    def __init__(self, scalar_dimension: int = len(SCALAR_FEATURE_NAMES)) -> None:
        super().__init__()
        if scalar_dimension != len(SCALAR_FEATURE_NAMES):
            raise ValueError("A2 scalar dimension differs from the frozen feature contract")
        self.scalar_dimension = scalar_dimension
        self.features = nn.Sequential(
            _StrideBlock(10, 24),
            _StrideBlock(24, 48),
            _StrideBlock(48, 96),
            _StrideBlock(96, 96),
        )
        self.scalar_mlp = nn.Sequential(nn.Linear(scalar_dimension, 32), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, image: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 10:
            raise ValueError("POST-AUDIT expects (N,10,H,W)")
        if scalar.ndim != 2 or scalar.shape != (len(image), self.scalar_dimension):
            raise ValueError("POST-AUDIT scalar feature shape mismatch")
        pooled = self.features(image).mean(dim=(-2, -1))
        scalar_hidden = self.scalar_mlp(scalar)
        return self.fusion(torch.cat((pooled, scalar_hidden), dim=1)).squeeze(1)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def signed_log1p(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.log1p(np.abs(array))


def _positive_centroid(image: np.ndarray) -> tuple[float, float] | None:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        return None
    weights = np.maximum(array.sum(axis=0), 0.0)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    yy, xx = np.mgrid[: array.shape[-2], : array.shape[-1]]
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def _prompt_centroid(prompt: np.ndarray) -> tuple[float, float] | None:
    array = np.asarray(prompt, dtype=np.float64)
    if array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not np.isfinite(array).all():
        return None
    weights = np.maximum(array, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return None
    yy, xx = np.mgrid[: array.shape[-2], : array.shape[-1]]
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def _distance(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 2.0 ** 0.5 * 60.0
    return float(np.linalg.norm(np.subtract(left, right)))


def deployable_scalar_features(
    observed_blend: np.ndarray,
    prompt: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """Return the fixed 25-scalar truth-free POST-AUDIT feature vector."""

    observed = np.asarray(observed_blend, dtype=np.float64)
    request = np.asarray(prompt, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    if observed.shape != candidate.shape or observed.ndim != 3 or observed.shape[0] != 3:
        raise ValueError("blend and reconstruction must have shape (3,H,W)")
    if request.shape != (1, *observed.shape[-2:]):
        raise ValueError("prompt must have shape (1,H,W)")
    finite = bool(np.isfinite(candidate).all())
    nonnegative = bool(finite and np.min(candidate) >= 0.0)
    safe_candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    safe_observed = np.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
    residual = safe_observed - safe_candidate

    reconstruction_flux = safe_candidate.sum(axis=(-2, -1))
    residual_l1 = np.mean(np.abs(residual), axis=(-2, -1))
    residual_l2 = np.sqrt(np.mean(residual ** 2, axis=(-2, -1)))
    reconstruction_peak = np.max(safe_candidate, axis=(-2, -1))
    abs_candidate = np.abs(safe_candidate)
    abs_peak = np.max(abs_candidate, axis=(-2, -1))
    sparsity = np.asarray([
        float(np.mean(abs_candidate[band] <= 0.01 * abs_peak[band])) if abs_peak[band] > 0 else 1.0
        for band in range(3)
    ])
    observation_centroid = _positive_centroid(safe_observed)
    reconstruction_centroid = _positive_centroid(safe_candidate)
    prompt_centroid = _prompt_centroid(request)
    observation_displacement = _distance(observation_centroid, reconstruction_centroid) / 60.0
    prompt_displacement = _distance(prompt_centroid, reconstruction_centroid) / 60.0
    reconstruction_abs_flux = np.sum(abs_candidate, axis=(-2, -1))
    residual_abs_flux = np.sum(np.abs(residual), axis=(-2, -1))
    observation_abs_flux = np.sum(np.abs(safe_observed), axis=(-2, -1))
    reconstruction_residual_ratio = np.log1p(reconstruction_abs_flux / np.maximum(residual_abs_flux, 1e-12))
    reconstruction_observation_ratio = np.log1p(reconstruction_abs_flux / np.maximum(observation_abs_flux, 1e-12))
    features = np.concatenate((
        signed_log1p(reconstruction_flux),
        np.log1p(residual_l1),
        np.log1p(residual_l2),
        signed_log1p(reconstruction_peak),
        sparsity,
        np.asarray([observation_displacement, prompt_displacement]),
        reconstruction_residual_ratio,
        reconstruction_observation_ratio,
        np.asarray([float(finite), float(nonnegative)]),
    )).astype(np.float32)
    if features.shape != (len(SCALAR_FEATURE_NAMES),) or not np.isfinite(features).all():
        raise RuntimeError("deployable scalar feature construction produced invalid output")
    return features


def normalized_pre_image(blend: np.ndarray, prompt: np.ndarray, scales: Sequence[float]) -> np.ndarray:
    observed = np.asarray(blend, dtype=np.float32)
    request = np.asarray(prompt, dtype=np.float32)
    denominator = np.asarray(scales, dtype=np.float32)[:, None, None]
    if observed.shape != (3, 60, 60) or request.shape != (1, 60, 60) or denominator.shape != (3, 1, 1):
        raise ValueError("unexpected PRE-AUDIT tensor shape")
    normalized = np.nan_to_num(observed / denominator, nan=0.0, posinf=IMAGE_CHANNEL_CLIP, neginf=-IMAGE_CHANNEL_CLIP)
    return np.concatenate((np.clip(normalized, -IMAGE_CHANNEL_CLIP, IMAGE_CHANNEL_CLIP), request), axis=0).astype(np.float32)


def normalized_post_image(
    blend: np.ndarray,
    prompt: np.ndarray,
    reconstruction: np.ndarray,
    scales: Sequence[float],
) -> np.ndarray:
    observed = np.asarray(blend, dtype=np.float32)
    request = np.asarray(prompt, dtype=np.float32)
    candidate = np.asarray(reconstruction, dtype=np.float32)
    denominator = np.asarray(scales, dtype=np.float32)[:, None, None]
    if observed.shape != (3, 60, 60) or candidate.shape != observed.shape or request.shape != (1, 60, 60):
        raise ValueError("unexpected POST-AUDIT tensor shape")
    arrays = []
    for value in (observed, candidate, observed - candidate):
        normalized = np.nan_to_num(value / denominator, nan=0.0, posinf=IMAGE_CHANNEL_CLIP, neginf=-IMAGE_CHANNEL_CLIP)
        arrays.append(np.clip(normalized, -IMAGE_CHANNEL_CLIP, IMAGE_CHANNEL_CLIP))
    return np.concatenate((arrays[0], request, arrays[1], arrays[2]), axis=0).astype(np.float32)


@dataclass(frozen=True)
class PostAuditSupervision:
    unsafe_to_catalog: bool
    catastrophic: bool
    catastrophic_image: bool
    catastrophic_flux: bool
    catastrophic_color: bool
    catastrophic_centroid: bool
    physical_output_contract_failure: bool
    false_subtraction_failure: bool
    false_subtraction_applicable: bool
    worse_than_baseline_catastrophic: bool
    source_confusion: bool
    scientific_primary_distance: float
    baseline_primary_distance: float
    false_subtraction_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _support(image: np.ndarray) -> np.ndarray:
    brightness = np.max(np.maximum(np.asarray(image, dtype=np.float64), 0.0), axis=0)
    peak = float(np.max(brightness))
    return np.zeros(brightness.shape, dtype=bool) if not np.isfinite(peak) or peak <= 0 else brightness > SOURCE_SUPPORT_FRACTION * peak


def post_audit_supervision(
    reconstruction: np.ndarray,
    requested_truth: np.ndarray,
    alternate_truth: np.ndarray,
    observed_blend: np.ndarray,
    *,
    mean_psf_fwhm_pixels: float = MEAN_PSF_FWHM_PIXELS,
) -> PostAuditSupervision:
    """Construct the valid-query safety label from truth, never model inputs."""

    candidate = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(requested_truth, dtype=np.float64)
    alternate = np.asarray(alternate_truth, dtype=np.float64)
    observed = np.asarray(observed_blend, dtype=np.float64)
    shapes_valid = candidate.shape == truth.shape == alternate.shape == observed.shape == (3, 60, 60)
    physical_failure = bool(not shapes_valid or not np.isfinite(candidate).all() or (shapes_valid and np.min(candidate) < 0.0))
    if not shapes_valid or not (np.isfinite(truth).all() and np.isfinite(alternate).all() and np.isfinite(observed).all()):
        return PostAuditSupervision(True, True, True, True, True, True, True, True, False, True, True, math.inf, math.inf, None)

    candidate_for_metrics = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    distance = scientific_distance(candidate_for_metrics, truth, mean_psf_fwhm_pixel=mean_psf_fwhm_pixels)
    baseline = scientific_distance(observed, truth, mean_psf_fwhm_pixel=mean_psf_fwhm_pixels)
    catastrophic_image = bool(not np.isfinite(distance.image) or distance.image > SCIENTIFIC_IMAGE_LIMIT)
    catastrophic_flux = bool(any(not np.isfinite(value) or value > SCIENTIFIC_FLUX_LIMIT for value in distance.relative_flux_grz))
    catastrophic_color = bool(any(value is not None and (not np.isfinite(value) or value > SCIENTIFIC_COLOR_LIMIT_MAG) for value in distance.color_gr_rz_magnitude))
    catastrophic_centroid = bool(distance.centroid_psf is not None and (not np.isfinite(distance.centroid_psf) or distance.centroid_psf > SCIENTIFIC_CENTROID_LIMIT_PSF))
    source_confusion = bool(np.mean((candidate_for_metrics - alternate) ** 2) < np.mean((candidate_for_metrics - truth) ** 2))
    catastrophic = bool(catastrophic_image or catastrophic_flux or catastrophic_color or catastrophic_centroid or source_confusion)

    protected = _support(truth) & ~_support(alternate)
    false_subtraction_applicable = bool(np.any(protected))
    if false_subtraction_applicable:
        truth_positive = np.maximum(truth[:, protected], 0.0)
        removed = np.maximum(truth[:, protected] - candidate_for_metrics[:, protected], 0.0)
        denominator = float(np.sum(truth_positive))
        fraction = math.inf if denominator <= 0 else float(np.sum(removed) / denominator)
        false_subtraction_failure = bool(not np.isfinite(fraction) or fraction > FALSE_SUBTRACTION_LIMIT)
    else:
        fraction = None
        false_subtraction_failure = False

    candidate_mse = float(np.mean((candidate_for_metrics - truth) ** 2))
    baseline_mse = float(np.mean((observed - truth) ** 2))
    worse_than_baseline = bool(catastrophic and candidate_mse > baseline_mse)
    unsafe = bool(catastrophic or physical_failure or false_subtraction_failure or worse_than_baseline)
    return PostAuditSupervision(
        unsafe, catastrophic, catastrophic_image, catastrophic_flux, catastrophic_color,
        catastrophic_centroid, physical_failure, false_subtraction_failure,
        false_subtraction_applicable, worse_than_baseline, source_confusion,
        float(distance.primary_normalized), float(baseline.primary_normalized), fraction,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def confusion_matrix(truth: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    y = np.asarray(truth, dtype=int).reshape(-1)
    pred = np.asarray(prediction, dtype=int).reshape(-1)
    if y.shape != pred.shape or np.any(y < 0) or np.any(y >= classes) or np.any(pred < 0) or np.any(pred >= classes):
        raise ValueError("invalid multiclass arrays")
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    return matrix


def multiclass_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    y = np.asarray(truth, dtype=int).reshape(-1)
    scores = np.asarray(probability, dtype=np.float64)
    if scores.shape != (len(y), len(PRE_CLASSES)) or not np.isfinite(scores).all():
        raise ValueError("invalid multiclass probabilities")
    pred = np.argmax(scores, axis=1)
    matrix = confusion_matrix(y, pred, len(PRE_CLASSES))
    recalls = np.divide(np.diag(matrix), matrix.sum(axis=1), out=np.zeros(len(PRE_CLASSES)), where=matrix.sum(axis=1) > 0)
    precisions = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(len(PRE_CLASSES)), where=matrix.sum(axis=0) > 0)
    f1 = np.divide(2 * recalls * precisions, recalls + precisions, out=np.zeros(len(PRE_CLASSES)), where=(recalls + precisions) > 0)
    onehot = np.eye(len(PRE_CLASSES))[y]
    brier = float(np.mean(np.sum((scores - onehot) ** 2, axis=1)))
    confidence = np.max(scores, axis=1)
    correct = (pred == y).astype(float)
    return {
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.mean(pred == y)),
        "recall_by_class": recalls.astype(float).tolist(),
        "precision_by_class": precisions.astype(float).tolist(),
        "f1_by_class": f1.astype(float).tolist(),
        "confusion_matrix": matrix.tolist(),
        "cross_entropy": float(-np.mean(np.log(np.clip(scores[np.arange(len(y)), y], 1e-12, 1.0)))),
        "brier": brier,
        "ece": expected_calibration_error(confidence, correct),
    }


def binary_auroc(probability: np.ndarray, truth: np.ndarray) -> float:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=int).reshape(-1)
    if score.shape != y.shape or not np.isfinite(score).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("invalid binary arrays")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def binary_auprc(probability: np.ndarray, truth: np.ndarray) -> float:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=int).reshape(-1)
    if score.shape != y.shape or not np.isfinite(score).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("invalid binary arrays")
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float(np.sum(precision * sorted_y) / positives)


def expected_calibration_error(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    if score.shape != y.shape or not np.isfinite(score).all() or not np.isfinite(y).all():
        raise ValueError("invalid ECE arrays")
    total = len(y)
    if total == 0:
        return math.nan
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (score >= edges[index]) & (score < edges[index + 1] if index + 1 < bins else score <= edges[index + 1])
        if np.any(selected):
            result += float(np.mean(selected)) * abs(float(np.mean(score[selected])) - float(np.mean(y[selected])))
    return float(result)


def binary_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=int).reshape(-1)
    return {
        "prevalence": float(np.mean(y)),
        "auroc": binary_auroc(score, y),
        "auprc": binary_auprc(score, y),
        "brier": float(np.mean((score - y) ** 2)),
        "ece": expected_calibration_error(score, y),
    }


def fit_multiclass_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    y = np.asarray(truth, dtype=int)
    def objective(log_temperature: float) -> float:
        probability = softmax(values / math.exp(log_temperature))
        return float(-np.mean(np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1.0))))
    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError("multiclass temperature fit failed")
    return float(math.exp(result.x))


def fit_binary_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=int).reshape(-1)
    def objective(log_temperature: float) -> float:
        probability = sigmoid(values / math.exp(log_temperature))
        return float(-np.mean(y * np.log(np.clip(probability, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - probability, 1e-12, 1.0))))
    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError("binary temperature fit failed")
    return float(math.exp(result.x))


def inverse_frequency_class_weights(labels: np.ndarray, classes: int) -> np.ndarray:
    y = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(y, minlength=classes).astype(float)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (float(np.sum(present)) * counts[present])
    return weights


def policy_metrics(
    query_truth: np.ndarray,
    pre_prediction: np.ndarray,
    post_unsafe_probability: np.ndarray,
    unsafe_truth: np.ndarray,
    catastrophic_truth: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    query = np.asarray(query_truth, dtype=int).reshape(-1)
    pre = np.asarray(pre_prediction, dtype=int).reshape(-1)
    post = np.asarray(post_unsafe_probability, dtype=np.float64).reshape(-1)
    unsafe = np.asarray(unsafe_truth, dtype=int).reshape(-1)
    catastrophic = np.asarray(catastrophic_truth, dtype=int).reshape(-1)
    if not (query.shape == pre.shape == post.shape == unsafe.shape == catastrophic.shape):
        raise ValueError("policy arrays must align")
    valid = query == PRE_TO_INDEX["VALID"]
    null = query == PRE_TO_INDEX["NULL_OR_WRONG"]
    ambiguous = query == PRE_TO_INDEX["AMBIGUOUS_OR_UNSUPPORTED"]
    accepted = (pre == PRE_TO_INDEX["VALID"]) & (post <= threshold)
    accepted_valid = accepted & valid
    valid_count = int(valid.sum())
    accepted_valid_count = int(accepted_valid.sum())
    baseline_unsafe = float(np.mean(unsafe[valid])) if valid_count else math.nan
    accepted_unsafe = float(np.mean(unsafe[accepted_valid])) if accepted_valid_count else 0.0
    baseline_catastrophic = float(np.mean(catastrophic[valid])) if valid_count else math.nan
    accepted_catastrophic = float(np.mean(catastrophic[accepted_valid])) if accepted_valid_count else 0.0
    unsafe_reduction = 0.0 if baseline_unsafe <= 0 else 1.0 - accepted_unsafe / baseline_unsafe
    catastrophic_reduction = 0.0 if baseline_catastrophic <= 0 else 1.0 - accepted_catastrophic / baseline_catastrophic
    return {
        "threshold": float(threshold),
        "valid_count": valid_count,
        "accepted_valid_count": accepted_valid_count,
        "accepted_coverage": float(accepted_valid_count / valid_count) if valid_count else math.nan,
        "accept_all_valid_unsafe_rate": baseline_unsafe,
        "accepted_unsafe_rate": accepted_unsafe,
        "unsafe_rate_reduction": float(unsafe_reduction),
        "accept_all_valid_catastrophic_rate": baseline_catastrophic,
        "accepted_catastrophic_rate": accepted_catastrophic,
        "catastrophic_rate_reduction": float(catastrophic_reduction),
        "null_acceptance": float(np.mean(accepted[null])) if np.any(null) else math.nan,
        "ambiguous_acceptance": float(np.mean(accepted[ambiguous])) if np.any(ambiguous) else math.nan,
    }


def threshold_constraints(metrics: dict[str, float | int]) -> dict[str, bool]:
    return {
        "unsafe_rate_reduction_at_least_0_50": float(metrics["unsafe_rate_reduction"]) >= 0.50,
        "catastrophic_rate_reduction_at_least_0_50": float(metrics["catastrophic_rate_reduction"]) >= 0.50,
        "accepted_coverage_at_least_0_50": float(metrics["accepted_coverage"]) >= 0.50,
        "null_acceptance_at_most_0_05": float(metrics["null_acceptance"]) <= 0.05,
        "ambiguous_acceptance_at_most_0_10": float(metrics["ambiguous_acceptance"]) <= 0.10,
    }


def select_fail_closed_threshold(
    query_truth: np.ndarray,
    pre_prediction: np.ndarray,
    post_probability: np.ndarray,
    unsafe_truth: np.ndarray,
    catastrophic_truth: np.ndarray,
) -> tuple[float, dict[str, float | int], bool, list[dict[str, object]]]:
    score = np.asarray(post_probability, dtype=np.float64).reshape(-1)
    candidates = np.r_[np.nextafter(np.min(score), -np.inf), np.unique(score)]
    rows: list[dict[str, object]] = []
    feasible: list[tuple[float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = policy_metrics(query_truth, pre_prediction, score, unsafe_truth, catastrophic_truth, float(threshold))
        constraints = threshold_constraints(metrics)
        rows.append({**metrics, **constraints, "all_constraints": bool(all(constraints.values()))})
        if all(constraints.values()):
            feasible.append((float(threshold), metrics))
    if feasible:
        threshold, metrics = max(feasible, key=lambda item: (float(item[1]["accepted_coverage"]), item[0]))
        return threshold, metrics, True, rows
    threshold = float(np.nextafter(np.min(score), -np.inf))
    return threshold, policy_metrics(query_truth, pre_prediction, score, unsafe_truth, catastrophic_truth, threshold), False, rows


def connected_components(group_pairs: Iterable[tuple[str, str]]) -> np.ndarray:
    pairs = list(group_pairs)
    parent: dict[str, str] = {}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for left, right in pairs:
        union(str(left), str(right))
    roots = [find(str(left)) for left, _ in pairs]
    unique = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([unique[root] for root in roots], dtype=int)


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    return float(np.mean(array)), float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha))

