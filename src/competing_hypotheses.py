"""Truth-free finite-candidate consistency and ambiguity-witness utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


EPSILON = np.finfo(np.float64).eps
FORBIDDEN_AUDITOR_TOKENS = {
    "architecture",
    "checkpoint",
    "family",
    "generator_difficulty",
    "latent",
    "path",
    "private_activation",
    "source_id",
    "target",
    "true_error",
    "true_obstruction",
    "true_snr",
    "training_loss",
}


@dataclass(frozen=True)
class ForwardConsistency:
    global_chi_square_mean: float
    per_band_chi_square_mean: tuple[float, float, float]
    residual_neighbor_correlation: tuple[float, float, float]
    relative_flux_residual: float
    finite: bool


@dataclass(frozen=True)
class PlausibilityThresholds:
    global_chi_square_mean: float
    per_band_chi_square_mean: tuple[float, float, float]
    absolute_relative_flux_residual: float
    calibration_count: int
    quantile_global: float = 0.99
    quantile_per_band: float = 0.995
    quantile_flux: float = 0.99


@dataclass(frozen=True)
class SourceMeasurements:
    flux_grz: tuple[float, float, float]
    centroid_xy: tuple[float, float] | None
    size_rms_pixel: float | None
    ellipticity: tuple[float, float] | None


@dataclass(frozen=True)
class PairwiseScientificDistance:
    image: float
    relative_flux_grz: tuple[float, float, float]
    color_gr_rz_magnitude: tuple[float | None, float | None]
    centroid_pixel: float | None
    centroid_psf: float | None
    shape_size_relative: float | None
    ellipticity_distance: float | None
    primary_normalized: float


@dataclass(frozen=True)
class AmbiguityWitness:
    exists: bool
    retained_candidate_ids: tuple[str, ...]
    maximizing_pair: tuple[str, str] | None
    primary_diameter: float
    artifact_audit_passed: bool
    reason: str


def _as_image(array: np.ndarray, name: str) -> np.ndarray:
    output = np.asarray(array, dtype=np.float64)
    if output.ndim != 3 or output.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3,H,W), got {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} contains non-finite values")
    return output


def _as_layers(array: np.ndarray, name: str = "candidate_layers") -> np.ndarray:
    output = np.asarray(array, dtype=np.float64)
    if output.ndim != 4 or output.shape[0] < 1 or output.shape[1] != 3:
        raise ValueError(f"{name} must have shape (K,3,H,W), got {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} contains non-finite values")
    return output


def recompose(candidate_layers: np.ndarray) -> np.ndarray:
    """Sum source layers without clipping or hidden background duplication."""

    return _as_layers(candidate_layers).sum(axis=0, dtype=np.float64)


def poisson_variance(recomposed_noiseless: np.ndarray, sky_electrons: Sequence[float]) -> np.ndarray:
    """Return the frozen BTK source-plus-sky Poisson variance contract."""

    recomposed = _as_image(recomposed_noiseless, "recomposed_noiseless")
    sky = np.asarray(sky_electrons, dtype=np.float64)
    if sky.shape != (3,) or not np.all(np.isfinite(sky)) or np.any(sky < 0):
        raise ValueError("sky_electrons must be three finite nonnegative values")
    return np.maximum(recomposed + sky[:, None, None], 1.0)


def _neighbor_correlation(image: np.ndarray) -> float:
    pairs = []
    if image.shape[0] > 1:
        pairs.append((image[:-1, :].ravel(), image[1:, :].ravel()))
    if image.shape[1] > 1:
        pairs.append((image[:, :-1].ravel(), image[:, 1:].ravel()))
    if not pairs:
        return 0.0
    left = np.concatenate([pair[0] for pair in pairs])
    right = np.concatenate([pair[1] for pair in pairs])
    if np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def forward_consistency(
    observed_blend: np.ndarray,
    candidate_layers: np.ndarray,
    sky_electrons: Sequence[float],
) -> ForwardConsistency:
    """Score a decomposition using only the observation and known noise contract."""

    observed = _as_image(observed_blend, "observed_blend")
    recomposed = recompose(candidate_layers)
    if recomposed.shape != observed.shape:
        raise ValueError("candidate and observed shapes differ")
    variance = poisson_variance(recomposed, sky_electrons)
    residual = observed - recomposed
    whitened = residual / np.sqrt(variance)
    squared = whitened**2
    band_scores = tuple(float(value) for value in squared.mean(axis=(1, 2)))
    correlations = tuple(_neighbor_correlation(whitened[band]) for band in range(3))
    denominator = float(np.sum(np.abs(observed))) + EPSILON
    flux_residual = float(np.sum(residual) / denominator)
    values = (float(squared.mean()), *band_scores, *correlations, flux_residual)
    return ForwardConsistency(
        global_chi_square_mean=values[0],
        per_band_chi_square_mean=band_scores,
        residual_neighbor_correlation=correlations,
        relative_flux_residual=flux_residual,
        finite=bool(np.all(np.isfinite(values))),
    )


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("calibration values must be a nonempty finite vector")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, quantile, interpolation="higher"))


def calibrate_plausibility(results: Sequence[ForwardConsistency]) -> PlausibilityThresholds:
    """Freeze conservative calibration quantiles from known-truth decompositions."""

    if not results or not all(result.finite for result in results):
        raise ValueError("all calibration consistency results must be finite")
    global_values = np.asarray([result.global_chi_square_mean for result in results])
    band_values = np.asarray([result.per_band_chi_square_mean for result in results])
    flux_values = np.abs(np.asarray([result.relative_flux_residual for result in results]))
    return PlausibilityThresholds(
        global_chi_square_mean=_higher_quantile(global_values, 0.99),
        per_band_chi_square_mean=tuple(
            _higher_quantile(band_values[:, band], 0.995) for band in range(3)
        ),
        absolute_relative_flux_residual=_higher_quantile(flux_values, 0.99),
        calibration_count=len(results),
    )


def is_plausible(result: ForwardConsistency, thresholds: PlausibilityThresholds) -> bool:
    return bool(
        result.finite
        and result.global_chi_square_mean <= thresholds.global_chi_square_mean
        and all(
            value <= limit
            for value, limit in zip(
                result.per_band_chi_square_mean,
                thresholds.per_band_chi_square_mean,
            )
        )
        and abs(result.relative_flux_residual) <= thresholds.absolute_relative_flux_residual
    )


def source_measurements(source: np.ndarray) -> SourceMeasurements:
    image = _as_image(source, "source")
    flux = tuple(float(value) for value in image.sum(axis=(1, 2)))
    weights = np.maximum(image.sum(axis=0), 0.0)
    total = float(weights.sum())
    if total <= EPSILON:
        return SourceMeasurements(flux, None, None, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    x = float(np.sum(weights * xx) / total)
    y = float(np.sum(weights * yy) / total)
    dx = xx - x
    dy = yy - y
    mxx = float(np.sum(weights * dx * dx) / total)
    myy = float(np.sum(weights * dy * dy) / total)
    mxy = float(np.sum(weights * dx * dy) / total)
    trace = mxx + myy
    size = float(np.sqrt(max(trace, 0.0)))
    if trace <= EPSILON:
        ellipticity = (0.0, 0.0)
    else:
        ellipticity = ((mxx - myy) / trace, 2.0 * mxy / trace)
    return SourceMeasurements(flux, (x, y), size, ellipticity)


def _color(flux_a: float, flux_b: float) -> float | None:
    if flux_a <= 0 or flux_b <= 0:
        return None
    return float(-2.5 * np.log10(flux_a / flux_b))


def scientific_distance(
    source_a: np.ndarray,
    source_b: np.ndarray,
    *,
    mean_psf_fwhm_pixel: float,
    image_floor: float = 1e-12,
    flux_floor: float = 1e-12,
) -> PairwiseScientificDistance:
    """Compute the preregistered requested-source distance components."""

    a = _as_image(source_a, "source_a")
    b = _as_image(source_b, "source_b")
    if a.shape != b.shape:
        raise ValueError("requested-source shapes differ")
    if mean_psf_fwhm_pixel <= 0 or image_floor <= 0 or flux_floor <= 0:
        raise ValueError("PSF and floors must be positive")
    image = float(np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + image_floor))
    measure_a = source_measurements(a)
    measure_b = source_measurements(b)
    relative_flux = tuple(
        float(abs(left - right) / (abs(0.5 * (left + right)) + flux_floor))
        for left, right in zip(measure_a.flux_grz, measure_b.flux_grz)
    )
    colors_a = (_color(measure_a.flux_grz[0], measure_a.flux_grz[1]), _color(measure_a.flux_grz[1], measure_a.flux_grz[2]))
    colors_b = (_color(measure_b.flux_grz[0], measure_b.flux_grz[1]), _color(measure_b.flux_grz[1], measure_b.flux_grz[2]))
    colors = tuple(
        None if left is None or right is None else float(abs(left - right))
        for left, right in zip(colors_a, colors_b)
    )
    if measure_a.centroid_xy is None or measure_b.centroid_xy is None:
        centroid = None
        centroid_psf = None
    else:
        centroid = float(np.linalg.norm(np.subtract(measure_a.centroid_xy, measure_b.centroid_xy)))
        centroid_psf = centroid / mean_psf_fwhm_pixel
    if measure_a.size_rms_pixel is None or measure_b.size_rms_pixel is None:
        size_relative = None
        ellipticity_distance = None
    else:
        size_relative = float(
            abs(measure_a.size_rms_pixel - measure_b.size_rms_pixel)
            / (0.5 * (measure_a.size_rms_pixel + measure_b.size_rms_pixel) + EPSILON)
        )
        ellipticity_distance = float(np.linalg.norm(np.subtract(measure_a.ellipticity, measure_b.ellipticity)))
    normalized_components = [image / 0.25, *(value / 0.20 for value in relative_flux)]
    normalized_components.extend(value / 0.20 for value in colors if value is not None)
    if centroid_psf is not None:
        normalized_components.append(centroid_psf / 0.5)
    return PairwiseScientificDistance(
        image=image,
        relative_flux_grz=relative_flux,
        color_gr_rz_magnitude=colors,
        centroid_pixel=centroid,
        centroid_psf=centroid_psf,
        shape_size_relative=size_relative,
        ellipticity_distance=ellipticity_distance,
        primary_normalized=float(max(normalized_components)),
    )


def empirical_ambiguity_witness(
    requested_sources: Mapping[str, np.ndarray],
    consistency: Mapping[str, ForwardConsistency],
    thresholds: PlausibilityThresholds,
    *,
    mean_psf_fwhm_pixel: float,
    artifact_audit_passed: bool,
    image_floor: float = 1e-12,
    flux_floor: float = 1e-12,
) -> AmbiguityWitness:
    """Find the maximum diameter among retained finite candidates."""

    if set(requested_sources) != set(consistency):
        raise ValueError("candidate IDs differ between sources and consistency scores")
    retained = tuple(sorted(candidate for candidate, result in consistency.items() if is_plausible(result, thresholds)))
    if len(retained) < 2:
        return AmbiguityWitness(False, retained, None, 0.0, artifact_audit_passed, "fewer_than_two_plausible_candidates")
    maximum = -np.inf
    maximum_pair: tuple[str, str] | None = None
    for left, right in combinations(retained, 2):
        distance = scientific_distance(
            requested_sources[left],
            requested_sources[right],
            mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
            image_floor=image_floor,
            flux_floor=flux_floor,
        )
        if distance.primary_normalized > maximum:
            maximum = distance.primary_normalized
            maximum_pair = (left, right)
    exists = bool(maximum > 1.0 and artifact_audit_passed)
    reason = "witness" if exists else ("artifact_audit_failed" if maximum > 1.0 else "diameter_within_limits")
    return AmbiguityWitness(exists, retained, maximum_pair, float(maximum), artifact_audit_passed, reason)


def assert_black_box_feature_contract(features: Mapping[str, object]) -> None:
    """Reject provenance, oracle, and private-model fields from auditor tensors."""

    for key, value in features.items():
        normalized = key.lower().replace("-", "_")
        if any(token == normalized or token in normalized for token in FORBIDDEN_AUDITOR_TOKENS):
            raise ValueError(f"forbidden black-box auditor field: {key}")
        if isinstance(value, Mapping):
            assert_black_box_feature_contract(value)


def serialize_dataclass(value: object) -> dict[str, object]:
    """Return JSON-ready scalar/list structures for campaign tables and logs."""

    payload = asdict(value)
    return json_compatible(payload)


def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
