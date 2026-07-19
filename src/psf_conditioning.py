"""Auditable PSF summaries for the Thayer-Select information campaign.

The functions here operate only on PSF kernels and partition labels.  They do
not accept source truth, source identifiers, pixels, reconstruction errors, or
generator difficulty as deployable model inputs.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


def normalized_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative, unit-sum float64 PSF kernel."""

    value = np.asarray(kernel, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 3:
        raise ValueError("PSF kernel must be a two-dimensional image")
    if not np.all(np.isfinite(value)):
        raise ValueError("PSF kernel contains non-finite values")
    if float(value.min()) < -1e-12:
        raise ValueError("PSF kernel contains material negative values")
    value = np.maximum(value, 0.0)
    total = float(value.sum(dtype=np.float64))
    if not total > 0.0:
        raise ValueError("PSF kernel has nonpositive flux")
    return value / total


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous bytes of an array."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def kernel_moments(kernel: np.ndarray, pixel_scale_arcsec: float) -> dict[str, float]:
    """Compute centered second moments and compact concentration summaries."""

    value = normalized_kernel(kernel)
    yy, xx = np.indices(value.shape, dtype=np.float64)
    cx = float(np.sum(value * xx))
    cy = float(np.sum(value * yy))
    dx = xx - cx
    dy = yy - cy
    qxx = float(np.sum(value * dx * dx))
    qyy = float(np.sum(value * dy * dy))
    qxy = float(np.sum(value * dx * dy))
    trace = qxx + qyy
    e1 = (qxx - qyy) / trace if trace > 0 else 0.0
    e2 = 2.0 * qxy / trace if trace > 0 else 0.0
    ellipticity = math.hypot(e1, e2)
    # Even-sized finite stamps leave approximately 1e-6 pixel-grid anisotropy
    # for an exactly axisymmetric analytic profile.  Such an angle is not a
    # stable physical orientation and must not be exposed as a feature.
    orientation = 0.5 * math.atan2(e2, e1) if ellipticity > 1e-5 else math.nan
    determinant = max(qxx * qyy - qxy * qxy, 0.0)
    second_moment_size_pixels = math.sqrt(max(trace, 0.0))
    second_moment_area_pixels2 = 4.0 * math.pi * math.sqrt(determinant)
    noise_equivalent_area_pixels2 = 1.0 / float(np.sum(value * value))
    radius = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(radius.ravel(), kind="stable")
    cumulative = np.cumsum(value.ravel()[order])
    sorted_radius = radius.ravel()[order]

    def encircled_radius(fraction: float) -> float:
        index = min(int(np.searchsorted(cumulative, fraction, side="left")), len(order) - 1)
        return float(sorted_radius[index])

    central = (np.abs(dx) <= 1.5) & (np.abs(dy) <= 1.5)
    return {
        "centroid_x_pixel": cx,
        "centroid_y_pixel": cy,
        "qxx_pixels2": qxx,
        "qyy_pixels2": qyy,
        "qxy_pixels2": qxy,
        "second_moment_size_pixels": second_moment_size_pixels,
        "second_moment_size_arcsec": second_moment_size_pixels * float(pixel_scale_arcsec),
        "second_moment_area_pixels2": second_moment_area_pixels2,
        "second_moment_area_arcsec2": second_moment_area_pixels2 * float(pixel_scale_arcsec) ** 2,
        "ellipticity_e1": e1,
        "ellipticity_e2": e2,
        "ellipticity_magnitude": ellipticity,
        "orientation_radians": orientation,
        "noise_equivalent_area_pixels2": noise_equivalent_area_pixels2,
        "central_3x3_fraction": float(value[central].sum()),
        "half_energy_radius_pixels": encircled_radius(0.5),
        "ninety_energy_radius_pixels": encircled_radius(0.9),
    }


def effective_configuration_count(hashes: list[str]) -> float:
    """Return exp(Shannon entropy) of the empirical configuration counts."""

    if not hashes:
        return 0.0
    counts = np.asarray(list(Counter(hashes).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def kernel_distance(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Return deterministic distances between equal-grid normalized kernels."""

    left = normalized_kernel(first)
    right = normalized_kernel(second)
    if left.shape != right.shape:
        raise ValueError("Kernel distance requires equal shapes")
    delta = left - right
    return {
        "l1": float(np.sum(np.abs(delta))),
        "l2": float(np.sqrt(np.sum(delta * delta))),
        "cosine_distance": float(1.0 - np.sum(left * right) / np.sqrt(np.sum(left * left) * np.sum(right * right))),
    }


def meaningful_scene_variation(configuration_hashes: list[str], scalar_std_max: float) -> bool:
    """Prospective construction gate: more than one scene config and variation."""

    return len(set(configuration_hashes)) > 1 and float(scalar_std_max) > 1e-8
