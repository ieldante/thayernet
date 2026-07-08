"""Synthetic foreground-only blending routines for galaxy images."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def _validate_image(image: np.ndarray, name: str) -> None:
    if image.ndim != 3:
        raise ValueError(f"{name} must have shape (H, W, C).")
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have three channels.")


def estimate_background(image: np.ndarray, border_width: int = 20) -> np.ndarray:
    """Estimate a per-channel background from image borders."""
    _validate_image(image, "image")
    height, width, channels = image.shape
    border = max(1, min(border_width, height // 2, width // 2))

    border_pixels = np.concatenate(
        [
            image[:border, :, :].reshape(-1, channels),
            image[-border:, :, :].reshape(-1, channels),
            image[:, :border, :].reshape(-1, channels),
            image[:, -border:, :].reshape(-1, channels),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0).reshape(1, 1, channels)


def estimate_central_source_mask(
    image: np.ndarray,
    dilation_iters: int = 14,
    soft_sigma: float = 8.0,
    aperture_radius: float = 120.0,
    aperture_soft_edge: float = 40.0,
) -> np.ndarray:
    """Estimate a soft central-source mask that retains diffuse halo light."""
    _validate_image(image, "image")
    gray = image.mean(axis=-1)
    height, width = gray.shape
    border = max(1, min(20, height // 2, width // 2))

    border_values = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    background = float(np.median(border_values))
    mad = float(np.median(np.abs(border_values - background)) + 1e-6)

    try:
        otsu_threshold = float(threshold_otsu(gray))
    except ValueError:
        otsu_threshold = float(np.percentile(gray, 95))

    core_threshold = max(
        background + 5.0 * mad,
        otsu_threshold,
        float(np.percentile(gray, 95)),
    )
    core_mask = gray > core_threshold

    labeled = label(core_mask)
    props = regionprops(labeled)
    if not props:
        return np.zeros((height, width), dtype=np.float32)

    center_y, center_x = height / 2.0, width / 2.0

    def component_score(prop: Any) -> float:
        y, x = prop.centroid
        distance = np.hypot(x - center_x, y - center_y)
        return float(distance - 0.02 * prop.area)

    chosen = min(props, key=component_score)
    main_mask = labeled == chosen.label
    if dilation_iters > 0:
        main_mask = binary_dilation(main_mask, iterations=dilation_iters)

    soft_mask = gaussian_filter(main_mask.astype(np.float32), sigma=soft_sigma)
    if soft_mask.max() > 0:
        soft_mask = soft_mask / soft_mask.max()

    y_grid, x_grid = np.ogrid[:height, :width]
    distance = np.hypot(x_grid - center_x, y_grid - center_y)
    aperture = 1.0 - np.clip(
        (distance - aperture_radius) / aperture_soft_edge,
        0.0,
        1.0,
    )
    soft_mask = soft_mask * aperture
    soft_mask[soft_mask < 0.008] = 0.0
    return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)


def extract_source_foreground(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Extract central-source foreground light and simple size metadata."""
    _validate_image(image, "image")
    background = estimate_background(image)
    foreground = np.clip(image - background, 0.0, 1.0)

    mask = estimate_central_source_mask(foreground)
    foreground = foreground * mask[..., None]
    foreground[foreground < 0.002] = 0.0

    area = float((mask > 0.1).sum())
    radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    return foreground.astype(np.float32), {"area": area, "radius": radius}


def shift_foreground(foreground: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate foreground light without wrapping around image boundaries."""
    _validate_image(foreground, "foreground")
    shifted = ndi_shift(
        foreground,
        shift=(dy, dx, 0),
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def _compute_difficulty(
    shift: tuple[int, int],
    brightness: float,
    blur_sigma: float,
    noise_std: float,
    size_ratio: float | None = None,
) -> str:
    """Assign a heuristic analysis label from sampled blend parameters."""
    dx, dy = shift
    shift_mag = abs(dx) + abs(dy)
    score = 0

    if shift_mag <= 24:
        score += 2
    elif shift_mag <= 56:
        score += 1

    if brightness >= 1.05:
        score += 2
    elif brightness >= 0.75:
        score += 1

    if blur_sigma >= 0.5:
        score += 1
    if noise_std >= 0.015:
        score += 1
    if size_ratio is not None and np.isfinite(size_ratio):
        if size_ratio >= 2.0 or size_ratio <= 0.5:
            score += 1

    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def blend_pair(
    target: np.ndarray,
    contaminant: np.ndarray,
    shift: tuple[int, int] = (0, 0),
    rotation: float = 0.0,
    brightness: float = 1.0,
    blur_sigma: float = 0.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend a target image with an extracted contaminant foreground."""
    _validate_image(target, "target")
    _validate_image(contaminant, "contaminant")
    if target.shape != contaminant.shape:
        raise ValueError("target and contaminant must have the same shape.")

    rng = np.random.default_rng() if rng is None else rng
    target = np.asarray(target, dtype=np.float32)
    contaminant = np.asarray(contaminant, dtype=np.float32)

    if blur_sigma > 0:
        target_blurred = gaussian_filter(target, sigma=(blur_sigma, blur_sigma, 0))
    else:
        target_blurred = target.copy()
    target_blurred = np.clip(target_blurred, 0.0, 1.0)

    _, target_size = extract_source_foreground(target)
    # Extract before rotation so interpolation acts on galaxy light, not cutout edges.
    contaminant_foreground, contaminant_size = extract_source_foreground(contaminant)

    target_radius = target_size["radius"]
    contaminant_radius = contaminant_size["radius"]
    size_ratio = (
        contaminant_radius / target_radius if target_radius > 0 else float("nan")
    )

    contaminant_foreground = np.clip(contaminant_foreground * brightness, 0.0, 1.0)

    if abs(rotation) > 1e-12:
        contaminant_foreground = ndi_rotate(
            contaminant_foreground,
            rotation,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        contaminant_foreground[contaminant_foreground < 0.002] = 0.0

    shifted_contaminant = shift_foreground(
        contaminant_foreground,
        dx=int(shift[0]),
        dy=int(shift[1]),
    )
    blended = np.clip(target_blurred + shifted_contaminant, 0.0, 1.0)

    if noise_std > 0.0:
        noise = rng.normal(scale=noise_std, size=blended.shape)
        blended = np.clip(blended + noise, 0.0, 1.0)

    info = {
        "shift": (int(shift[0]), int(shift[1])),
        "rotation": float(rotation),
        "brightness": float(brightness),
        "blur_sigma": float(blur_sigma),
        "noise_std": float(noise_std),
        "difficulty": _compute_difficulty(
            shift=(int(shift[0]), int(shift[1])),
            brightness=float(brightness),
            blur_sigma=float(blur_sigma),
            noise_std=float(noise_std),
            size_ratio=float(size_ratio),
        ),
        "target_area": float(target_size["area"]),
        "target_radius": float(target_radius),
        "contaminant_area": float(contaminant_size["area"]),
        "contaminant_radius": float(contaminant_radius),
        "size_ratio": float(size_ratio),
    }
    return blended.astype(np.float32), info


def _validate_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return low, high


def generate_blends(
    images: np.ndarray,
    n_blends: int,
    max_shift: int = 56,
    brightness_range: tuple[float, float] = (0.4, 1.0),
    blur_range: tuple[float, float] = (0.0, 0.3),
    noise_range: tuple[float, float] = (0.0, 0.01),
    rotation_range: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic blends as dictionaries with target and metadata."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (N, H, W, 3).")
    if images.shape[0] < 2:
        raise ValueError("At least two images are required to generate blends.")
    if n_blends < 0:
        raise ValueError("n_blends must be non-negative.")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")

    brightness_bounds = _validate_range("brightness_range", brightness_range)
    blur_bounds = _validate_range("blur_range", blur_range)
    noise_bounds = _validate_range("noise_range", noise_range)
    rotation_bounds = _validate_range("rotation_range", rotation_range)

    rng = np.random.default_rng() if rng is None else rng
    images = np.asarray(images, dtype=np.float32)
    blends: list[dict[str, Any]] = []

    for _ in range(n_blends):
        target_idx, contaminant_idx = rng.choice(images.shape[0], size=2, replace=False)
        target = images[target_idx]
        contaminant = images[contaminant_idx]

        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        brightness = float(rng.uniform(*brightness_bounds))
        blur_sigma = float(rng.uniform(*blur_bounds))
        noise_std = float(rng.uniform(*noise_bounds))
        rotation = (
            float(rng.uniform(*rotation_bounds))
            if rotation_bounds != (0.0, 0.0)
            else 0.0
        )

        blended, info = blend_pair(
            target=target,
            contaminant=contaminant,
            shift=(dx, dy),
            rotation=rotation,
            brightness=brightness,
            blur_sigma=blur_sigma,
            noise_std=noise_std,
            rng=rng,
        )
        blends.append(
            {
                "target": target,
                "contaminant": contaminant,
                "blended": blended,
                "info": info,
            }
        )

    return blends
