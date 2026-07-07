"""Simple non-learning baselines for synthetic galaxy deblending."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def identity_baseline(blended: np.ndarray) -> np.ndarray:
    """Return the blended image unchanged."""
    return np.asarray(blended, dtype=np.float32).copy()


def threshold_baseline(blended: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Keep the largest thresholded connected component in the blend."""
    if blended.ndim != 3 or blended.shape[-1] != 3:
        raise ValueError("blended must have shape (H, W, 3).")

    blended = np.asarray(blended, dtype=np.float32)
    gray = blended.mean(axis=-1)
    if threshold is None:
        threshold = float(np.percentile(gray, 90))

    mask = gray >= threshold
    labeled, num_components = label(mask)
    if num_components == 0:
        return np.zeros_like(blended, dtype=np.float32)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    primary_mask = labeled == int(np.argmax(sizes))
    return (blended * primary_mask[..., None]).astype(np.float32)
