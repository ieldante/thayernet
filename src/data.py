"""Data loading and splitting utilities for Galaxy10 DECaLS experiments."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_galaxy10(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load images, labels, and available metadata from Galaxy10 DECaLS.

    Parameters
    ----------
    h5_path:
        Path to `Galaxy10_DECals.h5`. The file is expected to contain `images`
        and `ans`; optional metadata arrays are returned when present.
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Galaxy10 DECaLS file not found at {path}. "
            "Download it separately and place it under data/."
        )

    with h5py.File(path, "r") as handle:
        missing = [key for key in ("images", "ans") if key not in handle]
        if missing:
            raise KeyError(f"Missing required dataset(s): {', '.join(missing)}")

        images = handle["images"][:]
        labels = handle["ans"][:]
        metadata_keys = ("ra", "dec", "redshift", "pxscale")
        metadata = {key: handle[key][:] for key in metadata_keys if key in handle}

    return images, labels, metadata


def normalise_images(images: np.ndarray) -> np.ndarray:
    """Convert image arrays to `float32` values in `[0, 1]`."""
    if images.size == 0:
        raise ValueError("Cannot normalise an empty image array.")
    return images.astype(np.float32) / 255.0


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split original images before synthetic blends are generated."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images and labels must have the same first dimension.")
    if images.shape[0] < 3:
        raise ValueError("At least three samples are required for train/val/test splits.")

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")

    n_samples = images.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Each split must contain at least one sample.")

    train_end = n_train
    val_end = train_end + n_val
    return (
        (images_shuffled[:train_end], labels_shuffled[:train_end]),
        (images_shuffled[train_end:val_end], labels_shuffled[train_end:val_end]),
        (images_shuffled[val_end:], labels_shuffled[val_end:]),
    )
