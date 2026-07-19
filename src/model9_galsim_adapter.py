"""Read-only adapter from known GalSim PSFs to Model-9 tensor kernels."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.model9_structured import normalize_psf


def sample_galsim_psf_kernels(
    psfs: Sequence[object],
    *,
    pixel_scale_arcsec: float = 0.2,
    kernel_size: int = 31,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw and explicitly normalize three already-authorized known PSFs.

    The adapter does not discover PSFs, inspect catalogs, or access datasets.
    It only converts three caller-supplied GalSim objects to a deterministic
    g/r/z tensor.  The odd kernel is centered by GalSim and then normalized in
    float64 before conversion to the requested dtype.
    """

    if len(psfs) != 3:
        raise ValueError("exactly three g/r/z PSF objects are required")
    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel scale must be positive")
    if kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("PSF kernel size must be odd and at least three")
    arrays = []
    for band, psf in enumerate(psfs):
        draw = getattr(psf, "drawImage", None)
        if not callable(draw):
            raise TypeError(f"PSF {band} does not provide GalSim drawImage")
        image = draw(nx=kernel_size, ny=kernel_size, scale=pixel_scale_arcsec)
        array = np.asarray(image.array, dtype=np.float64)
        if array.shape != (kernel_size, kernel_size):
            raise RuntimeError("GalSim PSF draw returned an unexpected shape")
        if not np.isfinite(array).all() or float(array.sum()) <= 0:
            raise RuntimeError("GalSim PSF draw is not finite with positive mass")
        # FFT-drawn physical GalSim PSFs can contain tiny signed ringing.  The
        # frozen adapter permits a negative extremum below 1e-4 of the positive
        # peak and total negative mass below 5e-4 of positive mass, clips only
        # that numerical ringing, and then explicitly renormalizes.
        negative_tolerance = 1.0e-4 * float(np.max(array))
        negative_mass_fraction = float(-np.sum(array[array < 0]) / np.sum(array[array > 0]))
        if float(np.min(array)) < -negative_tolerance or negative_mass_fraction > 5.0e-4:
            raise RuntimeError("GalSim PSF draw has material negative mass")
        arrays.append(np.maximum(array, 0.0))
    return normalize_psf(torch.as_tensor(np.stack(arrays), dtype=dtype))
