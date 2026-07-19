"""Frozen PSF-B construction and authoritative paired BTK acquisition helpers."""

from __future__ import annotations

import copy
from hashlib import sha256
import math
from typing import Any

import galsim
import numpy as np
import torch
from astropy import units as u
from btk.draw_blends import CatsimGenerator
from btk.survey import get_surveys

from src.btk_scene import (
    BAND_ORDER,
    PIXEL_SCALE_ARCSEC,
    FixedSceneSampling,
    SceneRender,
    SceneSpec,
)
from src.model9_galsim_adapter import sample_galsim_psf_kernels


PSF_B_NOMINAL_FWHM_ARCSEC = {"g": 0.70, "r": 0.68, "z": 0.66}
PSF_B_ELLIPTICITY = 0.10
PSF_B_ORIENTATION_DEGREES = 30.0
PSF_KERNEL_SIZE = 31


def frozen_psf_b_survey():
    """Return an LSST survey copy whose only changed acquisition field is PSF."""

    survey = copy.deepcopy(get_surveys("LSST"))
    for band in BAND_ORDER:
        filt = survey.get_filter(band)
        original_fwhm = float(filt.psf_fwhm.to_value("arcsec"))
        target_fwhm = PSF_B_NOMINAL_FWHM_ARCSEC[band]
        original_psf = filt.psf(survey, filt) if callable(filt.psf) else filt.psf
        filt.psf = original_psf.dilate(target_fwhm / original_fwhm).shear(
            e=PSF_B_ELLIPTICITY,
            beta=PSF_B_ORIENTATION_DEGREES * galsim.degrees,
        )
        filt.psf_fwhm = target_fwhm * u.arcsec
    return survey


def frozen_psf_pair_kernels() -> tuple[torch.Tensor, torch.Tensor]:
    survey_a = get_surveys("LSST")
    survey_b = frozen_psf_b_survey()
    objects_a = []
    objects_b = []
    for band in BAND_ORDER:
        filt_a = survey_a.get_filter(band)
        filt_b = survey_b.get_filter(band)
        objects_a.append(filt_a.psf(survey_a, filt_a) if callable(filt_a.psf) else filt_a.psf)
        objects_b.append(filt_b.psf(survey_b, filt_b) if callable(filt_b.psf) else filt_b.psf)
    return (
        sample_galsim_psf_kernels(
            objects_a, pixel_scale_arcsec=PIXEL_SCALE_ARCSEC, kernel_size=PSF_KERNEL_SIZE
        ),
        sample_galsim_psf_kernels(
            objects_b, pixel_scale_arcsec=PIXEL_SCALE_ARCSEC, kernel_size=PSF_KERNEL_SIZE
        ),
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    digest = sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _kernel_moments(kernel: np.ndarray) -> dict[str, float]:
    value = np.asarray(kernel, dtype=np.float64)
    value = value / value.sum()
    yy, xx = np.indices(value.shape, dtype=np.float64)
    cx = float(np.sum(value * xx))
    cy = float(np.sum(value * yy))
    dx = xx - cx
    dy = yy - cy
    ixx = float(np.sum(value * dx * dx))
    iyy = float(np.sum(value * dy * dy))
    ixy = float(np.sum(value * dx * dy))
    trace = ixx + iyy
    e1 = (ixx - iyy) / trace
    e2 = 2.0 * ixy / trace
    orientation = 0.5 * math.degrees(math.atan2(e2, e1))
    sigma_equivalent_pixels = math.sqrt(trace / 2.0)
    return {
        "centroid_x_pixel": cx,
        "centroid_y_pixel": cy,
        "second_moment_ixx_pixel2": ixx,
        "second_moment_iyy_pixel2": iyy,
        "second_moment_ixy_pixel2": ixy,
        "equivalent_gaussian_fwhm_arcsec": 2.0
        * math.sqrt(2.0 * math.log(2.0))
        * sigma_equivalent_pixels
        * PIXEL_SCALE_ARCSEC,
        "moment_ellipticity": math.hypot(e1, e2),
        "moment_orientation_degrees": orientation,
    }


def psf_diversity_metrics(
    psf_a: torch.Tensor | np.ndarray,
    psf_b: torch.Tensor | np.ndarray,
) -> list[dict[str, Any]]:
    left = np.asarray(psf_a, dtype=np.float64)
    right = np.asarray(psf_b, dtype=np.float64)
    if left.shape != (3, PSF_KERNEL_SIZE, PSF_KERNEL_SIZE) or right.shape != left.shape:
        raise ValueError("frozen PSF pair must use 3x31x31 kernels")
    rows = []
    for index, band in enumerate(BAND_ORDER):
        a = left[index] / left[index].sum()
        b = right[index] / right[index].sum()
        fft_a = np.fft.fft2(np.fft.ifftshift(a))
        fft_b = np.fft.fft2(np.fft.ifftshift(b))
        a0 = a.reshape(-1) - float(np.mean(a))
        b0 = b.reshape(-1) - float(np.mean(b))
        correlation = float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0)))
        moment_a = _kernel_moments(a)
        moment_b = _kernel_moments(b)
        rows.append(
            {
                "band": band,
                "psf_a_sha256": array_sha256(a),
                "psf_b_sha256": array_sha256(b),
                "psf_a_sum": float(a.sum()),
                "psf_b_sum": float(b.sum()),
                "kernel_l2_distance": float(np.linalg.norm(a - b)),
                "kernel_relative_l2_distance": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
                "kernel_cross_correlation": correlation,
                "fourier_transfer_relative_l2_distance": float(
                    np.linalg.norm(fft_a - fft_b) / np.linalg.norm(fft_a)
                ),
                "psf_a_nominal_fwhm_arcsec": float(
                    get_surveys("LSST").get_filter(band).psf_fwhm.to_value("arcsec")
                ),
                "psf_b_nominal_fwhm_arcsec": PSF_B_NOMINAL_FWHM_ARCSEC[band],
                "nominal_fwhm_difference_arcsec": PSF_B_NOMINAL_FWHM_ARCSEC[band]
                - float(get_surveys("LSST").get_filter(band).psf_fwhm.to_value("arcsec")),
                "effective_resolution_ratio_b_over_a": moment_b[
                    "equivalent_gaussian_fwhm_arcsec"
                ]
                / moment_a["equivalent_gaussian_fwhm_arcsec"],
                **{f"psf_a_{key}": value for key, value in moment_a.items()},
                **{f"psf_b_{key}": value for key, value in moment_b.items()},
            }
        )
    return rows


def render_fixed_scene_with_survey(
    catalog,
    spec: SceneSpec,
    *,
    survey,
    add_noise: str,
) -> SceneRender:
    """Render through BTK's authoritative CatsimGenerator using a supplied survey."""

    if add_noise not in {"none", "all"}:
        raise ValueError("paired scenes allow only add_noise='none' or 'all'")
    sampler = FixedSceneSampling(spec.catalog_rows, spec.positions_arcsec)
    generator = CatsimGenerator(
        catalog,
        sampler,
        survey,
        batch_size=1,
        njobs=1,
        verbose=False,
        use_bar=False,
        add_noise=add_noise,
        seed=spec.noise_seed,
        apply_shear=False,
        augment_data=False,
    )
    batch = next(generator)
    full_bands = tuple(batch.survey.available_filters)
    band_indices = [full_bands.index(band) for band in BAND_ORDER]
    blend = np.asarray(batch.blend_images[0, band_indices], dtype=np.float64)
    # BTK always constructs isolated layers internally.  They are copied only
    # to satisfy SceneRender's simulator return type and must never enter the
    # solver or a persisted paired-observation artifact.
    isolated = np.asarray(
        batch.isolated_images[0, : len(spec.catalog_rows)][:, band_indices], dtype=np.float64
    )
    image_size = blend.shape[-1]
    scale = float(batch.survey.pixel_scale.to_value("arcsec"))
    psf = np.asarray(
        [
            batch.psf[index]
            .drawImage(nx=image_size, ny=image_size, scale=scale)
            .array.astype(np.float64)
            for index in band_indices
        ]
    )
    rendered_catalog = batch.catalog_list[0]
    if list(np.asarray(rendered_catalog["catalog_row"], dtype=int)) != list(spec.catalog_rows):
        raise RuntimeError("BTK output catalog rows do not match the fixed request")
    if blend.shape != (3, image_size, image_size) or not np.isfinite(blend).all():
        raise RuntimeError("invalid paired BTK blend")
    return SceneRender(
        blend=blend,
        isolated=isolated,
        psf=psf,
        catalog=rendered_catalog,
        bands=BAND_ORDER,
        full_survey_bands=full_bands,
        pixel_scale_arcsec=scale,
    )
