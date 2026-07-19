"""Pre-science tests for the frozen PSF-diverse acquisition definition."""

from __future__ import annotations

import numpy as np
from btk.survey import get_surveys

from src.psf_diverse_acquisition import (
    PSF_B_ELLIPTICITY,
    PSF_B_NOMINAL_FWHM_ARCSEC,
    PSF_B_ORIENTATION_DEGREES,
    frozen_psf_b_survey,
    frozen_psf_pair_kernels,
    psf_diversity_metrics,
)


def test_psf_b_changes_only_psf_fields_on_a_survey_copy() -> None:
    original = get_surveys("LSST")
    diverse = frozen_psf_b_survey()
    assert diverse is not original
    assert diverse.pixel_scale == original.pixel_scale
    for band in ("g", "r", "z"):
        left = original.get_filter(band)
        right = diverse.get_filter(band)
        assert float(right.psf_fwhm.to_value("arcsec")) == PSF_B_NOMINAL_FWHM_ARCSEC[band]
        assert right.zeropoint == left.zeropoint
        assert right.sky_brightness == left.sky_brightness
        assert right.full_exposure_time == left.full_exposure_time
        assert right.effective_wavelength == left.effective_wavelength


def test_psf_pair_is_normalized_distinct_and_not_unrealistically_sharp() -> None:
    left, right = frozen_psf_pair_kernels()
    assert left.shape == right.shape == (3, 31, 31)
    np.testing.assert_allclose(left.sum(dim=(-2, -1)).numpy(), 1.0, atol=2e-15)
    np.testing.assert_allclose(right.sum(dim=(-2, -1)).numpy(), 1.0, atol=2e-15)
    assert np.all(np.asarray(right) >= 0.0)
    assert np.linalg.norm(np.asarray(left) - np.asarray(right)) > 0.01
    assert min(PSF_B_NOMINAL_FWHM_ARCSEC.values()) >= 0.6
    assert PSF_B_ELLIPTICITY == 0.10
    assert PSF_B_ORIENTATION_DEGREES == 30.0


def test_psf_diversity_metrics_capture_resolution_and_operator_change() -> None:
    left, right = frozen_psf_pair_kernels()
    rows = psf_diversity_metrics(left, right)
    assert [row["band"] for row in rows] == ["g", "r", "z"]
    for row in rows:
        assert row["kernel_relative_l2_distance"] > 0.1
        assert row["fourier_transfer_relative_l2_distance"] > 0.1
        assert row["kernel_cross_correlation"] < 0.999
        assert row["effective_resolution_ratio_b_over_a"] < 1.0
        assert row["psf_a_sha256"] != row["psf_b_sha256"]
