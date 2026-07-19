from __future__ import annotations

import numpy as np
import pytest

from src.competing_hypotheses import (
    ForwardConsistency,
    PlausibilityThresholds,
    assert_black_box_feature_contract,
    calibrate_plausibility,
    empirical_ambiguity_witness,
    forward_consistency,
    is_plausible,
    poisson_variance,
    recompose,
    scientific_distance,
    source_measurements,
)


def test_recomposition_and_poisson_variance_follow_frozen_contract() -> None:
    layers = np.stack([np.full((3, 2, 2), 2.0), np.full((3, 2, 2), 3.0)])
    recomposed = recompose(layers)
    assert np.array_equal(recomposed, np.full((3, 2, 2), 5.0))
    variance = poisson_variance(recomposed, [10.0, 20.0, 30.0])
    assert np.array_equal(variance[:, 0, 0], [15.0, 25.0, 35.0])


def test_forward_consistency_is_truth_free_and_exact_for_zero_residual() -> None:
    layers = np.stack([np.full((3, 4, 4), 2.0), np.full((3, 4, 4), 3.0)])
    observed = layers.sum(axis=0)
    result = forward_consistency(observed, layers, [100.0, 100.0, 100.0])
    assert result.global_chi_square_mean == 0.0
    assert result.per_band_chi_square_mean == (0.0, 0.0, 0.0)
    assert result.relative_flux_residual == 0.0
    assert result.finite


def test_forward_consistency_rejects_nonfinite_candidate() -> None:
    layers = np.zeros((1, 3, 2, 2))
    layers[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        forward_consistency(np.zeros((3, 2, 2)), layers, [1.0, 1.0, 1.0])


def test_calibration_uses_conservative_higher_quantiles() -> None:
    results = [
        ForwardConsistency(float(i), (float(i), float(i + 1), float(i + 2)), (0.0, 0.0, 0.0), i / 100.0, True)
        for i in range(1, 101)
    ]
    thresholds = calibrate_plausibility(results)
    assert thresholds.global_chi_square_mean == 100.0
    assert thresholds.per_band_chi_square_mean == (100.0, 101.0, 102.0)
    assert thresholds.absolute_relative_flux_residual == 1.0
    assert is_plausible(results[-1], thresholds)


def test_source_measurement_centroid_and_distance() -> None:
    left = np.zeros((3, 7, 7))
    right = np.zeros_like(left)
    left[:, 3, 2] = [1.0, 2.0, 3.0]
    right[:, 3, 4] = [1.0, 2.0, 3.0]
    measurement = source_measurements(left)
    assert measurement.centroid_xy == pytest.approx((2.0, 3.0))
    distance = scientific_distance(left, right, mean_psf_fwhm_pixel=4.0)
    assert distance.centroid_pixel == pytest.approx(2.0)
    assert distance.centroid_psf == pytest.approx(0.5)
    assert distance.primary_normalized > 1.0


def test_color_not_applicable_for_nonpositive_flux() -> None:
    left = np.zeros((3, 2, 2))
    right = np.ones((3, 2, 2))
    distance = scientific_distance(left, right, mean_psf_fwhm_pixel=4.0)
    assert distance.color_gr_rz_magnitude == (None, None)


def test_empirical_witness_requires_two_plausible_candidates_and_artifact_pass() -> None:
    a = np.zeros((3, 7, 7))
    b = np.zeros_like(a)
    a[:, 3, 2] = 10.0
    b[:, 3, 4] = 10.0
    score = ForwardConsistency(0.2, (0.2, 0.2, 0.2), (0.0, 0.0, 0.0), 0.0, True)
    thresholds = PlausibilityThresholds(1.0, (1.0, 1.0, 1.0), 0.1, 10)
    witness = empirical_ambiguity_witness(
        {"a": a, "b": b},
        {"a": score, "b": score},
        thresholds,
        mean_psf_fwhm_pixel=4.0,
        artifact_audit_passed=True,
    )
    assert witness.exists
    assert witness.maximizing_pair == ("a", "b")
    failed_audit = empirical_ambiguity_witness(
        {"a": a, "b": b},
        {"a": score, "b": score},
        thresholds,
        mean_psf_fwhm_pixel=4.0,
        artifact_audit_passed=False,
    )
    assert not failed_audit.exists
    assert failed_audit.reason == "artifact_audit_failed"


@pytest.mark.parametrize(
    "field",
    ["target_image", "family_id", "checkpoint_name", "source_ids", "true_snr", "private_latent"],
)
def test_black_box_contract_rejects_oracle_and_provenance_fields(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_black_box_feature_contract({field: np.zeros(1)})


def test_black_box_contract_accepts_deployable_fields() -> None:
    assert_black_box_feature_contract(
        {
            "observed_blend": np.zeros((3, 2, 2)),
            "coordinate_prompt": np.zeros((1, 2, 2)),
            "candidate_reconstruction": np.zeros((3, 2, 2)),
            "forward_consistency_score": 1.0,
            "plausible_set_diameter": 0.5,
        }
    )
