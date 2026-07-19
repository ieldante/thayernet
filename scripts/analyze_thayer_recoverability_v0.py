#!/usr/bin/env python3
"""Training-free identifiability audit for the frozen Family-E1 micro scenes.

This script reads only the Family-E1 difficult/mixed-eight training rows and
their frozen manifests.  It never imports or constructs a reconstruction
model.  The numerical experiment works directly with two nonnegative source
images and the exact signed-noise residual already implied by the observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.competing_hypotheses import scientific_distance, source_measurements


UPSTREAM = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
FAMILY_E1 = REPO / "outputs/runs/thayer_family_e1_v0_20260714_214715"
FAMILY_E1P = REPO / "outputs/runs/thayer_family_e1p_v0_20260714_225228"
SELECTOR = FAMILY_E1 / "manifests/training_manifest.csv"
MANIFEST = UPSTREAM / "manifests/v2_r_training_scene_manifest.csv"
SCENES = UPSTREAM / "manifests/v2_r_training_scenes.h5"
DIFFICULT_MANIFEST = FAMILY_E1P / "manifests/difficult_one_scene_paired_scene_manifest.csv"
MIXED_MANIFEST = FAMILY_E1P / "manifests/mixed_eight_scene_paired_scene_manifest.csv"

INDICES = (0, 3, 5, 6, 18, 51, 73, 81)
DIFFICULT_INDEX = 6
PIXEL_SCALE_ARCSEC = 0.2
PSF_FWHM_ARCSEC = np.asarray([0.86, 0.81, 0.77], dtype=np.float64)
PSF_FWHM_PIXEL = PSF_FWHM_ARCSEC / PIXEL_SCALE_ARCSEC
MEAN_PSF_FWHM_PIXEL = float(np.mean(PSF_FWHM_PIXEL))
PIXELS_PER_SOURCE_PAIR = 3 * 60 * 60
SCIENTIFIC_DIAMETER_GATE = 1.0


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_csv_rows(path: Path, zero_based_rows: tuple[int, ...] | list[int] | np.ndarray) -> pd.DataFrame:
    """Materialize only the authorized data rows plus the CSV header."""

    authorized = {int(value) for value in zero_based_rows}
    return pd.read_csv(
        path,
        low_memory=False,
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in authorized,
    )


def colors(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    return np.asarray(
        [-2.5 * np.log10(flux[0] / flux[1]), -2.5 * np.log10(flux[1] / flux[2])],
        dtype=np.float64,
    )


def centroid(source: np.ndarray) -> np.ndarray:
    measurement = source_measurements(np.asarray(source, dtype=np.float64))
    if measurement.centroid_xy is None:
        raise RuntimeError("positive frozen source has no centroid")
    return np.asarray(measurement.centroid_xy, dtype=np.float64)


def prompt_margin(requested: np.ndarray, companion: np.ndarray, q_requested: np.ndarray, q_companion: np.ndarray) -> float:
    requested_centroid = centroid(requested)
    companion_centroid = centroid(companion)
    requested_margin = np.linalg.norm(requested_centroid - q_companion) - np.linalg.norm(requested_centroid - q_requested)
    companion_margin = np.linalg.norm(companion_centroid - q_requested) - np.linalg.norm(companion_centroid - q_companion)
    return float(min(requested_margin, companion_margin))


def transferred_pair(requested: np.ndarray, companion: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    requested_alt = requested + fraction * companion
    companion_alt = (1.0 - fraction) * companion
    return requested_alt, companion_alt


def maximum_prompt_preserving_transfer(
    requested: np.ndarray,
    companion: np.ndarray,
    q_requested: np.ndarray,
    q_companion: np.ndarray,
) -> float:
    if prompt_margin(requested, companion, q_requested, q_companion) <= 0:
        return 0.0
    upper = 1.0 - 1e-9
    upper_pair = transferred_pair(requested, companion, upper)
    if prompt_margin(*upper_pair, q_requested, q_companion) > 0:
        return upper
    lower = 0.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        pair = transferred_pair(requested, companion, middle)
        if prompt_margin(*pair, q_requested, q_companion) > 0:
            lower = middle
        else:
            upper = middle
    return lower


def diameter_at_transfer(requested: np.ndarray, companion: np.ndarray, fraction: float) -> float:
    requested_alt, _ = transferred_pair(requested, companion, fraction)
    return float(
        scientific_distance(
            requested,
            requested_alt,
            mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
        ).primary_normalized
    )


def minimum_scientifically_distinct_transfer(
    requested: np.ndarray,
    companion: np.ndarray,
    maximum_fraction: float,
) -> float | None:
    if maximum_fraction <= 0 or diameter_at_transfer(requested, companion, maximum_fraction) <= SCIENTIFIC_DIAMETER_GATE:
        return None
    lower = 0.0
    upper = maximum_fraction
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if diameter_at_transfer(requested, companion, middle) > SCIENTIFIC_DIAMETER_GATE:
            upper = middle
        else:
            lower = middle
    return upper


def view_witness(
    requested: np.ndarray,
    companion: np.ndarray,
    observed: np.ndarray,
    q_requested: np.ndarray,
    q_companion: np.ndarray,
) -> dict[str, float | bool | None]:
    requested = np.asarray(requested, dtype=np.float64)
    companion = np.asarray(companion, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    signed_noise = observed - requested - companion
    maximum_fraction = maximum_prompt_preserving_transfer(requested, companion, q_requested, q_companion)
    witness_fraction = 0.999999 * maximum_fraction
    requested_alt, companion_alt = transferred_pair(requested, companion, witness_fraction)
    reconstructed = requested_alt + companion_alt + signed_noise
    relative_objective = float(
        np.sum(np.square(reconstructed - observed))
        / max(float(np.sum(np.square(observed))), np.finfo(np.float64).tiny)
    )
    distance = scientific_distance(
        requested,
        requested_alt,
        mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
    )
    critical_fraction = minimum_scientifically_distinct_transfer(requested, companion, maximum_fraction)

    # Direct output-space multistart experiment.  Allocation is the optimized
    # coordinate; every value on the interval is an exact data-fit minimizer.
    rng = np.random.default_rng(2026071500)
    starts = np.concatenate(([0.0, witness_fraction], rng.uniform(0.0, witness_fraction, size=30)))
    objectives = []
    for fraction in starts:
        left, right = transferred_pair(requested, companion, float(fraction))
        trial = left + right + signed_noise
        objectives.append(
            np.sum(np.square(trial - observed))
            / max(float(np.sum(np.square(observed))), np.finfo(np.float64).tiny)
        )

    # An arbitrarily small sign-changing tilt selects opposite endpoints of
    # this flat interval.  The base data objective remains unchanged.
    tilt = 1e-12
    positive_tilt_solution = 0.0
    negative_tilt_solution = witness_fraction
    perturbation_jump = diameter_at_transfer(requested, companion, negative_tilt_solution)

    return {
        "truth_prompt_margin_pixel": prompt_margin(requested, companion, q_requested, q_companion),
        "maximum_prompt_preserving_transfer": maximum_fraction,
        "minimum_transfer_for_scientific_difference": critical_fraction,
        "witness_fraction": witness_fraction,
        "witness_prompt_margin_pixel": prompt_margin(requested_alt, companion_alt, q_requested, q_companion),
        "witness_primary_scientific_diameter": float(distance.primary_normalized),
        "witness_image_distance": float(distance.image),
        "witness_max_relative_flux_distance": float(max(distance.relative_flux_grz)),
        "witness_max_color_distance_mag": float(max(value for value in distance.color_gr_rz_magnitude if value is not None)),
        "witness_centroid_distance_psf": float(distance.centroid_psf or 0.0),
        "witness_relative_data_objective": relative_objective,
        "multistart_count": int(len(starts)),
        "multistart_max_relative_data_objective": float(max(objectives)),
        "tilt_magnitude": tilt,
        "tilt_endpoint_scientific_jump": perturbation_jump,
        "scientifically_distinct_exact_witness": bool(distance.primary_normalized > SCIENTIFIC_DIAMETER_GATE),
        "positive_tilt_solution": positive_tilt_solution,
        "negative_tilt_solution": negative_tilt_solution,
    }


def direct_output_multistart(
    source_a: np.ndarray,
    source_b: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float | int]:
    """Run one exact projected-gradient step from 32 output-space starts."""

    source_a = np.asarray(source_a, dtype=np.float64)
    source_b = np.asarray(source_b, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    total = source_a + source_b
    signed_noise = observed - total
    denominator = max(float(np.sum(np.square(observed))), np.finfo(np.float64).tiny)
    rng = np.random.default_rng(2026071501)
    pre_objectives = []
    post_objectives = []
    endpoint_diameters = []
    for _ in range(32):
        allocation_a = rng.uniform(0.25, 0.75, size=total.shape)
        allocation_b = rng.uniform(0.25, 0.75, size=total.shape)
        left = allocation_a * total
        right = allocation_b * total
        pre_residual = left + right + signed_noise - observed
        pre_objectives.append(float(np.sum(np.square(pre_residual)) / denominator))

        # Gradient of 0.5*||left+right-total||^2 is the same for both
        # layers.  Step size 0.5 projects the sum exactly onto total.  The
        # bounded starts make both projected layers nonnegative.
        gradient = left + right - total
        left = left - 0.5 * gradient
        right = right - 0.5 * gradient
        if float(left.min()) < 0 or float(right.min()) < 0:
            raise RuntimeError("direct output-space projection left the nonnegative cone")
        post_residual = left + right + signed_noise - observed
        post_objectives.append(float(np.sum(np.square(post_residual)) / denominator))
        endpoint_diameters.append(
            float(
                scientific_distance(
                    source_a,
                    left,
                    mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                ).primary_normalized
            )
        )
    return {
        "starts": 32,
        "gradient_steps_per_start": 1,
        "maximum_pre_projection_relative_objective": float(max(pre_objectives)),
        "maximum_post_projection_relative_objective": float(max(post_objectives)),
        "minimum_post_projection_relative_objective": float(min(post_objectives)),
        "minimum_endpoint_truth_diameter": float(min(endpoint_diameters)),
        "maximum_endpoint_truth_diameter": float(max(endpoint_diameters)),
    }


def fisher_diagnostics(observed: np.ndarray, source_a: np.ndarray, source_b: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=np.float64)
    source_a = np.asarray(source_a, dtype=np.float64)
    source_b = np.asarray(source_b, dtype=np.float64)
    residual = observed - source_a - source_b
    median = np.median(residual, axis=(1, 2), keepdims=True)
    sigma = 1.4826 * np.median(np.abs(residual - median), axis=(1, 2))
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise RuntimeError("invalid robust noise approximation")
    jacobian = np.stack(
        (
            (source_a / sigma[:, None, None]).ravel(),
            (source_b / sigma[:, None, None]).ravel(),
        ),
        axis=1,
    )
    fisher = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(fisher)
    singular_values = np.sqrt(eigenvalues)
    rank = int(np.linalg.matrix_rank(jacobian))
    inverse = np.linalg.inv(fisher)
    norms = np.sqrt(np.diag(fisher))
    cosine = float(fisher[0, 1] / (norms[0] * norms[1]))
    normalized_hessian_condition = float((1.0 + abs(cosine)) / (1.0 - abs(cosine)))
    return {
        "noise_sigma_g": float(sigma[0]),
        "noise_sigma_r": float(sigma[1]),
        "noise_sigma_z": float(sigma[2]),
        "fisher_lambda_min": float(eigenvalues[0]),
        "fisher_lambda_max": float(eigenvalues[1]),
        "jacobian_rank": rank,
        "jacobian_condition": float(singular_values[-1] / singular_values[0]),
        "hessian_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "oracle_null_dimension": int(2 - rank),
        "noise_weighted_template_cosine": cosine,
        "shape_only_hessian_condition": normalized_hessian_condition,
        "crlb_std_amplitude_a": float(np.sqrt(inverse[0, 0])),
        "crlb_std_amplitude_b": float(np.sqrt(inverse[1, 1])),
        "worst_direction_crlb_std": float(1.0 / singular_values[0]),
    }


def scene_metrics(source_a: np.ndarray, source_b: np.ndarray, separation_pixel: float) -> dict[str, float]:
    a = np.asarray(source_a, dtype=np.float64)
    b = np.asarray(source_b, dtype=np.float64)
    flux_a = a.sum(axis=(1, 2), dtype=np.float64)
    flux_b = b.sum(axis=(1, 2), dtype=np.float64)
    total_a = float(flux_a.sum())
    total_b = float(flux_b.sum())
    overlap = float(np.minimum(a, b).sum() / min(total_a, total_b))
    symmetric_flux_ratio = float(min(total_a, total_b) / max(total_a, total_b))
    color_cosine = float(np.dot(flux_a, flux_b) / (np.linalg.norm(flux_a) * np.linalg.norm(flux_b)))
    color_distance = float(np.linalg.norm(colors(flux_a) - colors(flux_b)))
    psf_sigma = PSF_FWHM_PIXEL / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    per_band_psf_overlap = np.exp(-separation_pixel**2 / (4.0 * np.square(psf_sigma)))
    psf_overlap = float(np.mean(per_band_psf_overlap))
    ingredients = np.asarray([overlap, symmetric_flux_ratio, max(color_cosine, 0.0), psf_overlap])
    ambiguity_score = float(np.prod(np.maximum(ingredients, np.finfo(np.float64).tiny)) ** 0.25)
    morphology_cosine = float(np.vdot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {
        "overlap_fraction": overlap,
        "symmetric_flux_ratio": symmetric_flux_ratio,
        "color_similarity_cosine": color_cosine,
        "color_distance_mag": color_distance,
        "centroid_separation_pixel": separation_pixel,
        "centroid_separation_psf": float(separation_pixel / MEAN_PSF_FWHM_PIXEL),
        "psf_overlap": psf_overlap,
        "morphology_cosine": morphology_cosine,
        "ambiguity_score": ambiguity_score,
    }


def main() -> None:
    selector = read_csv_rows(SELECTOR, INDICES).set_index("family_e_index")
    paired_manifest = pd.concat(
        (pd.read_csv(DIFFICULT_MANIFEST), pd.read_csv(MIXED_MANIFEST)),
        ignore_index=True,
    ).drop_duplicates(subset=["family_e1_index"], keep="last")
    paired_manifest = paired_manifest.set_index("family_e1_index")
    upstream_indices = selector.loc[list(INDICES)].upstream_index.to_numpy(dtype=np.int64)
    if not np.array_equal(upstream_indices, paired_manifest.loc[list(INDICES)].upstream_index.to_numpy(dtype=np.int64)):
        raise RuntimeError("Family-E1 and Family-E1P frozen upstream indices disagree")
    if not np.array_equal(upstream_indices, np.asarray(INDICES, dtype=np.int64)):
        raise RuntimeError("Family-E1 selector no longer maps the frozen indices identically")
    selected = read_csv_rows(MANIFEST, upstream_indices).set_index("dataset_index").loc[list(upstream_indices)].reset_index()
    with h5py.File(SCENES, "r") as handle:
        observed = np.asarray(handle["blend"][upstream_indices], dtype=np.float32)
        isolated = np.asarray(handle["isolated"][upstream_indices], dtype=np.float32)
        xy = np.asarray(handle["xy"][upstream_indices], dtype=np.float64)

    results = []
    for local, family_index in enumerate(INDICES):
        row = selected.iloc[local]
        e1p = paired_manifest.loc[family_index]
        if sha256_array(observed[local]) != str(e1p.observation_sha256):
            raise RuntimeError(f"observation hash mismatch at Family-E1 index {family_index}")
        if sha256_array(isolated[local, 0]) != str(row.isolated_source_a_sha256):
            raise RuntimeError(f"source-A hash mismatch at Family-E1 index {family_index}")
        if sha256_array(isolated[local, 1]) != str(row.isolated_source_b_sha256):
            raise RuntimeError(f"source-B hash mismatch at Family-E1 index {family_index}")

        separation = float(np.linalg.norm(xy[local, 0] - xy[local, 1]))
        metrics = scene_metrics(isolated[local, 0], isolated[local, 1], separation)
        fisher = fisher_diagnostics(observed[local], isolated[local, 0], isolated[local, 1])
        matched = int(row.matched_source_index)
        alternate = 1 - matched
        q_a = np.asarray([float(e1p.prompt_a_x), float(e1p.prompt_a_y)])
        q_b = np.asarray([float(e1p.prompt_b_x), float(e1p.prompt_b_y)])
        view_a = view_witness(isolated[local, matched], isolated[local, alternate], observed[local], q_a, q_b)
        view_b = view_witness(isolated[local, alternate], isolated[local, matched], observed[local], q_b, q_a)
        output_multistart = direct_output_multistart(isolated[local, 0], isolated[local, 1], observed[local])
        both_witness = bool(view_a["scientifically_distinct_exact_witness"] and view_b["scientifically_distinct_exact_witness"])
        any_witness = bool(view_a["scientifically_distinct_exact_witness"] or view_b["scientifically_distinct_exact_witness"])
        if any_witness:
            classification = "FUNDAMENTALLY_UNIDENTIFIABLE"
        else:
            # The exact allocation set is still non-singleton.  With no
            # scientifically distinct witness it is ambiguous only below the
            # inherited scientific-resolution boundary.
            classification = "AMBIGUOUS"
        results.append(
            {
                "family_e1_index": family_index,
                "scene_id": str(row.scene_id),
                "condition_membership": "difficult+mixed-eight" if family_index == DIFFICULT_INDEX else "mixed-eight",
                "matched_source_index": matched,
                "source_a_id": str(row.source_a_id),
                "source_b_id": str(row.source_b_id),
                "observation_sha256": str(e1p.observation_sha256),
                "manifest_core_obstruction": float(row.core_obstruction),
                "manifest_flux_ratio_requested_to_companion": float(row.flux_ratio),
                **metrics,
                "oracle_template_fisher": fisher,
                "prompt_view_a": view_a,
                "prompt_view_b": view_b,
                "prompt_view_a_classification": "FUNDAMENTALLY_UNIDENTIFIABLE" if view_a["scientifically_distinct_exact_witness"] else "AMBIGUOUS",
                "prompt_view_b_classification": "FUNDAMENTALLY_UNIDENTIFIABLE" if view_b["scientifically_distinct_exact_witness"] else "AMBIGUOUS",
                "both_prompt_views_have_scientifically_distinct_witnesses": both_witness,
                "direct_output_multistart": output_multistart,
                "minimum_view_witness_diameter": float(min(view_a["witness_primary_scientific_diameter"], view_b["witness_primary_scientific_diameter"])),
                "maximum_view_witness_diameter": float(max(view_a["witness_primary_scientific_diameter"], view_b["witness_primary_scientific_diameter"])),
                "classification": classification,
            }
        )

    output = {
        "campaign": "Thayer-Recoverability-v0",
        "scope": {
            "unique_observations": len(INDICES),
            "difficult_condition_entries": 1,
            "mixed_eight_condition_entries": 8,
            "duplicated_index": DIFFICULT_INDEX,
            "accessed_family_e1_indices": list(INDICES),
            "development_access_count": 0,
            "atlas_access_count": 0,
            "lockbox_access_count": 0,
            "model_import_count": 0,
            "model_construction_count": 0,
            "model_optimizer_step_count": 0,
            "materialized_training_selector_rows": len(INDICES),
            "materialized_training_manifest_rows": len(INDICES),
        },
        "full_output_space": {
            "source_pixels_per_scene": PIXELS_PER_SOURCE_PAIR,
            "two_source_parameter_dimension": 2 * PIXELS_PER_SOURCE_PAIR,
            "two_source_observation_jacobian_rank": PIXELS_PER_SOURCE_PAIR,
            "two_source_observation_jacobian_null_dimension": PIXELS_PER_SOURCE_PAIR,
            "two_source_observation_jacobian_condition": "infinity",
            "two_source_data_hessian_rank": PIXELS_PER_SOURCE_PAIR,
            "two_source_data_hessian_null_dimension": PIXELS_PER_SOURCE_PAIR,
            "two_source_data_hessian_condition": "infinity",
            "family_e1_source_plus_residual_parameter_dimension": 3 * PIXELS_PER_SOURCE_PAIR,
            "family_e1_source_plus_residual_jacobian_rank": PIXELS_PER_SOURCE_PAIR,
            "family_e1_source_plus_residual_null_dimension": 2 * PIXELS_PER_SOURCE_PAIR,
            "family_e1_source_plus_residual_condition": "infinity",
            "fixed_residual_exact_solution_set": "(S_A + delta, S_B - delta), with -S_A <= delta <= S_B elementwise",
            "fixed_residual_solution_set_local_dimension": PIXELS_PER_SOURCE_PAIR,
            "sensitivity_ratio_along_null_space": "infinity",
        },
        "classification_rule": {
            "source": "unchanged empirical ambiguity certificate",
            "scientifically_distinct_primary_diameter": ">1.0",
            "fundamentally_unidentifiable": "the source-pair decomposition has at least one prompt-consistent exact-data-fit component alternative with primary diameter >1.0",
            "ambiguous": "the exact source-pair solution set is non-singleton but no tested prompt view has a >1.0 certificate",
            "thresholds_changed": False,
        },
        "scenes": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
