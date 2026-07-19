#!/usr/bin/env python3
"""Apply prior-gap, forward-consistency, and control-concentration gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from surveycodex.utilities import mean_sky_level


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import (  # noqa: E402
    BATCH_SIZE,
    K,
    PRIOR_EPSILON_SEED,
    load_model,
    prompts,
    read_csv,
    require_mps,
    sample_outputs,
    validation_arrays,
)
from scripts.thayer_select_prompt_ablation_common import CompactSelectNet  # noqa: E402
from src.btk_scene import BAND_ORDER, validated_lsst_survey  # noqa: E402
from src.canonical_tensor_hash import canonical_tensor_sha256  # noqa: E402


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
BOOTSTRAP_SEED = 2026078301
BOOTSTRAP_REPLICATES = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_run(path: Path, phase: str) -> Path:
    run_dir = path.resolve()
    promptability = json.loads((run_dir / "logs/pre_atlas_promptability_complete.json").read_text())
    if promptability["status"] != "PASS" or not promptability["next_gate_authorized"]:
        raise RuntimeError("promptability blocks subsequent gates")
    prerequisites = {
        "forward": run_dir / "logs/prior_posterior_gap_complete.json",
        "control": run_dir / "logs/forward_consistency_gate_complete.json",
    }
    if phase in prerequisites:
        record = json.loads(prerequisites[phase].read_text())
        required_key = "forward_consistency_authorized" if phase == "forward" else "control_concentration_authorized"
        if record["status"] != "PASS" or not record[required_key]:
            raise RuntimeError(f"previous gate blocks {phase}")
    return run_dir


def prior_gap(run_dir: Path) -> None:
    summary = read_csv(run_dir / "tables/pre_atlas_promptability_summary.csv")[0]
    mse_ratio = float(summary["prior_best_to_posterior_mse_ratio_diagnostic"])
    identity_gap = float(summary["posterior_minus_prior_identity_gap_diagnostic"])
    gates = [
        {"gate": "prior_best_of_16_to_posterior_mse", "threshold": "<=2.0", "observed": mse_ratio, "pass": mse_ratio <= 2.0},
        {"gate": "posterior_minus_prior_identity", "threshold": "<=0.15", "observed": identity_gap, "pass": identity_gap <= 0.15},
    ]
    passed = all(bool(row["pass"]) for row in gates)
    write_csv_fresh(run_dir / "tables/prior_posterior_gap_gates.csv", gates)
    report = f"""# Prior versus posterior gap

Status: **{'PASS' if passed else 'FAIL — FORWARD CONSISTENCY AND ATLAS BLOCKED'}**.

- Posterior requested MSE: {float(summary['mean_posterior_requested_mse_normalized']):.6g}.
- Prior best-of-16 requested MSE: {float(summary['prior_best_of_16_requested_mse_normalized']):.6g}.
- Prior-best/posterior MSE ratio: {mse_ratio:.6f} (gate <=2.0).
- Posterior identity success: {float(summary['posterior_identity_success']):.6f}.
- Individual prior identity success: {float(summary['individual_prior_sample_requested_success']):.6f}.
- Posterior-minus-prior identity gap: {identity_gap:.6f} (gate <=0.15).

Posterior samples remain training diagnostics only. This pass is based on inference-time
prior performance rather than posterior reconstruction quality.
"""
    write_text_fresh(run_dir / "diagnostics/prior_posterior_gap_report.md", report)
    write_json_fresh(run_dir / "logs/prior_posterior_gap_complete.json", {
        "status": "PASS" if passed else "FAIL", "forward_consistency_authorized": passed,
        "prior_best_to_posterior_mse_ratio": mse_ratio, "posterior_minus_prior_identity_gap": identity_gap,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "gates": gates}, sort_keys=True))


def score_candidates(observed: np.ndarray, layers: np.ndarray, sky: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # observed (N,3,H,W), layers (N,K,6,H,W)
    recomposed = layers[:, :, :3] + layers[:, :, 3:]
    variance = np.maximum(recomposed + sky[None, None, :, None, None], 1.0)
    residual = observed[:, None] - recomposed
    squared = residual.square() if isinstance(residual, torch.Tensor) else residual**2
    whitened_squared = squared / variance
    global_score = whitened_squared.mean(axis=(2, 3, 4))
    band_score = whitened_squared.mean(axis=(3, 4))
    denominator = np.maximum(np.abs(observed).sum(axis=(1, 2, 3)), np.finfo(np.float64).eps)
    flux_score = residual.sum(axis=(2, 3, 4)) / denominator[:, None]
    return global_score, band_score, flux_score


def higher(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


def calibration_thresholds(run_dir: Path, device: torch.device, scales: np.ndarray, sky: np.ndarray) -> dict[str, object]:
    definitions = [
        row for row in read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
        if row["partition"] == "calibration"
    ]
    with h5py.File(run_dir / "manifests/probabilistic_unet_calibration_scenes.h5", "r") as handle:
        observed = np.asarray(handle["blend"], dtype=np.float64)
        isolated = np.asarray(handle["isolated"], dtype=np.float64)
        xy = np.asarray(handle["xy"], dtype=np.float64)
    truth_layers = isolated.reshape(len(isolated), 1, 6, 60, 60)
    truth_global, truth_band, truth_flux = score_candidates(observed, truth_layers, sky)
    payload = torch.load(CONDITION_C, map_location="cpu", weights_only=False)
    condition = CompactSelectNet(4).to(device)
    condition.load_state_dict(payload["state_dict"], strict=True)
    condition.eval()
    prompt_a, prompt_b = prompts(xy)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(observed), 32):
            stop = min(start + 32, len(observed))
            blend = torch.from_numpy(np.ascontiguousarray((observed[start:stop] / scales[None, :, None, None]).astype(np.float32))).to(device)
            pa = torch.from_numpy(np.ascontiguousarray(prompt_a[start:stop])).to(device)
            pb = torch.from_numpy(np.ascontiguousarray(prompt_b[start:stop])).to(device)
            result = condition(torch.cat((torch.cat((blend, pa), dim=1), torch.cat((blend, pb), dim=1))))
            a, b = result.chunk(2)
            outputs.append(torch.cat((a, b), dim=1).cpu().numpy() * np.tile(scales, 2)[None, :, None, None])
    condition_layers = np.concatenate(outputs)[:, None].astype(np.float64)
    condition_global, condition_band, condition_flux = score_candidates(observed, condition_layers, sky)
    thresholds = {
        "global_chi_square_mean": max(higher(truth_global.ravel(), 0.99), higher(condition_global.ravel(), 0.95)),
        "per_band_chi_square_mean": [
            max(higher(truth_band[:, 0, band], 0.995), higher(condition_band[:, 0, band], 0.95))
            for band in range(3)
        ],
        "absolute_relative_flux_residual": max(higher(np.abs(truth_flux).ravel(), 0.99), higher(np.abs(condition_flux).ravel(), 0.95)),
        "truth_global_99": higher(truth_global.ravel(), 0.99),
        "condition_c_global_95": higher(condition_global.ravel(), 0.95),
        "truth_per_band_995": [higher(truth_band[:, 0, band], 0.995) for band in range(3)],
        "condition_c_per_band_95": [higher(condition_band[:, 0, band], 0.95) for band in range(3)],
        "truth_absolute_flux_99": higher(np.abs(truth_flux).ravel(), 0.99),
        "condition_c_absolute_flux_95": higher(np.abs(condition_flux).ravel(), 0.95),
        "calibration_scene_count": len(definitions),
        "selection": "componentwise max of truth-only high quantile and Condition-C 95th percentile",
        "validation_or_atlas_used": False,
    }
    return thresholds


def plausible(global_score: np.ndarray, band_score: np.ndarray, flux_score: np.ndarray, thresholds: dict[str, object]) -> np.ndarray:
    return (
        (global_score <= float(thresholds["global_chi_square_mean"]))
        & np.all(band_score <= np.asarray(thresholds["per_band_chi_square_mean"])[None, None, :], axis=2)
        & (np.abs(flux_score) <= float(thresholds["absolute_relative_flux_residual"]))
    )


def forward_consistency_gate(run_dir: Path) -> None:
    device = require_mps()
    model = load_model(run_dir, device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    thresholds = calibration_thresholds(run_dir, device, scales, sky)
    write_json_fresh(run_dir / "manifests/forward_consistency_thresholds.json", thresholds)
    rows, observed_physical, isolated_physical, xy = validation_arrays(run_dir)
    blend_normalized = observed_physical / scales[None, :, None, None]
    prompt_a_all, _ = prompts(xy)
    rng = np.random.default_rng(PRIOR_EPSILON_SEED)
    sample_path = run_dir / "prior_samples/non_atlas_validation_k16.h5"
    sample_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    all_plausible = []
    started = time.time()
    with h5py.File(sample_path, "x") as sample_file:
        dataset = sample_file.create_dataset(
            "decomposition", shape=(len(rows), K, 6, 60, 60), dtype="f4",
            chunks=(1, 1, 6, 60, 60), compression="lzf",
        )
        sample_file.attrs["complete"] = False
        sample_file.attrs["prior_epsilon_seed"] = PRIOR_EPSILON_SEED
        sample_file.attrs["checkpoint_sha256"] = sha256_file(run_dir / "checkpoints/thayer_pu_best.pth")
        with torch.no_grad():
            for start in range(0, len(rows), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(rows))
                blend = torch.from_numpy(np.ascontiguousarray(blend_normalized[start:stop])).to(device)
                prompt_a = torch.from_numpy(np.ascontiguousarray(prompt_a_all[start:stop])).to(device)
                mean, log_variance = model.encode_prior(blend)
                epsilon = torch.from_numpy(rng.standard_normal((len(blend), K, 8)).astype(np.float32)).to(device)
                output = sample_outputs(model, blend, prompt_a, mean, log_variance, epsilon).cpu().numpy()
                output_physical = output * np.tile(scales, 2)[None, None, :, None, None]
                dataset[start:stop] = output_physical.astype(np.float32)
                global_score, band_score, flux_score = score_candidates(
                    observed_physical[start:stop].astype(np.float64), output_physical.astype(np.float64), sky
                )
                pass_mask = plausible(global_score, band_score, flux_score, thresholds)
                all_plausible.extend(pass_mask.ravel().tolist())
                for local in range(stop - start):
                    index = start + local
                    count = int(pass_mask[local].sum())
                    scene_rows.append({
                        "scene_id": rows[index]["scene_id"], "kind": rows[index]["kind"],
                        "near_collision_pair_id": rows[index]["near_collision_pair_id"],
                        "plausible_sample_count": count, "plausible_fraction": count / K,
                        "at_least_one_plausible": count >= 1,
                    })
                    for sample_index in range(K):
                        sample_rows.append({
                            "scene_id": rows[index]["scene_id"], "kind": rows[index]["kind"],
                            "sample_index": sample_index, "global_chi_square_mean": float(global_score[local, sample_index]),
                            "band_g_chi_square_mean": float(band_score[local, sample_index, 0]),
                            "band_r_chi_square_mean": float(band_score[local, sample_index, 1]),
                            "band_z_chi_square_mean": float(band_score[local, sample_index, 2]),
                            "relative_flux_residual": float(flux_score[local, sample_index]),
                            "plausible": bool(pass_mask[local, sample_index]),
                            "decomposition_sha256": canonical_tensor_sha256(output_physical[local, sample_index]),
                        })
                sample_file.attrs["completed_count"] = stop
                print(json.dumps({"phase": "forward", "completed": stop, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
        sample_file.attrs["complete"] = True
    overall_rate = float(np.mean(all_plausible))
    median_count = float(np.median([row["plausible_sample_count"] for row in scene_rows]))
    at_least_one_rate = float(np.mean([row["at_least_one_plausible"] for row in scene_rows]))
    gates = [
        {"gate": "overall_prior_sample_plausibility", "threshold": ">=0.50", "observed": overall_rate, "pass": overall_rate >= 0.50},
        {"gate": "median_plausible_samples", "threshold": ">=4 of 16", "observed": median_count, "pass": median_count >= 4},
        {"gate": "scenes_with_any_plausible", "threshold": ">=0.75", "observed": at_least_one_rate, "pass": at_least_one_rate >= 0.75},
    ]
    passed = all(bool(row["pass"]) for row in gates)
    write_csv_fresh(run_dir / "tables/forward_consistency_prior_samples.csv", sample_rows)
    write_csv_fresh(run_dir / "tables/forward_consistency_per_scene.csv", scene_rows)
    write_csv_fresh(run_dir / "tables/forward_consistency_gates.csv", gates)
    report = f"""# Non-Atlas prior forward-consistency gate

Status: **{'PASS' if passed else 'FAIL — CONTROL CONCENTRATION AND ATLAS BLOCKED'}**.

- Calibration-only thresholds: global {thresholds['global_chi_square_mean']:.6g};
  g/r/z {', '.join(f'{value:.6g}' for value in thresholds['per_band_chi_square_mean'])};
  absolute relative flux {thresholds['absolute_relative_flux_residual']:.6g}.
- Plausible prior samples: {overall_rate:.6f} (gate >=0.50).
- Median plausible samples per K=16 scene: {median_count:.1f} (gate >=4).
- Scenes with at least one plausible sample: {at_least_one_rate:.6f} (gate >=0.75).

Thresholds use calibration only. Every prior decomposition was recomposed without
clipping and compared with the observed blend under the exact frozen source-plus-sky
Poisson variance contract. Forward-inconsistent diversity is not retained.
"""
    write_text_fresh(run_dir / "diagnostics/forward_consistency_gate_report.md", report)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ordinary = [row["plausible_sample_count"] for row in scene_rows if row["kind"] == "ordinary"]
    near = [row["plausible_sample_count"] for row in scene_rows if row["kind"] == "near_collision"]
    bins = np.arange(-0.5, 17.5, 1)
    axis.hist(ordinary, bins=bins, alpha=0.6, label="ordinary", density=True)
    axis.hist(near, bins=bins, alpha=0.6, label="near collision", density=True)
    axis.set(xlabel="plausible samples of 16", ylabel="density", title="Forward-consistent prior hypotheses")
    axis.legend(); axis.grid(alpha=0.25)
    figure.savefig(run_dir / "figures/forward_consistency_plausible_counts.png", dpi=170)
    plt.close(figure)
    write_json_fresh(run_dir / "logs/forward_consistency_gate_complete.json", {
        "status": "PASS" if passed else "FAIL", "control_concentration_authorized": passed,
        "overall_plausibility_rate": overall_rate, "median_plausible_samples": median_count,
        "scenes_with_any_plausible": at_least_one_rate, "thresholds_sha256": sha256_file(run_dir / "manifests/forward_consistency_thresholds.json"),
        "prior_samples_sha256": sha256_file(sample_path), "runtime_seconds": time.time() - started,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "gates": gates}, sort_keys=True))


def pairwise_scientific(samples: np.ndarray, mean_psf_pixel: float) -> tuple[np.ndarray, dict[str, float]]:
    count = len(samples)
    matrix = np.zeros((count, count), dtype=np.float64)
    if count < 2:
        return matrix, {"image": 0.0, "flux": 0.0, "color": 0.0, "centroid_psf": 0.0, "primary": 0.0}
    norms = np.linalg.norm(samples.reshape(count, -1), axis=1)
    flux = samples.sum(axis=(-2, -1), dtype=np.float64)
    colors = np.full((count, 2), np.nan)
    valid_gr = (flux[:, 0] > 0) & (flux[:, 1] > 0)
    valid_rz = (flux[:, 1] > 0) & (flux[:, 2] > 0)
    colors[valid_gr, 0] = -2.5 * np.log10(flux[valid_gr, 0] / flux[valid_gr, 1])
    colors[valid_rz, 1] = -2.5 * np.log10(flux[valid_rz, 1] / flux[valid_rz, 2])
    weights = np.maximum(samples.sum(axis=1), 0.0)
    yy, xx = np.indices(samples.shape[-2:], dtype=np.float64)
    total = weights.sum(axis=(1, 2))
    centroids = np.full((count, 2), np.nan)
    valid_centroid = total > np.finfo(np.float64).eps
    centroids[valid_centroid, 0] = (weights[valid_centroid] * xx).sum(axis=(1, 2)) / total[valid_centroid]
    centroids[valid_centroid, 1] = (weights[valid_centroid] * yy).sum(axis=(1, 2)) / total[valid_centroid]
    maxima = {"image": 0.0, "flux": 0.0, "color": 0.0, "centroid_psf": 0.0, "primary": 0.0}
    for left in range(count):
        for right in range(left + 1, count):
            image = float(np.linalg.norm(samples[left] - samples[right]) / (0.5 * (norms[left] + norms[right]) + 1e-12))
            relative_flux = np.abs(flux[left] - flux[right]) / (np.abs(0.5 * (flux[left] + flux[right])) + 1e-12)
            color = np.abs(colors[left] - colors[right])
            color_max = float(np.nanmax(color)) if np.any(np.isfinite(color)) else 0.0
            centroid = float(np.linalg.norm(centroids[left] - centroids[right]) / mean_psf_pixel) if np.all(np.isfinite(centroids[[left, right]])) else 0.0
            primary = max(image / 0.25, float(np.max(relative_flux)) / 0.20, color_max / 0.20, centroid / 0.5)
            matrix[left, right] = matrix[right, left] = primary
            maxima["image"] = max(maxima["image"], image)
            maxima["flux"] = max(maxima["flux"], float(np.max(relative_flux)))
            maxima["color"] = max(maxima["color"], color_max)
            maxima["centroid_psf"] = max(maxima["centroid_psf"], centroid)
            maxima["primary"] = max(maxima["primary"], primary)
    return matrix, maxima


def cluster_count(distance: np.ndarray) -> int:
    if len(distance) < 2:
        return len(distance)
    condensed = squareform(distance, checks=False)
    return int(len(np.unique(fcluster(linkage(condensed, method="complete"), t=1.0, criterion="distance"))))


def control_concentration(run_dir: Path) -> None:
    rows, _, isolated, _ = validation_arrays(run_dir)
    forward_rows = read_csv(run_dir / "tables/forward_consistency_prior_samples.csv")
    plausible_by_scene: dict[str, list[bool]] = {}
    for row in forward_rows:
        plausible_by_scene.setdefault(row["scene_id"], []).append(row["plausible"] == "True")
    survey = validated_lsst_survey()
    mean_psf = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in BAND_ORDER]) / 0.2)
    diversity_rows: list[dict[str, object]] = []
    started = time.time()
    with h5py.File(run_dir / "prior_samples/non_atlas_validation_k16.h5", "r") as handle:
        if not bool(handle.attrs["complete"]) or int(handle.attrs["completed_count"]) != len(rows):
            raise RuntimeError("prior sample store incomplete")
        for index, row in enumerate(rows):
            mask = np.asarray(plausible_by_scene[row["scene_id"]], dtype=bool)
            samples = np.asarray(handle["decomposition"][index, :, :3], dtype=np.float64)[mask]
            distance, maxima = pairwise_scientific(samples, mean_psf)
            clusters = cluster_count(distance)
            diversity_rows.append({
                "scene_id": row["scene_id"], "kind": row["kind"], "near_collision_pair_id": row["near_collision_pair_id"],
                "separation_arcsec": float(row["separation_arcsec"]),
                "truth_flux_ratio": float(np.sum(isolated[index, 0]) / max(float(np.sum(isolated[index, 1])), 1e-12)),
                "plausible_sample_count": len(samples), "scientific_cluster_count": clusters,
                "image_diameter": maxima["image"], "flux_diameter": maxima["flux"],
                "color_diameter_magnitude": maxima["color"], "centroid_diameter_psf": maxima["centroid_psf"],
                "primary_scientific_diameter": maxima["primary"],
                "empirical_witness": bool(len(samples) >= 2 and maxima["primary"] > 1.0 and clusters >= 2),
            })
            if (index + 1) % 100 == 0:
                print(json.dumps({"phase": "control", "completed": index + 1, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
    write_csv_fresh(run_dir / "tables/non_atlas_candidate_set_diversity.csv", diversity_rows)
    ordinary = [row for row in diversity_rows if row["kind"] == "ordinary"]
    near = [row for row in diversity_rows if row["kind"] == "near_collision"]
    ordinary_features = np.asarray([[row["separation_arcsec"], np.log(max(row["truth_flux_ratio"], 1e-12))] for row in ordinary])
    mean = ordinary_features.mean(axis=0)
    std = ordinary_features.std(axis=0) + 1e-12
    ordinary_standard = (ordinary_features - mean) / std
    matched_rows = []
    matched_diameters = []
    for near_row in near:
        feature = (np.asarray([near_row["separation_arcsec"], np.log(max(near_row["truth_flux_ratio"], 1e-12))]) - mean) / std
        match_index = int(np.argmin(np.linalg.norm(ordinary_standard - feature[None], axis=1)))
        match = ordinary[match_index]
        matched_diameters.append(float(match["primary_scientific_diameter"]))
        matched_rows.append({
            "near_scene_id": near_row["scene_id"], "near_pair_id": near_row["near_collision_pair_id"],
            "ordinary_scene_id": match["scene_id"], "near_separation": near_row["separation_arcsec"],
            "ordinary_separation": match["separation_arcsec"], "near_flux_ratio": near_row["truth_flux_ratio"],
            "ordinary_flux_ratio": match["truth_flux_ratio"], "near_diameter": near_row["primary_scientific_diameter"],
            "ordinary_diameter": match["primary_scientific_diameter"],
        })
    write_csv_fresh(run_dir / "tables/non_atlas_near_control_matches.csv", matched_rows)
    near_diameters = np.asarray([float(row["primary_scientific_diameter"]) for row in near])
    matched_diameters_array = np.asarray(matched_diameters)
    ratio = float(np.median(near_diameters) / max(np.median(matched_diameters_array), 1e-12))
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(near):
        groups.setdefault(row["near_collision_pair_id"], []).append(index)
    group_ids = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[group], dtype=int) for group in sampled])
        bootstrap.append(float(np.median(near_diameters[indices]) / max(np.median(matched_diameters_array[indices]), 1e-12)))
    bootstrap_lower, bootstrap_upper = np.quantile(bootstrap, [0.025, 0.975])
    prompt_rows = {row["scene_id"]: row for row in read_csv(run_dir / "tables/pre_atlas_promptability_per_scene.csv")}
    near_identity = float(np.mean([float(prompt_rows[row["scene_id"]]["individual_prior_sample_identity"]) for row in near]))
    ordinary_false_witness = float(np.mean([row["empirical_witness"] for row in ordinary]))
    near_median_plausible = float(np.median([row["plausible_sample_count"] for row in near]))
    gates = [
        {"gate": "ordinary_false_witness_rate", "threshold": "<=0.10", "observed": ordinary_false_witness, "pass": ordinary_false_witness <= 0.10},
        {"gate": "near_to_matched_control_diameter_ratio", "threshold": ">=1.25", "observed": ratio, "pass": ratio >= 1.25},
        {"gate": "diameter_ratio_bootstrap_lower", "threshold": ">1.0", "observed": float(bootstrap_lower), "pass": float(bootstrap_lower) > 1.0},
        {"gate": "near_median_plausible_set_size", "threshold": ">=4", "observed": near_median_plausible, "pass": near_median_plausible >= 4},
        {"gate": "near_retained_prompt_identity", "threshold": ">=0.70", "observed": near_identity, "pass": near_identity >= 0.70},
    ]
    passed = all(bool(row["pass"]) for row in gates)
    write_csv_fresh(run_dir / "tables/control_concentration_gates.csv", gates)
    summary = [{
        "ordinary_scene_count": len(ordinary), "near_collision_scene_count": len(near),
        "ordinary_median_primary_diameter": float(np.median([row["primary_scientific_diameter"] for row in ordinary])),
        "near_median_primary_diameter": float(np.median(near_diameters)),
        "matched_ordinary_median_primary_diameter": float(np.median(matched_diameters_array)),
        "near_to_matched_median_diameter_ratio": ratio,
        "bootstrap_95_lower": float(bootstrap_lower), "bootstrap_95_upper": float(bootstrap_upper),
        "ordinary_false_witness_rate": ordinary_false_witness,
        "near_false_or_true_witness_rate": float(np.mean([row["empirical_witness"] for row in near])),
        "near_median_plausible_set_size": near_median_plausible,
        "near_identity_rate": near_identity, "status": "PASS" if passed else "FAIL",
    }]
    write_csv_fresh(run_dir / "tables/control_concentration_summary.csv", summary)
    classification = "VALID_SELECTIVE_DIVERSITY" if passed else ("UNCONTROLLED_STOCHASTICITY" if ratio < 1.25 else "CONTROL_FALSE_WITNESS_EXCESS")
    report = f"""# Safe-control concentration and non-Atlas sensitivity gate

Status: **{'PASS' if passed else 'FAIL — ATLAS NOT AUTHORIZED'}**.

- Ordinary / near-collision scenes: {len(ordinary)} / {len(near)}.
- Ordinary / near median primary diameter: {summary[0]['ordinary_median_primary_diameter']:.6f} / {summary[0]['near_median_primary_diameter']:.6f}.
- Matched ordinary median diameter: {summary[0]['matched_ordinary_median_primary_diameter']:.6f}.
- Near/matched median diameter ratio: {ratio:.6f} (gate >=1.25).
- Pair-cluster bootstrap 95% interval: [{bootstrap_lower:.6f}, {bootstrap_upper:.6f}] (lower gate >1.0).
- Ordinary false-witness rate: {ordinary_false_witness:.6f} (gate <=0.10).
- Near median plausible-set size / retained identity: {near_median_plausible:.1f} / {near_identity:.6f}.
- Classification: **{classification}**.

Candidate diameters and clusters use only forward-consistent prior samples.
Matching uses separation and truth flux ratio only, before any diversity outcome.
No Atlas observation was opened.
"""
    write_text_fresh(run_dir / "diagnostics/control_concentration_report.md", report)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist([row["primary_scientific_diameter"] for row in ordinary], bins=40, alpha=0.6, label="ordinary", density=True)
    axis.hist(near_diameters, bins=40, alpha=0.6, label="near collision", density=True)
    axis.set(xlabel="plausible-set primary scientific diameter", ylabel="density", title="Selective hypothesis diversity")
    axis.legend(); axis.grid(alpha=0.25)
    figure.savefig(run_dir / "figures/non_atlas_control_concentration.png", dpi=170)
    plt.close(figure)
    write_json_fresh(run_dir / "logs/control_concentration_gate_complete.json", {
        "status": "PASS" if passed else "FAIL", "atlas_protocol_freeze_authorized": passed,
        "classification": classification, "near_to_matched_diameter_ratio": ratio,
        "bootstrap_95_lower": float(bootstrap_lower), "ordinary_false_witness_rate": ordinary_false_witness,
        "runtime_seconds": time.time() - started, "atlas_evaluation_count": 0,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "classification": classification, "gates": gates}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("gap", "forward", "control"), required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir, args.phase)
    if args.phase == "gap":
        prior_gap(run_dir)
    elif args.phase == "forward":
        forward_consistency_gate(run_dir)
    else:
        control_concentration(run_dir)


if __name__ == "__main__":
    main()
