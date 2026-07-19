#!/usr/bin/env python3
"""Freeze, calibrate, and execute the one-time Thayer-PU Atlas protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from btk.draw_blends import CatsimGenerator
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_hypotheses import (  # noqa: E402
    cluster_count,
    higher,
    pairwise_scientific,
    plausible,
    score_candidates,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)
from scripts.evaluate_probabilistic_unet_pre_atlas import load_model, prompts, require_mps, sample_outputs  # noqa: E402
from scripts.prepare_ambiguity_atlas_v0 import DefinitionSampling  # noqa: E402
from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene, validated_lsst_survey  # noqa: E402
from src.canonical_tensor_hash import canonical_tensor_sha256  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402


ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
K = 32
SAMPLE_SEEDS = list(range(2026077600, 2026077632))
PREFIXES = (1, 2, 4, 8, 16, 32)
BOOTSTRAP_SEED = 2026078401
BOOTSTRAP_REPLICATES = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array_historical(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_non_atlas_pass(run_dir: Path) -> dict[str, object]:
    record = json.loads((run_dir / "logs/control_concentration_gate_complete.json").read_text())
    if record["status"] != "PASS" or not record["atlas_protocol_freeze_authorized"]:
        raise RuntimeError("control-concentration gate blocks Atlas")
    if json.loads((run_dir / "logs/forward_consistency_gate_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("forward-consistency gate blocks Atlas")
    return record


def protocol_freeze(run_dir: Path) -> None:
    require_non_atlas_pass(run_dir)
    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_INITIAL_ATLAS_PASS" or freeze["pair_count"] != 25:
        raise RuntimeError("authoritative Atlas freeze unresolved")
    pair_manifest = ATLAS / "tables/atlas_pair_manifest.csv"
    visual_audit = ATLAS / "tables/atlas_initial_visual_audit.csv"
    if sha256_file(pair_manifest) != freeze["numerical_manifest_sha256"] or sha256_file(visual_audit) != freeze["visual_audit_sha256"]:
        raise RuntimeError("frozen Atlas artifact altered")
    checkpoint = run_dir / "checkpoints/thayer_pu_best.pth"
    thresholds = run_dir / "manifests/forward_consistency_thresholds.json"
    control_definitions = ATLAS / "manifests/fresh_validation_scene_definitions.csv"
    control_manifest = ATLAS / "tables/fresh_validation_scene_manifest.csv"
    protocol = f"""# Frozen Thayer-PU Atlas protocol

Frozen UTC: `{datetime.now(timezone.utc).isoformat()}` before matched-control
sampling or any Thayer-PU Atlas inference.

- Selected checkpoint SHA-256: `{sha256_file(checkpoint)}` (validation-selected epoch 27).
- K=32 truth-free prior samples; seeds: {SAMPLE_SEEDS[0]} through {SAMPLE_SEEDS[-1]}.
- The same scene-level epsilon sample is queried under prompt A and prompt B.
- Posterior samples, target-guided resampling, adaptive rejection, and post-Atlas tuning are prohibited.
- Forward tolerance SHA-256: `{sha256_file(thresholds)}`; calibration-only and already frozen.
- Scientific distances: image 0.25, per-band flux 0.20, color 0.20 mag,
  centroid 0.5 mean-PSF FWHM; complete-linkage cluster cut at primary distance 1.0.
- Candidate diameter is the maximum primary scientific distance among retained
  forward-consistent requested layers; fewer than two retained candidates gives zero.
- Matched control set: first 25 scenes of the frozen Atlas fresh-validation manifest.
  For each K prefix, its control 95th percentile is frozen with `higher`; positive
  classification is strict `diameter > threshold`, yielding the authoritative 4% FPR scale.
- Atlas: exactly the 25 frozen pair IDs and both observation sides (50 observations),
  exact BTK noisy replay, prior only, one execution.
- Coverage requires at least one retained sample within primary distance <=1.0 of
  own truth or the paired alternate requested truth.
- Gates: witnesses >=30/50 and >19/50; AUROC >=0.60 with pair-cluster bootstrap
  95% lower endpoint >0.5; recall at 4% control FPR >=0.10; safe-control false
  witnesses <=0.10; own coverage >=0.70; alternate coverage >=0.30; Atlas forward
  rate >=0.50.
"""
    protocol_path = run_dir / "preregistration/frozen_atlas_protocol.md"
    write_text_fresh(protocol_path, protocol)
    atlas_hash_rows = []
    for pair_id in freeze["pair_ids"]:
        path = ATLAS / f"atlas/{pair_id}.npz"
        atlas_hash_rows.append({"pair_id": pair_id, "path": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
    write_csv_fresh(run_dir / "tables/frozen_atlas_artifact_hashes.csv", atlas_hash_rows)
    write_json_fresh(run_dir / "preregistration/frozen_atlas_protocol_record.json", {
        "status": "FROZEN_BEFORE_CONTROL_OR_ATLAS_SAMPLING", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol_path), "checkpoint_sha256": sha256_file(checkpoint),
        "forward_thresholds_sha256": sha256_file(thresholds), "atlas_pair_manifest_sha256": sha256_file(pair_manifest),
        "atlas_visual_audit_sha256": sha256_file(visual_audit), "atlas_artifact_hash_table_sha256": sha256_file(run_dir / "tables/frozen_atlas_artifact_hashes.csv"),
        "control_definitions_sha256": sha256_file(control_definitions), "control_manifest_sha256": sha256_file(control_manifest),
        "k": K, "sample_seeds": SAMPLE_SEEDS, "atlas_evaluation_count": 0,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(sha256_file(protocol_path))


def make_epsilon(scene_count: int) -> np.ndarray:
    values = []
    for seed in SAMPLE_SEEDS:
        values.append(np.random.default_rng(seed).standard_normal((scene_count, 8)).astype(np.float32))
    return np.stack(values, axis=1)


def inference(
    model,
    observed: np.ndarray,
    xy: np.ndarray,
    scales: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_a, prompt_b = prompts(xy)
    epsilon_all = make_epsilon(len(observed))
    outputs_a, outputs_b = [], []
    with torch.no_grad():
        for start in range(0, len(observed), 4):
            stop = min(start + 4, len(observed))
            blend = torch.from_numpy(np.ascontiguousarray((observed[start:stop] / scales[None, :, None, None]).astype(np.float32))).to(device)
            pa = torch.from_numpy(np.ascontiguousarray(prompt_a[start:stop])).to(device)
            pb = torch.from_numpy(np.ascontiguousarray(prompt_b[start:stop])).to(device)
            mean, log_variance = model.encode_prior(blend)
            epsilon = torch.from_numpy(epsilon_all[start:stop]).to(device)
            outputs_a.append(sample_outputs(model, blend, pa, mean, log_variance, epsilon).cpu().numpy())
            outputs_b.append(sample_outputs(model, blend, pb, mean, log_variance, epsilon).cpu().numpy())
    scale6 = np.tile(scales, 2)[None, None, :, None, None]
    return np.concatenate(outputs_a) * scale6, np.concatenate(outputs_b) * scale6


def scene_metrics(
    decomposition: np.ndarray,
    observed: np.ndarray,
    sky: np.ndarray,
    thresholds: dict[str, object],
    mean_psf: float,
    own_truth: np.ndarray | None = None,
    alternate_truth: np.ndarray | None = None,
    prefix: int = K,
) -> dict[str, object]:
    candidates = decomposition[:prefix].astype(np.float64)
    global_score, band_score, flux_score = score_candidates(observed[None].astype(np.float64), candidates[None], sky)
    mask = plausible(global_score, band_score, flux_score, thresholds)[0]
    requested = candidates[mask, :3]
    distance, maxima = pairwise_scientific(requested, mean_psf)
    clusters = cluster_count(distance)
    own_distances: list[float] = []
    alternate_distances: list[float] = []
    if own_truth is not None:
        own_distances = [scientific_distance(sample, own_truth, mean_psf_fwhm_pixel=mean_psf).primary_normalized for sample in requested]
    if alternate_truth is not None:
        alternate_distances = [scientific_distance(sample, alternate_truth, mean_psf_fwhm_pixel=mean_psf).primary_normalized for sample in requested]
    return {
        "plausible_sample_count": int(mask.sum()),
        "forward_consistency_rate": float(mask.mean()),
        "primary_scientific_diameter": float(maxima["primary"]),
        "image_diameter": float(maxima["image"]),
        "flux_diameter": float(maxima["flux"]),
        "color_diameter_magnitude": float(maxima["color"]),
        "centroid_diameter_psf": float(maxima["centroid_psf"]),
        "scientific_cluster_count": clusters,
        "model_generated_witness": bool(len(requested) >= 2 and maxima["primary"] > 1.0 and clusters >= 2),
        "own_truth_coverage": bool(own_distances and min(own_distances) <= 1.0),
        "alternate_truth_coverage": bool(alternate_distances and min(alternate_distances) <= 1.0),
        "best_own_truth_distance": min(own_distances) if own_distances else math.nan,
        "best_alternate_truth_distance": min(alternate_distances) if alternate_distances else math.nan,
        "plausible_mask": mask,
    }


def render_controls() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_definitions = read_csv(ATLAS / "manifests/fresh_validation_scene_definitions.csv")
    definitions = all_definitions[:25]
    manifest = {row["scene_id"]: row for row in read_csv(ATLAS / "tables/fresh_validation_scene_manifest.csv")}
    catalog, _ = load_catsim_catalog(CATALOG)
    survey = validated_lsst_survey()
    frozen_batch = all_definitions[:250]
    batch = next(CatsimGenerator(
        catalog, DefinitionSampling(frozen_batch), survey, batch_size=250, njobs=1,
        verbose=False, use_bar=False, add_noise="all", seed=2026071223000,
        apply_shear=False, augment_data=False,
    ))
    band_indices = [tuple(survey.available_filters).index(band) for band in ("g", "r", "z")]
    observed = np.asarray(batch.blend_images[:25, band_indices], dtype=np.float64)
    isolated = np.asarray(batch.isolated_images[:25, :2][:, :, band_indices], dtype=np.float64)
    xy = np.asarray([
        [[source["x_peak"], source["y_peak"]] for source in catalog_rows]
        for catalog_rows in batch.catalog_list[:25]
    ], dtype=np.float64)
    for index, row in enumerate(definitions):
        if sha256_array_historical(observed[index]) != manifest[row["scene_id"]]["noisy_blend_sha256"]:
            raise RuntimeError(f"frozen 250-scene control replay failed: {row['scene_id']}")
    with np.load(ATLAS / "atlas/atlas_pair_0001.npz", allow_pickle=False) as arrays:
        sky = np.asarray(arrays["sky_electrons"], dtype=np.float64)
    return definitions, observed.astype(np.float32), isolated.astype(np.float32), xy, np.repeat(sky[None], 25, axis=0)


def control_freeze(run_dir: Path) -> None:
    protocol_record = json.loads((run_dir / "preregistration/frozen_atlas_protocol_record.json").read_text())
    if protocol_record["status"] != "FROZEN_BEFORE_CONTROL_OR_ATLAS_SAMPLING":
        raise RuntimeError("Atlas protocol not frozen")
    device = require_mps()
    model = load_model(run_dir, device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    thresholds = json.loads((run_dir / "manifests/forward_consistency_thresholds.json").read_text())
    survey = validated_lsst_survey()
    mean_psf = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in ("g", "r", "z")]) / 0.2)
    definitions, observed, isolated, xy, sky = render_controls()
    output_a, output_b = inference(model, observed, xy, scales, device)
    rows: list[dict[str, object]] = []
    prefix_diameter_rows: list[dict[str, object]] = []
    prefix_values: dict[int, list[float]] = {prefix: [] for prefix in PREFIXES}
    started = time.time()
    for index, definition in enumerate(definitions):
        full = scene_metrics(output_a[index], observed[index], sky[index], thresholds, mean_psf)
        rows.append({
            "scene_id": definition["scene_id"], "primary_scientific_diameter": full["primary_scientific_diameter"],
            "plausible_sample_count": full["plausible_sample_count"], "scientific_cluster_count": full["scientific_cluster_count"],
            "model_generated_witness": full["model_generated_witness"], "forward_consistency_rate": full["forward_consistency_rate"],
            "matched_prompt_swap_mse": float(np.mean((output_a[index] - np.concatenate((output_b[index, :, 3:], output_b[index, :, :3]), axis=1)) ** 2)),
            "candidate_hash_sequence": hashlib.sha256("\n".join(canonical_tensor_sha256(value) for value in output_a[index]).encode()).hexdigest(),
        })
        for prefix in PREFIXES:
            value = float(scene_metrics(output_a[index], observed[index], sky[index], thresholds, mean_psf, prefix=prefix)["primary_scientific_diameter"])
            prefix_values[prefix].append(value)
            prefix_diameter_rows.append({"scene_id": definition["scene_id"], "k_prefix": prefix, "primary_scientific_diameter": value})
    write_csv_fresh(run_dir / "tables/frozen_atlas_matched_control_results.csv", rows)
    write_csv_fresh(run_dir / "tables/frozen_atlas_control_prefix_diameters.csv", prefix_diameter_rows)
    thresholds_rows = []
    for prefix in PREFIXES:
        threshold = higher(np.asarray(prefix_values[prefix]), 0.95)
        fpr = float(np.mean(np.asarray(prefix_values[prefix]) > threshold))
        thresholds_rows.append({"k_prefix": prefix, "control_95th_percentile": threshold, "strict_greater_control_fpr": fpr})
    write_csv_fresh(run_dir / "tables/frozen_atlas_operating_thresholds.csv", thresholds_rows)
    false_witness = float(np.mean([row["model_generated_witness"] for row in rows]))
    write_json_fresh(run_dir / "preregistration/frozen_atlas_operating_record.json", {
        "status": "FROZEN_AFTER_CONTROLS_BEFORE_ATLAS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_results_sha256": sha256_file(run_dir / "tables/frozen_atlas_matched_control_results.csv"),
        "control_prefix_diameters_sha256": sha256_file(run_dir / "tables/frozen_atlas_control_prefix_diameters.csv"),
        "operating_thresholds_sha256": sha256_file(run_dir / "tables/frozen_atlas_operating_thresholds.csv"),
        "k32_threshold": float(thresholds_rows[-1]["control_95th_percentile"]),
        "k32_observed_control_fpr": float(thresholds_rows[-1]["strict_greater_control_fpr"]),
        "safe_control_false_witness_rate": false_witness,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    write_json_fresh(run_dir / "logs/atlas_controls_complete.json", {
        "status": "PASS", "control_count": len(rows), "safe_control_false_witness_rate": false_witness,
        "runtime_seconds": time.time() - started, "atlas_evaluation_count": 0,
    })
    print(json.dumps({"control_count": len(rows), "k32_threshold": thresholds_rows[-1], "false_witness_rate": false_witness}, sort_keys=True))


def render_atlas_observations() -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    pairs = {row["pair_id"]: row for row in read_csv(ATLAS / "tables/atlas_pair_manifest.csv")}
    definitions = read_csv(ATLAS / "manifests/atlas_pool_scene_definitions.csv")
    catalog, _ = load_catsim_catalog(CATALOG)
    records, observed, own_truth, alternate_truth, xy, sky_values = [], [], [], [], [], []
    for pair_id in freeze["pair_ids"]:
        pair = pairs[pair_id]
        with np.load(ATLAS / f"atlas/{pair_id}.npz", allow_pickle=False) as arrays:
            stored = {
                "left": {"blend": np.asarray(arrays["left_blend"], dtype=np.float32), "isolated": np.asarray(arrays["left_isolated"], dtype=np.float32)},
                "right": {"blend": np.asarray(arrays["right_blend"], dtype=np.float32), "isolated": np.asarray(arrays["right_isolated"], dtype=np.float32)},
            }
            sky = np.asarray(arrays["sky_electrons"], dtype=np.float64)
        rendered = {}
        for side in ("left", "right"):
            definition = definitions[int(pair[f"{side}_pool_index"])]
            spec = SceneSpec(
                scene_id=definition["scene_id"],
                catalog_rows=(int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])),
                positions_arcsec=((float(definition["target_x_arcsec"]), float(definition["target_y_arcsec"])), (float(definition["contaminant_x_arcsec"]), float(definition["contaminant_y_arcsec"]))),
                source_selection_seed=int(definition["source_selection_seed"]), position_seed=int(definition["position_seed"]), noise_seed=int(definition["noise_seed"]),
            )
            noiseless = render_fixed_scene(catalog, spec, add_noise="none")
            noisy = render_fixed_scene(catalog, spec, add_noise="all")
            if not np.array_equal(noiseless.blend.astype(np.float32), stored[side]["blend"]):
                raise RuntimeError(f"Atlas exact replay failed: {pair_id} {side}")
            rendered[side] = noisy
        for side, alternate in (("left", "right"), ("right", "left")):
            record = {"pair_id": pair_id, "side": side, "observation_id": f"{pair_id}:{side}"}
            records.append(record)
            observed.append(rendered[side].blend.astype(np.float32))
            own_truth.append(stored[side]["isolated"][0])
            alternate_truth.append(stored[alternate]["isolated"][0])
            xy.append(np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered[side].catalog], dtype=np.float64))
            sky_values.append(sky)
    return records, np.stack(observed), np.stack(own_truth), np.stack(alternate_truth), np.stack(xy), np.stack(sky_values)


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    combined = np.concatenate((positive, negative))
    ranks = rankdata(combined, method="average")
    count_positive = len(positive)
    return float((ranks[:count_positive].sum() - count_positive * (count_positive + 1) / 2) / (count_positive * len(negative)))


def atlas_evaluation(run_dir: Path) -> None:
    operating = json.loads((run_dir / "preregistration/frozen_atlas_operating_record.json").read_text())
    if operating["status"] != "FROZEN_AFTER_CONTROLS_BEFORE_ATLAS" or operating["atlas_evaluation_count"] != 0:
        raise RuntimeError("Atlas operating point is not frozen")
    protocol = json.loads((run_dir / "preregistration/frozen_atlas_protocol_record.json").read_text())
    if sha256_file(run_dir / "checkpoints/thayer_pu_best.pth") != protocol["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint changed after protocol freeze")
    if sha256_file(run_dir / "manifests/forward_consistency_thresholds.json") != protocol["forward_thresholds_sha256"]:
        raise RuntimeError("forward threshold changed after protocol freeze")
    for row in read_csv(run_dir / "tables/frozen_atlas_artifact_hashes.csv"):
        if sha256_file(REPO / row["path"]) != row["sha256"]:
            raise RuntimeError(f"frozen Atlas artifact changed: {row['pair_id']}")
    started_guard = run_dir / "atlas_evaluation/atlas_inference_started.json"
    write_json_fresh(started_guard, {"status": "ONE_TIME_ATLAS_INFERENCE_STARTED", "started_at_utc": datetime.now(timezone.utc).isoformat(), "prior_only": True})
    device = require_mps()
    model = load_model(run_dir, device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    thresholds = json.loads((run_dir / "manifests/forward_consistency_thresholds.json").read_text())
    survey = validated_lsst_survey()
    mean_psf = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in ("g", "r", "z")]) / 0.2)
    records, observed, own_truth, alternate_truth, xy, sky = render_atlas_observations()
    output_a, output_b = inference(model, observed, xy, scales, device)
    sample_path = run_dir / "atlas_evaluation/atlas_prior_samples_k32.h5"
    with h5py.File(sample_path, "x") as handle:
        handle.create_dataset("prompt_a_decomposition", data=output_a.astype(np.float32), chunks=(1, 1, 6, 60, 60), compression="lzf")
        handle.create_dataset("prompt_b_decomposition", data=output_b.astype(np.float32), chunks=(1, 1, 6, 60, 60), compression="lzf")
        handle.attrs["complete"] = True
        handle.attrs["checkpoint_sha256"] = sha256_file(run_dir / "checkpoints/thayer_pu_best.pth")
    control_rows = read_csv(run_dir / "tables/frozen_atlas_matched_control_results.csv")
    control_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in control_rows])
    control_prefix_rows = read_csv(run_dir / "tables/frozen_atlas_control_prefix_diameters.csv")
    control_prefix_diameters = {
        prefix: np.asarray([float(row["primary_scientific_diameter"]) for row in control_prefix_rows if int(row["k_prefix"]) == prefix])
        for prefix in PREFIXES
    }
    operating_rows = {int(row["k_prefix"]): row for row in read_csv(run_dir / "tables/frozen_atlas_operating_thresholds.csv")}
    atlas_rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    prefix_metrics: dict[int, list[dict[str, object]]] = {prefix: [] for prefix in PREFIXES}
    started = time.time()
    for index, record in enumerate(records):
        for prefix in PREFIXES:
            metrics = scene_metrics(output_a[index], observed[index], sky[index], thresholds, mean_psf, own_truth[index], alternate_truth[index], prefix)
            prefix_metrics[prefix].append(metrics)
        full = prefix_metrics[K][-1]
        atlas_rows.append({
            **record,
            "plausible_sample_count": full["plausible_sample_count"],
            "forward_consistency_rate": full["forward_consistency_rate"],
            "primary_scientific_diameter": full["primary_scientific_diameter"],
            "image_diameter": full["image_diameter"], "flux_diameter": full["flux_diameter"],
            "color_diameter_magnitude": full["color_diameter_magnitude"], "centroid_diameter_psf": full["centroid_diameter_psf"],
            "scientific_cluster_count": full["scientific_cluster_count"],
            "model_generated_witness": full["model_generated_witness"],
            "own_truth_coverage": full["own_truth_coverage"], "alternate_truth_coverage": full["alternate_truth_coverage"],
            "best_own_truth_distance": full["best_own_truth_distance"], "best_alternate_truth_distance": full["best_alternate_truth_distance"],
            "matched_prompt_swap_mse": float(np.mean((output_a[index] - np.concatenate((output_b[index, :, 3:], output_b[index, :, :3]), axis=1)) ** 2)),
            "sample_sequence_sha256": hashlib.sha256("\n".join(canonical_tensor_sha256(value) for value in output_a[index]).encode()).hexdigest(),
        })
    write_csv_fresh(run_dir / "tables/atlas_stochastic_hypothesis_results.csv", atlas_rows)
    atlas_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in atlas_rows])
    witness_count = int(sum(row["model_generated_witness"] for row in atlas_rows))
    own_rate = float(np.mean([row["own_truth_coverage"] for row in atlas_rows]))
    alternate_rate = float(np.mean([row["alternate_truth_coverage"] for row in atlas_rows]))
    forward_rate = float(np.mean([row["forward_consistency_rate"] for row in atlas_rows]))
    diameter_auc = auc(atlas_diameter, control_diameter)
    threshold = float(operating_rows[K]["control_95th_percentile"])
    recall = float(np.mean(atlas_diameter > threshold))
    control_false_witness = float(operating["safe_control_false_witness_rate"])
    groups = sorted({row["pair_id"] for row in records})
    indices_by_group = {group: np.asarray([index for index, row in enumerate(records) if row["pair_id"] == group]) for group in groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        atlas_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        control_indices = rng.choice(len(control_diameter), size=len(control_diameter), replace=True)
        auc_bootstrap.append(auc(atlas_diameter[atlas_indices], control_diameter[control_indices]))
    auc_lower, auc_upper = np.quantile(auc_bootstrap, [0.025, 0.975])
    gates = [
        {"gate": "model_generated_witness_count", "threshold": ">=30/50 and >19/50", "observed": witness_count, "pass": witness_count >= 30},
        {"gate": "candidate_diameter_auroc", "threshold": ">=0.60", "observed": diameter_auc, "pass": diameter_auc >= 0.60},
        {"gate": "auroc_bootstrap_lower", "threshold": ">0.5", "observed": float(auc_lower), "pass": float(auc_lower) > 0.5},
        {"gate": "recall_at_4pct_control_fpr", "threshold": ">=0.10", "observed": recall, "pass": recall >= 0.10},
        {"gate": "safe_control_false_witness", "threshold": "<=0.10", "observed": control_false_witness, "pass": control_false_witness <= 0.10},
        {"gate": "own_truth_coverage", "threshold": ">=0.70", "observed": own_rate, "pass": own_rate >= 0.70},
        {"gate": "alternate_truth_coverage", "threshold": ">=0.30", "observed": alternate_rate, "pass": alternate_rate >= 0.30},
        {"gate": "atlas_forward_consistency_rate", "threshold": ">=0.50", "observed": forward_rate, "pass": forward_rate >= 0.50},
    ]
    atlas_pass = all(bool(row["pass"]) for row in gates)
    for prefix in PREFIXES:
        values = prefix_metrics[prefix]
        diameters = np.asarray([float(value["primary_scientific_diameter"]) for value in values])
        prefix_rows.append({
            "k": prefix, "witness_count": int(sum(value["model_generated_witness"] for value in values)),
            "own_truth_coverage": float(np.mean([value["own_truth_coverage"] for value in values])),
            "alternate_truth_coverage": float(np.mean([value["alternate_truth_coverage"] for value in values])),
            "forward_consistency_rate": float(np.mean([value["forward_consistency_rate"] for value in values])),
            "candidate_diameter_auroc": auc(diameters, control_prefix_diameters[prefix]),
            "prefix_control_threshold": float(operating_rows[prefix]["control_95th_percentile"]),
            "recall_at_prefix_4pct_control_fpr": float(np.mean(diameters > float(operating_rows[prefix]["control_95th_percentile"]))),
        })
    write_csv_fresh(run_dir / "tables/atlas_success_gates.csv", gates)
    write_csv_fresh(run_dir / "tables/atlas_sample_efficiency.csv", prefix_rows)
    if atlas_pass:
        decision = "SUCCESS"
        next_experiment = "Preregister a leave-one-family-out Thayer-Audit feasibility study using Thayer-PU as one frozen candidate family, with no catalog admission or lockbox access."
    elif witness_count > 19:
        decision = "PARTIAL SUCCESS"
        next_experiment = "Preregister one focused conditional normalizing-flow prior correction on the frozen Thayer-PU representation, retaining every current non-Atlas and Atlas gate."
    else:
        decision = "FAILURE"
        next_experiment = "Preregister one explicit prompt-to-source assignment module before any further multimodal generator training."
    write_csv_fresh(run_dir / "tables/final_scientific_decision.csv", [{
        "decision": decision, "witness_count": witness_count, "baseline_witness_count": 19,
        "candidate_diameter_auroc": diameter_auc, "baseline_auroc": 0.4712,
        "recall_at_4pct_control_fpr": recall, "baseline_recall": 0.0,
        "own_truth_coverage": own_rate, "alternate_truth_coverage": alternate_rate,
        "safe_control_false_witness_rate": control_false_witness, "forward_consistency_rate": forward_rate,
        "exact_next_experiment": next_experiment,
    }])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(control_diameter, bins=20, alpha=0.7, label="controls")
    axes[0].hist(atlas_diameter, bins=20, alpha=0.7, label="Atlas")
    axes[0].axvline(threshold, color="black", linestyle="--", label="4% FPR threshold")
    axes[0].set(title="Candidate-set diameter", xlabel="primary diameter", ylabel="count"); axes[0].legend()
    axes[1].plot([row["k"] for row in prefix_rows], [row["witness_count"] for row in prefix_rows], marker="o", label="witnesses")
    axes[1].axhline(19, color="gray", linestyle="--", label="baseline 19")
    axes[1].set(xscale="log", xticks=list(PREFIXES), title="Sample efficiency", xlabel="K", ylabel="Atlas witness count"); axes[1].legend()
    for axis in axes: axis.grid(alpha=0.25)
    figure.savefig(run_dir / "figures/atlas_stochastic_hypothesis_results.png", dpi=170)
    plt.close(figure)
    chosen = np.arange(min(5, len(records)))
    figure, axes = plt.subplots(len(chosen), 6, figsize=(12, 2.2 * len(chosen)), constrained_layout=True)
    for row_index, index in enumerate(chosen):
        panels = [own_truth[index, 1], alternate_truth[index, 1], observed[index, 1], output_a[index, 0, 1], output_a[index, 1, 1], output_a[index, 2, 1]]
        scale = max(float(np.max(np.abs(panel))) for panel in panels)
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(np.arcsinh(panel / max(scale, 1e-12) * 20), origin="lower", cmap="coolwarm")
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
    for column, title in enumerate(("own truth", "alternate truth", "observed", "prior 1", "prior 2", "prior 3")):
        axes[0, column].set_title(title, fontsize=8)
    figure.savefig(run_dir / "example_grids/atlas_prior_sample_gallery.png", dpi=170)
    plt.close(figure)
    write_json_fresh(run_dir / "atlas_evaluation/atlas_one_pass_complete.json", {
        "status": "PASS", "atlas_evaluation_count": 1, "decision": decision,
        "witness_count": witness_count, "candidate_diameter_auroc": diameter_auc,
        "auroc_bootstrap_95": [float(auc_lower), float(auc_upper)], "recall_at_4pct_control_fpr": recall,
        "own_truth_coverage": own_rate, "alternate_truth_coverage": alternate_rate,
        "safe_control_false_witness_rate": control_false_witness, "forward_consistency_rate": forward_rate,
        "runtime_seconds": time.time() - started, "post_atlas_tuning": False,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"decision": decision, "witness_count": witness_count, "auroc": diameter_auc, "auroc_ci": [float(auc_lower), float(auc_upper)], "recall": recall, "gates": gates}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("protocol", "controls", "atlas"), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    require_non_atlas_pass(run_dir)
    if args.phase == "protocol":
        protocol_freeze(run_dir)
    elif args.phase == "controls":
        control_freeze(run_dir)
    else:
        atlas_evaluation(run_dir)


if __name__ == "__main__":
    main()
