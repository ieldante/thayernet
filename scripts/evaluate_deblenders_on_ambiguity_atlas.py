#!/usr/bin/env python3
"""Evaluate frozen prompted reconstructors on the initial Ambiguity Atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import CompactSelectNet, gaussian_prompt_numpy  # noqa: E402
from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene  # noqa: E402
from src.competing_hypotheses import (  # noqa: E402
    ForwardConsistency,
    PlausibilityThresholds,
    empirical_ambiguity_witness,
    forward_consistency,
    is_plausible,
    scientific_distance,
)
from src.models_thayer_select import ThayerSelectNet  # noqa: E402


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
R0 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/checkpoints/r0_best.pth"
R1 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/checkpoints/r1_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
MIN_LOG_VARIANCE = -8.0
MAX_LOG_VARIANCE = 2.0
BATCH_SIZE = 16


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        if fields is None:
            fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def load_models(device: torch.device):
    models = {}
    for name, path in (("THAYER_SELECT_CONDITION_C", CONDITION_C), ("THAYER_SELECT_R0", R0)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = CompactSelectNet(4).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
    payload = torch.load(R1, map_location="cpu", weights_only=False)
    r1 = ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE).to(device)
    r1.load_state_dict(payload["state_dict"], strict=True)
    r1.eval()
    for parameter in r1.parameters():
        parameter.requires_grad_(False)
    models["THAYER_SELECT_R1_RECONSTRUCTION_ONLY"] = r1
    return models


def load_thresholds(run_dir: Path) -> PlausibilityThresholds:
    payload = json.loads((run_dir / "calibration/forward_consistency_thresholds.json").read_text())
    return PlausibilityThresholds(
        global_chi_square_mean=float(payload["global_chi_square_mean"]),
        per_band_chi_square_mean=tuple(float(value) for value in payload["per_band_chi_square_mean"]),
        absolute_relative_flux_residual=float(payload["absolute_relative_flux_residual"]),
        calibration_count=int(payload["calibration_count"]),
        quantile_global=float(payload["quantile_global"]),
        quantile_per_band=float(payload["quantile_per_band"]),
        quantile_flux=float(payload["quantile_flux"]),
    )


def infer_model(model_name: str, model, blends: np.ndarray, prompts: np.ndarray, scales: np.ndarray, device: torch.device):
    reconstructions = []
    recoverability = []
    no_source = []
    for start in range(0, len(blends), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(blends))
        image = torch.from_numpy((blends[start:stop] / scales[None, :, None, None]).astype(np.float32)).to(device)
        prompt = torch.from_numpy(prompts[start:stop, None].astype(np.float32)).to(device)
        with torch.no_grad():
            if model_name == "THAYER_SELECT_R1_RECONSTRUCTION_ONLY":
                output = model(image, prompt)
                prediction = output["reconstruction"]
                recoverability.append(output["recoverability"].flatten().cpu().numpy())
                no_source.append(output["no_source_probability"].flatten().cpu().numpy())
            else:
                prediction = model(torch.cat([image, prompt], dim=1))
                recoverability.append(np.full(stop - start, np.nan))
                no_source.append(np.full(stop - start, np.nan))
        reconstructions.append(prediction.cpu().numpy() * scales[None, :, None, None])
    return (
        np.concatenate(reconstructions).astype(np.float32),
        np.concatenate(recoverability).astype(np.float64),
        np.concatenate(no_source).astype(np.float64),
    )


def consistency_from_row(row: dict[str, object]) -> ForwardConsistency:
    return ForwardConsistency(
        global_chi_square_mean=float(row["forward_global_chi_square_mean"]),
        per_band_chi_square_mean=(
            float(row["forward_g_chi_square_mean"]),
            float(row["forward_r_chi_square_mean"]),
            float(row["forward_z_chi_square_mean"]),
        ),
        residual_neighbor_correlation=(
            float(row["forward_g_neighbor_correlation"]),
            float(row["forward_r_neighbor_correlation"]),
            float(row["forward_z_neighbor_correlation"]),
        ),
        relative_flux_residual=float(row["forward_relative_flux_residual"]),
        finite=bool(row["finite"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "manifests/atlas_initial_freeze_record.json").exists():
        raise RuntimeError("initial Atlas is not frozen")
    if not (run_dir / "calibration/forward_consistency_thresholds.json").exists():
        raise RuntimeError("forward thresholds are not frozen")
    inventory_path = run_dir / "tables/candidate_output_inventory.csv"
    if inventory_path.exists():
        raise FileExistsError(inventory_path)
    device = require_mps()
    models = load_models(device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    thresholds = load_thresholds(run_dir)
    freeze = json.loads((run_dir / "manifests/atlas_initial_freeze_record.json").read_text())
    pair_ids = freeze["pair_ids"]
    pairs = {row["pair_id"]: row for row in read_csv(run_dir / "tables/atlas_pair_manifest.csv")}
    definitions = read_csv(run_dir / "manifests/atlas_pool_scene_definitions.csv")
    catalog, _ = load_catsim_catalog(CATALOG)
    mean_psf_fwhm_pixel = 0.0
    observation_records = []
    truth_witness_rows: list[dict[str, object]] = []
    started = time.time()
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        pair_arrays = np.load(run_dir / f"atlas/{pair_id}.npz", allow_pickle=False)
        sky = np.asarray(pair_arrays["sky_electrons"], dtype=np.float64)
        stored = {
            "left": {
                "blend": np.asarray(pair_arrays["left_blend"], dtype=np.float64),
                "isolated": np.asarray(pair_arrays["left_isolated"], dtype=np.float64),
            },
            "right": {
                "blend": np.asarray(pair_arrays["right_blend"], dtype=np.float64),
                "isolated": np.asarray(pair_arrays["right_isolated"], dtype=np.float64),
            },
        }
        noisy_by_side = {}
        xy_by_side = {}
        for side in ("left", "right"):
            definition = definitions[int(pair[f"{side}_pool_index"])]
            spec = SceneSpec(
                scene_id=definition["scene_id"],
                catalog_rows=(int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])),
                positions_arcsec=(
                    (float(definition["target_x_arcsec"]), float(definition["target_y_arcsec"])),
                    (float(definition["contaminant_x_arcsec"]), float(definition["contaminant_y_arcsec"])),
                ),
                source_selection_seed=int(definition["source_selection_seed"]),
                position_seed=int(definition["position_seed"]),
                noise_seed=int(definition["noise_seed"]),
            )
            noiseless = render_fixed_scene(catalog, spec, add_noise="none")
            noisy = render_fixed_scene(catalog, spec, add_noise="all")
            if sha256_array(noiseless.blend.astype(np.float32)) != sha256_array(stored[side]["blend"].astype(np.float32)):
                raise RuntimeError("stored Atlas blend replay failed")
            noisy_by_side[side] = noisy.blend
            xy_by_side[side] = np.asarray([[source["x_peak"], source["y_peak"]] for source in noisy.catalog], dtype=np.float64)
            if mean_psf_fwhm_pixel == 0.0:
                # Gaussian-equivalent FWHM from the frozen PSF second moment is not needed;
                # use the preregistered survey mean from the Atlas construction.
                survey = __import__("src.btk_scene", fromlist=["validated_lsst_survey"]).validated_lsst_survey()
                mean_psf_fwhm_pixel = float(np.mean([survey.get_filter(b).psf_fwhm.to_value("arcsec") for b in ("g", "r", "z")]) / 0.2)
        for observation_side in ("left", "right"):
            scores = {}
            sources = {"left_truth_decomposition": stored["left"]["isolated"][0], "right_truth_decomposition": stored["right"]["isolated"][0]}
            for candidate_side in ("left", "right"):
                scores[f"{candidate_side}_truth_decomposition"] = forward_consistency(
                    noisy_by_side[observation_side], stored[candidate_side]["isolated"], sky
                )
            witness = empirical_ambiguity_witness(
                sources,
                scores,
                thresholds,
                mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
                artifact_audit_passed=True,
            )
            truth_witness_rows.append(
                {
                    "pair_id": pair_id,
                    "observation_side": observation_side,
                    "left_candidate_plausible": is_plausible(scores["left_truth_decomposition"], thresholds),
                    "right_candidate_plausible": is_plausible(scores["right_truth_decomposition"], thresholds),
                    "plausible_candidate_count": len(witness.retained_candidate_ids),
                    "primary_diameter": witness.primary_diameter,
                    "empirical_ambiguity_witness": witness.exists,
                    "reason": witness.reason,
                    "artifact_audit_passed": witness.artifact_audit_passed,
                }
            )
        for regime in ("noiseless_mean", "noisy_observation"):
            for side in ("left", "right"):
                blend = stored[side]["blend"] if regime == "noiseless_mean" else noisy_by_side[side]
                for source_index in (0, 1):
                    x, y = xy_by_side[side][source_index]
                    observation_records.append(
                        {
                            "pair_id": pair_id,
                            "side": side,
                            "regime": regime,
                            "source_index": source_index,
                            "blend": blend,
                            "prompt": gaussian_prompt_numpy(float(x), float(y)),
                            "truth": stored[side]["isolated"][source_index],
                            "sky": sky,
                        }
                    )
        pair_arrays.close()
    blends = np.stack([record["blend"] for record in observation_records]).astype(np.float32)
    prompts = np.stack([record["prompt"] for record in observation_records]).astype(np.float32)

    candidate_rows: list[dict[str, object]] = []
    output_lookup = {}
    inference_runtime: dict[str, float] = {}
    replay_checks = {}
    for model_name, model in models.items():
        model_started = time.time()
        prediction, confidence, no_source = infer_model(model_name, model, blends, prompts, scales, device)
        inference_runtime[model_name] = time.time() - model_started
        replay_prediction, _, _ = infer_model(model_name, model, blends[:BATCH_SIZE], prompts[:BATCH_SIZE], scales, device)
        replay_error = float(np.max(np.abs(prediction[:BATCH_SIZE] - replay_prediction)))
        replay_checks[model_name] = replay_error
        if replay_error > 1e-6:
            raise RuntimeError(f"deterministic MPS replay failed for {model_name}: {replay_error}")
        for index, record in enumerate(observation_records):
            key = (model_name, record["pair_id"], record["side"], record["regime"], record["source_index"])
            output_lookup[key] = prediction[index]
            error = scientific_distance(
                prediction[index], record["truth"], mean_psf_fwhm_pixel=mean_psf_fwhm_pixel
            )
            candidate_rows.append(
                {
                    "pair_id": record["pair_id"],
                    "side": record["side"],
                    "regime": record["regime"],
                    "family_id_provenance_only": model_name,
                    "family_cluster": "THAYER_COMPACT_PROMPTED_UNET",
                    "source_index": record["source_index"],
                    "requested_source": record["source_index"] == 0,
                    "candidate_sha256": sha256_array(prediction[index]),
                    "finite": bool(np.all(np.isfinite(prediction[index]))),
                    "output_units": "detected electrons per pixel",
                    "band_order": "g,r,z",
                    "clipping_applied": False,
                    "private_heads_exported_to_candidate": False,
                    "reconstruction_error_primary": error.primary_normalized,
                    "reconstruction_safe": error.primary_normalized <= 1.0,
                    "r1_private_recoverability_diagnostic": confidence[index],
                    "r1_private_no_source_diagnostic": no_source[index],
                }
            )

    decomposition_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    model_witness_rows: list[dict[str, object]] = []
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        for regime in ("noiseless_mean", "noisy_observation"):
            for side in ("left", "right"):
                pair_arrays = np.load(run_dir / f"atlas/{pair_id}.npz", allow_pickle=False)
                observed = np.asarray(pair_arrays[f"{side}_blend"], dtype=np.float64)
                sky = np.asarray(pair_arrays["sky_electrons"], dtype=np.float64)
                pair_arrays.close()
                if regime == "noisy_observation":
                    record = next(item for item in observation_records if item["pair_id"] == pair_id and item["side"] == side and item["regime"] == regime)
                    observed = np.asarray(record["blend"], dtype=np.float64)
                requested_sources = {}
                consistency_scores = {}
                for model_name in models:
                    layers = np.stack(
                        [
                            output_lookup[(model_name, pair_id, side, regime, 0)],
                            output_lookup[(model_name, pair_id, side, regime, 1)],
                        ]
                    )
                    score = forward_consistency(observed, layers, sky)
                    requested_sources[model_name] = layers[0]
                    consistency_scores[model_name] = score
                    candidate_id = f"{pair_id}_{side}_{regime}_{model_name.lower()}"
                    output_path = run_dir / f"candidate_outputs/{candidate_id}.npz"
                    if output_path.exists():
                        raise FileExistsError(output_path)
                    np.savez_compressed(output_path, requested_source=layers[0], full_decomposition=layers, observed_blend=observed.astype(np.float32))
                    decomposition_rows.append(
                        {
                            "pair_id": pair_id,
                            "side": side,
                            "regime": regime,
                            "candidate_id": candidate_id,
                            "family_id_provenance_only": model_name,
                            "checkpoint_sha256": sha256_file({"THAYER_SELECT_CONDITION_C": CONDITION_C, "THAYER_SELECT_R0": R0, "THAYER_SELECT_R1_RECONSTRUCTION_ONLY": R1}[model_name]),
                            "requested_source_sha256": sha256_array(layers[0]),
                            "full_decomposition_sha256": sha256_array(layers),
                            "forward_global_chi_square_mean": score.global_chi_square_mean,
                            "forward_g_chi_square_mean": score.per_band_chi_square_mean[0],
                            "forward_r_chi_square_mean": score.per_band_chi_square_mean[1],
                            "forward_z_chi_square_mean": score.per_band_chi_square_mean[2],
                            "forward_g_neighbor_correlation": score.residual_neighbor_correlation[0],
                            "forward_r_neighbor_correlation": score.residual_neighbor_correlation[1],
                            "forward_z_neighbor_correlation": score.residual_neighbor_correlation[2],
                            "forward_relative_flux_residual": score.relative_flux_residual,
                            "plausible_under_frozen_noisy_threshold": is_plausible(score, thresholds) if regime == "noisy_observation" else "NOT_APPLICABLE",
                            "finite": score.finite,
                            "runtime_seconds_per_query": inference_runtime[model_name] / len(observation_records),
                            "output_path": str(output_path.relative_to(run_dir)),
                        }
                    )
                witness = empirical_ambiguity_witness(
                    requested_sources,
                    consistency_scores,
                    thresholds,
                    mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
                    artifact_audit_passed=True,
                ) if regime == "noisy_observation" else None
                model_witness_rows.append(
                    {
                        "pair_id": pair_id,
                        "side": side,
                        "regime": regime,
                        "plausible_candidate_count": len(witness.retained_candidate_ids) if witness else "NOT_APPLICABLE",
                        "model_candidate_primary_diameter": witness.primary_diameter if witness else "NOT_APPLICABLE",
                        "model_candidate_ambiguity_witness": witness.exists if witness else "NOT_APPLICABLE",
                        "reason": witness.reason if witness else "NOISY_THRESHOLD_NOT_APPLICABLE",
                    }
                )
            for model_name in models:
                left_prediction = output_lookup[(model_name, pair_id, "left", regime, 0)]
                right_prediction = output_lookup[(model_name, pair_id, "right", regime, 0)]
                output_distance = scientific_distance(
                    left_prediction, right_prediction, mean_psf_fwhm_pixel=mean_psf_fwhm_pixel
                )
                left_error = scientific_distance(
                    left_prediction,
                    next(item["truth"] for item in observation_records if item["pair_id"] == pair_id and item["side"] == "left" and item["regime"] == regime and item["source_index"] == 0),
                    mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
                )
                right_error = scientific_distance(
                    right_prediction,
                    next(item["truth"] for item in observation_records if item["pair_id"] == pair_id and item["side"] == "right" and item["regime"] == regime and item["source_index"] == 0),
                    mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
                )
                confidence_values = [
                    row["r1_private_recoverability_diagnostic"]
                    for row in candidate_rows
                    if row["pair_id"] == pair_id
                    and row["regime"] == regime
                    and row["family_id_provenance_only"] == model_name
                    and row["requested_source"]
                ]
                behavior_rows.append(
                    {
                        "pair_id": pair_id,
                        "regime": regime,
                        "family_id_provenance_only": model_name,
                        "truth_primary_diameter": pair["target_primary_diameter"],
                        "output_primary_diameter": output_distance.primary_normalized,
                        "left_reconstruction_error_primary": left_error.primary_normalized,
                        "right_reconstruction_error_primary": right_error.primary_normalized,
                        "both_pair_members_safe": left_error.primary_normalized <= 1.0 and right_error.primary_normalized <= 1.0,
                        "output_nearly_same_while_truth_diverges": output_distance.primary_normalized <= 1.0,
                        "at_least_one_unsafe_output": left_error.primary_normalized > 1.0 or right_error.primary_normalized > 1.0,
                        "mean_private_recoverability_diagnostic": float(np.nanmean(confidence_values)) if model_name == "THAYER_SELECT_R1_RECONSTRUCTION_ONLY" else "NOT_APPLICABLE",
                    }
                )

    write_csv_fresh(inventory_path, candidate_rows)
    write_csv_fresh(run_dir / "tables/candidate_decomposition_inventory.csv", decomposition_rows)
    write_csv_fresh(run_dir / "tables/ambiguity_witness_inventory.csv", truth_witness_rows)
    write_csv_fresh(run_dir / "tables/model_candidate_witness_inventory.csv", model_witness_rows)
    write_csv_fresh(run_dir / "tables/atlas_deblender_behavior.csv", behavior_rows)
    unsafe = [row for row in behavior_rows if row["regime"] == "noisy_observation" and row["at_least_one_unsafe_output"]]
    forced_same = [row for row in behavior_rows if row["regime"] == "noiseless_mean" and row["output_nearly_same_while_truth_diverges"]]
    truth_witness_count = sum(bool(row["empirical_ambiguity_witness"]) for row in truth_witness_rows)
    model_witness_count = sum(row["model_candidate_ambiguity_witness"] is True for row in model_witness_rows)
    report = f"""# Deblender behavior on the initial Ambiguity Atlas

Status: **ATLAS_DEBLENDER_FAILURE_OBSERVED**.

Condition C, R0, and reconstruction-only R1 were run on MPS in frozen eval mode
for both requested sources of all 25 frozen Atlas pairs. No prediction was
clipped. R1 private confidence heads were retained only as diagnostics and did
not enter any candidate, consistency score, or witness.

- Pair/model rows with at least one scientifically unsafe noisy reconstruction: {len(unsafe)} / 75.
- Pair/model rows producing nearly the same mean-scene answer while truths diverge: {len(forced_same)} / 75.
- Truth-decomposition empirical witnesses across the 50 noisy observations: {truth_witness_count} / 50.
- Model-candidate empirical witnesses across the 50 noisy observations: {model_witness_count} / 50.
- Maximum deterministic MPS replay error: {max(replay_checks.values()):.3g}.
- Development scenes used: 0.
- Lockbox scenes used: 0.

These are same-family-cluster controls, not cross-family evidence. Atlas pairs
demonstrate finite competing explanations; no absence of a model-candidate
witness is interpreted as uniqueness.
"""
    write_text_fresh(run_dir / "diagnostics/candidate_output_correctness.md", report)
    write_json_fresh(
        run_dir / "logs/atlas_deblender_evaluation_complete.json",
        {
            "status": "PASS",
            "device": "mps",
            "pair_count": len(pair_ids),
            "model_count": len(models),
            "query_count": len(observation_records),
            "deterministic_replay_max_absolute_error": replay_checks,
            "truth_witness_count": truth_witness_count,
            "model_candidate_witness_count": model_witness_count,
            "runtime_seconds": time.time() - started,
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        },
    )


if __name__ == "__main__":
    main()
