#!/usr/bin/env python3
"""MPS-only frozen Condition-C feature extraction for hierarchical safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.hierarchical_safety import RISK_LIMITS, metric_specific_risks, normalized_policy_violation
from thayer_select_prompt_ablation_common import CompactSelectNet


PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
CHECKPOINT = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PHASE1 / "manifests/normalization.json"
DATASETS = ("q_training", "q_validation", "r_training", "r_validation", "natural_calibration", "stratified_calibration")
ARTIFACT_VERSION = "v2"
IMAGE_SIZE = 60
PSF_FWHM_PIXELS = 4.066666666666666
BATCH_SIZE = 128


def artifact_stem(dataset: str) -> str:
    return f"{ARTIFACT_VERSION}_{dataset}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    frame.to_csv(path, index=False)


def load_model() -> CompactSelectNet:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing CPU neural fallback")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("condition") != "C_randomized_coordinate_prompt" or payload.get("selection") != "minimum validation MSE only":
        raise RuntimeError("Unexpected Condition-C checkpoint metadata")
    model = CompactSelectNet(4)
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model.to("mps")


def weighted_pool(feature: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
    weights = F.interpolate(prompt, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-8)
    return (feature * weights).sum(dim=(-2, -1)) / denominator


def frozen_forward(model: CompactSelectNet, image: torch.Tensor, prompt: torch.Tensor):
    inputs = torch.cat((image, prompt), dim=1)
    enc1 = model.enc1(inputs)
    enc2 = model.enc2(F.avg_pool2d(enc1, 2))
    bottleneck = model.bottleneck(F.avg_pool2d(enc2, 2))
    up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
    dec2 = model.dec2(torch.cat((up2, enc2), dim=1))
    up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
    dec1 = model.dec1(torch.cat((up1, enc1), dim=1))
    reconstruction = model.reconstruction_head(dec1)
    global_feature = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
    prompt_local = torch.cat((weighted_pool(enc1, prompt), weighted_pool(enc2, prompt), weighted_pool(bottleneck, prompt)), dim=1)
    return reconstruction, global_feature, prompt_local


def reconstruction_summary(prediction: np.ndarray, blend: np.ndarray, prompt_xy: np.ndarray) -> np.ndarray:
    batch, bands, height, width = prediction.shape
    yy, xx = np.mgrid[:height, :width]
    rows = []
    for index in range(batch):
        pred = np.asarray(prediction[index], dtype=np.float64)
        image = np.asarray(blend[index], dtype=np.float64)
        prompt = prompt_xy[index]
        radius = np.sqrt((xx - prompt[0]) ** 2 + (yy - prompt[1]) ** 2)
        aperture = radius <= 2.0 * PSF_FWHM_PIXELS
        flux = pred.sum(axis=(-2, -1))
        absolute = np.abs(pred)
        concentration = absolute[:, aperture].sum(axis=1) / np.maximum(absolute.sum(axis=(-2, -1)), 1e-30)
        offsets = []
        for band in range(bands):
            weight = np.maximum(pred[band], 0.0); total = float(weight.sum())
            if total <= 0 or not np.isfinite(total):
                offsets.extend((0.0, 0.0))
            else:
                cx = float(np.sum(xx * weight) / total); cy = float(np.sum(yy * weight) / total)
                offsets.extend(((cx - prompt[0]) / PSF_FWHM_PIXELS, (cy - prompt[1]) / PSF_FWHM_PIXELS))
        energy = np.sqrt(np.mean(pred**2, axis=(-2, -1)))
        local_pred = np.mean(np.abs(pred[:, aperture]), axis=1)
        local_input = np.mean(np.abs(image[:, aperture]), axis=1)
        contrast = local_pred / np.maximum(local_input, 1e-30)
        rows.append(np.concatenate((flux, concentration, np.asarray(offsets), energy, contrast)))
    return np.asarray(rows, dtype=np.float32)


def fit_flux_floors(run: Path) -> np.ndarray:
    path = run / f"manifests/{artifact_stem('r_training')}_scenes.h5"
    values = []
    with h5py.File(path, "r") as handle:
        for start in range(0, len(handle["isolated"]), 512):
            isolated = np.asarray(handle["isolated"][start:start + 512], dtype=np.float64)
            matched = np.asarray(handle["matched_index"][start:start + 512], dtype=int)
            selected = isolated[np.arange(len(isolated)), matched]
            values.append(np.abs(selected.sum(axis=(-2, -1))))
    median = np.median(np.concatenate(values), axis=0)
    floors = 0.001 * median
    if floors.shape != (3,) or np.any(~np.isfinite(floors)) or np.any(floors <= 0):
        raise RuntimeError("Invalid train-only flux floors")
    return floors.astype(np.float64)


def extraction_audit(model: CompactSelectNet, run: Path, scales: np.ndarray, checkpoint_before: str) -> dict:
    with h5py.File(run / f"manifests/{artifact_stem('q_validation')}_scenes.h5", "r") as handle:
        blend = np.asarray(handle["blend"][:64], dtype=np.float32)
        prompt = np.asarray(handle["prompt"][:64], dtype=np.float32)
    image = torch.from_numpy(np.ascontiguousarray(blend / scales[None, :, None, None])).to("mps")
    prompt_tensor = torch.from_numpy(np.ascontiguousarray(prompt)).to("mps")
    with torch.no_grad():
        first = frozen_forward(model, image, prompt_tensor)
        second = frozen_forward(model, image, prompt_tensor)
    differences = [float(torch.max(torch.abs(left - right)).cpu()) for left, right in zip(first, second)]
    return {
        "trainable_reconstruction_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_training_flag": model.training,
        "parameter_gradient_count": sum(parameter.grad is not None for parameter in model.parameters()),
        "output_requires_grad": any(value.requires_grad for value in first),
        "deterministic_exact": all(torch.equal(left, right) for left, right in zip(first, second)),
        "deterministic_max_abs_differences": differences,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": sha256_file(CHECKPOINT),
        "device": str(first[0].device),
        "global_dimension": int(first[1].shape[1]),
        "prompt_local_dimension": int(first[2].shape[1]),
        "prompt_local_scales": [60, 30, 15],
    }


def extract_dataset(run: Path, dataset: str, model: CompactSelectNet, scales: np.ndarray, floors: np.ndarray) -> dict:
    stem = artifact_stem(dataset)
    # Preserve the literal scientific class name NULL.
    manifest = pd.read_csv(run / f"manifests/{stem}_scene_manifest.csv", keep_default_na=False, low_memory=False)
    h5_path = run / f"manifests/{stem}_scenes.h5"
    reconstruction_path = run / f"features/{stem}_frozen_reconstructions.h5"
    feature_path = run / f"features/{stem}_features.npz"
    sample_path = run / f"features/{stem}_samples.csv"
    global_blocks = []; local_blocks = []; summary_blocks = []; sample_rows = []
    with h5py.File(h5_path, "r") as source, h5py.File(reconstruction_path, "x") as output, torch.no_grad():
        recon_store = output.create_dataset("reconstruction", shape=(len(manifest), 3, IMAGE_SIZE, IMAGE_SIZE), dtype="f4", chunks=(1, 3, IMAGE_SIZE, IMAGE_SIZE), compression="lzf")
        output.attrs["condition_c_checkpoint_sha256"] = sha256_file(CHECKPOINT)
        output.attrs["complete"] = False; output.attrs["completed_count"] = 0
        for start in range(0, len(manifest), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(manifest))
            blend = np.asarray(source["blend"][start:stop], dtype=np.float32)
            prompt = np.asarray(source["prompt"][start:stop], dtype=np.float32)
            prompt_xy = np.asarray(source["prompt_xy"][start:stop], dtype=np.float64)
            image = torch.from_numpy(np.ascontiguousarray(blend / scales[None, :, None, None])).to("mps")
            prompt_tensor = torch.from_numpy(np.ascontiguousarray(prompt)).to("mps")
            prediction_normalized, global_feature, prompt_local = frozen_forward(model, image, prompt_tensor)
            if prediction_normalized.device.type != "mps" or not torch.isfinite(prediction_normalized).all():
                raise RuntimeError("Invalid or non-MPS frozen inference")
            prediction = prediction_normalized.cpu().numpy() * scales[None, :, None, None]
            global_np = global_feature.cpu().numpy().astype(np.float32)
            local_np = prompt_local.cpu().numpy().astype(np.float32)
            summary_np = reconstruction_summary(prediction, blend, prompt_xy)
            recon_store[start:stop] = prediction.astype(np.float32)
            global_blocks.append(global_np); local_blocks.append(local_np); summary_blocks.append(summary_np)
            isolated = np.asarray(source["isolated"][start:stop], dtype=np.float32)
            matched = np.asarray(source["matched_index"][start:stop], dtype=int)
            for local_index, row_index in enumerate(range(start, stop)):
                row = manifest.iloc[row_index]
                applicable = row.query_state == "UNIQUE_VALID" and matched[local_index] in (0, 1)
                if applicable:
                    requested = isolated[local_index, matched[local_index]]; alternate = isolated[local_index, 1 - matched[local_index]]
                    risks = metric_specific_risks(prediction[local_index], requested, alternate, flux_floor_by_band=floors)
                    band_risks = risks["flux_risk_by_band"]
                    violation = {name: normalized_policy_violation(risks, limits) for name, limits in RISK_LIMITS.items()}
                else:
                    risks = {"image_risk": math.nan, "flux_risk_max": math.nan, "centroid_risk_pixels": math.nan, "centroid_risk_psf": math.nan, "confusion_risk": False}
                    band_risks = [math.nan, math.nan, math.nan]; violation = {name: math.nan for name in RISK_LIMITS}
                sample_rows.append({
                    "scene_id": row.scene_id, "dataset": dataset, "source_partition": row.source_partition, "query_state": row.query_state,
                    "sampling_stratum": row.sampling_stratum, "source_a_id": row.source_a_id, "source_b_id": row.source_b_id,
                    "source_a_group": row.source_a_group, "source_b_group": row.source_b_group, "matched_source_id": row.matched_source_id,
                    "matched_source_group": row.matched_source_group, "applicable_valid_risk": int(applicable),
                    "image_risk": risks["image_risk"], "flux_risk_g": band_risks[0], "flux_risk_r": band_risks[1], "flux_risk_z": band_risks[2],
                    "flux_risk_max": risks["flux_risk_max"], "centroid_risk_pixels": risks["centroid_risk_pixels"], "centroid_risk_psf": risks["centroid_risk_psf"],
                    "confusion_risk": int(risks["confusion_risk"]), "image_target_log1p": np.log1p(risks["image_risk"]) if applicable else math.nan,
                    "flux_target_log1p": np.log1p(risks["flux_risk_max"]) if applicable else math.nan,
                    "centroid_target_log1p": np.log1p(risks["centroid_risk_pixels"]) if applicable else math.nan,
                    "violation_strict": violation["strict"], "violation_moderate": violation["moderate"], "violation_permissive": violation["permissive"],
                    "frozen_reconstruction_sha256": sha256_array(prediction[local_index].astype(np.float32)),
                    "blend_sha256": row.blend_sha256, "prompt_sha256": row.prompt_sha256,
                })
            output.attrs["completed_count"] = stop
            if stop % 1024 == 0 or stop == len(manifest):
                output.flush(); print(f"{dataset}: {stop}/{len(manifest)}", flush=True)
        output.attrs["complete"] = True
    global_values = np.concatenate(global_blocks); local_values = np.concatenate(local_blocks); summary_values = np.concatenate(summary_blocks)
    combined = np.concatenate((global_values, local_values, summary_values), axis=1).astype(np.float32)
    if any(not np.isfinite(values).all() for values in (global_values, local_values, summary_values, combined)):
        raise RuntimeError(f"Nonfinite deployable features in {dataset}")
    if feature_path.exists():
        raise FileExistsError(feature_path)
    np.savez_compressed(feature_path, scene_id=manifest.scene_id.astype(str).to_numpy(), f_global=global_values, f_prompt_local=local_values, f_recon_summary=summary_values, f_combined=combined)
    samples = pd.DataFrame(sample_rows)
    if samples.scene_id.tolist() != manifest.scene_id.tolist():
        raise RuntimeError(f"Feature/sample alignment failed for {dataset}")
    write_csv_fresh(sample_path, samples)
    return {
        "dataset": dataset, "samples": len(manifest), "global_dimension": global_values.shape[1], "prompt_local_dimension": local_values.shape[1],
        "reconstruction_summary_dimension": summary_values.shape[1], "combined_dimension": combined.shape[1],
        "scene_id_sha256": sha256_array(manifest.scene_id.astype(str).to_numpy()), "global_sha256": sha256_array(global_values),
        "prompt_local_sha256": sha256_array(local_values), "reconstruction_summary_sha256": sha256_array(summary_values), "combined_sha256": sha256_array(combined),
        "feature_file": relative(feature_path), "feature_file_sha256": sha256_file(feature_path), "sample_file": relative(sample_path),
        "sample_file_sha256": sha256_file(sample_path), "reconstruction_file": relative(reconstruction_path), "reconstruction_file_sha256": sha256_file(reconstruction_path),
        "applicable_risk_rows": int(samples.applicable_valid_risk.sum()), "confusion_events": int(samples.confusion_risk.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if json.loads((run / "logs/data_preparation_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Fresh-data gate did not pass")
    if (run / "logs/feature_extraction_complete.json").exists():
        raise FileExistsError("Feature extraction already completed")
    started = time.time(); checkpoint_before = sha256_file(CHECKPOINT)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise RuntimeError("Invalid frozen normalization")
    floors = fit_flux_floors(run)
    write_json_fresh(run / "manifests/risk_flux_floors.json", {
        "fit_partition": "v2_r_training UNIQUE_VALID only", "fraction_of_median_absolute_band_flux": 0.001,
        "bands": ["g", "r", "z"], "floor_by_band": floors.tolist(), "development_used": False, "calibration_used": False,
    })
    model = load_model(); audit = extraction_audit(model, run, scales, checkpoint_before)
    if audit["trainable_reconstruction_parameters"] or audit["model_training_flag"] or audit["parameter_gradient_count"] or audit["output_requires_grad"] or not audit["deterministic_exact"] or audit["checkpoint_sha256_before"] != audit["checkpoint_sha256_after"] or audit["device"] != "mps:0":
        raise RuntimeError(f"Frozen feature-extraction audit failed: {audit}")
    write_json_fresh(run / "diagnostics/frozen_feature_extraction_audit.json", audit)
    rows = [extract_dataset(run, dataset, model, scales, floors) for dataset in DATASETS]
    write_csv_fresh(run / "tables/frozen_feature_inventory.csv", pd.DataFrame(rows))
    if sha256_file(CHECKPOINT) != checkpoint_before:
        raise RuntimeError("Condition-C checkpoint changed during feature extraction")
    write_json_fresh(run / "logs/feature_extraction_complete.json", {
        "status": "PASS", "device": "mps", "cpu_fallback": False, "datasets": list(DATASETS), "total_samples": sum(row["samples"] for row in rows),
        "runtime_seconds": time.time() - started, "condition_c_checkpoint_sha256": checkpoint_before,
        "development_accessed": False, "lockbox_accessed": False, "completed_at_unix": time.time(),
    })
    print(relative(run))


if __name__ == "__main__":
    main()
