#!/usr/bin/env python3
"""Evaluate frozen non-Atlas promptability, set coverage, and control gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import prompts  # noqa: E402
from scripts.thayer_select_prompt_ablation_common import CompactSelectNet  # noqa: E402
from src.competing_hypotheses import (  # noqa: E402
    PlausibilityThresholds, forward_consistency, is_plausible, scientific_distance,
)
from src.models_multiple_hypotheses import ThayerMultipleHypotheses  # noqa: E402


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
PU_THRESHOLDS = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/manifests/forward_consistency_thresholds.json"
ATLAS_NOISE = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json"
BATCH_SIZE = 8
MEAN_PSF_FWHM_PIXEL = float(np.mean([0.86, 0.81, 0.77]) / 0.2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS-only inference unavailable")
    return torch.device("mps")


def load_models(run_dir: Path, device: torch.device) -> tuple[ThayerMultipleHypotheses, CompactSelectNet]:
    payload = torch.load(run_dir / "checkpoints/thayer_mh_best.pth", map_location="cpu", weights_only=False)
    model = ThayerMultipleHypotheses(); model.load_state_dict(payload["state_dict"], strict=True); model.to(device).eval()
    condition_payload = torch.load(CONDITION_C, map_location="cpu", weights_only=False)
    condition = CompactSelectNet(4); condition.load_state_dict(condition_payload["state_dict"], strict=True); condition.to(device).eval()
    return model, condition


def thresholds() -> tuple[PlausibilityThresholds, np.ndarray]:
    raw = json.loads(PU_THRESHOLDS.read_text())
    threshold = PlausibilityThresholds(float(raw["global_chi_square_mean"]), tuple(float(x) for x in raw["per_band_chi_square_mean"]), float(raw["absolute_relative_flux_residual"]), int(raw["calibration_scene_count"]), 0.99, 0.995, 0.99)
    sky = np.asarray(json.loads(ATLAS_NOISE.read_text())["sky_electrons_grz"], dtype=np.float64)
    return threshold, sky


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative)); ranks = rankdata(values, method="average"); n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def unordered_mse(left: np.ndarray, right: np.ndarray) -> float:
    identity = float(np.mean((left[0] - right[0]) ** 2) + np.mean((left[1] - right[1]) ** 2))
    swapped = float(np.mean((left[0] - right[1]) ** 2) + np.mean((left[1] - right[0]) ** 2))
    return min(identity, swapped)


def evaluate_partition(run_dir: Path, partition: str, model: ThayerMultipleHypotheses, condition: CompactSelectNet, device: torch.device, scales: np.ndarray, *, save_candidates: bool) -> tuple[list[dict[str, object]], np.ndarray | None]:
    rows = [row for row in read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv") if row["partition"] == partition]
    threshold, sky = thresholds()
    output_rows: list[dict[str, object]] = []
    candidate_array = np.empty((len(rows), 2, 2, 6, 60, 60), dtype=np.float32) if save_candidates else None
    with h5py.File(run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5", "r") as scene, h5py.File(run_dir / f"target_sets/thayer_mh_{partition}_target_sets.h5", "r") as target:
        for start in range(0, len(rows), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(rows))
            blend_physical = np.asarray(scene["blend"][start:stop], dtype=np.float32)
            isolated_physical = np.asarray(scene["isolated"][start:stop], dtype=np.float32)
            xy = np.asarray(scene["xy"][start:stop], dtype=np.float64)
            targets_physical = np.asarray(target["targets"][start:stop], dtype=np.float32)
            blend_norm = blend_physical / scales[None, :, None, None]
            prompt_a, prompt_b = prompts(xy)
            with torch.no_grad():
                blend = torch.from_numpy(np.ascontiguousarray(blend_norm)).to(device)
                pa = torch.from_numpy(np.ascontiguousarray(prompt_a)).to(device); pb = torch.from_numpy(np.ascontiguousarray(prompt_b)).to(device)
                joined = model(torch.cat((blend, blend)), torch.cat((pa, pb))).cpu().numpy()
                output_norm = np.stack((joined[:len(blend)], joined[len(blend):]), axis=1)
                condition_input = torch.cat((torch.cat((blend, pa), dim=1), torch.cat((blend, pb), dim=1)))
                condition_joined = condition(condition_input).cpu().numpy()
                condition_norm = np.stack((condition_joined[:len(blend)], condition_joined[len(blend):]), axis=1)
            output_physical = output_norm * np.tile(scales, 2)[None, None, None, :, None, None]
            if candidate_array is not None: candidate_array[start:stop] = output_physical
            for local, row in enumerate(rows[start:stop]):
                identities: list[bool] = []
                plausible = np.zeros((2, 2), dtype=bool); own_cover = np.zeros((2, 2), dtype=bool); alt_cover = np.zeros((2, 2), dtype=bool)
                diameters: list[float] = []
                own_mse: list[float] = []; condition_mse: list[float] = []
                for prompt_index in (0, 1):
                    requested_truth_norm = isolated_physical[local, prompt_index] / scales[:, None, None]
                    companion_truth_norm = isolated_physical[local, 1 - prompt_index] / scales[:, None, None]
                    for hypothesis in (0, 1):
                        requested_norm = output_norm[local, prompt_index, hypothesis, :3]
                        mse_own = float(np.mean((requested_norm - requested_truth_norm) ** 2)); mse_alt_source = float(np.mean((requested_norm - companion_truth_norm) ** 2))
                        identities.append(mse_own < mse_alt_source); own_mse.append(mse_own)
                        candidate = output_physical[local, prompt_index, hypothesis]
                        fc = forward_consistency(blend_physical[local], np.stack((candidate[:3], candidate[3:])), sky)
                        plausible[prompt_index, hypothesis] = is_plausible(fc, threshold)
                        own_distance = scientific_distance(candidate[:3], targets_physical[local, prompt_index, 0, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
                        own_cover[prompt_index, hypothesis] = plausible[prompt_index, hypothesis] and own_distance <= 1.0
                        if row["kind"] == "near_collision":
                            alt_distance = scientific_distance(candidate[:3], targets_physical[local, prompt_index, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized
                            alt_cover[prompt_index, hypothesis] = plausible[prompt_index, hypothesis] and alt_distance <= 1.0
                    condition_mse.append(float(np.mean((condition_norm[local, prompt_index] - requested_truth_norm) ** 2)))
                    diameters.append(scientific_distance(output_physical[local, prompt_index, 0, :3], output_physical[local, prompt_index, 1, :3], mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL).primary_normalized)
                both_mode = [bool((own_cover[q, 0] and alt_cover[q, 1]) or (own_cover[q, 1] and alt_cover[q, 0])) for q in (0, 1)]
                swap_consistency = unordered_mse(np.concatenate((output_norm[local, 0, :, 3:], output_norm[local, 0, :, :3]), axis=1), output_norm[local, 1])
                output_rows.append({
                    "scene_id": row["scene_id"], "partition": partition, "kind": row["kind"], "near_collision_pair_id": row["near_collision_pair_id"], "near_collision_pair_side": row["near_collision_pair_side"],
                    "token0_prompt_swap_success": bool(identities[0] and identities[2]), "token1_prompt_swap_success": bool(identities[1] and identities[3]), "set_level_prompt_swap_success": bool(all(identities)),
                    "prompt_a_identity_token0": identities[0], "prompt_a_identity_token1": identities[1], "prompt_b_identity_token0": identities[2], "prompt_b_identity_token1": identities[3],
                    "mean_requested_mse_normalized": float(np.mean(own_mse)), "condition_c_requested_mse_normalized": float(np.mean(condition_mse)), "prompt_swap_set_mse": swap_consistency,
                    "forward_consistent_fraction": float(plausible.mean()), "both_hypotheses_forward_consistent": bool(plausible.all()),
                    "own_truth_coverage": bool(own_cover.any(axis=1).all()), "both_hypothesis_own_truth_coverage": bool(own_cover.all()),
                    "alternate_truth_coverage": bool(alt_cover.any(axis=1).all()) if row["kind"] == "near_collision" else False,
                    "both_mode_coverage": bool(all(both_mode)) if row["kind"] == "near_collision" else False,
                    "primary_set_diameter": float(np.mean(diameters)), "model_generated_witness": bool(plausible.all() and np.mean(diameters) > 1.0),
                    "finite": bool(np.isfinite(output_physical[local]).all()),
                })
            print(json.dumps({"phase": partition, "completed": stop, "total": len(rows)}, sort_keys=True), flush=True)
    return output_rows, candidate_array


def bootstrap_ratio(near_rows: list[dict[str, object]], ordinary: np.ndarray) -> tuple[float, float, float]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in near_rows: by_pair[str(row["near_collision_pair_id"])].append(float(row["primary_set_diameter"]))
    pair_values = np.asarray([np.mean(values) for values in by_pair.values()]); rng = np.random.default_rng(2026079701); values = []
    for _ in range(500):
        near_sample = rng.choice(pair_values, len(pair_values), replace=True); ordinary_sample = rng.choice(ordinary, len(ordinary), replace=True)
        values.append(float(np.median(near_sample) / max(np.median(ordinary_sample), 1e-12)))
    return float(np.median(pair_values) / max(np.median(ordinary), 1e-12)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run_dir = args.run_dir.resolve()
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    if training["status"] != "PASS" or training["fallback"] or any((run_dir / "atlas_evaluation").iterdir()): raise RuntimeError("training/Atlas precondition failed")
    device = require_mps(); scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32); model, condition = load_models(run_dir, device); started = time.time()
    validation, candidates = evaluate_partition(run_dir, "validation", model, condition, device, scales, save_candidates=True)
    with h5py.File(run_dir / "candidate_sets/non_atlas_validation_candidates.h5", "x") as handle:
        handle.create_dataset("decompositions", data=candidates, compression="lzf", chunks=(1, 1, 1, 6, 60, 60)); handle.attrs["complete"] = True
    ordinary = [row for row in validation if row["kind"] == "ordinary"]; near = [row for row in validation if row["kind"] == "near_collision"]
    prompt_rates = [float(np.mean([bool(row[f"token{token}_prompt_swap_success"]) for row in ordinary])) for token in (0, 1)]
    set_prompt = float(np.mean([bool(row["set_level_prompt_swap_success"]) for row in ordinary])); reconstruction_ratio = float(np.mean([float(row["mean_requested_mse_normalized"]) for row in ordinary]) / max(np.mean([float(row["condition_c_requested_mse_normalized"]) for row in ordinary]), 1e-12))
    prompt_gates = [
        {"gate": "token0_prompt_swap", "threshold": ">=0.80", "observed": prompt_rates[0], "pass": prompt_rates[0] >= 0.80},
        {"gate": "token1_prompt_swap", "threshold": ">=0.80", "observed": prompt_rates[1], "pass": prompt_rates[1] >= 0.80},
        {"gate": "set_level_prompt_swap", "threshold": ">=0.90", "observed": set_prompt, "pass": set_prompt >= 0.90},
        {"gate": "source_confusion_each_token", "threshold": "<=0.20", "observed": ";".join(str(1 - value) for value in prompt_rates), "pass": all(1 - value <= 0.20 for value in prompt_rates)},
        {"gate": "reconstruction_factor_to_condition_c", "threshold": "<=3.0", "observed": reconstruction_ratio, "pass": reconstruction_ratio <= 3.0},
    ]
    write_csv_fresh(run_dir / "tables/non_atlas_promptability_gates.csv", prompt_gates); write_csv_fresh(run_dir / "tables/non_atlas_validation_per_scene.csv", validation)
    prompt_pass = all(bool(row["pass"]) for row in prompt_gates)
    write_text_fresh(run_dir / "diagnostics/non_atlas_promptability.md", f"# Non-Atlas promptability\n\nStatus: **{'PASS' if prompt_pass else 'FAIL — STOP; ATLAS PROHIBITED'}**. Token prompt-swap rates are {prompt_rates[0]:.6f} / {prompt_rates[1]:.6f}; set-level rate {set_prompt:.6f}; reconstruction ratio to Condition C {reconstruction_ratio:.6f}.\n")
    if not prompt_pass:
        write_json_fresh(run_dir / "logs/pre_atlas_evaluation_complete.json", {"status": "FAIL_PROMPTABILITY", "atlas_authorized": False, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "runtime_seconds": time.time() - started}); return
    ordinary_diameter = np.asarray([float(row["primary_set_diameter"]) for row in ordinary]); near_diameter = np.asarray([float(row["primary_set_diameter"]) for row in near])
    own_near = float(np.mean([bool(row["own_truth_coverage"]) for row in near])); alt_near = float(np.mean([bool(row["alternate_truth_coverage"]) for row in near])); both_near = float(np.mean([bool(row["both_mode_coverage"]) for row in near]))
    own_ordinary = float(np.mean([bool(row["own_truth_coverage"]) for row in ordinary])); ordinary_forward = float(np.mean([float(row["forward_consistent_fraction"]) for row in ordinary])); near_forward = float(np.mean([float(row["forward_consistent_fraction"]) for row in near]))
    set_gates = [
        {"gate": "ordinary_own_truth_coverage", "threshold": ">0", "observed": own_ordinary, "pass": own_ordinary > 0},
        {"gate": "near_own_truth_coverage", "threshold": ">0", "observed": own_near, "pass": own_near > 0},
        {"gate": "near_alternate_truth_coverage", "threshold": ">0", "observed": alt_near, "pass": alt_near > 0},
        {"gate": "near_both_mode_coverage", "threshold": ">0", "observed": both_near, "pass": both_near > 0},
        {"gate": "ordinary_forward_consistency", "threshold": ">=0.50", "observed": ordinary_forward, "pass": ordinary_forward >= 0.50},
        {"gate": "near_forward_consistency", "threshold": ">=0.50", "observed": near_forward, "pass": near_forward >= 0.50},
    ]
    write_csv_fresh(run_dir / "tables/non_atlas_set_coverage_gates.csv", set_gates)
    set_pass = all(bool(row["pass"]) for row in set_gates)
    write_text_fresh(run_dir / "diagnostics/non_atlas_set_coverage.md", f"# Non-Atlas set coverage\n\nStatus: **{'PASS' if set_pass else 'FAIL — STOP; ATLAS PROHIBITED'}**. Ordinary own coverage {own_ordinary:.6f}; near own / alternate / both-mode coverage {own_near:.6f} / {alt_near:.6f} / {both_near:.6f}; ordinary / near forward-consistent fractions {ordinary_forward:.6f} / {near_forward:.6f}.\n")
    if not set_pass:
        write_json_fresh(run_dir / "logs/pre_atlas_evaluation_complete.json", {"status": "FAIL_SET_COVERAGE", "atlas_authorized": False, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "runtime_seconds": time.time() - started}); return
    calibration, _ = evaluate_partition(run_dir, "calibration", model, condition, device, scales, save_candidates=False)
    calibration_ordinary = np.asarray([float(row["primary_set_diameter"]) for row in calibration if row["kind"] == "ordinary"]); operating = float(np.quantile(calibration_ordinary, 0.95, method="higher"))
    false_witness = float(np.mean([bool(row["model_generated_witness"]) for row in ordinary])); recall = float(np.mean(near_diameter > operating)); ratio, ci_low, ci_high = bootstrap_ratio(near, ordinary_diameter)
    pair_consistency = []
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(validation):
        if row["kind"] == "near_collision": by_pair[str(row["near_collision_pair_id"])].append(index)
    for indices in by_pair.values():
        pair_consistency.append(0.5 * (unordered_mse(candidates[indices[0], 0], candidates[indices[1], 0]) + unordered_mse(candidates[indices[0], 1], candidates[indices[1], 1])))
    median_pair_consistency = float(np.median(pair_consistency)); diameter_auc = auc(near_diameter, ordinary_diameter)
    control_gates = [
        {"gate": "ordinary_false_witness", "threshold": "<=0.10", "observed": false_witness, "pass": false_witness <= 0.10},
        {"gate": "near_control_diameter_ratio", "threshold": ">=1.25 and bootstrap lower >1", "observed": ratio, "bootstrap_low": ci_low, "pass": ratio >= 1.25 and ci_low > 1.0},
        {"gate": "low_fpr_near_recall", "threshold": ">0", "observed": recall, "pass": recall > 0},
        {"gate": "pair_set_consistency", "threshold": "median <=0.10 normalized MSE", "observed": median_pair_consistency, "pass": median_pair_consistency <= 0.10},
    ]
    write_csv_fresh(run_dir / "tables/control_concentration_gates.csv", control_gates)
    write_csv_fresh(run_dir / "tables/calibration_control_metrics.csv", calibration)
    control_pass = all(bool(row["pass"]) for row in control_gates)
    write_json_fresh(run_dir / "manifests/non_atlas_operating_threshold.json", {"calibration_control_95th_percentile": operating, "calibration_ordinary_count": len(calibration_ordinary), "candidate_diameter_auroc": diameter_auc, "near_recall": recall, "ordinary_false_witness": false_witness, "near_control_median_ratio": ratio, "bootstrap_95": [ci_low, ci_high], "pair_set_consistency_median": median_pair_consistency})
    write_text_fresh(run_dir / "diagnostics/control_concentration.md", f"# Control concentration\n\nStatus: **{'PASS' if control_pass else 'FAIL — STOP; ATLAS PROHIBITED'}**. Ordinary false witnesses {false_witness:.6f}; near/control median diameter ratio {ratio:.6f} (bootstrap 95% [{ci_low:.6f}, {ci_high:.6f}]); near recall at the calibration-control 95th percentile {recall:.6f}; diameter AUROC {diameter_auc:.6f}; median pair-set consistency {median_pair_consistency:.6g}.\n")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(ordinary_diameter, bins=40, alpha=0.6, label="ordinary"); ax.hist(near_diameter, bins=40, alpha=0.6, label="near-collision"); ax.axvline(operating, color="black", linestyle="--", label="calibration 95th"); ax.set_xlabel("primary hypothesis-set diameter"); ax.set_ylabel("count"); ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "figures/non_atlas_set_diameter.png", dpi=160); plt.close(fig)
    write_json_fresh(run_dir / "logs/pre_atlas_evaluation_complete.json", {"status": "PASS" if control_pass else "FAIL_CONTROL_CONCENTRATION", "atlas_authorized": control_pass, "promptability_pass": True, "set_coverage_pass": True, "control_concentration_pass": control_pass, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "runtime_seconds": time.time() - started})


if __name__ == "__main__": main()
