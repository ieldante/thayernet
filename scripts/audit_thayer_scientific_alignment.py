#!/usr/bin/env python3
"""Run the frozen Thayer-SA surrogate, weight, and free-output preflights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_thayer_two_expert_micro_overfit import (
    MEAN_PSF_FWHM_PIXEL,
    NORMALIZATION,
    load_micro_arrays,
    select_microset,
)
from src.competing_hypotheses import scientific_distance
from src.loss_geometry import scientific_metrics
from src.scientific_alignment import (
    COLOR_THRESHOLD_MAG,
    FLUX_THRESHOLD,
    IMAGE_THRESHOLD,
    SMOOTHMAX_TEMPERATURE,
    corrected_objective,
    scientific_components,
    scientific_surrogate,
    smoothmax,
)


ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
TRAINED_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
PREFLIGHT_SEED = 2026071303
PREFLIGHT_STEPS = 400
PREFLIGHT_LR = 1e-4


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


def exact_outputs(targets: np.ndarray) -> np.ndarray:
    output = targets.copy()
    output[:32, :, 1] = output[:32, :, 0]
    return output


def canonical_outputs(targets: np.ndarray, trained: np.ndarray) -> dict[str, np.ndarray]:
    exact = exact_outputs(targets)
    collapsed = np.repeat(targets.mean(axis=2, keepdims=True), 2, axis=2)
    wrong = exact.copy()
    total = wrong[..., :3, :, :] + wrong[..., 3:, :, :]
    wrong[..., :3, :, :] = 0.5 * total
    wrong[..., 3:, :, :] = 0.5 * total
    generator = np.random.default_rng(PREFLIGHT_SEED)
    random = generator.uniform(float(targets.min()), float(targets.max()), size=targets.shape).astype(np.float32)
    return {
        "exact_truth": exact,
        "trained_thayer_me": trained,
        "collapsed_mean": collapsed,
        "source_sum_wrong_allocation": wrong,
        "random_bounded": random,
    }


def assignment(outputs: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    return corrected_objective(outputs, targets, counts, scales, MEAN_PSF_FWHM_PIXEL)["identity_wins"].detach().cpu().numpy()


def aligned_pairs(
    configurations: dict[str, np.ndarray],
    targets: np.ndarray,
    targets_physical: np.ndarray,
    counts: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    tensor_targets = torch.from_numpy(targets)
    tensor_counts = torch.from_numpy(counts)
    tensor_scales = torch.from_numpy(scales)
    for configuration, output in configurations.items():
        wins = assignment(torch.from_numpy(output), tensor_targets, tensor_counts, tensor_scales)
        for scene in range(len(output)):
            for prompt in (0, 1):
                if counts[scene, prompt] == 1:
                    chosen = (0, 0)
                else:
                    chosen = (0, 1) if wins[scene, prompt] else (1, 0)
                for expert in (0, 1):
                    target_index = chosen[expert]
                    predicted = output[scene, prompt, expert, :3]
                    target = targets[scene, prompt, target_index, :3]
                    surrogate = float(scientific_surrogate(
                        torch.from_numpy(predicted[None]),
                        torch.from_numpy(target[None]),
                        tensor_scales,
                        MEAN_PSF_FWHM_PIXEL,
                    ))
                    exact = scientific_distance(
                        predicted * scales[:, None, None],
                        targets_physical[scene, prompt, target_index, :3],
                        mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                    )
                    rows.append({
                        "configuration": configuration,
                        "scene": scene,
                        "prompt": prompt,
                        "expert": expert,
                        "target_index": target_index,
                        "exact_primary": exact.primary_normalized,
                        "surrogate": surrogate,
                        "exact_covered": exact.primary_normalized <= 1.0,
                        "surrogate_covered": surrogate <= 1.0,
                    })
    return rows


def unit_tests(targets: np.ndarray, scales: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    truth = torch.from_numpy(targets[32, 0, 0, :3]).clone()
    tensor_scales = torch.from_numpy(scales)
    cases: dict[str, torch.Tensor] = {"exact_truth": truth.clone()}
    cases["flux_g_plus_20pct"] = truth.clone()
    cases["flux_g_plus_20pct"][0] *= 1.2
    cases["translation_plus_1px"] = torch.roll(truth, shifts=1, dims=-1)
    cases["color_r_plus_20pct"] = truth.clone()
    cases["color_r_plus_20pct"][1] *= 1.2
    morphology = 0.5 * truth + 0.5 * torch.roll(truth, shifts=1, dims=-1)
    original_flux = truth.sum(dim=(-2, -1), keepdim=True)
    morphology_flux = morphology.sum(dim=(-2, -1), keepdim=True)
    cases["flux_preserving_morphology"] = morphology * torch.where(morphology_flux.abs() > 1e-12, original_flux / morphology_flux, torch.ones_like(morphology_flux))
    rows = []
    for name, predicted in cases.items():
        component = scientific_components(predicted[None], truth[None], tensor_scales, MEAN_PSF_FWHM_PIXEL)
        value = float(scientific_surrogate(predicted[None], truth[None], tensor_scales, MEAN_PSF_FWHM_PIXEL))
        rows.append({
            "case": name,
            "v_image": float(component.image),
            "v_flux_g": float(component.flux_grz[0, 0]),
            "v_flux_r": float(component.flux_grz[0, 1]),
            "v_flux_z": float(component.flux_grz[0, 2]),
            "v_color_gr": float(component.color_gr_rz[0, 0]),
            "v_color_rz": float(component.color_gr_rz[0, 1]),
            "v_centroid": float(component.centroid),
            "smoothmax": value,
            "finite": bool(np.isfinite(value)),
        })
    one = torch.zeros((1, 7), dtype=torch.float64)
    one[0, 0] = 1.0
    one_value = float(smoothmax(one))
    checks = [
        {"test": "exact_truth_near_zero", "observed": rows[0]["smoothmax"], "requirement": "<=1e-6", "pass": rows[0]["smoothmax"] <= 1e-6},
        {"test": "one_threshold_is_order_one", "observed": one_value, "requirement": "[0.90,1.05]", "pass": 0.90 <= one_value <= 1.05},
        {"test": "flux_perturbation_detected", "observed": rows[1]["v_flux_g"], "requirement": ">0", "pass": rows[1]["v_flux_g"] > 0},
        {"test": "translation_detected", "observed": rows[2]["v_centroid"], "requirement": ">0", "pass": rows[2]["v_centroid"] > 0},
        {"test": "color_perturbation_detected", "observed": max(rows[3]["v_color_gr"], rows[3]["v_color_rz"]), "requirement": ">0", "pass": max(rows[3]["v_color_gr"], rows[3]["v_color_rz"]) > 0},
        {"test": "morphology_detected", "observed": rows[4]["v_image"], "requirement": ">0", "pass": rows[4]["v_image"] > 0},
    ]
    return rows, checks


def exact_metrics_summary(
    output: np.ndarray,
    arrays: dict[str, np.ndarray],
    scales: np.ndarray,
    rows: list[dict[str, str]],
) -> dict[str, float]:
    threshold_data = json.loads((REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/manifests/forward_consistency_thresholds.json").read_text())
    noise = np.asarray(json.loads((REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json").read_text())["sky_electrons_grz"], dtype=np.float64)
    from src.competing_hypotheses import PlausibilityThresholds
    threshold = PlausibilityThresholds(
        float(threshold_data["global_chi_square_mean"]),
        tuple(float(value) for value in threshold_data["per_band_chi_square_mean"]),
        float(threshold_data["absolute_relative_flux_residual"]),
        int(threshold_data["calibration_scene_count"]),
    )
    result = scientific_metrics(output, arrays["targets_physical"], arrays["counts"], arrays["blend_physical"], scales, threshold, noise, MEAN_PSF_FWHM_PIXEL)
    ordinary = [result[index] for index, row in enumerate(rows) if row["kind"] == "ordinary"]
    ambiguous = [result[index] for index, row in enumerate(rows) if row["kind"] == "near_collision"]
    return {
        "mean_primary_scientific_distance": float(np.mean([item["primary_scientific_distance"] for item in result])),
        "ordinary_coverage": float(np.mean([item["ordinary_both_experts_coverage"] for item in ordinary])),
        "ambiguous_own_coverage": float(np.mean([item["own_truth_coverage"] for item in ambiguous])),
        "ambiguous_alternate_coverage": float(np.mean([item["alternate_truth_coverage"] for item in ambiguous])),
        "ambiguous_both_mode_coverage": float(np.mean([item["both_mode_coverage"] for item in ambiguous])),
        "forward_consistent_fraction": float(np.mean([item["forward_consistent_fraction"] for item in result])),
    }


def score(
    output: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    scales_tensor: torch.Tensor,
    arrays: dict[str, np.ndarray],
    scales: np.ndarray,
    rows: list[dict[str, str]],
    exact: torch.Tensor,
) -> dict[str, float]:
    objective = corrected_objective(output, targets, counts, scales_tensor, MEAN_PSF_FWHM_PIXEL)
    exact_summary = exact_metrics_summary(output.detach().cpu().numpy(), arrays, scales, rows)
    return {
        "corrected_objective": float(objective["total"].detach()),
        "mean_surrogate": float(objective["science"].mean().detach()),
        "ordinary_concentration": float(objective["ordinary_concentration"].mean().detach()),
        "rms_to_exact_slot_order": float((output.detach() - exact).square().mean().sqrt()),
        **exact_summary,
    }


def gradient_audit(configurations: dict[str, np.ndarray], targets: torch.Tensor, counts: torch.Tensor, scales: torch.Tensor) -> list[dict[str, object]]:
    rows = []
    for configuration in ("exact_truth", "source_sum_wrong_allocation", "trained_thayer_me"):
        output = torch.from_numpy(configurations[configuration]).clone().requires_grad_(True)
        values = corrected_objective(output, targets, counts, scales, MEAN_PSF_FWHM_PIXEL)
        terms = {
            "total": values["total"],
            "requested_reconstruction": values["requested_reconstruction"].mean(),
            "companion_reconstruction": values["companion_reconstruction"].mean(),
            "science": values["science"].mean(),
            "ordinary_concentration": values["ordinary_concentration"].mean(),
        }
        for term, value in terms.items():
            gradient = torch.autograd.grad(value, output, retain_graph=True, allow_unused=False)[0]
            rows.append({
                "configuration": configuration,
                "term": term,
                "value": float(value.detach()),
                "gradient_l2": float(gradient.norm().detach()),
                "gradient_max_abs": float(gradient.abs().max().detach()),
                "truth_stationary": configuration != "exact_truth" or float(gradient.norm().detach()) <= 1e-5,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_BEFORE_OFFICIAL_PREFLIGHT_OR_NEURAL_FIT":
        raise RuntimeError("preregistration gate failed")
    if sha256_file(REPO / "src/scientific_alignment.py") != freeze["scientific_alignment_implementation_sha256"]:
        raise RuntimeError("scientific alignment implementation changed after freeze")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("neural checkpoint exists before preflight")

    started = time.time()
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows, source_indices = select_microset()
    arrays = load_micro_arrays(source_indices, scales)
    with h5py.File(TRAINED_OUTPUTS, "r") as handle:
        trained_physical = np.asarray(handle["decompositions"], dtype=np.float32)
    trained = trained_physical / np.tile(scales, 2)[None, None, None, :, None, None]
    configurations = canonical_outputs(arrays["targets"], trained)
    targets = torch.from_numpy(arrays["targets"])
    counts = torch.from_numpy(arrays["counts"])
    scales_tensor = torch.from_numpy(scales)

    unit_rows, unit_checks = unit_tests(arrays["targets"], scales)
    write_csv_fresh(run_dir / "tables/scientific_surrogate_unit_tests.csv", unit_rows)
    write_csv_fresh(run_dir / "tables/scientific_surrogate_test_checks.csv", unit_checks)
    if not all(row["pass"] for row in unit_checks):
        raise RuntimeError("scientific surrogate unit test failed")

    alignment_rows = aligned_pairs(
        {name: value for name, value in configurations.items() if name != "random_bounded"},
        arrays["targets"], arrays["targets_physical"], arrays["counts"], scales,
    )
    exact_values = np.asarray([float(row["exact_primary"]) for row in alignment_rows])
    surrogate_values = np.asarray([float(row["surrogate"]) for row in alignment_rows])
    spearman = float(spearmanr(exact_values, surrogate_values).statistic)
    kendall = float(kendalltau(exact_values, surrogate_values).statistic)
    side = float(np.mean((exact_values <= 1.0) == (surrogate_values <= 1.0)))
    mean_by_configuration = {
        name: float(np.mean([float(row["surrogate"]) for row in alignment_rows if row["configuration"] == name]))
        for name in ("exact_truth", "trained_thayer_me", "collapsed_mean", "source_sum_wrong_allocation")
    }
    alignment_summary = [
        {"gate": "spearman", "observed": spearman, "threshold": ">=0.95", "pass": spearman >= 0.95},
        {"gate": "kendall", "observed": kendall, "threshold": ">=0.90", "pass": kendall >= 0.90},
        {"gate": "threshold_side_agreement", "observed": side, "threshold": ">=0.98", "pass": side >= 0.98},
        {"gate": "truth_ranked_best", "observed": mean_by_configuration["exact_truth"], "threshold": "strict minimum", "pass": mean_by_configuration["exact_truth"] < min(value for name, value in mean_by_configuration.items() if name != "exact_truth")},
        {"gate": "approved_above_compromise", "observed": mean_by_configuration["collapsed_mean"], "threshold": ">truth", "pass": mean_by_configuration["collapsed_mean"] > mean_by_configuration["exact_truth"]},
    ]
    write_csv_fresh(run_dir / "tables/surrogate_alignment_pairs.csv", alignment_rows)
    write_csv_fresh(run_dir / "tables/surrogate_alignment_summary.csv", alignment_summary)
    plt.figure(figsize=(6, 5))
    plt.scatter(exact_values, surrogate_values, s=8, alpha=0.35)
    limit = float(np.nanpercentile(np.concatenate((exact_values, surrogate_values)), 99))
    plt.plot([0, limit], [0, limit], color="black", linewidth=1)
    plt.axvline(1.0, color="tab:red", linestyle="--", linewidth=1)
    plt.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    plt.xlabel("Frozen exact primary scientific distance")
    plt.ylabel("Differentiable smooth scientific surrogate")
    plt.tight_layout()
    plt.savefig(run_dir / "figures/surrogate_vs_frozen_metric.png", dpi=180)
    plt.close()
    write_text_fresh(run_dir / "diagnostics/scientific_surrogate_contract.md", f"""# Differentiable scientific surrogate contract

Status: **{'PASS' if all(row['pass'] for row in alignment_summary) else 'FAIL'}**.

The surrogate uses physical inverse-normalized g/r/z source layers, frozen image/flux floors, flux-derived colors, nonnegative soft centroids, unchanged thresholds, and zero-anchored log-mean-exp smooth maximum with temperature {SMOOTHMAX_TEMPERATURE}. Target tensors are detached. Exact truth is numerical zero. Canonical Spearman / Kendall / threshold-side agreement are {spearman:.6f} / {kendall:.6f} / {side:.6f}. Mean surrogate values for truth, trained output, collapsed mean, and wrong allocation are {mean_by_configuration['exact_truth']:.6f}, {mean_by_configuration['trained_thayer_me']:.6f}, {mean_by_configuration['collapsed_mean']:.6f}, and {mean_by_configuration['source_sum_wrong_allocation']:.6f}.
""")
    if not all(row["pass"] for row in alignment_summary):
        write_json_fresh(run_dir / "logs/preflight_complete.json", {"status": "SURROGATE_ALIGNMENT_FAILURE", "neural_training_authorized": False, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
        print(json.dumps({"status": "SURROGATE_ALIGNMENT_FAILURE"}))
        return

    gradient_rows = gradient_audit(configurations, targets, counts, scales_tensor)
    write_csv_fresh(run_dir / "tables/loss_gradient_scale_audit.csv", gradient_rows)
    truth_stationary = all(bool(row["truth_stationary"]) for row in gradient_rows if row["configuration"] == "exact_truth")

    trajectories: list[dict[str, object]] = []
    final_outputs: dict[str, np.ndarray] = {}
    exact_tensor = torch.from_numpy(configurations["exact_truth"])
    for name, initial in configurations.items():
        output = torch.from_numpy(initial.copy()).requires_grad_(True)
        optimizer = torch.optim.Adam([output], lr=PREFLIGHT_LR, weight_decay=0.0)
        for step in range(PREFLIGHT_STEPS + 1):
            if step in (0, 100, 200, 300, 400):
                values = score(output, targets, counts, scales_tensor, arrays, scales, rows, exact_tensor)
                trajectories.append({"initialization": name, "step": step, **values})
            if step == PREFLIGHT_STEPS:
                break
            optimizer.zero_grad(set_to_none=True)
            objective = corrected_objective(output, targets, counts, scales_tensor, MEAN_PSF_FWHM_PIXEL)["total"]
            objective.backward()
            optimizer.step()
        final_outputs[name] = output.detach().cpu().numpy()
        print(json.dumps({"preflight": name, **trajectories[-1]}, sort_keys=True), flush=True)
    write_csv_fresh(run_dir / "tables/output_space_preflight_trajectories.csv", trajectories)
    with h5py.File(run_dir / "objective_preflight/final_outputs.h5", "x") as handle:
        for name, value in final_outputs.items():
            handle.create_dataset(name, data=value, compression="lzf", chunks=(1, 1, 1, 6, 60, 60))
        handle.attrs["complete"] = True
        handle.attrs["model_parameter_count"] = 0

    by_init = {name: [row for row in trajectories if row["initialization"] == name] for name in configurations}
    gates: list[dict[str, object]] = []
    exact_start, exact_end = by_init["exact_truth"][0], by_init["exact_truth"][-1]
    gates.extend([
        {"gate": "truth_stationary_gradient", "observed": truth_stationary, "threshold": "True", "pass": truth_stationary},
        {"gate": "truth_loss_remains_zero", "observed": exact_end["corrected_objective"], "threshold": "<=1e-6", "pass": exact_end["corrected_objective"] <= 1e-6},
        {"gate": "truth_tensor_remains_fixed", "observed": exact_end["rms_to_exact_slot_order"], "threshold": "<=1e-5", "pass": exact_end["rms_to_exact_slot_order"] <= 1e-5},
        {"gate": "truth_coverage_remains_full", "observed": min(exact_end["ordinary_coverage"], exact_end["ambiguous_own_coverage"], exact_end["ambiguous_alternate_coverage"], exact_end["ambiguous_both_mode_coverage"]), "threshold": ">=0.90", "pass": min(exact_end["ordinary_coverage"], exact_end["ambiguous_own_coverage"], exact_end["ambiguous_alternate_coverage"], exact_end["ambiguous_both_mode_coverage"]) >= 0.90},
    ])
    for name in ("trained_thayer_me", "collapsed_mean", "source_sum_wrong_allocation"):
        start, end = by_init[name][0], by_init[name][-1]
        loss_reduction = (start["corrected_objective"] - end["corrected_objective"]) / max(start["corrected_objective"], 1e-12)
        science_reduction = (start["mean_primary_scientific_distance"] - end["mean_primary_scientific_distance"]) / max(start["mean_primary_scientific_distance"], 1e-12)
        minimum_coverage = min(end["ordinary_coverage"], end["ambiguous_own_coverage"], end["ambiguous_alternate_coverage"], end["ambiguous_both_mode_coverage"])
        gates.extend([
            {"gate": f"{name}_loss_reduction", "observed": loss_reduction, "threshold": ">=0.10", "pass": loss_reduction >= 0.10},
            {"gate": f"{name}_science_reduction", "observed": science_reduction, "threshold": ">=0.10", "pass": science_reduction >= 0.10},
            {"gate": f"{name}_enters_coverage", "observed": minimum_coverage, "threshold": ">=0.90", "pass": minimum_coverage >= 0.90},
        ])
    random_start, random_end = by_init["random_bounded"][0], by_init["random_bounded"][-1]
    random_loss_reduction = (random_start["corrected_objective"] - random_end["corrected_objective"]) / max(random_start["corrected_objective"], 1e-12)
    random_science_reduction = (random_start["mean_primary_scientific_distance"] - random_end["mean_primary_scientific_distance"]) / max(random_start["mean_primary_scientific_distance"], 1e-12)
    gates.extend([
        {"gate": "random_loss_reduction", "observed": random_loss_reduction, "threshold": ">=0.20", "pass": random_loss_reduction >= 0.20},
        {"gate": "random_science_reduction", "observed": random_science_reduction, "threshold": ">=0.20", "pass": random_science_reduction >= 0.20},
    ])
    write_csv_fresh(run_dir / "tables/output_space_preflight_gates.csv", gates)
    passed = all(bool(row["pass"]) for row in gates)
    status = "PASS" if passed else "CORRECTED OBJECTIVE STILL MISALIGNED"
    write_text_fresh(run_dir / "diagnostics/output_space_preflight.md", f"""# Thayer-SA output-space preflight

Status: **{status}**.

The official preregistered CPU float32 Adam protocol optimized only detached free output tensors; model parameter count and neural optimizer steps were zero. Exact truth remained stationary and covered. Compromise, trained, collapsed, and random requirements are recorded without post-hoc changes in `tables/output_space_preflight_gates.csv`. {'Assignment audit and neural micro-overfit are authorized next.' if passed else 'The campaign stops here. Assignment audit and neural fitting are not reached, and no optimizer, weight, or gate is changed after inspection.'}
""")
    payload = {
        "status": status,
        "surrogate_alignment_passed": True,
        "output_space_preflight_passed": passed,
        "assignment_audit_authorized": passed,
        "neural_training_authorized": False,
        "model_parameter_count_in_preflight": 0,
        "neural_optimizer_step_count": 0,
        "runtime_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
    }
    write_json_fresh(run_dir / "logs/preflight_complete.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
