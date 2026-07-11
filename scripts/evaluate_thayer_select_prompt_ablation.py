#!/usr/bin/env python3
"""One-time development evaluation for the first Thayer-Select baseline."""

from __future__ import annotations

import argparse
import json
import math
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
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from thayer_select_prompt_ablation_common import (
    CompactSelectNet,
    PROMPT_SIGMA_PIXELS,
    gaussian_prompt_numpy,
    inverse_normalize,
    load_scales,
    read_csv,
    require_mps,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
)

CONDITIONS = {
    "A_centered_no_prompt": {"variant": "centered", "prompted": False, "channels": 3},
    "B_randomized_no_prompt": {"variant": "random", "prompted": False, "channels": 3},
    "C_randomized_coordinate_prompt": {"variant": "random", "prompted": True, "channels": 4},
}
BATCH_SIZE = 16
BOOTSTRAP_SEED = 2026076101


def load_model(run_dir: Path, condition: str, channels: int, device: torch.device) -> CompactSelectNet:
    path = run_dir / f"checkpoints/{condition.lower()}_best.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["condition"] != condition or payload["selection"] != "minimum validation MSE only":
        raise RuntimeError(f"Invalid checkpoint label/selection for {condition}")
    model = CompactSelectNet(channels).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if model.training or next(model.parameters()).device.type != "mps":
        raise RuntimeError("Evaluation model is not frozen on MPS")
    return model


def infer(
    model: CompactSelectNet,
    blends: np.ndarray,
    scales: np.ndarray,
    device: torch.device,
    coordinates: np.ndarray | None = None,
    empty_prompt: bool = False,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(blends), BATCH_SIZE):
            stop = min(len(blends), start + BATCH_SIZE)
            normalized = np.asarray(blends[start:stop], dtype=np.float32) / scales[None, :, None, None]
            if model.in_channels == 4:
                if empty_prompt:
                    prompts = np.zeros((stop - start, 1, blends.shape[-2], blends.shape[-1]), dtype=np.float32)
                else:
                    if coordinates is None:
                        raise RuntimeError("Prompted inference requires coordinates")
                    prompts = np.stack([
                        gaussian_prompt_numpy(float(x), float(y), sigma_pixels=PROMPT_SIGMA_PIXELS)
                        for x, y in coordinates[start:stop]
                    ])[:, None]
                normalized = np.concatenate((normalized, prompts), axis=1)
            tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(device)
            if tensor.device.type != "mps":
                raise RuntimeError("Inference device fallback")
            prediction = model(tensor)
            if prediction.device.type != "mps" or not torch.isfinite(prediction).all():
                raise RuntimeError("Invalid MPS prediction")
            physical = prediction.detach().cpu().numpy() * scales[None, :, None, None]
            outputs.append(physical.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def masks(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    brightness = np.max(np.abs(target), axis=0)
    peak = float(brightness.max())
    if peak <= 0 or not np.isfinite(peak):
        source = np.ones(brightness.shape, dtype=bool)
        core = np.zeros(brightness.shape, dtype=bool)
    else:
        source = brightness > 0.01 * peak
        core = brightness > 0.50 * peak
    return source, core, source & ~core


def masked_values(error: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return error[:, mask]


def centroid(image: np.ndarray) -> tuple[float, float]:
    weight = np.maximum(np.sum(image, axis=0), 0.0)
    total = float(weight.sum())
    if total <= 0 or not np.isfinite(total):
        return math.nan, math.nan
    yy, xx = np.mgrid[: image.shape[-2], : image.shape[-1]]
    return float((xx * weight).sum() / total), float((yy * weight).sum() / total)


def color(flux_one: float, flux_two: float) -> float:
    if flux_one <= 0 or flux_two <= 0:
        return math.nan
    return float(-2.5 * np.log10(flux_one / flux_two))


def sample_metrics(prediction: np.ndarray, target: np.ndarray, blend: np.ndarray) -> dict:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    blend64 = np.asarray(blend, dtype=np.float64)
    error = prediction64 - target64
    source, core, noncore = masks(target64)
    squared = error**2
    absolute = np.abs(error)
    data_range = float(target64.max() - target64.min())
    if data_range <= 0:
        data_range = max(float(np.max(np.abs(target64))), 1.0)
    mse = float(np.mean(squared))
    psnr = math.inf if mse == 0 else float(20 * np.log10(data_range) - 10 * np.log10(mse))
    ssim_values = []
    for band in range(3):
        band_range = float(target64[band].max() - target64[band].min())
        band_range = band_range if band_range > 0 else data_range
        ssim_values.append(structural_similarity(target64[band], prediction64[band], data_range=band_range))
    target_flux = target64.sum(axis=(-2, -1))
    prediction_flux = prediction64.sum(axis=(-2, -1))
    flux_fraction = (prediction_flux - target_flux) / np.maximum(np.abs(target_flux), 1e-30)
    target_gr = color(target_flux[0], target_flux[1]); prediction_gr = color(prediction_flux[0], prediction_flux[1])
    target_rz = color(target_flux[1], target_flux[2]); prediction_rz = color(prediction_flux[1], prediction_flux[2])
    target_centroid = centroid(target64); prediction_centroid = centroid(prediction64)
    centroid_error = math.nan
    if all(np.isfinite(target_centroid + prediction_centroid)):
        centroid_error = float(np.hypot(prediction_centroid[0] - target_centroid[0], prediction_centroid[1] - target_centroid[1]))
    source_squared = masked_values(squared, source); source_absolute = masked_values(absolute, source)
    core_squared = masked_values(squared, core); core_absolute = masked_values(absolute, core)
    noncore_squared = masked_values(squared, noncore); noncore_absolute = masked_values(absolute, noncore)
    identity_mse = float(np.mean((blend64 - target64) ** 2))
    return {
        "whole_mse": mse, "whole_mae": float(np.mean(absolute)),
        "source_mse": float(np.mean(source_squared)), "source_mae": float(np.mean(source_absolute)),
        "core_mse": float(np.mean(core_squared)) if core_squared.size else math.nan,
        "core_mae": float(np.mean(core_absolute)) if core_absolute.size else math.nan,
        "noncore_mse": float(np.mean(noncore_squared)) if noncore_squared.size else math.nan,
        "noncore_mae": float(np.mean(noncore_absolute)) if noncore_absolute.size else math.nan,
        "psnr": psnr, "ssim": float(np.mean(ssim_values)),
        "g_flux_fraction_error": float(flux_fraction[0]), "r_flux_fraction_error": float(flux_fraction[1]), "z_flux_fraction_error": float(flux_fraction[2]),
        "g_flux_absolute_error": float(abs(prediction_flux[0] - target_flux[0])),
        "r_flux_absolute_error": float(abs(prediction_flux[1] - target_flux[1])),
        "z_flux_absolute_error": float(abs(prediction_flux[2] - target_flux[2])),
        "g_minus_r_color_error": float(prediction_gr - target_gr) if np.isfinite(prediction_gr) and np.isfinite(target_gr) else math.nan,
        "r_minus_z_color_error": float(prediction_rz - target_rz) if np.isfinite(prediction_rz) and np.isfinite(target_rz) else math.nan,
        "centroid_error_pixels": centroid_error,
        "identity_mse": identity_mse, "worse_than_input": int(mse > identity_mse),
        "whole_squared_sum": float(squared.sum()), "whole_absolute_sum": float(absolute.sum()), "whole_value_count": int(squared.size),
        "source_squared_sum": float(source_squared.sum()), "source_absolute_sum": float(source_absolute.sum()), "source_value_count": int(source_squared.size),
        "core_squared_sum": float(core_squared.sum()), "core_absolute_sum": float(core_absolute.sum()), "core_value_count": int(core_squared.size),
        "noncore_squared_sum": float(noncore_squared.sum()), "noncore_absolute_sum": float(noncore_absolute.sum()), "noncore_value_count": int(noncore_squared.size),
    }


def finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else math.nan


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(2000)
    for index in range(len(estimates)):
        estimates[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def aggregate(condition: str, rows: list[dict]) -> dict:
    record = {"condition": condition, "scene_count": len(rows), "aggregation": "macro_per_scene"}
    for field in ("whole_mse", "whole_mae", "source_mse", "source_mae", "core_mse", "core_mae", "noncore_mse", "noncore_mae", "psnr", "ssim", "g_flux_fraction_error", "r_flux_fraction_error", "z_flux_fraction_error", "g_minus_r_color_error", "r_minus_z_color_error", "centroid_error_pixels"):
        record[field] = finite_mean([row[field] for row in rows])
    for field in ("whole_mse", "source_mse"):
        low, high = bootstrap_ci(np.asarray([row[field] for row in rows]), BOOTSTRAP_SEED + (0 if field == "whole_mse" else 1))
        record[f"{field}_ci95_low"] = low; record[f"{field}_ci95_high"] = high
    record["worse_than_input_count"] = sum(int(row["worse_than_input"]) for row in rows)
    return record


def micro_aggregate(condition: str, rows: list[dict]) -> dict:
    record = {"condition": condition, "scene_count": len(rows), "aggregation": "micro_affected_or_source_pixels"}
    for region in ("whole", "source", "core", "noncore"):
        count = sum(int(row[f"{region}_value_count"]) for row in rows)
        record[f"{region}_value_count"] = count
        record[f"{region}_mse"] = sum(float(row[f"{region}_squared_sum"]) for row in rows) / count if count else math.nan
        record[f"{region}_mae"] = sum(float(row[f"{region}_absolute_sum"]) for row in rows) / count if count else math.nan
    return record


def prompt_swap(
    run_dir: Path,
    model: CompactSelectNet,
    blends: np.ndarray,
    isolated: np.ndarray,
    xy: np.ndarray,
    scales: np.ndarray,
    device: torch.device,
    scene_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    prediction_a = infer(model, blends, scales, device, xy[:, 0])
    prediction_b = infer(model, blends, scales, device, xy[:, 1])
    offsets = np.asarray([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    perturb_sensitivity = np.zeros((len(blends), 2), dtype=float)
    for source_index, base_prediction in ((0, prediction_a), (1, prediction_b)):
        values = []
        for offset in offsets:
            shifted = infer(model, blends, scales, device, xy[:, source_index] + offset[None],)
            values.append(np.mean(np.abs(shifted - base_prediction), axis=(1, 2, 3)))
        perturb_sensitivity[:, source_index] = np.mean(values, axis=0)
    rows = []
    for index, scene in enumerate(scene_rows):
        mse_a_a = float(np.mean((prediction_a[index] - isolated[index, 0]) ** 2))
        mse_a_b = float(np.mean((prediction_a[index] - isolated[index, 1]) ** 2))
        mse_b_b = float(np.mean((prediction_b[index] - isolated[index, 1]) ** 2))
        mse_b_a = float(np.mean((prediction_b[index] - isolated[index, 0]) ** 2))
        output_difference = float(np.mean(np.abs(prediction_a[index] - prediction_b[index])))
        truth_difference = float(np.mean(np.abs(isolated[index, 0] - isolated[index, 1])))
        collapse_ratio = output_difference / max(truth_difference, 1e-30)
        rows.append({
            "scene_id": scene["scene_id"], "mse_query_a_to_a": mse_a_a, "mse_query_a_to_b": mse_a_b,
            "mse_query_b_to_b": mse_b_b, "mse_query_b_to_a": mse_b_a,
            "query_a_closer_requested": int(mse_a_a < mse_a_b), "query_b_closer_requested": int(mse_b_b < mse_b_a),
            "source_swap_success": int(mse_a_a < mse_a_b and mse_b_b < mse_b_a),
            "changing_prompt_changes_output": int(output_difference > 1e-8),
            "output_mean_absolute_change": output_difference, "truth_mean_absolute_difference": truth_difference,
            "prompt_sensitivity_ratio": collapse_ratio, "output_collapse": int(collapse_ratio < 0.1),
            "a_perturbation_sensitivity": perturb_sensitivity[index, 0], "b_perturbation_sensitivity": perturb_sensitivity[index, 1],
        })
    write_csv_fresh(run_dir / "tables/prompt_swap_per_scene.csv", rows)
    summary = {
        "scene_count": len(rows),
        "source_swap_success_rate": finite_mean([row["source_swap_success"] for row in rows]),
        "changing_prompt_changes_output_rate": finite_mean([row["changing_prompt_changes_output"] for row in rows]),
        "output_collapse_rate": finite_mean([row["output_collapse"] for row in rows]),
        "prompt_sensitivity_ratio_mean": finite_mean([row["prompt_sensitivity_ratio"] for row in rows]),
        "small_offset_absolute_sensitivity_mean": finite_mean([(row["a_perturbation_sensitivity"] + row["b_perturbation_sensitivity"]) / 2 for row in rows]),
    }
    write_json_fresh(run_dir / "reports/prompt_swap_summary.json", summary)
    return prediction_a, prediction_b, rows


def flagship_figure(run_dir: Path, blends, isolated, xy, prediction_a, prediction_b, swap_rows) -> None:
    path = run_dir / "figures/prompt_swap_flagship_grid.png"
    order = np.argsort([float(row["prompt_sensitivity_ratio"]) for row in swap_rows])
    chosen = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(len(chosen), 9, figsize=(19, 13), constrained_layout=True)
    titles = ["isolated A", "isolated B", "blend", "prompt A", "predicted A", "prompt B", "predicted B", "|A error|", "|B error|"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=9)
    for row_index, index in enumerate(chosen):
        prompt_a = gaussian_prompt_numpy(*xy[index, 0]); prompt_b = gaussian_prompt_numpy(*xy[index, 1])
        panels = [isolated[index, 0, 1], isolated[index, 1, 1], blends[index, 1], prompt_a,
                  prediction_a[index, 1], prompt_b, prediction_b[index, 1],
                  np.abs(prediction_a[index, 1] - isolated[index, 0, 1]), np.abs(prediction_b[index, 1] - isolated[index, 1, 1])]
        scale = max(float(np.max(np.abs(panel))) for panel in panels[:3] + panels[4:5] + panels[6:])
        for column, panel in enumerate(panels):
            axis = axes[row_index, column]
            if column in (3, 5):
                axis.imshow(panel, origin="lower", cmap="magma", vmin=0, vmax=1)
            else:
                axis.imshow(np.arcsinh(panel / max(scale, 1e-30) * 20), origin="lower", cmap="coolwarm")
            axis.set_xticks([]); axis.set_yticks([])
        axes[row_index, 0].set_ylabel(swap_rows[index]["scene_id"], fontsize=7)
    fig.suptitle("Condition C prompt swap (r band; signed asinh except prompts/errors)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def centrality_analysis(run_dir: Path, scene_rows, blends, isolated, xy, predictions, primary_rows) -> None:
    b_prediction = predictions["B_randomized_no_prompt"]
    c_prediction = predictions["C_randomized_coordinate_prompt"]
    b_rows = [row for row in primary_rows if row["condition"] == "B_randomized_no_prompt"]
    c_rows = [row for row in primary_rows if row["condition"] == "C_randomized_coordinate_prompt"]
    behavior = Counter()
    stratified_samples = []
    for index, scene in enumerate(scene_rows):
        distances = np.linalg.norm(xy[index] - np.asarray([(blends.shape[-1] - 1) / 2, (blends.shape[-2] - 1) / 2]), axis=1)
        fluxes = isolated[index].sum(axis=(1, 2, 3))
        sizes = np.asarray([float(scene["source_a_size_arcsec"]), float(scene["source_b_size_arcsec"])])
        mse_to = [float(np.mean((b_prediction[index] - isolated[index, source]) ** 2)) for source in (0, 1)]
        mse_average = float(np.mean((b_prediction[index] - isolated[index].mean(axis=0)) ** 2))
        chosen = int(np.argmin(mse_to))
        behavior["closer_source_a" if chosen == 0 else "closer_source_b"] += 1
        behavior["closer_central_source"] += int(chosen == int(np.argmin(distances)))
        behavior["closer_brighter_source"] += int(chosen == int(np.argmax(fluxes)))
        behavior["closer_larger_source"] += int(chosen == int(np.argmax(sizes)))
        behavior["closer_average_than_either"] += int(mse_average < min(mse_to))
        target = int(scene["target_index"]); alternate = 1 - target
        target_image = isolated[index, target]; alternate_image = isolated[index, alternate]
        source_mask, core_mask, _ = masks(target_image)
        obstruction = float(np.maximum(alternate_image[:, core_mask], 0).sum() / max(np.maximum(target_image[:, core_mask], 0).sum(), 1e-30))
        color_a = np.asarray([float(scene["source_a_g_ab"]) - float(scene["source_a_r_ab"]), float(scene["source_a_r_ab"]) - float(scene["source_a_z_ab"])])
        color_b = np.asarray([float(scene["source_b_g_ab"]) - float(scene["source_b_r_ab"]), float(scene["source_b_r_ab"]) - float(scene["source_b_z_ab"])])
        similarity_distance = float(np.linalg.norm(color_a - color_b) + abs(np.log(max(sizes[0], 1e-9) / max(sizes[1], 1e-9))))
        stratified_samples.append({
            "scene_id": scene["scene_id"], "separation_pixels": float(scene["separation_pixels"]),
            "separation_psf_units": float(scene.get("separation_psf_units", math.nan)),
            "flux_ratio": float(fluxes[target] / max(fluxes[alternate], 1e-30)),
            "target_distance_from_center_pixels": float(distances[target]),
            "size_ratio": float(sizes[target] / max(sizes[alternate], 1e-30)),
            "source_similarity_distance": similarity_distance, "core_obstruction_ratio": obstruction,
            "b_source_mse": float(b_rows[index]["source_mse"]), "c_source_mse": float(c_rows[index]["source_mse"]),
            "c_minus_b_source_mse": float(c_rows[index]["source_mse"]) - float(b_rows[index]["source_mse"]),
            "c_wins": int(float(c_rows[index]["source_mse"]) < float(b_rows[index]["source_mse"])),
        })
    write_json_fresh(run_dir / "reports/randomized_unprompted_behavior.json", {"scene_count": len(scene_rows), **behavior})
    write_csv_fresh(run_dir / "tables/centrality_stratification_per_scene.csv", stratified_samples)
    output = []
    variables = ["separation_pixels", "separation_psf_units", "flux_ratio", "target_distance_from_center_pixels", "size_ratio", "source_similarity_distance", "core_obstruction_ratio"]
    for variable in variables:
        values = np.asarray([row[variable] for row in stratified_samples], dtype=float)
        finite = np.isfinite(values)
        edges = np.unique(np.quantile(values[finite], [0, 0.25, 0.5, 0.75, 1]))
        for bin_index in range(max(0, len(edges) - 1)):
            low, high = edges[bin_index], edges[bin_index + 1]
            mask = finite & (values >= low) & (values <= high if bin_index == len(edges) - 2 else values < high)
            selected = [row for row, keep in zip(stratified_samples, mask) if keep]
            output.append({"variable": variable, "bin": bin_index, "low": low, "high": high, "scene_count": len(selected),
                           "b_source_mse": finite_mean([row["b_source_mse"] for row in selected]),
                           "c_source_mse": finite_mean([row["c_source_mse"] for row in selected]),
                           "c_minus_b_source_mse": finite_mean([row["c_minus_b_source_mse"] for row in selected]),
                           "c_win_rate": finite_mean([row["c_wins"] for row in selected])})
    write_csv_fresh(run_dir / "tables/centrality_stratified_summary.csv", output)


def no_harm_tests(run_dir: Path, model, blends, isolated, xy, scene_rows, scales, device, correct_predictions) -> None:
    target_indices = np.asarray([int(row["target_index"]) for row in scene_rows])
    alternate_indices = 1 - target_indices
    targets = isolated[np.arange(len(isolated)), target_indices]
    target_xy = xy[np.arange(len(xy)), target_indices]
    alternate_xy = xy[np.arange(len(xy)), alternate_indices]
    isolated_correct = infer(model, targets, scales, device, target_xy)
    empty = infer(model, blends, scales, device, empty_prompt=True)
    wrong = infer(model, blends, scales, device, alternate_xy)
    between = infer(model, blends, scales, device, (target_xy + alternate_xy) / 2)
    perturb_predictions = []
    for offset in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        perturb_predictions.append(infer(model, blends, scales, device, target_xy + np.asarray(offset)))
    perturbed = np.mean(perturb_predictions, axis=0)
    case_predictions = {"isolated_correct": isolated_correct, "empty_prompt": empty, "wrong_prompt": wrong, "between_sources": between, "one_pixel_perturbed": perturbed}
    rows = []
    for case, values in case_predictions.items():
        for index, scene in enumerate(scene_rows):
            target = targets[index]; alternate = isolated[index, alternate_indices[index]]
            metrics = sample_metrics(values[index], target, blends[index] if case != "isolated_correct" else target)
            predicted_flux = float(np.sum(np.abs(values[index]))); target_flux = float(np.sum(np.abs(target)))
            confusion = int(np.mean((values[index] - alternate) ** 2) < np.mean((values[index] - target) ** 2))
            sensitivity = float(np.mean(np.abs(values[index] - correct_predictions[index])))
            rows.append({"scene_id": scene["scene_id"], "case": case, "whole_mse": metrics["whole_mse"], "source_mse": metrics["source_mse"],
                         "predicted_absolute_flux": predicted_flux, "target_absolute_flux": target_flux,
                         "predicted_to_target_absolute_flux_ratio": predicted_flux / max(target_flux, 1e-30),
                         "hallucinated_source": int(case == "empty_prompt" and predicted_flux > 0.1 * target_flux),
                         "source_confusion": confusion, "mean_absolute_change_from_correct_prompt": sensitivity})
    write_csv_fresh(run_dir / "tables/no_harm_per_sample.csv", rows)
    summary = []
    for case in case_predictions:
        selected = [row for row in rows if row["case"] == case]
        summary.append({"case": case, "scene_count": len(selected), "whole_mse": finite_mean([row["whole_mse"] for row in selected]),
                        "source_mse": finite_mean([row["source_mse"] for row in selected]),
                        "predicted_to_target_absolute_flux_ratio": finite_mean([row["predicted_to_target_absolute_flux_ratio"] for row in selected]),
                        "hallucinated_source_rate": finite_mean([row["hallucinated_source"] for row in selected]),
                        "source_confusion_rate": finite_mean([row["source_confusion"] for row in selected]),
                        "prompt_sensitivity": finite_mean([row["mean_absolute_change_from_correct_prompt"] for row in selected])})
    write_csv_fresh(run_dir / "tables/no_harm_summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (run_dir / "logs/development_evaluation_complete.json").exists():
        raise RuntimeError("Development test was already evaluated; refusing a second inspection")
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    if training.get("status") != "PASS" or not training.get("all_epochs_completed"):
        raise RuntimeError("All conditions must be trained and frozen before development evaluation")
    device = require_mps(); scales = load_scales(run_dir)
    scene_rows = [row for row in read_csv(run_dir / "manifests/rendered_scene_manifest.csv") if row["partition"] == "development_test"]
    if len(scene_rows) != 1000 or any("lockbox" in row["partition"] for row in scene_rows):
        raise RuntimeError("Invalid development-test manifest")
    with h5py.File(run_dir / "manifests/development_test_scenes.h5", "r") as handle:
        random_blend = np.asarray(handle["random_blend"], dtype=np.float32)
        random_isolated = np.asarray(handle["random_isolated"], dtype=np.float32)
        centered_blend = np.asarray(handle["centered_blend"], dtype=np.float32)
        centered_isolated = np.asarray(handle["centered_isolated"], dtype=np.float32)
        random_xy = np.asarray(handle["random_xy"], dtype=np.float64)
    target_indices = np.asarray([int(row["target_index"]) for row in scene_rows])
    predictions = {}; models = {}
    for condition, config in CONDITIONS.items():
        model = load_model(run_dir, condition, config["channels"], device); models[condition] = model
        blend = centered_blend if config["variant"] == "centered" else random_blend
        isolated = centered_isolated if config["variant"] == "centered" else random_isolated
        coordinates = None if not config["prompted"] else random_xy[np.arange(len(random_xy)), target_indices]
        predictions[condition] = infer(model, blend, scales, device, coordinates)
    primary_rows = []
    for condition, config in CONDITIONS.items():
        blend = centered_blend if config["variant"] == "centered" else random_blend
        isolated = centered_isolated if config["variant"] == "centered" else random_isolated
        for index, scene in enumerate(scene_rows):
            target = isolated[index, target_indices[index]]
            primary_rows.append({"condition": condition, "scene_id": scene["scene_id"], "target_source_id": scene["target_source_id"],
                                 **sample_metrics(predictions[condition][index], target, blend[index])})
    write_csv_fresh(run_dir / "tables/primary_metrics_per_sample.csv", primary_rows)
    macro_rows = []; micro_rows = []
    for condition in CONDITIONS:
        selected = [row for row in primary_rows if row["condition"] == condition]
        macro_rows.append(aggregate(condition, selected)); micro_rows.append(micro_aggregate(condition, selected))
    write_csv_fresh(run_dir / "tables/primary_metrics_macro.csv", macro_rows)
    write_csv_fresh(run_dir / "tables/primary_metrics_micro.csv", micro_rows)
    by_condition = {condition: [row for row in primary_rows if row["condition"] == condition] for condition in CONDITIONS}
    b = np.asarray([row["source_mse"] for row in by_condition["B_randomized_no_prompt"]]); c = np.asarray([row["source_mse"] for row in by_condition["C_randomized_coordinate_prompt"]])
    delta = c - b; low, high = bootstrap_ci(delta, BOOTSTRAP_SEED + 9)
    effects = {
        "metric": "source_mse", "B_mean": float(b.mean()), "C_mean": float(c.mean()), "C_minus_B_mean": float(delta.mean()),
        "C_minus_B_ci95": [low, high], "C_wins": int(np.sum(c < b)), "C_losses": int(np.sum(c > b)), "ties": int(np.sum(c == b)),
        "A_centered_mean": finite_mean([row["source_mse"] for row in by_condition["A_centered_no_prompt"]]),
        "B_randomized_minus_A_centered_absolute_difference": float(b.mean() - finite_mean([row["source_mse"] for row in by_condition["A_centered_no_prompt"]])),
        "note": "A uses the aligned centered-position variant; B/C use one identical randomized manifest. No cross-manifest ratio is reported.",
    }
    write_json_fresh(run_dir / "reports/paired_effects.json", effects)
    prediction_a, prediction_b, swap_rows = prompt_swap(run_dir, models["C_randomized_coordinate_prompt"], random_blend, random_isolated, random_xy, scales, device, scene_rows)
    flagship_figure(run_dir, random_blend, random_isolated, random_xy, prediction_a, prediction_b, swap_rows)
    centrality_analysis(run_dir, scene_rows, random_blend, random_isolated, random_xy, predictions, primary_rows)
    no_harm_tests(run_dir, models["C_randomized_coordinate_prompt"], random_blend, random_isolated, random_xy, scene_rows, scales, device, predictions["C_randomized_coordinate_prompt"])
    write_json_fresh(run_dir / "logs/development_evaluation_complete.json", {
        "status": "PASS", "evaluated_once": True, "scene_count": 1000, "device": "mps",
        "model_eval_used": True, "torch_no_grad_used": True, "calibration_data_used": False,
        "lockbox_accessed": False, "completed_at_unix": time.time(),
        "development_manifest_sha256": sha256_file(run_dir / "manifests/rendered_scene_manifest.csv"),
    })


if __name__ == "__main__":
    main()
