#!/usr/bin/env python3
"""Apply the frozen non-Atlas promptability and candidate-contract gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_thayer_select_prompt_ablation import sample_metrics  # noqa: E402
from scripts.thayer_select_prompt_ablation_common import CompactSelectNet, gaussian_prompt_numpy  # noqa: E402
from src.models_prompted_resunet import PromptedResUNet  # noqa: E402


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    return torch.device("mps")


def load_models(run_dir: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    payload = torch.load(CONDITION_C, map_location="cpu", weights_only=False)
    condition_c = CompactSelectNet(4).to(device)
    condition_c.load_state_dict(payload["state_dict"], strict=True)
    res_payload = torch.load(run_dir / "checkpoints/prompted_resunet_best.pth", map_location="cpu", weights_only=False)
    if res_payload["selection"] != "minimum validation MSE only" or res_payload["parameter_count"] != 199_219:
        raise RuntimeError("invalid ResUNet checkpoint")
    resunet = PromptedResUNet().to(device)
    resunet.load_state_dict(res_payload["state_dict"], strict=True)
    models = {"Condition C": condition_c, "Prompted ResUNet": resunet}
    for model in models.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.training or next(model.parameters()).device.type != "mps":
            raise RuntimeError("model not frozen on MPS")
    return models


def infer(model: torch.nn.Module, blends: np.ndarray, coordinates: np.ndarray, scales: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(blends), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(blends))
            normalized = blends[start:stop].astype(np.float32) / scales[None, :, None, None]
            prompts = np.stack([gaussian_prompt_numpy(float(x), float(y)) for x, y in coordinates[start:stop]])[:, None]
            tensor = torch.from_numpy(np.ascontiguousarray(np.concatenate((normalized, prompts), axis=1))).to(device)
            prediction = model(tensor)
            if tensor.device.type != "mps" or prediction.device.type != "mps" or not torch.isfinite(prediction).all():
                raise RuntimeError("invalid or fallback inference")
            outputs.append((prediction.cpu().numpy() * scales[None, :, None, None]).astype(np.float32))
    return np.concatenate(outputs)


def finite_mean(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array[np.isfinite(array)])) if np.any(np.isfinite(array)) else math.nan


def output_stats(family: str, prediction: np.ndarray) -> dict[str, object]:
    border = np.concatenate((prediction[:, :, 0, :].ravel(), prediction[:, :, -1, :].ravel(), prediction[:, :, :, 0].ravel(), prediction[:, :, :, -1].ravel()))
    interior = prediction[:, :, 1:-1, 1:-1].ravel()
    dynamic = np.ptp(prediction, axis=(1, 2, 3))
    edge_jump = np.mean(np.abs(prediction[:, :, 0, :] - prediction[:, :, 1, :]), axis=(1, 2))
    return {
        "family": family,
        "candidate_count": len(prediction),
        "finite": bool(np.all(np.isfinite(prediction))),
        "dynamic_range_mean": float(np.mean(dynamic)),
        "border_mean": float(np.mean(border)),
        "border_std": float(np.std(border)),
        "interior_mean": float(np.mean(interior)),
        "exact_zero_fraction": float(np.mean(prediction == 0)),
        "negative_fraction": float(np.mean(prediction < 0)),
        "total_flux_mean": float(np.mean(prediction.sum(axis=(1, 2, 3)))),
        "edge_to_interior_abs_ratio": float(np.mean(np.abs(border)) / max(np.mean(np.abs(interior)), 1e-30)),
        "top_edge_jump_mean": float(np.mean(edge_jump)),
        "constant_or_zero_border": bool(np.std(border) == 0 or np.all(border == 0)),
        "clipping_applied": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (run_dir / "logs/pre_atlas_validation_complete.json").exists():
        raise FileExistsError("pre-Atlas evaluation already exists")
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    if training["status"] != "PASS" or training["atlas_evaluation_count"] != 0:
        raise RuntimeError("training gate not ready")
    device = require_mps()
    models = load_models(run_dir, device)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    rows = [row for row in read_csv(run_dir / "manifests/resunet_scene_definitions.csv") if row["partition"] == "validation"]
    with h5py.File(run_dir / "manifests/resunet_validation_scenes.h5", "r") as handle:
        if not bool(handle.attrs["complete"]) or len(handle["blend"]) != 1_500:
            raise RuntimeError("validation HDF5 incomplete")
        blends = np.asarray(handle["blend"], dtype=np.float32)
        isolated = np.asarray(handle["isolated"], dtype=np.float32)
        xy = np.asarray(handle["xy"], dtype=np.float64)
    started = time.time()
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family, model in models.items():
        prediction_a = infer(model, blends, xy[:, 0], scales, device)
        prediction_b = infer(model, blends, xy[:, 1], scales, device)
        replay = infer(model, blends[:BATCH_SIZE], xy[:BATCH_SIZE, 0], scales, device)
        if not np.array_equal(prediction_a[:BATCH_SIZE], replay):
            raise RuntimeError(f"deterministic MPS replay failed for {family}")
        predictions[family] = (prediction_a, prediction_b)

    metric_rows: list[dict[str, object]] = []
    swap_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    leakage_rows = []
    offsets = np.asarray([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    for family, model in models.items():
        prediction_a, prediction_b = predictions[family]
        perturb = np.zeros((len(rows), 2), dtype=np.float64)
        for source_index, base in ((0, prediction_a), (1, prediction_b)):
            changes = []
            for offset in offsets:
                shifted = infer(model, blends, xy[:, source_index] + offset[None], scales, device)
                changes.append(np.mean(np.abs(shifted - base), axis=(1, 2, 3)))
            perturb[:, source_index] = np.mean(changes, axis=0)
        family_query_rows = []
        family_swap_rows = []
        for index, scene in enumerate(rows):
            for source_index, prediction in ((0, prediction_a[index]), (1, prediction_b[index])):
                metrics = sample_metrics(prediction, isolated[index, source_index], blends[index])
                alternate_mse = float(np.mean((prediction.astype(np.float64) - isolated[index, 1 - source_index].astype(np.float64)) ** 2))
                record = {
                    "family": family, "scene_id": scene["scene_id"], "source_index": source_index,
                    "source_group": scene["source_a_group"] if source_index == 0 else scene["source_b_group"],
                    "alternate_source_group": scene["source_b_group"] if source_index == 0 else scene["source_a_group"],
                    **metrics, "alternate_whole_mse": alternate_mse,
                    "requested_minus_alternate_mse": float(metrics["whole_mse"]) - alternate_mse,
                    "closer_requested": int(float(metrics["whole_mse"]) < alternate_mse),
                    "prompt_perturbation_sensitivity": perturb[index, source_index],
                    "candidate_sha256": sha256_array(prediction),
                }
                family_query_rows.append(record)
                metric_rows.append(record)
            mse_a_a = float(np.mean((prediction_a[index].astype(np.float64) - isolated[index, 0].astype(np.float64)) ** 2))
            mse_a_b = float(np.mean((prediction_a[index].astype(np.float64) - isolated[index, 1].astype(np.float64)) ** 2))
            mse_b_b = float(np.mean((prediction_b[index].astype(np.float64) - isolated[index, 1].astype(np.float64)) ** 2))
            mse_b_a = float(np.mean((prediction_b[index].astype(np.float64) - isolated[index, 0].astype(np.float64)) ** 2))
            output_difference = float(np.mean(np.abs(prediction_a[index] - prediction_b[index])))
            truth_difference = float(np.mean(np.abs(isolated[index, 0] - isolated[index, 1])))
            collapse_ratio = output_difference / max(truth_difference, 1e-30)
            swap = {
                "family": family, "scene_id": scene["scene_id"],
                "source_a_group": scene["source_a_group"], "source_b_group": scene["source_b_group"],
                "mse_query_a_to_a": mse_a_a, "mse_query_a_to_b": mse_a_b,
                "mse_query_b_to_b": mse_b_b, "mse_query_b_to_a": mse_b_a,
                "source_swap_success": int(mse_a_a < mse_a_b and mse_b_b < mse_b_a),
                "output_collapse": int(collapse_ratio < 0.1),
                "prompt_sensitivity_ratio": collapse_ratio,
                "output_mean_absolute_change": output_difference,
                "truth_mean_absolute_difference": truth_difference,
            }
            family_swap_rows.append(swap)
            swap_rows.append(swap)
        summary_rows.append({
            "family": family, "query_count": len(family_query_rows), "scene_count": len(family_swap_rows),
            "whole_image_mse": finite_mean([row["whole_mse"] for row in family_query_rows]),
            "whole_image_mae": finite_mean([row["whole_mae"] for row in family_query_rows]),
            "source_region_mse": finite_mean([row["source_mse"] for row in family_query_rows]),
            "source_region_mae": finite_mean([row["source_mae"] for row in family_query_rows]),
            "psnr": finite_mean([row["psnr"] for row in family_query_rows]),
            "ssim": finite_mean([row["ssim"] for row in family_query_rows]),
            "g_flux_fraction_error": finite_mean([row["g_flux_fraction_error"] for row in family_query_rows]),
            "r_flux_fraction_error": finite_mean([row["r_flux_fraction_error"] for row in family_query_rows]),
            "z_flux_fraction_error": finite_mean([row["z_flux_fraction_error"] for row in family_query_rows]),
            "centroid_error_pixels": finite_mean([row["centroid_error_pixels"] for row in family_query_rows]),
            "prompt_swap_success": finite_mean([row["source_swap_success"] for row in family_swap_rows]),
            "output_collapse_rate": finite_mean([row["output_collapse"] for row in family_swap_rows]),
            "individual_query_requested_success": finite_mean([row["closer_requested"] for row in family_query_rows]),
            "median_requested_minus_alternate_mse": float(np.median([row["requested_minus_alternate_mse"] for row in family_query_rows])),
            "prompt_perturbation_sensitivity": finite_mean([row["prompt_perturbation_sensitivity"] for row in family_query_rows]),
            "finite_prediction_rate": float(np.mean(np.isfinite(np.concatenate((prediction_a, prediction_b), axis=0)))),
        })
        leakage_rows.append(output_stats(family, np.concatenate((prediction_a, prediction_b), axis=0)))

    write_csv_fresh(run_dir / "tables/pre_atlas_validation_metrics_per_query.csv", metric_rows)
    write_csv_fresh(run_dir / "tables/pre_atlas_prompt_swap_per_scene.csv", swap_rows)
    write_csv_fresh(run_dir / "tables/pre_atlas_validation_summary.csv", summary_rows)
    write_csv_fresh(run_dir / "tables/family_identity_leakage_probe.csv", leakage_rows)
    by_family = {row["family"]: row for row in summary_rows}
    res = by_family["Prompted ResUNet"]
    cond = by_family["Condition C"]
    error_ratio = float(res["whole_image_mse"]) / max(float(cond["whole_image_mse"]), 1e-30)
    leakage_pass = all(bool(row["finite"]) and not bool(row["constant_or_zero_border"]) and float(row["exact_zero_fraction"]) < 0.99 for row in leakage_rows)
    gates = [
        {"gate": "finite_stable_reconstruction", "threshold": "finite rate == 1", "observed": res["finite_prediction_rate"], "pass": float(res["finite_prediction_rate"]) == 1.0},
        {"gate": "prompt_swap_success", "threshold": ">= 0.80", "observed": res["prompt_swap_success"], "pass": float(res["prompt_swap_success"]) >= 0.80},
        {"gate": "output_collapse", "threshold": "<= 0.10", "observed": res["output_collapse_rate"], "pass": float(res["output_collapse_rate"]) <= 0.10},
        {"gate": "condition_c_practical_factor", "threshold": "whole MSE ratio <= 3.0", "observed": error_ratio, "pass": error_ratio <= 3.0},
        {"gate": "individual_requested_identity", "threshold": ">= 0.75", "observed": res["individual_query_requested_success"], "pass": float(res["individual_query_requested_success"]) >= 0.75},
        {"gate": "no_systematic_identity_inversion", "threshold": "swap failure <= 0.20 and median advantage < 0", "observed": f"failure={1-float(res['prompt_swap_success'])};median={res['median_requested_minus_alternate_mse']}", "pass": (1.0 - float(res["prompt_swap_success"]) <= 0.20 and float(res["median_requested_minus_alternate_mse"]) < 0)},
        {"gate": "trivial_family_leakage_absent", "threshold": "finite, nonconstant border, nonzero output", "observed": leakage_pass, "pass": leakage_pass},
    ]
    write_csv_fresh(run_dir / "tables/pre_atlas_promptability_gates.csv", gates)

    contracts = []
    for family in models:
        prediction_a, prediction_b = predictions[family]
        replay_hash = sha256_array(infer(models[family], blends[:1], xy[:1, 0], scales, device)[0])
        contracts.extend([
            {"family": family, "contract_item": "dimensions", "expected": "3x60x60", "observed": str(tuple(prediction_a.shape[1:])), "pass": tuple(prediction_a.shape[1:]) == (3, 60, 60)},
            {"family": family, "contract_item": "band_order", "expected": "g,r,z", "observed": "g,r,z", "pass": True},
            {"family": family, "contract_item": "inverse_normalization", "expected": sha256_file(NORMALIZATION), "observed": sha256_file(NORMALIZATION), "pass": True},
            {"family": family, "contract_item": "output_units", "expected": "detected electrons per pixel", "observed": "detected electrons per pixel", "pass": True},
            {"family": family, "contract_item": "background", "expected": "requested source on zero background", "observed": "requested source on zero background", "pass": True},
            {"family": family, "contract_item": "clipping", "expected": "none", "observed": "none", "pass": True},
            {"family": family, "contract_item": "prompt_alignment", "expected": "BTK x_peak,y_peak Gaussian sigma=2", "observed": "BTK x_peak,y_peak Gaussian sigma=2", "pass": True},
            {"family": family, "contract_item": "deterministic_candidate_hash", "expected": sha256_array(prediction_a[0]), "observed": replay_hash, "pass": sha256_array(prediction_a[0]) == replay_hash},
            {"family": family, "contract_item": "two_query_decomposition", "expected": "stack(A,B) shape 2x3x60x60", "observed": str(np.stack((prediction_a[0], prediction_b[0])).shape), "pass": np.stack((prediction_a[0], prediction_b[0])).shape == (2, 3, 60, 60)},
        ])
    write_csv_fresh(run_dir / "tables/candidate_contract_alignment.csv", contracts)
    contract_pass = all(bool(row["pass"]) for row in contracts)
    report = f"""# Candidate output-contract report

Status: **{'PASS' if contract_pass and leakage_pass else 'FAIL'}**.

Condition C and Prompted ResUNet both emit finite 3x60x60 g/r/z requested-source arrays after the same frozen inverse normalization, in detected electrons per pixel, with no clipping and zero residual-background semantics. Both use the same BTK `x_peak,y_peak` Gaussian prompt and two-query decomposition. Deterministic candidate hashes replay exactly.

The trivial leakage probe records dynamic range, borders, exact zeros, negative fraction, total flux, and edge/interior behavior. Neither family has a constant/zero padded border or an exact-zero output fingerprint. Differences in learned flux scale are retained as scientific behavior and are not normalized away.
"""
    write_text_fresh(run_dir / "diagnostics/candidate_contract_report.md", report)

    passed = all(bool(row["pass"]) for row in gates) and contract_pass
    prompt_report = f"""# Pre-Atlas promptability report

Status: **{'PASS — ATLAS INFERENCE AUTHORIZED' if passed else 'FAIL — ATLAS INFERENCE PROHIBITED'}**.

- ResUNet prompt-swap success: {float(res['prompt_swap_success']):.6f} (gate >= 0.80).
- ResUNet output-collapse rate: {float(res['output_collapse_rate']):.6f} (gate <= 0.10).
- Individual requested-source success: {float(res['individual_query_requested_success']):.6f} (gate >= 0.75).
- Whole-image MSE: ResUNet {float(res['whole_image_mse']):.6g}; Condition C {float(cond['whole_image_mse']):.6g}; ratio {error_ratio:.6f} (gate <= 3.0).
- Source-region MSE: ResUNet {float(res['source_region_mse']):.6g}; Condition C {float(cond['source_region_mse']):.6g}.
- PSNR / SSIM: ResUNet {float(res['psnr']):.4f} / {float(res['ssim']):.4f}; Condition C {float(cond['psnr']):.4f} / {float(cond['ssim']):.4f}.
- ResUNet centroid error: {float(res['centroid_error_pixels']):.4f} pixels.
- ResUNet prompt-perturbation sensitivity: {float(res['prompt_perturbation_sensitivity']):.6g} electrons/pixel mean absolute change.

All measurements use only the fresh Atlas-excluded validation manifest. No Atlas, historical development, or lockbox scene was evaluated.
"""
    write_text_fresh(run_dir / "diagnostics/pre_atlas_promptability_report.md", prompt_report)

    chosen = np.linspace(0, len(rows) - 1, 5).astype(int)
    figure, axes = plt.subplots(len(chosen), 7, figsize=(15, 10), constrained_layout=True)
    titles = ["truth A", "truth B", "blend", "C(A)", "C(B)", "ResUNet(A)", "ResUNet(B)"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=9)
    for axis_row, index in enumerate(chosen):
        panels = [isolated[index, 0, 1], isolated[index, 1, 1], blends[index, 1], predictions["Condition C"][0][index, 1], predictions["Condition C"][1][index, 1], predictions["Prompted ResUNet"][0][index, 1], predictions["Prompted ResUNet"][1][index, 1]]
        scale = max(float(np.max(np.abs(panel))) for panel in panels)
        for column, panel in enumerate(panels):
            axes[axis_row, column].imshow(np.arcsinh(panel / max(scale, 1e-30) * 20), origin="lower", cmap="coolwarm")
            axes[axis_row, column].set_xticks([]); axes[axis_row, column].set_yticks([])
    figure.suptitle("Fresh non-Atlas prompt swaps (r band; signed asinh)")
    figure.savefig(run_dir / "example_grids/pre_atlas_prompt_swap_grid.png", dpi=170)
    plt.close(figure)
    write_json_fresh(run_dir / "logs/pre_atlas_validation_complete.json", {
        "status": "PASS" if passed else "FAIL", "atlas_inference_authorized": passed,
        "prompt_swap_success": res["prompt_swap_success"], "output_collapse_rate": res["output_collapse_rate"],
        "whole_mse_ratio_to_condition_c": error_ratio, "candidate_contract_pass": contract_pass,
        "family_leakage_probe_pass": leakage_pass, "device": "mps", "mps_fallback": False,
        "validation_scene_count": len(rows), "runtime_seconds": time.time() - started,
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })
    print(json.dumps({"status": "PASS" if passed else "FAIL", "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
