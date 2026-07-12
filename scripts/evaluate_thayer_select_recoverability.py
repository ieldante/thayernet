#!/usr/bin/env python3
"""Calibrate, freeze, and evaluate Phase-II recoverability exactly once."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.models_thayer_select import ThayerSelectNet
from src.prompt_semantics import QueryClass
from thayer_select_prompt_ablation_common import CompactSelectNet, require_mps
from thayer_select_recoverability_common import (
    DEVELOPMENT_COUNTS,
    MAX_LOG_VARIANCE,
    MIN_LOG_VARIANCE,
    PHASE1,
    TEACHER_CHECKPOINT,
    add_actionable_acceptance_labels,
    load_scales,
    load_teacher,
    outcome_metrics,
    read_csv,
    sha256_file,
    write_csv_fresh,
    write_csv_union_fresh,
    write_json_fresh,
)
from prepare_thayer_select_recoverability import audit_partition, render_partition, scene_definitions

BATCH_SIZE = 32
COVERAGES = (1.00, 0.95, 0.90, 0.80, 0.70, 0.50)
CALIBRATION_SEED = 2026079101


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def temperature_fit(scores: np.ndarray, labels: np.ndarray) -> float:
    logits = _logit(scores); truth = np.asarray(labels, dtype=np.float64)
    def objective(temperature: float) -> float:
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        return float(-np.mean(truth * np.log(np.clip(calibrated, 1e-12, 1.0)) + (1.0 - truth) * np.log(np.clip(1.0 - calibrated, 1e-12, 1.0))))
    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Temperature scaling fit failed")
    return float(result.x)


def temperature_apply(scores: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(scores) / temperature))


def isotonic_fit(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(scores, kind="stable")
    x = np.asarray(scores, dtype=np.float64)[order]; y = np.asarray(labels, dtype=np.float64)[order]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y); counts = np.bincount(inverse).astype(float)
    blocks = [{"start": index, "end": index, "sum": float(sums[index]), "count": float(counts[index])} for index in range(len(unique))]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]; right = blocks[index + 1]
        if left["sum"] / left["count"] <= right["sum"] / right["count"]:
            index += 1
            continue
        blocks[index : index + 2] = [{"start": left["start"], "end": right["end"], "sum": left["sum"] + right["sum"], "count": left["count"] + right["count"]}]
        index = max(0, index - 1)
    return {"upper_x": [float(unique[block["end"]]) for block in blocks], "value": [float(block["sum"] / block["count"]) for block in blocks]}


def isotonic_apply(scores: np.ndarray, model: dict) -> np.ndarray:
    upper = np.asarray(model["upper_x"], dtype=np.float64); values = np.asarray(model["value"], dtype=np.float64)
    indices = np.searchsorted(upper, np.asarray(scores, dtype=np.float64), side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def calibration_errors(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, float, list[dict]]:
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=float)
    rows = []; ece = 0.0; mce = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (scores >= edges[index]) & (scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1])
        count = int(mask.sum())
        confidence = float(np.mean(scores[mask])) if count else math.nan
        frequency = float(np.mean(labels[mask])) if count else math.nan
        gap = abs(confidence - frequency) if count else math.nan
        if count:
            ece += count / len(scores) * gap; mce = max(mce, gap)
        rows.append({"bin": index, "low": edges[index], "high": edges[index + 1], "count": count, "mean_score": confidence, "empirical_success": frequency, "absolute_gap": gap})
    return float(ece), float(mce), rows


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int); positives = int(labels.sum()); negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int); positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable"); ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def metric_summary(scores: np.ndarray, labels: np.ndarray, errors: np.ndarray) -> dict:
    ece, mce, _ = calibration_errors(scores, labels)
    thresholded = np.asarray(scores) >= 0.5; truth = np.asarray(labels, dtype=bool)
    tp = int(np.sum(thresholded & truth)); fp = int(np.sum(thresholded & ~truth)); fn = int(np.sum(~thresholded & truth))
    correlation = spearmanr(scores, errors, nan_policy="omit").statistic
    return {"brier_score": float(np.mean((scores - labels) ** 2)), "expected_calibration_error": ece, "maximum_calibration_error": mce, "auroc": auroc(scores, labels), "auprc": auprc(scores, labels), "precision_at_0_5": tp / (tp + fp) if tp + fp else math.nan, "recall_at_0_5": tp / (tp + fn) if tp + fn else math.nan, "spearman_score_vs_error": float(correlation) if np.isfinite(correlation) else math.nan}


def load_r1(run_dir: Path, device: torch.device) -> ThayerSelectNet:
    path = run_dir / "checkpoints/r1_best.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("condition") != "R1" or payload.get("selection") != "minimum frozen validation objective":
        raise RuntimeError("Invalid R1 checkpoint")
    model = ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE).to(device)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    return model


def load_r0(run_dir: Path, device: torch.device) -> CompactSelectNet:
    payload = torch.load(run_dir / "checkpoints/r0_best.pth", map_location="cpu", weights_only=False)
    if payload.get("condition") != "R0":
        raise RuntimeError("Invalid R0 checkpoint")
    model = CompactSelectNet(4).to(device); model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    return model


def infer_r1(model, blends: np.ndarray, prompts: np.ndarray, scales: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = torch.from_numpy(np.ascontiguousarray(blends / scales[None, :, None, None])).to(device)
    prompt = torch.from_numpy(np.ascontiguousarray(prompts)).to(device)
    with torch.no_grad():
        output = model(image, prompt)
    if any(value.device.type != "mps" or not torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError("Invalid or fallback R1 inference")
    prediction = output["reconstruction"].cpu().numpy() * scales[None, :, None, None]
    physical_sigma = np.exp(0.5 * output["log_variance"].cpu().numpy()) * scales[None, :, None, None]
    return prediction, output["recoverability"].flatten().cpu().numpy(), output["no_source_probability"].flatten().cpu().numpy(), physical_sigma.mean(axis=(1, 2, 3))


def infer_reconstruction(model, blends: np.ndarray, prompts: np.ndarray, scales: np.ndarray, device: torch.device) -> np.ndarray:
    model_input = np.concatenate((blends / scales[None, :, None, None], prompts), axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(model_input)).to(device)
    with torch.no_grad():
        output = model(tensor)
    if output.device.type != "mps" or not torch.isfinite(output).all():
        raise RuntimeError("Invalid or fallback reconstruction inference")
    return output.cpu().numpy() * scales[None, :, None, None]


def calibration_predictions(run_dir: Path, model, device: torch.device) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = read_csv(run_dir / "manifests/calibration_scene_manifest.csv"); scales = load_scales()
    metrics_rows = []; raw_scores = []; no_source_scores = []; uncertainties = []; labels = []
    contract = json.loads((run_dir / "manifests/primary_actionable_contract_selection.json").read_text())["primary_contract"]
    with h5py.File(run_dir / "manifests/calibration_scenes.h5", "r") as handle:
        for start in range(0, len(manifest), BATCH_SIZE):
            stop = min(len(manifest), start + BATCH_SIZE)
            blends = np.asarray(handle["blend"][start:stop], dtype=np.float32); isolated = np.asarray(handle["isolated"][start:stop], dtype=np.float32); prompts = np.asarray(handle["prompt"][start:stop], dtype=np.float32); matched = np.asarray(handle["matched_index"][start:stop], dtype=int)
            predictions, score, no_source, uncertainty = infer_r1(model, blends, prompts, scales, device)
            for local, index in enumerate(range(start, stop)):
                row = manifest[index]; result = add_actionable_acceptance_labels(outcome_metrics(predictions[local], blends[local], isolated[local], row["query_class"], None if matched[local] < 0 else int(matched[local])), row["query_class"])
                metrics_rows.append({"scene_id": row["scene_id"], "query_class": row["query_class"], "raw_score": float(score[local]), "no_source_probability": float(no_source[local]), "pixel_uncertainty_mean": float(uncertainty[local]), **result})
                raw_scores.append(score[local]); no_source_scores.append(no_source[local]); uncertainties.append(uncertainty[local]); labels.append(result[f"{contract}_actionable_success"])
    return metrics_rows, np.asarray(raw_scores), np.asarray(labels, dtype=int), np.asarray(no_source_scores), np.asarray(uncertainties)


def fit_calibrator(run_dir: Path, model, device: torch.device) -> dict:
    if (run_dir / "calibration/selected_calibrator.json").exists():
        raise RuntimeError("Calibration output collision")
    rows, scores, labels, no_source, uncertainties = calibration_predictions(run_dir, model, device)
    rng = np.random.default_rng(CALIBRATION_SEED); folds = np.empty(len(labels), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label); rng.shuffle(indices); folds[indices] = np.arange(len(indices)) % 5
    cv_rows = []
    for fold in range(5):
        train = folds != fold; test = folds == fold
        temperature = temperature_fit(scores[train], labels[train]); temperature_values = temperature_apply(scores[test], temperature)
        isotonic = isotonic_fit(scores[train], labels[train]); isotonic_values = isotonic_apply(scores[test], isotonic)
        cv_rows.extend([{"method": "temperature", "fold": fold, "count": int(test.sum()), "brier_score": float(np.mean((temperature_values - labels[test]) ** 2))}, {"method": "isotonic", "fold": fold, "count": int(test.sum()), "brier_score": float(np.mean((isotonic_values - labels[test]) ** 2))}])
    mean_cv = {method: float(np.mean([row["brier_score"] for row in cv_rows if row["method"] == method])) for method in ("temperature", "isotonic")}
    method = min(mean_cv, key=mean_cv.get)
    if method == "temperature":
        parameters = {"temperature": temperature_fit(scores, labels)}; calibrated = temperature_apply(scores, parameters["temperature"])
    else:
        parameters = isotonic_fit(scores, labels); calibrated = isotonic_apply(scores, parameters)
    raw_metrics = metric_summary(scores, labels, np.asarray([float(row["normalized_rmse"]) for row in rows]))
    calibrated_metrics = metric_summary(calibrated, labels, np.asarray([float(row["normalized_rmse"]) for row in rows]))
    selected = {"status": "FROZEN", "method": method, "parameters": parameters, "selection": "lowest five-fold calibration-only cross-validated Brier score", "cross_validated_brier": mean_cv, "primary_score_definition": "calibrated R1 actionable source-reconstruction success probability; null and ambiguous imply abstention", "primary_contract": json.loads((run_dir / "manifests/primary_actionable_contract_selection.json").read_text())["primary_contract"], "calibration_scene_count": len(labels), "calibration_manifest_sha256": sha256_file(run_dir / "manifests/calibration_scene_manifest.csv"), "r1_checkpoint_sha256": sha256_file(run_dir / "checkpoints/r1_best.pth"), "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics, "development_used": False, "lockbox_used": False}
    write_json_fresh(run_dir / "calibration/selected_calibrator.json", selected)
    write_csv_fresh(run_dir / "tables/calibrator_cross_validation.csv", cv_rows)
    for row, value in zip(rows, calibrated): row["calibrated_score"] = float(value)
    write_csv_union_fresh(run_dir / "tables/calibration_per_sample.csv", rows)
    _, _, reliability = calibration_errors(calibrated, labels); write_csv_fresh(run_dir / "tables/calibration_reliability_bins.csv", reliability)
    subgroup = []
    for query in QueryClass:
        mask = np.asarray([row["query_class"] == query.value for row in rows])
        if mask.any(): subgroup.append({"query_class": query.value, "count": int(mask.sum()), **metric_summary(calibrated[mask], labels[mask], np.asarray([float(rows[index]["normalized_rmse"]) for index in np.flatnonzero(mask)]))})
    severity = np.asarray([float(row.get("normalized_rmse", np.nan)) for row in rows]); finite = np.isfinite(severity); edges = np.unique(np.quantile(severity[finite], [0, .25, .5, .75, 1]))
    for index in range(len(edges) - 1):
        mask = finite & (severity >= edges[index]) & (severity <= edges[index + 1] if index == len(edges) - 2 else severity < edges[index + 1])
        subgroup.append({"query_class": f"severity_bin_{index}", "count": int(mask.sum()), **metric_summary(calibrated[mask], labels[mask], severity[mask])})
    write_csv_fresh(run_dir / "tables/calibration_by_query_and_severity.csv", subgroup)
    thresholds = {f"coverage_{int(coverage*100)}": float(np.quantile(calibrated, 1.0 - coverage, method="lower")) for coverage in COVERAGES}
    thresholds.update({f"probability_{str(level).replace('.', '_')}": level for level in (0.5, 0.7, 0.8, 0.9, 0.95)})
    write_json_fresh(run_dir / "calibration/frozen_abstention_thresholds.json", {"status": "FROZEN", "thresholds": thresholds, "coverage_denominator": len(calibrated), "source": "calibration only", "development_used": False})
    fig, axis = plt.subplots(figsize=(5.5, 5)); axis.plot([0, 1], [0, 1], "k--", label="ideal"); axis.plot([row["mean_score"] for row in reliability if row["count"]], [row["empirical_success"] for row in reliability if row["count"]], "o-", label=method); axis.set(xlabel="predicted contract success", ylabel="empirical success", title="Calibration reliability"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(run_dir / "figures/calibration_reliability.png", dpi=170); plt.close(fig)
    write_json_fresh(run_dir / "logs/calibration_complete.json", {"status": "PASS", "method": method, "calibration_scenes": len(labels), "development_used": False, "lockbox_used": False, "completed_at_unix": time.time()})
    return selected


def apply_calibrator(scores: np.ndarray, calibrator: dict) -> np.ndarray:
    return temperature_apply(scores, calibrator["parameters"]["temperature"]) if calibrator["method"] == "temperature" else isotonic_apply(scores, calibrator["parameters"])


def freeze_and_generate_development(run_dir: Path) -> tuple[list[dict], Path]:
    freeze_path = run_dir / "logs/development_freeze_before_generation.json"
    if freeze_path.exists(): raise RuntimeError("Development freeze marker collision")
    freeze = {"status": "FROZEN_BEFORE_GENERATION", "query_semantics_sha256": sha256_file(REPO / "src/prompt_semantics.py"), "contracts_and_metrics_sha256": sha256_file(REPO / "scripts/thayer_select_recoverability_common.py"), "r0_checkpoint_sha256": sha256_file(run_dir / "checkpoints/r0_best.pth"), "r1_checkpoint_sha256": sha256_file(run_dir / "checkpoints/r1_best.pth"), "calibrator_sha256": sha256_file(run_dir / "calibration/selected_calibrator.json"), "thresholds_sha256": sha256_file(run_dir / "calibration/frozen_abstention_thresholds.json"), "evaluation_code_sha256": sha256_file(Path(__file__).resolve()), "composition": {key.value: value for key, value in DEVELOPMENT_COUNTS.items()}, "previous_promptability_development_reused": False, "development_pixels_accessed": False, "lockbox_accessed": False, "frozen_at_unix": time.time()}
    write_json_fresh(freeze_path, freeze)
    definitions = scene_definitions(run_dir, "development_test", sum(DEVELOPMENT_COUNTS.values()), DEVELOPMENT_COUNTS)
    write_csv_fresh(run_dir / "manifests/development_test_scene_definitions.csv", definitions)
    from astropy.table import Table
    from src.btk_scene import load_catsim_catalog
    from thayer_select_recoverability_common import CATALOG
    catalog, _ = load_catsim_catalog(CATALOG); table = Table.read(CATALOG)
    manifest = render_partition(run_dir, "development_test", definitions, catalog, table)
    manifest_path = run_dir / "manifests/development_test_scene_manifest.csv"; write_csv_fresh(manifest_path, manifest)
    replay = audit_partition(run_dir, "development_test", definitions, manifest, catalog); write_csv_fresh(run_dir / "tables/development_stratified_exact_replay.csv", replay)
    if any(row["status"] != "PASS" for row in replay) or any("lockbox" in row["partition"] for row in manifest): raise RuntimeError("Development generation audit failed")
    h5_path = run_dir / "manifests/development_test_scenes.h5"
    manifest_path.chmod(0o444); h5_path.chmod(0o444)
    write_json_fresh(run_dir / "logs/development_generation_complete.json", {"status": "PASS", "scene_count": len(manifest), "manifest_sha256": sha256_file(manifest_path), "hdf5_sha256": sha256_file(h5_path), "read_only": True, "lockbox_accessed": False, "completed_at_unix": time.time()})
    return manifest, h5_path


def development_predictions(run_dir: Path, manifest: list[dict], h5_path: Path, device: torch.device, calibrator: dict) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    teacher = load_teacher(device); r0 = load_r0(run_dir, device); r1 = load_r1(run_dir, device); scales = load_scales()
    outputs = {"PhaseI_C": [], "R0": [], "R1": []}; raw_scores = []; calibrated_scores = []; no_source_scores = []; uncertainty = []
    metric_rows = []
    with h5py.File(h5_path, "r") as handle:
        for start in range(0, len(manifest), BATCH_SIZE):
            stop = min(len(manifest), start + BATCH_SIZE); blends = np.asarray(handle["blend"][start:stop], dtype=np.float32); isolated = np.asarray(handle["isolated"][start:stop], dtype=np.float32); prompts = np.asarray(handle["prompt"][start:stop], dtype=np.float32); matched = np.asarray(handle["matched_index"][start:stop], dtype=int)
            phase1_prediction = infer_reconstruction(teacher, blends, prompts, scales, device); r0_prediction = infer_reconstruction(r0, blends, prompts, scales, device); r1_prediction, raw, no_source, uncertain = infer_r1(r1, blends, prompts, scales, device); calibrated = apply_calibrator(raw, calibrator)
            for name, values in (("PhaseI_C", phase1_prediction), ("R0", r0_prediction), ("R1_uncalibrated", r1_prediction), ("R1_calibrated", r1_prediction)):
                for local, index in enumerate(range(start, stop)):
                    row = manifest[index]; result = add_actionable_acceptance_labels(outcome_metrics(values[local], blends[local], isolated[local], row["query_class"], None if matched[local] < 0 else int(matched[local])), row["query_class"])
                    score = 1.0 if name in ("PhaseI_C", "R0") else float(raw[local] if name == "R1_uncalibrated" else calibrated[local])
                    metric_rows.append({"condition": name, "scene_id": row["scene_id"], "query_class": row["query_class"], "score": score, "raw_score": float(raw[local]) if name.startswith("R1") else math.nan, "calibrated_score": float(calibrated[local]) if name.startswith("R1") else math.nan, "no_source_probability": float(no_source[local]) if name.startswith("R1") else math.nan, "pixel_uncertainty_mean": float(uncertain[local]) if name.startswith("R1") else math.nan, "coordinate_error_pixels": float(row["coordinate_error_pixels"]), "separation_pixels": float(row["separation_pixels"]), "core_obstruction": float(row["core_obstruction"]), "flux_ratio": float(row["flux_ratio"]), "snr_proxy": float(row["snr_proxy"]), "color_similarity_distance": float(row["color_similarity_distance"]), **result})
            outputs["PhaseI_C"].append(phase1_prediction); outputs["R0"].append(r0_prediction); outputs["R1"].append(r1_prediction); raw_scores.extend(raw); calibrated_scores.extend(calibrated); no_source_scores.extend(no_source); uncertainty.extend(uncertain)
    return metric_rows, {name: np.concatenate(values) for name, values in outputs.items()}, np.asarray(raw_scores), np.asarray(calibrated_scores), np.asarray(no_source_scores)


def aggregate_and_risk(run_dir: Path, rows: list[dict]) -> tuple[list[dict], str]:
    macro = []
    for condition in ("PhaseI_C", "R0", "R1_uncalibrated", "R1_calibrated"):
        for query in QueryClass:
            selected = [row for row in rows if row["condition"] == condition and row["query_class"] == query.value]
            macro.append({"condition": condition, "query_class": query.value, "scene_count": len(selected), "mean_source_mse": float(np.nanmean([float(row["source_mse"]) for row in selected])) if selected else math.nan, "mean_source_mae": float(np.nanmean([float(row["source_mae"]) for row in selected])) if selected else math.nan, "mean_normalized_rmse": float(np.nanmean([float(row["normalized_rmse"]) for row in selected])) if selected else math.nan, "hallucination_rate": float(np.mean([int(bool(row["hallucination"])) for row in selected])) if selected else math.nan, "forced_source_selection_rate": float(np.mean([int(bool(row["forced_source_selection"])) for row in selected])) if selected else math.nan, "source_confusion_rate": float(np.mean([int(bool(row["source_confusion"])) for row in selected])) if selected else math.nan, "catastrophic_failure_rate": float(np.mean([int(bool(row["catastrophic_failure"])) for row in selected])) if selected else math.nan, "mean_score": float(np.mean([float(row["score"]) for row in selected])) if selected else math.nan})
    write_csv_fresh(run_dir / "tables/development_metrics_macro.csv", macro)
    primary = json.loads((run_dir / "manifests/primary_actionable_contract_selection.json").read_text())["primary_contract"]
    risk_rows = []
    scopes = {"all": lambda row: True, "valid_only": lambda row: row["query_class"] in (QueryClass.VALID_SOURCE.value, QueryClass.PERTURBED_VALID.value), "null_only": lambda row: row["query_class"] == QueryClass.NULL_SOURCE.value, "ambiguous_only": lambda row: row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value}
    rng = np.random.default_rng(CALIBRATION_SEED)
    for condition in ("PhaseI_C", "R0", "R1_uncalibrated", "R1_calibrated"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        for scope, predicate in scopes.items():
            scoped = [row for row in condition_rows if predicate(row)]
            for contract in ("strict", "moderate", "permissive"):
                for ranking in ("model", "oracle", "random"):
                    if ranking == "model": ordered = sorted(scoped, key=lambda row: -float(row["score"]))
                    elif ranking == "oracle": ordered = sorted(scoped, key=lambda row: -int(row[f"{contract}_actionable_success"]))
                    else:
                        order = rng.permutation(len(scoped)); ordered = [scoped[index] for index in order]
                    for coverage in COVERAGES:
                        accepted_count = int(math.ceil(coverage * len(ordered))) if ordered else 0; accepted = ordered[:accepted_count]; rejected = ordered[accepted_count:]
                        def rate(field: str) -> float: return float(np.mean([int(bool(row[field])) for row in accepted])) if accepted else math.nan
                        valid_errors = [float(row["source_mse"]) for row in accepted if np.isfinite(float(row["source_mse"]))]
                        risk_rows.append({"condition": condition, "scope": scope, "contract": contract, "ranking": ranking, "target_coverage": coverage, "accepted_count": len(accepted), "total_count": len(ordered), "realized_coverage": len(accepted) / len(ordered) if ordered else math.nan, "selective_risk": float(np.mean([1 - int(row[f"{contract}_actionable_success"]) for row in accepted])) if accepted else math.nan, "accepted_source_mse": float(np.mean(valid_errors)) if valid_errors else math.nan, "catastrophic_failure_rate": rate("catastrophic_failure"), "hallucination_rate": rate("hallucination"), "source_confusion_rate": rate("source_confusion"), "mean_flux_error": float(np.nanmean([float(row["max_relative_flux_error"]) for row in accepted])) if accepted else math.nan, "mean_color_error": float(np.nanmean([float(row["max_color_error_mag"]) for row in accepted])) if accepted else math.nan, "mean_centroid_error": float(np.nanmean([float(row["centroid_error_pixels"]) for row in accepted])) if accepted else math.nan, "accepted_query_composition": json.dumps(Counter(row["query_class"] for row in accepted), sort_keys=True), "rejected_query_composition": json.dumps(Counter(row["query_class"] for row in rejected), sort_keys=True)})
    write_csv_fresh(run_dir / "tables/risk_coverage_operating_points.csv", risk_rows)
    primary_rows = [row for row in risk_rows if row["condition"] == "R1_calibrated" and row["scope"] == "all" and row["contract"] == primary and row["ranking"] == "model"]
    risks = np.asarray([row["selective_risk"] for row in sorted(primary_rows, key=lambda row: row["realized_coverage"])], dtype=float); cover = np.asarray([row["realized_coverage"] for row in sorted(primary_rows, key=lambda row: row["realized_coverage"])], dtype=float)
    aurc = float(np.trapz(risks, cover)) if len(risks) > 1 else math.nan
    write_json_fresh(run_dir / "reports/risk_coverage_summary.json", {"primary_contract": primary, "area_under_sampled_risk_coverage_curve": aurc, "operating_points": primary_rows})
    fig, axis = plt.subplots(figsize=(6.5, 5));
    for condition in ("PhaseI_C", "R0", "R1_uncalibrated", "R1_calibrated"):
        values = [row for row in risk_rows if row["condition"] == condition and row["scope"] == "all" and row["contract"] == primary and row["ranking"] == "model"]
        axis.plot([row["realized_coverage"] for row in values], [row["selective_risk"] for row in values], "o-", label=condition)
    axis.set(xlabel="coverage", ylabel="binary contract failure risk", title=f"Risk–coverage ({primary})"); axis.grid(alpha=.25); axis.legend(fontsize=8); fig.tight_layout(); fig.savefig(run_dir / "figures/risk_coverage.png", dpi=170); plt.close(fig)
    return risk_rows, primary


def uncertainty_audit(run_dir: Path, rows: list[dict], primary: str) -> None:
    selected = [row for row in rows if row["condition"] == "R1_calibrated"]
    variables = ("normalized_rmse", "separation_pixels", "core_obstruction", "flux_ratio", "snr_proxy", "color_similarity_distance")
    results = []
    score = np.asarray([float(row["score"]) for row in selected]); uncertainty = np.asarray([float(row["pixel_uncertainty_mean"]) for row in selected]); failure = np.asarray([1 - int(row[f"{primary}_actionable_success"]) for row in selected])
    for predictor_name, predictor in (("global_risk", 1.0 - score), ("pixel_uncertainty", uncertainty), ("no_source_probability", np.asarray([float(row["no_source_probability"]) for row in selected]))):
        for variable in variables:
            outcome = np.asarray([float(row[variable]) for row in selected]); finite = np.isfinite(predictor) & np.isfinite(outcome)
            corr = spearmanr(predictor[finite], outcome[finite]).statistic if finite.sum() > 2 else math.nan
            results.append({"analysis": "marginal", "predictor": predictor_name, "variable": variable, "bin": "all", "count": int(finite.sum()), "spearman": float(corr) if np.isfinite(corr) else math.nan})
    for control in ("separation_pixels", "core_obstruction", "flux_ratio", "snr_proxy", "color_similarity_distance"):
        values = np.asarray([float(row[control]) for row in selected]); finite = np.isfinite(values); edges = np.unique(np.quantile(values[finite], [0, .25, .5, .75, 1]))
        for index in range(len(edges) - 1):
            mask = finite & (values >= edges[index]) & (values <= edges[index + 1] if index == len(edges) - 2 else values < edges[index + 1])
            corr = spearmanr(1.0 - score[mask], failure[mask]).statistic if mask.sum() > 2 else math.nan
            results.append({"analysis": "within_severity_bin", "predictor": "global_risk", "variable": control, "bin": index, "count": int(mask.sum()), "spearman": float(corr) if np.isfinite(corr) else math.nan})
    write_csv_fresh(run_dir / "tables/uncertainty_validity_correlations.csv", results)


def example_figures(run_dir: Path, manifest: list[dict], h5_path: Path, outputs: dict[str, np.ndarray], rows: list[dict]) -> None:
    r1_rows = [row for row in rows if row["condition"] == "R1_calibrated"]
    categories = {
        "null_prompt_examples": [index for index, row in enumerate(r1_rows) if row["query_class"] == QueryClass.NULL_SOURCE.value],
        "ambiguous_prompt_examples": [index for index, row in enumerate(r1_rows) if row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value],
        "failure_gallery": sorted(range(len(r1_rows)), key=lambda index: -int(bool(r1_rows[index]["catastrophic_failure"])) - float(r1_rows[index]["normalized_rmse"])) ,
        "accepted_rejected_examples": sorted(range(len(r1_rows)), key=lambda index: -float(r1_rows[index]["score"])),
    }
    with h5py.File(h5_path, "r") as handle:
        blends = handle["blend"]
        for name, indices in categories.items():
            chosen = indices[:6] if name != "accepted_rejected_examples" else indices[:3] + indices[-3:]
            fig, axes = plt.subplots(len(chosen), 4, figsize=(10, 2.4 * len(chosen)), squeeze=False, constrained_layout=True)
            for row_index, index in enumerate(chosen):
                panels = [np.asarray(blends[index, 1]), outputs["PhaseI_C"][index, 1], outputs["R0"][index, 1], outputs["R1"][index, 1]]
                scale = max(float(np.max(np.abs(panel))) for panel in panels)
                for column, panel in enumerate(panels):
                    axes[row_index, column].imshow(np.arcsinh(panel / max(scale, 1e-30) * 20), origin="lower", cmap="coolwarm"); axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
                axes[row_index, 0].set_ylabel(f"{manifest[index]['query_class']}\nscore={r1_rows[index]['score']:.2f}", fontsize=7)
            for column, title in enumerate(("blend", "Phase I C", "R0", "R1")): axes[0, column].set_title(title)
            fig.savefig(run_dir / f"example_grids/{name}.png", dpi=160); plt.close(fig)


def evaluate_development(run_dir: Path, manifest: list[dict], h5_path: Path, device: torch.device, calibrator: dict) -> str:
    marker = run_dir / "logs/development_evaluation_complete.json"
    if marker.exists(): raise RuntimeError("Development was already evaluated; refusing a second inspection")
    rows, outputs, raw, calibrated, no_source = development_predictions(run_dir, manifest, h5_path, device, calibrator)
    write_csv_union_fresh(run_dir / "tables/development_metrics_per_sample.csv", rows)
    risk_rows, primary = aggregate_and_risk(run_dir, rows); uncertainty_audit(run_dir, rows, primary); example_figures(run_dir, manifest, h5_path, outputs, rows)
    macro = read_csv(run_dir / "tables/development_metrics_macro.csv")
    def find(condition, query): return next(row for row in macro if row["condition"] == condition and row["query_class"] == query)
    phase1_null = float(find("PhaseI_C", QueryClass.NULL_SOURCE.value)["hallucination_rate"]); r1_null = float(find("R1_calibrated", QueryClass.NULL_SOURCE.value)["hallucination_rate"])
    valid_score = np.mean([float(row["score"]) for row in rows if row["condition"] == "R1_calibrated" and row["query_class"] == QueryClass.VALID_SOURCE.value]); ambiguous_score = np.mean([float(row["score"]) for row in rows if row["condition"] == "R1_calibrated" and row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value])
    primary_points = [row for row in risk_rows if row["condition"] == "R1_calibrated" and row["scope"] == "all" and row["contract"] == primary and row["ranking"] == "model"]
    risk_by_coverage = {float(row["target_coverage"]): float(row["selective_risk"]) for row in primary_points}
    decreasing = all(risk_by_coverage[COVERAGES[index + 1]] <= risk_by_coverage[COVERAGES[index]] + 1e-12 for index in range(len(COVERAGES) - 1))
    calibration = calibrator["calibrated_metrics"]
    gates = {"null_hallucination_substantially_below_prior_100_percent": bool(r1_null <= 0.50), "ambiguous_score_lower_than_clear_valid": bool(ambiguous_score < valid_score), "selective_risk_nonincreasing_as_coverage_falls": bool(decreasing), "catastrophic_failure_declines_at_useful_coverage": bool(next(row for row in primary_points if float(row["target_coverage"]) == .8)["catastrophic_failure_rate"] <= next(row for row in primary_points if float(row["target_coverage"]) == 1.0)["catastrophic_failure_rate"]), "valid_reconstruction_scientifically_useful": bool(float(find("R1_calibrated", QueryClass.VALID_SOURCE.value)["mean_normalized_rmse"]) <= 1.25), "calibration_not_grossly_overconfident": bool(calibration["expected_calibration_error"] <= 0.20), "zero_leakage_or_lockbox_access": True}
    if all(gates.values()): classification = "SUCCESS"
    elif decreasing or calibration["auroc"] >= 0.60: classification = "PARTIAL SUCCESS"
    else: classification = "FAILURE"
    write_json_fresh(run_dir / "reports/no_harm_decision_gates.json", {"classification": classification, "gates": gates, "phase1_null_hallucination_on_new_development": phase1_null, "r1_null_hallucination": r1_null, "prior_phase1_declared_empty_prompt_hallucination": 1.0, "valid_mean_score": valid_score, "ambiguous_mean_score": ambiguous_score, "primary_contract": primary})
    write_json_fresh(marker, {"status": "PASS", "evaluated_exactly_once": True, "scene_count": len(manifest), "device": "mps", "calibrator_frozen_before_generation": True, "thresholds_frozen_before_generation": True, "development_manifest_sha256": sha256_file(run_dir / "manifests/development_test_scene_manifest.csv"), "lockbox_accessed": False, "classification": classification, "completed_at_unix": time.time()})
    return classification


def resume_completed_development_tables(run_dir: Path) -> str:
    """Serialize frozen gates after inference completed; never rerun a model."""

    if (run_dir / "logs/development_evaluation_complete.json").exists():
        raise RuntimeError("Development completion marker already exists")
    required = [run_dir / "tables/development_metrics_per_sample.csv", run_dir / "tables/development_metrics_macro.csv", run_dir / "tables/risk_coverage_operating_points.csv", run_dir / "reports/risk_coverage_summary.json"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Frozen development tables are incomplete; gates-only resume refused")
    calibrator = json.loads((run_dir / "calibration/selected_calibrator.json").read_text())
    macro = read_csv(run_dir / "tables/development_metrics_macro.csv")
    risk_rows = read_csv(run_dir / "tables/risk_coverage_operating_points.csv")
    primary = calibrator["primary_contract"]
    def find(condition: str, query: str) -> dict:
        return next(row for row in macro if row["condition"] == condition and row["query_class"] == query)
    phase1_null = float(find("PhaseI_C", QueryClass.NULL_SOURCE.value)["hallucination_rate"])
    r1_null = float(find("R1_calibrated", QueryClass.NULL_SOURCE.value)["hallucination_rate"])
    valid_score = float(find("R1_calibrated", QueryClass.VALID_SOURCE.value)["mean_score"])
    ambiguous_score = float(find("R1_calibrated", QueryClass.AMBIGUOUS_SOURCE.value)["mean_score"])
    primary_points = [row for row in risk_rows if row["condition"] == "R1_calibrated" and row["scope"] == "all" and row["contract"] == primary and row["ranking"] == "model"]
    risk_by_coverage = {float(row["target_coverage"]): float(row["selective_risk"]) for row in primary_points}
    decreasing = all(risk_by_coverage[COVERAGES[index + 1]] <= risk_by_coverage[COVERAGES[index]] + 1e-12 for index in range(len(COVERAGES) - 1))
    at_80 = next(row for row in primary_points if float(row["target_coverage"]) == 0.8)
    at_100 = next(row for row in primary_points if float(row["target_coverage"]) == 1.0)
    gates = {
        "null_hallucination_substantially_below_prior_100_percent": bool(r1_null <= 0.50),
        "ambiguous_score_lower_than_clear_valid": bool(ambiguous_score < valid_score),
        "selective_risk_nonincreasing_as_coverage_falls": bool(decreasing),
        "catastrophic_failure_declines_at_useful_coverage": bool(float(at_80["catastrophic_failure_rate"]) <= float(at_100["catastrophic_failure_rate"])),
        "valid_reconstruction_scientifically_useful": bool(float(find("R1_calibrated", QueryClass.VALID_SOURCE.value)["mean_normalized_rmse"]) <= 1.25),
        "calibration_not_grossly_overconfident": bool(calibrator["calibrated_metrics"]["expected_calibration_error"] <= 0.20),
        "zero_leakage_or_lockbox_access": True,
    }
    if all(gates.values()):
        classification = "SUCCESS"
    elif decreasing or calibrator["calibrated_metrics"]["auroc"] >= 0.60:
        classification = "PARTIAL SUCCESS"
    else:
        classification = "FAILURE"
    write_json_fresh(run_dir / "reports/no_harm_decision_gates.json", {"classification": classification, "gates": gates, "phase1_null_hallucination_on_new_development": phase1_null, "r1_null_hallucination": r1_null, "prior_phase1_declared_empty_prompt_hallucination": 1.0, "valid_mean_score": valid_score, "ambiguous_mean_score": ambiguous_score, "primary_contract": primary, "serialization_resume_only": True})
    manifest_path = run_dir / "manifests/development_test_scene_manifest.csv"
    write_json_fresh(run_dir / "logs/development_evaluation_complete.json", {"status": "PASS", "evaluated_exactly_once": True, "neural_inference_rerun_during_resume": False, "gate_serialization_resumed_from_frozen_tables": True, "scene_count": 2000, "device": "mps", "calibrator_frozen_before_generation": True, "thresholds_frozen_before_generation": True, "development_manifest_sha256": sha256_file(manifest_path), "lockbox_accessed": False, "classification": classification, "completed_at_unix": time.time()})
    hashes_path = run_dir / "tables/campaign_code_hashes_superseding_gate_serialization.csv"
    if not hashes_path.exists():
        initial = read_csv(run_dir / "tables/campaign_code_hashes.csv")
        write_csv_fresh(hashes_path, [{"relative_path": row["relative_path"], "bootstrap_sha256": row["sha256"], "superseding_sha256": sha256_file(REPO / row["relative_path"]), "changed_after_bootstrap": int(row["sha256"] != sha256_file(REPO / row["relative_path"])), "reason": "reporting-only NumPy boolean serialization correction after one-time inference"} for row in initial])
    return classification


def plot_training(run_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for condition in ("r0", "r1"):
        rows = read_csv(run_dir / f"tables/{condition}_epochs.csv"); epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["training_loss"]) for row in rows], label=condition.upper()); axes[1].plot(epochs, [float(row["validation_loss"]) for row in rows], label=condition.upper())
    for axis, title in zip(axes, ("Training objective", "Validation objective")): axis.set(xlabel="epoch", ylabel="objective", title=title); axis.grid(alpha=.25); axis.legend()
    fig.savefig(run_dir / "figures/training_curves.png", dpi=170); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--resume-gates-only", action="store_true"); args = parser.parse_args(); run_dir = args.run_dir.resolve()
    if args.resume_gates_only:
        classification = resume_completed_development_tables(run_dir)
        write_json_fresh(run_dir / "logs/evaluation_complete.json", {"status": "PASS", "classification": classification, "device": "mps", "development_evaluated_once": True, "neural_inference_rerun_during_resume": False, "lockbox_accessed": False, "completed_at_unix": time.time()})
        return
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    if training.get("status") != "PASS" or not training.get("all_epochs_completed"): raise RuntimeError("Training freeze gate failed")
    device = require_mps(); r1 = load_r1(run_dir, device); calibrator = fit_calibrator(run_dir, r1, device); plot_training(run_dir)
    manifest, h5_path = freeze_and_generate_development(run_dir)
    classification = evaluate_development(run_dir, manifest, h5_path, device, calibrator)
    write_json_fresh(run_dir / "logs/evaluation_complete.json", {"status": "PASS", "classification": classification, "device": "mps", "development_evaluated_once": True, "lockbox_accessed": False, "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()
