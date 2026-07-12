#!/usr/bin/env python3
"""Calibration-only feasibility analysis for frozen hierarchical heads.

No model or operational accept/abstain threshold is selected here.  The
natural calibration partition is opened only after all validation selections
and five-seed head fits are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from scripts.train_hierarchical_query_gate import CLASS_TO_INDEX, QueryNet, SEEDS as QUERY_SEEDS
from scripts.train_hierarchical_risk_heads import BinaryNet, RiskNet, SEEDS as RISK_SEEDS, transformed
from src.hierarchical_safety import conformal_upper_offset


ALPHA = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, value: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    value.to_csv(path, index=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -40, 40)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int)
    positives = int(truth.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def negative_log_likelihood(probability: np.ndarray, truth: np.ndarray) -> float:
    return float(-np.mean(np.log(probability[np.arange(len(truth)), truth] + 1e-12)))


def brier_multiclass(probability: np.ndarray, truth: np.ndarray) -> float:
    one_hot = np.eye(probability.shape[1])[truth]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def ece_multiclass(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(axis=1)
    correct = np.argmax(probability, axis=1) == truth
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if mask.any():
            total += mask.mean() * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(total)


def score_structure(values: np.ndarray) -> dict:
    rounded = np.round(np.asarray(values, dtype=np.float64).reshape(-1), 12)
    _, counts = np.unique(rounded, return_counts=True)
    return {"unique_score_count": int(len(counts)), "tie_fraction": float(1.0 - len(counts) / len(rounded)),
            "largest_plateau": int(counts.max()), "largest_plateau_fraction": float(counts.max() / len(rounded))}


def fit_temperature_multiclass(logits: np.ndarray, truth: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(softmax(logits / math.exp(log_temperature)), truth)
    result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature scaling failed")
    return float(math.exp(result.x))


def fit_vector_scaling(logits: np.ndarray, truth: np.ndarray) -> dict:
    x = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(truth, dtype=np.int64))
    log_scale = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(x.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale, bias], lr=0.5, max_iter=150, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        adjusted = x / torch.exp(log_scale).clamp(0.05, 20.0) + (bias - bias.mean())
        loss = F.cross_entropy(adjusted, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return {"temperature_by_class": torch.exp(log_scale).detach().numpy().astype(float).tolist(),
            "centered_bias_by_class": (bias - bias.mean()).detach().numpy().astype(float).tolist()}


def apply_vector(logits: np.ndarray, parameters: dict) -> np.ndarray:
    adjusted = np.asarray(logits) / np.asarray(parameters["temperature_by_class"])[None]
    adjusted += np.asarray(parameters["centered_bias_by_class"])[None]
    return softmax(adjusted)


def query_logits(run: Path, features: np.lib.npyio.NpzFile) -> np.ndarray:
    selection = json.loads((run / "manifests/query_gate_selection.json").read_text())
    feature = selection["selected_feature_family"]
    family = selection["selected_head_family"]
    blocks = []
    for seed in QUERY_SEEDS:
        payload = torch.load(run / f"models/query_gate_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model = QueryNet(features[feature].shape[1], family)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            blocks.append(model(torch.from_numpy(transformed(features[feature], payload["mean"], payload["scale"]))).numpy())
    return np.mean(blocks, axis=0)


def risk_outputs(run: Path, features: np.lib.npyio.NpzFile, task: str) -> np.ndarray:
    selection = json.loads((run / "manifests/risk_head_selection.json").read_text())["risk_heads"][task]
    feature = selection["feature_family"]
    family = selection["head_family"]
    blocks = []
    for seed in RISK_SEEDS:
        payload = torch.load(run / f"models/{task}_risk_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model = RiskNet(features[feature].shape[1], family)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            blocks.append(model(torch.from_numpy(transformed(features[feature], payload["mean"], payload["scale"]))).numpy())
    return np.mean(blocks, axis=0)


def binary_logits(run: Path, features: np.lib.npyio.NpzFile, task: str) -> np.ndarray:
    if task == "confusion":
        selection = json.loads((run / "manifests/risk_head_selection.json").read_text())["confusion_head"]
        stem = "confusion"
    else:
        selection = json.loads((run / "manifests/catastrophic_head_selection.json").read_text())
        stem = "catastrophic"
    feature = selection["feature_family"]
    family = selection["head_family"]
    blocks = []
    for seed in RISK_SEEDS:
        payload = torch.load(run / f"models/{stem}_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model = BinaryNet(features[feature].shape[1], family)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            blocks.append(model(torch.from_numpy(transformed(features[feature], payload["mean"], payload["scale"]))).numpy())
    return np.mean(blocks, axis=0)


def fit_binary_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        probability = sigmoid(logits / math.exp(log_temperature))
        return float(-np.mean(truth * np.log(probability + 1e-12) + (1 - truth) * np.log(1 - probability + 1e-12)))
    result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    if not result.success:
        raise RuntimeError("Binary temperature scaling failed")
    return float(math.exp(result.x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    required = ["query_gate_training_complete.json", "risk_head_training_complete.json", "catastrophic_head_training_complete.json"]
    if any(json.loads((run / "logs" / name).read_text())["status"] != "PASS" for name in required):
        raise RuntimeError("All validation-selected heads must be frozen before calibration")
    if (run / "logs/calibration_feasibility_complete.json").exists():
        raise FileExistsError("Calibration already completed")
    write_json_fresh(run / "logs/calibration_code_freeze.json", {
        "status": "HASHED_BEFORE_CALIBRATION_ACCESS",
        "relative_path": relative(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "preregistered_methods_unchanged": True,
        "used_for_head_fitting_or_selection": False,
        "calibration_access_started": False,
        "development_accessed": False,
        "lockbox_accessed": False,
    })
    started = time.time()
    features = np.load(run / "features/v2_natural_calibration_features.npz", allow_pickle=True)
    samples = pd.read_csv(run / "features/v4_natural_calibration_samples.csv", keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(run / "manifests/v2_natural_calibration_scene_manifest.csv", keep_default_na=False)
    if features["scene_id"].astype(str).tolist() != samples.scene_id.astype(str).tolist() or samples.scene_id.tolist() != manifest.scene_id.tolist():
        raise RuntimeError("Calibration alignment failed")
    truth_query = np.asarray([CLASS_TO_INDEX[value] for value in samples.query_state], dtype=int)
    logits = query_logits(run, features)
    raw = softmax(logits)
    temperature = fit_temperature_multiclass(logits, truth_query)
    temp_probability = softmax(logits / temperature)
    vector_parameters = fit_vector_scaling(logits, truth_query)
    vector_probability = apply_vector(logits, vector_parameters)
    query_rows = []
    for method, probability, parameters in (("raw", raw, {}), ("temperature", temp_probability, {"temperature": temperature}),
                                             ("vector", vector_probability, vector_parameters)):
        structure = score_structure(probability)
        query_rows.append({"method": method, "negative_log_likelihood": negative_log_likelihood(probability, truth_query),
                           "brier": brier_multiclass(probability, truth_query), "ece": ece_multiclass(probability, truth_query),
                           **structure, "parameters": json.dumps(parameters, sort_keys=True)})
    query_frame = pd.DataFrame(query_rows)
    write_csv_fresh(run / "tables/query_calibration_comparison.csv", query_frame)
    selected_method = str(query_frame.sort_values("negative_log_likelihood", kind="stable").iloc[0].method)
    selected_probability = {"raw": raw, "temperature": temp_probability, "vector": vector_probability}[selected_method]
    query_predictions = pd.DataFrame({"scene_id": samples.scene_id, "query_state": samples.query_state,
                                      "p_unique": selected_probability[:, 0], "p_null": selected_probability[:, 1],
                                      "p_ambiguous": selected_probability[:, 2]})
    write_csv_fresh(run / "calibration/query_state_calibration_predictions.csv", query_predictions)
    write_json_fresh(run / "calibration/query_state_calibrator.json", {"selected_for_probability_calibration_only": selected_method,
        "selection_criterion": "minimum natural-calibration negative log likelihood", "temperature": temperature,
        "vector_parameters": vector_parameters, "operational_threshold_selected": False})

    valid = samples.query_state == "UNIQUE_VALID"
    valid_indices = np.flatnonzero(valid.to_numpy())
    subgroup = pd.DataFrame({"snr_bin": pd.qcut(manifest.loc[valid, "snr_proxy"], 4, labels=["snr_q1", "snr_q2", "snr_q3", "snr_q4"], duplicates="drop"),
                             "overlap_bin": pd.qcut(manifest.loc[valid, "core_obstruction"], 4, labels=["overlap_q1", "overlap_q2", "overlap_q3", "overlap_q4"], duplicates="drop")}, index=valid_indices)
    risk_rows = []
    subgroup_rows = []
    risk_predictions = pd.DataFrame({"scene_id": samples.loc[valid, "scene_id"].to_numpy()})
    for task, truth_column in (("image", "image_risk"), ("flux", "flux_risk_max"), ("centroid", "centroid_risk_pixels")):
        output = risk_outputs(run, features, task)[valid.to_numpy()]
        truth_raw = samples.loc[valid, truth_column].to_numpy(dtype=float)
        truth_log = np.log1p(truth_raw)
        median_log = output[:, 0]
        quantile_log = output[:, 1]
        median_raw = np.maximum(np.expm1(np.clip(median_log, -30, 30)), 0.0)
        base_quantile_raw = np.maximum(np.expm1(np.clip(quantile_log, -30, 30)), 0.0)
        conformal_offset = conformal_upper_offset(truth_log - median_log, ALPHA)
        quantile_offset = conformal_upper_offset(truth_log - quantile_log, ALPHA)
        conformal_upper = np.maximum(np.expm1(np.clip(median_log + conformal_offset, -30, 30)), 0.0)
        quantile_corrected = np.maximum(np.expm1(np.clip(quantile_log + quantile_offset, -30, 30)), 0.0)
        for method, upper, offset in (("split_conformal_median_residual", conformal_upper, conformal_offset),
                                      ("quantile_residual_correction", quantile_corrected, quantile_offset)):
            width = np.maximum(upper - median_raw, 0.0)
            structure = score_structure(upper)
            risk_rows.append({"risk": task, "method": method, "calibration_rows": len(truth_raw), "offset_log1p": offset,
                              "empirical_coverage": float(np.mean(truth_raw <= upper)), "mean_interval_width": float(np.mean(width)),
                              "median_interval_width": float(np.median(width)), "mae": float(np.mean(np.abs(median_raw - truth_raw))),
                              "median_absolute_error": float(np.median(np.abs(median_raw - truth_raw))),
                              "spearman": float(spearmanr(median_raw, truth_raw).statistic), **structure})
            for kind in ("snr_bin", "overlap_bin"):
                for level in subgroup[kind].dropna().unique():
                    mask = subgroup[kind].to_numpy() == level
                    subgroup_rows.append({"risk": task, "method": method, "subgroup_family": kind, "subgroup": str(level),
                                          "rows": int(mask.sum()), "empirical_coverage": float(np.mean(truth_raw[mask] <= upper[mask])),
                                          "mean_interval_width": float(np.mean(width[mask]))})
        risk_predictions[f"{task}_truth"] = truth_raw
        risk_predictions[f"{task}_median"] = median_raw
        risk_predictions[f"{task}_base_quantile"] = base_quantile_raw
        risk_predictions[f"{task}_conformal_upper"] = conformal_upper
        risk_predictions[f"{task}_quantile_corrected"] = quantile_corrected
    risk_frame = pd.DataFrame(risk_rows)
    write_csv_fresh(run / "tables/risk_calibration_summary.csv", risk_frame)
    write_csv_fresh(run / "tables/risk_calibration_subgroup_coverage.csv", pd.DataFrame(subgroup_rows))
    write_csv_fresh(run / "calibration/valid_risk_calibration_predictions.csv", risk_predictions)

    binary_rows = []
    for task, truth_column in (("confusion", "confusion_risk"), ("catastrophic", "catastrophic_valid_failure")):
        truth = samples.loc[valid, truth_column].to_numpy(dtype=int)
        task_logits = binary_logits(run, features, task)[valid.to_numpy()]
        temp = fit_binary_temperature(task_logits, truth)
        probability = sigmoid(task_logits / temp)
        binary_rows.append({"task": task, "calibration_rows": len(truth), "temperature": temp, "prevalence": float(truth.mean()),
                            "auroc": auroc(probability, truth), "auprc": auprc(probability, truth),
                            "brier": float(np.mean((probability - truth) ** 2)), **score_structure(probability)})
    write_csv_fresh(run / "tables/binary_risk_calibration_summary.csv", pd.DataFrame(binary_rows))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task in zip(axes, ("image", "flux", "centroid")):
        truth = risk_predictions[f"{task}_truth"].to_numpy()
        pred = risk_predictions[f"{task}_median"].to_numpy()
        keep = np.argsort(truth)[::max(1, len(truth) // 1000)]
        ax.scatter(truth[keep], pred[keep], s=5, alpha=0.35)
        ax.set_xscale("symlog", linthresh=1.0); ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(task); ax.set_xlabel("natural calibration risk"); ax.set_ylabel("frozen median prediction")
    fig.tight_layout(); fig.savefig(run / "figures/natural_calibration_transfer.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ("image", "flux", "centroid")):
        rows = risk_frame[risk_frame.risk == task]
        ax.bar(rows.method, rows.empirical_coverage); ax.axhline(0.9, color="black", linestyle="--")
        ax.set_ylim(0, 1); ax.tick_params(axis="x", labelrotation=25); ax.set_title(task)
    fig.tight_layout(); fig.savefig(run / "figures/calibration_transfer_coverage.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    catastrophic = pd.read_csv(run / "tables/catastrophic_head_seed_stability.csv")
    ax.scatter(catastrophic.auroc, catastrophic.auprc, c=catastrophic.seed, cmap="viridis", s=50)
    ax.axvline(0.654, color="black", linestyle="--", label="prior AUROC 0.654")
    ax.set_xlabel("validation AUROC"); ax.set_ylabel("validation AUPRC"); ax.legend(); fig.tight_layout()
    fig.savefig(run / "figures/catastrophic_ranking_seed_stability.png", dpi=180); plt.close(fig)

    write_json_fresh(run / "logs/calibration_feasibility_complete.json", {
        "status": "PASS", "runtime_seconds": time.time() - started, "heads_frozen_before_calibration": True,
        "query_probability_method": selected_method, "operational_policy_selected": False,
        "calibration_used_for_model_selection": False, "development_accessed": False, "lockbox_accessed": False,
    })
    print(json.dumps({"query_method": selected_method, "query_ece": float(query_frame.loc[query_frame.method == selected_method, "ece"].iloc[0]),
                      "risk_coverage": risk_frame.groupby("risk").empirical_coverage.mean().to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
