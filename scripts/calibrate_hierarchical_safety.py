#!/usr/bin/env python3
"""Natural-calibration-only hierarchical policy calibration and freeze."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.hierarchical_safety import RISK_LIMITS, conformal_upper_offset
from src.models_thayer_select import ThayerSelectNet


PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
CONDITION_C = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
R1_CHECKPOINT = PHASE2 / "checkpoints/r1_best.pth"
NORMALIZATION = PHASE1 / "manifests/normalization.json"
CLASSES = ("UNIQUE_VALID", "NULL", "AMBIGUOUS")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
SEEDS_QUERY = (2026071201, 2026071202, 2026071203, 2026071204, 2026071205)
SEEDS_RISK = (2026071211, 2026071212, 2026071213, 2026071214, 2026071215)
ALPHA = 0.10
CONFUSION_LIMIT = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str: return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL; descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle: handle.write(value)


def write_json_fresh(path: Path, value: object) -> None: write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)


def sigmoid(value: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -40, 40)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int); positive = int(truth.sum()); negative = len(truth) - positive
    if not positive or not negative: return math.nan
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[truth == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int); positives = int(truth.sum())
    if not positives: return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable"); ordered = truth[order]; precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


class QueryNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__(); self.network = nn.Linear(dimensions, 3) if family == "multinomial_logistic" else nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, values): return self.network(values)


class RiskNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__(); self.network = nn.Linear(dimensions, 2) if family == "linear" else nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, values): return self.network(values)


class BinaryNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__(); self.network = nn.Linear(dimensions, 1) if family == "linear" else nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, values): return self.network(values).flatten()


def transform(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray: return ((values - mean) / scale).astype(np.float32)


def load_dataset(run: Path, dataset: str):
    npz = np.load(run / f"features/v2_{dataset}_features.npz", allow_pickle=True); samples = pd.read_csv(run / f"features/v3_{dataset}_samples.csv", keep_default_na=False, low_memory=False); manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
    if npz["scene_id"].astype(str).tolist() != samples.scene_id.tolist() or samples.scene_id.tolist() != manifest.scene_id.tolist(): raise RuntimeError(f"Alignment failure: {dataset}")
    return npz, samples, manifest


def query_logits(run: Path, npz) -> np.ndarray:
    selection = json.loads((run / "manifests/query_gate_selection.json").read_text()); feature = selection["selected_feature_family"]; family = selection["selected_head_family"]; blocks = []
    for seed in SEEDS_QUERY:
        payload = torch.load(run / f"models/query_gate_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False); model = QueryNet(npz[feature].shape[1], family); model.load_state_dict(payload["state_dict"]); model.eval()
        with torch.no_grad(): blocks.append(model(torch.from_numpy(transform(npz[feature], payload["mean"], payload["scale"]))).numpy())
    return np.mean(blocks, axis=0)


def risk_outputs(run: Path, npz, task: str) -> np.ndarray:
    selection = json.loads((run / "manifests/risk_head_selection.json").read_text())["risk_heads"][task]; feature = selection["feature_family"]; family = selection["head_family"]; blocks = []
    for seed in SEEDS_RISK:
        payload = torch.load(run / f"models/{task}_risk_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False); model = RiskNet(npz[feature].shape[1], family); model.load_state_dict(payload["state_dict"]); model.eval()
        with torch.no_grad(): blocks.append(model(torch.from_numpy(transform(npz[feature], payload["mean"], payload["scale"]))).numpy())
    # Average in the trained log1p target space, not after exponential inversion.
    return np.mean(blocks, axis=0)


def confusion_logits(run: Path, npz) -> np.ndarray:
    selection = json.loads((run / "manifests/risk_head_selection.json").read_text())["confusion_head"]; feature = selection["feature_family"]; family = selection["head_family"]; blocks = []
    for seed in SEEDS_RISK:
        payload = torch.load(run / f"models/confusion_{feature}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False); model = BinaryNet(npz[feature].shape[1], family); model.load_state_dict(payload["state_dict"]); model.eval()
        with torch.no_grad(): blocks.append(model(torch.from_numpy(transform(npz[feature], payload["mean"], payload["scale"]))).numpy())
    return np.mean(blocks, axis=0)


def fit_vector_scaling(logits: np.ndarray, truth: np.ndarray) -> dict:
    x = torch.from_numpy(np.asarray(logits, dtype=np.float32)); y = torch.from_numpy(np.asarray(truth, dtype=np.int64)); log_scale = torch.zeros(3, requires_grad=True); bias = torch.zeros(3, requires_grad=True); optimizer = torch.optim.LBFGS([log_scale, bias], lr=0.5, max_iter=150, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(); centered_bias = bias - bias.mean(); output = x / torch.exp(log_scale).clamp(0.05, 20.0) + centered_bias; loss = F.cross_entropy(output, y); loss.backward(); return loss
    optimizer.step(closure)
    return {"temperature_by_class": torch.exp(log_scale).detach().numpy().astype(float).tolist(), "centered_bias_by_class": (bias - bias.mean()).detach().numpy().astype(float).tolist()}


def apply_vector(logits: np.ndarray, parameters: dict) -> np.ndarray:
    adjusted = np.asarray(logits) / np.asarray(parameters["temperature_by_class"])[None] + np.asarray(parameters["centered_bias_by_class"])[None]
    adjusted -= adjusted.max(axis=1, keepdims=True); exp = np.exp(adjusted); return exp / exp.sum(axis=1, keepdims=True)


def multiclass_ece(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(axis=1); correct = np.argmax(probability, axis=1) == truth; total = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if mask.any(): total += mask.mean() * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(total)


def fit_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(logits, dtype=float); labels = np.asarray(truth, dtype=float)
    def objective(log_t):
        scores = sigmoid(values / math.exp(log_t)); return float(-np.mean(labels * np.log(scores + 1e-12) + (1 - labels) * np.log(1 - scores + 1e-12)))
    result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    if not result.success: raise RuntimeError("Temperature fit failed")
    return float(math.exp(result.x))


def select_query_thresholds(probability: np.ndarray, truth: np.ndarray) -> dict:
    quantiles = np.linspace(0, 1, 21)
    unique_values = np.unique(np.quantile(probability[:, 0], quantiles)); null_values = np.unique(np.quantile(probability[:, 1], quantiles)); ambiguous_values = np.unique(np.quantile(probability[:, 2], quantiles))
    best = None
    for t_unique in unique_values:
        for t_null in null_values:
            base = (probability[:, 0] >= t_unique) & (probability[:, 1] <= t_null)
            for t_ambiguous in ambiguous_values:
                accepted = base & (probability[:, 2] <= t_ambiguous)
                valid_coverage = float(accepted[truth == 0].mean()); null_far = float(accepted[truth == 1].mean()); ambiguous_far = float(accepted[truth == 2].mean())
                if null_far <= 0.05 and ambiguous_far <= 0.10:
                    key = (valid_coverage, -(null_far + ambiguous_far), t_unique, -t_null, -t_ambiguous)
                    if best is None or key > best[0]: best = (key, {"unique_minimum": float(t_unique), "null_maximum": float(t_null), "ambiguous_maximum": float(t_ambiguous), "valid_coverage": valid_coverage, "null_false_accept_rate": null_far, "ambiguous_false_accept_rate": ambiguous_far})
    if best is None: raise RuntimeError("No query threshold candidate satisfies invalid-query constraints")
    best[1]["nondegenerate"] = bool(best[1]["valid_coverage"] > 0); return best[1]


def fresh_r1_baseline(run: Path, npz, samples: pd.DataFrame) -> dict:
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS required for fresh R1 baseline inference")
    payload = torch.load(R1_CHECKPOINT, map_location="cpu", weights_only=False); model = ThayerSelectNet(min_log_variance=-8.0, max_log_variance=2.0); model.load_state_dict(payload["state_dict"], strict=True); model.eval();
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.to("mps"); scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32); scores = []
    with h5py.File(run / "manifests/v2_r_validation_scenes.h5", "r") as handle, torch.no_grad():
        for start in range(0, len(samples), 128):
            blend = np.asarray(handle["blend"][start:start + 128], dtype=np.float32); prompt = np.asarray(handle["prompt"][start:start + 128], dtype=np.float32)
            output = model(torch.from_numpy(np.ascontiguousarray(blend / scales[None, :, None, None])).to("mps"), torch.from_numpy(np.ascontiguousarray(prompt)).to("mps")); scores.append(output["recoverability"].flatten().cpu().numpy())
    success_score = np.concatenate(scores); catastrophic = ((samples.violation_moderate >= 2) | (samples.confusion_risk == 1)).to_numpy(dtype=int); risk_score = 1.0 - success_score
    return {"baseline": "original_monolithic_R1_fresh_r_validation", "valid_catastrophic_auroc": auroc(risk_score, catastrophic), "valid_catastrophic_auprc": auprc(risk_score, catastrophic), "prevalence": float(catastrophic.mean()), "deployable": True, "checkpoint_sha256": sha256_file(R1_CHECKPOINT), "historical_development_inference_regenerated": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run = args.run_dir.resolve()
    if json.loads((run / "logs/risk_head_training_complete.json").read_text())["status"] != "PASS": raise RuntimeError("Risk-head gate missing")
    if (run / "manifests/hierarchical_policy_freeze.json").exists(): raise FileExistsError("Policy already frozen")
    started = time.time(); natural_npz, natural_samples, natural_manifest = load_dataset(run, "natural_calibration"); strat_npz, strat_samples, strat_manifest = load_dataset(run, "stratified_calibration"); valid_npz, valid_samples, valid_manifest = load_dataset(run, "r_validation")
    natural_truth = np.asarray([CLASS_TO_INDEX[value] for value in natural_samples.query_state], dtype=int); strat_truth = np.asarray([CLASS_TO_INDEX[value] for value in strat_samples.query_state], dtype=int)
    natural_query_logits = query_logits(run, natural_npz); strat_query_logits = query_logits(run, strat_npz); vector = fit_vector_scaling(natural_query_logits, natural_truth); natural_probability = apply_vector(natural_query_logits, vector); strat_probability = apply_vector(strat_query_logits, vector)
    query_thresholds = select_query_thresholds(natural_probability, natural_truth)
    raw_probability = torch.softmax(torch.from_numpy(natural_query_logits), dim=1).numpy(); query_calibration = {"method": "vector_scaling", "parameters": vector, "raw_ece": multiclass_ece(raw_probability, natural_truth), "calibrated_ece": multiclass_ece(natural_probability, natural_truth), "raw_log_loss": float(-np.mean(np.log(raw_probability[np.arange(len(natural_truth)), natural_truth] + 1e-12))), "calibrated_log_loss": float(-np.mean(np.log(natural_probability[np.arange(len(natural_truth)), natural_truth] + 1e-12))), "score_unique_value_count": [int(len(np.unique(natural_probability[:, index]))) for index in range(3)], "natural_calibration_only": True}
    write_json_fresh(run / "calibration/query_vector_scaling.json", query_calibration)

    result_natural = pd.DataFrame({"scene_id": natural_samples.scene_id, "query_state": natural_samples.query_state, "p_unique": natural_probability[:, 0], "p_null": natural_probability[:, 1], "p_ambiguous": natural_probability[:, 2]})
    result_strat = pd.DataFrame({"scene_id": strat_samples.scene_id, "query_state": strat_samples.query_state, "p_unique": strat_probability[:, 0], "p_null": strat_probability[:, 1], "p_ambiguous": strat_probability[:, 2]})
    conformal_rows = []; risk_calibrators = {}
    for task, truth_column in (("image", "image_risk"), ("flux", "flux_risk_max"), ("centroid", "centroid_risk_pixels")):
        natural_output = risk_outputs(run, natural_npz, task); strat_output = risk_outputs(run, strat_npz, task); valid = natural_samples.applicable_valid_risk.to_numpy(dtype=int).astype(bool); strat_valid = strat_samples.applicable_valid_risk.to_numpy(dtype=int).astype(bool)
        truth_log = np.log1p(natural_samples.loc[valid, truth_column].to_numpy(dtype=float)); residual = truth_log - natural_output[valid, 1]; offset = conformal_upper_offset(residual, ALPHA)
        natural_upper = np.maximum(np.expm1(natural_output[:, 1] + offset), 0.0); natural_median = np.maximum(np.expm1(natural_output[:, 0]), 0.0); strat_upper = np.maximum(np.expm1(strat_output[:, 1] + offset), 0.0); strat_median = np.maximum(np.expm1(strat_output[:, 0]), 0.0)
        result_natural[f"{task}_median"] = natural_median; result_natural[f"{task}_upper"] = natural_upper; result_strat[f"{task}_median"] = strat_median; result_strat[f"{task}_upper"] = strat_upper
        coverage = float(np.mean(natural_samples.loc[valid, truth_column].to_numpy(dtype=float) <= natural_upper[valid])); diagnostic_coverage = float(np.mean(strat_samples.loc[strat_valid, truth_column].to_numpy(dtype=float) <= strat_upper[strat_valid]))
        risk_calibrators[task] = {"method": "split_conformal_upper_residual", "space": "log1p empirical risk", "miscoverage": ALPHA, "offset": offset, "natural_valid_rows": int(valid.sum()), "empirical_natural_coverage": coverage, "stratified_diagnostic_coverage": diagnostic_coverage, "mean_natural_upper_width_from_median": float(np.mean(np.maximum(natural_upper[valid] - natural_median[valid], 0)))}
        conformal_rows.append({"risk": task, **risk_calibrators[task]})
    write_json_fresh(run / "calibration/risk_split_conformal.json", risk_calibrators); write_csv_fresh(run / "tables/conformal_calibration_summary.csv", pd.DataFrame(conformal_rows))

    natural_conf_logits = confusion_logits(run, natural_npz); strat_conf_logits = confusion_logits(run, strat_npz); valid = natural_samples.applicable_valid_risk.to_numpy(dtype=int).astype(bool); temperature = fit_temperature(natural_conf_logits[valid], natural_samples.loc[valid, "confusion_risk"].to_numpy(dtype=int)); natural_conf = sigmoid(natural_conf_logits / temperature); strat_conf = sigmoid(strat_conf_logits / temperature); result_natural["p_confusion"] = natural_conf; result_strat["p_confusion"] = strat_conf
    write_json_fresh(run / "calibration/confusion_temperature.json", {"method": "temperature", "temperature": temperature, "limit": CONFUSION_LIMIT, "natural_valid_rows": int(valid.sum()), "natural_auroc": auroc(natural_conf[valid], natural_samples.loc[valid, "confusion_risk"]), "natural_auprc": auprc(natural_conf[valid], natural_samples.loc[valid, "confusion_risk"]), "natural_prevalence": float(natural_samples.loc[valid, "confusion_risk"].mean()), "natural_calibration_only": True})

    query_pass = (result_natural.p_unique >= query_thresholds["unique_minimum"]) & (result_natural.p_null <= query_thresholds["null_maximum"]) & (result_natural.p_ambiguous <= query_thresholds["ambiguous_maximum"])
    limits = RISK_LIMITS["moderate"]
    full_accept = query_pass & (result_natural.image_upper < limits.image) & (result_natural.flux_upper < limits.flux) & (result_natural.centroid_upper < limits.centroid_pixels) & (result_natural.p_confusion < CONFUSION_LIMIT)
    result_natural["query_gate_accept"] = query_pass.astype(int); result_natural["full_policy_accept"] = full_accept.astype(int)
    result_strat["query_gate_accept"] = ((result_strat.p_unique >= query_thresholds["unique_minimum"]) & (result_strat.p_null <= query_thresholds["null_maximum"]) & (result_strat.p_ambiguous <= query_thresholds["ambiguous_maximum"])).astype(int)
    result_strat["full_policy_accept"] = (result_strat.query_gate_accept.astype(bool) & (result_strat.image_upper < limits.image) & (result_strat.flux_upper < limits.flux) & (result_strat.centroid_upper < limits.centroid_pixels) & (result_strat.p_confusion < CONFUSION_LIMIT)).astype(int)
    write_csv_fresh(run / "calibration/natural_calibration_predictions.csv", result_natural); write_csv_fresh(run / "calibration/stratified_diagnostic_predictions.csv", result_strat)
    calibration_behavior = []
    for population, frame in (("natural", result_natural), ("stratified_diagnostic", result_strat)):
        for state, group in frame.groupby("query_state"):
            calibration_behavior.append({"population": population, "query_state": state, "samples": len(group), "query_gate_acceptance": float(group.query_gate_accept.mean()), "full_policy_acceptance": float(group.full_policy_accept.mean()), "mean_p_unique": float(group.p_unique.mean()), "mean_p_null": float(group.p_null.mean()), "mean_p_ambiguous": float(group.p_ambiguous.mean()), "mean_p_confusion": float(group.p_confusion.mean())})
    write_csv_fresh(run / "tables/calibration_behavior_by_query_state.csv", pd.DataFrame(calibration_behavior))

    baseline = fresh_r1_baseline(run, valid_npz, valid_samples); write_csv_fresh(run / "tables/valid_risk_baselines_superseding_fresh_r1.csv", pd.DataFrame([baseline]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ("image", "flux", "centroid")):
        valid = natural_samples.applicable_valid_risk.to_numpy(dtype=int).astype(bool); residual = np.log1p(natural_samples.loc[valid, {"image":"image_risk","flux":"flux_risk_max","centroid":"centroid_risk_pixels"}[task]].to_numpy(dtype=float)) - risk_outputs(run, natural_npz, task)[valid, 1]; ax.hist(residual, bins=50, color="#4472c4", alpha=0.8); ax.axvline(risk_calibrators[task]["offset"], color="black", linestyle="--"); ax.set_title(task); ax.set_xlabel("log1p truth - q prediction")
    fig.tight_layout(); fig.savefig(run / "figures/conformal_residual_distributions.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); summary = pd.DataFrame(conformal_rows); ax.bar(summary.risk, summary.empirical_natural_coverage, label="natural"); ax.scatter(summary.risk, summary.stratified_diagnostic_coverage, color="orange", label="stratified diagnostic"); ax.axhline(0.9, color="black", linestyle="--"); ax.set_ylim(0, 1); ax.set_ylabel("upper-bound coverage"); ax.legend(); fig.tight_layout(); fig.savefig(run / "figures/conformal_quantile_coverage.png", dpi=180); plt.close(fig)

    selected_models = []
    risk_selection = json.loads((run / "manifests/risk_head_selection.json").read_text()); query_selection = json.loads((run / "manifests/query_gate_selection.json").read_text())
    for path in sorted((run / "models").glob("*.pth")):
        if "candidate" not in path.name: selected_models.append({"relative_path": relative(path), "sha256": sha256_file(path)})
    freeze = {
        "status": "FROZEN_BEFORE_DEVELOPMENT_GENERATION", "condition_c_checkpoint_sha256": sha256_file(CONDITION_C), "condition_c_checkpoint_path": relative(CONDITION_C),
        "query_head": query_selection, "risk_heads": risk_selection, "query_calibration": query_calibration, "query_thresholds": query_thresholds,
        "risk_calibration": risk_calibrators, "confusion_temperature": temperature, "confusion_limit": CONFUSION_LIMIT,
        "primary_limits": {"name": "moderate", "image": limits.image, "flux": limits.flux, "centroid_pixels": limits.centroid_pixels},
        "natural_calibration_full_policy_coverage": float(full_accept.mean()), "natural_calibration_unique_valid_full_policy_coverage": float(full_accept[natural_truth == 0].mean()),
        "natural_calibration_null_false_acceptance": float(full_accept[natural_truth == 1].mean()), "natural_calibration_ambiguous_false_acceptance": float(full_accept[natural_truth == 2].mean()),
        "nondegenerate": bool(full_accept[natural_truth == 0].sum() > 0), "selected_model_hashes": selected_models,
        "code_hashes": [{"relative_path": relative(path), "sha256": sha256_file(path)} for path in (REPO / "src/hierarchical_safety.py", REPO / "scripts/calibrate_hierarchical_safety.py", REPO / "scripts/train_hierarchical_query_gate.py", REPO / "scripts/train_hierarchical_risk_heads.py")],
        "calibration_manifest_sha256": sha256_file(run / "manifests/v2_natural_calibration_scene_manifest.csv"), "stratified_calibration_used_for_thresholds": False,
        "development_generated": False, "development_used": False, "lockbox_used": False, "frozen_at_unix": time.time(),
    }
    write_json_fresh(run / "manifests/hierarchical_policy_freeze.json", freeze)
    write_json_fresh(run / "logs/calibration_and_policy_freeze_complete.json", {"status": "PASS", "runtime_seconds": time.time() - started, "natural_calibration_only_for_operational_parameters": True, "policy_nondegenerate": freeze["nondegenerate"], "natural_unique_valid_coverage": freeze["natural_calibration_unique_valid_full_policy_coverage"], "development_used": False, "lockbox_used": False})
    print(json.dumps({"query_thresholds": query_thresholds, "full_policy_unique_coverage": freeze["natural_calibration_unique_valid_full_policy_coverage"], "nondegenerate": freeze["nondegenerate"], "r1_baseline": baseline}, sort_keys=True))


if __name__ == "__main__": main()
