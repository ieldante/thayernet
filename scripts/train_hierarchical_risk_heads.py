#!/usr/bin/env python3
"""CPU-only valid-query metric-risk and confusion heads."""

from __future__ import annotations

import argparse
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
from scipy.stats import rankdata, spearmanr
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
FEATURE_KEYS = ("f_global", "f_prompt_local", "f_recon_summary", "f_combined")
FAMILIES = ("linear", "small_mlp")
TARGETS = {
    "image": ("image_target_log1p", "image_risk"),
    "flux": ("flux_target_log1p", "flux_risk_max"),
    "centroid": ("centroid_target_log1p", "centroid_risk_pixels"),
}
SEEDS = (2026071211, 2026071212, 2026071213, 2026071214, 2026071215)
BASE_SEED = SEEDS[0]
QUANTILE = 0.90


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists(): raise FileExistsError(path)
    frame.to_csv(path, index=False)


def sigmoid(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(scores, dtype=float); truth = np.asarray(labels, dtype=int)
    positive = int(truth.sum()); negative = len(truth) - positive
    if positive == 0 or negative == 0: return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[truth == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(scores, dtype=float); truth = np.asarray(labels, dtype=int)
    positives = int(truth.sum())
    if positives == 0: return math.nan
    order = np.argsort(-score, kind="stable"); ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def pinball_numpy(prediction: np.ndarray, target: np.ndarray, quantile: float = QUANTILE) -> float:
    residual = np.asarray(target) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def top_risk_recall(score: np.ndarray, truth: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(math.ceil(len(truth) * fraction)))
    predicted_top = set(np.argsort(-score, kind="stable")[:count]); true_top = set(np.argsort(-truth, kind="stable")[:count])
    return len(predicted_top & true_top) / len(true_top)


class RiskNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__()
        self.network = nn.Linear(dimensions, 2) if family == "linear" else nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.network(values)


class BinaryNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__(); self.network = nn.Linear(dimensions, 1) if family == "linear" else nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.network(values).flatten()


def standardizer(values: np.ndarray):
    mean = np.mean(values, axis=0, dtype=np.float64); scale = np.std(values, axis=0, dtype=np.float64); scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transformed(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray: return ((values - mean) / scale).astype(np.float32)


def fit_risk(family: str, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray):
    torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1)); mean, scale = standardizer(x_train)
    train_x = torch.from_numpy(transformed(x_train, mean, scale)); train_y = torch.from_numpy(y_train.astype(np.float32))
    valid_x = torch.from_numpy(transformed(x_valid, mean, scale)); model = RiskNet(train_x.shape[1], family).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); generator = torch.Generator().manual_seed(seed)
    best_state = None; best_loss = math.inf; best_epoch = -1; stale = 0
    for epoch in range(80):
        model.train(); indices = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(indices), 512):
            batch = indices[start:start + 512]; optimizer.zero_grad(set_to_none=True); output = model(train_x[batch])
            residual = train_y[batch] - output[:, 1]
            loss = F.huber_loss(output[:, 0], train_y[batch]) + torch.maximum(QUANTILE * residual, (QUANTILE - 1.0) * residual).mean()
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): output = model(valid_x).numpy()
        valid_loss = pinball_numpy(output[:, 1], y_valid)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss; best_epoch = epoch + 1; stale = 0; best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        if epoch >= 20 and stale >= 12: break
    if best_state is None: raise RuntimeError("No risk checkpoint")
    model.load_state_dict(best_state); return model, mean, scale, best_epoch


def predict_risk(model: RiskNet, values: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    model.eval()
    with torch.no_grad(): output = model(torch.from_numpy(transformed(values, mean, scale))).numpy()
    raw = np.maximum(np.expm1(output), 0.0)
    return output, raw


def fit_binary(family: str, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray):
    torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1)); mean, scale = standardizer(x_train)
    train_x = torch.from_numpy(transformed(x_train, mean, scale)); train_y = torch.from_numpy(y_train.astype(np.float32)); valid_x = torch.from_numpy(transformed(x_valid, mean, scale))
    model = BinaryNet(train_x.shape[1], family).cpu(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = max(int(y_train.sum()), 1); negative = len(y_train) - positive; pos_weight = torch.tensor([negative / positive], dtype=torch.float32); generator = torch.Generator().manual_seed(seed)
    best_state = None; best_ap = -math.inf; best_epoch = -1; stale = 0
    for epoch in range(80):
        model.train(); indices = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(indices), 512):
            batch = indices[start:start + 512]; optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(train_x[batch]), train_y[batch], pos_weight=pos_weight); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): scores = sigmoid(model(valid_x).numpy())
        ap = auprc(scores, y_valid)
        if ap > best_ap + 1e-5:
            best_ap = ap; best_epoch = epoch + 1; stale = 0; best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        if epoch >= 20 and stale >= 12: break
    if best_state is None: raise RuntimeError("No confusion checkpoint")
    model.load_state_dict(best_state); return model, mean, scale, best_epoch


def predict_binary(model: BinaryNet, values: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    model.eval()
    with torch.no_grad(): logits = model(torch.from_numpy(transformed(values, mean, scale))).numpy()
    return logits, sigmoid(logits)


def save_model(path: Path, model: nn.Module, mean: np.ndarray, scale: np.ndarray, metadata: dict) -> None:
    if path.exists(): raise FileExistsError(path)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale, "device": "cpu", **metadata}, path)


def load(run: Path, dataset: str):
    npz = np.load(run / f"features/v2_{dataset}_features.npz", allow_pickle=True)
    samples = pd.read_csv(run / f"features/v3_{dataset}_samples.csv", keep_default_na=False)
    manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
    if npz["scene_id"].astype(str).tolist() != samples.scene_id.tolist() or samples.scene_id.tolist() != manifest.scene_id.tolist(): raise RuntimeError(f"Alignment failed: {dataset}")
    return npz, samples, manifest


def risk_metrics(prediction_transformed: np.ndarray, prediction_raw: np.ndarray, target_transformed: np.ndarray, target_raw: np.ndarray, catastrophic: np.ndarray) -> dict:
    median = prediction_raw[:, 0]; upper = prediction_raw[:, 1]
    return {
        "median_mae": float(np.mean(np.abs(median - target_raw))),
        "median_spearman": float(spearmanr(median, target_raw).statistic),
        "upper_pinball_log1p": pinball_numpy(prediction_transformed[:, 1], target_transformed),
        "upper_empirical_coverage": float(np.mean(target_raw <= upper)),
        "upper_mean_width_from_median": float(np.mean(np.maximum(upper - median, 0))),
        "upper_spearman": float(spearmanr(upper, target_raw).statistic),
        "top_10_percent_recall": top_risk_recall(upper, target_raw),
        "catastrophic_failure_auroc": auroc(upper, catastrophic),
        "catastrophic_failure_auprc": auprc(upper, catastrophic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); run = args.run_dir.resolve()
    gate = json.loads((run / "manifests/query_gate_selection.json").read_text())
    if gate["status"] != "PASS": raise RuntimeError("Query-state gate failed; risk heads prohibited")
    if (run / "logs/risk_head_training_complete.json").exists(): raise FileExistsError("Risk heads already trained")
    started = time.time(); train_npz, train_samples, train_manifest = load(run, "r_training"); valid_npz, valid_samples, valid_manifest = load(run, "r_validation")
    train_mask = (train_samples.applicable_valid_risk == 1).to_numpy(); valid_mask = (valid_samples.applicable_valid_risk == 1).to_numpy()
    catastrophic_train = ((train_samples.violation_moderate >= 2) | (train_samples.confusion_risk == 1)).to_numpy(dtype=int)
    catastrophic_valid = ((valid_samples.violation_moderate >= 2) | (valid_samples.confusion_risk == 1)).to_numpy(dtype=int)
    candidate_rows = []; selected = {}; validation_predictions = {}
    for target_name, (transformed_column, raw_column) in TARGETS.items():
        y_train_t = train_samples.loc[train_mask, transformed_column].to_numpy(dtype=float); y_valid_t = valid_samples.loc[valid_mask, transformed_column].to_numpy(dtype=float)
        y_train_raw = train_samples.loc[train_mask, raw_column].to_numpy(dtype=float); y_valid_raw = valid_samples.loc[valid_mask, raw_column].to_numpy(dtype=float)
        if not np.isfinite(np.r_[y_train_t, y_valid_t, y_train_raw, y_valid_raw]).all(): raise RuntimeError(f"Nonfinite applicable target: {target_name}")
        target_candidates = []
        for feature in FEATURE_KEYS:
            for family in FAMILIES:
                model, mean, scale, best_epoch = fit_risk(family, BASE_SEED, train_npz[feature][train_mask], y_train_t, valid_npz[feature][valid_mask], y_valid_t)
                prediction_t, prediction_raw = predict_risk(model, valid_npz[feature][valid_mask], mean, scale)
                metrics = risk_metrics(prediction_t, prediction_raw, y_valid_t, y_valid_raw, catastrophic_valid[valid_mask])
                row = {"target": target_name, "feature_family": feature, "head_family": family, "seed": BASE_SEED, "best_epoch": best_epoch, **metrics, "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
                candidate_rows.append(row); target_candidates.append(row)
                save_model(run / f"models/{target_name}_risk_candidate_{feature}_{family}.pth", model, mean, scale, {"task": target_name, "feature_family": feature, "family": family, "seed": BASE_SEED, "quantile": QUANTILE, "best_epoch": best_epoch})
        ranked = sorted(target_candidates, key=lambda row: (row["upper_pinball_log1p"], -row["upper_spearman"], row["median_mae"]))
        selected[target_name] = {"feature_family": ranked[0]["feature_family"], "head_family": ranked[0]["head_family"]}

    write_csv_fresh(run / "tables/risk_head_candidate_comparison.csv", pd.DataFrame(candidate_rows))
    seed_rows = []
    for target_name, choice in selected.items():
        transformed_column, raw_column = TARGETS[target_name]; y_train_t = train_samples.loc[train_mask, transformed_column].to_numpy(dtype=float); y_valid_t = valid_samples.loc[valid_mask, transformed_column].to_numpy(dtype=float); y_valid_raw = valid_samples.loc[valid_mask, raw_column].to_numpy(dtype=float)
        feature = choice["feature_family"]; family = choice["head_family"]; transformed_blocks = []; raw_blocks = []
        for seed in SEEDS:
            model, mean, scale, best_epoch = fit_risk(family, seed, train_npz[feature][train_mask], y_train_t, valid_npz[feature][valid_mask], y_valid_t)
            prediction_t, prediction_raw = predict_risk(model, valid_npz[feature][valid_mask], mean, scale); transformed_blocks.append(prediction_t); raw_blocks.append(prediction_raw)
            metrics = risk_metrics(prediction_t, prediction_raw, y_valid_t, y_valid_raw, catastrophic_valid[valid_mask]); seed_rows.append({"task": target_name, "seed": seed, "feature_family": feature, "head_family": family, "best_epoch": best_epoch, **metrics})
            save_model(run / f"models/{target_name}_risk_{feature}_{family}_seed_{seed}.pth", model, mean, scale, {"task": target_name, "feature_family": feature, "family": family, "seed": seed, "quantile": QUANTILE, "best_epoch": best_epoch})
        ensemble_t = np.mean(transformed_blocks, axis=0); ensemble_raw = np.mean(raw_blocks, axis=0)
        validation_predictions[target_name] = {"transformed": ensemble_t, "raw": ensemble_raw, "truth_transformed": y_valid_t, "truth_raw": y_valid_raw}
    write_csv_fresh(run / "tables/risk_head_seed_stability.csv", pd.DataFrame(seed_rows))

    confusion_candidates = []
    y_conf_train = train_samples.confusion_risk.to_numpy(dtype=int); y_conf_valid = valid_samples.confusion_risk.to_numpy(dtype=int)
    for feature in FEATURE_KEYS:
        for family in FAMILIES:
            model, mean, scale, best_epoch = fit_binary(family, BASE_SEED, train_npz[feature], y_conf_train, valid_npz[feature], y_conf_valid)
            logits, scores = predict_binary(model, valid_npz[feature], mean, scale)
            confusion_candidates.append({"feature_family": feature, "head_family": family, "seed": BASE_SEED, "best_epoch": best_epoch, "auroc": auroc(scores, y_conf_valid), "auprc": auprc(scores, y_conf_valid), "prevalence": float(y_conf_valid.mean()), "parameter_count": sum(parameter.numel() for parameter in model.parameters())})
            save_model(run / f"models/confusion_candidate_{feature}_{family}.pth", model, mean, scale, {"task": "confusion", "feature_family": feature, "family": family, "seed": BASE_SEED, "best_epoch": best_epoch})
    confusion_frame = pd.DataFrame(confusion_candidates); write_csv_fresh(run / "tables/confusion_head_candidate_comparison.csv", confusion_frame)
    best_confusion = confusion_frame.sort_values(["auprc", "auroc"], ascending=False, kind="stable").iloc[0]
    conf_feature = str(best_confusion.feature_family); conf_family = str(best_confusion.head_family); conf_logits = []; conf_scores = []; conf_seed_rows = []
    for seed in SEEDS:
        model, mean, scale, best_epoch = fit_binary(conf_family, seed, train_npz[conf_feature], y_conf_train, valid_npz[conf_feature], y_conf_valid)
        logits, scores = predict_binary(model, valid_npz[conf_feature], mean, scale); conf_logits.append(logits); conf_scores.append(scores)
        conf_seed_rows.append({"task": "confusion", "seed": seed, "feature_family": conf_feature, "head_family": conf_family, "best_epoch": best_epoch, "auroc": auroc(scores, y_conf_valid), "auprc": auprc(scores, y_conf_valid), "prevalence": float(y_conf_valid.mean())})
        save_model(run / f"models/confusion_{conf_feature}_{conf_family}_seed_{seed}.pth", model, mean, scale, {"task": "confusion", "feature_family": conf_feature, "family": conf_family, "seed": seed, "best_epoch": best_epoch})
    write_csv_fresh(run / "tables/confusion_head_seed_stability.csv", pd.DataFrame(conf_seed_rows))

    prediction_path = run / "features/risk_head_validation_predictions.npz"
    if prediction_path.exists(): raise FileExistsError(prediction_path)
    np.savez_compressed(prediction_path, scene_id=valid_samples.scene_id.astype(str).to_numpy(), catastrophic=catastrophic_valid, confusion_truth=y_conf_valid,
        image_median=validation_predictions["image"]["raw"][:, 0], image_upper=validation_predictions["image"]["raw"][:, 1],
        flux_median=validation_predictions["flux"]["raw"][:, 0], flux_upper=validation_predictions["flux"]["raw"][:, 1],
        centroid_median=validation_predictions["centroid"]["raw"][:, 0], centroid_upper=validation_predictions["centroid"]["raw"][:, 1],
        confusion_logit=np.mean(conf_logits, axis=0), confusion_score=np.mean(conf_scores, axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, target_name in zip(axes, TARGETS):
        truth = validation_predictions[target_name]["truth_raw"]; pred = validation_predictions[target_name]["raw"][:, 0]
        keep = np.argsort(truth)[::max(1, len(truth) // 1000)]; ax.scatter(truth[keep], pred[keep], s=5, alpha=0.35); maximum = float(np.quantile(np.r_[truth, pred], 0.98)); ax.plot([0, maximum], [0, maximum], "k--", linewidth=0.8); ax.set_title(target_name); ax.set_xlabel("empirical risk"); ax.set_ylabel("median prediction")
    fig.tight_layout(); fig.savefig(run / "figures/valid_risk_regression.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); coverages = [float(np.mean(validation_predictions[name]["truth_raw"] <= validation_predictions[name]["raw"][:, 1])) for name in TARGETS]
    ax.bar(list(TARGETS), coverages, color="#4472c4"); ax.axhline(QUANTILE, color="black", linestyle="--"); ax.set_ylim(0, 1); ax.set_ylabel("uncalibrated q=0.90 coverage"); fig.tight_layout(); fig.savefig(run / "figures/uncalibrated_quantile_coverage.png", dpi=180); plt.close(fig)

    selection = {"status": "FROZEN_FROM_VALIDATION", "quantile": QUANTILE, "risk_heads": selected, "confusion_head": {"feature_family": conf_feature, "head_family": conf_family}, "seeds": list(SEEDS), "ensemble": "unweighted probability/prediction mean", "development_used": False, "calibration_used_for_selection": False, "lockbox_used": False}
    write_json_fresh(run / "manifests/risk_head_selection.json", selection)
    baseline_rows = [{"baseline": "random_ranking", "valid_catastrophic_auprc": float(catastrophic_valid.mean()), "deployable": True}, {"baseline": "output_energy", "valid_catastrophic_auprc": auprc(valid_npz["f_recon_summary"][:, 12:15].sum(axis=1), catastrophic_valid), "deployable": True}, {"baseline": "oracle_generator_variables", "valid_catastrophic_auprc": math.nan, "deployable": False, "note": "fit and evaluated in final calibration/evaluation stage"}, {"baseline": "original_monolithic_R1", "valid_catastrophic_auprc": math.nan, "deployable": True, "note": "fresh-scene MPS inference deferred to frozen comparison stage; historical development inference is not regenerated"}]
    write_csv_fresh(run / "tables/valid_risk_baselines.csv", pd.DataFrame(baseline_rows))
    write_json_fresh(run / "logs/risk_head_training_complete.json", {"status": "PASS", "device": "cpu", "runtime_seconds": time.time() - started, "risk_head_selection": selected, "confusion_head_selection": {"feature_family": conf_feature, "head_family": conf_family}, "r_training_rows": len(train_samples), "r_validation_rows": len(valid_samples), "null_or_ambiguous_risk_loss_rows": 0, "development_used": False, "lockbox_used": False})
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
