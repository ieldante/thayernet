#!/usr/bin/env python3
"""CPU-only query-state gate selection and five-seed stability campaign."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
CLASSES = ("UNIQUE_VALID", "NULL", "AMBIGUOUS")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
FEATURE_KEYS = ("f_global", "f_prompt_local", "f_recon_summary", "f_combined")
FAMILIES = ("multinomial_logistic", "small_mlp")
BASE_SEED = 2026071201
SEEDS = (2026071201, 2026071202, 2026071203, 2026071204, 2026071205)


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


class QueryNet(nn.Module):
    def __init__(self, dimensions: int, family: str) -> None:
        super().__init__()
        if family == "multinomial_logistic":
            self.network = nn.Linear(dimensions, len(CLASSES))
        elif family == "small_mlp":
            self.network = nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))
        else:
            raise ValueError(family)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def predict(model: QueryNet, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(transform(values, mean, scale))).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probabilities


def average_precision(truth: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels = np.asarray(truth, dtype=bool)
    sample_weight = np.ones(len(labels), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    ordered_truth = labels[order]; ordered_weight = sample_weight[order]
    positive_weight = ordered_weight * ordered_truth
    denominator = float(positive_weight.sum())
    if denominator <= 0:
        return math.nan
    precision = np.cumsum(positive_weight) / np.cumsum(ordered_weight)
    return float(np.sum(precision * positive_weight) / denominator)


def pr_curve(truth: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(truth, dtype=bool)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    ordered = labels[order]
    tp = np.cumsum(ordered); precision = tp / np.arange(1, len(ordered) + 1)
    recall = tp / max(int(labels.sum()), 1)
    return np.r_[1.0, recall], np.r_[1.0, precision]


def confusion(truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None, *, normalize: bool = False) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    sample_weight = np.ones(len(truth), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    for actual, guess, weight in zip(truth, predicted, sample_weight):
        matrix[int(actual), int(guess)] += weight
    if normalize:
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-30)
    return matrix


def macro_metrics(probabilities: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> dict:
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion(truth, predicted, weights)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1e-30)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1e-30)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-30)
    support = matrix.sum(axis=1)
    auprc = [average_precision(truth == index, probabilities[:, index], weights) for index in range(3)]
    return {
        "macro_f1": float(np.mean(f1)),
        "macro_auprc": float(np.mean(auprc)),
        "per_class_precision": precision.astype(float).tolist(),
        "per_class_recall": recall.astype(float).tolist(),
        "per_class_f1": f1.astype(float).tolist(),
        "per_class_auprc": [float(value) for value in auprc],
        "support": support.astype(float).tolist(),
        "null_false_accept_rate": float(np.average(predicted[truth == 1] == 0, weights=None if weights is None else weights[truth == 1])),
        "ambiguous_false_accept_rate": float(np.average(predicted[truth == 2] == 0, weights=None if weights is None else weights[truth == 2])),
        "unique_false_reject_rate": float(np.average(predicted[truth == 0] != 0, weights=None if weights is None else weights[truth == 0])),
    }


def fit_model(family: str, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray):
    torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1))
    mean, scale = standardizer(x_train)
    train_x = torch.from_numpy(transform(x_train, mean, scale)); train_y = torch.from_numpy(y_train.astype(np.int64))
    valid_x = torch.from_numpy(transform(x_valid, mean, scale))
    model = QueryNet(train_x.shape[1], family).cpu()
    counts = np.bincount(y_train, minlength=3)
    class_weight = torch.from_numpy((len(y_train) / (len(CLASSES) * counts)).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state = None; best_score = -math.inf; best_epoch = -1; stale = 0
    for epoch in range(100):
        model.train(); indices = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(indices), 512):
            batch = indices[start:start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[batch]), train_y[batch], weight=class_weight)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(valid_x), dim=1).numpy()
        score = macro_metrics(probabilities, y_valid)["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score; best_epoch = epoch + 1; stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 20 and stale >= 15:
            break
    if best_state is None:
        raise RuntimeError("No valid query checkpoint")
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch


def save_model(path: Path, model: QueryNet, mean: np.ndarray, scale: np.ndarray, *, family: str, feature: str, seed: int, best_epoch: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "state_dict": model.state_dict(), "mean": mean, "scale": scale, "family": family, "feature_family": feature,
        "seed": seed, "best_epoch": best_epoch, "classes": CLASSES, "device": "cpu",
    }, path)


def load_data(run: Path, dataset: str):
    # The scene-id array is an object string array written by pandas; the file
    # is a locally hashed campaign artifact, never an external input.
    feature = np.load(run / f"features/v2_{dataset}_features.npz", allow_pickle=True)
    # Query state "NULL" is a scientific class label, not a missing token.
    samples = pd.read_csv(run / f"features/v3_{dataset}_samples.csv", keep_default_na=False)
    if feature["scene_id"].astype(str).tolist() != samples.scene_id.astype(str).tolist():
        raise RuntimeError(f"Feature/sample misalignment: {dataset}")
    labels = np.asarray([CLASS_TO_INDEX[value] for value in samples.query_state], dtype=int)
    return feature, samples, labels


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run = args.run_dir.resolve()
    if json.loads((run / "logs/feature_extraction_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Feature extraction gate did not pass")
    if (run / "logs/query_gate_training_complete.json").exists():
        raise FileExistsError("Query gate training already completed")
    started = time.time()
    train_npz, train_samples, y_train = load_data(run, "q_training")
    valid_npz, valid_samples, y_valid = load_data(run, "q_validation")
    valid_manifest = pd.read_csv(run / "manifests/v2_q_validation_scene_manifest.csv")
    if valid_manifest.scene_id.tolist() != valid_samples.scene_id.tolist():
        raise RuntimeError("Query validation manifest misalignment")
    natural_weights = valid_manifest.inverse_sampling_weight.to_numpy(dtype=float)

    candidate_rows = []; fitted = {}
    for feature in FEATURE_KEYS:
        for family in FAMILIES:
            model, mean, scale, best_epoch = fit_model(family, BASE_SEED, train_npz[feature], y_train, valid_npz[feature], y_valid)
            logits, probabilities = predict(model, valid_npz[feature], mean, scale)
            stratified = macro_metrics(probabilities, y_valid)
            natural = macro_metrics(probabilities, y_valid, natural_weights)
            row = {
                "feature_family": feature, "head_family": family, "seed": BASE_SEED, "best_epoch": best_epoch,
                **{f"stratified_{key}": value for key, value in stratified.items() if not isinstance(value, list)},
                **{f"natural_{key}": value for key, value in natural.items() if not isinstance(value, list)},
                "ambiguous_minus_unique_mean_p_unique": float(probabilities[y_valid == 2, 0].mean() - probabilities[y_valid == 0, 0].mean()),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
            candidate_rows.append(row); fitted[(feature, family)] = (model, mean, scale, best_epoch)
            save_model(run / f"models/query_candidate_{feature}_{family}.pth", model, mean, scale, family=family, feature=feature, seed=BASE_SEED, best_epoch=best_epoch)
    candidates = pd.DataFrame(candidate_rows)
    write_csv_fresh(run / "tables/query_gate_candidate_comparison.csv", candidates)
    ranked = candidates.sort_values(["stratified_macro_f1", "stratified_macro_auprc", "natural_macro_f1"], ascending=False, kind="stable")
    selected_feature = str(ranked.iloc[0].feature_family); selected_family = str(ranked.iloc[0].head_family)

    seed_rows = []; per_class_rows = []; seed_probabilities = []; seed_logits = []
    for seed in SEEDS:
        model, mean, scale, best_epoch = fit_model(selected_family, seed, train_npz[selected_feature], y_train, valid_npz[selected_feature], y_valid)
        logits, probabilities = predict(model, valid_npz[selected_feature], mean, scale)
        metrics = macro_metrics(probabilities, y_valid); natural = macro_metrics(probabilities, y_valid, natural_weights)
        gap = float(probabilities[y_valid == 2, 0].mean() - probabilities[y_valid == 0, 0].mean())
        seed_rows.append({
            "seed": seed, "best_epoch": best_epoch, "macro_f1": metrics["macro_f1"], "macro_auprc": metrics["macro_auprc"],
            "null_recall": metrics["per_class_recall"][1], "ambiguous_recall": metrics["per_class_recall"][2], "unique_recall": metrics["per_class_recall"][0],
            "null_false_accept_rate": metrics["null_false_accept_rate"], "ambiguous_false_accept_rate": metrics["ambiguous_false_accept_rate"],
            "unique_false_reject_rate": metrics["unique_false_reject_rate"], "ambiguous_minus_unique_mean_p_unique": gap,
            "natural_macro_f1": natural["macro_f1"], "natural_macro_auprc": natural["macro_auprc"],
        })
        for index, class_name in enumerate(CLASSES):
            per_class_rows.append({"seed": seed, "class": class_name, "precision": metrics["per_class_precision"][index], "recall": metrics["per_class_recall"][index], "f1": metrics["per_class_f1"][index], "one_vs_rest_auprc": metrics["per_class_auprc"][index], "support": metrics["support"][index]})
        save_model(run / f"models/query_gate_{selected_feature}_{selected_family}_seed_{seed}.pth", model, mean, scale, family=selected_family, feature=selected_feature, seed=seed, best_epoch=best_epoch)
        seed_probabilities.append(probabilities); seed_logits.append(logits)
    seed_frame = pd.DataFrame(seed_rows); write_csv_fresh(run / "tables/query_gate_seed_stability.csv", seed_frame)
    write_csv_fresh(run / "tables/query_gate_per_class_metrics.csv", pd.DataFrame(per_class_rows))
    ensemble = np.mean(seed_probabilities, axis=0); ensemble_logits = np.mean(seed_logits, axis=0)
    ensemble_metrics = macro_metrics(ensemble, y_valid); ensemble_natural = macro_metrics(ensemble, y_valid, natural_weights)
    prediction_path = run / "features/query_gate_validation_predictions.npz"
    if prediction_path.exists(): raise FileExistsError(prediction_path)
    np.savez_compressed(prediction_path, scene_id=valid_samples.scene_id.astype(str).to_numpy(), truth=y_valid, ensemble_logits=ensemble_logits, ensemble_probability=ensemble, seed_probability=np.stack(seed_probabilities))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, seed, probabilities in zip(axes.flat, SEEDS, seed_probabilities):
        matrix = confusion(y_valid, np.argmax(probabilities, axis=1), normalize=True)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        ax.set_title(f"seed {seed}"); ax.set_xticks(range(3), ["U", "N", "A"]); ax.set_yticks(range(3), ["U", "N", "A"])
    axes.flat[-1].axis("off"); fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7); fig.suptitle("Query-state validation confusion matrices (row normalized)")
    fig.savefig(run / "figures/query_gate_confusion_matrices.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for index, class_name in enumerate(CLASSES):
        recall, precision = pr_curve((y_valid == index).astype(int), ensemble[:, index])
        ax.plot(recall, precision, label=f"{class_name} AP={ensemble_metrics['per_class_auprc'][index]:.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(); fig.tight_layout()
    fig.savefig(run / "figures/query_gate_per_class_pr.png", dpi=180); plt.close(fig)

    requirements = {
        "ambiguity_inversion_removed_all_seeds": bool((seed_frame.ambiguous_minus_unique_mean_p_unique < 0).all()),
        "null_recall_meaningful_all_seeds": bool((seed_frame.null_recall > 0.50).all()),
        "ambiguous_recall_meaningful_all_seeds": bool((seed_frame.ambiguous_recall > 0.50).all()),
        "macro_f1_seed_standard_deviation_at_most_0_05": bool(seed_frame.macro_f1.std(ddof=1) <= 0.05),
        "macro_f1_above_balanced_chance": bool(ensemble_metrics["macro_f1"] > 1 / 3),
    }
    passed = all(requirements.values())
    selection = {
        "status": "PASS" if passed else "FAIL", "selection_split": "q_validation only", "selected_feature_family": selected_feature,
        "selected_head_family": selected_family, "seeds": list(SEEDS), "deployment_head": "unweighted probability ensemble of five selected-family seeds",
        "selection_order": ["stratified_macro_f1", "stratified_macro_auprc", "natural_macro_f1"],
        "ensemble_stratified_metrics": ensemble_metrics, "ensemble_natural_weighted_metrics": ensemble_natural,
        "requirements": requirements, "calibration_used": False, "development_used": False, "lockbox_used": False,
    }
    write_json_fresh(run / "manifests/query_gate_selection.json", selection)
    write_json_fresh(run / "reports/query_gate_gate.json", selection)
    write_json_fresh(run / "logs/query_gate_training_complete.json", {
        "status": "PASS" if passed else "FAIL", "runtime_seconds": time.time() - started, "device": "cpu", "candidate_count": len(candidates),
        "selected_feature_family": selected_feature, "selected_head_family": selected_family, "development_used": False, "lockbox_used": False,
    })
    print(json.dumps({"passed": passed, "selected_feature": selected_feature, "selected_family": selected_family, "ensemble_macro_f1": ensemble_metrics["macro_f1"]}, sort_keys=True))
    if not passed:
        raise RuntimeError("Query-state gate failed; stopping before valid-risk head training or policy evaluation")


if __name__ == "__main__":
    main()
