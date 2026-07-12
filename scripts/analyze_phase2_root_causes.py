#!/usr/bin/env python3
"""Read-only diagnosis of the frozen Thayer-Select Phase-II campaigns.

This script never reads the future lockbox, never trains or changes a neural
network, and never regenerates development reconstructions.  The sole neural
operation is an encoder-only pass through the frozen original R1 checkpoint to
obtain its exact pooled bottleneck features for diagnostic linear probes.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models_thayer_select import ThayerSelectNet


PRIMARY = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
REPLICATION = REPO / "outputs/runs/thayer_select_recoverability_seed_replication_20260711_203115"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
OUT = REPO / "outputs/runs/thayer_select_root_cause_analysis_20260711"
FIG = OUT / "figures"
TABLE = OUT / "tables"

QUERY_ORDER = ["VALID_SOURCE", "NULL_SOURCE", "AMBIGUOUS_SOURCE", "PERTURBED_VALID"]
SOURCE_QUERIES = {"VALID_SOURCE", "PERTURBED_VALID"}
SEED_NAMES = ["R1_original", "R1_seed_2", "R1_seed_3"]
EPS = 1e-12


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def auc_binary(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    mask = np.isfinite(score) & np.isfinite(label)
    score, label = score[mask], label[mask]
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def directionless_auc(value: np.ndarray, failure: np.ndarray) -> tuple[float, float, str]:
    raw = auc_binary(value, failure.astype(int))
    if not np.isfinite(raw):
        return math.nan, math.nan, "undefined"
    return max(raw, 1.0 - raw), raw, "higher in failures" if raw >= 0.5 else "lower in failures"


def bootstrap_auc(value: np.ndarray, failure: np.ndarray, seed: int = 7711) -> tuple[float, float]:
    value = np.asarray(value, dtype=float)
    failure = np.asarray(failure, dtype=bool)
    mask = np.isfinite(value)
    value, failure = value[mask], failure[mask]
    rng = np.random.default_rng(seed)
    observed = []
    positive = np.flatnonzero(failure)
    negative = np.flatnonzero(~failure)
    for _ in range(1000):
        indices = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        power, _, _ = directionless_auc(value[indices], failure[indices])
        observed.append(power)
    return tuple(np.quantile(observed, [0.025, 0.975]).tolist())


def centroid(image: np.ndarray) -> tuple[float, float]:
    weight = np.maximum(np.asarray(image, dtype=float), 0.0)
    total = float(weight.sum())
    if total <= 0 or not np.isfinite(total):
        return math.nan, math.nan
    yy, xx = np.indices(weight.shape)
    return float(np.sum(xx * weight) / total), float(np.sum(yy * weight) / total)


def cross_band_shift(scene: np.ndarray) -> float:
    centers = np.asarray([centroid(scene[band]) for band in range(3)], dtype=float)
    if not np.isfinite(centers).all():
        return math.nan
    return float(max(np.linalg.norm(centers[i] - centers[j]) for i in range(3) for j in range(i + 1, 3)))


def morphology_metrics(scene: np.ndarray) -> dict[str, float]:
    band_flux = np.sum(np.maximum(scene, 0.0), axis=(1, 2))
    scaled = np.stack([np.maximum(scene[i], 0.0) / max(float(band_flux[i]), EPS) for i in range(3)])
    image = scaled.mean(axis=0)
    image /= max(float(image.sum()), EPS)
    peak = float(image.max())
    support = image > 0.01 * peak
    gx = ndimage.sobel(image, axis=1, mode="nearest")
    gy = ndimage.sobel(image, axis=0, mode="nearest")
    gradient = np.hypot(gx, gy)
    edge_density = float(np.mean(gradient[support] > 0.10 * float(gradient.max()))) if support.any() else math.nan
    smooth = ndimage.gaussian_filter(image, sigma=1.0, mode="nearest")
    high_frequency = float(np.sum((image - smooth) ** 2) / max(np.sum(image**2), EPS))
    cx, cy = centroid(image)
    yy, xx = np.indices(image.shape)
    rotated = ndimage.map_coordinates(image, [2.0 * cy - yy, 2.0 * cx - xx], order=1, mode="constant", cval=0.0)
    asymmetry = float(np.sum(np.abs(image - rotated)) / max(2.0 * np.sum(np.abs(image)), EPS))
    radius = np.hypot(xx - cx, yy - cy)
    radial_bin = np.floor(radius).astype(int)
    radial_model = np.zeros_like(image)
    for index in range(int(radial_bin.max()) + 1):
        mask = radial_bin == index
        if mask.any():
            radial_model[mask] = float(np.mean(image[mask]))
    radial_smoothness = float(1.0 - np.sum(np.abs(image - radial_model)) / max(np.sum(np.abs(image)), EPS))
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(image.ravel()[order])
    sorted_radius = radius.ravel()[order]
    r20 = float(sorted_radius[min(np.searchsorted(cumulative, 0.20), len(sorted_radius) - 1)])
    r80 = float(sorted_radius[min(np.searchsorted(cumulative, 0.80), len(sorted_radius) - 1)])
    concentration = float(5.0 * np.log10(max(r80, 0.25) / max(r20, 0.25)))
    return {
        "edge_density": edge_density,
        "high_frequency_energy": high_frequency,
        "asymmetry": asymmetry,
        "radial_smoothness": radial_smoothness,
        "concentration": concentration,
    }


def source_overlap(isolated: np.ndarray) -> float:
    maps = np.sum(np.abs(isolated), axis=1)
    numerator = float(np.minimum(maps[0], maps[1]).sum())
    denominator = min(float(maps[0].sum()), float(maps[1].sum()))
    return numerator / max(denominator, EPS)


def load_development() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray, np.ndarray]:
    manifest = read_csv(PRIMARY / "manifests/development_test_scene_manifest.csv")
    manifest = numeric(
        manifest,
        [
            "source_a_row", "source_b_row", "matched_source_index", "separation_pixels",
            "separation_psf_units", "flux_ratio", "size_ratio", "core_obstruction",
            "snr_proxy", "color_similarity_distance", "coordinate_error_pixels",
        ],
    )
    original = read_csv(PRIMARY / "tables/development_metrics_per_sample.csv")
    original = original[original["condition"] == "R1_calibrated"].copy()
    original["condition"] = "R1_original"
    seeds = {
        "R1_original": original,
        "R1_seed_2": read_csv(REPLICATION / "tables/r1_seed_2_development_per_sample.csv"),
        "R1_seed_3": read_csv(REPLICATION / "tables/r1_seed_3_development_per_sample.csv"),
    }
    metric_columns = [
        "raw_score", "calibrated_score", "pixel_uncertainty_mean", "normalized_rmse",
        "source_mse", "whole_mse", "max_relative_flux_error", "max_color_error_mag",
        "centroid_error_pixels", "permissive_actionable_success",
    ]
    for name, frame in seeds.items():
        frame = numeric(frame.copy(), metric_columns)
        manifest_columns = [column for column in manifest.columns if column == "scene_id" or (column != "query_class" and column not in frame.columns)]
        seeds[name] = frame.merge(
            manifest[manifest_columns], on="scene_id", how="left", validate="one_to_one"
        )
        seeds[name]["catastrophic_failure_bool"] = bool_values(seeds[name]["catastrophic_failure"])
    with h5py.File(PRIMARY / "manifests/development_test_scenes.h5", "r") as handle:
        blends = handle["blend"][:]
        isolated = handle["isolated"][:]
        prompts = handle["prompt"][:]
    return manifest, seeds, blends, isolated, prompts


def add_catalog_and_image_features(manifest: pd.DataFrame, seeds: dict[str, pd.DataFrame], blends: np.ndarray, isolated: np.ndarray) -> pd.DataFrame:
    catalog_path = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
    with fits.open(catalog_path, memmap=True) as hdul:
        catalog = hdul[1].data
        rows_a = manifest["source_a_row"].astype(int).to_numpy()
        rows_b = manifest["source_b_row"].astype(int).to_numpy()
        g_r_a = np.asarray(catalog["g_ab"][rows_a] - catalog["r_ab"][rows_a], dtype=float)
        g_r_b = np.asarray(catalog["g_ab"][rows_b] - catalog["r_ab"][rows_b], dtype=float)
        r_z_a = np.asarray(catalog["r_ab"][rows_a] - catalog["z_ab"][rows_a], dtype=float)
        r_z_b = np.asarray(catalog["r_ab"][rows_b] - catalog["z_ab"][rows_b], dtype=float)
        bulge_a = np.asarray(catalog["fluxnorm_bulge"][rows_a], dtype=float)
        disk_a = np.asarray(catalog["fluxnorm_disk"][rows_a], dtype=float)
        agn_a = np.asarray(catalog["fluxnorm_agn"][rows_a], dtype=float)
        bulge_b = np.asarray(catalog["fluxnorm_bulge"][rows_b], dtype=float)
        disk_b = np.asarray(catalog["fluxnorm_disk"][rows_b], dtype=float)
        agn_b = np.asarray(catalog["fluxnorm_agn"][rows_b], dtype=float)
    features = manifest[["scene_id"]].copy()
    features["g_minus_r_difference"] = np.abs(g_r_a - g_r_b)
    features["r_minus_z_difference"] = np.abs(r_z_a - r_z_b)
    features["color_difference"] = np.hypot(features["g_minus_r_difference"], features["r_minus_z_difference"])
    bt_a = bulge_a / np.maximum(bulge_a + disk_a + agn_a, EPS)
    bt_b = bulge_b / np.maximum(bulge_b + disk_b + agn_b, EPS)
    features["morphology_difference"] = np.abs(bt_a - bt_b)
    overlaps = []
    shifts = []
    morph = []
    for index in range(len(manifest)):
        overlaps.append(source_overlap(isolated[index]))
        noiseless_scene = isolated[index, 0] + isolated[index, 1]
        shifts.append(cross_band_shift(noiseless_scene))
        morph.append(morphology_metrics(noiseless_scene))
    features["overlap"] = overlaps
    features["cross_band_centroid_shift"] = shifts
    features = pd.concat([features, pd.DataFrame(morph)], axis=1)
    for name in seeds:
        seeds[name] = seeds[name].merge(features, on="scene_id", how="left", validate="one_to_one")
    return features


def label_audit(original: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query in QUERY_ORDER:
        frame = original[original["query_class"] == query]
        labels = frame["permissive_actionable_success"].to_numpy(dtype=int)
        score = np.clip(frame["raw_score"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
        bce = -(labels * np.log(score) + (1 - labels) * np.log(1 - score))
        rows.append({
            "query_class": query,
            "samples": len(frame),
            "positive_labels": int(labels.sum()),
            "negative_labels": int(len(frame) - labels.sum()),
            "success_rate": float(labels.mean()),
            "catastrophic_failure_rate": float(frame["catastrophic_failure_bool"].mean()),
            "average_reconstruction_error_normalized_rmse": float(frame["normalized_rmse"].mean()),
            "average_recoverability_raw_score": float(frame["raw_score"].mean()),
            "average_calibrated_probability": float(frame["calibrated_score"].mean()),
            "average_uncertainty": float(frame["pixel_uncertainty_mean"].mean()),
            "average_recoverability_bce_loss": float(np.mean(bce)),
        })
    return pd.DataFrame(rows)


def calibration_audit(seeds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    calibration_files = {
        "R1_original": PRIMARY / "tables/calibration_per_sample.csv",
        "R1_seed_2": REPLICATION / "tables/r1_seed_2_calibration_per_sample.csv",
        "R1_seed_3": REPLICATION / "tables/r1_seed_3_calibration_per_sample.csv",
    }
    calibration = {name: numeric(read_csv(path), ["raw_score", "calibrated_score", "permissive_actionable_success"]) for name, path in calibration_files.items()}
    rows = []
    for name in SEED_NAMES:
        for split, frame in (("calibration", calibration[name]), ("development", seeds[name])):
            raw = frame["raw_score"].to_numpy(dtype=float)
            calibrated = frame["calibrated_score"].to_numpy(dtype=float)
            labels = frame["permissive_actionable_success"].to_numpy(dtype=int)
            counts = Counter(calibrated.tolist())
            rows.append({
                "seed": name,
                "split": split,
                "samples": len(frame),
                "positive_rate": float(labels.mean()),
                "unique_raw_probabilities": int(np.unique(raw).size),
                "unique_calibrated_probabilities": int(np.unique(calibrated).size),
                "tie_percentage_one_minus_unique_over_n": float(100.0 * (1.0 - np.unique(calibrated).size / len(calibrated))),
                "largest_plateau_samples": int(max(counts.values())),
                "largest_plateau_percentage": float(100.0 * max(counts.values()) / len(calibrated)),
                "zero_probability_percentage": float(100.0 * np.mean(calibrated == 0.0)),
                "raw_auroc": auc_binary(raw, labels),
                "calibrated_auroc": auc_binary(calibrated, labels),
                "raw_vs_calibrated_spearman": float(spearmanr(raw, calibrated).statistic),
            })
    return pd.DataFrame(rows), calibration


PREDICTORS = {
    "overlap": "overlap",
    "separation": "separation_pixels",
    "separation / PSF": "separation_psf_units",
    "flux ratio": "flux_ratio_log_abs",
    "size ratio": "size_ratio_log_abs",
    "SNR": "snr_proxy_log",
    "core obstruction": "core_obstruction",
    "color difference": "color_difference",
    "morphology difference": "morphology_difference",
}


def add_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["flux_ratio_log_abs"] = np.abs(np.log10(np.maximum(frame["flux_ratio"], EPS)))
    frame["size_ratio_log_abs"] = np.abs(np.log10(np.maximum(frame["size_ratio"], EPS)))
    frame["snr_proxy_log"] = np.log10(np.maximum(frame["snr_proxy"], EPS))
    return frame


def failure_correlations(seeds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = []
    for name in SEED_NAMES:
        frame = add_transforms(seeds[name])
        frame = frame[frame["query_class"].isin(SOURCE_QUERIES)].copy()
        failure = frame["catastrophic_failure_bool"].to_numpy()
        for label, column in PREDICTORS.items():
            values = frame[column].to_numpy(dtype=float)
            mask = np.isfinite(values)
            power, raw, direction = directionless_auc(values[mask], failure[mask])
            success_values = values[mask & ~failure]
            failure_values = values[mask & failure]
            p = float(mannwhitneyu(failure_values, success_values, alternative="two-sided").pvalue)
            row = {
                "seed": name,
                "variable": label,
                "n": int(mask.sum()),
                "failure_count": int(failure[mask].sum()),
                "predictive_auc": power,
                "signed_auc_failure_high": raw,
                "direction": direction,
                "failure_median": float(np.median(failure_values)),
                "nonfailure_median": float(np.median(success_values)),
                "mann_whitney_p": p,
            }
            if name == "R1_original":
                row["bootstrap_auc_ci_low"], row["bootstrap_auc_ci_high"] = bootstrap_auc(values[mask], failure[mask])
            all_rows.append(row)
    result = pd.DataFrame(all_rows)
    mean_power = result.groupby("variable")["predictive_auc"].mean().sort_values(ascending=False)
    result["mean_three_seed_auc"] = result["variable"].map(mean_power)
    result["rank"] = result["variable"].map({name: rank + 1 for rank, name in enumerate(mean_power.index)})
    return result.sort_values(["rank", "seed"])


def two_group_audit(original: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    source = original[original["query_class"].isin(SOURCE_QUERIES)].copy()
    success = source["permissive_actionable_success"].to_numpy(dtype=int) == 1
    catastrophic = source["catastrophic_failure_bool"].to_numpy()
    keep = success | catastrophic
    success = success[keep]
    rows = []
    for label, column in columns.items():
        values = source.loc[keep, column].to_numpy(dtype=float)
        mask = np.isfinite(values)
        failure = ~success[mask]
        power, raw, direction = directionless_auc(values[mask], failure)
        failed_values = values[mask][failure]
        successful_values = values[mask][~failure]
        rows.append({
            "variable": label,
            "successful_count": int((~failure).sum()),
            "catastrophic_count": int(failure.sum()),
            "successful_mean": float(np.mean(successful_values)),
            "successful_median": float(np.median(successful_values)),
            "catastrophic_mean": float(np.mean(failed_values)),
            "catastrophic_median": float(np.median(failed_values)),
            "predictive_auc": power,
            "signed_auc_failure_high": raw,
            "direction": direction,
            "mann_whitney_p": float(mannwhitneyu(failed_values, successful_values, alternative="two-sided").pvalue),
        })
    return pd.DataFrame(rows).sort_values("predictive_auc", ascending=False)


def extract_latent(blends: np.ndarray, prompts: np.ndarray) -> tuple[np.ndarray, dict]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the authorized encoder-only feature extraction")
    device = torch.device("mps")
    checkpoint = torch.load(PRIMARY / "checkpoints/r1_best.pth", map_location="cpu", weights_only=False)
    model = ThayerSelectNet(min_log_variance=-8.0, max_log_variance=2.0).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    normalization = json.loads((PRIMARY / "manifests/normalization.json").read_text())
    scales = np.asarray(normalization["per_band_scale"], dtype=np.float32)
    blocks = []
    with torch.no_grad():
        for start in range(0, len(blends), 128):
            image = torch.from_numpy(np.ascontiguousarray(blends[start:start + 128] / scales[None, :, None, None])).to(device)
            prompt = torch.from_numpy(np.ascontiguousarray(prompts[start:start + 128])).to(device)
            inputs = torch.cat((image, prompt), dim=1)
            enc1 = model.enc1(inputs)
            enc2 = model.enc2(F.avg_pool2d(enc1, 2))
            bottleneck = model.bottleneck(F.avg_pool2d(enc2, 2))
            pooled = F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(1)
            blocks.append(pooled.cpu().numpy())
    features = np.concatenate(blocks, axis=0)
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probability = eigenvalues / max(float(eigenvalues.sum()), EPS)
    effective_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    diagnostics = {
        "device": "mps",
        "operation": "encoder-only through pooled bottleneck; decoder and heads not executed",
        "samples": int(features.shape[0]),
        "dimensions": int(features.shape[1]),
        "effective_covariance_rank": effective_rank,
        "near_constant_dimensions_std_lt_1e_6": int(np.sum(features.std(axis=0) < 1e-6)),
        "checkpoint_sha256_declared": checkpoint.get("best_checkpoint_sha256", "not embedded"),
    }
    return features, diagnostics


def stratified_folds(labels: np.ndarray, folds: int = 5, seed: int = 20260711) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for index, value in enumerate(indices):
            buckets[index % folds].append(int(value))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets]


def fit_multinomial(x: np.ndarray, y: np.ndarray, classes: int, l2: float = 1e-2) -> np.ndarray:
    counts = np.bincount(y, minlength=classes).astype(float)
    sample_weight = len(y) / (classes * counts[y])
    xb = np.column_stack((np.ones(len(x)), x))
    dimensions = xb.shape[1]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        weight = flat.reshape(dimensions, classes)
        logits = xb @ weight
        log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(sample_weight * log_probability[np.arange(len(y)), y]) / np.sum(sample_weight)
        loss += 0.5 * l2 * float(np.sum(weight[1:] ** 2))
        probability = np.exp(log_probability)
        probability[np.arange(len(y)), y] -= 1.0
        gradient = xb.T @ (sample_weight[:, None] * probability) / np.sum(sample_weight)
        gradient[1:] += l2 * weight[1:]
        return float(loss), gradient.ravel()

    result = minimize(objective, np.zeros(dimensions * classes), method="L-BFGS-B", jac=True, options={"maxiter": 600, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Linear probe optimization failed: {result.message}")
    return result.x.reshape(dimensions, classes)


def cross_validated_probe(x: np.ndarray, labels: np.ndarray, names: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    folds = stratified_folds(labels)
    probabilities = np.zeros((len(labels), len(names)), dtype=float)
    predictions = np.zeros(len(labels), dtype=int)
    for test in folds:
        train = np.setdiff1d(np.arange(len(labels)), test, assume_unique=True)
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = (x[train] - mean) / scale
        test_x = (x[test] - mean) / scale
        weight = fit_multinomial(train_x, labels[train], len(names))
        logits = np.column_stack((np.ones(len(test_x)), test_x)) @ weight
        probabilities[test] = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        predictions[test] = np.argmax(probabilities[test], axis=1)
    recalls = [float(np.mean(predictions[labels == index] == index)) for index in range(len(names))]
    aucs = [auc_binary(probabilities[:, index], labels == index) for index in range(len(names))]
    confusion = np.zeros((len(names), len(names)), dtype=int)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    metrics = {
        "samples": len(labels),
        "class_counts": {name: int(np.sum(labels == index)) for index, name in enumerate(names)},
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_one_vs_rest_auroc": float(np.nanmean(aucs)),
        "per_class_recall": dict(zip(names, recalls)),
        "per_class_auroc": dict(zip(names, aucs)),
    }
    return metrics, confusion, probabilities


def latent_audit(features: np.ndarray, original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    confusion_rows = []
    details = {}

    query = original["query_class"].to_numpy()
    positive = original["permissive_actionable_success"].to_numpy(dtype=int) == 1
    catastrophic = original["catastrophic_failure_bool"].to_numpy()
    groups = np.full(len(original), "excluded_intermediate_source", dtype=object)
    groups[query == "NULL_SOURCE"] = "null"
    groups[query == "AMBIGUOUS_SOURCE"] = "ambiguous"
    source = np.isin(query, list(SOURCE_QUERIES))
    groups[source & catastrophic] = "catastrophic_source"
    groups[source & positive] = "successful_source"
    names = ["successful_source", "catastrophic_source", "ambiguous", "null"]
    keep = np.isin(groups, names)
    labels = np.asarray([names.index(value) for value in groups[keep]], dtype=int)
    metrics, confusion, probability = cross_validated_probe(features[keep], labels, names)
    details["four_group"] = metrics
    rows.append({"probe": "four_group", **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}})
    for i, truth in enumerate(names):
        for j, prediction in enumerate(names):
            confusion_rows.append({"probe": "four_group", "truth": truth, "prediction": prediction, "count": int(confusion[i, j])})

    binary_specs = {
        "successful_vs_catastrophic_source": (source & (positive | catastrophic), positive, ["catastrophic", "successful"]),
        "ambiguous_vs_source_query": (source | (query == "AMBIGUOUS_SOURCE"), query == "AMBIGUOUS_SOURCE", ["source", "ambiguous"]),
        "null_vs_source_query": (source | (query == "NULL_SOURCE"), query == "NULL_SOURCE", ["source", "null"]),
    }
    for probe, (mask, target, class_names) in binary_specs.items():
        labels = target[mask].astype(int)
        metrics, confusion, _ = cross_validated_probe(features[mask], labels, class_names)
        details[probe] = metrics
        existing_auc = math.nan
        if probe == "successful_vs_catastrophic_source":
            existing_auc = auc_binary(original.loc[mask, "raw_score"].to_numpy(dtype=float), labels)
        rows.append({"probe": probe, **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}, "existing_head_auroc": existing_auc})
        for i, truth in enumerate(class_names):
            for j, prediction in enumerate(class_names):
                confusion_rows.append({"probe": probe, "truth": truth, "prediction": prediction, "count": int(confusion[i, j])})
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows), details


def teacher_audit() -> pd.DataFrame:
    actionable = pd.concat([
        read_csv(PRIMARY / "tables/training_actionable_acceptance_labels.csv"),
        read_csv(PRIMARY / "tables/validation_actionable_acceptance_labels.csv"),
    ], ignore_index=True)
    outcomes = pd.concat([
        read_csv(PRIMARY / "tables/training_teacher_reliability_labels.csv"),
        read_csv(PRIMARY / "tables/validation_teacher_reliability_labels.csv"),
    ], ignore_index=True)
    teacher = outcomes.merge(
        actionable[["scene_id", "permissive_actionable_success"]],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )
    manifest = read_csv(PRIMARY / "manifests/all_scene_manifest.csv")
    manifest = numeric(manifest, ["separation_psf_units", "flux_ratio", "size_ratio", "core_obstruction", "snr_proxy", "color_similarity_distance", "coordinate_error_pixels"])
    teacher = numeric(teacher, ["permissive_actionable_success", "normalized_rmse", "max_relative_flux_error", "max_color_error_mag", "centroid_error_pixels"])
    teacher = teacher.merge(manifest[[
        "scene_id", "separation_psf_units", "flux_ratio", "size_ratio", "core_obstruction",
        "snr_proxy", "color_similarity_distance", "coordinate_error_pixels",
    ]], on="scene_id", how="left", validate="one_to_one")
    teacher = teacher[teacher["query_class"].isin(SOURCE_QUERIES)].copy()
    teacher = add_transforms(teacher)
    labels = teacher["permissive_actionable_success"].to_numpy(dtype=int)
    predictors = {
        "reconstruction: normalized RMSE": "normalized_rmse",
        "reconstruction: max flux error": "max_relative_flux_error",
        "reconstruction: max color error": "max_color_error_mag",
        "reconstruction: centroid error": "centroid_error_pixels",
        "generator: separation / PSF": "separation_psf_units",
        "generator: flux ratio": "flux_ratio_log_abs",
        "generator: size ratio": "size_ratio_log_abs",
        "generator: core obstruction": "core_obstruction",
        "generator: SNR": "snr_proxy_log",
        "generator: source color distance": "color_similarity_distance",
        "generator: prompt offset": "coordinate_error_pixels",
    }
    rows = []
    for label, column in predictors.items():
        values = teacher[column].to_numpy(dtype=float)
        mask = np.isfinite(values)
        raw = auc_binary(values[mask], labels[mask])
        rows.append({
            "predictor": label,
            "family": label.split(":", 1)[0],
            "n": int(mask.sum()),
            "positive_labels": int(labels[mask].sum()),
            "positive_rate": float(labels[mask].mean()),
            "predictive_auc": max(raw, 1.0 - raw),
            "signed_auc_positive_high": raw,
            "direction": "higher predicts success" if raw >= 0.5 else "lower predicts success",
        })
    return pd.DataFrame(rows).sort_values("predictive_auc", ascending=False)


def plot_outputs(label: pd.DataFrame, calibration: pd.DataFrame, calibration_frames: dict[str, pd.DataFrame], failures: pd.DataFrame, color: pd.DataFrame, morphology: pd.DataFrame, probes: pd.DataFrame, confusion: pd.DataFrame, teacher: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(label))
    axes[0].bar(x, label["success_rate"], color="#377eb8")
    axes[0].set_ylabel("Positive actionable-label rate")
    axes[1].bar(x - 0.18, label["average_recoverability_raw_score"], width=0.36, label="raw")
    axes[1].bar(x + 0.18, label["average_calibrated_probability"], width=0.36, label="isotonic")
    axes[1].set_ylabel("Mean recoverability probability")
    axes[1].legend()
    axes[2].bar(x, label["average_recoverability_bce_loss"], color="#e41a1c")
    axes[2].set_ylabel("Mean unweighted recoverability BCE")
    for axis in axes:
        axis.set_xticks(x, [q.replace("_SOURCE", "").replace("PERTURBED_VALID", "PERTURBED") for q in label["query_class"]], rotation=25, ha="right")
    fig.suptitle("Phase-II label and score audit (original R1 development)")
    fig.tight_layout()
    fig.savefig(FIG / "label_audit.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    for row, name in enumerate(SEED_NAMES):
        frame = calibration_frames[name]
        axes[row, 0].hist(frame["raw_score"], bins=np.linspace(0, 1, 51), color="#4daf4a")
        axes[row, 1].hist(frame["calibrated_score"], bins=np.linspace(0, 1, 51), color="#984ea3")
        axes[row, 0].set_ylabel(f"{name}\ncount")
        zero = 100 * np.mean(frame["calibrated_score"].to_numpy() == 0)
        axes[row, 1].text(0.98, 0.86, f"zero plateau: {zero:.1f}%", ha="right", transform=axes[row, 1].transAxes)
    axes[0, 0].set_title("Raw head probability")
    axes[0, 1].set_title("After isotonic calibration")
    axes[-1, 0].set_xlabel("Probability")
    axes[-1, 1].set_xlabel("Probability")
    fig.suptitle("Calibration-set histogram collapse")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_collapse.png", dpi=180)
    plt.close(fig)

    summary = failures.groupby("variable").agg(mean=("predictive_auc", "mean"), low=("predictive_auc", "min"), high=("predictive_auc", "max")).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(summary))
    ax.errorbar(summary["mean"], y, xerr=[summary["mean"] - summary["low"], summary["high"] - summary["mean"]], fmt="o", color="#e41a1c", capsize=3)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_yticks(y, summary.index)
    ax.set_xlabel("Directionless univariate AUROC for catastrophic failure\n(mean and range across three seeds)")
    ax.set_xlim(0.48, 1.01)
    fig.tight_layout()
    fig.savefig(FIG / "failure_predictive_power.png", dpi=180)
    plt.close(fig)

    combined = pd.concat([color.assign(family="color"), morphology.assign(family="morphology")], ignore_index=True)
    combined = combined.sort_values("predictive_auc")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = combined["family"].map({"color": "#377eb8", "morphology": "#ff7f00"})
    ax.barh(np.arange(len(combined)), combined["predictive_auc"] - 0.5, left=0.5, color=colors)
    ax.axvline(0.5, color="black", lw=1)
    ax.set_yticks(np.arange(len(combined)), combined["variable"])
    ax.set_xlabel("Directionless AUROC: successful vs catastrophic source reconstructions")
    ax.set_xlim(0.48, 1.0)
    fig.tight_layout()
    fig.savefig(FIG / "color_morphology_audit.png", dpi=180)
    plt.close(fig)

    four = confusion[confusion["probe"] == "four_group"]
    names = ["successful_source", "catastrophic_source", "ambiguous", "null"]
    matrix = four.pivot(index="truth", columns="prediction", values="count").reindex(index=names, columns=names).to_numpy()
    matrix_fraction = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(matrix_fraction, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_xticks(range(4), ["success", "catastrophic", "ambiguous", "null"], rotation=25)
    axes[0].set_yticks(range(4), ["success", "catastrophic", "ambiguous", "null"])
    axes[0].set_xlabel("Probe prediction")
    axes[0].set_ylabel("Truth")
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{matrix_fraction[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].barh(np.arange(len(probes)), probes["macro_one_vs_rest_auroc"], color="#4daf4a")
    axes[1].set_yticks(np.arange(len(probes)), probes["probe"].str.replace("_", " "))
    axes[1].set_xlim(0.45, 1.0)
    axes[1].axvline(0.5, color="black", ls="--", lw=1)
    axes[1].set_xlabel("Cross-validated probe AUROC")
    fig.suptitle("Frozen pooled-encoder linear probes")
    fig.tight_layout()
    fig.savefig(FIG / "latent_probe.png", dpi=180)
    plt.close(fig)

    display = teacher.sort_values("predictive_auc")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors = display["family"].map({"reconstruction": "#984ea3", "generator": "#ff7f00"})
    ax.barh(np.arange(len(display)), display["predictive_auc"], color=colors)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_yticks(np.arange(len(display)), display["predictor"])
    ax.set_xlim(0.48, 1.01)
    ax.set_xlabel("Directionless AUROC for frozen teacher actionable label")
    fig.tight_layout()
    fig.savefig(FIG / "teacher_audit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    for path in (OUT, FIG, TABLE):
        path.mkdir(parents=True, exist_ok=True)
    manifest, seeds, blends, isolated, prompts = load_development()
    features = add_catalog_and_image_features(manifest, seeds, blends, isolated)
    original = seeds["R1_original"]

    labels = label_audit(original)
    calibration, calibration_frames = calibration_audit(seeds)
    failures = failure_correlations(seeds)
    color = two_group_audit(original, {
        "|delta(g-r)|": "g_minus_r_difference",
        "|delta(r-z)|": "r_minus_z_difference",
        "cross-band centroid shift": "cross_band_centroid_shift",
    })
    morphology = two_group_audit(original, {
        "edge density": "edge_density",
        "high-frequency energy": "high_frequency_energy",
        "asymmetry": "asymmetry",
        "radial smoothness": "radial_smoothness",
        "concentration": "concentration",
    })
    latent, latent_diagnostics = extract_latent(blends, prompts)
    probes, confusion, probe_details = latent_audit(latent, original)
    teacher = teacher_audit()

    outputs = {
        "label_audit.csv": labels,
        "calibration_audit.csv": calibration,
        "failure_predictors.csv": failures,
        "color_audit.csv": color,
        "morphology_audit.csv": morphology,
        "latent_probe_metrics.csv": probes,
        "latent_probe_confusion.csv": confusion,
        "teacher_audit.csv": teacher,
        "derived_scene_features.csv": features,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLE / name, index=False)

    plot_outputs(labels, calibration, calibration_frames, failures, color, morphology, probes, confusion, teacher)
    original_failure = failures[failures["seed"] == "R1_original"].sort_values("rank")
    evidence = {
        "scope": {
            "lockbox_accessed": False,
            "development_reconstruction_regenerated": False,
            "neural_training_performed": False,
            "new_checkpoint_created": False,
            "latent_extraction": latent_diagnostics,
        },
        "label_audit": labels.to_dict(orient="records"),
        "calibration_audit": calibration.to_dict(orient="records"),
        "failure_ranking_original": original_failure.to_dict(orient="records"),
        "color_audit": color.to_dict(orient="records"),
        "morphology_audit": morphology.to_dict(orient="records"),
        "latent_probes": probe_details,
        "teacher_audit": teacher.to_dict(orient="records"),
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "top_failure_predictors": original_failure[["variable", "predictive_auc", "direction"]].head(5).to_dict(orient="records"),
        "latent_probes": probes.to_dict(orient="records"),
        "teacher_top": teacher.head(5).to_dict(orient="records"),
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
