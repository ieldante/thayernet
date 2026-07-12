#!/usr/bin/env python3
"""Append-only risk-calibration continuation after covariate dtype failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.calibrate_hierarchical_feasibility import (
    ALPHA,
    binary_logits,
    fit_binary_temperature,
    risk_outputs,
    score_structure,
    sha256_file,
    sigmoid,
    auprc,
    auroc,
)
from src.hierarchical_safety import conformal_upper_offset


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not (run / "tables/query_calibration_comparison.csv").is_file():
        raise RuntimeError("Query calibration did not finish before the incident")
    forbidden = [run / "tables/risk_calibration_summary.csv", run / "tables/risk_calibration_subgroup_coverage.csv",
                 run / "tables/binary_risk_calibration_summary.csv", run / "calibration/valid_risk_calibration_predictions.csv"]
    if any(path.exists() for path in forbidden):
        raise FileExistsError("Risk continuation output already exists")
    write_json_fresh(run / "logs/risk_calibration_continuation_code_freeze.json", {
        "status": "HASHED_BEFORE_RISK_CALIBRATION_CONTINUATION", "relative_path": str(Path(__file__).resolve().relative_to(REPO)),
        "sha256": sha256_file(Path(__file__).resolve()), "correction": "explicit numeric conversion for snr_proxy and core_obstruction",
        "query_artifacts_recomputed": False, "risk_artifacts_existing_before_continuation": False,
        "development_accessed": False, "lockbox_accessed": False,
    })
    write_json_fresh(run / "logs/calibration_covariate_dtype_incident.json", {
        "status": "APPEND_ONLY_CORRECTION", "exception": "pandas qcut received string-valued manifest covariates",
        "query_calibration_completed": True, "risk_residual_calibration_completed_before_exception": False,
        "heads_changed": False, "query_artifacts_changed": False, "operational_policy_selected": False,
        "development_accessed": False, "lockbox_accessed": False,
    })
    started = time.time()
    features = np.load(run / "features/v2_natural_calibration_features.npz", allow_pickle=True)
    samples = pd.read_csv(run / "features/v4_natural_calibration_samples.csv", keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(run / "manifests/v2_natural_calibration_scene_manifest.csv", keep_default_na=False)
    valid = samples.query_state == "UNIQUE_VALID"
    valid_indices = np.flatnonzero(valid.to_numpy())
    snr = pd.to_numeric(manifest.loc[valid, "snr_proxy"], errors="raise")
    overlap = pd.to_numeric(manifest.loc[valid, "core_obstruction"], errors="raise")
    subgroup = pd.DataFrame({
        "snr_bin": pd.qcut(snr, 4, labels=["snr_q1", "snr_q2", "snr_q3", "snr_q4"], duplicates="drop").to_numpy(),
        "overlap_bin": pd.qcut(overlap, 4, labels=["overlap_q1", "overlap_q2", "overlap_q3", "overlap_q4"], duplicates="drop").to_numpy(),
    }, index=valid_indices)
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
            risk_rows.append({"risk": task, "method": method, "calibration_rows": len(truth_raw), "offset_log1p": offset,
                              "empirical_coverage": float(np.mean(truth_raw <= upper)), "mean_interval_width": float(np.mean(width)),
                              "median_interval_width": float(np.median(width)), "mae": float(np.mean(np.abs(median_raw - truth_raw))),
                              "median_absolute_error": float(np.median(np.abs(median_raw - truth_raw))),
                              "spearman": float(spearmanr(median_raw, truth_raw).statistic), **score_structure(upper)})
            for family in ("snr_bin", "overlap_bin"):
                for level in subgroup[family].dropna().unique():
                    mask = subgroup[family].to_numpy() == level
                    subgroup_rows.append({"risk": task, "method": method, "subgroup_family": family, "subgroup": str(level),
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
        logits = binary_logits(run, features, task)[valid.to_numpy()]
        temperature = fit_binary_temperature(logits, truth)
        probability = sigmoid(logits / temperature)
        binary_rows.append({"task": task, "calibration_rows": len(truth), "temperature": temperature,
                            "prevalence": float(truth.mean()), "auroc": auroc(probability, truth), "auprc": auprc(probability, truth),
                            "brier": float(np.mean((probability - truth) ** 2)), **score_structure(probability)})
    write_csv_fresh(run / "tables/binary_risk_calibration_summary.csv", pd.DataFrame(binary_rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task in zip(axes, ("image", "flux", "centroid")):
        truth = risk_predictions[f"{task}_truth"].to_numpy(); pred = risk_predictions[f"{task}_median"].to_numpy()
        keep = np.argsort(truth)[::max(1, len(truth) // 1000)]
        ax.scatter(truth[keep], pred[keep], s=5, alpha=0.35); ax.set_xscale("symlog", linthresh=1.0); ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(task); ax.set_xlabel("natural calibration risk"); ax.set_ylabel("frozen median prediction")
    fig.tight_layout(); fig.savefig(run / "figures/natural_calibration_transfer.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ("image", "flux", "centroid")):
        rows = risk_frame[risk_frame.risk == task]
        ax.bar(rows.method, rows.empirical_coverage); ax.axhline(0.9, color="black", linestyle="--")
        ax.set_ylim(0, 1); ax.tick_params(axis="x", labelrotation=25); ax.set_title(task)
    fig.tight_layout(); fig.savefig(run / "figures/calibration_transfer_coverage.png", dpi=180); plt.close(fig)
    write_json_fresh(run / "logs/calibration_feasibility_complete.json", {
        "status": "PASS_WITH_APPEND_ONLY_DTYPE_CORRECTION", "runtime_seconds": time.time() - started,
        "heads_frozen_before_calibration": True, "query_artifacts_recomputed": False, "operational_policy_selected": False,
        "calibration_used_for_model_selection": False, "development_accessed": False, "lockbox_accessed": False,
    })
    print(json.dumps({"risk_coverage": risk_frame.groupby("risk").empirical_coverage.mean().to_dict(),
                      "binary_transfer": {row["task"]: {"auroc": row["auroc"], "auprc": row["auprc"]} for row in binary_rows}}, sort_keys=True))


if __name__ == "__main__":
    main()
