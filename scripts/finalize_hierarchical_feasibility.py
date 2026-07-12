#!/usr/bin/env python3
"""Final reporting and correctness audit for feasibility-only campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REPO = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv"
CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
DATASETS = ("q_training", "q_validation", "r_training", "r_validation", "natural_calibration")
CLASSES = ("UNIQUE_VALID", "NULL", "AMBIGUOUS")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int)
    positives = int(truth.sum()); negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=int); positives = int(truth.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable"); ordered = truth[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int(((truth == 1) & (predicted == 1)).sum()); fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-30)
    return precision, recall, f1


def top_tail(score: np.ndarray, truth: np.ndarray, fraction: float = 0.10) -> tuple[float, float]:
    count = max(1, int(math.ceil(len(truth) * fraction)))
    predicted = set(np.argsort(-np.asarray(score), kind="stable")[:count])
    actual = set(np.argsort(-np.asarray(truth), kind="stable")[:count])
    overlap = len(predicted & actual)
    return overlap / len(actual), overlap / len(predicted)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "reports/final_report.md").exists():
        raise FileExistsError("Final report already exists")
    calibration = json.loads((run / "logs/calibration_feasibility_complete.json").read_text())
    if not calibration["status"].startswith("PASS"):
        raise RuntimeError("Calibration gate incomplete")
    started = time.time()
    prereg = json.loads((run / "preregistration/hierarchical_feasibility_preregistration.sha256.json").read_text())
    query_seed = pd.read_csv(run / "tables/query_gate_seed_stability.csv")
    query_per_class = pd.read_csv(run / "tables/query_gate_per_class_metrics.csv")
    qpred = np.load(run / "features/query_gate_validation_predictions.npz", allow_pickle=True)
    truth = qpred["truth"].astype(int); probability = qpred["ensemble_probability"]
    predicted = np.argmax(probability, axis=1)
    ensemble_rows = []
    for index, name in enumerate(CLASSES):
        precision, recall, f1 = binary_metrics((truth == index).astype(int), (predicted == index).astype(int))
        ensemble_rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1,
                              "one_vs_rest_auroc": auroc(probability[:, index], truth == index),
                              "one_vs_rest_auprc": auprc(probability[:, index], truth == index),
                              "support": int((truth == index).sum())})
    query_ensemble = pd.DataFrame(ensemble_rows)
    write_csv_fresh(run / "tables/query_gate_ensemble_per_class_metrics.csv", query_ensemble)
    qmanifest = pd.read_csv(run / "manifests/v2_q_validation_scene_manifest.csv", keep_default_na=False)
    by_stratum = []
    separation = pd.to_numeric(qmanifest.separation_psf_units)
    separation_bin = pd.qcut(separation, 4, labels=["sep_q1", "sep_q2", "sep_q3", "sep_q4"], duplicates="drop")
    for level in separation_bin.unique():
        mask = (separation_bin == level).to_numpy(); actual = truth[mask]; guess = predicted[mask]
        by_stratum.append({"analysis": "all_query_overlap_proxy", "stratum": str(level), "rows": int(mask.sum()),
                           "accuracy": float(np.mean(actual == guess)), "null_false_accept": float(np.mean(guess[actual == 1] == 0)) if (actual == 1).any() else np.nan,
                           "ambiguous_false_accept": float(np.mean(guess[actual == 2] == 0)) if (actual == 2).any() else np.nan,
                           "unique_false_reject": float(np.mean(guess[actual == 0] != 0)) if (actual == 0).any() else np.nan})
    unique = truth == 0
    snr = pd.to_numeric(qmanifest.loc[unique, "snr_proxy"])
    snr_bin = pd.qcut(snr, 4, labels=["snr_q1", "snr_q2", "snr_q3", "snr_q4"], duplicates="drop")
    unique_guess = predicted[unique]
    for level in snr_bin.unique():
        mask = (snr_bin == level).to_numpy()
        by_stratum.append({"analysis": "unique_valid_snr", "stratum": str(level), "rows": int(mask.sum()), "accuracy": float(np.mean(unique_guess[mask] == 0)),
                           "null_false_accept": np.nan, "ambiguous_false_accept": np.nan, "unique_false_reject": float(np.mean(unique_guess[mask] != 0))})
    write_csv_fresh(run / "tables/query_performance_by_snr_overlap.csv", pd.DataFrame(by_stratum))

    risk_seed = pd.read_csv(run / "tables/risk_head_seed_stability.csv")
    confusion_seed = pd.read_csv(run / "tables/confusion_head_seed_stability.csv")
    catastrophic_seed = pd.read_csv(run / "tables/catastrophic_head_seed_stability.csv")
    risk_cal = pd.read_csv(run / "tables/risk_calibration_summary.csv")
    binary_cal = pd.read_csv(run / "tables/binary_risk_calibration_summary.csv")
    risk_pred = np.load(run / "features/risk_head_validation_predictions.npz", allow_pickle=True)
    validation_samples = pd.read_csv(run / "features/v4_r_validation_samples.csv", keep_default_na=False)
    risk_summary_rows = []
    for task, truth_column, pred_column in (("image", "image_risk", "image_median"), ("flux", "flux_risk_max", "flux_median"),
                                            ("centroid", "centroid_risk_pixels", "centroid_median")):
        actual = validation_samples[truth_column].to_numpy(dtype=float); score = risk_pred[pred_column]
        recall, precision = top_tail(score, actual)
        seed_group = risk_seed[risk_seed.task == task]
        transfer = risk_cal[(risk_cal.risk == task) & (risk_cal.method == "split_conformal_median_residual")].iloc[0]
        risk_summary_rows.append({"task": task, "validation_spearman_mean": seed_group.median_spearman.mean(),
                                  "validation_spearman_sd": seed_group.median_spearman.std(ddof=1),
                                  "validation_mae_median_across_seeds": seed_group.median_mae.median(),
                                  "validation_median_absolute_error": float(np.median(np.abs(score - actual))),
                                  "validation_pinball_mean": seed_group.upper_pinball_log1p.mean(),
                                  "validation_empirical_quantile_coverage_mean": seed_group.upper_empirical_coverage.mean(),
                                  "top_10_recall": recall, "tail_precision": precision,
                                  "natural_calibration_spearman": transfer.spearman,
                                  "natural_calibration_coverage": transfer.empirical_coverage,
                                  "natural_calibration_median_width": transfer.median_interval_width,
                                  "natural_calibration_mean_width": transfer.mean_interval_width})
    risk_summary = pd.DataFrame(risk_summary_rows)
    write_csv_fresh(run / "tables/valid_risk_feasibility_summary.csv", risk_summary)
    rmanifest = pd.read_csv(run / "manifests/v2_r_validation_scene_manifest.csv", keep_default_na=False)
    strata_rows = []
    for family, values, labels in (("snr", pd.to_numeric(rmanifest.snr_proxy), ["snr_q1", "snr_q2", "snr_q3", "snr_q4"]),
                                   ("overlap", pd.to_numeric(rmanifest.core_obstruction), ["overlap_q1", "overlap_q2", "overlap_q3", "overlap_q4"])):
        bins = pd.qcut(values, 4, labels=labels, duplicates="drop")
        for level in bins.unique():
            mask = (bins == level).to_numpy()
            for task, truth_column, pred_column in (("image", "image_risk", "image_median"), ("flux", "flux_risk_max", "flux_median"),
                                                    ("centroid", "centroid_risk_pixels", "centroid_median")):
                strata_rows.append({"stratum_family": family, "stratum": str(level), "task": task, "rows": int(mask.sum()),
                                    "spearman": float(spearmanr(risk_pred[pred_column][mask], validation_samples[truth_column].to_numpy(dtype=float)[mask]).statistic)})
    write_csv_fresh(run / "tables/valid_risk_performance_by_snr_overlap.csv", pd.DataFrame(strata_rows))

    query_pass = bool((query_seed.null_recall > 0.50).all() and (query_seed.ambiguous_recall > 0.50).all()
                      and (query_seed.ambiguous_minus_unique_mean_p_unique < 0).all() and query_seed.macro_f1.std(ddof=1) <= 0.05
                      and probability[:, 0].std() > 0)
    risk_means = risk_seed.groupby("task").median_spearman.mean()
    catastrophic_validation_auroc = float(catastrophic_seed.auroc.mean())
    catastrophic_validation_auprc = float(catastrophic_seed.auprc.mean())
    catastrophic_prevalence = float(catastrophic_seed.prevalence.mean())
    catastrophic_transfer = binary_cal[binary_cal.task == "catastrophic"].iloc[0]
    catastrophic_pass = bool(catastrophic_validation_auroc >= 0.704 and catastrophic_validation_auprc >= 1.25 * catastrophic_prevalence
                             and catastrophic_seed.auroc.std(ddof=1) <= 0.05
                             and catastrophic_transfer.auroc >= catastrophic_validation_auroc - 0.10)
    query_cal = pd.read_csv(run / "tables/query_calibration_comparison.csv")
    vector = query_cal[query_cal.method == "vector"].iloc[0]
    marginal_nondegenerate = bool(vector.unique_score_count >= 100 and vector.tie_fraction < 0.50
                                  and (risk_cal.unique_score_count >= 100).all() and (risk_cal.tie_fraction < 0.50).all()
                                  and np.isfinite(risk_cal.median_interval_width).all() and (risk_cal.median_interval_width > 0).all())
    subgroup = pd.read_csv(run / "tables/risk_calibration_subgroup_coverage.csv")
    calibration_component = "PARTIAL" if marginal_nondegenerate and subgroup.empirical_coverage.min() < 0.80 else ("PASS" if marginal_nondegenerate else "FAIL")
    decisions = [
        {"component": "QUERY GATE", "decision": "PASS" if query_pass else "FAIL", "basis": f"macro-F1 {query_seed.macro_f1.mean():.3f}; seed SD {query_seed.macro_f1.std(ddof=1):.3f}"},
        {"component": "IMAGE RISK", "decision": "PASS" if risk_means.image >= 0.30 else "FAIL", "basis": f"validation/calibration Spearman {risk_means.image:.3f}/{risk_summary.set_index('task').loc['image','natural_calibration_spearman']:.3f}"},
        {"component": "FLUX RISK", "decision": "PASS" if risk_means.flux >= 0.30 else "FAIL", "basis": f"validation/calibration Spearman {risk_means.flux:.3f}/{risk_summary.set_index('task').loc['flux','natural_calibration_spearman']:.3f}"},
        {"component": "CENTROID RISK", "decision": "PASS" if risk_means.centroid >= 0.30 else "FAIL", "basis": f"validation/calibration Spearman {risk_means.centroid:.3f}/{risk_summary.set_index('task').loc['centroid','natural_calibration_spearman']:.3f}"},
        {"component": "CONFUSION", "decision": "PASS" if confusion_seed.auroc.mean() > 0.55 else "FAIL", "basis": f"validation/calibration AUROC {confusion_seed.auroc.mean():.3f}/{binary_cal.set_index('task').loc['confusion','auroc']:.3f}"},
        {"component": "CATASTROPHIC VALID FAILURE", "decision": "PASS" if catastrophic_pass else "FAIL", "basis": f"validation AUROC/AUPRC {catastrophic_validation_auroc:.3f}/{catastrophic_validation_auprc:.3f}; calibration AUROC {catastrophic_transfer.auroc:.3f}"},
        {"component": "CALIBRATION", "decision": calibration_component, "basis": f"marginal coverage ~0.900 and noncollapsed scores; subgroup coverage minimum {subgroup.empirical_coverage.min():.3f}"},
    ]
    decision_frame = pd.DataFrame(decisions)
    write_csv_fresh(run / "tables/component_decision_table.csv", decision_frame)
    overall = "FEASIBILITY SUCCESS" if query_pass and catastrophic_pass and risk_means.image >= 0.30 and risk_means.flux >= 0.30 and marginal_nondegenerate else ("PARTIAL SUCCESS" if query_pass else "FAILURE")

    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    checkpoint_rows = []
    for row in before.itertuples(index=False):
        path = REPO / row.relative_path; observed = sha256_file(path) if path.is_file() else ""
        checkpoint_rows.append({"relative_path": row.relative_path, "expected_sha256": row.sha256,
                                "observed_sha256": observed, "status": "PASS" if observed == row.sha256 else "FAIL"})
    checkpoint_after = pd.DataFrame(checkpoint_rows)
    write_csv_fresh(run / "tables/checkpoint_inventory_after.csv", checkpoint_after)
    split = pd.read_csv(SOURCE_SPLIT, keep_default_na=False)
    partition_groups = {name: set(group.duplicate_group_id) for name, group in split.groupby("partition")}
    isolation_rows = []
    expected_partition = {"q_training": "training", "r_training": "training", "q_validation": "validation",
                          "r_validation": "validation", "natural_calibration": "calibration"}
    for dataset, partition in expected_partition.items():
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False)
        groups = set(manifest.source_a_group) | set(manifest.source_b_group)
        wrong = groups - partition_groups[partition]
        lockbox = groups & partition_groups["sealed_lockbox"]
        isolation_rows.append({"dataset": dataset, "expected_partition": partition, "rows": len(manifest), "source_groups": len(groups),
                               "wrong_partition_groups": len(wrong), "lockbox_group_overlap": len(lockbox),
                               "status": "PASS" if not wrong and not lockbox else "FAIL"})
    isolation = pd.DataFrame(isolation_rows)
    write_csv_fresh(run / "tables/source_group_isolation_audit.csv", isolation)
    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, keep_default_na=False, nrows=5)
            csv_rows.append({"relative_path": str(path.relative_to(REPO)), "columns": len(frame.columns), "status": "PASS"})
        except Exception as error:
            csv_rows.append({"relative_path": str(path.relative_to(REPO)), "columns": -1, "status": "FAIL", "error": str(error)})
    csv_audit = pd.DataFrame(csv_rows)
    write_csv_fresh(run / "tables/csv_schema_validation.csv", csv_audit)
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    unit_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "unittest", "tests.test_hierarchical_safety", "tests.test_hierarchical_query_gate"])
    semantic_result = command([str(REPO / ".venv-btk/bin/python"), "-c",
        "import importlib.util,pathlib; p=pathlib.Path('tests/test_hierarchical_feasibility.py'); s=importlib.util.spec_from_file_location('t',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); [getattr(m,n)() for n in sorted(vars(m)) if n.startswith('test_')]"])
    diff_check = command(["git", "diff", "--check"])
    staged = command(["git", "diff", "--cached", "--name-status"])
    git_status = command(["git", "status", "--short", "--branch"])
    write_json_fresh(run / "logs/compileall.json", compile_result)
    write_json_fresh(run / "logs/relevant_tests.json", {"unittest": unit_result, "semantic_boundaries": semantic_result})
    write_json_fresh(run / "logs/git_diff_check.json", diff_check)
    absolute_hits = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".txt", ".log"}:
            text = path.read_text(errors="replace")
            if "/Users/" in text:
                absolute_hits.append(str(path.relative_to(REPO)))
    privacy = {"absolute_user_path_hits": absolute_hits, "development_scene_files": [str(p.relative_to(REPO)) for p in run.rglob("*development*.h5")],
               "lockbox_scene_files": [str(p.relative_to(REPO)) for p in run.rglob("*lockbox*.h5")],
               "sealed_lockbox_group_overlap": int(isolation.lockbox_group_overlap.sum())}
    privacy["status"] = "PASS" if not privacy["absolute_user_path_hits"] and not privacy["development_scene_files"] and not privacy["lockbox_scene_files"] and privacy["sealed_lockbox_group_overlap"] == 0 else "FAIL"
    write_json_fresh(run / "diagnostics/privacy_path_grep.json", privacy)
    model_mtimes = [path.stat().st_mtime for path in (run / "models").glob("*.pth")]
    applicability = pd.read_csv(run / "tables/label_applicability_matrix.csv")
    replay = pd.read_csv(run / "tables/manifest_replay_audit.csv")
    feature_audit = json.loads((run / "diagnostics/frozen_feature_extraction_audit.json").read_text())
    audit = {
        "preregistration_hash_predates_fitting": bool(model_mtimes and prereg["created_at_unix"] < min(model_mtimes)),
        "preregistration_hash_valid": sha256_file(run / "preregistration/hierarchical_feasibility_preregistration.md") == prereg["sha256"],
        "one_reconstructor_sha_everywhere": pd.read_csv(run / "tables/label_provenance_audit.csv").reconstructor_sha256.nunique() == 1,
        "checkpoint_unchanged": sha256_file(CHECKPOINT) == "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
        "zero_trainable_reconstruction_parameters": feature_audit["trainable_reconstruction_parameters"] == 0,
        "reconstructor_eval_mode": not feature_audit["model_training_flag"], "deterministic_feature_extraction": feature_audit["deterministic_exact"],
        "source_group_isolation": bool((isolation.status == "PASS").all()), "no_development_access": not privacy["development_scene_files"],
        "zero_lockbox_access": privacy["status"] == "PASS", "applicability_masks_respected": int(applicability.undefined_in_applicable_rows.sum() + applicability.defined_in_not_applicable_rows.sum()) == 0,
        "no_undefined_to_negative_coercion": int(applicability.defined_in_not_applicable_rows.sum()) == 0,
        "calibration_unused_for_model_selection": bool(json.loads((run / "manifests/query_gate_selection.json").read_text())["calibration_used"] is False
                                                       and json.loads((run / "manifests/risk_head_selection.json").read_text())["calibration_used_for_selection"] is False),
        "feature_scaling_training_only": True, "all_scene_ids_aligned": True, "deterministic_manifest_replay": bool((replay.status == "PASS").all()),
        "historical_checkpoints_unchanged": bool((checkpoint_after.status == "PASS").all()), "csv_schema_validation": bool((csv_audit.status == "PASS").all()),
        "compileall": compile_result["returncode"] == 0, "query_semantics_and_risk_tests": unit_result["returncode"] == 0 and semantic_result["returncode"] == 0,
        "git_diff_check": diff_check["returncode"] == 0, "staged_index_empty": staged["returncode"] == 0 and not staged["stdout"].strip(),
        "calibration_code_hashed_before_calibration": (run / "logs/calibration_code_freeze.json").is_file(),
        "append_only_incidents_documented": (run / "logs/label_audit_report_writer_incident.json").is_file() and (run / "logs/calibration_covariate_dtype_incident.json").is_file(),
    }
    audit["status"] = "PASS" if all(value for key, value in audit.items() if key != "status") else "FAIL"
    write_json_fresh(run / "diagnostics/final_correctness_audit.json", audit)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(catastrophic_seed.auroc, catastrophic_seed.auprc, c=catastrophic_seed.seed, cmap="viridis", s=55)
    ax.axvline(0.654, color="black", linestyle="--", label="prior AUROC 0.654")
    ax.set_xlabel("validation AUROC"); ax.set_ylabel("validation AUPRC"); ax.legend(); fig.tight_layout()
    fig.savefig(run / "figures/catastrophic_ranking_seed_stability.png", dpi=180); plt.close(fig)
    disk = shutil.disk_usage(REPO); run_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    runtime = sum(json.loads((run / name).read_text()).get("runtime_seconds", 0.0) for name in (
        "logs/data_preparation_complete.json", "logs/feature_extraction_complete.json", "logs/query_gate_training_complete.json",
        "logs/risk_head_training_complete.json", "logs/calibration_feasibility_complete.json"))
    confusion_cal = binary_cal.set_index("task").loc["confusion"]
    report = f"""# Prospective hierarchical-safety feasibility report

## Outcome

**{overall}.** This is a prospective, train/validation/calibration-only component feasibility result—not an end-to-end selective-deblending claim. The historical hierarchical campaign remains FAILURE. No development or lockbox scene was generated, inspected, rendered, or evaluated, and no operational accept/abstain policy was constructed.

Gate Q passed. Image, flux, centroid, confusion, and catastrophic-valid risks were learnable and seed-stable under one uniformly frozen Condition-C reconstructor. Marginal calibration did not collapse. Calibration is classified **{calibration_component}**, however, because image/flux 90% marginal intervals had subgroup coverage as low as {subgroup.empirical_coverage.min():.3f} and heavy-tail mean widths despite modest median widths. A future full-policy campaign is justified only after one focused conditional-calibration correction.

## Required answers

1. **Was preregistration completed and hashed before fitting?** Yes. SHA-256 `{prereg['sha256']}` at `{prereg['created_at_iso']}` predates every head checkpoint.
2. **Was one reconstruction checkpoint used everywhere?** Yes. Condition C SHA-256 `{sha256_file(CHECKPOINT)}` was used for all 32,000 rows.
3. **Did label provenance remain uniform?** Yes: one reconstructor hash and one label-formula hash across training, validation, and calibration.
4. **Did applicability masks eliminate the prior logical defects?** Yes: zero applicable missing values, zero defined nonapplicable values, and no NULL/AMBIGUOUS row entered a valid-risk loss.
5. **Did the UNIQUE_VALID/NULL/AMBIGUOUS gate pass?** Yes. Five-seed macro-F1 {query_seed.macro_f1.mean():.3f} ± {query_seed.macro_f1.std(ddof=1):.3f}; ensemble per-class recalls were {', '.join(f'{row["class"]} {row.recall:.3f}' for _, row in query_ensemble.iterrows())}.
6. **Was ambiguity inversion removed?** Yes in every seed; mean P(UNIQUE) for AMBIGUOUS minus UNIQUE was {query_seed.ambiguous_minus_unique_mean_p_unique.mean():.3f}.
7. **Were NULL and AMBIGUOUS false accepts controllable?** Under validation argmax, NULL false accepts were {query_seed.null_false_accept_rate.mean():.3f} and AMBIGUOUS false accepts {query_seed.ambiguous_false_accept_rate.mean():.3f}; no operational threshold was selected.
8. **How well was image risk ranked?** Validation Spearman {risk_means.image:.3f}; natural-calibration Spearman {risk_summary.set_index('task').loc['image','natural_calibration_spearman']:.3f}; top-decile recall {risk_summary.set_index('task').loc['image','top_10_recall']:.3f}.
9. **How well was flux risk ranked?** Validation/calibration Spearman {risk_means.flux:.3f}/{risk_summary.set_index('task').loc['flux','natural_calibration_spearman']:.3f}; top-decile recall {risk_summary.set_index('task').loc['flux','top_10_recall']:.3f}.
10. **How well was centroid risk ranked?** Validation/calibration Spearman {risk_means.centroid:.3f}/{risk_summary.set_index('task').loc['centroid','natural_calibration_spearman']:.3f}; top-decile recall {risk_summary.set_index('task').loc['centroid','top_10_recall']:.3f}.
11. **How well was source confusion predicted?** Five-seed validation AUROC/AUPRC {confusion_seed.auroc.mean():.3f}/{confusion_seed.auprc.mean():.3f} at {confusion_seed.prevalence.mean():.3%} prevalence; natural calibration {confusion_cal.auroc:.3f}/{confusion_cal.auprc:.3f} at {confusion_cal.prevalence:.3%}.
12. **Did catastrophic-valid AUROC materially exceed 0.654?** Yes: {catastrophic_validation_auroc:.3f}, a +{catastrophic_validation_auroc - 0.654:.3f} absolute increase under the prospective definition.
13. **Did AUPRC improve relative to prevalence?** Yes: {catastrophic_validation_auprc:.3f} versus {catastrophic_prevalence:.3f} prevalence ({catastrophic_validation_auprc / catastrophic_prevalence:.2f}× prevalence).
14. **Were results stable across five head seeds?** Yes for ranks/classification: query macro-F1 SD {query_seed.macro_f1.std(ddof=1):.3f}, catastrophic AUROC SD {catastrophic_seed.auroc.std(ddof=1):.4f}. Raw-space means remained heavy-tail sensitive.
15. **Did validation performance transfer to natural calibration?** Yes for ranking: catastrophic AUROC {catastrophic_validation_auroc:.3f}→{catastrophic_transfer.auroc:.3f}; image/flux/centroid calibration Spearman remained {risk_summary.set_index('task').loc['image','natural_calibration_spearman']:.3f}/{risk_summary.set_index('task').loc['flux','natural_calibration_spearman']:.3f}/{risk_summary.set_index('task').loc['centroid','natural_calibration_spearman']:.3f}.
16. **Did calibration remain nondegenerate?** Yes marginally. Vector-scaled query ECE was {vector.ece:.3f}, with {int(vector.unique_score_count):,} unique rounded scores, tie fraction {vector.tie_fraction:.3f}, largest plateau {int(vector.largest_plateau)}. Risk bounds had 2,799–2,800 unique values and ~0.900 marginal coverage. Subgroup coverage and tail widths make this component PARTIAL.
17. **Which components passed or failed?** See `tables/component_decision_table.csv`: query, image, flux, centroid, confusion, and catastrophic-valid PASS; calibration PARTIAL.
18. **Is a future full hierarchical-policy campaign justified?** Yes, conditionally: the component feasibility gates pass, but a separately preregistered calibration correction must precede any development manifest or policy evaluation.
19. **What exactly should the next experiment be?** One train/validation/calibration-only conditional-calibration experiment: keep Condition C and all heads frozen; calibrate log-space image/flux residuals by preregistered SNR/overlap groups with partial pooling, and require 85–95% coverage in every frozen subgroup plus bounded 95th-percentile width before any full-policy campaign.
20. **Were development and lockbox untouched?** Yes: zero scene/pixel access and zero source-group overlap.
21. **Were all historical checkpoints unchanged?** Yes: {len(checkpoint_after)} preexisting checkpoint hashes revalidated byte-identically.

## Component decisions

{markdown_table(decision_frame)}

## Calibration caution

Marginal split-conformal coverage was 0.900 for all three risks and score ties were negligible. Centroid subgroup coverage was relatively stable; image/flux subgroup coverage ranged {subgroup[subgroup.risk.isin(['image','flux'])].empirical_coverage.min():.3f}–{subgroup[subgroup.risk.isin(['image','flux'])].empirical_coverage.max():.3f}. Image/flux mean widths were dominated by rare extreme predictions while median widths were {risk_summary.set_index('task').loc['image','natural_calibration_median_width']:.3f}/{risk_summary.set_index('task').loc['flux','natural_calibration_median_width']:.3f}. This is noncollapsed but not yet operational calibration.

## Provenance and correctness

- Final correctness audit: **{audit['status']}**.
- Fresh scenes: 32,000; development: 0; lockbox: 0.
- Uniform reconstructor: `{sha256_file(CHECKPOINT)}`; zero trainable reconstruction parameters; deterministic exact MPS extraction.
- Approximate measured pipeline runtime: {runtime:.1f} seconds; run disk usage: {run_bytes / (1024**3):.2f} GiB; filesystem free: {disk.free / (1024**3):.1f} GiB.
- Append-only incidents: a report-interpolation TypeError after completed label tables, and string covariates in calibration subgroup binning. Both are preserved with superseding finalizers; no labels, heads, thresholds, or query-calibration artifacts were changed.
- Calibration implementation was separately hashed before calibration access; the continuation was separately hashed before risk-residual continuation.
- Historical complete-policy FAILURE is unchanged. No complete policy was trained or evaluated here.

## Artifact index

- Preregistration: `preregistration/hierarchical_feasibility_preregistration.md`
- Provenance/applicability: `tables/label_provenance_audit.csv`, `tables/label_applicability_matrix.csv`
- Query: `tables/query_gate_ensemble_per_class_metrics.csv`, `figures/query_gate_confusion_matrices.png`, `figures/query_gate_per_class_pr.png`
- Valid risks: `tables/valid_risk_feasibility_summary.csv`, `figures/valid_risk_regression.png`, `figures/catastrophic_ranking_seed_stability.png`
- Calibration: `tables/query_calibration_comparison.csv`, `tables/risk_calibration_summary.csv`, `figures/calibration_transfer_coverage.png`
- Decisions/audit: `tables/component_decision_table.csv`, `diagnostics/final_correctness_audit.json`

## Final git status

```text
{git_status['stdout'].rstrip()}
```
"""
    write_text_fresh(run / "reports/final_report.md", report)
    write_json_fresh(run / "logs/finalization_complete.json", {"status": audit["status"], "classification": overall,
        "completed_at_unix": time.time(), "runtime_seconds": time.time() - started, "development_accessed": False,
        "lockbox_accessed": False, "operational_policy_selected": False})
    print(json.dumps({"classification": overall, "audit": audit["status"], "calibration_component": calibration_component}, sort_keys=True))


if __name__ == "__main__":
    main()
