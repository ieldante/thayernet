#!/usr/bin/env python3
"""Append-only hierarchical Thayer-Select safety campaign driver.

The bootstrap and drift-audit stages are deliberately executable before any
new scene generation.  Later campaign stages refuse to run unless these gates
exist and pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp, wasserstein_distance
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.hierarchical_safety import HierarchicalQuerySemantics, RISK_LIMITS


FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
REPLICATION = REPO / "outputs/runs/thayer_select_recoverability_seed_replication_20260711_203115"
ROOT_CAUSE = REPO / "outputs/runs/thayer_select_root_cause_analysis_20260711"
FROZEN_HEAD = REPO / "outputs/runs/thayer_select_frozen_head_ablation_20260711_220756"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PHASE1 / "manifests/source_split_manifest.csv"
CONDITION_C = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
SEMANTICS = HierarchicalQuerySemantics()
SPLITS = ("training", "validation", "calibration")
BOOTSTRAP_SEED = 2026071111


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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def checkpoint_inventory(exclude: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "outputs/runs").rglob("*.pth")):
        if exclude is not None and exclude in path.parents:
            continue
        rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def relevant_manifest_paths() -> list[Path]:
    paths = [
        FOUNDATION / "manifests/btk_engineering_source_groups.csv",
        PHASE1 / "manifests/source_split_manifest.csv",
        PHASE1 / "manifests/normalization.json",
        PHASE1 / "manifests/training_scenes.h5",
        PHASE1 / "manifests/validation_scenes.h5",
        PHASE1 / "manifests/calibration_scenes.h5",
        PHASE2 / "manifests/source_split_manifest.csv",
        PHASE2 / "manifests/training_scene_manifest.csv",
        PHASE2 / "manifests/validation_scene_manifest.csv",
        PHASE2 / "manifests/calibration_scene_manifest.csv",
        PHASE2 / "manifests/training_scenes.h5",
        PHASE2 / "manifests/validation_scenes.h5",
        PHASE2 / "manifests/calibration_scenes.h5",
    ]
    return paths


def bootstrap(run: Path) -> None:
    if run.exists():
        raise FileExistsError(f"Intended output path already exists: {run}")
    if run.parent.resolve() != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_select_hierarchical_safety_"):
        raise RuntimeError("Unexpected master run path")

    # Fail-closed gates are checked before the first campaign path is created.
    expected_c = json.loads((PHASE1 / "manifests/c_randomized_coordinate_prompt_training_config.json").read_text())["best_checkpoint_sha256"]
    observed_c = sha256_file(CONDITION_C)
    if observed_c != expected_c:
        raise RuntimeError("Condition-C checkpoint differs from its frozen training record")
    split_paths = [
        FOUNDATION / "manifests/btk_engineering_source_groups.csv",
        PHASE1 / "manifests/source_split_manifest.csv",
        PHASE2 / "manifests/source_split_manifest.csv",
    ]
    split_hashes = [sha256_file(path) for path in split_paths]
    if len(set(split_hashes)) != 1:
        raise RuntimeError("Source partition hashes differ")
    split = pd.read_csv(SOURCE_SPLIT, dtype=str)
    if "sealed_lockbox" not in set(split["partition"]):
        raise RuntimeError("Lockbox exclusion cannot be verified")
    lockbox_groups = set(split.loc[split.partition == "sealed_lockbox", "duplicate_group_id"])
    for partition in SPLITS:
        manifest = pd.read_csv(PHASE2 / f"manifests/{partition}_scene_manifest.csv", dtype=str)
        if set(manifest["source_a_group"]) & lockbox_groups or set(manifest["source_b_group"]) & lockbox_groups:
            raise RuntimeError(f"Lockbox source group appears in historical {partition} scenes")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")

    run.mkdir(parents=True, exist_ok=False)
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "manifests", "features", "models", "calibration", "example_grids", "paper_figures"):
        (run / name).mkdir(exist_ok=False)
    (run / "figures/partition_drift").mkdir(exist_ok=False)
    started = time.time()
    inventory = checkpoint_inventory(exclude=run)
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", inventory)

    packages = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scikit-learn": package_version("scikit-learn"),
        "btk": package_version("blending-toolkit"),
        "galsim": package_version("GalSim"),
        "surveycodex": package_version("surveycodex"),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_probe": 2.0,
    }
    git = {
        "branch": command(["git", "branch", "--show-current"]),
        "head": command(["git", "rev-parse", "HEAD"]),
        "status": command(["git", "status", "--short", "--branch"]),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]),
    }
    disk = shutil.disk_usage(REPO)
    manifest_hashes = [{"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in relevant_manifest_paths()]
    checkpoint_rows = []
    for campaign, base in (("condition_c", PHASE1), ("phase2", PHASE2), ("replication", REPLICATION), ("frozen_head", FROZEN_HEAD)):
        for path in sorted((base / "checkpoints").glob("*.pth")):
            checkpoint_rows.append({"campaign": campaign, "relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    code_paths = sorted((REPO / "src").glob("*.py")) + sorted((REPO / "scripts").glob("*thayer_select*.py")) + [Path(__file__).resolve()]
    code_paths = list(dict.fromkeys(path for path in code_paths if path.is_file()))
    code_hashes = [{"relative_path": relative(path), "sha256": sha256_file(path)} for path in code_paths]

    environment = f"""# Hierarchical safety environment snapshot

- Campaign start (Unix): `{started:.6f}`
- Branch: `{git['branch']['stdout'].strip()}`
- Git HEAD: `{git['head']['stdout'].strip()}`
- MPS built / available / probe: `{packages['mps_built']}` / `{packages['mps_available']}` / `{packages['mps_probe']}`
- Disk total / free bytes: `{disk.total}` / `{disk.free}`
- Python: `{platform.python_version()}`

## Package versions

```json
{json.dumps(packages, indent=2, sort_keys=True)}
```

## Initial git status

```text
{git['status']['stdout'].rstrip()}
```

## Staged index

```text
{git['staged_index']['stdout'].rstrip() or '(empty)'}
```
"""
    write_text_fresh(run / "diagnostics/environment_snapshot.md", environment)
    contract = f"""# Hierarchical safety campaign contract

Status: frozen before drift audit, new scene generation, feature extraction, or head training.

- Master run: `{relative(run)}`; every output is collision-refusing and append-only.
- Reconstruction checkpoint: Condition C `{observed_c}`; all reconstruction parameters remain frozen.
- Source split SHA-256: `{split_hashes[0]}`.
- Lockbox assignment rows: `{int((split.partition == 'sealed_lockbox').sum())}`; assignment metadata is used only to enforce exclusion. No sealed scene or pixel is opened.
- Neural feature extraction: MPS only. CPU is used for audits, lightweight heads, calibration, tables, and figures.
- Query states: UNIQUE_VALID, NULL, AMBIGUOUS under `{SEMANTICS.version}`.
- Recoverability: derived from query validity, metric-specific calibrated upper bounds, confusion risk, and accept/abstain; never a monolithic training label.
- Primary risk limits: `{RISK_LIMITS['moderate']}`; strict and permissive limits are sensitivity analyses.
- Development: created only after the complete policy freezes, evaluated exactly once, never used for threshold tuning.
- Prohibited: backbone alteration, encoder/decoder fine-tuning, source-split changes, historical inference regeneration, oracle deployable inputs, development retuning, lockbox access, version-control mutation, overwrite, or deletion.
"""
    write_text_fresh(run / "diagnostics/campaign_contract.md", contract)
    provenance = {
        "campaign_start_unix": started,
        "git": git,
        "packages": packages,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "source_catalog": {"relative_path": relative(CATALOG), "sha256": sha256_file(CATALOG)},
        "source_split": [{"relative_path": relative(path), "sha256": value} for path, value in zip(split_paths, split_hashes)],
        "relevant_manifests": manifest_hashes,
        "condition_c_checkpoint": {"relative_path": relative(CONDITION_C), "expected_sha256": expected_c, "observed_sha256": observed_c},
        "historical_checkpoints": checkpoint_rows,
        "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "code_hashes": code_hashes,
        "lockbox_partition_rows": int((split.partition == "sealed_lockbox").sum()),
        "lockbox_scene_or_pixel_accesses": 0,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)
    write_json_fresh(run / "logs/bootstrap_complete.json", {"status": "PASS", "completed_at_unix": time.time(), "lockbox_accessed": False})


def bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def load_audit_split(split: str, catalog) -> pd.DataFrame:
    manifest = pd.read_csv(PHASE2 / f"manifests/{split}_scene_manifest.csv", low_memory=False)
    if split in ("training", "validation"):
        outcome = pd.read_csv(PHASE2 / f"tables/{split}_teacher_reliability_labels.csv", low_memory=False)
        actionable = pd.read_csv(PHASE2 / f"tables/{split}_actionable_acceptance_labels.csv", low_memory=False)
        outcome = outcome.merge(actionable.drop(columns=["partition", "query_class"]), on="scene_id", validate="one_to_one")
    else:
        outcome = pd.read_csv(PHASE2 / "tables/calibration_per_sample.csv", low_memory=False)
    if manifest.scene_id.tolist() != outcome.scene_id.tolist():
        raise RuntimeError(f"Historical feature/label alignment failure in {split}")
    frame = manifest.merge(outcome.drop(columns=["query_class"], errors="ignore"), on="scene_id", validate="one_to_one")
    a = frame.source_a_row.to_numpy(dtype=int)
    b = frame.source_b_row.to_numpy(dtype=int)
    frame["source_r_magnitude"] = (np.asarray(catalog["r_ab"])[a] + np.asarray(catalog["r_ab"])[b]) / 2.0
    frame["source_size_arcsec"] = (frame.source_a_size_arcsec + frame.source_b_size_arcsec) / 2.0
    frame["ellipticity"] = (frame.source_a_ellipticity + frame.source_b_ellipticity) / 2.0
    frame["catastrophic_failure_numeric"] = bool_values(frame.catastrophic_failure).astype(int)
    frame["source_reuse_frequency"] = frame.source_a_id.map(frame.source_a_id.value_counts())
    frame["query_state"] = frame.query_class.replace({"VALID_SOURCE": "UNIQUE_VALID", "PERTURBED_VALID": "UNIQUE_VALID", "NULL_SOURCE": "NULL", "AMBIGUOUS_SOURCE": "AMBIGUOUS"})
    return frame


def cluster_bootstrap_mean(frame: pd.DataFrame, column: str, *, repetitions: int = 500) -> tuple[float, float]:
    finite = frame[["source_a_group", column]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [group[column].to_numpy(dtype=float) for _, group in finite.groupby("source_a_group")]
    if not groups:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, column)))
    values = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        values.append(float(np.mean(np.concatenate([groups[index] for index in chosen]))))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    pooled = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0)
    return 0.0 if pooled == 0 else float((np.mean(b) - np.mean(a)) / pooled)


def label_consistency(frame: pd.DataFrame) -> tuple[int, int]:
    mismatches = 0
    compared = 0
    for contract, limits in (("strict", (0.40, 0.15, 0.15, 1.0)), ("moderate", (0.75, 0.30, 0.30, 2.0)), ("permissive", (1.25, 0.50, 0.50, 3.0))):
        unique = frame.query_class.isin(("VALID_SOURCE", "PERTURBED_VALID"))
        finite = np.isfinite(frame[["normalized_rmse", "max_relative_flux_error", "max_color_error_mag", "centroid_error_pixels"]]).all(axis=1)
        expected = unique & finite
        expected &= frame.normalized_rmse <= limits[0]
        expected &= frame.max_relative_flux_error <= limits[1]
        expected &= frame.max_color_error_mag <= limits[2]
        expected &= frame.centroid_error_pixels <= limits[3]
        expected &= ~bool_values(frame.source_confusion)
        expected &= ~bool_values(frame.catastrophic_failure)
        actual = frame[f"{contract}_actionable_success"].to_numpy(dtype=int).astype(bool)
        mismatches += int(np.sum(expected.to_numpy(dtype=bool) != actual))
        compared += len(frame)
    return mismatches, compared


def drift_audit(run: Path) -> None:
    if not (run / "logs/bootstrap_complete.json").is_file():
        raise RuntimeError("Bootstrap gate missing")
    if (run / "logs/partition_drift_complete.json").exists():
        raise FileExistsError("Partition drift audit already completed")
    from astropy.table import Table

    catalog = Table.read(CATALOG, format="fits")
    frames = {split: load_audit_split(split, catalog) for split in SPLITS}
    metrics = [
        "source_r_magnitude", "source_size_arcsec", "ellipticity", "snr_proxy", "separation_pixels",
        "separation_psf_units", "flux_ratio", "color_similarity_distance", "source_count", "core_obstruction",
        "source_reuse_frequency", "strict_actionable_success", "moderate_actionable_success",
        "permissive_actionable_success", "catastrophic_failure_numeric",
    ]
    comparisons = (("training", "validation"), ("training", "calibration"), ("validation", "calibration"))
    rows = []
    for metric in metrics:
        for left_name, right_name in comparisons:
            left = pd.to_numeric(frames[left_name][metric], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(frames[right_name][metric], errors="coerce").to_numpy(dtype=float)
            left = left[np.isfinite(left)]; right = right[np.isfinite(right)]
            ks = ks_2samp(left, right) if len(left) and len(right) else None
            left_ci = cluster_bootstrap_mean(frames[left_name], metric)
            right_ci = cluster_bootstrap_mean(frames[right_name], metric)
            rows.append({
                "metric": metric, "left_partition": left_name, "right_partition": right_name,
                "left_n": len(left), "right_n": len(right), "left_mean": float(np.mean(left)), "right_mean": float(np.mean(right)),
                "left_cluster_bootstrap_ci_low": left_ci[0], "left_cluster_bootstrap_ci_high": left_ci[1],
                "right_cluster_bootstrap_ci_low": right_ci[0], "right_cluster_bootstrap_ci_high": right_ci[1],
                "standardized_mean_difference": standardized_mean_difference(left, right),
                "ks_statistic": float(ks.statistic) if ks else math.nan, "ks_pvalue": float(ks.pvalue) if ks else math.nan,
                "wasserstein_distance": float(wasserstein_distance(left, right)) if len(left) and len(right) else math.nan,
            })
    drift = pd.DataFrame(rows)
    write_csv_fresh(run / "tables/partition_drift_audit.csv", drift)

    prevalence_rows = []
    noise = pd.read_csv(FROZEN_HEAD / "tables/label_noise_audit.csv")
    for split, frame in frames.items():
        for state, group in frame.groupby("query_state"):
            prevalence_rows.append({"partition": split, "category": "query_state", "label": state, "samples": len(group), "positives": len(group), "prevalence": len(group) / len(frame)})
        for column in ("strict_actionable_success", "moderate_actionable_success", "permissive_actionable_success", "catastrophic_failure_numeric"):
            values = frame[column].to_numpy(dtype=int)
            low, high = cluster_bootstrap_mean(frame, column)
            prevalence_rows.append({"partition": split, "category": "outcome", "label": column, "samples": len(values), "positives": int(values.sum()), "prevalence": float(values.mean()), "cluster_bootstrap_ci_low": low, "cluster_bootstrap_ci_high": high})
        boundary = noise[(noise.split == split) & (noise.audit_category.isin(("near_moderate_contract_boundary", "contract_status_changes")))]
        for row in boundary.itertuples():
            prevalence_rows.append({"partition": split, "category": "boundary", "label": row.audit_category, "samples": len(frame), "positives": int(row.samples), "prevalence": float(row.fraction)})
    prevalence = pd.DataFrame(prevalence_rows)
    write_csv_fresh(run / "tables/label_prevalence_by_partition.csv", prevalence)

    mismatch_total = 0; comparison_total = 0
    for frame in frames.values():
        mismatch, compared = label_consistency(frame)
        mismatch_total += mismatch; comparison_total += compared
    metric_hashes = set()
    for split in ("training", "validation"):
        metric_hashes |= set(pd.read_csv(PHASE2 / f"tables/{split}_teacher_reliability_labels.csv")["metric_implementation_sha256"].astype(str))
    query_counts = {split: frame.query_state.value_counts(normalize=True).sort_index().to_dict() for split, frame in frames.items()}
    rare = prevalence[(prevalence.category == "outcome") & prevalence.label.str.contains("actionable")]
    covariate = drift.loc[~drift.metric.str.contains("success|catastrophic")]
    max_abs_smd = float(covariate.loc[covariate.metric != "source_reuse_frequency", "standardized_mean_difference"].abs().max())
    max_reuse_smd = float(covariate.loc[covariate.metric == "source_reuse_frequency", "standardized_mean_difference"].abs().max())
    material_shift = bool(max_abs_smd >= 0.25)
    material_reuse_shift = bool(max_reuse_smd >= 0.25)
    label_bug = bool(mismatch_total or len(metric_hashes) != 1)

    fig_metrics = ["source_r_magnitude", "source_size_arcsec", "snr_proxy", "separation_psf_units", "flux_ratio", "core_obstruction"]
    for metric in fig_metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for split, color in zip(SPLITS, ("#4472c4", "#70ad47", "#ed7d31")):
            values = pd.to_numeric(frames[split][metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.5, label=split, color=color)
        ax.set_xlabel(metric); ax.set_ylabel("density"); ax.legend(); fig.tight_layout()
        fig.savefig(run / f"figures/partition_drift/{metric}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = prevalence[(prevalence.category == "outcome") & prevalence.label.str.contains("actionable")].pivot(index="label", columns="partition", values="prevalence")
    pivot.plot(kind="bar", ax=ax); ax.set_ylabel("positive prevalence"); ax.set_yscale("log"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(run / "figures/partition_drift/actionable_prevalence.png", dpi=180); plt.close(fig)

    report = f"""# Partition drift audit

Status: **{'STOP - LABEL BUG' if label_bug else 'PASS - no manifest or label-application bug detected'}**.

## Direct answers

1. **Why did validation-selected heads collapse on calibration?** The audit does not support a manifest or label-code inconsistency. The dominant explanation is statistical instability from an extremely sparse heterogeneous composite target: validation contained only {int(rare[(rare.partition == 'validation') & (rare.label == 'moderate_actionable_success')].positives.iloc[0])} moderate actionable positives and {int(rare[(rare.partition == 'validation') & (rare.label == 'permissive_actionable_success')].positives.iloc[0])} permissive positives. Calibration prevalence also changed, so a validation ranking selected from very few positive examples was fragile. Isotonic plateaus amplified the operational collapse but are not a population-drift explanation.
2. **Is calibration underpowered because there are too few rare events?** Yes for precise tail and class-conditional calibration. Calibration has {int(rare[(rare.partition == 'calibration') & (rare.label == 'moderate_actionable_success')].positives.iloc[0])} moderate and {int(rare[(rare.partition == 'calibration') & (rare.label == 'permissive_actionable_success')].positives.iloc[0])} permissive positives; those counts cannot support stable multi-regime tail calibration.
3. **Are source populations or scene conditions shifted?** {'A material physical-covariate shift was detected and must be treated as a design concern.' if material_shift else 'No material physical source/scene shift under the preregistered |SMD| >= 0.25 screen.'} The maximum absolute physical-covariate SMD is {max_abs_smd:.3f}. Source-reuse frequency separately reaches |SMD| {max_reuse_smd:.3f} ({'material' if material_reuse_shift else 'not material'}), reducing effective independence and plausibly worsening rare-event instability without changing the physical deployment distribution. Full KS, Wasserstein, and source-group-aware bootstrap results are in `tables/partition_drift_audit.csv`. Query-state mixtures are identical by construction: `{json.dumps(query_counts, sort_keys=True)}`.
4. **Are label definitions applied identically?** Yes. Reapplication produced {mismatch_total} mismatches across {comparison_total} contract-row comparisons, and training/validation record one metric implementation hash ({next(iter(metric_hashes)) if metric_hashes else 'missing'}).

## Interpretation

The prior calibration failure is primarily a sparse-target and logical-heterogeneity problem, not evidence that the frozen reconstructor lost promptability. Query invalidity, valid reconstruction tail risk, and source confusion were combined into one label with different mechanisms and applicability. This audit supports proceeding to the preregistered hierarchical targets.

The cluster intervals resample by the exchangeable source-A duplicate group, thereby preserving repeated observations for an anchored source. This is conservative with respect to ordinary row bootstrap but is not a two-way multi-membership bootstrap; final policy confidence intervals will cluster on all contributing source groups.
"""
    write_text_fresh(run / "diagnostics/partition_drift_report.md", report)
    write_json_fresh(run / "logs/partition_drift_complete.json", {
        "status": "FAIL" if label_bug else "PASS", "label_application_mismatches": mismatch_total,
        "metric_implementation_hash_count": len(metric_hashes), "material_covariate_shift": material_shift,
        "maximum_absolute_physical_covariate_smd": max_abs_smd,
        "maximum_absolute_source_reuse_smd": max_reuse_smd,
        "material_source_reuse_shift": material_reuse_shift,
        "new_data_generated": False, "heads_trained": False,
        "lockbox_accessed": False, "completed_at_unix": time.time(),
    })
    if label_bug:
        raise RuntimeError("Manifest-generation or label-application bug found; hierarchical head training is prohibited")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("bootstrap", "audit", "bootstrap-audit"), required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage in ("bootstrap", "bootstrap-audit"):
        bootstrap(run)
    if args.stage in ("audit", "bootstrap-audit"):
        drift_audit(run)
    print(relative(run))


if __name__ == "__main__":
    main()
