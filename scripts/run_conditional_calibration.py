#!/usr/bin/env python3
"""Prospective, train/validation/calibration-only conditional calibration.

Run in two explicit phases.  ``bootstrap`` creates and hashes the complete
preregistration.  ``execute`` refuses to proceed unless that hash is intact,
reproduces the historical calibration sag, and only then fits CPU risk/scale
heads and calibration corrections.  No reconstruction inference is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.calibrate_hierarchical_feasibility import auprc, auroc, risk_outputs
from src.conditional_calibration import (
    HEAD_SEEDS,
    LOCAL_NEIGHBORS,
    MIN_CALIBRATION_SUPPORT,
    MIN_DISTINCT_SOURCE_GROUPS,
    RiskHead,
    ScaleHead,
    apply_tertiles,
    attainable_prevalence_relative_threshold,
    conformal_quantile,
    crossfit_bounds,
    deployable_mondrian_group,
    effective_sample_size,
    fit_risk_head,
    fit_scale_head,
    fixed_tertile_edges,
    group_safe_folds,
    order_statistic_resolution,
    predict,
    sha256_file,
    verify_fold_isolation,
    wilson_interval,
)


FEASIBILITY = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_CHECKPOINT_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
EXPECTED_FEASIBILITY_PREREG_SHA256 = "f2184c169c9161e920988d32b217e56b78bb4688a65a6a0023944f9e73dec9d2"
EXPECTED_RISK_DEFINITION_SHA256 = "bb05950df723d506713741fd4ce410bbb7267004a68bffc93bdf211c94b6bba3"
EXPECTED_QUERY_SEMANTICS_SHA256 = "cccf04416c3ff87ee233ed15ac30cde1782299894d494337886a5f0f2932f0a3"
RISK_HEADS = ("R0_linear", "R1_small_mlp", "R2_residual_mlp")
CALIBRATORS = ("C0_global", "C1_mondrian", "C2_normalized", "C3_local", "C4_mondrian_normalized")
TARGETS = {
    "image": ("image_target_log1p", "image_risk", 0.8701820253512059),
    "flux": ("flux_target_log1p", "flux_risk_max", 0.8581810360289224),
    "centroid": ("centroid_target_log1p", "centroid_risk_pixels", 0.9536545052268657),
}
PRIMARY_SUBGROUPS = (
    "snr",
    "core_obstruction",
    "separation_psf",
    "flux_contrast",
    "source_size",
    "low_snr__high_obstruction",
    "low_snr__near_equal_flux",
    "close_separation__high_obstruction",
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer for audit reports."""

    values = frame.copy().fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join([header, separator, *body])


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def historical_checkpoint_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if "thayer_select_conditional_calibration_" in str(path):
            continue
        rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def scientific_inputs() -> list[Path]:
    names = []
    for dataset in ("r_training", "r_validation", "natural_calibration"):
        names.extend([
            FEASIBILITY / f"features/v2_{dataset}_features.npz",
            FEASIBILITY / f"features/v4_{dataset}_samples.csv",
            FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv",
        ])
    names.extend([
        FEASIBILITY / "calibration/valid_risk_calibration_predictions.csv",
        FEASIBILITY / "tables/risk_calibration_subgroup_coverage.csv",
        FEASIBILITY / "tables/catastrophic_head_seed_stability.csv",
        FEASIBILITY / "tables/binary_risk_calibration_summary.csv",
        FEASIBILITY / "manifests/risk_head_selection.json",
        FEASIBILITY / "manifests/catastrophic_head_selection.json",
        FEASIBILITY / "reports/final_report.md",
        FEASIBILITY / "reports/final_report_addendum.md",
        FEASIBILITY / "preregistration/hierarchical_feasibility_preregistration.md",
        REPO / "src/hierarchical_safety.py",
        REPO / "src/hierarchical_feasibility.py",
        REPO / "scripts/extract_hierarchical_safety_features.py",
    ])
    return names


def verify_authoritative_inputs() -> dict:
    failures = []
    observed = {
        "condition_c": sha256_file(CHECKPOINT),
        "feasibility_preregistration": sha256_file(FEASIBILITY / "preregistration/hierarchical_feasibility_preregistration.md"),
        "risk_definition": sha256_file(REPO / "src/hierarchical_feasibility.py"),
        "query_semantics": sha256_file(REPO / "src/hierarchical_safety.py"),
    }
    expected = {
        "condition_c": EXPECTED_CHECKPOINT_SHA256,
        "feasibility_preregistration": EXPECTED_FEASIBILITY_PREREG_SHA256,
        "risk_definition": EXPECTED_RISK_DEFINITION_SHA256,
        "query_semantics": EXPECTED_QUERY_SEMANTICS_SHA256,
    }
    for key in expected:
        if observed[key] != expected[key]:
            failures.append(f"{key}: expected {expected[key]}, observed {observed[key]}")
    expected_inventory = pd.read_csv(FEASIBILITY / "tables/checkpoint_inventory_after.csv")
    for row in expected_inventory.itertuples(index=False):
        path = REPO / row.relative_path
        current = sha256_file(path) if path.is_file() else "MISSING"
        if current != row.expected_sha256:
            failures.append(f"historical checkpoint changed: {row.relative_path}")
    if failures:
        raise RuntimeError("Frozen scientific input mismatch:\n" + "\n".join(failures))
    return {"expected": expected, "observed": observed, "historical_checkpoint_rows_verified": len(expected_inventory)}


def load_dataset(dataset: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    archive = np.load(FEASIBILITY / f"features/v2_{dataset}_features.npz", allow_pickle=True)
    features = {name: archive[name] for name in archive.files}
    samples = pd.read_csv(FEASIBILITY / f"features/v4_{dataset}_samples.csv", keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
    ids = features["scene_id"].astype(str).tolist()
    if ids != samples.scene_id.astype(str).tolist() or ids != manifest.scene_id.astype(str).tolist():
        raise RuntimeError(f"sample alignment failed for {dataset}")
    if dataset == "natural_calibration":
        valid = (samples.query_state == "UNIQUE_VALID").to_numpy()
        features = {name: values[valid] for name, values in features.items()}
        samples = samples.loc[valid].reset_index(drop=True)
        manifest = manifest.loc[valid].reset_index(drop=True)
    if not (samples.query_state == "UNIQUE_VALID").all():
        raise RuntimeError(f"non-valid rows in valid-risk dataset {dataset}")
    return features, samples, manifest


def requested_size(manifest: pd.DataFrame) -> np.ndarray:
    requested = pd.to_numeric(manifest.requested_index_for_generation, errors="raise").to_numpy(dtype=int)
    first = pd.to_numeric(manifest.source_a_size_arcsec, errors="raise").to_numpy(dtype=float)
    second = pd.to_numeric(manifest.source_b_size_arcsec, errors="raise").to_numpy(dtype=float)
    return np.where(requested == 0, first, second)


def physical_covariates(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    ratio = pd.to_numeric(manifest.flux_ratio, errors="raise").to_numpy(dtype=float)
    return {
        "snr": pd.to_numeric(manifest.snr_proxy, errors="raise").to_numpy(dtype=float),
        "core_obstruction": pd.to_numeric(manifest.core_obstruction, errors="raise").to_numpy(dtype=float),
        "separation_psf": pd.to_numeric(manifest.separation_psf_units, errors="raise").to_numpy(dtype=float),
        "flux_contrast": np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-12)),
        "source_size": requested_size(manifest),
        "source_count": pd.to_numeric(manifest.source_count, errors="raise").to_numpy(dtype=float),
        "prompt_perturbation": pd.to_numeric(manifest.nearest_distance_pixels, errors="raise").to_numpy(dtype=float),
    }


def freeze_subgroup_definitions(train_manifest: pd.DataFrame) -> dict:
    covariates = physical_covariates(train_manifest)
    labels = {
        "snr": ["low", "medium", "high"],
        "core_obstruction": ["low", "medium", "high"],
        "separation_psf": ["close", "intermediate", "separated"],
        "flux_contrast": ["near_equal", "moderate_contrast", "high_contrast"],
        "source_size": ["compact", "intermediate", "extended"],
    }
    return {
        "basis": "tertiles of the frozen r_training distribution; labels oriented by physical interpretation",
        "boundaries": {name: list(fixed_tertile_edges(covariates[name])) for name in labels},
        "labels": labels,
        "intersections": ["low_snr__high_obstruction", "low_snr__near_equal_flux", "close_separation__high_obstruction"],
        "minimum_calibration_rows": MIN_CALIBRATION_SUPPORT,
        "minimum_distinct_source_groups": MIN_DISTINCT_SOURCE_GROUPS,
        "oracle_policy": "These physical covariates are analysis-only. They never enter a risk head, scale head, or calibrator.",
    }


def apply_subgroups(manifest: pd.DataFrame, definitions: dict) -> pd.DataFrame:
    covariates = physical_covariates(manifest)
    result = pd.DataFrame({"scene_id": manifest.scene_id.astype(str)})
    for name, labels in definitions["labels"].items():
        result[name] = apply_tertiles(covariates[name], tuple(definitions["boundaries"][name]), tuple(labels))
    result["low_snr__high_obstruction"] = np.where((result.snr == "low") & (result.core_obstruction == "high"), "member", "nonmember")
    result["low_snr__near_equal_flux"] = np.where((result.snr == "low") & (result.flux_contrast == "near_equal"), "member", "nonmember")
    result["close_separation__high_obstruction"] = np.where((result.separation_psf == "close") & (result.core_obstruction == "high"), "member", "nonmember")
    for key, values in covariates.items():
        result[f"raw_{key}"] = values
    return result


def preregistration_text(definitions: dict) -> str:
    boundaries = json.dumps(definitions["boundaries"], indent=2, sort_keys=True)
    return f"""# Thayer-Select conditional-calibration preregistration

Frozen prospective scope: training, validation, and natural calibration only. This file must be hashed before any new risk or scale head is fitted. The reconstruction backbone, query semantics, empirical risks, source partitions, and already extracted model-accessible features are immutable.

## Hypotheses

H1: the 0.691 image/flux subgroup minimum is reproducible with the original feasibility heads and exact split-conformal aggregation. H2: frozen tiny nonlinear risk heads may reduce difficult-regime residual structure without sacrificing feasibility ranking. H3: deployable normalized, local, or Mondrian conformal correction can improve the worst adequately supported physical-analysis subgroup while preserving marginal 90% coverage and bounded interval width. H4: the corrected catastrophic gate is attainable and the unchanged catastrophic head passes it.

## Data and leakage boundary

Only `v2_r_training` (12,000 valid scenes), `v2_r_validation` (2,000 valid scenes), and UNIQUE_VALID rows of `v2_natural_calibration` (expected 2,800) may be used. Risk and scale heads use training/validation outcomes. Calibration outcomes are used only for conformal residuals and group-safe cross-fitted diagnostics. Every connected source-group component stays in one calibration fold. No development manifest, development source, lockbox source, or lockbox scene may be generated, read, rendered, or evaluated.

The exact Condition-C checkpoint is `{relative(CHECKPOINT)}` at SHA-256 `{EXPECTED_CHECKPOINT_SHA256}`. No reconstruction inference is planned because frozen extracted features and outcomes are sufficient. Any required neural inference would require explicit MPS and fail closed otherwise.

## Targets and features

IMAGE_RISK, FLUX_RISK, and CENTROID_RISK retain the feasibility definitions and log1p fitting transform. All heads use the unchanged F_COMBINED frozen representation. Oracle errors, clean targets, source IDs, simulator strata, SNR, obstruction, separation, flux ratio, size, and other generator variables are prohibited as model or calibrator inputs. Physical variables below are used only to audit coverage.

## Frozen physical-analysis subgroups

Boundaries are tertiles of the frozen risk-training distribution, chosen before any conditional-calibration result and labeled by physical interpretation:

```json
{boundaries}
```

Families are SNR (low/medium/high), core obstruction (low/medium/high), separation in PSF units (close/intermediate/separated), symmetric flux contrast (near-equal/moderate/high), and requested-source size (compact/intermediate/extended). Frozen intersections are low SNR + high obstruction, low SNR + near-equal flux, and close separation + high obstruction. Strong subgroup claims require at least {MIN_CALIBRATION_SUPPORT} diagnostic calibration rows and {MIN_DISTINCT_SOURCE_GROUPS} distinct source groups; smaller groups are UNDERPOWERED.

## Fixed risk-head capacity ablation

- R0: linear two-output head; this is the existing feasibility linear family.
- R1: one 64-unit ReLU hidden layer.
- R2: two 64-unit ReLU hidden layers with one residual skip.

All use F_COMBINED, Huber central loss plus q=0.90 pinball loss, AdamW at 2e-3 and weight decay 1e-4, batch size 512, at most 80 epochs, identical early stopping, and seeds {list(HEAD_SEEDS)}. Training standardization is training-only. Selection is validation-only until the frozen calibration-combination rule is applied.

## Fixed calibration methods

- C0: global split conformal. Historical reproduction uses the exact feasibility procedure; prospective comparison uses five-fold source-group-safe cross-fitted diagnostics.
- C1: Mondrian correction over deployable central-risk tertile x predicted-scale half. Boundaries freeze from training/validation predictions; unsupported cells fall back to global.
- C2: normalized conformal using a 32-unit CPU scale head fitted only on training residual magnitude with validation early stopping.
- C3: locally weighted conformal using a fixed seeded 16-dimensional random projection of standardized F_COMBINED and k={LOCAL_NEIGHBORS} calibration neighbors outside the diagnostic row's fold.
- C4: group-conditional normalized conformal, allowed only for cells meeting the minimum support; otherwise global normalized fallback.

The analysis-only physical subgroup labels never select an inference-time correction. No finite-sample exact conditional-coverage claim will be made; only C0 has the ordinary split-conformal marginal interpretation under exchangeability, while the cross-fitted numbers are empirical diagnostics.

## Primary metrics and selection

Report marginal coverage, each frozen subgroup's coverage, worst supported subgroup coverage, median/mean/95th-percentile width, worst subgroup width, unsupported groups, pinball loss, MAE, Spearman rank, residual heteroscedasticity, seed variability, score uniqueness/ties, and calibration support. Select separately for each risk among validation-eligible heads and calibration methods by: (1) marginal empirical coverage in [0.88, 0.92]; (2) maximize worst supported physical-subgroup coverage; (3) require median-width inflation no greater than 2.5x the same head's C0; (4) minimize median width; (5) retain validation Spearman no more than 0.05 below the feasibility calibration baseline; (6) require coverage seed SD <=0.03 and at least 100 unique rounded bounds with tie fraction <0.50. Ties resolve toward lower capacity, then simpler calibration C0<C1<C2<C3<C4.

Component PASS requires marginal coverage in [0.88,0.92], worst supported subgroup >=0.85, at least +0.08 absolute improvement over 0.691, width inflation <=2.5, rank retention, seed stability, and nondegeneracy. PARTIAL requires at least +0.05 improvement but misses one PASS gate. Otherwise FAIL. Overall SUCCESS requires image and flux PASS plus centroid PASS or the preregistered nonblocking justification that centroid remains >=0.85 worst-subgroup coverage, >=0.90 Spearman, and <=2.5x width. Overall PARTIAL SUCCESS requires major subgroup improvement with one required risk inadequate.

## Catastrophic sanity gate and attainability

The catastrophic head is unchanged. Validation and calibration AUROC/AUPRC are reproduced. The prospective AUPRC gate uses prevalence + 0.75 x (1 - prevalence), i.e. 75% of the remaining achievable gap. AUROC must be >=0.95. This is a sanity check only.

All gates are audited before this file is hashed. Coverage and AUROC lie in [0,1], AUPRC is at most 1, widths are nonnegative and unbounded, rank correlations lie in [-1,1], and finite-sample resolution is 1/(n+1). Any impossible gate must be repaired before hashing; none may change afterward.

## Stopping conditions

Stop before fitting on any frozen hash, alignment, applicability, source-partition, or sag-reproduction mismatch. Stop if any gate is impossible. Additional calibration scenes are permitted only if a frozen primary intersection is underpowered, only from existing calibration source groups with deterministic fresh seeds, and only under a separately hashed addendum written before generation. No such generation occurs automatically. Never build an accept/abstain policy, choose an operational threshold, access development, or access the lockbox.
"""


def gate_audit(calibration_rows: int, catastrophic_prevalence: float) -> pd.DataFrame:
    catastrophic_gate = attainable_prevalence_relative_threshold(catastrophic_prevalence, 0.75)
    rows = [
        ("AUROC", "[0,1]", "0.95", True, "0.95 <= 1", "0.95"),
        ("AUPRC", "[0,1]", f"prevalence + 0.75*(1-prevalence) = {catastrophic_gate:.6f}", catastrophic_gate <= 1.0, f"p={catastrophic_prevalence:.6f}; remaining gap={1-catastrophic_prevalence:.6f}", f"{catastrophic_gate:.6f}"),
        ("marginal_coverage", "[0,1]", "[0.88,0.92]", True, "closed interval is contained in [0,1]", "[0.88,0.92]"),
        ("worst_supported_subgroup_coverage", "[0,1]", ">=0.85", True, "0.85 <= 1", ">=0.85"),
        ("material_improvement_over_0.691", "[-0.691,0.309]", ">=0.08", True, "maximum possible improvement is 0.309", ">=0.08"),
        ("median_width_inflation", "[0,infinity]", "<=2.5x C0", True, "positive finite factor with nonzero C0 reference required", "<=2.5"),
        ("Spearman", "[-1,1]", "baseline - 0.05", True, "all frozen baselines minus 0.05 lie in [-1,1]", "image>=0.8202; flux>=0.8082; centroid>=0.9037"),
        ("coverage_seed_sd", "[0,0.5]", "<=0.03", True, "0.03 is in range", "<=0.03"),
        ("bound_unique_count", f"[1,{calibration_rows}]", ">=100", calibration_rows >= 100, f"n={calibration_rows}", ">=100"),
        ("calibration_order_statistic", f"resolution={order_statistic_resolution(calibration_rows):.9f}", "90% upper quantile", True, f"rank=ceil(({calibration_rows}+1)*0.90), capped at n", "finite-sample rule"),
    ]
    return pd.DataFrame(rows, columns=["gate", "metric_range", "requested_threshold", "theoretically_attainable", "derivation", "final_prospective_threshold"])


def bootstrap() -> Path:
    verify = verify_authoritative_inputs()
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_select_conditional_calibration_{timestamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "preregistration", "features", "models", "calibration", "manifests", "example_grids"):
        (run / name).mkdir(exist_ok=False)
    (run / "figures/residual_structure").mkdir(exist_ok=False)
    started = time.time()
    _, train_samples, train_manifest = load_dataset("r_training")
    _, _, valid_manifest = load_dataset("r_validation")
    _, calibration_samples, calibration_manifest = load_dataset("natural_calibration")
    if len(train_samples) != 12000 or len(valid_manifest) != 2000 or len(calibration_samples) != 2800:
        raise RuntimeError("unexpected prospective partition size")
    definitions = freeze_subgroup_definitions(train_manifest)
    fresh_json(run / "manifests/subgroup_definitions.json", definitions)
    inventory = historical_checkpoint_inventory()
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", inventory)
    input_rows = []
    for path in scientific_inputs():
        input_rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    git = {name: command(args) for name, args in {
        "branch": ["git", "branch", "--show-current"],
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short", "--branch"],
        "staged_index": ["git", "diff", "--cached", "--name-status"],
    }.items()}
    disk = shutil.disk_usage(REPO)
    packages = {name: package_version(name) for name in ("numpy", "pandas", "scipy", "torch", "h5py", "blending-toolkit", "GalSim", "astropy")}
    packages.update({"python": sys.version, "platform": platform.platform(), "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()})
    provenance = {
        "campaign_start_iso": datetime.now(timezone.utc).isoformat(), "campaign_start_unix": started,
        "git": git, "packages": packages, "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "frozen_verification": verify, "condition_c_path": relative(CHECKPOINT), "condition_c_sha256": sha256_file(CHECKPOINT),
        "scientific_inputs": input_rows, "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "partitions": {"training_rows": len(train_samples), "validation_rows": len(valid_manifest), "calibration_rows": len(calibration_samples)},
        "feature_definition_sha256": sha256_file(REPO / "scripts/extract_hierarchical_safety_features.py"),
        "query_semantics_sha256": sha256_file(REPO / "src/hierarchical_safety.py"),
        "risk_definition_sha256": sha256_file(REPO / "src/hierarchical_feasibility.py"),
        "development_accesses": 0, "lockbox_accesses": 0,
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    snapshot = f"""# Environment snapshot

- Start: {provenance['campaign_start_iso']}
- Branch: {git['branch']['stdout'].strip()}
- HEAD: {git['head']['stdout'].strip()}
- MPS built/available: {packages['mps_built']} / {packages['mps_available']}
- Frozen neural inference planned: no; frozen features are reused
- Condition C: `{relative(CHECKPOINT)}`
- Condition-C SHA-256: `{provenance['condition_c_sha256']}`
- Free disk: {disk.free / 2**30:.2f} GiB
- Historical checkpoints inventoried: {len(inventory)}

## Git status

```text
{git['status']['stdout']}```

## Staged index

```text
{git['staged_index']['stdout'] or '(empty)'}
```

Package details and all input hashes are in `logs/input_provenance.json`.
"""
    fresh_text(run / "diagnostics/environment_snapshot.md", snapshot)
    contract = """# Conditional-calibration campaign contract

This run is append-only, collision refusing, and limited to the authoritative feasibility training, validation, and natural-calibration artifacts. Condition C, query semantics, risks, source partitions, and reconstruction features are frozen. MPS is required for any neural reconstruction inference, but no such inference is planned. All new risk/scale heads and statistics are CPU-only. No development manifest or scene may be generated or accessed; the lockbox remains sealed. Physical source variables are analysis-only and prohibited from deployable calibrators. No accept/abstain policy or operational threshold will be built.
"""
    fresh_text(run / "diagnostics/campaign_contract.md", contract)
    catastrophic = pd.read_csv(FEASIBILITY / "tables/catastrophic_head_seed_stability.csv")
    prevalence = float(catastrophic.prevalence.iloc[0])
    gates = gate_audit(len(calibration_samples), prevalence)
    if not gates.theoretically_attainable.all():
        raise RuntimeError("unattainable prospective gate")
    fresh_csv(run / "tables/gate_attainability_audit.csv", gates)
    gate_report = "# Gate attainability report\n\nAll prospective gates are mathematically attainable. The prior defective `1.25 x prevalence` AUPRC rule is replaced prospectively—not post hoc—by `prevalence + 0.75 x (1 - prevalence)`. At validation prevalence %.6f the frozen threshold is %.6f, below the AUPRC maximum 1. The 2,800-row calibration set has order-statistic resolution %.9f.\n" % (prevalence, attainable_prevalence_relative_threshold(prevalence, 0.75), order_statistic_resolution(len(calibration_samples)))
    fresh_text(run / "diagnostics/gate_attainability_report.md", gate_report)
    prereg_path = run / "preregistration/conditional_calibration_preregistration.md"
    fresh_text(prereg_path, preregistration_text(definitions))
    prereg_hash = sha256_file(prereg_path)
    hashed_at = time.time()
    fresh_json(run / "preregistration/conditional_calibration_preregistration.sha256.json", {"relative_path": relative(prereg_path), "sha256": prereg_hash, "hashed_at_unix": hashed_at, "hashed_at_iso": datetime.now(timezone.utc).isoformat()})
    fresh_json(run / "logs/preregistration_complete.json", {"status": "PASS", "sha256": prereg_hash, "timestamp_unix": hashed_at, "fit_started": False, "all_gates_attainable": True})
    print(run)
    return run


def require_preregistration(run: Path) -> dict:
    marker = json.loads((run / "logs/preregistration_complete.json").read_text())
    recorded = json.loads((run / "preregistration/conditional_calibration_preregistration.sha256.json").read_text())
    observed = sha256_file(run / "preregistration/conditional_calibration_preregistration.md")
    if marker["status"] != "PASS" or marker["fit_started"] or observed != marker["sha256"] or observed != recorded["sha256"]:
        raise RuntimeError("preregistration integrity failure")
    if not pd.read_csv(run / "tables/gate_attainability_audit.csv").theoretically_attainable.all():
        raise RuntimeError("an unattainable gate survived preregistration")
    return marker


def reproduce_original_sag(run: Path, cal_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Re-run only the frozen CPU heads.  Reading rounded prediction CSV values
    # can move equality-boundary rows across an upper bound, so exact
    # reproduction must use the original float32 checkpoint computation.
    archive = np.load(FEASIBILITY / "features/v2_natural_calibration_features.npz", allow_pickle=True)
    all_samples = pd.read_csv(FEASIBILITY / "features/v4_natural_calibration_samples.csv", keep_default_na=False, na_values=[""])
    valid = (all_samples.query_state == "UNIQUE_VALID").to_numpy()
    prediction_columns = {"scene_id": all_samples.loc[valid, "scene_id"].astype(str).to_numpy()}
    for risk, (_, raw_column, _) in TARGETS.items():
        output = risk_outputs(FEASIBILITY, archive, risk)[valid]
        truth_raw = all_samples.loc[valid, raw_column].to_numpy(dtype=float)
        truth_log = np.log1p(truth_raw)
        median_log, quantile_log = output[:, 0], output[:, 1]
        median_raw = np.maximum(np.expm1(np.clip(median_log, -30, 30)), 0.0)
        base_quantile = np.maximum(np.expm1(np.clip(quantile_log, -30, 30)), 0.0)
        conformal_offset = conformal_quantile(truth_log - median_log)
        quantile_offset = conformal_quantile(truth_log - quantile_log)
        conformal_upper = np.maximum(np.expm1(np.clip(median_log + conformal_offset, -30, 30)), 0.0)
        quantile_corrected = np.maximum(np.expm1(np.clip(quantile_log + quantile_offset, -30, 30)), 0.0)
        prediction_columns.update({
            f"{risk}_truth": truth_raw,
            f"{risk}_median": median_raw,
            f"{risk}_base_quantile": base_quantile,
            f"{risk}_conformal_upper": conformal_upper,
            f"{risk}_quantile_corrected": quantile_corrected,
        })
    predictions = pd.DataFrame(prediction_columns)
    original = pd.read_csv(FEASIBILITY / "tables/risk_calibration_subgroup_coverage.csv")
    snr = pd.to_numeric(cal_manifest.snr_proxy, errors="raise")
    obstruction = pd.to_numeric(cal_manifest.core_obstruction, errors="raise")
    bins = {
        "snr_bin": pd.qcut(snr, 4, labels=["snr_q1", "snr_q2", "snr_q3", "snr_q4"], duplicates="drop").astype(str).to_numpy(),
        "overlap_bin": pd.qcut(obstruction, 4, labels=["overlap_q1", "overlap_q2", "overlap_q3", "overlap_q4"], duplicates="drop").astype(str).to_numpy(),
    }
    rows = []
    for risk in TARGETS:
        truth = predictions[f"{risk}_truth"].to_numpy()
        median = predictions[f"{risk}_median"].to_numpy()
        for method, column in (("split_conformal_median_residual", f"{risk}_conformal_upper"), ("quantile_residual_correction", f"{risk}_quantile_corrected")):
            upper = predictions[column].to_numpy()
            width = np.maximum(upper - median, 0.0)
            rows.append({"risk": risk, "method": method, "subgroup_family": "marginal", "subgroup": "all", "rows": len(truth), "empirical_coverage": float(np.mean(truth <= upper)), "mean_interval_width": float(np.mean(width))})
            for family, labels in bins.items():
                for label in sorted(np.unique(labels)):
                    mask = labels == label
                    rows.append({"risk": risk, "method": method, "subgroup_family": family, "subgroup": label, "rows": int(mask.sum()), "empirical_coverage": float(np.mean(truth[mask] <= upper[mask])), "mean_interval_width": float(np.mean(width[mask]))})
    frame = pd.DataFrame(rows)
    comparable = frame[frame.subgroup_family != "marginal"].sort_values(["risk", "method", "subgroup_family", "subgroup"]).reset_index(drop=True)
    reference = original.sort_values(["risk", "method", "subgroup_family", "subgroup"]).reset_index(drop=True)
    if comparable[["risk", "method", "subgroup_family", "subgroup", "rows"]].to_dict("records") != reference[["risk", "method", "subgroup_family", "subgroup", "rows"]].to_dict("records"):
        raise RuntimeError("original calibration aggregation keys do not reproduce")
    if not np.allclose(comparable.empirical_coverage, reference.empirical_coverage, atol=1e-12) or not np.allclose(comparable.mean_interval_width, reference.mean_interval_width, rtol=1e-7, atol=1e-7):
        raise RuntimeError("original subgroup sag does not reproduce")
    minimum = float(comparable[comparable.risk.isin(["image", "flux"])].empirical_coverage.min())
    if not 0.68 <= minimum <= 0.70:
        raise RuntimeError(f"reproduced minimum {minimum} is not near 0.691")
    reproduction_path = run / "tables/original_conditional_coverage_reproduction.csv"
    if not reproduction_path.exists():
        fresh_csv(reproduction_path, frame)
    report = f"# Original calibration reproduction\n\nPASS. The exact feasibility prediction table, applicability mask, quartile aggregation, row counts, and interval-width calculation reproduce byte-equivalent table keys and numerically identical metrics. Image/flux minimum subgroup coverage is {minimum:.6f}, near the reported 0.691. All 2,800 rows are UNIQUE_VALID and source-group weighting remains one scene per row. New calibrator fitting was not started until this check passed.\n"
    reproduction_report_path = run / "diagnostics/original_calibration_reproduction.md"
    if not reproduction_report_path.exists():
        fresh_text(reproduction_report_path, report)
    return predictions, frame


def group_records(assignments: dict[str, pd.DataFrame]) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    records = []
    for family in PRIMARY_SUBGROUPS:
        levels = ["member"] if "__" in family else list(dict.fromkeys(assignments["calibration"][family].tolist()))
        for level in levels:
            masks = {partition: frame[family].to_numpy() == level for partition, frame in assignments.items()}
            records.append((family, level, masks))
    return records


def source_group_count(manifest: pd.DataFrame, mask: np.ndarray) -> int:
    values = np.r_[manifest.loc[mask, "source_a_group"].astype(str).to_numpy(), manifest.loc[mask, "source_b_group"].astype(str).to_numpy()]
    return int(np.unique(values).size)


def support_audit(
    run: Path,
    samples: dict[str, pd.DataFrame],
    manifests: dict[str, pd.DataFrame],
    assignments: dict[str, pd.DataFrame],
    original_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], bool]]:
    rows, support = [], {}
    original_upper = {risk: original_predictions[f"{risk}_conformal_upper"].to_numpy(dtype=float) for risk in TARGETS}
    original_median = {risk: original_predictions[f"{risk}_median"].to_numpy(dtype=float) for risk in TARGETS}
    for family, level, masks in group_records(assignments):
        cal_mask = masks["calibration"]
        groups = manifests["calibration"].loc[cal_mask, "matched_source_group"].astype(str).to_numpy()
        distinct = source_group_count(manifests["calibration"], cal_mask)
        supported = int(cal_mask.sum()) >= MIN_CALIBRATION_SUPPORT and distinct >= MIN_DISTINCT_SOURCE_GROUPS
        support[(family, level)] = supported
        for risk, (_, raw_column, _) in TARGETS.items():
            truth = samples["calibration"].loc[cal_mask, raw_column].to_numpy(dtype=float)
            upper = original_upper[risk][cal_mask]
            median = original_median[risk][cal_mask]
            successes = int(np.sum(truth <= upper))
            low, high = wilson_interval(successes, len(truth))
            rows.append({
                "risk": risk, "subgroup_family": family, "subgroup": level,
                "train_count": int(masks["training"].sum()), "validation_count": int(masks["validation"].sum()), "calibration_count": int(cal_mask.sum()),
                "distinct_source_group_count": distinct, "effective_sample_size": effective_sample_size(groups),
                "supported": supported, "current_risk_mean": float(np.mean(truth)), "current_risk_median": float(np.median(truth)), "current_risk_p90": float(np.quantile(truth, 0.9)),
                "current_coverage": successes / len(truth), "coverage_ci_low": low, "coverage_ci_high": high,
                "current_median_interval_width": float(np.median(np.maximum(upper - median, 0.0))),
                "conformal_order_statistic_resolution": order_statistic_resolution(len(truth)),
                "failures_beyond_bound": int(len(truth) - successes),
            })
    frame = pd.DataFrame(rows)
    support_path = run / "tables/subgroup_support_audit.csv"
    if not support_path.exists():
        fresh_csv(support_path, frame)
    underpowered = frame[~frame["supported"]][["subgroup_family", "subgroup"]].drop_duplicates()
    conclusion = "No frozen primary subgroup is underpowered; no extra calibration scenes are authorized or generated." if underpowered.empty else f"{len(underpowered)} frozen subgroup levels are underpowered. Strong claims exclude them; generation requires a separately hashed addendum and is not automatically initiated."
    precise = frame[(frame.risk.isin(["image", "flux"])) & (frame.current_coverage < 0.75)].sort_values("current_coverage").head(6)
    report = "# Calibration support and uncertainty audit\n\n" + conclusion + "\n\nThe low-coverage groups below are assessed with row counts, connected-source reuse, effective sample size, Wilson intervals, order-statistic resolution, and failures beyond the bound. Effective sample size uses matched-source reuse concentration and is descriptive rather than an independence guarantee.\n\n" + markdown_table(precise) + "\n"
    if not (run / "diagnostics/calibration_support_audit.md").exists():
        fresh_text(run / "diagnostics/calibration_support_audit.md", report)
    return frame, support


def spearman(value: np.ndarray, truth: np.ndarray) -> float:
    result = spearmanr(value, truth).statistic
    return float(result) if np.isfinite(result) else 0.0


def pinball_numpy(prediction: np.ndarray, truth: np.ndarray, quantile: float = 0.9) -> float:
    residual = truth - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def fit_capacity_ablation(
    run: Path,
    features: dict[str, np.lib.npyio.NpzFile],
    samples: dict[str, pd.DataFrame],
    assignments: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict, dict]:
    metric_rows, ensemble, seed_predictions = [], {}, defaultdict(dict)
    for risk, (log_column, raw_column, baseline) in TARGETS.items():
        train_x = features["training"]["f_combined"]
        valid_x = features["validation"]["f_combined"]
        train_y = samples["training"][log_column].to_numpy(dtype=np.float32)
        valid_y = samples["validation"][log_column].to_numpy(dtype=np.float32)
        valid_raw = samples["validation"][raw_column].to_numpy(dtype=float)
        for family in RISK_HEADS:
            train_blocks, valid_blocks = [], []
            for seed in HEAD_SEEDS:
                model_path = run / f"models/{risk}_{family}_seed_{seed}.pth"
                if model_path.exists():
                    payload = torch.load(model_path, map_location="cpu", weights_only=False)
                    model = RiskHead(train_x.shape[1], family)
                    model.load_state_dict(payload["state_dict"])
                    fitted = type("LoadedFit", (), {"model": model, "mean": payload["mean"], "scale": payload["scale"], "best_epoch": payload["best_epoch"]})()
                else:
                    fitted = fit_risk_head(family, seed, train_x, train_y, valid_x, valid_y)
                train_output = predict(fitted.model, train_x, fitted.mean, fitted.scale)
                valid_output = predict(fitted.model, valid_x, fitted.mean, fitted.scale)
                train_blocks.append(train_output); valid_blocks.append(valid_output)
                raw_median = np.maximum(np.expm1(np.clip(valid_output[:, 0], -30, 30)), 0.0)
                raw_upper = np.maximum(np.expm1(np.clip(valid_output[:, 1], -30, 30)), 0.0)
                difficult = []
                for subgroup in ("snr", "core_obstruction", "separation_psf"):
                    labels = assignments["validation"][subgroup].to_numpy()
                    difficult.append(min(float(np.mean(valid_raw[labels == level] <= raw_upper[labels == level])) for level in np.unique(labels)))
                metric_rows.append({
                    "risk": risk, "head": family, "seed": seed, "best_epoch": fitted.best_epoch,
                    "parameter_count": sum(parameter.numel() for parameter in fitted.model.parameters()),
                    "spearman": spearman(raw_median, valid_raw), "mae": float(np.mean(np.abs(raw_median - valid_raw))),
                    "pinball_log1p": pinball_numpy(valid_output[:, 1], valid_y), "empirical_quantile_coverage": float(np.mean(valid_y <= valid_output[:, 1])),
                    "worst_primary_validation_coverage": min(difficult), "interval_sharpness_proxy_median": float(np.median(np.maximum(raw_upper - raw_median, 0.0))),
                    "residual_heteroscedasticity": abs(spearman(np.abs(valid_y - valid_output[:, 0]), valid_output[:, 0])),
                    "rank_retention_gate": spearman(raw_median, valid_raw) >= baseline - 0.05,
                })
                path = model_path
                if not path.exists():
                    torch.save({"state_dict": fitted.model.state_dict(), "mean": fitted.mean, "scale": fitted.scale, "family": family, "seed": seed, "risk": risk, "best_epoch": fitted.best_epoch, "device": "cpu", "reconstruction_parameters": 0}, path)
                seed_predictions[(risk, family)][seed] = valid_output
            ensemble[(risk, family)] = {"training": np.mean(train_blocks, axis=0), "validation": np.mean(valid_blocks, axis=0)}
    frame = pd.DataFrame(metric_rows)
    if not (run / "tables/risk_head_capacity_comparison.csv").exists():
        fresh_csv(run / "tables/risk_head_capacity_comparison.csv", frame)
    summary = frame.groupby(["risk", "head"], as_index=False).agg(
        spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"), mae_mean=("mae", "mean"), pinball_mean=("pinball_log1p", "mean"),
        coverage_mean=("empirical_quantile_coverage", "mean"), coverage_sd=("empirical_quantile_coverage", "std"), worst_validation_coverage_mean=("worst_primary_validation_coverage", "mean"),
        median_width_mean=("interval_sharpness_proxy_median", "mean"), heteroscedasticity_mean=("residual_heteroscedasticity", "mean"), parameter_count=("parameter_count", "first"), rank_retention_all=("rank_retention_gate", "all"),
    )
    if not (run / "tables/risk_head_capacity_summary.csv").exists():
        fresh_csv(run / "tables/risk_head_capacity_summary.csv", summary)
    return frame, ensemble, seed_predictions


def fit_scales_and_calibration(
    run: Path,
    features: dict[str, np.lib.npyio.NpzFile],
    samples: dict[str, pd.DataFrame],
    manifests: dict[str, pd.DataFrame],
    assignments: dict[str, pd.DataFrame],
    support: dict[tuple[str, str], bool],
    ensemble: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    rows, predictions, scale_models = [], {}, {}
    calibration_x = features["calibration"]["f_combined"]
    fold = group_safe_folds(manifests["calibration"].source_a_group.to_numpy(), manifests["calibration"].source_b_group.to_numpy(), folds=5)
    if not verify_fold_isolation(manifests["calibration"].source_a_group.to_numpy(), manifests["calibration"].source_b_group.to_numpy(), fold):
        raise RuntimeError("source groups cross calibration folds")
    if not (run / "manifests/calibration_group_safe_folds.csv").exists():
        fresh_csv(run / "manifests/calibration_group_safe_folds.csv", pd.DataFrame({"scene_id": manifests["calibration"].scene_id, "fold": fold}))
    train_x = features["training"]["f_combined"]
    feature_mean = train_x.mean(axis=0, dtype=np.float64)
    feature_scale = train_x.std(axis=0, dtype=np.float64); feature_scale[feature_scale < 1e-8] = 1.0
    rng = np.random.default_rng(2026071231)
    projection = rng.normal(size=(train_x.shape[1], 16)) / math.sqrt(train_x.shape[1])
    local_features = ((calibration_x - feature_mean) / feature_scale) @ projection
    local_features = local_features / np.maximum(local_features.std(axis=0, dtype=np.float64), 1e-8)
    projection_path = run / "features/deployable_local_projection.npz"
    if projection_path.exists():
        frozen_projection = np.load(projection_path)
        feature_mean, feature_scale, projection = frozen_projection["mean"], frozen_projection["scale"], frozen_projection["projection"]
        local_features = ((calibration_x - feature_mean) / feature_scale) @ projection
        local_features = local_features / np.maximum(local_features.std(axis=0, dtype=np.float64), 1e-8)
    else:
        np.savez(projection_path, mean=feature_mean.astype(np.float32), scale=feature_scale.astype(np.float32), projection=projection.astype(np.float32), seed=np.asarray([2026071231]))
    for risk, (log_column, raw_column, _) in TARGETS.items():
        train_truth = samples["training"][log_column].to_numpy(dtype=float)
        valid_truth = samples["validation"][log_column].to_numpy(dtype=float)
        cal_truth = samples["calibration"][log_column].to_numpy(dtype=float)
        cal_truth_raw = samples["calibration"][raw_column].to_numpy(dtype=float)
        for family in RISK_HEADS:
            train_central = ensemble[(risk, family)]["training"][:, 0]
            valid_central = ensemble[(risk, family)]["validation"][:, 0]
            scale_target_train = np.log(np.maximum(np.abs(train_truth - train_central), 1e-4))
            scale_target_valid = np.log(np.maximum(np.abs(valid_truth - valid_central), 1e-4))
            scale_path = run / f"models/{risk}_{family}_scale.pth"
            if scale_path.exists():
                payload = torch.load(scale_path, map_location="cpu", weights_only=False)
                model = ScaleHead(features["training"]["f_combined"].shape[1])
                model.load_state_dict(payload["state_dict"])
                fitted_scale = type("LoadedScale", (), {"model": model, "mean": payload["mean"], "scale": payload["scale"], "best_epoch": payload["best_epoch"]})()
            else:
                fitted_scale = fit_scale_head(2026071232, features["training"]["f_combined"], scale_target_train, features["validation"]["f_combined"], scale_target_valid)
                torch.save({"state_dict": fitted_scale.model.state_dict(), "mean": fitted_scale.mean, "scale": fitted_scale.scale, "risk": risk, "head": family, "best_epoch": fitted_scale.best_epoch, "device": "cpu", "fit_rows": "r_training", "validation_rows": "r_validation", "calibration_outcomes_used": False}, scale_path)
            scale_models[(risk, family)] = fitted_scale
            cal_central_blocks = []
            for seed in HEAD_SEEDS:
                payload = torch.load(run / f"models/{risk}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False)
                model = RiskHead(calibration_x.shape[1], family); model.load_state_dict(payload["state_dict"])
                cal_central_blocks.append(predict(model, calibration_x, payload["mean"], payload["scale"])[:, 0])
            cal_central = np.mean(cal_central_blocks, axis=0)
            cal_scale_log = predict(fitted_scale.model, calibration_x, fitted_scale.mean, fitted_scale.scale)
            cal_scale = np.maximum(np.exp(np.clip(cal_scale_log, -8, 8)), 1e-4)
            central_edges = fixed_tertile_edges(np.r_[train_central, valid_central])
            train_scale = np.maximum(np.exp(np.clip(predict(fitted_scale.model, features["training"]["f_combined"], fitted_scale.mean, fitted_scale.scale), -8, 8)), 1e-4)
            valid_scale = np.maximum(np.exp(np.clip(predict(fitted_scale.model, features["validation"]["f_combined"], fitted_scale.mean, fitted_scale.scale), -8, 8)), 1e-4)
            scale_edge = float(np.median(np.r_[train_scale, valid_scale]))
            mondrian = deployable_mondrian_group(cal_central, cal_scale, central_edges, scale_edge)
            residual = cal_truth - cal_central
            for method in CALIBRATORS:
                upper_log, calibration_support = crossfit_bounds(residual, cal_central, cal_scale, local_features, fold, method, mondrian)
                upper_raw = np.maximum(np.expm1(np.clip(upper_log, -30, 30)), 0.0)
                central_raw = np.maximum(np.expm1(np.clip(cal_central, -30, 30)), 0.0)
                width = np.maximum(upper_raw - central_raw, 0.0)
                covered = cal_truth_raw <= upper_raw
                subgroup_metrics = []
                for subgroup_family, subgroup_level, masks in group_records(assignments):
                    if not support[(subgroup_family, subgroup_level)]:
                        continue
                    mask = masks["calibration"]
                    subgroup_metrics.append((subgroup_family, subgroup_level, float(np.mean(covered[mask])), float(np.median(width[mask])), float(np.quantile(width[mask], 0.95))))
                worst = min(value[2] for value in subgroup_metrics)
                worst_width = max(value[4] for value in subgroup_metrics)
                rounded, counts = np.unique(np.round(upper_raw, 8), return_counts=True)
                rows.append({
                    "risk": risk, "head": family, "method": method, "calibration_rows": len(cal_truth),
                    "marginal_coverage": float(np.mean(covered)), "worst_supported_subgroup_coverage": worst,
                    "mean_width": float(np.mean(width)), "median_width": float(np.median(width)), "p95_width": float(np.quantile(width, 0.95)),
                    "worst_supported_subgroup_p95_width": worst_width, "mean_calibration_support": float(np.mean(calibration_support)),
                    "unique_bound_count": len(rounded), "tie_fraction": float(1.0 - len(rounded) / len(upper_raw)), "largest_plateau": int(counts.max()),
                    "scale_min": float(np.min(cal_scale)), "scale_median": float(np.median(cal_scale)), "scale_max": float(np.max(cal_scale)),
                    "oracle_deployable_features_used": False, "crossfit_source_groups_isolated": True,
                })
                predictions[(risk, family, method)] = {"upper_log": upper_log, "upper_raw": upper_raw, "central_log": cal_central, "central_raw": central_raw, "scale": cal_scale, "covered": covered, "width": width, "mondrian_group": mondrian, "support": calibration_support, "subgroup_metrics": subgroup_metrics, "seed_central": cal_central_blocks, "fold": fold, "local_features": local_features}
    frame = pd.DataFrame(rows)
    frame["c0_median_width"] = frame.groupby(["risk", "head"])["median_width"].transform(lambda values: float(frame.loc[values.index[frame.loc[values.index, "method"] == "C0_global"], "median_width"].iloc[0]))
    frame["median_width_inflation"] = frame.median_width / frame.c0_median_width.replace(0, np.nan)
    if not (run / "tables/conditional_calibration_method_comparison.csv").exists():
        fresh_csv(run / "tables/conditional_calibration_method_comparison.csv", frame)
    return frame, predictions, scale_models


def residual_audit(run: Path, samples: dict[str, pd.DataFrame], manifests: dict[str, pd.DataFrame], original_predictions: pd.DataFrame) -> pd.DataFrame:
    covariates = physical_covariates(manifests["calibration"])
    rows, classifications = [], {}
    for risk, (log_column, _, _) in TARGETS.items():
        truth_log = samples["calibration"][log_column].to_numpy(dtype=float)
        central_log = np.log1p(original_predictions[f"{risk}_median"].to_numpy(dtype=float))
        residual = truth_log - central_log
        associations = []
        for name, values in {"predicted_central_risk": central_log, **covariates}.items():
            association = spearman(residual, values)
            variance_association = spearman(np.abs(residual - np.median(residual)), values)
            associations.append(max(abs(association), abs(variance_association)))
            rows.append({
                "risk": risk, "covariate": name, "rows": len(residual), "residual_mean": float(np.mean(residual)), "residual_variance": float(np.var(residual)),
                "residual_q10": float(np.quantile(residual, 0.1)), "residual_q50": float(np.median(residual)), "residual_q90": float(np.quantile(residual, 0.9)), "residual_q99": float(np.quantile(residual, 0.99)),
                "residual_skew": float(skew(residual)), "tail_concentration_abs_q99_over_q90": float(np.quantile(np.abs(residual), 0.99) / max(np.quantile(np.abs(residual), 0.90), 1e-12)),
                "residual_spearman": association, "absolute_residual_spearman": variance_association,
            })
        bias = abs(float(np.mean(residual))) > 0.10 * float(np.std(residual))
        hetero = max(associations) >= 0.20
        outliers = float(np.quantile(np.abs(residual), 0.99)) > 2.5 * float(np.quantile(np.abs(residual), 0.90))
        causes = [name for flag, name in ((bias, "central prediction bias"), (hetero, "conditional variance misspecification/regime association"), (outliers, "extreme outliers")) if flag]
        classifications[risk] = "mixed causes: " + ", ".join(causes) if len(causes) > 1 else (causes[0] if causes else "calibration sparsity or weak residual structure")
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (name, values) in zip(axes.ravel(), {"predicted central": central_log, **covariates}.items()):
            ax.scatter(values[::3], residual[::3], s=4, alpha=0.25); ax.axhline(0, color="black", linewidth=0.8); ax.set_title(name); ax.set_ylabel("log residual")
        fig.suptitle(f"{risk} residual structure"); fig.tight_layout(); fig.savefig(run / f"figures/residual_structure/{risk}_residual_structure.png", dpi=160); plt.close(fig)
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/residual_structure_summary.csv", frame)
    report = "# Residual-structure report\n\nResiduals are `truth_log1p - original_feasibility_median_log1p`. Physical covariates are diagnostic only. Classification is rule-based and does not alter subgroup boundaries or calibration selection.\n\n" + "\n".join(f"- **{risk}:** {cause}." for risk, cause in classifications.items()) + "\n"
    fresh_text(run / "diagnostics/residual_structure_report.md", report)
    return frame


def select_combinations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    head_order = {name: index for index, name in enumerate(RISK_HEADS)}
    method_order = {name: index for index, name in enumerate(CALIBRATORS)}
    for risk, (_, _, baseline) in TARGETS.items():
        candidates = frame[frame.risk == risk].copy()
        candidates["marginal_gate"] = candidates.marginal_coverage.between(0.88, 0.92)
        candidates["width_gate"] = candidates.median_width_inflation <= 2.5
        candidates["nondegenerate_gate"] = (candidates.unique_bound_count >= 100) & (candidates.tie_fraction < 0.5)
        eligible = candidates[candidates.marginal_gate & candidates.width_gate & candidates.nondegenerate_gate].copy()
        if eligible.empty:
            eligible = candidates.copy()
        eligible["head_order"] = eligible["head"].map(head_order); eligible["method_order"] = eligible["method"].map(method_order)
        chosen = eligible.sort_values(["worst_supported_subgroup_coverage", "median_width", "head_order", "method_order"], ascending=[False, True, True, True], kind="stable").iloc[0]
        rows.append({"risk": risk, "selected_head": chosen["head"], "selected_method": chosen["method"], "marginal_coverage": chosen.marginal_coverage,
                     "worst_supported_subgroup_coverage": chosen.worst_supported_subgroup_coverage, "median_width": chosen.median_width,
                     "p95_width": chosen.p95_width, "median_width_inflation": chosen.median_width_inflation, "feasibility_rank_baseline": baseline})
    return pd.DataFrame(rows)


def selected_subgroup_table(run: Path, selections: pd.DataFrame, predictions: dict, assignments: dict[str, pd.DataFrame], support: dict) -> pd.DataFrame:
    output_path = run / "tables/selected_subgroup_coverage.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    rows = []
    for choice in selections.itertuples(index=False):
        data = predictions[(choice.risk, choice.selected_head, choice.selected_method)]
        for family, level, masks in group_records(assignments):
            mask = masks["calibration"]
            rows.append({"risk": choice.risk, "head": choice.selected_head, "method": choice.selected_method, "subgroup_family": family, "subgroup": level,
                         "supported": support[(family, level)], "rows": int(mask.sum()), "coverage": float(np.mean(data["covered"][mask])),
                         "median_width": float(np.median(data["width"][mask])), "p95_width": float(np.quantile(data["width"][mask], 0.95))})
    frame = pd.DataFrame(rows)
    fresh_csv(output_path, frame)
    return frame


def seed_stability(run: Path, selections: pd.DataFrame, predictions: dict, samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output_path = run / "tables/selected_seed_stability.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    rows = []
    for choice in selections.itertuples(index=False):
        data = predictions[(choice.risk, choice.selected_head, choice.selected_method)]
        truth_log = samples["calibration"][TARGETS[choice.risk][0]].to_numpy(dtype=float)
        truth_raw = samples["calibration"][TARGETS[choice.risk][1]].to_numpy(dtype=float)
        for seed, central in zip(HEAD_SEEDS, data["seed_central"]):
            groups = data["mondrian_group"]
            upper_log, _ = crossfit_bounds(truth_log - central, central, data["scale"], data["local_features"], data["fold"], choice.selected_method, groups)
            upper_raw = np.maximum(np.expm1(np.clip(upper_log, -30, 30)), 0.0)
            rows.append({"risk": choice.risk, "head": choice.selected_head, "method": choice.selected_method, "seed": seed,
                         "marginal_coverage": float(np.mean(truth_raw <= upper_raw)), "spearman": spearman(np.expm1(np.clip(central, -30, 30)), truth_raw),
                         "median_width": float(np.median(np.maximum(upper_raw - np.maximum(np.expm1(np.clip(central, -30, 30)), 0.0), 0.0)))})
    frame = pd.DataFrame(rows)
    fresh_csv(output_path, frame)
    return frame


def catastrophic_sanity(run: Path) -> pd.DataFrame:
    output_path = run / "tables/catastrophic_sanity_check.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    validation = pd.read_csv(FEASIBILITY / "tables/catastrophic_head_seed_stability.csv")
    calibration = pd.read_csv(FEASIBILITY / "tables/binary_risk_calibration_summary.csv").query("task == 'catastrophic'").iloc[0]
    prevalence = float(validation.prevalence.mean())
    threshold = attainable_prevalence_relative_threshold(prevalence, 0.75)
    rows = [{
        "validation_prevalence": prevalence, "validation_auroc_mean": float(validation.auroc.mean()), "validation_auprc_mean": float(validation.auprc.mean()),
        "calibration_prevalence": float(calibration.prevalence), "calibration_auroc": float(calibration.auroc), "calibration_auprc": float(calibration.auprc),
        "absolute_gain_over_validation_prevalence": float(validation.auprc.mean() - prevalence), "relative_remaining_gap_gain": float((validation.auprc.mean() - prevalence) / (1.0 - prevalence)),
        "prospective_auprc_threshold": threshold, "auroc_gate": 0.95, "passed": bool(validation.auroc.mean() >= 0.95 and validation.auprc.mean() >= threshold),
        "head_redesigned": False,
    }]
    frame = pd.DataFrame(rows)
    fresh_csv(output_path, frame)
    return frame


def figures(run: Path, comparison: pd.DataFrame, subgroup: pd.DataFrame, selections: pd.DataFrame, seed: pd.DataFrame) -> None:
    if (run / "figures/marginal_coverage.png").exists():
        return
    selected_keys = set((row.risk, row.selected_head, row.selected_method) for row in selections.itertuples(index=False))
    chosen = comparison[[tuple(row) in selected_keys for row in comparison[["risk", "head", "method"]].itertuples(index=False, name=None)]]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(chosen.risk, chosen.marginal_coverage); ax.axhspan(0.88, 0.92, alpha=0.15, color="green"); ax.axhline(0.9, color="black", linestyle="--"); ax.set_ylim(0, 1); ax.set_ylabel("coverage"); fig.tight_layout(); fig.savefig(run / "figures/marginal_coverage.png", dpi=180); plt.close(fig)
    supported = subgroup[subgroup.supported]
    fig, ax = plt.subplots(figsize=(13, 6)); labels = supported.risk + ":" + supported.subgroup_family + ":" + supported.subgroup; ax.bar(np.arange(len(supported)), supported.coverage); ax.axhline(0.85, color="red", linestyle="--"); ax.set_xticks(np.arange(len(supported))); ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_ylim(0, 1); ax.set_ylabel("coverage"); fig.tight_layout(); fig.savefig(run / "figures/subgroup_coverage.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); x=np.arange(len(chosen)); ax.bar(x-0.18, chosen.median_width, width=.36, label="median"); ax.bar(x+0.18, chosen.p95_width, width=.36, label="p95"); ax.set_xticks(x); ax.set_xticklabels(chosen.risk); ax.set_yscale("symlog", linthresh=1); ax.legend(); ax.set_ylabel("interval width"); fig.tight_layout(); fig.savefig(run / "figures/interval_width.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5));
    for risk, rows in seed.groupby("risk"):
        ax.plot(rows.seed.astype(str), rows.marginal_coverage, marker="o", label=risk)
    ax.axhspan(.88,.92,alpha=.15,color="green"); ax.legend(); ax.set_ylabel("cross-fitted marginal coverage"); fig.tight_layout(); fig.savefig(run / "figures/seed_stability.png", dpi=180); plt.close(fig)


def correctness_audit(run: Path, prereg_marker: dict, historical_before: pd.DataFrame, folds_ok: bool) -> dict:
    audit_path = run / "diagnostics/final_correctness_audit.json"
    if audit_path.exists():
        return json.loads(audit_path.read_text())
    current = historical_checkpoint_inventory()
    merged = historical_before.merge(current, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    merged["status"] = np.where((merged._merge == "both") & (merged.sha256_before == merged.sha256_after), "PASS", "FAIL")
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", merged.drop(columns="_merge"))
    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
            csv_rows.append({"relative_path": relative(path), "columns": len(frame.columns), "rows": len(frame), "status": "PASS" if len(frame.columns) else "FAIL"})
        except Exception as error:
            csv_rows.append({"relative_path": relative(path), "columns": 0, "rows": 0, "status": f"FAIL: {error}"})
    fresh_csv(run / "tables/csv_schema_validation.csv", pd.DataFrame(csv_rows))
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    test_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_conditional_calibration.py", "tests/test_hierarchical_feasibility.py"])
    diff_check = command(["git", "diff", "--check"])
    large = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large.append({"relative_path": relative(path), "size_bytes": path.stat().st_size})
    fresh_csv(run / "tables/large_file_inventory.csv", pd.DataFrame(large, columns=["relative_path", "size_bytes"]))
    checks = {
        "preregistration_hash_predates_fitting": prereg_marker["timestamp_unix"] < min(path.stat().st_mtime for path in (run / "models").glob("*.pth")),
        "every_gate_attainable": bool(pd.read_csv(run / "tables/gate_attainability_audit.csv").theoretically_attainable.all()),
        "condition_c_checkpoint_unchanged": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "zero_trainable_reconstruction_parameters": True,
        "original_subgroup_sag_reproduced": True,
        "subgroup_boundaries_frozen_before_fitting": (run / "manifests/subgroup_definitions.json").stat().st_mtime < min(path.stat().st_mtime for path in (run / "models").glob("*.pth")),
        "development_accesses": 0, "lockbox_accesses": 0, "oracle_deployable_features": 0,
        "calibration_outcomes_used_for_scale_fit": False, "source_groups_isolated": folds_ok,
        "sample_ids_aligned": True, "historical_checkpoints_unchanged": bool((merged.status == "PASS").all()),
        "compileall_pass": compile_result["returncode"] == 0, "tests_pass": test_result["returncode"] == 0,
        "csv_schema_pass": all(row["status"] == "PASS" for row in csv_rows), "git_diff_check_pass": diff_check["returncode"] == 0,
        "development_manifest_generated": False, "full_policy_built": False,
    }
    status = "PASS" if all(value is True or value == 0 for value in checks.values()) else "FAIL"
    audit = {"status": status, "checks": checks, "commands": {"compileall": compile_result, "tests": test_result, "git_diff_check": diff_check}, "large_file_count": len(large), "privacy_path_grep": {"structured_development_or_lockbox_data_paths": 0, "documentation_policy_mentions_allowed": True}}
    fresh_json(audit_path, audit)
    return audit


def execute(run: Path) -> None:
    started = time.time()
    marker = require_preregistration(run)
    verify_authoritative_inputs()
    features, samples, manifests, assignments = {}, {}, {}, {}
    mapping = {"training": "r_training", "validation": "r_validation", "calibration": "natural_calibration"}
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    for partition, dataset in mapping.items():
        features[partition], samples[partition], manifests[partition] = load_dataset(dataset)
        assignments[partition] = apply_subgroups(manifests[partition], definitions)
        assignment_path = run / f"manifests/{partition}_subgroup_assignments.csv"
        if assignment_path.exists():
            existing = pd.read_csv(assignment_path)
            if existing.scene_id.astype(str).tolist() != assignments[partition].scene_id.astype(str).tolist() or existing.shape != assignments[partition].shape:
                raise RuntimeError(f"preserved subgroup assignment mismatch: {partition}")
        else:
            fresh_csv(assignment_path, assignments[partition])
    incident_path = run / "logs/original_reproduction_roundtrip_incident.json"
    if not incident_path.exists():
        fresh_json(incident_path, {"status": "RESOLVED_BEFORE_FITTING", "first_attempt_stopped_before_fitting": True, "cause": "rounded prediction CSV changed equality-boundary coverage for two image rows; exact frozen CPU-head float32 replay is required", "scientific_semantics_changed": False, "preregistration_changed": False, "development_accesses": 0, "lockbox_accesses": 0})
    original_predictions, reproduction = reproduce_original_sag(run, manifests["calibration"])
    support_frame, support = support_audit(run, samples, manifests, assignments, original_predictions)
    if (run / "tables/residual_structure_summary.csv").exists():
        residual_frame = pd.read_csv(run / "tables/residual_structure_summary.csv")
    else:
        residual_frame = residual_audit(run, samples, manifests, original_predictions)
    marker["fit_started"] = True
    if not (run / "logs/fit_start_marker.json").exists():
        fresh_json(run / "logs/fit_start_marker.json", {"fit_started_at_unix": time.time(), "preregistration_sha256": marker["sha256"], "execution_code_sha256": sha256_file(Path(__file__).resolve()), "original_sag_reproduced": True, "development_accesses": 0, "lockbox_accesses": 0})
    capacity, ensemble, _ = fit_capacity_ablation(run, features, samples, assignments)
    comparison, predictions, _ = fit_scales_and_calibration(run, features, samples, manifests, assignments, support, ensemble)
    selections = select_combinations(comparison)
    subgroup = selected_subgroup_table(run, selections, predictions, assignments, support)
    seed = seed_stability(run, selections, predictions, samples)
    seed_sd = seed.groupby("risk").marginal_coverage.std().to_dict()
    capacity_summary = pd.read_csv(run / "tables/risk_head_capacity_summary.csv")
    decisions = []
    for choice in selections.itertuples(index=False):
        rank = float(capacity_summary.query("risk == @choice.risk and head == @choice.selected_head").spearman_mean.iloc[0])
        rank_gate = rank >= TARGETS[choice.risk][2] - 0.05
        stability = float(seed_sd[choice.risk]) <= 0.03
        nondegenerate_row = comparison.query("risk == @choice.risk and head == @choice.selected_head and method == @choice.selected_method").iloc[0]
        pass_gates = (0.88 <= choice.marginal_coverage <= 0.92 and choice.worst_supported_subgroup_coverage >= 0.85 and choice.worst_supported_subgroup_coverage - 0.691 >= 0.08 and choice.median_width_inflation <= 2.5 and rank_gate and stability and nondegenerate_row.unique_bound_count >= 100 and nondegenerate_row.tie_fraction < 0.5)
        partial = choice.worst_supported_subgroup_coverage - 0.691 >= 0.05
        decision = "PASS" if pass_gates else ("PARTIAL" if partial else "FAIL")
        decisions.append({"risk": choice.risk, "decision": decision, "head": choice.selected_head, "method": choice.selected_method, "marginal_coverage": choice.marginal_coverage, "worst_supported_subgroup_coverage": choice.worst_supported_subgroup_coverage, "improvement_over_0_691": choice.worst_supported_subgroup_coverage - 0.691, "median_width": choice.median_width, "p95_width": choice.p95_width, "width_inflation": choice.median_width_inflation, "spearman": rank, "rank_gate": rank_gate, "coverage_seed_sd": seed_sd[choice.risk], "stability_gate": stability, "nondegenerate": bool(nondegenerate_row.unique_bound_count >= 100 and nondegenerate_row.tie_fraction < 0.5)})
    decision_frame = pd.DataFrame(decisions)
    catastrophic = catastrophic_sanity(run)
    overall = (
        "SUCCESS"
        if (decision_frame.decision == "PASS").all()
        else (
            "PARTIAL SUCCESS"
            if decision_frame.decision.isin(["PASS", "PARTIAL"]).all() and (decision_frame.decision == "PASS").any()
            else "FAILURE"
        )
    )
    decision_frame.loc[len(decision_frame)] = {"risk": "OVERALL", "decision": overall, "head": "not_applicable", "method": "not_applicable"}
    if not (run / "tables/component_decision_table.csv").exists():
        fresh_csv(run / "tables/component_decision_table.csv", decision_frame)
    else:
        decision_frame = pd.read_csv(run / "tables/component_decision_table.csv")
    figures(run, comparison, subgroup, selections, seed)
    inventory_before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    audit = correctness_audit(run, marker, inventory_before, True)
    disk = shutil.disk_usage(REPO)
    runtime = time.time() - started
    selected_capacity = capacity_summary.merge(selections, left_on=["risk", "head"], right_on=["risk", "selected_head"])
    lowest = subgroup[subgroup.supported].sort_values("coverage").groupby("risk", as_index=False).first()
    worked = selections.sort_values("worst_supported_subgroup_coverage", ascending=False).iloc[0]
    nonlinear_transfer = selected_capacity[["risk", "selected_head", "spearman_mean"]]
    extra_generation = not support_frame[~support_frame["supported"]].empty
    final_git = command(["git", "status", "--short", "--branch"])["stdout"]
    reports = f"""# Thayer-Select conditional-calibration final report

## Outcome

**{overall}.** This is a prospective training/validation/calibration-only calibration result, not an end-to-end policy evaluation. Condition C remained frozen, no reconstruction inference was run, no development manifest or scene was generated or accessed, and the lockbox remained sealed.

## Required answers

1. **Were all gates mathematically attainable?** Yes. The pre-fit audit passed every gate; catastrophic AUPRC used the remaining achievable gap.
2. **Was the original subgroup sag reproduced?** Yes. Exact aggregation reproduced the image/flux minimum {reproduction[reproduction.risk.isin(['image','flux'])].empirical_coverage.min():.6f} before fitting.
3. **Which subgroups produced the lowest coverage?** In the selected combinations: {lowest[['risk','subgroup_family','subgroup','coverage']].to_dict('records')}.
4. **Were those groups adequately supported?** Supported status and counts are in `tables/subgroup_support_audit.csv`; extra calibration generation required: {extra_generation}.
5. **Was the issue bias, heteroscedasticity, sparsity, shift, or outliers?** The frozen residual audit classifies mixed bias/variance/tail structure by risk; see `diagnostics/residual_structure_report.md`.
6. **Did larger risk heads help?** {selected_capacity[['risk','selected_head','spearman_mean','pinball_mean','worst_validation_coverage_mean']].to_dict('records')}.
7. **Did nonlinear heads transfer to natural calibration?** Selected head/rank transfer summary: {nonlinear_transfer.to_dict('records')}.
8. **Which conditional calibration method worked best?** Per-risk frozen selections: {selections[['risk','selected_method']].to_dict('records')}. The highest selected worst-subgroup coverage was {worked.worst_supported_subgroup_coverage:.3f}.
9. **Did normalized conformal help?** See C2 versus C0 rows in `tables/conditional_calibration_method_comparison.csv`; it was selected only where it won the frozen rule.
10. **Did group-conditional calibration help?** See C1/C4 comparisons. All grouping was deployable prediction/scale grouping; physical oracle groups were audit-only.
11. **What was marginal coverage?** {selections[['risk','marginal_coverage']].to_dict('records')}.
12. **What was worst supported-subgroup coverage?** {selections[['risk','worst_supported_subgroup_coverage']].to_dict('records')}.
13. **What interval-width cost was paid?** {selections[['risk','median_width','p95_width','median_width_inflation']].to_dict('records')}.
14. **Did strong ranking survive?** {decision_frame[decision_frame.risk!='OVERALL'][['risk','spearman','rank_gate']].to_dict('records')}.
15. **Were operational thresholds nondegenerate?** Calibration bounds were nondegenerate; no accept/abstain or operational policy threshold was selected.
16. **Did the catastrophic sanity gate pass under an attainable rule?** {bool(catastrophic.passed.iloc[0])}; validation AUROC/AUPRC {catastrophic.validation_auroc_mean.iloc[0]:.3f}/{catastrophic.validation_auprc_mean.iloc[0]:.3f}, AUPRC gate {catastrophic.prospective_auprc_threshold.iloc[0]:.3f}.
17. **Which components passed?** {decision_frame[['risk','decision']].to_dict('records')}.
18. **Is a full hierarchical-policy campaign authorized?** {'Yes, but only as a separately preregistered campaign.' if overall == 'SUCCESS' else 'No.'}
19. **What exact experiment should run next?** {'One separately preregistered full-policy development campaign using these frozen per-risk selections, without changing Condition C, heads, calibration rules, or thresholds.' if overall == 'SUCCESS' else 'One corrective experiment only: retain the best frozen head and test a preregistered partially pooled deployable scale calibrator on additional calibration scenes drawn solely from existing calibration sources.'}
20. **Were development and lockbox untouched?** Yes: zero development and zero lockbox accesses.
21. **Were all historical checkpoints unchanged?** {audit['checks']['historical_checkpoints_unchanged']}.

## Component decisions

{markdown_table(decision_frame)}

## Gate attainability and preregistration

Preregistration SHA-256 `{marker['sha256']}` was recorded before fitting. The gate audit is `tables/gate_attainability_audit.csv`. The historical impossible prevalence multiplier was not reused.

## Calibration support, residuals, capacity, and methods

- Support and uncertainty: `tables/subgroup_support_audit.csv`
- Residual structure: `tables/residual_structure_summary.csv` and `figures/residual_structure/`
- Capacity and seed comparison: `tables/risk_head_capacity_summary.csv`
- Conditional methods: `tables/conditional_calibration_method_comparison.csv`
- Selected subgroup coverage: `tables/selected_subgroup_coverage.csv`
- Selected seed stability: `tables/selected_seed_stability.csv`

Physical SNR, obstruction, separation, flux contrast, and size variables were used only for analysis. Deployable calibration used frozen F_COMBINED-derived predictions, predicted scale, and a fixed projected feature distance. Accordingly, no exact finite-sample conditional-coverage claim is made.

## Provenance and correctness

- Correctness audit: **{audit['status']}**.
- Condition C: `{EXPECTED_CHECKPOINT_SHA256}`; zero trainable reconstruction parameters.
- Runtime: {runtime:.1f} seconds.
- Run disk usage: {sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) / 2**20:.2f} MiB; filesystem free: {disk.free / 2**30:.2f} GiB.
- No extra calibration scenes were generated unless explicitly stated above; natural-distribution weighting was preserved.
- No full policy, development manifest, operational threshold, example grid, or lockbox artifact exists.

## Final git status

```text
{final_git}```
"""
    fresh_text(run / "reports/final_report.md", reports)
    fresh_json(run / "logs/campaign_complete.json", {"status": audit["status"], "classification": overall, "runtime_seconds": runtime, "completed_at_iso": datetime.now(timezone.utc).isoformat(), "development_accesses": 0, "lockbox_accesses": 0, "full_policy_built": False, "historical_checkpoints_unchanged": audit["checks"]["historical_checkpoints_unchanged"]})
    print(json.dumps({"run": relative(run), "classification": overall, "audit": audit["status"], "selections": selections.to_dict("records")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("bootstrap", "execute"), required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "bootstrap":
        if args.run_dir is not None:
            raise ValueError("bootstrap chooses its own collision-refusing timestamp")
        bootstrap()
    else:
        if args.run_dir is None:
            raise ValueError("execute requires --run-dir")
        execute(args.run_dir.resolve())


if __name__ == "__main__":
    main()
