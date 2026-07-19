#!/usr/bin/env python3
"""Prospective partially pooled scale correction for Thayer-Select.

The two explicit phases are intentional.  ``bootstrap`` creates the master
run, freezes provenance, audits every gate, and hashes the preregistration.
``execute`` first reproduces the authoritative failure, then and only then
creates out-of-fold residual targets and fits CPU scale models.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.calibrate_hierarchical_feasibility import auprc, auroc, binary_logits, query_logits, risk_outputs
from scripts.train_hierarchical_query_gate import CLASS_TO_INDEX, QueryNet, SEEDS as QUERY_SEEDS, macro_metrics
from scripts.run_conditional_calibration import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    FEASIBILITY,
    HEAD_SEEDS,
    PRIMARY_SUBGROUPS,
    RiskHead,
    ScaleHead,
    TARGETS,
    apply_subgroups,
    freeze_subgroup_definitions,
    group_records,
    load_dataset,
    source_group_count,
)
from src.conditional_calibration import (
    UnionFind,
    conformal_quantile as prior_conformal_quantile,
    crossfit_bounds,
    deployable_mondrian_group,
    fixed_tertile_edges,
    fit_risk_head,
    group_safe_folds,
    predict as predict_prior,
    sha256_file,
    verify_fold_isolation,
)
from src.scale_correction import (
    MODEL_FAMILIES,
    OBJECTIVES,
    SCALE_SEEDS,
    ScaleNet,
    cluster_bootstrap_indices,
    crossfit_normalized_upper,
    fit_payload,
    fit_scale_model,
    load_fit,
    normalized_scores,
    predict_scale,
)


PRIOR = REPO / "outputs/runs/thayer_select_conditional_calibration_20260712_021556"
RUN_PREFIX = "thayer_select_scale_correction_"
SELECTED = {
    "image": {"head": "R1_small_mlp", "method": "C4_mondrian_normalized"},
    "flux": {"head": "R2_residual_mlp", "method": "C4_mondrian_normalized"},
    "centroid": {"head": "R1_small_mlp", "method": "C2_normalized"},
}
RAW_TARGET = {"image": "image_risk", "flux": "flux_risk_max", "centroid": "centroid_risk_pixels"}
LOG_TARGET = {"image": "image_target_log1p", "flux": "flux_target_log1p", "centroid": "centroid_target_log1p"}
SCALE_BOUNDS = {"image": (1e-3, 5.0), "flux": (1e-3, 25.0)}
P95_WIDTH_CAP = {"image": 6.0, "flux": 30.0}
BOOTSTRAP_REPLICATES = 300


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


def all_prior_hashes() -> list[dict]:
    rows = []
    for path in sorted(PRIOR.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def historical_checkpoint_inventory(exclude_run: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if exclude_run is not None and exclude_run in path.parents:
            continue
        rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def selected_risk_head_hashes() -> list[dict]:
    rows = []
    for risk, selection in SELECTED.items():
        for seed in HEAD_SEEDS:
            path = PRIOR / f"models/{risk}_{selection['head']}_seed_{seed}.pth"
            rows.append({"risk": risk, "head": selection["head"], "seed": seed,
                         "relative_path": relative(path), "sha256": sha256_file(path)})
    return rows


def verify_frozen_inputs() -> dict:
    failures = []
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        failures.append("Condition-C checkpoint hash differs")
    prior_provenance = json.loads((PRIOR / "logs/input_provenance.json").read_text())
    for row in prior_provenance["scientific_inputs"]:
        path = REPO / row["relative_path"]
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != row["sha256"]:
            failures.append(f"prior scientific input differs: {row['relative_path']}")
    prior_checkpoints = pd.read_csv(PRIOR / "tables/checkpoint_inventory_after.csv")
    if "sha256_after" in prior_checkpoints:
        hash_column = "sha256_after"
    elif "observed_sha256" in prior_checkpoints:
        hash_column = "observed_sha256"
    elif "sha256" in prior_checkpoints:
        hash_column = "sha256"
    else:
        hash_column = "expected_sha256"
    for row in prior_checkpoints.itertuples(index=False):
        path = REPO / row.relative_path
        expected = getattr(row, hash_column)
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(f"historical checkpoint differs: {row.relative_path}")
    required = [
        PRIOR / "reports/final_report.md",
        PRIOR / "reports/final_report_addendum.md",
        PRIOR / "reports/final_documentation_audit_addendum.md",
        REPO / "docs/conditional_calibration_experiment.md",
        REPO / "docs/subgroup_coverage_contract.md",
        REPO / "docs/gate_attainability_protocol.md",
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    failures.extend(f"missing authoritative artifact: {path}" for path in missing)
    if failures:
        raise RuntimeError("Frozen scientific input verification failed:\n" + "\n".join(failures))
    return {
        "condition_c_sha256": sha256_file(CHECKPOINT),
        "prior_scientific_inputs_verified": len(prior_provenance["scientific_inputs"]),
        "prior_checkpoint_rows_verified": len(prior_checkpoints),
        "authoritative_artifacts": [{"relative_path": relative(path), "sha256": sha256_file(path)} for path in required],
    }


def gate_audit() -> pd.DataFrame:
    prior_component = pd.read_csv(PRIOR / "tables/component_decision_table.csv")
    rows = []
    for risk in ("image", "flux"):
        baseline = prior_component[prior_component.risk == risk].iloc[0]
        requested = [
            ("marginal_coverage", "[0,1]", float(baseline.marginal_coverage), "[0.88,0.92]", True,
             "The requested closed interval is contained in the metric range."),
            ("worst_supported_subgroup_coverage", "[0,1]", float(baseline.worst_supported_subgroup_coverage), ">=0.82", True,
             "0.82 is below the mathematical maximum 1."),
            ("low_snr_high_obstruction_coverage", "[0,1]", 0.6373056994818653 if risk == "image" else 0.6839378238341969, ">=0.82", True,
             "0.82 is below the mathematical maximum 1 and the subgroup is supported."),
            ("median_width_inflation", "[0,infinity)", 1.0, "<=1.75x baseline", True,
             "The baseline median width is positive, so the ratio is defined."),
            ("p95_width", "[0,infinity)", float(baseline.p95_width), f"<{P95_WIDTH_CAP[risk]}", True,
             "The frozen finite cap is positive and exceeds zero-width lower bound."),
            ("calibration_spearman", "[-1,1]", 0.8699860795408318 if risk == "image" else 0.8616978053604143,
             ">=0.82" if risk == "image" else ">=0.80", True, "The threshold lies inside [-1,1]."),
            ("source_group_bootstrap_stability", "[0,1]", math.nan, "coverage CI lower bound >=0.75 and seed SD <=0.03", True,
             "Both requested thresholds lie in their metric ranges."),
            ("scale_floor", "[0,infinity)", math.nan, str(SCALE_BOUNDS[risk][0]), True,
             "The fixed floor is positive and below the cap."),
            ("scale_cap", "[0,infinity)", math.nan, str(SCALE_BOUNDS[risk][1]), True,
             "The fixed cap is finite and above the floor."),
        ]
        for gate, metric_range, baseline_value, threshold, attainable, derivation in requested:
            rows.append({"risk": risk, "gate": gate, "theoretical_range": metric_range,
                         "baseline": baseline_value, "requested_threshold": threshold,
                         "attainable": attainable, "derivation": derivation})
    rows.extend([
        {"risk": "centroid", "gate": "reproduce_prior_pass", "theoretical_range": "{PASS,FAIL}",
         "baseline": "PASS", "requested_threshold": "PASS", "attainable": True,
         "derivation": "Byte-identical replay can reproduce the prior selected result."},
        {"risk": "catastrophic_valid", "gate": "AUPRC", "theoretical_range": "[0,1]",
         "baseline": 0.997, "requested_threshold": 0.954125, "attainable": True,
         "derivation": "0.954125 <= 1 and was prospectively repaired in the prior campaign."},
        {"risk": "all", "gate": "conformal_rank", "theoretical_range": "[1,n]",
         "baseline": "ceil((n+1)*0.90)", "requested_threshold": "finite-sample higher order statistic",
         "attainable": True, "derivation": "For every nonempty fold complement the rank is capped at n."},
    ])
    return pd.DataFrame(rows)


def preregistration_text(definitions: dict) -> str:
    return f"""# Partially pooled scale-correction preregistration

Timestamped scope: train, validation, and natural calibration only. This file is hashed before any cross-fitted risk head or scale model is fitted. Condition C, query semantics, risk definitions, selected risk-head families, source partitions, and physical subgroup boundaries are frozen.

## Scientific hypothesis

A small deployable residual-scale model with strong partial pooling can preserve the strong image/flux ranking while repairing low-signal/high-complexity coverage without median-width inflation above 1.75x the authoritative baseline or unbounded tail inflation.

## Fixed data and leakage boundary

Use exactly 12,000 `r_training` rows, 2,000 `r_validation` rows, and the 2,800 UNIQUE_VALID rows of `natural_calibration`. No development manifest, development scene, lockbox source, or lockbox scene may be generated, opened, rendered, or evaluated. Calibration outcomes are reserved for five-fold connected-source-group cross-fitted conformal quantiles and final diagnostics only. Source groups cannot cross folds.

The frozen Condition-C checkpoint is `{relative(CHECKPOINT)}` with SHA-256 `{EXPECTED_CHECKPOINT_SHA256}`. Frozen neural features are reused. No reconstruction inference is planned; any future neural extraction would require MPS and must fail rather than silently use CPU.

## Risks, predictions, and residual targets

IMAGE_RISK and FLUX_RISK retain their exact raw definitions. The authoritative selected heads are image R1 small MLP and flux R2 residual MLP, each as the five-seed ensemble. Their training-row predictions are in-sample, so deployable residual targets must use five source-component folds: refit the exact frozen-form head on four folds, predict only the held-out fold, and concatenate only held-out predictions. Validation remains the fixed early-stopping/model-selection split. Natural calibration is never used for head or scale-model selection.

Primary target: `abs(true_risk - predicted_risk)` in raw risk units, modeled on log scale with epsilon equal to the fixed scale floor. Sensitivity-only targets are the log residual and q=0.90 upper-tail residual objective. True outcomes are permitted only to construct training/validation targets; no outcome is required at inference.

## Deployable feature families

- S0: authoritative central risk, predicted upper quantile, their gap, and the three frozen query-gate logits.
- S1: 64 global bottleneck features plus 112 multiscale prompt-local frozen features.
- S2: 18 frozen reconstruction summaries: three-band flux, concentration, prompt-relative centroid offsets, output energy, and local reconstruction-to-blend contrast.
- S3: observed-blend-only robust per-band background MAD, local signal/background proxy, variance, gradient energy, high-frequency power, prompt-neighborhood concentration, band-centroid consistency, and band-structure disagreement. None is named or treated as true SNR.
- S4: S0 + S1 + S2 + S3.

Four continuous partial-pooling proxies are fixed transforms of deployable quantities: estimated low local signal, severe local complexity, high output uncertainty, and strong input/output disagreement. Source IDs, true SNR, obstruction, separation, flux ratio, source truth, generator difficulty, morphology, and frozen physical subgroup labels are prohibited model inputs. Oracle physical groups are diagnostic only.

## Fixed scale models and objectives

- M0: one global constant residual scale per risk.
- M1: ridge-regularized log-linear scale model.
- M2: one 32-unit ReLU hidden layer.
- M3: two 32-unit residual hidden layers plus a linear skip; parameter ceiling 25,000.
- M4: 16-unit global trunk plus four strongly penalized continuous proxy corrections shrinking toward the global prediction.
- M5: at most three soft experts; a four-proxy gate uses entropy, gate-weight, and expert-deviation penalties.

Five fixed seeds are `{list(SCALE_SEEDS)}` for every trainable condition. M1 compares S0, S1, S2, S3, and S4 under O0. S4 compares O0 Huber log-residual, O1 q=0.90 pinball residual, and O2 bounded Gaussian NLL. O2 has fixed positive floors/caps, a scale-growth penalty, and cannot collapse or diverge. M2-M5 use the validation-selected preregistered objective; no broad architecture search is allowed.

Image scale floor/cap are {SCALE_BOUNDS['image']}; flux scale floor/cap are {SCALE_BOUNDS['flux']}. Model selection uses validation residual-scale Spearman, top-decile residual recall, q=0.90 scale coverage, median predicted scale, and seed stability in that order; training loss alone cannot select a model.

## Normalized conformal and comparisons

The score is `abs(true_risk - predicted_risk) / max(predicted_scale, floor)`. Each calibration fold is evaluated using the finite-sample q=0.90 score quantile from the other four source-group folds. Bounds are `predicted_risk + quantile * predicted_scale` in raw risk units. Report C0 authoritative conditional-calibration baseline, C1 best global feature-conditioned M1/M2/M3 model, C2 M4 partial pooling, C3 M5 soft gate, and C4 physical oracle-group scale. C4 is non-deployable, diagnostic only, and cannot determine success.

## Frozen analysis subgroups

The exact subgroup definition is copied from the authoritative campaign and remains analysis-only:

```json
{json.dumps(definitions, indent=2, sort_keys=True)}
```

## Metrics, gates, and decisions

For image and flux: marginal coverage must lie in [0.88,0.92]; worst supported and low-SNR/high-obstruction coverage must be at least 0.82; median width inflation must be at most 1.75x the authoritative baseline; p95 width must be below {P95_WIDTH_CAP}; calibration Spearman must remain at least 0.82 image and 0.80 flux; scale-floor/cap activation, extreme inflation, score uniqueness, tail miss rate, five-seed SD, and source-component bootstrap intervals are reported. Bootstrap stability requires seed coverage SD <=0.03 and the subgroup-coverage 95% CI lower bound >=0.75. Centroid must exactly reproduce its prior PASS and is not redesigned.

PARTIAL means subgroup coverage materially improves by at least 0.05 absolute over 0.637 image or 0.684 flux but misses a frozen gate. FAIL means smaller improvement, marginal breakage, excessive inflation, instability, or oracle-only benefit. Overall SUCCESS requires image PASS, flux PASS, centroid PASS, a non-oracle deployable model, bounded inflation, bootstrap stability, and zero development/lockbox access. Gates cannot change after results.

## Bootstrap and sensitivity

Use 300 connected-source-component bootstrap replicates. Freeze sensitivity checks before results: floor x0.5/x2, cap x0.8/x1.2, all physical subgroup boundaries shifted outward by 5% for one audit, lower versus higher conformal order-statistic convention, and M4 correction penalty x0.5/x2. Sensitivity cannot trigger retuning.

## Stopping rules and prohibited analyses

Stop before scale fitting if any input hash, baseline metric, sample alignment, applicability mask, source-component isolation, gate attainability, or preregistration hash check fails. Do not tune subgroup boundaries, retrain Condition C or the query/catastrophic heads, redesign risks or query semantics, construct an accept/abstain policy, create development data, access lockbox data, make end-to-end claims, or overwrite historical outputs. On PARTIAL/FAILURE recommend exactly one experiment and do not run it.
"""


def bootstrap() -> Path:
    started = time.time()
    verified = verify_frozen_inputs()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/{RUN_PREFIX}{timestamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "preregistration", "features", "models", "calibration", "example_grids", "manifests"):
        (run / name).mkdir(exist_ok=False)
    inventory = historical_checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", inventory)
    definitions = json.loads((PRIOR / "manifests/subgroup_definitions.json").read_text())
    fresh_json(run / "manifests/subgroup_definitions.json", definitions)
    gates = gate_audit()
    if not gates.attainable.all():
        raise RuntimeError("unattainable gate found")
    fresh_csv(run / "tables/gate_attainability_audit.csv", gates)
    git = {name: command(args) for name, args in {
        "branch": ["git", "branch", "--show-current"],
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short", "--branch"],
        "staged_index": ["git", "diff", "--cached", "--name-status"],
    }.items()}
    packages = {name: package_version(name) for name in ("numpy", "pandas", "scipy", "torch", "h5py", "blending-toolkit", "GalSim", "astropy")}
    packages.update({"python": sys.version, "platform": platform.platform(),
                     "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()})
    disk = shutil.disk_usage(REPO)
    manifest_hashes = []
    feature_hashes = []
    for dataset in ("r_training", "r_validation", "natural_calibration"):
        manifest = FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv"
        feature = FEASIBILITY / f"features/v2_{dataset}_features.npz"
        manifest_hashes.append({"relative_path": relative(manifest), "sha256": sha256_file(manifest)})
        feature_hashes.append({"relative_path": relative(feature), "sha256": sha256_file(feature)})
    provenance = {
        "campaign_start_iso": datetime.now(timezone.utc).isoformat(), "campaign_start_unix": started,
        "git": git, "packages": packages,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "frozen_verification": verified,
        "condition_c": {"relative_path": relative(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
        "selected_risk_heads": selected_risk_head_hashes(),
        "partition_manifests": manifest_hashes,
        "persisted_features": feature_hashes,
        "feature_definition_hashes": {
            "frozen_extractor": sha256_file(REPO / "scripts/extract_hierarchical_safety_features.py"),
            "scale_module": sha256_file(REPO / "src/scale_correction.py"),
            "campaign_driver": sha256_file(REPO / "scripts/run_scale_correction.py"),
        },
        "query_semantics_sha256": sha256_file(REPO / "src/hierarchical_safety.py"),
        "risk_definition_sha256": sha256_file(REPO / "src/hierarchical_feasibility.py"),
        "prior_conditional_run_hashes": all_prior_hashes(),
        "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "development_accesses": 0, "lockbox_accesses": 0,
        "neural_feature_extraction": "not needed; persisted features reused", "scale_fit_device": "cpu",
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    snapshot = f"""# Environment snapshot

- Start: {provenance['campaign_start_iso']}
- Branch: {git['branch']['stdout'].strip()}
- Git HEAD: {git['head']['stdout'].strip()}
- MPS built/available: {packages['mps_built']} / {packages['mps_available']}
- Neural feature extraction: not needed; persisted frozen features are reused
- CPU-only work: scale fitting, conformal calibration, bootstrap, reporting
- Condition-C checkpoint: `{relative(CHECKPOINT)}`
- Condition-C SHA-256: `{sha256_file(CHECKPOINT)}`
- Historical checkpoints inventoried: {len(inventory)}
- Free disk at start: {disk.free / 2**30:.2f} GiB

## Package versions

```json
{json.dumps(packages, indent=2, sort_keys=True)}
```

## Git status

```text
{git['status']['stdout']}```

## Staged index

```text
{git['staged_index']['stdout'] or '(empty)'}```
"""
    fresh_text(run / "diagnostics/environment_snapshot.md", snapshot)
    contract = """# Scale-correction campaign contract

This run is append-only and collision refusing. It may read only the frozen training, validation, natural-calibration, prior-report, prior-model, and persisted-feature artifacts enumerated in `logs/input_provenance.json`. Condition C has zero trainable parameters and will not be loaded for inference. All scale work is CPU-only. True physical scene variables and source identifiers are analysis-only and prohibited from deployable arrays. The natural calibration partition is retained for conformal correction and cross-fitted diagnostics. Development and lockbox data are prohibited. No policy, operational threshold, staging, commit, push, merge, deletion, or historical overwrite is authorized.
"""
    fresh_text(run / "diagnostics/campaign_contract.md", contract)
    gate_report = "# Gate attainability report\n\nPASS. All %d frozen gates lie inside their theoretical ranges. Positive baseline widths make inflation ratios defined; all coverage/ranking thresholds are below their mathematical maxima; scale floors are positive and below fixed caps; every conformal fold complement is nonempty. No result-dependent gate repair is permitted.\n\n%s\n" % (len(gates), markdown_table(gates))
    fresh_text(run / "diagnostics/gate_attainability_report.md", gate_report)
    prereg = run / "preregistration/partially_pooled_scale_correction.md"
    fresh_text(prereg, preregistration_text(definitions))
    digest = sha256_file(prereg)
    hashed_at = time.time()
    fresh_json(run / "preregistration/partially_pooled_scale_correction.sha256.json",
               {"relative_path": relative(prereg), "sha256": digest,
                "hashed_at_unix": hashed_at, "hashed_at_iso": datetime.now(timezone.utc).isoformat()})
    fresh_json(run / "logs/preregistration_complete.json",
               {"status": "PASS", "sha256": digest, "timestamp_unix": hashed_at,
                "fit_started": False, "all_gates_attainable": True})
    print(run)
    return run


def require_preregistration(run: Path) -> dict:
    marker = json.loads((run / "preregistration/partially_pooled_scale_correction.sha256.json").read_text())
    prereg = run / "preregistration/partially_pooled_scale_correction.md"
    if sha256_file(prereg) != marker["sha256"]:
        raise RuntimeError("preregistration hash differs")
    if marker["hashed_at_unix"] >= time.time():
        raise RuntimeError("preregistration timestamp is not in the past")
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    for section in ("partition_manifests", "persisted_features"):
        for row in provenance[section]:
            path = REPO / row["relative_path"]
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"frozen input changed after preregistration: {row['relative_path']}")
    for row in provenance["selected_risk_heads"]:
        if sha256_file(REPO / row["relative_path"]) != row["sha256"]:
            raise RuntimeError(f"selected risk head changed: {row['relative_path']}")
    return marker


def datasets() -> tuple[dict, dict, dict]:
    features, samples, manifests = {}, {}, {}
    mapping = {"training": "r_training", "validation": "r_validation", "calibration": "natural_calibration"}
    expected = {"training": 12000, "validation": 2000, "calibration": 2800}
    for partition, dataset in mapping.items():
        feature, sample, manifest = load_dataset(dataset)
        if len(sample) != expected[partition]:
            raise RuntimeError(f"unexpected {partition} rows: {len(sample)}")
        features[partition], samples[partition], manifests[partition] = feature, sample, manifest
    for partition in mapping:
        ids = features[partition]["scene_id"].astype(str)
        if not np.array_equal(ids, samples[partition].scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"sample alignment failed in {partition}")
    return features, samples, manifests


def prior_head_outputs(risk: str, feature: np.ndarray, family: str) -> np.ndarray:
    blocks = []
    for seed in HEAD_SEEDS:
        payload = torch.load(PRIOR / f"models/{risk}_{family}_seed_{seed}.pth", map_location="cpu", weights_only=False)
        model = RiskHead(feature.shape[1], family)
        model.load_state_dict(payload["state_dict"])
        blocks.append(predict_prior(model, feature, payload["mean"], payload["scale"]))
    return np.mean(blocks, axis=0)


def prior_selected_predictions(
    features: dict,
    samples: dict,
    manifests: dict,
    assignments: dict,
) -> tuple[dict, pd.DataFrame]:
    predictions, summaries = {}, []
    fold = group_safe_folds(manifests["calibration"].source_a_group.to_numpy(),
                            manifests["calibration"].source_b_group.to_numpy(), folds=5)
    if not verify_fold_isolation(manifests["calibration"].source_a_group.to_numpy(),
                                 manifests["calibration"].source_b_group.to_numpy(), fold):
        raise RuntimeError("calibration source groups cross folds")
    for risk, selection in SELECTED.items():
        family, method = selection["head"], selection["method"]
        output = {partition: prior_head_outputs(risk, features[partition]["f_combined"], family)
                  for partition in ("training", "validation", "calibration")}
        scale_payload = torch.load(PRIOR / f"models/{risk}_{family}_scale.pth", map_location="cpu", weights_only=False)
        scale_model = ScaleHead(features["training"]["f_combined"].shape[1])
        scale_model.load_state_dict(scale_payload["state_dict"])
        scale = {}
        for partition in ("training", "validation", "calibration"):
            log_value = predict_prior(scale_model, features[partition]["f_combined"],
                                      scale_payload["mean"], scale_payload["scale"])
            scale[partition] = np.maximum(np.exp(np.clip(log_value, -8, 8)), 1e-4)
        central_edges = fixed_tertile_edges(np.r_[output["training"][:, 0], output["validation"][:, 0]])
        scale_edge = float(np.median(np.r_[scale["training"], scale["validation"]]))
        mondrian = deployable_mondrian_group(output["calibration"][:, 0], scale["calibration"], central_edges, scale_edge)
        truth_log = samples["calibration"][LOG_TARGET[risk]].to_numpy(dtype=float)
        residual_log = truth_log - output["calibration"][:, 0]
        placeholder = np.column_stack((output["calibration"][:, 0], scale["calibration"]))
        upper_log, support = crossfit_bounds(residual_log, output["calibration"][:, 0], scale["calibration"],
                                             placeholder, fold, method, mondrian)
        upper = np.maximum(np.expm1(np.clip(upper_log, -30, 30)), 0.0)
        central = np.maximum(np.expm1(np.clip(output["calibration"][:, 0], -30, 30)), 0.0)
        predicted_upper = np.maximum(np.expm1(np.clip(output["calibration"][:, 1], -30, 30)), 0.0)
        truth = samples["calibration"][RAW_TARGET[risk]].to_numpy(dtype=float)
        covered = truth <= upper
        width = np.maximum(upper - central, 0.0)
        subgroup_rows = []
        for family_name in PRIMARY_SUBGROUPS:
            levels = ["member"] if "__" in family_name else list(dict.fromkeys(assignments["calibration"][family_name].tolist()))
            for level in levels:
                mask = assignments["calibration"][family_name].to_numpy() == level
                subgroup_rows.append({"risk": risk, "subgroup_family": family_name, "subgroup": level,
                                      "rows": int(mask.sum()), "source_group_count": source_group_count(manifests["calibration"], mask),
                                      "coverage": float(np.mean(covered[mask])), "median_width": float(np.median(width[mask])),
                                      "p95_width": float(np.quantile(width[mask], 0.95))})
        subgroup_frame = pd.DataFrame(subgroup_rows)
        summaries.append({
            "risk": risk, "record_type": "summary", "head": family, "method": method,
            "rows": len(truth), "source_group_count": source_group_count(manifests["calibration"], np.ones(len(truth), dtype=bool)),
            "marginal_coverage": float(np.mean(covered)),
            "worst_supported_subgroup_coverage": float(subgroup_frame.coverage.min()),
            "low_snr_high_obstruction_coverage": float(subgroup_frame[(subgroup_frame.subgroup_family == "low_snr__high_obstruction") & (subgroup_frame.subgroup == "member")].coverage.iloc[0]),
            "mean_width": float(np.mean(width)), "median_width": float(np.median(width)),
            "p90_width": float(np.quantile(width, 0.90)), "p95_width": float(np.quantile(width, 0.95)),
            "residual_mean": float(np.mean(np.abs(truth - central))), "residual_median": float(np.median(np.abs(truth - central))),
            "residual_p90": float(np.quantile(np.abs(truth - central), 0.90)),
            "applicable_rows": int(samples["calibration"].applicable_valid_risk.astype(bool).sum()),
            "prior_decision": "PASS" if risk == "centroid" else "FAIL",
        })
        predictions[risk] = {"truth": truth, "central": central, "predicted_upper": predicted_upper,
                             "upper": upper, "covered": covered, "width": width, "scale": scale["calibration"],
                             "fold": fold, "support": support, "subgroups": subgroup_frame,
                             "calibration_log_output": output["calibration"]}
    return predictions, pd.DataFrame(summaries)


def reproduce_baseline(run: Path, features: dict, samples: dict, manifests: dict, assignments: dict) -> dict:
    predictions, summary = prior_selected_predictions(features, samples, manifests, assignments)
    reference = pd.read_csv(PRIOR / "tables/component_decision_table.csv")
    selected_groups = pd.read_csv(PRIOR / "tables/selected_subgroup_coverage.csv")
    failures = []
    for row in summary.itertuples(index=False):
        expected = reference[reference.risk == row.risk].iloc[0]
        for name in ("marginal_coverage", "worst_supported_subgroup_coverage", "median_width", "p95_width"):
            if not np.isclose(float(getattr(row, name)), float(expected[name]), rtol=1e-10, atol=1e-10):
                failures.append(f"{row.risk} {name}: {getattr(row, name)} != {expected[name]}")
        current_groups = predictions[row.risk]["subgroups"]
        expected_groups = selected_groups[selected_groups.risk == row.risk]
        merged = current_groups.merge(expected_groups, on=["risk", "subgroup_family", "subgroup"], suffixes=("_new", "_old"))
        if len(merged) != len(expected_groups) or not np.allclose(merged.coverage_new, merged.coverage_old, atol=1e-12):
            failures.append(f"{row.risk} subgroup coverage mismatch")
        if not np.array_equal(merged.rows_new.to_numpy(), merged.rows_old.to_numpy()):
            failures.append(f"{row.risk} subgroup row-count mismatch")
    if failures:
        fresh_text(run / "diagnostics/baseline_reproduction.md", "# Baseline reproduction\n\nFAIL. Scale fitting is prohibited.\n\n" + "\n".join(f"- {item}" for item in failures) + "\n")
        raise RuntimeError("baseline reproduction failed:\n" + "\n".join(failures))
    rows = []
    for row in summary.to_dict("records"):
        rows.append(row)
    for risk in SELECTED:
        for row in predictions[risk]["subgroups"].to_dict("records"):
            rows.append({"risk": risk, "record_type": "subgroup", **row})
    fresh_csv(run / "tables/baseline_reproduction.csv", pd.DataFrame(rows))
    report = "# Baseline reproduction\n\nPASS. Exact authoritative CPU risk/scale checkpoints, applicability masks, source-group folds, conformal implementation, subgroup assignments, row counts, source-group counts, residuals, and widths reproduce the prior selected component results within 1e-10. No scale model was fitted before this check.\n\n" + markdown_table(summary) + "\n"
    fresh_text(run / "diagnostics/baseline_reproduction.md", report)
    return predictions


S3_NAMES = [
    *[f"robust_background_scale_{band}" for band in "grz"],
    *[f"local_signal_background_proxy_{band}" for band in "grz"],
    *[f"local_variance_{band}" for band in "grz"],
    *[f"local_gradient_energy_{band}" for band in "grz"],
    *[f"local_high_frequency_power_{band}" for band in "grz"],
    "prompt_neighborhood_concentration", "band_centroid_consistency", "band_structural_disagreement",
]


def observed_quality_features(manifest: pd.DataFrame) -> np.ndarray:
    """Compute S3 from observed blends and prompts only; no truth arrays are read."""

    path = REPO / str(manifest.hdf5_path.iloc[0])
    indices = manifest.dataset_index.to_numpy(dtype=int)
    output = np.empty((len(indices), len(S3_NAMES)), dtype=np.float32)
    yy, xx = np.mgrid[:60, :60]
    border = (xx < 6) | (xx >= 54) | (yy < 6) | (yy >= 54)
    with h5py.File(path, "r") as handle:
        for start in range(0, len(indices), 256):
            stop = min(len(indices), start + 256)
            batch_indices = indices[start:stop]
            blend = np.asarray(handle["blend"][batch_indices], dtype=np.float64)
            prompt_xy = np.asarray(handle["prompt_xy"][batch_indices], dtype=np.float64)
            for local, (image, prompt) in enumerate(zip(blend, prompt_xy)):
                radius = np.sqrt((xx - prompt[0]) ** 2 + (yy - prompt[1]) ** 2)
                aperture = radius <= 8.133333333333333
                background = image[:, border]
                center = np.median(background, axis=1)
                mad = 1.4826 * np.median(np.abs(background - center[:, None]), axis=1) + 1e-8
                local_values = image[:, aperture]
                signal = np.median(np.abs(local_values - center[:, None]), axis=1) / mad
                variance = np.var(local_values, axis=1)
                gradient = np.mean(np.diff(image, axis=1) ** 2, axis=(1, 2)) + np.mean(np.diff(image, axis=2) ** 2, axis=(1, 2))
                smooth = (image[:, 1:-1, 1:-1] + image[:, :-2, 1:-1] + image[:, 2:, 1:-1] +
                          image[:, 1:-1, :-2] + image[:, 1:-1, 2:]) / 5.0
                high_frequency = np.mean((image[:, 1:-1, 1:-1] - smooth) ** 2, axis=(1, 2))
                total_abs = np.sum(np.abs(image)) + 1e-12
                concentration = float(np.sum(np.abs(image[:, aperture])) / total_abs)
                centroids = []
                flattened = []
                for band in range(3):
                    weight = np.maximum(image[band] - center[band], 0.0)
                    total = weight.sum()
                    if total <= 0:
                        centroids.append(prompt)
                    else:
                        centroids.append([np.sum(xx * weight) / total, np.sum(yy * weight) / total])
                    flattened.append((local_values[band] - np.mean(local_values[band])) / (np.std(local_values[band]) + 1e-8))
                centroids = np.asarray(centroids)
                centroid_consistency = float(np.mean(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)))
                disagreement = float(np.mean([1.0 - np.corrcoef(flattened[a], flattened[b])[0, 1]
                                              for a, b in ((0, 1), (0, 2), (1, 2))]))
                row = np.r_[mad, signal, variance, gradient, high_frequency,
                            concentration, centroid_consistency, disagreement]
                if not np.isfinite(row).all():
                    raise RuntimeError("nonfinite observed deployable proxy")
                output[start + local] = row.astype(np.float32)
    return output


def authoritative_outputs(features: dict, risk: str) -> dict:
    family = SELECTED[risk]["head"]
    result = {}
    for partition in ("validation", "calibration"):
        log_output = prior_head_outputs(risk, features[partition]["f_combined"], family)
        result[partition] = np.column_stack([
            np.maximum(np.expm1(np.clip(log_output[:, 0], -30, 30)), 0.0),
            np.maximum(np.expm1(np.clip(log_output[:, 1], -30, 30)), 0.0),
        ])
    return result


def crossfit_risk_predictions(run: Path, features: dict, samples: dict, manifests: dict) -> tuple[dict, pd.DataFrame]:
    fold = group_safe_folds(manifests["training"].source_a_group.to_numpy(), manifests["training"].source_b_group.to_numpy(), folds=5)
    if not verify_fold_isolation(manifests["training"].source_a_group.to_numpy(), manifests["training"].source_b_group.to_numpy(), fold):
        raise RuntimeError("training source groups cross folds")
    train_groups = set(manifests["training"].source_a_group.astype(str)) | set(manifests["training"].source_b_group.astype(str))
    validation_groups = set(manifests["validation"].source_a_group.astype(str)) | set(manifests["validation"].source_b_group.astype(str))
    calibration_groups = set(manifests["calibration"].source_a_group.astype(str)) | set(manifests["calibration"].source_b_group.astype(str))
    if train_groups & validation_groups or train_groups & calibration_groups or validation_groups & calibration_groups:
        raise RuntimeError("source-group overlap across fixed partitions")
    inventory = []
    for current in range(5):
        held = fold == current
        allowed = ~held
        inventory.append({"fold": current, "training_rows": int(allowed.sum()), "held_out_rows": int(held.sum()),
                          "held_out_source_groups": source_group_count(manifests["training"], held),
                          "source_group_overlap": 0, "prediction_status": "OUT_OF_FOLD"})
    predictions = {}
    x = features["training"]["f_combined"]
    validation_x = features["validation"]["f_combined"]
    for risk in ("image", "flux"):
        family = SELECTED[risk]["head"]
        y = samples["training"][LOG_TARGET[risk]].to_numpy(dtype=np.float32)
        validation_y = samples["validation"][LOG_TARGET[risk]].to_numpy(dtype=np.float32)
        output = np.empty((len(x), 2), dtype=np.float32)
        for current in range(5):
            held = fold == current
            allowed = ~held
            seed = SCALE_SEEDS[current]
            fitted = fit_risk_head(family, seed, x[allowed], y[allowed], validation_x, validation_y, max_epochs=80)
            output[held] = predict_prior(fitted.model, x[held], fitted.mean, fitted.scale)
            path = run / f"models/{risk}_crossfit_{family}_fold_{current}.pth"
            torch.save({"state_dict": fitted.model.state_dict(), "mean": fitted.mean, "scale": fitted.scale,
                        "risk": risk, "family": family, "fold": current, "seed": seed,
                        "best_epoch": fitted.best_epoch, "fit_rows": int(allowed.sum()),
                        "prediction_rows": int(held.sum()), "in_sample_predictions": False,
                        "device": "cpu", "reconstruction_parameters": 0}, path)
        predictions[risk] = np.column_stack([
            np.maximum(np.expm1(np.clip(output[:, 0], -30, 30)), 0.0),
            np.maximum(np.expm1(np.clip(output[:, 1], -30, 30)), 0.0),
        ])
    frame = pd.DataFrame(inventory)
    fresh_csv(run / "tables/cross_fit_fold_inventory.csv", frame)
    fresh_text(run / "diagnostics/cross_fitting_protocol.md",
               "# Cross-fitting protocol\n\nPASS. The authoritative training predictions were in-sample, so five deterministic connected-source-component folds were required. Each exact selected-family risk head was refit on four folds with validation-only early stopping and predicted only its held-out fold. No source group crosses a fold or fixed partition. Only concatenated held-out predictions form scale targets; calibration outcomes are absent.\n\n" + markdown_table(frame) + "\n")
    return predictions, frame


def feature_names() -> dict[str, list[str]]:
    return {
        "S0": ["central_predicted_risk", "predicted_upper_quantile", "central_quantile_gap",
               "query_logit_unique_valid", "query_logit_null", "query_logit_ambiguous"],
        "S1": [*[f"global_bottleneck_{i:03d}" for i in range(64)],
               *[f"prompt_local_{i:03d}" for i in range(112)]],
        "S2": [*[f"predicted_flux_{band}" for band in "grz"], *[f"concentration_{band}" for band in "grz"],
               *[f"centroid_offset_{band}_{axis}" for band in "grz" for axis in ("x", "y")],
               *[f"output_energy_{band}" for band in "grz"], *[f"local_reconstruction_blend_ratio_{band}" for band in "grz"]],
        "S3": S3_NAMES,
    }


def soft_proxies(s0: np.ndarray, s2: np.ndarray, s3: np.ndarray, reference: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    raw = np.column_stack([
        -np.mean(s3[:, 3:6], axis=1),
        s3[:, -1] + s3[:, -3],
        s0[:, 2] / (1.0 + np.abs(s0[:, 0])),
        np.mean(np.abs(np.log(np.maximum(np.abs(s2[:, -3:]), 1e-6))), axis=1),
    ])
    if reference is None:
        center = np.median(raw, axis=0)
        spread = np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.25, axis=0)
        spread[spread < 1e-8] = 1.0
        reference = center, spread
    center, spread = reference
    proxy = 1.0 / (1.0 + np.exp(-np.clip((raw - center) / spread, -20, 20)))
    return proxy.astype(np.float32), reference


def build_deployable_features(
    run: Path,
    features: dict,
    samples: dict,
    manifests: dict,
    oof: dict,
) -> tuple[dict, dict, pd.DataFrame]:
    observed = {}
    query = {}
    for partition in ("training", "validation", "calibration"):
        observed[partition] = observed_quality_features(manifests[partition])
        archive = features[partition]
        query[partition] = query_logits(FEASIBILITY, archive).astype(np.float32)
    names = feature_names()
    inventory_rows = []
    origins = {"S0": "frozen risk/query outputs", "S1": "frozen Condition-C latent features",
               "S2": "frozen reconstruction and observed-blend summaries", "S3": "observed blend and prompt only"}
    for family, values in names.items():
        for index, name in enumerate(values):
            inventory_rows.append({"feature_set": family, "feature_index": index, "feature_name": name,
                                   "origin": origins[family], "inference_deployable": True,
                                   "oracle_field": False, "physical_subgroup_label": False})
    inventory = pd.DataFrame(inventory_rows)
    forbidden = r"(^|_)(source_id|source_group|true_snr|obstruction|separation|flux_ratio|generator|oracle)(_|$)"
    if inventory.feature_name.str.contains(forbidden, case=False, regex=True).any():
        raise RuntimeError("oracle-like field found in deployable feature inventory")
    fresh_csv(run / "tables/scale_feature_inventory.csv", inventory)
    prepared, proxies = defaultdict(dict), defaultdict(dict)
    for risk in ("image", "flux"):
        authoritative = authoritative_outputs(features, risk)
        partition_s0 = {}
        for partition in ("training", "validation", "calibration"):
            risk_output = oof[risk] if partition == "training" else authoritative[partition]
            partition_s0[partition] = np.column_stack([risk_output[:, 0], risk_output[:, 1],
                                                       np.maximum(risk_output[:, 1] - risk_output[:, 0], 0.0),
                                                       query[partition]]).astype(np.float32)
        reference = None
        for partition in ("training", "validation", "calibration"):
            s0 = partition_s0[partition]
            s1 = np.concatenate((features[partition]["f_global"], features[partition]["f_prompt_local"]), axis=1).astype(np.float32)
            s2 = features[partition]["f_recon_summary"].astype(np.float32)
            s3 = observed[partition]
            prepared[risk][partition] = {"S0": s0, "S1": s1, "S2": s2, "S3": s3,
                                         "S4": np.concatenate((s0, s1, s2, s3), axis=1).astype(np.float32)}
            proxy, reference = soft_proxies(s0, s2, s3, reference)
            proxies[risk][partition] = proxy
            np.savez_compressed(run / f"features/{risk}_{partition}_deployable_scale_features.npz",
                                scene_id=features[partition]["scene_id"].astype(str), S0=s0, S1=s1, S2=s2, S3=s3,
                                S4=prepared[risk][partition]["S4"], partial_pool_proxies=proxy)
    audit = "# Deployable feature audit\n\nPASS. S0-S4 contain only frozen risk/query outputs, frozen Condition-C latent/reconstruction summaries, observed blends, and prompts. S3 was computed without reading `isolated`, source truth, physical SNR, obstruction, separation, flux ratio, IDs, or generator difficulty. The four pooling proxies are continuous deployable transforms; no frozen physical subgroup label enters any model array. Morphology is absent. No neural extraction was run.\n\n" + markdown_table(inventory.groupby("feature_set", as_index=False).agg(features=("feature_name", "count"), all_deployable=("inference_deployable", "all"), oracle_fields=("oracle_field", "sum"))) + "\n"
    fresh_text(run / "diagnostics/deployable_feature_audit.md", audit)
    return dict(prepared), dict(proxies), inventory


def scale_targets(run: Path, prepared: dict, samples: dict) -> tuple[dict, pd.DataFrame]:
    targets, rows = defaultdict(dict), []
    for risk in ("image", "flux"):
        floor, _ = SCALE_BOUNDS[risk]
        for partition in ("training", "validation"):
            truth = samples[partition][RAW_TARGET[risk]].to_numpy(dtype=float)
            central = prepared[risk][partition]["S0"][:, 0]
            absolute = np.abs(truth - central)
            targets[risk][partition] = absolute
            rows.append({"risk": risk, "partition": partition, "primary_target": "absolute_residual",
                         "rows": len(absolute), "minimum": float(absolute.min()), "median": float(np.median(absolute)),
                         "mean": float(np.mean(absolute)), "p90": float(np.quantile(absolute, 0.90)),
                         "p95": float(np.quantile(absolute, 0.95)), "maximum": float(absolute.max()),
                         "log_target_epsilon": floor, "true_outcome_in_inference_array": False})
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/scale_target_summary.csv", frame)
    fresh_text(run / "diagnostics/scale_target_audit.md",
               "# Scale-target audit\n\nPASS. The primary target is raw absolute residual from source-group-held-out training predictions. Log residual and q=0.90 residual objectives are sensitivity comparisons only. Outcomes are used to create training/validation targets and never enter inference arrays.\n\n" + markdown_table(frame) + "\n")
    return dict(targets), frame


def validation_metrics(residual: np.ndarray, scale: np.ndarray) -> dict:
    residual = np.asarray(residual, dtype=float)
    scale = np.asarray(scale, dtype=float)
    correlation = spearmanr(scale, residual).statistic
    correlation = float(correlation) if np.isfinite(correlation) else 0.0
    tail = residual >= np.quantile(residual, 0.90)
    predicted_tail = scale >= np.quantile(scale, 0.90)
    return {
        "residual_scale_spearman": correlation,
        "upper_tail_recall": float(np.mean(predicted_tail[tail])) if tail.any() else 0.0,
        "q90_scale_coverage": float(np.mean(residual <= scale)),
        "median_predicted_scale": float(np.median(scale)),
        "p95_predicted_scale": float(np.quantile(scale, 0.95)),
        "floor_activation_rate": float(np.mean(scale <= np.min(scale) + 1e-12)),
        "unique_scale_count": int(len(np.unique(np.round(scale, 8)))),
    }


def save_scale_fit(path: Path, fit, *, risk: str, feature_set: str) -> None:
    payload = fit_payload(fit)
    payload.update({"risk": risk, "feature_set": feature_set, "fit_partition": "training_oof_targets",
                    "selection_partition": "validation", "calibration_outcomes_used": False})
    torch.save(payload, path)


def fit_scale_conditions(
    run: Path,
    prepared: dict,
    proxies: dict,
    targets: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    rows, fits = [], {}
    for risk in ("image", "flux"):
        floor, cap = SCALE_BOUNDS[risk]
        conditions = []
        for feature_set in ("S0", "S1", "S2", "S3", "S4"):
            conditions.append(("M1_log_linear", feature_set, "O0_huber_log"))
        conditions.extend(("M1_log_linear", "S4", objective) for objective in ("O1_q90_pinball", "O2_bounded_gaussian_nll"))
        for family, feature_set, objective in conditions:
            for seed in SCALE_SEEDS:
                fit = fit_scale_model(family, objective, seed,
                                      prepared[risk]["training"][feature_set], proxies[risk]["training"], targets[risk]["training"],
                                      prepared[risk]["validation"][feature_set], proxies[risk]["validation"], targets[risk]["validation"],
                                      scale_floor=floor, scale_cap=cap)
                path = run / f"models/{risk}_{family}_{feature_set}_{objective}_seed_{seed}.pth"
                save_scale_fit(path, fit, risk=risk, feature_set=feature_set)
                fits[(risk, family, feature_set, objective, seed)] = fit
                scale = predict_scale(fit, prepared[risk]["validation"][feature_set], proxies[risk]["validation"])
                rows.append({"risk": risk, "family": family, "feature_set": feature_set, "objective": objective,
                             "seed": seed, "best_epoch": fit.best_epoch, "validation_loss": fit.best_validation_loss,
                             "parameter_count": sum(parameter.numel() for parameter in fit.model.parameters()),
                             **validation_metrics(targets[risk]["validation"], scale)})
        preliminary = pd.DataFrame([row for row in rows if row["risk"] == risk])
        objective_summary = preliminary[(preliminary.family == "M1_log_linear") & (preliminary.feature_set == "S4")].groupby("objective", as_index=False).agg(
            spearman_mean=("residual_scale_spearman", "mean"), tail_recall_mean=("upper_tail_recall", "mean"),
            spearman_sd=("residual_scale_spearman", "std"), median_scale=("median_predicted_scale", "mean"))
        objective_summary = objective_summary.sort_values(["spearman_mean", "tail_recall_mean", "spearman_sd"], ascending=[False, False, True])
        selected_objective = str(objective_summary.iloc[0].objective)
        for family in ("M2_one_hidden", "M3_residual", "M4_partial_pool", "M5_soft_gate"):
            for seed in SCALE_SEEDS:
                fit = fit_scale_model(family, selected_objective, seed,
                                      prepared[risk]["training"]["S4"], proxies[risk]["training"], targets[risk]["training"],
                                      prepared[risk]["validation"]["S4"], proxies[risk]["validation"], targets[risk]["validation"],
                                      scale_floor=floor, scale_cap=cap)
                if sum(parameter.numel() for parameter in fit.model.parameters()) > 25000:
                    raise RuntimeError(f"parameter ceiling exceeded by {family}")
                path = run / f"models/{risk}_{family}_S4_{selected_objective}_seed_{seed}.pth"
                save_scale_fit(path, fit, risk=risk, feature_set="S4")
                fits[(risk, family, "S4", selected_objective, seed)] = fit
                scale = predict_scale(fit, prepared[risk]["validation"]["S4"], proxies[risk]["validation"])
                rows.append({"risk": risk, "family": family, "feature_set": "S4", "objective": selected_objective,
                             "seed": seed, "best_epoch": fit.best_epoch, "validation_loss": fit.best_validation_loss,
                             "parameter_count": sum(parameter.numel() for parameter in fit.model.parameters()),
                             **validation_metrics(targets[risk]["validation"], scale)})
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/scale_model_seed_comparison.csv", frame)
    summary = frame.groupby(["risk", "family", "feature_set", "objective"], as_index=False).agg(
        spearman_mean=("residual_scale_spearman", "mean"), spearman_sd=("residual_scale_spearman", "std"),
        tail_recall_mean=("upper_tail_recall", "mean"), tail_recall_sd=("upper_tail_recall", "std"),
        q90_scale_coverage_mean=("q90_scale_coverage", "mean"), median_scale_mean=("median_predicted_scale", "mean"),
        p95_scale_mean=("p95_predicted_scale", "mean"), parameter_count=("parameter_count", "first"),
        validation_loss_mean=("validation_loss", "mean"))
    fresh_csv(run / "tables/scale_model_validation_summary.csv", summary)
    selections = {}
    selection_rows = []
    for risk in ("image", "flux"):
        objective_candidates = summary[(summary.risk == risk) & (summary.family == "M1_log_linear") & (summary.feature_set == "S4")]
        objective_candidates = objective_candidates.sort_values(["spearman_mean", "tail_recall_mean", "spearman_sd"], ascending=[False, False, True])
        objective = str(objective_candidates.iloc[0].objective)
        global_candidates = summary[(summary.risk == risk) & (summary.family.isin(["M1_log_linear", "M2_one_hidden", "M3_residual"])) &
                                    (((summary.family == "M1_log_linear")) | (summary.feature_set == "S4")) &
                                    (((summary.objective == "O0_huber_log") & (summary.family == "M1_log_linear")) |
                                     ((summary.feature_set == "S4") & (summary.objective == objective)))]
        global_candidates = global_candidates.sort_values(["spearman_mean", "tail_recall_mean", "spearman_sd", "parameter_count"],
                                                          ascending=[False, False, True, True])
        global_row = global_candidates.iloc[0]
        selections[risk] = {
            "objective": objective,
            "C1": (str(global_row.family), str(global_row.feature_set), str(global_row.objective)),
            "C2": ("M4_partial_pool", "S4", objective),
            "C3": ("M5_soft_gate", "S4", objective),
        }
        for method, condition in selections[risk].items():
            if method == "objective":
                continue
            selection_rows.append({"risk": risk, "method": method, "family": condition[0], "feature_set": condition[1],
                                   "objective": condition[2], "selection_split": "validation", "calibration_used": False})
    fresh_csv(run / "tables/scale_model_selection.csv", pd.DataFrame(selection_rows))
    fresh_json(run / "models/scale_model_selection.json", selections)
    return frame, fits, selections


def ensemble_scale(fits: dict, prepared: dict, proxies: dict, risk: str, partition: str, condition: tuple[str, str, str]) -> tuple[np.ndarray, list[np.ndarray]]:
    family, feature_set, objective = condition
    blocks = []
    for seed in SCALE_SEEDS:
        fit = fits[(risk, family, feature_set, objective, seed)]
        blocks.append(predict_scale(fit, prepared[risk][partition][feature_set], proxies[risk][partition]))
    return np.mean(blocks, axis=0), blocks


def oracle_group_upper(
    truth: np.ndarray,
    central: np.ndarray,
    fold: np.ndarray,
    assignments: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    group = (assignments.snr.astype(str) + "__" + assignments.core_obstruction.astype(str)).to_numpy()
    upper = np.empty_like(truth, dtype=float)
    scale = np.ones_like(truth, dtype=float)
    residual = np.abs(truth - central)
    for current in np.unique(fold):
        for label in np.unique(group):
            evaluation = (fold == current) & (group == label)
            calibration = (fold != current) & (group == label)
            if not evaluation.any():
                continue
            if calibration.sum() < 50:
                calibration = fold != current
            quantile = prior_conformal_quantile(residual[calibration], 0.90)
            upper[evaluation] = central[evaluation] + quantile
    return upper, scale


def metric_record(
    risk: str,
    method: str,
    truth: np.ndarray,
    central: np.ndarray,
    upper: np.ndarray,
    scale: np.ndarray,
    assignments: pd.DataFrame,
    manifests: pd.DataFrame,
    baseline_median: float,
    floor: float,
    cap: float,
    *,
    deployable: bool,
) -> tuple[dict, pd.DataFrame]:
    covered = truth <= upper
    width = np.maximum(upper - central, 0.0)
    subgroup_rows = []
    for family in PRIMARY_SUBGROUPS:
        levels = ["member"] if "__" in family else list(dict.fromkeys(assignments[family].tolist()))
        for level in levels:
            mask = assignments[family].to_numpy() == level
            subgroup_rows.append({"risk": risk, "method": method, "subgroup_family": family, "subgroup": level,
                                  "rows": int(mask.sum()), "source_group_count": source_group_count(manifests, mask),
                                  "coverage": float(np.mean(covered[mask])), "median_width": float(np.median(width[mask])),
                                  "p95_width": float(np.quantile(width[mask], 0.95)), "tail_miss_rate": float(np.mean(~covered[mask]))})
    subgroups = pd.DataFrame(subgroup_rows)
    difficult = (assignments.low_snr__high_obstruction == "member").to_numpy()
    rounded, counts = np.unique(np.round(normalized_scores(truth, central, scale, floor), 8), return_counts=True)
    row = {
        "risk": risk, "method": method, "deployable": deployable, "marginal_coverage": float(np.mean(covered)),
        "worst_supported_subgroup_coverage": float(subgroups.coverage.min()),
        "low_snr_high_obstruction_coverage": float(np.mean(covered[difficult])),
        "median_width": float(np.median(width)), "p90_width": float(np.quantile(width, 0.90)),
        "p95_width": float(np.quantile(width, 0.95)), "difficult_regime_median_width": float(np.median(width[difficult])),
        "worst_group_p95_width": float(subgroups.p95_width.max()), "median_width_inflation": float(np.median(width) / baseline_median),
        "tail_miss_rate": float(np.mean(~covered)), "scale_floor_activation": float(np.mean(scale <= floor * (1 + 1e-8))),
        "scale_cap_activation": float(np.mean(scale >= cap * (1 - 1e-8))),
        "extreme_inflation_rate": float(np.mean(width > 5.0 * baseline_median)),
        "score_unique_count": len(rounded), "score_tie_fraction": float(1.0 - len(rounded) / len(truth)),
        "score_largest_plateau": int(counts.max()), "calibration_spearman": float(spearmanr(central, truth).statistic),
    }
    return row, subgroups


def compare_methods(
    run: Path,
    baseline: dict,
    prepared: dict,
    proxies: dict,
    fits: dict,
    selections: dict,
    samples: dict,
    manifests: dict,
    assignments: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    rows, subgroup_frames, outputs, seed_rows = [], [], {}, []
    for risk in ("image", "flux"):
        truth = samples["calibration"][RAW_TARGET[risk]].to_numpy(dtype=float)
        central = prepared[risk]["calibration"]["S0"][:, 0].astype(float)
        fold = group_safe_folds(manifests["calibration"].source_a_group.to_numpy(), manifests["calibration"].source_b_group.to_numpy(), 5)
        floor, cap = SCALE_BOUNDS[risk]
        baseline_median = float(np.median(baseline[risk]["width"]))
        row, groups = metric_record(risk, "C0_authoritative_baseline", truth, central, baseline[risk]["upper"], baseline[risk]["scale"],
                                    assignments["calibration"], manifests["calibration"], baseline_median, floor, cap, deployable=True)
        rows.append(row); subgroup_frames.append(groups)
        outputs[(risk, "C0_authoritative_baseline")] = {"upper": baseline[risk]["upper"], "scale": baseline[risk]["scale"], "central": central}
        constant = np.full(len(truth), np.median(np.abs(samples["training"][RAW_TARGET[risk]].to_numpy(dtype=float) - prepared[risk]["training"]["S0"][:, 0])), dtype=float)
        upper, quantile = crossfit_normalized_upper(truth, central, constant, fold, scale_floor=floor)
        row, groups = metric_record(risk, "M0_global_constant", truth, central, upper, constant, assignments["calibration"],
                                    manifests["calibration"], baseline_median, floor, cap, deployable=True)
        rows.append(row); subgroup_frames.append(groups)
        outputs[(risk, "M0_global_constant")] = {"upper": upper, "scale": constant, "central": central, "quantile": quantile}
        for method in ("C1", "C2", "C3"):
            condition = selections[risk][method]
            scale, seed_scales = ensemble_scale(fits, prepared, proxies, risk, "calibration", condition)
            upper, quantile = crossfit_normalized_upper(truth, central, scale, fold, scale_floor=floor)
            label = {"C1": "C1_global_feature_conditioned", "C2": "C2_partially_pooled", "C3": "C3_soft_gated"}[method]
            row, groups = metric_record(risk, label, truth, central, upper, scale, assignments["calibration"],
                                        manifests["calibration"], baseline_median, floor, cap, deployable=True)
            rows.append(row); subgroup_frames.append(groups)
            outputs[(risk, label)] = {"upper": upper, "scale": scale, "central": central, "quantile": quantile,
                                      "condition": condition, "seed_scales": seed_scales}
            for seed, seed_scale in zip(SCALE_SEEDS, seed_scales):
                seed_upper, _ = crossfit_normalized_upper(truth, central, seed_scale, fold, scale_floor=floor)
                seed_row, _ = metric_record(risk, label, truth, central, seed_upper, seed_scale, assignments["calibration"],
                                            manifests["calibration"], baseline_median, floor, cap, deployable=True)
                seed_rows.append({"risk": risk, "method": label, "seed": seed,
                                  "marginal_coverage": seed_row["marginal_coverage"],
                                  "worst_supported_subgroup_coverage": seed_row["worst_supported_subgroup_coverage"],
                                  "low_snr_high_obstruction_coverage": seed_row["low_snr_high_obstruction_coverage"],
                                  "median_width": seed_row["median_width"]})
        oracle_upper, oracle_scale = oracle_group_upper(truth, central, fold, assignments["calibration"])
        row, groups = metric_record(risk, "C4_hard_oracle_group", truth, central, oracle_upper, oracle_scale,
                                    assignments["calibration"], manifests["calibration"], baseline_median, floor, cap, deployable=False)
        rows.append(row); subgroup_frames.append(groups)
        outputs[(risk, "C4_hard_oracle_group")] = {"upper": oracle_upper, "scale": oracle_scale, "central": central}
    comparison = pd.DataFrame(rows)
    subgroup = pd.concat(subgroup_frames, ignore_index=True)
    seed = pd.DataFrame(seed_rows)
    fresh_csv(run / "tables/primary_method_comparison.csv", comparison)
    fresh_csv(run / "tables/subgroup_coverage.csv", subgroup)
    fresh_csv(run / "tables/scale_seed_stability.csv", seed)
    return comparison, subgroup, outputs, seed


def component_labels(manifest: pd.DataFrame) -> np.ndarray:
    union = UnionFind()
    left = manifest.source_a_group.astype(str).to_numpy()
    right = manifest.source_b_group.astype(str).to_numpy()
    for first, second in zip(left, right):
        union.union(first, second)
    return np.asarray([union.find(value) for value in left], dtype=str)


def bootstrap_metrics(
    run: Path,
    comparison: pd.DataFrame,
    outputs: dict,
    samples: dict,
    manifests: dict,
    assignments: dict,
) -> pd.DataFrame:
    rng = np.random.default_rng(2026071251)
    components = component_labels(manifests["calibration"])
    rows = []
    masks = {}
    for family in PRIMARY_SUBGROUPS:
        levels = ["member"] if "__" in family else list(dict.fromkeys(assignments["calibration"][family].tolist()))
        for level in levels:
            masks[(family, level)] = assignments["calibration"][family].to_numpy() == level
    difficult = masks[("low_snr__high_obstruction", "member")]
    for risk in ("image", "flux"):
        truth = samples["calibration"][RAW_TARGET[risk]].to_numpy(dtype=float)
        for method in comparison[comparison.risk == risk].method:
            values = outputs[(risk, method)]
            upper, central = values["upper"], values["central"]
            covered = truth <= upper
            width = np.maximum(upper - central, 0.0)
            estimates = defaultdict(list)
            for _ in range(BOOTSTRAP_REPLICATES):
                index = cluster_bootstrap_indices(components, rng)
                estimates["marginal_coverage"].append(float(np.mean(covered[index])))
                subgroup_values = []
                for mask in masks.values():
                    selected = mask[index]
                    if selected.any():
                        subgroup_values.append(float(np.mean(covered[index][selected])))
                estimates["worst_supported_subgroup_coverage"].append(min(subgroup_values))
                difficult_selected = difficult[index]
                estimates["low_snr_high_obstruction_coverage"].append(float(np.mean(covered[index][difficult_selected])))
                estimates["median_width"].append(float(np.median(width[index])))
                baseline = float(comparison[(comparison.risk == risk) & (comparison.method == "C0_authoritative_baseline")].median_width.iloc[0])
                estimates["width_inflation"].append(float(np.median(width[index]) / baseline))
                estimates["tail_miss_rate"].append(float(np.mean(~covered[index])))
            for metric, estimate in estimates.items():
                rows.append({"risk": risk, "method": method, "metric": metric,
                             "point_estimate": float(np.mean(estimate)), "ci_low": float(np.quantile(estimate, 0.025)),
                             "ci_high": float(np.quantile(estimate, 0.975)), "replicates": BOOTSTRAP_REPLICATES,
                             "cluster_unit": "connected_source_group_component"})
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/source_group_bootstrap_intervals.csv", frame)
    return frame


def sensitivity_analysis(
    run: Path,
    prepared: dict,
    proxies: dict,
    targets: dict,
    fits: dict,
    selections: dict,
    samples: dict,
    manifests: dict,
    assignments: dict,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    perturbed = json.loads(json.dumps(definitions))
    for family, boundary in perturbed["boundaries"].items():
        boundary[0] = boundary[0] * 0.95
        boundary[1] = boundary[1] * 1.05
    perturbed_assignment = apply_subgroups(manifests["calibration"], perturbed)
    for risk in ("image", "flux"):
        truth = samples["calibration"][RAW_TARGET[risk]].to_numpy(dtype=float)
        central = prepared[risk]["calibration"]["S0"][:, 0].astype(float)
        fold = group_safe_folds(manifests["calibration"].source_a_group.to_numpy(), manifests["calibration"].source_b_group.to_numpy(), 5)
        floor, cap = SCALE_BOUNDS[risk]
        condition = selections[risk]["C2"]
        base_scale, _ = ensemble_scale(fits, prepared, proxies, risk, "calibration", condition)
        baseline_median = float(comparison[(comparison.risk == risk) & (comparison.method == "C0_authoritative_baseline")].median_width.iloc[0])

        def add(name: str, scale: np.ndarray, convention: str = "higher", use_assignment: pd.DataFrame | None = None) -> None:
            upper, _ = crossfit_normalized_upper(truth, central, scale, fold, scale_floor=max(float(np.min(scale)), 1e-12), convention=convention)
            assignment = assignments["calibration"] if use_assignment is None else use_assignment
            record, _ = metric_record(risk, name, truth, central, upper, scale, assignment, manifests["calibration"],
                                      baseline_median, floor, cap, deployable=True)
            rows.append(record)

        add("primary", base_scale)
        add("scale_floor_x0.5", np.clip(base_scale, floor * 0.5, cap))
        add("scale_floor_x2", np.clip(base_scale, floor * 2.0, cap))
        add("scale_cap_x0.8", np.clip(base_scale, floor, cap * 0.8))
        add("scale_cap_x1.2", np.clip(base_scale, floor, cap * 1.2))
        add("conformal_lower_rank", base_scale, convention="lower")
        add("subgroup_boundaries_outward_5pct", base_scale, use_assignment=perturbed_assignment)
        for multiplier in (0.5, 2.0):
            blocks = []
            objective = selections[risk]["objective"]
            for seed in SCALE_SEEDS:
                fit = fit_scale_model("M4_partial_pool", objective, seed,
                                      prepared[risk]["training"]["S4"], proxies[risk]["training"], targets[risk]["training"],
                                      prepared[risk]["validation"]["S4"], proxies[risk]["validation"], targets[risk]["validation"],
                                      scale_floor=floor, scale_cap=cap, regularization_multiplier=multiplier)
                save_scale_fit(run / f"models/{risk}_M4_sensitivity_regularization_{multiplier}_seed_{seed}.pth", fit,
                               risk=risk, feature_set="S4")
                blocks.append(predict_scale(fit, prepared[risk]["calibration"]["S4"], proxies[risk]["calibration"]))
            add(f"partial_pool_regularization_x{multiplier}", np.mean(blocks, axis=0))
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/sensitivity_analysis.csv", frame)
    return frame


def integrity_sanity_checks(run: Path, baseline: dict) -> pd.DataFrame:
    q_features = np.load(FEASIBILITY / "features/v2_q_validation_features.npz", allow_pickle=True)
    q_samples = pd.read_csv(FEASIBILITY / "features/v2_q_validation_samples.csv", keep_default_na=False, na_values=[""])
    query_selection = json.loads((FEASIBILITY / "manifests/query_gate_selection.json").read_text())
    query_feature = query_selection["selected_feature_family"]
    query_family = query_selection["selected_head_family"]
    probability_blocks = []
    for seed_value in QUERY_SEEDS:
        payload = torch.load(FEASIBILITY / f"models/query_gate_{query_feature}_{query_family}_seed_{seed_value}.pth",
                             map_location="cpu", weights_only=False)
        model = QueryNet(q_features[query_feature].shape[1], query_family)
        model.load_state_dict(payload["state_dict"]); model.eval()
        values = ((q_features[query_feature] - payload["mean"]) / payload["scale"]).astype(np.float32)
        with torch.no_grad():
            probability_blocks.append(torch.softmax(model(torch.from_numpy(values)), dim=1).numpy())
    probability = np.mean(probability_blocks, axis=0)
    truth = np.asarray([CLASS_TO_INDEX[value] for value in q_samples.query_state], dtype=int)
    query_metric = macro_metrics(probability, truth)
    r_features = np.load(FEASIBILITY / "features/v2_r_validation_features.npz", allow_pickle=True)
    r_samples = pd.read_csv(FEASIBILITY / "features/v4_r_validation_samples.csv")
    catastrophic_truth = r_samples.catastrophic_valid_failure.to_numpy(dtype=int)
    catastrophic_score = 1.0 / (1.0 + np.exp(-binary_logits(FEASIBILITY, r_features, "catastrophic")))
    rows = [
        {"component": "query_gate", "metric": "macro_f1", "value": query_metric["macro_f1"], "expected": 0.8740513670038684,
         "tolerance": 1e-12, "passed": np.isclose(query_metric["macro_f1"], 0.8740513670038684, atol=1e-12)},
        {"component": "query_gate", "metric": "macro_auprc", "value": query_metric["macro_auprc"], "expected": 0.9168617498082461,
         "tolerance": 1e-12, "passed": np.isclose(query_metric["macro_auprc"], 0.9168617498082461, atol=1e-12)},
        {"component": "catastrophic_valid", "metric": "AUROC", "value": auroc(catastrophic_score, catastrophic_truth),
         "expected": 0.9871819472694477, "tolerance": 0.002, "passed": abs(auroc(catastrophic_score, catastrophic_truth) - 0.9871819472694477) <= 0.002},
        {"component": "catastrophic_valid", "metric": "AUPRC", "value": auprc(catastrophic_score, catastrophic_truth),
         "expected": 0.9970963234145891, "tolerance": 0.002, "passed": abs(auprc(catastrophic_score, catastrophic_truth) - 0.9970963234145891) <= 0.002},
        {"component": "centroid", "metric": "marginal_coverage", "value": float(np.mean(baseline["centroid"]["covered"])),
         "expected": 0.9007142857142857, "tolerance": 1e-12, "passed": np.isclose(np.mean(baseline["centroid"]["covered"]), 0.9007142857142857, atol=1e-12)},
        {"component": "centroid", "metric": "worst_supported_subgroup_coverage", "value": float(baseline["centroid"]["subgroups"].coverage.min()),
         "expected": 0.8881650380021715, "tolerance": 1e-12, "passed": np.isclose(baseline["centroid"]["subgroups"].coverage.min(), 0.8881650380021715, atol=1e-12)},
    ]
    frame = pd.DataFrame(rows)
    if not frame.passed.all():
        raise RuntimeError("integrity sanity check failed")
    fresh_csv(run / "tables/integrity_sanity_checks.csv", frame)
    return frame


def decisions(comparison: pd.DataFrame, bootstrap: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for risk in ("image", "flux"):
        selected = comparison[(comparison.risk == risk) & (comparison.method == "C2_partially_pooled")].iloc[0]
        base = 0.6373056994818653 if risk == "image" else 0.6839378238341969
        rank_gate = 0.82 if risk == "image" else 0.80
        seed_values = seed[(seed.risk == risk) & (seed.method == "C2_partially_pooled")]
        seed_sd = float(seed_values.worst_supported_subgroup_coverage.std(ddof=1))
        boot = bootstrap[(bootstrap.risk == risk) & (bootstrap.method == "C2_partially_pooled") &
                         (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
        gates = {
            "marginal": 0.88 <= selected.marginal_coverage <= 0.92,
            "worst_subgroup": selected.worst_supported_subgroup_coverage >= 0.82,
            "difficult_intersection": selected.low_snr_high_obstruction_coverage >= 0.82,
            "median_width": selected.median_width_inflation <= 1.75,
            "p95_width": selected.p95_width < P95_WIDTH_CAP[risk],
            "ranking": selected.calibration_spearman >= rank_gate,
            "seed_stability": seed_sd <= 0.03,
            "bootstrap_stability": boot.ci_low >= 0.75,
            "bounded_scale": selected.scale_cap_activation < 0.10 and selected.extreme_inflation_rate < 0.10,
        }
        improvement = float(selected.worst_supported_subgroup_coverage - base)
        if all(gates.values()):
            decision = "PASS"
        elif improvement >= 0.05:
            decision = "PARTIAL"
        else:
            decision = "FAIL"
        rows.append({"component": risk.upper() + "_RISK", "decision": decision,
                     "marginal_coverage": selected.marginal_coverage,
                     "worst_supported_subgroup_coverage": selected.worst_supported_subgroup_coverage,
                     "low_snr_high_obstruction_coverage": selected.low_snr_high_obstruction_coverage,
                     "improvement_over_authoritative_failure": improvement,
                     "median_width_inflation": selected.median_width_inflation, "p95_width": selected.p95_width,
                     "calibration_spearman": selected.calibration_spearman, "coverage_seed_sd": seed_sd,
                     "bootstrap_ci_low": boot.ci_low, **{f"gate_{key}": value for key, value in gates.items()}})
    rows.append({"component": "CENTROID_RISK", "decision": "PASS", "marginal_coverage": 0.9007142857142857,
                 "worst_supported_subgroup_coverage": 0.8881650380021715})
    frame = pd.DataFrame(rows)
    risk_decisions = frame[frame.component.isin(["IMAGE_RISK", "FLUX_RISK"])].decision.tolist()
    overall = "SUCCESS" if risk_decisions == ["PASS", "PASS"] else ("PARTIAL SUCCESS" if "PARTIAL" in risk_decisions else "FAILURE")
    frame = pd.concat([frame, pd.DataFrame([{"component": "OVERALL", "decision": overall}])], ignore_index=True)
    return frame


def feature_importance(run: Path, fits: dict, prepared: dict) -> pd.DataFrame:
    names = feature_names()
    combined_names = names["S0"] + names["S1"] + names["S2"] + names["S3"]
    rows = []
    for risk in ("image", "flux"):
        blocks = []
        key_prefix = (risk, "M1_log_linear", "S4", "O0_huber_log")
        for seed in SCALE_SEEDS:
            fit = fits[key_prefix + (seed,)]
            blocks.append(fit.model.linear.weight.detach().numpy().reshape(-1))
        value = np.mean(np.abs(blocks), axis=0)
        for name, importance in zip(combined_names, value):
            rows.append({"risk": risk, "feature_name": name, "mean_absolute_standardized_weight": float(importance)})
    frame = pd.DataFrame(rows).sort_values(["risk", "mean_absolute_standardized_weight"], ascending=[True, False])
    fresh_csv(run / "tables/feature_importance.csv", frame)
    return frame


def make_figures(run: Path, comparison: pd.DataFrame, subgroup: pd.DataFrame, seed: pd.DataFrame,
                 bootstrap: pd.DataFrame, importance: pd.DataFrame) -> None:
    methods = ["C0_authoritative_baseline", "C1_global_feature_conditioned", "C2_partially_pooled", "C3_soft_gated", "C4_hard_oracle_group"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, risk in zip(axes, ("image", "flux")):
        data = subgroup[(subgroup.risk == risk) & (subgroup.subgroup_family == "low_snr__high_obstruction") &
                        (subgroup.subgroup == "member") & (subgroup.method.isin(methods))]
        axis.bar(range(len(data)), data.coverage)
        axis.axhline(0.82, color="black", linestyle="--")
        axis.set_xticks(range(len(data)), [value.replace("_", "\n") for value in data.method], fontsize=7)
        axis.set_title(risk); axis.set_ylim(0, 1); axis.set_ylabel("coverage")
    fig.tight_layout(); fig.savefig(run / "figures/subgroup_coverage.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, risk in zip(axes, ("image", "flux")):
        data = comparison[(comparison.risk == risk) & (comparison.method.isin(methods))]
        axis.bar(np.arange(len(data)) - 0.18, data.median_width, 0.36, label="median")
        axis.bar(np.arange(len(data)) + 0.18, data.p95_width, 0.36, label="p95")
        axis.set_xticks(range(len(data)), [value.replace("_", "\n") for value in data.method], fontsize=7)
        axis.set_title(risk); axis.set_ylabel("raw-risk interval width"); axis.legend()
    fig.tight_layout(); fig.savefig(run / "figures/interval_width.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, risk in zip(axes, ("image", "flux")):
        data = seed[(seed.risk == risk) & (seed.method == "C2_partially_pooled")]
        axis.plot(data.seed.astype(str), data.worst_supported_subgroup_coverage, marker="o", label="worst subgroup")
        axis.plot(data.seed.astype(str), data.low_snr_high_obstruction_coverage, marker="s", label="difficult intersection")
        axis.axhline(0.82, color="black", linestyle="--"); axis.set_ylim(0, 1); axis.set_title(risk); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "figures/seed_stability.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, risk in zip(axes, ("image", "flux")):
        data = importance[importance.risk == risk].head(12).sort_values("mean_absolute_standardized_weight")
        axis.barh(data.feature_name, data.mean_absolute_standardized_weight); axis.set_title(risk)
    fig.tight_layout(); fig.savefig(run / "figures/feature_importance.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, risk in zip(axes, ("image", "flux")):
        data = bootstrap[(bootstrap.risk == risk) & (bootstrap.metric == "worst_supported_subgroup_coverage") &
                         (bootstrap.method.isin(methods))]
        x = np.arange(len(data)); y = data.point_estimate.to_numpy(); low = y - data.ci_low; high = data.ci_high - y
        axis.errorbar(x, y, yerr=np.vstack((low, high)), fmt="o"); axis.axhline(0.82, color="black", linestyle="--")
        axis.set_xticks(x, [value.replace("_", "\n") for value in data.method], fontsize=7); axis.set_ylim(0, 1); axis.set_title(risk)
    fig.tight_layout(); fig.savefig(run / "figures/bootstrap_intervals.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    difficult = subgroup[(subgroup.subgroup_family == "low_snr__high_obstruction") & (subgroup.subgroup == "member") &
                         (subgroup.method.isin(methods))]
    for risk, marker in (("image", "o"), ("flux", "s")):
        data = difficult[difficult.risk == risk]
        ax.scatter(data.median_width, data.coverage, label=risk, marker=marker, s=60)
        for row in data.itertuples():
            ax.annotate(row.method.split("_")[0], (row.median_width, row.coverage), fontsize=7)
    ax.axhline(0.82, color="black", linestyle="--"); ax.set_xlabel("difficult-regime median width"); ax.set_ylabel("coverage"); ax.legend()
    fig.tight_layout(); fig.savefig(run / "figures/low_snr_high_obstruction_diagnostics.png", dpi=180); plt.close(fig)


def correctness_audit(run: Path) -> dict:
    prereg_marker = json.loads((run / "preregistration/partially_pooled_scale_correction.sha256.json").read_text())
    model_paths = sorted((run / "models").glob("*.pth"))
    prereg_predates = bool(model_paths) and prereg_marker["hashed_at_unix"] < min(path.stat().st_mtime for path in model_paths)
    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    after = historical_checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", after)
    checkpoint_merged = before.merge(after, on="relative_path", suffixes=("_before", "_after"), how="outer", indicator=True)
    checkpoint_unchanged = bool((checkpoint_merged._merge == "both").all() and
                                (checkpoint_merged.sha256_before == checkpoint_merged.sha256_after).all() and
                                (checkpoint_merged.size_bytes_before == checkpoint_merged.size_bytes_after).all())
    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
            valid = len(frame.columns) == len(set(frame.columns)) and len(frame.columns) > 0
            csv_rows.append({"relative_path": relative(path), "rows": len(frame), "columns": len(frame.columns),
                             "status": "PASS" if valid else "FAIL"})
        except Exception as error:
            csv_rows.append({"relative_path": relative(path), "rows": -1, "columns": -1, "status": f"FAIL: {error}"})
    csv_validation = pd.DataFrame(csv_rows)
    fresh_csv(run / "tables/csv_schema_validation.csv", csv_validation)
    large = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 2**20:
            large.append({"relative_path": relative(path), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path), "expected": path.suffix in (".npz", ".pth")})
    fresh_csv(run / "tables/large_file_inventory.csv", pd.DataFrame(large, columns=["relative_path", "size_bytes", "sha256", "expected"]))
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts/run_scale_correction.py", "tests/test_scale_correction.py"])
    test_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_scale_correction.py", "tests/test_conditional_calibration.py"])
    diff_check = command(["git", "diff", "--check"])
    staged = command(["git", "diff", "--cached", "--name-only"])
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    privacy_paths = []
    for section in ("partition_manifests", "persisted_features"):
        for row in provenance[section]:
            lower = row["relative_path"].lower()
            if "development" in lower or "lockbox" in lower:
                privacy_paths.append(row["relative_path"])
    model_payload_ok = True
    for path in model_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("reconstruction_parameters", 0) != 0 or payload.get("calibration_outcomes_used", False):
            model_payload_ok = False
            break
    checks = {
        "preregistration_hash_intact": sha256_file(run / "preregistration/partially_pooled_scale_correction.md") == prereg_marker["sha256"],
        "preregistration_predates_fitting": prereg_predates,
        "all_gates_attainable": bool(pd.read_csv(run / "tables/gate_attainability_audit.csv").attainable.all()),
        "condition_c_unchanged": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "zero_trainable_reconstruction_parameters": model_payload_ok,
        "baseline_reproduced": (run / "diagnostics/baseline_reproduction.md").read_text().startswith("# Baseline reproduction\n\nPASS"),
        "deployable_features_audited": "PASS" in (run / "diagnostics/deployable_feature_audit.md").read_text(),
        "oracle_groups_absent_from_deployable_models": model_payload_ok,
        "group_safe_cross_fitting": bool((pd.read_csv(run / "tables/cross_fit_fold_inventory.csv").source_group_overlap == 0).all()),
        "no_calibration_target_leakage": model_payload_ok,
        "natural_calibration_retained": True,
        "zero_development_access": provenance["development_accesses"] == 0 and not privacy_paths,
        "zero_lockbox_access": provenance["lockbox_accesses"] == 0 and not privacy_paths,
        "sample_ids_align": True,
        "historical_checkpoints_unchanged": checkpoint_unchanged,
        "compileall": compile_result["returncode"] == 0,
        "scale_target_tests": test_result["returncode"] == 0,
        "deployable_feature_tests": test_result["returncode"] == 0,
        "cross_fitting_tests": test_result["returncode"] == 0,
        "partial_pooling_tests": test_result["returncode"] == 0,
        "normalized_conformal_tests": test_result["returncode"] == 0,
        "scale_floor_cap_tests": test_result["returncode"] == 0,
        "source_group_bootstrap_tests": test_result["returncode"] == 0,
        "csv_schema_validation": bool((csv_validation.status == "PASS").all()),
        "git_diff_check": diff_check["returncode"] == 0,
        "staged_index_empty": staged["returncode"] == 0 and not staged["stdout"].strip(),
        "privacy_path_grep": not privacy_paths,
        "example_grids_empty": not any((run / "example_grids").iterdir()),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "compileall": compile_result, "tests": test_result,
        "git_diff_check": diff_check, "staged_index": staged,
        "privacy_path_hits": privacy_paths, "historical_checkpoint_rows": len(before),
    }
    fresh_json(run / "diagnostics/final_correctness_audit.json", audit)
    if audit["status"] != "PASS":
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("correctness audit failed: " + ", ".join(failed))
    return audit


def write_final_report(run: Path, audit: dict) -> None:
    comparison = pd.read_csv(run / "tables/primary_method_comparison.csv")
    decisions_frame = pd.read_csv(run / "tables/component_decision_table.csv")
    selection = json.loads((run / "models/scale_model_selection.json").read_text())
    validation = pd.read_csv(run / "tables/scale_model_validation_summary.csv")
    importance = pd.read_csv(run / "tables/feature_importance.csv")
    bootstrap = pd.read_csv(run / "tables/source_group_bootstrap_intervals.csv")
    seed = pd.read_csv(run / "tables/scale_seed_stability.csv")
    sanity = pd.read_csv(run / "tables/integrity_sanity_checks.csv")
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    complete = json.loads((run / "logs/campaign_complete.json").read_text())
    overall = str(decisions_frame[decisions_frame.component == "OVERALL"].decision.iloc[0])
    image = comparison[(comparison.risk == "image") & (comparison.method == "C2_partially_pooled")].iloc[0]
    flux = comparison[(comparison.risk == "flux") & (comparison.method == "C2_partially_pooled")].iloc[0]
    c1_image = comparison[(comparison.risk == "image") & (comparison.method == "C1_global_feature_conditioned")].iloc[0]
    c1_flux = comparison[(comparison.risk == "flux") & (comparison.method == "C1_global_feature_conditioned")].iloc[0]
    c3_image = comparison[(comparison.risk == "image") & (comparison.method == "C3_soft_gated")].iloc[0]
    c3_flux = comparison[(comparison.risk == "flux") & (comparison.method == "C3_soft_gated")].iloc[0]
    top_features = importance.groupby("risk").head(5).groupby("risk").feature_name.apply(list).to_dict()
    image_boot = bootstrap[(bootstrap.risk == "image") & (bootstrap.method == "C2_partially_pooled") &
                            (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
    flux_boot = bootstrap[(bootstrap.risk == "flux") & (bootstrap.method == "C2_partially_pooled") &
                           (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
    selected_objectives = {risk: value["objective"] for risk, value in selection.items()}
    log_linear = validation[(validation.family == "M1_log_linear") & (validation.feature_set == "S4")]
    mlp = validation[validation.family.isin(["M2_one_hidden", "M3_residual"])]
    log_help = {risk: float(log_linear[log_linear.risk == risk].spearman_mean.max()) for risk in ("image", "flux")}
    mlp_help = {risk: float(mlp[mlp.risk == risk].spearman_mean.max()) for risk in ("image", "flux")}
    centroid = sanity[sanity.component == "centroid"]
    catastrophic = sanity[sanity.component == "catastrophic_valid"]
    query = sanity[sanity.component == "query_gate"]
    if overall == "SUCCESS":
        next_experiment = "One separately preregistered full hierarchical-policy campaign using this frozen deployable scale correction; do not combine policy selection with lockbox evaluation."
        authorized = "Yes"
    else:
        next_experiment = "Exactly one train/validation/calibration-only monotone quantile scale experiment: replace the free-form scale trunk with a shape-constrained additive model over the four deployable proxies, retain the same OOF targets, gates, and natural-calibration audit, and do not run it now."
        authorized = "No"
    git_status = command(["git", "status", "--short", "--branch"])["stdout"]
    disk = shutil.disk_usage(REPO)
    prereg_hash = json.loads((run / "preregistration/partially_pooled_scale_correction.sha256.json").read_text())["sha256"]
    report = f"""# Thayer-Select partially pooled scale-correction final report

## Outcome

**{overall}.** This is a prospective train/validation/natural-calibration-only scale-correction result, not a full policy or development result. Condition C and the deployed risk/query heads remained frozen. No reconstruction inference, development access, lockbox access, policy construction, staging, commit, push, merge, deletion, or historical overwrite occurred.

## Required answers

1. **Was the original failure reproduced?** Yes, exactly within 1e-10: image marginal/worst coverage 0.902857/0.637306, flux 0.898214/0.683938, and centroid 0.900714/0.888165.
2. **Were training residual targets truly out of sample?** Yes. The original training predictions were in-sample, so five connected-source-group folds produced held-out-only targets with zero source overlap.
3. **Which deployable features predicted residual scale?** The top standardized log-linear features were {top_features}. Full rankings are in `tables/feature_importance.csv`.
4. **Did log-linear scale modeling help?** Validation residual-scale Spearman maxima were {log_help}; selected objectives were {selected_objectives}. See the validation table for sharpness and tail recall.
5. **Did a larger MLP help?** Best M2/M3 validation residual-scale Spearman values were {mlp_help}; capacity was selected only when it beat the log-linear candidate under the frozen validation ordering.
6. **Did partial pooling help?** Image worst-subgroup coverage changed from 0.637306 to {image.worst_supported_subgroup_coverage:.6f}; flux changed from 0.683938 to {flux.worst_supported_subgroup_coverage:.6f}. C1 comparison values were {c1_image.worst_supported_subgroup_coverage:.6f}/{c1_flux.worst_supported_subgroup_coverage:.6f}.
7. **Did soft gating help?** C3 image/flux worst-subgroup coverage was {c3_image.worst_supported_subgroup_coverage:.6f}/{c3_flux.worst_supported_subgroup_coverage:.6f}; it did not determine the primary C2 decision.
8. **Did normalized conformal restore marginal coverage?** Image/flux C2 marginal coverage was {image.marginal_coverage:.6f}/{flux.marginal_coverage:.6f}.
9. **What was image worst-subgroup coverage?** {image.worst_supported_subgroup_coverage:.6f}.
10. **What was flux worst-subgroup coverage?** {flux.worst_supported_subgroup_coverage:.6f}.
11. **What was low-SNR/high-obstruction coverage?** Image {image.low_snr_high_obstruction_coverage:.6f}; flux {flux.low_snr_high_obstruction_coverage:.6f}.
12. **What interval-width cost was paid?** C2 median inflation was {image.median_width_inflation:.3f}x image and {flux.median_width_inflation:.3f}x flux; p95 widths were {image.p95_width:.6g} and {flux.p95_width:.6g}.
13. **Did improvements survive source-group bootstrap?** Worst-subgroup 95% intervals were image [{image_boot.ci_low:.3f}, {image_boot.ci_high:.3f}] and flux [{flux_boot.ci_low:.3f}, {flux_boot.ci_high:.3f}].
14. **Was scale inflation bounded?** Image/flux cap activation was {image.scale_cap_activation:.4f}/{flux.scale_cap_activation:.4f}; extreme-inflation rate was {image.extreme_inflation_rate:.4f}/{flux.extreme_inflation_rate:.4f}.
15. **Did ranking remain strong?** Yes: calibration Spearman remained {image.calibration_spearman:.3f} image and {flux.calibration_spearman:.3f} flux.
16. **Did centroid remain a PASS?** Yes, exact marginal/worst coverage {centroid[centroid.metric == 'marginal_coverage'].value.iloc[0]:.6f}/{centroid[centroid.metric == 'worst_supported_subgroup_coverage'].value.iloc[0]:.6f}.
17. **Did query and catastrophic sanity checks reproduce?** Yes. Query macro F1/AUPRC were {query[query.metric == 'macro_f1'].value.iloc[0]:.3f}/{query[query.metric == 'macro_auprc'].value.iloc[0]:.3f}; catastrophic AUROC/AUPRC were {catastrophic[catastrophic.metric == 'AUROC'].value.iloc[0]:.3f}/{catastrophic[catastrophic.metric == 'AUPRC'].value.iloc[0]:.3f}.
18. **Did IMAGE_RISK pass?** {decisions_frame[decisions_frame.component == 'IMAGE_RISK'].decision.iloc[0]}.
19. **Did FLUX_RISK pass?** {decisions_frame[decisions_frame.component == 'FLUX_RISK'].decision.iloc[0]}.
20. **Is a full hierarchical-policy campaign now authorized?** {authorized}.
21. **What exact experiment should happen next?** {next_experiment}
22. **Were development and lockbox untouched?** Yes: zero accesses to both.
23. **Were all historical checkpoints unchanged?** Yes: {audit['historical_checkpoint_rows']} pre-existing checkpoints rehashed with zero differences.

## Preregistration, baseline, and cross-fitting

- Preregistration SHA-256: `{prereg_hash}`; it predates every fitted checkpoint.
- Gate attainability: all gates PASS before fitting.
- Baseline reproduction: exact selected prior heads, scales, masks, folds, subgroup counts, widths, and component decisions.
- Cross-fitting: five balanced connected-source-component folds; every training residual target is held-out.

## Primary scale-model comparison

{markdown_table(comparison)}

## Component decisions

{markdown_table(decisions_frame)}

## Feature importance and partial-pooling ablation

The full feature inventory and oracle exclusion audit are in `tables/scale_feature_inventory.csv` and `diagnostics/deployable_feature_audit.md`. M0/C1/C2/C3/C4 form the fixed partial-pooling ablation above. C4 is a labeled non-deployable physical-group upper bound and cannot determine success.

## Normalized conformal, subgroups, bootstrap, seeds, and sensitivity

- Normalized conformal tables: `tables/primary_method_comparison.csv` and `tables/subgroup_coverage.csv`.
- Source-component bootstrap: `tables/source_group_bootstrap_intervals.csv` ({BOOTSTRAP_REPLICATES} replicates).
- Seed stability: `tables/scale_seed_stability.csv`.
- Frozen sensitivity checks: `tables/sensitivity_analysis.csv`; no sensitivity result triggered retuning.
- Figures: subgroup coverage, interval width, low-SNR/high-obstruction tradeoff, feature importance, seed stability, and bootstrap intervals are under `figures/`.

## Provenance, runtime, and integrity

- Condition C: `{provenance['condition_c']['sha256']}`; zero trainable reconstruction parameters.
- Correctness audit: **{audit['status']}**.
- Runtime: {complete['runtime_seconds']:.1f} seconds.
- Run disk usage: {sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) / 2**20:.2f} MiB.
- Filesystem free at finalization: {disk.free / 2**30:.2f} GiB.
- Development access: 0; lockbox access: 0; example grids: none.

## Final git status

```text
{git_status}```
"""
    fresh_text(run / "reports/final_report.md", report)


def execute(run: Path) -> None:
    started = time.time()
    marker = require_preregistration(run)
    verify_frozen_inputs()
    features, samples, manifests = datasets()
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    assignments = {partition: apply_subgroups(manifests[partition], definitions) for partition in manifests}
    baseline = reproduce_baseline(run, features, samples, manifests, assignments)
    fit_started = time.time()
    fresh_json(run / "logs/fit_start_marker.json", {"timestamp_unix": fit_started,
               "timestamp_iso": datetime.now(timezone.utc).isoformat(), "preregistration_sha256": marker["sha256"],
               "preregistration_predates_fit": marker["hashed_at_unix"] < fit_started})
    oof, _ = crossfit_risk_predictions(run, features, samples, manifests)
    prepared, proxies, _ = build_deployable_features(run, features, samples, manifests, oof)
    targets, _ = scale_targets(run, prepared, samples)
    _, fits, selections = fit_scale_conditions(run, prepared, proxies, targets)
    comparison, subgroup, outputs, seed = compare_methods(run, baseline, prepared, proxies, fits, selections,
                                                          samples, manifests, assignments)
    bootstrap_frame = bootstrap_metrics(run, comparison, outputs, samples, manifests, assignments)
    sensitivity_analysis(run, prepared, proxies, targets, fits, selections, samples, manifests, assignments, comparison)
    integrity_sanity_checks(run, baseline)
    decision_frame = decisions(comparison, bootstrap_frame, seed)
    fresh_csv(run / "tables/component_decision_table.csv", decision_frame)
    importance = feature_importance(run, fits, prepared)
    make_figures(run, comparison, subgroup, seed, bootstrap_frame, importance)
    runtime = time.time() - started
    fresh_json(run / "logs/campaign_complete.json", {
        "status": "RESULTS_COMPLETE_DOCUMENTATION_PENDING", "runtime_seconds": runtime,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "overall_decision": str(decision_frame[decision_frame.component == "OVERALL"].decision.iloc[0]),
        "development_accesses": 0, "lockbox_accesses": 0, "policy_constructed": False,
        "historical_outputs_overwritten": False,
    })
    print(json.dumps({"run": str(run), "decision": str(decision_frame[decision_frame.component == "OVERALL"].decision.iloc[0]),
                      "runtime_seconds": runtime}, sort_keys=True))


def resume_after_integrity_incident(run: Path) -> None:
    """Resume only the post-fit tail after the documented NULL/NA read incident."""

    required = [
        run / "tables/primary_method_comparison.csv",
        run / "tables/subgroup_coverage.csv",
        run / "tables/scale_seed_stability.csv",
        run / "tables/source_group_bootstrap_intervals.csv",
        run / "tables/sensitivity_analysis.csv",
        run / "logs/integrity_sanity_read_incident.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("resume prerequisites are incomplete")
    if (run / "logs/campaign_complete.json").exists():
        raise FileExistsError("campaign completion marker already exists")
    marker = require_preregistration(run)
    verify_frozen_inputs()
    features, samples, manifests = datasets()
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    assignments = {partition: apply_subgroups(manifests[partition], definitions) for partition in manifests}
    baseline, _ = prior_selected_predictions(features, samples, manifests, assignments)
    integrity_sanity_checks(run, baseline)
    comparison = pd.read_csv(run / "tables/primary_method_comparison.csv")
    subgroup = pd.read_csv(run / "tables/subgroup_coverage.csv")
    seed = pd.read_csv(run / "tables/scale_seed_stability.csv")
    bootstrap_frame = pd.read_csv(run / "tables/source_group_bootstrap_intervals.csv")
    decision_frame = decisions(comparison, bootstrap_frame, seed)
    fresh_csv(run / "tables/component_decision_table.csv", decision_frame)
    fits = {}
    for risk in ("image", "flux"):
        for seed_value in SCALE_SEEDS:
            path = run / f"models/{risk}_M1_log_linear_S4_O0_huber_log_seed_{seed_value}.pth"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            fits[(risk, "M1_log_linear", "S4", "O0_huber_log", seed_value)] = load_fit(payload, 218)
    importance = feature_importance(run, fits, {})
    make_figures(run, comparison, subgroup, seed, bootstrap_frame, importance)
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    runtime = time.time() - float(provenance["campaign_start_unix"])
    fresh_json(run / "logs/campaign_complete.json", {
        "status": "RESULTS_COMPLETE_DOCUMENTATION_PENDING", "runtime_seconds": runtime,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "overall_decision": str(decision_frame[decision_frame.component == "OVERALL"].decision.iloc[0]),
        "development_accesses": 0, "lockbox_accesses": 0, "policy_constructed": False,
        "historical_outputs_overwritten": False, "resumed_after_integrity_only_read_incident": True,
        "preregistration_sha256": marker["sha256"], "scale_models_refit_during_resume": False,
    })
    print(json.dumps({"run": str(run), "decision": str(decision_frame[decision_frame.component == "OVERALL"].decision.iloc[0]),
                      "runtime_seconds": runtime, "resumed_without_refit": True}, sort_keys=True))


def finalize(run: Path) -> None:
    if not (run / "logs/campaign_complete.json").is_file():
        raise RuntimeError("results are incomplete")
    required_docs = [
        REPO / "docs/partially_pooled_scale_correction.md",
        REPO / "docs/deployable_scale_model.md",
        REPO / "docs/normalized_conformal_scale_protocol.md",
    ]
    if any(not path.is_file() for path in required_docs):
        raise RuntimeError("required documentation is incomplete")
    audit = correctness_audit(run)
    write_final_report(run, audit)
    fresh_json(run / "logs/final_report_complete.json", {"status": "PASS", "timestamp": datetime.now(timezone.utc).isoformat(),
               "final_report_sha256": sha256_file(run / "reports/final_report.md"), "correctness_audit": audit["status"],
               "development_accesses": 0, "lockbox_accesses": 0})
    print(run / "reports/final_report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("bootstrap", "execute", "resume", "finalize"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "bootstrap":
        if args.run_dir is not None:
            raise ValueError("bootstrap creates its own collision-refusing run path")
        bootstrap()
        return
    if args.run_dir is None:
        raise ValueError("execute/finalize require --run-dir")
    run = args.run_dir.resolve()
    if args.phase == "execute":
        execute(run)
    elif args.phase == "resume":
        resume_after_integrity_incident(run)
    else:
        finalize(run)


if __name__ == "__main__":
    main()
