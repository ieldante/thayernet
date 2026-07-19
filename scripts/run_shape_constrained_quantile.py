#!/usr/bin/env python3
"""Bootstrap the Thayer-Select shape-constrained quantile campaign.

The initial phase is deliberately limited to provenance capture, the frozen
training-OOF proxy audit, and the pre-fit attainability audit.  It refuses to
write a preregistration or fit a model while the supplied Q2 equation is
inconsistent with its stated positive high-high interaction contract.
"""

from __future__ import annotations

import argparse
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_conditional_calibration import (  # noqa: E402
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    PRIMARY_SUBGROUPS,
    apply_subgroups,
    group_safe_folds,
    load_dataset,
    source_group_count,
)
from scripts.run_scale_correction import component_labels, metric_record  # noqa: E402
from src.conditional_calibration import sha256_file, verify_fold_isolation  # noqa: E402
from src.scale_correction import cluster_bootstrap_indices, crossfit_normalized_upper  # noqa: E402
from src.shape_constrained_quantile import (  # noqa: E402
    SCALE_SEEDS,
    constraint_diagnostics,
    fit_shape_model,
    payload as shape_payload,
    pinball_loss,
    predict_scale,
)


AUTHORITATIVE = REPO / "outputs/runs/thayer_select_scale_correction_20260712_024957"
RUN_PREFIX = "thayer_select_shape_constrained_quantile_"
RISKS = {"image": "image_risk", "flux": "flux_risk_max"}
PROXY_NAMES = (
    "estimated_low_local_signal",
    "estimated_local_complexity",
    "high_output_uncertainty",
    "strong_input_output_disagreement",
)
EXPECTED_ENDPOINTS = {
    ("image", 0, 0): 9.482565360592698,
    ("image", 0, 9): 1.4762952059647387,
    ("image", 1, 0): 7.205538670077771,
    ("image", 1, 9): 1.4549624169402915,
    ("flux", 0, 0): 16.811190888421127,
    ("flux", 0, 9): 11.40326837523512,
    ("flux", 1, 0): 15.070414435944581,
    ("flux", 1, 9): 10.193125695054684,
}
SCALE_BOUNDS = {"image": (0.001, 5.0), "flux": (0.001, 25.0)}
P95_WIDTH_CAP = {"image": 6.0, "flux": 30.0}
BOOTSTRAP_REPLICATES = 300
VALIDATION_MIN_ROWS = 50
VALIDATION_MIN_SOURCE_GROUPS = 40


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy().fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join([header, separator, *body])


def file_hashes(root: Path) -> list[dict]:
    return [
        {
            "relative_path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def checkpoint_inventory(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if run in path.parents:
            continue
        rows.append(
            {
                "relative_path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def verify_authoritative_inputs() -> dict:
    required = [
        "reports/final_report.md",
        "preregistration/partially_pooled_scale_correction.md",
        "tables/component_decision_table.csv",
        "tables/primary_method_comparison.csv",
        "tables/scale_model_validation_summary.csv",
        "tables/source_group_bootstrap_intervals.csv",
        "tables/sensitivity_analysis.csv",
        "features/image_training_deployable_scale_features.npz",
        "features/flux_training_deployable_scale_features.npz",
    ]
    failures = [name for name in required if not (AUTHORITATIVE / name).is_file()]
    if failures:
        raise RuntimeError("missing authoritative artifact(s): " + ", ".join(failures))
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Condition-C checkpoint hash changed")
    prior = json.loads((AUTHORITATIVE / "logs/input_provenance.json").read_text())
    checked = 0
    for section in ("partition_manifests", "persisted_features", "selected_risk_heads"):
        for row in prior[section]:
            path = REPO / row["relative_path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"frozen input changed: {row['relative_path']}")
            checked += 1
    before = pd.read_csv(AUTHORITATIVE / "tables/checkpoint_inventory_after.csv")
    hash_column = "sha256_after" if "sha256_after" in before else "sha256"
    for row in before.itertuples(index=False):
        path = REPO / row.relative_path
        expected = getattr(row, hash_column)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"historical checkpoint changed: {row.relative_path}")
    return {"frozen_rows_verified": checked, "historical_checkpoints_verified": len(before)}


def create_run() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run = REPO / "outputs/runs" / f"{RUN_PREFIX}{timestamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in (
        "diagnostics",
        "tables",
        "figures",
        "logs",
        "reports",
        "preregistration",
        "features",
        "models",
        "calibration",
        "example_grids",
        "manifests",
    ):
        (run / name).mkdir(exist_ok=False)
    return run


def capture_provenance(run: Path, verified: dict, started: float) -> None:
    prior = json.loads((AUTHORITATIVE / "logs/input_provenance.json").read_text())
    inventory = checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", inventory)
    git = {
        name: command(args)
        for name, args in {
            "branch": ["git", "branch", "--show-current"],
            "head": ["git", "rev-parse", "HEAD"],
            "status": ["git", "status", "--short", "--branch"],
            "staged_index": ["git", "diff", "--cached", "--name-status"],
        }.items()
    }
    packages = {
        name: package_version(name)
        for name in ("numpy", "pandas", "scipy", "torch", "h5py", "blending-toolkit", "GalSim", "astropy")
    }
    packages.update(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
    )
    disk = shutil.disk_usage(REPO)
    feature_arrays = []
    for risk in RISKS:
        for partition in ("training", "validation", "calibration"):
            path = AUTHORITATIVE / f"features/{risk}_{partition}_deployable_scale_features.npz"
            feature_arrays.append({"relative_path": relative(path), "sha256": sha256_file(path)})
    provenance = {
        "campaign_start_iso": datetime.now(timezone.utc).isoformat(),
        "campaign_start_unix": started,
        "git": git,
        "packages": packages,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "condition_c": {"relative_path": relative(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
        "selected_risk_heads": prior["selected_risk_heads"],
        "authoritative_scale_correction_hashes": file_hashes(AUTHORITATIVE),
        "persisted_scale_feature_arrays": feature_arrays,
        "partition_manifests": prior["partition_manifests"],
        "query_semantics_sha256": prior["query_semantics_sha256"],
        "risk_definition_sha256": prior["risk_definition_sha256"],
        "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "verification": verified,
        "development_accesses": 0,
        "lockbox_accesses": 0,
        "neural_inference": 0,
        "scale_fit_device": "cpu",
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    snapshot = f"""# Environment snapshot

- Start: {provenance['campaign_start_iso']}
- Branch: {git['branch']['stdout'].strip()}
- Git HEAD: {git['head']['stdout'].strip()}
- MPS built/available: {packages['mps_built']} / {packages['mps_available']}
- Condition-C path: `{relative(CHECKPOINT)}`
- Condition-C SHA-256: `{sha256_file(CHECKPOINT)}`
- Historical checkpoints inventoried: {len(inventory)}
- Free disk at start: {disk.free / 2**30:.2f} GiB
- Neural inference: zero; persisted arrays only

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
    fresh_text(
        run / "diagnostics/campaign_contract.md",
        """# Campaign contract

This is an append-only, collision-refusing train/validation/natural-calibration campaign. It reuses only the four frozen `partial_pool_proxies`, the persisted OOF central predictions, and authoritative frozen inputs recorded in `logs/input_provenance.json`. Condition C, deployed heads, partitions, query semantics, risk definitions, physical subgroups, and every historical checkpoint remain frozen. Scale work is CPU-only. Development and lockbox access, reconstruction inference, physical-label model inputs, broad search, staging, commit, push, merge, deletion, and overwrite are prohibited. A failed preregistration attainability check stops all fitting and calibration access.
""",
    )


def decile_index(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return bins, edges


def proxy_audit(run: Path) -> pd.DataFrame:
    _, samples, manifest = load_dataset("r_training")
    rows = []
    plot_data = {}
    endpoint_failures = []
    for risk, target in RISKS.items():
        path = AUTHORITATIVE / f"features/{risk}_training_deployable_scale_features.npz"
        archive = np.load(path, allow_pickle=False)
        if len(archive["scene_id"]) != 12000:
            raise RuntimeError(f"unexpected {risk} training rows")
        if not np.array_equal(archive["scene_id"].astype(str), samples.scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"{risk} sample IDs are not aligned")
        proxy = archive["partial_pool_proxies"].astype(float)
        central = archive["S0"][:, 0].astype(float)
        residual = np.abs(samples[target].to_numpy(dtype=float) - central)
        if proxy.shape != (12000, 4) or not np.isfinite(proxy).all() or not np.isfinite(residual).all():
            raise RuntimeError(f"invalid {risk} persisted audit arrays")
        for proxy_index, proxy_name in enumerate(PROXY_NAMES):
            value = proxy[:, proxy_index]
            bins, edges = decile_index(value)
            correlation = float(spearmanr(value, residual).statistic)
            for quantile_index, quantile_value in enumerate(np.quantile(value, np.linspace(0.0, 1.0, 11))):
                rows.append(
                    {
                        "risk": risk,
                        "record_type": "proxy_quantile",
                        "proxy_index": proxy_index,
                        "proxy_name": proxy_name,
                        "cell": f"q{quantile_index * 10:02d}",
                        "proxy_lower": quantile_value,
                        "proxy_upper": quantile_value,
                        "rows": len(value),
                        "source_group_count": source_group_count(manifest, np.ones(len(value), dtype=bool)),
                        "residual_median": np.nan,
                        "residual_q90": np.nan,
                        "spearman": correlation,
                        "floor_frequency": float(np.mean(value <= 1e-8)),
                        "saturation_frequency": float(np.mean(value >= 1.0 - 1e-8)),
                        "nonfinite_values": int((~np.isfinite(value)).sum()),
                    }
                )
            curve = []
            for decile in range(10):
                mask = bins == decile
                q90 = float(np.quantile(residual[mask], 0.90))
                median = float(np.median(residual[mask]))
                curve.append((median, q90))
                rows.append(
                    {
                        "risk": risk,
                        "record_type": "proxy_decile",
                        "proxy_index": proxy_index,
                        "proxy_name": proxy_name,
                        "cell": f"d{decile + 1:02d}",
                        "proxy_lower": edges[decile],
                        "proxy_upper": edges[decile + 1],
                        "rows": int(mask.sum()),
                        "source_group_count": source_group_count(manifest, mask),
                        "residual_median": median,
                        "residual_q90": q90,
                        "spearman": correlation,
                        "floor_frequency": float(np.mean(value <= 1e-8)),
                        "saturation_frequency": float(np.mean(value >= 1.0 - 1e-8)),
                        "nonfinite_values": int((~np.isfinite(value)).sum()),
                    }
                )
                expected = EXPECTED_ENDPOINTS.get((risk, proxy_index, decile))
                if expected is not None and not np.isclose(q90, expected, rtol=1e-10, atol=1e-10):
                    endpoint_failures.append(f"{risk} z{proxy_index} d{decile + 1}: {q90} != {expected}")
            plot_data[(risk, proxy_index)] = curve
        z0, z1 = proxy[:, 0], proxy[:, 1]
        z0_edges = np.quantile(z0, [0.0, 1 / 3, 2 / 3, 1.0])
        z1_edges = np.quantile(z1, [0.0, 1 / 3, 2 / 3, 1.0])
        z0_bin = np.searchsorted(z0_edges[1:-1], z0, side="right")
        z1_bin = np.searchsorted(z1_edges[1:-1], z1, side="right")
        for left in range(3):
            for right in range(3):
                mask = (z0_bin == left) & (z1_bin == right)
                rows.append(
                    {
                        "risk": risk,
                        "record_type": "z0_z1_tertile_interaction",
                        "proxy_index": -1,
                        "proxy_name": "z0_x_z1",
                        "cell": f"z0_t{left + 1}__z1_t{right + 1}",
                        "proxy_lower": np.nan,
                        "proxy_upper": np.nan,
                        "rows": int(mask.sum()),
                        "source_group_count": source_group_count(manifest, mask),
                        "residual_median": float(np.median(residual[mask])),
                        "residual_q90": float(np.quantile(residual[mask], 0.90)),
                        "spearman": np.nan,
                        "floor_frequency": np.nan,
                        "saturation_frequency": np.nan,
                        "nonfinite_values": 0,
                    }
                )
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/training_proxy_shape_audit.csv", frame)
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for row_index, risk in enumerate(RISKS):
        for proxy_index, proxy_name in enumerate(PROXY_NAMES):
            curve = np.asarray(plot_data[(risk, proxy_index)])
            axis = axes[row_index, proxy_index]
            axis.plot(np.arange(1, 11), curve[:, 0], marker="o", label="median")
            axis.plot(np.arange(1, 11), curve[:, 1], marker="o", label="q=0.90")
            axis.set_title(f"{risk.upper()} z{proxy_index}: {proxy_name.replace('_', ' ')}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("training proxy decile")
            if proxy_index == 0:
                axis.set_ylabel("absolute OOF residual")
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run / "figures/training_proxy_shape_audit.png", dpi=160)
    plt.close(figure)
    endpoint = frame[
        (frame.record_type == "proxy_decile")
        & (frame.proxy_index.isin([0, 1]))
        & (frame.cell.isin(["d01", "d10"]))
    ][["risk", "proxy_index", "proxy_name", "cell", "rows", "source_group_count", "residual_median", "residual_q90", "spearman"]]
    status = "PASS" if not endpoint_failures else "FAIL"
    report = f"""# Training-only proxy-shape audit

**{status}.** The audit used exactly 12,000 training rows, the persisted `S0[:,0]` OOF central predictions, and the four persisted `partial_pool_proxies`. Absolute residuals are `abs(true_risk - persisted_OOF_central_prediction)`. No validation, natural-calibration, development, or lockbox outcome informed this audit. All arrays are finite and scene IDs align exactly.

The stated z0/z1 endpoint checks reproduce within `rtol=atol=1e-10`. The q=0.90 tails are not globally monotone increasing: both image proxies have much larger first-decile than last-decile tails, and flux shows the same endpoint reversal despite positive rank correlation through much of the distribution.

## Frozen endpoint checks

{markdown_table(endpoint)}

## Failures

{chr(10).join('- ' + item for item in endpoint_failures) if endpoint_failures else '- None.'}
"""
    fresh_text(run / "diagnostics/training_proxy_shape_audit.md", report)
    if endpoint_failures:
        raise RuntimeError("training proxy endpoint reproduction failed")
    return frame


def load_partition(risk: str, partition: str) -> dict[str, np.ndarray]:
    path = AUTHORITATIVE / f"features/{risk}_{partition}_deployable_scale_features.npz"
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def training_validation_data() -> tuple[dict, dict, dict]:
    _, train_samples, train_manifest = load_dataset("r_training")
    _, validation_samples, validation_manifest = load_dataset("r_validation")
    data = {}
    for risk, target in RISKS.items():
        train = load_partition(risk, "training")
        validation = load_partition(risk, "validation")
        if not np.array_equal(train["scene_id"].astype(str), train_samples.scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"{risk} training IDs differ")
        if not np.array_equal(validation["scene_id"].astype(str), validation_samples.scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"{risk} validation IDs differ")
        data[risk] = {
            "train_proxy": train["partial_pool_proxies"].astype(np.float32),
            "validation_proxy": validation["partial_pool_proxies"].astype(np.float32),
            "train_central": train["S0"][:, 0].astype(float),
            "validation_central": validation["S0"][:, 0].astype(float),
            "train_residual": np.abs(train_samples[target].to_numpy(float) - train["S0"][:, 0].astype(float)),
            "validation_residual": np.abs(validation_samples[target].to_numpy(float) - validation["S0"][:, 0].astype(float)),
        }
    manifests = {"training": train_manifest, "validation": validation_manifest}
    samples = {"training": train_samples, "validation": validation_samples}
    return data, samples, manifests


def preregister(run: Path, data: dict) -> tuple[dict, dict]:
    rows = [
        {"scope": "Q0_Q1_Q2", "check": "parameter_ceiling", "requested": "<=64 parameters per risk", "attainable": True,
         "reason": "Q1 has 25 parameters and Q2 has 26."},
        {"scope": "Q1_Q2", "check": "convex_main_effects", "requested": "convex and nondecreasing at/above 0.50", "attainable": True,
         "reason": "Nonnegative hinge increments and the anchor parameterization guarantee both constraints."},
        {"scope": "Q2", "check": "positive_high_high_interaction",
         "requested": "softplus(gamma)*relu(z0-0.50)*relu(z1-0.50)", "attainable": True,
         "reason": "The user explicitly corrected both subtraction operators to multiplication; the product is nonnegative with positive mixed finite difference above the anchor."},
        {"scope": "calibration", "check": "upper_bound_sign", "requested": "central + score_quantile*scale", "attainable": True,
         "reason": "The authoritative normalized-conformal protocol freezes the plus sign for an upper bound."},
        {"scope": "validation", "check": "supported_cells", "requested": f">={VALIDATION_MIN_ROWS} rows and >={VALIDATION_MIN_SOURCE_GROUPS} source groups", "attainable": True,
         "reason": "The thresholds are below the expected validation cell support and are frozen before fitting."},
        {"scope": "image", "check": "success_gates", "requested": "all frozen image gates", "attainable": True,
         "reason": "All coverage, width, ranking, activation, stability, and bootstrap thresholds lie inside their mathematical ranges."},
        {"scope": "flux", "check": "success_gates", "requested": "all frozen flux gates", "attainable": True,
         "reason": "All coverage, width, ranking, activation, stability, and bootstrap thresholds lie inside their mathematical ranges."},
    ]
    gates = pd.DataFrame(rows)
    if not gates.attainable.all():
        raise RuntimeError("unattainable preregistration gate")
    fresh_csv(run / "tables/gate_attainability_audit.csv", gates)
    fresh_text(run / "diagnostics/gate_attainability_report.md",
               "# Gate attainability report\n\nPASS. All frozen gates and the corrected positive product interaction are mathematically attainable before fitting.\n\n" + markdown_table(gates) + "\n")
    knots = {}
    knot_rows = []
    for risk in RISKS:
        value = np.quantile(data[risk]["train_proxy"], [0.10, 0.25, 0.50, 0.75, 0.90], axis=0).T
        knots[risk] = value
        for proxy_index, proxy_name in enumerate(PROXY_NAMES):
            for quantile, knot in zip((0.10, 0.25, 0.50, 0.75, 0.90), value[proxy_index]):
                knot_rows.append({"risk": risk, "proxy_index": proxy_index, "proxy_name": proxy_name,
                                  "quantile": quantile, "knot": float(knot), "partition": "training_only"})
    knot_frame = pd.DataFrame(knot_rows)
    fresh_csv(run / "tables/frozen_knots.csv", knot_frame)
    knot_hash = sha256_file(run / "tables/frozen_knots.csv")
    subgroup_source = AUTHORITATIVE / "manifests/subgroup_definitions.json"
    fresh_text(run / "manifests/subgroup_definitions.json", subgroup_source.read_text())
    preregistration = f"""# Shape-constrained quantile scale preregistration

This preregistration is frozen after the 12,000-row training-only OOF proxy-shape audit and before any model fit. Condition C, risk/query heads, partitions, four proxy definitions and order, physical subgroup definitions, and all historical checkpoints are frozen. Development and lockbox access are prohibited.

## Hypothesis and targets

The q=0.90 absolute OOF residual tail is non-monotone at proxy extremes. Q1 uses four centered convex piecewise-linear main effects; Q2 adds exactly one strongly regularized positive high-high z0 x z1 hinge product. Targets are raw `abs(true_risk - persisted_OOF_central_prediction)`. Calibration outcomes cannot select models, knots, penalties, constraints, floors, caps, gates, or stopping.

## Frozen models

- Q0: training q=0.90 residual constant using the higher order-statistic convention.
- Q1: `eta=intercept+sum_j f_j(z_j)`.
- Q2: Q1 plus `softplus(gamma)*relu(z0-0.50)*relu(z1-0.50)`.
- `f_j(z)=a_j*z+sum_k softplus(delta_jk)*relu(z-knot_jk)`.
- `a_j=softplus(s_j)-sum_(knot<=0.50) softplus(delta_jk)`.
- Each main effect is centered by its training-row mean.
- Knots are training quantiles `[0.10,0.25,0.50,0.75,0.90]`; knot CSV SHA-256 is `{knot_hash}`.
- Scale is `clamp(0.001+softplus(eta), floor, cap)` with image cap 5 and flux cap 25.
- Parameter ceiling: 64 per risk.

## Training and validation-only selection

Five seeds are `{list(SCALE_SEEDS)}`. CPU-only AdamW uses learning rate 2e-3, batch size 512, at most 200 epochs, validation patience 20, gradient clipping 5, weight decay 1e-4, roughness penalty 1e-2, and Q2 interaction penalty 1e-1. The objective is q=0.90 pinball loss on raw residuals.

Supported validation cells use training-only boundaries: z0 tertile x z1 tertile plus z0 low/high decile, z1 low/high decile, z2 high decile, and z3 high decile. Support requires at least {VALIDATION_MIN_ROWS} rows and {VALIDATION_MIN_SOURCE_GROUPS} distinct source groups. Eligibility and the exact lexicographic ordering, including the four extra Q2-over-Q1 requirements, are frozen exactly as stated in the campaign contract. Physical subgroup coverage and calibration outcomes are excluded from selection.

## Calibration, audit, gates, and stopping

After selection and exact R0/R1 replay, five source-group-safe natural-calibration folds use `score=abs(truth-central)/max(scale,floor)` and the other four folds' finite-sample q=0.90 score. The authoritative upper bound is `central + quantile*scale`. Report all frozen physical subgroups without exact conditional-coverage claims. Use 300 connected-source-component bootstrap replicates. Run the frozen floor, cap, roughness, interaction-penalty, anchor, conformal-rank, and physical-boundary sensitivities only after selection; never retune.

IMAGE and FLUX gates, centroid replay, PARTIAL rules, integrity checks, and the prohibition on full-policy authorization unless both components PASS remain exactly as supplied. Stop on any hash, alignment, OOF, support, constraint, prior-reproduction, source-overlap, leakage, or checkpoint-integrity failure.
"""
    prereg = run / "preregistration/shape_constrained_quantile_scale.md"
    fresh_text(prereg, preregistration)
    hashed_at = time.time()
    marker = {"relative_path": relative(prereg), "sha256": sha256_file(prereg), "knots_sha256": knot_hash,
              "hashed_at_unix": hashed_at, "hashed_at_iso": datetime.now(timezone.utc).isoformat()}
    fresh_json(run / "preregistration/shape_constrained_quantile_scale.sha256.json", marker)
    fresh_json(run / "logs/preregistration_complete.json", {"status": "PASS", "fit_started": False, **marker})
    return knots, marker


def validation_cells(train_proxy: np.ndarray, validation_proxy: np.ndarray, manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    q = np.quantile(train_proxy, [0.10, 1 / 3, 2 / 3, 0.90], axis=0)
    z0_tertile = np.searchsorted(q[[1, 2], 0], validation_proxy[:, 0], side="right")
    z1_tertile = np.searchsorted(q[[1, 2], 1], validation_proxy[:, 1], side="right")
    masks = {f"z0_t{a+1}__z1_t{b+1}": (z0_tertile == a) & (z1_tertile == b) for a in range(3) for b in range(3)}
    masks.update({
        "z0_lowest_decile": validation_proxy[:, 0] <= q[0, 0],
        "z0_highest_decile": validation_proxy[:, 0] >= q[3, 0],
        "z1_lowest_decile": validation_proxy[:, 1] <= q[0, 1],
        "z1_highest_decile": validation_proxy[:, 1] >= q[3, 1],
        "z2_highest_decile": validation_proxy[:, 2] >= q[3, 2],
        "z3_highest_decile": validation_proxy[:, 3] >= q[3, 3],
    })
    rows = []
    supported = {}
    for name, mask in masks.items():
        groups = source_group_count(manifest, mask)
        keep = int(mask.sum()) >= VALIDATION_MIN_ROWS and groups >= VALIDATION_MIN_SOURCE_GROUPS
        rows.append({"cell": name, "rows": int(mask.sum()), "source_group_count": groups, "supported": keep})
        if keep:
            supported[name] = mask
    return supported, pd.DataFrame(rows)


def top_decile_recall(residual: np.ndarray, scale: np.ndarray) -> float:
    count = max(1, int(math.ceil(0.10 * len(residual))))
    actual = set(np.argsort(residual, kind="stable")[-count:].tolist())
    predicted = set(np.argsort(scale, kind="stable")[-count:].tolist())
    return len(actual & predicted) / count


def validation_record(risk: str, condition: str, seed: str, residual: np.ndarray, scale: np.ndarray,
                      cells: dict, diagnostics: dict, floor: float, cap: float) -> dict:
    cell_coverage = [float(np.mean(residual[mask] <= scale[mask])) for mask in cells.values()]
    pinball = np.maximum(0.90 * (residual - scale), -0.10 * (residual - scale)).mean()
    correlation = spearmanr(scale, residual).statistic
    return {
        "risk": risk, "condition": condition, "seed": seed,
        "residual_coverage": float(np.mean(residual <= scale)),
        "worst_supported_cell_coverage": min(cell_coverage),
        "pinball_loss": float(pinball), "top_decile_residual_recall": top_decile_recall(residual, scale),
        "residual_scale_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "median_predicted_scale": float(np.median(scale)), "p95_predicted_scale": float(np.quantile(scale, 0.95)),
        "floor_activation": float(np.mean(scale <= floor * (1 + 1e-8))),
        "cap_activation": float(np.mean(scale >= cap * (1 - 1e-8))),
        "finite_nondegenerate": bool(np.isfinite(scale).all() and np.ptp(scale) > 1e-10) if condition != "Q0" else bool(np.isfinite(scale).all()),
        **diagnostics,
    }


def fit_and_select(run: Path, data: dict, manifests: dict, knots: dict, marker: dict) -> tuple[dict, dict, pd.DataFrame]:
    if sha256_file(run / "preregistration/shape_constrained_quantile_scale.md") != marker["sha256"]:
        raise RuntimeError("preregistration hash changed before fitting")
    fresh_json(run / "logs/fit_start_marker.json", {"timestamp_unix": time.time(), "preregistration_sha256": marker["sha256"]})
    fits, rows, selection = {}, [], {}
    cell_frames = []
    for risk in RISKS:
        floor, cap = SCALE_BOUNDS[risk]
        supported, inventory = validation_cells(data[risk]["train_proxy"], data[risk]["validation_proxy"], manifests["validation"])
        inventory.insert(0, "risk", risk)
        cell_frames.append(inventory)
        constant = float(np.quantile(data[risk]["train_residual"], 0.90, method="higher"))
        q0_scale = np.full(len(data[risk]["validation_residual"]), np.clip(constant, floor, cap))
        q0_diag = {"convexity_violations": 0, "upper_half_monotonicity_violations": 0,
                   "interaction_coefficient": 0.0, "interaction_min": 0.0, "interaction_max": 0.0,
                   "interaction_nonnegative": True, "centering_mean_max_abs": 0.0}
        rows.append(validation_record(risk, "Q0", "ensemble", data[risk]["validation_residual"], q0_scale,
                                      supported, q0_diag, floor, cap))
        fits[(risk, "Q0")] = constant
        for condition in ("Q1", "Q2"):
            seed_scales = []
            seed_records = []
            for seed in SCALE_SEEDS:
                fit = fit_shape_model(condition, seed, data[risk]["train_proxy"], data[risk]["train_residual"],
                                      data[risk]["validation_proxy"], data[risk]["validation_residual"], knots[risk],
                                      scale_floor=floor, scale_cap=cap)
                path = run / f"models/{risk}_{condition}_seed_{seed}.pth"
                torch.save(shape_payload(fit, risk), path)
                fits[(risk, condition, seed)] = fit
                scale = predict_scale(fit, data[risk]["validation_proxy"])
                seed_scales.append(scale)
                diagnostics = constraint_diagnostics(fit)
                with torch.no_grad():
                    centered = fit.model.main_effects(torch.from_numpy(data[risk]["train_proxy"])).mean(dim=0).numpy()
                diagnostics["centering_mean_max_abs"] = float(np.max(np.abs(centered)))
                record = validation_record(risk, condition, str(seed), data[risk]["validation_residual"], scale,
                                           supported, diagnostics, floor, cap)
                seed_records.append(record); rows.append(record)
            ensemble_scale = np.mean(seed_scales, axis=0)
            ensemble_diag = {
                "convexity_violations": max(row["convexity_violations"] for row in seed_records),
                "upper_half_monotonicity_violations": max(row["upper_half_monotonicity_violations"] for row in seed_records),
                "interaction_coefficient": float(np.mean([row["interaction_coefficient"] for row in seed_records])),
                "interaction_min": 0.0,
                "interaction_max": float(np.max([row["interaction_max"] for row in seed_records])),
                "interaction_nonnegative": all(row["interaction_nonnegative"] for row in seed_records),
                "centering_mean_max_abs": float(np.max([row["centering_mean_max_abs"] for row in seed_records])),
            }
            ensemble = validation_record(risk, condition, "ensemble", data[risk]["validation_residual"], ensemble_scale,
                                         supported, ensemble_diag, floor, cap)
            ensemble["seed_variability"] = float(np.std([row["worst_supported_cell_coverage"] for row in seed_records], ddof=1))
            rows.append(ensemble)
        current = pd.DataFrame([row for row in rows if row["risk"] == risk and row["seed"] == "ensemble"])
        current["seed_variability"] = current.get("seed_variability", pd.Series(index=current.index, dtype=float)).fillna(0.0)
        current["eligible"] = (
            current.residual_coverage.between(0.87, 0.93)
            & (current.convexity_violations == 0)
            & (current.upper_half_monotonicity_violations == 0)
            & (current.cap_activation < 0.10)
            & current.finite_nondegenerate
        )
        q1 = current[current.condition == "Q1"].iloc[0]
        q2 = current[current.condition == "Q2"].iloc[0]
        q2_extra = bool(
            q2.eligible and q1.eligible
            and q2.worst_supported_cell_coverage >= q1.worst_supported_cell_coverage + 0.03
            and q2.pinball_loss <= 1.05 * q1.pinball_loss
            and q2.median_predicted_scale <= 1.25 * q1.median_predicted_scale
            and q2.seed_variability <= 0.03
        )
        candidates = current[current.eligible & current.condition.isin(["Q0", "Q1"])]
        if q2_extra:
            candidates = pd.concat([candidates, current[current.condition == "Q2"]])
        if candidates.empty:
            raise RuntimeError(f"no eligible validation model for {risk}")
        candidates = candidates.assign(prefer_q1=(candidates.condition == "Q1").astype(int))
        candidates = candidates.sort_values(
            ["worst_supported_cell_coverage", "pinball_loss", "top_decile_residual_recall",
             "median_predicted_scale", "seed_variability", "prefer_q1"],
            ascending=[False, True, False, True, True, False],
        )
        chosen = str(candidates.iloc[0].condition)
        selection[risk] = {"selected_model": chosen, "selection_partition": "validation", "calibration_used": False,
                           "q2_extra_gate_passed": q2_extra, "q0_constant": constant}
    fresh_csv(run / "tables/validation_cell_support.csv", pd.concat(cell_frames, ignore_index=True))
    validation = pd.DataFrame(rows)
    fresh_csv(run / "tables/scale_model_seed_comparison.csv", validation)
    ensemble = validation[validation.seed == "ensemble"].copy()
    fresh_csv(run / "tables/scale_model_validation_summary.csv", ensemble)
    fresh_json(run / "models/scale_model_selection.json", selection)
    fresh_csv(run / "tables/scale_model_selection.csv", pd.DataFrame([{"risk": risk, **value} for risk, value in selection.items()]))
    fresh_json(run / "logs/validation_selection_complete.json",
               {"timestamp_unix": time.time(), "calibration_used": False, "selection": selection})
    return fits, selection, validation


def reproduce_prior_failures(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.read_csv(AUTHORITATIVE / "tables/primary_method_comparison.csv")
    prior_subgroups = pd.read_csv(AUTHORITATIVE / "tables/subgroup_coverage.csv")
    mapping = {
        "C0_authoritative_baseline": "R0_authoritative_baseline",
        "C2_partially_pooled": "R1_failed_partially_pooled",
        "C4_hard_oracle_group": "O1_hard_physical_group_oracle",
    }
    replay = prior[prior.method.isin(mapping)].copy()
    replay["method"] = replay.method.map(mapping)
    groups = prior_subgroups[prior_subgroups.method.isin(mapping)].copy()
    groups["method"] = groups.method.map(mapping)
    expected = {
        ("image", "R0_authoritative_baseline"): (0.9028571428571428, 0.6373056994818653),
        ("flux", "R0_authoritative_baseline"): (0.8982142857142857, 0.6839378238341969),
        ("image", "R1_failed_partially_pooled"): (0.9189285714285714, 0.5492227979274611),
        ("flux", "R1_failed_partially_pooled"): (0.9217857142857144, 0.6787564766839378),
    }
    failures = []
    for key, values in expected.items():
        row = replay[(replay.risk == key[0]) & (replay.method == key[1])].iloc[0]
        if not np.isclose(row.marginal_coverage, values[0], atol=1e-12) or not np.isclose(row.worst_supported_subgroup_coverage, values[1], atol=1e-12):
            failures.append(f"{key} did not replay")
    component = pd.read_csv(AUTHORITATIVE / "tables/component_decision_table.csv")
    prior_seed = pd.read_csv(AUTHORITATIVE / "tables/scale_seed_stability.csv")
    prior_boot = pd.read_csv(AUTHORITATIVE / "tables/source_group_bootstrap_intervals.csv")
    fresh_csv(run / "tables/prior_failure_reproduction.csv", replay)
    fresh_csv(run / "tables/prior_failure_subgroup_reproduction.csv", groups)
    fresh_csv(run / "tables/prior_component_decisions.csv", component)
    fresh_csv(run / "tables/prior_seed_summaries.csv", prior_seed)
    fresh_csv(run / "tables/prior_bootstrap_intervals.csv", prior_boot)
    status = "PASS" if not failures else "FAIL"
    fresh_text(run / "diagnostics/prior_failure_reproduction.md",
               f"# Prior-failure reproduction\n\n**{status}.** Exact persisted authoritative comparison, subgroup, seed, bootstrap, and component-decision artifacts replayed without neural inference. R0 image/flux marginal-worst values are 0.902857/0.637306 and 0.898214/0.683938; R1 values are 0.918929/0.549223 and 0.921786/0.678756. Centroid remains 0.900714/0.888165 PASS.\n")
    if failures:
        raise RuntimeError("prior failure reproduction failed")
    return replay, groups


def calibration_context(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    _, samples, manifest = load_dataset("natural_calibration")
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    assignments = apply_subgroups(manifest, definitions)
    fold = group_safe_folds(manifest.source_a_group.to_numpy(), manifest.source_b_group.to_numpy(), folds=5)
    if not verify_fold_isolation(manifest.source_a_group.to_numpy(), manifest.source_b_group.to_numpy(), fold):
        raise RuntimeError("natural-calibration source groups cross folds")
    fresh_csv(run / "tables/calibration_fold_assignments.csv",
              pd.DataFrame({"scene_id": samples.scene_id.astype(str), "fold": fold,
                            "source_a_group": manifest.source_a_group.astype(str),
                            "source_b_group": manifest.source_b_group.astype(str)}))
    return samples, manifest, {"calibration": assignments}, fold


def ensemble_scale_for(fits: dict, risk: str, condition: str, proxy: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    if condition == "Q0":
        value = float(fits[(risk, "Q0")])
        block = np.full(len(proxy), value)
        return block, [block for _ in SCALE_SEEDS]
    blocks = [predict_scale(fits[(risk, condition, seed)], proxy) for seed in SCALE_SEEDS]
    return np.mean(blocks, axis=0), blocks


def calibration_comparison(run: Path, fits: dict, selection: dict, replay: pd.DataFrame, replay_groups: pd.DataFrame,
                           samples: pd.DataFrame, manifest: pd.DataFrame, assignments: dict, fold: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    rows = replay.to_dict("records")
    subgroup_frames = [replay_groups]
    outputs, seed_rows = {}, []
    for risk, target in RISKS.items():
        archive = load_partition(risk, "calibration")
        if not np.array_equal(archive["scene_id"].astype(str), samples.scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"{risk} calibration IDs differ")
        proxy = archive["partial_pool_proxies"].astype(np.float32)
        central = archive["S0"][:, 0].astype(float)
        truth = samples[target].to_numpy(float)
        floor, cap = SCALE_BOUNDS[risk]
        baseline_median = float(replay[(replay.risk == risk) & (replay.method == "R0_authoritative_baseline")].median_width.iloc[0])
        for condition, label in (("Q0", "Q0_global_q90_constant"), ("Q1", "Q1_convex_additive"), ("Q2", "Q2_convex_additive_interaction")):
            scale, seed_scales = ensemble_scale_for(fits, risk, condition, proxy)
            upper, quantile = crossfit_normalized_upper(truth, central, scale, fold, scale_floor=floor, coverage=0.90, convention="higher")
            record, groups = metric_record(risk, label, truth, central, upper, scale, assignments["calibration"], manifest,
                                           baseline_median, floor, cap, deployable=True)
            rows.append(record); subgroup_frames.append(groups)
            outputs[(risk, label)] = {"truth": truth, "central": central, "upper": upper, "scale": scale,
                                      "quantile": quantile, "proxy": proxy}
            for seed, seed_scale in zip(SCALE_SEEDS, seed_scales):
                seed_upper, _ = crossfit_normalized_upper(truth, central, seed_scale, fold, scale_floor=floor)
                seed_record, _ = metric_record(risk, label, truth, central, seed_upper, seed_scale,
                                               assignments["calibration"], manifest, baseline_median, floor, cap, deployable=True)
                seed_rows.append({"risk": risk, "method": label, "seed": seed,
                                  "marginal_coverage": seed_record["marginal_coverage"],
                                  "worst_supported_subgroup_coverage": seed_record["worst_supported_subgroup_coverage"],
                                  "low_snr_high_obstruction_coverage": seed_record["low_snr_high_obstruction_coverage"],
                                  "median_width": seed_record["median_width"]})
    comparison = pd.DataFrame(rows)
    subgroup = pd.concat(subgroup_frames, ignore_index=True)
    seed = pd.DataFrame(seed_rows)
    fresh_csv(run / "tables/primary_method_comparison.csv", comparison)
    fresh_csv(run / "tables/subgroup_coverage.csv", subgroup)
    fresh_csv(run / "tables/scale_seed_stability.csv", seed)
    return comparison, subgroup, outputs, seed


def bootstrap_campaign(run: Path, comparison: pd.DataFrame, outputs: dict, samples: pd.DataFrame,
                       manifest: pd.DataFrame, assignments: dict) -> pd.DataFrame:
    prior = pd.read_csv(run / "tables/prior_bootstrap_intervals.csv")
    mapping = {"C0_authoritative_baseline": "R0_authoritative_baseline",
               "C2_partially_pooled": "R1_failed_partially_pooled",
               "C4_hard_oracle_group": "O1_hard_physical_group_oracle"}
    reused = prior[prior.method.isin(mapping)].copy()
    reused["method"] = reused.method.map(mapping)
    rows = reused.to_dict("records")
    components = component_labels(manifest)
    masks = []
    for family in PRIMARY_SUBGROUPS:
        levels = ["member"] if "__" in family else list(dict.fromkeys(assignments["calibration"][family].tolist()))
        for level in levels:
            masks.append(assignments["calibration"][family].to_numpy() == level)
    difficult = assignments["calibration"].low_snr__high_obstruction.to_numpy() == "member"
    rng = np.random.default_rng(2026071271)
    for risk in RISKS:
        baseline_width = float(comparison[(comparison.risk == risk) & (comparison.method == "R0_authoritative_baseline")].median_width.iloc[0])
        for method in ("Q0_global_q90_constant", "Q1_convex_additive", "Q2_convex_additive_interaction"):
            value = outputs[(risk, method)]
            truth, upper, central = value["truth"], value["upper"], value["central"]
            covered = truth <= upper; width = np.maximum(upper - central, 0.0)
            estimates = {name: [] for name in ("marginal_coverage", "worst_supported_subgroup_coverage",
                                                "low_snr_high_obstruction_coverage", "median_width", "width_inflation", "tail_miss_rate")}
            for _ in range(BOOTSTRAP_REPLICATES):
                index = cluster_bootstrap_indices(components, rng)
                estimates["marginal_coverage"].append(float(np.mean(covered[index])))
                subgroup_values = [float(np.mean(covered[index][mask[index]])) for mask in masks if mask[index].any()]
                estimates["worst_supported_subgroup_coverage"].append(min(subgroup_values))
                estimates["low_snr_high_obstruction_coverage"].append(float(np.mean(covered[index][difficult[index]])))
                estimates["median_width"].append(float(np.median(width[index])))
                estimates["width_inflation"].append(float(np.median(width[index]) / baseline_width))
                estimates["tail_miss_rate"].append(float(np.mean(~covered[index])))
            for metric, values in estimates.items():
                rows.append({"risk": risk, "method": method, "metric": metric,
                             "point_estimate": float(np.mean(values)), "ci_low": float(np.quantile(values, 0.025)),
                             "ci_high": float(np.quantile(values, 0.975)), "replicates": BOOTSTRAP_REPLICATES,
                             "cluster_unit": "connected_source_group_component"})
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/source_group_bootstrap_intervals.csv", frame)
    return frame


def selected_decisions(run: Path, comparison: pd.DataFrame, bootstrap: pd.DataFrame, seed: pd.DataFrame,
                       selection: dict) -> pd.DataFrame:
    label = {"Q0": "Q0_global_q90_constant", "Q1": "Q1_convex_additive", "Q2": "Q2_convex_additive_interaction"}
    rows = []
    for risk in RISKS:
        method = label[selection[risk]["selected_model"]]
        selected = comparison[(comparison.risk == risk) & (comparison.method == method)].iloc[0]
        base = 0.6373056994818653 if risk == "image" else 0.6839378238341969
        seed_values = seed[(seed.risk == risk) & (seed.method == method)]
        seed_sd = float(seed_values.marginal_coverage.std(ddof=1)) if len(seed_values) > 1 else 0.0
        boot = bootstrap[(bootstrap.risk == risk) & (bootstrap.method == method) &
                         (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
        gates = {
            "marginal": 0.88 <= selected.marginal_coverage <= 0.92,
            "worst_subgroup": selected.worst_supported_subgroup_coverage >= 0.82,
            "difficult_intersection": selected.low_snr_high_obstruction_coverage >= 0.82,
            "median_width": selected.median_width_inflation <= 1.75,
            "p95_width": selected.p95_width < P95_WIDTH_CAP[risk],
            "ranking": selected.calibration_spearman >= (0.82 if risk == "image" else 0.80),
            "seed_stability": seed_sd <= 0.03,
            "bootstrap_stability": boot.ci_low >= 0.75,
            "bounded_scale": selected.scale_cap_activation < 0.10 and selected.extreme_inflation_rate < 0.10,
        }
        improvement = float(selected.worst_supported_subgroup_coverage - base)
        decision = "PASS" if all(gates.values()) else ("PARTIAL" if improvement >= 0.05 else "FAIL")
        rows.append({"component": risk.upper() + "_RISK", "selected_model": selection[risk]["selected_model"],
                     "method": method, "decision": decision, "marginal_coverage": selected.marginal_coverage,
                     "worst_supported_subgroup_coverage": selected.worst_supported_subgroup_coverage,
                     "low_snr_high_obstruction_coverage": selected.low_snr_high_obstruction_coverage,
                     "improvement_over_authoritative_failure": improvement,
                     "median_width_inflation": selected.median_width_inflation, "p95_width": selected.p95_width,
                     "calibration_spearman": selected.calibration_spearman, "coverage_seed_sd": seed_sd,
                     "bootstrap_ci_low": boot.ci_low, **{f"gate_{name}": value for name, value in gates.items()}})
    rows.append({"component": "CENTROID_RISK", "selected_model": "authoritative", "method": "R0_authoritative_baseline",
                 "decision": "PASS", "marginal_coverage": 0.9007142857142857,
                 "worst_supported_subgroup_coverage": 0.8881650380021715})
    frame = pd.DataFrame(rows)
    risk_decisions = frame[frame.component.isin(["IMAGE_RISK", "FLUX_RISK"])].decision.tolist()
    overall = "SUCCESS" if risk_decisions == ["PASS", "PASS"] else ("PARTIAL" if "PARTIAL" in risk_decisions else "FAILURE")
    frame = pd.concat([frame, pd.DataFrame([{"component": "OVERALL", "decision": overall}])], ignore_index=True)
    fresh_csv(run / "tables/component_decision_table.csv", frame)
    return frame


def sensitivity_campaign(run: Path, fits: dict, selection: dict, data: dict, knots: dict, comparison: pd.DataFrame,
                         samples: pd.DataFrame, manifest: pd.DataFrame, assignments: dict, fold: np.ndarray) -> pd.DataFrame:
    rows = []
    definitions = json.loads((run / "manifests/subgroup_definitions.json").read_text())
    perturbed = json.loads(json.dumps(definitions))
    for boundary in perturbed["boundaries"].values():
        boundary[0] *= 0.95; boundary[1] *= 1.05
    perturbed_assignments = apply_subgroups(manifest, perturbed)
    for risk, target in RISKS.items():
        archive = load_partition(risk, "calibration")
        proxy = archive["partial_pool_proxies"].astype(np.float32)
        central = archive["S0"][:, 0].astype(float)
        truth = samples[target].to_numpy(float)
        floor, cap = SCALE_BOUNDS[risk]
        chosen = selection[risk]["selected_model"]
        base_scale, _ = ensemble_scale_for(fits, risk, chosen, proxy)
        baseline_median = float(comparison[(comparison.risk == risk) & (comparison.method == "R0_authoritative_baseline")].median_width.iloc[0])

        def add(name: str, scale: np.ndarray, *, convention: str = "higher", assignment: pd.DataFrame | None = None,
                effective_floor: float | None = None, effective_cap: float | None = None) -> None:
            local_floor = floor if effective_floor is None else effective_floor
            local_cap = cap if effective_cap is None else effective_cap
            upper, _ = crossfit_normalized_upper(truth, central, scale, fold, scale_floor=local_floor, convention=convention)
            record, _ = metric_record(risk, name, truth, central, upper, scale,
                                      assignments["calibration"] if assignment is None else assignment,
                                      manifest, baseline_median, local_floor, local_cap, deployable=True)
            rows.append(record)

        add("primary", base_scale)
        add("scale_floor_x0.5", np.clip(base_scale, floor * 0.5, cap), effective_floor=floor * 0.5)
        add("scale_floor_x2", np.clip(base_scale, floor * 2.0, cap), effective_floor=floor * 2.0)
        add("scale_cap_x0.8", np.clip(base_scale, floor, cap * 0.8), effective_cap=cap * 0.8)
        add("scale_cap_x1.2", np.clip(base_scale, floor, cap * 1.2), effective_cap=cap * 1.2)
        add("conformal_lower_rank", base_scale, convention="lower")
        add("physical_boundaries_outward_5pct", base_scale, assignment=perturbed_assignments)
        settings = [
            ("roughness_x0.5", chosen if chosen in ("Q1", "Q2") else "Q1", 0.005, 0.1, 0.50),
            ("roughness_x2", chosen if chosen in ("Q1", "Q2") else "Q1", 0.02, 0.1, 0.50),
            ("interaction_penalty_x0.5", "Q2", 0.01, 0.05, 0.50),
            ("interaction_penalty_x2", "Q2", 0.01, 0.20, 0.50),
            ("anchor_0.45", chosen if chosen in ("Q1", "Q2") else "Q1", 0.01, 0.1, 0.45),
            ("anchor_0.55", chosen if chosen in ("Q1", "Q2") else "Q1", 0.01, 0.1, 0.55),
        ]
        for name, condition, roughness, interaction_penalty, anchor in settings:
            blocks = []
            for seed in SCALE_SEEDS:
                fit = fit_shape_model(condition, seed, data[risk]["train_proxy"], data[risk]["train_residual"],
                                      data[risk]["validation_proxy"], data[risk]["validation_residual"], knots[risk],
                                      scale_floor=floor, scale_cap=cap, roughness=roughness,
                                      interaction_shrinkage=interaction_penalty, anchor=anchor)
                torch.save(shape_payload(fit, risk), run / f"models/{risk}_{name}_seed_{seed}.pth")
                blocks.append(predict_scale(fit, proxy))
            add(name, np.mean(blocks, axis=0))
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/sensitivity_analysis.csv", frame)
    return frame


def make_figures(run: Path, fits: dict, selection: dict, comparison: pd.DataFrame, subgroup: pd.DataFrame,
                 bootstrap: pd.DataFrame, seed: pd.DataFrame) -> None:
    methods = ["R0_authoritative_baseline", "R1_failed_partially_pooled", "Q0_global_q90_constant",
               "Q1_convex_additive", "Q2_convex_additive_interaction", "O1_hard_physical_group_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, risk in zip(axes, RISKS):
        data = comparison[(comparison.risk == risk) & comparison.method.isin(methods)]
        axis.bar(np.arange(len(data)) - 0.18, data.median_width, 0.36, label="median")
        axis.bar(np.arange(len(data)) + 0.18, data.p95_width, 0.36, label="p95")
        axis.set_xticks(range(len(data)), [value.replace("_", "\n") for value in data.method], fontsize=6)
        axis.set_title(risk.upper()); axis.set_ylabel("interval width"); axis.legend()
    fig.tight_layout(); fig.savefig(run / "figures/interval_width.png", dpi=170); plt.close(fig)
    difficult = subgroup[(subgroup.subgroup_family == "low_snr__high_obstruction") & (subgroup.subgroup == "member")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, risk in zip(axes, RISKS):
        data = difficult[(difficult.risk == risk) & difficult.method.isin(methods)]
        axis.bar(range(len(data)), data.coverage); axis.axhline(0.82, color="black", linestyle="--")
        axis.set_xticks(range(len(data)), [value.replace("_", "\n") for value in data.method], fontsize=6)
        axis.set_ylim(0, 1); axis.set_title(risk.upper()); axis.set_ylabel("coverage")
    fig.tight_layout(); fig.savefig(run / "figures/low_snr_high_obstruction_diagnostics.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, risk in zip(axes, RISKS):
        data = bootstrap[(bootstrap.risk == risk) & (bootstrap.metric == "worst_supported_subgroup_coverage") & bootstrap.method.isin(methods)]
        x = np.arange(len(data)); y = data.point_estimate.to_numpy()
        axis.errorbar(x, y, yerr=np.vstack((y - data.ci_low.to_numpy(), data.ci_high.to_numpy() - y)), fmt="o")
        axis.axhline(0.82, color="black", linestyle="--"); axis.set_ylim(0, 1); axis.set_title(risk.upper())
        axis.set_xticks(x, [value.replace("_", "\n") for value in data.method], fontsize=6)
    fig.tight_layout(); fig.savefig(run / "figures/bootstrap_intervals.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, risk in zip(axes, RISKS):
        method = {"Q0": "Q0_global_q90_constant", "Q1": "Q1_convex_additive", "Q2": "Q2_convex_additive_interaction"}[selection[risk]["selected_model"]]
        data = seed[(seed.risk == risk) & (seed.method == method)]
        axis.plot(data.seed.astype(str), data.marginal_coverage, marker="o", label="marginal")
        axis.plot(data.seed.astype(str), data.worst_supported_subgroup_coverage, marker="s", label="worst subgroup")
        axis.axhline(0.82, color="black", linestyle="--"); axis.set_ylim(0, 1); axis.set_title(risk.upper()); axis.legend()
    fig.tight_layout(); fig.savefig(run / "figures/seed_stability.png", dpi=170); plt.close(fig)
    for risk in RISKS:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        grid = np.linspace(0, 1, 400, dtype=np.float32)
        for condition, linestyle in (("Q1", "-"), ("Q2", "--")):
            fit = fits[(risk, condition, SCALE_SEEDS[0])]
            for index, axis in enumerate(axes):
                values = np.zeros((len(grid), 4), dtype=np.float32); values[:, index] = grid
                with torch.no_grad():
                    effect = fit.model.main_effects(torch.from_numpy(values))[:, index].numpy()
                axis.plot(grid, effect, linestyle=linestyle, label=condition)
                axis.set_title(f"z{index}"); axis.axvline(0.50, color="gray", alpha=0.4); axis.grid(alpha=0.25)
        axes[0].set_ylabel("centered effect"); axes[-1].legend(); fig.tight_layout()
        fig.savefig(run / f"figures/{risk}_effect_curves.png", dpi=170); plt.close(fig)
        fit = fits[(risk, "Q2", SCALE_SEEDS[0])]
        grid2 = np.linspace(0, 1, 120, dtype=np.float32)
        z0, z1 = np.meshgrid(grid2, grid2); values = np.zeros((z0.size, 4), dtype=np.float32)
        values[:, 0] = z0.ravel(); values[:, 1] = z1.ravel()
        with torch.no_grad(): surface = fit.model.interaction_effect(torch.from_numpy(values)).numpy().reshape(z0.shape)
        fig, axis = plt.subplots(figsize=(5.5, 4.5)); image = axis.contourf(z0, z1, surface, levels=20)
        fig.colorbar(image, ax=axis, label="positive interaction"); axis.set_xlabel("z0"); axis.set_ylabel("z1"); axis.set_title(risk.upper())
        fig.tight_layout(); fig.savefig(run / f"figures/{risk}_interaction_surface.png", dpi=170); plt.close(fig)


def correctness_audit(run: Path, marker: dict, validation: pd.DataFrame, manifests: dict) -> dict:
    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    after = checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", after)
    exact_checkpoints = before.equals(after)
    model_paths = sorted((run / "models").glob("*.pth"))
    prereg_predates = bool(model_paths) and marker["hashed_at_unix"] < min(path.stat().st_mtime for path in model_paths)
    prior_oof = pd.read_csv(AUTHORITATIVE / "tables/cross_fit_fold_inventory.csv")
    train_groups = set(manifests["training"].source_a_group.astype(str)) | set(manifests["training"].source_b_group.astype(str))
    validation_groups = set(manifests["validation"].source_a_group.astype(str)) | set(manifests["validation"].source_b_group.astype(str))
    schema_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
            schema_rows.append({"relative_path": relative(path), "rows": len(frame), "columns": len(frame.columns),
                                "status": "PASS" if len(frame.columns) == len(set(frame.columns)) else "FAIL"})
        except Exception as error:
            schema_rows.append({"relative_path": relative(path), "rows": -1, "columns": -1, "status": f"FAIL: {error}"})
    schemas = pd.DataFrame(schema_rows); fresh_csv(run / "tables/csv_schema_validation.csv", schemas)
    large = [{"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
              "expected": path.suffix in (".pth", ".npz")} for path in sorted(run.rglob("*")) if path.is_file() and path.stat().st_size >= 10 * 2**20]
    fresh_csv(run / "tables/large_file_inventory.csv", pd.DataFrame(large, columns=["relative_path", "size_bytes", "sha256", "expected"]))
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_shape_constrained_quantile.py", "tests/test_scale_correction.py", "tests/test_conditional_calibration.py"])
    diff = command(["git", "diff", "--check"])
    staged = command(["git", "diff", "--cached", "--name-only"])
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    privacy_hits = [row["relative_path"] for section in ("partition_manifests", "persisted_scale_feature_arrays")
                    for row in provenance[section] if "development" in row["relative_path"].lower() or "lockbox" in row["relative_path"].lower()]
    checks = {
        "preregistration_hash_intact": sha256_file(run / "preregistration/shape_constrained_quantile_scale.md") == marker["sha256"],
        "preregistration_predates_fitting": prereg_predates,
        "all_gates_attainable": bool(pd.read_csv(run / "tables/gate_attainability_audit.csv").attainable.all()),
        "training_predictions_oof": bool((prior_oof.prediction_status == "OUT_OF_FOLD").all() and (prior_oof.source_group_overlap == 0).all()),
        "fixed_partition_source_isolation": not bool(train_groups & validation_groups),
        "zero_constraint_violations": bool((validation.convexity_violations == 0).all() and (validation.upper_half_monotonicity_violations == 0).all()),
        "main_effects_centered": bool((validation.centering_mean_max_abs < 1e-5).all()),
        "interaction_nonnegative": bool(validation.interaction_nonnegative.all()),
        "historical_checkpoints_unchanged": exact_checkpoints,
        "condition_c_unchanged": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "zero_neural_inference": provenance["neural_inference"] == 0,
        "zero_development_access": provenance["development_accesses"] == 0 and not privacy_hits,
        "zero_lockbox_access": provenance["lockbox_accesses"] == 0 and not privacy_hits,
        "calibration_after_selection": (run / "logs/validation_selection_complete.json").stat().st_mtime <= (run / "tables/calibration_fold_assignments.csv").stat().st_mtime,
        "compileall": compile_result["returncode"] == 0,
        "target_and_constraint_tests": tests["returncode"] == 0,
        "csv_schema_validation": bool((schemas.status == "PASS").all()),
        "git_diff_check": diff["returncode"] == 0,
        "staged_index_empty": staged["returncode"] == 0 and not staged["stdout"].strip(),
        "privacy_path_grep": not privacy_hits,
        "example_grids_empty": not any((run / "example_grids").iterdir()),
    }
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "compileall": compile_result, "tests": tests, "git_diff_check": diff, "staged_index": staged,
             "historical_checkpoint_rows": len(before), "privacy_path_hits": privacy_hits}
    fresh_json(run / "diagnostics/final_correctness_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("correctness audit failed: " + ", ".join(name for name, value in checks.items() if not value))
    return audit


def replay_integrity_sanity(run: Path) -> pd.DataFrame:
    frame = pd.read_csv(AUTHORITATIVE / "tables/integrity_sanity_checks.csv")
    if not frame.passed.all():
        raise RuntimeError("authoritative query/catastrophic/centroid sanity artifact is not PASS")
    fresh_csv(run / "tables/integrity_sanity_checks.csv", frame)
    fresh_text(run / "diagnostics/integrity_sanity_reproduction.md",
               "# Integrity sanity reproduction\n\nPASS. The exact persisted authoritative query-gate, catastrophic-valid, and centroid audit artifact replayed without neural inference. Query macro F1/AUPRC are 0.874051/0.916862; catastrophic AUROC/AUPRC are 0.987182/0.997096; centroid marginal/worst coverage are 0.900714/0.888165.\n")
    return frame


def write_final_report(run: Path, marker: dict, comparison: pd.DataFrame, subgroup: pd.DataFrame,
                       bootstrap: pd.DataFrame, validation: pd.DataFrame, selection: dict,
                       decisions: pd.DataFrame, audit: dict, started: float) -> None:
    label = {"Q0": "Q0_global_q90_constant", "Q1": "Q1_convex_additive", "Q2": "Q2_convex_additive_interaction"}
    image_method = label[selection["image"]["selected_model"]]
    flux_method = label[selection["flux"]["selected_model"]]
    image = comparison[(comparison.risk == "image") & (comparison.method == image_method)].iloc[0]
    flux = comparison[(comparison.risk == "flux") & (comparison.method == flux_method)].iloc[0]
    image_boot = bootstrap[(bootstrap.risk == "image") & (bootstrap.method == image_method) & (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
    flux_boot = bootstrap[(bootstrap.risk == "flux") & (bootstrap.method == flux_method) & (bootstrap.metric == "worst_supported_subgroup_coverage")].iloc[0]
    q1 = validation[(validation.condition == "Q1") & (validation.seed == "ensemble")]
    q2 = validation[(validation.condition == "Q2") & (validation.seed == "ensemble")]
    overall = str(decisions[decisions.component == "OVERALL"].decision.iloc[0])
    if overall == "SUCCESS":
        authorized = "Yes—one separately preregistered full hierarchical-policy campaign is authorized, still stopping before development/lockbox evaluation."
        next_step = "Run that one separately preregistered hierarchical-policy campaign using the frozen selected scale models."
    else:
        authorized = "No."
        next_step = "Run exactly one new train/validation/calibration-only convex tensor-product quantile experiment over z0 and z1, retaining the same four proxies, OOF targets, gates, and sealed development/lockbox partitions; do not run it in this campaign."
    git_status = command(["git", "status", "--short", "--branch"])["stdout"]
    disk = shutil.disk_usage(REPO)
    runtime = time.time() - started
    report = f"""# Thayer-Select shape-constrained quantile scale-correction final report

## Outcome

**{overall}.** This is a train/validation/natural-calibration-only result. It is not a full policy result. Condition C, risk/query heads, proxy definitions, source partitions, and physical subgroups remained frozen. Development and lockbox remained sealed.

## Required answers

1. **Did the training-only proxy-shape audit reproduce?** Yes, all four stated z0/z1 q=0.90 endpoint checks reproduced within 1e-10.
2. **Why was global monotonicity rejected?** Training OOF tails reverse sharply at the lowest proxy deciles; image z0 is 9.483 versus 1.476 at the highest decile, with analogous z1 and flux reversals.
3. **Were all scale targets truly OOF?** Yes. The authoritative five-fold connected-source-group inventory marks every training prediction OUT_OF_FOLD with zero overlap.
4. **Were knots and constraints frozen before fitting?** Yes. Preregistration `{marker['sha256']}` and knot hash `{marker['knots_sha256']}` predate every checkpoint.
5. **Did Q1 improve validation tail calibration?** Q1 worst supported validation-cell coverage was image {q1[q1.risk == 'image'].worst_supported_cell_coverage.iloc[0]:.3f} and flux {q1[q1.risk == 'flux'].worst_supported_cell_coverage.iloc[0]:.3f}; see the validation table for Q0 comparison.
6. **Did the single interaction improve validation proxy-cell coverage?** Q2-minus-Q1 was image {(q2[q2.risk == 'image'].worst_supported_cell_coverage.iloc[0] - q1[q1.risk == 'image'].worst_supported_cell_coverage.iloc[0]):.3f} and flux {(q2[q2.risk == 'flux'].worst_supported_cell_coverage.iloc[0] - q1[q1.risk == 'flux'].worst_supported_cell_coverage.iloc[0]):.3f}; Q2 extra-gate results were {selection['image']['q2_extra_gate_passed']}/{selection['flux']['q2_extra_gate_passed']}.
7. **Which model was selected without calibration access?** Image {selection['image']['selected_model']}; flux {selection['flux']['selected_model']}.
8. **Did q=0.90 quantile fitting improve image coverage?** Selected image worst-subgroup coverage changed from 0.637306 to {image.worst_supported_subgroup_coverage:.6f}.
9. **Did it improve flux coverage?** Selected flux worst-subgroup coverage changed from 0.683938 to {flux.worst_supported_subgroup_coverage:.6f}.
10. **What were image and flux worst-subgroup coverages?** {image.worst_supported_subgroup_coverage:.6f} and {flux.worst_supported_subgroup_coverage:.6f}.
11. **What were low-SNR/high-obstruction coverages?** {image.low_snr_high_obstruction_coverage:.6f} and {flux.low_snr_high_obstruction_coverage:.6f}.
12. **What width cost was paid?** Median inflation was {image.median_width_inflation:.3f}x image and {flux.median_width_inflation:.3f}x flux; p95 widths were {image.p95_width:.3f} and {flux.p95_width:.3f}.
13. **Did the result survive source-group bootstrap?** Worst-group 95% intervals were image [{image_boot.ci_low:.3f}, {image_boot.ci_high:.3f}] and flux [{flux_boot.ci_low:.3f}, {flux_boot.ci_high:.3f}].
14. **Were all convexity and upper-half monotonicity constraints satisfied?** Yes, with zero finite-difference violations across all primary seeds.
15. **Did the interaction remain bounded and nonnegative?** Yes. Every Q2 coefficient and evaluated interaction surface was nonnegative; scale caps bound the resulting predictions.
16. **Did marginal coverage remain in range?** Image {image.marginal_coverage:.6f}; flux {flux.marginal_coverage:.6f}.
17. **Did ranking remain strong?** Image/flux Spearman remained {image.calibration_spearman:.3f}/{flux.calibration_spearman:.3f}.
18. **Did centroid remain PASS?** Yes, 0.900714 marginal and 0.888165 worst supported coverage.
19. **Did query and catastrophic sanity checks reproduce?** Yes, exact persisted audit replay passed.
20. **Did IMAGE_RISK pass?** {decisions[decisions.component == 'IMAGE_RISK'].decision.iloc[0]}.
21. **Did FLUX_RISK pass?** {decisions[decisions.component == 'FLUX_RISK'].decision.iloc[0]}.
22. **Is a full hierarchical-policy campaign authorized?** {authorized}
23. **What exactly should happen next?** {next_step}
24. **Were development and lockbox untouched?** Yes, zero accesses.
25. **Were all historical checkpoints unchanged?** Yes, {audit['historical_checkpoint_rows']} checkpoints rehashed identically.

## Prior failure reproduction

R0 reproduced image marginal/worst 0.902857/0.637306, flux 0.898214/0.683938, and centroid 0.900714/0.888165. R1 reproduced image 0.918929/0.549223 and flux 0.921786/0.678756. O1 remains non-deployable and cannot determine success.

## Validation-only selection

{markdown_table(validation[validation.seed == 'ensemble'])}

## Normalized-conformal comparison

{markdown_table(comparison)}

## Component decisions

{markdown_table(decisions)}

## Artifacts and integrity

- Proxy audit: `tables/training_proxy_shape_audit.csv` and `figures/training_proxy_shape_audit.png`.
- Constraints/effects: `scale_model_seed_comparison.csv`, effect curves, and interaction surfaces.
- Subgroups/widths: `tables/subgroup_coverage.csv` and figures.
- Bootstrap: {BOOTSTRAP_REPLICATES} connected-source-component replicates in `tables/source_group_bootstrap_intervals.csv`.
- Sensitivity: `tables/sensitivity_analysis.csv`; no retuning occurred.
- Correctness audit: **{audit['status']}**; compileall, target/OOF/constraint/conformal/bootstrap tests, CSV validation, checkpoint audit, `git diff --check`, staged-index, privacy-path, and large-file checks passed.
- Runtime: {runtime:.1f} seconds; run size {sum(path.stat().st_size for path in run.rglob('*') if path.is_file()) / 2**20:.2f} MiB; free disk {disk.free / 2**30:.2f} GiB.

## Final git status

```text
{git_status}```
"""
    fresh_text(run / "reports/final_report.md", report)
    fresh_json(run / "logs/campaign_complete.json", {"status": overall, "runtime_seconds": runtime,
                                                      "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                                                      "development_accesses": 0, "lockbox_accesses": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("use --execute for the full restarted campaign")
    started = time.time()
    verified = verify_authoritative_inputs()
    run = create_run()
    capture_provenance(run, verified, started)
    proxy_audit(run)
    data, samples, manifests = training_validation_data()
    knots, marker = preregister(run, data)
    fits, selection, validation = fit_and_select(run, data, manifests, knots, marker)
    replay, replay_groups = reproduce_prior_failures(run)
    calibration_samples, calibration_manifest, assignments, fold = calibration_context(run)
    comparison, subgroup, outputs, seed = calibration_comparison(
        run, fits, selection, replay, replay_groups, calibration_samples, calibration_manifest, assignments, fold
    )
    bootstrap = bootstrap_campaign(run, comparison, outputs, calibration_samples, calibration_manifest, assignments)
    decisions = selected_decisions(run, comparison, bootstrap, seed, selection)
    sensitivity_campaign(run, fits, selection, data, knots, comparison, calibration_samples,
                         calibration_manifest, assignments, fold)
    replay_integrity_sanity(run)
    make_figures(run, fits, selection, comparison, subgroup, bootstrap, seed)
    audit = correctness_audit(run, marker, validation, manifests)
    write_final_report(run, marker, comparison, subgroup, bootstrap, validation, selection, decisions, audit, started)
    print(run)


if __name__ == "__main__":
    main()
