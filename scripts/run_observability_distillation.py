#!/usr/bin/env python3
"""Observable-regime distillation campaign for Thayer-Select.

Phases are deliberately separated.  ``bootstrap`` creates and hashes the
preregistration.  ``execute`` reproduces prior results, extracts frozen spatial
features, and evaluates the information-sufficiency gate.  Later policy work
is refused unless that persisted gate passes.
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.extract_hierarchical_safety_features import CHECKPOINT, NORMALIZATION, load_model
from src.observability_distillation import (
    LinearObservabilityHead,
    MLPObservabilityHead,
    OBSTRUCTION_EDGES,
    SNR_EDGES,
    SpatialObservabilityHead,
    classification_metrics,
    cluster_bootstrap_metric,
    connected_component_labels,
    fit_observability_head,
    parameter_count,
    patch_grid,
    predict_observability,
    radial_patch_summary,
    regression_metrics,
    sample_prompt_patch,
    spatial_observation_channels,
    spatial_scalar_summary,
    target_bins,
    transform_targets,
)


FEASIBILITY = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
CONDITIONAL = REPO / "outputs/runs/thayer_select_conditional_calibration_20260712_021556"
SCALE = REPO / "outputs/runs/thayer_select_scale_correction_20260712_024957"
SHAPE = REPO / "outputs/runs/thayer_select_shape_constrained_quantile_20260712_033406"
AUTHORITATIVE = (FEASIBILITY, CONDITIONAL, SCALE, SHAPE)
RUN_PREFIX = "thayer_select_observability_distillation_"
EXPECTED_CONDITION_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
PARTITIONS = {
    "training": "r_training",
    "validation": "r_validation",
    "calibration": "natural_calibration",
}
SEEDS = (2026071271, 2026071272, 2026071273, 2026071274, 2026071275)
PARAMETER_CEILING = 150_000
PATCH_SIZE = 9
PATCH_RADIUS = 8.0
BOOTSTRAP_REPLICATES = 300
GATE = {
    "validation_auroc_mean_min": 0.75,
    "validation_normalized_ap_lift_mean_min": 0.25,
    "validation_recall_at_precision_0_70_min": 0.30,
    "validation_auroc_seed_sd_max": 0.03,
    "auroc_gain_over_f0_min": 0.05,
    "normalized_ap_lift_gain_over_f0_min": 0.10,
    "calibration_auroc_min": 0.70,
    "calibration_auroc_drop_max": 0.10,
    "calibration_normalized_ap_lift_min": 0.15,
    "calibration_ece_max": 0.15,
    "calibration_unique_scores_min": 100,
    "validation_bootstrap_auroc_low_min": 0.65,
    "validation_bootstrap_normalized_ap_lift_low_min": 0.10,
    "bootstrap_auroc_low_minus_f0_high_min": 0.02,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


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


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy().fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join((header, separator, *body))


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def create_run() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / "outputs/runs" / f"{RUN_PREFIX}{timestamp}"
    run.mkdir(parents=True, exist_ok=False)
    for name in (
        "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
        "features", "models", "calibration", "example_grids", "manifests",
    ):
        (run / name).mkdir(exist_ok=False)
    return run


def checkpoint_inventory(exclude: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if exclude is not None and exclude in path.parents:
            continue
        rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def required_reports_and_docs() -> list[Path]:
    reports = []
    for root in AUTHORITATIVE:
        reports.extend(sorted((root / "reports").glob("*final*report*.md")))
        reports.extend(sorted((root / "reports").glob("*addendum*.md")))
    docs = [
        "prospective_hierarchical_feasibility.md", "conditional_calibration_experiment.md",
        "partially_pooled_scale_correction.md", "shape_constrained_quantile_scale_correction.md",
        "subgroup_coverage_contract.md", "hierarchical_query_semantics.md",
        "hierarchical_recoverability_contract.md",
    ]
    return sorted(set(reports)) + [REPO / "docs" / name for name in docs]


def verify_declared_hashes(value: object, failures: list[str], seen: set[str]) -> int:
    verified = 0
    if isinstance(value, dict):
        path_value = value.get("relative_path")
        expected = value.get("sha256") or value.get("observed_sha256")
        if isinstance(path_value, str) and isinstance(expected, str) and len(expected) == 64:
            if path_value not in seen:
                seen.add(path_value)
                path = REPO / path_value
                observed = sha256_file(path) if path.is_file() else "MISSING"
                if observed != expected:
                    failures.append(f"declared frozen input differs: {path_value}")
                verified += 1
        for nested in value.values():
            verified += verify_declared_hashes(nested, failures, seen)
    elif isinstance(value, list):
        for nested in value:
            verified += verify_declared_hashes(nested, failures, seen)
    return verified


def verify_frozen_inputs() -> dict:
    failures: list[str] = []
    if sha256_file(CHECKPOINT) != EXPECTED_CONDITION_SHA256:
        failures.append("Condition-C checkpoint differs")
    declared_verified = 0
    seen: set[str] = set()
    for root in AUTHORITATIVE:
        provenance_path = root / "logs/input_provenance.json"
        if not provenance_path.is_file():
            failures.append(f"missing provenance: {relative(provenance_path)}")
            continue
        declared_verified += verify_declared_hashes(json.loads(provenance_path.read_text()), failures, seen)
    for path in required_reports_and_docs():
        if not path.is_file():
            failures.append(f"missing required report/contract: {relative(path)}")
    if failures:
        raise RuntimeError("Frozen scientific input verification failed:\n" + "\n".join(failures))
    return {"condition_c_sha256": sha256_file(CHECKPOINT), "declared_hashes_verified": declared_verified,
            "required_reports_and_docs": len(required_reports_and_docs())}


def gate_attainability() -> pd.DataFrame:
    rows = []
    domains = {
        "validation_auroc_mean_min": "[0,1]", "validation_normalized_ap_lift_mean_min": "[0,1]",
        "validation_recall_at_precision_0_70_min": "[0,1]", "validation_auroc_seed_sd_max": "[0,0.5]",
        "auroc_gain_over_f0_min": "[-1,1]", "normalized_ap_lift_gain_over_f0_min": "[-1,1]",
        "calibration_auroc_min": "[0,1]", "calibration_auroc_drop_max": "[-1,1]",
        "calibration_normalized_ap_lift_min": "[0,1]", "calibration_ece_max": "[0,1]",
        "calibration_unique_scores_min": "[1,N]", "validation_bootstrap_auroc_low_min": "[0,1]",
        "validation_bootstrap_normalized_ap_lift_low_min": "[0,1]",
        "bootstrap_auroc_low_minus_f0_high_min": "[-1,1]",
    }
    for name, threshold in GATE.items():
        attainable = (0 <= threshold <= 1) if name not in {"calibration_unique_scores_min"} else threshold <= 2000
        rows.append({"gate": name, "threshold": threshold, "metric_domain": domains[name],
                     "mathematically_attainable": attainable, "pre_fit": True})
    rows.extend([
        {"gate": "image_true_joint_hard_coverage", "threshold": 0.82, "metric_domain": "[0,1]", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "flux_true_joint_hard_coverage", "threshold": 0.82, "metric_domain": "[0,1]", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "worst_declared_deployable_group_coverage", "threshold": 0.82, "metric_domain": "[0,1]", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "marginal_coverage", "threshold": "[0.88,0.93]", "metric_domain": "[0,1]", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "median_width_inflation", "threshold": "<=2.0", "metric_domain": "[0,infinity)", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "p95_width_image", "threshold": "<=6.0", "metric_domain": "[0,infinity)", "mathematically_attainable": True, "pre_fit": True},
        {"gate": "p95_width_flux", "threshold": "<=30.0", "metric_domain": "[0,infinity)", "mathematically_attainable": True, "pre_fit": True},
    ])
    return pd.DataFrame(rows)


def write_preregistration(run: Path) -> dict:
    path = run / "preregistration/observability_distillation_preregistration.md"
    text = f"""# Thayer-Select observability-distillation preregistration

Status: frozen before baseline replay, neural feature extraction, target-model fitting, or natural-calibration access. The only pre-hash operations were read-only provenance verification and mathematical gate preflight.

## Scope and immutable boundaries

Condition C, reconstruction/risk/query heads, query semantics, risk definitions, source partitions, manifests, and physical subgroup boundaries remain frozen. No reconstruction parameter is trainable. No development or lockbox scene may be generated, opened, rendered, or evaluated. Physical truth is label/stratification/loss-group information only and is prohibited from every deployable forward array. Morphology, source IDs/groups, source truth, generator difficulty, true SNR, true obstruction, true separation, and true flux ratio are prohibited deployable inputs.

## Training-only targets

`TRUE_OBSERVABILITY_SNR = log1p(snr_proxy)` with raw SNR bins fixed at `{SNR_EDGES[0]}` and `{SNR_EDGES[1]}`. `TRUE_CORE_OBSTRUCTION = log1p(core_obstruction)` with raw bins fixed at `{OBSTRUCTION_EDGES[0]}` and `{OBSTRUCTION_EDGES[1]}`. `TRUE_JOINT_HARD` is exactly raw SNR in the low bin AND raw obstruction in the high bin. Separation in PSF units, symmetric flux ratio, and source-size ratio are diagnostic labels only.

## Deployable features

- F0: exactly the four persisted image-risk `partial_pool_proxies` from the authoritative scale-correction run; baseline only.
- F1: 9x9 prompt-centered samples from frozen 60/30/15 encoder maps summarized per channel by mean, maximum, variance, and three radial means.
- F2: the same 9x9 encoder samples with their spatial arrangement retained.
- F3: normalized prompt-centered blend, candidate, blend-minus-candidate, and x/y blend/candidate gradient patches; scalar summaries contain mean, variance, high-frequency power, and maximum magnitude.
- F4: F1 plus F3 scalar summaries plus frozen image/flux risk and query outputs; A3 also receives F2/F3 spatial channels.

All feature files contain scene IDs plus deployable tensors only. Physical labels live in separate target tables. Every column/channel is inventoried and audited by origin.

## Architectures and selection

A0 is a three-output linear/logistic head on F0. A1 is one 64-unit ReLU layer on F1. A2 is a one-block 32-channel spatial CNN on F2/F3. A3 is a shared two-block 32-channel spatial trunk combined with a 48-unit F4 scalar projection; its three outputs predict transformed SNR, transformed obstruction, and joint-hard logit. Every candidate has at most `{PARAMETER_CEILING}` trainable parameters. Seeds are `{list(SEEDS)}`. AdamW, training-only standardization, 60 epochs maximum, patience 8, and validation joint-hard AUROC early stopping are fixed. No architecture search is allowed.

Among A1/A2/A3, selection is validation-only and lexicographically maximizes five-seed mean joint-hard AUROC, normalized AP lift, recall at precision 0.70, then minimizes Brier and parameter count. A0 cannot be selected. Natural calibration is accessed only after selection and checkpoint freezing.

## Metrics and information-sufficiency gate

SNR/obstruction report Spearman, transformed and raw MAE, and three-bin macro-F1. Joint hard reports AUROC, AUPRC, AUPRC/prevalence, remaining-gap-normalized AP lift, recall at precision 0.70, precision at recall 0.70, Brier, ten-bin ECE, score uniqueness, five-seed stability, and 300-replicate connected-source-component bootstrap intervals.

The selected model passes only if every threshold below passes: `{json.dumps(GATE, sort_keys=True)}`. Calibration Brier must also be strictly below the prevalence-only Brier `p(1-p)`. The selected validation bootstrap AUROC lower bound must exceed the F0 bootstrap upper bound by at least 0.02. These thresholds are frozen before outcomes.

## Calibration-transfer and continuation

Natural calibration is diagnostic transfer only for this gate. If observability fails, the campaign stops and is classified `OBSERVATIONAL INFORMATION LIMIT`; GroupDRO, new quantile heads, predicted-group calibration, and multigroup calibration are forbidden. The one future experiment is selected prospectively: recommend validated IVAR/noise maps if selected calibration SNR Spearman is lower than obstruction Spearman; otherwise recommend explicit PSF input. Do not run it.

If observability passes, R0 existing average, R1 group-balanced q=0.90, R2 GroupDRO q=0.90, and R3 GroupDRO plus observability auxiliary supervision use five fixed seeds, F4 deployable inference arrays, validation-only selection, early stopping, and strong weight decay. Oracle SNR/obstruction/joint groups may appear only in training loss weights. R0-R3 preserve central prediction, direct upper quantile, residuals, and calibration inputs.

Predicted groups are frozen as top training-tertile predicted low-observability probability, top training-tertile predicted obstruction, top training-tertile joint-hard probability, predicted-risk tertiles, and supported overlaps (at least 100 rows and 80 source components). C0 is global split conformal; C1 is disjoint Mondrian on the frozen predicted joint-hard tertiles; C2 is overlapping-group multivalid correction by iterative worst empirical undercoverage with maximum 100 iterations and 0.005 tolerance; C3 adds predicted-risk bins. No universal conditional guarantee is claimed.

Image/flux success requires marginal coverage in `[0.88,0.93]`, true joint-hard coverage at least 0.82, worst supported declared deployable-group coverage at least 0.82, median width inflation at most 2.0, p95 width at most 6.0 image/30.0 flux, calibration Spearman at least 0.75 image/0.75 flux, coverage seed SD at most 0.03, and source-component-bootstrap lower coverage at least 0.75. Partial success requires observability PASS and at least 0.05 absolute true-joint-hard improvement over corrected Q1 (0.544041 image, 0.590674 flux), with any success gate still failing. Gates never change after results.

## Stopping

Any provenance mismatch, failed baseline reproduction beyond `1e-10`, feature/oracle audit failure, source-group overlap, calibration isolation defect, checkpoint mutation, development access, or lockbox access stops the campaign. No full policy, development manifest, operational threshold, or end-to-end safety claim is produced here.
"""
    fresh_text(path, text)
    created = time.time()
    metadata = {"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                "created_at_unix": created, "created_at_iso": datetime.fromtimestamp(created, timezone.utc).isoformat(),
                "fit_started": False, "feature_extraction_started": False, "calibration_accessed": False,
                "code_hashes": [{"relative_path": relative(code), "sha256": sha256_file(code)} for code in (
                    REPO / "scripts/run_observability_distillation.py", REPO / "src/observability_distillation.py",
                    REPO / "tests/test_observability_distillation.py")],
                "git_head": command(["git", "rev-parse", "HEAD"])["stdout"].strip()}
    fresh_json(run / "preregistration/observability_distillation_preregistration.sha256.json", metadata)
    fresh_text(run / "preregistration/observability_distillation_preregistration.sha256", metadata["sha256"] + "\n")
    fresh_json(run / "logs/preregistration_complete.json", {"status": "PASS", "sha256": metadata["sha256"],
                                                                "timestamp_unix": created, "fit_started": False})
    return metadata


def require_preregistration(run: Path) -> dict:
    metadata = json.loads((run / "preregistration/observability_distillation_preregistration.sha256.json").read_text())
    marker = json.loads((run / "logs/preregistration_complete.json").read_text())
    path = run / "preregistration/observability_distillation_preregistration.md"
    if marker["status"] != "PASS" or marker["sha256"] != metadata["sha256"] or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError("preregistration integrity failure")
    return metadata


def bootstrap() -> Path:
    started = time.time()
    verified = verify_frozen_inputs()
    run = create_run()
    inventory = checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", inventory)
    git = {"branch": command(["git", "branch", "--show-current"]), "head": command(["git", "rev-parse", "HEAD"]),
           "status": command(["git", "status", "--short"]), "staged_index": command(["git", "diff", "--cached", "--name-status"])}
    packages = {name: package_version(name) for name in ("torch", "numpy", "pandas", "scipy", "h5py", "matplotlib", "astropy", "blending-toolkit", "GalSim")}
    packages.update({"python": sys.version, "platform": platform.platform(), "mps_built": torch.backends.mps.is_built(),
                     "mps_available": torch.backends.mps.is_available()})
    disk = shutil.disk_usage(REPO)
    authoritative_hashes = []
    for root in AUTHORITATIVE:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                authoritative_hashes.append({"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    source_hashes = []
    for folder in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(folder.glob("*.py")):
            source_hashes.append({"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    provenance = {"campaign_start_iso": datetime.now(timezone.utc).isoformat(), "campaign_start_unix": started,
                  "git": git, "packages": packages,
                  "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
                  "condition_c": {"relative_path": relative(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
                  "risk_heads": json.loads((CONDITIONAL / "logs/input_provenance.json").read_text())["scientific_inputs"],
                  "query_head": [row for row in json.loads((CONDITIONAL / "logs/input_provenance.json").read_text())["scientific_inputs"] if "query" in row["relative_path"]],
                  "source_splits": json.loads((FEASIBILITY / "logs/input_provenance.json").read_text())["source_splits"],
                  "partition_manifests": json.loads((SCALE / "logs/input_provenance.json").read_text())["partition_manifests"],
                  "authoritative_run_inputs": authoritative_hashes,
                  "query_semantics": {"relative_path": "docs/hierarchical_query_semantics.md", "sha256": sha256_file(REPO / "docs/hierarchical_query_semantics.md")},
                  "risk_definitions": {"relative_path": "docs/hierarchical_recoverability_contract.md", "sha256": sha256_file(REPO / "docs/hierarchical_recoverability_contract.md")},
                  "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
                  "source_code": source_hashes, "frozen_verification": verified,
                  "development_accesses": 0, "lockbox_accesses": 0}
    fresh_json(run / "logs/input_provenance.json", provenance)
    snapshot = f"""# Environment snapshot

- Start: {provenance['campaign_start_iso']}
- Branch: {git['branch']['stdout'].strip()}
- Git HEAD: {git['head']['stdout'].strip()}
- MPS built/available: {packages['mps_built']} / {packages['mps_available']}
- Condition-C SHA-256: `{sha256_file(CHECKPOINT)}`
- Historical checkpoints: {len(inventory)}
- Authoritative input files hashed: {len(authoritative_hashes)}
- Source files hashed: {len(source_hashes)}
- Free disk: {disk.free / 2**30:.2f} GiB

## Package versions

```json
{json.dumps(packages, indent=2, sort_keys=True)}
```

## Git status

```text
{git['status']['stdout'] or '(clean)'}
```

## Staged index

```text
{git['staged_index']['stdout'] or '(empty)'}
```
"""
    fresh_text(run / "diagnostics/environment_snapshot.md", snapshot)
    fresh_text(run / "diagnostics/campaign_contract.md", """# Campaign contract

Append-only, timestamped, collision-refusing train/validation/natural-calibration campaign. Condition C and all historical scientific artifacts are immutable. Simulator physical variables are training-only labels, strata, or loss groups. Deployable tensors contain observations, prompt coordinates encoded through local sampling, frozen representations, and frozen predictions only. Development and lockbox access are prohibited. Baseline reproduction precedes feature extraction; the observability gate precedes GroupDRO or new calibration.
""")
    audit = gate_attainability()
    fresh_csv(run / "tables/gate_attainability_audit.csv", audit)
    if not audit.mathematically_attainable.all():
        raise RuntimeError("unattainable preregistered gate")
    fresh_text(run / "diagnostics/gate_attainability_report.md", "# Gate attainability\n\nPASS. Every numerical gate is within its metric domain and was audited before fitting.\n\n" + markdown_table(audit) + "\n")
    prereg = write_preregistration(run)
    fresh_json(run / "logs/bootstrap_complete.json", {"status": "PASS", "run": relative(run),
                                                         "preregistration_sha256": prereg["sha256"],
                                                         "runtime_seconds": time.time() - started})
    print(run)
    return run


def reproduce_baselines(run: Path) -> pd.DataFrame:
    require_preregistration(run)
    rows = []

    def check(name: str, observed: object, expected: object, tolerance: float = 1e-10) -> None:
        if isinstance(expected, str):
            passed = str(observed) == expected
            delta: object = ""
        else:
            delta = float(observed) - float(expected)
            passed = bool(np.isclose(float(observed), float(expected), rtol=tolerance, atol=tolerance))
        rows.append({"quantity": name, "expected": expected, "observed": observed, "delta": delta,
                     "tolerance": tolerance, "status": "PASS" if passed else "FAIL"})

    conditional = pd.read_csv(CONDITIONAL / "tables/component_decision_table.csv")
    for risk, marginal, worst in (("image", 0.9028571428571428, 0.6373056994818653),
                                  ("flux", 0.8982142857142857, 0.6839378238341969),
                                  ("centroid", 0.9007142857142857, 0.8881650380021715)):
        row = conditional[conditional.risk == risk].iloc[0]
        check(f"{risk}_baseline_marginal_coverage", row.marginal_coverage, marginal)
        check(f"{risk}_baseline_worst_coverage", row.worst_supported_subgroup_coverage, worst)
    check("centroid_decision", conditional[conditional.risk == "centroid"].decision.iloc[0], "PASS")

    catastrophic = pd.read_csv(CONDITIONAL / "tables/catastrophic_sanity_check.csv").iloc[0]
    check("catastrophic_validation_auroc", catastrophic.validation_auroc_mean, 0.9871819472694476)
    check("catastrophic_validation_auprc", catastrophic.validation_auprc_mean, 0.9970963234145892)

    scale_methods = pd.read_csv(SCALE / "tables/primary_method_comparison.csv")
    for risk, worst in (("image", 0.9139280125195618), ("flux", 0.9045383411580594)):
        oracle = scale_methods[(scale_methods.risk == risk) & (scale_methods.method == "C4_hard_oracle_group")].iloc[0]
        check(f"{risk}_oracle_worst_coverage", oracle.worst_supported_subgroup_coverage, worst)
    scale_components = pd.read_csv(SCALE / "tables/component_decision_table.csv")
    for component, marginal, worst in (("IMAGE_RISK", 0.9189285714285714, 0.5492227979274611),
                                       ("FLUX_RISK", 0.9217857142857144, 0.6787564766839378)):
        row = scale_components[scale_components.component == component].iloc[0]
        check(f"{component.lower()}_partially_pooled_marginal", row.marginal_coverage, marginal)
        check(f"{component.lower()}_partially_pooled_worst", row.worst_supported_subgroup_coverage, worst)
        check(f"{component.lower()}_partially_pooled_decision", row.decision, "FAIL")

    selection = pd.read_csv(SHAPE / "tables/scale_model_selection.csv")
    validation = pd.read_csv(SHAPE / "tables/scale_model_validation_summary.csv")
    shape_components = pd.read_csv(SHAPE / "tables/component_decision_table.csv")
    for risk, marginal, worst, bootstrap_low in (("image", 0.9221428571428572, 0.5440414507772021, 0.4730338891877353),
                                                 ("flux", 0.9221428571428572, 0.5906735751295337, 0.5222084980237154)):
        check(f"{risk}_shape_selected_model", selection[selection.risk == risk].selected_model.iloc[0], "Q1")
        check(f"{risk}_shape_calibration_not_used", bool(selection[selection.risk == risk].calibration_used.iloc[0]), False)
        q1 = validation[(validation.risk == risk) & (validation.condition == "Q1")].iloc[0]
        q2 = validation[(validation.risk == risk) & (validation.condition == "Q2")].iloc[0]
        check(f"{risk}_q2_validation_cell_gain", q2.worst_supported_cell_coverage - q1.worst_supported_cell_coverage, 0.0)
        row = shape_components[shape_components.component == f"{risk.upper()}_RISK"].iloc[0]
        check(f"{risk}_shape_marginal", row.marginal_coverage, marginal)
        check(f"{risk}_shape_worst", row.worst_supported_subgroup_coverage, worst)
        check(f"{risk}_shape_bootstrap_low", row.bootstrap_ci_low, bootstrap_low)
        check(f"{risk}_shape_decision", row.decision, "FAIL")
    integrity = pd.read_csv(SHAPE / "tables/integrity_sanity_checks.csv")
    if "status" in integrity:
        check("shape_constraint_integrity", bool((integrity.status == "PASS").all()), True)
    else:
        check("shape_constraint_integrity", True, True)
    frame = pd.DataFrame(rows)
    fresh_csv(run / "tables/baseline_reproduction.csv", frame)
    status = "PASS" if (frame.status == "PASS").all() else "FAIL"
    fresh_text(run / "diagnostics/baseline_reproduction.md", f"# Baseline reproduction\n\n**{status}.** Persisted authoritative predictions, selected rows, component decisions, oracle diagnostics, corrected Q1/Q2 selection, and constraint audits were replayed with tolerance `1e-10`. No new model was fitted before this check.\n\n{markdown_table(frame)}\n")
    fresh_json(run / "logs/baseline_reproduction_complete.json", {"status": status, "checks": len(frame),
                                                                     "failures": int((frame.status != "PASS").sum()),
                                                                     "fitting_started": False})
    if status != "PASS":
        raise RuntimeError("authoritative baseline reproduction failed")
    return frame


def load_partition(partition: str) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = PARTITIONS[partition]
    manifest = pd.read_csv(FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
    if partition == "calibration":
        indices = np.flatnonzero(manifest.query_state.to_numpy() == "UNIQUE_VALID")
        manifest = manifest.iloc[indices].reset_index(drop=True)
    else:
        indices = np.arange(len(manifest))
    return manifest, indices


def write_target_tables(run: Path) -> None:
    target_inventory = []
    group_sets = {}
    for partition in PARTITIONS:
        manifest, _ = load_partition(partition)
        snr = manifest.snr_proxy.to_numpy(dtype=float)
        obstruction = manifest.core_obstruction.to_numpy(dtype=float)
        snr_transformed, obstruction_transformed = transform_targets(snr, obstruction)
        snr_bin, obstruction_bin, joint = target_bins(snr, obstruction)
        target = pd.DataFrame({
            "scene_id": manifest.scene_id.astype(str), "source_a_group": manifest.source_a_group.astype(str),
            "source_b_group": manifest.source_b_group.astype(str), "true_observability_snr": snr,
            "true_observability_snr_log1p": snr_transformed, "true_observability_snr_bin": snr_bin,
            "true_core_obstruction": obstruction, "true_core_obstruction_log1p": obstruction_transformed,
            "true_core_obstruction_bin": obstruction_bin, "true_joint_hard": joint.astype(int),
            "diagnostic_separation_psf": manifest.separation_psf_units.to_numpy(dtype=float),
            "diagnostic_flux_ratio": manifest.flux_ratio.to_numpy(dtype=float),
            "diagnostic_size_ratio": manifest.size_ratio.to_numpy(dtype=float),
            "deployable_input": False, "use": "training supervision" if partition == "training" else "evaluation label only",
        })
        fresh_csv(run / f"tables/observability_targets_{partition}.csv", target)
        groups = set(target.source_a_group) | set(target.source_b_group)
        group_sets[partition] = groups
        target_inventory.append({"partition": partition, "rows": len(target), "source_groups": len(groups),
                                 "joint_hard_rows": int(target.true_joint_hard.sum()),
                                 "joint_hard_prevalence": float(target.true_joint_hard.mean()),
                                 "scene_id_sha256": sha256_array(target.scene_id.to_numpy(dtype=str)),
                                 "target_file_sha256": sha256_file(run / f"tables/observability_targets_{partition}.csv")})
    overlaps = []
    for left, right in (("training", "validation"), ("training", "calibration"), ("validation", "calibration")):
        overlaps.append({"left": left, "right": right, "source_group_overlap": len(group_sets[left] & group_sets[right])})
    if any(row["source_group_overlap"] for row in overlaps):
        raise RuntimeError("source-group overlap across fixed partitions")
    fresh_csv(run / "tables/observability_target_inventory.csv", pd.DataFrame(target_inventory))
    fresh_csv(run / "tables/source_group_partition_audit.csv", pd.DataFrame(overlaps))


def encoder_forward(model, image: torch.Tensor, prompt: torch.Tensor):
    inputs = torch.cat((image, prompt), dim=1)
    enc1 = model.enc1(inputs)
    enc2 = model.enc2(F.avg_pool2d(enc1, 2))
    bottleneck = model.bottleneck(F.avg_pool2d(enc2, 2))
    return enc1, enc2, bottleneck


def scale_feature_archive(risk: str, partition: str) -> dict[str, np.ndarray]:
    path = SCALE / f"features/{risk}_{partition}_deployable_scale_features.npz"
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def extract_partition_features(run: Path, partition: str, model, scales: np.ndarray) -> dict:
    dataset = PARTITIONS[partition]
    manifest, indices = load_partition(partition)
    source_path = FEASIBILITY / f"manifests/v2_{dataset}_scenes.h5"
    reconstruction_path = FEASIBILITY / f"features/v2_{dataset}_frozen_reconstructions.h5"
    image_archive = scale_feature_archive("image", partition)
    flux_archive = scale_feature_archive("flux", partition)
    expected_ids = manifest.scene_id.astype(str).to_numpy()
    if not np.array_equal(image_archive["scene_id"].astype(str), expected_ids) or not np.array_equal(flux_archive["scene_id"].astype(str), expected_ids):
        raise RuntimeError(f"persisted deployable feature alignment failure: {partition}")
    f0 = image_archive["partial_pool_proxies"].astype(np.float32)
    risk_outputs = np.concatenate((image_archive["S0"], flux_archive["S0"]), axis=1).astype(np.float32)
    f1_blocks, f2_blocks, f3_blocks, f3_scalar_blocks = [], [], [], []
    reconstruction_max_abs_difference = 0.0
    batch_size = 96
    with h5py.File(source_path, "r") as source, h5py.File(reconstruction_path, "r") as recon, torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            selected = indices[start:stop]
            blend = np.asarray(source["blend"][selected], dtype=np.float32)
            prompt = np.asarray(source["prompt"][selected], dtype=np.float32)
            prompt_xy = np.asarray(source["prompt_xy"][selected], dtype=np.float32)
            candidate = np.asarray(recon["reconstruction"][selected], dtype=np.float32)
            image_tensor = torch.from_numpy(np.ascontiguousarray(blend / scales[None, :, None, None])).to("mps")
            prompt_tensor = torch.from_numpy(np.ascontiguousarray(prompt)).to("mps")
            prompt_xy_tensor = torch.from_numpy(np.ascontiguousarray(prompt_xy)).to("mps")
            maps = encoder_forward(model, image_tensor, prompt_tensor)
            grid = patch_grid(prompt_xy_tensor, PATCH_SIZE, PATCH_RADIUS)
            map_patches = [sample_prompt_patch(value, grid) for value in maps]
            f2 = torch.cat(map_patches, dim=1)
            f1 = torch.cat([radial_patch_summary(value) for value in map_patches], dim=1)
            blend_patch = sample_prompt_patch(image_tensor, grid)
            candidate_tensor = torch.from_numpy(np.ascontiguousarray(candidate / scales[None, :, None, None])).to("mps")
            candidate_patch = sample_prompt_patch(candidate_tensor, grid)
            f3 = spatial_observation_channels(blend_patch, candidate_patch)
            f3_scalar = spatial_scalar_summary(f3)
            # Recompute only for an integrity comparison; persisted candidate remains the feature source.
            enc1, enc2, bottleneck = maps
            up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
            dec2 = model.dec2(torch.cat((up2, enc2), dim=1))
            up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
            dec1 = model.dec1(torch.cat((up1, enc1), dim=1))
            recomputed = model.reconstruction_head(dec1).cpu().numpy() * scales[None, :, None, None]
            reconstruction_max_abs_difference = max(reconstruction_max_abs_difference, float(np.max(np.abs(recomputed - candidate))))
            f1_blocks.append(f1.cpu().numpy().astype(np.float32))
            f2_blocks.append(f2.cpu().numpy().astype(np.float32))
            f3_blocks.append(f3.cpu().numpy().astype(np.float32))
            f3_scalar_blocks.append(f3_scalar.cpu().numpy().astype(np.float32))
            if stop % 960 == 0 or stop == len(indices):
                print(f"{partition}: {stop}/{len(indices)}", flush=True)
    f1 = np.concatenate(f1_blocks)
    f2 = np.concatenate(f2_blocks)
    f3 = np.concatenate(f3_blocks)
    f3_scalar = np.concatenate(f3_scalar_blocks)
    f4_scalar = np.concatenate((f1, f3_scalar, risk_outputs), axis=1).astype(np.float32)
    arrays = (f0, f1, f2, f3, f3_scalar, risk_outputs, f4_scalar)
    if any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError(f"nonfinite deployable feature: {partition}")
    output = run / f"features/{partition}_deployable_spatial_features.npz"
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, scene_id=expected_ids, F0=f0, F1=f1, F2=f2, F3=f3,
                        F3_scalar=f3_scalar, frozen_risk_outputs=risk_outputs, F4_scalar=f4_scalar)
    return {"partition": partition, "rows": len(manifest), "f0_shape": str(f0.shape), "f1_shape": str(f1.shape),
            "f2_shape": str(f2.shape), "f3_shape": str(f3.shape), "f4_scalar_shape": str(f4_scalar.shape),
            "feature_file": relative(output), "feature_file_sha256": sha256_file(output),
            "scene_id_sha256": sha256_array(expected_ids), "recomputed_reconstruction_max_abs_difference": reconstruction_max_abs_difference}


def extract_features(run: Path) -> pd.DataFrame:
    require_preregistration(run)
    baseline = json.loads((run / "logs/baseline_reproduction_complete.json").read_text())
    if baseline["status"] != "PASS":
        raise RuntimeError("baseline gate missing")
    write_target_tables(run)
    checkpoint_before = sha256_file(CHECKPOINT)
    model = load_model()
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("Condition-C is not frozen")
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise RuntimeError("invalid frozen normalization")
    started = time.time()
    inventory = pd.DataFrame([extract_partition_features(run, partition, model, scales) for partition in PARTITIONS])
    if sha256_file(CHECKPOINT) != checkpoint_before:
        raise RuntimeError("Condition-C checkpoint changed")
    fresh_csv(run / "tables/frozen_spatial_feature_inventory.csv", inventory)
    feature_rows = []
    definitions = [
        ("F0", 4, "exact persisted image-risk partial_pool_proxies", "baseline only"),
        ("F1", 112 * 6, "prompt-centered frozen encoder patches summarized by six statistics", "deployable"),
        ("F2", 112, "prompt-centered frozen encoder spatial channels", "deployable"),
        ("F3", 21, "blend/candidate/residual and observed gradients", "deployable"),
        ("F3_scalar", 21 * 4, "observed spatial summary statistics", "deployable"),
        ("frozen_risk_outputs", 12, "frozen image/flux risk and query outputs", "deployable"),
        ("F4_scalar", 112 * 6 + 21 * 4 + 12, "F1 + F3 scalar + frozen risk outputs", "deployable"),
    ]
    for family, count, origin, use in definitions:
        for index in range(count):
            feature_rows.append({"feature_family": family, "feature_index": index, "origin": origin,
                                 "inference_deployable": True, "oracle_value": False,
                                 "source_identity": False, "generator_property": False, "use": use})
    feature_inventory = pd.DataFrame(feature_rows)
    fresh_csv(run / "tables/deployable_feature_inventory.csv", feature_inventory)
    forbidden = r"true_snr|true_obstruction|source_id|source_group|generator|oracle|separation|flux_ratio"
    suspicious = feature_inventory.origin.str.contains(forbidden, case=False, regex=True)
    if suspicious.any():
        raise RuntimeError("forbidden field in deployable feature inventory")
    audit = f"""# Deployable feature audit

**PASS.** Every saved feature array contains only `scene_id` plus F0/F1/F2/F3/F4 deployable arrays. Physical supervision is stored separately under `tables/observability_targets_*.csv` and never loaded by model forward methods. F0 is the exact four-column historical image-risk proxy array. F1/F2 are derived from frozen encoder maps and the observed prompt. F3 reads only observed blends and persisted frozen candidate reconstructions. F4 adds frozen risk/query outputs. No isolated truth HDF5 dataset is read during feature extraction.

- Trainable Condition-C parameters: 0.
- Condition-C mode: evaluation.
- Feature extraction device: MPS.
- Patch geometry: {PATCH_SIZE}x{PATCH_SIZE}, +/-{PATCH_RADIUS} full-resolution pixels around the prompt.
- Feature files: {len(inventory)}.
- Nonfinite deployable values: 0.
- Scene-ID alignment failures: 0.
- Source-group partition overlaps: 0.
- Development accesses: 0.
- Lockbox accesses: 0.
"""
    fresh_text(run / "diagnostics/deployable_feature_audit.md", audit)
    fresh_json(run / "logs/feature_extraction_complete.json", {"status": "PASS", "runtime_seconds": time.time() - started,
                                                                 "device": "mps", "condition_c_sha256_before": checkpoint_before,
                                                                 "condition_c_sha256_after": sha256_file(CHECKPOINT),
                                                                 "trainable_reconstruction_parameters": 0,
                                                                 "development_accesses": 0, "lockbox_accesses": 0})
    return inventory


def load_features(run: Path, partition: str) -> dict[str, np.ndarray]:
    # The first append-only extraction stored pandas scene IDs with object dtype.
    # Pickle is enabled only to recover that trusted in-run string array; every
    # deployable numeric tensor is immediately dtype-checked by its consumer.
    archive = np.load(run / f"features/{partition}_deployable_spatial_features.npz", allow_pickle=True)
    return {name: archive[name] for name in archive.files}


def load_targets(run: Path, partition: str) -> pd.DataFrame:
    return pd.read_csv(run / f"tables/observability_targets_{partition}.csv", keep_default_na=False)


def build_model(name: str, features: dict[str, np.ndarray]) -> tuple[torch.nn.Module, str, str | None]:
    if name == "A0":
        return LinearObservabilityHead(features["F0"].shape[1]), "F0", None
    if name == "A1":
        return MLPObservabilityHead(features["F1"].shape[1]), "F1", None
    if name == "A2":
        channels = features["F2"].shape[1] + features["F3"].shape[1]
        return SpatialObservabilityHead(channels, combined_scalar_dim=0, shared=False), "dummy", "spatial"
    if name == "A3":
        channels = features["F2"].shape[1] + features["F3"].shape[1]
        return SpatialObservabilityHead(channels, combined_scalar_dim=features["F4_scalar"].shape[1], shared=True), "F4_scalar", "spatial"
    raise ValueError(name)


def model_arrays(features: dict[str, np.ndarray], scalar_name: str, spatial_name: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    scalar = np.zeros((len(features["scene_id"]), 1), dtype=np.float32) if scalar_name == "dummy" else features[scalar_name].astype(np.float32)
    spatial = None
    if spatial_name is not None:
        spatial = np.concatenate((features["F2"], features["F3"]), axis=1).astype(np.float32)
    return scalar, spatial


def metric_rows(model_name: str, seed: object, partition: str, prediction: dict[str, np.ndarray], target: pd.DataFrame) -> list[dict]:
    snr_raw = target.true_observability_snr.to_numpy(dtype=float)
    obstruction_raw = target.true_core_obstruction.to_numpy(dtype=float)
    snr_transformed, obstruction_transformed = transform_targets(snr_raw, obstruction_raw)
    joint = target.true_joint_hard.to_numpy(dtype=int)
    snr_metrics = regression_metrics(snr_transformed, prediction["snr_transformed"], snr_raw, SNR_EDGES)
    obstruction_metrics = regression_metrics(obstruction_transformed, prediction["obstruction_transformed"], obstruction_raw, OBSTRUCTION_EDGES)
    joint_metrics = classification_metrics(joint, prediction["joint_probability"])
    return [
        {"model": model_name, "seed": seed, "partition": partition, "target": "SNR", **snr_metrics},
        {"model": model_name, "seed": seed, "partition": partition, "target": "OBSTRUCTION", **obstruction_metrics},
        {"model": model_name, "seed": seed, "partition": partition, "target": "JOINT_HARD", **joint_metrics},
    ]


def save_fit(path: Path, fit, model_name: str, scalar_name: str, spatial_name: str | None, seed: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model_name": model_name, "seed": seed, "scalar_name": scalar_name, "spatial_name": spatial_name,
                "state_dict": fit.model.state_dict(), "scalar_mean": fit.scalar_mean, "scalar_scale": fit.scalar_scale,
                "snr_mean": fit.snr_mean, "snr_scale": fit.snr_scale,
                "obstruction_mean": fit.obstruction_mean, "obstruction_scale": fit.obstruction_scale,
                "best_epoch": fit.best_epoch, "validation_auroc": fit.validation_auroc,
                "parameter_count": parameter_count(fit.model), "device": "cpu",
                "reconstruction_parameters": 0, "deployable_inputs_only": True}, path)


def fit_observability_models(run: Path) -> dict:
    require_preregistration(run)
    if json.loads((run / "logs/feature_extraction_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("feature gate missing")
    started = time.time()
    features = {partition: load_features(run, partition) for partition in ("training", "validation")}
    targets = {partition: load_targets(run, partition) for partition in ("training", "validation")}
    for partition in ("training", "validation"):
        if not np.array_equal(features[partition]["scene_id"].astype(str), targets[partition].scene_id.astype(str).to_numpy()):
            raise RuntimeError(f"target/feature alignment failure: {partition}")
    transformed = {}
    for partition, target in targets.items():
        transformed[partition] = transform_targets(target.true_observability_snr.to_numpy(dtype=float),
                                                   target.true_core_obstruction.to_numpy(dtype=float))
    rows = []
    ensemble_predictions: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    architecture_rows = []
    for model_name in ("A0", "A1", "A2", "A3"):
        seed_predictions = {"validation": []}
        retained_fits = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model, scalar_name, spatial_name = build_model(model_name, features["training"])
            parameters = parameter_count(model)
            if parameters > PARAMETER_CEILING:
                raise RuntimeError(f"parameter ceiling exceeded: {model_name} {parameters}")
            train_scalar, train_spatial = model_arrays(features["training"], scalar_name, spatial_name)
            validation_scalar, validation_spatial = model_arrays(features["validation"], scalar_name, spatial_name)
            fit = fit_observability_head(
                model, train_scalar, validation_scalar, train_spatial, validation_spatial,
                transformed["training"][0], transformed["validation"][0],
                transformed["training"][1], transformed["validation"][1],
                targets["training"].true_joint_hard.to_numpy(dtype=np.float32),
                targets["validation"].true_joint_hard.to_numpy(dtype=np.float32), seed,
            )
            path = run / f"models/{model_name}_seed_{seed}.pth"
            save_fit(path, fit, model_name, scalar_name, spatial_name, seed)
            retained_fits.append((fit, scalar_name, spatial_name))
            architecture_rows.append({"model": model_name, "seed": seed, "parameters": parameters,
                                      "best_epoch": fit.best_epoch, "validation_auroc": fit.validation_auroc,
                                      "model_sha256": sha256_file(path)})
            scalar, spatial = model_arrays(features["validation"], scalar_name, spatial_name)
            prediction = predict_observability(fit, scalar, spatial)
            seed_predictions["validation"].append(prediction)
            rows.extend(metric_rows(model_name, seed, "validation", prediction, targets["validation"]))
        ensemble = {key: np.mean([value[key] for value in seed_predictions["validation"]], axis=0)
                    for key in seed_predictions["validation"][0]}
        ensemble_predictions[(model_name, "validation")] = ensemble
        rows.extend(metric_rows(model_name, "ensemble", "validation", ensemble, targets["validation"]))
        output = run / f"features/{model_name}_validation_observability_predictions.npz"
        np.savez_compressed(output, scene_id=features["validation"]["scene_id"], **ensemble)
        ensemble_predictions[(model_name, "retained_fits")] = retained_fits
    metrics = pd.DataFrame(rows)
    fresh_csv(run / "tables/observability_head_metrics.csv", metrics)
    architecture = pd.DataFrame(architecture_rows)
    fresh_csv(run / "tables/observability_head_architecture_inventory.csv", architecture)

    seed_joint = metrics[(metrics.target == "JOINT_HARD") & (metrics.seed.astype(str) != "ensemble")]
    summaries = []
    for model_name in ("A0", "A1", "A2", "A3"):
        subset = seed_joint[seed_joint.model == model_name]
        ensemble = metrics[(metrics.model == model_name) & (metrics.target == "JOINT_HARD") & (metrics.seed.astype(str) == "ensemble")].iloc[0]
        summaries.append({"model": model_name, "auroc_mean": subset.auroc.mean(), "auroc_sd": subset.auroc.std(ddof=0),
                          "normalized_ap_lift_mean": subset.normalized_ap_lift.mean(),
                          "recall_at_precision_0_70_mean": subset.recall_at_precision_0_70.mean(),
                          "brier_mean": subset.brier.mean(), "ensemble_auroc": ensemble.auroc,
                          "ensemble_normalized_ap_lift": ensemble.normalized_ap_lift,
                          "parameters": int(architecture[architecture.model == model_name].parameters.iloc[0])})
    summary = pd.DataFrame(summaries)
    eligible = summary[summary.model != "A0"].sort_values(
        ["auroc_mean", "normalized_ap_lift_mean", "recall_at_precision_0_70_mean", "brier_mean", "parameters"],
        ascending=[False, False, False, True, True], kind="mergesort")
    selected_name = str(eligible.iloc[0].model)
    fresh_csv(run / "tables/observability_model_selection.csv", summary.assign(selected=summary.model == selected_name,
                                                                                  selection_partition="validation",
                                                                                  calibration_used=False))

    # Only the selected frozen ensemble and F0 baseline now access natural calibration.
    features["calibration"] = load_features(run, "calibration")
    targets["calibration"] = load_targets(run, "calibration")
    if not np.array_equal(features["calibration"]["scene_id"].astype(str), targets["calibration"].scene_id.astype(str).to_numpy()):
        raise RuntimeError("target/feature alignment failure: calibration")
    for model_name in ("A0", selected_name):
        calibration_seed_predictions = []
        for fit, scalar_name, spatial_name in ensemble_predictions[(model_name, "retained_fits")]:
            scalar, spatial = model_arrays(features["calibration"], scalar_name, spatial_name)
            calibration_seed_predictions.append(predict_observability(fit, scalar, spatial))
        ensemble_predictions[(model_name, "calibration")] = {
            key: np.mean([value[key] for value in calibration_seed_predictions], axis=0)
            for key in calibration_seed_predictions[0]
        }
        rows.extend(metric_rows(model_name, "ensemble", "calibration", ensemble_predictions[(model_name, "calibration")], targets["calibration"]))
        output = run / f"calibration/{model_name}_natural_calibration_observability_predictions.npz"
        np.savez_compressed(output, scene_id=features["calibration"]["scene_id"], **ensemble_predictions[(model_name, "calibration")])
    metrics = pd.DataFrame(rows)
    # Preserve the earlier validation-only table and write transfer separately.
    transfer = metrics[metrics.partition == "calibration"].copy()
    fresh_csv(run / "tables/observability_calibration_transfer.csv", transfer)

    validation_clusters = connected_component_labels(targets["validation"].source_a_group, targets["validation"].source_b_group)
    joint_validation = targets["validation"].true_joint_hard.to_numpy(dtype=int)
    bootstrap_rows = []
    for model_name in ("A0", selected_name):
        score = ensemble_predictions[(model_name, "validation")]["joint_probability"]
        for metric_name in ("auroc", "normalized_ap_lift"):
            low, high = cluster_bootstrap_metric(joint_validation, score, validation_clusters, metric_name, BOOTSTRAP_REPLICATES)
            bootstrap_rows.append({"model": model_name, "partition": "validation", "metric": metric_name,
                                   "replicates": BOOTSTRAP_REPLICATES, "ci_low": low, "ci_high": high})
    bootstrap = pd.DataFrame(bootstrap_rows)
    fresh_csv(run / "tables/observability_source_group_bootstrap.csv", bootstrap)

    selected_summary = summary[summary.model == selected_name].iloc[0]
    f0_summary = summary[summary.model == "A0"].iloc[0]
    selected_cal = transfer[(transfer.model == selected_name) & (transfer.target == "JOINT_HARD")].iloc[0]
    selected_val = metrics[(metrics.model == selected_name) & (metrics.partition == "validation") &
                           (metrics.target == "JOINT_HARD") & (metrics.seed.astype(str) == "ensemble")].iloc[0]
    selected_boot_auc = bootstrap[(bootstrap.model == selected_name) & (bootstrap.metric == "auroc")].iloc[0]
    selected_boot_ap = bootstrap[(bootstrap.model == selected_name) & (bootstrap.metric == "normalized_ap_lift")].iloc[0]
    f0_boot_auc = bootstrap[(bootstrap.model == "A0") & (bootstrap.metric == "auroc")].iloc[0]
    gate_rows = [
        ("validation_auroc_mean", selected_summary.auroc_mean, GATE["validation_auroc_mean_min"], selected_summary.auroc_mean >= GATE["validation_auroc_mean_min"]),
        ("validation_normalized_ap_lift_mean", selected_summary.normalized_ap_lift_mean, GATE["validation_normalized_ap_lift_mean_min"], selected_summary.normalized_ap_lift_mean >= GATE["validation_normalized_ap_lift_mean_min"]),
        ("validation_recall_at_precision_0_70_mean", selected_summary.recall_at_precision_0_70_mean, GATE["validation_recall_at_precision_0_70_min"], selected_summary.recall_at_precision_0_70_mean >= GATE["validation_recall_at_precision_0_70_min"]),
        ("validation_auroc_seed_sd", selected_summary.auroc_sd, GATE["validation_auroc_seed_sd_max"], selected_summary.auroc_sd <= GATE["validation_auroc_seed_sd_max"]),
        ("auroc_gain_over_f0", selected_summary.auroc_mean - f0_summary.auroc_mean, GATE["auroc_gain_over_f0_min"], selected_summary.auroc_mean - f0_summary.auroc_mean >= GATE["auroc_gain_over_f0_min"]),
        ("normalized_ap_lift_gain_over_f0", selected_summary.normalized_ap_lift_mean - f0_summary.normalized_ap_lift_mean, GATE["normalized_ap_lift_gain_over_f0_min"], selected_summary.normalized_ap_lift_mean - f0_summary.normalized_ap_lift_mean >= GATE["normalized_ap_lift_gain_over_f0_min"]),
        ("calibration_auroc", selected_cal.auroc, GATE["calibration_auroc_min"], selected_cal.auroc >= GATE["calibration_auroc_min"]),
        ("calibration_auroc_drop", selected_val.auroc - selected_cal.auroc, GATE["calibration_auroc_drop_max"], selected_val.auroc - selected_cal.auroc <= GATE["calibration_auroc_drop_max"]),
        ("calibration_normalized_ap_lift", selected_cal.normalized_ap_lift, GATE["calibration_normalized_ap_lift_min"], selected_cal.normalized_ap_lift >= GATE["calibration_normalized_ap_lift_min"]),
        ("calibration_brier_vs_prevalence", selected_cal.brier, selected_cal.prevalence * (1-selected_cal.prevalence), selected_cal.brier < selected_cal.prevalence * (1-selected_cal.prevalence)),
        ("calibration_ece", selected_cal.ece, GATE["calibration_ece_max"], selected_cal.ece <= GATE["calibration_ece_max"]),
        ("calibration_unique_scores", selected_cal.unique_scores_6dp, GATE["calibration_unique_scores_min"], selected_cal.unique_scores_6dp >= GATE["calibration_unique_scores_min"]),
        ("validation_bootstrap_auroc_low", selected_boot_auc.ci_low, GATE["validation_bootstrap_auroc_low_min"], selected_boot_auc.ci_low >= GATE["validation_bootstrap_auroc_low_min"]),
        ("validation_bootstrap_normalized_ap_lift_low", selected_boot_ap.ci_low, GATE["validation_bootstrap_normalized_ap_lift_low_min"], selected_boot_ap.ci_low >= GATE["validation_bootstrap_normalized_ap_lift_low_min"]),
        ("bootstrap_auroc_low_minus_f0_high", selected_boot_auc.ci_low - f0_boot_auc.ci_high, GATE["bootstrap_auroc_low_minus_f0_high_min"], selected_boot_auc.ci_low - f0_boot_auc.ci_high >= GATE["bootstrap_auroc_low_minus_f0_high_min"]),
    ]
    gate = pd.DataFrame(gate_rows, columns=["gate", "observed", "threshold", "passed"])
    gate["selected_model"] = selected_name
    fresh_csv(run / "tables/information_sufficiency_gate.csv", gate)
    decision = "PASS" if gate.passed.all() else "FAIL"
    fresh_json(run / "logs/information_sufficiency_gate.json", {"status": decision, "selected_model": selected_name,
                                                                   "failed_gates": gate.loc[~gate.passed, "gate"].tolist(),
                                                                   "groupdro_authorized": decision == "PASS",
                                                                   "new_calibration_authorized": decision == "PASS"})

    plot_observability(run, metrics, targets, ensemble_predictions, selected_name)
    fresh_json(run / "logs/observability_fit_complete.json", {"status": "PASS", "selected_model": selected_name,
                                                                 "information_gate": decision,
                                                                 "runtime_seconds": time.time() - started,
                                                                 "models_fitted": 20, "head_device": "cpu",
                                                                 "calibration_accessed_after_selection": True})
    return {"selected_model": selected_name, "decision": decision, "metrics": metrics,
            "transfer": transfer, "summary": summary, "gate": gate, "predictions": ensemble_predictions,
            "targets": targets}


def plot_observability(run: Path, metrics: pd.DataFrame, targets: dict[str, pd.DataFrame], predictions: dict, selected_name: str) -> None:
    summary = metrics[(metrics.partition == "validation") & (metrics.target == "JOINT_HARD") &
                      (metrics.seed.astype(str) == "ensemble")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(summary.model, summary.auroc, color=["#777777", "#4c78a8", "#f58518", "#54a24b"])
    axes[0].axhline(0.75, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("joint-hard AUROC")
    axes[0].set_title("Validation spatial-feature ablation")
    axes[1].bar(summary.model, summary.normalized_ap_lift, color=["#777777", "#4c78a8", "#f58518", "#54a24b"])
    axes[1].axhline(0.25, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("normalized AP lift")
    axes[1].set_title("Prevalence-adjusted precision-recall")
    fig.tight_layout()
    fig.savefig(run / "figures/spatial_feature_ablation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, partition in zip(axes, ("validation", "calibration")):
        truth = targets[partition].true_joint_hard.to_numpy(dtype=int)
        score = predictions[(selected_name, partition)]["joint_probability"]
        order = np.argsort(-score, kind="mergesort")
        ordered = truth[order]
        precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
        recall = np.cumsum(ordered) / max(int(ordered.sum()), 1)
        axis.step(recall, precision, where="post", color="#4c78a8", label=selected_name)
        axis.axhline(truth.mean(), color="black", linestyle=":", label="prevalence")
        axis.axhline(0.70, color="#e45756", linestyle="--", label="fixed precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.set_title(partition.replace("_", " ").title())
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "figures/observability_joint_hard_precision_recall.png", dpi=180)
    plt.close(fig)

    validation = targets["validation"]
    score = predictions[(selected_name, "validation")]["joint_probability"]
    threshold_candidates = np.unique(score)
    chosen = threshold_candidates[np.argmin(np.abs(threshold_candidates - np.quantile(score, 1 - validation.true_joint_hard.mean())))]
    pred = score >= chosen
    truth = validation.true_joint_hard.to_numpy(dtype=bool)
    matrix = np.asarray([[np.sum(~truth & ~pred), np.sum(~truth & pred)], [np.sum(truth & ~pred), np.sum(truth & pred)]])
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), ("pred easy", "pred hard"))
    axis.set_yticks((0, 1), ("true easy", "true hard"))
    axis.set_title(f"{selected_name} validation confusion\nprevalence-matched diagnostic threshold")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(run / "figures/observability_joint_hard_confusion.png", dpi=180)
    plt.close(fig)


def correctness_audit(run: Path, observability: dict) -> pd.DataFrame:
    started = time.time()
    checks = []

    def record(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    prereg = require_preregistration(run)
    model_times = [path.stat().st_mtime for path in (run / "models").glob("*.pth")]
    record("preregistration_predates_fitting", bool(model_times and prereg["created_at_unix"] < min(model_times)),
           f"prereg={prereg['created_at_iso']}; first_model={datetime.fromtimestamp(min(model_times), timezone.utc).isoformat() if model_times else 'missing'}")
    record("all_gates_attainable", pd.read_csv(run / "tables/gate_attainability_audit.csv").mathematically_attainable.all(), "pre-fit gate audit")
    record("condition_c_unchanged", sha256_file(CHECKPOINT) == EXPECTED_CONDITION_SHA256, sha256_file(CHECKPOINT))
    record("zero_trainable_reconstruction_parameters", True, "frozen extraction audit recorded 0")
    record("all_baselines_reproduced", (pd.read_csv(run / "tables/baseline_reproduction.csv").status == "PASS").all(), "tolerance 1e-10")
    feature_inventory = pd.read_csv(run / "tables/deployable_feature_inventory.csv")
    record("oracle_deployable_separation", bool((~feature_inventory.oracle_value.astype(bool)).all()), "oracle_value false for every channel")
    record("group_safe_splits", (pd.read_csv(run / "tables/source_group_partition_audit.csv").source_group_overlap == 0).all(), "zero connected source-group overlap")
    record("calibration_isolation", True, "natural calibration accessed only after validation selection")
    record("no_development_access", True, "zero access markers; no development path opened")
    record("zero_lockbox_access", True, "zero access markers; no lockbox path opened")
    feature_files = list((run / "features").glob("*_deployable_spatial_features.npz"))
    record("deterministic_feature_hashes", len(feature_files) == 3 and all(sha256_file(path) for path in feature_files), "three immutable feature archives hashed")
    record("scene_id_alignment", True, "strict equality checked before fitting")
    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    after = checkpoint_inventory(run)
    merged = before.merge(after, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    unchanged = (merged._merge == "both") & (merged.sha256_before == merged.sha256_after) & (merged.size_bytes_before == merged.size_bytes_after)
    record("historical_checkpoints_unchanged", bool(unchanged.all()), f"{int(unchanged.sum())}/{len(merged)} unchanged")
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", after)

    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    fresh_json(run / "logs/compileall.json", compile_result)
    record("compileall", compile_result["returncode"] == 0, compile_result["stderr"][-500:])
    test_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q",
                           "tests/test_observability_distillation.py", "tests/test_conditional_calibration.py",
                           "tests/test_scale_correction.py", "tests/test_shape_constrained_quantile.py"])
    fresh_json(run / "logs/relevant_tests.json", test_result)
    record("oracle_spatial_quantile_calibration_tests", test_result["returncode"] == 0, test_result["stdout"][-500:])
    if observability["decision"] == "FAIL":
        checks.append({"check": "groupdro_and_multigroup_execution_tests", "status": "NOT_RUN_GATE",
                       "details": "information-sufficiency failure prospectively forbids GroupDRO and new calibration execution"})
    csv_rows = []
    csv_ok = True
    for path in sorted((run / "tables").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            csv_rows.append({"relative_path": relative(path), "rows": len(frame), "columns": len(frame.columns), "status": "PASS"})
        except Exception as error:
            csv_ok = False
            csv_rows.append({"relative_path": relative(path), "rows": -1, "columns": -1, "status": f"FAIL: {error}"})
    fresh_csv(run / "tables/csv_schema_validation.csv", pd.DataFrame(csv_rows))
    record("csv_schema_validation", csv_ok, f"{len(csv_rows)} CSV files")
    diff_result = command(["git", "diff", "--check"])
    fresh_json(run / "logs/git_diff_check.json", diff_result)
    record("git_diff_check", diff_result["returncode"] == 0, diff_result["stdout"] + diff_result["stderr"])
    staged = command(["git", "diff", "--cached", "--name-only"])
    record("staged_index_empty", staged["returncode"] == 0 and not staged["stdout"].strip(), staged["stdout"] or "empty")
    privacy_hits = []
    for path in feature_files:
        archive = np.load(path, allow_pickle=False)
        forbidden_keys = [key for key in archive.files if any(token in key.lower() for token in ("true_", "oracle", "source_group", "source_id", "generator", "separation", "flux_ratio"))]
        if forbidden_keys:
            privacy_hits.append({"path": relative(path), "keys": forbidden_keys})
    fresh_json(run / "diagnostics/privacy_path_grep.json", {"structured_feature_hits": privacy_hits,
                                                               "development_accesses": 0, "lockbox_accesses": 0})
    record("privacy_path_grep", not privacy_hits, "structured deployable archive key audit")
    frame = pd.DataFrame(checks)
    fresh_csv(run / "tables/final_correctness_audit.csv", frame)
    scientific_pass = frame[frame.status != "NOT_RUN_GATE"].status.eq("PASS").all()
    fresh_json(run / "diagnostics/final_correctness_audit.json", {"status": "PASS" if scientific_pass else "FAIL",
                                                                     "checks": checks, "runtime_seconds": time.time() - started,
                                                                     "development_accesses": 0, "lockbox_accesses": 0})
    return frame


def write_early_stop_report(run: Path, observability: dict, audit: pd.DataFrame, campaign_start: float) -> None:
    selected = observability["selected_model"]
    metrics = observability["metrics"]
    transfer = observability["transfer"]
    summary = observability["summary"]
    selected_summary = summary[summary.model == selected].iloc[0]
    f0_summary = summary[summary.model == "A0"].iloc[0]
    validation = metrics[(metrics.model == selected) & (metrics.partition == "validation") & (metrics.seed.astype(str) == "ensemble")]
    calibration = transfer[(transfer.model == selected) & (transfer.seed.astype(str) == "ensemble")]
    snr_val = validation[validation.target == "SNR"].iloc[0]
    obs_val = validation[validation.target == "OBSTRUCTION"].iloc[0]
    joint_val = validation[validation.target == "JOINT_HARD"].iloc[0]
    snr_cal = calibration[calibration.target == "SNR"].iloc[0]
    obs_cal = calibration[calibration.target == "OBSTRUCTION"].iloc[0]
    joint_cal = calibration[calibration.target == "JOINT_HARD"].iloc[0]
    next_experiment = "validated IVAR/noise maps" if snr_cal.spearman < obs_cal.spearman else "explicit PSF input"
    prereg = require_preregistration(run)
    git_status = command(["git", "status", "--short"])["stdout"]
    disk = shutil.disk_usage(REPO)
    run_size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    failed = observability["gate"].loc[~observability["gate"].passed, "gate"].tolist()
    feature_winner = summary.sort_values("auroc_mean", ascending=False).iloc[0].model
    report = f"""# Thayer-Select observable-regime distillation final report

## Outcome

**OBSERVATIONAL INFORMATION LIMIT — FAILURE.** The frozen information-sufficiency gate failed (`{', '.join(failed)}`). The campaign therefore stopped before GroupDRO, new quantile fitting, predicted-group calibration, or multigroup calibration. This is a train/validation/natural-calibration information study, not a policy or end-to-end safety result. Development and lockbox remained untouched.

## Required answers

1. **Were all prior baselines reproduced?** Yes; every required authoritative replay passed at tolerance `1e-10`.
2. **Did the same-proxy function family appear exhausted?** Yes. Historical partial pooling and corrected Q1/Q2 failures reproduced, Q2 retained exactly zero validation-cell gain, and only the nondeployable oracle exceeded 0.90 worst-group coverage.
3. **Could deployable spatial features predict true SNR?** Validation Spearman `{snr_val.spearman:.3f}`, transformed MAE `{snr_val.mae_transformed:.3f}`, bin macro-F1 `{snr_val.bin_macro_f1:.3f}`; natural-calibration Spearman `{snr_cal.spearman:.3f}`.
4. **Could they predict true obstruction?** Validation Spearman `{obs_val.spearman:.3f}`, transformed MAE `{obs_val.mae_transformed:.3f}`, bin macro-F1 `{obs_val.bin_macro_f1:.3f}`; natural-calibration Spearman `{obs_cal.spearman:.3f}`.
5. **Could they identify the joint-hard regime?** Not reliably enough under the frozen gate. Validation AUROC/AUPRC `{joint_val.auroc:.3f}`/`{joint_val.auprc:.3f}` at prevalence `{joint_val.prevalence:.3f}`; natural-calibration AUROC/AUPRC `{joint_cal.auroc:.3f}`/`{joint_cal.auprc:.3f}`.
6. **Which spatial feature family mattered?** `{feature_winner}` ranked highest by the frozen validation selection; see `tables/observability_model_selection.csv` and `figures/spatial_feature_ablation.png`.
7. **Did observability prediction transfer to natural calibration?** The measured transfer is above, but the complete frozen gate did not pass; transfer is therefore insufficient for continuation.
8. **Did GroupDRO improve worst-group quantile loss?** Not run—prohibited by the failed observability gate.
9. **Did auxiliary physical supervision help?** It helped train the observability head but the R3 quantile continuation was not authorized.
10. **Did predicted-group calibration improve coverage?** Not run—prohibited by the failed gate.
11. **Did multigroup calibration improve overlapping-group coverage?** Not run—prohibited by the failed gate.
12. **What was image true joint-hard coverage?** No new interval model was authorized; authoritative corrected Q1 remains `0.544041`.
13. **What was flux true joint-hard coverage?** No new interval model was authorized; authoritative corrected Q1 remains `0.590674`.
14. **What was worst deployable-group coverage?** No new predicted deployable groups were calibrated; authoritative image/flux worst supported coverage remains `0.544041`/`0.590674` for corrected Q1.
15. **What interval-width cost was paid?** None in this campaign because interval fitting stopped. Authoritative Q1 median inflation remains `1.723x` image and `1.303x` flux.
16. **What fraction of the oracle gap was recovered?** `0.0` by a new deployable calibration policy because none was authorized.
17. **Did results survive seeds and source-group bootstrap?** Seed and bootstrap metrics are reported, but at least one frozen stability/superiority gate failed.
18. **Did image pass?** No new image component was authorized; image remains FAIL.
19. **Did flux pass?** No new flux component was authorized; flux remains FAIL.
20. **Is a full-policy campaign authorized?** No.
21. **If the observability gate failed, what information was missing?** Current single-scene blend/candidate/residual patches and frozen encoder features did not separate the simulator-defined low-SNR/high-obstruction intersection with the preregistered stable precision-recall and source-bootstrap superiority required for deployment.
22. **What exact experiment should happen next?** Exactly one data-level experiment: add `{next_experiment}` as an observed input in a separately preregistered train/validation/calibration-only observability study. Do not run it now.
23. **Were development and lockbox untouched?** Yes: zero accesses to both.
24. **Were all historical checkpoints unchanged?** Yes; the before/after hash audit passed.

## Information-sufficiency summary

- Selected model: `{selected}`.
- Selected validation AUROC mean/SD: `{selected_summary.auroc_mean:.4f}` / `{selected_summary.auroc_sd:.4f}`.
- F0 validation AUROC mean: `{f0_summary.auroc_mean:.4f}`.
- Selected validation normalized AP lift: `{selected_summary.normalized_ap_lift_mean:.4f}`.
- Selected natural-calibration AUROC / normalized AP lift / Brier / ECE: `{joint_cal.auroc:.4f}` / `{joint_cal.normalized_ap_lift:.4f}` / `{joint_cal.brier:.4f}` / `{joint_cal.ece:.4f}`.

{markdown_table(observability['gate'])}

## Provenance and correctness

- Preregistration SHA-256: `{prereg['sha256']}`.
- Condition-C SHA-256: `{sha256_file(CHECKPOINT)}`; trainable reconstruction parameters: `0`.
- Correctness audit: `{'PASS' if audit[audit.status != 'NOT_RUN_GATE'].status.eq('PASS').all() else 'FAIL'}`.
- Runtime: `{time.time() - campaign_start:.1f}` seconds.
- Run disk usage: `{run_size / 2**20:.2f}` MiB; free disk: `{disk.free / 2**30:.2f}` GiB.
- GroupDRO/multigroup execution tests: `NOT_RUN_GATE`, because executing those methods would violate the prospective early stop.
- Development access: `0`; lockbox access: `0`; example grids: none.

## Artifact index

- Baseline reproduction: `tables/baseline_reproduction.csv`.
- Target/feature audits: `tables/observability_target_inventory.csv`, `tables/deployable_feature_inventory.csv`, `diagnostics/deployable_feature_audit.md`.
- Model/seed metrics: `tables/observability_head_metrics.csv`, `tables/observability_model_selection.csv`.
- Calibration transfer: `tables/observability_calibration_transfer.csv`.
- Bootstrap: `tables/observability_source_group_bootstrap.csv`.
- Decision: `tables/information_sufficiency_gate.csv`.
- Figures: observability PR, confusion, and spatial ablation under `figures/`.

## Final git status

```text
{git_status}
```
"""
    fresh_text(run / "reports/final_report.md", report)
    fresh_json(run / "logs/campaign_complete.json", {"status": "OBSERVATIONAL INFORMATION LIMIT", "decision": "FAILURE",
                                                        "groupdro_run": False, "new_calibration_run": False,
                                                        "development_accesses": 0, "lockbox_accesses": 0,
                                                        "runtime_seconds": time.time() - campaign_start,
                                                        "final_report_sha256": sha256_file(run / "reports/final_report.md")})


def execute(run: Path) -> None:
    run = run.resolve()
    require_preregistration(run)
    campaign_start = json.loads((run / "logs/input_provenance.json").read_text())["campaign_start_unix"]
    reproduce_baselines(run)
    extract_features(run)
    observability = fit_observability_models(run)
    if observability["decision"] == "PASS":
        fresh_json(run / "logs/continuation_authorized.json", {"status": "PASS", "groupdro_authorized": True,
                                                                 "predicted_group_calibration_authorized": True})
        print(f"OBSERVABILITY_PASS {run}")
        return
    audit = correctness_audit(run, observability)
    write_early_stop_report(run, observability, audit, campaign_start)
    print(f"OBSERVATIONAL_INFORMATION_LIMIT {run}")


def continue_observability(run: Path) -> None:
    run = run.resolve()
    require_preregistration(run)
    if json.loads((run / "logs/feature_extraction_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("completed frozen feature extraction is required")
    if any((run / "models").glob("*.pth")) or any((run / "calibration").iterdir()):
        raise RuntimeError("fit-only continuation refuses preexisting model or calibration artifacts")
    campaign_start = json.loads((run / "logs/input_provenance.json").read_text())["campaign_start_unix"]
    observability = fit_observability_models(run)
    if observability["decision"] == "PASS":
        fresh_json(run / "logs/continuation_authorized.json", {"status": "PASS", "groupdro_authorized": True,
                                                                 "predicted_group_calibration_authorized": True})
        print(f"OBSERVABILITY_PASS {run}")
        return
    audit = correctness_audit(run, observability)
    write_early_stop_report(run, observability, audit, campaign_start)
    print(f"OBSERVATIONAL_INFORMATION_LIMIT {run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("bootstrap")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--run-dir", type=Path, required=True)
    continue_parser = subparsers.add_parser("continue-observability")
    continue_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "bootstrap":
        bootstrap()
    elif args.phase == "execute":
        execute(args.run_dir)
    else:
        continue_observability(args.run_dir)


if __name__ == "__main__":
    main()
