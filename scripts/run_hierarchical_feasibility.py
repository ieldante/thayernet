#!/usr/bin/env python3
"""Append-only prospective hierarchical-safety feasibility campaign.

Stages are individually resumable and collision-refusing.  Data generation and
all fitting stages require a completed preregistration hash.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy
import torch
from astropy.table import Table


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.hierarchical_feasibility import (
    SEMANTICS,
    THRESHOLDS,
    ambiguous_forced_output,
    catastrophic_valid_failure,
    null_hallucination_outcomes,
    threshold_record,
)

import prepare_hierarchical_safety_data as legacy_data
import extract_hierarchical_safety_features as legacy_features


FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
PHASE2 = REPO / "outputs/runs/thayer_select_recoverability_20260711_191518"
REPLICATION = REPO / "outputs/runs/thayer_select_recoverability_seed_replication_20260711_203115"
FROZEN_HEAD = REPO / "outputs/runs/thayer_select_frozen_head_ablation_20260711_220756"
CORRECTIVE = REPO / "outputs/runs/thayer_select_hierarchical_safety_20260712_001405"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PHASE1 / "manifests/source_split_manifest.csv"
CHECKPOINT = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PHASE1 / "manifests/normalization.json"
ARTIFACT_VERSION = "v2"
DATASETS = {
    "q_training": {"source_partition": "training", "state_counts": {"UNIQUE_VALID": 4000, "NULL": 4000, "AMBIGUOUS": 4000}},
    "q_validation": {"source_partition": "validation", "state_counts": {"UNIQUE_VALID": 668, "NULL": 666, "AMBIGUOUS": 666}},
    "r_training": {"source_partition": "training", "stratum_counts": {"natural": 6000, "low_snr": 2400, "high_overlap": 1800, "equal_flux_similar_size": 900, "confusion_prone": 900}},
    "r_validation": {"source_partition": "validation", "stratum_counts": {"natural": 2000}},
    "natural_calibration": {"source_partition": "calibration", "state_counts": {"UNIQUE_VALID": 2800, "NULL": 800, "AMBIGUOUS": 400}},
}
BASE_SEED = 202607121300


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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def expected_checkpoint_hash() -> str:
    config = json.loads((PHASE1 / "manifests/c_randomized_coordinate_prompt_training_config.json").read_text())
    return str(config["best_checkpoint_sha256"])


def source_split_paths() -> list[Path]:
    return [
        FOUNDATION / "manifests/btk_engineering_source_groups.csv",
        PHASE1 / "manifests/source_split_manifest.csv",
        PHASE2 / "manifests/source_split_manifest.csv",
    ]


def input_code_paths() -> list[Path]:
    return [
        REPO / "src/hierarchical_safety.py",
        REPO / "src/hierarchical_feasibility.py",
        REPO / "scripts/prepare_hierarchical_safety_data.py",
        REPO / "scripts/extract_hierarchical_safety_features.py",
        REPO / "scripts/train_hierarchical_query_gate.py",
        REPO / "scripts/train_hierarchical_risk_heads.py",
        Path(__file__).resolve(),
        REPO / "tests/test_hierarchical_safety.py",
        REPO / "tests/test_hierarchical_query_gate.py",
        REPO / "tests/test_hierarchical_feasibility.py",
    ]


def verify_gates() -> dict:
    observed = sha256_file(CHECKPOINT)
    expected = expected_checkpoint_hash()
    if observed != expected:
        raise RuntimeError("Selected Condition-C checkpoint differs from frozen training record")
    split_hashes = [sha256_file(path) for path in source_split_paths()]
    if len(set(split_hashes)) != 1:
        raise RuntimeError("Source split differs across historical records")
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged["returncode"] != 0 or staged["stdout"].strip():
        raise RuntimeError("Unexpected staged changes exist")
    split = pd.read_csv(SOURCE_SPLIT, dtype=str)
    if "sealed_lockbox" not in set(split.partition):
        raise RuntimeError("Lockbox exclusion cannot be verified")
    lockbox = set(split.loc[split.partition == "sealed_lockbox", "duplicate_group_id"])
    for partition in ("training", "validation", "calibration"):
        manifest = pd.read_csv(PHASE2 / f"manifests/{partition}_scene_manifest.csv", dtype=str)
        if set(manifest.source_a_group) & lockbox or set(manifest.source_b_group) & lockbox:
            raise RuntimeError(f"Lockbox group present in historical {partition} manifest")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return {"checkpoint_sha256": observed, "source_split_sha256": split_hashes[0],
            "lockbox_rows": int((split.partition == "sealed_lockbox").sum()), "mps_probe": 2.0}


def bootstrap(run: Path) -> None:
    if run.exists():
        raise FileExistsError(f"Intended output path exists: {run}")
    if run.parent.resolve() != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_select_hierarchical_feasibility_"):
        raise RuntimeError("Unexpected feasibility run path")
    gates = verify_gates()
    run.mkdir(parents=True, exist_ok=False)
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "preregistration", "manifests", "features", "models", "calibration", "example_grids"):
        (run / name).mkdir(exist_ok=False)
    started = time.time()
    checkpoints = []
    for path in sorted((REPO / "outputs/runs").rglob("*.pth")):
        if run not in path.parents:
            checkpoints.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    inventory = pd.DataFrame(checkpoints)
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", inventory)
    packages = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "blending-toolkit": package_version("blending-toolkit"),
        "GalSim": package_version("GalSim"), "surveycodex": package_version("surveycodex"),
        "astropy": package_version("astropy"), "h5py": package_version("h5py"),
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "mps_probe": 2.0,
    }
    git = {"branch": command(["git", "branch", "--show-current"]), "head": command(["git", "rev-parse", "HEAD"]),
           "status": command(["git", "status", "--short", "--branch"]),
           "staged_index": command(["git", "diff", "--cached", "--name-status"])}
    disk = shutil.disk_usage(REPO)
    code_hashes = [{"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                   for path in input_code_paths()]
    historical = []
    for name, base in (("BTK foundation", FOUNDATION), ("promptability", PHASE1), ("Phase-II", PHASE2),
                       ("seed replication", REPLICATION), ("frozen head", FROZEN_HEAD), ("corrective audit", CORRECTIVE)):
        for path in sorted((base / "checkpoints").glob("*.pth")) if (base / "checkpoints").is_dir() else []:
            historical.append({"campaign": name, "relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    provenance = {
        "campaign_start_unix": started, "campaign_start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "git": git, "packages": packages, "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "source_catalog": {"relative_path": relative(CATALOG), "sha256": sha256_file(CATALOG), "size_bytes": CATALOG.stat().st_size},
        "source_splits": [{"relative_path": relative(path), "sha256": sha256_file(path)} for path in source_split_paths()],
        "selected_frozen_reconstructor": {"relative_path": relative(CHECKPOINT), "expected_sha256": expected_checkpoint_hash(),
                                           "observed_sha256": gates["checkpoint_sha256"]},
        "normalization": {"relative_path": relative(NORMALIZATION), "sha256": sha256_file(NORMALIZATION)},
        "historical_checkpoints": historical, "checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "input_code_hashes": code_hashes, "lockbox_partition_rows": gates["lockbox_rows"],
        "lockbox_scene_or_pixel_accesses": 0, "development_scene_accesses": 0,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)
    environment = f"""# Hierarchical feasibility environment snapshot

- Start: `{provenance['campaign_start_iso']}` (`{started:.6f}` Unix)
- Branch: `{git['branch']['stdout'].strip()}`
- HEAD: `{git['head']['stdout'].strip()}`
- Source catalog SHA-256: `{provenance['source_catalog']['sha256']}`
- Source split SHA-256: `{gates['source_split_sha256']}`
- Condition-C SHA-256: `{gates['checkpoint_sha256']}`
- MPS built / available / execution probe: `{packages['mps_built']}` / `{packages['mps_available']}` / `{packages['mps_probe']}`
- Disk total / free bytes: `{disk.total}` / `{disk.free}`

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
    contract = f"""# Prospective hierarchical-safety feasibility contract

Status: frozen before preregistration, data generation, feature extraction, or fitting.

- This is a prospective train/validation/calibration-only feasibility campaign, not an end-to-end selective-deblending claim.
- The historical hierarchical campaign remains classified FAILURE and is not modified or reinterpreted.
- One reconstruction checkpoint is used everywhere: `{relative(CHECKPOINT)}` SHA-256 `{gates['checkpoint_sha256']}`.
- Reconstruction stays frozen and in evaluation mode; MPS is mandatory for reconstruction and feature extraction.
- CPU is restricted to generation bookkeeping, audits, small heads, statistics, calibration, tables, and figures.
- Query state and reconstruction quality are distinct; no combined recoverability label is permitted.
- Calibration is natural-mixture only, is unavailable for selection, and begins after all heads freeze.
- Development and sealed lockbox scenes are prohibited; access counts start at zero.
- All run artifacts are collision-refusing. No staging, committing, pushing, merging, deleting, or historical overwrite is authorized.
"""
    write_text_fresh(run / "diagnostics/campaign_contract.md", contract)
    config = json.loads((PHASE1 / "manifests/c_randomized_coordinate_prompt_training_config.json").read_text())
    normalization = json.loads(NORMALIZATION.read_text())
    recon_report = f"""# Uniform reconstruction provenance

- Checkpoint: `{relative(CHECKPOINT)}`
- SHA-256: `{gates['checkpoint_sha256']}`
- Architecture: `{config['architecture']}`; 4 inputs (three scientific bands plus Gaussian coordinate prompt); 3-band linear reconstruction head.
- Normalization: training-only per-band division by `{normalization['per_band_scale']}`; no clipping; inverse scaling multiplies output by the same values.
- Prompt: one Gaussian coordinate channel, sigma 2 pixels, generated by `gaussian_prompt_numpy`.
- Selection: `{config['validation_rule']}`.
- Frozen inference: every parameter has `requires_grad=False`, model is in eval mode, and MPS-only deterministic replay is mandatory before extraction.
- R1 reconstruction outputs are prohibited for all partitions.
"""
    write_text_fresh(run / "diagnostics/uniform_reconstruction_provenance.md", recon_report)
    write_csv_fresh(run / "tables/reconstructor_provenance_audit.csv", pd.DataFrame([{
        "checkpoint_path": relative(CHECKPOINT), "checkpoint_sha256": gates["checkpoint_sha256"], "expected_sha256": expected_checkpoint_hash(),
        "architecture": config["architecture"], "normalization_sha256": sha256_file(NORMALIZATION),
        "prompt_implementation_sha256": sha256_file(REPO / "scripts/thayer_select_prompt_ablation_common.py"),
        "inference_implementation_sha256": sha256_file(REPO / "scripts/extract_hierarchical_safety_features.py"),
        "output_scaling": "multiply each output band by frozen training scale", "parameters_frozen": True, "eval_mode_required": True,
        "r1_used": False, "status": "PASS",
    }]))
    write_json_fresh(run / "logs/bootstrap_complete.json", {"status": "PASS", "completed_at_unix": time.time(),
                                                                "development_accessed": False, "lockbox_accessed": False})


def preregister(run: Path) -> None:
    if json.loads((run / "logs/bootstrap_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Bootstrap gate missing")
    if any((run / name).exists() for name in ("logs/data_preparation_complete.json", "logs/feature_extraction_complete.json",
                                               "logs/query_gate_training_complete.json", "logs/risk_head_training_complete.json")):
        raise RuntimeError("Preregistration attempted after data generation or fitting")
    path = run / "preregistration/hierarchical_feasibility_preregistration.md"
    text = f"""# Hierarchical-safety feasibility preregistration

Frozen scope: prospective feasibility only. This document must be hashed before scene generation or any classifier/regressor fit.

## Scientific question and provenance

Under uniform reconstruction provenance and prospective failure-specific definitions, are query validity and valid-source reconstruction risks learnable, stable across five head seeds, and calibratable without score collapse enough to justify a later separately preregistered full-policy campaign?

The only reconstructor is Condition C at `{relative(CHECKPOINT)}`, SHA-256 `{sha256_file(CHECKPOINT)}`. It is a frozen compact prompted U-Net using the training-only per-band normalization in `{relative(NORMALIZATION)}`. Every reconstruction parameter is frozen; eval mode and exact repeated MPS inference are required. R1 outputs are prohibited.

## Source partitions and sizes

Only duplicate-group-isolated `training`, `validation`, and `calibration` source partitions from source-split SHA-256 `{sha256_file(SOURCE_SPLIT)}` may be used. Development, sealed lockbox, and engineering-excluded groups are prohibited.

- Query training: 12,000 scenes, balanced 4,000 UNIQUE_VALID / 4,000 NULL / 4,000 AMBIGUOUS.
- Query validation: 2,000 scenes, approximately balanced 668 / 666 / 666; report both stratified and inverse-weighted natural-mixture summaries.
- Natural calibration: 4,000 scenes, frozen operational mixture 70% UNIQUE_VALID / 20% NULL / 10% AMBIGUOUS; no balancing.
- Valid-risk training: 12,000 UNIQUE_VALID scenes: 6,000 natural, 2,400 low-SNR, 1,800 high-overlap, 900 equal-flux/similar-size, 900 confusion-prone. Store stratum and inverse sampling metadata.
- Valid-risk validation: 2,000 natural UNIQUE_VALID scenes. All naturally occurring calibration UNIQUE_VALID rows (expected 2,800) are reserved for calibration transfer only.

## Query semantics

Source association uses Euclidean source-peak distance. The matching radius is exactly 4.0 pixels = {SEMANTICS.matching_radius_psf:.9f} PSF FWHM for the frozen 4.066666667-pixel reference PSF. No source within the inclusive radius gives NULL. A nearest source within radius is AMBIGUOUS when the second-minus-first distance is at most the inclusive 1.0-pixel ambiguity margin, including exact ties and even when the second source is just outside the radius. Otherwise the nearest source is UNIQUE_VALID. Stable distance ordering is used, but ties never receive truth. Prompt centers must be finite inside inclusive image edges. Alternate-source requests use the same rule. Valid perturbations are capped at 3.5 pixels and must replay as the same unique source.

NULL and AMBIGUOUS receive no requested-source reconstruction truth. AMBIGUOUS_FORCED_OUTPUT is descriptive only and is never fitted.

## Empirical targets and applicability

- QUERY_STATE: UNIQUE_VALID / NULL / AMBIGUOUS, applicable to every row.
- UNIQUE_VALID only: normalized image RMSE; per-band and maximum absolute relative flux risk using training-only per-band floors equal to 0.1% of median absolute training truth flux; centroid risk in pixels and PSF FWHM; binary source confusion (alternate isolated-source MSE lower than requested-source MSE); and CATASTROPHIC_VALID_FAILURE.
- Catastrophe is source confusion, any non-finite valid risk, or image/maximum-flux/centroid risk at least twice its moderate primary limit. Equality is catastrophic.
- NULL only: continuous output RMS-energy divided by blend RMS-energy, continuous maximum per-band absolute-flux exposure divided by blend absolute flux, and binary NULL_HALLUCINATION when either moderate exposure ratio is at least 0.10.
- AMBIGUOUS only: descriptive AMBIGUOUS_FORCED_OUTPUT when a source-like exposed output is deterministically closer in MSE to one isolated source. It assigns no requested-source truth and is not trained.
- All nonapplicable values remain missing with explicit masks. No undefined value may become false or zero.

## Threshold families

Thresholds are fixed for scientific interpretation, not optimized against AUROC, AUPRC, or coverage:

```json
{json.dumps({name: threshold_record(name) for name in THRESHOLDS}, indent=2, sort_keys=True)}
```

Continuous values are always preserved. Strict and permissive thresholds are sensitivity analyses; moderate is primary.

## Deployable features

F_GLOBAL is pooled bottleneck; F_PROMPT_LOCAL is Gaussian-weighted pooling at encoder 60/30/15-pixel scales; F_RECON contains predicted per-band flux, centroid-to-prompt offsets, concentration, RMS energy, and local input/output contrast; F_COMBINED concatenates these. Ground truth, oracle errors, true SNR/separation/flux ratio, generator difficulty, IDs, and labels are prohibited as inputs. Feature extraction must be deterministic on MPS.

## Fitting and validation selection

Training-only standardization is mandatory. CPU heads compare multinomial logistic regression and a 64-hidden-unit ReLU MLP for query state; continuous heads compare linear Huber/median plus linear 0.90-quantile outputs and the corresponding one-hidden-layer MLP; binary confusion and catastrophic heads compare balanced logistic regression and a 64-hidden-unit MLP. Candidate feature/head families use validation only. Selected heads run seeds 2026071201-1205 (query) and 2026071211-1215 (risk). Calibration never selects models.

Query metrics: macro F1, per-class precision/recall, one-vs-rest AUROC/AUPRC, NULL and AMBIGUOUS false-accept rates, UNIQUE_VALID false-reject rate, confusion matrices, seed variability, SNR/overlap strata. Gate Q requires every seed NULL and AMBIGUOUS recall >0.50, every seed mean P(UNIQUE|AMBIGUOUS) below P(UNIQUE|UNIQUE), ensemble macro F1 >1/3, seed SD <=0.05, and nonconstant scores. Failure stops later policy work; it does not authorize development.

Continuous metrics: MAE, median absolute error, Spearman correlation, pinball loss, empirical quantile coverage, top-decile recall and precision, seed variability, SNR/overlap strata. Binary metrics: AUROC/AUPRC, prevalence, top-risk recall/tail precision. Catastrophic feasibility requires mean AUROC >=0.704 (a preregistered material +0.05 over 0.654), mean AUPRC >=1.25 times validation prevalence, AUROC seed SD <=0.05, and natural-calibration AUROC no more than 0.10 below validation. Primary image and flux rankability require mean Spearman >=0.30; centroid and confusion are classified separately. A useless ranker (Spearman <0.10 or binary AUROC <=0.55) is not calibrated.

## Calibration

After all model hashes freeze, use only the natural calibration set. Compare scalar temperature scaling and vector scaling for query probabilities; select lower calibration negative log likelihood and report Brier/ECE plus unique score count, tie fraction, and largest plateau. For useful risk regressors, compare split-conformal 90% upper residual calibration and 0.90 quantile residual correction in log1p space; report empirical bound coverage, mean/median width, subgroup coverage, natural transfer, ties, and plateaus. Isotonic is descriptive only. No operational hierarchical accept/abstain policy or threshold is selected.

## Feasibility decisions and stopping

Each component is PASS/PARTIAL/FAIL. Overall SUCCESS requires Gate Q PASS, catastrophic conditions above, image and flux mean Spearman >=0.30, calibration score unique count >=100 with tie fraction <0.50 and finite nonzero interval widths, seed stability, and uniform provenance. PARTIAL means query state works but valid-only tail risk remains weak. FAILURE follows query failure, provenance failure, unlearnable valid risk, or calibration collapse. Gates do not change after results.

Stop before fitting on any provenance/applicability mismatch. Stop before risk fitting if Gate Q fails. Never generate or inspect development or lockbox scenes; never construct a full policy. Prohibited analyses include generator variables as deployable inputs, historical mixed labels, R1 outputs, calibration-based selection, threshold optimization for ranking/coverage, backbone fine-tuning, and retrospective repair.
"""
    write_text_fresh(path, text)
    created = time.time()
    metadata = {
        "relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "created_at_unix": created, "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
        "git_head": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "code_hashes": [{"relative_path": relative(code), "sha256": sha256_file(code)} for code in input_code_paths()],
        "fitting_artifacts_existing": [], "development_accessed": False, "lockbox_accessed": False,
    }
    write_json_fresh(run / "preregistration/hierarchical_feasibility_preregistration.sha256.json", metadata)
    write_text_fresh(run / "preregistration/hierarchical_feasibility_preregistration.sha256", metadata["sha256"] + "\n")
    write_json_fresh(run / "logs/preregistration_complete.json", {"status": "PASS", "sha256": metadata["sha256"],
                                                                     "timestamp_unix": created, "fit_started": False})


def require_preregistration(run: Path) -> dict:
    marker = json.loads((run / "logs/preregistration_complete.json").read_text())
    metadata = json.loads((run / "preregistration/hierarchical_feasibility_preregistration.sha256.json").read_text())
    path = run / "preregistration/hierarchical_feasibility_preregistration.md"
    if marker["status"] != "PASS" or sha256_file(path) != metadata["sha256"] or marker["sha256"] != metadata["sha256"]:
        raise RuntimeError("Preregistration integrity gate failed")
    return metadata


def generate_data(run: Path) -> None:
    require_preregistration(run)
    if (run / "logs/data_preparation_complete.json").exists():
        raise FileExistsError("Data already generated")
    split = pd.read_csv(SOURCE_SPLIT)
    forbidden = set(split.loc[split.partition.isin(("development_test", "sealed_lockbox", "engineering_excluded")), "duplicate_group_id"])
    table = Table.read(CATALOG, format="fits")
    catalog, observed_catalog_hash = legacy_data.load_catsim_catalog(CATALOG)
    if observed_catalog_hash != sha256_file(CATALOG):
        raise RuntimeError("Loaded source catalog hash mismatch")
    legacy_data.DATASETS = DATASETS
    legacy_data.ARTIFACT_VERSION = ARTIFACT_VERSION
    legacy_data.BASE_SEED = BASE_SEED
    legacy_data.SEMANTICS = SEMANTICS
    replay_rows = []
    inventory = []
    started = time.time()
    for dataset, spec in DATASETS.items():
        pool = split[(split.partition == spec["source_partition"]) & (split.engineering_excluded == 0)].copy()
        if set(pool.duplicate_group_id) & forbidden:
            raise RuntimeError(f"Forbidden source group in {dataset}")
        definitions = legacy_data.build_definitions(dataset, spec, pool, table)
        write_csv_fresh(run / f"manifests/v2_{dataset}_scene_definitions.csv", definitions)
        manifest = legacy_data.render_dataset(run, dataset, definitions, catalog, table)
        write_csv_fresh(run / f"manifests/v2_{dataset}_scene_manifest.csv", manifest)
        replay_rows.extend(legacy_data.replay_audit(run, dataset, manifest, catalog))
        inventory.append({
            "dataset": dataset, "source_partition": spec["source_partition"], "scenes": len(manifest),
            "unique_scene_ids": manifest.scene_id.nunique(), "query_state_counts": json.dumps(manifest.query_state.value_counts().sort_index().to_dict(), sort_keys=True),
            "sampling_stratum_counts": json.dumps(manifest.sampling_stratum.value_counts().sort_index().to_dict(), sort_keys=True),
            "manifest_sha256": sha256_file(run / f"manifests/v2_{dataset}_scene_manifest.csv"),
            "hdf5_sha256": sha256_file(run / f"manifests/v2_{dataset}_scenes.h5"), "artifact_version": ARTIFACT_VERSION,
        })
    replay = pd.DataFrame(replay_rows)
    write_csv_fresh(run / "tables/manifest_replay_audit.csv", replay)
    write_csv_fresh(run / "tables/fresh_dataset_inventory.csv", pd.DataFrame(inventory))
    if not (replay.status == "PASS").all():
        raise RuntimeError("Deterministic manifest replay failed")
    write_json_fresh(run / "logs/data_preparation_complete.json", {
        "status": "PASS", "datasets": {name: int(sum(spec.get("state_counts", spec.get("stratum_counts")).values())) for name, spec in DATASETS.items()},
        "total_scenes": int(sum(sum(spec.get("state_counts", spec.get("stratum_counts")).values()) for spec in DATASETS.values())),
        "runtime_seconds": time.time() - started, "source_split_sha256": sha256_file(SOURCE_SPLIT),
        "preregistration_sha256": require_preregistration(run)["sha256"], "preregistration_predates_data": True,
        "development_accessed": False, "lockbox_accessed": False,
    })


def extract_features(run: Path) -> None:
    prereg = require_preregistration(run)
    if json.loads((run / "logs/data_preparation_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Data gate missing")
    if (run / "logs/feature_extraction_complete.json").exists():
        raise FileExistsError("Features already extracted")
    legacy_features.DATASETS = tuple(DATASETS)
    started = time.time()
    checkpoint_before = sha256_file(CHECKPOINT)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    floors = legacy_features.fit_flux_floors(run)
    write_json_fresh(run / "manifests/risk_flux_floors.json", {
        "fit_partition": "r_training UNIQUE_VALID only", "fraction_of_median_absolute_band_flux": 0.001,
        "bands": ["g", "r", "z"], "floor_by_band": floors.tolist(), "calibration_used": False,
        "development_used": False, "lockbox_used": False,
    })
    model = legacy_features.load_model()
    audit = legacy_features.extraction_audit(model, run, scales, checkpoint_before)
    if (audit["trainable_reconstruction_parameters"] or audit["model_training_flag"] or audit["parameter_gradient_count"]
            or audit["output_requires_grad"] or not audit["deterministic_exact"] or audit["device"] != "mps:0"
            or audit["checkpoint_sha256_before"] != audit["checkpoint_sha256_after"]):
        raise RuntimeError(f"Frozen reconstructor audit failed: {audit}")
    write_json_fresh(run / "diagnostics/frozen_feature_extraction_audit.json", audit)
    rows = [legacy_features.extract_dataset(run, dataset, model, scales, floors) for dataset in DATASETS]
    write_csv_fresh(run / "tables/frozen_feature_inventory.csv", pd.DataFrame(rows))
    if sha256_file(CHECKPOINT) != checkpoint_before:
        raise RuntimeError("Reconstructor checkpoint changed during extraction")
    # Correct the literal NULL class before any fitting; v2 remains preserved.
    corrections = []
    for dataset in DATASETS:
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
        sample = pd.read_csv(run / f"features/v2_{dataset}_samples.csv", keep_default_na=False, low_memory=False)
        if manifest.scene_id.tolist() != sample.scene_id.tolist():
            raise RuntimeError(f"Sample alignment failed: {dataset}")
        sample["query_state"] = manifest.query_state.astype(str)
        sample["applicable_valid_risk"] = (sample.query_state == "UNIQUE_VALID").astype(int)
        write_csv_fresh(run / f"features/v3_{dataset}_samples.csv", sample)
        corrections.append({"dataset": dataset, "rows": len(sample), "supersedes": f"features/v2_{dataset}_samples.csv",
                            "blank_query_states_after": int((sample.query_state == "").sum()), "applicability_mismatches": 0,
                            "sha256": sha256_file(run / f"features/v3_{dataset}_samples.csv")})
    write_csv_fresh(run / "tables/sample_metadata_correction_inventory.csv", pd.DataFrame(corrections))
    write_json_fresh(run / "logs/feature_extraction_complete.json", {
        "status": "PASS", "device": "mps", "cpu_fallback": False, "datasets": list(DATASETS),
        "total_samples": int(sum(row["samples"] for row in rows)), "runtime_seconds": time.time() - started,
        "condition_c_checkpoint_sha256": checkpoint_before, "preregistration_sha256": prereg["sha256"],
        "development_accessed": False, "lockbox_accessed": False,
    })


def audit_labels(run: Path) -> None:
    prereg = require_preregistration(run)
    if json.loads((run / "logs/feature_extraction_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Feature gate missing")
    if (run / "logs/prospective_label_audit_complete.json").exists():
        raise FileExistsError("Label audit already exists")
    formula_hash = sha256_file(REPO / "src/hierarchical_feasibility.py")
    recon_hash = sha256_file(CHECKPOINT)
    prevalence_rows = []
    applicability_rows = []
    provenance_rows = []
    overlap_rows = []
    all_scene_ids = set()
    for dataset in DATASETS:
        manifest = pd.read_csv(run / f"manifests/v2_{dataset}_scene_manifest.csv", keep_default_na=False, low_memory=False)
        sample = pd.read_csv(run / f"features/v3_{dataset}_samples.csv", keep_default_na=False, low_memory=False)
        if manifest.scene_id.tolist() != sample.scene_id.tolist():
            raise RuntimeError(f"Scene/sample misalignment: {dataset}")
        duplicate_ids = all_scene_ids & set(manifest.scene_id)
        if duplicate_ids:
            raise RuntimeError(f"Cross-dataset duplicate scene IDs: {len(duplicate_ids)}")
        all_scene_ids.update(manifest.scene_id)
        recon_path = run / f"features/v2_{dataset}_frozen_reconstructions.h5"
        scene_path = run / f"manifests/v2_{dataset}_scenes.h5"
        rows = []
        with h5py.File(recon_path, "r") as predictions, h5py.File(scene_path, "r") as scenes:
            if predictions.attrs["condition_c_checkpoint_sha256"] != recon_hash:
                raise RuntimeError(f"Reconstructor provenance mismatch: {dataset}")
            for index, (mrow, srow) in enumerate(zip(manifest.itertuples(index=False), sample.itertuples(index=False))):
                state = str(mrow.query_state)
                unique = state == "UNIQUE_VALID"
                null = state == "NULL"
                ambiguous = state == "AMBIGUOUS"
                if sum((unique, null, ambiguous)) != 1:
                    raise RuntimeError(f"Invalid query state {state}")
                prediction = np.asarray(predictions["reconstruction"][index], dtype=np.float32)
                blend = np.asarray(scenes["blend"][index], dtype=np.float32)
                isolated = np.asarray(scenes["isolated"][index], dtype=np.float32)
                record = {**mrow._asdict(),
                    "reconstructor_checkpoint_sha256": recon_hash, "frozen_reconstruction_sha256": srow.frozen_reconstruction_sha256,
                    "label_formula_sha256": formula_hash, "query_state_applicable": 1,
                    "image_risk_applicable": int(unique), "flux_risk_applicable": int(unique), "centroid_risk_applicable": int(unique),
                    "confusion_applicable": int(unique), "catastrophic_valid_failure_applicable": int(unique),
                    "null_hallucination_applicable": int(null), "ambiguous_forced_output_applicable": int(ambiguous),
                    "image_risk": float(srow.image_risk) if unique else np.nan,
                    "flux_risk_g": float(srow.flux_risk_g) if unique else np.nan,
                    "flux_risk_r": float(srow.flux_risk_r) if unique else np.nan,
                    "flux_risk_z": float(srow.flux_risk_z) if unique else np.nan,
                    "flux_risk_max": float(srow.flux_risk_max) if unique else np.nan,
                    "centroid_risk_pixels": float(srow.centroid_risk_pixels) if unique else np.nan,
                    "centroid_risk_psf": float(srow.centroid_risk_psf) if unique else np.nan,
                    "confusion": int(srow.confusion_risk) if unique else np.nan,
                    "catastrophic_valid_failure": int(catastrophic_valid_failure(
                        image_risk=float(srow.image_risk), flux_risk_max=float(srow.flux_risk_max),
                        centroid_risk_pixels=float(srow.centroid_risk_pixels), confusion=bool(srow.confusion_risk))) if unique else np.nan,
                    "null_output_energy_ratio": np.nan, "null_absolute_flux_ratio": np.nan, "null_hallucination": np.nan,
                    "ambiguous_forced_output": np.nan, "ambiguous_exposed_source_index": np.nan,
                    "ambiguous_exposed_source_mse_margin": np.nan,
                }
                if null:
                    outcome = null_hallucination_outcomes(prediction, blend)
                    record.update({"null_output_energy_ratio": outcome["null_output_energy_ratio"],
                                   "null_absolute_flux_ratio": outcome["null_absolute_flux_ratio"],
                                   "null_hallucination": int(outcome["null_hallucination"])})
                if ambiguous:
                    outcome = ambiguous_forced_output(prediction, isolated, blend)
                    record.update({"ambiguous_forced_output": int(outcome["ambiguous_forced_output"]),
                                   "ambiguous_exposed_source_index": outcome["exposed_source_index"],
                                   "ambiguous_exposed_source_mse_margin": outcome["exposed_source_mse_margin"]})
                rows.append(record)
        outcomes = pd.DataFrame(rows)
        output = run / f"tables/prospective_outcomes_{dataset}.csv"
        write_csv_fresh(output, outcomes)
        # Authoritative head-training samples retain only deployable routing and targets.
        v4 = sample.copy()
        v4["query_state"] = outcomes.query_state
        v4["reconstructor_checkpoint_sha256"] = recon_hash
        v4["label_formula_sha256"] = formula_hash
        for column in ("image_risk", "flux_risk_g", "flux_risk_r", "flux_risk_z", "flux_risk_max", "centroid_risk_pixels", "centroid_risk_psf"):
            v4[column] = outcomes[column]
        v4["confusion_risk"] = outcomes.confusion
        v4["catastrophic_valid_failure"] = outcomes.catastrophic_valid_failure
        for target, raw in (("image", "image_risk"), ("flux", "flux_risk_max"), ("centroid", "centroid_risk_pixels")):
            v4[f"{target}_target_log1p"] = np.where(outcomes[f"{raw.split('_risk')[0]}_risk_applicable"] == 1, np.log1p(outcomes[raw]), np.nan)
        write_csv_fresh(run / f"features/v4_{dataset}_samples.csv", v4)
        for state, group in outcomes.groupby("query_state", dropna=False):
            prevalence_rows.append({"dataset": dataset, "source_partition": group.source_partition.iloc[0], "query_state": state,
                                    "rows": len(group), "fraction": len(group) / len(outcomes),
                                    "catastrophic_valid_prevalence": float(group.catastrophic_valid_failure.mean()) if state == "UNIQUE_VALID" else np.nan,
                                    "confusion_prevalence": float(group.confusion.mean()) if state == "UNIQUE_VALID" else np.nan,
                                    "null_hallucination_prevalence": float(group.null_hallucination.mean()) if state == "NULL" else np.nan,
                                    "ambiguous_forced_output_prevalence": float(group.ambiguous_forced_output.mean()) if state == "AMBIGUOUS" else np.nan})
        for target, mask in (("QUERY_STATE", "query_state_applicable"), ("IMAGE_RISK", "image_risk_applicable"),
                             ("FLUX_RISK", "flux_risk_applicable"), ("CENTROID_RISK", "centroid_risk_applicable"),
                             ("CONFUSION", "confusion_applicable"), ("CATASTROPHIC_VALID_FAILURE", "catastrophic_valid_failure_applicable"),
                             ("NULL_HALLUCINATION", "null_hallucination_applicable"), ("AMBIGUOUS_FORCED_OUTPUT", "ambiguous_forced_output_applicable")):
            applicable = outcomes[mask].astype(bool)
            applicability_rows.append({"dataset": dataset, "target": target, "rows": len(outcomes),
                                       "applicable_rows": int(applicable.sum()), "not_applicable_rows": int((~applicable).sum()),
                                       "undefined_in_applicable_rows": 0 if target == "QUERY_STATE" else int(outcomes.loc[applicable, {
                                           "IMAGE_RISK":"image_risk", "FLUX_RISK":"flux_risk_max", "CENTROID_RISK":"centroid_risk_pixels",
                                           "CONFUSION":"confusion", "CATASTROPHIC_VALID_FAILURE":"catastrophic_valid_failure",
                                           "NULL_HALLUCINATION":"null_hallucination", "AMBIGUOUS_FORCED_OUTPUT":"ambiguous_forced_output"}.get(target, "query_state")].isna().sum()),
                                       "defined_in_not_applicable_rows": 0 if target == "QUERY_STATE" else int(outcomes.loc[~applicable, {
                                           "IMAGE_RISK":"image_risk", "FLUX_RISK":"flux_risk_max", "CENTROID_RISK":"centroid_risk_pixels",
                                           "CONFUSION":"confusion", "CATASTROPHIC_VALID_FAILURE":"catastrophic_valid_failure",
                                           "NULL_HALLUCINATION":"null_hallucination", "AMBIGUOUS_FORCED_OUTPUT":"ambiguous_forced_output"}.get(target, "query_state")].notna().sum())})
        provenance_rows.append({"dataset": dataset, "source_partition": outcomes.source_partition.iloc[0], "rows": len(outcomes),
                                "unique_reconstructor_hashes": outcomes.reconstructor_checkpoint_sha256.nunique(),
                                "reconstructor_sha256": outcomes.reconstructor_checkpoint_sha256.iloc[0],
                                "unique_formula_hashes": outcomes.label_formula_sha256.nunique(),
                                "formula_sha256": outcomes.label_formula_sha256.iloc[0], "status": "PASS"})
        valid = outcomes[outcomes.query_state == "UNIQUE_VALID"].copy()
        flags = {
            "image_moderate_failure": valid.image_risk >= THRESHOLDS["moderate"].risks.image,
            "flux_moderate_failure": valid.flux_risk_max >= THRESHOLDS["moderate"].risks.flux,
            "centroid_moderate_failure": valid.centroid_risk_pixels >= THRESHOLDS["moderate"].risks.centroid_pixels,
            "confusion": valid.confusion == 1, "catastrophic_valid_failure": valid.catastrophic_valid_failure == 1,
        }
        for left, left_values in flags.items():
            for right, right_values in flags.items():
                overlap_rows.append({"dataset": dataset, "left": left, "right": right,
                                     "left_positive": int(left_values.sum()), "right_positive": int(right_values.sum()),
                                     "intersection": int((left_values & right_values).sum()),
                                     "union": int((left_values | right_values).sum())})
    applicability = pd.DataFrame(applicability_rows)
    provenance = pd.DataFrame(provenance_rows)
    if (applicability.undefined_in_applicable_rows > 0).any() or (applicability.defined_in_not_applicable_rows > 0).any():
        raise RuntimeError("Applicability audit failed")
    if (provenance.unique_reconstructor_hashes != 1).any() or provenance.reconstructor_sha256.nunique() != 1 or (provenance.unique_formula_hashes != 1).any():
        raise RuntimeError("Label provenance audit failed")
    write_csv_fresh(run / "tables/label_provenance_audit.csv", provenance)
    write_csv_fresh(run / "tables/label_prevalence_by_partition.csv", pd.DataFrame(prevalence_rows))
    write_csv_fresh(run / "tables/label_applicability_matrix.csv", applicability)
    write_csv_fresh(run / "tables/failure_overlap_matrix.csv", pd.DataFrame(overlap_rows))
    report = f"""# Prospective label audit

Status: **PASS before fitting**.

- Rows audited: {sum(DATASETS[name].get('state_counts', DATASETS[name].get('stratum_counts')).values() for name in DATASETS):,}.
- Reconstructor SHA-256 across every partition: `{recon_hash}` (one unique value).
- Label implementation SHA-256 across every partition: `{formula_hash}` (one unique value).
- NULL and AMBIGUOUS valid-risk values: explicitly not applicable; zero values were defined outside applicability.
- UNIQUE_VALID requested truth: present on every applicable row.
- Undefined-to-negative/zero coercions in authoritative v4 targets: zero.
- Scene IDs: globally unique across all five datasets.
- Calibration mixture: frozen natural 70/20/10 distribution; no balancing and no model selection.
- Preregistration SHA-256 `{prereg['sha256']}` existed before data, inference, or fitting.
- Development/lockbox access: zero / zero.
"""
    write_text_fresh(run / "diagnostics/prospective_label_audit.md", report)
    write_json_fresh(run / "logs/prospective_label_audit_complete.json", {"status": "PASS", "completed_at_unix": time.time(),
                                                                           "fitting_started": False, "reconstructor_sha256": recon_hash,
                                                                           "formula_sha256": formula_hash, "development_accessed": False,
                                                                           "lockbox_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("bootstrap", "preregister", "data", "features", "label-audit"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    {"bootstrap": bootstrap, "preregister": preregister, "data": generate_data,
     "features": extract_features, "label-audit": audit_labels}[args.stage](run)
    print(relative(run))


if __name__ == "__main__":
    main()
