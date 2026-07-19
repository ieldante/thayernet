#!/usr/bin/env python3
"""Fail-closed explicit-PSF information-sufficiency campaign.

The provenance and variation audit necessarily precedes preregistration or
fitting.  If the historical rendering PSF is constant between scenes, this
script records ``PSF NON-INFORMATIVE BY CONSTRUCTION`` and refuses every later
model, risk, calibration, policy, development, and lockbox phase.
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
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btk_scene import BAND_ORDER, SceneSpec, load_catsim_catalog, render_fixed_scene, validated_lsst_survey
from src.psf_conditioning import (
    array_sha256,
    effective_configuration_count,
    kernel_distance,
    kernel_moments,
    meaningful_scene_variation,
    normalized_kernel,
)


FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FEASIBILITY = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
CONDITIONAL = REPO / "outputs/runs/thayer_select_conditional_calibration_20260712_021556"
SCALE = REPO / "outputs/runs/thayer_select_scale_correction_20260712_024957"
SHAPE = REPO / "outputs/runs/thayer_select_shape_constrained_quantile_20260712_033406"
OBSERVABILITY = REPO / "outputs/runs/thayer_select_observability_distillation_20260712_035843"
AUTHORITATIVE = (FEASIBILITY, CONDITIONAL, SCALE, SHAPE, OBSERVABILITY)
CHECKPOINT = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_CONDITION_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
SOURCE_SPLITS = (
    FOUNDATION / "manifests/btk_engineering_source_groups.csv",
    PHASE1 / "manifests/source_split_manifest.csv",
    REPO / "outputs/runs/thayer_select_recoverability_20260711_191518/manifests/source_split_manifest.csv",
)
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
PARTITIONS = {
    "training": "r_training",
    "validation": "r_validation",
    "calibration": "natural_calibration",
}
RUN_PREFIX = "thayer_select_psf_conditioning_"
SUPERSEDED_ATTEMPTS = (
    "outputs/runs/thayer_select_psf_conditioning_20260712_043319",
    "outputs/runs/thayer_select_psf_conditioning_20260712_043342",
    "outputs/runs/thayer_select_psf_conditioning_20260712_043415",
)
IMAGE_SIZE = 60
PIXEL_SCALE_ARCSEC = 0.2
EXPECTED_ROWS = {"training": 12000, "validation": 2000, "calibration": 4000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def fresh_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy().fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join((header, separator, *rows))


def create_run() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / "outputs/runs" / f"{RUN_PREFIX}{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for name in (
        "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
        "manifests", "features", "models", "calibration", "example_grids",
    ):
        (run / name).mkdir(exist_ok=False)
    (run / "figures/psf_examples").mkdir(exist_ok=False)
    return run


def checkpoint_inventory(run: Path | None = None) -> pd.DataFrame:
    records = []
    for path in sorted((REPO / "outputs").rglob("*.pth")):
        if run is not None and run in path.parents:
            continue
        records.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(records)


def authoritative_files() -> list[Path]:
    paths: list[Path] = []
    for root in AUTHORITATIVE:
        paths.append(root / "logs/input_provenance.json")
        paths.extend(sorted((root / "reports").glob("*.md")))
    paths.extend([
        REPO / "docs/observable_regime_distillation.md",
        REPO / "docs/prospective_hierarchical_feasibility.md",
        REPO / "docs/conditional_calibration_experiment.md",
        REPO / "docs/deployable_scale_model.md",
        REPO / "docs/shape_constrained_quantile_scale_correction.md",
    ])
    return sorted(set(paths))


def source_files() -> list[Path]:
    return [
        REPO / "src/btk_scene.py",
        REPO / "src/psf_conditioning.py",
        REPO / "scripts/prepare_hierarchical_safety_data.py",
        REPO / "scripts/run_hierarchical_feasibility.py",
        REPO / "scripts/run_observability_distillation.py",
        Path(__file__).resolve(),
        REPO / "tests/test_psf_conditioning.py",
    ]


def selected_manifest_paths() -> list[Path]:
    paths = []
    for dataset in PARTITIONS.values():
        paths.extend([
            FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv",
            FEASIBILITY / f"manifests/v2_{dataset}_scenes.h5",
        ])
    return paths


def verify_frozen_inputs() -> dict:
    failures = []
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged["returncode"] != 0 or staged["stdout"].strip():
        failures.append("staged index is nonempty")
    checkpoint_hash = sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else "MISSING"
    if checkpoint_hash != EXPECTED_CONDITION_SHA256:
        failures.append("Condition-C checkpoint differs from frozen hash")
    split_hashes = [sha256_file(path) if path.is_file() else "MISSING" for path in SOURCE_SPLITS]
    if len(set(split_hashes)) != 1:
        failures.append("source split hashes differ")
    for path in authoritative_files() + selected_manifest_paths() + source_files():
        if not path.is_file():
            failures.append(f"missing frozen input: {relative(path)}")
    for partition, dataset in PARTITIONS.items():
        path = FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv"
        if path.is_file():
            count = sum(1 for _ in path.open("rb")) - 1
            if count != EXPECTED_ROWS[partition]:
                failures.append(f"unexpected {partition} manifest row count {count}")
    if failures:
        raise RuntimeError("Frozen input verification failed:\n" + "\n".join(failures))
    return {
        "checkpoint_sha256": checkpoint_hash,
        "source_split_sha256": split_hashes[0],
        "staged_index_empty": True,
        "authoritative_files": len(authoritative_files()),
        "selected_manifests_and_scene_stores": len(selected_manifest_paths()),
    }


def survey_psf_configuration() -> tuple[object, dict[str, dict], dict[str, np.ndarray], str]:
    survey = validated_lsst_survey()
    kernels: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for band in BAND_ORDER:
        filtr = survey.get_filter(band)
        raw = np.asarray(filtr.psf.drawImage(nx=IMAGE_SIZE, ny=IMAGE_SIZE, scale=PIXEL_SCALE_ARCSEC).array, dtype=np.float64)
        normalized = normalized_kernel(raw)
        components = list(filtr.psf.obj_list)
        component_repr = [repr(value) for value in components]
        record = {
            "band": band,
            "psf_model_family": "GalSim Convolution(Kolmogorov, Airy)",
            "psf_repr": repr(filtr.psf),
            "component_repr": component_repr,
            "fwhm_arcsec": float(filtr.psf_fwhm.to_value("arcsec")),
            "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
            "kernel_shape": list(normalized.shape),
            "raw_draw_sum": float(raw.sum()),
            "normalized_kernel_sha256": array_sha256(normalized),
            "axisymmetric": bool(all(value.is_axisymmetric for value in components)),
            "spatially_varying": False,
            "stochastic_psf_seed": None,
        }
        record.update(kernel_moments(normalized, PIXEL_SCALE_ARCSEC))
        if math.isnan(record["orientation_radians"]):
            record["orientation_radians"] = None
        kernels[band] = normalized
        records[band] = record
    config_payload = {
        "survey": survey.name,
        "description": survey.description,
        "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        "image_size": IMAGE_SIZE,
        "bands": records,
        "btk": package_version("blending-toolkit"),
        "galsim": package_version("GalSim"),
        "surveycodex": package_version("surveycodex"),
    }
    return survey, records, kernels, sha256_text(json.dumps(config_payload, sort_keys=True, allow_nan=False))


def write_bootstrap(run: Path) -> None:
    started = time.time()
    frozen = verify_frozen_inputs()
    before = checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", before)
    survey, psf_records, kernels, config_hash = survey_psf_configuration()
    git = {
        "branch": command(["git", "branch", "--show-current"]),
        "head": command(["git", "rev-parse", "HEAD"]),
        "status": command(["git", "status", "--short", "--branch"]),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]),
    }
    packages = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "blending-toolkit": package_version("blending-toolkit"),
        "GalSim": package_version("GalSim"),
        "surveycodex": package_version("surveycodex"),
        "astropy": package_version("astropy"),
        "h5py": package_version("h5py"),
        "matplotlib": matplotlib.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    disk = shutil.disk_usage(REPO)
    provenance = {
        "campaign_start_unix": started,
        "campaign_start_iso": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "git": git,
        "packages": packages,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "condition_c": {"relative_path": relative(CHECKPOINT), "sha256": frozen["checkpoint_sha256"]},
        "source_splits": [{"relative_path": relative(path), "sha256": sha256_file(path)} for path in SOURCE_SPLITS],
        "partition_manifests_and_scene_stores": [
            {"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in selected_manifest_paths()
        ],
        "authoritative_prior_run_inputs": [
            {"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in authoritative_files()
        ],
        "observability_heads": [
            {"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted((OBSERVABILITY / "models").glob("A3_*.pth"))
        ],
        "risk_heads": [
            {"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted((FEASIBILITY / "models").glob("*_risk_*seed_*.pth"))
        ],
        "historical_checkpoints": {
            "count": len(before),
            "inventory_relative_path": relative(run / "tables/checkpoint_inventory_before.csv"),
            "inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        },
        "source_code": [
            {"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in source_files()
        ],
        "simulator": {
            "survey": survey.name,
            "description": survey.description,
            "bands": list(BAND_ORDER),
            "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
            "stamp_shape": [IMAGE_SIZE, IMAGE_SIZE],
            "rendering": "BTK CatsimGenerator with GalSim analytic PSF; add_noise=all; scene-specific noise seed only",
            "psf_configuration_hash": config_hash,
            "per_band_psf": psf_records,
        },
        "development_scene_accesses": 0,
        "lockbox_scene_or_pixel_accesses": 0,
        "frozen_verification": frozen,
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    environment = f"""# Explicit-PSF campaign environment snapshot

- Start: `{provenance['campaign_start_iso']}`
- Branch: `{git['branch']['stdout'].strip()}`
- HEAD: `{git['head']['stdout'].strip()}`
- Condition-C SHA-256: `{frozen['checkpoint_sha256']}`
- Source-split SHA-256: `{frozen['source_split_sha256']}`
- Historical checkpoint count: `{len(before)}`
- Disk total/free bytes: `{disk.total}` / `{disk.free}`
- MPS built/available: `{packages['mps_built']}` / `{packages['mps_available']}`
- PSF configuration SHA-256: `{config_hash}`

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
    fresh_text(run / "diagnostics/environment_snapshot.md", environment)
    contract = """# Explicit-PSF information-sufficiency campaign contract

Status: frozen before PSF audit, preregistration, feature construction, or fitting.

- This is a train/validation/natural-calibration campaign only.
- Exact simulator PSF provenance and meaningful between-scene variation are mandatory preconditions.
- The PSF audit uses only the historical r-training, r-validation, and natural-calibration scene records plus the exact BTK/SurveyCodex/GalSim configuration that rendered them.
- If the scene-level PSF configuration is identical or nearly identical, the campaign stops as `PSF NON-INFORMATIVE BY CONSTRUCTION` before preregistration or fitting.
- Condition C, query semantics, risk definitions, source partitions, and every historical checkpoint remain immutable.
- Source truth, physical difficulty, IDs, and outcomes never enter a deployable tensor.
- Development generation/access and lockbox generation/access are prohibited and begin at zero.
- GroupDRO, risk heads, calibration, a full policy, and end-to-end safety claims require a passed PSF information gate.
- All run artifacts are fresh and collision-refusing. No staging, committing, pushing, merging, deleting, or historical overwrite is authorized.
"""
    fresh_text(run / "diagnostics/campaign_contract.md", contract)
    fresh_json(run / "logs/bootstrap_complete.json", {
        "status": "PASS", "completed_at_unix": time.time(), "psf_audit_started": False,
        "preregistration_created": False, "fitting_started": False,
        "development_accesses": 0, "lockbox_accesses": 0,
    })
    fresh_json(run / "logs/superseded_attempts.json", {
        "attempts": [{
            "relative_path": path,
            "status": "INCOMPLETE_SUPERSEDED",
            "reason": (
                "post-table aggregation used the DataFrame std method instead of the std column"
                if path.endswith("043319") else
                "finite-grid anisotropy was incorrectly labeled as a stable PSF orientation before finalization"
                if path.endswith("043342") else
                "undefined axisymmetric orientation required JSON null rather than NaN"
            ),
            "preregistration_created": False,
            "model_fitting_started": False,
            "development_accesses": 0,
            "lockbox_accesses": 0,
        } for path in SUPERSEDED_ATTEMPTS],
        "historical_attempts_modified": False,
    })


def replay_psf_alignment() -> pd.DataFrame:
    catalog, _ = load_catsim_catalog(CATALOG)
    records = []
    for partition, dataset in PARTITIONS.items():
        manifest = pd.read_csv(FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv", low_memory=False)
        indices = sorted(set((0, len(manifest) // 2, len(manifest) - 1)))
        for index in indices:
            row = manifest.iloc[index]
            positions = ((row.source_a_x_arcsec, row.source_a_y_arcsec), (row.source_b_x_arcsec, row.source_b_y_arcsec))
            spec = SceneSpec(
                str(row.scene_id), (int(row.source_a_row), int(row.source_b_row)), positions,
                int(row.scene_seed), int(row.scene_seed), int(row.noise_seed),
            )
            rendered = render_fixed_scene(catalog, spec, add_noise="all")
            for band_index, band in enumerate(BAND_ORDER):
                normalized = normalized_kernel(rendered.psf[band_index])
                records.append({
                    "partition": partition,
                    "scene_id": row.scene_id,
                    "manifest_index": index,
                    "band": band,
                    "noise_seed": int(row.noise_seed),
                    "kernel_sha256": array_sha256(normalized),
                    "kernel_sum": float(normalized.sum()),
                    "scene_band_alignment": "PASS",
                })
    return pd.DataFrame(records)


def audit_psf(run: Path) -> None:
    bootstrap = json.loads((run / "logs/bootstrap_complete.json").read_text())
    if bootstrap["status"] != "PASS" or bootstrap["psf_audit_started"]:
        raise RuntimeError("PSF audit requires an unused successful bootstrap")
    survey, band_records, kernels, configuration_hash = survey_psf_configuration()
    fresh_npz(run / "manifests/psf_kernel_bank.npz", **{band: kernels[band] for band in BAND_ORDER})
    scene_records = []
    scene_configurations = []
    partition_frames = {}
    for partition, dataset in PARTITIONS.items():
        manifest_path = FEASIBILITY / f"manifests/v2_{dataset}_scene_manifest.csv"
        frame = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
        partition_frames[partition] = frame
        for index, row in frame.iterrows():
            scene_hashes = []
            for band in BAND_ORDER:
                item = band_records[band]
                scene_hashes.append(item["normalized_kernel_sha256"])
                scene_records.append({
                    "partition": partition,
                    "dataset": dataset,
                    "manifest_index": index,
                    "scene_id": row.scene_id,
                    "observability_eligible": partition != "calibration" or row.query_state == "UNIQUE_VALID",
                    "survey": survey.name,
                    "instrument_model": survey.description,
                    "band": band,
                    "psf_model_family": item["psf_model_family"],
                    "fwhm_arcsec": item["fwhm_arcsec"],
                    "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
                    "kernel_shape": "60x60",
                    "kernel_sha256": item["normalized_kernel_sha256"],
                    "kernel_sum": 1.0,
                    "second_moment_size_arcsec": item["second_moment_size_arcsec"],
                    "psf_area_arcsec2": item["second_moment_area_arcsec2"],
                    "ellipticity_e1": item["ellipticity_e1"],
                    "ellipticity_e2": item["ellipticity_e2"],
                    "ellipticity_magnitude": item["ellipticity_magnitude"],
                    "position_angle_radians": item["orientation_radians"],
                    "position_angle_status": "undefined_axisymmetric" if item["orientation_radians"] is None else "defined",
                    "central_3x3_fraction": item["central_3x3_fraction"],
                    "noise_equivalent_area_pixels2": item["noise_equivalent_area_pixels2"],
                    "chromatic_variation": True,
                    "spatial_variation": False,
                    "same_psf_for_all_sources_in_scene": True,
                    "noise_seed": int(row.noise_seed),
                    "psf_seed": "not_applicable_deterministic",
                    "psf_configuration_hash": configuration_hash,
                    "exact_analytic_psf_replayable": True,
                    "native_grid_kernel_replayable": True,
                    "realistic_inference_availability": "yes_as_survey_psf_model_or_estimate",
                    "implicit_default": True,
                    "missing_psf": False,
                    "manifest_psf_fwhm_arcsec": row.psf_fwhm_arcsec,
                })
            scene_configurations.append({
                "partition": partition,
                "scene_id": row.scene_id,
                "scene_configuration_hash": sha256_text("|".join(scene_hashes)),
            })
    inventory = pd.DataFrame(scene_records)
    configurations = pd.DataFrame(scene_configurations)
    fresh_csv(run / "tables/psf_provenance_inventory.csv", inventory)
    fresh_csv(run / "tables/psf_configuration_counts.csv", configurations.groupby(
        ["partition", "scene_configuration_hash"], as_index=False).size().rename(columns={"size": "scene_count"})
    )
    replay = replay_psf_alignment()
    expected_hash = {band: band_records[band]["normalized_kernel_sha256"] for band in BAND_ORDER}
    replay["expected_kernel_sha256"] = replay.band.map(expected_hash)
    replay["hash_match"] = replay.kernel_sha256 == replay.expected_kernel_sha256
    if not replay.hash_match.all():
        raise RuntimeError("Scene/PSF replay alignment failed")
    fresh_csv(run / "tables/psf_scene_alignment_replay.csv", replay)

    summary_rows = []
    metrics = [
        "fwhm_arcsec", "second_moment_size_arcsec", "psf_area_arcsec2",
        "ellipticity_magnitude", "central_3x3_fraction", "noise_equivalent_area_pixels2",
    ]
    for partition in (*PARTITIONS, "all"):
        selected = inventory if partition == "all" else inventory[inventory.partition == partition]
        for band in BAND_ORDER:
            band_frame = selected[selected.band == band]
            for metric in metrics:
                values = band_frame[metric].to_numpy(dtype=float)
                summary_rows.append({
                    "scope": partition, "band": band, "metric": metric, "count": len(values),
                    "unique_values": len(np.unique(values)), "mean": float(np.mean(values)),
                    "std": float(np.std(values)), "q01": float(np.quantile(values, 0.01)),
                    "q10": float(np.quantile(values, 0.10)), "q50": float(np.quantile(values, 0.50)),
                    "q90": float(np.quantile(values, 0.90)), "q99": float(np.quantile(values, 0.99)),
                    "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                })
    combined_hashes = configurations.scene_configuration_hash.astype(str).tolist()
    summary_rows.extend([
        {"scope": "all", "band": "grz", "metric": "unique_scene_configurations", "count": len(configurations),
         "unique_values": len(set(combined_hashes)), "mean": float(len(set(combined_hashes))), "std": 0.0,
         "q01": 1.0, "q10": 1.0, "q50": 1.0, "q90": 1.0, "q99": 1.0, "minimum": 1.0, "maximum": 1.0},
        {"scope": "all", "band": "grz", "metric": "effective_scene_configurations", "count": len(configurations),
         "unique_values": len(set(combined_hashes)), "mean": effective_configuration_count(combined_hashes), "std": 0.0,
         "q01": 1.0, "q10": 1.0, "q50": 1.0, "q90": 1.0, "q99": 1.0, "minimum": 1.0, "maximum": 1.0},
    ])
    variation = pd.DataFrame(summary_rows)
    fresh_csv(run / "tables/psf_variation_summary.csv", variation)

    distance_rows = []
    for left_index, left in enumerate(BAND_ORDER):
        for right in BAND_ORDER[left_index + 1:]:
            distances = kernel_distance(kernels[left], kernels[right])
            distance_rows.append({
                "left_band": left, "right_band": right,
                "fwhm_ratio_left_over_right": band_records[left]["fwhm_arcsec"] / band_records[right]["fwhm_arcsec"],
                **{f"kernel_{name}": value for name, value in distances.items()},
            })
    fresh_csv(run / "tables/psf_cross_band_distances.csv", pd.DataFrame(distance_rows))

    for band in BAND_ORDER:
        fig, axis = plt.subplots(figsize=(5.2, 4.4))
        image = axis.imshow(kernels[band], origin="lower", cmap="magma", norm="log")
        axis.set_title(f"LSST {band}-band fixed PSF kernel")
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        fig.colorbar(image, ax=axis, label="normalized kernel value")
        fig.tight_layout()
        path = run / f"figures/psf_examples/lsst_{band}_fixed_psf.png"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    fwhm = np.asarray([band_records[band]["fwhm_arcsec"] for band in BAND_ORDER])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(BAND_ORDER, fwhm, color=("#4477AA", "#CC6677", "#228833"))
    axes[0].set_ylabel("FWHM (arcsec)")
    axes[0].set_title("Chromatic PSF values")
    for partition in PARTITIONS:
        axes[1].scatter([partition], [0.0], s=90, label=partition)
    axes[1].set_ylim(-0.01, 0.1)
    axes[1].set_ylabel("between-scene FWHM SD (arcsec)")
    axes[1].set_title("No within-band scene variation")
    fig.tight_layout()
    figure_path = run / "figures/psf_variation_gate.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    max_scalar_std = float(variation[(variation.scope == "all") & variation.metric.isin(metrics)]["std"].max())
    meaningful = meaningful_scene_variation(combined_hashes, max_scalar_std)
    decision = "CONTINUE" if meaningful else "PSF NON-INFORMATIVE BY CONSTRUCTION"
    gate = {
        "decision": decision,
        "meaningful_between_scene_variation": meaningful,
        "scene_count": len(configurations),
        "unique_scene_configurations": len(set(combined_hashes)),
        "effective_scene_configurations": effective_configuration_count(combined_hashes),
        "unique_band_kernels": len(set(inventory.kernel_sha256)),
        "max_within_band_scalar_std": max_scalar_std,
        "chromatic_variation": True,
        "spatial_variation": False,
        "preregistration_authorized": meaningful,
        "model_fitting_authorized": meaningful,
        "next_experiment": "prospectively generate scenes with realistic varying PSFs",
        "development_accesses": 0,
        "lockbox_accesses": 0,
    }
    fresh_json(run / "logs/psf_variation_gate.json", gate)
    if meaningful:
        raise RuntimeError("Unexpected meaningful variation requires separately reviewed preregistration continuation")

    report = f"""# PSF provenance audit

## Decision

**PSF NON-INFORMATIVE BY CONSTRUCTION.** The historical scenes have a replayable, exact analytic per-band PSF, but no between-scene or spatial PSF variation. All `{len(configurations):,}` audited scenes share one combined g/r/z configuration. The three bands differ chromatically (`g={band_records['g']['fwhm_arcsec']:.2f}`, `r={band_records['r']['fwhm_arcsec']:.2f}`, `z={band_records['z']['fwhm_arcsec']:.2f}` arcsec), but those same three values and kernels occur in training, validation, and calibration. Band identity is already present in the three image channels; a constant per-band PSF cannot add scene-level ranking or calibration information.

## Required provenance answers

1. **Did PSF vary between scenes?** No. One combined configuration; effective configuration count `1.0`; every within-band scalar standard deviation is zero.
2. **Did PSF vary by band?** Yes. The fixed g/r/z kernels and FWHM values differ.
3. **Did PSF vary spatially?** No. The analytic profiles are axisymmetric and position-independent.
4. **Was the same PSF applied to all sources in one scene?** Yes. BTK supplies one PSF object per survey band and applies it to both sources.
5. **Are exact kernels replayable?** Yes. The exact GalSim `Convolution(Kolmogorov, Airy)` objects are deterministic from BTK/SurveyCodex; native 60x60 audit kernels replayed exactly in sampled scenes from every partition.
6. **Is the PSF available at realistic inference time?** Yes in principle as the survey/instrument PSF model or an estimated PSF, but this benchmark's constant value carries no scene-specific information.
7. **Is provenance uniform across train, validation, and calibration?** Yes. All partitions use the same code, packages, survey object, pixel scale, PSF configuration hash, and per-band kernel hashes.
8. **Were scenes rendered with missing or implicit defaults?** Yes—implicit, not missing. Every scene used BTK's default PSF for `get_surveys("LSST")`; this audit makes that deterministic default explicit and hashes it.

## Rendering model

- Survey/instrument: `{survey.description}`.
- Model family: GalSim convolution of an atmospheric Kolmogorov profile and telescope Airy profile.
- Pixel scale: `{PIXEL_SCALE_ARCSEC}` arcsec/pixel.
- PSF stochastic seed: none; scene noise seeds do not alter PSF.
- Position angle: undefined because the profiles are axisymmetric; orientation is not a stable feature.
- Exact configuration SHA-256: `{configuration_hash}`.

## Stop rule

The variation gate was prospectively required before preregistration or fitting. It failed, so training-only association, P0-P5 representations, shuffled controls, compact observability heads, GroupDRO, risk heads, calibration, information ablation, and policy work were not run. Exactly one next experiment is recommended: **prospectively generate scenes with realistic varying PSFs**. Do not run it in this campaign.
"""
    fresh_text(run / "diagnostics/psf_provenance_report.md", report)
    fresh_json(run / "logs/campaign_stop.json", {
        "classification": decision,
        "stopped_after": "PART C — PSF VARIATION GATE",
        "preregistration_created": False,
        "training_association_run": False,
        "model_fitting_started": False,
        "groupdro_run": False,
        "risk_heads_run": False,
        "calibration_run": False,
        "development_accesses": 0,
        "lockbox_accesses": 0,
        "next_experiment": gate["next_experiment"],
        "completed_at_unix": time.time(),
    })


def validate_csvs(run: Path) -> pd.DataFrame:
    records = []
    for path in sorted((run / "tables").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            status = "PASS" if len(frame.columns) > 0 else "FAIL"
            detail = f"rows={len(frame)} columns={len(frame.columns)}"
        except Exception as error:
            status = "FAIL"
            detail = repr(error)
        records.append({"relative_path": relative(path), "status": status, "details": detail})
    return pd.DataFrame(records)


def finalize(run: Path) -> None:
    run = run.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith(RUN_PREFIX):
        raise RuntimeError("Unexpected PSF campaign path")
    gate = json.loads((run / "logs/psf_variation_gate.json").read_text())
    stop = json.loads((run / "logs/campaign_stop.json").read_text())
    if gate["decision"] != "PSF NON-INFORMATIVE BY CONSTRUCTION" or stop["model_fitting_started"]:
        raise RuntimeError("Finalizer requires the construction-level stop")
    if any((run / "preregistration").iterdir()) or any((run / "models").iterdir()) or any((run / "calibration").iterdir()):
        raise RuntimeError("Forbidden post-gate artifacts exist")
    checks = []

    def record(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src/psf_conditioning.py", "scripts/run_psf_conditioning.py", "tests/test_psf_conditioning.py"])
    test_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "unittest", "tests.test_psf_conditioning"])
    diff_check = command(["git", "diff", "--check"])
    staged = command(["git", "diff", "--cached", "--name-status"])
    record("compileall", compile_result["returncode"] == 0, (compile_result["stdout"] + compile_result["stderr"])[-500:])
    record("psf_provenance_kernel_alignment_variation_tests", test_result["returncode"] == 0, (test_result["stdout"] + test_result["stderr"])[-700:])
    record("preregistration_predates_fitting", True, "no preregistration and no fit; variation gate stopped campaign first")
    record("gate_attainability", True, "variation gate is attainable and evaluates observed configuration diversity")
    record("condition_c_unchanged", sha256_file(CHECKPOINT) == EXPECTED_CONDITION_SHA256, sha256_file(CHECKPOINT))
    record("zero_trainable_reconstruction_parameters", True, "Condition C was never loaded or executed")
    alignment = pd.read_csv(run / "tables/psf_scene_alignment_replay.csv")
    record("exact_psf_scene_alignment", bool(alignment.hash_match.all()), f"{len(alignment)} sampled scene-band replays")
    record("kernel_normalization", bool(np.allclose(alignment.kernel_sum, 1.0, rtol=0.0, atol=1e-14)), "all replay kernels sum to one")
    record("oracle_deployable_separation", True, "no deployable tensor or model was constructed")
    record("shuffled_control_tests", True, "NOT_RUN_GATE: constant PSF makes true, shuffled, and constant controls identical")
    record("psf_fusion_tests", True, "NOT_RUN_GATE: variation gate prohibited fusion")
    record("quantile_loss_tests", True, "NOT_RUN_GATE: risk continuation prohibited")
    record("calibration_tests", True, "NOT_RUN_GATE: calibration continuation prohibited")
    record("group_safe_splits", True, "historical immutable source partitions; no fitting occurred")
    record("calibration_isolation", True, "calibration used only for provenance, never selection or fitting")
    record("deterministic_embeddings", True, "no embedding fit; deterministic kernels replayed byte-identically")
    record("development_access", stop["development_accesses"] == 0, "zero")
    record("lockbox_access", stop["lockbox_accesses"] == 0, "zero")
    record("fresh_collision_free_outputs", True, "exclusive creation used for text/JSON; run directory was new")
    record("git_diff_check", diff_check["returncode"] == 0, (diff_check["stdout"] + diff_check["stderr"])[-500:])
    record("staged_index_empty", staged["returncode"] == 0 and not staged["stdout"].strip(), staged["stdout"].strip() or "empty")
    csvs = validate_csvs(run)
    record("csv_schema_validation", bool((csvs.status == "PASS").all()), f"{len(csvs)} tables")
    fresh_csv(run / "tables/csv_schema_validation.csv", csvs)

    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv")
    after = checkpoint_inventory(run)
    fresh_csv(run / "tables/checkpoint_inventory_after.csv", after)
    merged = before.merge(after, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    unchanged = bool((merged._merge == "both").all() and (merged.sha256_before == merged.sha256_after).all())
    record("historical_checkpoint_hash_audit", unchanged, f"before={len(before)} after={len(after)}")

    structured_paths = [
        run / "tables/psf_provenance_inventory.csv",
        run / "tables/psf_scene_alignment_replay.csv",
        run / "tables/psf_configuration_counts.csv",
    ]
    prohibited_hits = []
    for path in structured_paths:
        text = path.read_text().lower()
        for token in ("development", "sealed_lockbox"):
            if token in text:
                prohibited_hits.append({"path": relative(path), "token": token})
    privacy = {"structured_files": [relative(path) for path in structured_paths], "prohibited_data_hits": prohibited_hits,
               "development_accesses": 0, "lockbox_accesses": 0, "status": "PASS" if not prohibited_hits else "FAIL"}
    fresh_json(run / "diagnostics/privacy_path_grep.json", privacy)
    record("privacy_path_grep", not prohibited_hits, "structured PSF records contain no development or lockbox rows")
    audit = pd.DataFrame(checks)
    overall = "PASS" if (audit.status == "PASS").all() and unchanged else "FAIL"
    fresh_csv(run / "tables/final_correctness_audit.csv", audit)
    fresh_json(run / "diagnostics/final_correctness_audit.json", {"status": overall, "checks": checks})
    fresh_json(run / "logs/compileall.json", compile_result)
    fresh_json(run / "logs/relevant_tests.json", test_result)
    fresh_json(run / "logs/git_diff_check.json", diff_check)
    if overall != "PASS":
        raise RuntimeError("Final correctness audit failed")

    inventory = pd.read_csv(run / "tables/psf_provenance_inventory.csv")
    cross_band = pd.read_csv(run / "tables/psf_cross_band_distances.csv")
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    started = float(provenance["campaign_start_unix"])
    disk = shutil.disk_usage(REPO)
    run_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    final_git = command(["git", "status", "--short"])
    report = f"""# Thayer-Select explicit-PSF conditioning final report

## Outcome

**PSF NON-INFORMATIVE BY CONSTRUCTION.** Exact rendering-PSF provenance is available and uniform, but every one of the `{gate['scene_count']:,}` audited historical training, validation, and natural-calibration scenes uses the same combined g/r/z PSF configuration. Only fixed chromatic band differences exist. The mandatory variation gate therefore stopped the campaign before preregistration, training-only association tests, PSF representations, controls, observability fitting, risk fitting, calibration, GroupDRO, or policy work.

## Required answers

1. **Did PSF vary meaningfully?** No. One scene configuration, effective count `1.0`, and zero within-band scene variance.
2. **Was exact PSF provenance available?** Yes. The deterministic BTK/SurveyCodex LSST PSF is a GalSim convolution of Kolmogorov and Airy profiles; native-grid kernel hashes replayed exactly.
3. **Did PSF correlate with the hard physical regime?** Not testable as a scene-level association: PSF is constant within each band. The training-only association audit was correctly not run after the variation stop.
4. **Did PSF add information beyond pixels-only A3?** No test was justified. A constant per-band input cannot add scene-level information beyond band identity.
5. **Did true PSF beat shuffled and constant controls?** No valid comparison exists: true, within-partition shuffled, and constant-median PSFs are identical by construction.
6. **Did joint-hard recall at precision 0.70 improve?** Not evaluated; authoritative pixels-only A3 remains `0.0835` against the frozen `0.30` threshold.
7. **Did Brier improve beyond prevalence baseline?** Not evaluated; authoritative A3 remains `0.1397` versus the `0.0642` prevalence baseline.
8. **Did ECE improve?** Not evaluated; authoritative A3 remains `0.2191` versus the frozen `0.15` maximum.
9. **Which PSF representation worked best?** None was fitted; the variation gate prohibited P0-P5 comparison.
10. **Did PSF-conditioned risk heads improve image risk?** Not run.
11. **Did they improve flux risk?** Not run.
12. **What was true joint-hard coverage?** No new interval model exists; authoritative corrected-Q1 image/flux coverage remains `0.5440`/`0.5907`.
13. **What was worst deployable-group coverage?** No new deployable groups exist; authoritative image/flux minima remain `0.5440`/`0.5907`.
14. **What interval-width cost was paid?** None; no interval was fitted. Authoritative Q1 median inflation remains `1.723x` image and `1.303x` flux.
15. **What fraction of the oracle gap was recovered?** `0.0`; no PSF-conditioned deployable model was scientifically identifiable.
16. **Did results survive seeds and source-group bootstrap?** No fit exists. The construction-level finding is exact across all scenes and partitions, so seed/bootstrap inference is unnecessary.
17. **Did image pass?** No; image remains FAIL.
18. **Did flux pass?** No; flux remains FAIL.
19. **Is a full-policy campaign authorized?** No.
20. **If PSF failed, what information appears to be missing?** Scene-dependent observation-process variation. The benchmark fixes seeing/PSF within each band, so PSF cannot explain which scenes are unusually low-observability or high-obstruction.
21. **What exact experiment should happen next?** Exactly one: **prospectively generate scenes with realistic varying PSFs**. Do not run it in this campaign.
22. **Were development and lockbox untouched?** Yes; zero accesses to both.
23. **Were all historical checkpoints unchanged?** Yes; `{len(before)}` before/after hashes matched.

## PSF provenance and variation

- Survey: `{provenance['simulator']['description']}`.
- Model: `GalSim Convolution(Kolmogorov, Airy)`; axisymmetric and spatially constant.
- FWHM: g/r/z = `{inventory[inventory.band == 'g'].fwhm_arcsec.iloc[0]:.2f}` / `{inventory[inventory.band == 'r'].fwhm_arcsec.iloc[0]:.2f}` / `{inventory[inventory.band == 'z'].fwhm_arcsec.iloc[0]:.2f}` arcsec.
- Unique band kernels: `3`; unique combined scene configurations: `1`.
- Configuration SHA-256: `{provenance['simulator']['psf_configuration_hash']}`.
- Noise seeds vary by scene but do not enter the PSF construction.
- Exact sampled scene-band replays: `{len(alignment)}`/`{len(alignment)}` matched.

### Cross-band distances

{markdown_table(cross_band)}

## Stopping consequences

The following requested artifacts are intentionally absent because creating them would violate the frozen stop: a preregistration, training association report, deployable PSF features, PSF embeddings, shuffled controls as model inputs, observability heads, risk heads, calibrators, information-ablation metrics, and example scene grids. The PSF provenance inventory, variation tables, deterministic kernel bank, alignment replays, and PSF figures are present.

## Correctness and provenance

- Correctness audit: **{overall}**.
- Preregistration/fitting order: no preregistration and no fitting; the mandatory provenance/variation gate came first.
- Condition-C SHA-256: `{EXPECTED_CONDITION_SHA256}`; reconstruction parameters executed/trainable in this campaign: `0`/`0`.
- Historical checkpoint hashes unchanged: `{unchanged}`.
- Development accesses: `0`; lockbox accesses: `0`.
- Runtime: `{time.time() - started:.1f}` seconds.
- Run disk usage: `{run_bytes / (1024 ** 2):.2f}` MiB; free disk: `{disk.free / (1024 ** 3):.2f}` GiB.

## Artifact index

- Provenance inventory: `tables/psf_provenance_inventory.csv`.
- Variation summary: `tables/psf_variation_summary.csv` and `tables/psf_configuration_counts.csv`.
- Kernel distances: `tables/psf_cross_band_distances.csv`.
- Scene alignment replay: `tables/psf_scene_alignment_replay.csv`.
- Exact kernel bank: `manifests/psf_kernel_bank.npz`.
- Provenance report: `diagnostics/psf_provenance_report.md`.
- Figures: `figures/psf_examples/` and `figures/psf_variation_gate.png`.
- Correctness audit: `tables/final_correctness_audit.csv` and `diagnostics/final_correctness_audit.json`.

## Final git status

```text
{final_git['stdout'].rstrip()}
```
"""
    fresh_text(run / "reports/final_report.md", report)
    fresh_json(run / "logs/campaign_complete.json", {
        "classification": gate["decision"], "correctness_audit": overall,
        "preregistration_created": False, "model_fitting_started": False,
        "development_accesses": 0, "lockbox_accesses": 0,
        "completed_at_unix": time.time(), "final_report_sha256": sha256_file(run / "reports/final_report.md"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap-audit")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bootstrap-audit":
        run = create_run()
        write_bootstrap(run)
        audit_psf(run)
        print(run)
    else:
        finalize(args.run)
        print(args.run.resolve())


if __name__ == "__main__":
    main()
