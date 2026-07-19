#!/usr/bin/env python3
"""Append-only scene-stratified external-photometry campaign."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift as image_shift
from scipy.stats import beta, spearmanr
import torch


RUN_DIR = Path(__file__).resolve().parents[1]
REPO = RUN_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from btk.survey import get_surveys
from surveycodex.utilities import mag2counts
from scripts import run_thayer_external_photometry_preflight_v0 as preflight
from scripts import run_thayer_external_photometry_convergence_correction_v0 as correction
from scripts.analyze_thayer_recoverability_v0 import centroid, scene_metrics
from scripts.run_thayer_flux_free_identifiability_v0 import (
    load_science_inputs,
    selected_manifest_rows,
    sha256_file,
)
from src.btk_scene import load_catsim_catalog
from src.model9_optimizer import deterministic_starts
from src.model9_structured import (
    BANDS,
    FAMILY_BULGE_DISK,
    FrozenSolverProtocol,
    oracle_information_audit,
)


CAMPAIGN = "Thayer-External-Photometry-Scene-Stratification-v0"
SCENES = (0, 3, 5, 6, 18, 51, 73, 81)
NEW_SCENES = (3, 5, 18, 51, 73, 81)
CONDITIONS = ("TOTAL_SOURCE_PHOTOMETRY", "PER_BAND_SOURCE_PHOTOMETRY")
PRIMARY_CONDITION = "TOTAL_SOURCE_PHOTOMETRY"
STARTS = 4
MAX_NFEV = 500
RELATIVE_SIGMA = 0.05
MEASUREMENT_SEED = 2026071805
TOTAL_WEIGHTS = np.ones(3, dtype=np.float64)
DIAMETERS = correction.DIAMETERS

PRIOR_RUN = REPO / "outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638"
PRIOR_MEASUREMENTS = REPO / "outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/tables/external_photometry_measurements.csv"
S1_METRICS = preflight.S1_METRICS
P2_METRICS = preflight.P2_METRICS
DEFINITIONS = preflight.DEFINITIONS
SCENE_MANIFEST = preflight.SCENE_MANIFEST
SCIENCE_H5 = preflight.SCIENCE_H5
CATALOG_PATH = preflight.CATALOG_PATH
PROTOCOL_PATH = RUN_DIR / "preregistration/frozen_protocol.md"

AUTHORITATIVE_REPORTS = (
    REPO / "reports/thayer_recoverability_v0.md",
    REPO / "outputs/runs/thayer_identifiability_v1_20260715_003220/reports/final_report.md",
    REPO / "outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/reports/final_report.md",
    REPO / "outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/reports/final_report.md",
    REPO / "outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/reports/final_report.md",
    PRIOR_RUN / "reports/final_report.md",
)

FEATURES = (
    "overlap_fraction",
    "centroid_separation_psf",
    "source_flux_ratio",
    "log10_total_brightness_electrons",
    "morphology_similarity",
    "color_similarity",
    "sersic_parameter_similarity",
    "bulge_fraction_difference",
    "effective_radius_ratio",
    "psf_sensitivity_log10_condition_gain",
    "log10_s1_condition_number",
    "s1_endpoint_multiplicity",
    "s1_image_diameter",
    "s1_morphology_diameter",
    "s1_flux_allocation_diameter",
)

CLASS_SCORE = {
    "NON_IDENTIFIABLE": 0,
    "PARTIALLY_IDENTIFIABLE": 1,
    "NEAR_UNIQUE": 2,
    "UNIQUE": 3,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def fresh_json(path: Path, value: Any) -> None:
    fresh_text(path, json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(safe(item), sort_keys=True, separators=(",", ":"))
                if isinstance(item, (dict, list, tuple, np.ndarray)) else safe(item)
                for key, item in row.items()
            })
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "head": command(["git", "rev-parse", "HEAD"]),
        "readme_sha256": sha256_file(REPO / "README.md"),
        "git_index_entries": command(["git", "diff", "--cached", "--name-only"]),
        "authoritative_report_hashes": {
            str(path.relative_to(REPO)): sha256_file(path) for path in AUTHORITATIVE_REPORTS
        },
        "protected_data_access": {"development": 0, "atlas": 0, "lockbox": 0},
        "neural_networks_loaded": 0,
        "neural_training_steps": 0,
    }


def ensure_output_directories() -> None:
    for name in ("tables", "figures", "fit_records", "manifests", "reports"):
        (RUN_DIR / name).mkdir(exist_ok=False)


def definition_rows() -> dict[int, dict[str, str]]:
    return {
        int(row["dataset_index"]): row
        for row in read_rows(DEFINITIONS)
        if int(row["dataset_index"]) in SCENES
    }


def load_historical_baselines() -> dict[tuple[int, str], dict[str, Any]]:
    keys = (
        "distinct_solution_classes",
        "requested_image_diameter",
        "companion_image_diameter",
        "flux_allocation_diameter",
        "morphology_parameter_diameter",
    )
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_rows(S1_METRICS):
        if int(row["scene_index"]) in SCENES and row["family"] == FAMILY_BULGE_DISK:
            result[(int(row["scene_index"]), "S1")] = {
                "classification": row["classification"],
                "condition_number": float(row["condition_number"]),
                "rank": int(row["rank"]),
                "nullity": int(row["null_space_dimension"]),
                **{
                    key: int(row[key]) if key == "distinct_solution_classes" else float(row[key])
                    for key in keys
                },
            }
    for row in read_rows(P2_METRICS):
        if int(row["scene_index"]) not in SCENES or row["family"] != FAMILY_BULGE_DISK:
            continue
        condition = row["condition"]
        if condition not in {"S2", "P2"}:
            continue
        result[(int(row["scene_index"]), condition)] = {
            "classification": row["classification"],
            "condition_number": float(row["condition_number"]),
            "rank": int(row["rank"]),
            "nullity": int(row["null_space_dimension"]),
            **{
                key: int(row[key]) if key == "distinct_solution_classes" else float(row[key])
                for key in keys
            },
        }
    expected = {(scene, condition) for scene in SCENES for condition in ("S1", "S2", "P2")}
    if set(result) != expected:
        raise RuntimeError(f"historical baseline mismatch: {expected - set(result)}")
    return result


def catalog_fluxes(entry: Any, survey: Any) -> np.ndarray:
    return np.asarray([
        float(mag2counts(entry[f"{band}_ab"], survey, survey.get_filter(band)).to_value("electron"))
        for band in BANDS
    ], dtype=np.float64)


def load_prior_measurements() -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    rows = read_rows(PRIOR_MEASUREMENTS)
    for row in rows:
        key = (int(row["scene_index"]), row["condition"])
        if key[0] in {0, 6}:
            grouped.setdefault(key, []).append(row)
    output = {}
    preserved_rows: list[dict[str, Any]] = []
    for key in ((scene, condition) for scene in (0, 6) for condition in CONDITIONS):
        group = grouped[key]
        output[key] = {
            "measured": np.asarray([float(row["measured_flux_electrons"]) for row in group]),
            "sigma": np.asarray([float(row["sigma_electrons"]) for row in group]),
        }
        for row in group:
            preserved_rows.append({**row, "measurement_source": "authoritative_reuse_no_regeneration"})
    return output, preserved_rows


def generate_measurements() -> tuple[
    dict[tuple[int, str], dict[str, np.ndarray]],
    list[dict[str, Any]],
    dict[int, list[Any]],
]:
    measurements, rows = load_prior_measurements()
    catalog, catalog_hash = load_catsim_catalog(CATALOG_PATH)
    definitions = definition_rows()
    manifests = selected_manifest_rows()
    survey = get_surveys("LSST")
    catalog_entries: dict[int, list[Any]] = {}
    for scene in SCENES:
        definition = definitions[scene]
        source_rows = (int(definition["source_a_row"]), int(definition["source_b_row"]))
        catalog_entries[scene] = [catalog.table[index] for index in source_rows]
        if scene not in NEW_SCENES:
            continue
        true_by_ab = [catalog_fluxes(catalog.table[index], survey) for index in source_rows]
        requested_index = int(manifests[scene]["matched_source_index"])
        identity_order = (requested_index, 1 - requested_index)
        true_flux = np.stack([true_by_ab[index] for index in identity_order])
        for condition_index, condition in enumerate(CONDITIONS):
            rng = np.random.default_rng(MEASUREMENT_SEED + 100 * scene + condition_index)
            latent = true_flux @ TOTAL_WEIGHTS if condition == PRIMARY_CONDITION else true_flux.reshape(-1)
            sigma = RELATIVE_SIGMA * latent
            z = rng.standard_normal(latent.shape)
            measured = latent + sigma * z
            if np.any(measured <= 0) or np.any(sigma <= 0):
                raise RuntimeError("nonpositive external photometry")
            measurements[(scene, condition)] = {"measured": measured, "sigma": sigma}
            labels = (
                (("requested", "total"), ("companion", "total"))
                if condition == PRIMARY_CONDITION
                else tuple((identity, band) for identity in ("requested", "companion") for band in BANDS)
            )
            for index, (identity, band) in enumerate(labels):
                rows.append({
                    "scene_index": scene,
                    "scene_id": manifests[scene]["scene_id"],
                    "condition": condition,
                    "source_identity": identity,
                    "measurement_band": band,
                    "combination_weights_g_r_z": TOTAL_WEIGHTS if band == "total" else [1.0 if item == band else 0.0 for item in BANDS],
                    "measured_flux_electrons": measured[index],
                    "sigma_electrons": sigma[index],
                    "relative_sigma": RELATIVE_SIGMA,
                    "frozen_noise_z": z[index],
                    "measurement_seed": MEASUREMENT_SEED + 100 * scene + condition_index,
                    "catalog_sha256": catalog_hash,
                    "latent_catalog_value_persisted": False,
                    "measurement_source": "scene_stratification_extension",
                })
        del true_flux, true_by_ab
    if set(measurements) != {(scene, condition) for scene in SCENES for condition in CONDITIONS}:
        raise RuntimeError("measurement population incomplete")
    return measurements, rows, catalog_entries


def endpoint_value(endpoint: Any, key: str) -> Any:
    return endpoint[key] if isinstance(endpoint, dict) else getattr(endpoint, key)


def classify_external(result: dict[str, Any]) -> str:
    protocol = FrozenSolverProtocol()
    successful = [endpoint for endpoint in result["endpoints"] if bool(endpoint_value(endpoint, "success"))]
    if not successful:
        return "OPTIMIZATION_UNRESOLVED"
    if not result["replay"]["exact_match"] or not np.isfinite(float(result["condition_number"])):
        return "NUMERICALLY_UNSTABLE"
    best = min(successful, key=lambda endpoint: float(endpoint_value(endpoint, "likelihood_objective")))
    if float(endpoint_value(best, "chi_square")) > float(result["support_threshold"]):
        return "OUT_OF_SUPPORT"
    geometry = result["geometry"]
    image_diameter = max(float(geometry["requested_image_diameter"]), float(geometry["companion_image_diameter"]))
    allocation_distinct = (
        image_diameter > protocol.unique_image_diameter
        or float(geometry["flux_allocation_diameter"]) > protocol.unique_flux_allocation_diameter
        or float(geometry["morphology_parameter_diameter"]) > protocol.unique_morphology_diameter
    )
    multiple = int(geometry["distinct_solution_classes"]) > 1 or int(result["nullity"]) > 0
    if multiple:
        return "NON_IDENTIFIABLE" if allocation_distinct else "PARTIALLY_IDENTIFIABLE"
    strict = (
        int(result["rank"]) == int(result["active_parameter_count"])
        and int(result["nullity"]) == 0
        and float(result["condition_number"]) <= protocol.maximum_condition_number
        and float(result["gradient_norm"]) <= protocol.acceptable_gradient_norm
        and image_diameter <= protocol.unique_image_diameter
        and float(geometry["flux_allocation_diameter"]) <= protocol.unique_flux_allocation_diameter
        and float(geometry["morphology_parameter_diameter"]) <= protocol.unique_morphology_diameter
        and bool(geometry["prompt_identity_consistent"])
        and not bool(result["boundary_contact_flags"]["invalid_zero_flux_collapse"])
    )
    return "UNIQUE" if strict else "NEAR_UNIQUE"


def prior_fit(scene: int, condition: str) -> dict[str, Any]:
    path = PRIOR_RUN / "fit_records" / f"scene_{scene:03d}_{condition.lower()}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "scene_index": scene,
        "scene_id": record["scene_id"],
        "condition": condition,
        "family": record["family"],
        "level": record["level"],
        "starts": np.asarray(record["starts"], dtype=np.float64),
        "endpoints": record["endpoints"],
        "gradient_norm": float(record["gradient_norm"]),
        "rank": int(record["rank"]),
        "active_parameter_count": int(record["active_parameter_count"]),
        "nullity": int(record["nullity"]),
        "condition_number": float(record["condition_number"]),
        "geometry": record["geometry"],
        "support_start_indices": record["support_start_indices"],
        "support_threshold": float(record["support_threshold"]),
        "boundary_contact_flags": record["boundary_contact_flags"],
        "replay": record["replay"],
        "optimizer_limited": bool(record["optimizer_limited"]),
        "optimizer_successful_starts": sum(bool(endpoint["success"]) for endpoint in record["endpoints"]),
        "max_budget_starts": sum(int(endpoint["nfev"]) >= MAX_NFEV for endpoint in record["endpoints"]),
        "fit_runtime_seconds": float(record["fit_runtime_seconds"]),
        "source": "authoritative_convergence_correction_reuse",
        "source_record_sha256": sha256_file(path),
    }
    result["scientific_classification"] = classify_external(result)
    return result


def serialize_fit(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "scene_index": result["scene_index"],
        "scene_id": result["scene_id"],
        "condition": result["condition"],
        "family": result["family"],
        "level": result["level"],
        "source": result["source"],
        "starts": result["starts"],
        "endpoints": [endpoint if isinstance(endpoint, dict) else endpoint.record() for endpoint in result["endpoints"]],
        "gradient_norm": result["gradient_norm"],
        "rank": result["rank"],
        "active_parameter_count": result["active_parameter_count"],
        "nullity": result["nullity"],
        "condition_number": result["condition_number"],
        "geometry": result["geometry"],
        "support_start_indices": result["support_start_indices"],
        "support_threshold": result["support_threshold"],
        "boundary_contact_flags": result["boundary_contact_flags"],
        "replay": result["replay"],
        "optimizer_limited": result["optimizer_limited"],
        "optimizer_successful_starts": result["optimizer_successful_starts"],
        "max_budget_starts": result["max_budget_starts"],
        "fit_runtime_seconds": result["fit_runtime_seconds"],
        "scientific_classification": result["scientific_classification"],
        "source_record_sha256": result.get("source_record_sha256", ""),
    }


def execute_fits(
    measurements: dict[tuple[int, str], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifests = selected_manifest_rows()
    results = [prior_fit(scene, condition) for scene in (0, 6) for condition in CONDITIONS]
    endpoint_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for scene in NEW_SCENES:
        inputs, metadata = load_science_inputs(scene, FAMILY_BULGE_DISK)
        starts = deterministic_starts(inputs, FrozenSolverProtocol(), count=STARTS)
        for condition in CONDITIONS:
            audit = oracle_information_audit(
                inputs,
                FrozenSolverProtocol(),
                extra_named_inputs={"scene_index": scene, "external_measurements": measurements[(scene, condition)]["measured"].tolist()},
            )
            if audit["status"] != "PASS":
                raise RuntimeError(f"oracle information audit failed for {scene} {condition}")
            fit = correction.fit_one(scene, condition, inputs, measurements[(scene, condition)], starts)
            fit.update({
                "scene_id": metadata["scene_id"],
                "source": "new_scene_stratification_fit",
                "scientific_classification": "",
            })
            fit["scientific_classification"] = classify_external(fit)
            results.append(fit)
            audit_rows.append({
                "scene_index": scene,
                "scene_id": metadata["scene_id"],
                "condition": condition,
                "status": audit["status"],
                "blend_rows_accessed": 1,
                "coordinate_rows_accessed": 1,
                "catalog_rows_used_by_measurement_generator": 2,
                "catalog_truth_exposed_to_fit": False,
                "isolated_source_images_exposed_to_fit": 0,
                "morphology_truth_exposed_to_fit": 0,
                "truth_initialization_exposed_to_fit": 0,
                "development_access": 0,
                "atlas_access": 0,
                "lockbox_access": 0,
            })
    results.sort(key=lambda item: (int(item["scene_index"]), CONDITIONS.index(item["condition"])))
    for result in results:
        fresh_json(
            RUN_DIR / "fit_records" / f"scene_{result['scene_index']:03d}_{result['condition'].lower()}.json",
            serialize_fit(result),
        )
        geometry = result["geometry"]
        support = set(int(value) for value in result["support_start_indices"])
        acceptable = set(int(value) for value in geometry["acceptable_endpoint_indices"])
        for endpoint in result["endpoints"]:
            record = endpoint if isinstance(endpoint, dict) else endpoint.record()
            endpoint_rows.append({
                "scene_index": result["scene_index"],
                "scene_id": result["scene_id"],
                "condition": result["condition"],
                "source": result["source"],
                **record,
                "observation_support_acceptable": int(record["start_index"]) in support,
                "total_objective_acceptable": int(record["start_index"]) in acceptable,
                "hit_500_evaluation_ceiling": int(record["nfev"]) >= MAX_NFEV,
            })
    for scene in (0, 6):
        for condition in CONDITIONS:
            audit_rows.append({
                "scene_index": scene,
                "scene_id": manifests[scene]["scene_id"],
                "condition": condition,
                "status": "PASS_REUSED_AUTHORITY",
                "blend_rows_accessed": 0,
                "coordinate_rows_accessed": 0,
                "catalog_rows_used_by_measurement_generator": 0,
                "catalog_truth_exposed_to_fit": False,
                "isolated_source_images_exposed_to_fit": 0,
                "morphology_truth_exposed_to_fit": 0,
                "truth_initialization_exposed_to_fit": 0,
                "development_access": 0,
                "atlas_access": 0,
                "lockbox_access": 0,
            })
    return results, endpoint_rows, audit_rows


def component_morphology(entry: Any) -> dict[str, float]:
    disk_flux = float(entry["fluxnorm_disk"])
    bulge_flux = float(entry["fluxnorm_bulge"])
    total = disk_flux + bulge_flux
    if total <= 0 or float(entry["fluxnorm_agn"]) != 0:
        raise RuntimeError("unsupported catalog morphology for scene descriptor")
    bt = bulge_flux / total
    components = []
    if disk_flux > 0:
        components.append((disk_flux / total, 1.0, math.sqrt(float(entry["a_d"]) * float(entry["b_d"])), float(entry["b_d"]) / float(entry["a_d"]), float(entry["pa_disk"])))
    if bulge_flux > 0:
        components.append((bulge_flux / total, 4.0, math.sqrt(float(entry["a_b"]) * float(entry["b_b"])), float(entry["b_b"]) / float(entry["a_b"]), float(entry["pa_bulge"])))
    radius = sum(weight * value for weight, _, value, _, _ in components)
    axis_ratio = sum(weight * value for weight, _, _, value, _ in components)
    n_eff = sum(weight * value for weight, value, _, _, _ in components)
    x = sum(weight * math.cos(math.radians(2.0 * angle)) for weight, _, _, _, angle in components)
    y = sum(weight * math.sin(math.radians(2.0 * angle)) for weight, _, _, _, angle in components)
    angle = (0.5 * math.degrees(math.atan2(y, x))) % 180.0
    return {"bulge_fraction": bt, "n_eff": n_eff, "radius": radius, "axis_ratio": axis_ratio, "angle": angle}


def angle_distance(left: float, right: float) -> float:
    raw = abs(left - right) % 180.0
    return min(raw, 180.0 - raw)


def sersic_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    distances = (
        abs(left["n_eff"] - right["n_eff"]) / 5.5,
        abs(math.log(left["radius"] / right["radius"])) / math.log(100.0),
        abs(left["axis_ratio"] - right["axis_ratio"]) / 0.9,
        angle_distance(left["angle"], right["angle"]) / 90.0,
    )
    return float(1.0 - np.mean(np.clip(distances, 0.0, 1.0)))


def aligned_morphology_similarity(source_a: np.ndarray, source_b: np.ndarray) -> float:
    aligned = []
    target_xy = np.asarray([(source_a.shape[-1] - 1) / 2.0, (source_a.shape[-2] - 1) / 2.0])
    for source in (source_a, source_b):
        center_xy = centroid(source)
        shift_yx = (target_xy[1] - center_xy[1], target_xy[0] - center_xy[0])
        bands = []
        for band in source:
            moved = image_shift(np.asarray(band, dtype=np.float64), shift=shift_yx, order=1, mode="constant", cval=0.0, prefilter=False)
            bands.append(moved / max(float(moved.sum()), np.finfo(np.float64).tiny))
        aligned.append(np.stack(bands).ravel())
    return float(np.dot(aligned[0], aligned[1]) / (np.linalg.norm(aligned[0]) * np.linalg.norm(aligned[1])))


def compute_features(
    baselines: dict[tuple[int, str], dict[str, Any]],
    catalog_entries: dict[int, list[Any]],
) -> list[dict[str, Any]]:
    with h5py.File(SCIENCE_H5, "r") as handle:
        isolated = np.asarray(handle["isolated"][list(SCENES)], dtype=np.float64)
        xy = np.asarray(handle["xy"][list(SCENES)], dtype=np.float64)
    manifests = selected_manifest_rows()
    rows = []
    for local, scene in enumerate(SCENES):
        separation = float(np.linalg.norm(xy[local, 0] - xy[local, 1]))
        recoverability = scene_metrics(isolated[local, 0], isolated[local, 1], separation)
        left = component_morphology(catalog_entries[scene][0])
        right = component_morphology(catalog_entries[scene][1])
        total_brightness = float(isolated[local].sum())
        s1 = baselines[(scene, "S1")]
        s2 = baselines[(scene, "S2")]
        p2 = baselines[(scene, "P2")]
        rows.append({
            "scene_index": scene,
            "scene_id": manifests[scene]["scene_id"],
            "overlap_fraction": recoverability["overlap_fraction"],
            "centroid_separation_psf": recoverability["centroid_separation_psf"],
            "source_flux_ratio": recoverability["symmetric_flux_ratio"],
            "total_brightness_electrons": total_brightness,
            "log10_total_brightness_electrons": math.log10(total_brightness),
            "morphology_similarity": aligned_morphology_similarity(isolated[local, 0], isolated[local, 1]),
            "color_similarity": recoverability["color_similarity_cosine"],
            "sersic_parameter_similarity": sersic_similarity(left, right),
            "bulge_fraction_difference": abs(left["bulge_fraction"] - right["bulge_fraction"]),
            "effective_radius_ratio": min(left["radius"], right["radius"]) / max(left["radius"], right["radius"]),
            "psf_sensitivity_log10_condition_gain": math.log10(s2["condition_number"] / p2["condition_number"]),
            "s1_condition_number": s1["condition_number"],
            "log10_s1_condition_number": math.log10(s1["condition_number"]),
            "s1_endpoint_multiplicity": s1["distinct_solution_classes"],
            "s1_image_diameter": max(s1["requested_image_diameter"], s1["companion_image_diameter"]),
            "s1_morphology_diameter": s1["morphology_parameter_diameter"],
            "s1_flux_allocation_diameter": s1["flux_allocation_diameter"],
            "source_a_bulge_fraction": left["bulge_fraction"],
            "source_b_bulge_fraction": right["bulge_fraction"],
            "source_a_effective_radius_arcsec": left["radius"],
            "source_b_effective_radius_arcsec": right["radius"],
            "source_a_effective_sersic_n": left["n_eff"],
            "source_b_effective_sersic_n": right["n_eff"],
            "isolated_arrays_used_for_post_fit_features_only": True,
        })
    return rows


def external_summary(result: dict[str, Any]) -> dict[str, Any]:
    geometry = result["geometry"]
    return {
        "classification": result["scientific_classification"],
        "condition_number": float(result["condition_number"]),
        "rank": int(result["rank"]),
        "nullity": int(result["nullity"]),
        "gradient_norm": float(result["gradient_norm"]),
        "distinct_solution_classes": int(geometry["distinct_solution_classes"]),
        **{key: float(geometry[key]) for key in DIAMETERS},
    }


def fractional_reduction(before: float, after: float) -> float:
    if before > 0:
        return (before - after) / before
    return 0.0 if after == 0 else -1.0


def response_rows(
    results: list[dict[str, Any]],
    baselines: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(int(result["scene_index"]), result["condition"]): external_summary(result) for result in results}
    rows = []
    for scene in SCENES:
        s1 = baselines[(scene, "S1")]
        p2 = baselines[(scene, "P2")]
        for condition in CONDITIONS:
            external = lookup[(scene, condition)]
            helpful = correction.materially_better_geometry(external, p2)
            row: dict[str, Any] = {
                "scene_index": scene,
                "condition": condition,
                "primary_response_condition": condition == PRIMARY_CONDITION,
                "photometry_helpful": helpful,
                "response_label": "Photometry Helpful" if helpful else "Photometry Not Helpful",
                "s1_classification": s1["classification"],
                "p2_classification": p2["classification"],
                "external_classification": external["classification"],
                "s1_to_p2_classification_improvement": CLASS_SCORE[p2["classification"]] - CLASS_SCORE[s1["classification"]],
                "p2_to_external_classification_improvement": CLASS_SCORE.get(external["classification"], -9) - CLASS_SCORE[p2["classification"]],
                "s1_to_external_classification_improvement": CLASS_SCORE.get(external["classification"], -9) - CLASS_SCORE[s1["classification"]],
                "s1_endpoint_classes": s1["distinct_solution_classes"],
                "p2_endpoint_classes": p2["distinct_solution_classes"],
                "external_endpoint_classes": external["distinct_solution_classes"],
                "s1_to_p2_endpoint_reduction": s1["distinct_solution_classes"] - p2["distinct_solution_classes"],
                "p2_to_external_endpoint_reduction": p2["distinct_solution_classes"] - external["distinct_solution_classes"],
                "s1_to_external_endpoint_reduction": s1["distinct_solution_classes"] - external["distinct_solution_classes"],
                "s1_condition_number": s1["condition_number"],
                "p2_condition_number": p2["condition_number"],
                "external_condition_number": external["condition_number"],
                "s1_to_p2_condition_improvement_factor": s1["condition_number"] / p2["condition_number"],
                "p2_to_external_condition_improvement_factor": p2["condition_number"] / external["condition_number"],
                "s1_to_external_condition_improvement_factor": s1["condition_number"] / external["condition_number"],
                "s1_to_p2_log10_condition_improvement": math.log10(s1["condition_number"] / p2["condition_number"]),
                "p2_to_external_log10_condition_improvement": math.log10(p2["condition_number"] / external["condition_number"]),
                "s1_to_external_log10_condition_improvement": math.log10(s1["condition_number"] / external["condition_number"]),
                "external_rank": external["rank"],
                "external_nullity": external["nullity"],
                "external_gradient_norm": external["gradient_norm"],
            }
            for label, key in (
                ("image", None),
                ("morphology", "morphology_parameter_diameter"),
                ("flux_allocation", "flux_allocation_diameter"),
            ):
                if label == "image":
                    s1_value = max(s1["requested_image_diameter"], s1["companion_image_diameter"])
                    p2_value = max(p2["requested_image_diameter"], p2["companion_image_diameter"])
                    external_value = max(external["requested_image_diameter"], external["companion_image_diameter"])
                else:
                    s1_value, p2_value, external_value = s1[key], p2[key], external[key]
                row.update({
                    f"s1_{label}_diameter": s1_value,
                    f"p2_{label}_diameter": p2_value,
                    f"external_{label}_diameter": external_value,
                    f"s1_to_p2_{label}_diameter_reduction": s1_value - p2_value,
                    f"p2_to_external_{label}_diameter_reduction": p2_value - external_value,
                    f"p2_to_external_{label}_diameter_fraction_reduction": fractional_reduction(p2_value, external_value),
                })
            row["mean_p2_to_external_diameter_fraction_reduction"] = float(np.mean([
                row["p2_to_external_image_diameter_fraction_reduction"],
                row["p2_to_external_morphology_diameter_fraction_reduction"],
                row["p2_to_external_flux_allocation_diameter_fraction_reduction"],
            ]))
            rows.append(row)
    return rows


def auc_statistic(values: np.ndarray, labels: np.ndarray) -> tuple[float, str, float]:
    positive = values[labels == 1]
    negative = values[labels == 0]
    raw = np.mean([1.0 if left > right else 0.5 if left == right else 0.0 for left in positive for right in negative])
    if raw >= 0.5:
        return float(raw), "higher predicts helpful", float(raw)
    return float(1.0 - raw), "lower predicts helpful", float(raw)


def exact_auc_pvalue(values: np.ndarray, labels: np.ndarray, observed_auc: float) -> float:
    count = int(labels.sum())
    statistics = []
    for indices in itertools.combinations(range(len(labels)), count):
        permuted = np.zeros(len(labels), dtype=int)
        permuted[list(indices)] = 1
        statistics.append(auc_statistic(values, permuted)[0])
    return float(sum(value >= observed_auc - 1e-12 for value in statistics) / len(statistics))


def bh_qvalues(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(pvalues) - reverse_rank + 1
        running = min(running, pvalues[index] * len(pvalues) / rank)
        adjusted[index] = running
    return adjusted.tolist()


class TransparentTree:
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.root: dict[str, Any] | None = None
        self.importance = {feature: 0.0 for feature in FEATURES}

    @staticmethod
    def gini(labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.0
        probability = float(labels.mean())
        return 1.0 - probability**2 - (1.0 - probability)**2

    def build(self, matrix: np.ndarray, labels: np.ndarray, depth: int, indices: np.ndarray) -> dict[str, Any]:
        prediction = int(labels.mean() >= 0.5)
        node: dict[str, Any] = {
            "samples": int(len(labels)),
            "helpful": int(labels.sum()),
            "prediction": prediction,
            "gini": self.gini(labels),
        }
        if depth >= self.max_depth or len(np.unique(labels)) == 1 or len(labels) < 2:
            return node
        best = None
        parent = self.gini(labels)
        for feature_index, feature in enumerate(FEATURES):
            unique = np.unique(matrix[:, feature_index])
            for threshold in (unique[:-1] + unique[1:]) / 2.0:
                left = matrix[:, feature_index] <= threshold
                if not left.any() or left.all():
                    continue
                weighted = (left.sum() * self.gini(labels[left]) + (~left).sum() * self.gini(labels[~left])) / len(labels)
                gain = parent - weighted
                candidate = (gain, -feature_index, -float(threshold), feature_index, float(threshold), left)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 0:
            return node
        gain, _, _, feature_index, threshold, left = best
        feature = FEATURES[feature_index]
        self.importance[feature] += len(labels) * gain
        node.update({
            "feature": feature,
            "threshold": threshold,
            "gain": gain,
            "left": self.build(matrix[left], labels[left], depth + 1, indices[left]),
            "right": self.build(matrix[~left], labels[~left], depth + 1, indices[~left]),
        })
        return node

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "TransparentTree":
        self.importance = {feature: 0.0 for feature in FEATURES}
        self.root = self.build(matrix, labels, 0, np.arange(len(labels)))
        total = sum(self.importance.values())
        if total > 0:
            self.importance = {key: value / total for key, value in self.importance.items()}
        return self

    def predict_one(self, row: np.ndarray) -> int:
        if self.root is None:
            raise RuntimeError("tree not fit")
        node = self.root
        while "feature" in node:
            index = FEATURES.index(node["feature"])
            node = node["left"] if row[index] <= node["threshold"] else node["right"]
        return int(node["prediction"])

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray([self.predict_one(row) for row in matrix], dtype=int)


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        mask = truth == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls))


def extract_rules(node: dict[str, Any], conditions: list[str] | None = None) -> list[dict[str, Any]]:
    conditions = [] if conditions is None else conditions
    if "feature" not in node:
        return [{
            "rule": " AND ".join(conditions) if conditions else "all scenes",
            "prediction": "Photometry Helpful" if node["prediction"] else "Photometry Not Helpful",
            "samples": node["samples"],
            "helpful_samples": node["helpful"],
            "empirical_precision": node["helpful"] / node["samples"] if node["prediction"] else (node["samples"] - node["helpful"]) / node["samples"],
        }]
    feature, threshold = node["feature"], node["threshold"]
    return extract_rules(node["left"], conditions + [f"{feature} <= {threshold:.6g}"]) + extract_rules(node["right"], conditions + [f"{feature} > {threshold:.6g}"])


def best_stump(matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best = None
    for feature_index, feature in enumerate(FEATURES):
        unique = np.unique(matrix[:, feature_index])
        for threshold in (unique[:-1] + unique[1:]) / 2.0:
            for high_helpful in (False, True):
                prediction = (matrix[:, feature_index] > threshold).astype(int)
                if not high_helpful:
                    prediction = 1 - prediction
                balanced = balanced_accuracy(labels, prediction)
                accuracy = float(np.mean(labels == prediction))
                candidate = (balanced, accuracy, -feature_index, -float(threshold), feature, float(threshold), high_helpful, prediction)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
    if best is None:
        raise RuntimeError("no stump split")
    balanced, accuracy, _, _, feature, threshold, high_helpful, prediction = best
    return {
        "feature": feature,
        "threshold": threshold,
        "direction": ">" if high_helpful else "<=",
        "rule": f"Photometry Helpful when {feature} {'>' if high_helpful else '<='} {threshold:.6g}",
        "resubstitution_accuracy": accuracy,
        "resubstitution_balanced_accuracy": balanced,
        "true_positive": int(np.sum((labels == 1) & (prediction == 1))),
        "false_positive": int(np.sum((labels == 0) & (prediction == 1))),
        "true_negative": int(np.sum((labels == 0) & (prediction == 0))),
        "false_negative": int(np.sum((labels == 1) & (prediction == 0))),
    }


def statistical_analysis(
    features: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    primary = {int(row["scene_index"]): row for row in responses if row["condition"] == PRIMARY_CONDITION}
    labels = np.asarray([int(primary[scene]["photometry_helpful"]) for scene in SCENES], dtype=int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise RuntimeError("binary scene stratification response has only one class")
    matrix = np.asarray([[float(row[feature]) for feature in FEATURES] for row in features], dtype=np.float64)
    condition_response = np.asarray([primary[scene]["p2_to_external_log10_condition_improvement"] for scene in SCENES])
    ranking = []
    pvalues = []
    for feature_index, feature in enumerate(FEATURES):
        values = matrix[:, feature_index]
        auc, direction, raw_auc = auc_statistic(values, labels)
        pvalue = exact_auc_pvalue(values, labels, auc)
        pvalues.append(pvalue)
        rho = spearmanr(values, condition_response).statistic
        ranking.append({
            "feature": feature,
            "orientation_free_auc": auc,
            "raw_auc_higher_helpful": raw_auc,
            "direction": direction,
            "exact_permutation_p": pvalue,
            "helpful_median": float(np.median(values[labels == 1])),
            "not_helpful_median": float(np.median(values[labels == 0])),
            "helpful_minus_not_helpful_median": float(np.median(values[labels == 1]) - np.median(values[labels == 0])),
            "spearman_rho_with_log_condition_improvement": 0.0 if not np.isfinite(rho) else float(rho),
        })
    qvalues = bh_qvalues(pvalues)
    for row, qvalue in zip(ranking, qvalues):
        row["bh_qvalue"] = qvalue
    tree = TransparentTree(max_depth=2).fit(matrix, labels)
    predictions = tree.predict(matrix)
    loo = []
    for held_out in range(len(labels)):
        mask = np.arange(len(labels)) != held_out
        fitted = TransparentTree(max_depth=2).fit(matrix[mask], labels[mask])
        loo.append(fitted.predict_one(matrix[held_out]))
    loo_array = np.asarray(loo, dtype=int)
    tree_summary = {
        "algorithm": "exhaustive_midpoint_gini_tree",
        "maximum_depth": 2,
        "root": tree.root,
        "feature_importance": tree.importance,
        "resubstitution_predictions": predictions,
        "resubstitution_accuracy": float(np.mean(predictions == labels)),
        "resubstitution_balanced_accuracy": balanced_accuracy(labels, predictions),
        "leave_one_out_predictions": loo_array,
        "leave_one_out_accuracy": float(np.mean(loo_array == labels)),
        "leave_one_out_balanced_accuracy": balanced_accuracy(labels, loo_array),
        "labels": labels,
    }
    for row in ranking:
        row["decision_tree_impurity_importance"] = tree.importance[row["feature"]]
    ranking.sort(key=lambda row: (-row["orientation_free_auc"], row["exact_permutation_p"], -abs(row["spearman_rho_with_log_condition_improvement"]), FEATURES.index(row["feature"])))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    rules = extract_rules(tree.root or {})
    stump = best_stump(matrix, labels)
    rules.insert(0, {"rule_type": "best_single_split", **stump})
    for row in rules[1:]:
        row["rule_type"] = "decision_tree_terminal_path"

    analysis_rows = []
    for feature_row, scene, label, prediction, loo_prediction in zip(features, SCENES, labels, predictions, loo_array):
        response = primary[scene]
        analysis_rows.append({
            **feature_row,
            "photometry_helpful": bool(label),
            "tree_prediction": "Photometry Helpful" if prediction else "Photometry Not Helpful",
            "leave_one_out_prediction": "Photometry Helpful" if loo_prediction else "Photometry Not Helpful",
            "p2_to_external_endpoint_reduction": response["p2_to_external_endpoint_reduction"],
            "p2_to_external_classification_improvement": response["p2_to_external_classification_improvement"],
            "p2_to_external_log10_condition_improvement": response["p2_to_external_log10_condition_improvement"],
            "mean_p2_to_external_diameter_fraction_reduction": response["mean_p2_to_external_diameter_fraction_reduction"],
        })
    frame = pd.DataFrame(analysis_rows).set_index("scene_index")
    correlation_columns = list(FEATURES) + [
        "photometry_helpful",
        "p2_to_external_endpoint_reduction",
        "p2_to_external_classification_improvement",
        "p2_to_external_log10_condition_improvement",
        "mean_p2_to_external_diameter_fraction_reduction",
    ]
    correlation = frame[correlation_columns].astype(float).corr(method="spearman").fillna(0.0)
    np.fill_diagonal(correlation.values, 1.0)
    return ranking, tree_summary, rules, analysis_rows, correlation


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return lower, upper


def scene_ranking(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in responses if row["condition"] == PRIMARY_CONDITION]
    rows.sort(key=lambda row: (
        -int(row["photometry_helpful"]),
        -int(row["p2_to_external_endpoint_reduction"]),
        -int(row["p2_to_external_classification_improvement"]),
        -float(row["p2_to_external_log10_condition_improvement"]),
        -float(row["mean_p2_to_external_diameter_fraction_reduction"]),
        int(row["scene_index"]),
    ))
    return [{"scene_rank": index, **row} for index, row in enumerate(rows, start=1)]


def short_label(name: str) -> str:
    replacements = {
        "centroid_separation_psf": "separation / PSF",
        "log10_total_brightness_electrons": "log brightness",
        "sersic_parameter_similarity": "Sérsic similarity",
        "bulge_fraction_difference": "|Δ bulge fraction|",
        "effective_radius_ratio": "effective-radius ratio",
        "psf_sensitivity_log10_condition_gain": "PSF sensitivity",
        "log10_s1_condition_number": "log S1 condition",
        "s1_endpoint_multiplicity": "S1 endpoints",
        "s1_image_diameter": "S1 image diameter",
        "s1_morphology_diameter": "S1 morphology diameter",
        "s1_flux_allocation_diameter": "S1 flux-allocation diameter",
        "source_flux_ratio": "flux ratio",
        "morphology_similarity": "morphology similarity",
        "color_similarity": "color similarity",
        "overlap_fraction": "overlap",
    }
    return replacements.get(name, name.replace("_", " "))


def save_figures(
    analysis_rows: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    tree: dict[str, Any],
    correlation: pd.DataFrame,
    scene_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"font.size": 9})
    labels = [short_label(column) for column in correlation.columns]
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    image = ax.imshow(correlation.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Spearman correlation matrix (n=8 frozen scenes)")
    fig.colorbar(image, ax=ax, label="Spearman ρ", fraction=0.04)
    fig.savefig(RUN_DIR / "figures/correlation_matrix.png", dpi=180)
    plt.close(fig)

    top = [row["feature"] for row in ranking[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), top):
        for row in analysis_rows:
            helpful = bool(row["photometry_helpful"])
            ax.scatter(float(row[feature]), float(row["p2_to_external_log10_condition_improvement"]), marker="o" if helpful else "x", s=55, color="#d95f02" if helpful else "#1b9e77")
            ax.annotate(str(row["scene_index"]), (float(row[feature]), float(row["p2_to_external_log10_condition_improvement"])), xytext=(4, 3), textcoords="offset points")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_xlabel(short_label(feature))
        ax.set_ylabel("log10(P2 condition / phot condition)")
    handles = [
        plt.Line2D([], [], color="#d95f02", marker="o", linestyle="None", label="Helpful"),
        plt.Line2D([], [], color="#1b9e77", marker="x", linestyle="None", label="Not helpful"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.savefig(RUN_DIR / "figures/top_feature_scatter_plots.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    scenes = [int(row["scene_index"]) for row in scene_rows]
    x = np.arange(len(scenes))
    colors = ["#d95f02" if row["photometry_helpful"] else "#1b9e77" for row in scene_rows]
    axes[0].bar(x, [row["p2_to_external_endpoint_reduction"] for row in scene_rows], color=colors)
    axes[0].set_ylabel("P2 → photometry endpoint reduction")
    axes[1].bar(x, [row["p2_to_external_log10_condition_improvement"] for row in scene_rows], color=colors)
    axes[1].set_ylabel("log10 condition improvement")
    axes[2].bar(x, [row["p2_to_external_classification_improvement"] for row in scene_rows], color=colors)
    axes[2].set_ylabel("classification-level improvement")
    for ax in axes:
        ax.set_xticks(x, [f"S{scene}" for scene in scenes])
        ax.axhline(0.0, color="0.4", linewidth=0.8)
    fig.savefig(RUN_DIR / "figures/scene_response_ranking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.axis("off")
    positions: dict[str, tuple[float, float]] = {}
    def draw_node(node: dict[str, Any], x_value: float, y_value: float, width: float, path: str) -> None:
        positions[path] = (x_value, y_value)
        if "feature" in node:
            text = f"{short_label(node['feature'])} ≤ {node['threshold']:.4g}\nn={node['samples']}, helpful={node['helpful']}"
        else:
            label = "Helpful" if node["prediction"] else "Not helpful"
            text = f"{label}\nn={node['samples']}, helpful={node['helpful']}"
        ax.text(x_value, y_value, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3f3f3", "edgecolor": "0.35"})
        if "feature" in node:
            for child_key, direction, child_x in (("left", "yes", x_value - width), ("right", "no", x_value + width)):
                child_y = y_value - 0.32
                ax.annotate("", xy=(child_x, child_y + 0.06), xytext=(x_value, y_value - 0.06), arrowprops={"arrowstyle": "->", "color": "0.35"})
                ax.text((x_value + child_x) / 2.0, (y_value + child_y) / 2.0 + 0.02, direction, ha="center", va="center")
                draw_node(node[child_key], child_x, child_y, width / 2.2, path + child_key[0])
    draw_node(tree["root"], 0.5, 0.86, 0.25, "r")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Depth-2 transparent decision tree")
    fig.savefig(RUN_DIR / "figures/decision_tree.png", dpi=180)
    plt.close(fig)


def tree_text(node: dict[str, Any], indent: str = "") -> str:
    if "feature" not in node:
        label = "Photometry Helpful" if node["prediction"] else "Photometry Not Helpful"
        return f"{indent}predict {label} (n={node['samples']}, helpful={node['helpful']})\n"
    output = f"{indent}if {node['feature']} <= {node['threshold']:.6g}:\n"
    output += tree_text(node["left"], indent + "  ")
    output += f"{indent}else:\n"
    output += tree_text(node["right"], indent + "  ")
    return output


def final_report(
    features: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    tree: dict[str, Any],
    rules: list[dict[str, Any]],
    ranked_scenes: list[dict[str, Any]],
    start_snapshot: dict[str, Any],
    end_snapshot: dict[str, Any],
    runtime: float,
) -> str:
    primary = [row for row in responses if row["condition"] == PRIMARY_CONDITION]
    secondary = [row for row in responses if row["condition"] != PRIMARY_CONDITION]
    helpful = sum(bool(row["photometry_helpful"]) for row in primary)
    secondary_helpful = sum(bool(row["photometry_helpful"]) for row in secondary)
    concordant = sum(
        next(row for row in primary if int(row["scene_index"]) == scene)["photometry_helpful"]
        == next(row for row in secondary if int(row["scene_index"]) == scene)["photometry_helpful"]
        for scene in SCENES
    )
    lower, upper = clopper_pearson(helpful, len(primary))
    top = ranking[:5]
    integrity = (
        start_snapshot["head"] == end_snapshot["head"]
        and start_snapshot["readme_sha256"] == end_snapshot["readme_sha256"]
        and start_snapshot["authoritative_report_hashes"] == end_snapshot["authoritative_report_hashes"]
        and not end_snapshot["git_index_entries"]
    )
    best_rule = rules[0]
    report = f"""# {CAMPAIGN} final report

## Outcome

**SCENE_STRATIFICATION_EXPLORATORY_RULE_IDENTIFIED**

Under the prespecified total-photometry primary response, photometry was helpful in **{helpful}/8 scenes ({100*helpful/8:.1f}%; exact 95% binomial CI {100*lower:.1f}%–{100*upper:.1f}%)**. Per-band photometry was helpful in **{secondary_helpful}/8** scenes, and the two conditions agreed on helpful/not-helpful status for **{concordant}/8** scenes.

The sample is the complete frozen eight-scene population but is too small for a stable general deployment rule. All feature p-values are exact conditional label-permutation results; multiplicity-adjusted q-values and leave-one-out tree performance are reported explicitly.

## Scene ranking and S1 → P2 → external response

Primary response is 5%-uncertainty total source photometry. Positive endpoint/classification/log-condition changes denote improvement.

| Rank | Scene | Helpful | Endpoint classes S1→P2→Ext | Classification S1→P2→Ext | P2→Ext log10 condition gain | P2→Ext mean diameter fraction reduction |
| ---: | ---: | --- | --- | --- | ---: | ---: |
"""
    for row in ranked_scenes:
        report += f"| {row['scene_rank']} | {row['scene_index']} | {'yes' if row['photometry_helpful'] else 'no'} | {row['s1_endpoint_classes']}→{row['p2_endpoint_classes']}→{row['external_endpoint_classes']} | {row['s1_classification']}→{row['p2_classification']}→{row['external_classification']} | {row['p2_to_external_log10_condition_improvement']:.4f} | {row['mean_p2_to_external_diameter_fraction_reduction']:.4f} |\n"
    report += """

Full condition-number factors and image, morphology, and flux-allocation diameter transitions for both total and per-band photometry are in `tables/photometry_response.csv`.

## Feature ranking

| Rank | Feature | Oriented AUC | Direction | Exact p | BH q | Helpful median | Not-helpful median | ρ with log-condition gain | Tree importance |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for row in top:
        report += f"| {row['rank']} | {row['feature']} | {row['orientation_free_auc']:.3f} | {row['direction']} | {row['exact_permutation_p']:.4f} | {row['bh_qvalue']:.4f} | {row['helpful_median']:.5g} | {row['not_helpful_median']:.5g} | {row['spearman_rho_with_log_condition_improvement']:.3f} | {row['decision_tree_impurity_importance']:.3f} |\n"
    report += f"""

No adjusted result is treated as confirmatory. The ranking is descriptive feature importance over eight observations.

## Decision tree

```text
{tree_text(tree['root'])}```

- Resubstitution accuracy: **{tree['resubstitution_accuracy']:.3f}**; balanced accuracy: **{tree['resubstitution_balanced_accuracy']:.3f}**.
- Leave-one-out accuracy: **{tree['leave_one_out_accuracy']:.3f}**; leave-one-out balanced accuracy: **{tree['leave_one_out_balanced_accuracy']:.3f}**.
- Best simple rule: **{best_rule['rule']}**. In-sample confusion: TP={best_rule['true_positive']}, FP={best_rule['false_positive']}, TN={best_rule['true_negative']}, FN={best_rule['false_negative']}; balanced accuracy={best_rule['resubstitution_balanced_accuracy']:.3f}.

These predictor cut points are acquisition heuristics only. They do not modify the inherited uniqueness, diameter, optimizer, or classification thresholds.

## When should additional photometric information be acquired?

Acquire total external source photometry for scenes that satisfy the learned low-complexity rule **`{best_rule['rule'].replace('Photometry Helpful when ', '')}`**, provided a current S1/P2 structural fit supplies the required descriptor. Do not acquire it routinely for the complementary stratum: across this frozen population only {helpful}/8 scenes met the unchanged material-improvement rule, and the leave-one-out balanced accuracy of the depth-2 rule is {tree['leave_one_out_balanced_accuracy']:.3f}. Use per-band measurements only when they are already available; their helpful/not-helpful decisions agreed with total photometry in {concordant}/8 scenes and were not the prespecified primary acquisition.

This answer is conditional on simulated CatSim/BTK scenes, exact source coordinates, known PSF, Level-5 support, and 5% external errors. It is not validated for real-survey acquisition.

## Exactly one next experiment

Run **Thayer-External-Photometry-Rule-Validation-v0**: preregister the best single rule above and prospectively test it, without refitting the rule, on a fresh training-only simulated cohort balanced across the rule boundary; acquire only 5% total source photometry, retain the same Level-5 solver and P2 control, and use rule precision plus P2-to-photometry endpoint reduction as the primary validation endpoints.

## Provenance and integrity

- All six available authoritative reports were read. The requested `Thayer-Project-Synthesis-v1` artifact is not present under that title; the authoritative Flux-Free report itself records that absence.
- Scene 0 and 6 measurements/fits were reused from the convergence-correction authority. The other six scenes used the frozen measurement seed rule, four starts, and 500-evaluation ceiling.
- Isolated training-source arrays and catalog morphology were used only for post-fit scene descriptors, never as solver inputs.
- Integrity status: **{'PASS' if integrity else 'FAIL'}**. HEAD, README, authoritative report hashes, and the empty staged index were unchanged.
- Development, Atlas, and lockbox access: zero. Neural networks loaded/trained: zero. Nothing was staged or committed.
- Runtime: **{runtime:.3f} seconds ({runtime/60:.2f} minutes)**.
"""
    return report


def main() -> None:
    started = time.perf_counter()
    started_utc = now()
    ensure_output_directories()
    start_snapshot = snapshot()
    if start_snapshot["git_index_entries"]:
        raise RuntimeError("staged index is not empty")
    fresh_json(RUN_DIR / "manifests/start_snapshot.json", start_snapshot)
    input_paths = (
        Path(__file__), PROTOCOL_PATH, S1_METRICS, P2_METRICS, DEFINITIONS,
        SCENE_MANIFEST, SCIENCE_H5, CATALOG_PATH, PRIOR_MEASUREMENTS,
        REPO / "src/model9_structured.py", REPO / "src/model9_optimizer.py",
        REPO / "scripts/run_thayer_external_photometry_preflight_v0.py",
        REPO / "scripts/run_thayer_external_photometry_convergence_correction_v0.py",
        *AUTHORITATIVE_REPORTS,
    )
    fresh_json(RUN_DIR / "manifests/input_hashes.json", {
        "campaign": CAMPAIGN,
        "frozen_at_utc": started_utc,
        "files": {str(path.relative_to(REPO)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in input_paths},
        "scenes": SCENES,
        "new_fit_scenes": NEW_SCENES,
        "reused_fit_scenes": (0, 6),
        "family": FAMILY_BULGE_DISK,
        "conditions": CONDITIONS,
        "primary_condition": PRIMARY_CONDITION,
        "starts": STARTS,
        "max_nfev": MAX_NFEV,
        "relative_sigma": RELATIVE_SIGMA,
        "measurement_seed_rule": "2026071805 + 100*scene_index + condition_index",
    })

    baselines = load_historical_baselines()
    measurements, measurement_rows, catalog_entries = generate_measurements()
    fresh_csv(RUN_DIR / "tables/external_photometry_measurements.csv", measurement_rows)
    results, endpoints, audits = execute_fits(measurements)
    fresh_csv(RUN_DIR / "tables/multistart_endpoints.csv", endpoints)
    fresh_csv(RUN_DIR / "tables/oracle_information_audit.csv", audits)

    features = compute_features(baselines, catalog_entries)
    responses = response_rows(results, baselines)
    ranking, tree, rules, analysis_rows, correlation = statistical_analysis(features, responses)
    ranked_scenes = scene_ranking(responses)
    fresh_csv(RUN_DIR / "tables/scene_features.csv", features)
    fresh_csv(RUN_DIR / "tables/photometry_response.csv", responses)
    fresh_csv(RUN_DIR / "tables/scene_ranking.csv", ranked_scenes)
    fresh_csv(RUN_DIR / "tables/feature_ranking.csv", ranking)
    fresh_json(RUN_DIR / "tables/decision_tree.json", tree)
    fresh_csv(RUN_DIR / "tables/decision_rules.csv", rules)
    fresh_csv(RUN_DIR / "tables/summary_table.csv", analysis_rows)
    correlation.to_csv(RUN_DIR / "tables/correlation_matrix.csv", float_format="%.12g")
    save_figures(analysis_rows, ranking, tree, correlation, ranked_scenes)

    end_snapshot = snapshot()
    fresh_json(RUN_DIR / "manifests/end_snapshot.json", end_snapshot)
    runtime = time.perf_counter() - started
    report = final_report(features, responses, ranking, tree, rules, ranked_scenes, start_snapshot, end_snapshot, runtime)
    fresh_text(RUN_DIR / "reports/final_report.md", report)
    produced = sorted(path for path in RUN_DIR.rglob("*") if path.is_file())
    fresh_json(RUN_DIR / "manifests/final_manifest.json", {
        "campaign": CAMPAIGN,
        "completed_at_utc": now(),
        "runtime_seconds": runtime,
        "files": {str(path.relative_to(RUN_DIR)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in produced},
        "head_unchanged": start_snapshot["head"] == end_snapshot["head"],
        "readme_unchanged": start_snapshot["readme_sha256"] == end_snapshot["readme_sha256"],
        "authoritative_reports_unchanged": start_snapshot["authoritative_report_hashes"] == end_snapshot["authoritative_report_hashes"],
        "git_index_empty": not end_snapshot["git_index_entries"],
        "protected_data_access": {"development": 0, "atlas": 0, "lockbox": 0},
        "commits_created": 0,
    })
    print(json.dumps({
        "event": "campaign_complete",
        "run_dir": str(RUN_DIR),
        "runtime_seconds": runtime,
        "helpful_scenes_total_photometry": [row["scene_index"] for row in responses if row["condition"] == PRIMARY_CONDITION and row["photometry_helpful"]],
        "tree_leave_one_out_balanced_accuracy": tree["leave_one_out_balanced_accuracy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
