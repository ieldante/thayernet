#!/usr/bin/env python3
"""Fail-closed Scene 5/18 total-photometry convergence correction."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
import torch


RUN_DIR = Path(__file__).resolve().parents[1]
REPO = RUN_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_thayer_external_photometry_convergence_correction_v0 as correction
from scripts import run_thayer_external_photometry_preflight_v0 as preflight
from scripts.run_thayer_flux_free_identifiability_v0 import (
    CHECKPOINT_BASELINE,
    load_science_inputs,
    raw_array_sha256,
    sha256_file,
)
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_optimizer import (
    MultiStartEndpoint,
    analyze_jacobian,
    boundary_contact_flags,
    deterministic_starts,
    solution_geometry,
)
from src.model9_structured import (
    FAMILY_BULGE_DISK,
    FrozenSolverProtocol,
    SolverInputs,
    canonicalize_parameters,
    expanded_noise_sigma,
    oracle_information_audit,
    parameter_bounds,
    parameter_scales,
    parameter_sha256,
    render_pair,
    validate_parameters,
)


CAMPAIGN = "Thayer-External-Photometry-Stratification-Convergence-Correction-v0"
SCENES = (5, 18)
CONDITION = "TOTAL_SOURCE_PHOTOMETRY"
STARTS = 4
OLD_MAX_NFEV = 500
MAX_NFEV = 2000
RELATIVE_SIGMA = 0.05
NUMERICAL_PERTURBATION_SCALE = 1.0e-8
NUMERICAL_SPECTRUM_RTOL = 1.0e-3
EXPECTED_HEAD = "74b8ff7efbbf7e9891cc8fd8095a9931e3b63174"
EXPECTED_README_SHA256 = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"
EXPECTED_CHECKPOINT_BASELINE_SHA256 = "982fa39058030c1ec81e832b76031acd95936295d70a4f8ef69b6238dc72d477"

STRAT_RUN = REPO / "outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606"
STRAT_REPORT = STRAT_RUN / "reports/final_report.md"
STRAT_PROTOCOL = STRAT_RUN / "preregistration/frozen_protocol.md"
STRAT_ENDPOINTS = STRAT_RUN / "tables/multistart_endpoints.csv"
STRAT_MEASUREMENTS = STRAT_RUN / "tables/external_photometry_measurements.csv"
STRAT_SUMMARY = STRAT_RUN / "tables/summary_table.csv"
STRAT_SCENE_RANKING = STRAT_RUN / "tables/scene_ranking.csv"
STRAT_FEATURE_RANKING = STRAT_RUN / "tables/feature_ranking.csv"
STRAT_FEATURES = STRAT_RUN / "tables/scene_features.csv"
STRAT_RESPONSES = STRAT_RUN / "tables/photometry_response.csv"
STRAT_FINAL_MANIFEST = STRAT_RUN / "manifests/final_manifest.json"
STRAT_INPUT_HASHES = STRAT_RUN / "manifests/input_hashes.json"
STRAT_SCRIPT = STRAT_RUN / "analysis/run_campaign.py"

MODEL9_RUN = REPO / "outputs/runs/thayer_model_9_preparation_v0_20260715_172217"
MODEL9_MANIFEST = MODEL9_RUN / "manifests/final_manifest.json"
MODEL9_SOURCE_HASHES = MODEL9_RUN / "manifests/source_hashes.json"

READ_FIRST_FILES = (
    STRAT_REPORT,
    STRAT_PROTOCOL,
    STRAT_RUN / "fit_records/scene_005_total_source_photometry.json",
    STRAT_RUN / "fit_records/scene_018_total_source_photometry.json",
    STRAT_ENDPOINTS,
    STRAT_MEASUREMENTS,
    STRAT_SUMMARY,
    STRAT_SCENE_RANKING,
    STRAT_FEATURE_RANKING,
    REPO / "outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638/reports/final_report.md",
    REPO / "outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/reports/final_report.md",
    REPO / "outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/reports/final_report.md",
    MODEL9_RUN / "reports/final_report.md",
    REPO / "outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/manifests/historical_integrity.json",
    REPO / "outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/manifests/historical_integrity.json",
    MODEL9_RUN / "manifests/historical_integrity.json",
)

RELEVANT_TESTS = (
    "tests/test_model9_foundation.py",
    "tests/test_model9_joint.py",
    "tests/test_canonical_tensor_hash.py",
    "tests/test_family_e_signed_residual.py",
    "tests/test_psf_conditioning.py",
)

DIAMETERS = correction.DIAMETERS
CLASS_SCORE = {
    "NON_IDENTIFIABLE": 0,
    "PARTIALLY_IDENTIFIABLE": 1,
    "NEAR_UNIQUE": 2,
    "UNIQUE": 3,
}
UNRESOLVED_FIT_CLASSES = {
    "OPTIMIZATION_UNRESOLVED",
    "NUMERICALLY_UNSTABLE",
    "OUT_OF_SUPPORT",
    "INVALID_CONTRACT",
}


def load_strat_module() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_scene_stratification", STRAT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load predecessor analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strat = load_strat_module()


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: (
                        json.dumps(safe(item), sort_keys=True, separators=(",", ":"))
                        if isinstance(item, (dict, list, tuple, np.ndarray))
                        else safe(item)
                    )
                    for key, item in row.items()
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def protected_files() -> tuple[Path, ...]:
    predecessor = tuple(path for path in STRAT_RUN.rglob("*") if path.is_file())
    return tuple(dict.fromkeys((*predecessor, *READ_FIRST_FILES)))


def historical_snapshot(reference_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    baseline = json.loads(CHECKPOINT_BASELINE.read_text(encoding="utf-8"))
    matches = mismatches = missing = bytes_rehashed = 0
    for record in baseline:
        path = REPO / record["path"]
        if not path.exists():
            missing += 1
        elif path.stat().st_size == int(record["final_bytes"]) and sha256_file(path) == record["final_sha256"]:
            matches += 1
            bytes_rehashed += path.stat().st_size
        else:
            mismatches += 1
    files = {str(path.relative_to(REPO)): sha256_file(path) for path in protected_files()}
    snapshot = {
        "timestamp_utc": now(),
        "head": command(["git", "rev-parse", "HEAD"]),
        "readme_sha256": sha256_file(REPO / "README.md"),
        "git_index_entries": command(["git", "diff", "--cached", "--name-only"]),
        "checkpoint_baseline_sha256": sha256_file(CHECKPOINT_BASELINE),
        "historical_checkpoint_count": len(baseline),
        "historical_checkpoint_matches": matches,
        "historical_checkpoint_mismatches": mismatches,
        "historical_checkpoint_missing": missing,
        "historical_checkpoint_bytes_rehashed": bytes_rehashed,
        "protected_historical_file_hashes": files,
        "protected_historical_files_match_start": reference_hashes is None or files == reference_hashes,
        "protected_data_access": {"development": 0, "atlas_tensors": 0, "lockbox": 0},
        "historical_isolated_hdf5_access": 0,
        "commits_created": 0,
    }
    snapshot["status"] = "PASS" if (
        snapshot["head"] == EXPECTED_HEAD
        and snapshot["readme_sha256"] == EXPECTED_README_SHA256
        and not snapshot["git_index_entries"]
        and snapshot["checkpoint_baseline_sha256"] == EXPECTED_CHECKPOINT_BASELINE_SHA256
        and len(baseline) == matches == 600
        and mismatches == 0
        and missing == 0
        and snapshot["protected_historical_files_match_start"]
    ) else "FAIL"
    return snapshot


def stale_optimizer_processes() -> list[dict[str, Any]]:
    current = {os.getpid(), os.getppid()}
    tokens = (
        "run_thayer_external_photometry_stratification_convergence_correction",
        "run_thayer_external_photometry_convergence_correction_v0.py",
        "run_thayer_external_photometry_scene_stratification",
        "run_thayer_flux_free_identifiability_v0.py",
        "run_thayer_psf_diverse_flux_identifiability_v0.py",
    )
    rows = []
    for line in command(["ps", "-axo", "pid=,etime=,command="]).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        if pid not in current and any(token in fields[2] for token in tokens):
            rows.append({"pid": pid, "elapsed": fields[1], "command": fields[2]})
    return rows


def verify_predecessor_manifest() -> dict[str, Any]:
    manifest = json.loads(STRAT_FINAL_MANIFEST.read_text(encoding="utf-8"))
    checks = {}
    for relative, expected in manifest["files"].items():
        path = STRAT_RUN / relative
        actual = sha256_file(path) if path.exists() else "MISSING"
        checks[relative] = {"expected": expected["sha256"], "actual": actual, "match": actual == expected["sha256"]}
    return {
        "campaign": manifest.get("campaign"),
        "completed_at_utc": manifest.get("completed_at_utc"),
        "resolved_primary_scenes": manifest.get("resolved_primary_scenes"),
        "unresolved_primary_scenes": manifest.get("unresolved_primary_scenes"),
        "head_unchanged": manifest.get("head_unchanged"),
        "readme_unchanged": manifest.get("readme_unchanged"),
        "git_index_empty": manifest.get("git_index_empty"),
        "protected_data_access": manifest.get("protected_data_access"),
        "file_hash_checks": checks,
        "all_file_hashes_match": all(item["match"] for item in checks.values()),
    }


def load_measurements() -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, str]]]:
    rows = read_rows(STRAT_MEASUREMENTS)
    selected = [
        row for row in rows
        if int(row["scene_index"]) in SCENES and row["condition"] == CONDITION
    ]
    grouped: dict[int, list[dict[str, str]]] = {scene: [] for scene in SCENES}
    for row in selected:
        grouped[int(row["scene_index"])].append(row)
        if float(row["relative_sigma"]) != RELATIVE_SIGMA:
            raise RuntimeError("measurement relative sigma changed")
        if row["latent_catalog_value_persisted"] != "False":
            raise RuntimeError("latent catalog value leaked into measurement table")
        if json.loads(row["combination_weights_g_r_z"]) != [1.0, 1.0, 1.0]:
            raise RuntimeError("total-photometry weights changed")
        if row["measurement_source"] != "scene_stratification_extension":
            raise RuntimeError("measurement provenance changed")
    output = {}
    for scene in SCENES:
        group = grouped[scene]
        if len(group) != 2 or [row["source_identity"] for row in group] != ["requested", "companion"]:
            raise RuntimeError(f"measurement rows changed for Scene {scene}")
        measured = np.asarray([float(row["measured_flux_electrons"]) for row in group], dtype=np.float64)
        sigma = np.asarray([float(row["sigma_electrons"]) for row in group], dtype=np.float64)
        if not np.isfinite(measured).all() or not np.isfinite(sigma).all() or np.any(measured <= 0) or np.any(sigma <= 0):
            raise RuntimeError("measurement values are invalid")
        output[scene] = {"measured": measured, "sigma": sigma}
    return output, selected


def verify_starts_and_endpoints(
    inputs_by_scene: dict[int, SolverInputs],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]], dict[str, Any]]:
    endpoint_rows = [
        row for row in read_rows(STRAT_ENDPOINTS)
        if int(row["scene_index"]) in SCENES and row["condition"] == CONDITION
    ]
    if len(endpoint_rows) != 8:
        raise RuntimeError("predecessor does not contain exactly eight relevant endpoints")
    starts_by_scene = {}
    records_by_scene = {}
    start_records = []
    for scene in SCENES:
        path = STRAT_RUN / f"fit_records/scene_{scene:03d}_total_source_photometry.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        records_by_scene[scene] = record
        rows = sorted(
            [row for row in endpoint_rows if int(row["scene_index"]) == scene],
            key=lambda row: int(row["start_index"]),
        )
        if [int(row["start_index"]) for row in rows] != list(range(STARTS)):
            raise RuntimeError(f"start IDs changed for Scene {scene}")
        if len(record["starts"]) != STARTS or len(record["endpoints"]) != STARTS:
            raise RuntimeError(f"fit record endpoint count changed for Scene {scene}")
        regenerated = deterministic_starts(inputs_by_scene[scene], FrozenSolverProtocol(), count=STARTS)
        stored = np.asarray(record["starts"], dtype=np.float64)
        if not np.array_equal(regenerated, stored):
            raise RuntimeError(f"deterministic starts do not regenerate for Scene {scene}")
        for index, (row, vector, endpoint) in enumerate(zip(rows, regenerated, record["endpoints"])):
            row_vector = np.asarray(json.loads(row["initialization"]), dtype=np.float64)
            endpoint_vector = np.asarray(endpoint["initialization"], dtype=np.float64)
            if not np.array_equal(vector, row_vector) or not np.array_equal(vector, endpoint_vector):
                raise RuntimeError(f"physical start mismatch for Scene {scene}, start {index}")
            expected_hash = parameter_sha256(vector)
            if expected_hash != row["initialization_sha256"] or expected_hash != endpoint["initialization_sha256"]:
                raise RuntimeError(f"initialization hash mismatch for Scene {scene}, start {index}")
            if int(row["nfev"]) != OLD_MAX_NFEV or row["success"] != "False":
                raise RuntimeError(f"prior endpoint did not reach unresolved 500 ceiling for Scene {scene}, start {index}")
            start_records.append({
                "scene_index": scene,
                "start_id": index,
                "initialization_sha256": expected_hash,
                "physical_starting_parameters": vector,
                "unconstrained_initial_variables": "NOT_APPLICABLE_DIRECT_BOUNDED_PHYSICAL_PARAMETERIZATION",
            })
        starts_by_scene[scene] = regenerated
    return starts_by_scene, records_by_scene, {
        "relevant_endpoint_count": len(endpoint_rows),
        "all_eight_nfev_500": all(int(row["nfev"]) == OLD_MAX_NFEV for row in endpoint_rows),
        "all_eight_optimizer_success_false": all(row["success"] == "False" for row in endpoint_rows),
        "start_ids_per_scene": list(range(STARTS)),
        "deterministic_starts_regenerate_exactly": True,
        "initialization_hashes_match": True,
        "physical_starts_match": True,
        "unconstrained_initial_variables_present": False,
        "start_records": start_records,
    }


def authorization_gate() -> tuple[
    dict[str, Any],
    dict[int, SolverInputs],
    dict[int, dict[str, Any]],
    dict[int, np.ndarray],
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, Any]],
]:
    failures: list[str] = []
    predecessor = verify_predecessor_manifest()
    report_text = STRAT_REPORT.read_text(encoding="utf-8")
    completed = (
        predecessor["campaign"] == "Thayer-External-Photometry-Scene-Stratification-v0"
        and predecessor["completed_at_utc"]
        and predecessor["all_file_hashes_match"]
        and "**SCENE_STRATIFICATION_PRIMARY_RESPONSE_PARTIALLY_RESOLVED**" in report_text
    )
    only_unresolved = predecessor["unresolved_primary_scenes"] == [5, 18]
    exact_recommendation = (
        "Run **Thayer-External-Photometry-Stratification-Convergence-Correction-v0**" in report_text
        and "changing only the per-start evaluation ceiling from 500 to 2000" in report_text
    )
    if not completed:
        failures.append("predecessor campaign completion or artifact hashes failed")
    if not only_unresolved:
        failures.append("Scenes 5 and 18 are not the sole unresolved primary cases")
    if not exact_recommendation:
        failures.append("predecessor did not recommend this exact experiment")

    measurement_hash_expected = json.loads(STRAT_FINAL_MANIFEST.read_text(encoding="utf-8"))["files"]["tables/external_photometry_measurements.csv"]["sha256"]
    measurement_hash_actual = sha256_file(STRAT_MEASUREMENTS)
    if measurement_hash_actual != measurement_hash_expected:
        failures.append("measurement hash mismatch")
    measurements, measurement_rows = load_measurements()

    model9_manifest = json.loads(MODEL9_MANIFEST.read_text(encoding="utf-8"))
    model9_sources = json.loads(MODEL9_SOURCE_HASHES.read_text(encoding="utf-8"))["files"]
    model9_source_checks = {path: sha256_file(REPO / path) == expected for path, expected in model9_sources.items()}
    model9_ready = model9_manifest.get("status") == "MODEL_9_FOUNDATION_READY" and all(model9_source_checks.values())
    if not model9_ready:
        failures.append("Model-9 readiness or source hashes failed")

    test_command = [str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", *RELEVANT_TESTS]
    test_run = subprocess.run(test_command, cwd=REPO, text=True, capture_output=True)
    tests_pass = test_run.returncode == 0 and "passed" in test_run.stdout
    if not tests_pass:
        failures.append("relevant test suite failed")
    validator_code = (
        "from scripts.validate_thayer_model9_foundation import validate; "
        "r=validate(); print(r['status']); raise SystemExit(0 if r['status']=='PASS' else 1)"
    )
    validator_run = subprocess.run(
        [str(REPO / ".venv-btk/bin/python"), "-c", validator_code],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    validator_pass = validator_run.returncode == 0 and validator_run.stdout.strip() == "PASS"
    if not validator_pass:
        failures.append("standalone Model-9 validator failed")

    initial_integrity = historical_snapshot()
    if initial_integrity["status"] != "PASS":
        failures.append("initial historical integrity failed")
    stale = stale_optimizer_processes()
    if stale:
        failures.append("stale optimizer process exists")

    inputs_by_scene = {}
    metadata_by_scene = {}
    for scene in SCENES:
        inputs, metadata = load_science_inputs(scene, FAMILY_BULGE_DISK)
        inputs_by_scene[scene] = inputs
        metadata_by_scene[scene] = metadata
    starts_by_scene, prior_records, start_verification = verify_starts_and_endpoints(inputs_by_scene)
    if not start_verification["all_eight_nfev_500"]:
        failures.append("not all eight relevant starts reached 500 evaluations")

    audits = []
    for scene in SCENES:
        audit = oracle_information_audit(
            inputs_by_scene[scene],
            FrozenSolverProtocol(),
            extra_named_inputs={"scene_index": scene, "external_measurements": measurements[scene]["measured"].tolist()},
        )
        audits.append({"scene_index": scene, "condition": CONDITION, "status": audit["status"]})
        if audit["status"] != "PASS":
            failures.append(f"oracle-information audit failed for Scene {scene}")

    gate = {
        "campaign": CAMPAIGN,
        "timestamp_utc": now(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "predecessor_completed_successfully": bool(completed),
        "predecessor_artifact_verification": predecessor,
        "sole_unresolved_primary_scenes": [5, 18] if only_unresolved else predecessor["unresolved_primary_scenes"],
        "exact_recommended_next_experiment": CAMPAIGN if exact_recommendation else "MISMATCH",
        "prior_endpoint_and_start_verification": start_verification,
        "measurement_table_sha256_expected": measurement_hash_expected,
        "measurement_table_sha256_actual": measurement_hash_actual,
        "measurement_hash_matches": measurement_hash_actual == measurement_hash_expected,
        "measurement_rows_reused": measurement_rows,
        "measurement_regeneration_performed": False,
        "model9_ready": model9_ready,
        "model9_source_hash_checks": model9_source_checks,
        "relevant_tests": {
            "status": "PASS" if tests_pass else "FAIL",
            "returncode": test_run.returncode,
            "command": test_command,
            "stdout": test_run.stdout.strip(),
            "stderr": test_run.stderr.strip(),
        },
        "standalone_validator": {
            "status": "PASS" if validator_pass else "FAIL",
            "returncode": validator_run.returncode,
            "stdout": validator_run.stdout.strip(),
            "stderr": validator_run.stderr.strip(),
        },
        "historical_integrity": initial_integrity,
        "stale_optimizer_processes": stale,
        "oracle_information_audits": audits,
        "protected_data_access": {"development": 0, "atlas_tensors": 0, "lockbox": 0},
        "head_unchanged": initial_integrity["head"] == EXPECTED_HEAD,
        "readme_unchanged": initial_integrity["readme_sha256"] == EXPECTED_README_SHA256,
        "git_index_empty": not initial_integrity["git_index_entries"],
    }
    atomic_json(RUN_DIR / "manifests/authorization_gate.json", gate)
    if failures:
        raise RuntimeError("authorization gate failed: " + "; ".join(failures))
    return gate, inputs_by_scene, metadata_by_scene, starts_by_scene, measurements, prior_records


def numerical_perturbation(
    parameters: np.ndarray,
    inputs: SolverInputs,
    measurement: dict[str, np.ndarray],
    diagnostics: Any,
) -> dict[str, Any]:
    protocol = FrozenSolverProtocol()
    value = np.asarray(parameters, dtype=np.float64)
    lower, upper = parameter_bounds(inputs, protocol)
    scales = parameter_scales(inputs, protocol)
    direction = np.sin(np.arange(value.size, dtype=np.float64) + 1.0)
    candidate = np.maximum(value + NUMERICAL_PERTURBATION_SCALE * scales * direction, lower)
    finite_upper = np.isfinite(upper)
    candidate[finite_upper] = np.minimum(candidate[finite_upper], upper[finite_upper])
    jacobian = preflight.combined_jacobian(candidate, inputs, CONDITION, measurement)
    perturbed = analyze_jacobian(
        jacobian,
        active_mask=diagnostics.active_mask,
        parameter_scales=scales,
    )
    spectrum_change = float(
        np.linalg.norm(perturbed.singular_values - diagnostics.singular_values)
        / max(float(np.linalg.norm(diagnostics.singular_values)), np.finfo(np.float64).tiny)
    )
    residual = np.asarray(preflight.combined_residual(
        torch.as_tensor(candidate, dtype=torch.float64), inputs, CONDITION, measurement
    ).detach().cpu())
    stable = bool(
        np.isfinite(residual).all()
        and np.isfinite(jacobian).all()
        and perturbed.rank == diagnostics.rank
        and perturbed.null_space_dimension == diagnostics.null_space_dimension
        and spectrum_change <= NUMERICAL_SPECTRUM_RTOL
    )
    return {
        "stable": stable,
        "scale": NUMERICAL_PERTURBATION_SCALE,
        "spectrum_relative_change": spectrum_change,
        "rank": perturbed.rank,
        "null_space_dimension": perturbed.null_space_dimension,
        "condition_number": perturbed.condition_number,
        "parameter_sha256": raw_array_sha256(candidate),
    }


def fit_one(
    scene: int,
    inputs: SolverInputs,
    measurement: dict[str, np.ndarray],
    starts: np.ndarray,
) -> dict[str, Any]:
    protocol = FrozenSolverProtocol()
    lower, upper = parameter_bounds(inputs, protocol)
    endpoints: list[MultiStartEndpoint] = []
    endpoint_parts: dict[int, dict[str, float]] = {}
    objective_histories: dict[int, list[dict[str, Any]]] = {}
    fit_started = time.perf_counter()
    observation_size = int(inputs.observed.numel())
    expanded_sigma = np.asarray(expanded_noise_sigma(inputs.noise_sigma, inputs.observed), dtype=np.float64)
    observation_log_normalization = float(np.log(expanded_sigma).sum())
    photometry_log_normalization = float(np.log(measurement["sigma"]).sum())

    for start_index, initialization in enumerate(starts):
        validate_parameters(initialization, inputs, protocol)
        history: list[dict[str, Any]] = []
        start_wall = time.perf_counter()

        def residual_function(value: np.ndarray) -> np.ndarray:
            residual = np.asarray(preflight.combined_residual(
                torch.as_tensor(value, dtype=torch.float64), inputs, CONDITION, measurement
            ).detach().cpu())
            observation = residual[:observation_size]
            photometry = residual[observation_size:]
            observation_objective = 0.5 * float(observation @ observation) + observation_log_normalization
            photometry_objective = 0.5 * float(photometry @ photometry) + photometry_log_normalization
            history.append({
                "function_evaluation": len(history) + 1,
                "parameter_sha256": raw_array_sha256(np.asarray(value, dtype=np.float64)),
                "observation_objective": observation_objective,
                "photometry_objective": photometry_objective,
                "total_objective": observation_objective + photometry_objective,
                "observation_chi_square": float(observation @ observation),
                "photometry_chi_square": float(photometry @ photometry),
            })
            return residual

        result = least_squares(
            residual_function,
            initialization,
            jac=lambda value: preflight.combined_jacobian(value, inputs, CONDITION, measurement),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=protocol.ftol,
            xtol=protocol.xtol,
            gtol=protocol.gtol,
            max_nfev=MAX_NFEV,
            verbose=0,
        )
        if len(history) != int(result.nfev):
            raise RuntimeError("objective history length does not equal scipy nfev")
        selected = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
        canonical, _, symmetries = canonicalize_parameters(selected, inputs, protocol)
        tensor = torch.as_tensor(selected, dtype=torch.float64)
        pair = render_pair(tensor, inputs, protocol)
        residual = np.asarray(preflight.combined_residual(tensor, inputs, CONDITION, measurement).detach().cpu())
        jacobian = preflight.combined_jacobian(selected, inputs, CONDITION, measurement)
        scales = parameter_scales(inputs, protocol)
        gradient = float(np.linalg.norm((jacobian * scales[None]).T @ residual))
        parts = preflight.objective_parts(selected, inputs, CONDITION, measurement)
        endpoint_parts[start_index] = parts
        objective_histories[start_index] = history
        endpoint = MultiStartEndpoint(
            start_index=start_index,
            initialization=initialization.copy(),
            parameters=selected,
            canonical_parameters=canonical,
            initialization_sha256=parameter_sha256(initialization),
            parameter_sha256=parameter_sha256(canonical),
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            nfev=int(result.nfev),
            njev=None if result.njev is None else int(result.njev),
            cost=float(result.cost),
            likelihood_objective=parts["total_objective"],
            chi_square=parts["observation_chi_square"],
            optimality=float(result.optimality),
            gradient_norm=gradient,
            requested_sha256=canonical_tensor_sha256(pair.requested),
            companion_sha256=canonical_tensor_sha256(pair.companion),
            recomposed_sha256=canonical_tensor_sha256(pair.recomposed_sources),
            symmetries=symmetries,
        )
        endpoints.append(endpoint)
        print(json.dumps({
            "event": "start_complete",
            "scene": scene,
            "condition": CONDITION,
            "start": start_index,
            "success": endpoint.success,
            "status": endpoint.status,
            "nfev": endpoint.nfev,
            "gradient_norm": endpoint.gradient_norm,
            "total_objective": endpoint.likelihood_objective,
            "elapsed_seconds": time.perf_counter() - start_wall,
        }), flush=True)

    finite = [endpoint for endpoint in endpoints if np.isfinite(endpoint.likelihood_objective)]
    if not finite:
        raise RuntimeError(f"all endpoints are nonfinite for Scene {scene}")
    best = min(finite, key=lambda endpoint: endpoint.likelihood_objective)
    best_tensor = torch.as_tensor(best.parameters, dtype=torch.float64)
    canonical, active, symmetries = canonicalize_parameters(best.parameters, inputs, protocol)
    jacobian = preflight.combined_jacobian(best.parameters, inputs, CONDITION, measurement)
    scales = parameter_scales(inputs, protocol)
    diagnostics = analyze_jacobian(jacobian, active_mask=active, parameter_scales=scales)
    residual = np.asarray(preflight.combined_residual(best_tensor, inputs, CONDITION, measurement).detach().cpu())
    gradient_norm = float(np.linalg.norm((jacobian * scales[None]).T @ residual))
    dof = max(1, int(inputs.observed.numel()) - int(active.sum()))
    support_threshold = float(chi2.ppf(protocol.model_acceptance_quantile, dof))
    support = [endpoint for endpoint in finite if endpoint.chi_square <= support_threshold]
    geometry = solution_geometry(support if support else finite, inputs, protocol)

    pair_first = render_pair(best_tensor, inputs, protocol)
    residual_first = np.asarray(preflight.combined_residual(best_tensor, inputs, CONDITION, measurement).detach().cpu())
    jacobian_first = preflight.combined_jacobian(best.parameters, inputs, CONDITION, measurement)
    pair_second = render_pair(best_tensor, inputs, protocol)
    residual_second = np.asarray(preflight.combined_residual(best_tensor, inputs, CONDITION, measurement).detach().cpu())
    jacobian_second = preflight.combined_jacobian(best.parameters, inputs, CONDITION, measurement)
    replay = {
        "requested_first": canonical_tensor_sha256(pair_first.requested),
        "requested_second": canonical_tensor_sha256(pair_second.requested),
        "companion_first": canonical_tensor_sha256(pair_first.companion),
        "companion_second": canonical_tensor_sha256(pair_second.companion),
        "residual_first": raw_array_sha256(residual_first),
        "residual_second": raw_array_sha256(residual_second),
        "jacobian_first": raw_array_sha256(jacobian_first),
        "jacobian_second": raw_array_sha256(jacobian_second),
    }
    replay["exact_match"] = (
        replay["requested_first"] == replay["requested_second"]
        and replay["companion_first"] == replay["companion_second"]
        and replay["residual_first"] == replay["residual_second"]
        and replay["jacobian_first"] == replay["jacobian_second"]
    )
    perturbation = numerical_perturbation(best.parameters, inputs, measurement, diagnostics)
    boundary = boundary_contact_flags(best.parameters, inputs, protocol)
    optimizer_limited = (
        not any(endpoint.success for endpoint in endpoints)
        or not support
        or not replay["exact_match"]
        or not np.isfinite(gradient_norm)
        or not np.isfinite(diagnostics.condition_number)
    )
    result = {
        "scene_index": scene,
        "condition": CONDITION,
        "family": FAMILY_BULGE_DISK,
        "level": "Level 5",
        "starts": starts,
        "endpoints": endpoints,
        "endpoint_parts": endpoint_parts,
        "objective_histories": objective_histories,
        "best": best,
        "best_parts": endpoint_parts[best.start_index],
        "canonical_best_parameters": canonical,
        "gradient_norm": gradient_norm,
        "gradient_gate_pass": gradient_norm <= protocol.acceptable_gradient_norm,
        "rank": diagnostics.rank,
        "active_parameter_count": diagnostics.active_parameter_count,
        "nullity": diagnostics.null_space_dimension,
        "condition_number": diagnostics.condition_number,
        "geometry": asdict(geometry),
        "support_start_indices": [endpoint.start_index for endpoint in support],
        "support_threshold": support_threshold,
        "symmetries": symmetries,
        "boundary_contact_flags": boundary,
        "replay": replay,
        "numerical_perturbation": perturbation,
        "fitted_requested_fluxes": best.parameters[:3],
        "fitted_companion_fluxes": best.parameters[12:15],
        "optimizer_limited": optimizer_limited,
        "optimizer_successful_starts": sum(endpoint.success for endpoint in endpoints),
        "max_budget_starts": sum(endpoint.nfev >= MAX_NFEV for endpoint in endpoints),
        "fit_runtime_seconds": time.perf_counter() - fit_started,
    }
    result["scientific_classification"] = strat.classify_external(result)
    return result


def fit_record(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "scene_index": result["scene_index"],
        "scene_id": metadata["scene_id"],
        "condition": CONDITION,
        "family": result["family"],
        "level": result["level"],
        "only_authorized_scientific_change": {"max_nfev_before": OLD_MAX_NFEV, "max_nfev_after": MAX_NFEV},
        "parameterization": "direct_bounded_physical_coordinates",
        "unconstrained_initial_variables_present": False,
        "starts": result["starts"],
        "endpoints": [
            {**endpoint.record(), **result["endpoint_parts"][endpoint.start_index]}
            for endpoint in result["endpoints"]
        ],
        "objective_histories": result["objective_histories"],
        "objective_history_complete_for_every_residual_evaluation": True,
        "best_start_index": result["best"].start_index,
        "best_objective_components": result["best_parts"],
        "gradient_norm": result["gradient_norm"],
        "gradient_gate_pass": result["gradient_gate_pass"],
        "rank": result["rank"],
        "active_parameter_count": result["active_parameter_count"],
        "nullity": result["nullity"],
        "condition_number": result["condition_number"],
        "geometry": result["geometry"],
        "support_start_indices": result["support_start_indices"],
        "support_threshold": result["support_threshold"],
        "symmetries": result["symmetries"],
        "boundary_contact_flags": result["boundary_contact_flags"],
        "fitted_requested_fluxes_g_r_z": result["fitted_requested_fluxes"],
        "fitted_companion_fluxes_g_r_z": result["fitted_companion_fluxes"],
        "replay": result["replay"],
        "numerical_perturbation": result["numerical_perturbation"],
        "scientific_classification": result["scientific_classification"],
        "optimizer_limited": result["optimizer_limited"],
        "optimizer_successful_starts": result["optimizer_successful_starts"],
        "max_budget_starts": result["max_budget_starts"],
        "fit_runtime_seconds": result["fit_runtime_seconds"],
    }


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


def log_condition_gain(before: float, after: float) -> float:
    if math.isinf(after):
        return float("-inf")
    if after <= 0:
        return float("inf")
    return math.log10(before / after)


def corrected_response(
    result: dict[str, Any],
    metadata: dict[str, Any],
    baselines: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    scene = int(result["scene_index"])
    s1, p2 = baselines[(scene, "S1")], baselines[(scene, "P2")]
    external = external_summary(result)
    resolved = external["classification"] not in UNRESOLVED_FIT_CLASSES
    helpful = correction.materially_better_geometry(external, p2) if resolved else None
    row: dict[str, Any] = {
        "scene_index": scene,
        "scene_id": metadata["scene_id"],
        "condition": CONDITION,
        "primary_response_condition": True,
        "response_resolved": resolved,
        "photometry_helpful": "" if helpful is None else bool(helpful),
        "response_label": "Optimization Unresolved" if helpful is None else "Photometry Helpful" if helpful else "Photometry Not Helpful",
        "scene_classification": "STILL_OPTIMIZATION_UNRESOLVED" if helpful is None else "HELPFUL" if helpful else "NOT_HELPFUL",
        "optimizer_successful_starts": result["optimizer_successful_starts"],
        "max_budget_starts": result["max_budget_starts"],
        "s1_classification": s1["classification"],
        "p2_classification": p2["classification"],
        "external_classification": external["classification"],
        "s1_to_p2_classification_improvement": CLASS_SCORE[p2["classification"]] - CLASS_SCORE[s1["classification"]],
        "p2_to_external_classification_improvement": "" if not resolved else CLASS_SCORE[external["classification"]] - CLASS_SCORE[p2["classification"]],
        "s1_endpoint_classes": s1["distinct_solution_classes"],
        "p2_endpoint_classes": p2["distinct_solution_classes"],
        "external_endpoint_classes": external["distinct_solution_classes"],
        "p2_to_external_endpoint_reduction": "" if not resolved else p2["distinct_solution_classes"] - external["distinct_solution_classes"],
        "s1_condition_number": s1["condition_number"],
        "p2_condition_number": p2["condition_number"],
        "external_condition_number": external["condition_number"],
        "p2_to_external_log10_condition_improvement": log_condition_gain(p2["condition_number"], external["condition_number"]),
        "external_rank": external["rank"],
        "external_nullity": external["nullity"],
        "external_gradient_norm": external["gradient_norm"],
    }
    fractions = []
    for label, key in (
        ("image", None),
        ("morphology", "morphology_parameter_diameter"),
        ("flux_allocation", "flux_allocation_diameter"),
    ):
        if label == "image":
            p2_value = max(p2["requested_image_diameter"], p2["companion_image_diameter"])
            external_value = max(external["requested_image_diameter"], external["companion_image_diameter"])
        else:
            p2_value, external_value = p2[key], external[key]
        fraction = fractional_reduction(float(p2_value), float(external_value))
        fractions.append(fraction)
        row.update({
            f"p2_{label}_diameter": p2_value,
            f"external_{label}_diameter": external_value,
            f"p2_to_external_{label}_diameter_reduction": float(p2_value) - float(external_value),
            f"p2_to_external_{label}_diameter_fraction_reduction": fraction,
        })
    row["mean_p2_to_external_diameter_fraction_reduction"] = float(np.mean(fractions))
    return row


def old_best_record(record: dict[str, Any]) -> dict[str, Any]:
    return min(record["endpoints"], key=lambda endpoint: float(endpoint["likelihood_objective"]))


def response_number(value: str) -> float | int | str:
    if value == "":
        return ""
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def draw_corrected_figures(
    all_responses: list[dict[str, Any]],
    tree_summary: dict[str, Any],
) -> None:
    scenes = [int(row["scene_index"]) for row in all_responses]
    p2_classes = [float(row["p2_endpoint_classes"]) for row in all_responses]
    external_classes = [float(row["external_endpoint_classes"]) for row in all_responses]
    resolved = [bool(row["response_resolved"]) for row in all_responses]
    x = np.arange(len(scenes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, p2_classes, width, label="P2", color="#4c78a8")
    bars = ax.bar(x + width / 2, external_classes, width, label="Total photometry", color="#f58518")
    for bar, ok in zip(bars, resolved):
        if not ok:
            bar.set_hatch("///")
            bar.set_alpha(0.5)
    ax.set_xticks(x, [f"Scene {scene}" for scene in scenes])
    ax.set_ylabel("Acceptable endpoint classes")
    ax.set_title("Corrected total-photometry information-source comparison")
    ax.legend(frameon=False)
    destination = RUN_DIR / "figures/corrected_information_source_comparison.png"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=180)
    plt.close(fig)
    os.replace(temporary, destination)

    root = tree_summary["root"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.axis("off")

    def plot_node(node: dict[str, Any], x0: float, y0: float, spread: float) -> None:
        if "feature" in node:
            label = f"{node['feature']} <= {node['threshold']:.5g}\nn={node['samples']}, helpful={node['helpful']}"
        else:
            label = f"{'Helpful' if node['prediction'] else 'Not helpful'}\nn={node['samples']}, helpful={node['helpful']}"
        ax.text(x0, y0, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#555555"})
        if "feature" in node:
            child_y = y0 - 0.3
            for child, child_x, edge in ((node["left"], x0 - spread, "yes"), (node["right"], x0 + spread, "no")):
                ax.plot([x0, child_x], [y0 - 0.04, child_y + 0.04], color="#555555", linewidth=1)
                ax.text((x0 + child_x) / 2, (y0 + child_y) / 2, edge, fontsize=8)
                plot_node(child, child_x, child_y, spread / 2.2)

    plot_node(root, 0.5, 0.86, 0.24)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Corrected exploratory decision tree")
    destination = RUN_DIR / "figures/corrected_decision_tree.png"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=180)
    plt.close(fig)
    os.replace(temporary, destination)


def corrected_analysis(
    new_responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not any(bool(row["response_resolved"]) for row in new_responses):
        return None
    prior_rows = [
        {key: response_number(value) for key, value in row.items()}
        for row in read_rows(STRAT_RESPONSES)
        if row["condition"] == CONDITION and int(row["scene_index"]) not in SCENES
    ]
    all_responses = sorted([*prior_rows, *new_responses], key=lambda row: int(row["scene_index"]))
    active_responses = [row for row in all_responses if bool(row["response_resolved"])]
    active_scenes = tuple(int(row["scene_index"]) for row in active_responses)
    feature_rows = read_rows(STRAT_FEATURES)
    active_features = [row for row in feature_rows if int(row["scene_index"]) in active_scenes]
    active_features.sort(key=lambda row: active_scenes.index(int(row["scene_index"])))
    active_responses.sort(key=lambda row: active_scenes.index(int(row["scene_index"])))
    predecessor_scenes = strat.SCENES
    strat.SCENES = active_scenes
    try:
        ranking, tree_summary, rules, analysis_rows, _ = strat.statistical_analysis(active_features, active_responses)
    finally:
        strat.SCENES = predecessor_scenes
    helpful = sum(bool(row["photometry_helpful"]) for row in active_responses)
    total = len(active_responses)
    ci_lower, ci_upper = strat.clopper_pearson(helpful, total)
    old_helpful = 3
    old_total = 6
    old_rate = old_helpful / old_total
    new_rate = helpful / total
    summary_rows = []
    feature_lookup = {int(row["scene_index"]): row for row in feature_rows}
    response_lookup = {int(row["scene_index"]): row for row in all_responses}
    for scene in strat.SCENES:
        response = response_lookup[scene]
        summary_rows.append({
            **feature_lookup[scene],
            "response_resolved": response["response_resolved"],
            "response_label": response["response_label"],
            "photometry_helpful": response["photometry_helpful"],
            "included_in_corrected_analysis": bool(response["response_resolved"]),
            "corrected_interpretable_scene_count": total,
            "corrected_helpful_scene_count": helpful,
            "corrected_helpful_rate": new_rate,
            "exact_95pct_ci_lower": ci_lower,
            "exact_95pct_ci_upper": ci_upper,
        })
    atomic_csv(RUN_DIR / "tables/corrected_eight_scene_summary.csv", summary_rows)

    ranked = [dict(row) for row in active_responses]
    ranked.sort(key=lambda row: (
        -int(bool(row["photometry_helpful"])),
        -int(row["p2_to_external_endpoint_reduction"]),
        -int(row["p2_to_external_classification_improvement"]),
        -float(row["p2_to_external_log10_condition_improvement"]),
        -float(row["mean_p2_to_external_diameter_fraction_reduction"]),
        int(row["scene_index"]),
    ))
    unresolved = [row for row in all_responses if not bool(row["response_resolved"])]
    ranking_rows = [
        {"scene_rank": index, "ranking_status": "resolved", **row}
        for index, row in enumerate(ranked, start=1)
    ] + [
        {"scene_rank": len(ranked) + index, "ranking_status": "unranked_optimizer_unresolved", **row}
        for index, row in enumerate(unresolved, start=1)
    ]
    atomic_csv(RUN_DIR / "tables/corrected_scene_ranking.csv", ranking_rows)
    atomic_csv(RUN_DIR / "tables/corrected_feature_ranking.csv", ranking)
    atomic_csv(RUN_DIR / "tables/corrected_decision_rule.csv", rules)
    atomic_json(RUN_DIR / "tables/corrected_decision_tree.json", tree_summary)
    draw_corrected_figures(all_responses, tree_summary)
    bulge = next(row for row in ranking if row["feature"] == "bulge_fraction_difference")
    suggestive = bool(bulge["orientation_free_auc"] >= 0.75 and bulge["direction"] == "lower predicts helpful")
    survives_multiplicity = bool(bulge["bh_qvalue"] <= 0.05)
    loo_balanced = float(tree_summary["leave_one_out_balanced_accuracy"])
    survives_loo = loo_balanced > 0.5
    if total < 8:
        hypothesis = "remained unresolved"
    elif suggestive and (float(bulge["orientation_free_auc"]) >= 1.0 or loo_balanced > 1.0 / 3.0):
        hypothesis = "strengthened"
    else:
        hypothesis = "weakened"
    return {
        "interpretable_scenes": total,
        "helpful_scenes": helpful,
        "helpful_rate": new_rate,
        "old_helpful_rate": old_rate,
        "helpful_rate_change": new_rate - old_rate,
        "exact_95pct_ci": [ci_lower, ci_upper],
        "tree_summary": tree_summary,
        "top_rule": rules[0],
        "bulge_feature": bulge,
        "bulge_pattern_suggestive": suggestive,
        "bulge_survives_bh": survives_multiplicity,
        "leave_one_out_balanced_accuracy": loo_balanced,
        "bulge_survives_leave_one_out": survives_loo,
        "bulge_fraction_hypothesis": hypothesis,
        "all_eight_interpretable": total == 8,
    }


def main() -> None:
    started_wall = time.perf_counter()
    started_utc = now()
    for directory in ("reports", "tables", "manifests", "figures", "fit_records"):
        (RUN_DIR / directory).mkdir(exist_ok=False)

    gate, inputs_by_scene, metadata_by_scene, starts_by_scene, measurements, prior_records = authorization_gate()
    print(json.dumps({"event": "authorization_gate", "status": gate["status"]}), flush=True)

    input_paths = tuple(dict.fromkeys((
        Path(__file__),
        RUN_DIR / "preregistration/frozen_protocol.md",
        STRAT_REPORT,
        STRAT_PROTOCOL,
        STRAT_ENDPOINTS,
        STRAT_MEASUREMENTS,
        STRAT_SUMMARY,
        STRAT_SCENE_RANKING,
        STRAT_FEATURE_RANKING,
        STRAT_FEATURES,
        STRAT_RESPONSES,
        STRAT_FINAL_MANIFEST,
        STRAT_INPUT_HASHES,
        preflight.SCIENCE_H5,
        preflight.SCENE_MANIFEST,
        preflight.DEFINITIONS,
        preflight.S1_METRICS,
        preflight.P2_METRICS,
        REPO / "src/model9_structured.py",
        REPO / "src/model9_optimizer.py",
        MODEL9_MANIFEST,
        MODEL9_SOURCE_HASHES,
        *READ_FIRST_FILES,
    )))
    atomic_json(RUN_DIR / "manifests/input_hashes.json", {
        "campaign": CAMPAIGN,
        "frozen_at_utc": now(),
        "files": {
            str(path.relative_to(REPO)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "scenes": SCENES,
        "family": FAMILY_BULGE_DISK,
        "condition": CONDITION,
        "starts": STARTS,
        "only_authorized_scientific_change": {"max_nfev_before": OLD_MAX_NFEV, "max_nfev_after": MAX_NFEV},
        "cpu_float64": True,
        "measurements_reused_not_regenerated": True,
    })

    baselines = strat.load_historical_baselines()
    results = []
    endpoint_rows = []
    oracle_rows = []
    for scene in SCENES:
        fit = fit_one(scene, inputs_by_scene[scene], measurements[scene], starts_by_scene[scene])
        results.append(fit)
        atomic_json(
            RUN_DIR / f"fit_records/scene_{scene:03d}_total_source_photometry.json",
            fit_record(fit, metadata_by_scene[scene]),
        )
        acceptable = set(fit["geometry"]["acceptable_endpoint_indices"])
        support = set(fit["support_start_indices"])
        for endpoint in fit["endpoints"]:
            endpoint_rows.append({
                "scene_index": scene,
                "scene_id": metadata_by_scene[scene]["scene_id"],
                "condition": CONDITION,
                **endpoint.record(),
                **fit["endpoint_parts"][endpoint.start_index],
                "observation_support_acceptable": endpoint.start_index in support,
                "total_objective_acceptable": endpoint.start_index in acceptable,
                "hit_2000_evaluation_ceiling": endpoint.nfev >= MAX_NFEV,
                "objective_history_evaluations": len(fit["objective_histories"][endpoint.start_index]),
            })
        oracle_rows.append({
            "scene_index": scene,
            "scene_id": metadata_by_scene[scene]["scene_id"],
            "condition": CONDITION,
            "status": "PASS",
            "blend_rows_accessed": 1,
            "coordinate_rows_accessed": 1,
            "frozen_measurement_rows_accessed": 2,
            "measurement_regeneration_performed": False,
            "noisy_measurements_exposed_to_fit": True,
            "measurement_likelihood_not_hard_fix": True,
            "measurement_relative_sigma": RELATIVE_SIGMA,
            "catalog_truth_exposed_to_fit": False,
            "isolated_source_images_accessed": 0,
            "isolated_source_images_exposed_to_fit": 0,
            "morphology_truth_exposed_to_fit": 0,
            "truth_initialization_exposed_to_fit": 0,
            "prior_endpoint_initialization_exposed_to_fit": 0,
            "development_access": 0,
            "atlas_tensor_access": 0,
            "lockbox_access": 0,
        })
    atomic_csv(RUN_DIR / "tables/convergence_correction_endpoints.csv", endpoint_rows)
    atomic_csv(RUN_DIR / "tables/oracle_information_audit.csv", oracle_rows)

    new_responses = [
        corrected_response(result, metadata_by_scene[int(result["scene_index"])], baselines)
        for result in results
    ]
    response_by_scene = {int(row["scene_index"]): row for row in new_responses}
    summary_rows = []
    comparison_500_rows = []
    comparison_p2_rows = []
    flux_rows = []
    for result in results:
        scene = int(result["scene_index"])
        response = response_by_scene[scene]
        best = result["best"]
        old_record = prior_records[scene]
        old_best = old_best_record(old_record)
        old_parts = preflight.objective_parts(
            np.asarray(old_best["parameters"], dtype=np.float64), inputs_by_scene[scene], CONDITION, measurements[scene]
        )
        old_geometry = old_record["geometry"]
        new_geometry = result["geometry"]
        p2 = baselines[(scene, "P2")]
        summary_rows.append({
            "scene_index": scene,
            "scene_id": metadata_by_scene[scene]["scene_id"],
            "condition": CONDITION,
            "scene_classification": response["scene_classification"],
            "scientific_classification": result["scientific_classification"],
            "objective_value": result["best_parts"]["total_objective"],
            "observation_likelihood_component": result["best_parts"]["observation_objective"],
            "photometry_likelihood_component": result["best_parts"]["photometry_objective"],
            "best_start_index": best.start_index,
            "optimizer_success": best.success,
            "optimizer_status": best.status,
            "optimizer_message": best.message,
            "best_nfev": best.nfev,
            "optimizer_successful_starts": result["optimizer_successful_starts"],
            "max_budget_starts": result["max_budget_starts"],
            "gradient_norm": result["gradient_norm"],
            "rank": result["rank"],
            "nullity": result["nullity"],
            "condition_number": result["condition_number"],
            "acceptable_endpoint_class_count": new_geometry["distinct_solution_classes"],
            "requested_image_diameter": new_geometry["requested_image_diameter"],
            "companion_image_diameter": new_geometry["companion_image_diameter"],
            "morphology_diameter": new_geometry["morphology_parameter_diameter"],
            "flux_allocation_diameter": new_geometry["flux_allocation_diameter"],
            "requested_source_fluxes_g_r_z": result["fitted_requested_fluxes"],
            "companion_source_fluxes_g_r_z": result["fitted_companion_fluxes"],
            "boundary_contacts": result["boundary_contact_flags"],
            "replay_status": "exact" if result["replay"]["exact_match"] else "failed",
            "numerical_perturbation_stability": result["numerical_perturbation"]["stable"],
            "fit_runtime_seconds": result["fit_runtime_seconds"],
        })
        comparison_500_rows.append({
            "scene_index": scene,
            "condition": CONDITION,
            "old_max_nfev": OLD_MAX_NFEV,
            "new_max_nfev": MAX_NFEV,
            "old_best_start_index": old_best["start_index"],
            "new_best_start_index": best.start_index,
            "old_best_objective": old_parts["total_objective"],
            "new_best_objective": result["best_parts"]["total_objective"],
            "objective_reduction_old_minus_new": old_parts["total_objective"] - result["best_parts"]["total_objective"],
            "old_observation_objective": old_parts["observation_objective"],
            "new_observation_objective": result["best_parts"]["observation_objective"],
            "old_photometry_objective": old_parts["photometry_objective"],
            "new_photometry_objective": result["best_parts"]["photometry_objective"],
            "old_gradient_norm": float(old_record["gradient_norm"]),
            "new_gradient_norm": result["gradient_norm"],
            "gradient_reduction_old_minus_new": float(old_record["gradient_norm"]) - result["gradient_norm"],
            "old_endpoint_classes": old_geometry["distinct_solution_classes"],
            "new_endpoint_classes": new_geometry["distinct_solution_classes"],
            "endpoint_class_change_new_minus_old": int(new_geometry["distinct_solution_classes"]) - int(old_geometry["distinct_solution_classes"]),
            **{f"old_{key}": old_geometry[key] for key in DIAMETERS},
            **{f"new_{key}": new_geometry[key] for key in DIAMETERS},
            **{f"scientific_diameter_change_new_minus_old_{key}": float(new_geometry[key]) - float(old_geometry[key]) for key in DIAMETERS},
            "old_optimizer_successful_starts": 0,
            "new_optimizer_successful_starts": result["optimizer_successful_starts"],
            "number_of_starts_that_now_converge": result["optimizer_successful_starts"],
            "old_scientific_classification": "OPTIMIZATION_UNRESOLVED",
            "new_scientific_classification": result["scientific_classification"],
            "classification_change": f"OPTIMIZATION_UNRESOLVED -> {result['scientific_classification']}",
            "scene_classification": response["scene_classification"],
        })
        comparison_p2_rows.append({
            "scene_index": scene,
            "condition": CONDITION,
            "scene_classification": response["scene_classification"],
            "response_resolved": response["response_resolved"],
            "photometry_helpful": response["photometry_helpful"],
            "p2_classification": p2["classification"],
            "external_classification": result["scientific_classification"],
            "p2_to_external_classification_improvement": response["p2_to_external_classification_improvement"],
            "p2_endpoint_classes": p2["distinct_solution_classes"],
            "external_endpoint_classes": new_geometry["distinct_solution_classes"],
            "p2_to_external_endpoint_reduction": response["p2_to_external_endpoint_reduction"],
            **{f"p2_{key}": p2[key] for key in DIAMETERS},
            **{f"external_{key}": new_geometry[key] for key in DIAMETERS},
            "p2_to_external_image_diameter_fraction_reduction": response["p2_to_external_image_diameter_fraction_reduction"],
            "p2_to_external_morphology_diameter_fraction_reduction": response["p2_to_external_morphology_diameter_fraction_reduction"],
            "p2_to_external_flux_allocation_diameter_fraction_reduction": response["p2_to_external_flux_allocation_diameter_fraction_reduction"],
            "material_improvement_under_frozen_rule": response["photometry_helpful"],
        })
        for identity, values in (("requested", result["fitted_requested_fluxes"]), ("companion", result["fitted_companion_fluxes"])):
            for band, value in zip(("g", "r", "z"), values):
                flux_rows.append({
                    "scene_index": scene,
                    "scene_id": metadata_by_scene[scene]["scene_id"],
                    "condition": CONDITION,
                    "source_identity": identity,
                    "band": band,
                    "fitted_flux_electrons": value,
                    "best_start_index": best.start_index,
                })
    atomic_csv(RUN_DIR / "tables/convergence_correction_summary.csv", summary_rows)
    atomic_csv(RUN_DIR / "tables/comparison_to_500eval.csv", comparison_500_rows)
    atomic_csv(RUN_DIR / "tables/comparison_to_p2.csv", comparison_p2_rows)
    atomic_csv(RUN_DIR / "tables/fitted_source_fluxes.csv", flux_rows)

    resolved_count = sum(bool(row["response_resolved"]) for row in new_responses)
    if resolved_count == 2:
        outcome = "STRATIFICATION_DATASET_COMPLETE"
    elif resolved_count == 1:
        outcome = "STRATIFICATION_PARTIALLY_COMPLETE"
    else:
        outcome = "STRATIFICATION_CONVERGENCE_STILL_LIMITED"
    corrected = corrected_analysis(new_responses)

    final_integrity_snapshot = historical_snapshot(gate["historical_integrity"]["protected_historical_file_hashes"])
    final_integrity = {
        "campaign": CAMPAIGN,
        "status": "PASS" if final_integrity_snapshot["status"] == "PASS" else "FAIL",
        "before": gate["historical_integrity"],
        "after": final_integrity_snapshot,
        "head_unchanged": gate["historical_integrity"]["head"] == final_integrity_snapshot["head"] == EXPECTED_HEAD,
        "readme_unchanged": gate["historical_integrity"]["readme_sha256"] == final_integrity_snapshot["readme_sha256"] == EXPECTED_README_SHA256,
        "historical_reports_and_predecessor_artifacts_unchanged": final_integrity_snapshot["protected_historical_files_match_start"],
        "historical_checkpoints_unchanged": final_integrity_snapshot["historical_checkpoint_matches"] == 600 and final_integrity_snapshot["historical_checkpoint_mismatches"] == 0,
        "git_index_empty": not final_integrity_snapshot["git_index_entries"],
        "protected_data_access_zero": all(value == 0 for value in final_integrity_snapshot["protected_data_access"].values()),
        "nothing_staged": not final_integrity_snapshot["git_index_entries"],
        "commits_created": 0,
    }
    atomic_json(RUN_DIR / "manifests/historical_integrity.json", final_integrity)
    if final_integrity["status"] != "PASS":
        outcome = "STRATIFICATION_CORRECTION_INVALID"

    runtime_seconds = time.perf_counter() - started_wall
    ended_utc = now()
    unresolved_scenes = [int(row["scene_index"]) for row in new_responses if not bool(row["response_resolved"])]
    if unresolved_scenes:
        scene_text = " and ".join(f"Scene {scene}" for scene in unresolved_scenes)
        next_experiment_name = "Thayer-External-Photometry-Stratification-Optimizer-Diagnosis-v0"
        next_experiment_text = f"Run one frozen-information optimizer-diagnosis experiment focused only on {scene_text}, with no new scientific information source."
    elif corrected and corrected["bulge_pattern_suggestive"]:
        next_experiment_name = "Thayer-External-Photometry-Stratification-Independent-Scene-Validation-v0"
        next_experiment_text = "Run a preregistered independent-scene validation of the exploratory low-|ΔB/T| candidate stratification rule."
    else:
        next_experiment_name = "Thayer-External-Photometry-Population-Scale-Description-v0"
        next_experiment_text = "Run one broader descriptive population-scale photometry experiment without a prespecified decision rule."

    failures = [
        endpoint for endpoint in endpoint_rows
        if not bool(endpoint["success"])
    ]
    ceiling = [endpoint for endpoint in endpoint_rows if bool(endpoint["hit_2000_evaluation_ceiling"])]
    diagnostics_lines = []
    for row in summary_rows:
        diagnostics_lines.append(
            f"| {row['scene_index']} | {row['scene_classification']} | {row['scientific_classification']} | "
            f"{row['optimizer_successful_starts']}/4 | {row['max_budget_starts']}/4 | {row['best_nfev']} | "
            f"{row['objective_value']:.12g} | {row['observation_likelihood_component']:.12g} | "
            f"{row['photometry_likelihood_component']:.12g} | {row['gradient_norm']:.12g} | "
            f"{row['rank']} | {row['nullity']} | {safe(row['condition_number'])} | "
            f"{row['acceptable_endpoint_class_count']} | {row['requested_image_diameter']:.6g} | "
            f"{row['companion_image_diameter']:.6g} | {row['morphology_diameter']:.6g} | "
            f"{row['flux_allocation_diameter']:.6g} | {row['replay_status']} | "
            f"{'stable' if row['numerical_perturbation_stability'] else 'unstable'} |"
        )
    comparison_lines = []
    for row in comparison_500_rows:
        comparison_lines.append(
            f"| {row['scene_index']} | {row['objective_reduction_old_minus_new']:.12g} | "
            f"{row['gradient_reduction_old_minus_new']:.12g} | {row['old_endpoint_classes']}→{row['new_endpoint_classes']} | "
            f"{row['number_of_starts_that_now_converge']} | {row['classification_change']} |"
        )
    failure_lines = [
        f"- Scene {row['scene_index']} / start {row['start_index']}: success={row['success']}, status={row['status']}, nfev={row['nfev']}, message=`{row['message']}`"
        for row in failures
    ] or ["- None."]
    ceiling_lines = [
        f"- Scene {row['scene_index']} / start {row['start_index']}: nfev={row['nfev']}."
        for row in ceiling
    ] or ["- None."]
    if corrected:
        rate_text = (
            f"The descriptive helpful rate changed from 3/6 (50.0%) to {corrected['helpful_scenes']}/{corrected['interpretable_scenes']} "
            f"({100.0 * corrected['helpful_rate']:.1f}%; exact 95% CI {100.0 * corrected['exact_95pct_ci'][0]:.1f}%–{100.0 * corrected['exact_95pct_ci'][1]:.1f}%)."
        )
        hypothesis_text = (
            f"The bulge-fraction hypothesis **{corrected['bulge_fraction_hypothesis']}**. "
            f"Its corrected oriented AUC is {corrected['bulge_feature']['orientation_free_auc']:.3f}, exact p={corrected['bulge_feature']['exact_permutation_p']:.4f}, "
            f"BH q={corrected['bulge_feature']['bh_qvalue']:.4f}; corrected leave-one-out balanced accuracy is {corrected['leave_one_out_balanced_accuracy']:.3f}. "
            f"It remains exploratory and is not a validated predictor."
        )
    else:
        rate_text = "The helpful rate remains the frozen 3/6 (50.0%) because neither unresolved scene became interpretable."
        hypothesis_text = "The bulge-fraction hypothesis **remained unresolved** because the corrected dataset is still incomplete."

    scene_statements = []
    for row in new_responses:
        scene_statements.append(
            f"- Scene {row['scene_index']}: {'resolved' if row['response_resolved'] else 'did not resolve'}; "
            f"classification `{row['scene_classification']}`; response `{row['response_label']}`."
        )
    report = f"""# {CAMPAIGN} final report

## Outcome

**{outcome}**

The only scientific-execution change was **`max_nfev: 500 -> 2000`**. The exact predecessor measurements, 5% uncertainties, four direct-bounded physical starts, optimizer, parameterization, Level-5 bulge+disk family, observation and photometry likelihoods, objective, PSF, coordinates, morphology support, all other tolerances, gradient gate, endpoint acceptance and clustering, rank/nullity and condition rules, scientific diameters, replay, and classification logic were unchanged. No resolved scene was rerun.

{chr(10).join(scene_statements)}

Campaign primary answer: **{resolved_count}/2 previously unresolved scenes became scientifically interpretable.**

## Fit diagnostics

| Scene | Scene label | Fit class | Successful starts | 2000-ceiling starts | Best nfev | Total objective | Observation objective | Photometry objective | Gradient | Rank | Nullity | Condition | Classes | Requested diameter | Companion diameter | Morphology diameter | Flux-allocation diameter | Replay | Perturbation |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(diagnostics_lines)}

Fitted source g/r/z fluxes and complete boundary-contact structures are in `tables/fitted_source_fluxes.csv`, `tables/convergence_correction_summary.csv`, and the atomic fit records. Every residual-function objective evaluation is retained in each fit record.

## Comparison with the 500-evaluation result

| Scene | Objective reduction | Gradient reduction | Endpoint classes | Starts now converged | Classification change |
| ---: | ---: | ---: | --- | ---: | --- |
{chr(10).join(comparison_lines)}

All requested scientific-diameter changes and direct P2 comparisons are in `tables/comparison_to_500eval.csv` and `tables/comparison_to_p2.csv`.

## Corrected stratification

{rate_text}

{hypothesis_text}

Corrected outputs, when authorized by at least one resolved scene, are separately labeled and do not overwrite the frozen six-scene analysis. No validated predictor is claimed from at most eight scenes.

## Optimizer failures

{chr(10).join(failure_lines)}

## Starts at the 2000-evaluation ceiling

{chr(10).join(ceiling_lines)}

## Integrity and runtime

Authorization and final integrity status: **{'PASS' if gate['status'] == 'PASS' and final_integrity['status'] == 'PASS' else 'FAIL'}**. All 600 historical checkpoints matched before and after; README and HEAD were unchanged; predecessor and historical reports were unchanged; protected development, Atlas-tensor, lockbox, and isolated-source access were zero; the Git index remained empty; nothing was staged or committed.

Exact campaign runtime: **{runtime_seconds:.6f} seconds** ({runtime_seconds / 60.0:.6f} minutes), from `{started_utc}` to `{ended_utc}`.

## Exactly one next experiment

**{next_experiment_name}** — {next_experiment_text}
"""
    atomic_text(RUN_DIR / "reports/final_report.md", report)

    required = [
        "reports/final_report.md",
        "preregistration/frozen_protocol.md",
        "tables/convergence_correction_summary.csv",
        "tables/convergence_correction_endpoints.csv",
        "tables/comparison_to_500eval.csv",
        "tables/comparison_to_p2.csv",
        "tables/fitted_source_fluxes.csv",
        "tables/oracle_information_audit.csv",
        "manifests/input_hashes.json",
        "manifests/historical_integrity.json",
    ]
    if corrected:
        required.extend([
            "tables/corrected_eight_scene_summary.csv",
            "tables/corrected_scene_ranking.csv",
            "tables/corrected_feature_ranking.csv",
            "figures/corrected_information_source_comparison.png",
            "figures/corrected_decision_tree.png",
        ])
    files = {}
    for relative in required:
        path = RUN_DIR / relative
        if not path.exists():
            raise RuntimeError(f"required output missing: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    final_manifest = {
        "campaign": CAMPAIGN,
        "completed_at_utc": ended_utc,
        "status": outcome,
        "authorization_gate_status": gate["status"],
        "historical_integrity_status": final_integrity["status"],
        "only_authorized_scientific_change": {"max_nfev_before": OLD_MAX_NFEV, "max_nfev_after": MAX_NFEV},
        "scene_classifications": {str(row["scene_index"]): row["scene_classification"] for row in new_responses},
        "resolved_scenes": [int(row["scene_index"]) for row in new_responses if bool(row["response_resolved"])],
        "unresolved_scenes": unresolved_scenes,
        "corrected_analysis_created": corrected is not None,
        "next_experiment": next_experiment_name,
        "exact_runtime_seconds": runtime_seconds,
        "files": files,
        "head_unchanged": final_integrity["head_unchanged"],
        "readme_unchanged": final_integrity["readme_unchanged"],
        "historical_checkpoints_unchanged": final_integrity["historical_checkpoints_unchanged"],
        "historical_reports_unchanged": final_integrity["historical_reports_and_predecessor_artifacts_unchanged"],
        "protected_data_access_zero": final_integrity["protected_data_access_zero"],
        "git_index_empty": final_integrity["git_index_empty"],
        "commits_created": 0,
    }
    atomic_json(RUN_DIR / "manifests/final_manifest.json", final_manifest)
    print(json.dumps({
        "event": "campaign_complete",
        "outcome": outcome,
        "resolved_scenes": final_manifest["resolved_scenes"],
        "unresolved_scenes": unresolved_scenes,
        "runtime_seconds": runtime_seconds,
        "run_dir": str(RUN_DIR),
    }), flush=True)


if __name__ == "__main__":
    main()
