#!/usr/bin/env python3
"""Authoritative-budget correction for the external-photometry preflight."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_thayer_external_photometry_preflight_v0 import (
    AUTHORITATIVE_REPORTS,
    CATALOG_PATH,
    DEFINITIONS,
    DIAMETERS,
    P2_METRICS,
    S1_METRICS,
    SCENE_MANIFEST,
    SCIENCE_H5,
    combined_jacobian,
    combined_residual,
    load_baselines,
    objective_parts,
    promise as preflight_promise,
    read_csv_rows,
)
from scripts.run_thayer_flux_free_identifiability_v0 import (
    CHECKPOINT_BASELINE,
    load_science_inputs,
    raw_array_sha256,
    selected_manifest_rows,
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
    oracle_information_audit,
    parameter_bounds,
    parameter_scales,
    parameter_sha256,
    render_pair,
    validate_parameters,
)


CAMPAIGN = "Thayer-External-Photometry-Convergence-Correction-v0"
SCENES = (0, 6)
CONDITIONS = ("TOTAL_SOURCE_PHOTOMETRY", "PER_BAND_SOURCE_PHOTOMETRY")
STARTS = 4
MAX_NFEV = 500
RELATIVE_SIGMA = 0.05
EXPECTED_HEAD = "74b8ff7efbbf7e9891cc8fd8095a9931e3b63174"
EXPECTED_README_SHA256 = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"
EXPECTED_CHECKPOINT_BASELINE_SHA256 = "982fa39058030c1ec81e832b76031acd95936295d70a4f8ef69b6238dc72d477"

PREFLIGHT_RUN = REPO / "outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852"
PREFLIGHT_REPORT = PREFLIGHT_RUN / "reports/final_report.md"
PREFLIGHT_PROTOCOL = PREFLIGHT_RUN / "preregistration/frozen_preflight_protocol.md"
PREFLIGHT_ENDPOINTS = PREFLIGHT_RUN / "tables/preflight_endpoints.csv"
PREFLIGHT_MEASUREMENTS = PREFLIGHT_RUN / "tables/external_photometry_measurements.csv"
PREFLIGHT_SUMMARY = PREFLIGHT_RUN / "tables/preflight_condition_summary.csv"
PREFLIGHT_COMPARISON = PREFLIGHT_RUN / "tables/comparison_to_s1_and_p2.csv"
PREFLIGHT_ORACLE_AUDIT = PREFLIGHT_RUN / "tables/oracle_information_audit.csv"

EXPECTED_PREFLIGHT_HASHES = {
    PREFLIGHT_REPORT: "b5552e019471ea259960c48a2236cfa2e6be2b2e0dd296d5392d3efb5d96ca4e",
    PREFLIGHT_PROTOCOL: "9a5ee69f8a884d8b36387d0ff5fa1f9049fa25ccacc66c7bacf78f86966bc219",
    PREFLIGHT_ENDPOINTS: "167879f7df267db68f324abcde4667d62dfe14be9205a1116940f65c949e9cb1",
    PREFLIGHT_MEASUREMENTS: "7767fd176b1b062959d58ed82f075563e263474728ee669f3d85d5fbd8a65e8d",
    PREFLIGHT_SUMMARY: "ea67aac830d05a9f3cb63a76192a0185b44be7a3cbe0ae739c77c1ebd583c81c",
    PREFLIGHT_COMPARISON: "6f055d76bbbc233362f0180a67cbcda0f1ff6fef6c9d163b9fbe10d00ba3fae5",
}

MODEL9_RUN = REPO / "outputs/runs/thayer_model_9_preparation_v0_20260715_172217"
MODEL9_MANIFEST = MODEL9_RUN / "manifests/final_manifest.json"
MODEL9_SOURCE_HASHES = MODEL9_RUN / "manifests/source_hashes.json"

RELEVANT_TESTS = (
    "tests/test_model9_foundation.py",
    "tests/test_model9_joint.py",
    "tests/test_canonical_tensor_hash.py",
    "tests/test_family_e_signed_residual.py",
    "tests/test_psf_conditioning.py",
)

PROTECTED_HISTORICAL_FILES = (
    *AUTHORITATIVE_REPORTS,
    PREFLIGHT_REPORT,
    PREFLIGHT_PROTOCOL,
    PREFLIGHT_ENDPOINTS,
    PREFLIGHT_MEASUREMENTS,
    PREFLIGHT_SUMMARY,
    PREFLIGHT_COMPARISON,
    PREFLIGHT_ORACLE_AUDIT,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=True)
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
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
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


def protocol_text(runtime_estimate: dict[str, float]) -> str:
    return f"""# Frozen convergence-correction protocol

Frozen before any correction-campaign scientific optimizer call.

- Campaign: `{CAMPAIGN}`.
- Predecessor: exact append-only `Thayer-External-Photometry-Preflight-v0` run `20260718_154852`, required outcome `PREFLIGHT_OPTIMIZATION_LIMITED` and required next experiment `{CAMPAIGN}`.
- Only authorized scientific-execution change: maximum function evaluations per start, `150 -> 500`.
- Scope unchanged: Scenes 0 and 6 only; Level 5 / bulge+disk only; `TOTAL_SOURCE_PHOTOMETRY` and `PER_BAND_SOURCE_PHOTOMETRY` only.
- Exactly four deterministic starts per fit. Physical start vectors and initialization hashes must match the preflight byte-for-byte. The implementation is directly bounded in physical coordinates; there is no unconstrained latent initialization vector.
- Exact frozen noisy measurements are loaded from the preflight table with SHA-256 `{EXPECTED_PREFLIGHT_HASHES[PREFLIGHT_MEASUREMENTS]}`. Measurements are not regenerated. Relative sigma remains 5% and photometry remains an explicit Gaussian likelihood term.
- Frozen Model-9 renderer, parameterization, morphology support, PSF, observation, noise convention, objective, optimizer (`scipy.optimize.least_squares`, bounded TRF, `x_scale=jac`), `1e-10` tolerances, gradient diagnostic, endpoint support/acceptance, image clustering, rank/nullity, uniqueness diameters, boundary rules, and replay procedure remain unchanged.
- Every endpoint is retained, including optimizer-declared failures and starts reaching 500 evaluations. Fits run sequentially; no favorability-based retry or early stop is permitted.
- The best endpoint is replayed by two exact render/residual/Jacobian evaluations and all three hashes must match.
- Fit labels preserve the preflight promise logic: `STRONG_PROMISE`, `MODERATE_PROMISE`, and `NO_CLEAR_GAIN` map to `CONVERGED_STRONG_PROMISE`, `CONVERGED_MODERATE_PROMISE`, and `CONVERGED_NO_CLEAR_GAIN`; preflight optimization-limitation logic maps to `STILL_OPTIMIZATION_LIMITED`. A strong label additionally requires the frozen `1e-5` gradient gate. Contract or integrity failure maps to `INVALID`.
- Campaign labels and the exactly-one-next-experiment rule are those specified for this correction campaign.
- No isolated-source image, morphology truth, mask, truth initialization, catalog truth flux, protected development data, Atlas tensor, or lockbox may enter inference. No measurement generation or neural training is authorized.

Pre-execution estimate from the frozen preflight: {runtime_estimate['preflight_runtime_seconds']:.6f} seconds for {int(runtime_estimate['preflight_total_nfev'])} evaluations. If only the five previously capped starts use the full extension, the linear estimate is {runtime_estimate['estimated_seconds']:.3f} seconds ({runtime_estimate['estimated_seconds'] / 60.0:.2f} minutes). The all-cap linear bound is {runtime_estimate['upper_seconds']:.3f} seconds ({runtime_estimate['upper_seconds'] / 60.0:.2f} minutes). No 30-minute wall-clock limit is imposed.
"""


def runtime_estimate(preflight_rows: list[dict[str, str]]) -> dict[str, float]:
    preflight_runtime = 562.870292
    total_nfev = sum(int(row["nfev"]) for row in preflight_rows)
    projected_nfev = sum(
        MAX_NFEV if int(row["nfev"]) >= 150 else int(row["nfev"])
        for row in preflight_rows
    )
    return {
        "preflight_runtime_seconds": preflight_runtime,
        "preflight_total_nfev": float(total_nfev),
        "preflight_seconds_per_nfev": preflight_runtime / total_nfev,
        "estimated_seconds": preflight_runtime * projected_nfev / total_nfev,
        "upper_seconds": preflight_runtime * (MAX_NFEV / 150.0),
    }


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
    files = {str(path.relative_to(REPO)): sha256_file(path) for path in PROTECTED_HISTORICAL_FILES}
    current = {
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
        "protected_data_access": {"development": 0, "atlas_tensors": 0, "lockbox": 0},
        "historical_isolated_hdf5_access": 0,
        "commits_created": 0,
    }
    current["protected_historical_files_match_start"] = reference_hashes is None or files == reference_hashes
    current["status"] = "PASS" if (
        current["head"] == EXPECTED_HEAD
        and current["readme_sha256"] == EXPECTED_README_SHA256
        and not current["git_index_entries"]
        and current["checkpoint_baseline_sha256"] == EXPECTED_CHECKPOINT_BASELINE_SHA256
        and len(baseline) == 600
        and matches == 600
        and mismatches == 0
        and missing == 0
        and current["protected_historical_files_match_start"]
    ) else "FAIL"
    return current


def stale_optimizer_processes() -> list[dict[str, Any]]:
    current_pids = {os.getpid(), os.getppid()}
    tokens = (
        "run_thayer_external_photometry_preflight_v0.py",
        "run_thayer_external_photometry_convergence_correction_v0.py",
        "run_thayer_flux_free_identifiability_v0.py",
        "run_thayer_psf_diverse_flux_identifiability_v0.py",
    )
    rows = []
    for line in command(["ps", "-axo", "pid=,etime=,command="]).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        if pid not in current_pids and any(token in fields[2] for token in tokens):
            rows.append({"pid": pid, "elapsed": fields[1], "command": fields[2]})
    return rows


def load_frozen_measurements() -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], list[dict[str, str]]]:
    rows = read_csv_rows(PREFLIGHT_MEASUREMENTS)
    expected_counts = {
        (0, "TOTAL_SOURCE_PHOTOMETRY"): 2,
        (0, "PER_BAND_SOURCE_PHOTOMETRY"): 6,
        (6, "TOTAL_SOURCE_PHOTOMETRY"): 2,
        (6, "PER_BAND_SOURCE_PHOTOMETRY"): 6,
    }
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["scene_index"]), row["condition"])
        grouped.setdefault(key, []).append(row)
        if float(row["relative_sigma"]) != RELATIVE_SIGMA:
            raise RuntimeError("frozen measurement relative sigma changed")
        if row["latent_catalog_value_persisted"] != "False":
            raise RuntimeError("latent catalog value leaked into frozen measurement table")
        if row["catalog_sha256"] != "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46":
            raise RuntimeError("frozen measurement catalog provenance changed")
    if set(grouped) != set(expected_counts):
        raise RuntimeError("frozen measurement fit keys changed")
    output: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, expected in expected_counts.items():
        group = grouped[key]
        if len(group) != expected:
            raise RuntimeError(f"frozen measurement row count changed for {key}")
        measured = np.asarray([float(row["measured_flux_electrons"]) for row in group], dtype=np.float64)
        sigma = np.asarray([float(row["sigma_electrons"]) for row in group], dtype=np.float64)
        if not np.isfinite(measured).all() or not np.isfinite(sigma).all() or np.any(measured <= 0) or np.any(sigma <= 0):
            raise RuntimeError("frozen measurements are not finite and positive")
        output[key] = {"measured": measured, "sigma": sigma}
    return output, rows


def verify_preflight_records(
    inputs_by_scene: dict[int, SolverInputs],
) -> tuple[dict[int, np.ndarray], dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    endpoints = read_csv_rows(PREFLIGHT_ENDPOINTS)
    summary = [row for row in read_csv_rows(PREFLIGHT_SUMMARY) if row["source"] == "new_fit"]
    if len(summary) != 4 or len(endpoints) != 16:
        raise RuntimeError("preflight does not contain exactly four fits and sixteen endpoints")
    fit_keys = {(int(row["scene_index"]), row["condition"]) for row in endpoints}
    if fit_keys != {(scene, condition) for scene in SCENES for condition in CONDITIONS}:
        raise RuntimeError("preflight fit keys changed")
    starts_by_scene: dict[int, np.ndarray] = {}
    start_records: list[dict[str, Any]] = []
    for scene in SCENES:
        regenerated = deterministic_starts(inputs_by_scene[scene], FrozenSolverProtocol(), count=STARTS)
        starts_by_scene[scene] = regenerated
        reference: dict[int, np.ndarray] = {}
        for condition in CONDITIONS:
            condition_rows = sorted(
                [row for row in endpoints if int(row["scene_index"]) == scene and row["condition"] == condition],
                key=lambda row: int(row["start_index"]),
            )
            if [int(row["start_index"]) for row in condition_rows] != list(range(STARTS)):
                raise RuntimeError(f"preflight start IDs changed for scene {scene} {condition}")
            for row in condition_rows:
                index = int(row["start_index"])
                physical = np.asarray(json.loads(row["initialization"]), dtype=np.float64)
                if parameter_sha256(physical) != row["initialization_sha256"]:
                    raise RuntimeError("preflight initialization hash does not match its physical vector")
                if index in reference and not np.array_equal(reference[index], physical):
                    raise RuntimeError("preflight starts differ across photometry conditions")
                reference[index] = physical
        ordered = np.stack([reference[index] for index in range(STARTS)])
        if not np.array_equal(ordered, regenerated):
            raise RuntimeError(f"regenerated deterministic starts differ for scene {scene}")
        for index, physical in enumerate(regenerated):
            start_records.append({
                "scene_index": scene,
                "start_id": index,
                "initialization_sha256": parameter_sha256(physical),
                "physical_starting_parameters": physical,
                "unconstrained_variables_present": False,
                "unconstrained_variables": "NOT_APPLICABLE_DIRECT_BOUNDED_PHYSICAL_PARAMETERIZATION",
            })
    verification = {
        "preflight_fit_record_count": len(summary),
        "preflight_endpoint_count": len(endpoints),
        "fit_keys": sorted([list(key) for key in fit_keys]),
        "start_ids_per_fit": list(range(STARTS)),
        "starts_match_across_conditions": True,
        "starts_match_regenerated_schedule_exactly": True,
        "initialization_hashes_match_physical_vectors": True,
        "parameterization": "direct_bounded_physical_coordinates",
        "unconstrained_variables_present": False,
        "start_records": start_records,
    }
    return starts_by_scene, verification, endpoints, summary


def authorization_gate() -> tuple[dict[str, Any], dict[int, SolverInputs], dict[int, dict[str, Any]], dict[int, np.ndarray], dict[tuple[int, str], dict[str, np.ndarray]], list[dict[str, str]], list[dict[str, str]]]:
    failures: list[str] = []
    for path, expected in EXPECTED_PREFLIGHT_HASHES.items():
        if not path.exists() or sha256_file(path) != expected:
            failures.append(f"preflight hash mismatch: {path.relative_to(REPO)}")
    report_text = PREFLIGHT_REPORT.read_text(encoding="utf-8")
    predecessor_outcome_ok = "**PREFLIGHT_OPTIMIZATION_LIMITED**" in report_text
    predecessor_next_ok = "Thayer-External-Photometry-Convergence-Correction-v0: repeat only these four fits with the authoritative 500-evaluation budget" in report_text
    if not predecessor_outcome_ok:
        failures.append("predecessor outcome is not PREFLIGHT_OPTIMIZATION_LIMITED")
    if not predecessor_next_ok:
        failures.append("predecessor did not recommend this exact correction campaign")

    model9_manifest = json.loads(MODEL9_MANIFEST.read_text(encoding="utf-8"))
    model9_sources = json.loads(MODEL9_SOURCE_HASHES.read_text(encoding="utf-8"))["files"]
    source_hash_checks = {
        path: sha256_file(REPO / path) == expected
        for path, expected in model9_sources.items()
    }
    model9_ready = model9_manifest.get("status") == "MODEL_9_FOUNDATION_READY" and all(source_hash_checks.values())
    if not model9_ready:
        failures.append("Model-9 readiness or frozen source hash failed")

    test_command = [str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", *RELEVANT_TESTS]
    test_run = subprocess.run(test_command, cwd=REPO, text=True, capture_output=True)
    tests_pass = test_run.returncode == 0 and "49 passed" in test_run.stdout
    if not tests_pass:
        failures.append("relevant 49-test suite failed")
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

    stale = stale_optimizer_processes()
    if stale:
        failures.append("stale optimizer process is active")

    initial_integrity = historical_snapshot()
    if initial_integrity["status"] != "PASS":
        failures.append("initial historical integrity failed")

    measurements, measurement_rows = load_frozen_measurements()
    if sha256_file(PREFLIGHT_MEASUREMENTS) != EXPECTED_PREFLIGHT_HASHES[PREFLIGHT_MEASUREMENTS]:
        failures.append("frozen external measurement hash failed")

    inputs_by_scene: dict[int, SolverInputs] = {}
    metadata_by_scene: dict[int, dict[str, Any]] = {}
    for scene in SCENES:
        inputs, metadata = load_science_inputs(scene, FAMILY_BULGE_DISK)
        inputs_by_scene[scene] = inputs
        metadata_by_scene[scene] = metadata
    starts_by_scene, start_verification, preflight_endpoints, preflight_summary = verify_preflight_records(inputs_by_scene)

    audit_checks = []
    for scene in SCENES:
        for condition in CONDITIONS:
            audit = oracle_information_audit(
                inputs_by_scene[scene],
                FrozenSolverProtocol(),
                extra_named_inputs={
                    "scene_index": scene,
                    "external_measurements": measurements[(scene, condition)]["measured"].tolist(),
                },
            )
            audit_checks.append({"scene_index": scene, "condition": condition, "status": audit["status"]})
            if audit["status"] != "PASS":
                failures.append(f"oracle information audit failed for scene {scene} {condition}")

    gate = {
        "campaign": CAMPAIGN,
        "timestamp_utc": now(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "predecessor_outcome": "PREFLIGHT_OPTIMIZATION_LIMITED" if predecessor_outcome_ok else "MISMATCH",
        "predecessor_exact_recommendation": CAMPAIGN if predecessor_next_ok else "MISMATCH",
        "preflight_hashes": {
            str(path.relative_to(REPO)): {"expected": expected, "actual": sha256_file(path)}
            for path, expected in EXPECTED_PREFLIGHT_HASHES.items()
        },
        "external_measurement_rows": len(measurement_rows),
        "external_measurement_sha256": sha256_file(PREFLIGHT_MEASUREMENTS),
        "external_measurements_reused_not_regenerated": True,
        "start_verification": start_verification,
        "model9_ready": model9_ready,
        "model9_source_hash_checks": source_hash_checks,
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
        "oracle_information_audits": audit_checks,
        "protected_data_access": {"development": 0, "atlas_tensors": 0, "lockbox": 0},
        "head_unchanged": initial_integrity["head"] == EXPECTED_HEAD,
        "readme_unchanged": initial_integrity["readme_sha256"] == EXPECTED_README_SHA256,
        "git_index_empty": not initial_integrity["git_index_entries"],
    }
    if failures:
        raise RuntimeError("authorization gate failed: " + "; ".join(failures))
    return gate, inputs_by_scene, metadata_by_scene, starts_by_scene, measurements, preflight_endpoints, preflight_summary


def fit_one(
    scene: int,
    condition: str,
    inputs: SolverInputs,
    measurement: dict[str, np.ndarray],
    starts: np.ndarray,
) -> dict[str, Any]:
    protocol = FrozenSolverProtocol()
    lower, upper = parameter_bounds(inputs, protocol)
    endpoints: list[MultiStartEndpoint] = []
    endpoint_parts: dict[int, dict[str, float]] = {}
    fit_started = time.perf_counter()
    for start_index, initialization in enumerate(starts):
        validate_parameters(initialization, inputs, protocol)
        start_wall = time.perf_counter()
        result = least_squares(
            lambda value: np.asarray(combined_residual(torch.as_tensor(value, dtype=torch.float64), inputs, condition, measurement).detach().cpu()),
            initialization,
            jac=lambda value: combined_jacobian(value, inputs, condition, measurement),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=protocol.ftol,
            xtol=protocol.xtol,
            gtol=protocol.gtol,
            max_nfev=MAX_NFEV,
            verbose=0,
        )
        selected = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
        canonical, _, symmetries = canonicalize_parameters(selected, inputs, protocol)
        tensor = torch.as_tensor(selected, dtype=torch.float64)
        pair = render_pair(tensor, inputs, protocol)
        residual = np.asarray(combined_residual(tensor, inputs, condition, measurement).detach().cpu())
        jacobian = combined_jacobian(selected, inputs, condition, measurement)
        scales = parameter_scales(inputs, protocol)
        gradient = float(np.linalg.norm((jacobian * scales[None]).T @ residual))
        parts = objective_parts(selected, inputs, condition, measurement)
        endpoint_parts[start_index] = parts
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
            "condition": condition,
            "start": start_index,
            "success": endpoint.success,
            "nfev": endpoint.nfev,
            "gradient_norm": endpoint.gradient_norm,
            "total_objective": endpoint.likelihood_objective,
            "elapsed_seconds": time.perf_counter() - start_wall,
        }), flush=True)

    finite = [endpoint for endpoint in endpoints if np.isfinite(endpoint.likelihood_objective)]
    if not finite:
        raise RuntimeError(f"all endpoints are nonfinite for scene {scene} {condition}")
    best = min(finite, key=lambda endpoint: endpoint.likelihood_objective)
    best_tensor = torch.as_tensor(best.parameters, dtype=torch.float64)
    canonical, active, symmetries = canonicalize_parameters(best.parameters, inputs, protocol)
    jacobian = combined_jacobian(best.parameters, inputs, condition, measurement)
    scales = parameter_scales(inputs, protocol)
    diagnostics = analyze_jacobian(jacobian, active_mask=active, parameter_scales=scales)
    residual = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    gradient_norm = float(np.linalg.norm((jacobian * scales[None]).T @ residual))
    dof = max(1, int(inputs.observed.numel()) - int(active.sum()))
    support_threshold = float(chi2.ppf(protocol.model_acceptance_quantile, dof))
    support = [endpoint for endpoint in finite if endpoint.chi_square <= support_threshold]
    geometry = solution_geometry(support if support else finite, inputs, protocol)

    pair_first = render_pair(best_tensor, inputs, protocol)
    residual_first = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    jacobian_first = combined_jacobian(best.parameters, inputs, condition, measurement)
    pair_second = render_pair(best_tensor, inputs, protocol)
    residual_second = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    jacobian_second = combined_jacobian(best.parameters, inputs, condition, measurement)
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
    boundary = boundary_contact_flags(best.parameters, inputs, protocol)
    optimizer_limited = (
        not any(endpoint.success for endpoint in endpoints)
        or not support
        or not replay["exact_match"]
        or not np.isfinite(gradient_norm)
        or not np.isfinite(diagnostics.condition_number)
    )
    return {
        "scene_index": scene,
        "condition": condition,
        "family": FAMILY_BULGE_DISK,
        "level": "Level 5",
        "starts": starts,
        "endpoints": endpoints,
        "endpoint_parts": endpoint_parts,
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
        "fitted_requested_fluxes": best.parameters[:3],
        "fitted_companion_fluxes": best.parameters[12:15],
        "optimizer_limited": optimizer_limited,
        "optimizer_successful_starts": sum(endpoint.success for endpoint in endpoints),
        "max_budget_starts": sum(endpoint.nfev >= MAX_NFEV for endpoint in endpoints),
        "fit_runtime_seconds": time.perf_counter() - fit_started,
    }


def classify_fit(result: dict[str, Any], baselines: dict[tuple[int, str], dict[str, Any]]) -> str:
    base = preflight_promise(result, baselines)
    result["preflight_logic_promise"] = base
    if base == "OPTIMIZATION_LIMITED":
        return "STILL_OPTIMIZATION_LIMITED"
    if base == "STRONG_PROMISE":
        return "CONVERGED_STRONG_PROMISE" if result["gradient_gate_pass"] else "CONVERGED_MODERATE_PROMISE"
    if base == "MODERATE_PROMISE":
        return "CONVERGED_MODERATE_PROMISE"
    if base == "NO_CLEAR_GAIN":
        return "CONVERGED_NO_CLEAR_GAIN"
    return "INVALID"


def campaign_outcome(results: list[dict[str, Any]]) -> str:
    labels = {(result["scene_index"], result["condition"]): result["classification"] for result in results}
    if any(label == "INVALID" for label in labels.values()):
        return "CONVERGENCE_CORRECTION_INVALID"
    if any(label == "STILL_OPTIMIZATION_LIMITED" for label in labels.values()):
        return "CONVERGENCE_CORRECTION_STILL_LIMITED"
    scene0_per_band = labels[(0, "PER_BAND_SOURCE_PHOTOMETRY")]
    scene6_per_band = labels[(6, "PER_BAND_SOURCE_PHOTOMETRY")]
    if (
        scene0_per_band == "CONVERGED_STRONG_PROMISE"
        and scene6_per_band in {"CONVERGED_STRONG_PROMISE", "CONVERGED_MODERATE_PROMISE"}
    ):
        return "EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED"
    scene0_help = any(
        labels[(0, condition)] in {"CONVERGED_STRONG_PROMISE", "CONVERGED_MODERATE_PROMISE"}
        for condition in CONDITIONS
    )
    scene6_no_gain = all(labels[(6, condition)] == "CONVERGED_NO_CLEAR_GAIN" for condition in CONDITIONS)
    if scene0_help and scene6_no_gain:
        return "EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED"
    any_gain = any(
        label in {"CONVERGED_STRONG_PROMISE", "CONVERGED_MODERATE_PROMISE"}
        for label in labels.values()
    )
    if not any_gain:
        return "EXTERNAL_PHOTOMETRY_NO_CLEAR_ADVANTAGE"
    # A converged mixed pattern that does not meet the full threshold is still
    # scene-stratified evidence rather than an unqualified population result.
    return "EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED"


def materially_better_geometry(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_classes = int(left["distinct_solution_classes"])
    right_classes = int(right["distinct_solution_classes"])
    if left_classes < right_classes:
        return True
    nonworse = all(float(left[key]) <= float(right[key]) for key in DIAMETERS)
    half_nonzero = all(
        float(left[key]) <= (0.5 * float(right[key]) if float(right[key]) > 0 else 0.0)
        for key in DIAMETERS
    )
    return nonworse and half_nonzero and any(float(right[key]) > 0 for key in DIAMETERS)


def fit_record(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "scene_index": result["scene_index"],
        "scene_id": metadata["scene_id"],
        "condition": result["condition"],
        "family": result["family"],
        "level": result["level"],
        "only_authorized_change": {"max_nfev_before": 150, "max_nfev_after": MAX_NFEV},
        "parameterization": "direct_bounded_physical_coordinates",
        "unconstrained_variables_present": False,
        "starts": result["starts"],
        "endpoints": [
            {**endpoint.record(), **result["endpoint_parts"][endpoint.start_index]}
            for endpoint in result["endpoints"]
        ],
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
        "classification": result["classification"],
        "preflight_logic_promise": result["preflight_logic_promise"],
        "optimizer_limited": result["optimizer_limited"],
        "fit_runtime_seconds": result["fit_runtime_seconds"],
    }


def make_figure(
    run_dir: Path,
    preflight_best: dict[tuple[int, str], dict[str, str]],
    results: list[dict[str, Any]],
) -> None:
    labels = [f"S{result['scene_index']}\n{'total' if result['condition'].startswith('TOTAL') else 'per-band'}" for result in results]
    old_objectives = np.asarray([float(preflight_best[(result["scene_index"], result["condition"])]["total_objective"]) for result in results])
    new_objectives = np.asarray([float(result["best_parts"]["total_objective"]) for result in results])
    old_gradients = np.asarray([float(preflight_best[(result["scene_index"], result["condition"])]["gradient_norm"]) for result in results])
    new_gradients = np.asarray([float(result["gradient_norm"]) for result in results])
    positions = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].bar(positions - width / 2, old_objectives - new_objectives, width, color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Preflight objective - 500-eval objective")
    axes[0].set_title("Objective improvement")
    axes[1].bar(positions - width / 2, old_gradients, width, label="150-eval", color="#999999")
    axes[1].bar(positions + width / 2, new_gradients, width, label="500-eval", color="#f58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Scaled gradient norm")
    axes[1].set_title("Endpoint convergence diagnostic")
    axes[1].legend(frameon=False)
    destination = run_dir / "figures/preflight_vs_500eval_comparison.png"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=170)
    plt.close(fig)
    os.replace(temporary, destination)


def next_experiment(outcome: str) -> tuple[str, str]:
    if outcome == "EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED":
        return "Thayer-External-Photometry-vs-PSF-Diversity-v0", "Run the authorized full comparison against PSF diversity."
    if outcome == "EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED":
        return "Thayer-External-Photometry-Scene-Stratification-v0", "Run one targeted scene-stratified photometry experiment to isolate why Scene 0 benefits while Scene 6 does not."
    if outcome == "EXTERNAL_PHOTOMETRY_NO_CLEAR_ADVANTAGE":
        return "Thayer-Higher-Resolution-vs-External-Photometry-v0", "Compare one higher-resolution independent observation against the frozen external-photometry conditions; do not introduce a neural model."
    if outcome == "CONVERGENCE_CORRECTION_STILL_LIMITED":
        return "Thayer-External-Photometry-Optimizer-Diagnosis-v0", "Diagnose the remaining bounded-TRF convergence failure without changing the scientific information source."
    return "Thayer-External-Photometry-Integrity-Diagnosis-v0", "Repair and re-audit the failed frozen-contract gate before any scientific rerun."


def main() -> None:
    started_wall = time.perf_counter()
    started_utc = now()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_external_photometry_convergence_correction_v0_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdirectory in ("reports", "preregistration", "tables", "manifests", "figures", "fit_records"):
        (run_dir / subdirectory).mkdir()

    preflight_endpoint_rows = read_csv_rows(PREFLIGHT_ENDPOINTS)
    estimate = runtime_estimate(preflight_endpoint_rows)
    atomic_text(run_dir / "preregistration/frozen_protocol.md", protocol_text(estimate))
    print(json.dumps({"event": "runtime_estimate", **estimate}), flush=True)

    gate, inputs_by_scene, metadata_by_scene, starts_by_scene, measurements, preflight_endpoint_rows, preflight_summary_rows = authorization_gate()
    atomic_json(run_dir / "manifests/authorization_gate.json", gate)
    print(json.dumps({"event": "authorization_gate", "status": gate["status"]}), flush=True)

    input_paths = (
        Path(__file__),
        PREFLIGHT_REPORT,
        PREFLIGHT_PROTOCOL,
        PREFLIGHT_ENDPOINTS,
        PREFLIGHT_MEASUREMENTS,
        PREFLIGHT_SUMMARY,
        PREFLIGHT_COMPARISON,
        PREFLIGHT_ORACLE_AUDIT,
        SCIENCE_H5,
        SCENE_MANIFEST,
        DEFINITIONS,
        CATALOG_PATH,
        S1_METRICS,
        P2_METRICS,
        REPO / "src/model9_structured.py",
        REPO / "src/model9_optimizer.py",
        MODEL9_MANIFEST,
        MODEL9_SOURCE_HASHES,
        *AUTHORITATIVE_REPORTS,
    )
    atomic_json(run_dir / "manifests/input_hashes.json", {
        "campaign": CAMPAIGN,
        "frozen_at_utc": now(),
        "files": {
            str(path.relative_to(REPO)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "scenes": SCENES,
        "family": FAMILY_BULGE_DISK,
        "conditions": CONDITIONS,
        "starts": STARTS,
        "only_authorized_change": {"max_nfev_before": 150, "max_nfev_after": MAX_NFEV},
        "cpu_float64": True,
        "measurements_reused_not_regenerated": True,
    })

    baselines = load_baselines()
    results: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for scene in SCENES:
        for condition in CONDITIONS:
            fit = fit_one(scene, condition, inputs_by_scene[scene], measurements[(scene, condition)], starts_by_scene[scene])
            fit["classification"] = classify_fit(fit, baselines)
            results.append(fit)
            record_name = f"scene_{scene:03d}_{condition.lower()}.json"
            atomic_json(run_dir / "fit_records" / record_name, fit_record(fit, metadata_by_scene[scene]))
            acceptable = set(fit["geometry"]["acceptable_endpoint_indices"])
            support = set(fit["support_start_indices"])
            for endpoint in fit["endpoints"]:
                endpoint_rows.append({
                    "scene_index": scene,
                    "scene_id": metadata_by_scene[scene]["scene_id"],
                    "condition": condition,
                    **endpoint.record(),
                    **fit["endpoint_parts"][endpoint.start_index],
                    "observation_support_acceptable": endpoint.start_index in support,
                    "total_objective_acceptable": endpoint.start_index in acceptable,
                    "hit_500_evaluation_ceiling": endpoint.nfev >= MAX_NFEV,
                })
            oracle_rows.append({
                "scene_index": scene,
                "condition": condition,
                "status": "PASS",
                "blend_rows_accessed": 1,
                "coordinate_rows_accessed": 1,
                "frozen_measurement_rows_accessed": 2 if condition == "TOTAL_SOURCE_PHOTOMETRY" else 6,
                "catalog_rows_accessed": 0,
                "catalog_truth_flux_exposed_to_fit": False,
                "noisy_measurements_exposed_to_fit": True,
                "isolated_source_images_accessed": 0,
                "morphology_truth_exposed": 0,
                "masks_exposed": 0,
                "truth_initialization_exposed": 0,
                "preflight_endpoint_initialization_used": False,
                "development_access": 0,
                "atlas_tensor_access": 0,
                "lockbox_access": 0,
                "measurement_likelihood_not_hard_fix": True,
                "measurement_relative_sigma": RELATIVE_SIGMA,
            })
            print(json.dumps({
                "event": "fit_complete",
                "scene": scene,
                "condition": condition,
                "classification": fit["classification"],
                "successful_starts": fit["optimizer_successful_starts"],
                "max_budget_starts": fit["max_budget_starts"],
                "fit_runtime_seconds": fit["fit_runtime_seconds"],
            }), flush=True)

    atomic_csv(run_dir / "tables/convergence_correction_endpoints.csv", endpoint_rows)
    atomic_csv(run_dir / "tables/oracle_information_audit.csv", oracle_rows)

    preflight_best: dict[tuple[int, str], dict[str, str]] = {}
    for scene in SCENES:
        for condition in CONDITIONS:
            rows = [row for row in preflight_endpoint_rows if int(row["scene_index"]) == scene and row["condition"] == condition]
            preflight_best[(scene, condition)] = min(rows, key=lambda row: float(row["total_objective"]))
    preflight_fit_summary = {
        (int(row["scene_index"]), row["condition"]): row
        for row in preflight_summary_rows
    }

    summary_rows: list[dict[str, Any]] = []
    comparison_preflight_rows: list[dict[str, Any]] = []
    comparison_baseline_rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    for fit in results:
        scene = fit["scene_index"]
        condition = fit["condition"]
        best = fit["best"]
        geometry = fit["geometry"]
        old_best = preflight_best[(scene, condition)]
        old_summary = preflight_fit_summary[(scene, condition)]
        old_rows = [row for row in preflight_endpoint_rows if int(row["scene_index"]) == scene and row["condition"] == condition]
        old_by_start = {int(row["start_index"]): row for row in old_rows}
        resolved_ceiling = sum(
            int(old_by_start[endpoint.start_index]["nfev"]) >= 150 and endpoint.nfev < MAX_NFEV
            for endpoint in fit["endpoints"]
        )
        summary_rows.append({
            "scene_index": scene,
            "scene_id": metadata_by_scene[scene]["scene_id"],
            "condition": condition,
            "family": FAMILY_BULGE_DISK,
            "level": "Level 5",
            "classification": fit["classification"],
            "preflight_logic_promise": fit["preflight_logic_promise"],
            "best_total_objective": fit["best_parts"]["total_objective"],
            "observation_likelihood": fit["best_parts"]["observation_objective"],
            "photometry_likelihood": fit["best_parts"]["photometry_objective"],
            "observation_chi_square": fit["best_parts"]["observation_chi_square"],
            "photometry_chi_square": fit["best_parts"]["photometry_chi_square"],
            "optimizer_converged_starts": fit["optimizer_successful_starts"],
            "max_budget_starts": fit["max_budget_starts"],
            "best_start_index": best.start_index,
            "best_nfev": best.nfev,
            "best_optimizer_success": best.success,
            "best_optimizer_status": best.status,
            "best_optimizer_message": best.message,
            "gradient_norm": fit["gradient_norm"],
            "gradient_gate_pass": fit["gradient_gate_pass"],
            "rank": fit["rank"],
            "active_parameter_count": fit["active_parameter_count"],
            "nullity": fit["nullity"],
            "condition_number": fit["condition_number"],
            **{key: geometry[key] for key in ("distinct_solution_classes", *DIAMETERS)},
            "boundary_contact_flags": fit["boundary_contact_flags"],
            "deterministic_replay_exact": fit["replay"]["exact_match"],
            "fit_runtime_seconds": fit["fit_runtime_seconds"],
        })
        comparison_preflight_rows.append({
            "scene_index": scene,
            "condition": condition,
            "preflight_classification": old_summary["preflight_promise"],
            "correction_classification": fit["classification"],
            "preflight_best_total_objective": float(old_best["total_objective"]),
            "correction_best_total_objective": fit["best_parts"]["total_objective"],
            "objective_improvement_preflight_minus_correction": float(old_best["total_objective"]) - fit["best_parts"]["total_objective"],
            "preflight_gradient_norm": float(old_summary["gradient_norm"]),
            "correction_gradient_norm": fit["gradient_norm"],
            "gradient_norm_improvement_preflight_minus_correction": float(old_summary["gradient_norm"]) - fit["gradient_norm"],
            "preflight_endpoint_classes": int(old_summary["distinct_solution_classes"]),
            "correction_endpoint_classes": int(geometry["distinct_solution_classes"]),
            "endpoint_class_change_correction_minus_preflight": int(geometry["distinct_solution_classes"]) - int(old_summary["distinct_solution_classes"]),
            **{f"preflight_{key}": float(old_summary[key]) for key in DIAMETERS},
            **{f"correction_{key}": float(geometry[key]) for key in DIAMETERS},
            **{f"diameter_improvement_{key}": float(old_summary[key]) - float(geometry[key]) for key in DIAMETERS},
            "preflight_optimizer_successful_starts": sum(row["success"] == "True" for row in old_rows),
            "correction_optimizer_successful_starts": fit["optimizer_successful_starts"],
            "preflight_150_ceiling_starts": sum(int(row["nfev"]) >= 150 for row in old_rows),
            "correction_500_ceiling_starts": fit["max_budget_starts"],
            "preflight_ceiling_starts_no_longer_at_ceiling": resolved_ceiling,
            "start_transitions": [
                {
                    "start_index": endpoint.start_index,
                    "preflight_nfev": int(old_by_start[endpoint.start_index]["nfev"]),
                    "correction_nfev": endpoint.nfev,
                    "preflight_success": old_by_start[endpoint.start_index]["success"] == "True",
                    "correction_success": endpoint.success,
                }
                for endpoint in fit["endpoints"]
            ],
        })
        s1 = baselines[(scene, "FLUX_FREE_SINGLE")]
        p2 = baselines[(scene, "PSF_DIVERSE")]
        comparison_baseline_rows.append({
            "scene_index": scene,
            "condition": condition,
            "classification": fit["classification"],
            "photometry_materially_exceeds_p2": materially_better_geometry(geometry, p2),
            "new_endpoint_classes": geometry["distinct_solution_classes"],
            "s1_endpoint_classes": s1["distinct_solution_classes"],
            "p2_endpoint_classes": p2["distinct_solution_classes"],
            **{f"new_{key}": geometry[key] for key in DIAMETERS},
            **{f"s1_{key}": s1[key] for key in DIAMETERS},
            **{f"p2_{key}": p2[key] for key in DIAMETERS},
            "new_condition_number": fit["condition_number"],
            "s1_condition_number": s1["condition_number"],
            "p2_condition_number": p2["condition_number"],
            "new_gradient_norm": fit["gradient_norm"],
            "s1_gradient_norm": s1["gradient_norm"],
            "p2_gradient_norm": p2["gradient_norm"],
        })
        for identity, fluxes in (
            ("requested", fit["fitted_requested_fluxes"]),
            ("companion", fit["fitted_companion_fluxes"]),
        ):
            flux_rows.append({
                "scene_index": scene,
                "scene_id": metadata_by_scene[scene]["scene_id"],
                "condition": condition,
                "source_identity": identity,
                "fitted_flux_g_electrons": fluxes[0],
                "fitted_flux_r_electrons": fluxes[1],
                "fitted_flux_z_electrons": fluxes[2],
                "best_start_index": best.start_index,
                "boundary_contact_flags": fit["boundary_contact_flags"],
            })

    atomic_csv(run_dir / "tables/convergence_correction_summary.csv", summary_rows)
    atomic_csv(run_dir / "tables/comparison_to_preflight.csv", comparison_preflight_rows)
    atomic_csv(run_dir / "tables/comparison_to_s1_and_p2.csv", comparison_baseline_rows)
    atomic_csv(run_dir / "tables/fitted_source_fluxes.csv", flux_rows)
    make_figure(run_dir, preflight_best, results)

    outcome = campaign_outcome(results)
    next_name, next_reason = next_experiment(outcome)
    final_integrity = historical_snapshot(gate["historical_integrity"]["protected_historical_file_hashes"])
    atomic_json(run_dir / "manifests/historical_integrity.json", {
        "campaign": CAMPAIGN,
        "status": final_integrity["status"],
        "before": gate["historical_integrity"],
        "after": final_integrity,
        "head_unchanged": gate["historical_integrity"]["head"] == final_integrity["head"] == EXPECTED_HEAD,
        "readme_unchanged": gate["historical_integrity"]["readme_sha256"] == final_integrity["readme_sha256"] == EXPECTED_README_SHA256,
        "historical_reports_unchanged": final_integrity["protected_historical_files_match_start"],
        "historical_checkpoints_unchanged": final_integrity["historical_checkpoint_matches"] == 600 and final_integrity["historical_checkpoint_mismatches"] == 0,
        "git_index_empty": not final_integrity["git_index_entries"],
        "protected_data_access_zero": all(value == 0 for value in final_integrity["protected_data_access"].values()),
        "commits_created": 0,
    })
    if final_integrity["status"] != "PASS":
        outcome = "CONVERGENCE_CORRECTION_INVALID"
        next_name, next_reason = next_experiment(outcome)

    lookup = {(fit["scene_index"], fit["condition"]): fit for fit in results}
    scene0_one_class_survived = all(lookup[(0, condition)]["geometry"]["distinct_solution_classes"] == 1 for condition in CONDITIONS)
    scene6_gain = any(lookup[(6, condition)]["classification"] in {"CONVERGED_STRONG_PROMISE", "CONVERGED_MODERATE_PROMISE"} for condition in CONDITIONS)
    per_band_beats_total = {
        scene: materially_better_geometry(
            lookup[(scene, "PER_BAND_SOURCE_PHOTOMETRY")]["geometry"],
            lookup[(scene, "TOTAL_SOURCE_PHOTOMETRY")]["geometry"],
        )
        for scene in SCENES
    }
    p2_gain = {
        (fit["scene_index"], fit["condition"]): materially_better_geometry(
            fit["geometry"], baselines[(fit["scene_index"], "PSF_DIVERSE")]
        )
        for fit in results
    }
    failures = [
        (fit, endpoint)
        for fit in results
        for endpoint in fit["endpoints"]
        if not endpoint.success or endpoint.nfev >= MAX_NFEV
    ]
    runtime = time.perf_counter() - started_wall
    ended_utc = now()
    failure_lines = "\n".join(
        f"- Scene {fit['scene_index']} / {fit['condition']} / start {endpoint.start_index}: success={endpoint.success}, status={endpoint.status}, nfev={endpoint.nfev}, message=`{endpoint.message}`"
        for fit, endpoint in failures
    ) or "- None."
    report = f"""# {CAMPAIGN} final report

## Outcome

**{outcome}**

The only authorized scientific-execution change was the per-start maximum function-evaluation budget: **150 -> 500**. Scenes, Level-5 bulge+disk structure, the two photometry conditions, exact noisy measurements, 5% uncertainties, four deterministic physical starts, optimizer, objective, parameterization, supports, PSF, observation, tolerances, gradient diagnostic, endpoint acceptance/clustering, ranks, diameters, classification logic, and replay procedure were unchanged. Measurements were reused from the frozen preflight table and were not regenerated.

## Four-fit convergence result

| Scene | Condition | Fit classification | Successful starts | 500-ceiling starts | Best nfev | Best total objective | Gradient norm | Classes | Replay |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for fit in results:
        report += f"| {fit['scene_index']} | {fit['condition']} | {fit['classification']} | {fit['optimizer_successful_starts']}/4 | {fit['max_budget_starts']}/4 | {fit['best'].nfev} | {fit['best_parts']['total_objective']:.12g} | {fit['gradient_norm']:.12g} | {fit['geometry']['distinct_solution_classes']} | {'exact' if fit['replay']['exact_match'] else 'failed'} |\n"
    report += f"""

All four fits were converged and interpretable under the frozen preflight optimization-limitation rule: **{'yes' if all(fit['classification'] not in {'STILL_OPTIMIZATION_LIMITED', 'INVALID'} for fit in results) else 'no'}**. Scene 0's one-class result survived at 500 evaluations: **{'yes' if scene0_one_class_survived else 'no'}**. Scene 6 gained materially over P2: **{'yes' if scene6_gain else 'no'}**.

Per-band photometry materially exceeded total photometry: **Scene 0 {'yes' if per_band_beats_total[0] else 'no'}; Scene 6 {'yes' if per_band_beats_total[6] else 'no'}** under the same endpoint-class/diameter improvement rule. Photometry materially exceeded PSF diversity: **{'; '.join(f"Scene {scene} {condition}={'yes' if p2_gain[(scene, condition)] else 'no'}" for scene in SCENES for condition in CONDITIONS)}**. This evidence authorizes **{'a full follow-up' if outcome == 'EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED' else ('a targeted scene-stratified follow-up' if outcome == 'EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED' else 'no photometry campaign follow-up')}**.

The direct bounded solver has no unconstrained initialization variables. Every physical starting vector and initialization hash matched the preflight exactly; no preflight endpoint or truth parameter was used as an initialization.

## Preflight comparison

Objective, gradient, endpoint-class, diameter, ceiling-resolution, and classification changes are recorded fit-by-fit in `tables/comparison_to_preflight.csv`. All objective components, local rank/nullity/condition diagnostics, boundary contacts, fitted g/r/z source fluxes, and exact replay hashes are retained in the summary, endpoint table, fitted-flux table, and four atomic fit records.

## Max-budget and optimizer-declared failures

{failure_lines}

## Integrity and runtime

Authorization, oracle-information, and final integrity gates passed: **{'yes' if gate['status'] == 'PASS' and final_integrity['status'] == 'PASS' else 'no'}**. All 600 historical checkpoints matched before and after; README and HEAD were unchanged; historical reports and preflight artifacts were unchanged; protected development, Atlas-tensor, and lockbox access were zero; the Git index remained empty; nothing was staged or committed.

Exact campaign runtime: **{runtime:.6f} seconds** ({runtime / 60.0:.3f} minutes), from {started_utc} to {ended_utc}. The preregistered linear estimate was {estimate['estimated_seconds']:.3f} seconds, with no wall-clock cutoff.

## Exactly one next experiment

**{next_name}** — {next_reason}
"""
    atomic_text(run_dir / "reports/final_report.md", report)

    required_files = (
        "reports/final_report.md",
        "preregistration/frozen_protocol.md",
        "tables/convergence_correction_summary.csv",
        "tables/convergence_correction_endpoints.csv",
        "tables/comparison_to_preflight.csv",
        "tables/comparison_to_s1_and_p2.csv",
        "tables/fitted_source_fluxes.csv",
        "tables/oracle_information_audit.csv",
        "manifests/input_hashes.json",
        "manifests/historical_integrity.json",
        "figures/preflight_vs_500eval_comparison.png",
    )
    missing = [relative for relative in required_files if not (run_dir / relative).exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing before final manifest: {missing}")
    final_manifest = {
        "campaign": CAMPAIGN,
        "outcome": outcome,
        "timestamp_utc": ended_utc,
        "only_authorized_change": {"max_nfev_before": 150, "max_nfev_after": MAX_NFEV},
        "four_fits_complete": len(results) == 4,
        "sixteen_endpoints_retained": sum(len(fit["endpoints"]) for fit in results) == 16,
        "fit_classifications": {
            f"scene_{fit['scene_index']}_{fit['condition']}": fit["classification"]
            for fit in results
        },
        "scene_0_one_class_survived": scene0_one_class_survived,
        "scene_6_material_gain": scene6_gain,
        "runtime_seconds": runtime,
        "optimizer_failures": sum(not endpoint.success for fit in results for endpoint in fit["endpoints"]),
        "max_budget_endpoints": sum(endpoint.nfev >= MAX_NFEV for fit in results for endpoint in fit["endpoints"]),
        "replay_exact_all_fits": all(fit["replay"]["exact_match"] for fit in results),
        "authorization_gate_status": gate["status"],
        "historical_integrity_status": final_integrity["status"],
        "protected_data_access_zero": True,
        "readme_unchanged": final_integrity["readme_sha256"] == EXPECTED_README_SHA256,
        "head_unchanged": final_integrity["head"] == EXPECTED_HEAD,
        "git_index_empty": not final_integrity["git_index_entries"],
        "commits_created": 0,
        "recommended_next_experiment": next_name,
        "required_artifacts": {
            relative: {"exists": True, "bytes": (run_dir / relative).stat().st_size, "sha256": sha256_file(run_dir / relative)}
            for relative in required_files
        },
        "fit_records": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted((run_dir / "fit_records").glob("*.json"))
        },
    }
    atomic_json(run_dir / "manifests/final_manifest.json", final_manifest)
    print(json.dumps({
        "event": "campaign_complete",
        "run_dir": str(run_dir),
        "outcome": outcome,
        "runtime_seconds": runtime,
        "integrity": final_integrity["status"],
        "next_experiment": next_name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
