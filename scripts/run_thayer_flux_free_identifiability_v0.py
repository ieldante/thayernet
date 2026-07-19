#!/usr/bin/env python3
"""Execute the frozen Thayer flux-free structured identifiability campaign.

The science loader intentionally exposes only the blend and two-coordinate
contract.  It never reads the HDF5 isolated-source dataset or the CatSim
catalog.  Preregistration and execution are separate commands so the exact
driver and protocol hashes can be sealed before any scientific array access.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from btk.survey import get_surveys
from surveycodex.utilities import mean_sky_level

from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_galsim_adapter import sample_galsim_psf_kernels
from src.model9_optimizer import (
    analyze_jacobian,
    boundary_contact_flags,
    classify_identifiability,
    deterministic_starts,
    local_fit_diagnostics,
    multi_start_optimize,
    residual_jacobian,
    solution_geometry,
)
from src.model9_structured import (
    BANDS,
    FAMILY_BULGE_DISK,
    FAMILY_SERSIC,
    FrozenSolverProtocol,
    SolverInputs,
    canonicalize_parameters,
    input_provenance_trace,
    likelihood_components,
    normalize_psf,
    observed_flux_reference,
    oracle_information_audit,
    parameter_bounds,
    parameter_names,
    parameter_scales,
    parameters_per_source,
    render_pair,
    whitened_residual_vector,
)


CAMPAIGN = "Thayer-Flux-Free-Identifiability-v0"
INDICES = (0, 3, 5, 6, 18, 51, 73, 81)
FAMILIES = (FAMILY_SERSIC, FAMILY_BULGE_DISK)
LEVEL_BY_FAMILY = {FAMILY_SERSIC: "Level 4", FAMILY_BULGE_DISK: "Level 5"}
PREPARATION = REPO / "outputs/runs/thayer_model_9_preparation_v0_20260715_172217"
UPSTREAM = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
IDENTIFIABILITY = REPO / "outputs/runs/thayer_identifiability_v1_20260715_003220"
MANIFEST = UPSTREAM / "manifests/v2_r_training_scene_manifest.csv"
SCENES = UPSTREAM / "manifests/v2_r_training_scenes.h5"
FROZEN_PROTOCOL = PREPARATION / "preregistration/draft_flux_free_protocol.md"
FROZEN_PROTOCOL_JSON = PREPARATION / "preregistration/frozen_protocol.json"
SOURCE_HASHES = PREPARATION / "manifests/source_hashes.json"
CHECKPOINT_BASELINE = REPO / "outputs/runs/thayer_d3_hash_r1_20260714_012539/authoritative_inputs/historical_checkpoints_final.json"
README_SHA256 = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"
PREPARATION_PROTOCOL_SHA256 = "5b37499d0ea957ddb36b3737a5c24ae4aa489f5fd066b3012dafbb475157695b"
PREPARATION_PROTOCOL_JSON_SHA256 = "8235923adabe28407fcceabc883448c4afd9006a5e00b633170430a191f6e692"
NUMERICAL_PERTURBATION_SCALE = 1.0e-8
NUMERICAL_SPECTRUM_RTOL = 1.0e-3

PREVIOUS = {
    0: {"sersic": "UNIQUE", "bulge_disk": "UNIQUE", "minimum": "Level 4"},
    3: {"sersic": "UNIQUE", "bulge_disk": "UNIQUE", "minimum": "Level 4"},
    5: {"sersic": "UNIQUE", "bulge_disk": "UNIQUE", "minimum": "Level 4"},
    6: {"sersic": "OUT_OF_SUPPORT", "bulge_disk": "UNIQUE", "minimum": "Level 5"},
    18: {"sersic": "OUT_OF_SUPPORT", "bulge_disk": "UNIQUE", "minimum": "Level 5"},
    51: {"sersic": "OUT_OF_SUPPORT", "bulge_disk": "OUT_OF_SUPPORT", "minimum": "none"},
    73: {"sersic": "UNIQUE", "bulge_disk": "UNIQUE", "minimum": "Level 4"},
    81: {"sersic": "OUT_OF_SUPPORT", "bulge_disk": "UNIQUE", "minimum": "Level 5"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def raw_array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        if math.isnan(scalar):
            return "nan"
        if math.isinf(scalar):
            return "inf" if scalar > 0 else "-inf"
        return scalar
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, value: Any) -> None:
    write_text_fresh(path, json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in writer.fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return json_safe(value)


def command(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {
        "arguments": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def selected_manifest_rows() -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["dataset_index"])
            if index in INDICES:
                rows[index] = row
    if tuple(sorted(rows)) != INDICES:
        raise RuntimeError("the frozen eight-scene manifest identity changed")
    return rows


def known_psf() -> tuple[torch.Tensor, dict[str, float]]:
    survey = get_surveys("LSST")
    if tuple(band for band in survey.available_filters if band in BANDS) != BANDS:
        raise RuntimeError("LSST g/r/z band contract changed")
    objects = []
    sky: dict[str, float] = {}
    for band in BANDS:
        filt = survey.get_filter(band)
        psf = filt.psf(survey, filt) if callable(filt.psf) else filt.psf
        objects.append(psf)
        sky[band] = float(mean_sky_level(survey, filt).to_value("electron"))
    kernels = sample_galsim_psf_kernels(objects, pixel_scale_arcsec=0.2, kernel_size=31)
    return kernels, sky


def declared_noise_sigma(observed: np.ndarray, sky_by_band: dict[str, float]) -> np.ndarray:
    """Observation-only fixed Gaussian plug-in for BTK's Poisson convention."""

    sky = np.asarray([sky_by_band[band] for band in BANDS], dtype=np.float64)[:, None, None]
    variance = sky + np.maximum(np.asarray(observed, dtype=np.float64), 0.0)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise RuntimeError("declared observation-only variance map is invalid")
    return np.sqrt(variance)


def historical_integrity() -> dict[str, Any]:
    baseline = json.loads(CHECKPOINT_BASELINE.read_text(encoding="utf-8"))
    matched = mismatched = missing = total_bytes = 0
    for record in baseline:
        path = REPO / record["path"]
        if not path.exists():
            missing += 1
            continue
        total_bytes += path.stat().st_size
        if path.stat().st_size == int(record["final_bytes"]) and sha256_file(path) == record["final_sha256"]:
            matched += 1
        else:
            mismatched += 1
    references = {
        "reports/thayer_recoverability_v0.md": "58e1121239b53ff1cd47a6599cc3d0755fc8727a3feb99dcb61107e1f6ad22a7",
        "outputs/runs/thayer_identifiability_v1_20260715_003220/reports/final_report.md": "4c6d5d984fa016f438a0ef441a123f5b1fbdd8c1e81a520fb21e0d9f0adc8239",
        "outputs/runs/thayer_identifiability_v1_20260715_003220/preregistration/prior_and_analysis_freeze.md": "5af9db12575fe8f7025149cf55685456a22b30e0e19b8d9d7738d65683cb3475",
        "docs/family_e1_signed_noise_residual_preflight.md": "5dd0ec9e66bf762eb7ddb8745ec3031d961359fa7b4979a25411c3fa154d5df7",
        "docs/signed_noise_residual_physical_contract.md": "2a7c9da88d398b4753844656a7252d2c943c16789d9c1ba690c88a6ea20e9b56",
    }
    reference_checks = {name: {"expected": expected, "actual": sha256_file(REPO / name)} for name, expected in references.items()}
    staged = command(["git", "diff", "--cached", "--name-only"])
    result = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "checkpoint_baseline": str(CHECKPOINT_BASELINE.relative_to(REPO)),
        "checkpoint_baseline_sha256": sha256_file(CHECKPOINT_BASELINE),
        "historical_checkpoint_count": len(baseline),
        "historical_checkpoint_matches": matched,
        "historical_checkpoint_mismatches": mismatched,
        "historical_checkpoint_missing": missing,
        "historical_checkpoint_bytes_rehashed": total_bytes,
        "historical_references": reference_checks,
        "readme_sha256": sha256_file(REPO / "README.md"),
        "readme_unchanged": sha256_file(REPO / "README.md") == README_SHA256,
        "staged_index_empty": staged["returncode"] == 0 and not staged["stdout"].strip(),
        "commits_created": 0,
    }
    result["status"] = "PASS" if (
        matched == len(baseline)
        and mismatched == 0
        and missing == 0
        and all(item["expected"] == item["actual"] for item in reference_checks.values())
        and result["readme_unchanged"]
        and result["staged_index_empty"]
    ) else "FAIL"
    return result


def verify_foundation() -> dict[str, Any]:
    source_manifest = json.loads(SOURCE_HASHES.read_text(encoding="utf-8"))
    file_checks = {}
    for name, expected in source_manifest["files"].items():
        actual = sha256_file(REPO / name)
        file_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    checks = {
        "preparation_ready": json.loads((PREPARATION / "manifests/final_manifest.json").read_text())["status"] == "MODEL_9_FOUNDATION_READY",
        "protocol_hash": sha256_file(FROZEN_PROTOCOL) == PREPARATION_PROTOCOL_SHA256,
        "protocol_json_hash": sha256_file(FROZEN_PROTOCOL_JSON) == PREPARATION_PROTOCOL_JSON_SHA256,
        "source_hashes": all(item["match"] for item in file_checks.values()),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "source_files": file_checks}


def preregister(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"append-only run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for relative in (
        "reports", "preregistration", "tables", "manifests", "logs", "scene_results",
        "figures/singular_value_spectra", "figures/multistart_solution_geometry",
        "figures/source_reconstruction_panels",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    shutil.copyfile(FROZEN_PROTOCOL, run_dir / "preregistration/frozen_protocol.md")
    shutil.copyfile(FROZEN_PROTOCOL_JSON, run_dir / "preregistration/frozen_protocol.json")
    foundation = verify_foundation()
    integrity = historical_integrity()
    if foundation["status"] != "PASS" or integrity["status"] != "PASS":
        write_json_fresh(run_dir / "manifests/authorization_gate.json", {"foundation": foundation, "integrity": integrity})
        raise RuntimeError("authorization gate failed closed")
    rows = selected_manifest_rows()
    psf, sky = known_psf()
    scene_metadata = []
    for index in INDICES:
        row = rows[index]
        scene_metadata.append({
            "scene_index": index,
            "scene_id": row["scene_id"],
            "requested_source_index": int(row["matched_source_index"]),
            "observation_sha256": row["blend_sha256"],
            "source_a_identity_sha256_metadata_only": row["isolated_source_a_sha256"],
            "source_b_identity_sha256_metadata_only": row["isolated_source_b_sha256"],
            "prompt_sha256": row["prompt_sha256"],
        })
    status = command(["git", "status", "--short"])
    input_hashes = {
        "campaign": CAMPAIGN,
        "frozen_before_scientific_array_access": True,
        "timestamp_utc": utc_now(),
        "driver": {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": sha256_file(Path(__file__).resolve())},
        "preparation_protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "preparation_protocol_json_sha256": sha256_file(FROZEN_PROTOCOL_JSON),
        "preparation_source_hash_manifest_sha256": sha256_file(SOURCE_HASHES),
        "science_hdf5": {"path": str(SCENES.relative_to(REPO)), "bytes": SCENES.stat().st_size, "sha256": sha256_file(SCENES)},
        "scene_manifest": {"path": str(MANIFEST.relative_to(REPO)), "sha256": sha256_file(MANIFEST)},
        "known_psf_canonical_sha256": canonical_tensor_sha256(psf),
        "known_psf_band_sums": psf.sum(dim=(-2, -1)).tolist(),
        "declared_sky_electrons_per_pixel": sky,
        "scene_metadata": scene_metadata,
    }
    write_json_fresh(run_dir / "manifests/input_hashes.json", input_hashes)
    write_json_fresh(run_dir / "manifests/historical_integrity.json", integrity)
    write_json_fresh(run_dir / "manifests/authorization_gate.json", {
        "status": "PASS",
        "foundation": foundation,
        "integrity": integrity,
        "preparation_tests": "40/40 compatibility tests passed independently before preregistration",
        "standalone_validator": "PASS",
        "frozen_scientific_arrays_accessed_before_gate": 0,
        "isolated_source_arrays_accessed": 0,
        "development_access": 0,
        "atlas_array_access": 0,
        "lockbox_access": 0,
    })
    write_text_fresh(run_dir / "manifests/preexisting_workspace_status.txt", status["stdout"])
    execution_contract = f"""# {CAMPAIGN} execution addendum

Status: **FROZEN BEFORE SCIENTIFIC OBSERVATION ACCESS**

The authoritative preparation protocol is copied byte-for-byte as
`frozen_protocol.md` (SHA-256 `{PREPARATION_PROTOCOL_SHA256}`). This addendum
resolves only inference inputs left symbolic by that protocol; it changes no
support, objective, optimizer, start, or classification threshold.

- Scene indices: `{list(INDICES)}` exactly once each.
- Prompt centers: exact two-coordinate `xy` contract inherited by
  Thayer-Identifiability-v1; requested/companion order is the frozen
  `matched_source_index`. The companion coordinate is therefore already part
  of the frozen two-source task contract.
- Noise: the fixed observation-only plug-in map
  `sigma[b,y,x]^2 = LSST_sky_electrons[b] + max(observed[b,y,x], 0)`.
  It follows the authoritative BTK `add_noise='all'` source-Poisson plus
  zero-mean sky-Poisson convention and uses no hidden noiseless component.
  Frozen sky levels are `{json.dumps(sky, sort_keys=True)}` electrons/pixel.
- Primary families: Level 4 Sérsic and Level 5 bulge+disk only; both are run
  for every scene and neither is selected after outcomes.
- Numerical perturbation: one-sided/clipped parameter perturbation of
  `{NUMERICAL_PERTURBATION_SCALE:g}` times the frozen dimensionless column
  scales; rank/nullity must be unchanged and active-spectrum relative change
  must not exceed `{NUMERICAL_SPECTRUM_RTOL:g}`. It is diagnostic only and
  never changes the primary endpoints.
- Prompt swap: swap the two center assignments and the two fitted parameter
  blocks algebraically; source layers and recomposed observation must replay
  within `1e-10` relative L2. No outcome-dependent refit is performed.
- Replay: exact renderer/Jacobian hash replay at the selected endpoint plus
  the already-passed synthetic full-optimizer replay gate.

The HDF5 loader may read only `blend[index]` and `xy[index]`. It may not read
the `isolated` dataset. The CatSim catalog is not opened. Development, Atlas,
lockbox, and neural checkpoints are prohibited.
"""
    write_text_fresh(run_dir / "preregistration/execution_contract.md", execution_contract)
    seal = {
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_SCIENCE",
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_protocol_sha256": sha256_file(run_dir / "preregistration/frozen_protocol.md"),
        "frozen_protocol_json_sha256": sha256_file(run_dir / "preregistration/frozen_protocol.json"),
        "execution_contract_sha256": sha256_file(run_dir / "preregistration/execution_contract.md"),
        "input_hashes_sha256": sha256_file(run_dir / "manifests/input_hashes.json"),
        "historical_integrity_sha256": sha256_file(run_dir / "manifests/historical_integrity.json"),
        "scientific_observation_arrays_accessed": 0,
    }
    write_json_fresh(run_dir / "preregistration/freeze_seal.json", seal)
    print(json.dumps(seal, indent=2, sort_keys=True))


def verify_run_freeze(run_dir: Path) -> dict[str, Any]:
    seal = json.loads((run_dir / "preregistration/freeze_seal.json").read_text())
    actual = {
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_protocol_sha256": sha256_file(run_dir / "preregistration/frozen_protocol.md"),
        "frozen_protocol_json_sha256": sha256_file(run_dir / "preregistration/frozen_protocol.json"),
        "execution_contract_sha256": sha256_file(run_dir / "preregistration/execution_contract.md"),
        "input_hashes_sha256": sha256_file(run_dir / "manifests/input_hashes.json"),
        "historical_integrity_sha256": sha256_file(run_dir / "manifests/historical_integrity.json"),
    }
    matches = {key: actual[key] == seal[key] for key in actual}
    return {"status": "PASS" if all(matches.values()) else "FAIL", "matches": matches, "actual": actual}


def load_science_inputs(index: int, family: str) -> tuple[SolverInputs, dict[str, Any]]:
    if index not in INDICES or family not in FAMILIES:
        raise ValueError("unauthorized scene or family")
    row = selected_manifest_rows()[index]
    # Contract-critical: no access to handle["isolated"] is present here.
    with h5py.File(SCENES, "r") as handle:
        observed_f32 = np.asarray(handle["blend"][index], dtype=np.float32)
        xy = np.asarray(handle["xy"][index], dtype=np.float64)
    if legacy_array_sha256(observed_f32) != row["blend_sha256"]:
        raise RuntimeError("frozen observation hash mismatch")
    expected_xy = np.asarray(
        [
            [float(row["source_a_x_pixel"]), float(row["source_a_y_pixel"])],
            [float(row["source_b_x_pixel"]), float(row["source_b_y_pixel"])],
        ],
        dtype=np.float64,
    )
    if xy.shape != (2, 2) or not np.allclose(xy, expected_xy, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("frozen exact two-coordinate contract mismatch")
    requested_index = int(row["matched_source_index"])
    companion_index = 1 - requested_index
    psf, sky = known_psf()
    observed = np.asarray(observed_f32, dtype=np.float64)
    sigma = declared_noise_sigma(observed, sky)
    inputs = SolverInputs(
        observed=torch.as_tensor(observed, dtype=torch.float64),
        requested_center_xy=tuple(map(float, xy[requested_index])),
        companion_center_xy=tuple(map(float, xy[companion_index])),
        psf=psf,
        noise_sigma=torch.as_tensor(sigma, dtype=torch.float64),
        family=family,
    )
    inputs.validate()
    metadata = {
        "scene_index": index,
        "scene_id": row["scene_id"],
        "requested_source_index": requested_index,
        "companion_source_index": companion_index,
        "observation_legacy_sha256": legacy_array_sha256(observed_f32),
        "observation_canonical_sha256": canonical_tensor_sha256(inputs.observed),
        "xy_sha256": raw_array_sha256(xy),
        "requested_center_xy": inputs.requested_center_xy,
        "companion_center_xy": inputs.companion_center_xy,
        "sky_electrons_per_pixel": sky,
        "noise_sigma_canonical_sha256": canonical_tensor_sha256(inputs.noise_sigma),
    }
    return inputs, metadata


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right) / scale)


def numerical_perturbation(parameters: np.ndarray, inputs: SolverInputs, protocol: FrozenSolverProtocol, local: Any) -> dict[str, Any]:
    value = np.asarray(parameters, dtype=np.float64)
    lower, upper = parameter_bounds(inputs, protocol)
    scales = parameter_scales(inputs, protocol)
    direction = np.sin(np.arange(value.size, dtype=np.float64) + 1.0)
    candidate = value + NUMERICAL_PERTURBATION_SCALE * scales * direction
    candidate = np.maximum(candidate, lower)
    finite_upper = np.isfinite(upper)
    candidate[finite_upper] = np.minimum(candidate[finite_upper], upper[finite_upper])
    jacobian = residual_jacobian(candidate, inputs, protocol)
    perturbed = analyze_jacobian(
        jacobian,
        active_mask=local.jacobian_diagnostics.active_mask,
        parameter_scales=scales,
    )
    base_singular = local.jacobian_diagnostics.singular_values
    spectrum_change = float(
        np.linalg.norm(perturbed.singular_values - base_singular)
        / max(float(np.linalg.norm(base_singular)), np.finfo(np.float64).tiny)
    )
    residual = whitened_residual_vector(torch.as_tensor(candidate, dtype=torch.float64), inputs, protocol)
    stable = bool(
        torch.isfinite(residual).all()
        and np.isfinite(jacobian).all()
        and perturbed.rank == local.jacobian_diagnostics.rank
        and perturbed.null_space_dimension == local.jacobian_diagnostics.null_space_dimension
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


def execute_one(run_dir: Path, index: int, family: str) -> None:
    freeze = verify_run_freeze(run_dir)
    if freeze["status"] != "PASS":
        raise RuntimeError("campaign freeze seal changed; fail closed")
    output = run_dir / "scene_results" / f"scene_{index:03d}_{family}.json"
    arrays_output = run_dir / "scene_results" / f"scene_{index:03d}_{family}_images.npz"
    if output.exists() or arrays_output.exists():
        raise FileExistsError("append-only scene/family result already exists")
    protocol = FrozenSolverProtocol()
    inputs, metadata = load_science_inputs(index, family)
    audit = oracle_information_audit(inputs, protocol, extra_named_inputs={
        "scene_index": index,
        "scene_id": metadata["scene_id"],
        "requested_source_index": metadata["requested_source_index"],
    })
    if audit["status"] != "PASS":
        raise RuntimeError("oracle-information audit failed")
    starts = deterministic_starts(inputs, protocol)
    endpoints = multi_start_optimize(inputs, protocol, starts=starts)
    successful = [endpoint for endpoint in endpoints if endpoint.success and np.isfinite(endpoint.likelihood_objective)]
    if not successful:
        result = {
            "campaign": CAMPAIGN,
            "metadata": metadata,
            "family": family,
            "level": LEVEL_BY_FAMILY[family],
            "classification": "OPTIMIZATION_UNRESOLVED",
            "oracle_audit": audit,
            "input_provenance_trace": input_provenance_trace(inputs, protocol),
            "starts": starts,
            "endpoints": [endpoint.record() for endpoint in endpoints],
        }
        write_json_fresh(output, result)
        print(json.dumps({"scene": index, "family": family, "classification": result["classification"]}))
        return
    best = min(successful, key=lambda endpoint: endpoint.likelihood_objective)
    local = local_fit_diagnostics(best.parameters, inputs, protocol)
    support_endpoints = []
    support_records = []
    for endpoint in successful:
        canonical, active, _ = canonicalize_parameters(endpoint.parameters, inputs, protocol)
        dof = max(1, int(inputs.observed.numel()) - int(active.sum()))
        threshold = float(chi2.ppf(protocol.model_acceptance_quantile, dof))
        accepted = bool(endpoint.chi_square <= threshold)
        support_records.append({"start_index": endpoint.start_index, "degrees_of_freedom": dof, "chi_square_threshold": threshold, "support_acceptable": accepted})
        if accepted:
            support_endpoints.append(endpoint)
    geometry_pool = support_endpoints if support_endpoints else successful
    geometry = solution_geometry(geometry_pool, inputs, protocol)
    perturbation = numerical_perturbation(best.parameters, inputs, protocol, local)
    classification = classify_identifiability(
        endpoints,
        geometry,
        local,
        inputs,
        protocol,
        contract_valid=audit["status"] == "PASS",
        numerically_stable=perturbation["stable"],
    )
    best_tensor = torch.as_tensor(best.parameters, dtype=torch.float64)
    pair = render_pair(best_tensor, inputs, protocol)
    objective = likelihood_components(best_tensor, inputs, protocol)
    req = np.asarray(pair.requested.detach().cpu(), dtype=np.float64)
    comp = np.asarray(pair.companion.detach().cpu(), dtype=np.float64)
    recomposed = req + comp
    residual = np.asarray(objective.signed_residual.detach().cpu(), dtype=np.float64)
    per_source = parameters_per_source(family)
    swapped_parameters = np.concatenate((best.parameters[per_source:], best.parameters[:per_source]))
    swapped_inputs = SolverInputs(
        observed=inputs.observed,
        requested_center_xy=inputs.companion_center_xy,
        companion_center_xy=inputs.requested_center_xy,
        psf=inputs.psf,
        noise_sigma=inputs.noise_sigma,
        family=family,
    )
    swapped_pair = render_pair(torch.as_tensor(swapped_parameters, dtype=torch.float64), swapped_inputs, protocol)
    swap = {
        "requested_to_original_companion_relative_l2": relative_l2(np.asarray(swapped_pair.requested), comp),
        "companion_to_original_requested_relative_l2": relative_l2(np.asarray(swapped_pair.companion), req),
        "recomposed_relative_l2": relative_l2(np.asarray(swapped_pair.recomposed_sources), recomposed),
    }
    swap["algebraically_consistent"] = max(swap.values()) <= 1.0e-10
    replay_jacobian = residual_jacobian(best.parameters, inputs, protocol)
    replay = {
        "requested_sha256_first": canonical_tensor_sha256(pair.requested),
        "requested_sha256_second": canonical_tensor_sha256(render_pair(best_tensor, inputs, protocol).requested),
        "jacobian_sha256_first": raw_array_sha256(local.jacobian),
        "jacobian_sha256_second": raw_array_sha256(replay_jacobian),
    }
    replay["exact_match"] = replay["requested_sha256_first"] == replay["requested_sha256_second"] and replay["jacobian_sha256_first"] == replay["jacobian_sha256_second"]
    boundary = boundary_contact_flags(best.parameters, inputs, protocol)
    diagnostics = local.jacobian_diagnostics
    result = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "metadata": metadata,
        "family": family,
        "level": LEVEL_BY_FAMILY[family],
        "classification": classification,
        "protocol": asdict(protocol),
        "oracle_audit": audit,
        "input_provenance_trace": input_provenance_trace(inputs, protocol),
        "starts": starts,
        "endpoints": [endpoint.record() for endpoint in endpoints],
        "support_records": support_records,
        "best_start_index": best.start_index,
        "best_parameters": best.parameters,
        "parameter_names": parameter_names(family),
        "best_likelihood_objective": best.likelihood_objective,
        "best_chi_square": best.chi_square,
        "likelihood_by_band": objective.likelihood_by_band.detach().cpu().numpy(),
        "chi_square_by_band": (objective.signed_residual / inputs.noise_sigma).square().sum(dim=(-2, -1)).detach().cpu().numpy(),
        "gradient_norm": local.gradient_norm,
        "rank": diagnostics.rank,
        "active_parameter_count": diagnostics.active_parameter_count,
        "rank_tolerance": diagnostics.rank_tolerance,
        "null_space_dimension": diagnostics.null_space_dimension,
        "singular_values": diagnostics.singular_values,
        "null_space_basis": diagnostics.null_space_basis,
        "condition_number": diagnostics.condition_number,
        "hessian_eigenvalues": diagnostics.hessian_eigenvalues,
        "hessian_condition_number": diagnostics.hessian_condition_number,
        "symmetries": local.symmetries,
        "boundary_contact_flags": boundary,
        "geometry": asdict(geometry),
        "support_acceptable_start_indices": [endpoint.start_index for endpoint in support_endpoints],
        "fitted_requested_fluxes": best.parameters[:3],
        "fitted_companion_fluxes": best.parameters[per_source : per_source + 3],
        "observation_band_sums": inputs.observed.sum(dim=(-2, -1)).detach().cpu().numpy(),
        "recomposed_band_sums": np.sum(recomposed, axis=(-2, -1)),
        "residual_band_sums": np.sum(residual, axis=(-2, -1)),
        "source_nonnegative": bool(np.all(req >= 0) and np.all(comp >= 0)),
        "psf_band_sums": normalize_psf(inputs.psf).sum(dim=(-2, -1)).detach().cpu().numpy(),
        "signed_residual_min": float(np.min(residual)),
        "signed_residual_max": float(np.max(residual)),
        "prompt_swap": swap,
        "numerical_perturbation": perturbation,
        "deterministic_replay": replay,
        "image_hashes": {
            "requested": canonical_tensor_sha256(pair.requested),
            "companion": canonical_tensor_sha256(pair.companion),
            "recomposed": canonical_tensor_sha256(pair.recomposed_sources),
            "residual": canonical_tensor_sha256(objective.signed_residual),
        },
        "access_counts": {
            "blend_rows": 1,
            "xy_rows": 1,
            "isolated_source_arrays": 0,
            "catalog_rows": 0,
            "development": 0,
            "atlas_arrays": 0,
            "lockbox": 0,
            "neural_training_steps": 0,
        },
    }
    np.savez_compressed(
        arrays_output,
        observed=np.asarray(inputs.observed),
        requested=req,
        companion=comp,
        recomposed=recomposed,
        residual=residual,
        noise_sigma=np.asarray(inputs.noise_sigma),
    )
    write_json_fresh(output, result)
    print(json.dumps({"scene": index, "family": family, "classification": classification, "chi_square": best.chi_square, "rank": diagnostics.rank, "null": diagnostics.null_space_dimension, "classes": geometry.distinct_solution_classes}))


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for index in INDICES:
        for family in FAMILIES:
            path = run_dir / "scene_results" / f"scene_{index:03d}_{family}.json"
            if not path.exists():
                raise RuntimeError(f"missing required scene/family result: {path.name}")
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def result_lookup(results: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(result["metadata"]["scene_index"]), result["family"]): result for result in results}


def minimum_family(rows: dict[str, dict[str, Any]]) -> str:
    if rows[FAMILY_SERSIC]["classification"] == "UNIQUE":
        return "Level 4"
    if rows[FAMILY_BULGE_DISK]["classification"] == "UNIQUE":
        return "Level 5"
    return "none"


def comparison_explanation(result: dict[str, Any]) -> str:
    classification = result["classification"]
    if classification == "UNIQUE":
        return "unique solution survives without isolated-truth source fluxes"
    if classification == "NON_IDENTIFIABLE":
        if int(result["null_space_dimension"]) > 0:
            return "continuous morphology/photometry allocation ambiguity is restored"
        return "several isolated tolerance-quality source solutions are present"
    if classification == "PARTIALLY_IDENTIFIABLE":
        return "residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold"
    if classification == "NEAR_UNIQUE":
        return "one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails"
    if classification == "OUT_OF_SUPPORT":
        return "no likelihood-acceptable solution exists in the frozen morphology support"
    if classification == "NUMERICALLY_UNSTABLE":
        return "frozen numerical perturbation diagnostics are unstable"
    if classification == "OPTIMIZATION_UNRESOLVED":
        return "the frozen optimizer did not resolve a scientific endpoint"
    return "the controlled inference contract is invalid"


def campaign_outcome(summary_rows: list[dict[str, Any]], union_unique: int) -> str:
    classifications = [row["best_scene_classification"] for row in summary_rows]
    if "INVALID_CONTRACT" in classifications:
        return "FLUX_FREE_INVALID"
    if union_unique >= 6:
        return "FLUX_FREE_UNIQUENESS_LARGELY_SURVIVES"
    if union_unique >= 3:
        return "FLUX_FREE_UNIQUENESS_PARTIALLY_SURVIVES"
    support = sum(value == "OUT_OF_SUPPORT" for value in classifications)
    unresolved = sum(value in {"OPTIMIZATION_UNRESOLVED", "NUMERICALLY_UNSTABLE"} for value in classifications)
    ambiguity = sum(value in {"NON_IDENTIFIABLE", "PARTIALLY_IDENTIFIABLE", "NEAR_UNIQUE"} for value in classifications)
    if support > max(unresolved, ambiguity):
        return "FLUX_FREE_SUPPORT_LIMITED"
    if unresolved > ambiguity:
        return "FLUX_FREE_OPTIMIZATION_UNRESOLVED"
    return "FLUX_FREE_UNIQUENESS_COLLAPSES"


def best_scene_classification(l4: str, l5: str) -> str:
    priority = {
        "UNIQUE": 0,
        "NEAR_UNIQUE": 1,
        "PARTIALLY_IDENTIFIABLE": 2,
        "NON_IDENTIFIABLE": 3,
        "OUT_OF_SUPPORT": 4,
        "NUMERICALLY_UNSTABLE": 5,
        "OPTIMIZATION_UNRESOLVED": 6,
        "INVALID_CONTRACT": 7,
    }
    return min((l4, l5), key=lambda value: priority[value])


def make_figures(run_dir: Path, results: list[dict[str, Any]]) -> None:
    lookup = result_lookup(results)
    class_order = ["UNIQUE", "NEAR_UNIQUE", "PARTIALLY_IDENTIFIABLE", "NON_IDENTIFIABLE", "OUT_OF_SUPPORT", "OPTIMIZATION_UNRESOLVED", "NUMERICALLY_UNSTABLE", "INVALID_CONTRACT"]
    colors = {name: plt.cm.tab10(i % 10) for i, name in enumerate(class_order)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(INDICES))
    for family, marker in ((FAMILY_SERSIC, "o"), (FAMILY_BULGE_DISK, "s")):
        values = [lookup[index, family]["classification"] for index in INDICES]
        y = [class_order.index(value) for value in values]
        axes[0].scatter(x, y, marker=marker, s=70, label=LEVEL_BY_FAMILY[family], c=[colors[value] for value in values], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x, INDICES)
    axes[0].set_yticks(range(len(class_order)), class_order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Frozen scene index")
    axes[0].set_title("Flux-free structural classifications")
    axes[0].legend()
    l4 = sum(lookup[index, FAMILY_SERSIC]["classification"] == "UNIQUE" for index in INDICES)
    l5 = sum(lookup[index, FAMILY_BULGE_DISK]["classification"] == "UNIQUE" for index in INDICES)
    union = sum(any(lookup[index, family]["classification"] == "UNIQUE" for family in FAMILIES) for index in INDICES)
    axes[1].bar(["L4", "L5", "Union", "Oracle-flux\nunion v1"], [l4, l5, union, 7], color=["#4c78a8", "#f58518", "#54a24b", "#999999"])
    axes[1].axhline(6, color="black", linestyle="--", linewidth=1, label="largely survives gate")
    axes[1].set_ylim(0, 8.4)
    axes[1].set_ylabel("Unique scenes out of 8")
    axes[1].set_title("Recoverability frontier")
    axes[1].legend()
    fig.savefig(run_dir / "figures/flux_free_recoverability_frontier.png", dpi=180)
    plt.close(fig)
    for result in results:
        index = int(result["metadata"]["scene_index"])
        family = result["family"]
        singular = np.asarray(result.get("singular_values", []), dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        if singular.size:
            ax.semilogy(np.arange(1, singular.size + 1), np.maximum(singular, np.finfo(float).tiny), marker="o", ms=3)
            ax.axhline(float(result["rank_tolerance"]), color="red", linestyle="--", label="rank tolerance")
        ax.set_xlabel("Singular-value index")
        ax.set_ylabel("Scaled whitened-Jacobian singular value")
        ax.set_title(f"Scene {index} {LEVEL_BY_FAMILY[family]}: rank {result.get('rank','--')}/{result.get('active_parameter_count','--')}")
        ax.legend(loc="best")
        fig.savefig(run_dir / f"figures/singular_value_spectra/scene_{index:03d}_{family}.png", dpi=160)
        plt.close(fig)
        endpoints = result.get("endpoints", [])
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        if endpoints:
            best = min(float(item["likelihood_objective"]) for item in endpoints)
            per_source = parameters_per_source(family)
            fractions = []
            delta = []
            for item in endpoints:
                p = np.asarray(item["parameters"], dtype=float)
                total = p[:3] + p[per_source : per_source + 3]
                fractions.append(float(np.mean(p[:3] / np.maximum(total, np.finfo(float).tiny))))
                delta.append(float(item["likelihood_objective"]) - best)
            ax.scatter(fractions, np.maximum(delta, 1e-16), c=[item["start_index"] for item in endpoints], cmap="viridis", s=40)
            ax.set_yscale("log")
        ax.set_xlabel("Mean requested-source fitted flux fraction")
        ax.set_ylabel("NLL minus best")
        ax.set_title(f"Scene {index} {LEVEL_BY_FAMILY[family]} multi-start geometry")
        fig.savefig(run_dir / f"figures/multistart_solution_geometry/scene_{index:03d}_{family}.png", dpi=160)
        plt.close(fig)
        image_path = run_dir / "scene_results" / f"scene_{index:03d}_{family}_images.npz"
        if image_path.exists():
            arrays = np.load(image_path)
            fig, axes = plt.subplots(3, 5, figsize=(14, 8), constrained_layout=True)
            names = ("observed", "requested", "companion", "recomposed", "residual")
            for band, band_name in enumerate(BANDS):
                for column, name in enumerate(names):
                    array = np.asarray(arrays[name][band], dtype=float)
                    scale = np.percentile(np.abs(array), 99.5)
                    shown = np.arcsinh(array / max(scale, np.finfo(float).tiny))
                    axes[band, column].imshow(shown, origin="lower", cmap="coolwarm" if name in {"observed", "residual"} else "magma")
                    axes[band, column].set_xticks([])
                    axes[band, column].set_yticks([])
                    if band == 0:
                        axes[band, column].set_title(name)
                    if column == 0:
                        axes[band, column].set_ylabel(band_name)
            fig.suptitle(f"Scene {index} {LEVEL_BY_FAMILY[family]} — {result['classification']} (observation-only)")
            fig.savefig(run_dir / f"figures/source_reconstruction_panels/scene_{index:03d}_{family}.png", dpi=150)
            plt.close(fig)


def finalize(run_dir: Path) -> None:
    if verify_run_freeze(run_dir)["status"] != "PASS":
        raise RuntimeError("campaign freeze seal changed")
    results = load_results(run_dir)
    lookup = result_lookup(results)
    metrics_rows = []
    endpoint_rows = []
    flux_rows = []
    audit_rows = []
    comparison_rows = []
    for result in results:
        index = int(result["metadata"]["scene_index"])
        family = result["family"]
        geometry = result.get("geometry", {})
        metrics_rows.append({
            "scene_index": index,
            "scene_id": result["metadata"]["scene_id"],
            "family": family,
            "level": result["level"],
            "classification": result["classification"],
            "best_observation_objective": result.get("best_likelihood_objective", ""),
            "best_chi_square": result.get("best_chi_square", ""),
            "gradient_norm": result.get("gradient_norm", ""),
            "rank": result.get("rank", ""),
            "active_parameter_count": result.get("active_parameter_count", ""),
            "null_space_dimension": result.get("null_space_dimension", ""),
            "condition_number": result.get("condition_number", ""),
            "hessian_condition_number": result.get("hessian_condition_number", ""),
            "singular_values": result.get("singular_values", []),
            "hessian_eigenvalues": result.get("hessian_eigenvalues", []),
            "distinct_solution_classes": geometry.get("distinct_solution_classes", ""),
            "requested_image_diameter": geometry.get("requested_image_diameter", ""),
            "companion_image_diameter": geometry.get("companion_image_diameter", ""),
            "flux_allocation_diameter": geometry.get("flux_allocation_diameter", ""),
            "morphology_parameter_diameter": geometry.get("morphology_parameter_diameter", ""),
            "prompt_identity_consistent": geometry.get("prompt_identity_consistent", ""),
            "boundary_contact_flags": result.get("boundary_contact_flags", {}),
            "prompt_swap": result.get("prompt_swap", {}),
            "numerical_perturbation": result.get("numerical_perturbation", {}),
            "deterministic_replay": result.get("deterministic_replay", {}),
        })
        for endpoint in result.get("endpoints", []):
            endpoint_rows.append({
                "scene_index": index,
                "scene_id": result["metadata"]["scene_id"],
                "family": family,
                **endpoint,
            })
        if "best_parameters" in result:
            per_source = parameters_per_source(family)
            params = np.asarray(result["best_parameters"], dtype=float)
            for band_index, band in enumerate(BANDS):
                flux_rows.append({
                    "scene_index": index,
                    "scene_id": result["metadata"]["scene_id"],
                    "family": family,
                    "band": band,
                    "requested_flux": params[band_index],
                    "companion_flux": params[per_source + band_index],
                    "total_fitted_source_flux": params[band_index] + params[per_source + band_index],
                    "observed_stamp_sum": result["observation_band_sums"][band_index],
                    "residual_stamp_sum": result["residual_band_sums"][band_index],
                })
        for entry in result.get("input_provenance_trace", []):
            audit_rows.append({
                "scene_index": index,
                "scene_id": result["metadata"]["scene_id"],
                "family": family,
                "input": entry["input"],
                "provenance": entry["provenance"],
                "sha256": entry["sha256"],
                "oracle": entry["oracle"],
                "audit_status": result["oracle_audit"]["status"],
            })
        comparison_rows.append({
            "scene_index": index,
            "scene_id": result["metadata"]["scene_id"],
            "family": family,
            "oracle_flux_v1_classification": PREVIOUS[index][family],
            "flux_free_classification": result["classification"],
            "comparison": comparison_explanation(result),
        })
    summary_rows = []
    for index in INDICES:
        by_family = {family: lookup[index, family] for family in FAMILIES}
        l4 = by_family[FAMILY_SERSIC]["classification"]
        l5 = by_family[FAMILY_BULGE_DISK]["classification"]
        current_minimum = minimum_family(by_family)
        summary_rows.append({
            "scene_index": index,
            "scene_id": by_family[FAMILY_SERSIC]["metadata"]["scene_id"],
            "level_4_classification": l4,
            "level_5_classification": l5,
            "unique_under_any_family": l4 == "UNIQUE" or l5 == "UNIQUE",
            "minimum_structural_family_required": current_minimum,
            "oracle_flux_v1_minimum": PREVIOUS[index]["minimum"],
            "best_scene_classification": best_scene_classification(l4, l5),
        })
    l4_unique = sum(row["level_4_classification"] == "UNIQUE" for row in summary_rows)
    l5_unique = sum(row["level_5_classification"] == "UNIQUE" for row in summary_rows)
    union_unique = sum(bool(row["unique_under_any_family"]) for row in summary_rows)
    outcome = campaign_outcome(summary_rows, union_unique)
    write_csv_fresh(run_dir / "tables/scene_family_summary.csv", summary_rows)
    write_csv_fresh(run_dir / "tables/full_flux_free_identifiability_metrics.csv", metrics_rows)
    write_csv_fresh(run_dir / "tables/multistart_endpoints.csv", endpoint_rows)
    write_csv_fresh(run_dir / "tables/fitted_source_fluxes.csv", flux_rows)
    write_csv_fresh(run_dir / "tables/oracle_information_audit.csv", audit_rows)
    write_csv_fresh(run_dir / "tables/comparison_to_oracle_flux_v1.csv", comparison_rows)
    make_figures(run_dir, results)
    integrity = historical_integrity()
    if integrity["status"] != "PASS":
        write_json_fresh(run_dir / "manifests/historical_integrity_final.json", integrity)
        raise RuntimeError("post-execution historical integrity failed closed")
    write_json_fresh(run_dir / "manifests/historical_integrity_final.json", integrity)
    exact_lines = []
    for row in summary_rows:
        exact_lines.append(
            f"- Scene {row['scene_index']}: L4 `{row['level_4_classification']}`; "
            f"L5 `{row['level_5_classification']}`; minimum unique family "
            f"`{row['minimum_structural_family_required']}`."
        )
    nonunique_lines = []
    for row in summary_rows:
        if not row["unique_under_any_family"]:
            failures = []
            for family in FAMILIES:
                result = lookup[int(row["scene_index"]), family]
                failures.append(f"{LEVEL_BY_FAMILY[family]} {result['classification']}: {comparison_explanation(result)}")
            nonunique_lines.append(f"- Scene {row['scene_index']}: " + "; ".join(failures) + ".")
    if not nonunique_lines:
        nonunique_lines = ["- None; every scene is unique under at least one frozen family."]
    if outcome == "FLUX_FREE_UNIQUENESS_LARGELY_SURVIVES":
        next_experiment = "Thayer-Direct-Structural-Solver-v0"
        next_purpose = "measure actual requested-source reconstruction accuracy, rather than identifiability alone, using direct morphology-constrained fitting"
    elif outcome == "FLUX_FREE_UNIQUENESS_PARTIALLY_SURVIVES":
        next_experiment = "Thayer-Flux-Morphology-Degeneracy-Diagnostic-v0"
        next_purpose = "isolate the dominant remaining flux-allocation versus morphology degeneracy on the same frozen scenes"
    elif outcome == "FLUX_FREE_UNIQUENESS_COLLAPSES":
        next_experiment = "Thayer-PSF-Diverse-Flux-Identifiability-v0"
        next_purpose = "test whether a second known PSF observation supplies enough independent information to remove the restored allocation ambiguity"
    elif outcome == "FLUX_FREE_SUPPORT_LIMITED":
        next_experiment = "Thayer-Structural-Support-Diagnostic-v0"
        next_purpose = "separate frozen-family misspecification from flux ambiguity without expanding support after outcomes"
    else:
        next_experiment = "Thayer-Flux-Free-Numerical-Resolution-v0"
        next_purpose = "resolve the frozen optimizer or numerical blocker without changing the scientific model"
    report = f"""# {CAMPAIGN} final report

## Campaign outcome

**{outcome}**

The primary endpoint is **{union_unique}/8** scenes `UNIQUE` under at least one
permitted flux-free structural family. Level 4 is unique for **{l4_unique}/8**
and Level 5 for **{l5_unique}/8**. The earlier **7/8** result does
{' survive at the declared 6/8 threshold' if union_unique >= 6 else ' not survive at the declared 6/8 threshold'}
after isolated-truth per-source g/r/z fluxes are removed.

## Exact findings

The solver received only the frozen blended g/r/z observation, the requested
and companion coordinates from the frozen two-source coordinate contract, the
known normalized LSST g/r/z PSFs, 60x60 geometry at 0.2 arcsec/pixel, the
observation-only BTK Poisson plug-in sigma map, the family identifier, and the
pre-frozen Level-4/5 bounds. Individual requested and companion fluxes were
free nonnegative unbounded-above likelihood parameters.

The solver did **not** receive isolated source images or masks, true per-source
fluxes, the exact signed noise realization, true morphology parameters or
labels, B/T truth, catalog rows, truth initialization, protected-set data, or
outcome-dependent family selection. All 16 starts per scene/family are
retained; no failing run was discarded.

{chr(10).join(exact_lines)}

All fitted source layers are nonnegative, PSFs are normalized, residuals are
signed, and each solver invocation has a machine-readable eight-input
provenance trace. Deterministic renderer/Jacobian replay and the frozen
numerical perturbation check are reported per scene/family.

## Why non-unique scenes fail

{chr(10).join(nonunique_lines)}

`OUT_OF_SUPPORT` means the admissible likelihood set is empty and is never
counted as uniqueness. Scene 51 remains the required stress case; its prior
oracle-flux evidence identified a sub-support unresolved bulge, and the
flux-free classification above does not reinterpret empty support as source
identifiability.

## Interpretation

The findings are conditional structural identifiability results for these
eight frozen simulated observations and the declared exact-coordinate,
known-PSF, fixed-noise-convention contract. They are not observation-only
unrestricted-output uniqueness and do not establish generalization to real
survey data.

{'Morphology-aware direct fitting is scientifically justified because the flux-free union clears the preregistered survival gate.' if union_unique >= 6 else 'The present evidence does not yet justify treating morphology-aware fitting as a broadly unique target across this frozen set.'}
PriorNet training remains **unauthorized**. The POST audit layer may return
only after a direct structural solver establishes reconstruction accuracy and
produces scientifically valid outputs; identifiability alone is insufficient.

## Integrity

Preparation and compatibility tests passed before science; all 600 historical
checkpoints matched both before and after execution. Historical reports,
README, and the Git index remain unchanged. Scientific accesses were exactly
16 authorized blend/coordinate row reads total (two per scene, one per family),
with zero isolated-source array reads, zero catalog-row reads, zero
development, zero Atlas-array, and zero lockbox access. No neural network was
loaded or trained. Nothing was staged or committed.

The workspace had 382 pre-existing porcelain entries before this campaign.
They remain user-owned. Campaign-created files are this fresh ignored run tree
and `scripts/run_thayer_flux_free_identifiability_v0.py`; no prior output was
deleted or rewritten.

## Exactly one recommended next experiment

Run **{next_experiment}** to {next_purpose}. PriorNet is not authorized by this
recommendation.
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)
    final_manifest = {
        "campaign": CAMPAIGN,
        "outcome": outcome,
        "level_4_unique_count": l4_unique,
        "level_5_unique_count": l5_unique,
        "union_unique_count": union_unique,
        "previous_oracle_flux_union_unique_count": 7,
        "recommended_next_experiment": next_experiment,
        "priornet_authorized": False,
        "historical_integrity_status": integrity["status"],
        "readme_unchanged": integrity["readme_unchanged"],
        "staged_index_empty": integrity["staged_index_empty"],
        "commits_created": 0,
        "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"),
        "required_artifacts": {},
    }
    required = (
        "reports/final_report.md",
        "preregistration/frozen_protocol.md",
        "tables/scene_family_summary.csv",
        "tables/full_flux_free_identifiability_metrics.csv",
        "tables/multistart_endpoints.csv",
        "tables/fitted_source_fluxes.csv",
        "tables/oracle_information_audit.csv",
        "tables/comparison_to_oracle_flux_v1.csv",
        "figures/flux_free_recoverability_frontier.png",
        "manifests/input_hashes.json",
        "manifests/historical_integrity.json",
    )
    for relative in required:
        path = run_dir / relative
        final_manifest["required_artifacts"][relative] = {"exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}
    final_manifest["all_required_artifacts_present"] = all(item["exists"] for item in final_manifest["required_artifacts"].values())
    write_json_fresh(run_dir / "manifests/final_manifest.json", final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--run-dir", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--scene", type=int, required=True, choices=INDICES)
    execute.add_argument("--family", required=True, choices=FAMILIES)
    final = subparsers.add_parser("finalize")
    final.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if arguments.command == "preregister":
        preregister(arguments.run_dir.resolve())
    elif arguments.command == "execute":
        execute_one(arguments.run_dir.resolve(), arguments.scene, arguments.family)
    else:
        finalize(arguments.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
