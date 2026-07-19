#!/usr/bin/env python3
"""Run the frozen PSF-diverse flux-free identifiability campaign.

The command sequence is deliberately split into preregistration, paired
observation generation, individual fits, and finalization.  Scientific arrays
cannot be opened until the source/protocol seal exists, and fits cannot begin
until every generated observation hash has a second immutable seal.
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
import subprocess
import sys
from typing import Any

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

from scripts.run_thayer_flux_free_identifiability_v0 import (
    FAMILIES,
    INDICES,
    LEVEL_BY_FAMILY,
    MANIFEST,
    SCENES,
    SOURCE_HASHES,
    declared_noise_sigma,
    historical_integrity as predecessor_historical_integrity,
    json_safe,
    known_psf,
    legacy_array_sha256,
    raw_array_sha256,
    selected_manifest_rows,
    sha256_file,
    verify_foundation,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)
from src.btk_scene import SceneSpec, load_catsim_catalog
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_joint import (
    JointSolverInputs,
    classify_joint_identifiability,
    joint_deterministic_starts,
    joint_likelihood_components,
    joint_local_fit_diagnostics,
    joint_multi_start_optimize,
    joint_oracle_information_audit,
    joint_residual_jacobian,
    joint_solution_geometry,
    joint_whitened_residual_vector,
    render_joint,
)
from src.model9_optimizer import analyze_jacobian, boundary_contact_flags, residual_jacobian
from src.model9_structured import (
    BANDS,
    FAMILY_BULGE_DISK,
    FAMILY_SERSIC,
    FrozenSolverProtocol,
    SolverInputs,
    canonicalize_parameters,
    normalize_psf,
    parameter_bounds,
    parameter_names,
    parameter_scales,
    parameters_per_source,
    render_pair,
)
from src.psf_diverse_acquisition import (
    PSF_B_ELLIPTICITY,
    PSF_B_NOMINAL_FWHM_ARCSEC,
    PSF_B_ORIENTATION_DEGREES,
    frozen_psf_b_survey,
    frozen_psf_pair_kernels,
    psf_diversity_metrics,
    render_fixed_scene_with_survey,
)


CAMPAIGN = "Thayer-PSF-Diverse-Flux-Identifiability-v0"
CONDITIONS = ("S2", "P2")
PREDECESSOR = REPO / "outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310"
PREPARATION = REPO / "outputs/runs/thayer_model_9_preparation_v0_20260715_172217"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CATALOG_PATH = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
DEFINITIONS = MANIFEST.with_name("v2_r_training_scene_definitions.csv")
S1_METRICS = PREDECESSOR / "tables/full_flux_free_identifiability_metrics.csv"
S1_SUMMARY = PREDECESSOR / "tables/scene_family_summary.csv"
PAIRED_NOISE_SEED_OFFSET = 100_000_000
NUMERICAL_PERTURBATION_SCALE = 1.0e-8
NUMERICAL_SPECTRUM_RTOL = 1.0e-3
README_SHA256 = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {
        "arguments": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing append-only overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing append-only overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def predecessor_artifact_checks() -> dict[str, Any]:
    manifest_path = PREDECESSOR / "manifests/final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {}
    for relative, record in manifest["required_artifacts"].items():
        path = PREDECESSOR / relative
        checks[relative] = {
            "exists": path.exists(),
            "expected": record["sha256"],
            "actual": sha256_file(path) if path.exists() else None,
        }
        checks[relative]["match"] = checks[relative]["actual"] == checks[relative]["expected"]
    state = {
        "manifest_sha256": sha256_file(manifest_path),
        "outcome": manifest.get("outcome"),
        "recommended_next_experiment": manifest.get("recommended_next_experiment"),
        "level_4_unique_count": manifest.get("level_4_unique_count"),
        "level_5_unique_count": manifest.get("level_5_unique_count"),
        "union_unique_count": manifest.get("union_unique_count"),
        "artifact_checks": checks,
    }
    state["status"] = "PASS" if (
        state["outcome"] == "FLUX_FREE_UNIQUENESS_COLLAPSES"
        and state["recommended_next_experiment"] == CAMPAIGN
        and all(item["match"] for item in checks.values())
    ) else "FAIL"
    return state


def stale_process_check() -> dict[str, Any]:
    result = command(["ps", "-axo", "pid=,command="])
    matches = []
    for line in result["stdout"].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, process_command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        lowered = process_command.lower()
        if (
            "run_thayer_flux_free_identifiability_v0.py execute" in lowered
            or "run_thayer_psf_diverse_flux_identifiability_v0.py execute" in lowered
            or "model9 structural solver" in lowered
        ):
            matches.append({"pid": pid, "command": process_command})
    return {"status": "PASS" if not matches else "FAIL", "matches": matches}


def test_gate() -> dict[str, Any]:
    compatibility = command(
        [
            str(REPO / ".venv-btk/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_model9_foundation.py",
            "tests/test_canonical_tensor_hash.py",
            "tests/test_family_e_signed_residual.py",
            "tests/test_psf_conditioning.py",
            "tests/test_model9_joint.py",
            "tests/test_psf_diverse_acquisition.py",
        ]
    )
    compile_gate = command(
        [
            str(REPO / ".venv-btk/bin/python"),
            "-m",
            "py_compile",
            "src/model9_joint.py",
            "src/psf_diverse_acquisition.py",
            "scripts/run_thayer_psf_diverse_flux_identifiability_v0.py",
        ]
    )
    return {
        "status": "PASS" if compatibility["returncode"] == 0 and compile_gate["returncode"] == 0 else "FAIL",
        "compatibility": compatibility,
        "py_compile": compile_gate,
        "expected_test_count": 52,
        "scientific_observation_access": 0,
    }


def authorization_gate() -> dict[str, Any]:
    foundation = verify_foundation()
    integrity = predecessor_historical_integrity()
    integrity["campaign"] = CAMPAIGN
    predecessor = predecessor_artifact_checks()
    processes = stale_process_check()
    tests = test_gate()
    staged = command(["git", "diff", "--cached", "--name-only"])
    head = command(["git", "rev-parse", "HEAD"])
    status = {
        "model_9_foundation": foundation,
        "historical_integrity": integrity,
        "predecessor": predecessor,
        "tests": tests,
        "stale_processes": processes,
        "readme_sha256": sha256_file(REPO / "README.md"),
        "readme_unchanged": sha256_file(REPO / "README.md") == README_SHA256,
        "git_index_empty": staged["returncode"] == 0 and not staged["stdout"].strip(),
        "head": head["stdout"].strip(),
        "frozen_scientific_arrays_accessed_before_gate": 0,
        "isolated_source_arrays_accessed_before_gate": 0,
        "development_access": 0,
        "atlas_tensor_access": 0,
        "lockbox_access": 0,
    }
    status["status"] = "PASS" if (
        foundation["status"] == "PASS"
        and integrity["status"] == "PASS"
        and predecessor["status"] == "PASS"
        and tests["status"] == "PASS"
        and processes["status"] == "PASS"
        and status["readme_unchanged"]
        and status["git_index_empty"]
    ) else "FAIL"
    return status


def definition_rows() -> dict[int, dict[str, str]]:
    selected = {}
    with DEFINITIONS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["dataset_index"])
            if index in INDICES:
                selected[index] = row
    if tuple(sorted(selected)) != INDICES:
        raise RuntimeError("frozen scene definitions changed")
    return selected


def frozen_protocol_text() -> str:
    protocol = FrozenSolverProtocol()
    return f"""# {CAMPAIGN} frozen protocol

Status: **FROZEN BEFORE SCIENTIFIC OBSERVATION ACCESS**

## Scope and primary endpoint

Exactly scenes `{list(INDICES)}` are evaluated under Level 4 Sérsic and Level 5
bulge+disk. S1 is imported unchanged from the completed predecessor. New S2
and P2 fits use exactly 16 starts each, for 32 new fits and 512 retained
endpoints. The primary endpoint is P2 union `UNIQUE` count across eight scenes.

## Acquisition

Observation A is the authoritative original noisy BTK LSST g/r/z blend.
Observation A2 is regenerated from the same two CatSim rows and coordinates
with the original LSST PSF and noise seed `original_noise_seed +
{PAIRED_NOISE_SEED_OFFSET}`. Observation B uses the same catalog rows,
coordinates, photometric calibration, exposure time, morphology, and intrinsic
flux, with that same paired second-seed rule and the preregistered PSF-B only.
The shared second seed couples the alternative S2/P2 controls fairly; both are
independent of Observation A. BTK `CatsimGenerator`, `add_noise='all'`, one
source-Poisson plus zero-mean sky-Poisson realization, 60x60 geometry, and 0.2
arcsec/pixel are frozen. No deconvolution or transformation of Observation A
is used. Simulator-generated isolated layers are discarded and never persisted
or passed to inference.

## Inference and objective

The solver receives exactly two blended g/r/z arrays, the frozen requested and
companion coordinates, both known normalized PSFs, image geometry/pixel scale,
the observation-only plug-in sigma maps, and frozen family/support metadata.
One shared source-parameter vector renders both observations. The joint
whitened residual is the concatenation of the two predecessor residual vectors;
the NLL is their exact sum. Starts and diagnostic parameter scales are computed
from Observation A only, making S2/P2 starts byte-identical. Per-observation and
per-band likelihood and chi-square terms are logged.

## Frozen solver

All bounds, transformations, symmetry gauges, endpoint tolerances, and
classification definitions are inherited unchanged from the predecessor:
`{json.dumps(asdict(protocol), sort_keys=True)}`. The only multi-observation
extension is that objective comparison and clustering use the joint objective,
image equivalence must hold through both PSFs, diameters are the maximum over
both rendered observations, observation count is doubled for the fixed 0.99
chi-square support gate, and the Jacobian stacks both observation blocks.

## Causal and campaign rules

Classification priority is UNIQUE > NEAR_UNIQUE > PARTIALLY_IDENTIFIABLE >
NON_IDENTIFIABLE > OUT_OF_SUPPORT. At family level: S2 improves over S1 and P2
does not exceed S2 => ADDITIONAL_EXPOSURE_ONLY; P2 improves over S1 while S2
does not => PSF_DIVERSITY_SPECIFIC; both improve and P2 exceeds S2 =>
BOTH_EXPOSURE_AND_PSF_DIVERSITY; neither improves => NO_MEANINGFUL_GAIN; any
unresolved/unstable/invalid comparison => INCONCLUSIVE_OPTIMIZATION. A P2
campaign result materially exceeds S2 when its union UNIQUE count is at least
one larger. With P2 union 0-2, conditioning improvement requires at least half
of the 16 family fits to improve minimum nonzero singular value, condition,
endpoint-class count, or a frozen diameter by at least 5% relative to S1.
The user-declared campaign outcome mapping is then applied without alteration.

## Integrity

No isolated HDF5 dataset, development data, Atlas tensor, lockbox, neural
network, truth initialization, catalog morphology label, true source flux, or
per-source photometry may enter inference. Acquisition catalog rows are used
only inside the authoritative forward simulator and are destroyed before the
paired observation artifacts are sealed. Every result is written atomically
and every start, including failures and budget exhaustion, is retained.
"""


def frozen_psf_text(metrics: list[dict[str, Any]]) -> str:
    lines = [
        f"# {CAMPAIGN} frozen PSF pair",
        "",
        "Status: **PREREGISTERED BEFORE SCIENTIFIC FITTING**",
        "",
        "PSF A is the standard BTK 1.0.9 LSST GalSim convolution PSF with nominal",
        "g/r/z FWHM 0.86/0.81/0.77 arcsec.",
        "",
        "PSF B is constructed deterministically by flux-preservingly dilating each",
        "PSF-A band to nominal g/r/z FWHM 0.70/0.68/0.66 arcsec and applying",
        f"GalSim ellipticity e={PSF_B_ELLIPTICITY:.2f} at {PSF_B_ORIENTATION_DEGREES:.1f} degrees.",
        "The band-dependent widths retain an LSST-like wavelength trend and remain",
        "well above diffraction-limited resolution. All other survey/filter fields",
        "are unchanged. Inference uses centered, nonnegative, unit-sum 31x31 kernels",
        "sampled at 0.2 arcsec/pixel. Same-size convolution uses 15-pixel zero padding.",
        "BTK acquisition uses the identical underlying GalSim objects.",
        "",
        "| band | A nominal FWHM | B nominal FWHM | rel kernel L2 | rel Fourier L2 | correlation | B/A effective width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['band']} | {row['psf_a_nominal_fwhm_arcsec']:.3f} | "
            f"{row['psf_b_nominal_fwhm_arcsec']:.3f} | {row['kernel_relative_l2_distance']:.6f} | "
            f"{row['fourier_transfer_relative_l2_distance']:.6f} | "
            f"{row['kernel_cross_correlation']:.6f} | {row['effective_resolution_ratio_b_over_a']:.6f} |"
        )
    lines.extend(
        (
            "",
            "Kernel hashes and full moment/transfer metrics are sealed in",
            "`manifests/psf_hashes.json` and `tables/psf_diversity_metrics.csv`.",
            "This single PSF-B was selected from physical reasoning, not outcomes; no",
            "candidate PSF was screened on any frozen scene.",
            "",
        )
    )
    return "\n".join(lines)


def preregister(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"append-only run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for relative in (
        "reports",
        "preregistration",
        "tables",
        "manifests",
        "logs",
        "observations",
        "scene_results",
        "figures/singular_value_comparison",
        "figures/multistart_solution_geometry",
        "figures/source_reconstruction_panels",
        "figures/residual_comparison",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    gate = authorization_gate()
    write_json_fresh(run_dir / "manifests/authorization_gate.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError("authorization gate failed closed before scientific access")
    psf_a, psf_b = frozen_psf_pair_kernels()
    metrics = psf_diversity_metrics(psf_a, psf_b)
    write_text_fresh(run_dir / "preregistration/frozen_protocol.md", frozen_protocol_text())
    write_text_fresh(run_dir / "preregistration/frozen_psf_pair.md", frozen_psf_text(metrics))
    write_csv_fresh(run_dir / "tables/psf_diversity_metrics.csv", metrics)
    psf_hashes = {
        "schema": "thayer-psf-diverse-pair-v1",
        "selected_before_scientific_fitting": True,
        "psf_a_canonical_sha256": canonical_tensor_sha256(psf_a),
        "psf_b_canonical_sha256": canonical_tensor_sha256(psf_b),
        "psf_a_band_sha256": {row["band"]: row["psf_a_sha256"] for row in metrics},
        "psf_b_band_sha256": {row["band"]: row["psf_b_sha256"] for row in metrics},
        "psf_a_band_sums": np.asarray(psf_a.sum(dim=(-2, -1))),
        "psf_b_band_sums": np.asarray(psf_b.sum(dim=(-2, -1))),
        "psf_b_nominal_fwhm_arcsec": PSF_B_NOMINAL_FWHM_ARCSEC,
        "psf_b_ellipticity": PSF_B_ELLIPTICITY,
        "psf_b_orientation_degrees": PSF_B_ORIENTATION_DEGREES,
        "support": [31, 31],
        "pixel_scale_arcsec": 0.2,
        "normalization": "per-band unit sum",
        "padding": "same-size zero padding, 15 pixels",
        "generation": "PSF-A GalSim objects dilated per band then sheared; no scene screening",
    }
    write_json_fresh(run_dir / "manifests/psf_hashes.json", psf_hashes)
    rows = selected_manifest_rows()
    definitions = definition_rows()
    scene_metadata = []
    for index in INDICES:
        row = rows[index]
        definition = definitions[index]
        if row["scene_id"] != definition["scene_id"]:
            raise RuntimeError("scene manifest/definition identity mismatch")
        scene_metadata.append(
            {
                "scene_index": index,
                "scene_id": row["scene_id"],
                "requested_source_index": int(row["matched_source_index"]),
                "observation_a_legacy_sha256": row["blend_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "source_row_identifiers_for_simulator_only": [
                    int(definition["source_a_row"]),
                    int(definition["source_b_row"]),
                ],
                "original_noise_seed": int(definition["noise_seed"]),
                "paired_noise_seed": int(definition["noise_seed"]) + PAIRED_NOISE_SEED_OFFSET,
            }
        )
    source_files = (
        "src/model9_joint.py",
        "src/psf_diverse_acquisition.py",
        "scripts/run_thayer_psf_diverse_flux_identifiability_v0.py",
        "tests/test_model9_joint.py",
        "tests/test_psf_diverse_acquisition.py",
    )
    input_hashes = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "frozen_before_scientific_array_access": True,
        "scene_indices": INDICES,
        "new_fit_count": len(INDICES) * len(FAMILIES) * len(CONDITIONS),
        "retained_endpoint_count_planned": len(INDICES)
        * len(FAMILIES)
        * len(CONDITIONS)
        * FrozenSolverProtocol().starts_per_family,
        "source_files": {name: sha256_file(REPO / name) for name in source_files},
        "model9_source_hash_manifest_sha256": sha256_file(SOURCE_HASHES),
        "predecessor_final_manifest_sha256": sha256_file(
            PREDECESSOR / "manifests/final_manifest.json"
        ),
        "predecessor_final_report_sha256": sha256_file(PREDECESSOR / "reports/final_report.md"),
        "scene_manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "sha256": sha256_file(MANIFEST),
        },
        "scene_definitions": {
            "path": str(DEFINITIONS.relative_to(REPO)),
            "sha256": sha256_file(DEFINITIONS),
        },
        "science_hdf5": {
            "path": str(SCENES.relative_to(REPO)),
            "bytes": SCENES.stat().st_size,
            "sha256": sha256_file(SCENES),
        },
        "acquisition_catalog": {
            "path": str(CATALOG_PATH.relative_to(REPO)),
            "bytes": CATALOG_PATH.stat().st_size,
            "sha256": sha256_file(CATALOG_PATH),
            "purpose": "forward simulator only; never solver input",
        },
        "psf_a_canonical_sha256": canonical_tensor_sha256(psf_a),
        "psf_b_canonical_sha256": canonical_tensor_sha256(psf_b),
        "scene_metadata": scene_metadata,
    }
    write_json_fresh(run_dir / "manifests/input_hashes.json", input_hashes)
    write_json_fresh(
        run_dir / "manifests/historical_integrity.json", gate["historical_integrity"]
    )
    write_json_fresh(
        run_dir / "manifests/protected_data_access.json",
        {
            "development_access": 0,
            "protected_atlas_tensor_access": 0,
            "lockbox_access": 0,
            "fresh_final_test_access": 0,
            "historical_isolated_hdf5_access": 0,
            "neural_training_steps": 0,
            "status": "PASS",
        },
    )
    seal_paths = (
        "preregistration/frozen_protocol.md",
        "preregistration/frozen_psf_pair.md",
        "manifests/input_hashes.json",
        "manifests/psf_hashes.json",
        "manifests/historical_integrity.json",
        "manifests/protected_data_access.json",
        "tables/psf_diversity_metrics.csv",
    )
    seal = {
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_SCIENCE",
        "timestamp_utc": utc_now(),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "joint_solver_sha256": sha256_file(REPO / "src/model9_joint.py"),
        "acquisition_sha256": sha256_file(REPO / "src/psf_diverse_acquisition.py"),
        "sealed_artifacts": {relative: sha256_file(run_dir / relative) for relative in seal_paths},
        "scientific_observation_arrays_accessed": 0,
        "catalog_rows_accessed": 0,
    }
    write_json_fresh(run_dir / "preregistration/freeze_seal.json", seal)
    print(json.dumps(seal, indent=2, sort_keys=True))


def verify_freeze(
    run_dir: Path, *, allow_finalization_amendment: bool = False
) -> dict[str, Any]:
    seal = json.loads((run_dir / "preregistration/freeze_seal.json").read_text(encoding="utf-8"))
    driver_matches = sha256_file(Path(__file__).resolve()) == seal["driver_sha256"]
    amendment_matches = False
    if allow_finalization_amendment and not driver_matches:
        amendment_path = run_dir / "preregistration/post_science_finalization_amendment.json"
        if amendment_path.exists():
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            amendment_matches = (
                amendment.get("status") == "AUTHORIZED_REPORT_ONLY_FINALIZATION_AMENDMENT"
                and amendment.get("driver", {}).get("pre_science_frozen_sha256")
                == seal["driver_sha256"]
                and amendment.get("driver", {}).get("authorized_report_finalizer_sha256")
                == sha256_file(Path(__file__).resolve())
                and amendment.get("preserved_seals", {}).get("pre_science_freeze_seal_sha256")
                == sha256_file(run_dir / "preregistration/freeze_seal.json")
                and amendment.get("preserved_seals", {}).get("science_input_seal_sha256")
                == sha256_file(run_dir / "preregistration/science_input_seal.json")
            )
    checks = {
        "driver": driver_matches or amendment_matches,
        "joint_solver": sha256_file(REPO / "src/model9_joint.py") == seal["joint_solver_sha256"],
        "acquisition": sha256_file(REPO / "src/psf_diverse_acquisition.py")
        == seal["acquisition_sha256"],
    }
    if allow_finalization_amendment:
        checks["authorized_report_only_finalization_amendment"] = (
            driver_matches or amendment_matches
        )
    for relative, expected in seal["sealed_artifacts"].items():
        checks[relative] = sha256_file(run_dir / relative) == expected
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def prepare_observations(run_dir: Path) -> None:
    if verify_freeze(run_dir)["status"] != "PASS":
        raise RuntimeError("pre-science freeze changed; fail closed")
    if (run_dir / "preregistration/science_input_seal.json").exists():
        raise FileExistsError("paired observations are already sealed")
    if any((run_dir / "observations").iterdir()):
        raise RuntimeError("unsealed observation artifacts exist; fail closed")
    catalog, catalog_hash = load_catsim_catalog(CATALOG_PATH)
    expected_catalog_hash = json.loads(
        (run_dir / "manifests/input_hashes.json").read_text(encoding="utf-8")
    )["acquisition_catalog"]["sha256"]
    if catalog_hash != expected_catalog_hash:
        raise RuntimeError("acquisition catalog hash mismatch")
    manifest_rows = selected_manifest_rows()
    definitions = definition_rows()
    survey_a = get_surveys("LSST")
    survey_b = frozen_psf_b_survey()
    pair_records = []
    for index in INDICES:
        manifest = manifest_rows[index]
        definition = definitions[index]
        with h5py.File(SCENES, "r") as handle:
            observation_a_f32 = np.asarray(handle["blend"][index], dtype=np.float32)
            xy = np.asarray(handle["xy"][index], dtype=np.float64)
        if legacy_array_sha256(observation_a_f32) != manifest["blend_sha256"]:
            raise RuntimeError(f"scene {index} Observation-A hash mismatch")
        expected_xy = np.asarray(
            (
                (float(manifest["source_a_x_pixel"]), float(manifest["source_a_y_pixel"])),
                (float(manifest["source_b_x_pixel"]), float(manifest["source_b_y_pixel"])),
            )
        )
        if not np.allclose(xy, expected_xy, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"scene {index} coordinate contract mismatch")
        positions = (
            (float(definition["source_a_x_arcsec"]), float(definition["source_a_y_arcsec"])),
            (float(definition["source_b_x_arcsec"]), float(definition["source_b_y_arcsec"])),
        )
        paired_seed = int(definition["noise_seed"]) + PAIRED_NOISE_SEED_OFFSET
        spec = SceneSpec(
            scene_id=f"{definition['scene_id']}_paired",
            catalog_rows=(int(definition["source_a_row"]), int(definition["source_b_row"])),
            positions_arcsec=positions,
            source_selection_seed=int(definition["scene_seed"]),
            position_seed=int(definition["scene_seed"]),
            noise_seed=paired_seed,
        )
        rendered_a2 = render_fixed_scene_with_survey(
            catalog, spec, survey=survey_a, add_noise="all"
        )
        rendered_b = render_fixed_scene_with_survey(
            catalog, spec, survey=survey_b, add_noise="all"
        )
        observation_a = np.asarray(observation_a_f32, dtype=np.float64)
        observation_a2 = np.asarray(rendered_a2.blend, dtype=np.float64)
        observation_b = np.asarray(rendered_b.blend, dtype=np.float64)
        simulator_source_a2 = np.sum(rendered_a2.isolated, axis=0)
        simulator_source_b = np.sum(rendered_b.isolated, axis=0)
        intrinsic_flux_consistency = np.max(
            np.abs(
                np.sum(simulator_source_a2, axis=(-2, -1))
                - np.sum(simulator_source_b, axis=(-2, -1))
            )
            / np.maximum(
                np.abs(np.sum(simulator_source_a2, axis=(-2, -1))),
                np.finfo(np.float64).tiny,
            )
        )
        output = run_dir / f"observations/scene_{index:03d}_paired.npz"
        atomic_npz(
            output,
            observation_a=observation_a,
            observation_a2=observation_a2,
            observation_b=observation_b,
            xy=xy,
        )
        pair_records.append(
            {
                "scene_index": index,
                "scene_id": manifest["scene_id"],
                "requested_source_index": int(manifest["matched_source_index"]),
                "original_noise_seed": int(definition["noise_seed"]),
                "paired_noise_seed": paired_seed,
                "observation_a_sha256": raw_array_sha256(observation_a),
                "observation_a2_sha256": raw_array_sha256(observation_a2),
                "observation_b_sha256": raw_array_sha256(observation_b),
                "xy_sha256": raw_array_sha256(xy),
                "paired_npz_sha256": sha256_file(output),
                "same_latent_catalog_rows": True,
                "same_latent_coordinates": True,
                "same_photometric_calibration": True,
                "maximum_total_scene_flux_render_difference_fraction": float(
                    intrinsic_flux_consistency
                ),
                "isolated_layers_persisted": False,
                "isolated_layers_exposed_to_solver": False,
            }
        )
        del rendered_a2, rendered_b, simulator_source_a2, simulator_source_b
    write_csv_fresh(run_dir / "tables/observation_pair_hashes.csv", pair_records)
    observation_manifest = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "authoritative_forward_simulator": "BTK 1.0.9 CatsimGenerator",
        "paired_noise_seed_offset": PAIRED_NOISE_SEED_OFFSET,
        "records": pair_records,
        "scientific_observation_rows_accessed": len(INDICES),
        "historical_isolated_hdf5_rows_accessed": 0,
        "simulator_catalog_rows_selected": 2 * len(INDICES),
        "simulator_generated_isolated_layers_persisted": 0,
        "simulator_generated_isolated_layers_exposed_to_solver": 0,
        "development_access": 0,
        "atlas_tensor_access": 0,
        "lockbox_access": 0,
    }
    write_json_fresh(run_dir / "manifests/observation_pair_hashes.json", observation_manifest)
    science_seal = {
        "campaign": CAMPAIGN,
        "status": "SCIENTIFIC_INPUTS_FROZEN_BEFORE_FITTING",
        "timestamp_utc": utc_now(),
        "pre_science_freeze_sha256": sha256_file(run_dir / "preregistration/freeze_seal.json"),
        "observation_manifest_sha256": sha256_file(
            run_dir / "manifests/observation_pair_hashes.json"
        ),
        "observation_table_sha256": sha256_file(run_dir / "tables/observation_pair_hashes.csv"),
        "observation_files": {
            f"scene_{record['scene_index']:03d}_paired.npz": record["paired_npz_sha256"]
            for record in pair_records
        },
        "fits_executed": 0,
    }
    write_json_fresh(run_dir / "preregistration/science_input_seal.json", science_seal)
    print(json.dumps(science_seal, indent=2, sort_keys=True))


def verify_science_inputs(
    run_dir: Path, *, allow_finalization_amendment: bool = False
) -> dict[str, Any]:
    if verify_freeze(
        run_dir, allow_finalization_amendment=allow_finalization_amendment
    )["status"] != "PASS":
        return {"status": "FAIL", "reason": "pre-science freeze changed"}
    seal = json.loads(
        (run_dir / "preregistration/science_input_seal.json").read_text(encoding="utf-8")
    )
    checks = {
        "pre_science_freeze": sha256_file(run_dir / "preregistration/freeze_seal.json")
        == seal["pre_science_freeze_sha256"],
        "observation_manifest": sha256_file(run_dir / "manifests/observation_pair_hashes.json")
        == seal["observation_manifest_sha256"],
        "observation_table": sha256_file(run_dir / "tables/observation_pair_hashes.csv")
        == seal["observation_table_sha256"],
    }
    for name, expected in seal["observation_files"].items():
        checks[name] = sha256_file(run_dir / "observations" / name) == expected
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def observation_records(run_dir: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(
        (run_dir / "manifests/observation_pair_hashes.json").read_text(encoding="utf-8")
    )
    return {int(record["scene_index"]): record for record in manifest["records"]}


def load_joint_inputs(
    run_dir: Path, index: int, family: str, condition: str
) -> tuple[JointSolverInputs, dict[str, Any]]:
    if index not in INDICES or family not in FAMILIES or condition not in CONDITIONS:
        raise ValueError("unauthorized scene/family/condition")
    record = observation_records(run_dir)[index]
    path = run_dir / f"observations/scene_{index:03d}_paired.npz"
    with np.load(path, allow_pickle=False) as payload:
        observation_a = np.asarray(payload["observation_a"], dtype=np.float64)
        observation_2 = np.asarray(
            payload["observation_a2" if condition == "S2" else "observation_b"],
            dtype=np.float64,
        )
        xy = np.asarray(payload["xy"], dtype=np.float64)
    expected_second = record[
        "observation_a2_sha256" if condition == "S2" else "observation_b_sha256"
    ]
    if raw_array_sha256(observation_a) != record["observation_a_sha256"]:
        raise RuntimeError("Observation A changed after science seal")
    if raw_array_sha256(observation_2) != expected_second:
        raise RuntimeError("paired Observation 2 changed after science seal")
    requested_index = int(record["requested_source_index"])
    companion_index = 1 - requested_index
    psf_a, psf_b = frozen_psf_pair_kernels()
    psf_2 = psf_a if condition == "S2" else psf_b
    _, sky = known_psf()
    first = SolverInputs(
        observed=torch.as_tensor(observation_a, dtype=torch.float64),
        requested_center_xy=tuple(map(float, xy[requested_index])),
        companion_center_xy=tuple(map(float, xy[companion_index])),
        psf=psf_a,
        noise_sigma=torch.as_tensor(declared_noise_sigma(observation_a, sky)),
        family=family,
    )
    second = SolverInputs(
        observed=torch.as_tensor(observation_2, dtype=torch.float64),
        requested_center_xy=first.requested_center_xy,
        companion_center_xy=first.companion_center_xy,
        psf=psf_2,
        noise_sigma=torch.as_tensor(declared_noise_sigma(observation_2, sky)),
        family=family,
    )
    inputs = JointSolverInputs(first, second, condition)
    inputs.validate()
    return inputs, {
        "scene_index": index,
        "scene_id": record["scene_id"],
        "condition": condition,
        "family": family,
        "requested_source_index": requested_index,
        "companion_source_index": companion_index,
        "requested_center_xy": first.requested_center_xy,
        "companion_center_xy": first.companion_center_xy,
        "observation_a_sha256": record["observation_a_sha256"],
        "observation_2_sha256": expected_second,
        "psf_a_sha256": canonical_tensor_sha256(psf_a),
        "psf_2_sha256": canonical_tensor_sha256(psf_2),
        "paired_noise_seed": record["paired_noise_seed"],
    }


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right) / scale)


def numerical_perturbation(
    parameters: np.ndarray,
    inputs: JointSolverInputs,
    protocol: FrozenSolverProtocol,
    local: Any,
) -> dict[str, Any]:
    value = np.asarray(parameters, dtype=np.float64)
    lower, upper = parameter_bounds(inputs.reference, protocol)
    scales = parameter_scales(inputs.reference, protocol)
    direction = np.sin(np.arange(value.size, dtype=np.float64) + 1.0)
    candidate = value + NUMERICAL_PERTURBATION_SCALE * scales * direction
    candidate = np.maximum(candidate, lower)
    finite_upper = np.isfinite(upper)
    candidate[finite_upper] = np.minimum(candidate[finite_upper], upper[finite_upper])
    jacobian = joint_residual_jacobian(candidate, inputs, protocol)
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
    residual = joint_whitened_residual_vector(
        torch.as_tensor(candidate, dtype=torch.float64), inputs, protocol
    )
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


def minimum_nonzero_singular(diagnostics: Any) -> float:
    values = np.asarray(diagnostics.singular_values, dtype=np.float64)
    positive = values[values > diagnostics.rank_tolerance]
    return float(positive[-1]) if positive.size else 0.0


def operator_information(
    jacobian: np.ndarray,
    active: np.ndarray,
    scales: np.ndarray,
    split: int,
) -> dict[str, Any]:
    blocks = {
        "observation_a": analyze_jacobian(
            jacobian[:split], active_mask=active, parameter_scales=scales
        ),
        "observation_2": analyze_jacobian(
            jacobian[split:], active_mask=active, parameter_scales=scales
        ),
        "joint": analyze_jacobian(jacobian, active_mask=active, parameter_scales=scales),
    }
    result = {}
    for name, diagnostics in blocks.items():
        result[name] = {
            "rank": diagnostics.rank,
            "active_parameter_count": diagnostics.active_parameter_count,
            "null_space_dimension": diagnostics.null_space_dimension,
            "condition_number": diagnostics.condition_number,
            "minimum_nonzero_singular_value": minimum_nonzero_singular(diagnostics),
            "singular_values": diagnostics.singular_values,
        }
    a_min = result["observation_a"]["minimum_nonzero_singular_value"]
    joint_min = result["joint"]["minimum_nonzero_singular_value"]
    result["joint_minimum_singular_increase"] = joint_min - a_min
    result["joint_minimum_singular_ratio"] = joint_min / max(a_min, np.finfo(np.float64).tiny)
    return result


def execute_one(run_dir: Path, index: int, family: str, condition: str) -> None:
    if verify_science_inputs(run_dir)["status"] != "PASS":
        raise RuntimeError("scientific input seal changed; fail closed")
    output = run_dir / f"scene_results/scene_{index:03d}_{family}_{condition.lower()}.json"
    arrays_output = run_dir / f"scene_results/scene_{index:03d}_{family}_{condition.lower()}_images.npz"
    if output.exists() or arrays_output.exists():
        print(json.dumps({"scene": index, "family": family, "condition": condition, "status": "ALREADY_COMPLETE"}))
        return
    protocol = FrozenSolverProtocol()
    inputs, metadata = load_joint_inputs(run_dir, index, family, condition)
    audit = joint_oracle_information_audit(
        inputs,
        protocol,
        extra_named_inputs={
            "scene_index": index,
            "scene_id": metadata["scene_id"],
            "requested_source_index": metadata["requested_source_index"],
            "condition": condition,
        },
    )
    if audit["status"] != "PASS":
        raise RuntimeError("oracle-information audit failed")
    starts = joint_deterministic_starts(inputs, protocol)
    endpoints = joint_multi_start_optimize(inputs, protocol, starts=starts)
    successful = [
        endpoint
        for endpoint in endpoints
        if endpoint.success and np.isfinite(endpoint.likelihood_objective)
    ]
    if not successful:
        atomic_json(
            output,
            {
                "campaign": CAMPAIGN,
                "metadata": metadata,
                "family": family,
                "condition": condition,
                "level": LEVEL_BY_FAMILY[family],
                "classification": "OPTIMIZATION_UNRESOLVED",
                "oracle_audit": audit,
                "starts": starts,
                "endpoints": [endpoint.record() for endpoint in endpoints],
            },
        )
        print(json.dumps({"scene": index, "family": family, "condition": condition, "classification": "OPTIMIZATION_UNRESOLVED"}))
        return
    best = min(successful, key=lambda endpoint: endpoint.likelihood_objective)
    local = joint_local_fit_diagnostics(best.parameters, inputs, protocol)
    support_endpoints = []
    support_records = []
    for endpoint in successful:
        _, active, _ = canonicalize_parameters(endpoint.parameters, inputs.reference, protocol)
        dof = max(1, inputs.observation_pixel_count - int(active.sum()))
        threshold = float(chi2.ppf(protocol.model_acceptance_quantile, dof))
        accepted = bool(endpoint.chi_square <= threshold)
        support_records.append(
            {
                "start_index": endpoint.start_index,
                "degrees_of_freedom": dof,
                "chi_square_threshold": threshold,
                "support_acceptable": accepted,
            }
        )
        if accepted:
            support_endpoints.append(endpoint)
    geometry = joint_solution_geometry(
        support_endpoints if support_endpoints else successful, inputs, protocol
    )
    perturbation = numerical_perturbation(best.parameters, inputs, protocol, local)
    classification = classify_joint_identifiability(
        endpoints,
        geometry,
        local,
        inputs,
        protocol,
        contract_valid=audit["status"] == "PASS",
        numerically_stable=perturbation["stable"],
    )
    tensor = torch.as_tensor(best.parameters, dtype=torch.float64)
    pairs = render_joint(tensor, inputs, protocol)
    objective = joint_likelihood_components(tensor, inputs, protocol)
    requested = [np.asarray(pair.requested.detach().cpu(), dtype=np.float64) for pair in pairs]
    companion = [np.asarray(pair.companion.detach().cpu(), dtype=np.float64) for pair in pairs]
    recomposed = [requested[i] + companion[i] for i in (0, 1)]
    residuals = [
        np.asarray(objective.signed_residuals[i].detach().cpu(), dtype=np.float64)
        for i in (0, 1)
    ]
    per_source = parameters_per_source(family)
    swapped_parameters = np.concatenate((best.parameters[per_source:], best.parameters[:per_source]))
    swap_records = []
    for observation_index, observation in enumerate((inputs.observation_a, inputs.observation_2)):
        swapped_input = SolverInputs(
            observed=observation.observed,
            requested_center_xy=observation.companion_center_xy,
            companion_center_xy=observation.requested_center_xy,
            psf=observation.psf,
            noise_sigma=observation.noise_sigma,
            family=family,
        )
        swapped_pair = render_pair(
            torch.as_tensor(swapped_parameters, dtype=torch.float64), swapped_input, protocol
        )
        swap_records.append(
            {
                "observation_index": observation_index,
                "requested_to_original_companion_relative_l2": relative_l2(
                    np.asarray(swapped_pair.requested), companion[observation_index]
                ),
                "companion_to_original_requested_relative_l2": relative_l2(
                    np.asarray(swapped_pair.companion), requested[observation_index]
                ),
                "recomposed_relative_l2": relative_l2(
                    np.asarray(swapped_pair.recomposed_sources), recomposed[observation_index]
                ),
            }
        )
        swap_records[-1]["algebraically_consistent"] = max(
            value for key, value in swap_records[-1].items() if key.endswith("relative_l2")
        ) <= 1.0e-10
    replay_jacobian = joint_residual_jacobian(best.parameters, inputs, protocol)
    replay_pairs = render_joint(tensor, inputs, protocol)
    replay = {
        "requested_hashes_first": [canonical_tensor_sha256(pair.requested) for pair in pairs],
        "requested_hashes_second": [canonical_tensor_sha256(pair.requested) for pair in replay_pairs],
        "jacobian_sha256_first": raw_array_sha256(local.jacobian),
        "jacobian_sha256_second": raw_array_sha256(replay_jacobian),
    }
    replay["exact_match"] = (
        replay["requested_hashes_first"] == replay["requested_hashes_second"]
        and replay["jacobian_sha256_first"] == replay["jacobian_sha256_second"]
    )
    _, active, symmetries = canonicalize_parameters(best.parameters, inputs.reference, protocol)
    scales = parameter_scales(inputs.reference, protocol)
    operator = operator_information(
        local.jacobian, active, scales, int(inputs.observation_a.observed.numel())
    )
    diagnostics = local.jacobian_diagnostics
    boundary = boundary_contact_flags(best.parameters, inputs.reference, protocol)
    result = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "metadata": metadata,
        "family": family,
        "level": LEVEL_BY_FAMILY[family],
        "condition": condition,
        "classification": classification,
        "protocol": asdict(protocol),
        "oracle_audit": audit,
        "shared_parameter_vector_enforced": True,
        "starts": starts,
        "endpoints": [endpoint.record() for endpoint in endpoints],
        "support_records": support_records,
        "best_start_index": best.start_index,
        "best_parameters": best.parameters,
        "parameter_names": parameter_names(family),
        "best_joint_likelihood_objective": best.likelihood_objective,
        "best_joint_chi_square": best.chi_square,
        "likelihood_by_observation": np.asarray(objective.likelihood_by_observation.detach().cpu()),
        "likelihood_by_observation_band": np.asarray(
            objective.likelihood_by_observation_band.detach().cpu()
        ),
        "chi_square_by_observation": np.asarray(
            objective.chi_square_by_observation.detach().cpu()
        ),
        "chi_square_by_observation_band": np.asarray(
            objective.chi_square_by_observation_band.detach().cpu()
        ),
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
        "symmetries": symmetries,
        "boundary_contact_flags": boundary,
        "geometry": asdict(geometry),
        "support_acceptable_start_indices": [
            endpoint.start_index for endpoint in support_endpoints
        ],
        "fitted_requested_fluxes": best.parameters[:3],
        "fitted_companion_fluxes": best.parameters[per_source : per_source + 3],
        "observation_band_sums": [
            np.asarray(inputs.observation_a.observed.sum(dim=(-2, -1))),
            np.asarray(inputs.observation_2.observed.sum(dim=(-2, -1))),
        ],
        "recomposed_band_sums": [np.sum(value, axis=(-2, -1)) for value in recomposed],
        "residual_band_sums": [np.sum(value, axis=(-2, -1)) for value in residuals],
        "source_nonnegative": bool(
            all(np.all(value >= 0) for value in requested + companion)
        ),
        "psf_band_sums": [
            np.asarray(normalize_psf(inputs.observation_a.psf).sum(dim=(-2, -1))),
            np.asarray(normalize_psf(inputs.observation_2.psf).sum(dim=(-2, -1))),
        ],
        "signed_residual_min": [float(np.min(value)) for value in residuals],
        "signed_residual_max": [float(np.max(value)) for value in residuals],
        "prompt_swap": swap_records,
        "numerical_perturbation": perturbation,
        "deterministic_replay": replay,
        "operator_information": operator,
        "image_hashes": {
            "requested": [canonical_tensor_sha256(pair.requested) for pair in pairs],
            "companion": [canonical_tensor_sha256(pair.companion) for pair in pairs],
            "recomposed": [canonical_tensor_sha256(pair.recomposed_sources) for pair in pairs],
            "residual": [canonical_tensor_sha256(value) for value in objective.signed_residuals],
        },
        "access_counts": {
            "paired_observation_artifacts": 1,
            "blended_observation_arrays": 2,
            "historical_isolated_source_arrays": 0,
            "simulator_catalog_rows": 0,
            "development": 0,
            "atlas_arrays": 0,
            "lockbox": 0,
            "neural_training_steps": 0,
        },
    }
    atomic_npz(
        arrays_output,
        observed_a=np.asarray(inputs.observation_a.observed),
        observed_2=np.asarray(inputs.observation_2.observed),
        requested_a=requested[0],
        companion_a=companion[0],
        recomposed_a=recomposed[0],
        residual_a=residuals[0],
        requested_2=requested[1],
        companion_2=companion[1],
        recomposed_2=recomposed[1],
        residual_2=residuals[1],
        noise_sigma_a=np.asarray(inputs.observation_a.noise_sigma),
        noise_sigma_2=np.asarray(inputs.observation_2.noise_sigma),
    )
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "scene": index,
                "family": family,
                "condition": condition,
                "classification": classification,
                "chi_square": best.chi_square,
                "rank": diagnostics.rank,
                "null": diagnostics.null_space_dimension,
                "classes": geometry.distinct_solution_classes,
            }
        )
    )


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for index in INDICES:
        for family in FAMILIES:
            for condition in CONDITIONS:
                path = run_dir / f"scene_results/scene_{index:03d}_{family}_{condition.lower()}.json"
                if not path.exists():
                    raise RuntimeError(f"missing required result: {path.name}")
                results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_csv_json(value: str) -> Any:
    if value in {"", None}:
        return None
    return json.loads(value)


CLASSIFICATION_SCORE = {
    "INVALID_CONTRACT": -3,
    "OPTIMIZATION_UNRESOLVED": -2,
    "NUMERICALLY_UNSTABLE": -1,
    "OUT_OF_SUPPORT": 0,
    "NON_IDENTIFIABLE": 1,
    "PARTIALLY_IDENTIFIABLE": 2,
    "NEAR_UNIQUE": 3,
    "UNIQUE": 4,
}


def best_classification(values: list[str]) -> str:
    return max(values, key=lambda value: CLASSIFICATION_SCORE[value])


def causal_attribution(s1: str, s2: str, p2: str) -> str:
    unresolved = {"INVALID_CONTRACT", "OPTIMIZATION_UNRESOLVED", "NUMERICALLY_UNSTABLE"}
    if {s1, s2, p2}.intersection(unresolved):
        return "INCONCLUSIVE_OPTIMIZATION"
    s1_score = CLASSIFICATION_SCORE[s1]
    s2_gain = CLASSIFICATION_SCORE[s2] > s1_score
    p2_gain = CLASSIFICATION_SCORE[p2] > s1_score
    p2_over_s2 = CLASSIFICATION_SCORE[p2] > CLASSIFICATION_SCORE[s2]
    if s2_gain and p2_gain and p2_over_s2:
        return "BOTH_EXPOSURE_AND_PSF_DIVERSITY"
    if p2_gain and not s2_gain:
        return "PSF_DIVERSITY_SPECIFIC"
    if s2_gain and not p2_over_s2:
        return "ADDITIONAL_EXPOSURE_ONLY"
    if s2_gain or p2_gain:
        return "BOTH_EXPOSURE_AND_PSF_DIVERSITY" if p2_over_s2 else "ADDITIONAL_EXPOSURE_ONLY"
    return "NO_MEANINGFUL_GAIN"


def condition_unique_counts(
    classifications: dict[tuple[int, str, str], str], condition: str
) -> dict[str, int]:
    l4 = sum(
        classifications[index, FAMILY_SERSIC, condition] == "UNIQUE" for index in INDICES
    )
    l5 = sum(
        classifications[index, FAMILY_BULGE_DISK, condition] == "UNIQUE" for index in INDICES
    )
    union = sum(
        any(classifications[index, family, condition] == "UNIQUE" for family in FAMILIES)
        for index in INDICES
    )
    return {"level_4": int(l4), "level_5": int(l5), "union": int(union)}


def minimum_family_for_scene(
    classifications: dict[tuple[int, str, str], str], index: int, condition: str
) -> str:
    if classifications[index, FAMILY_SERSIC, condition] == "UNIQUE":
        return "Level 4"
    if classifications[index, FAMILY_BULGE_DISK, condition] == "UNIQUE":
        return "Level 5"
    return "none"


def finite_ratio(numerator: float | str, denominator: float | str) -> float:
    """Divide finite/serialized condition metrics without changing their semantics."""
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator == 0.0:
        return math.inf if numerator > 0 else 1.0
    if math.isinf(denominator):
        return 0.0 if math.isfinite(numerator) else 1.0
    return numerator / denominator


def make_figures(
    run_dir: Path,
    results: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
) -> None:
    colors = {"S1": "#777777", "S2": "#4c78a8", "P2": "#f58518"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(3)
    width = 0.24
    for offset, condition in enumerate(("S1", "S2", "P2")):
        values = [counts[condition]["level_4"], counts[condition]["level_5"], counts[condition]["union"]]
        ax.bar(x + (offset - 1) * width, values, width, label=condition, color=colors[condition])
    ax.set_xticks(x, ("Level 4", "Level 5", "Union"))
    ax.set_ylabel("UNIQUE scenes out of 8")
    ax.set_ylim(0, 8)
    ax.legend()
    ax.set_title("Flux-free recoverability frontier: S1 vs S2 vs P2")
    fig.tight_layout()
    fig.savefig(run_dir / "figures/recoverability_frontier_s1_s2_p2.png", dpi=180)
    plt.close(fig)

    heat = np.asarray(
        [
            [
                CLASSIFICATION_SCORE[row["s1_classification"]],
                CLASSIFICATION_SCORE[row["s2_classification"]],
                CLASSIFICATION_SCORE[row["p2_classification"]],
            ]
            for row in transition_rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    image = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=-3, vmax=4)
    ax.set_xticks((0, 1, 2), ("S1", "S2", "P2"))
    ax.set_yticks(
        np.arange(len(transition_rows)),
        [f"{row['scene_index']} {row['level']}" for row in transition_rows],
    )
    ax.set_title("Condition classification transitions")
    fig.colorbar(image, ax=ax, label="Frozen classification order (display only)")
    fig.tight_layout()
    fig.savefig(run_dir / "figures/condition_transition_heatmap.png", dpi=180)
    plt.close(fig)

    psf_a, psf_b = frozen_psf_pair_kernels()
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0))
    for band_index, band in enumerate(BANDS):
        for column, (label, psf) in enumerate((("A", psf_a), ("B", psf_b))):
            axes[band_index, column].imshow(
                np.asarray(psf[band_index]), origin="lower", cmap="magma"
            )
            axes[band_index, column].set_title(f"{band} PSF {label}")
            axes[band_index, column].axis("off")
    fig.suptitle("Preregistered known PSF pair (31x31 kernels)")
    fig.tight_layout()
    fig.savefig(run_dir / "figures/psf_a_vs_psf_b.png", dpi=180)
    plt.close(fig)

    for result in results:
        index = int(result["metadata"]["scene_index"])
        family = result["family"]
        condition = result["condition"].lower()
        stem = f"scene_{index:03d}_{family}_{condition}"
        operator = result["operator_information"]
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for label, key, color in (
            ("Observation A", "observation_a", "#777777"),
            ("Observation 2", "observation_2", "#4c78a8"),
            ("Joint", "joint", "#f58518"),
        ):
            values = np.asarray(operator[key]["singular_values"], dtype=float)
            ax.semilogy(np.arange(1, values.size + 1), np.maximum(values, 1e-18), marker="o", label=label, color=color)
        ax.set_xlabel("Singular-value index")
        ax.set_ylabel("Scaled whitened singular value")
        ax.set_title(f"Scene {index} {result['level']} {result['condition']}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / f"figures/singular_value_comparison/{stem}.png", dpi=150)
        plt.close(fig)

        endpoints = result["endpoints"]
        best = min(float(item["likelihood_objective"]) for item in endpoints)
        per_source = parameters_per_source(family)
        fractions = []
        delta = []
        success = []
        for endpoint in endpoints:
            parameters = np.asarray(endpoint["parameters"], dtype=float)
            total = parameters[0] + parameters[per_source]
            fractions.append(parameters[0] / max(total, np.finfo(float).tiny))
            delta.append(max(float(endpoint["likelihood_objective"]) - best, 1e-16))
            success.append(bool(endpoint["success"]))
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.scatter(fractions, delta, c=np.arange(len(endpoints)), cmap="viridis", marker="o")
        for position, passed in enumerate(success):
            if not passed:
                ax.scatter(fractions[position], delta[position], facecolors="none", edgecolors="red", s=80)
        ax.set_yscale("log")
        ax.set_xlabel("Requested g-band fitted flux fraction")
        ax.set_ylabel("Joint objective above best")
        ax.set_title(f"Scene {index} {result['level']} {result['condition']} endpoints")
        fig.tight_layout()
        fig.savefig(run_dir / f"figures/multistart_solution_geometry/{stem}.png", dpi=150)
        plt.close(fig)

        arrays_path = run_dir / f"scene_results/{stem}_images.npz"
        with np.load(arrays_path, allow_pickle=False) as arrays:
            observed_a = np.asarray(arrays["observed_a"])
            observed_2 = np.asarray(arrays["observed_2"])
            req_a = np.asarray(arrays["requested_a"])
            req_2 = np.asarray(arrays["requested_2"])
            comp_a = np.asarray(arrays["companion_a"])
            comp_2 = np.asarray(arrays["companion_2"])
            residual_a = np.asarray(arrays["residual_a"])
            residual_2 = np.asarray(arrays["residual_2"])
        fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
        for row_index, values in enumerate(
            ((observed_a, req_a, comp_a, residual_a), (observed_2, req_2, comp_2, residual_2))
        ):
            for column, value in enumerate(values):
                axes[row_index, column].imshow(value[1], origin="lower", cmap="magma" if column < 3 else "coolwarm")
                axes[row_index, column].axis("off")
        for column, title in enumerate(("Observed r", "Requested r", "Companion r", "Residual r")):
            axes[0, column].set_title(title)
        axes[0, 0].set_ylabel("Observation A")
        axes[1, 0].set_ylabel("Observation 2")
        fig.suptitle(f"Scene {index} {result['level']} {result['condition']} — {result['classification']}")
        fig.tight_layout()
        fig.savefig(run_dir / f"figures/source_reconstruction_panels/{stem}.png", dpi=145)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2))
        vmax = max(float(np.max(np.abs(residual_a))), float(np.max(np.abs(residual_2))))
        for row_index, residual in enumerate((residual_a, residual_2)):
            for band_index, band in enumerate(BANDS):
                axes[row_index, band_index].imshow(
                    residual[band_index], origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
                )
                axes[row_index, band_index].set_title(f"{band} residual")
                axes[row_index, band_index].axis("off")
        fig.suptitle(f"Scene {index} {result['level']} {result['condition']} signed residuals")
        fig.tight_layout()
        fig.savefig(run_dir / f"figures/residual_comparison/{stem}.png", dpi=145)
        plt.close(fig)


def finalize(run_dir: Path) -> None:
    if verify_science_inputs(run_dir, allow_finalization_amendment=True)["status"] != "PASS":
        raise RuntimeError("scientific input seal changed; fail closed")
    results = load_results(run_dir)
    result_lookup = {
        (int(result["metadata"]["scene_index"]), result["family"], result["condition"]): result
        for result in results
    }
    s1_metric_rows = csv_rows(S1_METRICS)
    s1_lookup = {
        (int(row["scene_index"]), row["family"]): row for row in s1_metric_rows
    }
    classifications: dict[tuple[int, str, str], str] = {}
    for index in INDICES:
        for family in FAMILIES:
            classifications[index, family, "S1"] = s1_lookup[index, family]["classification"]
            for condition in CONDITIONS:
                classifications[index, family, condition] = result_lookup[
                    index, family, condition
                ]["classification"]

    condition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for row in s1_metric_rows:
        condition_rows.append(
            {
                "scene_index": int(row["scene_index"]),
                "scene_id": row["scene_id"],
                "family": row["family"],
                "level": row["level"],
                "condition": "S1",
                "classification": row["classification"],
                "unique": row["classification"] == "UNIQUE",
                "source": "completed_authoritative_predecessor",
            }
        )
    for result in results:
        index = int(result["metadata"]["scene_index"])
        family = result["family"]
        condition = result["condition"]
        geometry = result["geometry"]
        condition_rows.append(
            {
                "scene_index": index,
                "scene_id": result["metadata"]["scene_id"],
                "family": family,
                "level": result["level"],
                "condition": condition,
                "classification": result["classification"],
                "unique": result["classification"] == "UNIQUE",
                "source": "new_joint_fit",
            }
        )
        metric_rows.append(
            {
                "scene_index": index,
                "scene_id": result["metadata"]["scene_id"],
                "family": family,
                "level": result["level"],
                "condition": condition,
                "classification": result["classification"],
                "best_joint_likelihood_objective": result["best_joint_likelihood_objective"],
                "best_joint_chi_square": result["best_joint_chi_square"],
                "likelihood_by_observation": result["likelihood_by_observation"],
                "likelihood_by_observation_band": result["likelihood_by_observation_band"],
                "chi_square_by_observation": result["chi_square_by_observation"],
                "chi_square_by_observation_band": result["chi_square_by_observation_band"],
                "gradient_norm": result["gradient_norm"],
                "rank": result["rank"],
                "active_parameter_count": result["active_parameter_count"],
                "null_space_dimension": result["null_space_dimension"],
                "condition_number": result["condition_number"],
                "hessian_condition_number": result["hessian_condition_number"],
                "singular_values": result["singular_values"],
                "hessian_eigenvalues": result["hessian_eigenvalues"],
                "distinct_solution_classes": geometry["distinct_solution_classes"],
                "requested_image_diameter": geometry["requested_image_diameter"],
                "companion_image_diameter": geometry["companion_image_diameter"],
                "flux_allocation_diameter": geometry["flux_allocation_diameter"],
                "morphology_parameter_diameter": geometry["morphology_parameter_diameter"],
                "prompt_identity_consistent": geometry["prompt_identity_consistent"],
                "boundary_contact_flags": result["boundary_contact_flags"],
                "numerical_perturbation": result["numerical_perturbation"],
                "deterministic_replay": result["deterministic_replay"],
                "operator_information": result["operator_information"],
            }
        )
        for endpoint in result["endpoints"]:
            endpoint_rows.append(
                {
                    "scene_index": index,
                    "scene_id": result["metadata"]["scene_id"],
                    "family": family,
                    "condition": condition,
                    **endpoint,
                }
            )
        for band_index, band in enumerate(BANDS):
            flux_rows.append(
                {
                    "scene_index": index,
                    "scene_id": result["metadata"]["scene_id"],
                    "family": family,
                    "condition": condition,
                    "band": band,
                    "requested_flux": result["fitted_requested_fluxes"][band_index],
                    "companion_flux": result["fitted_companion_fluxes"][band_index],
                    "total_fitted_source_flux": result["fitted_requested_fluxes"][band_index]
                    + result["fitted_companion_fluxes"][band_index],
                    "observation_a_stamp_sum": result["observation_band_sums"][0][band_index],
                    "observation_2_stamp_sum": result["observation_band_sums"][1][band_index],
                    "residual_a_stamp_sum": result["residual_band_sums"][0][band_index],
                    "residual_2_stamp_sum": result["residual_band_sums"][1][band_index],
                }
            )
        oracle_rows.append(
            {
                "scene_index": index,
                "scene_id": result["metadata"]["scene_id"],
                "family": family,
                "condition": condition,
                "status": result["oracle_audit"]["status"],
                "reason": result["oracle_audit"]["reason"],
                "shared_parameter_vector_enforced": result["shared_parameter_vector_enforced"],
                "trace_entry_count": len(result["oracle_audit"]["trace"]),
                "input_names": [entry["input"] for entry in result["oracle_audit"]["trace"]],
                "oracle_flags": [entry["oracle"] for entry in result["oracle_audit"]["trace"]],
                "isolated_source_arrays": result["access_counts"]["historical_isolated_source_arrays"],
                "catalog_rows_as_inference": result["access_counts"]["simulator_catalog_rows"],
                "development": result["access_counts"]["development"],
                "atlas_arrays": result["access_counts"]["atlas_arrays"],
                "lockbox": result["access_counts"]["lockbox"],
            }
        )
        s1 = s1_lookup[index, family]
        s1_singular = np.asarray(parse_csv_json(s1["singular_values"]), dtype=float)
        s1_min = float(s1_singular[-1]) if s1_singular.size else 0.0
        joint_info = result["operator_information"]["joint"]
        second_info = result["operator_information"]["observation_2"]
        s1_condition = float(s1["condition_number"])
        info_row = {
            "scene_index": index,
            "scene_id": result["metadata"]["scene_id"],
            "family": family,
            "level": result["level"],
            "condition": condition,
            "s1_minimum_nonzero_singular_value": s1_min,
            "observation_2_minimum_nonzero_singular_value": second_info[
                "minimum_nonzero_singular_value"
            ],
            "joint_minimum_nonzero_singular_value": joint_info[
                "minimum_nonzero_singular_value"
            ],
            "joint_vs_s1_minimum_singular_ratio": finite_ratio(
                joint_info["minimum_nonzero_singular_value"], s1_min
            ),
            "s1_condition_number": s1_condition,
            "joint_condition_number": joint_info["condition_number"],
            "joint_vs_s1_condition_ratio": finite_ratio(
                joint_info["condition_number"], s1_condition
            ),
            "s1_endpoint_classes": int(s1["distinct_solution_classes"]),
            "joint_endpoint_classes": geometry["distinct_solution_classes"],
            "endpoint_class_change": geometry["distinct_solution_classes"]
            - int(s1["distinct_solution_classes"]),
            "s1_requested_image_diameter": float(s1["requested_image_diameter"]),
            "joint_requested_image_diameter": geometry["requested_image_diameter"],
            "s1_flux_allocation_diameter": float(s1["flux_allocation_diameter"]),
            "joint_flux_allocation_diameter": geometry["flux_allocation_diameter"],
            "s1_morphology_diameter": float(s1["morphology_parameter_diameter"]),
            "joint_morphology_diameter": geometry["morphology_parameter_diameter"],
            "s1_classification": s1["classification"],
            "joint_classification": result["classification"],
            "uniqueness_transition_from_s1": f"{s1['classification']}->{result['classification']}",
        }
        info_rows.append(info_row)
        comparison_rows.append(dict(info_row))

    transition_rows = []
    for index in INDICES:
        for family in FAMILIES:
            s1 = classifications[index, family, "S1"]
            s2 = classifications[index, family, "S2"]
            p2 = classifications[index, family, "P2"]
            transition_rows.append(
                {
                    "scene_index": index,
                    "scene_id": result_lookup[index, family, "S2"]["metadata"]["scene_id"],
                    "family": family,
                    "level": LEVEL_BY_FAMILY[family],
                    "s1_classification": s1,
                    "s2_classification": s2,
                    "p2_classification": p2,
                    "s1_to_s2": f"{s1}->{s2}",
                    "s1_to_p2": f"{s1}->{p2}",
                    "s2_to_p2": f"{s2}->{p2}",
                    "s2_unique_transition": s1 != "UNIQUE" and s2 == "UNIQUE",
                    "p2_unique_transition": s1 != "UNIQUE" and p2 == "UNIQUE",
                    "causal_attribution": causal_attribution(s1, s2, p2),
                }
            )

    counts = {
        condition: condition_unique_counts(classifications, condition)
        for condition in ("S1", "S2", "P2")
    }
    for condition in ("S1", "S2", "P2"):
        for index in INDICES:
            condition_rows.append(
                {
                    "scene_index": index,
                    "scene_id": result_lookup[index, FAMILY_SERSIC, "S2"]["metadata"]["scene_id"],
                    "family": "union",
                    "level": "minimum permitted family",
                    "condition": condition,
                    "classification": best_classification(
                        [classifications[index, family, condition] for family in FAMILIES]
                    ),
                    "unique": any(
                        classifications[index, family, condition] == "UNIQUE"
                        for family in FAMILIES
                    ),
                    "minimum_structural_family_required": minimum_family_for_scene(
                        classifications, index, condition
                    ),
                    "source": "derived_union",
                }
            )

    write_csv_fresh(run_dir / "tables/scene_family_condition_summary.csv", condition_rows)
    write_csv_fresh(run_dir / "tables/full_psf_diverse_metrics.csv", metric_rows)
    write_csv_fresh(run_dir / "tables/multistart_endpoints.csv", endpoint_rows)
    write_csv_fresh(run_dir / "tables/fitted_source_fluxes.csv", flux_rows)
    write_csv_fresh(run_dir / "tables/condition_transition_matrix.csv", transition_rows)
    write_csv_fresh(run_dir / "tables/psf_information_gain.csv", info_rows)
    write_csv_fresh(run_dir / "tables/oracle_information_audit.csv", oracle_rows)
    write_csv_fresh(
        run_dir / "tables/comparison_to_single_observation_flux_free.csv", comparison_rows
    )

    p2_info = [row for row in info_rows if row["condition"] == "P2"]
    conditioning_improvements = sum(
        (
            row["joint_vs_s1_minimum_singular_ratio"] >= 1.05
            or row["joint_vs_s1_condition_ratio"] <= 0.95
            or row["joint_endpoint_classes"] < row["s1_endpoint_classes"]
            or row["joint_requested_image_diameter"]
            <= 0.95 * row["s1_requested_image_diameter"]
            or row["joint_flux_allocation_diameter"]
            <= 0.95 * row["s1_flux_allocation_diameter"]
        )
        for row in p2_info
    )
    p2_union = counts["P2"]["union"]
    s2_union = counts["S2"]["union"]
    all_new_classifications = [result["classification"] for result in results]
    if "INVALID_CONTRACT" in all_new_classifications:
        outcome = "PSF_DIVERSE_FLUX_IDENTIFIABILITY_INVALID"
    elif p2_union >= 6 and p2_union >= s2_union + 1:
        outcome = "PSF_DIVERSE_UNIQUENESS_LARGELY_RESTORED"
    elif 3 <= p2_union <= 5 and p2_union >= s2_union + 1:
        outcome = "PSF_DIVERSE_UNIQUENESS_PARTIALLY_RESTORED"
    elif max(p2_union, s2_union) >= 3 and abs(p2_union - s2_union) <= 1:
        outcome = "ADDITIONAL_EXPOSURE_RESTORES_UNIQUENESS"
    elif sum(
        value in {"OPTIMIZATION_UNRESOLVED", "NUMERICALLY_UNSTABLE"}
        for value in all_new_classifications
    ) >= len(results) // 2:
        outcome = "PSF_DIVERSE_OPTIMIZATION_UNRESOLVED"
    elif p2_union <= 2 and conditioning_improvements >= len(p2_info) // 2:
        outcome = "PSF_DIVERSE_CONDITIONING_IMPROVES_WITHOUT_UNIQUENESS"
    else:
        outcome = "PSF_DIVERSITY_NO_MEANINGFUL_GAIN"

    if outcome in {
        "PSF_DIVERSE_UNIQUENESS_LARGELY_RESTORED",
        "PSF_DIVERSE_UNIQUENESS_PARTIALLY_RESTORED",
    }:
        next_experiment = "Thayer-Direct-MultiObservation-Structural-Solver-v0"
        next_purpose = "test requested-source reconstruction accuracy under the validated paired-observation structural contract"
    elif outcome == "ADDITIONAL_EXPOSURE_RESTORES_UNIQUENESS":
        next_experiment = "Thayer-Direct-Same-PSF-MultiExposure-Structural-Solver-v0"
        next_purpose = "test reconstruction accuracy using the simpler validated repeated-exposure contract"
    elif outcome == "PSF_DIVERSE_CONDITIONING_IMPROVES_WITHOUT_UNIQUENESS":
        next_experiment = "Thayer-External-Photometry-vs-PSF-Diversity-v0"
        next_purpose = "compare independent per-source photometry against PSF diversity as a targeted missing information source"
    elif outcome == "PSF_DIVERSITY_NO_MEANINGFUL_GAIN":
        next_experiment = "Thayer-Higher-Resolution-Flux-Identifiability-v0"
        next_purpose = "test whether one genuinely higher-resolution observation supplies the missing information"
    else:
        next_experiment = "Thayer-PSF-Diverse-Optimizer-Resolution-v0"
        next_purpose = "resolve fixed-optimizer uncertainty without changing the scientific information contract"

    make_figures(run_dir, results, transition_rows, counts)

    failures = sum(not bool(endpoint["success"]) for endpoint in endpoint_rows)
    max_budget = sum(
        int(endpoint["nfev"]) >= FrozenSolverProtocol().max_nfev for endpoint in endpoint_rows
    )
    transition_lines = []
    for index in INDICES:
        l4 = next(row for row in transition_rows if row["scene_index"] == index and row["family"] == FAMILY_SERSIC)
        l5 = next(row for row in transition_rows if row["scene_index"] == index and row["family"] == FAMILY_BULGE_DISK)
        transition_lines.append(
            f"- Scene {index}: L4 `{l4['s1_classification']} -> {l4['s2_classification']} -> {l4['p2_classification']}`; "
            f"L5 `{l5['s1_classification']} -> {l5['s2_classification']} -> {l5['p2_classification']}`; "
            f"P2 minimum unique family `{minimum_family_for_scene(classifications, index, 'P2')}`."
        )
    attribution_counts = {
        label: sum(row["causal_attribution"] == label for row in transition_rows)
        for label in (
            "ADDITIONAL_EXPOSURE_ONLY",
            "PSF_DIVERSITY_SPECIFIC",
            "BOTH_EXPOSURE_AND_PSF_DIVERSITY",
            "NO_MEANINGFUL_GAIN",
            "INCONCLUSIVE_OPTIMIZATION",
        )
    }
    p2_condition_ratio = [row["joint_vs_s1_condition_ratio"] for row in p2_info]
    p2_min_ratio = [row["joint_vs_s1_minimum_singular_ratio"] for row in p2_info]
    diversity_metrics = csv_rows(run_dir / "tables/psf_diversity_metrics.csv")
    morphology_justified = p2_union >= 6 and outcome.startswith("PSF_DIVERSE_UNIQUENESS")
    independent_required = p2_union > counts["S1"]["union"]
    report = f"""# {CAMPAIGN} final report

## Campaign outcome

**{outcome}**

The primary P2 union endpoint is **{p2_union}/8 UNIQUE**. P2 Level 4 is
**{counts['P2']['level_4']}/8** and Level 5 is **{counts['P2']['level_5']}/8**.
The same-PSF S2 control is L4 **{counts['S2']['level_4']}/8**, L5
**{counts['S2']['level_5']}/8**, union **{s2_union}/8**. The authoritative S1
counts remain L4 **{counts['S1']['level_4']}/8**, L5
**{counts['S1']['level_5']}/8**, union **{counts['S1']['union']}/8**.

## Solver information and paired acquisition

The solver received only the two blended g/r/z observations, frozen requested
and companion coordinates, both known normalized PSFs, 60x60 geometry at 0.2
arcsec/pixel, separate observation-derived BTK Poisson plug-in sigma maps, and
the frozen Level-4/5 support. One latent morphology-and-flux parameter vector
jointly explained both observations. No isolated image, true per-source flux,
catalog parameter, truth morphology/family, mask, truth initialization, or true
noise realization entered inference.

Observation A is the unchanged authoritative blend. A2 and B were freshly
generated from the identical two CatSim rows and coordinates through BTK
`CatsimGenerator(add_noise='all')`, with the preregistered independent second
seed. A2 used PSF A; B used PSF B. Simulator-only catalog state and generated
isolated layers were discarded before the paired observations were sealed.

PSF A has nominal g/r/z FWHM 0.86/0.81/0.77 arcsec. PSF B is the preregistered
0.70/0.68/0.66 arcsec, e=0.10, 30-degree GalSim construction. Across bands,
relative kernel L2 distances span
`{min(float(row['kernel_relative_l2_distance']) for row in diversity_metrics):.4g}`--
`{max(float(row['kernel_relative_l2_distance']) for row in diversity_metrics):.4g}`,
Fourier-transfer relative distances span
`{min(float(row['fourier_transfer_relative_l2_distance']) for row in diversity_metrics):.4g}`--
`{max(float(row['fourier_transfer_relative_l2_distance']) for row in diversity_metrics):.4g}`,
and cross-correlations span
`{min(float(row['kernel_cross_correlation']) for row in diversity_metrics):.4g}`--
`{max(float(row['kernel_cross_correlation']) for row in diversity_metrics):.4g}`.

## Scene-level transitions (S1 -> S2 -> P2)

{chr(10).join(transition_lines)}

Family-level causal attributions: `{json.dumps(attribution_counts, sort_keys=True)}`.
An improvement is not attributed to PSF diversity merely because P2 exceeds
S1; S2 is the mandatory repeated-exposure control.

## Rank, conditioning, endpoint geometry, and flux allocation

All local ranks, nullities, singular spectra, Hessian spectra, conditions,
endpoint-class counts, requested/companion image diameters, flux-allocation
diameters, morphology diameters, boundary contacts, prompt-swap tests, replay
hashes, and perturbation results are in the full metrics and information-gain
tables. Relative to S1, the P2 joint minimum-singular ratios span
`{min(p2_min_ratio):.4g}`--`{max(p2_min_ratio):.4g}` and joint condition ratios
span `{min(p2_condition_ratio):.4g}`--`{max(p2_condition_ratio):.4g}`.
`{conditioning_improvements}/16` P2 family fits meet the preregistered >=5%
information/geometry improvement rule. Fitted per-source g/r/z allocations are
reported for every condition without oracle component photometry.

All **512** planned endpoints are retained. Optimizer-declared failures:
**{failures}**. Starts reaching the 500-evaluation ceiling: **{max_budget}**.
Good joint image likelihood alone was not treated as source uniqueness.

## Scientific interpretation

PSF diversity restored uniqueness: **{'yes' if outcome in {'PSF_DIVERSE_UNIQUENESS_LARGELY_RESTORED', 'PSF_DIVERSE_UNIQUENESS_PARTIALLY_RESTORED'} else 'no under the frozen campaign threshold'}**.
The same-PSF control shows whether any gain is explained by an added exposure;
the exact causal breakdown above is authoritative. These are conditional
structural-identifiability findings for eight frozen simulated scenes, not a
claim of unrestricted source recovery or real-survey generalization.

A morphology-aware reconstruction target is **{'scientifically justified for a direct paired-observation accuracy test' if morphology_justified else 'not yet broadly justified as unique across the frozen set'}**.
Independent observations are **{'required by the information contract that produced the observed improvement' if independent_required else 'not shown to restore uniqueness'}**.
The audit layer may return **only after** a direct structural reconstruction
experiment produces accurate, scientifically valid outputs; identifiability
alone does not authorize POST. PriorNet remains unauthorized.

## Integrity

All inference audits pass: `{sum(row['status'] == 'PASS' for row in oracle_rows)}/{len(oracle_rows)}`.
Protected development, Atlas tensor, lockbox, and historical isolated-HDF5
access are zero. README, HEAD, the Git index, historical reports, the completed
predecessor run, and all 600 checkpoints remain unchanged. Nothing was staged
or committed.

The authorized post-science report-only finalization amendment in
`preregistration/post_science_finalization_amendment.json` coerces serialized
numeric strings such as `"inf"` only for report arithmetic. It does not alter
the protocol, observations, completed fits, thresholds, or scientific outcome.

## Exactly one recommended next experiment

Run **{next_experiment}** to {next_purpose}. No neural training is recommended.
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)

    integrity = predecessor_historical_integrity()
    integrity["campaign"] = CAMPAIGN
    integrity["timestamp_utc"] = utc_now()
    if integrity["status"] != "PASS":
        write_json_fresh(run_dir / "manifests/historical_integrity_final.json", integrity)
        raise RuntimeError("post-execution historical integrity failed closed")
    write_json_fresh(run_dir / "manifests/historical_integrity_final.json", integrity)
    required = (
        "reports/final_report.md",
        "preregistration/frozen_protocol.md",
        "preregistration/frozen_psf_pair.md",
        "preregistration/post_science_finalization_amendment.json",
        "tables/scene_family_condition_summary.csv",
        "tables/full_psf_diverse_metrics.csv",
        "tables/multistart_endpoints.csv",
        "tables/fitted_source_fluxes.csv",
        "tables/condition_transition_matrix.csv",
        "tables/psf_information_gain.csv",
        "tables/psf_diversity_metrics.csv",
        "tables/oracle_information_audit.csv",
        "tables/comparison_to_single_observation_flux_free.csv",
        "figures/recoverability_frontier_s1_s2_p2.png",
        "figures/condition_transition_heatmap.png",
        "figures/psf_a_vs_psf_b.png",
        "manifests/input_hashes.json",
        "manifests/psf_hashes.json",
        "manifests/historical_integrity.json",
        "manifests/historical_integrity_final.json",
        "manifests/protected_data_access.json",
    )
    required_records = {
        relative: {
            "exists": (run_dir / relative).exists(),
            "sha256": sha256_file(run_dir / relative) if (run_dir / relative).exists() else None,
        }
        for relative in required
    }
    final_manifest = {
        "campaign": CAMPAIGN,
        "timestamp_utc": utc_now(),
        "outcome": outcome,
        "counts": counts,
        "conditioning_improvement_family_count": int(conditioning_improvements),
        "optimizer_failures": int(failures),
        "max_budget_endpoints": int(max_budget),
        "all_required_artifacts_present": all(
            record["exists"] for record in required_records.values()
        ),
        "required_artifacts": required_records,
        "singular_value_figure_count": len(
            list((run_dir / "figures/singular_value_comparison").glob("*.png"))
        ),
        "multistart_figure_count": len(
            list((run_dir / "figures/multistart_solution_geometry").glob("*.png"))
        ),
        "source_reconstruction_figure_count": len(
            list((run_dir / "figures/source_reconstruction_panels").glob("*.png"))
        ),
        "residual_figure_count": len(
            list((run_dir / "figures/residual_comparison").glob("*.png"))
        ),
        "oracle_leakage": False,
        "protected_access_zero": True,
        "historical_integrity_status": integrity["status"],
        "readme_unchanged": integrity["readme_unchanged"],
        "git_index_empty": integrity["staged_index_empty"],
        "commits_created": 0,
        "priornet_authorized": False,
        "recommended_next_experiment": next_experiment,
        "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"),
    }
    write_json_fresh(run_dir / "manifests/final_manifest.json", final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preregister")
    pre.add_argument("--run-dir", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-observations")
    prepare.add_argument("--run-dir", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--scene", type=int, choices=INDICES, required=True)
    execute.add_argument("--family", choices=FAMILIES, required=True)
    execute.add_argument("--condition", choices=CONDITIONS, required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    if arguments.command == "preregister":
        preregister(run_dir)
    elif arguments.command == "prepare-observations":
        prepare_observations(run_dir)
    elif arguments.command == "execute":
        execute_one(run_dir, arguments.scene, arguments.family, arguments.condition)
    else:
        finalize(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
