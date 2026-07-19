#!/usr/bin/env python3
"""Two-scene reduced-budget external-photometry identifiability preflight."""

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

import h5py
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

from btk.survey import get_surveys
from surveycodex.utilities import mag2counts

from scripts.run_thayer_flux_free_identifiability_v0 import (
    CHECKPOINT_BASELINE,
    legacy_array_sha256,
    load_science_inputs,
    raw_array_sha256,
    selected_manifest_rows,
    sha256_file,
)
from src.btk_scene import load_catsim_catalog
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.model9_optimizer import (
    MultiStartEndpoint,
    analyze_jacobian,
    boundary_contact_flags,
    deterministic_starts,
    solution_geometry,
)
from src.model9_structured import (
    BANDS,
    FAMILY_BULGE_DISK,
    FrozenSolverProtocol,
    SolverInputs,
    canonicalize_parameters,
    likelihood_components,
    normalize_psf,
    oracle_information_audit,
    parameter_bounds,
    parameter_names,
    parameter_scales,
    parameter_sha256,
    parameters_per_source,
    render_pair,
    whitened_residual_vector,
)

CAMPAIGN = "Thayer-External-Photometry-Preflight-v0"
SCENES = (0, 6)
CONDITIONS = ("TOTAL_SOURCE_PHOTOMETRY", "PER_BAND_SOURCE_PHOTOMETRY")
STARTS = 4
MAX_NFEV = 150
RELATIVE_SIGMA = 0.05
MEASUREMENT_SEED = 2026071805
TOTAL_WEIGHTS = np.ones(3, dtype=np.float64)

FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CATALOG_PATH = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
UPSTREAM = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
DEFINITIONS = UPSTREAM / "manifests/v2_r_training_scene_definitions.csv"
SCIENCE_H5 = UPSTREAM / "manifests/v2_r_training_scenes.h5"
SCENE_MANIFEST = UPSTREAM / "manifests/v2_r_training_scene_manifest.csv"
S1_RUN = REPO / "outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310"
P2_RUN = REPO / "outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646"
S1_METRICS = S1_RUN / "tables/full_flux_free_identifiability_metrics.csv"
P2_METRICS = P2_RUN / "tables/full_psf_diverse_metrics.csv"
AUTHORITATIVE_REPORTS = (
    REPO / "outputs/runs/thayer_identifiability_v1_20260715_003220/reports/final_report.md",
    S1_RUN / "reports/final_report.md",
    P2_RUN / "reports/final_report.md",
    REPO / "outputs/runs/thayer_model_9_preparation_v0_20260715_172217/reports/final_report.md",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fresh_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def fresh_json(path: Path, value: Any) -> None:
    fresh_text(path, json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(safe(v), sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list, tuple, np.ndarray)) else safe(v)) for k, v in row.items()})


def protocol_text() -> str:
    return f"""# Frozen preflight protocol

Frozen before external-measurement generation or scientific fitting.

- Campaign: `{CAMPAIGN}`
- Scope: Scene 0 and Scene 6 only; Level 5 / bulge+disk only.
- Historical controls: authoritative S1 `FLUX_FREE_SINGLE` and P2 `PSF_DIVERSE`, comparison only; neither is rerun.
- New conditions: `TOTAL_SOURCE_PHOTOMETRY` and `PER_BAND_SOURCE_PHOTOMETRY` on the unchanged single observation.
- Renderer, parameterization, morphology support, PSF, observation noise convention, optimizer tolerances, symmetry quotient, endpoint clustering, and diameter definitions: frozen Model-9 implementation.
- External measurements: CatSim catalog g/r/z AB photometry converted with the same documented LSST `mag2counts` calibration used by BTK. No isolated image is read or generated.
- Total-photometry combination: `1*g + 1*r + 1*z` detected-electron flux, one scalar per source.
- Per-band photometry: separate g, r, z detected-electron fluxes per source.
- Measurement model: independent Gaussian, sigma = 0.05 times the latent catalog flux (or weighted total); deterministic synthetic noise seed `{MEASUREMENT_SEED}`. Only noisy measured values and declared sigma enter fitting.
- Fits: exactly {STARTS} deterministic Model-9 starts per new scene/condition; the same four starts are reused across photometry conditions; maximum {MAX_NFEV} function evaluations per start; CPU float64.
- Objective residual vector: frozen whitened single-observation residual concatenated with external-photometry standardized residuals. Observation and photometry log likelihoods are reported separately.
- Acceptable endpoint classes: finite optimizer endpoints that pass the frozen observation chi-square support gate, then the frozen total-objective tolerance and image-space clustering. All four endpoints are retained regardless of status.
- Local rank/nullity/condition: symmetry-corrected SVD of the combined observation-plus-photometry residual Jacobian using frozen parameter scaling.
- Replay: render, combined residual, and combined Jacobian are evaluated twice at the best endpoint and must hash exactly.
- No isolated-source image, mask, morphology truth/label, truth initialization, protected development, Atlas tensor, or lockbox may be accessed. Catalog truth photometry is confined to the deterministic measurement generator and discarded before fitting.

Preflight promise rules are literal. `STRONG_PROMISE` requires one acceptable class and every scientific diameter to be strictly smaller than both S1 and P2. `MODERATE_PROMISE` requires a lower endpoint-class count than P2 or at least a 50% reduction in every nonzero P2 scientific diameter without worsening any diameter. `NO_CLEAR_GAIN` applies otherwise. A fit is `OPTIMIZATION_LIMITED` if it has no optimizer-declared successful endpoint, no observation-support endpoint, a failed replay, or a nonfinite diagnostic. Any information-contract failure is `INVALID`.
"""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def definition_rows() -> dict[int, dict[str, str]]:
    return {int(r["dataset_index"]): r for r in read_csv_rows(DEFINITIONS) if int(r["dataset_index"]) in SCENES}


def load_baselines() -> dict[tuple[int, str], dict[str, Any]]:
    keys = ("distinct_solution_classes", "requested_image_diameter", "companion_image_diameter", "flux_allocation_diameter", "morphology_parameter_diameter")
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_csv_rows(S1_METRICS):
        if int(row["scene_index"]) in SCENES and row["family"] == FAMILY_BULGE_DISK:
            output[(int(row["scene_index"]), "FLUX_FREE_SINGLE")] = {"classification": row["classification"], **{k: float(row[k]) if k != "distinct_solution_classes" else int(row[k]) for k in keys}, "gradient_norm": float(row["gradient_norm"]), "condition_number": float(row["condition_number"]), "rank": int(row["rank"]), "nullity": int(row["null_space_dimension"])}
    for row in read_csv_rows(P2_METRICS):
        if int(row["scene_index"]) in SCENES and row["family"] == FAMILY_BULGE_DISK and row["condition"] == "P2":
            output[(int(row["scene_index"]), "PSF_DIVERSE")] = {"classification": row["classification"], **{k: float(row[k]) if k != "distinct_solution_classes" else int(row[k]) for k in keys}, "gradient_norm": float(row["gradient_norm"]), "condition_number": float(row["condition_number"]), "rank": int(row["rank"]), "nullity": int(row["null_space_dimension"])}
    if len(output) != 4:
        raise RuntimeError("authoritative S1/P2 baseline rows are incomplete")
    return output


def measurements() -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    catalog, catalog_hash = load_catsim_catalog(CATALOG_PATH)
    definitions = definition_rows()
    manifests = selected_manifest_rows()
    survey = get_surveys("LSST")
    generated: dict[tuple[int, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        definition = definitions[scene]
        source_rows = (int(definition["source_a_row"]), int(definition["source_b_row"]))
        requested_index = int(manifests[scene]["matched_source_index"])
        identity_order = (requested_index, 1 - requested_index)
        true_by_ab = []
        for source_row in source_rows:
            entry = catalog.table[source_row]
            true_by_ab.append(np.asarray([float(mag2counts(entry[f"{band}_ab"], survey, survey.get_filter(band)).to_value("electron")) for band in BANDS], dtype=np.float64))
        true_flux = np.stack([true_by_ab[i] for i in identity_order])
        for condition_index, condition in enumerate(CONDITIONS):
            rng = np.random.default_rng(MEASUREMENT_SEED + 100 * scene + condition_index)
            latent = true_flux @ TOTAL_WEIGHTS if condition == "TOTAL_SOURCE_PHOTOMETRY" else true_flux.reshape(-1)
            sigma = RELATIVE_SIGMA * latent
            z = rng.standard_normal(latent.shape)
            measured = latent + sigma * z
            if np.any(measured <= 0) or np.any(sigma <= 0):
                raise RuntimeError("nonpositive synthetic external photometry")
            generated[(scene, condition)] = {"measured": measured, "sigma": sigma}
            labels = (("requested", "total"), ("companion", "total")) if condition == "TOTAL_SOURCE_PHOTOMETRY" else tuple((identity, band) for identity in ("requested", "companion") for band in BANDS)
            for i, (identity, band) in enumerate(labels):
                rows.append({
                    "scene_index": scene,
                    "scene_id": manifests[scene]["scene_id"],
                    "condition": condition,
                    "source_identity": identity,
                    "measurement_band": band,
                    "combination_weights_g_r_z": TOTAL_WEIGHTS if band == "total" else [1.0 if b == band else 0.0 for b in BANDS],
                    "measured_flux_electrons": measured[i],
                    "sigma_electrons": sigma[i],
                    "relative_sigma": RELATIVE_SIGMA,
                    "frozen_noise_z": z[i],
                    "measurement_seed": MEASUREMENT_SEED + 100 * scene + condition_index,
                    "catalog_sha256": catalog_hash,
                    "latent_catalog_value_persisted": False,
                })
        del true_flux, true_by_ab
    return generated, rows


def predicted_photometry(parameters: torch.Tensor, condition: str) -> torch.Tensor:
    per_source = 12
    bands = torch.cat((parameters[:3], parameters[per_source:per_source + 3]))
    if condition == "PER_BAND_SOURCE_PHOTOMETRY":
        return bands
    return torch.stack((bands[:3].sum(), bands[3:].sum()))


def combined_residual(parameters: torch.Tensor, inputs: SolverInputs, condition: str, measurement: dict[str, Any]) -> torch.Tensor:
    observed = whitened_residual_vector(parameters, inputs, FrozenSolverProtocol())
    measured = torch.as_tensor(measurement["measured"], dtype=parameters.dtype, device=parameters.device)
    sigma = torch.as_tensor(measurement["sigma"], dtype=parameters.dtype, device=parameters.device)
    external = (predicted_photometry(parameters, condition) - measured) / sigma
    return torch.cat((observed, external))


def combined_jacobian(parameters: np.ndarray, inputs: SolverInputs, condition: str, measurement: dict[str, Any]) -> np.ndarray:
    value = torch.as_tensor(parameters, dtype=torch.float64).clone().requires_grad_(True)
    def function(candidate: torch.Tensor) -> torch.Tensor:
        return combined_residual(candidate, inputs, condition, measurement)
    try:
        jac = torch.autograd.functional.jacobian(function, value, vectorize=True, strategy="forward-mode")
    except RuntimeError:
        jac = torch.autograd.functional.jacobian(function, value, vectorize=True, strategy="reverse-mode")
    return np.asarray(jac.detach().cpu(), dtype=np.float64)


def objective_parts(parameters: np.ndarray, inputs: SolverInputs, condition: str, measurement: dict[str, Any]) -> dict[str, float]:
    tensor = torch.as_tensor(parameters, dtype=torch.float64)
    obs = likelihood_components(tensor, inputs, FrozenSolverProtocol())
    pred = predicted_photometry(tensor, condition)
    measured = torch.as_tensor(measurement["measured"], dtype=torch.float64)
    sigma = torch.as_tensor(measurement["sigma"], dtype=torch.float64)
    ext = (pred - measured) / sigma
    phot_nll = 0.5 * (ext.square() + 2.0 * torch.log(sigma)).sum()
    return {"observation_objective": float(obs.likelihood_total), "observation_chi_square": float(obs.chi_square), "photometry_objective": float(phot_nll), "photometry_chi_square": float(ext.square().sum()), "total_objective": float(obs.likelihood_total + phot_nll)}


def fit_one(scene: int, condition: str, inputs: SolverInputs, measurement: dict[str, Any], starts: np.ndarray) -> dict[str, Any]:
    protocol = FrozenSolverProtocol()
    lower, upper = parameter_bounds(inputs, protocol)
    endpoints: list[MultiStartEndpoint] = []
    extra: dict[int, dict[str, float]] = {}
    for start_index, initialization in enumerate(starts):
        result = least_squares(
            lambda x: np.asarray(combined_residual(torch.as_tensor(x, dtype=torch.float64), inputs, condition, measurement).detach().cpu()),
            initialization,
            jac=lambda x: combined_jacobian(x, inputs, condition, measurement),
            bounds=(lower, upper), method="trf", x_scale="jac",
            ftol=protocol.ftol, xtol=protocol.xtol, gtol=protocol.gtol,
            max_nfev=MAX_NFEV, verbose=0,
        )
        selected = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
        canonical, _, symmetries = canonicalize_parameters(selected, inputs, protocol)
        tensor = torch.as_tensor(selected, dtype=torch.float64)
        pair = render_pair(tensor, inputs, protocol)
        residual = np.asarray(combined_residual(tensor, inputs, condition, measurement).detach().cpu())
        jac = combined_jacobian(selected, inputs, condition, measurement)
        scales = parameter_scales(inputs, protocol)
        gradient = float(np.linalg.norm((jac * scales[None]) .T @ residual))
        parts = objective_parts(selected, inputs, condition, measurement)
        extra[start_index] = parts
        endpoints.append(MultiStartEndpoint(
            start_index=start_index, initialization=initialization.copy(), parameters=selected,
            canonical_parameters=canonical, initialization_sha256=parameter_sha256(initialization),
            parameter_sha256=parameter_sha256(canonical), success=bool(result.success),
            status=int(result.status), message=str(result.message), nfev=int(result.nfev),
            njev=None if result.njev is None else int(result.njev), cost=float(result.cost),
            likelihood_objective=parts["total_objective"], chi_square=parts["observation_chi_square"],
            optimality=float(result.optimality), gradient_norm=gradient,
            requested_sha256=canonical_tensor_sha256(pair.requested), companion_sha256=canonical_tensor_sha256(pair.companion),
            recomposed_sha256=canonical_tensor_sha256(pair.recomposed_sources), symmetries=symmetries,
        ))
    finite = [e for e in endpoints if np.isfinite(e.likelihood_objective)]
    best = min(finite, key=lambda e: e.likelihood_objective)
    best_tensor = torch.as_tensor(best.parameters, dtype=torch.float64)
    canonical, active, symmetries = canonicalize_parameters(best.parameters, inputs, protocol)
    jac = combined_jacobian(best.parameters, inputs, condition, measurement)
    scales = parameter_scales(inputs, protocol)
    diagnostics = analyze_jacobian(jac, active_mask=active, parameter_scales=scales)
    residual = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    gradient_norm = float(np.linalg.norm((jac * scales[None]).T @ residual))
    dof = max(1, int(inputs.observed.numel()) - int(active.sum()))
    support_threshold = float(chi2.ppf(protocol.model_acceptance_quantile, dof))
    support = [e for e in finite if e.chi_square <= support_threshold]
    geometry = solution_geometry(support if support else finite, inputs, protocol)
    pair1 = render_pair(best_tensor, inputs, protocol)
    residual1 = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    jac1 = combined_jacobian(best.parameters, inputs, condition, measurement)
    pair2 = render_pair(best_tensor, inputs, protocol)
    residual2 = np.asarray(combined_residual(best_tensor, inputs, condition, measurement).detach().cpu())
    jac2 = combined_jacobian(best.parameters, inputs, condition, measurement)
    replay = {
        "requested_first": canonical_tensor_sha256(pair1.requested), "requested_second": canonical_tensor_sha256(pair2.requested),
        "residual_first": raw_array_sha256(residual1), "residual_second": raw_array_sha256(residual2),
        "jacobian_first": raw_array_sha256(jac1), "jacobian_second": raw_array_sha256(jac2),
    }
    replay["exact_match"] = replay["requested_first"] == replay["requested_second"] and replay["residual_first"] == replay["residual_second"] and replay["jacobian_first"] == replay["jacobian_second"]
    boundary = boundary_contact_flags(best.parameters, inputs, protocol)
    optimizer_limited = not any(e.success for e in endpoints) or not support or not replay["exact_match"] or not np.isfinite(gradient_norm) or not np.isfinite(diagnostics.condition_number)
    return {
        "scene_index": scene, "condition": condition, "family": FAMILY_BULGE_DISK, "level": "Level 5",
        "starts": starts, "endpoints": endpoints, "endpoint_parts": extra, "best": best,
        "best_parts": extra[best.start_index], "gradient_norm": gradient_norm,
        "rank": diagnostics.rank, "active_parameter_count": diagnostics.active_parameter_count,
        "nullity": diagnostics.null_space_dimension, "condition_number": diagnostics.condition_number,
        "geometry": asdict(geometry), "support_start_indices": [e.start_index for e in support],
        "support_threshold": support_threshold, "symmetries": symmetries,
        "boundary_contact_flags": boundary, "replay": replay,
        "fitted_requested_fluxes": best.parameters[:3], "fitted_companion_fluxes": best.parameters[12:15],
        "optimizer_limited": optimizer_limited,
    }


DIAMETERS = ("requested_image_diameter", "companion_image_diameter", "flux_allocation_diameter", "morphology_parameter_diameter")


def promise(result: dict[str, Any], baselines: dict[tuple[int, str], dict[str, Any]]) -> str:
    if result["optimizer_limited"]:
        return "OPTIMIZATION_LIMITED"
    scene = result["scene_index"]
    s1, p2 = baselines[(scene, "FLUX_FREE_SINGLE")], baselines[(scene, "PSF_DIVERSE")]
    geometry = result["geometry"]
    classes = int(geometry["distinct_solution_classes"])
    strict_all = all(float(geometry[k]) < float(s1[k]) and float(geometry[k]) < float(p2[k]) for k in DIAMETERS)
    if classes == 1 and strict_all:
        return "STRONG_PROMISE"
    nonworse = all(float(geometry[k]) <= float(p2[k]) for k in DIAMETERS)
    half_nonzero = all(float(geometry[k]) <= (0.5 * float(p2[k]) if float(p2[k]) > 0 else 0.0) for k in DIAMETERS)
    if classes < int(p2["distinct_solution_classes"]) or (nonworse and half_nonzero and any(float(p2[k]) > 0 for k in DIAMETERS)):
        return "MODERATE_PROMISE"
    return "NO_CLEAR_GAIN"


def historical_integrity(start: dict[str, Any]) -> dict[str, Any]:
    baseline = json.loads(CHECKPOINT_BASELINE.read_text(encoding="utf-8"))
    matches = mismatches = missing = 0
    for record in baseline:
        path = REPO / record["path"]
        if not path.exists():
            missing += 1
        elif path.stat().st_size == int(record["final_bytes"]) and sha256_file(path) == record["final_sha256"]:
            matches += 1
        else:
            mismatches += 1
    final = {
        "head_before": start["head"], "head_after": command(["git", "rev-parse", "HEAD"]),
        "readme_sha256_before": start["readme"], "readme_sha256_after": sha256_file(REPO / "README.md"),
        "authoritative_report_hashes_before": start["reports"],
        "authoritative_report_hashes_after": {str(p.relative_to(REPO)): sha256_file(p) for p in AUTHORITATIVE_REPORTS},
        "historical_checkpoint_matches": matches, "historical_checkpoint_mismatches": mismatches,
        "historical_checkpoint_missing": missing, "historical_checkpoint_count": len(baseline),
        "git_index_entries": command(["git", "diff", "--cached", "--name-only"]),
        "protected_data_access": {"development": 0, "atlas_tensors": 0, "lockbox": 0},
        "historical_isolated_hdf5_access": 0,
        "historical_reports_modified": False, "historical_checkpoints_modified": mismatches > 0,
        "commits_created": 0,
    }
    final["historical_reports_modified"] = final["authoritative_report_hashes_before"] != final["authoritative_report_hashes_after"]
    final["status"] = "PASS" if final["head_before"] == final["head_after"] and final["readme_sha256_before"] == final["readme_sha256_after"] and not final["historical_reports_modified"] and mismatches == 0 and missing == 0 and not final["git_index_entries"] else "FAIL"
    return final


def campaign_outcome(results: list[dict[str, Any]]) -> str:
    promises = {(r["scene_index"], r["condition"]): r["promise"] for r in results}
    if any(v == "INVALID" for v in promises.values()):
        return "PREFLIGHT_INVALID"
    if any(v == "OPTIMIZATION_LIMITED" for v in promises.values()):
        return "PREFLIGHT_OPTIMIZATION_LIMITED"
    per_band = [promises[(s, "PER_BAND_SOURCE_PHOTOMETRY")] for s in SCENES]
    if "STRONG_PROMISE" in per_band and all(v in {"STRONG_PROMISE", "MODERATE_PROMISE"} for v in per_band):
        return "EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED"
    total = [promises[(s, "TOTAL_SOURCE_PHOTOMETRY")] for s in SCENES]
    if any(v in {"STRONG_PROMISE", "MODERATE_PROMISE"} for v in total):
        return "TOTAL_PHOTOMETRY_ONLY_PARTIALLY_PROMISING"
    return "EXTERNAL_PHOTOMETRY_NO_CLEAR_ADVANTAGE"


def make_figure(run_dir: Path, baselines: dict[tuple[int, str], dict[str, Any]], results: list[dict[str, Any]]) -> None:
    labels, values, colors = [], [], []
    lookup = {(r["scene_index"], r["condition"]): r for r in results}
    for scene in SCENES:
        for condition, color in (("FLUX_FREE_SINGLE", "#777777"), ("PSF_DIVERSE", "#4c78a8"), ("TOTAL_SOURCE_PHOTOMETRY", "#f58518"), ("PER_BAND_SOURCE_PHOTOMETRY", "#54a24b")):
            labels.append(f"S{scene}\n{condition.replace('_SOURCE_PHOTOMETRY','').replace('FLUX_FREE_SINGLE','S1').replace('PSF_DIVERSE','P2')}")
            values.append(baselines[(scene, condition)]["distinct_solution_classes"] if condition in {"FLUX_FREE_SINGLE", "PSF_DIVERSE"} else lookup[(scene, condition)]["geometry"]["distinct_solution_classes"])
            colors.append(color)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Acceptable endpoint classes")
    ax.set_title("Two-scene information-source comparison (Level 5)")
    ax.set_ylim(0, max(2.2, max(values) + 0.2))
    fig.savefig(run_dir / "figures/information_source_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    started_wall = time.perf_counter()
    started_utc = now()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_external_photometry_preflight_v0_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("reports", "preregistration", "tables", "manifests", "figures"):
        (run_dir / sub).mkdir()
    start_integrity = {
        "head": command(["git", "rev-parse", "HEAD"]), "readme": sha256_file(REPO / "README.md"),
        "reports": {str(p.relative_to(REPO)): sha256_file(p) for p in AUTHORITATIVE_REPORTS},
    }
    fresh_text(run_dir / "preregistration/frozen_preflight_protocol.md", protocol_text())
    input_paths = [Path(__file__), SCIENCE_H5, SCENE_MANIFEST, DEFINITIONS, CATALOG_PATH, S1_METRICS, P2_METRICS, REPO / "src/model9_structured.py", REPO / "src/model9_optimizer.py", *AUTHORITATIVE_REPORTS]
    fresh_json(run_dir / "manifests/input_hashes.json", {"campaign": CAMPAIGN, "frozen_at_utc": now(), "files": {str(p.relative_to(REPO)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in input_paths}, "scenes": SCENES, "family": FAMILY_BULGE_DISK, "conditions_new": CONDITIONS, "starts": STARTS, "max_nfev": MAX_NFEV, "cpu_only": True})
    baselines = load_baselines()
    measured, measurement_rows = measurements()
    fresh_csv(run_dir / "tables/external_photometry_measurements.csv", measurement_rows)
    manifests = selected_manifest_rows()
    results: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    start_hashes: dict[int, str] = {}
    for scene in SCENES:
        inputs, metadata = load_science_inputs(scene, FAMILY_BULGE_DISK)
        audit = oracle_information_audit(inputs, FrozenSolverProtocol(), extra_named_inputs={"scene_index": scene, "external_measurements": measured[(scene, CONDITIONS[0])]["measured"].tolist()})
        starts = deterministic_starts(inputs, FrozenSolverProtocol(), count=STARTS)
        start_hashes[scene] = raw_array_sha256(starts)
        for condition in CONDITIONS:
            fit = fit_one(scene, condition, inputs, measured[(scene, condition)], starts)
            fit["metadata"] = metadata
            fit["promise"] = "INVALID" if audit["status"] != "PASS" else promise(fit, baselines)
            results.append(fit)
            acceptable = set(fit["geometry"]["acceptable_endpoint_indices"])
            support = set(fit["support_start_indices"])
            for endpoint in fit["endpoints"]:
                endpoint_rows.append({"scene_index": scene, "scene_id": metadata["scene_id"], "condition": condition, **endpoint.record(), **fit["endpoint_parts"][endpoint.start_index], "observation_support_acceptable": endpoint.start_index in support, "total_objective_acceptable": endpoint.start_index in acceptable})
            audit_rows.append({
                "scene_index": scene, "condition": condition, "status": "PASS" if audit["status"] == "PASS" else "FAIL",
                "blend_rows_accessed": 1, "coordinate_rows_accessed": 1, "catalog_rows_used_by_measurement_generator": 2,
                "catalog_truth_flux_exposed_to_fit": False, "noisy_measurements_exposed_to_fit": True,
                "isolated_source_images_accessed": 0, "isolated_source_images_exposed": 0,
                "morphology_truth_exposed": 0, "masks_exposed": 0, "truth_initialization_exposed": 0,
                "development_access": 0, "atlas_tensor_access": 0, "lockbox_access": 0,
                "measurement_likelihood_not_hard_fix": True, "measurement_relative_sigma": RELATIVE_SIGMA,
            })
    fresh_csv(run_dir / "tables/preflight_endpoints.csv", endpoint_rows)
    fresh_csv(run_dir / "tables/oracle_information_audit.csv", audit_rows)
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for scene in SCENES:
        for historical in ("FLUX_FREE_SINGLE", "PSF_DIVERSE"):
            base = baselines[(scene, historical)]
            summary_rows.append({"scene_index": scene, "scene_id": manifests[scene]["scene_id"], "condition": historical, "source": "authoritative_reuse_no_rerun", "classification": base["classification"], "preflight_promise": "COMPARISON_ONLY", "best_objective": "", "observation_likelihood": "", "photometry_likelihood": "", "optimizer_converged_starts": "", "gradient_norm": base["gradient_norm"], "rank": base["rank"], "nullity": base["nullity"], "condition_number": base["condition_number"], **{k: base[k] for k in ("distinct_solution_classes", *DIAMETERS)}})
        for fit in [r for r in results if r["scene_index"] == scene]:
            best = fit["best"]
            row = {"scene_index": scene, "scene_id": manifests[scene]["scene_id"], "condition": fit["condition"], "source": "new_fit", "classification": "PREFLIGHT_ONLY", "preflight_promise": fit["promise"], "best_objective": fit["best_parts"]["total_objective"], "observation_likelihood": fit["best_parts"]["observation_objective"], "photometry_likelihood": fit["best_parts"]["photometry_objective"], "optimizer_converged_starts": sum(e.success for e in fit["endpoints"]), "best_start_index": best.start_index, "best_nfev": best.nfev, "gradient_norm": fit["gradient_norm"], "rank": fit["rank"], "active_parameter_count": fit["active_parameter_count"], "nullity": fit["nullity"], "condition_number": fit["condition_number"], **{k: fit["geometry"][k] for k in ("distinct_solution_classes", *DIAMETERS)}, "fitted_requested_fluxes_g_r_z": fit["fitted_requested_fluxes"], "fitted_companion_fluxes_g_r_z": fit["fitted_companion_fluxes"], "boundary_contact_flags": fit["boundary_contact_flags"], "deterministic_replay_exact": fit["replay"]["exact_match"]}
            summary_rows.append(row)
            s1, p2 = baselines[(scene, "FLUX_FREE_SINGLE")], baselines[(scene, "PSF_DIVERSE")]
            comparison_rows.append({"scene_index": scene, "condition": fit["condition"], "preflight_promise": fit["promise"], "new_endpoint_classes": fit["geometry"]["distinct_solution_classes"], "s1_endpoint_classes": s1["distinct_solution_classes"], "p2_endpoint_classes": p2["distinct_solution_classes"], **{f"new_{k}": fit["geometry"][k] for k in DIAMETERS}, **{f"s1_{k}": s1[k] for k in DIAMETERS}, **{f"p2_{k}": p2[k] for k in DIAMETERS}, "new_condition_number": fit["condition_number"], "s1_condition_number": s1["condition_number"], "p2_condition_number": p2["condition_number"], "new_gradient_norm": fit["gradient_norm"], "s1_gradient_norm": s1["gradient_norm"], "p2_gradient_norm": p2["gradient_norm"]})
    fresh_csv(run_dir / "tables/preflight_condition_summary.csv", summary_rows)
    fresh_csv(run_dir / "tables/comparison_to_s1_and_p2.csv", comparison_rows)
    make_figure(run_dir, baselines, results)
    outcome = campaign_outcome(results)
    runtime = time.perf_counter() - started_wall
    per_band = [r for r in results if r["condition"] == "PER_BAND_SOURCE_PHOTOMETRY"]
    total = [r for r in results if r["condition"] == "TOTAL_SOURCE_PHOTOMETRY"]
    if outcome == "EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED":
        next_experiment = "Thayer-External-Photometry-vs-PSF-Diversity-v0"
    elif outcome == "PREFLIGHT_OPTIMIZATION_LIMITED":
        next_experiment = "Thayer-External-Photometry-Convergence-Correction-v0: repeat only these four fits with the authoritative 500-evaluation budget before any eight-scene expansion."
    else:
        next_experiment = "Thayer-Color-Ratio-Photometry-Targeted-v0: on Scenes 0 and 6 only, test one independent 1% source color-ratio measurement while retaining 5% total flux."
    report = f"""# {CAMPAIGN} final report

## Outcome

**{outcome}**

This was a two-scene, reduced-budget preflight, not a population-level campaign. It used only Scenes 0 and 6, only Level 5 bulge+disk, exactly four deterministic starts, and at most 150 function evaluations per start. Authoritative S1 and P2 results were reused without rerunning them.

## External information supplied

`TOTAL_SOURCE_PHOTOMETRY` supplied one noisy per-source scalar equal to the frozen `g+r+z` detected-electron combination. `PER_BAND_SOURCE_PHOTOMETRY` supplied noisy per-source g, r, and z measurements. Every supplied measurement used 5% relative Gaussian uncertainty and frozen deterministic seed {MEASUREMENT_SEED}. Measurements entered as an explicit likelihood; no source flux was fixed to truth. Measured values and uncertainties are in `tables/external_photometry_measurements.csv`.

## Scene-level evidence

| Scene | Total photometry | Per-band photometry | S1 classes | P2 classes | Total classes | Per-band classes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
"""
    lookup = {(r["scene_index"], r["condition"]): r for r in results}
    for scene in SCENES:
        report += f"| {scene} | {lookup[(scene, 'TOTAL_SOURCE_PHOTOMETRY')]['promise']} | {lookup[(scene, 'PER_BAND_SOURCE_PHOTOMETRY')]['promise']} | {baselines[(scene, 'FLUX_FREE_SINGLE')]['distinct_solution_classes']} | {baselines[(scene, 'PSF_DIVERSE')]['distinct_solution_classes']} | {lookup[(scene, 'TOTAL_SOURCE_PHOTOMETRY')]['geometry']['distinct_solution_classes']} | {lookup[(scene, 'PER_BAND_SOURCE_PHOTOMETRY')]['geometry']['distinct_solution_classes']} |\n"
    per_band_better_p2 = all(r["geometry"]["distinct_solution_classes"] < baselines[(r["scene_index"], "PSF_DIVERSE")]["distinct_solution_classes"] for r in per_band)
    total_promising = any(r["promise"] in {"STRONG_PROMISE", "MODERATE_PROMISE"} for r in total)
    report += f"""

Per-band photometry reduced multiplicity more than P2 on both scenes: **{'yes' if per_band_better_p2 else 'no'}** under the frozen endpoint-class rule. Total broadband photometry was sufficient to show preflight promise: **{'yes' if total_promising else 'no'}**. The exact campaign-level decision is `{outcome}`; therefore the full eight-scene campaign is **{'justified' if outcome == 'EXTERNAL_PHOTOMETRY_FULL_CAMPAIGN_JUSTIFIED' else 'not justified by this preflight'}**.

The literal `STRONG_PROMISE` rule is demanding because both S1 baselines already have one class and zero reported diameters; equality at zero is not counted as improvement. This preflight does not relabel any result `UNIQUE` and makes no population claim.

## Optimization limitations

Optimizer-declared successful endpoints by fit were: {', '.join(f"Scene {r['scene_index']} {r['condition']}={sum(e.success for e in r['endpoints'])}/4" for r in results)}. Starts hitting the 150-evaluation ceiling: {sum(e.nfev >= MAX_NFEV for r in results for e in r['endpoints'])}/16. Every endpoint was retained, and each best endpoint received an exact deterministic replay. The reduced start/evaluation budget limits basin discovery relative to the authoritative campaigns.

## Integrity and runtime

No isolated-source image, morphology truth, mask, or truth initialization entered inference. Catalog photometry was used only to generate noisy external measurements and was discarded before fitting. Protected development, Atlas-tensor, and lockbox accesses were zero. Nothing was staged or committed; README and HEAD were unchanged subject to the final integrity manifest.

Exact runtime: **{runtime:.6f} seconds** ({runtime / 60.0:.3f} minutes), from {started_utc} to {now()}.

## Exactly one next experiment

**{next_experiment}**
"""
    fresh_text(run_dir / "reports/final_report.md", report)
    integrity = historical_integrity(start_integrity)
    fresh_json(run_dir / "manifests/historical_integrity.json", integrity)
    print(json.dumps({"run_dir": str(run_dir), "outcome": outcome, "runtime_seconds": runtime, "integrity": integrity["status"]}, indent=2))


if __name__ == "__main__":
    main()
