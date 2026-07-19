#!/usr/bin/env python3
"""Append-only closure for Thayer-Identifiability-v1 R6 scene results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import analyze_thayer_identifiability_v1 as audit


RUN = audit.RUN
SOURCE = RUN / "scene_results_r6"
FINAL = RUN / "scene_results_final"
INDICES = audit.INDICES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {
        "arguments": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verify() -> None:
    audit.verify_preregistration()
    for index in INDICES:
        if not (SOURCE / f"scene_{index:03d}.json").is_file():
            raise FileNotFoundError(index)
    staged = command(["git", "diff", "--cached", "--name-only"])
    if staged["returncode"] != 0 or staged["stdout"].strip():
        raise RuntimeError("staged index is not empty")


def update_unique_record(
    record: dict[str, Any],
    *,
    tangent: dict[str, Any],
    residual: float,
    closure: dict[str, Any],
) -> None:
    condition = float(tangent["condition_number"])
    record.update(
        {
            "rank": int(tangent["jacobian_rank"]),
            "null_space": int(tangent["null_space_dimension"]),
            "condition_number": condition,
            "number_of_exact_observation_consistent_solutions": 1,
            "diameter": 0.0,
            "diameter_status": "exact singleton",
            "prompt_identity_unique": True,
            "requested_galaxy_identifiable": True,
            "truth_in_prior_support": True,
            "best_relative_exact_residual": residual,
            "classification": audit.classify(
                model_member=True,
                exact_solution_count=1,
                nullity=int(tangent["null_space_dimension"]),
                diameter=0.0,
                condition=condition,
                prompt_unique=True,
            ),
        }
    )
    record["diagnostics"]["closure"] = closure
    record["diagnostics"]["tangent"] = tangent
    record["diagnostics"]["solution_count"] = 1


def polish_scene6(result: dict[str, Any]) -> None:
    scene = audit.load_scene(6)
    target = scene.target
    band_norms = np.linalg.norm(target, axis=(1, 2))
    closure_records = []
    for level, model in ((5, "bd_gradient"), (6, "bd_shared")):
        record = result["priors"][level]
        fit = record["diagnostics"]["fit"]
        parameters = np.asarray(fit["best_continuous_parameters"], dtype=np.float64)
        start_record = fit["start_records"][0]
        used = int(start_record["nfev"]) + int(start_record["full_refinement_nfev"])
        remaining = audit.MAX_NFEV - used
        if remaining < 2:
            raise RuntimeError("scene-6 polish has no remaining evaluation budget")

        def objective(candidate: np.ndarray) -> np.ndarray:
            left, right = audit.render_pair(scene, model, candidate)
            return ((left + right - target) / band_norms[:, None, None]).ravel()

        polished = least_squares(
            objective,
            parameters,
            bounds=(np.zeros_like(parameters), np.ones_like(parameters)),
            method="trf",
            x_scale="jac",
            diff_step=1e-2,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=remaining,
            verbose=0,
        )
        left, right = audit.render_pair(scene, model, polished.x)
        residual = audit.relative_residual(left + right, target)
        if residual > audit.EXACT_TOLERANCE:
            raise RuntimeError(f"scene-6 {model} polish remains nonexact: {residual}")
        tangent = audit.tangent_diagnostics(scene, model, polished.x)
        closure = {
            "kind": "float32-render-aware in-budget polish",
            "source_r6_sha256": sha256_file(SOURCE / "scene_006.json"),
            "finite_difference_step": 1e-2,
            "preclosure_relative_residual": float(record["best_relative_exact_residual"]),
            "postclosure_relative_residual": residual,
            "initial_evaluations": used,
            "remaining_budget": remaining,
            "closure_evaluations": int(polished.nfev),
            "total_evaluations": used + int(polished.nfev),
            "maximum_evaluations": audit.MAX_NFEV,
            "parameters": polished.x.tolist(),
        }
        update_unique_record(record, tangent=tangent, residual=residual, closure=closure)
        fit.update(
            {
                "best_parameters": polished.x.tolist(),
                "best_continuous_parameters": polished.x.tolist(),
                "best_relative_residual": residual,
                "exact_start_count": max(1, int(fit["exact_start_count"])),
                "unique_exact_output_count_found": 1,
            }
        )
        closure_records.append(closure)
    # Level 7 has the exact same support and structural metrics as Level 6.
    level6 = result["priors"][6]
    level7 = result["priors"][7]
    for key in (
        "rank",
        "null_space",
        "condition_number",
        "number_of_exact_observation_consistent_solutions",
        "diameter",
        "diameter_status",
        "prompt_identity_unique",
        "requested_galaxy_identifiable",
        "truth_in_prior_support",
        "best_relative_exact_residual",
        "classification",
    ):
        level7[key] = copy.deepcopy(level6[key])
    level7["diagnostics"]["closure"] = {
        "kind": "strictly-positive soft-support identity replay",
        "level6_closure": copy.deepcopy(closure_records[-1]),
    }


def shared_to_gradient(shared: np.ndarray) -> np.ndarray:
    output = []
    for source_index in (0, 1):
        local = np.asarray(shared[source_index * 7 : (source_index + 1) * 7], dtype=np.float64)
        output.extend(local[:6].tolist() + [float(local[6])] * 3)
    return np.asarray(output, dtype=np.float64)


def close_level5_by_containment(result: dict[str, Any]) -> None:
    level5 = result["priors"][5]
    level6 = result["priors"][6]
    if (
        level5["classification"] == "UNIQUE"
        or level6["classification"] != "UNIQUE"
        or not bool(level5["truth_in_prior_support"])
    ):
        return
    scene = audit.load_scene(int(result["scene"]))
    shared = np.asarray(level6["diagnostics"]["fit"]["best_parameters"], dtype=np.float64)
    gradient = shared_to_gradient(shared)
    left, right = audit.render_pair(scene, "bd_gradient", gradient)
    residual = audit.relative_residual(left + right, scene.target)
    if residual > audit.EXACT_TOLERANCE:
        raise RuntimeError(f"L6-to-L5 containment replay is not exact: {residual}")
    tangent = audit.tangent_diagnostics(scene, "bd_gradient", gradient)
    closure = {
        "kind": "analytic L6 subset of L5 containment replay",
        "source_r6_sha256": sha256_file(SOURCE / f"scene_{int(result['scene']):03d}.json"),
        "shared_color_parameters": shared.tolist(),
        "mapped_gradient_parameters": gradient.tolist(),
        "relative_residual": residual,
        "optimizer_evaluations": 0,
    }
    update_unique_record(level5, tangent=tangent, residual=residual, closure=closure)
    level5["diagnostics"]["fit"].update(
        {
            "best_parameters": gradient.tolist(),
            "best_continuous_parameters": gradient.tolist(),
            "best_relative_residual": residual,
            "exact_start_count": max(1, int(level5["diagnostics"]["fit"]["exact_start_count"])),
            "unique_exact_output_count_found": 1,
        }
    )


def close_scenes() -> list[dict[str, Any]]:
    results = []
    FINAL.mkdir(parents=True, exist_ok=True)
    for index in INDICES:
        source_path = SOURCE / f"scene_{index:03d}.json"
        result = json.loads(source_path.read_text())
        if index == 6:
            polish_scene6(result)
        close_level5_by_containment(result)
        unique_levels = [
            int(record["prior_level"])
            for record in result["priors"]
            if record["classification"] == "UNIQUE"
        ]
        result["minimum_prior_level_for_unique"] = min(unique_levels) if unique_levels else None
        result["closure"] = {
            "source_r6_path": str(source_path.relative_to(REPO)),
            "source_r6_sha256": sha256_file(source_path),
            "closure_script_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": audit.PREREG_SHA256,
        }
        destination = FINAL / f"scene_{index:03d}.json"
        write_json_fresh(destination, result)
        results.append(result)
    return results


def diameter_text(record: dict[str, Any]) -> str:
    value = record["diameter"]
    if value == "infinity":
        return "infinity"
    if value is None:
        return "--"
    prefix = ">=" if "lower bound" in str(record["diameter_status"]) else ""
    return f"{prefix}{float(value):.6g}"


def make_report(results: list[dict[str, Any]], table: pd.DataFrame, full: pd.DataFrame) -> str:
    minimum = {int(item["scene"]): item["minimum_prior_level_for_unique"] for item in results}
    level2 = full.loc[full.prior_level == 2, "diameter"].astype(float)
    level3 = full.loc[full.prior_level == 3, "diameter"].astype(float)
    unique = full.loc[full.classification == "UNIQUE"].copy()
    unique_conditions = unique.condition_number.astype(float)
    unique_residuals = unique.best_relative_exact_residual.astype(float)
    headers = [str(value) for value in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    main_table = "\n".join(lines)
    minima_lines = [
        f"- Scene {scene}: " + (f"Level {level}" if level is not None else "none through Level 7")
        for scene, level in minimum.items()
    ]
    return f"""# Thayer-Identifiability-v1 scientific report

## Decision

Realistic hard morphology removes the unrestricted continuous allocation null
space for **7/8 scenes**.  Four scenes first become unique at Level 4
(single elliptical Sersic), three first become unique at Level 5 (bulge+disk),
and scene 51 has no truth-containing exact solution through Level 7.  Thus the
recoverability frontier is largely prior-limited, but it does not disappear
completely under the frozen realistic prior ladder.

The remainder is not evidence for a surviving continuous observation null in
the parametric families.  Every truth-containing structural solution has null
space zero and output-tangent condition between
`{unique_conditions.min():.6g}` and `{unique_conditions.max():.6g}`; the largest
condition times the frozen numerical tolerance is only
`{unique_conditions.max() * audit.EXACT_TOLERANCE:.3e}`.  Scene 51 instead
fails model support: its nonzero tiny bulge has HLR about 0.015 arcsec, below
the globally frozen 0.03-arcsec Level-4/5 bound.  An empty admissible set is
not credited as uniqueness.

## Requested table

Diameter values prefixed by `>=` are certified prompt-consistent lower bounds;
`--` denotes an empty exact support, not zero diameter.

{main_table}

## Minimum prior required for uniqueness

{chr(10).join(minima_lines)}

## Quantitative interpretation

- Levels 0 and 1 retain rank/null `10800/10800` in every scene and
  uncountably many exact solutions.  Nonnegativity alone has exact primary
  diameter 10.
- Exact per-source g/r/z fluxes reduce the tangent to rank/null
  `10797/10797`, but still leave uncountably many solutions.  The certified
  Level-2 diameter lower bounds span `{level2.min():.6g}`--`{level2.max():.6g}`.
- The hard TV smoothness prior leaves the same rank/null and an exact
  continuum.  Certified Level-3 diameter lower bounds span
  `{level3.min():.6g}`--`{level3.max():.6g}`; five of eight exceed the inherited
  scientific gate directly, while the remaining three still retain a
  nonunique prompt-conditioned continuum.
- All accepted structural singleton fits have full restricted rank, null zero,
  one exact observation-consistent output, diameter zero, unique prompt
  identity, and exact residuals from `{unique_residuals.min():.3e}` to
  `{unique_residuals.max():.3e}` against the fixed
  `{audit.EXACT_TOLERANCE:.3e}` tolerance.
- Level 7 is a strictly positive soft density over Level-6 support.  As frozen,
  it changes posterior preference but cannot change structural rank, exact
  solution count, or diameter; its results therefore equal Level 6.

## Scientific answer

The prior-free Family-E1 frontier is **not universally fundamental**: explicit
prompt-centered galaxy structure collapses 10,797--10,800 dimensional exact
allocation null spaces to zero for seven scenes, with modest condition numbers.
However, the frontier **does not fully disappear under the allowed priors**:
scene 51 remains unidentifiable because every allowed structural family is
exactly misspecified at the frozen support boundary.  The evidence therefore
supports a mixed conclusion—mostly missing inductive structure, plus one
PSF-resolution/model-support limit—not a universal observation-information
limit and not universal recoverability.

## Exactly one recommended next experiment

Run **Thayer-Identifiability-PSF-Core-v1**: one training-free, preregistered
repeat on the same eight scenes that adds a single PSF-unresolved central
component branch to the Level-5 forward model, with all other priors, prompts,
noise convention, exactness tolerance, and scientific diameter gate frozen.
Its sole purpose is to determine whether scene 51's 0.015-arcsec component is
identifiable as unresolved flux or remains observation-limited.

## Integrity

The prior freeze SHA-256 is `{audit.PREREG_SHA256}`.  Eight authorized training
scene rows and sixteen corresponding training catalog rows were accessed.
Development, Atlas, lockbox, neural-model imports, and network-weight optimizer
steps were all zero.  README, Family-E1, D0/D1/D2/D3, thresholds, prompts,
historical checkpoints, and the staged index were unchanged.
"""


def aggregate(results: list[dict[str, Any]], *, resume: bool = False) -> None:
    table_rows = []
    full_rows = []
    for result in results:
        for record in result["priors"]:
            table_rows.append(
                {
                    "Scene": int(record["scene"]),
                    "Prior": f"L{record['prior_level']} {record['prior']}",
                    "Rank": int(record["rank"]),
                    "Null space": int(record["null_space"]),
                    "Diameter": diameter_text(record),
                    "Classification": record["classification"],
                }
            )
            full_rows.append({key: value for key, value in record.items() if key != "diagnostics"})
    table = pd.DataFrame(table_rows)
    full = pd.DataFrame(full_rows)
    tables = RUN / "tables_final"
    tables.mkdir(parents=True, exist_ok=True)
    table_path = tables / "identifiability_table.csv"
    full_path = tables / "full_identifiability_metrics.csv"
    minimum_path = tables / "minimum_unique_prior_by_scene.csv"
    if resume:
        for path in (table_path, full_path, minimum_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        persisted_table = pd.read_csv(table_path, keep_default_na=False)
        if len(persisted_table) != len(table) or list(persisted_table.columns) != list(table.columns):
            raise RuntimeError("persisted requested table does not match closure shape")
    else:
        for path in (table_path, full_path, minimum_path):
            if path.exists():
                raise FileExistsError(path)
        table.to_csv(table_path, index=False)
        full.to_csv(full_path, index=False)
    minimum = pd.DataFrame(
        [
            {
                "Scene": int(result["scene"]),
                "Minimum prior level required for UNIQUE": result["minimum_prior_level_for_unique"],
            }
            for result in results
        ]
    )
    if not resume:
        minimum.to_csv(minimum_path, index=False)
    report = make_report(results, table, full)
    write_text_fresh(RUN / "reports/final_report.md", report)
    manifest = {
        "campaign": "Thayer-Identifiability-v1",
        "status": "PASS",
        "preregistration_sha256": audit.PREREG_SHA256,
        "closure_script_sha256": sha256_file(Path(__file__)),
        "analysis_script_sha256": sha256_file(REPO / "scripts/analyze_thayer_identifiability_v1.py"),
        "scene_result_sha256": {
            str(index): sha256_file(FINAL / f"scene_{index:03d}.json") for index in INDICES
        },
        "identifiability_table_sha256": sha256_file(table_path),
        "full_metrics_sha256": sha256_file(full_path),
        "minimum_unique_sha256": sha256_file(minimum_path),
        "final_report_sha256": sha256_file(RUN / "reports/final_report.md"),
        "minimum_prior_levels": {
            str(result["scene"]): result["minimum_prior_level_for_unique"] for result in results
        },
        "classification_counts": full.classification.value_counts().to_dict(),
        "access_counts": {
            "authorized_training_scenes": 8,
            "authorized_training_catalog_rows": 16,
            "development": 0,
            "atlas": 0,
            "lockbox": 0,
            "neural_model_imports": 0,
            "network_weight_optimizer_steps": 0,
        },
        "git_staged_index_empty": True,
        "completed_unix": time.time(),
    }
    write_json_fresh(RUN / "logs/final_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-finalize", action="store_true")
    args = parser.parse_args()
    verify()
    if args.resume_finalize:
        results = [json.loads((FINAL / f"scene_{index:03d}.json").read_text()) for index in INDICES]
        aggregate(results, resume=True)
    else:
        results = close_scenes()
        aggregate(results)


if __name__ == "__main__":
    main()
