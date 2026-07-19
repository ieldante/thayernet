#!/usr/bin/env python3
"""Standard-library-only bootstrap for the Thayer-D3C contract capsule.

This phase creates a fresh append-only run, records exact-file provenance,
rechecks the frozen 600-checkpoint inventory, and freezes the scientific
metadata resolution contract before any scientific metadata value is read.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs/runs"
VENV = REPO / ".venv-btk"
CHECKPOINT_BASELINE = (
    REPO
    / "outputs/runs/thayer_full_l0_d3r_20260713_121652/tables/checkpoint_inventory_after.csv"
)

INITIAL_ALLOWLIST = (
    "outputs/runs/thayer_authoritative_d3_20260713_145040/reports/final_report.md",
    "outputs/runs/thayer_authoritative_d3_20260713_145040/preregistration/authoritative_square_full_l0_d3.md",
    "outputs/runs/thayer_authoritative_d3_20260713_145040/diagnostics/final_correctness_audit.json",
    "docs/d3_scientific_artifact_contract.md",
    "scripts/bootstrap_thayer_authoritative_d3.py",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/reports/final_report_superseding_v3.md",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/diagnostics/final_correctness_audit_superseding_v3.json",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/tables/deletion_rename_inventory.csv",
    "docs/d3_runtime_readiness.md",
    "docs/pure_forward_evaluator_contract.md",
    "docs/d3_runtime_bootstrap_contract.md",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/reports/final_report.md",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/replay_verification/d1_endpoint_manifest.json",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/diagnostics/final_correctness_audit_superseding_v2.json",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/diagnostics/final_access_log_manifest.json",
    "outputs/runs/thayer_repository_integrity_20260713_031653/reports/final_report.md",
    "outputs/runs/thayer_repository_integrity_20260713_031653/diagnostics/final_correctness_audit_superseding_v2.json",
    "outputs/runs/thayer_repository_integrity_20260713_031653/tables/d0_d3_summary_superseding_v2.csv",
    "docs/repository_integrity_audit.md",
    "docs/independent_scientific_oracles.md",
    "docs/allowlisted_file_access_contract.md",
    "outputs/runs/thayer_feasibility_projection_20260712_234216/reports/final_report.md",
    "outputs/runs/thayer_feasibility_projection_20260712_234216/diagnostics/final_correctness_audit.json",
    "docs/scientific_region_projection_contract.md",
    "docs/direct_scientific_feasibility_projection.md",
    "outputs/runs/thayer_output_parameterization_20260713_023120/reports/final_report.md",
    "outputs/runs/thayer_output_parameterization_20260713_023120/diagnostics/final_correctness_audit.json",
    "outputs/runs/thayer_output_parameterization_20260713_023120/preregistration/fixed_l0_output_parameterization.md",
    "docs/physical_source_output_contract.md",
    "docs/output_parameterization_selection.md",
    "docs/d1_endpoint_persistence.md",
    "docs/feature_endpoint_artifact_contract.md",
    "docs/fixed_feature_decoder_audit.md",
    "docs/current_status.md",
    "docs/scientific_region_projection_contract.md",
    "docs/direct_scientific_feasibility_projection.md",
    "docs/d1_endpoint_persistence.md",
    "docs/feature_endpoint_artifact_contract.md",
    "docs/fixed_feature_decoder_audit.md",
    "scripts/bootstrap_thayer_d3_scientific_capsule.py",
    "outputs/runs/thayer_full_l0_d3r_20260713_121652/tables/checkpoint_inventory_after.csv",
)

SUBDIRECTORIES = (
    "access_guard",
    "diagnostics",
    "tables",
    "figures",
    "logs",
    "reports",
    "preregistration",
    "dependency_inventory",
    "provenance_resolution",
    "extracted_metadata",
    "contract",
    "schema",
    "validator",
    "evaluator_tests",
    "launcher_tests",
    "future_d3_template",
)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    )
    return result.stdout


def write_text_x(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json_x(path: Path, payload: Any) -> None:
    write_text_x(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = RUNS / f"thayer_d3_scientific_capsule_{stamp}"
    run.mkdir(parents=False, exist_ok=False)
    for relative in SUBDIRECTORIES:
        (run / relative).mkdir(exist_ok=False)
    return run


def exact_file_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in dict.fromkeys(INITIAL_ALLOWLIST):
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"required exact allowlisted file missing: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "read_scope": "exact_path_metadata_or_text_only",
                "scientific_tensor_deserialized": False,
            }
        )
    return rows


def checkpoint_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        for frozen in csv.DictReader(handle):
            path = REPO / frozen["path"]
            expected_bytes = int(frozen["expected_bytes"])
            expected_sha = frozen["expected_sha256"]
            actual_bytes = path.stat().st_size if path.is_file() else -1
            actual_sha = sha256(path) if path.is_file() else "MISSING"
            rows.append(
                {
                    "path": frozen["path"],
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "status": "PASS"
                    if actual_bytes == expected_bytes and actual_sha == expected_sha
                    else "FAIL",
                }
            )
    if len(rows) != 600:
        raise RuntimeError(f"expected 600 frozen checkpoint rows; found {len(rows)}")
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint baseline mismatch")
    return rows


def preregistration_text(run: Path, snapshot: dict[str, Any]) -> str:
    source_rows = "\n".join(f"- `{p}`" for p in dict.fromkeys(INITIAL_ALLOWLIST))
    return f"""# Thayer-D3C Scientific Contract Capsule Preregistration

## Identity and ordering freeze

- Campaign: `Thayer-D3C` (`Thayer D3 Scientific Contract Capsule`).
- Fresh run: `{run.name}`.
- Frozen UTC: `{utcnow()}`.
- Repository branch / HEAD: `{snapshot['branch']}` / `{snapshot['git_head']}`.
- Scientific metadata values extracted before this freeze: `0`.
- Scientific tensors deserialized before this freeze: `0`.
- Models, optimizers, gradients, decoder forwards, and D3 steps before this freeze: `0`.

This is a metadata-and-contract campaign. It may create a complete immutable
scientific contract capsule, but it may not execute D3 or authorize itself to
continue into D3.

## Frozen authoritative seed artifacts

{source_rows}

Only exact additional paths explicitly named by these seed artifacts may be
opened. Dependency discovery is deterministic: parse exact repository-relative
path literals and explicit JSON/CSV manifest references from a seed artifact,
resolve them relative to that artifact's authoritative run root when needed,
require an existing regular file, record its size and SHA-256 before content
use, and refuse directory enumeration, wildcard expansion, or guessed paths.

## Frozen metadata-key resolution

Scientific metadata keys are identified before value extraction by AST and
schema inspection of the exact allowlisted evaluator, oracle, bootstrap,
preregistration, manifest, and contract sources. Candidate keys must be named
by a consuming function signature, explicit constant reference, manifest
field, CSV column, JSON pointer, or named NPZ/HDF5 member. The resolver may
read only the preidentified member; it may not iterate sibling members. A
payload is admissible only when it has no scene or spatial axis, rank at most
one except for a small named threshold mapping, and at most 64 scalar values.
Shape and dtype are checked before values are accepted. Every exact file and
member access is logged.

## Frozen source priority and discrepancy rules

1. Explicit value in a frozen authoritative preregistration.
2. Explicit value in a frozen authoritative manifest or JSON/CSV contract.
3. Explicit immutable Python constant in an allowlisted module.
4. Deterministic derivation from immutable constants with formula and code hash.
5. Exact small metadata member from an allowlisted NPZ/HDF5 container.

Machine-readable values outrank rounded narrative prose. No current library
default, inferred default, synthetic value, later incompatible value, or
post-hoc threshold choice is permitted. Exact disagreement is a terminal
provenance conflict. A required unresolved value is a terminal
`AUTHORITATIVE SOURCE MISSING`; no partial authoritative capsule is allowed.

## Frozen dependency schema and completeness gate

The inventory begins from the authoritative D3 bootstrap, pure forward
evaluator, independent reference evaluator, truth-coverage evaluator, hard
assignment, target loss, mapping, stopping gates, and D3 preregistration. It
must enumerate observation configuration; plausibility and forward gates;
truth-coverage gates; output contract; prompt/assignment semantics; runtime
implementation hashes; immutable artifact references; and row identity.
Each dependency records canonical name, type, shape, units, band order,
requiredness, default policy, consumer, source symbol, source path/key/hash,
and resolution state. Schema completeness is required, not only field validity.

## Frozen capsule schema

Required top-level sections are `capsule_identity`, `scientific_semantics`,
`observation_configuration`, `forward_plausibility`, `truth_coverage`,
`numerical_tolerances`, `implementation_hashes`,
`scientific_artifact_references`, `row_identity`, `provenance`,
`runtime_contract`, and `completeness`. Required values may not be null or use
`TODO`, `UNKNOWN`, `TBD`, implicit defaults, nonfinite numbers, ambiguous units,
or implicit band order. Every small scientific value is embedded directly;
large tensors are referenced only by exact path, byte size, schema/member
expectations, and SHA-256.

## Frozen validation and tests

The builder uses exact paths, canonical sorted JSON, explicit float values,
and attached field provenance. The validator rejects missing/extra required
semantics, nonfinite values, band or unit mismatch, threshold/operator drift,
code/artifact/runtime hash drift, protected paths, placeholders, hidden
defaults, and absence of `ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED`.
Negative tests corrupt every frozen dependency class. Evaluator validation
uses only the capsule, exact evaluator/reference sources, and twelve synthetic
cases; evaluation must perform zero filesystem I/O and be deterministic.
Fresh-process validation covers repository-root, temporary-cwd, cleared-env,
and frozen-runtime execution without scientific tensor deserialization.

## Gate attainability audit

Each capsule gate is mechanically attainable if and only if all consuming
symbols and exact values are reachable through the frozen exact-path procedure.
The fail-closed outcomes are attainable by construction: source absence,
conflict, hidden dependency, or validation failure stops construction before
an authoritative capsule marker. `SCIENTIFIC CONTRACT CAPSULE PASS` is
attainable only after required count equals resolved count, conflicts and
unresolved counts are zero, the hash chain and all tests pass, no prohibited
access occurs, and all 600 checkpoint hashes remain unchanged.

## No-scene / no-model / no-D3 policy

No scene, target-image, cached-feature, endpoint, checkpoint, or other
scientific tensor value may be deserialized. No encoder, decoder, project
model, optimizer, gradient, or decoder forward may be constructed or run.
Atlas, development, and lockbox scene paths remain prohibited. Capsule success
authorizes only a separately preregistered future D3 campaign; it does not run
D3 here.
"""


def main() -> int:
    run = make_run()
    started_utc = utcnow()
    snapshot = {
        "captured_utc": started_utc,
        "branch": git("branch", "--show-current").strip(),
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_status": git("status", "--short").splitlines(),
        "staged_index": git("diff", "--cached", "--name-status").splitlines(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "exact_environment_path": str(VENV),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "campaign_start_utc": started_utc,
        "third_party_imports": 0,
        "scientific_metadata_values_extracted": 0,
        "scientific_tensor_deserializations": 0,
    }
    write_text_x(
        run / "diagnostics/environment_snapshot_stdlib_only.md",
        "# Standard-library-only environment snapshot\n\n```json\n"
        + json.dumps(snapshot, indent=2, sort_keys=True)
        + "\n```\n",
    )
    write_text_x(
        run / "logs/command_log.sh",
        f"{sys.executable} -B scripts/bootstrap_thayer_d3_scientific_capsule.py\n",
    )
    checkpoint_rows = checkpoint_inventory()
    write_csv_x(run / "tables/checkpoint_inventory_before.csv", checkpoint_rows)
    exact_inputs = exact_file_inventory()
    write_json_x(
        run / "logs/input_provenance.json",
        {
            "campaign": "Thayer-D3C",
            "captured_before_scientific_metadata_value_extraction": True,
            "exact_inputs": exact_inputs,
            "checkpoint_baseline_rows": len(checkpoint_rows),
            "blocked_or_nonallowlisted_reads": 0,
            "scene_tensor_reads": 0,
            "target_tensor_reads": 0,
            "cached_feature_tensor_reads": 0,
            "endpoint_tensor_reads": 0,
            "atlas_access": 0,
            "development_access": 0,
            "lockbox_access": 0,
        },
    )
    write_text_x(
        run / "diagnostics/campaign_contract.md",
        "# Thayer-D3C campaign contract\n\n"
        "Fresh append-only metadata-and-contract campaign. Exact paths only; "
        "no scientific tensor deserialization, model construction, optimizer, "
        "decoder forward, gradients, D3, protected data access, historical "
        "writes, staging, or implicit scientific defaults.\n",
    )
    prereg_path = run / "preregistration/d3_scientific_contract_capsule.md"
    write_text_x(prereg_path, preregistration_text(run, snapshot))
    freeze = {
        "path": str(prereg_path.relative_to(run)),
        "sha256": sha256(prereg_path),
        "frozen_utc": utcnow(),
        "metadata_value_extractions_before_freeze": 0,
        "third_party_imports_before_freeze": 0,
        "scientific_tensor_deserializations_before_freeze": 0,
        "checkpoint_rows_verified": len(checkpoint_rows),
        "status": "FROZEN_BEFORE_METADATA_VALUE_EXTRACTION",
    }
    write_json_x(run / "preregistration/preregistration_freeze.json", freeze)
    print(run.relative_to(REPO))
    print(freeze["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
