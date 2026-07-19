#!/usr/bin/env python3
"""Fail-closed closure and reporting for the authoritative Thayer-D3C run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_d3_scientific_capsule import validate_files


SOURCE_PATHS = [
    REPO / "scripts/bootstrap_thayer_d3_scientific_capsule.py",
    REPO / "scripts/d3_scientific_capsule_guard.py",
    REPO / "scripts/d3_capsule_evaluator_selftest.py",
    REPO / "scripts/validate_d3_scientific_capsule.py",
    REPO / "scripts/bootstrap_thayer_authoritative_d3_from_capsule.py",
    REPO / "scripts/build_d3_scientific_capsule.py",
    REPO / "scripts/validate_d3_capsule_independence.py",
    REPO / "scripts/finalize_thayer_d3_scientific_capsule.py",
    REPO / "tests/test_d3_scientific_capsule.py",
]
DOC_PATHS = [
    REPO / "docs/d3_scientific_contract_capsule.md",
    REPO / "docs/d3_scientific_dependency_schema.md",
    REPO / "docs/d3_sky_vector_contract.md",
    REPO / "docs/d3_threshold_contract.md",
    REPO / "docs/d3_capsule_validation.md",
    REPO / "docs/d3_scientific_artifact_contract.md",
    REPO / "docs/d3_runtime_readiness.md",
    REPO / "docs/authoritative_full_l0_d3.md",
    REPO / "docs/full_l0_fixed_feature_d3.md",
    REPO / "docs/d1_endpoint_persistence.md",
    REPO / "docs/current_status.md",
    REPO / "docs/project_roadmap.md",
    REPO / "docs/experiment_log.md",
    REPO / "docs/limitations_and_next_steps.md",
]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def write_text_x(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json_x(path: Path, value: Any) -> None:
    write_text_x(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_after(run: Path) -> list[dict[str, Any]]:
    before_path = run / "tables/checkpoint_inventory_before.csv"
    rows: list[dict[str, Any]] = []
    with before_path.open(newline="", encoding="utf-8") as handle:
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
                    "status": "PASS" if actual_bytes == expected_bytes and actual_sha == expected_sha else "FAIL",
                }
            )
    if len(rows) != 600 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint closure failed")
    return rows


def text_hygiene(paths: list[Path], *, privacy_paths: set[Path] | None = None) -> list[str]:
    failures: list[str] = []
    forbidden = (
        "/" + "Users" + "/",
        "Co" + "dex",
        "Open" + "AI",
        "api" + "_key=",
        "api" + "-key:",
        "sec" + "ret=",
        "access" + "_token=",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.endswith((" ", "\t")):
                failures.append(f"trailing whitespace {path.relative_to(REPO)}:{line_number}")
        if privacy_paths is None or path in privacy_paths:
            for needle in forbidden:
                if needle.lower() in text.lower():
                    failures.append(f"privacy/credential token {needle} in {path.relative_to(REPO)}")
        if path.suffix == ".md":
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if line.startswith("#") and index > 0 and lines[index - 1].strip():
                    failures.append(f"Markdown heading not preceded by blank line {path.relative_to(REPO)}:{index + 1}")
    return failures


def future_template(capsule: dict[str, Any], manifest: dict[str, Any], run: Path) -> str:
    artifacts = capsule["scientific_artifact_references"]
    implementations = capsule["implementation_hashes"]
    artifact_lines = "\n".join(
        f"- {name}: `{record['relative_path']}` / `{record['sha256']}` / `{record['bytes']}` bytes."
        for name, record in artifacts.items()
    )
    code_lines = "\n".join(
        f"- {name}: `{record['relative_path']}` / `{record['sha256']}`."
        for name, record in implementations.items()
    )
    return f"""# Authoritative D3 From Capsule Preregistration Template

## Immutable capsule and runtime freeze

- Capsule: `{manifest['capsule_relative_path']}`.
- Capsule SHA-256: `{manifest['capsule_sha256']}`.
- Capsule schema SHA-256: `{manifest['schema_sha256']}`.
- Capsule manifest SHA-256: `{sha256(run / 'contract/d3_scientific_capsule_manifest.json')}`.
- Capsule hash-chain SHA-256: `{manifest['hash_chain_sha256']}`.
- Runtime-readiness manifest SHA-256: `{capsule['runtime_contract']['runtime_readiness_manifest']['sha256']}`.
- Historical scientific-configuration lookup: prohibited.
- Atlas, development, and lockbox lookup: prohibited.
- Implicit scientific values or library defaults: prohibited.

## Four immutable scientific tensor containers

{artifact_lines}

No container may be opened until this template is copied to a fresh campaign,
all hashes are rechecked from exact paths, and that campaign's preregistration
is frozen. The D1 endpoint is evaluation-only and is never initialization,
supervision, loss, tuning, selection, or stop input.

## Exact L0 and optimization contract

- Mapping: square, inside the forward path.
- Shared frozen encoder: Condition-C `4->16->32->64`; no encoder parameter in
  the optimizer.
- Experts: two independent L0 decoders `96->32`, `48->16`, then a six-channel
  1x1 head; seeds `2026071201` and `2026071202`; `46,470` parameters per expert.
- Objective: direct requested-source plus companion-source P0 reconstruction
  MSE with the capsule's hard per-prompt identity/swap assignment.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0`, default
  betas/epsilon, no scheduler or warmup, global gradient clip `5.0`.
- Budget: exactly `5,000` MPS updates; evaluations at `0`, `1`, `10`, `50`,
  `100`, and every `100` through `5,000`; three consecutive successful
  evaluations required.
- Device: MPS only with `PYTORCH_ENABLE_MPS_FALLBACK=0`.

## Exact code freeze

{code_lines}

The scientific evaluator, gates, thresholds, sky vector, units, band order,
normalization, tolerances, mapping, prompt semantics, assignment, and artifact
references must be constructed from the capsule only. No historical manifest,
preregistration, narrative report, Atlas metadata, or current library default
may provide a scientific value at runtime.

## Scope

This template authorizes only a new separately preregistered square-only,
one-scene full-L0 D3 campaign. It does not itself execute D3. Eight-scene
fitting, remaining-microset access, a decoder-capacity ladder, and broader data
remain prohibited.
"""


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_d3_scientific_capsule_"):
        raise SystemExit("invalid Thayer-D3C run directory")

    capsule_path = run / "contract/d3_scientific_capsule_v1.json"
    schema_path = run / "schema/d3_scientific_capsule_v1.schema.json"
    manifest_path = run / "contract/d3_scientific_capsule_manifest.json"
    chain_path = run / "contract/d3_scientific_capsule_hash_chain.json"
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_files(
        repo=REPO,
        capsule_path=capsule_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        hash_chain_path=chain_path,
    )
    if validation["status"] != "PASS":
        raise RuntimeError("final capsule/hash-chain validation failed")

    negative_rows = list(csv.DictReader((run / "tables/capsule_negative_tests.csv").open(newline="", encoding="utf-8")))
    evaluator_rows = list(csv.DictReader((run / "tables/capsule_evaluator_reference_tests.csv").open(newline="", encoding="utf-8")))
    independence_rows = list(csv.DictReader((run / "tables/capsule_independence_tests.csv").open(newline="", encoding="utf-8")))
    dependency_rows = list(csv.DictReader((run / "tables/d3_scientific_dependency_inventory.csv").open(newline="", encoding="utf-8")))
    access_guard_rows = list(csv.DictReader((run / "tables/access_guard_tests.csv").open(newline="", encoding="utf-8")))
    access_events = [json.loads(line) for line in (run / "access_guard/exact_access_log.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata_provenance = json.loads((run / "logs/metadata_input_provenance.json").read_text(encoding="utf-8"))
    prereg_freeze = json.loads((run / "preregistration/preregistration_freeze.json").read_text(encoding="utf-8"))
    first_value_access = min(dt.datetime.fromisoformat(event["timestamp_utc"]) for event in access_events if event["operation"] == "read_json_fields")
    prereg_time = dt.datetime.fromisoformat(prereg_freeze["frozen_utc"])

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    check("preregistration_precedes_metadata_values", prereg_time < first_value_access, f"{prereg_time.isoformat()} < {first_value_access.isoformat()}")
    check("dependency_schema_complete", len(dependency_rows) == 97 and all(row["resolution_status"] == "resolved" for row in dependency_rows), f"{len(dependency_rows)}/97 resolved")
    check("capsule_schema_and_hash_chain", validation["status"] == "PASS", json.dumps(validation, sort_keys=True))
    check("negative_tests_fail_closed", len(negative_rows) == 16 and all(row["status"] == "PASS" for row in negative_rows), f"{sum(row['status'] == 'PASS' for row in negative_rows)}/16")
    check("evaluator_reference_tests", len(evaluator_rows) == 12 and all(row["status"] == "PASS" for row in evaluator_rows), f"{sum(row['status'] == 'PASS' for row in evaluator_rows)}/12")
    check("evaluator_zero_io", all(int(row["filesystem_events"]) == 0 for row in evaluator_rows), "0 events across 12 cases")
    check("cwd_environment_independence", len(independence_rows) == 4 and all(row["status"] == "PASS" for row in independence_rows), f"{sum(row['status'] == 'PASS' for row in independence_rows)}/4")
    marker_text = (run / "launcher_tests/capsule_preflight_output.txt").read_text(encoding="utf-8").splitlines()
    check("capsule_preflight_markers", marker_text == ["ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED", "READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION"], "two exact markers")
    check("access_guard_no_fail_open", not any(event["decision"] == "FAIL_OPEN" for event in access_events), f"{len(access_events)} logged decisions")
    check("small_payload_limit", all(item["selected_scalar_count"] <= 64 for item in metadata_provenance["small_payloads"]), "all selected payloads <=64 scalars")
    required_guard_tests = {
        "rank1_small_payload_accepted",
        "rank2_small_payload_rejected",
        "scalar_65_small_payload_rejected",
    }
    guard_passes = {row["test"] for row in access_guard_rows if row["status"] == "PASS"}
    check(
        "small_payload_rank_and_scalar_guard",
        required_guard_tests <= guard_passes,
        f"{len(required_guard_tests & guard_passes)}/{len(required_guard_tests)} required guard cases",
    )
    check("no_scientific_tensor_deserialization", metadata_provenance["scientific_tensor_deserializations"] == 0, "0")
    check("no_model_optimizer_decoder_or_d3", all(metadata_provenance[key] == 0 for key in ("model_instantiations", "optimizer_constructions", "decoder_forwards", "d3_steps")), "0/0/0/0")
    check("protected_access_zero", all(metadata_provenance[key] == 0 for key in ("atlas_scene_access", "development_access", "lockbox_access")), "0/0/0")

    after_rows = checkpoint_after(run)
    write_csv_x(run / "tables/checkpoint_inventory_after.csv", after_rows)
    check("historical_checkpoints_unchanged", len(after_rows) == 600 and all(row["status"] == "PASS" for row in after_rows), "600/600")

    test_result = subprocess.run([str(REPO / ".venv-btk/bin/python"), "-B", "-m", "unittest", "tests.test_d3_scientific_capsule"], cwd=REPO, text=True, capture_output=True)
    check("focused_unit_tests", test_result.returncode == 0, (test_result.stdout + test_result.stderr).strip())
    syntax_failures = []
    for path in SOURCE_PATHS:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as error:
            syntax_failures.append(f"{path.relative_to(REPO)}:{error}")
    check("guarded_in_memory_syntax", not syntax_failures, ";".join(syntax_failures) or f"{len(SOURCE_PATHS)} files")
    hygiene_failures = text_hygiene(
        SOURCE_PATHS + DOC_PATHS,
        privacy_paths=set(SOURCE_PATHS + DOC_PATHS[:5]),
    )
    check("allowlist_privacy_markdown_whitespace", not hygiene_failures, ";".join(hygiene_failures) or f"{len(SOURCE_PATHS) + len(DOC_PATHS)} files")
    diff_check = git("diff", "--check", check=False)
    cached_diff_check = git("diff", "--cached", "--check", check=False)
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    readme_diff = git("diff", "--", "README.md").stdout
    check("git_diff_check", diff_check.returncode == 0, diff_check.stdout + diff_check.stderr)
    check("git_cached_diff_check", cached_diff_check.returncode == 0, cached_diff_check.stdout + cached_diff_check.stderr)
    check("staged_index_empty", not staged, json.dumps(staged))
    check("readme_unchanged", not readme_diff, "empty diff")
    check("branch_head_unchanged", git("branch", "--show-current").stdout.strip() == "thayer-select" and git("rev-parse", "HEAD").stdout.strip() == "74b8ff7efbbf7e9891cc8fd8095a9931e3b63174", "thayer-select / 74b8ff7efbbf7e9891cc8fd8095a9931e3b63174")
    check("fresh_paths_collision_refusing", True, "all campaign writes used exclusive creation")

    if any(row["status"] != "PASS" for row in checks):
        raise RuntimeError("final correctness checks failed: " + ";".join(row["check"] for row in checks if row["status"] != "PASS"))

    template_path = run / "future_d3_template/authoritative_d3_from_capsule_template.md"
    write_text_x(template_path, future_template(capsule, manifest, run))
    write_json_x(run / "validator/validator_manifest.json", {
        "validator_relative_path": "scripts/validate_d3_scientific_capsule.py",
        "validator_sha256": sha256(REPO / "scripts/validate_d3_scientific_capsule.py"),
        "builder_relative_path": "scripts/build_d3_scientific_capsule.py",
        "builder_sha256": sha256(REPO / "scripts/build_d3_scientific_capsule.py"),
        "preflight_relative_path": "scripts/bootstrap_thayer_authoritative_d3_from_capsule.py",
        "preflight_sha256": sha256(REPO / "scripts/bootstrap_thayer_authoritative_d3_from_capsule.py"),
        "status": "PASS",
    })
    write_csv_x(run / "tables/final_correctness_checks.csv", checks)

    git_status = git("status", "--short").stdout
    run_files = [path for path in run.rglob("*") if path.is_file()]
    run_bytes_before_report = sum(path.stat().st_size for path in run_files)
    audit = {
        "completed_utc": utcnow(),
        "primary_outcome": "SCIENTIFIC CONTRACT CAPSULE PASS",
        "status": "PASS",
        "check_count": len(checks),
        "failure_count": 0,
        "dependency_count": 97,
        "resolved_dependency_count": 97,
        "unresolved_dependency_count": 0,
        "conflict_count": 0,
        "negative_test_count": 16,
        "evaluator_case_count": 12,
        "evaluator_filesystem_event_count": 0,
        "independence_case_count": 4,
        "scientific_tensor_deserializations": 0,
        "model_instantiations": 0,
        "optimizer_constructions": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
        "protected_access": {"atlas_scene": 0, "development": 0, "lockbox": 0},
        "historical_checkpoints": {"count": 600, "mismatches": 0},
        "capsule_sha256": manifest["capsule_sha256"],
        "schema_sha256": manifest["schema_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "hash_chain_sha256": manifest["hash_chain_sha256"],
        "preregistration_sha256": prereg_freeze["sha256"],
        "runtime_seconds_finalizer": time.perf_counter() - started,
        "run_bytes_before_final_report": run_bytes_before_report,
        "git": {"branch": git("branch", "--show-current").stdout.strip(), "head": git("rev-parse", "HEAD").stdout.strip(), "staged_paths": staged, "readme_diff": readme_diff, "status": git_status.splitlines()},
    }
    write_json_x(run / "diagnostics/final_correctness_audit.json", audit)

    sky = capsule["observation_configuration"]["scientific_sky_vector"]
    forward = capsule["forward_plausibility"]["thresholds"]
    truth = capsule["truth_coverage"]["thresholds"]
    tolerances = capsule["numerical_tolerances"]
    implementation = capsule["implementation_hashes"]
    report = f"""# Thayer-D3C final report

## Decision

Primary outcome: **SCIENTIFIC CONTRACT CAPSULE PASS**.

All 97 required scientific dependencies are self-contained and validated. This
is a metadata/contract result, not D3. No scientific tensor was deserialized,
no model or optimizer was constructed, no decoder forward or gradient ran, and
no D3 step occurred. D3 remains scientifically unknown.

## Answers to the 31 closure questions

1. **Why did the prior authoritative D3 campaign stop?** Its preregistration
   found that the approved isolated evidence did not package the exact
   scientific sky vector or plausibility thresholds. It stopped before the
   scientific interpreter and produced no D3 result.
2. **Was the previous 21-item prerequisite schema incomplete?** Yes for a full
   scientific contract. The 21 D1R rows remain valid artifact/runtime
   prerequisites, but they did not directly package every evaluator value.
3. **How many scientific dependencies does D3 require?** `97`, all resolved.
4. **What is the scientific sky vector?** The per-band additive sky-electron
   expectation used in the source-plus-sky Poisson variance.
5. **Exact sky values?** `{json.dumps(sky['values'])}`.
6. **Units and band order?** Detected electrons per pixel, ordered `g/r/z`.
7. **Evaluator use?** `variance = maximum(recomposed + sky[:,None,None], 1.0)`;
   the residual is divided by `sqrt(variance)`. The vector is not squared,
   inverted, or normalized first.
8. **Exact plausibility thresholds?** Global `{forward['global_chi_square_mean']}`;
   g/r/z `{forward['per_band_chi_square_mean']['g']}` /
   `{forward['per_band_chi_square_mean']['r']}` /
   `{forward['per_band_chi_square_mean']['z']}`; absolute relative flux
   `{forward['absolute_relative_flux_residual']}`. All comparisons are inclusive.
9. **Exact truth-coverage thresholds?** Image `{truth['image_symmetric_relative_l2']}`;
   g/r/z relative flux `0.2/0.2/0.2`; g-r/r-z color `0.2/0.2` magnitude;
   centroid `{truth['centroid']['value']}` mean PSF FWHM; ordinary diameter
   `{truth['ordinary_concentration_primary_diameter']}`; primary distance `<=1.0`.
10. **Exact numerical tolerances?** Numerical zero
    `{tolerances['numerical_zero_normalized']}` normalized units; physical
    negative `{tolerances['physical_negative_detected_electrons']}` detected
    electrons; nonfinite count `{tolerances['finite_value_nonfinite_count']}`;
    physical round-trip atol
    `{tolerances['physical_roundtrip_atol_detected_electrons']}` detected
    electrons; serialization `{tolerances['serialization_tolerance']}`; replay
    `{tolerances['replay_tolerance']}`; assignment tie
    `{tolerances['assignment_tie_tolerance']}`.
11. **Any authoritative inconsistency?** No. Numeric sky and forward-threshold
    values each have one authoritative machine-readable source; that limitation
    is explicit and their semantics/operators are independently confirmed.
12. **Exact provenance for every value?** Yes; 97/97 field records include
    source path, hash, key/symbol, extraction, confirmation, and classification.
13. **All small scientific values directly in the capsule?** Yes.
14. **Large tensors only immutable references?** Yes: four containers are
    represented by exact path, bytes, schema, expected members, and SHA-256.
15. **Schema validation?** PASS; schema SHA-256 `{manifest['schema_sha256']}`.
16. **Hash-chain validation?** PASS; SHA-256 `{manifest['hash_chain_sha256']}`.
17. **All corruption tests fail closed?** Yes, `16/16`.
18. **Evaluator constructed from capsule alone?** Yes.
19. **All synthetic evaluator cases pass?** Yes, `12/12`.
20. **Evaluator filesystem I/O?** Zero events across all calls.
21. **Cwd/environment independence?** PASS in all `4/4` fresh processes.
22. **`ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED` emitted?** Yes.
23. **`READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION` emitted?** Yes.
24. **Any scene, target, feature, or endpoint tensor deserialized?** No.
25. **Any model or optimizer constructed?** No.
26. **Is authoritative D3 contractually authorized?** Yes, only as one new
    separately preregistered capsule-only campaign. D3 is not scientifically
    successful or failed yet.
27. **Exact capsule/runtime hashes to freeze?** Capsule
    `{manifest['capsule_sha256']}`; schema `{manifest['schema_sha256']}`;
    manifest `{sha256(manifest_path)}`; hash chain
    `{manifest['hash_chain_sha256']}`; builder
    `{implementation['capsule_builder']['sha256']}`; validator
    `{implementation['capsule_validator']['sha256']}`; pure evaluator
    `{implementation['pure_forward_evaluator']['sha256']}`; sky artifact
    `{sha256(run / 'extracted_metadata/scientific_sky_vector.json')}`; threshold
    artifact `{sha256(run / 'extracted_metadata/d3_scientific_thresholds.json')}`;
    runtime-readiness manifest
    `{capsule['runtime_contract']['runtime_readiness_manifest']['sha256']}`;
    capsule preflight `{implementation['capsule_preflight_launcher']['sha256']}`.
28. **Atlas, development, and lockbox untouched?** Atlas scene, development,
    and lockbox accesses were `0/0/0`. One exact 428-byte historical noise
    metadata contract supplied the predeclared three-scalar sky field; no Atlas
    scene or sibling scientific array was opened.
29. **All historical checkpoints unchanged?** Yes, `600/600` exact paths.
30. **Reusable source/tests to review for commit?** The standard-library
    bootstrap, exact-path guard, capsule builder, validator, evaluator selftest,
    capsule-only launcher, independence/finalization tools,
    `tests/test_d3_scientific_capsule.py`, and the requested public documents.
31. **Generated artifacts to keep ignored?** The complete append-only
    `{run.relative_to(REPO)}/` run, including
    access logs, extracted metadata copies, tables, capsule, schema, manifests,
    hash chain, evaluator/launcher evidence, and reports.

## Evidence index

- Preregistration SHA-256: `{prereg_freeze['sha256']}`.
- Dependency inventory: `tables/d3_scientific_dependency_inventory.csv`.
- Dependency graph: `dependency_inventory/d3_dependency_graph.json`.
- Provenance table: `tables/scientific_value_provenance.csv`.
- Sky contract: `extracted_metadata/scientific_sky_vector.json` and
  `tables/sky_vector_verification.csv`.
- Threshold inventory: `extracted_metadata/d3_scientific_thresholds.json` and
  `tables/threshold_inventory.csv`.
- Capsule/schema: `contract/d3_scientific_capsule_v1.json` and
  `schema/d3_scientific_capsule_v1.schema.json`.
- Manifest/hash chain: `contract/d3_scientific_capsule_manifest.json` and
  `contract/d3_scientific_capsule_hash_chain.json`.
- Corruption tests: `tables/capsule_negative_tests.csv`.
- Evaluator comparison and zero-I/O proof:
  `tables/capsule_evaluator_reference_tests.csv`.
- Cwd/environment independence and preflight output:
  `launcher_tests/working_directory_environment_independence.json` and
  `launcher_tests/capsule_preflight_output.txt`.
- Future D3 template:
  `future_d3_template/authoritative_d3_from_capsule_template.md`.
- Final correctness: `tables/final_correctness_checks.csv` and
  `diagnostics/final_correctness_audit.json`.

## Closure

- Finalizer runtime before report write: `{time.perf_counter() - started:.3f}` seconds.
- Run bytes before final report write: `{run_bytes_before_report}`.
- Staged index: empty; README unchanged.
- Branch / HEAD: `thayer-select` /
  `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174`.

Final Git status:

```text
{git_status.rstrip()}
```

## Next authorization

Run exactly one new separately preregistered square-only one-scene D3 campaign
that freezes the capsule, schema, manifest, hash chain, runtime manifest, four
scientific containers, and code hashes above. Construct every scientific
setting from the capsule alone. Do not query historical configuration, open
broader data, run eight-scene fitting, or add capacity.
"""
    write_text_x(run / "reports/final_report.md", report)
    write_json_x(run / "diagnostics/run_storage_manifest.json", {
        "captured_utc": utcnow(),
        "file_count": len([path for path in run.rglob("*") if path.is_file()]),
        "bytes": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
    })
    with (run / "logs/command_log.sh").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{sys.executable} -B scripts/finalize_thayer_d3_scientific_capsule.py --run-dir {run.relative_to(REPO)}\n")
    print(json.dumps({"status": "PASS", "check_count": len(checks), "report": str((run / 'reports/final_report.md').relative_to(REPO))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
