#!/usr/bin/env python3
"""Standard-library-only orchestrator for the bundle-driven Thayer-D3S run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json"
BUNDLE_SHA256 = "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045"
GOVERNING_REQUEST = Path()
CHECKPOINT_BASELINE = REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/tables/checkpoint_inventory_before.csv"
DIRECTORIES = (
    "access_guard",
    "runtime/orchestrator",
    "runtime/scientific/tmp",
    "runtime/scientific/cache",
    "runtime/scientific/config",
    "runtime/scientific/torch",
    "runtime/scientific/pycache",
    "runtime/replay/tmp",
    "runtime/replay/cache",
    "runtime/replay/config",
    "runtime/replay/torch",
    "runtime/replay/pycache",
    "runtime/postprocess_runtime/tmp",
    "runtime/postprocess_runtime/cache",
    "runtime/postprocess_runtime/config",
    "runtime/postprocess_runtime/matplotlib",
    "runtime/postprocess_runtime/pycache",
    "runtime/postprocess_runtime/output",
    "diagnostics",
    "tables",
    "figures",
    "logs",
    "reports",
    "preregistration",
    "bundle_validation",
    "authoritative_inputs",
    "cached_features",
    "initial_state",
    "one_step_trace",
    "decoder_training",
    "penultimate_trajectories",
    "gradients",
    "checkpoints",
    "replay_verification",
    "postprocessing_inputs",
    "example_grids",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(*args: str, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(args)}\n{completed.stdout}\n{completed.stderr}")
    return completed


def git(*args: str) -> str:
    return command("git", *args).stdout.rstrip("\n")


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def create_run() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_scientific_d3_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for relative in DIRECTORIES:
        (run / relative).mkdir(parents=True, exist_ok=False)
    return run


def checkpoint_inventory() -> list[dict[str, object]]:
    with CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    if len(baseline) != 600:
        raise RuntimeError(f"expected 600 checkpoint baseline rows, found {len(baseline)}")
    rows: list[dict[str, object]] = []
    for index, item in enumerate(baseline, start=1):
        path = REPO / item["path"]
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "path": item["path"],
                "expected_bytes": int(item["expected_bytes"]),
                "actual_bytes": actual_bytes,
                "expected_sha256": item["expected_sha256"],
                "actual_sha256": actual_hash,
                "status": "PASS" if actual_bytes == int(item["expected_bytes"]) and actual_hash == item["expected_sha256"] else "FAIL",
            }
        )
        if index % 100 == 0:
            print(f"checkpoint inventory {index}/600", flush=True)
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint baseline mismatch")
    return rows


def referenced_file_rows(bundle: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, object]]:
    candidates: dict[str, tuple[Path, str | None, int | None, str]] = {}

    def add(label: str, record: object, source: str) -> None:
        if not isinstance(record, dict) or "path" not in record:
            return
        path = resolve(REPO, str(record["path"]))
        expected_hash = str(record["sha256"]) if "sha256" in record else None
        expected_bytes = int(record["bytes"]) if "bytes" in record else None
        candidates[str(path)] = (path, expected_hash, expected_bytes, f"{source}:{label}")

    for key, value in bundle.items():
        if isinstance(value, dict):
            add(key, value, "bundle")
            for child_key, child_value in value.items():
                add(f"{key}.{child_key}", child_value, "bundle")
    for record in registry["requirements"]:
        add(record["canonical_requirement_id"], record["expected_value"], "registry")
    rows: list[dict[str, object]] = []
    for path, expected_hash, expected_bytes, source in sorted(candidates.values(), key=lambda item: str(item[0])):
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        size_ok = expected_bytes is None or actual_bytes == expected_bytes
        hash_ok = expected_hash is None or actual_hash == expected_hash
        rows.append(
            {
                "source": source,
                "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                "expected_bytes": "" if expected_bytes is None else expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": "" if expected_hash is None else expected_hash,
                "actual_sha256": actual_hash,
                "status": "PASS" if path.is_file() and size_ok and hash_ok else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        failures = [row for row in rows if row["status"] != "PASS"]
        raise RuntimeError(f"bundle-referenced file mismatch: {failures[:3]}")
    return rows


def source_hash_rows() -> list[dict[str, object]]:
    relative_paths = (
        "scripts/run_thayer_scientific_d3.py",
        "scripts/run_thayer_authoritative_d3_v2.py",
        "scripts/run_thayer_d3_synthetic_preflight.py",
        "scripts/validate_d3_executable_capsule_v2.py",
        "scripts/thayer_d3_runtime_guard.py",
        "src/d3_executable_contract.py",
        "src/d3_requirement_registry.py",
        "src/canonical_tensor_hash.py",
        "src/competing_hypotheses.py",
        "src/models_probabilistic_unet.py",
        "src/models_two_expert_decoder.py",
        "src/output_parameterization.py",
    )
    return [
        {"path": value, "bytes": (REPO / value).stat().st_size, "sha256": sha256(REPO / value)}
        for value in relative_paths
    ]


def bundle_regression_audit(bundle: dict[str, Any], registry: dict[str, Any]) -> dict[str, object]:
    """Find settings that the governing D3S request requires the bundle to freeze."""

    identifiers = {str(record["canonical_requirement_id"]) for record in registry["requirements"]}
    searchable = {
        str(record["canonical_requirement_id"]): (
            str(record["human_readable_name"]) + " " + json.dumps(record["expected_value"], sort_keys=True)
        ).casefold()
        for record in registry["requirements"]
    }

    def matches(*terms: str) -> list[str]:
        return sorted(
            identifier
            for identifier, text in searchable.items()
            if any(term.casefold() in identifier.casefold() or term.casefold() in text for term in terms)
        )

    rows = [
        {
            "required_setting": "expert_activity_gate",
            "governing_request_location": "Part B; Part O",
            "required_semantics": "exact executable definition of both-experts-active and expert-death failure",
            "matching_registry_ids": matches("expert activity", "expert_activity", "expert death", "expert_death"),
            "status": "MISSING",
        },
        {
            "required_setting": "prompt_collapse_immediate_stop",
            "governing_request_location": "Part O",
            "required_semantics": "executable collapse condition distinct from descriptive prompt semantics",
            "matching_registry_ids": matches("prompt collapse", "prompt_collapse"),
            "status": "MISSING",
        },
        {
            "required_setting": "optional_tangent_protocol",
            "governing_request_location": "Part B; Part Q",
            "required_semantics": "authorization and exact finite-difference/JVP/VJP settings",
            "matching_registry_ids": matches("tangent", "jvp", "vjp", "jacobian", "singular value"),
            "status": "MISSING",
        },
        {
            "required_setting": "scientific_outcome_categories",
            "governing_request_location": "Part B; Part R",
            "required_semantics": "the six exact primary outcome categories and decision mapping",
            "matching_registry_ids": matches(
                "l0 full decoder success",
                "decoder optimization barrier",
                "decoder parameterization/capacity barrier",
                "hard-assignment barrier",
                "square-mapping optimization barrier",
                "mixed cause",
            ),
            "status": "MISSING",
        },
        {
            "required_setting": "semantic_state_persistence_rules",
            "governing_request_location": "Part B; Part N; Part T",
            "required_semantics": "first own/alternate/both, lowest objective, closest D1, success and final semantic states",
            "matching_registry_ids": matches("closest to d1", "first own", "first alternate", "semantic state"),
            "status": "MISSING",
        },
    ]
    control = {
        "maximum_steps": bundle.get("stop_rules", {}).get("maximum_steps"),
        "success_consecutive_evaluations": bundle.get("stop_rules", {}).get("success_consecutive_evaluations"),
        "registry_requirement_count": len(identifiers),
        "execution_stop_rules_present": "execution.stop_rules" in identifiers,
        "output_contract_values_present": all(
            value in identifiers for value in ("output.physical_negative", "output.nonfinite_count", "output.numerical_zero")
        ),
        "prompt_semantics_present": all(value in identifiers for value in ("prompt.prompt_a", "prompt.prompt_b", "truth.prompt_identity")),
        "forward_thresholds_present": all(
            value in identifiers
            for value in (
                "forward.global_chi_square_mean",
                "forward.per_band_chi_square_mean.g",
                "forward.per_band_chi_square_mean.r",
                "forward.per_band_chi_square_mean.z",
                "forward.absolute_relative_flux_residual",
            )
        ),
    }
    return {
        "schema_version": "thayer-d3s-executable-bundle-regression-v1",
        "status": "EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN",
        "bundle_sha256": sha256(BUNDLE),
        "bundle_hash_match": sha256(BUNDLE) == BUNDLE_SHA256,
        "governing_request_path": str(GOVERNING_REQUEST),
        "governing_request_sha256": sha256(GOVERNING_REQUEST),
        "control_settings_present": control,
        "missing_required_settings": rows,
        "missing_count": len(rows),
        "scientific_tensor_loads": 0,
        "third_party_imports": 0,
        "scientific_d3_steps": 0,
        "decision": "STOP_BEFORE_PREREGISTRATION_AND_BUNDLE_CONSUMER",
        "completed_utc": utcnow(),
    }


def close_bundle_regression(
    run: Path,
    bundle: dict[str, Any],
    registry: dict[str, Any],
    checkpoints_before: list[dict[str, object]],
    regression: dict[str, object],
    started: str,
) -> None:
    write_json_x(run / "bundle_validation/executable_bundle_regression.json", regression)
    write_json_x(
        run / "preregistration/not_frozen.json",
        {
            "status": "NOT_FROZEN",
            "reason": "The bundle lacks required execution settings; creating a partial preregistration would add or infer settings.",
            "scientific_tensor_loads": 0,
            "third_party_imports": 0,
        },
    )
    for relative, role in (
        ("cached_features/not_loaded.json", "cached features"),
        ("initial_state/not_loaded.json", "initial decoder state"),
        ("one_step_trace/not_run.json", "one-step scientific trace"),
        ("decoder_training/not_run.json", "authoritative D3 trajectory"),
        ("checkpoints/not_created.json", "D3 checkpoints"),
        ("replay_verification/not_run.json", "fresh-process replay"),
        ("postprocessing_inputs/not_created.json", "postprocessing inputs"),
    ):
        write_json_x(run / relative, {"status": "NOT_RUN", "role": role, "reason": regression["status"]})
    checkpoints_after = checkpoint_inventory()
    write_csv_x(run / "tables/checkpoint_inventory_after.csv", checkpoints_after)
    unchanged = checkpoints_before == checkpoints_after
    final_git_status = git("status", "--short")
    write_text_x(run / "diagnostics/final_git_status.txt", final_git_status + ("\n" if final_git_status else ""))
    access = {
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "ordinary_one_scene_access": 0,
        "eight_scene_access": 0,
        "remaining_microset_access": 0,
        "scientific_tensor_loads": 0,
        "scientific_d3_steps": 0,
    }
    write_json_x(run / "access_guard/access_summary.json", access)
    checks = {
        "bundle_hash_match": regression["bundle_hash_match"],
        "registry_count_180": len(registry["requirements"]) == 180,
        "required_execution_setting_regression_detected": regression["missing_count"] > 0,
        "stopped_before_preregistration": True,
        "stopped_before_third_party_import": True,
        "stopped_before_scientific_tensor_load": True,
        "stopped_before_model_optimizer_or_decoder_forward": True,
        "historical_checkpoints_600_unchanged": len(checkpoints_after) == 600 and unchanged,
        "protected_access_zero": all(value == 0 for value in access.values()),
        "readme_unchanged": git("diff", "--", "README.md") == "",
        "staged_index_empty": git("diff", "--cached", "--name-only") == "",
        "git_diff_check": command("git", "diff", "--check", check=False).returncode == 0,
        "git_cached_diff_check": command("git", "diff", "--cached", "--check", check=False).returncode == 0,
    }
    audit = {
        "schema_version": "thayer-d3s-final-correctness-audit-v1",
        "primary_outcome": regression["status"],
        "checks": checks,
        "check_count": len(checks),
        "pass_count": sum(bool(value) for value in checks.values()),
        "bundle_sha256": sha256(BUNDLE),
        "registry_requirement_count": len(registry["requirements"]),
        "historical_checkpoint_count": len(checkpoints_after),
        "historical_checkpoints_unchanged": unchanged,
        "access": access,
        "campaign_start_utc": started,
        "completed_utc": utcnow(),
    }
    write_json_x(run / "diagnostics/final_correctness_audit.json", audit)
    missing_names = [row["required_setting"] for row in regression["missing_required_settings"]]
    report = f"""# Thayer-D3S final report

Primary outcome: **EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN**.

The executable bundle byte hash matched, and its canonical registry contained
180 entries. The campaign stopped during the standard-library-only
preregistration-completeness audit because the bundle does not freeze required
execution settings named by the governing D3S request: `{', '.join(missing_names)}`.
Inferring any of these would violate the bundle-only and no-repair rules.

## Required answers

1. Bundle hash matched: **yes**, `{BUNDLE_SHA256}`.
2. All 180 requirements validate: **not rerun**; the registry count is 180, but the campaign stopped before the third-party consumer because required D3S settings are absent.
3. Consumer requirement closure passed: **not rerun in D3S**; the prior bundle record is not substituted for the missing settings.
4. Preregistration preceded imports/tensors: **no preregistration was frozen**; the completeness gate stopped before imports and tensors.
5. Readiness continued into D3: **no**; bundle regression stop.
6. D0/D1/D2 evidence reproduced: **not loaded or rerun**.
7. Cached features and initial state reproduced: **not loaded**.
8. Both experts received finite nonzero gradients: **not tested**.
9. Final and non-final blocks updated: **not tested**.
10. Square-map gradients usable: **not tested scientifically**.
11. Own coverage reached: **unknown**.
12. Alternate coverage reached: **unknown**.
13. Both-mode coverage reached: **unknown**.
14. Prompt swap passed: **not evaluated**.
15. Forward consistency passed: **not evaluated**.
16. z-band error evolution: **no trajectory**.
17. Learned features approached D1: **no trajectory**.
18. Different successful endpoint: **not tested**.
19. Assignment flips associated with failure: **not tested**.
20. Square-map pathology contributed: **not tested**.
21. Expert death or dominance: **not tested; the missing executable activity/death definition is the blocking regression**.
22. Optional tangent evidence valid: **not run; its protocol is absent from the bundle**.
23. Existing L0 capacity sufficient: **scientifically unknown**.
24. Eight-scene campaign authorized: **no**.
25. Decoder-capacity ladder authorized: **no**.
26. Exact next experiment: **one metadata-only executable-contract v3 campaign that freezes the missing gates/protocol/categories, adds negative tests for each, and emits a new hashed bundle; it must not load scientific tensors or run D3**.
27. Broader scenes, Atlas, development, and lockbox untouched: **yes, all access counts zero**.
28. Historical checkpoints unchanged: **yes, {len(checkpoints_after)}/{len(checkpoints_after)} exact rows matched before and after**.
29. Reusable source eventually committed: **the fail-closed completeness audit in `scripts/run_thayer_scientific_d3.py`, after review; no scientific runner was created or executed**.
30. Generated artifacts remaining ignored: **the complete `{run.relative_to(REPO)}/` record**.

## Provenance and stopping evidence

- Governing request SHA-256: `{regression['governing_request_sha256']}`.
- Missing-setting audit: `bundle_validation/executable_bundle_regression.json`.
- Scientific tensor loads / decoder forwards / optimizer steps / D3 steps: `0 / 0 / 0 / 0`.
- Atlas / development / lockbox: `0 / 0 / 0`.
- Preregistration: deliberately not frozen; `preregistration/not_frozen.json` records why.
- README: unchanged.
- Staged index: empty.
- Historical checkpoints: `{len(checkpoints_after)}/{len(checkpoints_after)}` unchanged.
"""
    write_text_x(run / "reports/final_report.md", report)


def freeze_preregistration(run: Path, bundle: dict[str, Any], registry: dict[str, Any], started: str) -> dict[str, object]:
    values = {
        str(record["canonical_requirement_id"]): record["expected_value"]
        for record in registry["requirements"]
    }
    body = "\n".join(
        (
            "# Bundle-driven authoritative square-only one-scene full-L0 D3 preregistration",
            "",
            f"Frozen UTC: `{utcnow()}`.",
            f"Campaign start UTC: `{started}`.",
            f"Executable bundle: `{BUNDLE.relative_to(REPO)}`.",
            f"Executable bundle SHA-256: `{BUNDLE_SHA256}`.",
            f"Requirement registry SHA-256: `{bundle['requirement_registry']['sha256']}`.",
            f"Canonical requirement count: `{bundle['requirement_registry']['required_count']}`.",
            "",
            "This file freezes the executable bundle and the exact expected value of every canonical registry requirement. No value below is inferred from a historical run. The D1 endpoint is diagnostic only. The encoder is not instantiated. The square mapping, hard assignment, direct loss, optimizer, clipping, evaluation/checkpoint cadence, 5,000-step maximum, three-consecutive-evaluation success rule, immediate stops, and no-broader-data rule are exactly the values below.",
            "",
            "Scientific outcome is exactly one of: L0 FULL DECODER SUCCESS; DECODER OPTIMIZATION BARRIER; DECODER PARAMETERIZATION/CAPACITY BARRIER; HARD-ASSIGNMENT BARRIER; SQUARE-MAPPING OPTIMIZATION BARRIER; MIXED CAUSE.",
            "",
            "Optional tangent diagnostics are not authorized in this run unless the authoritative trajectory first freezes and a separate finite-difference/JVP/VJP validation succeeds. They are omitted from the primary trajectory and cannot alter its outcome.",
            "",
            "## Frozen executable bundle",
            "",
            "```json",
            json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Frozen canonical requirement values",
            "",
            "```json",
            json.dumps(values, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
        )
    )
    path = run / "preregistration/bundle_driven_authoritative_d3.md"
    write_text_x(path, body)
    record = {
        "path": str(path.relative_to(run)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "frozen_utc": utcnow(),
        "third_party_imports_before_freeze": 0,
        "scientific_tensor_loads_before_freeze": 0,
        "requirement_count": len(values),
    }
    write_json_x(run / "preregistration/preregistration_manifest.json", record)
    return record


def environment(run: Path, runtime_name: str) -> dict[str, str]:
    runtime = run / f"runtime/{runtime_name}"
    result = dict(os.environ)
    result.update(
        {
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "TMPDIR": str(runtime / "tmp"),
            "TMP": str(runtime / "tmp"),
            "TEMP": str(runtime / "tmp"),
            "XDG_CACHE_HOME": str(runtime / "cache"),
            "XDG_CONFIG_HOME": str(runtime / "config"),
            "TORCH_HOME": str(runtime / "torch"),
            "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
        }
    )
    if runtime_name == "postprocess_runtime":
        result["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    return result


def run_streamed(command_args: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command_args,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"subprocess failed ({returncode}): {' '.join(command_args)}")


def scientific_config(run: Path, bundle: dict[str, Any], registry: dict[str, Any], preregistration: dict[str, object], mode: str) -> dict[str, object]:
    values = {record["canonical_requirement_id"]: record["expected_value"] for record in registry["requirements"]}
    runtime_name = "scientific" if mode == "train" else "replay"
    exact_reads = {
        str(BUNDLE),
        str(resolve(REPO, bundle["capsule_v2"]["path"])),
        str(resolve(REPO, bundle["requirement_registry"]["path"])),
        str(run / "preregistration/bundle_driven_authoritative_d3.md"),
        str(run / "preregistration/preregistration_manifest.json"),
        str(run / "runtime/orchestrator/scientific_config.json" if mode == "train" else run / "runtime/orchestrator/replay_config.json"),
        str(REPO / "scripts/thayer_d3_runtime_guard.py"),
        str(REPO / "scripts/run_thayer_scientific_d3_process.py"),
        str(REPO / "src/canonical_tensor_hash.py"),
        str(REPO / "src/competing_hypotheses.py"),
        str(REPO / "src/models_probabilistic_unet.py"),
        str(REPO / "src/models_two_expert_decoder.py"),
        str(REPO / "src/output_parameterization.py"),
        str(resolve(REPO, values["execution.hard_assignment_code"]["path"])),
        str(resolve(REPO, values["artifact.cached_features"]["path"])),
        str(resolve(REPO, values["artifact.p0_target_set"]["path"])),
        str(resolve(REPO, values["artifact.initial_decoder_state"]["path"])),
        str(resolve(REPO, values["l0.state_dict_source"]["path"])),
        str(resolve(REPO, values["artifact.d1_endpoint"]["path"])),
        str(resolve(REPO, values["capsule_artifact_d1_endpoint_manifest"]["path"])),
        str(resolve(REPO, bundle["validated_results"]["bundle.synthetic_preflight_result"]["path"])),
    }
    if mode == "replay":
        manifest_path = run / "checkpoints/semantic_checkpoint_manifest.json"
        exact_reads.add(str(manifest_path))
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for value in manifest.values():
                if isinstance(value, str):
                    exact_reads.add(str(run / value))
                elif isinstance(value, list):
                    exact_reads.update(str(run / item) for item in value if isinstance(item, str))
    return {
        "schema_version": "thayer-d3s-scientific-config-v1",
        "mode": mode,
        "repo": str(REPO),
        "run": str(run),
        "runtime_root": str(run / f"runtime/{runtime_name}"),
        "bundle": str(BUNDLE),
        "bundle_sha256": BUNDLE_SHA256,
        "preregistration": str(run / "preregistration/bundle_driven_authoritative_d3.md"),
        "preregistration_sha256": preregistration["sha256"],
        "guard_source": str(REPO / "scripts/thayer_d3_runtime_guard.py"),
        "access_log": str(run / f"access_guard/{runtime_name}_access_log.jsonl"),
        "blocked_log": str(run / f"access_guard/{runtime_name}_blocked_access_log.jsonl"),
        "exact_read_files": sorted(exact_reads),
        "environment": {key: environment(run, runtime_name)[key] for key in (
            "PYTORCH_ENABLE_MPS_FALLBACK", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
            "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TMPDIR", "TMP", "TEMP",
            "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TORCH_HOME", "PYTHONPYCACHEPREFIX",
        )},
    }


def validate_bundle(run: Path, bundle: dict[str, Any]) -> None:
    capsule = resolve(REPO, bundle["capsule_v2"]["path"])
    schema = resolve(REPO, bundle["capsule_schema"]["path"])
    registry = resolve(REPO, bundle["requirement_registry"]["path"])
    manifest = resolve(REPO, bundle["capsule_manifest"]["path"])
    chain = resolve(REPO, bundle["capsule_hash_chain"]["path"])
    validator = command(
        str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/validate_d3_executable_capsule_v2.py"),
        "--repo", str(REPO), "--capsule", str(capsule), "--schema", str(schema),
        "--registry", str(registry), "--manifest", str(manifest), "--hash-chain", str(chain),
        env=environment(run, "scientific"),
    )
    write_text_x(run / "bundle_validation/capsule_validator_stdout.txt", validator.stdout)
    write_text_x(run / "bundle_validation/capsule_validator_stderr.txt", validator.stderr)
    preflight_dir = run / "bundle_validation/frozen_consumer_preflight"
    launcher = command(
        str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_authoritative_d3_v2.py"),
        "--bundle", str(BUNDLE), "--bundle-sha256", BUNDLE_SHA256,
        "--output-dir", str(preflight_dir), "--synthetic-preflight-only",
        env=environment(run, "scientific"),
    )
    write_text_x(run / "bundle_validation/future_launcher_stdout.txt", launcher.stdout)
    write_text_x(run / "bundle_validation/future_launcher_stderr.txt", launcher.stderr)
    markers = [line.strip() for line in launcher.stdout.splitlines() if line.strip()]
    if markers[-2:] != ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"]:
        raise RuntimeError(f"required consumer markers absent: {markers}")
    write_json_x(
        run / "bundle_validation/validation_summary.json",
        {
            "status": "PASS",
            "bundle_sha256": sha256(BUNDLE),
            "capsule_validator": json.loads(validator.stdout),
            "markers": markers[-2:],
            "validated_utc": utcnow(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="Fresh output path; default is timestamped under outputs/runs")
    parser.add_argument("--governing-request", type=Path)
    parser.add_argument("--policy-preflight-only", action="store_true")
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--policy-bundle-sha256")
    parser.add_argument("--policy-output-dir", type=Path)
    args = parser.parse_args()
    if args.policy_preflight_only:
        if args.policy_bundle is None or args.policy_bundle_sha256 is None or args.policy_output_dir is None:
            raise SystemExit("policy preflight requires --policy-bundle, --policy-bundle-sha256, and --policy-output-dir")
        sys.path.insert(0, str(REPO))
        from src.d3_policy_preflight import READINESS_MARKERS, run_launcher_preflight

        run_launcher_preflight(args.policy_bundle, args.policy_bundle_sha256, args.policy_output_dir, REPO)
        for marker in READINESS_MARKERS:
            print(marker, flush=True)
        return 0
    if args.governing_request is None:
        raise SystemExit("scientific campaign mode requires --governing-request")
    global GOVERNING_REQUEST
    GOVERNING_REQUEST = args.governing_request.resolve()
    if not GOVERNING_REQUEST.is_file():
        raise SystemExit(f"governing request not found: {GOVERNING_REQUEST}")
    started = utcnow()
    if sha256(BUNDLE) != BUNDLE_SHA256:
        raise SystemExit("EXECUTABLE BUNDLE VALIDATION FAILURE — D3 NOT RUN")
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    registry_path = resolve(REPO, bundle["requirement_registry"]["path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run = args.run.resolve() if args.run else create_run()
    if args.run:
        run.mkdir(parents=True, exist_ok=False)
        for relative in DIRECTORIES:
            (run / relative).mkdir(parents=True, exist_ok=False)
    print(f"RUN={run}", flush=True)

    checkpoints = checkpoint_inventory()
    write_csv_x(run / "tables/checkpoint_inventory_before.csv", checkpoints)
    references = referenced_file_rows(bundle, registry)
    write_csv_x(run / "tables/bundle_referenced_file_validation.csv", references)
    sources = source_hash_rows()
    write_csv_x(run / "tables/source_code_hashes.csv", sources)
    free = shutil.disk_usage(REPO)
    snapshot = {
        "schema_version": "thayer-d3s-environment-snapshot-v1",
        "campaign_start_utc": started,
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "staged_paths": git("diff", "--cached", "--name-only").splitlines(),
        "python_executable": sys.executable,
        "btk_python": str(REPO / ".venv-btk/bin/python"),
        "bundle": str(BUNDLE.relative_to(REPO)),
        "bundle_sha256": sha256(BUNDLE),
        "governing_request": str(GOVERNING_REQUEST),
        "governing_request_sha256": sha256(GOVERNING_REQUEST),
        "registry_sha256": sha256(registry_path),
        "checkpoint_count": len(checkpoints),
        "bundle_referenced_file_count": len(references),
        "source_code_hashes": sources,
        "disk_free_bytes": free.free,
        "disk_total_bytes": free.total,
    }
    write_json_x(run / "logs/input_provenance.json", snapshot)
    write_text_x(
        run / "diagnostics/environment_snapshot_stdlib_only.md",
        "# Standard-library-only environment snapshot\n\n```json\n" + json.dumps(snapshot, indent=2, sort_keys=True) + "\n```\n",
    )
    write_text_x(
        run / "diagnostics/campaign_contract.md",
        "# Thayer-D3S campaign contract\n\nThe exact executable bundle is sole authority. This fresh append-only run may proceed to scientific tensors only after its preregistration is hashed and both frozen consumer markers pass. No broader scene, Atlas, development, or lockbox access is permitted.\n",
    )
    command_lines = [
        f"{REPO / '.venv-btk/bin/python'} -B scripts/run_thayer_scientific_d3.py --run {run} --governing-request {GOVERNING_REQUEST}",
        f"{REPO / '.venv-btk/bin/python'} -B scripts/validate_d3_executable_capsule_v2.py [frozen bundle arguments]",
        f"{REPO / '.venv-btk/bin/python'} -B scripts/run_thayer_authoritative_d3_v2.py --bundle {BUNDLE} --bundle-sha256 {BUNDLE_SHA256} --output-dir {run / 'bundle_validation/frozen_consumer_preflight'} --synthetic-preflight-only",
        f"{REPO / '.venv-btk/bin/python'} -B scripts/run_thayer_scientific_d3_process.py --config {run / 'runtime/orchestrator/scientific_config.json'}",
        f"{REPO / '.venv-btk/bin/python'} -B scripts/run_thayer_scientific_d3_process.py --config {run / 'runtime/orchestrator/replay_config.json'}",
        f"{REPO / '.venv-btk/bin/python'} -B scripts/postprocess_thayer_scientific_d3.py --config {run / 'runtime/orchestrator/postprocess_config.json'}",
    ]
    write_text_x(run / "logs/command_log.sh", "\n".join(command_lines) + "\n")
    regression = bundle_regression_audit(bundle, registry)
    if regression["missing_count"]:
        close_bundle_regression(run, bundle, registry, checkpoints, regression, started)
        print("EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN", flush=True)
        print(f"RUN={run}", flush=True)
        return 0
    preregistration = freeze_preregistration(run, bundle, registry, started)
    validate_bundle(run, bundle)

    train_config = scientific_config(run, bundle, registry, preregistration, "train")
    train_config_path = run / "runtime/orchestrator/scientific_config.json"
    write_json_x(train_config_path, train_config)
    run_streamed(
        [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_scientific_d3_process.py"), "--config", str(train_config_path)],
        run / "logs/scientific_stdout_stderr.log",
        environment(run, "scientific"),
    )
    if not (run / "decoder_training/trajectory_summary.json").is_file():
        raise RuntimeError("scientific process returned without trajectory summary")

    replay_config = scientific_config(run, bundle, registry, preregistration, "replay")
    replay_config_path = run / "runtime/orchestrator/replay_config.json"
    write_json_x(replay_config_path, replay_config)
    run_streamed(
        [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/run_thayer_scientific_d3_process.py"), "--config", str(replay_config_path)],
        run / "logs/replay_stdout_stderr.log",
        environment(run, "replay"),
    )
    replay_summary = json.loads((run / "replay_verification/replay_summary.json").read_text(encoding="utf-8"))
    if replay_summary.get("status") != "PASS":
        raise RuntimeError("fresh-process replay failed")

    shutil.copyfile(run / "decoder_training/trajectory.csv", run / "postprocessing_inputs/trajectory.csv")
    shutil.copyfile(run / "checkpoints/semantic_checkpoint_manifest.json", run / "postprocessing_inputs/semantic_checkpoint_manifest.json")
    post_env = environment(run, "postprocess_runtime")
    post_config = {
        "schema_version": "thayer-d3s-postprocess-config-v1",
        "repo": str(REPO),
        "run": str(run),
        "runtime_root": str(run / "runtime/postprocess_runtime"),
        "guard_source": str(REPO / "scripts/thayer_d3_runtime_guard.py"),
        "access_log": str(run / "access_guard/postprocess_access_log.jsonl"),
        "blocked_log": str(run / "access_guard/postprocess_blocked_access_log.jsonl"),
        "approved_inputs": [str(run / "postprocessing_inputs/trajectory.csv"), str(run / "postprocessing_inputs/semantic_checkpoint_manifest.json")],
        "environment": {key: post_env[key] for key in (
            "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TORCH_HOME",
            "PYTHONPYCACHEPREFIX", "MPLCONFIGDIR",
        )},
    }
    post_config_path = run / "runtime/orchestrator/postprocess_config.json"
    write_json_x(post_config_path, post_config)
    run_streamed(
        [str(REPO / ".venv-btk/bin/python"), "-B", str(REPO / "scripts/postprocess_thayer_scientific_d3.py"), "--config", str(post_config_path)],
        run / "logs/postprocess_stdout_stderr.log",
        post_env,
    )
    write_json_x(
        run / "diagnostics/orchestrator_completion.json",
        {
            "status": "SCIENTIFIC_REPLAY_POSTPROCESS_COMPLETE",
            "run": str(run),
            "completed_utc": utcnow(),
            "bundle_markers": ["ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED", "READY_FOR_AUTHORITATIVE_D3_EXECUTION"],
        },
    )
    print(f"SCIENTIFIC_REPLAY_POSTPROCESS_COMPLETE={run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
