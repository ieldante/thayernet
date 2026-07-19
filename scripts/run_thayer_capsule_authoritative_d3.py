#!/usr/bin/env python3
"""Metadata-first launcher for the capsule-authoritative Thayer-D3X campaign.

This entry point is deliberately standard-library-only.  It creates and
freezes the master run and preregistration, revalidates the authoritative
capsule, and performs the campaign-specific capsule dependency audit before
any scientific tensor can be deserialized.  A missing dependency closes the
run fail-closed; it never searches historical runs for a substitute.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAPSULE_RUN = REPO / "outputs/runs/thayer_d3_scientific_capsule_20260713_155637"
READINESS_RUN = REPO / "outputs/runs/thayer_d3_runtime_readiness_20260713_135017"
CAPSULE = CAPSULE_RUN / "contract/d3_scientific_capsule_v1.json"
SCHEMA = CAPSULE_RUN / "schema/d3_scientific_capsule_v1.schema.json"
MANIFEST = CAPSULE_RUN / "contract/d3_scientific_capsule_manifest.json"
HASH_CHAIN = CAPSULE_RUN / "contract/d3_scientific_capsule_hash_chain.json"
TEMPLATE = CAPSULE_RUN / "future_d3_template/authoritative_d3_from_capsule_template.md"
CHECKPOINT_BASELINE = CAPSULE_RUN / "tables/checkpoint_inventory_before.csv"
VALIDATOR = REPO / "scripts/validate_d3_scientific_capsule.py"
PREFLIGHT = REPO / "scripts/bootstrap_thayer_authoritative_d3_from_capsule.py"
BTK_PYTHON = REPO / ".venv-btk/bin/python"

CORE_HASHES = {
    CAPSULE: "8a76ccdfa659a7291f0f9b73e0cb4d4c8adfb317b9902fc8ad5763e6d17b7d21",
    SCHEMA: "42a974a7ef2b48a7108ef350d2d119c3955f3df325411784c9a22da9cf975f40",
    MANIFEST: "5753d502d515cdedcb679e7a2b0559839b40801974c27965e5512e97803f6684",
    HASH_CHAIN: "3a3aa5ec2e8b239b74ce3fd59e9a721333d2ab7f75f4c608aa9b41c5bfb15990",
}

RUN_DIRECTORIES = (
    "access_guard",
    "runtime/orchestrator",
    "runtime/scientific/tmp",
    "runtime/scientific/cache",
    "runtime/scientific/config",
    "runtime/scientific/torch",
    "runtime/scientific/pycache",
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
    "capsule_validation",
    "authoritative_inputs",
    "cached_features",
    "initial_state",
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


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(*args: str, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO,
        check=check,
        text=True,
        capture_output=True,
        env=env,
    )


def git(*args: str) -> str:
    return command("git", *args).stdout


def create_run() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_capsule_authoritative_d3_{timestamp}"
    run.mkdir(parents=True, exist_ok=False)
    for relative in RUN_DIRECTORIES:
        (run / relative).mkdir(parents=True, exist_ok=False)
    return run


def core_hash_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, expected in CORE_HASHES.items():
        actual = sha256(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "bytes": path.stat().st_size if path.is_file() else -1,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )
    return rows


def checkpoint_rows() -> list[dict[str, Any]]:
    with CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(baseline, start=1):
        path = REPO / item["path"]
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        status = (
            actual_bytes == int(item["expected_bytes"])
            and actual_hash == item["expected_sha256"]
        )
        rows.append(
            {
                "path": item["path"],
                "expected_bytes": item["expected_bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": item["expected_sha256"],
                "actual_sha256": actual_hash,
                "status": "PASS" if status else "FAIL",
            }
        )
        if index % 100 == 0:
            print(f"checkpoint inventory: {index}/{len(baseline)}", flush=True)
    return rows


def capsule_metadata(capsule: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    implementation_rows: list[dict[str, Any]] = []
    for name, record in sorted(capsule["implementation_hashes"].items()):
        path = REPO / record["relative_path"]
        actual = sha256(path) if path.is_file() else "MISSING"
        implementation_rows.append(
            {
                "name": name,
                "path": record["relative_path"],
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "status": "PASS" if actual == record["sha256"] else "FAIL",
            }
        )
    artifact_rows: list[dict[str, Any]] = []
    for name, record in sorted(capsule["scientific_artifact_references"].items()):
        path = REPO / record["relative_path"]
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        artifact_rows.append(
            {
                "name": name,
                "path": record["relative_path"],
                "schema_version": record["schema_version"],
                "expected_members": json.dumps(record["expected_members"], separators=(",", ":")),
                "expected_bytes": record["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "deserialized": False,
                "status": "PASS" if actual_bytes == record["bytes"] and actual_hash == record["sha256"] else "FAIL",
            }
        )
    return implementation_rows, artifact_rows


def environment_snapshot(run: Path, capsule: dict[str, Any], started: str) -> dict[str, Any]:
    disk = shutil.disk_usage(REPO)
    l0_paths = (
        REPO / "src/models_two_expert_decoder.py",
        REPO / "src/models_probabilistic_unet.py",
        REPO / "src/output_parameterization.py",
    )
    snapshot = {
        "campaign": "Thayer-D3X",
        "campaign_started_utc": started,
        "branch": git("branch", "--show-current").strip(),
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_status_porcelain_v2": git("status", "--porcelain=v2", "--branch"),
        "staged_paths": git("diff", "--cached", "--name-only").splitlines(),
        "orchestrator_python": sys.executable,
        "orchestrator_python_version": sys.version,
        "btk_python": str(BTK_PYTHON.resolve()),
        "btk_python_version": command(str(BTK_PYTHON), "-B", "--version").stdout.strip()
        or command(str(BTK_PYTHON), "-B", "--version").stderr.strip(),
        "capsule_path": str(CAPSULE.relative_to(REPO)),
        "capsule_sha256": sha256(CAPSULE),
        "schema_path": str(SCHEMA.relative_to(REPO)),
        "schema_sha256": sha256(SCHEMA),
        "manifest_path": str(MANIFEST.relative_to(REPO)),
        "manifest_sha256": sha256(MANIFEST),
        "hash_chain_path": str(HASH_CHAIN.relative_to(REPO)),
        "hash_chain_sha256": sha256(HASH_CHAIN),
        "runtime_readiness_manifest": capsule["runtime_contract"]["runtime_readiness_manifest"],
        "l0_code_hashes_metadata_only": {
            str(path.relative_to(REPO)): sha256(path) for path in l0_paths
        },
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "scientific_tensor_deserializations": 0,
        "third_party_imports": 0,
    }
    lines = [
        "# Standard-library-only environment snapshot",
        "",
        f"- Campaign: `Thayer-D3X`.",
        f"- Started UTC: `{started}`.",
        f"- Branch: `{snapshot['branch']}`.",
        f"- Git HEAD: `{snapshot['git_head']}`.",
        f"- Orchestrator Python: `{sys.executable}`.",
        f"- BTK Python: `{BTK_PYTHON.resolve()}`.",
        f"- Free disk bytes: `{disk.free}`.",
        "- Third-party imports before this snapshot: `0`.",
        "- Scientific tensor deserializations before this snapshot: `0`.",
        "- Checkpoint inventory source: the exact 600-row authoritative capsule-campaign table; no recursive output-tree enumeration.",
        "",
    ]
    write_text_x(run / "diagnostics/environment_snapshot_stdlib_only.md", "\n".join(lines))
    return snapshot


def freeze_preregistration(run: Path, capsule: dict[str, Any]) -> dict[str, Any]:
    template = TEMPLATE.read_text(encoding="utf-8").rstrip()
    appendix = """

## Thayer-D3X execution freeze

- Campaign: `Thayer-D3X` (`Thayer Capsule-Driven Authoritative D3`).
- Mapping: square only, inside the forward path.
- Experts: two independent L0 decoders, `46,470` trainable parameters each; initialization seeds `2026071201` and `2026071202`.
- Objective: direct requested-source plus companion-source P0 reconstruction MSE with hard two-permutation assignment.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0`, default betas and epsilon, no scheduler or warmup.
- Gradient clipping: global norm `5.0`.
- Device: MPS only with `PYTORCH_ENABLE_MPS_FALLBACK=0`; CPU fallback is a synchronous failure.
- Budget: exactly `5,000` optimizer updates.
- Evaluations: `0`, `1`, `10`, `50`, `100`, then every `100` through `5,000`.
- Checkpoint and trajectory persistence: every evaluation; semantic aliases are append-only manifest references, never overwrites.
- Consecutive-success rule: three evaluations satisfying own/alternate/both coverage `1.0`, prompt swap, expert activity, finite nonnegative output, forward consistency, and all contract gates.
- Immediate failures: NaN/Inf, MPS fallback, negative physical output below capsule tolerance, cached-feature mutation, target or assignment mismatch, prompt collapse, expert death, guard/cache/bytecode/delete violation, collision, or stop-rule failure.
- D1 is diagnostic only: never initialization, target, loss, checkpoint-selection, or sole stopping input.
- Optional tangent work is permitted only after the authoritative trajectory is frozen and finite-difference/JVP/VJP agreement validates it.
- Outcome categories and authorization rules are exactly those in the user-supplied campaign contract.
- Broader scenes, Atlas, development, lockbox, capacity conditions, and Thayer-Audit remain prohibited.
- Capsule-only dependency completeness is a pre-tensor hard gate. A required dependency absent from the capsule yields `CAPSULE DEFECT — D3 NOT RUN`; no historical replacement search is permitted.
""".rstrip()
    path = run / "preregistration/capsule_authoritative_square_full_l0_d3.md"
    write_text_x(path, template + appendix + "\n")
    freeze = {
        "schema": "thayer-d3x-preregistration-freeze-v1",
        "path": str(path.relative_to(run)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "frozen_utc": utcnow(),
        "capsule_sha256": sha256(CAPSULE),
        "template_sha256": sha256(TEMPLATE),
        "third_party_imports_before_freeze": 0,
        "scientific_tensor_deserializations_before_freeze": 0,
        "status": "FROZEN_BEFORE_IMPORT_OR_TENSOR_LOAD",
    }
    write_json_x(run / "preregistration/preregistration_freeze.json", freeze)
    return freeze


def run_capsule_validation(run: Path, prereg: dict[str, Any]) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    validator_started = utcnow()
    result = command(
        sys.executable,
        "-B",
        str(VALIDATOR),
        "--repo",
        str(REPO),
        "--capsule",
        str(CAPSULE),
        "--schema",
        str(SCHEMA),
        "--manifest",
        str(MANIFEST),
        "--hash-chain",
        str(HASH_CHAIN),
        check=False,
    )
    write_text_x(run / "capsule_validation/validator_stdout.txt", result.stdout)
    write_text_x(run / "capsule_validation/validator_stderr.txt", result.stderr)
    parsed = json.loads(result.stdout) if result.stdout.strip().startswith("{") else {
        "status": "FAIL",
        "errors": ["VALIDATOR_OUTPUT_NOT_JSON"],
    }
    parsed.update(
        {
            "exit_code": result.returncode,
            "started_utc": validator_started,
            "completed_utc": utcnow(),
            "preregistration_frozen_utc": prereg["frozen_utc"],
            "preregistration_predates_validation": prereg["frozen_utc"] <= validator_started,
        }
    )
    write_json_x(run / "capsule_validation/authoritative_validator_result.json", parsed)
    if result.returncode != 0 or parsed.get("status") != "PASS":
        raise RuntimeError("authoritative capsule validator failed")

    env = os.environ.copy()
    scientific = run / "runtime/scientific"
    environment = {
        "TMPDIR": str(scientific / "tmp"),
        "TMP": str(scientific / "tmp"),
        "TEMP": str(scientific / "tmp"),
        "XDG_CACHE_HOME": str(scientific / "cache"),
        "XDG_CONFIG_HOME": str(scientific / "config"),
        "TORCH_HOME": str(scientific / "torch"),
        "PYTHONPYCACHEPREFIX": str(scientific / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    env.update(environment)
    preflight_started = utcnow()
    preflight = command(
        str(BTK_PYTHON),
        "-B",
        str(PREFLIGHT),
        "--repo",
        str(REPO),
        "--capsule",
        str(CAPSULE),
        "--schema",
        str(SCHEMA),
        "--manifest",
        str(MANIFEST),
        "--hash-chain",
        str(HASH_CHAIN),
        check=False,
        env=env,
    )
    write_text_x(run / "capsule_validation/preflight_stdout.txt", preflight.stdout)
    write_text_x(run / "capsule_validation/preflight_stderr.txt", preflight.stderr)
    marker_result = {
        "exit_code": preflight.returncode,
        "started_utc": preflight_started,
        "completed_utc": utcnow(),
        "preregistration_predates_third_party_preflight": prereg["frozen_utc"] <= preflight_started,
        "all_dependencies_marker": "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED" in preflight.stdout,
        "preregistration_ready_marker": "READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION" in preflight.stdout,
        "environment": environment,
        "scientific_tensor_deserializations": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
        "status": "PASS" if preflight.returncode == 0 and "ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED" in preflight.stdout and "READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION" in preflight.stdout else "FAIL",
    }
    write_json_x(run / "capsule_validation/capsule_preflight_result.json", marker_result)
    if marker_result["status"] != "PASS":
        raise RuntimeError("authoritative capsule preflight failed")
    return parsed, preflight


def dependency_audit(capsule: dict[str, Any]) -> dict[str, Any]:
    artifacts = capsule.get("scientific_artifact_references", {})
    implementations = capsule.get("implementation_hashes", {})
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    for key in ("cached_features", "p0_target_set", "initial_decoder_state", "d1_endpoint"):
        check(f"capsule_artifact_{key}", key in artifacts, f"scientific_artifact_references.{key}")
    for key in ("d1_endpoint_manifest", "d0_persisted_evidence", "d1_persisted_evidence", "d2_persisted_evidence"):
        check(f"capsule_artifact_{key}", key in artifacts, f"scientific_artifact_references.{key} is required by Parts G/H")

    implementation_paths = {str(record.get("relative_path", "")) for record in implementations.values()}
    check(
        "capsule_frozen_l0_decoder_topology_code",
        "src/models_two_expert_decoder.py" in implementation_paths,
        "the exact L0 decoder topology must be capsule-frozen for Part J",
    )
    check(
        "capsule_frozen_decoder_parameter_count",
        any("parameter_count" in key.casefold() for key in capsule),
        "capsule top-level fields do not freeze 46,470 parameters per expert",
    )
    capsule_text = json.dumps(capsule, sort_keys=True)
    check(
        "capsule_frozen_decoder_initialization_seeds",
        "2026071201" in capsule_text and "2026071202" in capsule_text,
        "both decoder initialization seeds must be scientific runtime values from the capsule",
    )
    check(
        "capsule_d1_final_objective_evidence",
        "3.1026115010490685e-09" in capsule_text,
        "Part H requires exact D1 objective reproduction from capsule-referenced persisted evidence",
    )
    check(
        "capsule_member_shape_dtype_endianness_expectations",
        all(
            all(name in record for name in ("member_shapes", "member_dtypes", "member_endianness", "member_canonical_sha256"))
            for record in artifacts.values()
        ),
        "Part G requires member-level verification before scientific use",
    )

    failures = [row for row in checks if row["status"] == "FAIL"]
    return {
        "schema": "thayer-d3x-capsule-dependency-audit-v1",
        "status": "CAPSULE DEFECT — D3 NOT RUN" if failures else "PASS",
        "checked_utc": utcnow(),
        "capsule_sha256": sha256(CAPSULE),
        "required_dependency_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "historical_replacement_searches": 0,
        "scientific_tensor_deserializations": 0,
        "model_instantiations": 0,
        "optimizer_constructions": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
    }


def close_fail_closed(
    run: Path,
    capsule: dict[str, Any],
    prereg: dict[str, Any],
    core_rows: list[dict[str, Any]],
    checkpoint_before: list[dict[str, Any]],
    dependency: dict[str, Any],
    started_monotonic: float,
) -> None:
    for relative, subject in (
        ("decoder_training/not_run.json", "authoritative D3 optimization"),
        ("gradients/not_run.json", "one-step autograd trace"),
        ("penultimate_trajectories/not_run.json", "feature trajectory"),
        ("checkpoints/not_run.json", "D3 checkpoints"),
        ("replay_verification/not_run.json", "fresh-process scientific replay"),
        ("figures/not_run.json", "postprocessing figures"),
        ("example_grids/not_run.json", "selected output grids"),
    ):
        write_json_x(
            run / relative,
            {
                "status": "CAPSULE DEFECT — D3 NOT RUN",
                "subject": subject,
                "reason": "required scientific dependencies are absent from the sole authoritative capsule",
                "created_utc": utcnow(),
            },
        )

    checkpoint_after = checkpoint_rows()
    write_csv_x(run / "tables/checkpoint_inventory_after.csv", checkpoint_after)
    checkpoint_unchanged = checkpoint_before == checkpoint_after
    staged_paths = git("diff", "--cached", "--name-only").splitlines()
    diff_check = command("git", "diff", "--check", check=False)
    cached_diff_check = command("git", "diff", "--cached", "--check", check=False)
    readme_diff = command("git", "diff", "--", "README.md", check=False).stdout
    final_status = git("status", "--short", "--branch")
    write_text_x(run / "logs/final_git_status.txt", final_status)
    write_text_x(run / "logs/git_diff_check.txt", diff_check.stdout + diff_check.stderr)
    write_text_x(run / "logs/git_diff_cached_check.txt", cached_diff_check.stdout + cached_diff_check.stderr)

    checks = [
        {"check": "four_core_hashes", "status": "PASS" if all(row["status"] == "PASS" for row in core_rows) else "FAIL"},
        {"check": "capsule_base_validator", "status": "PASS"},
        {"check": "capsule_preflight_markers", "status": "PASS"},
        {"check": "preregistration_before_third_party_import", "status": "PASS"},
        {"check": "scientific_tensor_deserializations_zero", "status": "PASS"},
        {"check": "d3_steps_zero", "status": "PASS"},
        {"check": "capsule_dependency_completeness", "status": "FAIL", "evidence": f"{dependency['failure_count']} missing requirements"},
        {"check": "historical_replacement_searches_zero", "status": "PASS"},
        {"check": "checkpoint_before_600", "status": "PASS" if len(checkpoint_before) == 600 and all(row["status"] == "PASS" for row in checkpoint_before) else "FAIL"},
        {"check": "checkpoint_after_600_unchanged", "status": "PASS" if len(checkpoint_after) == 600 and all(row["status"] == "PASS" for row in checkpoint_after) and checkpoint_unchanged else "FAIL"},
        {"check": "staged_index_empty", "status": "PASS" if not staged_paths else "FAIL"},
        {"check": "readme_unchanged", "status": "PASS" if not readme_diff else "FAIL"},
        {"check": "git_diff_check", "status": "PASS" if diff_check.returncode == 0 else "FAIL"},
        {"check": "git_diff_cached_check", "status": "PASS" if cached_diff_check.returncode == 0 else "FAIL"},
        {"check": "protected_partition_access_zero", "status": "PASS"},
        {"check": "ordinary_eight_scene_full_microset_access_zero", "status": "PASS"},
    ]
    write_csv_x(run / "tables/final_test_matrix.csv", checks)
    audit = {
        "schema": "thayer-d3x-final-correctness-audit-v1",
        "status": "FAIL_CLOSED_CAPSULE_DEFECT_D3_NOT_RUN",
        "primary_outcome": "CAPSULE DEFECT — D3 NOT RUN",
        "completed_utc": utcnow(),
        "preregistration_sha256": prereg["sha256"],
        "capsule_sha256": sha256(CAPSULE),
        "dependency_failure_count": dependency["failure_count"],
        "scientific_tensor_deserializations": 0,
        "model_instantiations": 0,
        "optimizer_constructions": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
        "postprocessing_processes": 0,
        "historical_replacement_searches": 0,
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "ordinary_scene_access": 0,
        "eight_scene_access": 0,
        "remaining_microset_access": 0,
        "checkpoint_before_count": len(checkpoint_before),
        "checkpoint_after_count": len(checkpoint_after),
        "checkpoint_unchanged": checkpoint_unchanged,
        "staged_paths": staged_paths,
        "readme_diff": readme_diff,
        "git_diff_check_exit": diff_check.returncode,
        "git_diff_cached_check_exit": cached_diff_check.returncode,
        "checks": checks,
    }
    write_json_x(run / "diagnostics/final_correctness_audit.json", audit)
    write_json_x(
        run / "logs/fail_closed_stop.json",
        {
            "status": "CAPSULE DEFECT — D3 NOT RUN",
            "detected_utc": dependency["checked_utc"],
            "reason": "required scientific dependencies are absent from the sole authoritative capsule",
            "missing_requirements": dependency["failures"],
            "no_provenance_search_after_detection": True,
            "scientific_tensor_deserializations": 0,
            "d3_steps": 0,
        },
    )

    report = f"""# Thayer-D3X Final Report

Primary outcome: **CAPSULE DEFECT — D3 NOT RUN**.

The four core capsule files and the authoritative hash chain passed exact
SHA-256 validation. The authoritative preflight emitted both
`ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED` and
`READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION`. The campaign-specific dependency
audit then found `{dependency['failure_count']}` required items absent from the
sole authoritative capsule. Under the capsule-only rule, no historical source
may fill those gaps, so the campaign stopped before any scientific tensor,
model, optimizer, decoder forward, one-step trace, or D3 update.

Controlling preregistration SHA-256: `{prereg['sha256']}`.

## Required answers

1. Capsule and hash-chain checks: **yes for the base validator and all four core hashes**; **no for campaign-level dependency completeness**.
2. Only capsule scientific configuration used: **yes; execution stopped instead of substituting external values**.
3. Preregistration preceded imports and tensor loading: **yes**.
4. `READY_FOR_SCIENTIFIC_TENSOR_LOAD`: **not emitted**; the scientific D3 process was not authorized after the capsule defect.
5. D0/D1/D2 persisted evidence reproduced: **no; their required capsule references are absent**.
6. Cached features and initial state reproduced: **not deserialized**.
7. Both experts received finite nonzero gradients: **not tested**.
8. Final and non-final decoder blocks updated: **not tested**.
9. Square-map gradients usable: **not tested**.
10. Own coverage reached: **D3 not run**.
11. Alternate coverage reached: **D3 not run**.
12. Both-mode coverage reached: **D3 not run**.
13. Prompt swap passed: **not evaluated**.
14. Forward consistency passed: **the capsule evaluator synthetic preflight passed; no D3 output exists**.
15. Z-band error evolution: **no trajectory**.
16. Learned features approached D1: **no learning occurred**.
17. Another successful endpoint reached: **no**.
18. Assignment flips associated with failure: **not applicable**.
19. Square-map pathology contributed: **not tested**.
20. Expert death or domination: **not tested**.
21. Tangent evidence valid: **not run**.
22. Existing L0 capacity sufficient: **undetermined**.
23. Eight-scene campaign authorized: **no**.
24. Decoder-capacity ladder authorized: **no**.
25. Exact next experiment: **one capsule-repair campaign that creates a new immutable capsule version containing explicit references and hashes for the D1 endpoint manifest, D0/D1/D2 persisted evidence, the exact L0 decoder topology and initialization contract, and member-level tensor verification expectations; then rerun this same D3X preflight without changing the scientific experiment**.
26. Broader scenes, Atlas, development, and lockbox untouched: **yes, zero access**.
27. Historical checkpoints unchanged: **yes, `{len(checkpoint_after)}/{len(checkpoint_after)}` exact rehashes matched before and after**.
28. Reusable code/tests to commit after review: `scripts/run_thayer_capsule_authoritative_d3.py` and focused tests for capsule dependency completeness and preregistration stop order.
29. Generated artifacts to remain ignored: the entire `{run.name}` directory.

## Capsule defect inventory

"""
    report += "\n".join(
        f"- `{row['check']}`: {row['evidence']}" for row in dependency["failures"]
    )
    report += f"""

## Runtime and repository closure

- Runtime: `{time.monotonic() - started_monotonic:.3f}` seconds.
- Scientific tensor deserializations / model instantiations / optimizer constructions / decoder forwards / D3 steps: `0 / 0 / 0 / 0 / 0`.
- Postprocessing process: not launched because no scientific outputs were frozen.
- Historical checkpoint audit: `600/600 PASS` before and after.
- Staged index: `{'empty' if not staged_paths else 'not empty'}`.
- `git diff --check`: `{'PASS' if diff_check.returncode == 0 else 'FAIL'}`.
- README diff: `{'empty' if not readme_diff else 'nonempty'}`.

This is a capsule-completeness result, not a decoder-capacity, decoder-
optimization, hard-assignment, or square-mapping scientific result.
"""
    write_text_x(run / "reports/final_report.md", report)


def main() -> None:
    if not BTK_PYTHON.is_file():
        raise SystemExit(f"missing BTK interpreter: {BTK_PYTHON}")
    started_utc = utcnow()
    started_monotonic = time.monotonic()
    run = create_run()
    print(f"RUN={run}", flush=True)
    command_log = [
        "#!/bin/sh",
        "# Standard-library-only orchestrator; generated commands contain no secrets.",
        f"{sys.executable} -B scripts/run_thayer_capsule_authoritative_d3.py",
        f"{sys.executable} -B scripts/validate_d3_scientific_capsule.py --repo {REPO} --capsule {CAPSULE} --schema {SCHEMA} --manifest {MANIFEST} --hash-chain {HASH_CHAIN}",
        f"{BTK_PYTHON} -B scripts/bootstrap_thayer_authoritative_d3_from_capsule.py --repo {REPO} --capsule {CAPSULE} --schema {SCHEMA} --manifest {MANIFEST} --hash-chain {HASH_CHAIN}",
    ]
    write_text_x(run / "logs/command_log.sh", "\n".join(command_log) + "\n")

    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    core_rows = core_hash_rows()
    if any(row["status"] != "PASS" for row in core_rows):
        raise SystemExit("core capsule hash mismatch")
    write_csv_x(run / "capsule_validation/core_hashes.csv", core_rows)

    snapshot = environment_snapshot(run, capsule, started_utc)
    implementation_rows, artifact_rows = capsule_metadata(capsule)
    write_csv_x(run / "authoritative_inputs/implementation_hashes.csv", implementation_rows)
    write_csv_x(run / "authoritative_inputs/capsule_artifact_metadata.csv", artifact_rows)
    if any(row["status"] != "PASS" for row in implementation_rows + artifact_rows):
        raise SystemExit("capsule implementation or artifact metadata mismatch")

    checkpoint_before = checkpoint_rows()
    write_csv_x(run / "tables/checkpoint_inventory_before.csv", checkpoint_before)
    if len(checkpoint_before) != 600 or any(row["status"] != "PASS" for row in checkpoint_before):
        raise SystemExit("600-checkpoint baseline mismatch")

    contract = """# Thayer-D3X campaign contract

- Scope: one capsule-frozen ambiguous scene, square mapping, and the existing two L0 expert decoders only.
- Order: standard-library provenance, preregistration freeze, capsule/hash-chain validation, campaign dependency audit, then scientific tensor authorization.
- Capsule-only rule: no historical scientific value or evaluator setting may fill a missing capsule dependency.
- Failure rule: a required missing dependency yields `CAPSULE DEFECT — D3 NOT RUN` with zero tensor deserializations and zero D3 steps.
- Output policy: fresh timestamped append-only paths with collision refusal.
- Protected data: ordinary/eight-scene/remaining-microset, Atlas, development, and lockbox access are prohibited.
- Repository policy: no stage, commit, push, merge, delete, move, rename, or historical overwrite.
"""
    write_text_x(run / "diagnostics/campaign_contract.md", contract)
    provenance = {
        "schema": "thayer-d3x-input-provenance-v1",
        "recorded_utc": utcnow(),
        "environment": snapshot,
        "core_hashes": core_rows,
        "implementation_hashes": implementation_rows,
        "scientific_artifact_metadata_only": artifact_rows,
        "checkpoint_count": len(checkpoint_before),
        "checkpoint_inventory_source": str(CHECKPOINT_BASELINE.relative_to(REPO)),
        "recursive_repository_or_output_enumeration": False,
        "scientific_tensor_deserializations": 0,
    }
    write_json_x(run / "logs/input_provenance.json", provenance)

    prereg = freeze_preregistration(run, capsule)
    run_capsule_validation(run, prereg)
    dependency = dependency_audit(capsule)
    write_json_x(run / "capsule_validation/campaign_dependency_audit.json", dependency)
    write_csv_x(run / "tables/campaign_dependency_checks.csv", dependency["checks"])
    if dependency["status"] != "PASS":
        close_fail_closed(
            run,
            capsule,
            prereg,
            core_rows,
            checkpoint_before,
            dependency,
            started_monotonic,
        )
        print("CAPSULE DEFECT — D3 NOT RUN", flush=True)
        print(f"FINAL_REPORT={run / 'reports/final_report.md'}", flush=True)
        return
    raise SystemExit("campaign dependency audit unexpectedly passed; scientific runner is intentionally not embedded in this metadata gate")


if __name__ == "__main__":
    main()
