#!/usr/bin/env python3
"""Standard-library bootstrap for the append-only Thayer-D3E campaign."""

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
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CAPSULE_RUN = REPO / "outputs/runs/thayer_d3_scientific_capsule_20260713_155637"
STOPPED_RUN = REPO / "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342"
READINESS_RUN = REPO / "outputs/runs/thayer_d3_runtime_readiness_20260713_135017"
D1_RUN = REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715"
RI_RUN = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653"

CAPSULE = CAPSULE_RUN / "contract/d3_scientific_capsule_v1.json"
RUNTIME_HASH_FREEZE = READINESS_RUN / "diagnostics/runtime_hash_freeze.json"
CHECKPOINT_BASELINE = CAPSULE_RUN / "tables/checkpoint_inventory_before.csv"

CORE_HASHES = {
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_v1.json":
        "8a76ccdfa659a7291f0f9b73e0cb4d4c8adfb317b9902fc8ad5763e6d17b7d21",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/schema/d3_scientific_capsule_v1.schema.json":
        "42a974a7ef2b48a7108ef350d2d119c3955f3df325411784c9a22da9cf975f40",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_manifest.json":
        "5753d502d515cdedcb679e7a2b0559839b40801974c27965e5512e97803f6684",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_hash_chain.json":
        "3a3aa5ec2e8b239b74ce3fd59e9a721333d2ab7f75f4c608aa9b41c5bfb15990",
}

MISSING_REQUIREMENTS = (
    ("capsule_artifact_d1_endpoint_manifest", "artifact reference"),
    ("capsule_artifact_d0_persisted_evidence", "evidence reference"),
    ("capsule_artifact_d1_persisted_evidence", "evidence reference"),
    ("capsule_artifact_d2_persisted_evidence", "evidence reference"),
    ("capsule_frozen_l0_decoder_topology_code", "model-construction requirement"),
    ("capsule_frozen_decoder_parameter_count", "model-construction requirement"),
    ("capsule_frozen_decoder_initialization_seeds", "model-construction requirement"),
    ("capsule_d1_final_objective_evidence", "evidence reference"),
    ("capsule_member_shape_dtype_endianness_expectations", "artifact member schema"),
)

RUN_DIRECTORIES = (
    "access_guard", "runtime", "diagnostics", "tables", "figures", "logs",
    "reports", "preregistration", "requirement_registry", "consumer_contract",
    "capsule_v2", "schema", "architecture_audit", "artifact_member_audit",
    "synthetic_inputs", "synthetic_execution", "optimizer_audit",
    "checkpoint_replay", "negative_tests", "launcher_tests", "future_d3_bundle",
)

EXPLICIT_ALLOWLIST = (
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/reports/final_report.md",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/capsule_validation/campaign_dependency_audit.json",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/capsule_validation/authoritative_validator_result.json",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/capsule_validation/capsule_preflight_result.json",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/tables/campaign_dependency_checks.csv",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/diagnostics/final_correctness_audit.json",
    "outputs/runs/thayer_capsule_authoritative_d3_20260713_161342/preregistration/capsule_authoritative_square_full_l0_d3.md",
    "scripts/run_thayer_capsule_authoritative_d3.py",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/reports/final_report.md",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_v1.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/schema/d3_scientific_capsule_v1.schema.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_manifest.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/contract/d3_scientific_capsule_hash_chain.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/diagnostics/final_correctness_audit.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/future_d3_template/authoritative_d3_from_capsule_template.md",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/tables/checkpoint_inventory_before.csv",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/tables/d3_scientific_dependency_inventory.csv",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/dependency_inventory/d3_dependency_graph.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/tables/scientific_value_provenance.csv",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/extracted_metadata/scientific_sky_vector.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/tables/sky_vector_verification.csv",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/extracted_metadata/d3_scientific_thresholds.json",
    "outputs/runs/thayer_d3_scientific_capsule_20260713_155637/tables/threshold_inventory.csv",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/reports/final_report_superseding_v3.md",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/diagnostics/final_correctness_audit_superseding_v3.json",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/diagnostics/runtime_hash_freeze.json",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/diagnostics/readiness_manifest.json",
    "outputs/runs/thayer_d3_runtime_readiness_20260713_135017/import_tests/scientific_postprocess_import_graph.json",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/reports/final_report.md",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/optimized_features/d1_penultimate_endpoints.npz",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/physical_outputs/d1_physical_outputs.npz",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/replay_verification/d1_endpoint_manifest.json",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/diagnostics/final_correctness_audit_superseding_v2.json",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/tables/d1_endpoint_inventory.csv",
    "outputs/runs/thayer_d1_endpoint_replay_20260713_113715/tables/downstream_d3_prerequisite_check.csv",
    "outputs/runs/thayer_repository_integrity_20260713_031653/reports/final_report.md",
    "outputs/runs/thayer_repository_integrity_20260713_031653/diagnostics/final_correctness_audit_superseding_v2.json",
    "outputs/runs/thayer_repository_integrity_20260713_031653/tables/d0_d3_summary_superseding_v2.csv",
    "outputs/runs/thayer_repository_integrity_20260713_031653/code_inventory/local_import_graph_v3.json",
    "outputs/runs/thayer_repository_integrity_20260713_031653/independent_oracles/reference_implementation.py",
    "outputs/runs/thayer_repository_integrity_20260713_031653/data_lineage/one_scene_lineage_superseding_v4.json",
    "outputs/runs/thayer_repository_integrity_20260713_031653/data_lineage/one_scene_payload.npz",
    "outputs/runs/thayer_repository_integrity_20260713_031653/fixed_feature_retry/cached_features_superseding_v4.pt",
    "outputs/runs/thayer_repository_integrity_20260713_031653/fixed_feature_retry/initial_state_square_superseding_v3.pt",
    "outputs/runs/thayer_full_l0_d3r_20260713_121652/authoritative_inputs/run_authoritative_d3.py",
    "docs/d3_scientific_contract_capsule.md", "docs/d3_scientific_dependency_schema.md",
    "docs/d3_scientific_artifact_contract.md", "docs/d3_runtime_readiness.md",
    "docs/d1_endpoint_persistence.md", "docs/feature_endpoint_artifact_contract.md",
    "docs/fixed_feature_decoder_audit.md", "docs/full_l0_fixed_feature_d3.md",
    "docs/decoder_capacity_ladder.md", "docs/current_status.md",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_x(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO, check=True, text=True, capture_output=True
    ).stdout


def create_run() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_d3_executable_contract_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for relative in RUN_DIRECTORIES:
        (run / relative).mkdir(parents=True, exist_ok=False)
    return run


def checkpoint_inventory() -> list[dict[str, Any]]:
    with CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    if len(baseline) != 600:
        raise RuntimeError(f"CHECKPOINT_BASELINE_COUNT_MISMATCH:{len(baseline)}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(baseline, start=1):
        path = REPO / item["path"]
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        passed = (
            actual_bytes == int(item["expected_bytes"])
            and actual_hash == item["expected_sha256"]
        )
        rows.append({
            "path": item["path"],
            "expected_bytes": item["expected_bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": item["expected_sha256"],
            "actual_sha256": actual_hash,
            "status": "PASS" if passed else "FAIL",
        })
        if index % 100 == 0:
            print(f"checkpoint inventory {index}/600", flush=True)
    return rows


def resolved_allowlist(capsule: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    paths = set(EXPLICIT_ALLOWLIST)
    paths.update(CORE_HASHES)
    paths.update(runtime["hashes"])
    for record in capsule["implementation_hashes"].values():
        paths.add(record["relative_path"])
    for record in capsule["scientific_artifact_references"].values():
        paths.add(record["relative_path"])
    return sorted(paths)


def artifact_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = REPO / relative
        rows.append({
            "path": relative,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else -1,
            "sha256": sha256(path) if path.is_file() else "MISSING",
        })
    return rows


def validate_frozen_inputs(
    capsule: dict[str, Any], runtime: dict[str, Any], checkpoints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = dict(CORE_HASHES)
    expected.update(runtime["hashes"])
    for record in capsule["implementation_hashes"].values():
        expected[record["relative_path"]] = record["sha256"]
    for record in capsule["scientific_artifact_references"].values():
        expected[record["relative_path"]] = record["sha256"]
    for relative, expected_hash in sorted(expected.items()):
        path = REPO / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        rows.append({
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected_hash else "FAIL",
        })
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("FROZEN_INPUT_HASH_MISMATCH")
    if any(row["status"] != "PASS" for row in checkpoints):
        raise RuntimeError("HISTORICAL_CHECKPOINT_MISMATCH")
    return rows


def preregistration_text(allowlist: list[str], start_utc: str) -> str:
    missing = "\n".join(f"- `{name}` ({category})" for name, category in MISSING_REQUIREMENTS)
    allowed = "\n".join(f"- `{path}`" for path in allowlist)
    return f"""# Thayer-D3E executable D3 contract and architecture audit

Campaign start: `{start_utc}`  
Experiment: `Thayer-D3E` (`Thayer Executable D3`)

## Scope and frozen scientific status

This is an executable contract, architecture, and deterministic synthetic full-stack audit. It is not scientific D3. Square D0/D1 remain 100/100/100 PASS, square D2 remains 0/0/0 FAIL, and square D3 remains UNKNOWN. Synthetic success has no scientific interpretation and cannot automatically continue into D3.

## Authoritative inputs and exact allowlist

{allowed}

No path outside this list, a record embedded in an allowlisted manifest/source, the fresh run, the approved reusable source/tests, or the approved documentation paths may be accessed. No recursive repository, outputs, or data enumeration is permitted.

## Reproduced missing-requirement hypothesis

The stopped capsule-driven consumer reports exactly these nine requirements:

{missing}

The campaign must reproduce all nine from persisted evidence before any capsule repair. A count or canonical-name mismatch stops the campaign.

## Canonical registry design

One machine-readable registry is the sole declaration of D3 requirements. Every record freezes identifier, name, category, requiredness, type, shape, dtype, units, semantic version, capsule location, provenance rule, consumer, validation function, representation kind, scientific-deserialization policy, protected-data restrictions, and failure message. The builder, validator, metadata/model preflights, synthetic consumer, scientific launcher, and future preregistration template derive their requirement sets from this registry. Their required sets must be exactly equal before capsule v2 is built.

## Capsule-v2 and consumer protocol

Capsule v2 is append-only and contains every valid capsule-v1 value plus the exact evidence, D1 manifest, model-construction, member-schema, state, execution, and runtime contracts. No required value may be absent, null, placeholder-like, implicit, or inferred from an unhashed current-code default. Schema, manifest, and hash chain are rebuilt as v2 and frozen with the registry and consumer hashes.

## Architecture and artifact protocol

Container inspection is metadata/header-only: ZIP central directory plus NPY headers for NPZ, named dataset metadata for HDF5, and weights-only loading for exact permitted model states. Scientific scene, target, cached-feature, and D1 endpoint array values are never deserialized. The exact L0 module/class/factory, signature, kwargs, topology, two independent experts, expected state keys/shapes/dtypes, square output mapping, 46,470 trainable parameters per expert, and 92,940 total trainable parameters are hard gates. Construction failure yields `ARCHITECTURE CONSTRUCTION DEFECT — D3 NOT AUTHORIZED` without topology changes.

## Synthetic execution protocol

Deterministic analytic tensors are generated solely from frozen production member shapes/dtypes. They distinguish prompts and target modes, include nonzero g/r/z structure, exercise source order and hard assignment, and contain no scientific values. The real production forward, mapping, assignment, loss, and pure evaluator are compared with independent references. Batch-size, reorder, contiguous/noncontiguous, finite/nonnegative, determinism, threshold/operator, and zero evaluator-I/O gates are frozen.

The exact optimizer contract is constructed over both and only the expert parameters. Exactly one synthetic MPS backward/step is run with fallback disabled. Both experts and final heads must have finite nonzero gradients, at least one non-final block per expert must update, and outputs must remain finite/nonnegative. Failure yields `SYNTHETIC AUTOGRAD/OPTIMIZER DEFECT`; CPU substitution is not allowed.

The post-step checkpoint uses the future-D3 schema and is reloaded in a fresh guarded process. State, output, assignment, loss, evaluator, and applicable gradient replay must pass. The actual D3 consumer then runs `--synthetic-preflight-only` through the same readiness/capsule/registry/artifact/model/optimizer/forward/loss/evaluator/backward/checkpoint/reload/closure path and must emit exactly `ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED` and `READY_FOR_AUTHORITATIVE_D3_EXECUTION`, then exit before scientific D3.

## Negative tests and closure

The 25 preregistered corruptions are: missing D0/D1/D2 evidence; missing D1 manifest; wrong module/class; missing constructor kwarg; wrong expert parameter count; missing or shape-changed state key; missing/shape-changed/dtype-changed cached member; missing or semantically swapped P0 member; missing D1 endpoint member; wrong optimizer class; missing budget; wrong mapping/assignment/evaluator/runtime hash; placeholder; protected path; and unexpected required member. Each must fail before model execution with its canonical requirement ID.

At successful closure, declared required IDs equal accessed/explicitly validated IDs, no undeclared dependency or implicit default was used, all 600 checkpoints rehash exactly, README and staged index remain unchanged, `git diff --check` passes, and all protected/ordinary/eight-scene/full-microset/Atlas/development/lockbox access counts remain zero.

## Attainability audit

- Exact nine-item reproduction: attainable from the stopped run's persisted dependency audit.
- Registry equality and capsule-v2 completeness: attainable by deriving every component set from one registry API.
- Header/member validation: attainable without array payload reads using format metadata.
- Exact architecture/state validation: attainable from allowlisted hashed source and permitted weights-only model-state loads.
- Synthetic forward/evaluator/reference comparisons: attainable without scientific arrays.
- Synthetic MPS optimizer gate: attainable only if the validated host exposes MPS; otherwise the frozen outcome is runtime/access or optimizer defect, never a lowered CPU gate.
- Checkpoint replay, consumer corruption, and closure equality: attainable with fresh run-local synthetic artifacts.
- Scientific D3, broader data, Atlas, development, lockbox, and automatic continuation: deliberately unattainable and prohibited in this campaign.

## Repair and stop policy

A correction requires an independently persisted failing test first and may change only non-scientific contract/infrastructure. It may not alter topology, parameter count, weights, mapping, assignment, loss, thresholds, evaluator, scientific values, or data. Before/after hashes and diffs are recorded, prior tests rerun, and changes remain unstaged/uncommitted. Gates are never lowered after execution. Any unresolved requirement, architecture, member, optimizer, replay, runtime, access, or set-equality failure stops the campaign in its corresponding frozen outcome category.
"""


def main() -> int:
    start_utc = utcnow()
    run = create_run()
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_HASH_FREEZE.read_text(encoding="utf-8"))
    allowlist = resolved_allowlist(capsule, runtime)
    checkpoints = checkpoint_inventory()
    write_csv_x(run / "tables/checkpoint_inventory_before.csv", checkpoints)
    frozen_rows = validate_frozen_inputs(capsule, runtime, checkpoints)
    write_csv_x(run / "tables/frozen_input_hash_validation.csv", frozen_rows)
    artifacts = artifact_rows(allowlist)
    write_json_x(run / "access_guard/repository_allowlist.json", {
        "schema": "thayer-d3e-exact-path-allowlist-v1",
        "created_utc": utcnow(),
        "paths": allowlist,
        "broad_search_permitted": False,
        "scientific_value_deserialization_permitted": False,
    })

    branch = git("branch", "--show-current").strip()
    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--short")
    staged = [line for line in git("diff", "--cached", "--name-only").splitlines() if line]
    if staged:
        raise RuntimeError(f"STAGED_INDEX_NOT_EMPTY:{staged}")
    disk = shutil.disk_usage(REPO)
    env_path = str(Path(sys.executable).resolve().parent.parent)
    snapshot = [
        "# Thayer-D3E standard-library-only environment snapshot", "",
        f"- Campaign start UTC: `{start_utc}`", f"- Branch: `{branch}`",
        f"- Git HEAD: `{head}`", f"- Python executable: `{sys.executable}`",
        f"- Environment path: `{env_path}`", f"- Expected BTK environment: `{REPO / '.venv-btk'}`",
        f"- Free disk bytes: `{disk.free}`", f"- Total disk bytes: `{disk.total}`",
        f"- Exact allowlisted artifacts: `{len(allowlist)}`", f"- Historical checkpoint rows: `{len(checkpoints)}`",
        "- Historical checkpoint mismatches: `0`", "- Staged index: `empty`", "",
        "## Git status at bootstrap", "", "```text", status.rstrip(), "```", "",
        "## Frozen capsule-v1 hashes", "",
    ]
    snapshot.extend(f"- `{path}`: `{digest}`" for path, digest in CORE_HASHES.items())
    snapshot.extend(["", "## Runtime-readiness frozen hashes", ""])
    snapshot.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(runtime["hashes"].items()))
    write_text_x(run / "diagnostics/environment_snapshot_stdlib_only.md", "\n".join(snapshot) + "\n")
    write_text_x(run / "diagnostics/campaign_contract.md", f"""# Thayer-D3E campaign contract

- Fresh append-only run: `{run.relative_to(REPO)}`
- Collision behavior: exclusive creation only; historical files are immutable.
- Access: exact allowlist only; no recursive repository, outputs, or data search.
- Scientific arrays: zero value deserializations permitted.
- Scientific D3: prohibited; synthetic preflight must exit after readiness markers.
- Mapping/topology/loss/assignment/threshold/seed changes: prohibited.
- Repair: only after an independent persisted failing test, with before/after evidence.
- Closure: 600/600 checkpoint rehash, empty staged index, README unchanged, protected access zero.
""")
    provenance = {
        "schema": "thayer-d3e-input-provenance-v1",
        "campaign_start_utc": start_utc,
        "run": str(run.relative_to(REPO)),
        "branch": branch,
        "git_head": head,
        "git_status": status.splitlines(),
        "staged_paths": staged,
        "python_executable": sys.executable,
        "environment_path": env_path,
        "capsule_v1_hashes": CORE_HASHES,
        "runtime_readiness_hashes": runtime["hashes"],
        "checkpoint_inventory_source": str(CHECKPOINT_BASELINE.relative_to(REPO)),
        "checkpoint_count": len(checkpoints),
        "checkpoint_mismatches": 0,
        "free_disk_bytes": disk.free,
        "allowlisted_artifacts": artifacts,
        "third_party_imports_before_preregistration": 0,
        "scientific_member_inspections_before_preregistration": 0,
        "scientific_tensor_deserializations": 0,
    }
    write_json_x(run / "logs/input_provenance.json", provenance)
    command_lines = [
        "#!/bin/sh", "set -eu",
        f"{sys.executable} -B scripts/bootstrap_thayer_d3_executable_contract.py",
        "# Subsequent commands are appended only after the preregistration freeze.",
    ]
    write_text_x(run / "logs/command_log.sh", "\n".join(command_lines) + "\n")

    prereg_path = run / "preregistration/executable_d3_contract_and_architecture_audit.md"
    write_text_x(prereg_path, preregistration_text(allowlist, start_utc))
    frozen_utc = utcnow()
    prereg_hash = sha256(prereg_path)
    write_json_x(run / "preregistration/preregistration_freeze.json", {
        "schema": "thayer-d3e-preregistration-freeze-v1",
        "path": str(prereg_path.relative_to(REPO)),
        "sha256": prereg_hash,
        "frozen_utc": frozen_utc,
        "predates_container_member_inspection": True,
        "predates_model_import": True,
        "predates_model_construction": True,
        "predates_optimizer_construction": True,
        "status": "FROZEN",
    })
    print(json.dumps({
        "run": str(run.relative_to(REPO)),
        "preregistration_sha256": prereg_hash,
        "frozen_utc": frozen_utc,
        "checkpoint_count": 600,
        "frozen_input_status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
