#!/usr/bin/env python3
"""Append-only v4.1 orchestrator over the frozen v4 authority bridge."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.run_thayer_scientific_d3_v4 as v4  # noqa: E402
from src.d3_execution_bridge_v4 import (  # noqa: E402
    IntegrationRequirementFailure,
    bridge_schema,
    sha256_file,
    validate_authority_chain,
    validate_bridge,
)


FROZEN_BRIDGE_V4 = (
    REPO
    / "outputs/runs/thayer_d3_integration_science_20260713_182315"
    / "execution_bridge/d3_execution_bridge_v4.json"
)
FROZEN_BRIDGE_V4_SHA256 = (
    "3ab6e4a525297f48cc7fd9428651c604aa1236ed0a4425f9953c5b5772345dc5"
)
V41_WORKER = REPO / "scripts/run_thayer_scientific_d3_process_v41.py"
V41_NORMALIZER = REPO / "src/d3_contract_tokens_v41.py"
BYTECODE_ENV_CONTRACT = {"PYTHONDONTWRITEBYTECODE": "1"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def environment(run: Path, role: str) -> dict[str, str]:
    result = v4.environment(run, role)
    result.update(BYTECODE_ENV_CONTRACT)
    return result


def next_candidate(run: Path) -> tuple[str, Path, str]:
    index = 1
    while True:
        candidate_id = f"candidate_{index:03d}"
        candidate_dir = run / f"execution_bridge/candidates/{candidate_id}"
        if not candidate_dir.exists():
            candidate_dir.mkdir(parents=False, exist_ok=False)
            tag = "candidate_bootstrap" if index == 1 else f"{candidate_id}_bootstrap"
            return candidate_id, candidate_dir, tag
        index += 1


def write_bootstrap_manifest(run: Path, candidate_dir: Path, tag: str) -> Path:
    manifest = candidate_dir / "serialization_bootstrap_import_manifest.json"
    v4.write_json_x(
        manifest,
        {
            "schema_version": "thayer-d3i41-serialization-bootstrap-import-v1",
            "tag": tag,
            "import_order": [
                "numpy",
                "torch",
                "frozen required torch and project modules",
                "torch.utils.serialization",
                "torch.utils.serialization.config",
            ],
            "required_modules": [
                "torch.utils.serialization",
                "torch.utils.serialization.config",
            ],
            "prewarm": {
                "object": "tiny synthetic state dictionary only",
                "save_count": 1,
                "load_count": 1,
                "load_mode": "weights_only=True",
                "scratch_root": str(
                    (run / f"runtime/scientific/tmp/serialization_prewarm/{tag}").resolve()
                ),
            },
            "scientific_checkpoint_permitted": False,
            "matplotlib_permitted": False,
            "strict_package_permissions_broadened": False,
            "bytecode_writing_permitted": False,
            "worker": source_record(V41_WORKER),
        },
    )
    return manifest


def run_candidate_prewarm(run: Path, tag: str) -> Path:
    code = (
        "from pathlib import Path; import sys; "
        "from scripts.run_thayer_scientific_d3_process_v41 import standalone_candidate_prewarm; "
        "standalone_candidate_prewarm(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])"
    )
    command = [
        str(REPO / ".venv-btk/bin/python"),
        "-B",
        "-c",
        code,
        str(run),
        str(run / "runtime/scientific"),
        tag,
    ]
    v4.run_process(
        command,
        run / f"logs/{tag}_serialization_prewarm.log",
        environment(run, "scientific"),
    )
    result = run / f"serialization_bootstrap/{tag}_result.json"
    if not result.is_file():
        raise IntegrationRequirementFailure(
            "D3I41-CANDIDATE-PREWARM-RESULT", "candidate prewarm result absent"
        )
    if json.loads(result.read_text(encoding="utf-8")).get("status") != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41-CANDIDATE-PREWARM-STATUS", "candidate prewarm did not pass"
        )
    return result


def build_candidate(
    args: argparse.Namespace, run: Path
) -> tuple[str, Path, str, Path, Path]:
    if sha256_file(FROZEN_BRIDGE_V4) != FROZEN_BRIDGE_V4_SHA256:
        raise IntegrationRequirementFailure(
            "D3I41-FROZEN-BRIDGE-V4-SHA", "frozen bridge v4 changed"
        )
    candidate_id, candidate_dir, tag = next_candidate(run)
    import_manifest = write_bootstrap_manifest(run, candidate_dir, tag)
    prewarm_result = run_candidate_prewarm(run, tag)
    frozen_bridge = json.loads(FROZEN_BRIDGE_V4.read_text(encoding="utf-8"))
    candidate = json.loads(json.dumps(frozen_bridge))
    candidate["phase"] = "frozen"
    candidate["producing_run"] = str(run.relative_to(REPO))
    candidate["created_utc"] = utcnow()
    candidate["repository_head"] = v4.git_head()
    candidate["authorities"] = frozen_bridge["authorities"]
    candidate["scientific_contract"] = frozen_bridge["scientific_contract"]
    candidate["launchers"]["orchestrator"] = source_record(Path(__file__))
    candidate["launchers"]["scientific_worker"] = source_record(V41_WORKER)
    candidate["launchers"]["dtype_normalizer"] = source_record(V41_NORMALIZER)
    candidate["cli_propagation"]["worker_output_root"] = str(run.resolve())
    candidate["cli_propagation"]["worker_runtime_root"] = str(
        (run / "runtime/scientific").resolve()
    )
    candidate["synthetic_preflight"] = {
        "status": "PENDING_CANDIDATE_EXECUTION",
        "inherited_v4_preflight": frozen_bridge["synthetic_preflight"],
        "required_markers": [
            "ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED",
            "ALL_D3_POLICIES_EXECUTABLY_DEFINED",
            "ALL_D3_CONTROL_BRANCHES_SYNTHETICALLY_COVERED",
            "DECLARED_DEFINED_ACCESSED_TESTED_PERSISTED_POLICIES_EQUAL",
            "BUNDLE_V3_PROPAGATED_TO_SCIENTIFIC_WORKER",
            "SCIENTIFIC_WORKER_AND_POSTPROCESS_WORKER_PRESENT",
            "V41_DTYPE_TOKEN_NORMALIZATION_PASS",
            "V41_SERIALIZATION_BOOTSTRAP_PASS",
            "V41_ZERO_STRICT_EXTERNAL_PYC_READS",
            "V41_FULL_STACK_SYNTHETIC_EXECUTION_PASS",
            "READY_FOR_AUTHORITATIVE_D3_EXECUTION",
        ],
        "scientific_array_values_loaded": 0,
    }
    candidate["scientific_contract"]["v41_corrections"] = {
        "integration_revision": "4.1.0",
        "candidate_id": candidate_id,
        "frozen_bridge_v4": {
            "path": str(FROZEN_BRIDGE_V4.relative_to(REPO)),
            "sha256": FROZEN_BRIDGE_V4_SHA256,
        },
        "dtype_normalizer": source_record(V41_NORMALIZER),
        "dtype_comparison_policy_version": "semantic-numpy-dtype-token-v1",
        "scientific_worker": source_record(V41_WORKER),
        "orchestrator": source_record(Path(__file__)),
        "serialization_bootstrap_manifest": {
            "path": str(import_manifest.relative_to(REPO)),
            "sha256": sha256_file(import_manifest),
        },
        "serialization_prewarm_result": {
            "path": str(prewarm_result.relative_to(REPO)),
            "sha256": sha256_file(prewarm_result),
        },
        "assertions": {
            "scientific_values_changed": False,
            "artifact_values_changed": False,
            "model_changed": False,
            "optimizer_changed": False,
            "policy_changed": False,
            "runtime_strict_permissions_broadened": False,
            "dtype_comparison_normalized": True,
            "serialization_preimport_added": True,
        },
    }
    candidate["flow_invariants"]["v41_orchestrator_sha"] = sha256_file(Path(__file__))
    candidate["flow_invariants"]["v41_scientific_worker_sha"] = sha256_file(V41_WORKER)
    candidate["flow_invariants"]["v41_dtype_normalizer_sha"] = sha256_file(V41_NORMALIZER)
    candidate_path = candidate_dir / "d3_execution_bridge_v41_candidate.json"
    v4.write_json_x(candidate_path, candidate)
    candidate_hash = sha256_file(candidate_path)
    v4.write_text_x(
        candidate_dir / "d3_execution_bridge_v41_candidate.sha256",
        f"{candidate_hash}  {candidate_path.name}\n",
    )
    schema = bridge_schema()
    schema["title"] = "Thayer D3 execution bridge v4 with v4.1 correction records"
    schema_path = candidate_dir / "d3_execution_bridge_v41.schema.json"
    v4.write_json_x(schema_path, schema)
    manifest_path = candidate_dir / "d3_execution_bridge_v41_manifest.json"
    v4.write_json_x(
        manifest_path,
        {
            "schema_version": "thayer-d3i41-bridge-candidate-manifest-v1",
            "candidate_id": candidate_id,
            "bridge_path": str(candidate_path.relative_to(REPO)),
            "bridge_sha256": candidate_hash,
            "schema_path": str(schema_path.relative_to(REPO)),
            "schema_sha256": sha256_file(schema_path),
            "serialization_bootstrap_manifest_sha256": sha256_file(import_manifest),
            "serialization_prewarm_result_sha256": sha256_file(prewarm_result),
            "exact_tested_bytes_must_not_be_rebuilt": True,
            "status": "FROZEN_FOR_CANDIDATE_TESTING",
        },
    )
    hash_chain_path = candidate_dir / "d3_execution_bridge_v41_hash_chain.json"
    v4.write_json_x(
        hash_chain_path,
        {
            "schema_version": "thayer-d3i41-bridge-hash-chain-v1",
            "frozen_bridge_v4_sha256": FROZEN_BRIDGE_V4_SHA256,
            "candidate_bridge_sha256": candidate_hash,
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "schema_sha256": sha256_file(schema_path),
            "prewarm_result_sha256": sha256_file(prewarm_result),
            "preregistration_sha256": json.loads(
                (run / "preregistration/preregistration_freeze.json").read_text(encoding="utf-8")
            )["sha256"],
        },
    )
    return candidate_id, candidate_path, candidate_hash, manifest_path, hash_chain_path


def worker_command(run: Path, bridge: Path, bridge_hash: str, mode: str) -> list[str]:
    return [
        str(REPO / ".venv-btk/bin/python"),
        "-B",
        str(V41_WORKER),
        "--bridge-v4",
        str(bridge),
        "--bridge-v4-sha256",
        bridge_hash,
        "--output-root",
        str(run),
        "--strict-runtime-root",
        str(run / "runtime/scientific"),
        "--mode",
        mode,
    ]


def copy_flow_tables(run: Path) -> None:
    for old_name, new_name in (
        ("v4_argument_flow_closure.csv", "v41_argument_flow_closure.csv"),
        ("v4_requirement_flow_closure.csv", "v41_requirement_flow_closure.csv"),
    ):
        with (run / f"tables/{old_name}").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        v4.write_csv_x(run / f"tables/{new_name}", rows)


def synthetic_campaign(args: argparse.Namespace, run: Path) -> int:
    candidate_id, bridge, bridge_hash, manifest, hash_chain = build_candidate(args, run)
    validate_bridge(repo=REPO, bridge_path=bridge, bridge_sha256=bridge_hash, require_frozen=True)
    command = worker_command(run, bridge, bridge_hash, "synthetic_integration_preflight")
    v4.run_process(
        command,
        run / "logs/synthetic_scientific_worker_v41.log",
        environment(run, "scientific"),
    )
    replay_env = environment(run, "scientific")
    replay_env["D3_V4_SYNTHETIC_REPLAY_ONLY"] = "1"
    v4.run_process(
        command,
        run / "logs/synthetic_checkpoint_replay_v41.log",
        replay_env,
    )
    result_path = run / "synthetic_preflight/v41_scientific_worker_result.json"
    replay_path = run / "synthetic_preflight/synthetic_checkpoint_replay_v41.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or replay.get("status") != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41-SYNTHETIC-FULL-STACK", "worker or replay failed"
        )
    if result.get("scientific_array_values_loaded") != 0:
        raise IntegrationRequirementFailure(
            "D3I41-SYNTHETIC-SCIENTIFIC-LOAD", "synthetic worker loaded scientific values"
        )
    post_manifest = v4.make_post_manifest(run, "synthetic", [result_path])
    v4.run_postprocessor(run, post_manifest, "synthetic")
    post = json.loads(
        (run / "synthetic_preflight/postprocessing_result.json").read_text(encoding="utf-8")
    )
    if post.get("status") != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41-SYNTHETIC-POSTPROCESS", "synthetic postprocessor failed"
        )
    v4.write_flow_closure(run, args, bridge, bridge_hash)
    copy_flow_tables(run)
    markers = [
        "ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED",
        "ALL_D3_POLICIES_EXECUTABLY_DEFINED",
        "ALL_D3_CONTROL_BRANCHES_SYNTHETICALLY_COVERED",
        "DECLARED_DEFINED_ACCESSED_TESTED_PERSISTED_POLICIES_EQUAL",
        "BUNDLE_V3_PROPAGATED_TO_SCIENTIFIC_WORKER",
        "SCIENTIFIC_WORKER_AND_POSTPROCESS_WORKER_PRESENT",
        "V41_DTYPE_TOKEN_NORMALIZATION_PASS",
        "V41_SERIALIZATION_BOOTSTRAP_PASS",
        "V41_ZERO_STRICT_EXTERNAL_PYC_READS",
        "V41_FULL_STACK_SYNTHETIC_EXECUTION_PASS",
        "READY_FOR_AUTHORITATIVE_D3_EXECUTION",
    ]
    summary = {
        "schema_version": "thayer-d3i41-full-stack-v1",
        "status": "PASS",
        "candidate_id": candidate_id,
        "bridge_candidate": str(bridge.relative_to(run)),
        "bridge_candidate_sha256": bridge_hash,
        "bridge_manifest_sha256": sha256_file(manifest),
        "bridge_hash_chain_sha256": sha256_file(hash_chain),
        "worker": result,
        "checkpoint_replay": replay,
        "postprocessing": post,
        "scientific_array_values_loaded": 0,
        "markers": markers,
        "completed_utc": utcnow(),
    }
    v4.write_json_x(run / "synthetic_preflight/full_stack_result.json", summary)
    for marker in markers:
        print(marker, flush=True)
    return 0


def scientific_campaign(args: argparse.Namespace, run: Path) -> int:
    authority_record_path = run / "execution_bridge/authoritative_candidate.json"
    if not authority_record_path.is_file():
        raise IntegrationRequirementFailure(
            "D3I41-AUTHORITATIVE-CANDIDATE", "authoritative candidate record absent"
        )
    authority_record = json.loads(authority_record_path.read_text(encoding="utf-8"))
    bridge = REPO / authority_record["bridge_path"]
    bridge_hash = authority_record["bridge_sha256"]
    validate_bridge(repo=REPO, bridge_path=bridge, bridge_sha256=bridge_hash, require_frozen=True)
    command = worker_command(run, bridge, bridge_hash, "authoritative_scientific_d3")
    v4.run_process(
        command,
        run / "logs/scientific_worker.log",
        environment(run, "scientific"),
    )
    replay_env = environment(run, "scientific")
    replay_env["D3_V4_REPLAY_ONLY"] = "1"
    v4.run_process(
        command,
        run / "logs/scientific_replay_worker.log",
        replay_env,
    )
    replay = json.loads(
        (run / "replay_verification/replay_summary.json").read_text(encoding="utf-8")
    )
    if replay.get("status") != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41-SCIENTIFIC-REPLAY", "fresh-process replay failed"
        )
    post_manifest = v4.make_post_manifest(
        run,
        "scientific",
        [
            run / "decoder_training/trajectory.csv",
            run / "decoder_training/trajectory_summary.json",
            run / "postprocessing_inputs/selected_outputs.npz",
        ],
    )
    v4.run_postprocessor(run, post_manifest, "scientific")
    completion = {
        "status": "SCIENTIFIC_REPLAY_POSTPROCESS_COMPLETE",
        "bridge_sha256": bridge_hash,
        "bundle_v3_sha256": args.bundle_v3_sha256,
        "completed_utc": utcnow(),
    }
    v4.write_json_x(run / "diagnostics/orchestrator_completion.json", completion)
    print("SCIENTIFIC_REPLAY_POSTPROCESS_COMPLETE", flush=True)
    return 0


def main() -> int:
    args = v4.parse_args()
    run = args.output_dir.resolve()
    request = args.governing_request.resolve()
    if not run.is_dir():
        raise IntegrationRequirementFailure("D3I41-OUTPUT-DIR", "master run is absent")
    if not request.is_file():
        raise IntegrationRequirementFailure("D3I41-GOVERNING-REQUEST", "request absent")
    validate_authority_chain(REPO, args.bundle_v3.resolve(), args.bundle_v3_sha256)
    provenance = json.loads((run / "logs/input_provenance.json").read_text(encoding="utf-8"))
    expected_request = next(
        item for item in provenance["authoritative_inputs"] if item["authority"] == "governing_request"
    )
    if sha256_file(request) != expected_request["actual_sha256"]:
        raise IntegrationRequirementFailure(
            "D3I41-GOVERNING-REQUEST-SHA", "governing request changed"
        )
    record = {
        "bundle_v3": str(args.bundle_v3.resolve()),
        "bundle_v3_sha256": args.bundle_v3_sha256,
        "output_dir": str(run),
        "governing_request": str(request),
        "synthetic_only": args.synthetic_integration_preflight_only,
        "validated_utc": utcnow(),
    }
    target = run / (
        "runtime/orchestrator/synthetic_cli_arguments.json"
        if args.synthetic_integration_preflight_only
        else "runtime/orchestrator/scientific_cli_arguments.json"
    )
    v4.write_json_x(v4.collision_free(target), record)
    if args.synthetic_integration_preflight_only:
        return synthetic_campaign(args, run)
    return scientific_campaign(args, run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrationRequirementFailure as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "canonical_integration_requirement_id": exc.requirement_id,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2)
