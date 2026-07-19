#!/usr/bin/env python3
"""Append-only orchestrator for independently validated D3 v4.1 R1."""

from __future__ import annotations

import argparse
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


FROZEN_BRIDGE = REPO / (
    "outputs/runs/thayer_d3_integration_science_20260713_182315/"
    "execution_bridge/d3_execution_bridge_v4.json"
)
FROZEN_BRIDGE_SHA256 = (
    "3ab6e4a525297f48cc7fd9428651c604aa1236ed0a4425f9953c5b5772345dc5"
)
R1_WORKER = REPO / "scripts/run_thayer_scientific_d3_process_v41r1.py"
R1_DTYPE = REPO / "src/d3_contract_tokens_v41r1.py"
R1_CHECKPOINT_ADAPTER = REPO / "src/d3_checkpoint_adapter_v41r1.py"
R1_VALIDATOR = REPO / "scripts/validate_thayer_d3_i41r1_candidate.py"
R1_TESTS = (
    REPO / "tests/test_d3_contract_tokens_v41r1.py",
    REPO / "tests/test_d3_serialization_v41r1.py",
    REPO / "tests/test_thayer_d3_i41r1_integration.py",
    REPO / "tests/test_thayer_d3_i41r1_independent_validator.py",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def environment(run: Path, role: str) -> dict[str, str]:
    values = v4.environment(run, role)
    values.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "D3_I41R1_RUN": str(run),
        }
    )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-v3", type=Path, required=True)
    parser.add_argument("--bundle-v3-sha256", required=True)
    parser.add_argument("--governing-request", type=Path, required=True)
    parser.add_argument("--synthetic-integration-preflight-only", action="store_true")
    return parser.parse_args()


def next_candidate(run: Path) -> tuple[str, Path, str]:
    index = 1
    while True:
        candidate_id = f"candidate_{index:03d}"
        candidate_dir = run / f"execution_bridge/candidates/{candidate_id}"
        if not candidate_dir.exists():
            candidate_dir.mkdir(parents=False, exist_ok=False)
            return candidate_id, candidate_dir, f"{candidate_id}_bootstrap"
        index += 1


def run_candidate_prewarm(run: Path, tag: str) -> Path:
    code = (
        "from pathlib import Path; import sys; "
        "from scripts.run_thayer_scientific_d3_process_v41r1 "
        "import standalone_candidate_prewarm; "
        "standalone_candidate_prewarm(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])"
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
    result = run / f"serialization_bootstrap/{tag}_r1_result.json"
    if not result.is_file() or json.loads(result.read_text())["status"] != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41R1-PREWARM", "candidate production adapter prewarm failed"
        )
    return result


def build_candidate(run: Path) -> tuple[str, Path, str, list[Path]]:
    if sha256_file(FROZEN_BRIDGE) != FROZEN_BRIDGE_SHA256:
        raise IntegrationRequirementFailure(
            "D3I41R1-FROZEN-BRIDGE", "frozen bridge hash mismatch"
        )
    candidate_id, candidate_dir, tag = next_candidate(run)
    checkpoint_contract = run / (
        "serialization_bootstrap/production_checkpoint_adapter_contract.json"
    )
    synthetic_manifest = run / (
        "serialization_bootstrap/synthetic_checkpoint_payload_manifest.json"
    )
    ledger = run / "compliance_ledger/v41r1_requirement_ledger.json"
    test_closure = run / "required_test_audit/required_test_closure.json"
    audit = REPO / (
        "outputs/runs/thayer_d3_v41_science_20260713_200621/"
        "diagnostics/independent_contract_audit_v2.json"
    )
    required_csv = REPO / (
        "outputs/runs/thayer_d3_v41_science_20260713_200621/"
        "tables/v41_required_test_name_audit_v2.csv"
    )
    required_paths = (checkpoint_contract, synthetic_manifest, ledger, test_closure)
    if not all(path.is_file() for path in required_paths):
        raise IntegrationRequirementFailure(
            "D3I41R1-CANDIDATE-EVIDENCE", "pre-candidate evidence incomplete"
        )
    import_manifest = candidate_dir / "serialization_bootstrap_import_manifest.json"
    v4.write_json_x(
        import_manifest,
        {
            "schema_version": "thayer-d3-i41r1-bootstrap-import-v1",
            "import_order": [
                "numpy",
                "torch",
                "torch.serialization",
                "torch.utils.serialization",
                "torch.utils.serialization.config",
                "src.d3_checkpoint_adapter_v41r1",
            ],
            "required_modules": [
                "torch.serialization",
                "torch.utils.serialization",
                "torch.utils.serialization.config",
            ],
            "production_writer": (
                "src.d3_checkpoint_adapter_v41r1.write_production_checkpoint"
            ),
            "production_reader": (
                "src.d3_checkpoint_adapter_v41r1.read_production_checkpoint"
            ),
            "complete_schema_manifest": source_record(synthetic_manifest),
            "scientific_checkpoint_permitted": False,
            "scientific_model_tensor_permitted": False,
            "strict_permissions_broadened": False,
        },
    )
    prewarm = run_candidate_prewarm(run, tag)
    frozen = json.loads(FROZEN_BRIDGE.read_text(encoding="utf-8"))
    candidate = json.loads(json.dumps(frozen))
    candidate["phase"] = "frozen"
    candidate["producing_run"] = str(run.relative_to(REPO))
    candidate["created_utc"] = utcnow()
    candidate["repository_head"] = v4.git_head()
    candidate["launchers"]["orchestrator"] = source_record(Path(__file__))
    candidate["launchers"]["scientific_worker"] = source_record(R1_WORKER)
    candidate["launchers"]["dtype_normalizer"] = source_record(R1_DTYPE)
    candidate["launchers"]["production_checkpoint_adapter"] = source_record(
        R1_CHECKPOINT_ADAPTER
    )
    candidate["launchers"]["independent_validator"] = source_record(R1_VALIDATOR)
    candidate["cli_propagation"]["worker_output_root"] = str(run.resolve())
    candidate["cli_propagation"]["worker_runtime_root"] = str(
        (run / "runtime/scientific").resolve()
    )
    candidate["synthetic_preflight"] = {
        "status": "PENDING_CANDIDATE_EXECUTION",
        "inherited_v4_preflight": frozen["synthetic_preflight"],
        "scientific_array_values_loaded": 0,
        "candidate_self_certified": False,
    }
    candidate["scientific_contract"]["v41r1_corrections"] = {
        "integration_revision": "4.1-r1",
        "candidate_id": candidate_id,
        "frozen_bridge_v4": {
            "path": str(FROZEN_BRIDGE.relative_to(REPO)),
            "sha256": FROZEN_BRIDGE_SHA256,
        },
        "independent_audit": {
            "path": str(audit.relative_to(REPO)),
            "sha256": sha256_file(audit),
        },
        "required_test_authority": {
            "path": str(required_csv.relative_to(REPO)),
            "sha256": sha256_file(required_csv),
        },
        "requirement_ledger": source_record(ledger),
        "required_test_closure": source_record(test_closure),
        "dtype_normalizer": source_record(R1_DTYPE),
        "production_checkpoint_adapter": source_record(R1_CHECKPOINT_ADAPTER),
        "production_checkpoint_adapter_contract": source_record(
            checkpoint_contract
        ),
        "synthetic_checkpoint_manifest": source_record(synthetic_manifest),
        "serialization_bootstrap_manifest": source_record(import_manifest),
        "serialization_prewarm_result": source_record(prewarm),
        "independent_validator": source_record(R1_VALIDATOR),
        "assertions": {
            "scientific_values_changed": False,
            "artifact_values_changed": False,
            "model_changed": False,
            "optimizer_changed": False,
            "policy_changed": False,
            "threshold_changed": False,
            "runtime_permissions_broadened": False,
            "dtype_object_equality_implemented": True,
            "complete_dtype_result_implemented": True,
            "subarray_rejection_implemented": True,
            "production_checkpoint_adapter_prewarmed": True,
            "all_required_test_names_collected_and_passed": True,
            "candidate_assertions_are_not_eligibility": True,
        },
    }
    candidate["flow_invariants"]["v41r1_orchestrator_sha"] = sha256_file(
        Path(__file__)
    )
    candidate["flow_invariants"]["v41r1_worker_sha"] = sha256_file(R1_WORKER)
    candidate["flow_invariants"]["v41r1_dtype_sha"] = sha256_file(R1_DTYPE)
    candidate_path = candidate_dir / "d3_execution_bridge_v41r1_candidate.json"
    v4.write_json_x(candidate_path, candidate)
    candidate_hash = sha256_file(candidate_path)
    v4.write_text_x(
        candidate_dir / "d3_execution_bridge_v41r1_candidate.sha256",
        f"{candidate_hash}  {candidate_path.name}\n",
    )
    schema_path = candidate_dir / "d3_execution_bridge_v41r1.schema.json"
    schema = bridge_schema()
    schema["title"] = "Thayer D3 independent contract-compliant bridge v4.1 R1"
    v4.write_json_x(schema_path, schema)
    sources = [
        Path(__file__),
        R1_WORKER,
        R1_DTYPE,
        R1_CHECKPOINT_ADAPTER,
        R1_VALIDATOR,
        *R1_TESTS,
    ]
    source_inventory = candidate_dir / "r1_source_inventory.json"
    v4.write_json_x(
        source_inventory,
        {
            "schema_version": "thayer-d3-i41r1-source-inventory-v1",
            "records": [source_record(path) for path in sources],
        },
    )
    manifest_path = candidate_dir / "d3_execution_bridge_v41r1_manifest.json"
    v4.write_json_x(
        manifest_path,
        {
            "schema_version": "thayer-d3-i41r1-candidate-manifest-v1",
            "candidate_id": candidate_id,
            "candidate_status": "FROZEN_FOR_INDEPENDENT_VALIDATION",
            "creation_reason": "first R1 contract-compliance candidate",
            "bridge": source_record(candidate_path),
            "schema": source_record(schema_path),
            "source_inventory": source_record(source_inventory),
            "requirement_ledger": source_record(ledger),
            "independent_validator": source_record(R1_VALIDATOR),
            "required_test_closure": source_record(test_closure),
            "exact_tested_bytes_must_not_be_rebuilt": True,
        },
    )
    hash_chain = candidate_dir / "d3_execution_bridge_v41r1_hash_chain.json"
    v4.write_json_x(
        hash_chain,
        {
            "schema_version": "thayer-d3-i41r1-hash-chain-v1",
            "frozen_bridge_v4_sha256": FROZEN_BRIDGE_SHA256,
            "candidate_bridge_sha256": candidate_hash,
            "candidate_schema_sha256": sha256_file(schema_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "source_inventory_sha256": sha256_file(source_inventory),
            "audit_ledger_sha256": sha256_file(ledger),
            "independent_validator_sha256": sha256_file(R1_VALIDATOR),
            "dtype_contract_sha256": sha256_file(R1_DTYPE),
            "checkpoint_adapter_contract_sha256": sha256_file(
                checkpoint_contract
            ),
            "synthetic_checkpoint_manifest_sha256": sha256_file(
                synthetic_manifest
            ),
            "serialization_prewarm_result_sha256": sha256_file(prewarm),
            "required_test_closure_sha256": sha256_file(test_closure),
        },
    )
    return candidate_id, candidate_path, candidate_hash, [
        schema_path,
        manifest_path,
        hash_chain,
        source_inventory,
    ]


def worker_command(
    run: Path, bridge: Path, bridge_hash: str, mode: str
) -> list[str]:
    return [
        str(REPO / ".venv-btk/bin/python"),
        "-B",
        str(R1_WORKER),
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


def synthetic_campaign(args: argparse.Namespace, run: Path) -> int:
    candidate_id, bridge, bridge_hash, records = build_candidate(run)
    validate_bridge(
        repo=REPO,
        bridge_path=bridge,
        bridge_sha256=bridge_hash,
        require_frozen=True,
    )
    command = worker_command(run, bridge, bridge_hash, "synthetic_integration_preflight")
    v4.run_process(
        command,
        run / "logs/synthetic_scientific_worker_v41r1.log",
        environment(run, "scientific"),
    )
    replay_environment = environment(run, "scientific")
    replay_environment["D3_V4_SYNTHETIC_REPLAY_ONLY"] = "1"
    v4.run_process(
        command,
        run / "logs/synthetic_checkpoint_replay_v41r1.log",
        replay_environment,
    )
    result_path = run / "synthetic_preflight/v41_scientific_worker_result.json"
    replay_path = run / "synthetic_preflight/synthetic_checkpoint_replay_v41.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or replay.get("status") != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41R1-SYNTHETIC", "scientific worker or replay failed"
        )
    if result.get("scientific_array_values_loaded") != 0:
        raise IntegrationRequirementFailure(
            "D3I41R1-SYNTHETIC-PAYLOAD", "scientific values loaded"
        )
    post_manifest = v4.make_post_manifest(run, "synthetic", [result_path])
    v4.run_postprocessor(run, post_manifest, "synthetic")
    post = json.loads(
        (run / "synthetic_preflight/postprocessing_result.json").read_text()
    )
    v4.write_flow_closure(run, args, bridge, bridge_hash)
    summary = {
        "schema_version": "thayer-d3-i41r1-full-stack-v1",
        "status": "PASS",
        "candidate_id": candidate_id,
        "candidate_bridge": str(bridge.relative_to(REPO)),
        "candidate_bridge_sha256": bridge_hash,
        "candidate_proof_records": [source_record(path) for path in records],
        "worker": result,
        "checkpoint_replay": replay,
        "postprocessing": post,
        "scientific_payload_values_loaded": 0,
        "candidate_self_certified": False,
        "eligibility_decision": "PENDING_INDEPENDENT_VALIDATOR",
        "completed_utc": utcnow(),
    }
    v4.write_json_x(run / "synthetic_preflight/full_stack_result.json", summary)
    print("V41R1_FULL_STACK_SYNTHETIC_EXECUTION_PASS", flush=True)
    print("ZERO_SCIENTIFIC_PAYLOADS_LOADED", flush=True)
    print("CANDIDATE_AWAITS_INDEPENDENT_VALIDATOR", flush=True)
    return 0


def scientific_campaign(args: argparse.Namespace, run: Path) -> int:
    authority_path = run / "execution_bridge/authoritative_candidate.json"
    final_validation = run / (
        "independent_validator/candidate_validation_final_hash_only.json"
    )
    if not authority_path.is_file() or not final_validation.is_file():
        raise IntegrationRequirementFailure(
            "D3I41R1-SCIENCE-AUTHORITY", "authoritative eligibility absent"
        )
    if json.loads(final_validation.read_text())["status"] != "PASS":
        raise IntegrationRequirementFailure(
            "D3I41R1-SCIENCE-FINAL-HASH", "final hash gate failed"
        )
    authority = json.loads(authority_path.read_text())
    bridge = REPO / authority["bridge_path"]
    bridge_hash = authority["bridge_sha256"]
    validate_bridge(
        repo=REPO,
        bridge_path=bridge,
        bridge_sha256=bridge_hash,
        require_frozen=True,
    )
    command = worker_command(run, bridge, bridge_hash, "authoritative_scientific_d3")
    v4.run_process(
        command,
        run / "logs/scientific_worker.log",
        environment(run, "scientific"),
    )
    return 0


def main() -> int:
    args = parse_args()
    run = args.output_dir.resolve()
    if not run.is_dir() or not args.governing_request.is_file():
        raise IntegrationRequirementFailure(
            "D3I41R1-INPUT", "run or governing request absent"
        )
    validate_authority_chain(
        REPO, args.bundle_v3.resolve(), args.bundle_v3_sha256
    )
    audit_manifest = json.loads(
        (run / "authoritative_audit/authority_manifest.json").read_text()
    )
    for record in audit_manifest.values():
        if sha256_file(REPO / record["path"]) != record["sha256"]:
            raise IntegrationRequirementFailure(
                "D3I41R1-AUDIT-AUTHORITY", record["path"]
            )
    cli_record = {
        "bundle_v3": str(args.bundle_v3.resolve()),
        "bundle_v3_sha256": args.bundle_v3_sha256,
        "governing_request": str(args.governing_request.resolve()),
        "governing_request_sha256": sha256_file(args.governing_request.resolve()),
        "synthetic_only": args.synthetic_integration_preflight_only,
        "validated_utc": utcnow(),
    }
    target = run / (
        "runtime/orchestrator/synthetic_cli_arguments.json"
        if args.synthetic_integration_preflight_only
        else "runtime/orchestrator/scientific_cli_arguments.json"
    )
    v4.write_json_x(target, cli_record)
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
