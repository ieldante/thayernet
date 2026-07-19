#!/usr/bin/env python3
"""Independent eligibility validator for Thayer-D3I41R1 candidates."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AUDIT = REPO / (
    "outputs/runs/thayer_d3_v41_science_20260713_200621/"
    "diagnostics/independent_contract_audit_v2.json"
)
REQUIRED_TEST_CSV = REPO / (
    "outputs/runs/thayer_d3_v41_science_20260713_200621/"
    "tables/v41_required_test_name_audit_v2.csv"
)
FROZEN_BRIDGE = REPO / (
    "outputs/runs/thayer_d3_integration_science_20260713_182315/"
    "execution_bridge/d3_execution_bridge_v4.json"
)
REQUIRED_RESULT_FIELDS = {
    "original_actual_token",
    "original_expected_token",
    "canonical_actual_token",
    "canonical_expected_token",
    "actual_kind",
    "expected_kind",
    "actual_itemsize",
    "expected_itemsize",
    "actual_byteorder",
    "expected_byteorder",
    "platform_byteorder",
    "equal",
    "equality_basis",
    "failure_reason",
    "status",
}
ELIGIBILITY_MARKERS = (
    "ALL_V41_INDEPENDENT_AUDIT_ROWS_PASS",
    "ALL_REQUIRED_REGRESSION_TESTS_COLLECTED_AND_PASSED",
    "NUMPY_DTYPE_OBJECT_EQUALITY_AUTHORITATIVE",
    "DTYPE_COERCION_RESULT_AND_SUBARRAY_RULES_PASS",
    "PRODUCTION_CHECKPOINT_ADAPTER_PREWARM_PASS",
    "STRICT_CHECKPOINT_PATH_ZERO_NEW_IMPORTS",
    "STRICT_CHECKPOINT_PATH_ZERO_EXTERNAL_PYC_READS",
    "ZERO_SCIENTIFIC_PAYLOADS_LOADED",
    "READY_FOR_V41_SCIENTIFIC_PAYLOAD_ACCESS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dtype_validation(dtype_path: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    module = importlib.import_module("src.d3_contract_tokens_v41r1")
    source = dtype_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "numpy_dtype_contract_equal"
    )
    object_compare = any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "actual_dtype"
        and any(isinstance(operator, ast.Eq) for operator in node.ops)
        and any(
            isinstance(comparator, ast.Name)
            and comparator.id == "expected_dtype"
            for comparator in node.comparators
        )
        for node in ast.walk(function)
    )
    canonical_compare = any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id.startswith("canonical_")
        for node in ast.walk(function)
    )
    pairs = [
        ("float32", "<f4"),
        (np.float32, "<f4"),
        (np.dtype("float32"), "<f4"),
        ("=f4", "<f4"),
        (">f4", "<f4"),
        ("float64", "<f4"),
        ("int32", "<f4"),
    ]
    pair_rows = []
    for actual, expected in pairs:
        independent = bool(np.dtype(actual) == np.dtype(expected))
        candidate = module.numpy_dtype_contract_equal(actual, expected)
        pair_rows.append(
            {
                "actual": str(actual),
                "expected": str(expected),
                "independent_dtype_object_equal": independent,
                "candidate_equal": candidate.equal,
                "equality_basis": candidate.equality_basis,
                "status": "PASS"
                if candidate.equal == independent
                and candidate.equality_basis == "numpy_dtype_object_equality"
                else "FAIL",
            }
        )
    rejection_rows = []
    for name, value in (
        ("structured", np.dtype([("value", "<f4")])),
        ("object", np.dtype(object)),
        ("subarray", np.dtype((np.float32, (2,)))),
    ):
        rejected = False
        try:
            module.coerce_numpy_dtype(value)
        except module.UnsupportedNumpyDType:
            rejected = True
        rejection_rows.append({"category": name, "rejected": rejected})
    result_fields = {field.name for field in fields(module.NumpyDTypeContractResult)}
    checks = {
        "object_equality_ast": object_compare and not canonical_compare,
        "coercion_function": callable(module.coerce_numpy_dtype),
        "result_fields": result_fields == REQUIRED_RESULT_FIELDS,
        "pair_oracle": all(row["status"] == "PASS" for row in pair_rows),
        "compound_rejection": all(row["rejected"] for row in rejection_rows),
    }
    return checks, {
        "pair_rows": pair_rows,
        "rejection_rows": rejection_rows,
        "result_fields": sorted(result_fields),
    }


def _serialization_validation(
    run: Path, candidate: dict[str, Any], worker_path: Path
) -> tuple[dict[str, bool], dict[str, Any]]:
    correction = candidate["scientific_contract"]["v41r1_corrections"]
    contract_path = REPO / correction["production_checkpoint_adapter_contract"]["path"]
    prewarm_path = REPO / correction["serialization_prewarm_result"]["path"]
    schema_manifest = REPO / correction["synthetic_checkpoint_manifest"]["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prewarm = json.loads(prewarm_path.read_text(encoding="utf-8"))
    manifest = json.loads(schema_manifest.read_text(encoding="utf-8"))
    adapter = importlib.import_module("src.d3_checkpoint_adapter_v41r1")
    payload = adapter.build_synthetic_production_checkpoint_payload()
    frozen_worker = REPO / contract["frozen_execution_source"]["path"]
    frozen_tree = ast.parse(frozen_worker.read_text(encoding="utf-8"))
    frozen_run = next(
        node
        for node in frozen_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_authoritative"
    )
    frozen_save = next(
        node
        for node in ast.walk(frozen_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and node.args
        and isinstance(node.args[0], ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "schema_version"
            for key in node.args[0].keys
        )
    )
    frozen_keys = {
        key.value for key in frozen_save.args[0].keys if isinstance(key, ast.Constant)
    }
    worker_tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    prewarm_functions = {
        node.name: node
        for node in worker_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_exercise_production_adapter", "serialization_bootstrap_prewarm_r1"}
    }
    direct_generic = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr in {"save", "load"}
        for function in prewarm_functions.values()
        for node in ast.walk(function)
    )
    adapter_calls = {
        node.func.attr
        for function in prewarm_functions.values()
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "checkpoint_adapter"
    }
    checks = {
        "contract_hash": sha256_file(contract_path)
        == correction["production_checkpoint_adapter_contract"]["sha256"],
        "prewarm_hash": sha256_file(prewarm_path)
        == correction["serialization_prewarm_result"]["sha256"],
        "schema_manifest_hash": sha256_file(schema_manifest)
        == correction["synthetic_checkpoint_manifest"]["sha256"],
        "complete_schema": set(payload) == frozen_keys
        == set(contract["checkpoint_schema"]["top_level_keys"]),
        "manifest_complete": bool(manifest["complete_schema_validated"]),
        "writer_invoked": prewarm["adapter_trace"]["writer_calls"] == 1,
        "reader_invoked": prewarm["adapter_trace"]["reader_calls"] >= 2,
        "strict_zero_imports": prewarm["strict_new_imports"] == [],
        "strict_zero_pyc": prewarm["strict_external_pyc_reads"] == [],
        "no_generic_substitute": not direct_generic,
        "adapter_symbols_called": {
            "write_production_checkpoint",
            "read_production_checkpoint",
        }.issubset(adapter_calls),
        "no_scientific_values": not prewarm["scientific_checkpoint_opened"]
        and not prewarm["scientific_model_tensor_used"],
        "frozen_reader_settings": contract["settings"]["weights_only"] is True
        and contract["settings"]["map_location"] == "cpu",
    }
    return checks, {
        "contract_path": str(contract_path.relative_to(REPO)),
        "prewarm_path": str(prewarm_path.relative_to(REPO)),
        "adapter_calls": sorted(adapter_calls),
        "frozen_schema_keys": sorted(frozen_keys),
        "prewarm_trace": prewarm["adapter_trace"],
    }


def _test_validation(run: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    with REQUIRED_TEST_CSV.open(newline="", encoding="utf-8") as handle:
        required_rows = list(csv.DictReader(handle))
    closure = json.loads(
        (run / "required_test_audit/required_test_closure.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {row["required_name"] for row in required_rows}
    actual = {row["required_name"] for row in closure["rows"]}
    exact = expected == actual and len(expected) == 19
    every = all(
        row["collected"]
        and row["executed"]
        and row["outcome"] == "PASSED"
        and not row["skipped"]
        for row in closure["rows"]
    )
    return {
        "authority_hash": closure["authority_sha256"]
        == sha256_file(REQUIRED_TEST_CSV),
        "exact_set": exact,
        "all_pass": every,
    }, {"expected": sorted(expected), "actual": sorted(actual)}


def validate_full(run: Path, candidate_path: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    frozen_bridge = json.loads(FROZEN_BRIDGE.read_text(encoding="utf-8"))
    correction = candidate["scientific_contract"]["v41r1_corrections"]
    dtype_path = REPO / candidate["launchers"]["dtype_normalizer"]["path"]
    worker_path = REPO / candidate["launchers"]["scientific_worker"]["path"]
    orchestrator_path = REPO / candidate["launchers"]["orchestrator"]["path"]
    dtype_checks, dtype_evidence = _dtype_validation(dtype_path)
    serialization_checks, serialization_evidence = _serialization_validation(
        run, candidate, worker_path
    )
    test_checks, test_evidence = _test_validation(run)
    full_stack = json.loads(
        (run / "synthetic_preflight/full_stack_result.json").read_text(
            encoding="utf-8"
        )
    )
    audit_hash_ok = sha256_file(AUDIT) == correction["independent_audit"]["sha256"]
    csv_hash_ok = (
        sha256_file(REQUIRED_TEST_CSV)
        == correction["required_test_authority"]["sha256"]
    )
    authorities_ok = candidate["authorities"] == frozen_bridge["authorities"]
    base_candidate_contract = json.loads(json.dumps(candidate["scientific_contract"]))
    base_candidate_contract.pop("v41r1_corrections")
    scientific_contract_ok = base_candidate_contract == frozen_bridge["scientific_contract"]
    no_self_write = all(
        "independent_validator/candidate_validation.json"
        not in path.read_text(encoding="utf-8")
        for path in (worker_path, orchestrator_path)
    )
    zero_payload = full_stack["scientific_payload_values_loaded"] == 0

    base_results = {
        "R1-DTYPE-001": dtype_checks["object_equality_ast"]
        and dtype_checks["pair_oracle"],
        "R1-DTYPE-002": dtype_checks["coercion_function"],
        "R1-DTYPE-003": dtype_checks["result_fields"],
        "R1-DTYPE-004": dtype_checks["compound_rejection"],
        "R1-SER-001": serialization_checks["writer_invoked"]
        and serialization_checks["reader_invoked"]
        and serialization_checks["adapter_symbols_called"],
        "R1-SER-002": serialization_checks["complete_schema"]
        and serialization_checks["manifest_complete"],
        "R1-SER-003": serialization_checks["strict_zero_imports"]
        and serialization_checks["strict_zero_pyc"],
        "R1-SER-004": serialization_checks["no_generic_substitute"]
        and serialization_checks["no_scientific_values"],
        "R1-TEST-000": all(test_checks.values()),
        "R1-SCI-001": zero_payload,
    }
    ledger = json.loads(
        (run / "compliance_ledger/v41r1_requirement_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    closure = json.loads(
        (run / "required_test_audit/required_test_closure.json").read_text(
            encoding="utf-8"
        )
    )
    closure_by_name = {row["required_name"]: row for row in closure["rows"]}
    row_results: dict[str, bool] = dict(base_results)
    for row in ledger["rows"]:
        requirement_id = row["canonical_requirement_id"]
        if requirement_id.startswith("R1-TEST-") and requirement_id != "R1-TEST-000":
            name = row["source_row_or_key"].split(":", 1)[1]
            test_row = closure_by_name.get(name)
            row_results[requirement_id] = bool(
                test_row
                and test_row["collected"]
                and test_row["executed"]
                and test_row["outcome"] == "PASSED"
                and not test_row["skipped"]
            )
    prerequisite_ok = (
        audit_hash_ok
        and csv_hash_ok
        and authorities_ok
        and scientific_contract_ok
        and no_self_write
        and all(row_results.values())
        and all(serialization_checks.values())
    )
    row_results["R1-AUDIT-001"] = prerequisite_ok
    validated_rows = []
    for row in ledger["rows"]:
        passed = row_results[row["canonical_requirement_id"]]
        updated = dict(row)
        updated["independent_validator_result"] = "PASS" if passed else "FAIL"
        updated["status"] = "PASS" if passed else "FAIL"
        updated["failure_reason"] = "" if passed else "independent check failed"
        validated_rows.append(updated)
    eligible = prerequisite_ok and all(row["status"] == "PASS" for row in validated_rows)
    markers = list(ELIGIBILITY_MARKERS) if eligible else []
    result = {
        "schema_version": "thayer-d3-i41r1-independent-candidate-validation-v1",
        "status": "ELIGIBLE" if eligible else "INELIGIBLE",
        "candidate_path": str(candidate_path.relative_to(REPO)),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_self_certified": False,
        "audit_hash_ok": audit_hash_ok,
        "required_test_csv_hash_ok": csv_hash_ok,
        "authorities_unchanged": authorities_ok,
        "scientific_contract_unchanged": scientific_contract_ok,
        "candidate_cannot_write_validator_result": no_self_write,
        "scientific_payload_values_loaded": full_stack[
            "scientific_payload_values_loaded"
        ],
        "ledger_row_count": len(validated_rows),
        "ledger_pass_count": sum(row["status"] == "PASS" for row in validated_rows),
        "dtype_checks": dtype_checks,
        "dtype_evidence": dtype_evidence,
        "serialization_checks": serialization_checks,
        "serialization_evidence": serialization_evidence,
        "test_checks": test_checks,
        "test_evidence": test_evidence,
        "markers": markers,
        "rows": validated_rows,
    }
    output_json = run / "independent_validator/candidate_validation.json"
    output_csv = run / "tables/independent_candidate_validation.csv"
    output_md = run / "diagnostics/independent_candidate_validation.md"
    validated_ledger = run / "compliance_ledger/v41r1_requirement_ledger_validated.json"
    validated_table = run / "tables/v41r1_requirement_ledger_validated.csv"
    write_json_x(output_json, result)
    write_csv_x(
        output_csv,
        [
            {
                "requirement_id": row["canonical_requirement_id"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
            }
            for row in validated_rows
        ],
    )
    write_json_x(
        validated_ledger,
        {
            **{key: value for key, value in ledger.items() if key != "rows"},
            "rows": validated_rows,
            "status": "PASS" if eligible else "FAIL",
        },
    )
    write_csv_x(validated_table, validated_rows)
    with output_md.open("x", encoding="utf-8") as handle:
        handle.write(
            "# Independent candidate validation\n\n"
            f"Status: `{'ELIGIBLE' if eligible else 'INELIGIBLE'}`. "
            f"Ledger: `{sum(row['status'] == 'PASS' for row in validated_rows)}/"
            f"{len(validated_rows)}` PASS. Candidate self-certification was not used.\n\n"
        )
        for marker in markers:
            handle.write(f"- `{marker}`\n")
    with (run / "independent_validator/candidate_validation.sha256").open(
        "x", encoding="utf-8"
    ) as handle:
        handle.write(f"{sha256_file(output_json)}  {output_json.name}\n")
    return result


def validate_hash_only(run: Path) -> dict[str, Any]:
    authority = json.loads(
        (run / "execution_bridge/authoritative_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = json.loads(
        (run / "diagnostics/r1_source_freeze_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    mismatches = []
    for record in freeze["records"]:
        path = REPO / record["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != record["sha256"]:
            mismatches.append(
                {"path": record["path"], "expected": record["sha256"], "actual": actual}
            )
    candidate = REPO / authority["bridge_path"]
    if sha256_file(candidate) != authority["bridge_sha256"]:
        mismatches.append({"path": authority["bridge_path"], "reason": "bridge hash"})
    result = {
        "schema_version": "thayer-d3-i41r1-independent-final-hash-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "record_count": len(freeze["records"]),
        "mismatches": mismatches,
        "marker": "READY_FOR_V41_SCIENTIFIC_PAYLOAD_ACCESS"
        if not mismatches
        else "NOT_READY",
    }
    write_json_x(
        run / "independent_validator/candidate_validation_final_hash_only.json",
        result,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--mode", choices=("full", "hash-only"), default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    if args.mode == "full":
        if args.candidate is None:
            raise SystemExit("--candidate is required in full mode")
        result = validate_full(run, args.candidate.resolve())
    else:
        result = validate_hash_only(run)
    for marker in result.get("markers", (result.get("marker"),)):
        if marker:
            print(marker, flush=True)
    return 0 if result["status"] in {"ELIGIBLE", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
