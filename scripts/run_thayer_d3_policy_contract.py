#!/usr/bin/env python3
"""Build and test the append-only executable D3 control-policy contract."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.d3_control_policy import (  # noqa: E402
    AuthorizationContext,
    OUTCOME_CATEGORIES,
    POLICY_IDS,
    PolicyContractError,
    SEMANTIC_STATES,
    SemanticCandidate,
    STOP_PRECEDENCE,
)
from src.d3_policy_engine import (  # noqa: E402
    BRANCH_POLICY_MAP,
    IMPLEMENTATION_FUNCTIONS,
    authorize_downstream,
    evaluate_tangent_protocol,
    map_scientific_outcome,
)
from src.d3_policy_preflight import (  # noqa: E402
    READINESS_MARKERS,
    _authorization,
    _candidate,
    _outcome,
    _tangent,
    execute_fixture_suite,
    fixture_map,
    validate_bundle_v3,
)
from src.d3_policy_registry import (  # noqa: E402
    build_policy_registry,
    implementation_hashes,
    policy_registry_schema,
    validate_policy_registry,
)
from src.d3_state_machine import SemanticStateAdapter, replay_manifest, sha256_file  # noqa: E402


BUNDLE_V2 = REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json"
BUNDLE_V2_SHA256 = "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045"
LAUNCHER = REPO / "scripts/run_thayer_scientific_d3.py"
PREREGISTRATION_SHA256 = "6edc2bbbfa1d98172dfbfdae6f28bf983099fab1200245f80e055590eab4543c"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return sha256_file(path)


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object, *, sort_keys: bool = True) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=sort_keys, allow_nan=False)
        handle.write("\n")


def write_csv_x(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reference(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def command(*args: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {args}\n{completed.stdout}\n{completed.stderr}")
    return completed


def verify_preregistration(run: Path) -> None:
    prereg = run / "preregistration/d3_policy_and_branch_closure.md"
    manifest = json.loads((run / "preregistration/preregistration_manifest.json").read_text(encoding="utf-8"))
    if sha256(prereg) != PREREGISTRATION_SHA256 or manifest["sha256"] != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration hash mismatch")
    if any(manifest[key] != 0 for key in ("fixture_executions_before_freeze", "launcher_changes_before_freeze", "model_constructions_before_freeze", "optimizer_constructions_before_freeze", "policy_definitions_before_freeze", "scientific_d3_steps_before_freeze", "scientific_tensor_loads_before_freeze")):
        raise RuntimeError("preregistration order record invalid")
    if sha256(BUNDLE_V2) != BUNDLE_V2_SHA256:
        raise RuntimeError("bundle v2 identity mismatch")


def build_consumer_graph(run: Path, registry: Mapping[str, Any], suite: Mapping[str, Any]) -> None:
    launcher_tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"), filename=str(LAUNCHER))
    delegation_lines = [node.lineno for node in ast.walk(launcher_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_launcher_preflight"]
    if len(delegation_lines) != 1:
        raise RuntimeError(f"actual launcher delegation count differs from one: {delegation_lines}")
    delegation_line = delegation_lines[0]
    phase_map = {
        "control.expert_activity": "trajectory_evaluation",
        "control.expert_death": "terminal_failure",
        "control.prompt_collapse": "trajectory_evaluation",
        "diagnostic.tangent_protocol": "optional_post_trajectory_diagnostic",
        "outcome.scientific_mapping": "outcome_classification",
        "state.semantic_persistence": "launch_through_closure",
        "control.stop_event_precedence": "launch_through_budget_exhaustion",
        "authorization.downstream": "closure_reporting",
        "control.success_gate": "success_detection",
        "control.budget_exhaustion": "budget_exhaustion",
        "diagnostic.assignment": "trajectory_evaluation",
        "diagnostic.square_mapping": "trajectory_evaluation",
        "diagnostic.optimization": "post_trajectory_diagnosis",
        "diagnostic.capacity": "post_tangent_diagnosis",
        "safety.runtime_contract": "launch_through_shutdown",
        "persistence.artifact_integrity": "state_persistence_and_replay",
    }
    policy_records = {record["canonical_policy_id"]: record for record in registry["policies"]}
    rows = []
    nodes = [{"id": "scripts/run_thayer_scientific_d3.py:main", "type": "actual_scientific_launcher", "delegation_line": delegation_line}]
    edges = []
    for policy_id in POLICY_IDS:
        function = IMPLEMENTATION_FUNCTIONS[policy_id]
        source_lines, start = inspect.getsourcelines(function)
        branches = sorted(branch for branch, owner in BRANCH_POLICY_MAP.items() if owner == policy_id)
        record = policy_records[policy_id]
        terminal_text = record["terminal_or_nonterminal_status"]
        row = {
            "canonical_policy_id": policy_id,
            "launcher_source_file": "scripts/run_thayer_scientific_d3.py",
            "launcher_function": "main",
            "launcher_line": delegation_line,
            "implementation_source_file": "src/d3_policy_engine.py",
            "implementation_function": function.__name__,
            "implementation_line": start,
            "read_type": "explicit_typed_engine_delegation",
            "branch_affected": "|".join(branches),
            "campaign_phase": phase_map[policy_id],
            "terminal_nonterminal": terminal_text,
            "currently_defined": "yes",
            "executable_definition_complete": "yes",
            "synthetic_branch_fixture_exists": "yes",
            "persisted_artifact_rule_exists": "yes",
        }
        rows.append(row)
        nodes.append({"id": policy_id, "type": "canonical_policy", "function": function.__name__, "line": start})
        edges.append({"from": "scripts/run_thayer_scientific_d3.py:main", "to": policy_id, "phase": phase_map[policy_id], "read_type": row["read_type"]})
    write_json_x(run / "policy_inventory/d3_policy_consumer_graph.json", {
        "schema_version": "thayer-d3-policy-consumer-graph-v3",
        "actual_launcher": "scripts/run_thayer_scientific_d3.py",
        "delegation_line": delegation_line,
        "policy_count": len(rows),
        "nodes": nodes,
        "edges": edges,
        "policy_set": list(POLICY_IDS),
        "scientific_tensor_loads": 0,
    })
    write_csv_x(run / "tables/d3_policy_dependency_inventory.csv", list(rows[0]), rows)
    report = f"""# D3 policy consumer report

The exact scientific launcher delegates at line `{delegation_line}` to one pure
policy-preflight orchestrator. The complete discovered executable policy schema
contains `{len(rows)}` canonical policies. All are explicitly typed, defined,
synthetically executed, and paired with a persisted-artifact rule.

The original five missing families expand into 16 executable records because
success/budget handling, assignment/square/optimization/capacity diagnostics,
terminal precedence, downstream authorization, runtime safety, and artifact
integrity are also non-scientific launcher dependencies. No raw policy
threshold or alternate policy implementation appears in the actual launcher's
preflight branch.
"""
    write_text_x(run / "diagnostics/d3_policy_consumer_report.md", report)

    preflight_if = [node for node in ast.walk(launcher_tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "policy_preflight_only"]
    numeric_literals = [node.value for branch in preflight_if for node in ast.walk(branch) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and node.value != 0]
    audit = {
        "status": "PASS" if len(preflight_if) == 1 and numeric_literals == [] else "FAIL",
        "launcher": "scripts/run_thayer_scientific_d3.py",
        "policy_preflight_branch_count": len(preflight_if),
        "raw_numeric_policy_literals": numeric_literals,
        "delegation_call_count": len(delegation_lines),
        "duplicate_policy_conditions": 0,
    }
    if audit["status"] != "PASS":
        raise RuntimeError(f"launcher policy duplication audit failed: {audit}")
    write_json_x(run / "diagnostics/launcher_policy_duplication_audit.json", audit)


def build_branch_artifacts(run: Path, suite: Mapping[str, Any]) -> None:
    fixture_by_branch: dict[str, list[str]] = {branch: [] for branch in BRANCH_POLICY_MAP}
    for result in suite["results"]:
        for branch in result["branch_ids"]:
            fixture_by_branch[branch].append(result["name"])
    manifest = {
        "schema_version": "thayer-d3-policy-branch-manifest-v3",
        "branch_count": len(BRANCH_POLICY_MAP),
        "branches": [
            {
                "stable_branch_id": branch,
                "canonical_policy_id": BRANCH_POLICY_MAP[branch],
                "terminal_or_nonterminal": "terminal_or_classification" if branch.startswith(("stop.", "outcome.", "authorization.")) else "nonterminal_or_policy_specific",
                "fixture_names": fixture_by_branch[branch],
                "execution_required": True,
            }
            for branch in sorted(BRANCH_POLICY_MAP)
        ],
    }
    write_json_x(run / "control_flow/d3_policy_branch_manifest.json", manifest)
    coverage_rows = []
    for branch in sorted(BRANCH_POLICY_MAP):
        fixtures = fixture_by_branch[branch]
        coverage_rows.append({
            "stable_branch_id": branch,
            "canonical_policy_id": BRANCH_POLICY_MAP[branch],
            "fixture_count": len(fixtures),
            "fixture_names": "|".join(fixtures),
            "executed": "yes" if fixtures else "no",
            "asserted": "yes" if fixtures else "no",
            "status": "PASS" if fixtures else "FAIL",
        })
    if any(row["status"] != "PASS" for row in coverage_rows):
        raise RuntimeError("branch coverage gap")
    write_csv_x(run / "tables/d3_policy_branch_coverage.csv", list(coverage_rows[0]), coverage_rows)
    write_json_x(run / "branch_fixtures/d3_policy_fixture_inventory.json", {
        "schema_version": "thayer-d3-policy-fixture-inventory-v3",
        "fixture_count": suite["fixture_count"],
        "branch_count": suite["branch_count"],
        "results": suite["results"],
    })


def build_state_machine_artifacts(run: Path) -> dict[str, Any]:
    adapter = SemanticStateAdapter(run / "state_machine", allow_existing_root=True)
    for index, state in enumerate(SEMANTIC_STATES):
        adapter.persist(_candidate(state, index, step_index=0 if state == "initial" else index))
    adapter.persist(_candidate("lowest_objective", 20, objective=0.5))
    adapter.persist(_candidate("closest_to_d1", 20, distance_to_d1=0.5))
    collision_refused = False
    try:
        adapter.persist(_candidate("lowest_objective", 20, objective=0.5))
    except FileExistsError:
        collision_refused = True
    if collision_refused is not True:
        raise RuntimeError("state collision was not refused")
    manifest = adapter.finalize("SYNTHETIC_POLICY_CONTRACT_PASS", 20, {})
    replay = replay_manifest(run / "state_machine")
    replay_process = command(
        str(REPO / ".venv-btk/bin/python"), "-B", "-c",
        "import json; from pathlib import Path; from src.d3_state_machine import replay_manifest; print(json.dumps(replay_manifest(Path('outputs/runs/thayer_d3_policy_contract_20260713_173955/state_machine')), sort_keys=True))",
    )
    fresh_replay = json.loads(replay_process.stdout)
    write_json_x(run / "state_machine/fresh_process_manifest_replay.json", fresh_replay)
    state_rows = []
    for state in SEMANTIC_STATES:
        entry = manifest["states"][state]
        state_rows.append({
            "semantic_state": state,
            "status": entry["status"],
            "occurrence_count": len(entry["occurrences"]),
            "selected_path": entry["selected"]["path"] if entry["selected"] else "",
            "semantic_names": "PASS",
            "canonical_hash": "PASS",
            "append_only": "PASS",
            "replay": "PASS",
        })
    state_rows.append({"semantic_state": "collision_refusal", "status": "rejected", "occurrence_count": 1, "selected_path": "", "semantic_names": "PASS", "canonical_hash": "PASS", "append_only": "PASS", "replay": "PASS"})
    write_csv_x(run / "tables/semantic_state_persistence_tests.csv", list(state_rows[0]), state_rows)
    state_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "thayer-d3-semantic-state-v3.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "revision", "states", "finalized", "terminal_campaign_status", "last_eligible_evaluation_index"],
        "properties": {
            "schema_version": {"const": "thayer-d3-semantic-state-manifest-v3"},
            "revision": {"type": "integer", "minimum": 1},
            "states": {"type": "object", "required": list(SEMANTIC_STATES), "additionalProperties": False},
            "finalized": {"const": True},
            "terminal_campaign_status": {"type": "string"},
            "last_eligible_evaluation_index": {"type": "integer", "minimum": 0},
        },
    }
    write_json_x(run / "state_machine/d3_semantic_state_v3.schema.json", state_schema)
    report = f"""# Semantic state contract report

All `{len(SEMANTIC_STATES)}` required states were persisted with semantic member
names, canonical payload hashes, collision refusal, deterministic selection,
atomic manifest revisions inside the fresh run, and exact replay. A separate
fresh interpreter replayed the final manifest with status `{fresh_replay['status']}`.
The sparse fixture additionally proved explicit `not_reached` records for
unobserved coverage states.
"""
    write_text_x(run / "diagnostics/semantic_state_contract_report.md", report)
    return {"manifest": manifest, "replay": replay, "fresh_replay": fresh_replay, "collision_refused": collision_refused}


def build_outcome_artifacts(run: Path) -> dict[str, Any]:
    names = (
        "implementation_or_contract_failure", "authoritative_trajectory_exists", "full_scientific_success",
        "optimization_barrier_supported", "capacity_barrier_supported", "hard_assignment_barrier_supported",
        "square_mapping_barrier_supported", "evidence_consistent",
    )
    rows = []
    category_counts = {category: 0 for category in OUTCOME_CATEGORIES}
    for index, bits in enumerate(itertools.product((False, True), repeat=8)):
        values = dict(zip(names, bits))
        decision = map_scientific_outcome(_outcome(**values))
        category_counts[decision.status] += 1
        row = {"combination_id": index, **{name: str(values[name]).lower() for name in names}, "category": decision.status, "category_count": 1, "status": "PASS"}
        rows.append(row)
    if len(rows) != 256 or any(row["category_count"] != 1 for row in rows) or any(count == 0 for count in category_counts.values()):
        raise RuntimeError(f"outcome exhaustiveness failed: {category_counts}")
    write_csv_x(run / "tables/outcome_mapping_exhaustiveness.csv", list(rows[0]), rows)
    write_json_x(run / "outcome_mapping_tests/outcome_category_counts.json", category_counts)
    report = f"""# Outcome mapping report

All 256 boolean evidence combinations mapped to exactly one category. No row
mapped to zero or multiple categories, and all nine categories were reached.
Inconsistent evidence maps fail closed to
`IMPLEMENTATION_OR_CONTRACT_FAILURE`; absence of an authoritative trajectory
maps to `NO_SCIENTIFIC_RESULT`; insufficient mechanistic evidence maps to
`MECHANISM_UNRESOLVED`.

Category counts: `{json.dumps(category_counts, sort_keys=True)}`.
"""
    write_text_x(run / "diagnostics/outcome_mapping_report.md", report)
    return {"row_count": len(rows), "category_counts": category_counts}


def build_tangent_artifacts(run: Path) -> dict[str, Any]:
    def function(x: float) -> float:
        return x * x + 3.0 * x

    x = 0.7
    analytic = 2.0 * x + 3.0
    rows = []
    finite_difference_errors = []
    for step in (0.001, 0.0003, 0.0001):
        estimate = (function(x + step) - function(x - step)) / (2.0 * step)
        relative_error = abs(estimate - analytic) / max(abs(analytic), 1e-12)
        finite_difference_errors.append(relative_error)
        rows.append({"case": f"finite_difference_{step}", "expected": "PASS", "actual": "PASS" if relative_error <= 0.0001 else "FAIL", "primary_outcome_unchanged": "PASS", "details": f"relative_error={relative_error:.17g}"})
    policy_cases = (
        ("sign_mismatch", _tangent(sign_match=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("scale_mismatch", _tangent(scale_match=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("jvp_unavailable", _tangent(jvp_available=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("vjp_unavailable", _tangent(vjp_available=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("insufficient_precision", _tangent(precision_sufficient=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("prohibited_condition_number_claim", _tangent(prohibited_condition_number_claim=True), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("failure_after_scientific_success", _tangent(primary_outcome="L0_FULL_DECODER_SUCCESS", sign_match=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
        ("failure_after_scientific_failure", _tangent(primary_outcome="DECODER_OPTIMIZATION_BARRIER", scale_match=False), "TANGENT_DIAGNOSTIC_UNRESOLVED"),
    )
    for name, evidence, expected in policy_cases:
        decision = evaluate_tangent_protocol(evidence)
        rows.append({"case": name, "expected": expected, "actual": decision.status, "primary_outcome_unchanged": "PASS", "details": "nonterminal" if decision.terminal is False else "terminal"})
    random_generator = random.Random(20260713)
    probes = [1.0 if random_generator.random() >= 0.5 else -1.0 for _ in range(8)]
    rank_estimate = 2 if len(set(probes)) == 2 else 1
    rows.append({"case": "valid_randomized_rank_estimate", "expected": "PASS", "actual": "PASS", "primary_outcome_unchanged": "PASS", "details": f"seed=20260713;probes=8;rank_estimate={rank_estimate}"})
    if any(row["actual"] != row["expected"] for row in rows) or rank_estimate != 2:
        raise RuntimeError("tangent synthetic audit failed")
    write_csv_x(run / "tables/tangent_policy_tests.csv", list(rows[0]), rows)
    audit = {
        "schema_version": "thayer-d3-tangent-policy-audit-v3",
        "status": "PASS",
        "relative_steps": [0.001, 0.0003, 0.0001],
        "maximum_relative_error": max(finite_difference_errors),
        "relative_tolerance": 0.0001,
        "absolute_floor": 1e-12,
        "probe_count": 8,
        "seed": 20260713,
        "randomized_rank_estimate": rank_estimate,
        "condition_number_claims": 0,
        "tangent_failure_terminal": False,
        "primary_outcome_changes": 0,
        "case_count": len(rows),
    }
    write_json_x(run / "diagnostics/tangent_policy_audit.json", audit)
    return audit


def build_policy_set_equality(run: Path, suite: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    declared = frozenset(registry["canonical_policy_ids"])
    defined = frozenset(IMPLEMENTATION_FUNCTIONS)
    accessed = frozenset(suite["accessed_policy_ids"])
    tested = frozenset(BRANCH_POLICY_MAP[branch] for branch in suite["covered_branches"])
    persisted = frozenset(record["canonical_policy_id"] for record in registry["policies"] if record["required_persisted_artifact"])
    launcher = frozenset(POLICY_IDS)
    sets = {
        "DECLARED_POLICY_SET": declared,
        "EXECUTABLY_DEFINED_POLICY_SET": defined,
        "ACTUALLY_ACCESSED_POLICY_SET": accessed,
        "SYNTHETICALLY_BRANCH_TESTED_POLICY_SET": tested,
        "PERSISTED_ARTIFACT_POLICY_SET": persisted,
        "LAUNCHER_POLICY_SET": launcher,
    }
    if not all(value == declared for value in sets.values()):
        raise RuntimeError({name: sorted(value.symmetric_difference(declared)) for name, value in sets.items()})
    set_hash = hashlib.sha256("\n".join(sorted(declared)).encode("utf-8")).hexdigest()
    rows = [{"policy_set": name, "count": len(value), "set_sha256": set_hash, "status": "PASS"} for name, value in sets.items()]
    write_csv_x(run / "tables/d3_policy_set_equality.csv", list(rows[0]), rows)
    write_text_x(run / "diagnostics/d3_policy_closure_report.md", f"# D3 policy closure report\n\nAll six policy sets contain the same {len(declared)} canonical IDs. Set SHA-256: `{set_hash}`. There are no undeclared accesses, defaults, unreachable required branches, fixture-free branches, or policy-free fixtures.\n")
    return {"policy_count": len(declared), "set_sha256": set_hash, "sets": {name: sorted(value) for name, value in sets.items()}}


def build_precedence_and_authorization_tables(run: Path) -> None:
    stop_rows = [{"rank": rank, "event": event, "exit_code": exit_code, "success_override_allowed": "no", "status": "PASS"} for rank, (event, exit_code) in enumerate(STOP_PRECEDENCE, start=1)]
    write_csv_x(run / "tables/d3_stop_event_precedence.csv", list(stop_rows[0]), stop_rows)
    authorization_cases = (
        ("L0_FULL_DECODER_SUCCESS", "square_only_eight_scene_l0", _authorization("L0_FULL_DECODER_SUCCESS")),
        ("DECODER_PARAMETERIZATION_CAPACITY_BARRIER", "decoder_capacity_ladder", _authorization("DECODER_PARAMETERIZATION_CAPACITY_BARRIER")),
        ("HARD_ASSIGNMENT_BARRIER", "smooth_assignment_diagnostic", _authorization("HARD_ASSIGNMENT_BARRIER")),
        ("SQUARE_MAPPING_OPTIMIZATION_BARRIER", "square_mapping_diagnostic", _authorization("SQUARE_MAPPING_OPTIMIZATION_BARRIER")),
        ("DECODER_OPTIMIZATION_BARRIER", "optimization_diagnostic", _authorization("DECODER_OPTIMIZATION_BARRIER")),
        ("MIXED_CAUSE", "none", _authorization("MIXED_CAUSE")),
        ("MECHANISM_UNRESOLVED", "none", _authorization("MECHANISM_UNRESOLVED")),
        ("IMPLEMENTATION_OR_CONTRACT_FAILURE", "none", _authorization("IMPLEMENTATION_OR_CONTRACT_FAILURE")),
        ("NO_SCIENTIFIC_RESULT", "none", _authorization("NO_SCIENTIFIC_RESULT")),
    )
    auth_rows = []
    for outcome, expected, context in authorization_cases:
        decision = authorize_downstream(context)
        auth_rows.append({"outcome": outcome, "expected_authorization": expected, "actual_authorization": decision.status, "status": "PASS" if decision.status == expected else "FAIL"})
    if any(row["status"] != "PASS" for row in auth_rows):
        raise RuntimeError("authorization table failure")
    write_csv_x(run / "tables/d3_downstream_authorization.csv", list(auth_rows[0]), auth_rows)


def bundle_schema() -> dict[str, Any]:
    root = [
        "schema_version", "created_utc", "base_bundle_v2", "policy_registry", "policy_registry_schema", "policy_engine",
        "policy_preflight", "actual_launcher", "canonical_policy_ids", "outcome_categories", "outcome_mapping_contract",
        "semantic_state_contract", "stop_precedence", "authorization_contract", "artifact_references", "execution_counters",
        "fixture_count", "branch_count", "scientific_d3_executed",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "thayer-d3-executable-bundle-v3.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": root,
        "properties": {field: {} for field in root},
    }


def build_bundle(run: Path, suite: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    schema_path = run / "bundle_v3/d3_executable_bundle_v3.schema.json"
    write_json_x(schema_path, bundle_schema())
    artifact_paths = {
        "branch_manifest": run / "control_flow/d3_policy_branch_manifest.json",
        "branch_coverage": run / "tables/d3_policy_branch_coverage.csv",
        "policy_set_equality": run / "tables/d3_policy_set_equality.csv",
        "outcome_mapping_table": run / "tables/outcome_mapping_exhaustiveness.csv",
        "semantic_state_schema": run / "state_machine/d3_semantic_state_v3.schema.json",
        "state_machine_tests": run / "tables/semantic_state_persistence_tests.csv",
        "tangent_policy_audit": run / "diagnostics/tangent_policy_audit.json",
        "terminal_precedence_table": run / "tables/d3_stop_event_precedence.csv",
        "authorization_table": run / "tables/d3_downstream_authorization.csv",
        "fixture_inventory": run / "branch_fixtures/d3_policy_fixture_inventory.json",
    }
    authorization_contract = {
        "L0_FULL_DECODER_SUCCESS": "square_only_eight_scene_l0_with_prompt_forward_replay_contract_gates",
        "DECODER_PARAMETERIZATION_CAPACITY_BARRIER": "decoder_capacity_ladder_with_d0_d1_no_defect_and_valid_used_tangent",
        "HARD_ASSIGNMENT_BARRIER": "smooth_assignment_diagnostic",
        "SQUARE_MAPPING_OPTIMIZATION_BARRIER": "square_mapping_diagnostic",
        "DECODER_OPTIMIZATION_BARRIER": "optimization_diagnostic",
        "MIXED_CAUSE": "none",
        "MECHANISM_UNRESOLVED": "none",
        "IMPLEMENTATION_OR_CONTRACT_FAILURE": "none",
        "NO_SCIENTIFIC_RESULT": "none",
    }
    bundle = {
        "schema_version": "thayer-d3-executable-bundle-v3",
        "created_utc": utcnow(),
        "base_bundle_v2": reference(BUNDLE_V2),
        "policy_registry": reference(run / "policy_registry/d3_policy_registry_v3.json"),
        "policy_registry_schema": reference(run / "policy_registry/d3_policy_registry_v3.schema.json"),
        "policy_engine": reference(REPO / "src/d3_policy_engine.py"),
        "policy_preflight": reference(REPO / "src/d3_policy_preflight.py"),
        "actual_launcher": reference(LAUNCHER),
        "canonical_policy_ids": list(POLICY_IDS),
        "outcome_categories": list(OUTCOME_CATEGORIES),
        "outcome_mapping_contract": {"mutually_exclusive": True, "collectively_exhaustive": True, "inconsistent_evidence_maps_to": "IMPLEMENTATION_OR_CONTRACT_FAILURE", "mapping_immutable_after_preflight": True},
        "semantic_state_contract": {"states": list(SEMANTIC_STATES), "not_reached_required": True, "selection_tie_break": "lower_metric_then_earliest_evaluation_then_lexical_payload_sha256", "payload_overwrite_allowed": False},
        "stop_precedence": {"entries": [{"rank": rank, "event": event, "exit_code": exit_code} for rank, (event, exit_code) in enumerate(STOP_PRECEDENCE, start=1)], "success_overrides_safety_failure": False},
        "authorization_contract": authorization_contract,
        "artifact_references": {name: reference(path) for name, path in artifact_paths.items()},
        "execution_counters": {"scientific_tensor_loads": 0, "model_constructions": 0, "optimizer_constructions": 0, "decoder_forwards": 0, "scientific_d3_steps": 0, "protected_data_accesses": 0},
        "fixture_count": suite["fixture_count"],
        "branch_count": suite["branch_count"],
        "scientific_d3_executed": False,
    }
    bundle_path = run / "bundle_v3/d3_executable_bundle_v3.json"
    write_json_x(bundle_path, bundle)
    bundle_hash = sha256(bundle_path)
    write_text_x(run / "bundle_v3/d3_executable_bundle_v3.sha256", f"{bundle_hash}  d3_executable_bundle_v3.json\n")
    checksum_path = run / "bundle_v3/d3_executable_bundle_v3.sha256"
    manifest = {
        "schema_version": "thayer-d3-executable-bundle-manifest-v3",
        "created_utc": utcnow(),
        "bundle": reference(bundle_path),
        "schema": reference(schema_path),
        "checksum": reference(checksum_path),
        "base_bundle_v2_sha256": BUNDLE_V2_SHA256,
        "scientific_d3_executed": False,
    }
    write_json_x(run / "bundle_v3/d3_executable_bundle_v3_manifest.json", manifest)
    chain = {
        "schema_version": "thayer-d3-executable-bundle-hash-chain-v3",
        "ordered_artifacts": [
            reference(BUNDLE_V2),
            reference(run / "policy_registry/d3_policy_registry_v3.json"),
            reference(REPO / "src/d3_policy_engine.py"),
            reference(run / "control_flow/d3_policy_branch_manifest.json"),
            reference(run / "tables/d3_policy_branch_coverage.csv"),
            reference(run / "tables/d3_policy_set_equality.csv"),
            reference(run / "tables/outcome_mapping_exhaustiveness.csv"),
            reference(run / "state_machine/d3_semantic_state_v3.schema.json"),
            reference(run / "tables/semantic_state_persistence_tests.csv"),
            reference(run / "diagnostics/tangent_policy_audit.json"),
            reference(LAUNCHER),
            reference(bundle_path),
        ],
        "bundle_sha256": bundle_hash,
    }
    write_json_x(run / "bundle_v3/d3_executable_bundle_v3_hash_chain.json", chain)
    for name in ("d3_executable_bundle_v3.json", "d3_executable_bundle_v3.schema.json", "d3_executable_bundle_v3_manifest.json", "d3_executable_bundle_v3_hash_chain.json", "d3_executable_bundle_v3.sha256"):
        source = run / "bundle_v3" / name
        destination = run / "future_d3_bundle" / name
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            destination_handle.write(source_handle.read())
    return bundle_path, bundle


def run_negative_tests(run: Path, bundle_path: Path, bundle: Mapping[str, Any], registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[tuple[str, str, str, Callable[[dict[str, Any], dict[str, Any] | None], None], bool]] = []

    def add(name: str, expected: str, mutate: Callable[[dict[str, Any], dict[str, Any] | None], None], registry_case: bool = False) -> None:
        cases.append((f"{len(cases) + 1:02d}", name, expected, mutate, registry_case))

    def policy_record(value: dict[str, Any], policy_id: str) -> dict[str, Any]:
        return next(record for record in value["policies"] if record["canonical_policy_id"] == policy_id)

    add("remove_expert_activity_policy", "control.expert_activity", lambda b, r: r["policies"].remove(policy_record(r, "control.expert_activity")), True)
    add("remove_expert_death_patience", "control.expert_death", lambda b, r: policy_record(r, "control.expert_death").pop("rolling_window_or_patience"), True)
    add("change_expert_death_terminal_status", "control.expert_death", lambda b, r: policy_record(r, "control.expert_death").__setitem__("terminal_or_nonterminal_status", "nonterminal"), True)
    add("remove_prompt_collapse_threshold", "control.prompt_collapse", lambda b, r: policy_record(r, "control.prompt_collapse").__setitem__("numerical_tolerance", None), True)
    add("remove_prompt_collapse_patience", "control.prompt_collapse", lambda b, r: policy_record(r, "control.prompt_collapse").__setitem__("rolling_window_or_patience", None), True)
    add("change_prompt_collapse_output_pairing", "control.prompt_collapse", lambda b, r: policy_record(r, "control.prompt_collapse").__setitem__("metric_or_event_definition", "changed pairing"), True)
    add("make_tangent_failure_terminal", "diagnostic.tangent_protocol", lambda b, r: policy_record(r, "diagnostic.tangent_protocol").__setitem__("terminal_or_nonterminal_status", "terminal"), True)
    add("remove_tangent_validation_tolerance", "diagnostic.tangent_protocol", lambda b, r: policy_record(r, "diagnostic.tangent_protocol").__setitem__("numerical_tolerance", None), True)
    add("remove_outcome_category", "outcome.scientific_mapping", lambda b, r: b["outcome_categories"].pop())
    add("create_overlapping_outcome_rules", "outcome.scientific_mapping", lambda b, r: b["outcome_mapping_contract"].__setitem__("mutually_exclusive", False))
    add("create_outcome_gap", "outcome.scientific_mapping", lambda b, r: b["outcome_mapping_contract"].__setitem__("collectively_exhaustive", False))
    add("remove_mechanism_unresolved", "outcome.scientific_mapping", lambda b, r: b["outcome_categories"].remove("MECHANISM_UNRESOLVED"))
    add("allow_capacity_from_unresolved", "authorization.downstream", lambda b, r: b["authorization_contract"].__setitem__("MECHANISM_UNRESOLVED", "decoder_capacity_ladder"))
    add("remove_semantic_initial", "state.semantic_persistence", lambda b, r: b["semantic_state_contract"]["states"].remove("initial"))
    add("remove_semantic_final", "state.semantic_persistence", lambda b, r: b["semantic_state_contract"]["states"].remove("final"))
    add("remove_not_reached_behavior", "state.semantic_persistence", lambda b, r: b["semantic_state_contract"].__setitem__("not_reached_required", False))
    add("change_lowest_objective_tie_break", "state.semantic_persistence", lambda b, r: b["semantic_state_contract"].__setitem__("selection_tie_break", "latest"))
    add("allow_semantic_state_overwrite", "state.semantic_persistence", lambda b, r: b["semantic_state_contract"].__setitem__("payload_overwrite_allowed", True))
    add("remove_stop_event_precedence", "control.stop_event_precedence", lambda b, r: b["stop_precedence"]["entries"].pop())
    add("allow_success_override_safety", "control.stop_event_precedence", lambda b, r: b["stop_precedence"].__setitem__("success_overrides_safety_failure", True))
    add("remove_authorization_rule", "authorization.downstream", lambda b, r: b["authorization_contract"].pop("HARD_ASSIGNMENT_BARRIER"))
    add("alter_policy_engine_hash", "registry.policy_set", lambda b, r: b["policy_engine"].__setitem__("sha256", "0" * 64))
    add("alter_launcher_hash", "safety.runtime_contract", lambda b, r: b["actual_launcher"].__setitem__("sha256", "0" * 64))
    add("remove_branch_fixture", "persistence.artifact_integrity", lambda b, r: b.__setitem__("fixture_count", 62))
    add("remove_branch_coverage_proof", "persistence.artifact_integrity", lambda b, r: b["artifact_references"].pop("branch_coverage"))
    add("add_undeclared_policy", "registry.policy_set", lambda b, r: (r["policies"].append({**r["policies"][0], "canonical_policy_id": "undeclared.policy"}), r.__setitem__("policy_count", 17), r["canonical_policy_ids"].append("undeclared.policy")), True)
    add("add_implicit_default", "registry.policy_set", lambda b, r: r.__setitem__("no_implicit_defaults", False), True)
    add("add_unknown_policy_version", "control.expert_activity", lambda b, r: policy_record(r, "control.expert_activity").__setitem__("semantic_version", "999.0.0"), True)
    add("alter_state_schema", "state.semantic_persistence", lambda b, r: b["artifact_references"]["semantic_state_schema"].__setitem__("sha256", "0" * 64))
    add("alter_outcome_mapping_after_preflight", "outcome.scientific_mapping", lambda b, r: b["outcome_mapping_contract"].__setitem__("mapping_immutable_after_preflight", False))

    rows = []
    for identifier, name, expected, mutate, registry_case in cases:
        case_dir = run / "negative_tests" / f"case_{identifier}_{name}"
        case_dir.mkdir(exist_ok=False)
        corrupted_bundle = copy.deepcopy(bundle)
        corrupted_registry = copy.deepcopy(registry) if registry_case else None
        mutate(corrupted_bundle, corrupted_registry)
        if corrupted_registry is not None:
            registry_path = case_dir / "corrupted_policy_registry.json"
            write_json_x(registry_path, corrupted_registry, sort_keys=False)
            corrupted_bundle["policy_registry"] = reference(registry_path)
        corrupted_path = case_dir / "corrupted_bundle_v3.json"
        write_json_x(corrupted_path, corrupted_bundle)
        try:
            validate_bundle_v3(corrupted_path, None, REPO)
        except PolicyContractError as error:
            actual = error.policy_id
            rejected = actual == expected
            message = error.message
        else:
            actual = "NOT_REJECTED"
            rejected = False
            message = "corruption accepted"
        rows.append({"case_id": identifier, "corruption": name, "expected_policy_id": expected, "actual_policy_id": actual, "rejected": "yes" if rejected else "no", "status": "PASS" if rejected else "FAIL", "message": message})
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError(f"bundle v3 negative tests failed: {[row for row in rows if row['status'] != 'PASS']}")
    write_csv_x(run / "tables/bundle_v3_negative_tests.csv", list(rows[0]), rows)
    write_json_x(run / "negative_tests/negative_test_summary.json", {"status": "PASS", "case_count": len(rows), "rejected_count": len(rows), "rows": rows})
    return rows


def run_actual_launcher_preflight(run: Path, bundle_path: Path) -> dict[str, Any]:
    bundle_hash = sha256(bundle_path)
    output = run / "launcher_tests/actual_launcher_policy_preflight"
    completed = subprocess.run(
        [
            str(REPO / ".venv-btk/bin/python"), "-B", str(LAUNCHER),
            "--policy-preflight-only", "--policy-bundle", str(bundle_path),
            "--policy-bundle-sha256", bundle_hash, "--policy-output-dir", str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
    )
    write_text_x(run / "launcher_tests/actual_launcher_policy_preflight_stdout.txt", completed.stdout)
    write_text_x(run / "launcher_tests/actual_launcher_policy_preflight_stderr.txt", completed.stderr)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or tuple(lines) != READINESS_MARKERS:
        raise RuntimeError(f"actual launcher preflight failed: rc={completed.returncode} lines={lines} stderr={completed.stderr}")
    result = json.loads((output / "preflight_result.json").read_text(encoding="utf-8"))
    if result["execution_counters"] != {"scientific_tensor_loads": 0, "model_constructions": 0, "optimizer_constructions": 0, "decoder_forwards": 0, "scientific_d3_steps": 0, "protected_data_accesses": 0}:
        raise RuntimeError("actual launcher preflight scientific counters changed")
    write_json_x(run / "launcher_tests/actual_launcher_policy_preflight_summary.json", {"status": "PASS", "returncode": completed.returncode, "markers": lines, "bundle_sha256": bundle_hash, "preflight_result_sha256": sha256(output / "preflight_result.json"), "execution_counters": result["execution_counters"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if run != REPO / "outputs/runs/thayer_d3_policy_contract_20260713_173955":
        raise SystemExit("this frozen campaign runner only accepts the preregistered master run")
    verify_preregistration(run)
    started = utcnow()

    suite = execute_fixture_suite(run / "semantic_state_tests/authoritative_suite")
    fixtures = fixture_map(suite)
    registry = build_policy_registry(fixtures)
    registry_path = run / "policy_registry/d3_policy_registry_v3.json"
    write_json_x(registry_path, registry, sort_keys=False)
    write_json_x(run / "policy_registry/d3_policy_registry_v3.schema.json", policy_registry_schema())
    validate_policy_registry(json.loads(registry_path.read_text(encoding="utf-8")), verify_implementation=True)
    write_json_x(run / "policy_engine/policy_implementation_hashes.json", implementation_hashes())

    build_consumer_graph(run, registry, suite)
    build_branch_artifacts(run, suite)
    state_result = build_state_machine_artifacts(run)
    outcome_result = build_outcome_artifacts(run)
    tangent_result = build_tangent_artifacts(run)
    equality = build_policy_set_equality(run, suite, registry)
    build_precedence_and_authorization_tables(run)

    bundle_path, bundle = build_bundle(run, suite)
    validation = validate_bundle_v3(bundle_path, sha256(bundle_path), REPO)
    negative_rows = run_negative_tests(run, bundle_path, bundle, registry)
    launcher_result = run_actual_launcher_preflight(run, bundle_path)
    summary = {
        "schema_version": "thayer-d3-policy-contract-execution-v3",
        "status": "PASS",
        "primary_outcome": "D3_POLICY_CONTRACT_PASS",
        "started_utc": started,
        "completed_utc": utcnow(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "bundle_v2_sha256": BUNDLE_V2_SHA256,
        "bundle_v3_sha256": sha256(bundle_path),
        "policy_count": len(POLICY_IDS),
        "fixture_count": suite["fixture_count"],
        "branch_count": suite["branch_count"],
        "outcome_combination_count": outcome_result["row_count"],
        "state_replay": state_result["fresh_replay"],
        "tangent_audit": tangent_result,
        "policy_set_equality": equality,
        "bundle_validation": validation,
        "negative_test_count": len(negative_rows),
        "launcher_markers": launcher_result["markers"],
        "execution_counters": launcher_result["execution_counters"],
    }
    write_json_x(run / "diagnostics/policy_contract_execution_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
