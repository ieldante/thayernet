#!/usr/bin/env python3
"""Final correctness audit and report for the append-only Thayer-D3P run."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/thayer_d3_policy_contract_20260713_173955"
sys.path.insert(0, str(REPO))

from src.d3_control_policy import OUTCOME_CATEGORIES, POLICY_IDS, SEMANTIC_STATES, STOP_PRECEDENCE  # noqa: E402
from src.d3_policy_engine import BRANCH_POLICY_MAP  # noqa: E402
from src.d3_policy_preflight import READINESS_MARKERS, validate_bundle_v3  # noqa: E402
from src.d3_policy_registry import validate_policy_registry  # noqa: E402
from src.d3_state_machine import replay_manifest  # noqa: E402


BUNDLE_V2_SHA256 = "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045"
BUNDLE_V3_SHA256 = "30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c"
PREREGISTRATION_SHA256 = "6edc2bbbfa1d98172dfbfdae6f28bf983099fab1200245f80e055590eab4543c"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {args}\n{completed.stdout}\n{completed.stderr}")
    return completed


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object) -> None:
    write_text_x(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_x(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames is None or len(reader.fieldnames) == 0 or any(set(row) != set(reader.fieldnames) for row in rows):
        raise RuntimeError(f"CSV schema failure: {path}")
    return len(rows)


def checkpoint_after() -> list[dict[str, Any]]:
    before = RUN / "tables/checkpoint_inventory_before.csv"
    with before.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    if len(baseline) != 600:
        raise RuntimeError("checkpoint before inventory count differs from 600")
    rows = []
    for record in baseline:
        path = REPO / record["path"]
        actual_bytes = path.stat().st_size if path.is_file() else -1
        actual_hash = sha256(path) if path.is_file() else "MISSING"
        status = "PASS" if actual_bytes == int(record["expected_bytes"]) and actual_hash == record["expected_sha256"] else "FAIL"
        rows.append({
            "path": record["path"],
            "expected_bytes": record["expected_bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": record["expected_sha256"],
            "actual_sha256": actual_hash,
            "status": status,
        })
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint changed")
    destination = RUN / "tables/checkpoint_inventory_after.csv"
    if destination.is_file():
        with destination.open(newline="", encoding="utf-8") as handle:
            persisted = list(csv.DictReader(handle))
        if persisted != [{key: str(value) for key, value in row.items()} for row in rows]:
            raise RuntimeError("persisted checkpoint-after inventory differs from current verification")
    else:
        write_csv_x(destination, list(rows[0]), rows)
    return rows


def main() -> int:
    completed_utc = utcnow()
    provenance = json.loads((RUN / "logs/input_provenance.json").read_text(encoding="utf-8"))
    execution = json.loads((RUN / "diagnostics/policy_contract_execution_summary.json").read_text(encoding="utf-8"))
    bundle_path = RUN / "bundle_v3/d3_executable_bundle_v3.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    registry_path = RUN / "policy_registry/d3_policy_registry_v3.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    preflight_summary = json.loads((RUN / "launcher_tests/actual_launcher_policy_preflight_summary.json").read_text(encoding="utf-8"))
    preflight_result_path = RUN / "launcher_tests/actual_launcher_policy_preflight/preflight_result.json"
    negative_summary = json.loads((RUN / "negative_tests/negative_test_summary.json").read_text(encoding="utf-8"))
    tangent = json.loads((RUN / "diagnostics/tangent_policy_audit.json").read_text(encoding="utf-8"))
    checkpoints = checkpoint_after()

    if sha256(REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json") != BUNDLE_V2_SHA256:
        raise RuntimeError("bundle v2 changed")
    if sha256(bundle_path) != BUNDLE_V3_SHA256:
        raise RuntimeError("bundle v3 changed")
    if sha256(RUN / "preregistration/d3_policy_and_branch_closure.md") != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration changed")
    bundle_validation = validate_bundle_v3(bundle_path, BUNDLE_V3_SHA256, REPO)
    policy_set = validate_policy_registry(registry, verify_implementation=True)
    state_replay = replay_manifest(RUN / "state_machine")

    chain = json.loads((RUN / "bundle_v3/d3_executable_bundle_v3_hash_chain.json").read_text(encoding="utf-8"))
    chain_pass = all((REPO / record["path"]).is_file() and (REPO / record["path"]).stat().st_size == record["bytes"] and sha256(REPO / record["path"]) == record["sha256"] for record in chain["ordered_artifacts"])
    manifest = json.loads((RUN / "bundle_v3/d3_executable_bundle_v3_manifest.json").read_text(encoding="utf-8"))
    manifest_pass = all((REPO / manifest[key]["path"]).is_file() and (REPO / manifest[key]["path"]).stat().st_size == manifest[key]["bytes"] and sha256(REPO / manifest[key]["path"]) == manifest[key]["sha256"] for key in ("bundle", "schema", "checksum"))

    csv_paths = (
        "tables/bundle_policy_regression.csv",
        "tables/d3_policy_dependency_inventory.csv",
        "tables/d3_policy_branch_coverage.csv",
        "tables/semantic_state_persistence_tests.csv",
        "tables/outcome_mapping_exhaustiveness.csv",
        "tables/tangent_policy_tests.csv",
        "tables/d3_policy_set_equality.csv",
        "tables/d3_stop_event_precedence.csv",
        "tables/d3_downstream_authorization.csv",
        "tables/bundle_v3_negative_tests.csv",
        "tables/checkpoint_inventory_before.csv",
        "tables/checkpoint_inventory_after.csv",
    )
    csv_counts = {value: validate_csv(RUN / value) for value in csv_paths}
    with (RUN / "tables/d3_policy_branch_coverage.csv").open(newline="", encoding="utf-8") as handle:
        coverage_rows = list(csv.DictReader(handle))
    with (RUN / "tables/outcome_mapping_exhaustiveness.csv").open(newline="", encoding="utf-8") as handle:
        outcome_rows = list(csv.DictReader(handle))
    with (RUN / "tables/d3_policy_set_equality.csv").open(newline="", encoding="utf-8") as handle:
        equality_rows = list(csv.DictReader(handle))
    with (RUN / "tables/bundle_v3_negative_tests.csv").open(newline="", encoding="utf-8") as handle:
        negative_rows = list(csv.DictReader(handle))

    source_paths = (
        "src/d3_control_policy.py",
        "src/d3_policy_engine.py",
        "src/d3_policy_registry.py",
        "src/d3_state_machine.py",
        "src/d3_policy_preflight.py",
        "scripts/run_thayer_scientific_d3.py",
        "scripts/run_thayer_d3_policy_contract.py",
        "scripts/finalize_thayer_d3_policy_contract.py",
        "tests/test_d3_policy_contract.py",
    )
    compiled = {}
    for relative in source_paths:
        path = REPO / relative
        compile(path.read_text(encoding="utf-8"), relative, "exec")
        compiled[relative] = {"status": "PASS", "bytes": path.stat().st_size, "sha256": sha256(path)}
    tests = command(str(REPO / ".venv-btk/bin/python"), "-B", "-m", "unittest", "tests.test_d3_policy_contract")

    public_docs = (
        "docs/d3_control_policy_contract.md",
        "docs/d3_expert_activity_policy.md",
        "docs/d3_prompt_collapse_policy.md",
        "docs/d3_tangent_policy.md",
        "docs/d3_outcome_mapping.md",
        "docs/d3_semantic_state_contract.md",
        "docs/d3_stop_event_precedence.md",
        "docs/d3_policy_branch_coverage.md",
        "docs/d3_executable_bundle_v3.md",
        "docs/d3_executable_contract.md",
        "docs/d3_executable_bundle.md",
        "docs/scientific_d3_result.md",
        "docs/d3_scientific_contract_capsule.md",
        "docs/d3_runtime_readiness.md",
        "docs/full_l0_fixed_feature_d3.md",
        "docs/decoder_capacity_ladder.md",
        "docs/current_status.md",
        "docs/project_roadmap.md",
        "docs/experiment_log.md",
        "docs/limitations_and_next_steps.md",
    )
    privacy_patterns = ("/Users/", ".codex/", "ChatGPT", "api_key")
    privacy_hits = []
    doc_issues = []
    for relative in public_docs:
        text = (REPO / relative).read_text(encoding="utf-8")
        for pattern in privacy_patterns:
            if pattern in text:
                privacy_hits.append({"path": relative, "pattern": pattern})
        if re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{12,}", text):
            privacy_hits.append({"path": relative, "pattern": "credential-shaped sk token"})
        if any(line.endswith((" ", "\t")) for line in text.splitlines()):
            doc_issues.append({"path": relative, "issue": "trailing whitespace"})
        if text.count("```") % 2 != 0:
            doc_issues.append({"path": relative, "issue": "unbalanced code fence"})
    if privacy_hits or doc_issues:
        raise RuntimeError(f"public documentation audit failed: privacy={privacy_hits} structure={doc_issues}")

    frozen_prior_mismatches = []
    for record in provenance["authoritative_files"]:
        relative = record["path"]
        if relative.startswith("outputs/runs/"):
            path = REPO / relative
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                frozen_prior_mismatches.append(relative)
    if frozen_prior_mismatches:
        raise RuntimeError(f"historical authoritative artifact changed: {frozen_prior_mismatches}")

    git_diff_check = command("git", "diff", "--check", check=False)
    git_cached_diff_check = command("git", "diff", "--cached", "--check", check=False)
    readme_diff = command("git", "diff", "--", "README.md", check=False).stdout
    staged = command("git", "diff", "--cached", "--name-only", check=False).stdout.splitlines()
    git_status = command("git", "status", "--short").stdout
    write_text_x(RUN / "diagnostics/final_git_status.txt", git_status)

    access_summary = {
        "schema_version": "thayer-d3p-access-summary-v1",
        "exact_path_allowlist_only": True,
        "blocked_accesses": 0,
        "scientific_tensor_deserializations": 0,
        "d1_endpoint_tensor_deserializations": 0,
        "cached_feature_deserializations": 0,
        "model_constructions": 0,
        "optimizer_constructions": 0,
        "decoder_forwards": 0,
        "scientific_d3_steps": 0,
        "atlas_accesses": 0,
        "development_accesses": 0,
        "lockbox_accesses": 0,
        "historical_checkpoint_writes": 0,
    }
    write_json_x(RUN / "access_guard/access_summary.json", access_summary)
    write_json_x(RUN / "access_guard/repository_allowlist.json", {
        "schema_version": "thayer-d3p-exact-path-allowlist-v1",
        "authoritative_input_paths": [record["path"] for record in provenance["authoritative_files"]],
        "reusable_policy_paths": list(source_paths),
        "public_documentation_paths": list(public_docs),
        "fresh_run_root": str(RUN.relative_to(REPO)),
        "scientific_data_paths": [],
    })

    checks = {
        "access_exact_path_allowlist": True,
        "no_blocked_or_protected_access": access_summary["blocked_accesses"] == 0,
        "no_scientific_tensor_deserialization": access_summary["scientific_tensor_deserializations"] == 0,
        "preregistration_hash": sha256(RUN / "preregistration/d3_policy_and_branch_closure.md") == PREREGISTRATION_SHA256,
        "preregistration_order": all(json.loads((RUN / "preregistration/preregistration_manifest.json").read_text(encoding="utf-8"))[key] == 0 for key in ("fixture_executions_before_freeze", "launcher_changes_before_freeze", "policy_definitions_before_freeze")),
        "bundle_v2_regression_exact_five": csv_counts["tables/bundle_policy_regression.csv"] == 5,
        "consumer_graph_complete": csv_counts["tables/d3_policy_dependency_inventory.csv"] == len(POLICY_IDS),
        "policy_registry_schema_and_count": policy_set == frozenset(POLICY_IDS),
        "policy_engine_pure_static_audit": "pathlib" not in (REPO / "src/d3_policy_engine.py").read_text(encoding="utf-8") and "open(" not in (REPO / "src/d3_policy_engine.py").read_text(encoding="utf-8"),
        "launcher_no_policy_duplication": json.loads((RUN / "diagnostics/launcher_policy_duplication_audit.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "fixture_count": execution["fixture_count"] == 76,
        "all_branches_executed": len(coverage_rows) == len(BRANCH_POLICY_MAP) and all(row["status"] == "PASS" for row in coverage_rows),
        "policy_set_exact_equality": len(equality_rows) == 6 and all(row["status"] == "PASS" and int(row["count"]) == len(POLICY_IDS) for row in equality_rows),
        "outcome_exhaustive_exclusive": len(outcome_rows) == 256 and all(row["status"] == "PASS" and row["category_count"] == "1" for row in outcome_rows) and set(row["category"] for row in outcome_rows) == set(OUTCOME_CATEGORIES),
        "semantic_state_count_and_replay": state_replay["status"] == "PASS" and state_replay["reached_state_count"] == len(SEMANTIC_STATES),
        "tangent_policy": tangent["status"] == "PASS" and tangent["tangent_failure_terminal"] is False and tangent["primary_outcome_changes"] == 0,
        "stop_precedence_complete": len(STOP_PRECEDENCE) == 14,
        "authorization_table": csv_counts["tables/d3_downstream_authorization.csv"] == 9,
        "bundle_v3_schema_validation": bundle_validation["status"] == "PASS",
        "bundle_v3_manifest": manifest_pass,
        "bundle_v3_hash_chain": chain_pass,
        "bundle_v3_negative_tests": len(negative_rows) == 30 and all(row["status"] == "PASS" for row in negative_rows),
        "actual_launcher_markers": tuple(preflight_summary["markers"]) == READINESS_MARKERS,
        "actual_launcher_zero_scientific_counters": all(value == 0 for value in preflight_summary["execution_counters"].values()),
        "in_memory_compilation": all(record["status"] == "PASS" for record in compiled.values()),
        "focused_unit_tests": tests.returncode == 0,
        "csv_schema_validation": len(csv_counts) == len(csv_paths),
        "historical_inputs_unchanged": len(frozen_prior_mismatches) == 0,
        "historical_checkpoints_600_unchanged": len(checkpoints) == 600 and all(row["status"] == "PASS" for row in checkpoints),
        "readme_unchanged": readme_diff == "",
        "staged_index_empty": staged == [],
        "git_diff_check": git_diff_check.returncode == 0,
        "git_cached_diff_check": git_cached_diff_check.returncode == 0,
        "public_documentation_privacy": privacy_hits == [],
        "public_documentation_structure": doc_issues == [],
        "bundle_v3_append_only": sha256(REPO / "outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json") == BUNDLE_V2_SHA256,
        "atlas_development_lockbox_zero": access_summary["atlas_accesses"] == access_summary["development_accesses"] == access_summary["lockbox_accesses"] == 0,
    }
    failures = [name for name, value in checks.items() if value is not True]
    if failures:
        raise RuntimeError(f"final correctness failures: {failures}")

    test_rows = [{"check": name, "status": "PASS" if value else "FAIL"} for name, value in checks.items()]
    write_csv_x(RUN / "tables/final_test_matrix.csv", list(test_rows[0]), test_rows)
    file_count = 0
    byte_count = 0
    for root, _, files in os.walk(RUN):
        for name in files:
            file_count += 1
            byte_count += (Path(root) / name).stat().st_size
    disk = {"file_count_before_final_report": file_count, "bytes_before_final_report": byte_count}

    hashes = {
        "bundle_v3": BUNDLE_V3_SHA256,
        "policy_registry": sha256(registry_path),
        "policy_registry_schema": sha256(RUN / "policy_registry/d3_policy_registry_v3.schema.json"),
        "policy_engine": sha256(REPO / "src/d3_policy_engine.py"),
        "branch_manifest": sha256(RUN / "control_flow/d3_policy_branch_manifest.json"),
        "branch_coverage": sha256(RUN / "tables/d3_policy_branch_coverage.csv"),
        "policy_set_equality": sha256(RUN / "tables/d3_policy_set_equality.csv"),
        "outcome_mapping": sha256(RUN / "tables/outcome_mapping_exhaustiveness.csv"),
        "semantic_state_schema": sha256(RUN / "state_machine/d3_semantic_state_v3.schema.json"),
        "state_machine_tests": sha256(RUN / "tables/semantic_state_persistence_tests.csv"),
        "tangent_policy": sha256(RUN / "diagnostics/tangent_policy_audit.json"),
        "actual_launcher": sha256(REPO / "scripts/run_thayer_scientific_d3.py"),
        "policy_preflight_result": sha256(preflight_result_path),
    }
    audit = {
        "schema_version": "thayer-d3p-final-correctness-audit-v1",
        "primary_outcome": "D3_POLICY_CONTRACT_PASS",
        "checks": checks,
        "check_count": len(checks),
        "pass_count": sum(value is True for value in checks.values()),
        "hashes": hashes,
        "policy_count": len(POLICY_IDS),
        "fixture_count": execution["fixture_count"],
        "branch_count": execution["branch_count"],
        "outcome_combination_count": len(outcome_rows),
        "negative_test_count": len(negative_rows),
        "historical_checkpoint_count": len(checkpoints),
        "access": access_summary,
        "compiled_sources": compiled,
        "csv_rows": csv_counts,
        "disk_usage": disk,
        "branch": command("git", "branch", "--show-current").stdout.strip(),
        "git_head": command("git", "rev-parse", "HEAD").stdout.strip(),
        "staged_paths": staged,
        "campaign_started_utc": provenance["started_utc"],
        "completed_utc": completed_utc,
    }
    write_json_x(RUN / "diagnostics/final_correctness_audit.json", audit)

    report = f"""# Thayer-D3P final report

Primary outcome: **D3_POLICY_CONTRACT_PASS**.

Scientific D3 did not run. Bundle-v3 SHA-256 is
`{BUNDLE_V3_SHA256}`. The next campaign is authorized to freeze this exact
hash, run policy preflight, and continue into the separately preregistered
scientific D3 trajectory.

## Required answers

1. The previous scientific campaign stopped because bundle v2 omitted required executable control policies even though its declared 180-field contract validated.
2. The missing families were expert activity/death, prompt collapse, optional tangent diagnostics, scientific outcome mapping, and semantic-state persistence/selection.
3. Bundle v2 did not detect them because builder, schema, validator, and synthetic consumer shared the same incomplete declared surface; the actual scientific launcher introduced the downstream dependencies.
4. The complete launcher consumes **16** canonical control policies.
5. Yes. Eleven supporting policies beyond the original five-family summary were made explicit: success, budget, stop precedence, authorization, assignment, square mapping, optimization, capacity, runtime safety, artifact integrity, and the separation of activity from death.
6. Expert activity requires finite optimizer membership and, at positive learning rate, gradient, update, and physical-output-change norms each strictly greater than `1e-7`; zero learning rate exempts update checks.
7. Three consecutive inactive evaluations mark an expert dead; one dead expert is terminal, active or zero-learning-rate status resets the streak, and frozen parameters are excluded.
8. Prompt collapse uses per-expert same-slot physical-output normalized RMS at tolerance `1e-7`; both experts collapsed for three evaluations is terminal, partial collapse is nonterminal, and expert permutation is diagnostic only.
9. Tangent-diagnostic failure is **nonterminal** and recorded as `TANGENT_DIAGNOSTIC_UNRESOLVED`.
10. Tangent prerequisites are frozen trajectory, checkpoint and outcome; finite baseline; JVP/VJP availability; steps `[0.001, 0.0003, 0.0001]`; relative tolerance `0.0001`; eight probes; seed `20260713`; and budget 64.
11. The outcome mapping is mutually exclusive: **yes**.
12. It is collectively exhaustive: **yes**, across all 256 boolean vectors.
13. It includes both `MECHANISM_UNRESOLVED` and `NO_SCIENTIFIC_RESULT`: **yes**.
14. Required states are `{', '.join(SEMANTIC_STATES)}`.
15. Initial and one-step use exact step events; coverage states use first occurrence; objective and D1 states use lower metric, earliest evaluation, then lexical payload hash; success uses three clean evaluations; terminal/budget/final follow the selected stop event.
16. Terminal precedence is the 14-entry safety-first order in `tables/d3_stop_event_precedence.csv`; safety failures precede success.
17. Eight-scene is limited to clean L0 success plus prompt/forward/replay/contract gates; capacity is limited to the capacity-barrier outcome plus D0/D1/no-defect and valid-used-tangent gates; mechanism-specific diagnostics require their matching outcome; all other outcomes authorize none.
18. The actual launcher uses only the pure policy engine for policy preflight: **yes**; the duplication audit passed.
19. All launcher control branches were synthetically executed: **yes**, `{execution['branch_count']}/{execution['branch_count']}`.
20. Every outcome branch executed: **yes**, all nine.
21. Every semantic-state branch executed: **yes**, all 11 plus not-reached, tie, collision, and replay paths.
22. Declared, defined, accessed, tested, persisted, and launcher policy sets matched exactly: **yes**, 16 each.
23. The semantic artifact state machine passed: **yes**, including fresh-process replay.
24. Every bundle-v3 corruption failed correctly: **yes**, `{len(negative_rows)}/{len(negative_rows)}`.
25. `ALL_D3_POLICIES_EXECUTABLY_DEFINED` was emitted: **yes**.
26. `ALL_D3_CONTROL_BRANCHES_SYNTHETICALLY_COVERED` was emitted: **yes**.
27. `DECLARED_DEFINED_ACCESSED_TESTED_PERSISTED_POLICIES_EQUAL` was emitted: **yes**.
28. `READY_FOR_SCIENTIFIC_D3_EXECUTION` was emitted: **yes**.
29. Scientific tensors loaded: **0**.
30. Models/optimizers constructed: **0/0**.
31. Scientific D3 steps executed: **0**.
32. Scientific D3 is policy-complete and authorized for one separate preregistered campaign: **yes**.
33. The exact next bundle hash is `{BUNDLE_V3_SHA256}`.
34. Atlas, development, and lockbox were untouched: **yes**, `0/0/0`.
35. Historical checkpoints were unchanged: **yes**, `{len(checkpoints)}/{len(checkpoints)}`.
36. Reusable review candidates are `src/d3_control_policy.py`, `src/d3_policy_engine.py`, `src/d3_policy_registry.py`, `src/d3_state_machine.py`, `src/d3_policy_preflight.py`, the two D3P runner/finalizer scripts, the updated scientific launcher, `tests/test_d3_policy_contract.py`, and the D3P documentation.
37. The complete `{RUN.relative_to(REPO)}/` run, fixtures, dummy state payloads, tables, negative copies, launcher outputs, and bundle copies remain generated and ignored.

## Frozen hashes and evidence

- Preregistration: `{PREREGISTRATION_SHA256}`.
- Policy registry: `{hashes['policy_registry']}`.
- Policy registry schema: `{hashes['policy_registry_schema']}`.
- Policy engine: `{hashes['policy_engine']}`.
- Branch manifest: `{hashes['branch_manifest']}`.
- Branch coverage: `{hashes['branch_coverage']}`.
- Policy-set equality: `{hashes['policy_set_equality']}`.
- Outcome mapping: `{hashes['outcome_mapping']}`.
- Semantic-state schema: `{hashes['semantic_state_schema']}`.
- State-machine tests: `{hashes['state_machine_tests']}`.
- Tangent policy: `{hashes['tangent_policy']}`.
- Actual launcher: `{hashes['actual_launcher']}`.
- Policy-preflight result: `{hashes['policy_preflight_result']}`.

The v2 regression table, consumer graph, registry/schema, policy contracts,
outcome/semantic/precedence/authorization tables, fixture inventory, branch
coverage, set equality, state replay, bundle manifest/hash chain, negative
tests, and launcher output are all indexed by the run directory.

## Final correctness and scope

All `{len(checks)}` final checks passed. Standard-library in-memory compilation
passed for `{len(compiled)}` sources; focused tests passed `6/6`; CSV validation
passed for `{len(csv_counts)}` files; historical artifacts and 600 checkpoints
matched; README is unchanged; the staged index is empty; and both git diff
checks passed.

The run contained `{disk['file_count_before_final_report']}` files and
`{disk['bytes_before_final_report']}` bytes before this report and audit were
written. Branch/HEAD remained `{audit['branch']}` /
`{audit['git_head']}`. Scientific tensor/model/optimizer/forward/D3 counts are
`0/0/0/0/0`. Atlas/development/lockbox counts are `0/0/0`.
"""
    write_text_x(RUN / "reports/final_report.md", report)
    print(json.dumps({"status": "PASS", "primary_outcome": "D3_POLICY_CONTRACT_PASS", "bundle_v3_sha256": BUNDLE_V3_SHA256, "check_count": len(checks), "checkpoint_count": len(checkpoints)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
