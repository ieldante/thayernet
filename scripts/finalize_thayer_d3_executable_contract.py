#!/usr/bin/env python3
"""Fail-closed final audit and report for one completed Thayer-D3E run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
EXPECTED_RUN = Path("outputs/runs/thayer_d3_executable_contract_20260713_164320")
EXPECTED_PREREGISTRATION = "b5a69f70c0f24f287da1f70a4a33e876fe9c8186be7c4e3c0eea67804bf1eede"
EXPECTED_REGISTRY = "a1af885bc8e1c6b6bc33395920eb4b279151e51663444e6e303c2f1cfc34660f"
EXPECTED_BUNDLE = "884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045"
EXPECTED_CAPSULE = "0cbfaab01451de732ad41094bfe1a8b489534a5f90164ce0f6c75d9db36ebe7d"

SOURCE_FILES = (
    "src/d3_artifact_metadata.py",
    "src/d3_requirement_registry.py",
    "src/d3_executable_contract.py",
    "scripts/bootstrap_thayer_d3_executable_contract.py",
    "scripts/build_d3_executable_capsule_v2.py",
    "scripts/validate_d3_executable_capsule_v2.py",
    "scripts/run_thayer_d3_synthetic_preflight.py",
    "scripts/replay_thayer_d3_synthetic_checkpoint.py",
    "scripts/run_thayer_authoritative_d3_v2.py",
    "scripts/run_thayer_d3_executable_contract.py",
    "scripts/finalize_thayer_d3_executable_contract.py",
    "tests/test_d3_executable_contract.py",
)

DOC_FILES = (
    "docs/d3_executable_contract.md",
    "docs/d3_requirement_registry.md",
    "docs/d3_l0_architecture_contract.md",
    "docs/d3_tensor_member_contract.md",
    "docs/d3_synthetic_full_stack_preflight.md",
    "docs/d3_executable_bundle.md",
    "docs/d3_scientific_contract_capsule.md",
    "docs/d3_scientific_artifact_contract.md",
    "docs/d3_runtime_readiness.md",
    "docs/d1_endpoint_persistence.md",
    "docs/full_l0_fixed_feature_d3.md",
    "docs/decoder_capacity_ladder.md",
    "docs/current_status.md",
    "docs/project_roadmap.md",
    "docs/experiment_log.md",
    "docs/limitations_and_next_steps.md",
)

NINE_MISSING = (
    "capsule_artifact_d1_endpoint_manifest",
    "capsule_artifact_d0_persisted_evidence",
    "capsule_artifact_d1_persisted_evidence",
    "capsule_artifact_d2_persisted_evidence",
    "capsule_frozen_l0_decoder_topology_code",
    "capsule_frozen_decoder_parameter_count",
    "capsule_frozen_decoder_initialization_seeds",
    "capsule_d1_final_objective_evidence",
    "capsule_member_shape_dtype_endianness_expectations",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def run_checked(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def exact_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def checkpoint_closure(before_path: Path) -> tuple[str, int]:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=(
            "path",
            "expected_bytes",
            "actual_bytes",
            "expected_sha256",
            "actual_sha256",
            "status",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    rows = list(csv.DictReader(before_path.open("r", encoding="utf-8", newline="")))
    if len(rows) != 600:
        raise RuntimeError(f"expected 600 checkpoint rows, found {len(rows)}")
    for row in rows:
        path = REPO / row["path"]
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        passed = (
            actual_bytes == int(row["expected_bytes"])
            and actual_sha256 == row["expected_sha256"]
        )
        writer.writerow(
            {
                "path": row["path"],
                "expected_bytes": row["expected_bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": row["expected_sha256"],
                "actual_sha256": actual_sha256,
                "status": "PASS" if passed else "FAIL",
            }
        )
        if not passed:
            raise RuntimeError(f"historical checkpoint changed: {row['path']}")
    return output.getvalue(), len(rows)


def validate_csvs(run: Path) -> int:
    names = (
        "capsule_consumer_drift.csv",
        "consumer_negative_tests.csv",
        "container_member_inventory.csv",
        "container_member_validation.csv",
        "d3_requirement_registry.csv",
        "frozen_input_hash_validation.csv",
        "l0_architecture_inventory.csv",
        "l0_parameter_inventory.csv",
        "l0_state_dict_validation.csv",
        "optimizer_parameter_inventory.csv",
        "requirement_set_equality.csv",
        "synthetic_tensor_inventory.csv",
    )
    for name in names:
        path = run / "tables" / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or any(not column for column in header):
                raise RuntimeError(f"invalid CSV header: {path}")
            list(reader)
    return len(names)


def disk_usage(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for directory, _, names in os.walk(path):
        for name in names:
            item = Path(directory) / name
            total += item.stat().st_size
            count += 1
    return total, count


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_relative = args.run
    if run_relative.is_absolute():
        run_relative = run_relative.relative_to(REPO)
    if run_relative != EXPECTED_RUN:
        raise RuntimeError(f"unexpected run path: {run_relative}")
    run = REPO / run_relative

    outputs = (
        run / "tables/checkpoint_inventory_after.csv",
        run / "diagnostics/final_correctness_audit.json",
        run / "diagnostics/final_git_status.txt",
        run / "reports/final_report.md",
    )
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        raise RuntimeError(f"exclusive output collision: {collisions}")

    summary = load_json(run / "diagnostics/campaign_execution_summary.json")
    prereg = load_json(run / "preregistration/preregistration_freeze.json")
    equality = load_json(run / "consumer_contract/requirement_set_equality.json")
    synthetic = load_json(run / "synthetic_execution/synthetic_preflight_result.json")
    architecture = load_json(run / "architecture_audit/l0_architecture_audit.json")
    optimizer = load_json(run / "optimizer_audit/synthetic_optimizer_audit.json")
    replay = load_json(run / "checkpoint_replay/fresh_process_replay.json")
    bundle_manifest = load_json(run / "future_d3_bundle/d3_executable_bundle_v2_manifest.json")

    assertions = {
        "campaign_pending_final_audit": summary["status"] == "PASS_PENDING_DOCUMENTATION_AND_FINAL_AUDIT",
        "preregistration_hash": prereg["sha256"] == EXPECTED_PREREGISTRATION,
        "preregistration_order": all(
            prereg[key]
            for key in (
                "predates_container_member_inspection",
                "predates_model_import",
                "predates_model_construction",
                "predates_optimizer_construction",
            )
        ),
        "nine_missing_exact": tuple(summary["exact_missing_requirements_reproduced"]) == NINE_MISSING,
        "registry_identity": summary["registry_sha256"] == EXPECTED_REGISTRY,
        "registry_count": summary["requirement_count"] == 180,
        "required_set_equality": summary["set_equality"] == "PASS" and equality["status"] == "PASS",
        "capsule_v2": summary["capsule_v2"]["capsule"]["sha256"] == EXPECTED_CAPSULE,
        "container_members": summary["container_member_audit"] == "PASS",
        "architecture": summary["architecture"] == "PASS" and architecture["status"] == "PASS",
        "parameter_counts": architecture["parameter_counts"] == [46470, 46470],
        "synthetic_preflight": synthetic["status"] == "PASS",
        "optimizer": optimizer["status"] == "PASS",
        "checkpoint_replay": replay["status"] == "PASS",
        "negative_tests": summary["negative_test_count"] == 25,
        "actual_consumer": summary["actual_consumer"] == "PASS",
        "requirement_closure": (
            synthetic["requirement_closure"]["declared_required_count"] == 180
            and synthetic["requirement_closure"]["accessed_or_validated_count"] == 180
            and not synthetic["requirement_closure"]["unaccessed_required"]
            and not synthetic["requirement_closure"]["undeclared_accesses"]
        ),
        "consumer_markers": synthetic["markers"] == [
            "ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED",
            "READY_FOR_AUTHORITATIVE_D3_EXECUTION",
        ],
        "no_scientific_values": summary["scientific_array_values_loaded"] == 0,
        "no_scientific_d3": summary["scientific_d3_steps"] == 0,
        "protected_access_zero": all(summary[key] == 0 for key in ("atlas_access", "development_access", "lockbox_access")),
        "bundle_identity": bundle_manifest["bundle"]["sha256"] == EXPECTED_BUNDLE,
    }
    failed = sorted(name for name, passed in assertions.items() if not passed)
    if failed:
        raise RuntimeError(f"campaign evidence gates failed: {failed}")

    compile_results: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = (REPO / relative).read_text(encoding="utf-8")
        compile(source, relative, "exec", dont_inherit=True)
        compile_results[relative] = "PASS"

    env = dict(os.environ)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "THAYER_D3E_REGISTRY": str(run / "requirement_registry/d3_requirement_registry.json"),
            "THAYER_D3E_CAPSULE": str(run / "capsule_v2/d3_executable_capsule_v2.json"),
        }
    )
    test_output = run_checked(
        [str(REPO / ".venv-btk/bin/python"), "-B", "-m", "unittest", "-v", "tests.test_d3_executable_contract"],
        env=env,
    )
    if "Ran 5 tests" not in test_output or "OK" not in test_output:
        raise RuntimeError(f"unexpected focused-test output:\n{test_output}")

    csv_count = validate_csvs(run)
    after_csv, checkpoint_count = checkpoint_closure(run / "tables/checkpoint_inventory_before.csv")

    forbidden_terms = (
        "/" + "Users/",
        "Chat" + "GPT",
        "Open" + "AI",
        "generated by " + "AI",
        "generated by an " + "AI",
        "artificial " + "intelligence",
    )
    privacy_pattern = re.compile(
        "|".join(re.escape(term) for term in forbidden_terms),
        flags=re.IGNORECASE,
    )
    privacy_findings: list[str] = []
    for relative in SOURCE_FILES + DOC_FILES:
        for line_number, line in enumerate((REPO / relative).read_text(encoding="utf-8").splitlines(), start=1):
            if privacy_pattern.search(line):
                privacy_findings.append(f"{relative}:{line_number}")
    if privacy_findings:
        raise RuntimeError(f"privacy/path audit failed: {privacy_findings}")

    run_checked(["git", "diff", "--check"])
    run_checked(["git", "diff", "--cached", "--check"])
    run_checked(["git", "diff", "--quiet", "--", "README.md"])
    run_checked(["git", "diff", "--cached", "--quiet"])
    git_status = run_checked(["git", "status", "--short"])
    staged_count = len(run_checked(["git", "diff", "--cached", "--name-only"]).splitlines())
    if staged_count != 0:
        raise RuntimeError("staged index is not empty")

    runtime_output = run_checked(
        [
            str(REPO / ".venv-btk/bin/python"),
            "-B",
            "-c",
            (
                "import json,platform,torch; "
                "print(json.dumps({'python':platform.python_version(),"
                "'platform':platform.platform(), 'torch':torch.__version__,"
                "'mps_available':torch.backends.mps.is_available()}))"
            ),
        ],
        env=env,
    )
    runtime = json.loads(runtime_output)
    run_bytes, run_files = disk_usage(run)
    completed_utc = datetime.now(timezone.utc).isoformat()

    correctness = {
        "schema_version": "thayer-d3e-final-correctness-audit-v1",
        "completed_utc": completed_utc,
        "primary_outcome": "EXECUTABLE D3 CONTRACT PASS",
        "checks": {
            **assertions,
            "standard_library_in_memory_compilation": True,
            "focused_unit_tests": True,
            "csv_schema_validation": True,
            "checkpoint_hash_closure": checkpoint_count == 600,
            "readme_unchanged": True,
            "staged_index_empty": staged_count == 0,
            "git_diff_check": True,
            "allowlist_privacy_path_audit": not privacy_findings,
            "exclusive_final_paths": True,
        },
        "compiled_sources": compile_results,
        "focused_test_count": 5,
        "csv_file_count": csv_count,
        "historical_checkpoint_count": checkpoint_count,
        "runtime": runtime,
        "disk_usage": {"bytes_before_final_outputs": run_bytes, "files_before_final_outputs": run_files},
        "scientific_array_values_loaded": 0,
        "scientific_d3_steps": 0,
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "readme_unchanged": True,
        "staged_index_count": staged_count,
        "bundle_sha256": EXPECTED_BUNDLE,
    }
    if not all(correctness["checks"].values()):
        raise RuntimeError("final correctness matrix contains a failure")

    final_report = f"""# Thayer-D3E Final Report

Primary outcome: **EXECUTABLE D3 CONTRACT PASS**.

This is an executable-contract authorization result. No scientific array value
was loaded, no scientific D3 step was executed, and D3 remains scientifically
unknown until the next separately preregistered campaign.

## Answers to the required questions

1. **Why did the previous capsule-driven D3 stop?** Its actual consumer found
   nine required declarations absent from capsule v1 and failed closed before
   scientific value loading, model construction, optimizer construction, or a
   decoder forward.
2. **What were the exact nine missing requirements?** `{NINE_MISSING[0]}`,
   `{NINE_MISSING[1]}`, `{NINE_MISSING[2]}`, `{NINE_MISSING[3]}`,
   `{NINE_MISSING[4]}`, `{NINE_MISSING[5]}`, `{NINE_MISSING[6]}`,
   `{NINE_MISSING[7]}`, and `{NINE_MISSING[8]}`.
3. **Why did capsule-v1 validation not detect them?** The v1 producer, schema,
   base validator, and hash-chain validator shared the same 97-entry contract;
   the nine dependencies existed only in the downstream consumer.
4. **Was producer-consumer contract drift confirmed?** Yes.
5. **How many canonical D3 requirements exist in the new registry?** 180.
6. **Did builder, validator, preflight, and consumer use identical sets?** Yes;
   each set contained the same 180 identifiers.
7. **Was capsule v2 built append-only?** Yes; capsule v1 and historical records
   were not modified.
8. **Did capsule v2 include D0/D1/D2 evidence references?** Yes.
9. **Did it include the D1 endpoint manifest?** Yes.
10. **Did it include the complete L0 construction contract?** Yes.
11. **Did it include exact member-level tensor schemas?** Yes: names, shapes,
    dtypes, endianness, roles, and hashes.
12. **Did every scientific container header validate?** Yes, without reading
    scientific array or tensor-storage values.
13. **Did the exact L0 experts instantiate?** Yes, as two independent
    `src.output_parameterization.MappedCompactExpertDecoder("square")` models.
14. **Were parameter counts exactly 46,470 per expert?** Yes; 92,940 total.
15. **Did exact initial states load cleanly?** Yes, with no missing or unexpected
    keys and exact shape, dtype, and hash agreement.
16. **Did synthetic production-shape forward pass?** Yes, on MPS with fallback
    disabled.
17. **Did production and reference assignment agree?** Yes.
18. **Did production and reference loss agree?** Yes within the preregistered
    numerical tolerance.
19. **Did the evaluator agree with the reference?** Yes, and it performed zero
    filesystem I/O.
20. **Did one synthetic MPS backward/optimizer step pass?** Yes; both experts
    had finite nonzero gradients and final and nonfinal parameters updated.
21. **Did checkpoint save/reload/replay pass?** Yes, including a fresh MPS
    process and exact state, gradient, assignment, loss, and output hashes.
22. **Did every consumer corruption test fail correctly?** Yes, 25/25 before
    model execution with the expected canonical requirement identifier.
23. **Did declared and accessed requirement sets match?** Yes, 180/180 with no
    unaccessed requirement or undeclared access.
24. **Did the actual consumer emit `ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED`?**
    Yes.
25. **Did it emit `READY_FOR_AUTHORITATIVE_D3_EXECUTION`?** Yes.
26. **Was any scientific array value loaded?** No; count 0.
27. **Was any scientific D3 step executed?** No; count 0.
28. **Is authoritative D3 now executable and authorized?** Yes, only as one new
    separately preregistered square one-scene L0 campaign freezing the exact
    bundle below.
29. **What exact bundle hash must the next campaign freeze?**
    `{EXPECTED_BUNDLE}`.
30. **Were Atlas, development, and lockbox untouched?** Yes; access counts were
    0/0/0.
31. **Were all historical checkpoints unchanged?** Yes, 600/600 matched byte
    sizes and SHA-256 values at closure.
32. **What reusable source/tests should eventually be committed?** The three
    `src/d3_*` modules; the bootstrap, capsule builder/validator, synthetic
    consumer, replay, future launcher, campaign runner, and finalizer scripts;
    `tests/test_d3_executable_contract.py`; and the requested D3 documentation,
    after ordinary review.
33. **What generated artifacts should remain ignored?** The complete
    `outputs/runs/thayer_d3_executable_contract_20260713_164320/` run record,
    including preregistration, registry copies, capsule and bundle copies,
    synthetic fixtures, synthetic checkpoint, logs, tables, diagnostics,
    figures, launcher outputs, and this report.

## Preregistration and drift

- Preregistration SHA-256: `{EXPECTED_PREREGISTRATION}`.
- Frozen before container-member inspection, project model import, model
  construction, and optimizer construction: **PASS**.
- Capsule-v1 base/schema/hash-chain checks: **PASS**.
- Capsule-v1 actual-consumer dependency audit: **FAIL**, exactly nine missing.
- Capsule-v2 actual-consumer dependency audit: **PASS**.

| Contract | Producer | Base validator | Hash chain | Actual consumer |
| --- | --- | --- | --- | --- |
| Capsule v1 | PASS | PASS | PASS | FAIL: nine missing |
| Capsule v2 | PASS | PASS | PASS | PASS: 180/180 |

The exact nine-row evidence is in `tables/capsule_consumer_drift.csv`.

## Registry and capsule-v2 proof

- Canonical registry: `requirement_registry/d3_requirement_registry.json`.
- Registry SHA-256: `{EXPECTED_REGISTRY}`.
- Requirement count: 180.
- Builder = validator = preflight = consumer = runtime closure: **PASS**.
- Capsule-v2 SHA-256: `{EXPECTED_CAPSULE}`.
- Capsule-v2 schema: `schema/d3_executable_capsule_v2.schema.json`.
- Manifest and hash chain: `capsule_v2/d3_executable_capsule_v2_manifest.json`
  and `capsule_v2/d3_executable_capsule_v2_hash_chain.json`.
- No placeholders, implicit defaults, missing requirements, or undeclared
  dependencies were accepted.

## Container-member inventory

| Scientific container | Validated members | Method | Result |
| --- | ---: | --- | --- |
| P0 target set | 11 | NPZ metadata and NPY headers only | PASS |
| D1 endpoint | 4 | NPZ metadata and NPY headers only | PASS |
| Cached encoder features | 6 | restricted PyTorch ZIP metadata only | PASS |
| Initial decoder states | 36 | restricted PyTorch ZIP metadata plus permitted state load | PASS |

Exact rows are in `tables/container_member_inventory.csv` and
`tables/container_member_validation.csv`. Scientific payload bytes read: 0.

## Exact L0 architecture

```mermaid
flowchart LR
    B["bottleneck 64 x 15 x 15"] --> U2["upsample 30 x 30"]
    E2["enc2 32 x 30 x 30"] --> C2["concat 96"]
    U2 --> C2 --> D2["ConvBlock 96 to 32"] --> U1["upsample 60 x 60"]
    E1["enc1 16 x 60 x 60"] --> C1["concat 48"]
    U1 --> C1 --> D1["ConvBlock 48 to 16"] --> H["1 x 1 head 16 to 6"] --> S["square mapping"]
```

Two independent experts instantiated. Each had 46,470 trainable parameters,
18 exact initial-state tensors, and no missing or unexpected state key. The
full architecture, parameter, and state inventories are in
`architecture_audit/l0_architecture_audit.json`,
`tables/l0_parameter_inventory.csv`, and `tables/l0_state_dict_validation.csv`.

## Synthetic full-stack execution

Analytic deterministic fixtures used production feature shapes and contained no
scientific values. MPS forward, exact square mapping, assignment, loss,
zero-I/O evaluator comparison, backward, gradient clipping, and one AdamW step
passed. Optimizer contract: AdamW, learning rate 0.001, weight decay 0, no
scheduler, and clip norm 5.0. Both experts had finite nonzero gradients and
updated final and nonfinal parameters.

The synthetic schema and trace are in `synthetic_inputs/`,
`tables/synthetic_tensor_inventory.csv`,
`synthetic_execution/synthetic_forward_contract.json`, and
`optimizer_audit/synthetic_optimizer_audit.json`.

Checkpoint save, strict reload, and fresh-process replay passed. Evidence is in
`checkpoint_replay/synthetic_checkpoint_manifest.json` and
`checkpoint_replay/fresh_process_replay.json`.

All 25 corruption cases are in `tables/consumer_negative_tests.csv`; each
stopped before model execution. Runtime closure was exactly 180 declared and
180 accessed or validated requirements with no difference. Evidence is in
`synthetic_execution/requirement_closure.json`.

## Executable bundle and final authorization

- Bundle: `future_d3_bundle/d3_executable_bundle_v2.json`.
- Bundle SHA-256: `{EXPECTED_BUNDLE}`.
- Bundle schema and manifest: `future_d3_bundle/d3_executable_bundle_v2.schema.json`
  and `future_d3_bundle/d3_executable_bundle_v2_manifest.json`.
- Actual future-launcher synthetic preflight: **PASS**.
- Final markers: **both emitted**.

One new separately preregistered authoritative D3 campaign may freeze this
bundle and then load only the authorized scientific values. Another readiness
or capsule campaign is not justified unless the bundle validator identifies a
concrete defect. No broader data, eight-scene stage, or capacity ladder is
authorized by this result.

## Final correctness audit

- In-memory compilation: {len(SOURCE_FILES)}/{len(SOURCE_FILES)} sources PASS.
- Focused executable-contract tests: 5/5 PASS.
- Campaign CSV/schema checks: {csv_count}/{csv_count} PASS.
- Historical checkpoint closure: {checkpoint_count}/{checkpoint_count} PASS.
- `git diff --check`: PASS.
- Staged index: empty.
- README: unchanged.
- Allowlist-restricted privacy/path audit: PASS.
- Collision-free final paths: PASS.
- Runtime: Python {runtime['python']}, PyTorch {runtime['torch']},
  MPS available `{str(runtime['mps_available']).lower()}`, platform
  `{runtime['platform']}`.
- Run disk usage before final artifacts: {human_bytes(run_bytes)} across
  {run_files} files.
- Final git status: dirty and unstaged by existing work plus this campaign;
  exact status is persisted in `diagnostics/final_git_status.txt`. Staged files:
  0.

Scientific array values loaded: **0**. Scientific D3 steps: **0**. Atlas,
development, and lockbox accesses: **0/0/0**. D3 scientific status:
**UNKNOWN UNTIL THE NEXT CAMPAIGN**.
"""

    exact_write(run / "tables/checkpoint_inventory_after.csv", after_csv)
    exact_write(
        run / "diagnostics/final_correctness_audit.json",
        json.dumps(correctness, indent=2, sort_keys=True) + "\n",
    )
    exact_write(run / "diagnostics/final_git_status.txt", git_status)
    exact_write(run / "reports/final_report.md", final_report)
    print(
        json.dumps(
            {
                "primary_outcome": "EXECUTABLE D3 CONTRACT PASS",
                "report": str(run_relative / "reports/final_report.md"),
                "bundle_sha256": EXPECTED_BUNDLE,
                "historical_checkpoints_unchanged": checkpoint_count,
                "scientific_array_values_loaded": 0,
                "scientific_d3_steps": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
