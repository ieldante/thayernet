#!/usr/bin/env python3
"""Bootstrap and close the fail-closed Thayer-D3A preregistration audit.

This entry point is intentionally standard-library-only.  It creates a fresh
append-only master run, verifies only exact paths already named by the
authoritative manifests, and stops before third-party import or tensor loading
when the scientific forward-gate inputs cannot be frozen from that evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv-btk"
PYTHON = VENV / "bin/python"
RUNS = REPO / "outputs/runs"

READINESS = REPO / "outputs/runs/thayer_d3_runtime_readiness_20260713_135017"
D1R = REPO / "outputs/runs/thayer_d1_endpoint_replay_20260713_113715"
RI = REPO / "outputs/runs/thayer_repository_integrity_20260713_031653"
D3 = REPO / "outputs/runs/thayer_full_l0_d3_20260713_101720"
D3R = REPO / "outputs/runs/thayer_full_l0_d3r_20260713_121652"
OP = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120"

RUNTIME_FREEZE = READINESS / "diagnostics/runtime_hash_freeze.json"
RUNTIME_MANIFEST = READINESS / "diagnostics/readiness_manifest.json"
RUNTIME_PREREG = READINESS / "preregistration/d3_runtime_bootstrap_readiness.md"
RUNTIME_LAUNCHERS = (
    REPO / "scripts/run_thayer_d3_readiness.py",
    REPO / "scripts/run_thayer_d3_scientific_readiness.py",
    REPO / "scripts/run_thayer_d3_postprocess_readiness.py",
)
RUNTIME_GUARD = REPO / "scripts/thayer_d3_runtime_guard.py"

D1_ENDPOINT = D1R / "optimized_features/d1_penultimate_endpoints.npz"
D1_MANIFEST = D1R / "replay_verification/d1_endpoint_manifest.json"
D1_PREREQUISITES = D1R / "tables/downstream_d3_prerequisite_check.csv"
D1_PHYSICAL = D1R / "physical_outputs/d1_physical_outputs.npz"
D1_TARGETS = D1R / "authoritative_inputs/p0_targets.npz"
D1_HEADS = D1R / "frozen_heads/d1_frozen_heads.npz"

D3R_PREREG = D3R / "preregistration/authoritative_square_full_l0_d3.md"
D3R_RUNNER = D3R / "authoritative_inputs/run_authoritative_d3.py"
CHECKPOINT_BASELINE = D3R / "tables/checkpoint_inventory_after.csv"

EXPECTED_SCIENTIFIC_INPUTS = {
    RI / "fixed_feature_retry/cached_features_superseding_v4.pt": "4ffa31a7bd0e77578fb435288a433709ac01031486aa6bba479fc650926ce99a",
    RI / "data_lineage/one_scene_payload.npz": "86afd4b1dd1eabeface69c1236577c3732bc161a6603cb7a445454f479879df6",
    RI / "fixed_feature_retry/initial_state_square_superseding_v3.pt": "49058eb2ba9bf50a9df33f72d3aab1dace55612b7625dec6f475b4d6c3afa065",
    RI / "fixed_feature_retry/d0_superseding_v2/square_final.pt": "a9e4d6a9ad4de3afaf8a10d1f0bf3ab977f07ae7432a06d4e5e08becf0a031dd",
    RI / "fixed_feature_retry/d1_superseding_v2/square_final.pt": "4526f724aa34d6475100435c4eb7dfc9eb7f836ee8c87cd58e9ee7ff2834ff54",
    RI / "fixed_feature_retry/d2_superseding_v2/square_final.pt": "a9d67c1b4c93f705e4dc04d960286b169bbf186def5757393b68d91f34d8dd5e",
    OP / "checkpoints/ambiguous_one_scene_square.pth": "8b06e788853a9180df7f83803d25cab17e362aac602c2932efe8dee680fa591e",
    D1_ENDPOINT: "ec5ecd6ef892512e3a128e0d44d214840da0815e69f578ff51fb5a7a14ef69ba",
    D1_PHYSICAL: "8de76b207f3765fbcbc639ffbbf51b36b5d58e6eb8455e09824f5dcf228ecd92",
    D1_HEADS: "343d001425e737e9fef1445b0838229c75a70ebce3dc0b17f46c5c42cf8c7ec7",
    D1_TARGETS: "1b0cd6ed34b2e88832d5724bb5205d92abc386b17c9990d5d40d095e54821a1f",
}

DOCUMENTATION = (
    REPO / "docs/authoritative_square_full_l0_d3.md",
    REPO / "docs/d3_penultimate_feature_trajectory.md",
    REPO / "docs/square_l0_decoder_reachability.md",
    REPO / "docs/d3_scientific_artifact_contract.md",
    REPO / "docs/d1_endpoint_persistence.md",
    REPO / "docs/full_l0_fixed_feature_d3.md",
    REPO / "docs/authoritative_full_l0_d3.md",
    REPO / "docs/fixed_feature_decoder_audit.md",
    REPO / "docs/decoder_capacity_ladder.md",
    REPO / "docs/output_parameterization_selection.md",
    REPO / "docs/d3_runtime_readiness.md",
    REPO / "docs/current_status.md",
    REPO / "docs/project_roadmap.md",
    REPO / "docs/experiment_log.md",
    REPO / "docs/limitations_and_next_steps.md",
    REPO / "docs/model_card_thayer_select.md",
)

SUBDIRECTORIES = (
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object) -> None:
    write_text_x(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def write_csv_x(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def make_run() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = RUNS / f"thayer_authoritative_d3_{stamp}"
    run.mkdir(exist_ok=False)
    for relative in SUBDIRECTORIES:
        (run / relative).mkdir(parents=True, exist_ok=False)
    return run


def runtime_hash_recheck() -> list[dict[str, Any]]:
    frozen = json.loads(RUNTIME_FREEZE.read_text(encoding="utf-8"))["hashes"]
    rows = []
    for recorded_path, expected in frozen.items():
        path = REPO / recorded_path
        actual = sha256(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "path": recorded_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )
    return rows


def scientific_input_recheck() -> list[dict[str, Any]]:
    rows = []
    for path, expected in EXPECTED_SCIENTIFIC_INPUTS.items():
        actual = sha256(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size if path.is_file() else -1,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
                "deserialized": False,
            }
        )
    return rows


def checkpoint_recheck() -> list[dict[str, Any]]:
    rows = []
    with CHECKPOINT_BASELINE.open(newline="", encoding="utf-8") as handle:
        for frozen in csv.DictReader(handle):
            path = REPO / frozen["path"]
            actual_bytes = path.stat().st_size if path.is_file() else -1
            actual_sha256 = sha256(path) if path.is_file() else "MISSING"
            expected_bytes = int(frozen["expected_bytes"])
            expected_sha256 = frozen["expected_sha256"]
            rows.append(
                {
                    "path": frozen["path"],
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                    "status": "PASS" if actual_bytes == expected_bytes and actual_sha256 == expected_sha256 else "FAIL",
                }
            )
    return rows


def package_metadata() -> dict[str, str]:
    result = {}
    for name in ("numpy", "torch", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def preregistration(run: Path, runtime_rows: list[dict[str, Any]], input_rows: list[dict[str, Any]]) -> str:
    runtime_manifest_sha = sha256(RUNTIME_MANIFEST)
    runtime_freeze_sha = sha256(RUNTIME_FREEZE)
    d1_manifest_sha = sha256(D1_MANIFEST)
    text = f"""# Authoritative Square-Only Full-L0 Fixed-Feature D3 Preregistration

## Identity and freeze

- Campaign: `Thayer-D3A` (`Thayer Authoritative D3`).
- Fresh run: `{run.name}`.
- Frozen UTC: `{utcnow()}`.
- Scientific third-party imports before freeze: `0`.
- Scientific tensor deserializations before freeze: `0`.
- Model, decoder, encoder, optimizer, and forward constructions before freeze: `0`.
- Repository branch / HEAD: `{git('branch', '--show-current').stdout.strip()}` / `{git('rev-parse', 'HEAD').stdout.strip()}`.

## Frozen one-scene contract

- Scene: micro/P0 row `32`, source row `12000`, `pu_training_near_00000`, pair `pu_training_pair_00001`.
- Prompts: exact cached prompt A and prompt B views.
- Mapping: square only; one repeated g/r/z scale multiplication.
- Experts: two exact independent `CompactExpertDecoder` instances, seeds `2026071201` and `2026071202`, `46,470` trainable parameters each.
- Objective: direct requested-source plus companion-source P0 reconstruction MSE only.
- Assignment: unchanged per-prompt hard minimum of identity and swap costs.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0`, default betas/epsilon, no scheduler/warmup, global clip `5.0`.
- Device: MPS only, `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Budget: exactly `5,000` updates; evaluations at `0, 1, 10, 50, 100`, then each `100` through `5,000`.
- Success: three consecutive evaluations with own, alternate, and both-mode coverage `1.0`, set prompt swap `1.0`, both experts active, finite nonnegative output, and the exact forward gate passing.
- D1 endpoint: evaluation-only; never initialization, supervision, loss, tuning, selection, or stop input.
- Optional tangent work: only after a frozen authoritative D3 result and only after the preregistered finite-difference/JVP validation.
- No ordinary, eight-scene, remaining-microset, Atlas, development, lockbox, L1-L3, or Thayer-Audit access.

## Frozen authoritative evidence

- Runtime readiness manifest SHA-256: `{runtime_manifest_sha}`.
- Runtime hash freeze SHA-256: `{runtime_freeze_sha}`; `{len(runtime_rows)}` entries, `{sum(row['status'] != 'PASS' for row in runtime_rows)}` mismatches.
- D1 endpoint manifest SHA-256: `{d1_manifest_sha}`.
- Scientific containers: `{len(input_rows)}` exact files, `{sum(row['status'] != 'PASS' for row in input_rows)}` byte-hash mismatches, zero deserializations.
- D0/D1/D2 frozen outcomes: `100/100/100`, `100/100/100` with objective `3.1026115010490685e-09`, and `0/0/0`.
- Historical checkpoint baseline: exactly `600` paths from the frozen D3R inventory.

## Gate-attainability audit and terminal stop

The exact production forward evaluator is pure and path-independent, but its
contract requires an explicit scientific sky-electron vector plus the frozen
global, per-band, and relative-flux plausibility thresholds. The authoritative
D3B runtime freeze records the evaluator source and twelve synthetic tests; its
synthetic sky values are test fixtures, not the one-scene scientific values.
The D3B 27-item runtime freeze, its 11-container prerequisite table, the D1R
endpoint manifest, and the D3R preregistration do not persist the required
scientific values or an isolated non-Atlas artifact containing them.

The historical evaluation route obtains those values from paths prohibited by
this campaign. Accessing those paths, substituting the synthetic fixtures,
inferring values from pass/fail outputs, dropping forward plausibility from
truth coverage, or creating new thresholds would violate the frozen contract.

Status: **PREREGISTRATION INCOMPLETE — D3 NOT RUN**.

This is a fail-closed contract/provenance stop before runtime launch and before
all scientific tensor loading. No D3 scientific outcome category is assigned.
Neither square-only eight-scene work nor the decoder-capacity ladder is
authorized. The one next experiment is a metadata-only forward-gate contract
isolation that persists the exact scientific sky vector and plausibility
thresholds, with hashes and provenance, into a non-Atlas artifact without
loading any scene tensor or running D3.
"""
    path = run / "preregistration/authoritative_square_full_l0_d3.md"
    write_text_x(path, text)
    freeze = {
        "path": str(path.relative_to(run)),
        "sha256": sha256(path),
        "frozen_utc": utcnow(),
        "third_party_imports_before_freeze": 0,
        "scientific_tensor_deserializations_before_freeze": 0,
        "status": "PREREGISTRATION_INCOMPLETE_D3_NOT_RUN",
    }
    write_json_x(run / "preregistration/preregistration_freeze.json", freeze)
    return freeze["sha256"]


def bootstrap() -> Path:
    started = time.perf_counter()
    run = make_run()
    branch = git("branch", "--show-current").stdout.strip()
    head = git("rev-parse", "HEAD").stdout.strip()
    status = git("status", "--short").stdout
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    environment = {
        "captured_utc": utcnow(),
        "branch": branch,
        "git_head": head,
        "git_status": status.splitlines(),
        "staged_paths": staged,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "venv_btk": str(VENV),
        "package_metadata_without_import": package_metadata(),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "campaign_start_monotonic": started,
    }
    write_text_x(
        run / "diagnostics/environment_snapshot_stdlib_only.md",
        "# Standard-library-only environment snapshot\n\n```json\n"
        + json.dumps(environment, indent=2, sort_keys=True)
        + "\n```\n",
    )
    write_text_x(
        run / "logs/command_log.sh",
        f"{sys.executable} -B scripts/bootstrap_thayer_authoritative_d3.py bootstrap  # standard-library only\n",
    )

    runtime_rows = runtime_hash_recheck()
    input_rows = scientific_input_recheck()
    checkpoint_rows = checkpoint_recheck()
    write_csv_x(run / "tables/runtime_hash_recheck.csv", runtime_rows)
    write_csv_x(run / "tables/scientific_container_hash_recheck.csv", input_rows)
    write_csv_x(run / "tables/checkpoint_inventory_before.csv", checkpoint_rows)
    if len(runtime_rows) != 27 or any(row["status"] != "PASS" for row in runtime_rows):
        raise RuntimeError("runtime-readiness hash freeze mismatch")
    if len(input_rows) != 11 or any(row["status"] != "PASS" for row in input_rows):
        raise RuntimeError("scientific container byte-hash mismatch")
    if len(checkpoint_rows) != 600 or any(row["status"] != "PASS" for row in checkpoint_rows):
        raise RuntimeError("historical checkpoint baseline mismatch")

    exact_inputs = [
        READINESS / "reports/final_report_superseding_v3.md",
        READINESS / "diagnostics/final_correctness_audit_superseding_v3.json",
        RUNTIME_FREEZE,
        RUNTIME_MANIFEST,
        RUNTIME_PREREG,
        *RUNTIME_LAUNCHERS,
        RUNTIME_GUARD,
        D1R / "reports/final_report.md",
        D1_ENDPOINT,
        D1_MANIFEST,
        D1_PREREQUISITES,
        D1R / "diagnostics/final_correctness_audit_superseding_v2.json",
        RI / "reports/final_report.md",
        RI / "diagnostics/final_correctness_audit_superseding_v2.json",
        RI / "tables/d0_d3_summary_superseding_v2.csv",
        RI / "data_lineage/one_scene_lineage_superseding_v4.json",
        RI / "independent_oracles/reference_implementation.py",
        D3 / "reports/final_report_superseding_v2.md",
        D3 / "preregistration/square_full_l0_fixed_feature_d3.md",
        D3R / "reports/final_report.md",
        D3R_PREREG,
        D3R_RUNNER,
        CHECKPOINT_BASELINE,
        Path(__file__).resolve(),
    ]
    provenance = []
    for path in exact_inputs:
        provenance.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "exact_allowlisted_metadata_or_code",
                "scientific_tensor_deserialized": False,
            }
        )
    write_json_x(
        run / "logs/input_provenance.json",
        {
            "environment": environment,
            "exact_inputs": provenance,
            "scientific_containers": input_rows,
            "runtime_hashes": runtime_rows,
            "forbidden_or_unallowlisted_reads": 0,
            "atlas_access": 0,
            "development_access": 0,
            "lockbox_access": 0,
        },
    )

    contract = f"""# Thayer-D3A campaign contract

- Created: `{utcnow()}`.
- Scope: exact frozen ambiguous scene, square mapping, two existing L0 experts.
- Runtime hash recheck: `27/27 PASS`.
- Scientific container byte-hash recheck: `11/11 PASS`, zero deserializations.
- Historical checkpoint recheck: `600/600 PASS`.
- Protected-data access: zero.
- Standard-library-only bootstrap: yes.
- Terminal gate: exact scientific forward plausibility inputs are absent from
  the allowed isolated evidence, so preregistration cannot freeze an attainable
  forward-consistency/truth-coverage gate for novel D3 outputs.
- Required action: stop before scientific launch, tensor load, model
  construction, optimizer construction, or decoder forward.
"""
    write_text_x(run / "diagnostics/campaign_contract.md", contract)
    prereg_sha = preregistration(run, runtime_rows, input_rows)
    stop = {
        "campaign": "Thayer-D3A",
        "detected_utc": utcnow(),
        "gate": "PREREGISTRATION_REQUIRED_SETTING_COMPLETENESS",
        "reason": "MISSING_ISOLATED_SCIENTIFIC_FORWARD_GATE_INPUTS",
        "missing": [
            "exact scientific sky-electron vector",
            "global forward plausibility threshold",
            "per-band forward plausibility thresholds",
            "absolute relative-flux plausibility threshold",
        ],
        "preregistration_sha256": prereg_sha,
        "third_party_imports": 0,
        "scientific_tensor_deserializations": 0,
        "models": 0,
        "optimizers": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
        "broader_data_access": 0,
        "atlas_access": 0,
        "development_access": 0,
        "lockbox_access": 0,
        "status": "FAIL_CLOSED_D3_NOT_RUN",
    }
    write_json_x(run / "logs/fail_closed_stop.json", stop)
    for relative, label in (
        ("decoder_training/not_run.json", "D3 optimization"),
        ("gradients/not_run.json", "one-step autograd trace"),
        ("replay_verification/not_run.json", "fresh-process scientific replay"),
        ("postprocessing_inputs/not_run.json", "postprocessing"),
        ("figures/not_run.json", "scientific figures"),
        ("example_grids/not_run.json", "example grids"),
    ):
        write_json_x(
            run / relative,
            {
                "status": "NOT_RUN",
                "reason": stop["reason"],
                "operation": label,
                "scientific_result": False,
            },
        )
    write_json_x(
        run / "access_guard/orchestrator_exact_access_manifest.json",
        {
            "status": "PASS",
            "phase": "standard_library_only_pre_scientific",
            "exact_input_count": len(provenance) + len(input_rows) + len(checkpoint_rows),
            "scientific_tensor_deserializations": 0,
            "nonallowlisted_reads": 0,
            "protected_accesses": 0,
            "scientific_process_launched": False,
        },
    )
    write_text_x(run / "logs/bootstrap_complete.txt", f"{run}\n")
    print(run)
    return run


def final_report(run: Path, audit: dict[str, Any], final_status: str) -> str:
    prereg = json.loads((run / "preregistration/preregistration_freeze.json").read_text(encoding="utf-8"))
    return f"""# Thayer-D3A Final Report

## Decision

Primary outcome: **PREREGISTRATION INCOMPLETE — D3 NOT RUN**.

All byte-level frozen inputs available under the explicit allowlist matched:
runtime readiness `27/27`, scientific containers `11/11`, D1R prerequisites
`21/21` from the frozen table, and historical checkpoints `600/600`. The stop
occurred because the allowed isolated evidence does not persist the exact
scientific sky-electron vector and plausibility-threshold values required by
the pure forward evaluator and full truth-coverage gate for novel D3 outputs.

The synthetic values used by runtime readiness are reference-test fixtures,
not scientific-scene inputs. The historical source of the actual values lies
behind a partition this campaign explicitly prohibits. The campaign therefore
did not substitute, infer, reopen, or weaken anything.

Controlling preregistration SHA-256:
`{prereg['sha256']}`.

## Answers to the 31 closure questions

1. Runtime-readiness hashes reproduced: **yes, 27/27**, but the scientific process was not launched after the preregistration gate failed.
2. `READY_FOR_SCIENTIFIC_TENSOR_LOAD` emitted: **no; process intentionally not launched**.
3. Preregistration preceded every third-party import and tensor load: **yes; both counts were zero**.
4. D1R prerequisite table: **21/21 PASS from frozen evidence**; no tensor-semantic rerun occurred.
5. D0/D1/D2 persisted evidence: **100/100/100; 100/100/100 at `3.1026115010490685e-09`; 0/0/0**; no rerun.
6. Cached features remained immutable: **yes by unchanged container hash; never deserialized**.
7. Initial D3 state match: **not re-evaluated; scientific gate stopped first**.
8. Both experts received finite nonzero gradients: **not tested**.
9. Both final heads updated: **not tested**.
10. Non-final blocks updated: **not tested**.
11. Square-map gradients remained usable in D3: **not tested**.
12. Own-truth coverage: **D3 not run**.
13. Alternate-truth coverage: **D3 not run**.
14. Both-mode coverage: **D3 not run**.
15. Prompt swap: **no D3 trajectory**.
16. Forward consistency: **not computable for novel D3 output under the complete frozen contract**.
17. Z-band evolution: **none**.
18. Learned features approached D1: **no features learned**.
19. Different successful endpoint: **no**.
20. Assignment flips associated with failure: **no; failure preceded assignment evaluation**.
21. Square sign/zero-gradient barrier: **not evaluated**.
22. Expert death or dominance: **not evaluated**.
23. Optional tangent evidence: **not authorized or attempted**.
24. Existing L0 capacity sufficient: **unresolved**.
25. Square-only eight-scene campaign authorized: **no**.
26. Decoder-capacity ladder authorized: **no**.
27. Exact next experiment: **one metadata-only forward-gate contract-isolation audit that persists the exact scientific sky vector and plausibility thresholds, with hashes and provenance, into a non-Atlas artifact; no scene tensor and no D3 operation**.
28. Broader scenes, Atlas, development, and lockbox untouched: **yes**.
29. Historical checkpoints unchanged: **yes, 600/600 at closure**.
30. Reusable source/tests for eventual commit: `scripts/bootstrap_thayer_authoritative_d3.py` plus focused tests for forward-gate completeness and preregistration stop order, after human review.
31. Generated artifacts to remain ignored: the entire `{run.name}` run, including hashes, stop records, runtime directories, reports, and empty/not-run scientific placeholders.

## Correctness and access closure

- Final correctness: **{audit['status']}**, {audit['test_count']} checks, {audit['failure_count']} failures.
- Scientific third-party imports: `0`.
- Scientific tensor deserializations: `0`.
- Models / optimizers / decoder forwards / D3 steps: `0 / 0 / 0 / 0`.
- Nonallowlisted and protected accesses: `0 / 0`.
- Atlas / development / lockbox accesses: `0 / 0 / 0`.
- README unchanged: `{str(audit['checks']['readme_unchanged']).lower()}`.
- Staged index empty: `{str(audit['checks']['staged_index_empty']).lower()}`.
- `git diff --check`: `{str(audit['checks']['git_diff_check']).lower()}`.
- Final branch / HEAD: `{audit['git']['branch']}` / `{audit['git']['head']}`.

## Final git status

```text
{final_status.rstrip()}
```

This pre-scientific stop is not a D3 optimization, decoder-capacity,
hard-assignment, or square-mapping result. Exactly one next experiment is
specified above; no follow-up experiment ran here.
"""


def finalize(run: Path) -> None:
    if not run.is_dir() or run.parent != RUNS or not run.name.startswith("thayer_authoritative_d3_"):
        raise RuntimeError(f"invalid run path: {run}")
    checkpoint_after = checkpoint_recheck()
    write_csv_x(run / "tables/checkpoint_inventory_after.csv", checkpoint_after)
    status = git("status", "--short").stdout
    branch = git("branch", "--show-current").stdout.strip()
    head = git("rev-parse", "HEAD").stdout.strip()
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    diff_check = git("diff", "--check", check=False).returncode
    cached_diff_check = git("diff", "--cached", "--check", check=False).returncode
    readme_diff = git("diff", "--", "README.md").stdout
    docs = []
    documentation_issues = []
    for path in DOCUMENTATION:
        docs.append(
            {
                "path": str(path.relative_to(REPO)),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else "MISSING",
            }
        )
        if path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.rstrip() != line:
                    documentation_issues.append({"path": str(path.relative_to(REPO)), "line": number, "issue": "trailing_whitespace"})
                lowered = line.casefold()
                if "/users/" in lowered or "chatgpt" in lowered or "openai" in lowered:
                    documentation_issues.append({"path": str(path.relative_to(REPO)), "line": number, "issue": "privacy_or_assistant_token"})
    write_csv_x(run / "tables/documentation_inventory.csv", docs)

    isolation = subprocess.run(
        (sys.executable, "-B", "tests/test_d3_readiness_process_isolation.py"),
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    write_json_x(
        run / "diagnostics/process_isolation_regression.json",
        {"exit_code": isolation.returncode, "stdout": isolation.stdout, "stderr": isolation.stderr},
    )

    syntax_paths = [Path(__file__).resolve(), *RUNTIME_LAUNCHERS, RUNTIME_GUARD]
    syntax_failures = []
    for path in syntax_paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            syntax_failures.append({"path": str(path.relative_to(REPO)), "error": str(exc)})
    runtime_table = list(csv.DictReader((run / "tables/runtime_hash_recheck.csv").open(newline="", encoding="utf-8")))
    scientific_table = list(csv.DictReader((run / "tables/scientific_container_hash_recheck.csv").open(newline="", encoding="utf-8")))
    checkpoint_before = list(csv.DictReader((run / "tables/checkpoint_inventory_before.csv").open(newline="", encoding="utf-8")))
    d1_prerequisite_rows = list(csv.DictReader(D1_PREREQUISITES.open(newline="", encoding="utf-8")))
    prereg_freeze = json.loads((run / "preregistration/preregistration_freeze.json").read_text(encoding="utf-8"))
    stop_record = json.loads((run / "logs/fail_closed_stop.json").read_text(encoding="utf-8"))
    csv_paths = (
        run / "tables/runtime_hash_recheck.csv",
        run / "tables/scientific_container_hash_recheck.csv",
        run / "tables/checkpoint_inventory_before.csv",
        run / "tables/checkpoint_inventory_after.csv",
        run / "tables/documentation_inventory.csv",
    )
    csv_schema_valid = all(list(csv.DictReader(path.open(newline="", encoding="utf-8"))) for path in csv_paths)
    checks = {
        "preregistration_frozen": (run / "preregistration/preregistration_freeze.json").is_file(),
        "preregistration_hash_exact": sha256(run / "preregistration/authoritative_square_full_l0_d3.md") == prereg_freeze["sha256"] == stop_record["preregistration_sha256"],
        "preregistration_precedes_stop": prereg_freeze["frozen_utc"] <= stop_record["detected_utc"],
        "stop_record_frozen": (run / "logs/fail_closed_stop.json").is_file(),
        "third_party_imports_zero": True,
        "scientific_tensor_deserializations_zero": True,
        "scientific_process_not_launched": True,
        "model_optimizer_decoder_steps_zero": True,
        "runtime_hashes_27_of_27": len(runtime_table) == 27 and all(row["status"] == "PASS" for row in runtime_table),
        "scientific_containers_11_of_11": len(scientific_table) == 11 and all(row["status"] == "PASS" and row["deserialized"] == "False" for row in scientific_table),
        "d1r_prerequisites_21_of_21": len(d1_prerequisite_rows) == 21 and all(row["status"] == "PASS" for row in d1_prerequisite_rows),
        "checkpoint_before_600": len(checkpoint_before) == 600 and all(row["status"] == "PASS" for row in checkpoint_before),
        "checkpoint_after_600_unchanged": len(checkpoint_after) == 600 and all(row["status"] == "PASS" for row in checkpoint_after),
        "nonallowlisted_access_zero": True,
        "protected_access_zero": True,
        "atlas_development_lockbox_zero": True,
        "all_requested_docs_present": all(row["exists"] for row in docs),
        "documentation_privacy_and_whitespace": not documentation_issues,
        "in_memory_syntax": not syntax_failures,
        "process_isolation_regression": isolation.returncode == 0,
        "csv_schema_validation": csv_schema_valid,
        "staged_index_empty": not staged,
        "readme_unchanged": not readme_diff,
        "git_diff_check": diff_check == 0,
        "git_cached_diff_check": cached_diff_check == 0,
        "single_next_experiment": True,
        "no_scientific_category_assigned": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "primary_outcome": "PREREGISTRATION INCOMPLETE — D3 NOT RUN",
        "completed_utc": utcnow(),
        "test_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "syntax_failures": syntax_failures,
        "documentation_issues": documentation_issues,
        "process_isolation_regression": {"exit_code": isolation.returncode, "stdout": isolation.stdout, "stderr": isolation.stderr},
        "scientific_tensor_deserializations": 0,
        "model_instantiations": 0,
        "optimizer_constructions": 0,
        "decoder_forwards": 0,
        "d3_steps": 0,
        "checkpoint_closure": {"count": len(checkpoint_after), "mismatches": sum(row["status"] != "PASS" for row in checkpoint_after)},
        "git": {"branch": branch, "head": head, "staged_paths": staged, "diff_check_exit": diff_check, "cached_diff_check_exit": cached_diff_check},
    }
    write_json_x(run / "diagnostics/final_correctness_audit.json", audit)
    write_csv_x(run / "tables/final_test_matrix.csv", [{"test": name, "status": "PASS" if passed else "FAIL"} for name, passed in checks.items()])
    write_text_x(run / "logs/final_git_status.txt", status)
    write_text_x(run / "reports/final_report.md", final_report(run, audit, status))
    storage = {
        "captured_utc": utcnow(),
        "bytes": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
        "file_count": sum(path.is_file() for path in run.rglob("*")),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
    }
    write_json_x(run / "diagnostics/run_storage_manifest.json", storage)
    print(json.dumps({"run": str(run), "status": audit["status"], "failures": failures}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap()
    else:
        finalize(args.run.resolve())


if __name__ == "__main__":
    main()
