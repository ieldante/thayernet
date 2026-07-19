#!/usr/bin/env python3
"""Record the fail-closed Thayer-FF prestart incident without loading arrays."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
OP_RUN = REPO / "outputs/runs/thayer_output_parameterization_20260713_023120"
FP_RUN = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
ME_RUN = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
CL_RUN = REPO / "outputs/runs/thayer_capacity_ladder_20260713_013132"
SUBDIRECTORIES = (
    "diagnostics",
    "tables",
    "figures",
    "logs",
    "reports",
    "preregistration",
    "cached_features",
    "initial_states",
    "direct_output",
    "free_features",
    "head_only",
    "decoder_optimization",
    "tangent_space",
    "trajectories",
    "example_grids",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def write_text_fresh(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_inventory() -> list[dict[str, object]]:
    rows = []
    for path in sorted((REPO / "outputs/runs").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return rows


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("astropy", "btk", "h5py", "matplotlib", "numpy", "scipy", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    run = args.run_dir or (
        REPO
        / "outputs/runs"
        / f"thayer_fixed_feature_audit_{started.astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    run = run.resolve()
    if run.parent != (REPO / "outputs/runs").resolve():
        raise ValueError("run directory must be a direct outputs/runs child")
    run.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run / name).mkdir(exist_ok=False)

    op_provenance = json.loads((OP_RUN / "logs/input_provenance.json").read_text())
    frozen_rows = list(
        csv.DictReader(
            (OP_RUN / "tables/frozen_row_selection.csv").open(newline="", encoding="utf-8")
        )
    )
    ambiguous = next(row for row in frozen_rows if row["one_scene_ambiguous"] == "True")
    checkpoints = checkpoint_inventory()
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", checkpoints)

    status = git("status", "--short").splitlines()
    staged = git("diff", "--cached", "--name-only").splitlines()
    frozen_hashes = op_provenance["frozen_contract_hashes"]
    relevant_paths = {
        "thayer_me_final_report": ME_RUN / "reports/final_report.md",
        "thayer_fp_final_report": FP_RUN / "reports/final_report.md",
        "thayer_cl_final_report": CL_RUN / "reports/final_report.md",
        "thayer_op_final_report": OP_RUN / "reports/final_report.md",
        "thayer_op_preregistration": OP_RUN / "preregistration/fixed_l0_output_parameterization.md",
        "thayer_op_condition_summary": OP_RUN / "tables/condition_summary.csv",
    }
    relevant_hashes = {
        name: {
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in relevant_paths.items()
    }
    free_disk = shutil.disk_usage(REPO).free
    incident = {
        "campaign": "Thayer-FF",
        "detected_at_utc": started.isoformat(),
        "status": "FAIL_CLOSED_BEFORE_PREREGISTRATION",
        "reason": "UNINTENDED_DEVELOPMENT_TABULAR_VALUE_ACCESS_DURING_METADATA_SEARCH",
        "accessed_relative_path": "outputs/runs/thayer_select_root_cause_analysis_20260711/tables/derived_scene_features.csv",
        "development_tabular_search_result_access_count": 1,
        "development_scene_array_access_count": 0,
        "atlas_array_access_count": 0,
        "lockbox_array_access_count": 0,
        "ambiguous_scene_tensor_load_count": 0,
        "p0_target_tensor_load_count": 0,
        "encoder_feature_extraction_count": 0,
        "neural_model_construction_count": 0,
        "optimizer_step_count": 0,
        "jvp_vjp_count": 0,
        "preregistration_written": False,
        "campaign_continued_after_incident": False,
        "capacity_ladder_authorized": False,
    }
    write_json_fresh(run / "logs/fail_closed_stop.json", incident)

    provenance = {
        "campaign": "Thayer-FF Fixed-Feature L0 Expert-Decoder Optimization Audit",
        "campaign_started_utc": started.isoformat(),
        "campaign_status": incident["status"],
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": status,
        "staged_index": staged,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "free_disk_bytes": free_disk,
        "frozen_ambiguous_row": ambiguous,
        "prompt_a_row_id": f"{ambiguous['scene_id']}:prompt_A",
        "prompt_b_row_id": f"{ambiguous['scene_id']}:prompt_B",
        "scene_container_sha256": op_provenance["frozen_inputs"][
            "outputs/runs/thayer_multiple_hypotheses_20260712_190701/manifests/probabilistic_unet_training_scenes.h5"
        ]["sha256"],
        "per_scene_blend_sha256": "NOT READ OR DERIVED AFTER FAIL_CLOSED_STOP",
        "p0_target_file_sha256": op_provenance["p0_projected_target_file_sha256"],
        "p0_target_hash_table_sha256": op_provenance["p0_projected_target_hash_table_sha256"],
        "condition_c_checkpoint_sha256": op_provenance["condition_c_checkpoint_sha256"],
        "expert_initialization_seeds": op_provenance["expert_initialization_seeds"],
        "frozen_contract_hashes": frozen_hashes,
        "relevant_source_hashes": relevant_hashes,
        "historical_checkpoint_count": len(checkpoints),
        "incident": incident,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)

    environment = f"""# Environment snapshot

- Campaign: Thayer-FF
- Started UTC: `{started.isoformat()}`
- Status: **FAIL CLOSED BEFORE PREREGISTRATION**
- Branch: `{provenance['branch']}`
- Git HEAD: `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Torch: `{torch.__version__}`
- MPS built/available: `{provenance['mps_built']}/{provenance['mps_available']}`
- Historical checkpoint inventory: `{len(checkpoints)}` files
- Free disk: `{free_disk}` bytes
- Staged index entries: `{len(staged)}`

No scene tensor, P0 target tensor, cached encoder feature, model, optimizer, or
Jacobian operation was opened or constructed by this run.
"""
    write_text_fresh(run / "diagnostics/environment_snapshot.md", environment)

    contract = """# Thayer-FF campaign contract

This run was created only to preserve a pre-preregistration fail-closed
incident. The intended one-scene fixed-feature audit retained the exact frozen
ambiguous row, both coordinate prompts, P0 target set, hard assignment, three
output mappings, Condition-C encoder, and two 46,470-parameter L0 experts.

During the read-only metadata inventory, a broad text search returned values
from an existing development feature table. Part N requires an entire-campaign
stop on development access. The stop therefore occurred before preregistration,
per-scene loading, feature extraction, model construction, optimization, or
tangent-space analysis. No interpretation category is assigned and no
capacity ladder is authorized.
"""
    write_text_fresh(run / "diagnostics/campaign_contract.md", contract)

    checks = [
        {"check": "stop_record_written", "status": "PASS", "evidence": "logs/fail_closed_stop.json"},
        {"check": "stopped_before_preregistration", "status": "PASS", "evidence": "preregistration_written=false"},
        {"check": "scene_tensor_load_count_zero", "status": "PASS", "evidence": "0"},
        {"check": "p0_tensor_load_count_zero", "status": "PASS", "evidence": "0"},
        {"check": "model_construction_count_zero", "status": "PASS", "evidence": "0"},
        {"check": "optimizer_step_count_zero", "status": "PASS", "evidence": "0"},
        {"check": "development_access_boundary", "status": "FAIL", "evidence": "one tabular search result exposed"},
        {"check": "capacity_ladder_authorization", "status": "PASS", "evidence": "false"},
    ]
    write_csv_fresh(run / "tables/final_correctness_checks.csv", checks)
    correctness = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL_CLOSED",
        "check_count": len(checks),
        "failure_count": 1,
        "failures": [checks[6]],
        "historical_checkpoint_count": len(checkpoints),
        "development_tabular_search_result_access_count": 1,
        "development_scene_array_access_count": 0,
        "atlas_array_access_count": 0,
        "lockbox_array_access_count": 0,
    }
    write_json_fresh(run / "diagnostics/final_correctness_audit.json", correctness)

    report = f"""# Thayer-FF fixed-feature decoder audit final report

Primary outcome: **FAIL-CLOSED PRESTART STOP — DEVELOPMENT TABULAR VALUES WERE
UNINTENTIONALLY EXPOSED DURING METADATA SEARCH**.

The stop occurred before preregistration, the frozen ambiguous-scene load, P0
target loading, encoder feature extraction, model construction, optimization,
or tangent-space analysis. This is a protocol result, not a scientific decoder
result. No interpretation category from the planned D0-D3 audit is assigned.

## Direct answers

1. Frozen input hashes were recorded from the authoritative provenance, but a
   full campaign match gate was not completed after the stop.
2. No. Preregistration was deliberately not written after the protected-data
   boundary incident.
3. Thayer-OP was not rerun.
4. Cached features were not created.
5. Initial-state alignment was not run.
6. D0 own-truth reachability was not run.
7. D0 alternate-truth reachability was not run.
8. D0 both-mode reachability was not run.
9. No mapping was audited for direct navigability.
10. D1 was not run.
11. D2 was not run.
12. D3 was not run.
13. Reachability was not evaluated.
14. Tangent residual capture was not estimated.
15. JVP/VJP checks were not run.
16. No condition-number claim is made.
17. No scientific blocker category is assigned.
18. The decoder-capacity ladder is not authorized.
19. Next experiment: restart the exact frozen Thayer-FF preregistration in a
    fresh task whose inventory commands are path-allowlisted before execution.
20. Atlas and lockbox arrays were untouched; development scene arrays were
    untouched, but one existing development feature-table search result was
    exposed, so the broader development-access claim fails.
21. This run hashed `{len(checkpoints)}` historical checkpoint files before
    stopping and did not write or mutate any checkpoint. A closure rehash was
    not needed because no model or checkpoint write occurred.

## Provenance and closure

- Stop record: `logs/fail_closed_stop.json`
- Input provenance: `logs/input_provenance.json`
- Checkpoint inventory: `tables/checkpoint_inventory_before.csv`
- Correctness audit: `diagnostics/final_correctness_audit.json`
- Preregistration hash: not applicable; no preregistration was written.
- Encoder/feature provenance: not created.
- Objective evaluations: `0`.
- MPS optimizer steps: `0`.
- Historical checkpoint writes: `0`.
- Capacity authorization: **false**.
- README and historical runs were not modified.
- Nothing was staged, committed, pushed, merged, deleted, or overwritten.
"""
    write_text_fresh(run / "reports/final_report.md", report)

    write_json_fresh(
        run / "logs/closure.json",
        {
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL_CLOSED",
            "run_relative_path": str(run.relative_to(REPO)),
            "run_bytes": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
            "historical_checkpoint_write_count": 0,
            "optimizer_step_count": 0,
            "staged_index_entry_count": len(staged),
        },
    )
    print(run)


if __name__ == "__main__":
    main()
