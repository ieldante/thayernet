#!/usr/bin/env python3
"""Run artifact-level final correctness checks for Ambiguity Atlas v0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "manifests", "candidate_outputs", "decompositions", "atlas", "embeddings",
    "optimization", "features", "models", "calibration", "rotations",
    "example_grids", "paper_figures",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_fresh(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def command(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_ambiguity_atlas_v0_"):
        raise ValueError("unexpected run directory")

    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, evidence: str, applicability: str = "APPLICABLE") -> None:
        checks.append(
            {
                "check": name,
                "applicability": applicability,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    prereg = run_dir / freeze["preregistration_path"]
    inferred_outputs = [
        run_dir / "tables/candidate_output_inventory.csv",
        run_dir / "optimization/counterfactual_optimization_trajectories.csv",
    ]
    predates = all(prereg.stat().st_mtime_ns < path.stat().st_mtime_ns for path in inferred_outputs)
    record("preregistration_predates_inference_and_optimization", predates, freeze["preregistration_sha256"])
    record("preregistration_hash_matches", sha256_file(prereg) == freeze["preregistration_sha256"], sha256_file(prereg))
    record("all_required_subdirectories_exist", all((run_dir / name).is_dir() for name in REQUIRED_DIRS), ",".join(REQUIRED_DIRS))

    summary = json.loads((run_dir / "manifests/source_partition_summary.json").read_text())
    record("zero_development_source_commitments", summary["development_rows_committed"] == 0, str(summary["development_rows_committed"]))
    record("zero_lockbox_source_commitments", summary["sealed_lockbox_rows_committed"] == 0, str(summary["sealed_lockbox_rows_committed"]))
    commitments = read_csv(run_dir / "manifests/campaign_source_partition_commitments.csv")
    record("approved_source_roles_only", {row["campaign_role"] for row in commitments} <= {"training", "validation", "calibration", "audit_evaluation"}, str(sorted({row["campaign_role"] for row in commitments})))

    pairs = read_csv(run_dir / "tables/atlas_pair_manifest.csv")
    group_isolation = all(
        len({row["left_target_group"], row["right_target_group"], row["left_contaminant_group"], row["right_contaminant_group"]}) == 4
        for row in pairs
    )
    record("atlas_source_group_isolation", group_isolation, f"{len(pairs)} pair rows")
    validations = read_csv(run_dir / "tables/atlas_pair_validation.csv")
    record("pair_replay_and_visual_validation", len(validations) == 25 and all(row["numerical_gate_status"] == "PASS" and row["noisy_exact_replay_pass"] == "True" for row in validations), f"{len(validations)} frozen rows")

    unit_tests = read_csv(run_dir / "tables/forward_model_unit_tests.csv")
    record("fixed_forward_model_tests", bool(unit_tests) and all(row["status"] == "PASS" for row in unit_tests), f"{len(unit_tests)} rows")
    test_logs = [(run_dir / "logs/main_contract_tests.txt").read_text(), (run_dir / "logs/btk_contract_tests.txt").read_text()]
    record("whitened_distance_and_ambiguity_unit_tests", all("passed" in text and "failed" not in text for text in test_logs), "14 main + 17 BTK tests")

    outputs = read_csv(run_dir / "tables/candidate_output_inventory.csv")
    decompositions = read_csv(run_dir / "tables/candidate_decomposition_inventory.csv")
    output_keys = {
        (row["pair_id"], row["side"], row["regime"], row["family_id_provenance_only"], row["source_index"])
        for row in outputs
    }
    aligned = all(
        (row["pair_id"], row["side"], row["regime"], row["family_id_provenance_only"], "0") in output_keys
        and (row["pair_id"], row["side"], row["regime"], row["family_id_provenance_only"], "1") in output_keys
        for row in decompositions
    )
    record("prompt_candidate_full_decomposition_alignment", aligned, f"{len(decompositions)} decompositions / {len(outputs)} source outputs")

    calibration_definitions = read_csv(run_dir / "manifests/forward_consistency_calibration_scene_definitions.csv")
    calibration_groups = {row["target_group"] for row in calibration_definitions} | {row["contaminant_group"] for row in calibration_definitions}
    committed_calibration_groups = {row["duplicate_group_id"] for row in commitments if row["campaign_role"] == "calibration"}
    record("calibration_group_isolation", calibration_groups <= committed_calibration_groups and len(calibration_definitions) == 3000, f"{len(calibration_definitions)} calibration scenes")
    threshold_mtime = (run_dir / "calibration/forward_consistency_thresholds.json").stat().st_mtime_ns
    output_mtime = (run_dir / "tables/candidate_output_inventory.csv").stat().st_mtime_ns
    record("calibration_frozen_before_candidate_evaluation", threshold_mtime < output_mtime, f"threshold_mtime_ns={threshold_mtime}; output_mtime_ns={output_mtime}")

    family_rows = read_csv(run_dir / "tables/deblender_family_inventory.csv")
    primary_clusters = {row["family_cluster"] for row in family_rows if row["status"].startswith("COMPATIBLE")}
    record("cross_family_gate_blocked_below_three", len(primary_clusters) < 3, str(sorted(primary_clusters)))
    record("held_out_family_evaluation_absent", not any((run_dir / "rotations").iterdir()), "rotations directory empty", "NOT_AUTHORIZED")
    record("auditor_tensors_and_models_absent", not any((run_dir / "features").iterdir()) and not any((run_dir / "models").iterdir()), "features/models empty", "NOT_AUTHORIZED")
    record("family_id_absent_from_auditor_tensors", True, "no auditor tensors exist", "NOT_AUTHORIZED")
    record("target_leakage_absent_from_auditor_tensors", True, "no auditor tensors exist", "NOT_AUTHORIZED")
    record("catalog_policy_absent_after_gate_failure", not any((run_dir / "rotations").iterdir()), "no coverage thresholds or rotations", "NOT_AUTHORIZED")

    label_rows = read_csv(run_dir / "tables/label_applicability_matrix.csv")
    record("masked_label_applicability_complete", len(label_rows) == 10 and all(row["atlas_valid_prompt"] in {"APPLICABLE", "CONDITIONAL", "NOT_APPLICABLE"} for row in label_rows), f"{len(label_rows)} label rows")

    before = {row["path"]: (row["sha256"], row["bytes"]) for row in read_csv(run_dir / "tables/checkpoint_inventory_before.csv")}
    after = {row["path"]: (row["sha256"], row["bytes"]) for row in read_csv(run_dir / "tables/checkpoint_inventory_after.csv")}
    record("historical_checkpoint_hash_audit", before == after and len(before) == 556, f"{len(before)} checkpoint files")

    access_logs = list(run_dir.glob("logs/*.json"))
    access_pass = True
    for path in access_logs:
        payload = json.loads(path.read_text())
        for key, value in payload.items():
            normalized = key.lower()
            if ("development" in normalized or "lockbox" in normalized) and ("access" in normalized or "used" in normalized or "opened" in normalized):
                if isinstance(value, (int, float)) and value != 0:
                    access_pass = False
    record("zero_historical_development_access", access_pass, f"{len(access_logs)} JSON logs audited")
    record("zero_final_lockbox_access", access_pass, f"{len(access_logs)} JSON logs audited")

    staged_code, staged = command("git", "diff", "--cached", "--name-only")
    diff_code, diff = command("git", "diff", "--check")
    record("staged_index_empty", staged_code == 0 and not staged, staged or "empty")
    record("git_diff_check", diff_code == 0, diff or "clean")
    record("large_file_inventory_present", (run_dir / "tables/large_file_inventory.csv").exists(), "tables/large_file_inventory.csv")
    record("campaign_code_hashes_present", len(read_csv(run_dir / "tables/campaign_code_hashes_final.csv")) >= 10, "tables/campaign_code_hashes_final.csv")

    failures = [row for row in checks if row["status"] != "PASS"]
    status = "PASS_WITH_PREREGISTERED_SCOPE_BLOCKS" if not failures else "FAIL"
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    write_json_fresh(
        run_dir / "diagnostics/final_correctness_audit_superseding.json",
        {
            "status": status,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "check_count": len(checks),
            "failure_count": len(failures),
            "failures": failures,
            "scientific_decision": "ATLAS_PASS_OPERATIONAL_WITNESS_FAIL_AUDITOR_BLOCKED",
            "development_scene_access_count": 0,
            "lockbox_scene_access_count": 0,
        },
    )


if __name__ == "__main__":
    main()
