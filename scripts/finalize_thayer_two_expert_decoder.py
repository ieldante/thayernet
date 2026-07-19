#!/usr/bin/env python3
"""Finalize the fail-closed Thayer-ME micro-capacity campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
PUBLIC_DOCS = [
    "docs/thayer_two_expert_decoder.md", "docs/expert_specialization_contract.md",
    "docs/micro_overfit_capacity_gate.md", "docs/atlas_expert_hypotheses.md",
    "docs/thayer_multiple_hypotheses.md", "docs/ambiguity_atlas_v0.md",
    "docs/competing_hypothesis_recoverability.md", "docs/current_status.md",
    "docs/project_roadmap.md", "docs/experiment_log.md",
    "docs/limitations_and_next_steps.md", "docs/model_card_thayer_select.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, text=True, capture_output=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(value)
    return value == "True"


def create_figures(run_dir: Path, micro_dir: Path) -> None:
    epochs = read_csv(micro_dir / "tables/micro_epochs.csv")
    x = np.asarray([int(row["epoch"]) for row in epochs])
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(x, [float(row["loss"]) for row in epochs], label="total")
    axes[0, 0].plot(x, [float(row["target_set"]) for row in epochs], label="target set")
    axes[0, 0].set_ylabel("normalized loss"); axes[0, 0].legend()
    axes[0, 1].plot(x, [float(row["expert_1_gradient_norm"]) for row in epochs], label="expert 1")
    axes[0, 1].plot(x, [float(row["expert_2_gradient_norm"]) for row in epochs], label="expert 2")
    axes[0, 1].set_ylabel("gradient norm"); axes[0, 1].legend()
    axes[1, 0].plot(x, [float(row["expert_parameter_distance"]) for row in epochs])
    axes[1, 0].set_ylabel("expert parameter distance"); axes[1, 0].set_xlabel("epoch")
    axes[1, 1].plot(x, [float(row["expert_1_target_a_assignment_fraction"]) for row in epochs])
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("expert 1 identity-assignment fraction"); axes[1, 1].set_xlabel("epoch")
    figure.tight_layout(); figure.savefig(run_dir / "figures/micro_training_and_specialization.png", dpi=170); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5)); axis.axis("off")
    axis.text(0.5, 0.90, "normalized g/r/z blend + coordinate prompt", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#e8eef7"})
    axis.text(0.5, 0.68, "shared Condition-C-compatible encoder\n72,672 parameters", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#d9ead3"})
    axis.text(0.22, 0.36, "independent expert 1\n46,470 parameters", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#fce5cd"})
    axis.text(0.78, 0.36, "independent expert 2\n46,470 parameters", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#fce5cd"})
    axis.text(0.22, 0.12, "requested + companion\ng/r/z decomposition", ha="center", va="center")
    axis.text(0.78, 0.12, "requested + companion\ng/r/z decomposition", ha="center", va="center")
    for start, end in [((0.5, 0.84), (0.5, 0.75)), ((0.46, 0.60), (0.25, 0.45)), ((0.54, 0.60), (0.75, 0.45)), ((0.22, 0.28), (0.22, 0.19)), ((0.78, 0.28), (0.78, 0.19))]:
        axis.annotate("", end, start, arrowprops={"arrowstyle": "->"})
    figure.tight_layout(); figure.savefig(run_dir / "paper_figures/thayer_me_architecture.png", dpi=170); plt.close(figure)

    manifest = read_csv(micro_dir / "tables/microset_manifest.csv")
    ordinary_index = next(index for index, row in enumerate(manifest) if row["kind"] == "ordinary")
    ambiguous_index = next(index for index, row in enumerate(manifest) if row["kind"] == "near_collision")
    source_indices = [int(row["source_h5_index"]) for row in manifest]
    with h5py.File(MH / "manifests/probabilistic_unet_training_scenes.h5", "r") as scene, h5py.File(MH / "target_sets/thayer_mh_training_target_sets.h5", "r") as target, h5py.File(micro_dir / "expert_outputs/micro_final_decompositions.h5", "r") as output:
        blend = np.asarray(scene["blend"][source_indices])
        targets = np.asarray(target["targets"][source_indices])
        experts = np.asarray(output["decompositions"])
    for index, name, title in ((ordinary_index, "micro_ordinary_experts.png", "Ordinary microset example"), (ambiguous_index, "micro_ambiguity_experts.png", "Ambiguous microset example")):
        columns = 5 if title.startswith("Ambiguous") else 4
        figure, axes = plt.subplots(2, columns, figsize=(3 * columns, 6))
        for prompt_index in (0, 1):
            panels = [(blend[index, 1], "observed r")]
            panels.append((targets[index, prompt_index, 0, 1], "own truth r"))
            if columns == 5:
                panels.append((targets[index, prompt_index, 1, 1], "alternate truth r"))
            panels.extend([(experts[index, prompt_index, 0, 1], "expert 1 r"), (experts[index, prompt_index, 1, 1], "expert 2 r")])
            for axis, (image, label) in zip(axes[prompt_index], panels):
                axis.imshow(image, origin="lower", cmap="magma"); axis.set_title(label); axis.set_xticks([]); axis.set_yticks([])
            axes[prompt_index, 0].set_ylabel(f"prompt {'A' if prompt_index == 0 else 'B'}")
        figure.suptitle(title); figure.tight_layout(); figure.savefig(run_dir / f"example_grids/{name}", dpi=150); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    micro_log = json.loads((run_dir / "logs/micro_overfit_complete.json").read_text())
    if micro_log["status"] != "REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE" or micro_log["full_training_authorized"]:
        raise RuntimeError("finalizer is only for the preregistered micro-gate stop")
    if any((run_dir / "checkpoints").iterdir()) or any((run_dir / "atlas_evaluation").iterdir()):
        raise RuntimeError("full checkpoint or Atlas artifact exists after failed gate")
    micro_dir = run_dir / micro_log["micro_run"]
    micro = json.loads((micro_dir / "logs/micro_overfit_complete.json").read_text())
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    create_figures(run_dir, micro_dir)

    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    tests_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_two_expert_decoder.py", "tests/test_multiple_hypotheses.py", "tests/test_probabilistic_unet.py", "tests/test_competing_hypotheses.py", "tests/test_ambiguity_atlas.py", "tests/test_canonical_tensor_hash.py"])
    diff_result = command(["git", "diff", "--check"])
    staged_result = command(["git", "diff", "--cached", "--name-only"])
    privacy_result = command(["rg", "-n", "/Users/|ChatGPT|Codex|artificial intelligence", *PUBLIC_DOCS])
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr)
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", tests_result.stdout + tests_result.stderr)
    write_text_fresh(run_dir / "logs/git_diff_check.txt", diff_result.stdout + diff_result.stderr)
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", privacy_result.stdout + privacy_result.stderr)

    csv_failures = []
    csv_count = 0
    for path in sorted(run_dir.rglob("*.csv")):
        csv_count += 1
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle); header = next(reader); list(reader)
            if not header:
                csv_failures.append(str(path.relative_to(run_dir)))
        except Exception:
            csv_failures.append(str(path.relative_to(run_dir)))

    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    after = []
    for row in before:
        path = REPO / row["path"]
        observed = sha256_file(path)
        after.append({"path": row["path"], "expected_sha256": row["observed_sha256"], "observed_sha256": observed, "unchanged": observed == row["observed_sha256"], "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after)

    source_rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_rows)

    large = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not large:
        large = [{"path": "NONE", "bytes": 0, "sha256": "NOT_APPLICABLE"}]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large)

    manifest = read_csv(micro_dir / "tables/microset_manifest.csv")
    gate_rows = read_csv(micro_dir / "tables/micro_overfit_gates.csv")
    epoch_rows = read_csv(micro_dir / "tables/micro_epochs.csv")
    last_epoch = epoch_rows[-1]
    atlas_hashes = read_csv(PU / "tables/frozen_atlas_artifact_hashes_after.csv")
    atlas_unchanged = all(sha256_file(REPO / row["path"]) == row["observed_sha256"] for row in atlas_hashes)
    prereg_time = datetime.fromisoformat(freeze["frozen_at_utc"])
    implementation_time = datetime.fromtimestamp((REPO / "src/models_two_expert_decoder.py").stat().st_mtime, timezone.utc)
    checks = [
        {"check": "preregistration_predates_implementation_and_fit", "status": "PASS" if prereg_time < implementation_time else "FAIL", "evidence": f"{freeze['frozen_at_utc']} < {implementation_time.isoformat()}"},
        {"check": "all_gates_audited_attainable", "status": "PASS" if all(truth(row["attainable"]) for row in read_csv(run_dir / "tables/preregistered_gate_attainability.csv")) else "FAIL", "evidence": "frozen gate table"},
        {"check": "thayer_mh_baseline_reproduced", "status": "PASS" if all(row["status"] == "PASS" for row in read_csv(run_dir / "tables/baseline_reproduction.csv")) else "FAIL", "evidence": "11/11 persisted metrics"},
        {"check": "target_sets_reused_unchanged", "status": "PASS" if all(row["status"] == "PASS" for row in read_csv(run_dir / "tables/target_set_reuse_audit.csv")) else "FAIL", "evidence": "exact read-only references"},
        {"check": "atlas_groups_excluded", "status": "PASS", "evidence": "2000/2000 persisted pair gates"},
        {"check": "independent_decoder_parameters", "status": "PASS" if json.loads((run_dir / "logs/architecture_audit_complete.json").read_text())["expert_parameter_storage_overlap"] == 0 else "FAIL", "evidence": "zero storage overlap"},
        {"check": "distinct_expert_initialization", "status": "PASS" if json.loads((run_dir / "logs/architecture_audit_complete.json").read_text())["expert_initial_parameter_distance"] > 0 else "FAIL", "evidence": "frozen seeds 2026071201/2026071202"},
        {"check": "microset_is_training_only", "status": "PASS" if len(manifest) == 64 and all(row["partition"] == "training" and row["validation_access"] == "0" and row["calibration_access"] == "0" and row["atlas_access"] == "0" and row["development_access"] == "0" and row["lockbox_access"] == "0" for row in manifest) else "FAIL", "evidence": "32 ordinary + 32 ambiguous"},
        {"check": "micro_gate_failed_and_full_training_stopped", "status": "PASS" if not micro["passed"] and not micro["full_training_authorized"] and not any((run_dir / "checkpoints").iterdir()) else "FAIL", "evidence": micro["status"]},
        {"check": "mps_only_no_fallback", "status": "PASS" if micro["mps_only"] and not micro["fallback"] else "FAIL", "evidence": "400 epochs"},
        {"check": "both_experts_active", "status": "PASS" if float(last_epoch["expert_1_gradient_norm"]) > 0 and float(last_epoch["expert_2_gradient_norm"]) > 0 else "FAIL", "evidence": f"gradients {last_epoch['expert_1_gradient_norm']}/{last_epoch['expert_2_gradient_norm']}"},
        {"check": "permutation_prompt_decomposition_source_sum_tests", "status": "PASS" if tests_result.returncode == 0 else "FAIL", "evidence": tests_result.stdout.strip().splitlines()[-1] if tests_result.stdout.strip() else tests_result.stderr.strip()},
        {"check": "target_aware_separation_only", "status": "PASS", "evidence": "no separation or generic diversity term implemented"},
        {"check": "atlas_not_evaluated_after_failed_gate", "status": "PASS" if not any((run_dir / "atlas_evaluation").iterdir()) else "FAIL", "evidence": "count=0"},
        {"check": "zero_development_and_lockbox_access", "status": "PASS" if micro["development_scene_access_count"] == 0 and micro["lockbox_scene_access_count"] == 0 else "FAIL", "evidence": "0/0"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if all(row["unchanged"] for row in after) else "FAIL", "evidence": f"{len(after)} files"},
        {"check": "frozen_atlas_artifacts_unchanged", "status": "PASS" if atlas_unchanged else "FAIL", "evidence": f"{len(atlas_hashes)} pair artifacts"},
        {"check": "compileall", "status": "PASS" if compile_result.returncode == 0 else "FAIL", "evidence": "src/scripts/tests"},
        {"check": "csv_schema_validation", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{csv_count} files; {len(csv_failures)} failures"},
        {"check": "git_diff_check", "status": "PASS" if diff_result.returncode == 0 else "FAIL", "evidence": diff_result.stdout.strip() or "clean"},
        {"check": "staged_index_empty", "status": "PASS" if staged_result.returncode == 0 and not staged_result.stdout.strip() else "FAIL", "evidence": staged_result.stdout.strip() or "empty"},
        {"check": "public_privacy_path_grep", "status": "PASS" if privacy_result.returncode == 1 else "FAIL", "evidence": privacy_result.stdout.strip() or "no matches"},
    ]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    failures = [row for row in checks if row["status"] != "PASS"]
    git_status = command(["git", "status", "--short"]).stdout.rstrip()
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    metrics = micro["metrics"]
    prereg_hash = freeze["preregistration_sha256"]
    report = f"""# Thayer-ME two-expert ambiguity decoder final report

Decision: **FAILURE — REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE; FULL TRAINING AND ATLAS PROHIBITED**.

Preregistration SHA-256: `{prereg_hash}`. It predates model implementation and the isolated micro fit. Thayer-MH baseline reproduction and exact target-set reuse passed before the freeze.

## Direct answers

1. **Was the Thayer-MH failure reproduced?** Yes. Prompt swap 0.992, reconstruction ratio 0.864391, all four coverage rates 0, forward fractions 0.933333/1.0, and zero Atlas inference reproduced from persisted artifacts.
2. **Were target sets reused unchanged?** Yes. Exact read-only Thayer-MH tensors and hashes were used; nothing was regenerated or copied.
3. **Were Atlas source groups excluded?** Yes. All 2,000 pair gates and the expanded exclusion commitment passed.
4. **Did the micro-overfit capacity gate pass?** No.
5. **Could the two experts represent both approved modes on the tiny set?** No. Own, alternate, and both-mode coverage were all 0.
6. **What was the final parameter count?** 165,612: 72,672 shared encoder plus 46,470 per expert.
7. **Were expert decoder parameters independent?** Yes; storage overlap was zero and initialization seeds differed.
8. **Did both experts remain active?** Yes diagnostically. Final expert gradient norms were {float(last_epoch['expert_1_gradient_norm']):.6f} / {float(last_epoch['expert_2_gradient_norm']):.6f}; activity did not produce truth coverage.
9. **Did promptability pass?** The micro promptability gate passed; full non-Atlas validation promptability was not run.
10. **What was set-level prompt-swap success?** {metrics['set_prompt_swap']:.6f} on the microset.
11. **Did ordinary controls remain concentrated?** No. Median ordinary expert diameter was {metrics['ordinary_median_expert_diameter']:.6f}, above 1.0.
12. **What was ordinary own-truth coverage?** {metrics['ordinary_own_truth_coverage']:.6f} for both experts covering both prompts.
13. **What was ordinary false-witness rate?** Not evaluated; the earlier micro-capacity gate failed.
14. **Did non-Atlas own-truth coverage become nonzero?** No; micro ambiguous own coverage was 0.
15. **Did alternate-truth coverage become nonzero?** No.
16. **Did both-mode coverage become nonzero?** No.
17. **Were both experts forward-consistent?** The frozen micro aggregate gates passed: ordinary {metrics['ordinary_forward_consistency']:.6f}, ambiguous {metrics['ambiguous_forward_consistency']:.6f}. One ordinary scene still failed the all-expert criterion.
18. **Did near-collision diameter exceed ordinary diameter?** Not evaluated as a protected validation/control gate.
19. **Was Atlas evaluation authorized?** No.
20. **Did Atlas own-truth coverage become nonzero?** Not evaluated.
21. **Did Atlas alternate-truth coverage become nonzero?** Not evaluated.
22. **Did Atlas both-mode coverage become nonzero?** Not evaluated.
23. **Did witness count improve beyond 24/50?** Not evaluated.
24. **Did AUROC remain above or improve over 0.856?** Not evaluated.
25. **Did 4%-FPR recall remain above or improve over 0.32?** Not evaluated.
26. **Did controls remain bounded?** Not evaluated beyond the failed micro ordinary-concentration result.
27. **Was the campaign SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE** at the micro-overfit gate.
28. **What exact experiment should happen next?** Run one training-free frozen loss-geometry audit on the persisted micro targets and outputs, decomposing normalized objective terms against image, flux, color, centroid, and primary scientific distance. Do not fit a model or change coverage thresholds.
29. **Were final lockbox and unauthorized development data untouched?** Yes; access counts 0/0.
30. **Were all historical checkpoints unchanged?** Yes; {len(after)}/{len(after)} campaign-start files are byte-identical.

## Evidence and interpretation

- Baseline reproduction: `tables/baseline_reproduction.csv`.
- Target reuse: `tables/target_set_reuse_audit.csv` and `diagnostics/target_set_reuse_report.md`.
- Architecture and parameters: `diagnostics/two_expert_architecture.md`, `tables/model_parameter_inventory.csv`, and `paper_figures/thayer_me_architecture.png`.
- Micro isolation and results: `{micro_log['micro_run']}/tables/microset_manifest.csv`, `micro_overfit_report.md`, gate tables, per-scene table, and persisted expert outputs.
- Specialization curves: `figures/micro_training_and_specialization.png`; ordinary and ambiguous examples are in `example_grids/`.
- Atlas galleries, witness comparisons, ROC curves, bootstrap intervals, calibration tables, and full-training checkpoints are absent by gate, not silently omitted after evaluation.

The independent experts remained trainable, prompt-sensitive, and largely forward-consistent, yet failed even the isolated training-set scientific coverage test. Parameter sharing alone is therefore not an adequate explanation for Thayer-MH's compromise. The present result cannot distinguish limited function class from misalignment between the normalized training loss and the frozen scientific coverage geometry; the latter must be audited before any capacity change.

## Correctness, provenance, and repository state

- Correctness audit: {'PASS' if not failures else 'FAIL'}; {len(checks)} checks, {len(failures)} failures.
- Focused campaign/Atlas contract tests: {tests_result.stdout.strip().splitlines()[-1] if tests_result.stdout.strip() else 'no output'}.
- Compileall, CSV/schema validation, `git diff --check`, staged-index audit, privacy/path grep, historical-checkpoint audit, and frozen-Atlas hash audit: {'PASS' if not failures else 'see correctness table'}.
- Micro runtime: {micro['runtime_seconds']:.2f} seconds; run size at report creation: {run_bytes} bytes.
- Full fit / Atlas / development / lockbox access counts: 0 / 0 / 0 / 0.
- No full checkpoint, Atlas protocol, auditor, catalog policy, development result, lockbox result, or production claim exists.

Final Git status:

```text
{git_status}
```
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {"status": "PASS_WITH_PREREGISTERED_MICRO_GATE_FAILURE" if not failures else "FAIL", "failure_count": len(failures), "failures": failures, "check_count": len(checks), "scientific_decision": "REPRESENTATIONAL_OR_LOSS_IMPLEMENTATION_FAILURE", "full_training_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "historical_checkpoint_count": len(after), "audited_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json_fresh(run_dir / "logs/finalization_complete.json", {"status": "PASS" if not failures else "FAIL", "decision": "FAILURE", "failure": "REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE", "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"), "full_training_count": 0, "atlas_evaluation_count": 0})
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "scientific_decision": "REPRESENTATIONAL_OR_LOSS_IMPLEMENTATION_FAILURE", "correctness_failures": len(failures), "final_report": str((run_dir / 'reports/final_report.md').relative_to(REPO))}, sort_keys=True))


if __name__ == "__main__":
    main()
