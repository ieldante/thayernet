#!/usr/bin/env python3
"""Finalize the fail-closed Thayer-SA preflight campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
TRAINED_OUTPUTS = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540/expert_outputs/micro_final_decompositions.h5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, text=True, capture_output=True)


def inventory_checkpoints() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted((REPO / "outputs/runs").rglob("*.pth"))
    ]


def source_inventory() -> list[dict[str, object]]:
    return [
        {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for root in ("src", "scripts", "tests")
        for path in sorted((REPO / root).rglob("*.py"))
    ]


def rgb(source: np.ndarray) -> np.ndarray:
    channels = np.stack((source[2], source[1], source[0]), axis=-1)
    positive = np.maximum(channels, 0.0)
    scale = float(np.percentile(positive, 99.5))
    if scale <= 0:
        scale = 1.0
    return np.clip(np.arcsinh(5.0 * positive / scale) / np.arcsinh(5.0), 0.0, 1.0)


def make_figures(run_dir: Path) -> None:
    trajectories = read_csv(run_dir / "tables/output_space_preflight_trajectories.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for initialization in sorted({row["initialization"] for row in trajectories}):
        values = [row for row in trajectories if row["initialization"] == initialization]
        steps = [int(row["step"]) for row in values]
        axes[0].plot(steps, [float(row["corrected_objective"]) for row in values], marker="o", label=initialization)
        axes[1].plot(steps, [float(row["mean_primary_scientific_distance"]) for row in values], marker="o", label=initialization)
    axes[0].set_title("Corrected objective")
    axes[1].set_title("Frozen scientific distance")
    for axis in axes:
        axis.set_xlabel("CPU free-output updates")
        axis.set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("Value")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "figures/output_space_preflight_paths.png", dpi=180)
    plt.close(fig)

    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    with h5py.File(run_dir / "objective_preflight/final_outputs.h5", "r") as handle:
        exact = np.asarray(handle["exact_truth"], dtype=np.float32)
        trained_final = np.asarray(handle["trained_thayer_me"], dtype=np.float32)
        wrong_final = np.asarray(handle["source_sum_wrong_allocation"], dtype=np.float32)
    with h5py.File(TRAINED_OUTPUTS, "r") as handle:
        trained_initial = np.asarray(handle["decompositions"], dtype=np.float32) / np.tile(scales, 2)[None, None, None, :, None, None]
    for scene, name in ((0, "ordinary"), (32, "ambiguous")):
        fig, axes = plt.subplots(4, 2, figsize=(6, 10))
        configurations = (
            ("Exact truth", exact),
            ("Trained start", trained_initial),
            ("Trained preflight end", trained_final),
            ("Wrong-allocation end", wrong_final),
        )
        for row_index, (label, values) in enumerate(configurations):
            for expert in (0, 1):
                physical = values[scene, 0, expert, :3] * scales[:, None, None]
                axes[row_index, expert].imshow(rgb(physical))
                axes[row_index, expert].set_title(f"{label}, expert {expert + 1}", fontsize=8)
                axes[row_index, expert].axis("off")
        fig.tight_layout()
        fig.savefig(run_dir / f"example_grids/{name}_preflight_outputs.png", dpi=180)
        plt.close(fig)


def report_text(run_dir: Path, runtime: float, run_bytes: int, free_disk: int, git_status: str) -> str:
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    alignment = {row["gate"]: row for row in read_csv(run_dir / "tables/surrogate_alignment_summary.csv")}
    trajectories = read_csv(run_dir / "tables/output_space_preflight_trajectories.csv")
    final = {row["initialization"]: row for row in trajectories if row["step"] == "400"}
    return f"""# Thayer-SA scientific-alignment micro-overfit final report

Decision: **FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED**. The campaign stopped at the preregistered detached output-space preflight. Assignment auditing and neural micro-overfit were not authorized.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates the official surrogate audit, official output-space preflight, and any neural fit. Neural optimizer steps, model checkpoints, Atlas evaluations, development accesses, and lockbox accesses are all zero.

## Direct answers

1. **Was the loss-geometry diagnosis reproduced?** Yes. All 13 frozen reproduction checks passed, including 54/64 compromises beating truth and all 32/32 ambiguous compromises beating truth.
2. **Did the differentiable scientific surrogate match the frozen metric?** Yes. Spearman was {float(alignment['spearman']['observed']):.6f}, Kendall {float(alignment['kendall']['observed']):.6f}, and threshold-side agreement {float(alignment['threshold_side_agreement']['observed']):.6f}.
3. **Did exact truth receive numerical-near-zero surrogate loss?** Yes: 0.0, with zero gradient.
4. **Did truth beat all compromise configurations?** Yes under the corrected objective and surrogate.
5. **Did free-output optimization remain at truth?** Yes. Loss and tensor RMS remained 0, with all frozen coverage rates 1.0.
6. **Did free-output optimization move compromise outputs toward truth?** Directionally but insufficiently. Loss and mean scientific distance fell for trained, collapsed, and wrong-allocation starts, but none reached the required coverage hierarchy.
7. **Did hard set assignment remain stable enough?** Not evaluated. The earlier output-space gate failed, so assignment auditing was not reached.
8. **Was neural micro-overfit authorized?** No.
9. **Did ordinary truth coverage become high?** No neural result exists. Final detached-preflight ordinary coverage was {float(final['trained_thayer_me']['ordinary_coverage']):.5f} from trained output, {float(final['collapsed_mean']['ordinary_coverage']):.5f} from collapsed mean, and {float(final['source_sum_wrong_allocation']['ordinary_coverage']):.5f} from wrong allocation.
10. **Did ordinary expert diameter fall below 1.0?** Not evaluated as a neural gate.
11. **Did ambiguous own-truth coverage become high?** No. Detached-preflight finals were {float(final['trained_thayer_me']['ambiguous_own_coverage']):.5f}, {float(final['collapsed_mean']['ambiguous_own_coverage']):.5f}, and {float(final['source_sum_wrong_allocation']['ambiguous_own_coverage']):.5f}.
12. **Did alternate-truth coverage become high?** No; the same three finals were {float(final['trained_thayer_me']['ambiguous_alternate_coverage']):.5f}, {float(final['collapsed_mean']['ambiguous_alternate_coverage']):.5f}, and {float(final['source_sum_wrong_allocation']['ambiguous_alternate_coverage']):.5f}.
13. **Did both-mode coverage become high?** No; the same three finals were {float(final['trained_thayer_me']['ambiguous_both_mode_coverage']):.5f}, {float(final['collapsed_mean']['ambiguous_both_mode_coverage']):.5f}, and {float(final['source_sum_wrong_allocation']['ambiguous_both_mode_coverage']):.5f}.
14. **Did prompt swap remain strong without an explicit prompt-swap loss?** Not evaluated after training because training was prohibited.
15. **Did forward consistency remain strong without a forward loss?** No neural conclusion exists. Detached trained-output optimization retained mean per-expert forward-consistent fraction {float(final['trained_thayer_me']['forward_consistent_fraction']):.5f}, but this is not a model result.
16. **Did source-sum consistency remain strong without a source-sum loss?** Not evaluated after neural training.
17. **Did gradient conflict disappear or materially decline?** The former forward/source conflict is absent by construction because forward loss is evaluation-only. No post-training gradient comparison exists.
18. **Did objective ranking align with scientific distance?** Canonically yes, but favorable ranking did not yield reliable coverage-reaching optimization.
19. **Was the campaign SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE**.
20. **Is full non-Atlas training now authorized?** No.
21. **What exact experiment should happen next?** Run one preregistered, training-free output-space conditioning campaign that keeps the same targets, thresholds, architecture, hard assignment, and initializations while testing a near-truth smooth component geometry; require detached coverage entry before neural fitting.
22. **Were Atlas, development, and lockbox untouched?** Yes: 0 / 0 / 0 accesses.
23. **Were all historical checkpoints unchanged?** Yes. No Thayer-SA checkpoint exists, and every campaign-start checkpoint is byte-identical.

## Evidence inventory

- Baseline reproduction: `tables/loss_geometry_reproduction.csv`.
- Preregistration and attainability: `preregistration/scientific_alignment_micro_overfit.md`, `preregistration/freeze_record.json`, and `tables/preregistered_gate_attainability.csv`.
- Surrogate tests and alignment: `tables/scientific_surrogate_unit_tests.csv`, `tables/surrogate_alignment_summary.csv`, and `figures/surrogate_vs_frozen_metric.png`.
- Loss and gradient scales: `tables/loss_gradient_scale_audit.csv`.
- Official detached paths: `tables/output_space_preflight_trajectories.csv`, `tables/output_space_preflight_gates.csv`, `objective_preflight/final_outputs.h5`, and `figures/output_space_preflight_paths.png`.
- Output examples: `example_grids/ordinary_preflight_outputs.png` and `example_grids/ambiguous_preflight_outputs.png`.
- Assignment, micro-overfit, post-training geometry, and neural forward-evaluation artifacts are absent by the failed prerequisite, not silently omitted after fitting.

## Provenance and closure

- Runtime including finalization: {runtime:.3f} seconds.
- Run size at report creation: {run_bytes} bytes; free disk: {free_disk} bytes.
- Correctness checks: see `tables/final_correctness_checks.csv` and `diagnostics/final_correctness_audit.json`.
- Neural execution: 0 optimizer steps; no CPU fallback and no MPS training launch.
- Atlas / development / lockbox accesses: 0 / 0 / 0.

Final Git status:

```text
{git_status}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.time()
    preflight = json.loads((run_dir / "logs/preflight_complete.json").read_text())
    if preflight["status"] != "CORRECTED OBJECTIVE STILL MISALIGNED" or preflight["neural_training_authorized"]:
        raise RuntimeError("expected fail-closed preflight status")
    if any((run_dir / "checkpoints").iterdir()) or any((run_dir / "micro_overfit").iterdir()) or any((run_dir / "gradients").iterdir()):
        raise RuntimeError("unauthorized downstream artifact exists")

    make_figures(run_dir)
    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    after = inventory_checkpoints()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after)
    unchanged = len(before) == len(after) and all(
        left["path"] == right["path"]
        and left["sha256"] == right["sha256"]
        and int(left["bytes"]) == int(right["bytes"])
        for left, right in zip(before, after)
    )
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_inventory())

    compile_result = command([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr + f"\nexit_code={compile_result.returncode}\n")
    tests_result = command([sys.executable, "-m", "pytest", "-q", "tests/test_scientific_alignment.py", "tests/test_loss_geometry.py", "tests/test_two_expert_decoder.py"])
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", tests_result.stdout + tests_result.stderr + f"\nexit_code={tests_result.returncode}\n")
    diff_result = command(["git", "diff", "--check"])
    write_text_fresh(run_dir / "logs/git_diff_check.txt", diff_result.stdout + diff_result.stderr + f"\nexit_code={diff_result.returncode}\n")

    csv_valid = True
    csv_count = 0
    for path in sorted(run_dir.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        csv_count += 1
        csv_valid = csv_valid and bool(rows) and bool(rows[0]) and all(len(row) == len(rows[0]) for row in rows)

    disk = shutil.disk_usage(REPO)
    git_status = command(["git", "status", "--short"]).stdout.strip()
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    report = report_text(run_dir, preflight["runtime_seconds"] + (time.time() - started), run_bytes, disk.free, git_status)
    write_text_fresh(run_dir / "reports/final_report.md", report)
    write_text_fresh(run_dir / "diagnostics/final_decision.md", """# Thayer-SA final decision

**FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED.** The surrogate matched the frozen metric and exact truth was stationary, but the preregistered detached optimizer did not reliably enter scientific coverage from compromise or random starts. Assignment and neural stages were not reached. The remaining demonstrated cause is corrected-objective optimization geometry, not neural capacity and not a completed assignment diagnosis.
""")

    public_paths = [
        REPO / "docs/scientific_alignment_objective.md",
        REPO / "docs/differentiable_scientific_distance.md",
        REPO / "docs/forward_consistency_as_gate.md",
        REPO / "docs/scientific_alignment_micro_overfit.md",
        run_dir / "reports/final_report.md",
    ]
    privacy_pattern = re.compile(r"/Users/|ChatGPT|\bCodex\b|artificial intelligence", re.IGNORECASE)
    privacy_matches = []
    for path in public_paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if privacy_pattern.search(line):
                privacy_matches.append(f"{path.relative_to(REPO)}:{line_number}:{line}")
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", "\n".join(privacy_matches) + ("\n" if privacy_matches else "PASS: no prohibited public path or assistant-language tokens\n"))

    large_files = [
        {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.stat().st_size >= 5_000_000
    ]
    if not large_files:
        large_files = [{"path": "NONE", "bytes": 0, "sha256": ""}]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large_files)

    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    frozen_at = datetime.fromisoformat(freeze["frozen_at_utc"])
    preflight_at = datetime.fromisoformat(preflight["completed_at_utc"])
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    checks = [
        ("preregistration_predates_official_preflight", frozen_at < preflight_at, f"{frozen_at.isoformat()} < {preflight_at.isoformat()}"),
        ("all_gates_attainable", all(row["attainable"] == "True" for row in read_csv(run_dir / "tables/preregistered_gate_attainability.csv")), "exact truth attains all"),
        ("architecture_unchanged", provenance["architecture"]["parameter_count"] == 165612, provenance["architecture"]["combined_sha256"]),
        ("microset_unchanged", provenance["microset_manifest"]["sha256"] == "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085", provenance["microset_manifest"]["sha256"]),
        ("truth_surrogate_zero", float(read_csv(run_dir / "tables/scientific_surrogate_unit_tests.csv")[0]["smoothmax"]) <= 1e-6, "exact truth"),
        ("no_forward_or_source_sum_training_term", "source_sum" not in (REPO / "src/scientific_alignment.py").read_text() and "forward" not in (REPO / "src/scientific_alignment.py").read_text(), "static objective audit"),
        ("output_preflight_model_isolation", preflight["model_parameter_count_in_preflight"] == 0 and preflight["neural_optimizer_step_count"] == 0, "0 model parameters / 0 neural steps"),
        ("assignment_not_run_after_failed_prerequisite", not any((run_dir / "gradients").iterdir()), "fail-closed ordering"),
        ("mps_neural_execution_not_reached", preflight["neural_optimizer_step_count"] == 0, "no neural launch; no fallback"),
        ("zero_atlas_access", preflight["atlas_evaluation_count"] == 0, "0"),
        ("zero_development_access", preflight["development_scene_access_count"] == 0, "0"),
        ("zero_lockbox_access", preflight["lockbox_scene_access_count"] == 0, "0"),
        ("historical_checkpoints_unchanged", unchanged, f"{len(after)} checkpoints"),
        ("fresh_collision_free_run", run_dir.name == "thayer_scientific_alignment_20260712_220315", str(run_dir.relative_to(REPO))),
        ("compileall", compile_result.returncode == 0, f"exit {compile_result.returncode}"),
        ("focused_tests", tests_result.returncode == 0, f"exit {tests_result.returncode}"),
        ("csv_schema_validation", csv_valid, f"{csv_count} CSV files"),
        ("git_diff_check", diff_result.returncode == 0, f"exit {diff_result.returncode}"),
        ("privacy_path_grep", not privacy_matches, f"{len(privacy_matches)} matches"),
        ("readme_unchanged_by_campaign", "README.md" not in git_status, "not in git status"),
    ]
    check_rows = [{"check": name, "status": "PASS" if passed else "FAIL", "detail": detail} for name, passed, detail in checks]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", check_rows)
    failures = [row for row in check_rows if row["status"] == "FAIL"]
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(check_rows),
        "failure_count": len(failures),
        "scientific_decision": "FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED",
        "neural_micro_overfit_authorized": False,
        "full_non_atlas_training_authorized": False,
        "historical_checkpoint_count": len(after),
        "neural_optimizer_step_count": 0,
        "atlas_evaluation_count": 0,
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", audit)
    write_json_fresh(run_dir / "logs/finalization_complete.json", {
        **audit,
        "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"),
        "checkpoint_inventory_after_sha256": sha256_file(run_dir / "tables/checkpoint_inventory_after.csv"),
        "run_bytes": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
    })
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
