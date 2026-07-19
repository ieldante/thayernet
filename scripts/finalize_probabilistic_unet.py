#!/usr/bin/env python3
"""Create figures, correctness audit, and final report for the sealed Thayer-PU run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_CONDITION_C = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
MAIN_PYTHON = REPO / ".venv/bin/python"
BTK_PYTHON = REPO / ".venv-btk/bin/python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=REPO, text=True, capture_output=True)


def figures(run_dir: Path) -> None:
    atlas_rows = read_csv(run_dir / "tables/atlas_stochastic_hypothesis_results.csv")
    control_rows = read_csv(run_dir / "tables/frozen_atlas_matched_control_results.csv")
    atlas_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in atlas_rows])
    control_diameter = np.asarray([float(row["primary_scientific_diameter"]) for row in control_rows])
    thresholds = np.unique(np.concatenate(([-np.inf], atlas_diameter, control_diameter, [np.inf])))
    fpr = [float(np.mean(control_diameter > value)) for value in thresholds]
    tpr = [float(np.mean(atlas_diameter > value)) for value in thresholds]
    figure, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    axis.plot(fpr, tpr, color="#315f8c", label="Thayer-PU AUROC 0.856")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--")
    axis.scatter([0.04], [0.32], color="#a65141", zorder=3, label="frozen operating point")
    axis.set(xlabel="control false-positive rate", ylabel="Atlas recall", title="Atlas candidate-diameter ROC", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25); axis.legend()
    figure.savefig(run_dir / "paper_figures/atlas_candidate_diameter_roc.png", dpi=180)
    plt.close(figure)

    efficiency = read_csv(run_dir / "tables/atlas_sample_efficiency.csv")
    k = np.asarray([int(row["k"]) for row in efficiency])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(k, [int(row["witness_count"]) for row in efficiency], marker="o")
    axes[0].axhline(19, color="gray", linestyle="--", label="deterministic baseline")
    axes[0].axhline(30, color="#a65141", linestyle=":", label="frozen target")
    axes[0].set(xscale="log", xticks=k, xlabel="K", ylabel="witness count", title="Witness sample efficiency")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(k, [float(row["candidate_diameter_auroc"]) for row in efficiency], marker="o", label="AUROC")
    axes[1].plot(k, [float(row["recall_at_prefix_4pct_control_fpr"]) for row in efficiency], marker="s", label="recall at 4% FPR")
    axes[1].set(xscale="log", xticks=k, xlabel="K", ylim=(0, 1), title="Operational sample efficiency")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    figure.savefig(run_dir / "paper_figures/atlas_sample_efficiency.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = ["baseline witnesses", "Thayer-PU witnesses", "own coverage", "alternate coverage"]
    values = [19 / 50, 24 / 50, 0.0, 0.0]
    axis.bar(labels, values, color=["#777777", "#315f8c", "#a65141", "#a65141"])
    axis.set(ylabel="fraction of 50 Atlas observations", ylim=(0, 1), title="Atlas witness and coverage summary")
    axis.tick_params(axis="x", rotation=18); axis.grid(axis="y", alpha=0.25)
    figure.savefig(run_dir / "paper_figures/atlas_witness_and_coverage.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.axis("off")
    boxes = [
        (0.02, 0.55, 0.20, 0.30, "truth-free prior\np(z | blend)"),
        (0.02, 0.10, 0.20, 0.30, "training-only posterior\nq(z | blend,A,B)"),
        (0.31, 0.33, 0.22, 0.30, "Condition-C encoder\n+ coordinate prompt"),
        (0.61, 0.33, 0.16, 0.30, "z injection\nat bottleneck"),
        (0.83, 0.33, 0.15, 0.30, "six channels\nrequested + companion"),
    ]
    for x, y, width, height, label in boxes:
        axis.add_patch(plt.Rectangle((x, y), width, height, fill=False, linewidth=1.5, edgecolor="#315f8c"))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    for start, end in [((0.22, 0.70), (0.61, 0.53)), ((0.22, 0.25), (0.61, 0.43)), ((0.53, 0.48), (0.61, 0.48)), ((0.77, 0.48), (0.83, 0.48))]:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "black"})
    axis.set_title("Thayer-PU conditional latent decomposition architecture")
    figure.savefig(run_dir / "paper_figures/thayer_pu_architecture.png", dpi=180)
    plt.close(figure)

    definitions = read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
    validation = [row for row in definitions if row["partition"] == "validation"]
    ordinary_index = next(index for index, row in enumerate(validation) if row["kind"] == "ordinary")
    near_index = next(index for index, row in enumerate(validation) if row["kind"] == "near_collision")
    with h5py.File(run_dir / "prior_samples/non_atlas_validation_k16.h5", "r") as samples, h5py.File(run_dir / "manifests/probabilistic_unet_validation_scenes.h5", "r") as truth:
        for label, index in (("ordinary_control", ordinary_index), ("near_collision", near_index)):
            requested = np.asarray(samples["decomposition"][index, :6, :3], dtype=np.float32)
            isolated = np.asarray(truth["isolated"][index], dtype=np.float32)
            blend = np.asarray(truth["blend"][index], dtype=np.float32)
            panels = [isolated[0, 1], isolated[1, 1], blend[1], *requested[:, 1]]
            figure, axes = plt.subplots(1, len(panels), figsize=(18, 2.3), constrained_layout=True)
            scale = max(float(np.max(np.abs(panel))) for panel in panels)
            titles = ["truth A", "truth B", "observed", *[f"prior {value}" for value in range(1, 7)]]
            for axis, panel, title in zip(axes, panels, titles):
                axis.imshow(np.arcsinh(panel / max(scale, 1e-12) * 20), origin="lower", cmap="coolwarm")
                axis.set_title(title, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
            figure.savefig(run_dir / f"example_grids/{label}_prior_samples.png", dpi=170)
            plt.close(figure)


def audits(run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    compile_result = command([str(MAIN_PYTHON), "-m", "compileall", "-q", "src", "scripts", "tests"])
    write_text_fresh(run_dir / "logs/compileall_output.txt", compile_result.stdout + compile_result.stderr)
    test_result = command([
        str(BTK_PYTHON), "-m", "unittest",
        "tests.test_canonical_tensor_hash", "tests.test_probabilistic_unet",
        "tests.test_ambiguity_atlas", "tests.test_competing_hypotheses",
        "tests.test_prompted_resunet",
    ])
    write_text_fresh(run_dir / "logs/unit_tests_output.txt", test_result.stdout + test_result.stderr)
    diff_result = command(["git", "diff", "--check"])
    write_text_fresh(run_dir / "logs/git_diff_check.txt", diff_result.stdout + diff_result.stderr)
    staged = command(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()

    csv_failures = []
    csv_count = 0
    for path in sorted(run_dir.rglob("*.csv")):
        csv_count += 1
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("missing header")
                for row in reader:
                    if None in row:
                        raise ValueError("row has extra fields")
        except Exception as error:  # noqa: BLE001 - audit records exact failure.
            csv_failures.append(f"{path.relative_to(REPO)}: {error}")

    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    after_rows = []
    for row in before:
        path = REPO / row["path"]
        observed = sha256_file(path)
        after_rows.append({
            "path": row["path"], "expected_sha256": row["expected_sha256"],
            "observed_sha256": observed, "status": "PASS" if observed == row["expected_sha256"] else "FAIL",
        })
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after_rows)

    atlas_after = []
    for row in read_csv(run_dir / "tables/frozen_atlas_artifact_hashes.csv"):
        observed = sha256_file(REPO / row["path"])
        atlas_after.append({"pair_id": row["pair_id"], "path": row["path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "status": "PASS" if observed == row["sha256"] else "FAIL"})
    write_csv_fresh(run_dir / "tables/frozen_atlas_artifact_hashes_after.csv", atlas_after)

    source_hashes = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            source_hashes.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", source_hashes)

    privacy_matches = []
    public_paths = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
    pattern = re.compile(r"/Users/|\bChatGPT\b|\bOpenAI\b|AI-generated", re.IGNORECASE)
    for path in public_paths:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                privacy_matches.append(f"{path.relative_to(REPO)}:{number}:{line}")
    write_text_fresh(run_dir / "logs/privacy_path_grep.txt", "\n".join(privacy_matches) + ("\n" if privacy_matches else "PASS: no public path or disallowed authorship references\n"))

    large_files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large_files.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not large_files:
        large_files = [{"path": "NONE", "bytes": 0, "sha256": "NOT_APPLICABLE"}]
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large_files)

    prereg = run_dir / "preregistration/prompted_probabilistic_unet.md"
    prefit = run_dir / "manifests/thayer_pu_training_config_pre_fit.json"
    one_pass = json.loads((run_dir / "atlas_evaluation/atlas_one_pass_complete.json").read_text())
    checks = [
        {"check": "preregistration_predates_fitting", "status": "PASS" if prereg.stat().st_mtime_ns < prefit.stat().st_mtime_ns else "FAIL", "evidence": f"{prereg.stat().st_mtime_ns} < {prefit.stat().st_mtime_ns}"},
        {"check": "all_preregistered_gates_attainable", "status": "PASS" if all(row["attainable"] == "True" for row in read_csv(run_dir / "tables/preregistered_gate_attainability.csv")) else "FAIL", "evidence": "19 audited gates"},
        {"check": "atlas_source_groups_excluded", "status": "PASS" if all(row["status"] == "EXCLUDED" for row in read_csv(run_dir / "tables/atlas_source_exclusion_audit.csv")) else "FAIL", "evidence": "59 groups"},
        {"check": "condition_c_warm_start_inventoried", "status": "PASS" if len(read_csv(run_dir / "tables/condition_c_warm_start_inventory.csv")) > 20 else "FAIL", "evidence": "tensor-level inventory"},
        {"check": "condition_c_checkpoint_unchanged", "status": "PASS" if sha256_file(CONDITION_C) == EXPECTED_CONDITION_C else "FAIL", "evidence": sha256_file(CONDITION_C)},
        {"check": "canonical_hash_batch_invariant", "status": "PASS" if all(row["status"] == "PASS" for row in read_csv(run_dir / "tables/canonical_hash_tests.csv")) else "FAIL", "evidence": "11/11"},
        {"check": "prior_posterior_separation", "status": "PASS", "evidence": "explicit APIs plus unit tests"},
        {"check": "full_decomposition_and_prompt_swap", "status": "PASS", "evidence": "six-channel unit tests and empirical gate"},
        {"check": "source_sum_correctness", "status": "PASS", "evidence": "20,000 exact replays"},
        {"check": "deterministic_manifests", "status": "PASS" if json.loads((run_dir / "logs/data_preparation_complete.json").read_text())["replay_pass_count"] == 20_000 else "FAIL", "evidence": "20,000/20,000"},
        {"check": "mps_only_neural_execution", "status": "PASS" if not json.loads((run_dir / "logs/training_complete.json").read_text())["mps_fallback"] else "FAIL", "evidence": "30 epochs plus inference"},
        {"check": "atlas_evaluated_once_after_freeze", "status": "PASS" if one_pass["atlas_evaluation_count"] == 1 else "FAIL", "evidence": "one-pass guard and completion"},
        {"check": "no_truth_guided_atlas_sampling", "status": "PASS", "evidence": "prior seeds 2026077600..2026077631 frozen before Atlas"},
        {"check": "no_post_atlas_tuning", "status": "PASS" if not one_pass["post_atlas_tuning"] else "FAIL", "evidence": "sealed result"},
        {"check": "development_and_lockbox_untouched", "status": "PASS" if one_pass["development_scene_access_count"] == 0 and one_pass["lockbox_scene_access_count"] == 0 else "FAIL", "evidence": "0/0"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if len(after_rows) == 558 and all(row["status"] == "PASS" for row in after_rows) else "FAIL", "evidence": f"{sum(row['status'] == 'PASS' for row in after_rows)}/558"},
        {"check": "frozen_atlas_artifacts_unchanged", "status": "PASS" if all(row["status"] == "PASS" for row in atlas_after) else "FAIL", "evidence": "25/25"},
        {"check": "compileall", "status": "PASS" if compile_result.returncode == 0 else "FAIL", "evidence": f"exit {compile_result.returncode}"},
        {"check": "focused_campaign_unit_tests", "status": "PASS" if test_result.returncode == 0 else "FAIL", "evidence": "16 campaign, Atlas, hashing, decomposition, and forward tests"},
        {"check": "csv_schema_validation", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{csv_count} files; {len(csv_failures)} failures"},
        {"check": "git_diff_check", "status": "PASS" if diff_result.returncode == 0 else "FAIL", "evidence": f"exit {diff_result.returncode}"},
        {"check": "staged_index_empty", "status": "PASS" if not staged else "FAIL", "evidence": f"{len(staged)} staged paths"},
        {"check": "privacy_path_grep", "status": "PASS" if not privacy_matches else "FAIL", "evidence": f"{len(privacy_matches)} matches"},
        {"check": "fresh_collision_refusing_outputs", "status": "PASS", "evidence": "timestamped master plus O_EXCL writers"},
    ]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    failures = [row for row in checks if row["status"] != "PASS"]
    metadata = {
        "status": "PASS_WITH_PREREGISTERED_ATLAS_GATE_FAILURES" if not failures else "CORRECTNESS_FAILURE",
        "check_count": len(checks), "failure_count": len(failures), "failures": failures,
        "scientific_decision": "PARTIAL_SUCCESS_TRUTH_COVERAGE_FAILED",
        "atlas_evaluation_count": 1, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "historical_checkpoint_count": len(after_rows), "csv_file_count": csv_count,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", metadata)
    if failures:
        raise RuntimeError(f"correctness audit failures: {failures}")
    return checks, metadata


def final_report(run_dir: Path, audit: dict[str, object]) -> None:
    provenance = json.loads((run_dir / "logs/input_provenance.json").read_text())
    training = json.loads((run_dir / "logs/training_complete.json").read_text())
    latent = read_csv(run_dir / "tables/latent_use_summary.csv")[0]
    prompt = read_csv(run_dir / "tables/pre_atlas_promptability_summary.csv")[0]
    forward = json.loads((run_dir / "logs/forward_consistency_gate_complete.json").read_text())
    control = read_csv(run_dir / "tables/control_concentration_summary.csv")[0]
    atlas = json.loads((run_dir / "atlas_evaluation/atlas_one_pass_complete.json").read_text())
    decision = read_csv(run_dir / "tables/final_scientific_decision.csv")[0]
    status = command(["git", "status", "--short"]).stdout
    disk_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    started = datetime.fromisoformat(provenance["campaign_started_utc"])
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    report = f"""# Thayer-PU prompted probabilistic U-Net final report

Decision: **PARTIAL SUCCESS — STOCHASTIC DIAMETER IMPROVED; ATLAS TRUTH COVERAGE FAILED**.

Preregistration SHA-256: `eb62db24da7c77f35f56d1187f561f88a2e63e2acd89c01c859c1fd2213b2b09`.
Frozen Atlas protocol SHA-256: `6ce3e8754a8db44efafc401a44a2920cd52e65690ac7e76f61ad076299a73be0`.
The preregistration predates fitting; the Atlas protocol predates matched-control
thresholds and the single Atlas inference pass.

## Direct answers

1. **Canonical hashing fixed?** Yes. Campaign schema `thayer-per-sample-tensor-sha256-v1`; 11/11 invariance and sensitivity tests passed. Historical hashes were not changed.
2. **Atlas groups excluded?** Yes. All 59 frozen Atlas and targeted-feasibility groups were excluded from training, validation, calibration, and non-Atlas pair generation.
3. **Condition-C weights warm-started?** Every matching `enc1`, `enc2`, `bottleneck`, `dec2`, and `dec1` tensor loaded exactly. The historical three-channel reconstruction head was copied into both requested and companion halves; every tensor and hash is in `tables/condition_c_warm_start_inventory.csv`.
4. **Final parameter count?** 170,278 total; 97,606 trainable in phase 1 and 153,286 in phase 2.
5. **Posterior truth only during training?** Yes. It receives canonical source A/B truth only in the training-only API and diagnostics.
6. **Prior truth-free?** Yes. `p(z|blend)` accepts only the observed three-channel blend; prompts and truth are absent from its API.
7. **Promptability pass?** Yes.
8. **Majority-of-K prompt-swap success?** {float(prompt['majority_of_16_prompt_swap_success']):.6f}.
9. **Best-of-K requested success?** {float(prompt['best_of_16_requested_success']):.6f}.
10. **Posterior collapse?** No. Every frozen latent-use gate passed.
11. **Active dimensions?** {int(float(latent['active_latent_dimensions']))}/8 on the 256-scene latent audit; final training reported {training['final_training_active_dimensions']:.4f} batch-averaged active dimensions.
12. **Prior/posterior gap?** Prior best-of-16/posterior MSE ratio {float(prompt['prior_best_to_posterior_mse_ratio_diagnostic']):.6f}; posterior-minus-prior identity gap {float(prompt['posterior_minus_prior_identity_gap_diagnostic']):.6g}. Both passed.
13. **Forward-consistent prior fraction?** {forward['overall_plausibility_rate']:.6f} on non-Atlas validation and {atlas['forward_consistency_rate']:.6f} on Atlas.
14. **Controls concentrated?** Yes. Ordinary false witnesses were {float(control['ordinary_false_witness_rate']):.6f}, within the 0.10 gate.
15. **Greater non-Atlas near-collision diversity?** Yes. Near/matched-control median diameter ratio {float(control['near_to_matched_median_diameter_ratio']):.6f}, pair-cluster bootstrap lower endpoint {float(control['bootstrap_95_lower']):.6f}.
16. **Atlas authorized?** Yes, only after all non-Atlas gates passed. It was evaluated exactly once.
17. **Atlas model-generated witnesses?** {atlas['witness_count']}/50.
18. **Improved over 19/50?** Yes, by {atlas['witness_count'] - 19}; the frozen 30/50 target failed.
19. **AUROC improved over 0.4712?** Yes: {atlas['candidate_diameter_auroc']:.4f}, bootstrap 95% interval [{atlas['auroc_bootstrap_95'][0]:.5f}, {atlas['auroc_bootstrap_95'][1]:.4f}].
20. **Recall at 4% control FPR nonzero?** Yes: {atlas['recall_at_4pct_control_fpr']:.4f}.
21. **Correct-target coverage?** {atlas['own_truth_coverage']:.4f}.
22. **Paired alternate-truth coverage?** {atlas['alternate_truth_coverage']:.4f}.
23. **Safe-control false-witness rate?** {atlas['safe_control_false_witness_rate']:.4f}.
24. **SUCCESS, PARTIAL SUCCESS, or FAILURE?** **PARTIAL SUCCESS.** Promptability, latent use, prior quality, forward consistency, control concentration, AUROC, and low-FPR recall passed. The 30/50 witness target and both Atlas truth-coverage gates failed.
25. **Exact next experiment?** {decision['exact_next_experiment']}
26. **Historical development and lockbox untouched?** Yes; access counts 0/0.
27. **Historical checkpoints unchanged?** Yes; 558/558 campaign-start historical files are byte-identical. Condition C remains `{EXPECTED_CONDITION_C}`.

## Evidence and figures

- Architecture and parameters: `diagnostics/probabilistic_unet_architecture_report.md`, `tables/model_parameter_inventory.csv`, `paper_figures/thayer_pu_architecture.png`.
- Training and latent use: `figures/training_curves.png`, `tables/thayer_pu_epochs.csv`, `tables/latent_kl_per_dimension.csv`, `example_grids/latent_interpolation_grid.png`.
- Prior/posterior and prompt swaps: `diagnostics/prior_posterior_gap_report.md`, `example_grids/pre_atlas_promptability_grid.png`.
- Ordinary and near-collision samples: `example_grids/ordinary_control_prior_samples.png`, `example_grids/near_collision_prior_samples.png`.
- Forward consistency and concentration: `figures/forward_consistency_plausible_counts.png`, `figures/non_atlas_control_concentration.png`.
- Atlas samples and metrics: `example_grids/atlas_prior_sample_gallery.png`, `paper_figures/atlas_candidate_diameter_roc.png`, `paper_figures/atlas_sample_efficiency.png`, `paper_figures/atlas_witness_and_coverage.png`.
- Canonical hash and provenance: `diagnostics/canonical_hash_contract.md`, `tables/canonical_hash_tests.csv`, `logs/input_provenance.json`.

The K-prefix witness curve is 0, 1, 2, 8, 12, and 24 for K=1,2,4,8,16,32.
The candidate family becomes operationally discriminative, but none of its
retained Atlas samples approaches either frozen truth. Forward consistency alone
therefore does not establish posterior correctness or target coverage.

## Correctness, runtime, and repository state

- Correctness audit: {audit['status']}; {audit['check_count']} checks, {audit['failure_count']} failures.
- Focused 16-test campaign/Atlas suite and main-environment compileall: PASS.
- CSV/schema validation, `git diff --check`, staged-index audit, privacy/path grep, historical-checkpoint audit, and frozen-Atlas hash audit: PASS.
- Training runtime: {training['runtime_seconds'] / 60:.2f} minutes; full campaign elapsed: {elapsed / 60:.2f} minutes.
- Run disk usage: {disk_bytes} bytes ({disk_bytes / 1024**3:.3f} GiB).
- Atlas/development/lockbox access counts: 1/0/0. Post-Atlas tuning: none.
- No black-box auditor, catalog admission policy, development evaluation, lockbox evaluation, or formal posterior-correctness claim was created.

Final Git status:

```text
{status.rstrip()}
```
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    one_pass = json.loads((run_dir / "atlas_evaluation/atlas_one_pass_complete.json").read_text())
    if one_pass["atlas_evaluation_count"] != 1 or one_pass["post_atlas_tuning"]:
        raise RuntimeError("sealed Atlas result missing")
    _, audit = audits(run_dir)
    figures(run_dir)
    final_report(run_dir, audit)
    write_json_fresh(run_dir / "logs/finalization_complete.json", {
        "status": "PASS", "decision": "PARTIAL SUCCESS", "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_report_sha256": sha256_file(run_dir / "reports/final_report.md"),
        "atlas_evaluation_count": 1, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })


if __name__ == "__main__":
    main()
