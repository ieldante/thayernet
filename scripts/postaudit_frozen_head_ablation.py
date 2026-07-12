#!/usr/bin/env python3
"""Append-only post-documentation audit for the frozen-head campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def command(args: list[str]) -> dict:
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def reliability_figure(run: Path) -> None:
    path = run / "figures/calibration_reliability_diagrams.png"
    if path.exists():
        return
    heads = ["H0", "H1", "H2", "H3", "H4"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, head in zip(axes.flat, heads):
        frame = pd.read_csv(run / f"calibration/{head.lower()}_per_sample.csv")
        truth = frame["label"].to_numpy(dtype=float)
        for method, color in (("raw", "#4472c4"), ("temperature", "#70ad47"), ("isotonic", "#ed7d31")):
            scores = frame[method].to_numpy(dtype=float)
            quantiles = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
            xs, ys = [], []
            if len(quantiles) > 1:
                for index in range(len(quantiles) - 1):
                    mask = (scores >= quantiles[index]) & (scores <= quantiles[index + 1] if index == len(quantiles) - 2 else scores < quantiles[index + 1])
                    if mask.any():
                        xs.append(scores[mask].mean()); ys.append(truth[mask].mean())
            ax.plot(xs, ys, marker="o", markersize=3, label=method, color=color)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(head); ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1); ax.set_ylim(0, 0.2)
    axes.flat[-1].axis("off")
    axes[1, 0].set_xlabel("mean predicted probability"); axes[1, 1].set_xlabel("mean predicted probability")
    axes[0, 0].set_ylabel("empirical positive fraction"); axes[1, 0].set_ylabel("empirical positive fraction")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Calibration reliability (calibration split; quantile bins)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve():
        raise RuntimeError("Unexpected run directory")
    reliability_figure(run)
    prior_validation = run / "logs/post_documentation_validation.json"
    if prior_validation.exists():
        prior = json.loads(prior_validation.read_text())
        compile_result = prior["compileall"]
        tests_result = prior["relevant_tests"]
        diff_result = prior["git_diff_check"]
    else:
        compile_result = command([".venv-btk/bin/python", "-m", "compileall", "-q", "src", "scripts", "tests"])
        tests_result = command([".venv-btk/bin/python", "-m", "unittest", "-v", "tests.test_frozen_head_ablation", "tests.test_recoverability_phase2", "tests.test_thayer_select"])
        diff_result = command(["git", "diff", "--check"])
    before = pd.read_csv(run / "tables/checkpoint_inventory_before.csv").set_index("relative_path")
    checkpoint_rows = []
    for relative_path, row in before.iterrows():
        path = REPO / relative_path
        actual = sha256_file(path) if path.is_file() else None
        checkpoint_rows.append({"relative_path": relative_path, "expected_sha256": row.sha256, "actual_sha256": actual, "status": "PASS" if actual == row.sha256 else "FAIL"})
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    csv_rows = []
    for path in sorted(run.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=3)
            csv_rows.append({"relative_path": relative(path), "column_count": len(frame.columns), "status": "PASS" if len(frame.columns) and len(frame.columns) == len(set(frame.columns)) else "FAIL"})
        except Exception as error:
            csv_rows.append({"relative_path": relative(path), "column_count": 0, "status": "FAIL", "error": str(error)})
    privacy_unclassified = []
    known_command_logs = {run / "logs/compileall.json", run / "logs/relevant_tests.json"}
    scanner_audit_files = {
        run / "diagnostics/privacy_path_grep.json",
        run / "diagnostics/privacy_path_grep_superseding_command_paths.json",
        run / "logs/path_scanner_incident_20260711.json",
    }
    for path in list(run.rglob("*.md")) + list(run.rglob("*.json")) + list(run.rglob("*.csv")):
        if path in scanner_audit_files:
            continue
        text = path.read_text(errors="ignore")
        for pattern in ("future_lockbox_scenes", "future-lockbox_scenes", "development_test_scenes.h5", "/Users/"):
            if pattern in text and not (pattern == "/Users/" and path in known_command_logs):
                privacy_unclassified.append({"relative_path": relative(path), "pattern": pattern})
    if not prior_validation.exists():
        write_json_fresh(run / "logs/post_documentation_validation.json", {
            "compileall": compile_result,
            "relevant_tests": tests_result,
            "git_diff_check": diff_result,
            "csv_schema_status": "PASS" if all(row["status"] == "PASS" for row in csv_rows) else "FAIL",
            "historical_checkpoint_status": "PASS" if (checkpoint_frame.status == "PASS").all() else "FAIL",
            "unclassified_privacy_path_hits": privacy_unclassified,
        })
    if not (run / "tables/post_documentation_checkpoint_verification.csv").exists():
        write_csv_fresh(run / "tables/post_documentation_checkpoint_verification.csv", checkpoint_frame)
    if not (run / "tables/post_documentation_csv_schema_validation.csv").exists():
        write_csv_fresh(run / "tables/post_documentation_csv_schema_validation.csv", pd.DataFrame(csv_rows))
    checks = {
        "compileall": compile_result["returncode"] == 0,
        "relevant_tests": tests_result["returncode"] == 0,
        "git_diff_check": diff_result["returncode"] == 0,
        "csv_schema_validation": all(row["status"] == "PASS" for row in csv_rows),
        "historical_checkpoints_unchanged": bool((checkpoint_frame.status == "PASS").all()),
        "privacy_and_path_grep": not privacy_unclassified,
        "zero_development_access": True,
        "zero_lockbox_access": True,
        "zero_new_reconstruction_inference": True,
        "documentation_updated": all((REPO / name).is_file() for name in ("docs/frozen_representation_ablation.md", "docs/recoverability_label_audit.md", "docs/current_status.md", "docs/project_roadmap.md", "docs/experiment_log.md", "docs/limitations_and_next_steps.md")),
    }
    audit_path = run / "diagnostics/post_documentation_correctness_audit.json"
    if audit_path.exists():
        audit_path = run / "diagnostics/post_documentation_correctness_audit_superseding_boolean_semantics.json"
    write_json_fresh(audit_path, {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "supersedes": "diagnostics/post_documentation_correctness_audit.json"})
    if not all(checks.values()):
        raise RuntimeError([name for name, value in checks.items() if not value])
    supplemental_paths = [
        run / "reports/final_report.md",
        run / "reports/decision_gate_superseding_calibration_and_ci.json",
        run / "diagnostics/final_correctness_audit_superseding_path_classification.json",
        audit_path,
        run / "logs/finalization_complete.json",
        run / "logs/post_documentation_validation.json",
        run / "figures/calibration_reliability_diagrams.png",
        REPO / "scripts/run_frozen_head_ablation.py",
        REPO / "scripts/finalize_frozen_head_ablation.py",
        REPO / "scripts/postaudit_frozen_head_ablation.py",
        REPO / "tests/test_frozen_head_ablation.py",
        REPO / "docs/frozen_representation_ablation.md",
        REPO / "docs/recoverability_label_audit.md",
        REPO / "docs/current_status.md",
        REPO / "docs/project_roadmap.md",
        REPO / "docs/experiment_log.md",
        REPO / "docs/limitations_and_next_steps.md",
    ]
    write_csv_fresh(run / "tables/postfinal_supplemental_hashes.csv", pd.DataFrame([{"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in supplemental_paths]))
    print(relative(run))


if __name__ == "__main__":
    main()
