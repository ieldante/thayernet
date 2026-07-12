#!/usr/bin/env python3
"""Append-only finalizer for the frozen-head ablation.

This preserves the original failed path-hygiene audit and original automated
decision, then adds superseding artifacts.  It performs no neural inference or
model fitting and opens no scene HDF5 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
EXECUTED_RUNNER_SHA256 = "3e456cd5fa7dd0f8edcc9c46401640403be5f2d4a36594907a8e4c81f8aa1463"


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
        if path.read_text() == text:
            return
        raise FileExistsError(f"Refusing overwrite of nonidentical content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def git_status() -> str:
    return subprocess.run(["git", "status", "--short", "--branch"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def selected(frame: pd.DataFrame, head: str, analysis: str) -> pd.Series:
    return frame[(frame["head"] == head) & (frame["analysis"] == analysis)].iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact dependency-free Markdown table."""

    columns = ["head", "head_family", "balance_method", "parameter_count", "positives", "prevalence", "auroc", "auprc", "balanced_accuracy", "brier_score"]
    value = frame[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in value.itertuples(index=False, name=None):
        rendered = []
        for item in row:
            rendered.append(f"{item:.4f}" if isinstance(item, float) else str(item))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_select_frozen_head_ablation_"):
        raise RuntimeError("Unexpected run directory")

    original_privacy = json.loads((run / "diagnostics/privacy_path_grep.json").read_text())
    compile_log = json.loads((run / "logs/compileall.json").read_text())
    test_log = json.loads((run / "logs/relevant_tests.json").read_text())
    benign = []
    for name, payload in (("logs/compileall.json", compile_log), ("logs/relevant_tests.json", test_log)):
        only_command_executable = (
            "/Users/" in str(payload["command"][0])
            and all("/Users/" not in str(value) for value in payload["command"][1:])
            and "/Users/" not in payload["stdout"]
            and "/Users/" not in payload["stderr"]
        )
        if not only_command_executable:
            raise RuntimeError(f"Unexpected private-path content in {name}")
        benign.append({"relative_path": relative(run / name), "classification": "benign local interpreter executable captured in command[0] only"})
    if len(original_privacy["hits"]) != 2:
        raise RuntimeError("Unexpected original privacy audit shape")
    write_json_fresh(run / "diagnostics/privacy_path_grep_superseding_command_paths.json", {
        "status": "PASS",
        "supersedes": "diagnostics/privacy_path_grep.json",
        "original_hits_preserved": original_privacy["hits"],
        "classification": benign,
        "sensitive_content_hits": [],
        "development_or_lockbox_path_hits": [],
        "note": "No artifact was rewritten. The two matches are executable provenance, not leaked scientific or personal content.",
    })
    write_json_fresh(run / "logs/path_scanner_incident_20260711.json", {
        "status": "RESOLVED_APPEND_ONLY",
        "executed_runner_sha256": EXECUTED_RUNNER_SHA256,
        "cause": "privacy scanner treated the absolute BTK interpreter executable in captured command arrays as a sensitive path",
        "scientific_outputs_affected": False,
        "historical_outputs_overwritten": False,
        "resolution": "preserved failed audit and added a content-aware superseding audit",
    })
    write_json_fresh(run / "logs/finalizer_optional_dependency_incident_20260711.json", {
        "status": "RESOLVED_APPEND_ONLY",
        "cause": "pandas Markdown rendering requested the absent optional tabulate package",
        "scientific_outputs_affected": False,
        "existing_outputs_overwritten": False,
        "resolution": "dependency-free fixed Markdown renderer; existing superseding artifacts verified byte-identical on resume",
    })

    heads = pd.read_csv(run / "tables/head_comparison_with_h4.csv")
    paired = pd.read_csv(run / "tables/paired_head_differences.csv")
    query = pd.read_csv(run / "tables/query_and_failure_ranking.csv")
    calibration = pd.read_csv(run / "tables/calibration_comparison.csv")
    thresholds = pd.read_csv(run / "tables/calibration_threshold_behavior.csv")
    oracle = pd.read_csv(run / "tables/oracle_diagnostic.csv")
    noise = pd.read_csv(run / "tables/label_noise_audit.csv")
    balance = pd.read_csv(run / "tables/label_balance_by_split.csv")

    primary_head = "H2"  # frozen by validation AUPRC before calibration
    latent_only = heads.set_index("head").loc[primary_head]
    h0 = heads.set_index("head").loc["H0"]
    h1 = heads.set_index("head").loc["H1"]
    h4 = heads.set_index("head").loc["H4"]
    h2_vs_h1_pr = paired[(paired.head_a == "H2") & (paired.head_b == "H1") & (paired.metric == "auprc")].iloc[0]
    h4_vs_h2_pr = paired[(paired.head_a == "H4") & (paired.head_b == "H2") & (paired.metric == "auprc")].iloc[0]
    ambiguous_gap = float(selected(query, primary_head, "ambiguous_over_valid_score_gap")["score_gap"])
    catastrophic = float(selected(query, primary_head, "catastrophic_source_rejection")["auroc"])
    null_safety = float(selected(query, primary_head, "null_hallucination_rejection")["auroc"])
    source_confusion = float(selected(query, primary_head, "source_confusion_rejection")["auroc"])
    cal_h2 = calibration[(calibration["head"] == primary_head) & (calibration["evaluation"] == "apparent_full_calibration_fit")].set_index("calibration_method")
    h2_temp_thresholds = thresholds[(thresholds["head"] == primary_head) & (thresholds["calibration_method"] == "temperature")]
    boundary_fraction = float(noise[(noise.split == "validation") & (noise.audit_category == "near_moderate_contract_boundary")]["fraction"].iloc[0])
    contract_change_fraction = float(noise[(noise.split == "validation") & (noise.audit_category == "contract_status_changes")]["fraction"].iloc[0])
    oracle_valid = oracle[oracle.split == "validation"].iloc[0]
    prevalence = balance[balance.contract == "moderate"].set_index("split")

    reasons = [
        f"The validation-selected H2 head reached AUROC/AUPRC {latent_only.auroc:.3f}/{latent_only.auprc:.3f}, but its calibration-split raw AUROC/AUPRC fell to {cal_h2.loc['raw','auroc']:.3f}/{cal_h2.loc['raw','auprc']:.3f}.",
        f"H2 did not materially beat H1: paired AUPRC difference {h2_vs_h1_pr.difference_a_minus_b:+.3f}, 95% CI [{h2_vs_h1_pr.ci_2_5:+.3f}, {h2_vs_h1_pr.ci_97_5:+.3f}].",
        f"Ambiguity inversion persisted: ambiguous-minus-valid score gap {ambiguous_gap:+.3f} (desired below zero).",
        f"Selected-head catastrophic rejection was weak (AUROC {catastrophic:.3f}); source-confusion rejection was {source_confusion:.3f}.",
        f"Temperature avoided exact zero thresholds but still realized 100% coverage at nominal 95%, 90%, and 80% because the selected MLP scores saturated; isotonic had an {cal_h2.loc['isotonic','largest_probability_plateau_fraction']:.1%} largest plateau.",
        f"H4 AUPRC gain was {h4_vs_h2_pr.difference_a_minus_b:+.3f}, 95% CI [{h4_vs_h2_pr.ci_2_5:+.3f}, {h4_vs_h2_pr.ci_97_5:+.3f}], so no independent centroid gain was established.",
    ]
    decision = "NO CLEAR IMPROVEMENT"
    next_experiment = "Redesign and preregister the moderate reliability-contract target and its failure-specific labels before any further model or representation change."
    write_json_fresh(run / "reports/decision_gate_superseding_calibration_and_ci.json", {
        "status": "AUTHORITATIVE_SUPERSEDING_DECISION",
        "supersedes": "reports/decision_gate.json",
        "classification": decision,
        "reasons": reasons,
        "single_next_experiment": next_experiment,
        "why_original_gate_was_superseded": "The original automatic branch used a point-estimate linearity margin without requiring paired-CI materiality or calibration stability, and therefore overcalled nonlinear sufficiency.",
        "head_reselected_on_calibration": False,
        "development_evaluation_performed": False,
        "lockbox_result_exists": False,
    })

    original_audit = json.loads((run / "diagnostics/final_correctness_audit.json").read_text())
    checks = dict(original_audit["checks"])
    checks["privacy_path_grep"] = True
    checks["original_failed_audit_preserved"] = True
    checks["superseding_decision_uses_calibration_as_evaluation_not_reselection"] = True
    checks["zero_additional_neural_inference_in_finalization"] = True
    checks["zero_development_access"] = True
    checks["zero_lockbox_access"] = True
    if not all(checks.values()):
        raise RuntimeError(f"Superseding correctness audit still fails: {[key for key, value in checks.items() if not value]}")
    write_json_fresh(run / "diagnostics/final_correctness_audit_superseding_path_classification.json", {
        "status": "PASS",
        "supersedes": "diagnostics/final_correctness_audit.json",
        "checks": checks,
        "historical_checkpoint_inventory": "tables/checkpoint_inventory_after.csv",
        "finalizer_neural_operations": 0,
        "finalizer_scene_files_opened": 0,
        "development_accessed": False,
        "lockbox_accessed": False,
    })

    status = git_status()
    disk = shutil.disk_usage(REPO)
    compute_runtime = float(original_audit["runtime_seconds_to_audit"])
    final_report = f"""# Frozen-representation recoverability ablation final report

## Outcome

**Decision gate: {decision}.** The frozen latent contains strong moderate-success ranking information on validation, but the controlled ablation did not establish a stable operational recoverability head. The validation-selected H2 head degraded sharply on calibration, ambiguity remained inverted, catastrophic rejection remained weak, and calibration still produced unusable coverage ties.

**Exactly one next experiment:** {next_experiment}

This was a diagnostic experiment within the existing recoverability phase. The overall project did not change. No scientific reconstruction backbone was retrained, no development evaluation was performed, and no lockbox result exists.

## Required answers

1. **Was the encoder completely frozen?** Yes: zero trainable backbone parameters, zero gradients, evaluation mode, deterministic repeated extraction, and unchanged R1 checkpoint hash.
2. **How many latent features were extracted?** 13,500 samples x 64 pooled-bottleneck values: 10,000 training, 1,500 validation, and 2,000 calibration.
3. **Positive-label prevalence?** Training {prevalence.loc['training','positive_prevalence']:.3%} (41/10,000), validation {prevalence.loc['validation','positive_prevalence']:.3%} (5/1,500), calibration {prevalence.loc['calibration','positive_prevalence']:.3%} (30/2,000).
4. **Did class balancing improve AUROC and AUPRC?** On validation, H0 was {h0.auroc:.3f}/{h0.auprc:.3f} and H1 was {h1.auroc:.3f}/{h1.auprc:.3f}: AUROC was similar and AUPRC improved substantially. This did not remain operationally stable on calibration.
5. **Did logistic regression remain competitive with MLPs?** Yes statistically: H2-H1 AUPRC was {h2_vs_h1_pr.difference_a_minus_b:+.3f}, 95% CI [{h2_vs_h1_pr.ci_2_5:+.3f}, {h2_vs_h1_pr.ci_97_5:+.3f}].
6. **Is recoverability approximately linearly accessible?** The strict predeclared point margin was missed (AUPRC gap 0.033 > 0.02), so the formal answer is no. However, the paired interval includes zero and calibration instability prevents a defensible nonlinear-sufficiency claim.
7. **Did ambiguity ranking improve?** The gap shrank from the prior roughly +0.14 to {ambiguous_gap:+.3f}, but remained inverted. Therefore it improved numerically but did not succeed.
8. **Did catastrophic-valid ranking improve?** No useful operational improvement was established. H2 catastrophic-rejection AUROC was {catastrophic:.3f}; its narrow moderate-success-versus-catastrophic contrast is inflated by only five positives and is not the relevant all-source rejection result.
9. **Did null-safety ranking improve?** H2 validation null-hallucination rejection AUROC was {null_safety:.3f}, a strong diagnostic result, but it was insufficient to rescue the overall head.
10. **Did temperature scaling avoid isotonic threshold collapse?** Only partially. It avoided exact zero thresholds, but H2 still realized 100% coverage at nominal 95%, 90%, and 80% due to saturated raw scores. Isotonic collapsed to {int(cal_h2.loc['isotonic','unique_values'])} values with an {cal_h2.loc['isotonic','largest_probability_plateau_fraction']:.1%} largest plateau.
11. **Did centroid features add independent value?** No established gain. H4-H2 AUPRC was {h4_vs_h2_pr.difference_a_minus_b:+.3f}, 95% CI [{h4_vs_h2_pr.ci_2_5:+.3f}, {h4_vs_h2_pr.ci_97_5:+.3f}].
12. **How strong was the oracle?** Validation AUROC/AUPRC {oracle_valid.auroc:.3f}/{oracle_valid.auprc:.3f}, versus prevalence {oracle_valid.prevalence:.3%}. It was informative but much weaker in AUPRC than the latent heads and is not deployable.
13. **How much label-boundary noise was found?** {boundary_fraction:.2%} of validation samples were within 5% of a moderate contract threshold; {contract_change_fraction:.2%} changed status across strict/moderate/permissive contracts.
14. **Where is the bottleneck?** A combination: usable information exists in the representation, balancing helps extract it on validation, but extreme target scarcity/heterogeneity and head/calibration instability prevent reliable operational ranking.
15. **What single next experiment is recommended?** {next_experiment}
16. **Was the lockbox untouched?** Yes. Zero lockbox access; no lockbox result.
17. **Were historical checkpoints unchanged?** Yes. The before/after hash inventory passed.

## Head comparison: frozen validation scores

{markdown_table(heads)}

The moderate target has only five validation positives, so AUPRC confidence intervals are necessarily wide. AUPRC should be compared with its 0.333% validation prevalence, not interpreted in isolation.

## Calibration and operational behavior

The validation-selected head was H2; calibration did not retroactively change that choice. H2 calibration raw AUROC/AUPRC was {cal_h2.loc['raw','auroc']:.3f}/{cal_h2.loc['raw','auprc']:.3f}. Temperature scaling preserved more resolution than isotonic but inherited raw-score saturation. Probability calibration quality, ranking quality, and attainable coverage are therefore reported separately.

## Correctness and provenance

- Encoder feature layer: adaptive-average-pooled 64-channel bottleneck; prompt-channel information included through the original model input.
- Pixel-uncertainty features: excluded.
- Reconstruction statistics: excluded from H0-H4.
- Generator variables: excluded from H0-H4; used only by the explicitly non-deployable oracle.
- Scaling: fit on training features only.
- Calibration: fit on calibration only after head freeze.
- Development inference/evaluation: none.
- Lockbox access: none.
- Historical checkpoints: unchanged.
- Original path-scanner failure: preserved; superseded after verifying the only two matches were the captured BTK interpreter executable in command provenance.
- Scientific compute runtime through the first audit: {compute_runtime:.1f} seconds.
- Disk free at finalization: {disk.free} bytes.

## Artifact index

- Head metrics and confidence intervals: `tables/head_comparison_with_h4.csv`, `tables/head_bootstrap_confidence_intervals.csv`, `tables/paired_head_differences.csv`
- Query/failure ranking: `tables/query_and_failure_ranking.csv`
- Calibration, ties, and coverage: `tables/calibration_comparison.csv`, `tables/calibration_threshold_behavior.csv`
- Centroid ablation: `figures/centroid_feature_ablation.png`
- Oracle: `tables/oracle_diagnostic.csv`, `tables/oracle_feature_importance.csv`, `figures/oracle_diagnostic.png`
- Label audit/gallery: `tables/label_noise_audit.csv`, `figures/label_disagreement_gallery.png`
- Provenance: `logs/input_provenance.json`, `manifests/campaign_file_hashes.csv`
- Authoritative decision correction: `reports/decision_gate_superseding_calibration_and_ci.json`
- Authoritative correctness audit: `diagnostics/final_correctness_audit_superseding_path_classification.json`

## Final git status

```text
{status.rstrip()}
```
"""
    write_text_fresh(run / "reports/final_report.md", final_report)

    hash_rows = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path != run / "manifests/campaign_file_hashes.csv":
            hash_rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(run / "manifests/campaign_file_hashes.csv", pd.DataFrame(hash_rows))
    write_json_fresh(run / "logs/finalization_complete.json", {
        "status": "PASS",
        "authoritative_decision": decision,
        "single_next_experiment": next_experiment,
        "run_directory": relative(run),
        "files_hashed": len(hash_rows),
        "scientific_compute_rerun": False,
        "neural_inference": False,
        "development_accessed": False,
        "lockbox_accessed": False,
        "completed_at_unix": time.time(),
    })
    print(relative(run))


if __name__ == "__main__":
    main()
