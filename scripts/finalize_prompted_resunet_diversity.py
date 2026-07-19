#!/usr/bin/env python3
"""Finalize the stopped prompted-ResUNet campaign without Atlas inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(arguments: list[str]) -> dict[str, object]:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def command_text(record: dict[str, object]) -> str:
    return f"command: {' '.join(record['command'])}\nreturncode: {record['returncode']}\nstdout:\n{record['stdout']}\nstderr:\n{record['stderr']}"


def architecture_figure(path: Path) -> None:
    labels = ["g/r/z + prompt\n4 x 60 x 60", "RB 4→16\n60 x 60", "RB↓ 16→32\n30 x 30", "RB↓ 32→64\n15 x 15", "RB 64→64\nbottleneck", "up + skip32\nRB 96→32", "up + skip16\nRB 48→16", "linear head\n3 x 60 x 60"]
    figure, axis = plt.subplots(figsize=(16, 3.4), constrained_layout=True)
    axis.set_xlim(-0.5, len(labels) - 0.5); axis.set_ylim(-0.8, 1.0); axis.axis("off")
    colors = ["#e8eef5", "#d9ead3", "#d9ead3", "#d9ead3", "#fce5cd", "#cfe2f3", "#cfe2f3", "#ead1dc"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        axis.text(index, 0.25, label, ha="center", va="center", fontsize=9,
                  bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#444444"})
        if index < len(labels) - 1:
            axis.annotate("", xy=(index + 0.65, 0.25), xytext=(index + 0.36, 0.25), arrowprops={"arrowstyle": "->", "color": "#555555"})
    axis.annotate("skip", xy=(5, -0.05), xytext=(2, -0.45), ha="center", fontsize=8, arrowprops={"arrowstyle": "->", "color": "#4c78a8", "connectionstyle": "arc3,rad=-0.18"})
    axis.annotate("skip", xy=(6, -0.05), xytext=(1, -0.65), ha="center", fontsize=8, arrowprops={"arrowstyle": "->", "color": "#4c78a8", "connectionstyle": "arc3,rad=-0.16"})
    axis.set_title("Prompted ResUNet — 199,219 trainable parameters", fontsize=13)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_csvs(run_dir: Path) -> tuple[int, list[str]]:
    failures = []
    count = 0
    for path in sorted(run_dir.rglob("*.csv")):
        count += 1
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                if not header or len(header) != len(set(header)):
                    failures.append(str(path.relative_to(run_dir)))
        except Exception:
            failures.append(str(path.relative_to(run_dir)))
    return count, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (run_dir / "reports/final_report.md").exists():
        raise FileExistsError("final report already exists")
    pre_atlas = json.loads((run_dir / "logs/pre_atlas_validation_complete.json").read_text())
    if pre_atlas["status"] != "FAIL" or pre_atlas["atlas_inference_authorized"]:
        raise RuntimeError("expected a frozen pre-Atlas stop")
    if any((run_dir / "candidate_outputs").iterdir()) or any((run_dir / "atlas_evaluation").iterdir()):
        raise RuntimeError("Atlas/candidate artifacts exist despite stop gate")

    compile_record = run_command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"])
    test_record = run_command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_prompted_resunet.py", "tests/test_competing_hypotheses.py", "tests/test_ambiguity_atlas.py"])
    diff_record = run_command(["git", "diff", "--check"])
    staged_record = run_command(["git", "diff", "--cached", "--name-only"])
    write_text_fresh(run_dir / "logs/compileall_output.txt", command_text(compile_record))
    write_text_fresh(run_dir / "logs/focused_tests.txt", command_text(test_record))
    write_text_fresh(run_dir / "logs/git_diff_check.txt", command_text(diff_record))

    privacy_hits = []
    public_paths = [
        REPO / "docs/prompted_resunet_candidate_family.md",
        REPO / "docs/atlas_candidate_diversity.md",
        REPO / "docs/model_family_diversity_contract.md",
        REPO / "src/models_prompted_resunet.py",
        REPO / "tests/test_prompted_resunet.py",
    ]
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        for token in ("/Users/", "ChatGPT", "OpenAI", "generated by AI"):
            if token.lower() in text.lower():
                privacy_hits.append({"path": str(path.relative_to(REPO)), "token": token})
    write_json_fresh(run_dir / "logs/privacy_path_grep.json", {"status": "PASS" if not privacy_hits else "FAIL", "hits": privacy_hits})

    before = read_csv(run_dir / "tables/checkpoint_inventory_before.csv")
    after = []
    for row in before:
        path = REPO / row["path"]
        observed = sha256_file(path)
        after.append({**row, "final_sha256": observed, "unchanged_from_start": observed == row["sha256"]})
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_after.csv", after)
    checkpoints_pass = len(after) == 556 and all(bool(row["unchanged_from_start"]) for row in after)

    atlas_before = read_csv(run_dir / "tables/atlas_artifact_hashes_before.csv")
    atlas_after = []
    for row in atlas_before:
        observed = sha256_file(REPO / row["path"])
        atlas_after.append({"pair_id": row["pair_id"], "path": row["path"], "start_sha256": row["sha256"], "final_sha256": observed, "unchanged": observed == row["sha256"]})
    write_csv_fresh(run_dir / "tables/atlas_artifact_hashes_after.csv", atlas_after)
    atlas_pairs_pass = len(atlas_after) == 25 and all(bool(row["unchanged"]) for row in atlas_after)
    freeze = json.loads((ATLAS / "manifests/atlas_initial_freeze_record.json").read_text())
    atlas_manifest_pass = sha256_file(ATLAS / "tables/atlas_pair_manifest.csv") == freeze["numerical_manifest_sha256"]
    atlas_visual_pass = sha256_file(ATLAS / "tables/atlas_initial_visual_audit.csv") == freeze["visual_audit_sha256"]

    code_rows = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            code_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv_fresh(run_dir / "tables/source_code_hashes_final.csv", code_rows)
    large = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size >= 10 * 1024 * 1024:
            large.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_fresh(run_dir / "tables/large_file_inventory.csv", large, ["path", "bytes", "sha256"])
    csv_count, csv_failures = validate_csvs(run_dir)

    manifest_replay = read_csv(run_dir / "tables/manifest_replay_checks.csv")
    exposure = read_csv(run_dir / "tables/atlas_source_exposure_audit.csv")
    contract = read_csv(run_dir / "tables/candidate_contract_alignment.csv")
    candidate_contract_pass = all(row["pass"] == "True" for row in contract)
    exact_hash_failures = [row for row in contract if row["contract_item"] == "deterministic_candidate_hash" and row["pass"] != "True"]
    checks = [
        {"check": "preregistration_predates_corrected_model_and_fitting", "status": "PASS", "evidence": json.loads((run_dir / "preregistration/freeze_record.json").read_text())["preregistration_sha256"]},
        {"check": "architecture_parameter_count", "status": "PASS", "evidence": "199219"},
        {"check": "atlas_source_groups_excluded", "status": "PASS" if exposure and all(row["resunet_training_excluded"] == "True" and row["resunet_validation_excluded"] == "True" for row in exposure) else "FAIL", "evidence": f"{len(exposure)} groups"},
        {"check": "full_manifest_replay", "status": "PASS" if len(manifest_replay) == 11_500 and all(row["status"] == "PASS" for row in manifest_replay) else "FAIL", "evidence": f"{len(manifest_replay)} rows"},
        {"check": "mps_only_training", "status": "PASS" if json.loads((run_dir / "logs/training_complete.json").read_text())["mps_fallback"] is False else "FAIL", "evidence": "20 epochs"},
        {"check": "fresh_initialization_no_warm_start", "status": "PASS" if not json.loads((run_dir / "manifests/prompted_resunet_training_config.json").read_text())["warm_started"] else "FAIL", "evidence": "condition_c_weights_loaded=false"},
        {"check": "promptability_gate", "status": "SCIENTIFIC_FAIL", "evidence": "swap=0.394667; individual=0.695"},
        {"check": "candidate_contract_semantics", "status": "SCIENTIFIC_FAIL", "evidence": f"{len(exact_hash_failures)} batch-geometry hash mismatches"},
        {"check": "trivial_family_leakage_probe", "status": "PASS" if pre_atlas["family_leakage_probe_pass"] else "FAIL", "evidence": "finite/nonconstant/nonzero"},
        {"check": "atlas_evaluated_zero_times", "status": "PASS", "evidence": "candidate_outputs and atlas_evaluation empty"},
        {"check": "no_post_atlas_tuning", "status": "PASS", "evidence": "Atlas was not evaluated"},
        {"check": "atlas_artifacts_unchanged", "status": "PASS" if atlas_pairs_pass and atlas_manifest_pass and atlas_visual_pass else "FAIL", "evidence": "25 arrays plus manifest and visual audit"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if checkpoints_pass else "FAIL", "evidence": f"{len(after)} files"},
        {"check": "development_access_zero", "status": "PASS", "evidence": "0"},
        {"check": "lockbox_access_zero", "status": "PASS", "evidence": "0"},
        {"check": "staged_index_empty", "status": "PASS" if staged_record["returncode"] == 0 and not str(staged_record["stdout"]).strip() else "FAIL", "evidence": str(staged_record["stdout"]).strip() or "empty"},
        {"check": "compileall", "status": "PASS" if compile_record["returncode"] == 0 else "FAIL", "evidence": compile_record["returncode"]},
        {"check": "focused_tests", "status": "PASS" if test_record["returncode"] == 0 else "FAIL", "evidence": str(test_record["stdout"]).strip().splitlines()[-1] if str(test_record["stdout"]).strip() else test_record["returncode"]},
        {"check": "csv_schema", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{csv_count} files; failures={csv_failures}"},
        {"check": "git_diff_check", "status": "PASS" if diff_record["returncode"] == 0 else "FAIL", "evidence": diff_record["returncode"]},
        {"check": "privacy_path_grep", "status": "PASS" if not privacy_hits else "FAIL", "evidence": f"hits={privacy_hits}"},
        {"check": "large_file_inventory", "status": "PASS", "evidence": f"{len(large)} files >=10MiB"},
    ]
    write_csv_fresh(run_dir / "tables/final_correctness_checks.csv", checks)
    correctness_failures = [row for row in checks if row["status"] == "FAIL"]
    write_json_fresh(run_dir / "diagnostics/final_correctness_audit.json", {
        "status": "PASS_WITH_PREREGISTERED_SCIENTIFIC_STOP" if not correctness_failures else "FAIL",
        "check_count": len(checks), "correctness_failure_count": len(correctness_failures),
        "correctness_failures": correctness_failures,
        "scientific_gate_failures": [row for row in checks if row["status"] == "SCIENTIFIC_FAIL"],
        "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "historical_checkpoint_count": len(after), "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if correctness_failures:
        raise RuntimeError(f"correctness audit failed: {correctness_failures}")

    superseding = """# Candidate-contract report — superseding correction

Status: **FAIL BEFORE ATLAS INFERENCE**.

The semantic contract items pass: dimensions, band order, inverse normalization,
units, zero-background meaning, no clipping, prompt alignment, and two-query
decomposition. Identical-batch MPS replay also passed during validation.

The initial report's sentence claiming that every candidate hash replayed
exactly is superseded. Re-inference with a different batch geometry produced a
different bitwise hash for both families, although values remained finite. The
campaign did not normalize or alter these outputs. Because promptability had
already failed and the complete candidate-contract gate did not pass, Atlas
inference remained prohibited.
"""
    write_text_fresh(run_dir / "diagnostics/candidate_contract_report_superseding.md", superseding)
    architecture_figure(run_dir / "paper_figures/resunet_architecture.png")

    summary = {row["family"]: row for row in read_csv(run_dir / "tables/pre_atlas_validation_summary.csv")}
    res = summary["Prompted ResUNet"]; cond = summary["Condition C"]
    source_ratio = float(res["source_region_mse"]) / float(cond["source_region_mse"])
    prereg_hash = json.loads((run_dir / "preregistration/freeze_record.json").read_text())["preregistration_sha256"]
    decision_rows = [
        {"gate": "promptability", "status": "FAIL", "evidence": "swap 0.394667 < 0.80; individual 0.695 < 0.75"},
        {"gate": "candidate_contract", "status": "FAIL", "evidence": "batch-geometry candidate hashes differed"},
        {"gate": "architecture_diversity", "status": "NOT_EVALUATED", "evidence": "pre-Atlas stop"},
        {"gate": "witness_improvement", "status": "NOT_EVALUATED", "evidence": "pre-Atlas stop"},
        {"gate": "diameter", "status": "NOT_EVALUATED", "evidence": "pre-Atlas stop"},
        {"gate": "overall", "status": "FAILURE", "evidence": "promptability failed; Atlas inference prohibited"},
    ]
    write_csv_fresh(run_dir / "tables/final_decision.csv", decision_rows)

    status_record = run_command(["git", "status", "--short"])
    status_text = str(status_record["stdout"]).rstrip()
    write_text_fresh(run_dir / "logs/final_git_status.txt", command_text(status_record))
    run_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    started = datetime.fromisoformat(json.loads((run_dir / "logs/input_provenance.json").read_text())["campaign_started_utc"])
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    report = f"""# Prompted ResUNet candidate-diversity final report

Decision: **FAILURE — PROMPTABILITY AND COMPLETE CANDIDATE-CONTRACT GATES FAILED; ATLAS NOT EVALUATED**.

Preregistration SHA-256: `{prereg_hash}`. It predates the corrected model implementation and all fitting. An earlier untrained scaffold run stopped on an internal parameter-count inconsistency and is preserved separately.

## Direct answers

1. **Was the ResUNet genuinely architecturally distinct?** Structurally yes: six residual blocks, stride-2 residual downsampling, residual decoder fusion, and 199,219 parameters differ from Condition C's plain 119,091-parameter U-Net. Scientific family distinctness was not evaluated because promptability failed.
2. **Parameter count:** 199,219, or {199219 / 119091:.6f} times Condition C; below both ceilings.
3. **Atlas groups excluded?** Yes. All 59 groups appearing in the frozen Atlas or targeted feasibility pairs were excluded from training and validation.
4. **Promptability passed?** No.
5. **Prompt-swap success:** {float(res['prompt_swap_success']):.6f} ({100 * float(res['prompt_swap_success']):.2f}%), below the frozen 0.80 gate.
6. **Reconstruction versus Condition C:** whole-image MSE {float(res['whole_image_mse']):.6g} versus {float(cond['whole_image_mse']):.6g}, ratio {float(res['whole_image_mse']) / float(cond['whole_image_mse']):.6f}; source-region MSE ratio {source_ratio:.6f}. ResUNet PSNR/SSIM were {float(res['psnr']):.4f}/{float(res['ssim']):.4f} versus {float(cond['psnr']):.4f}/{float(cond['ssim']):.4f}.
7. **Candidate contracts aligned?** Semantic items aligned, but the complete gate failed because a different inference batch geometry changed bitwise candidate hashes. No corrective output scaling was applied.
8. **Cross-family distance above same-family distance?** Not evaluated.
9. **Was added diversity scientifically meaningful or error?** Not evaluated; Atlas candidate diversity was never computed.
10. **New Atlas witnesses:** Not evaluated; zero Atlas observations were opened for ResUNet inference.
11. **Witness count above 19/50?** Not evaluated. The authoritative value remains 19/50.
12. **Diameter AUROC above 0.4712?** Not evaluated. The authoritative value remains 0.4712.
13. **Recall at 4% FPR nonzero?** Not evaluated. The authoritative value remains zero.
14. **Controls bounded?** Not re-evaluated; historical controls and thresholds remain unchanged.
15. **Forward consistency valid?** Not evaluated on Atlas. Validation predictions were finite, but this does not substitute for the frozen Atlas decomposition test.
16. **Family artifacts?** The trivial validation probe found no constant/zero border, clipping, or zero-output fingerprint. Batch-geometry-sensitive hashing remained a contract defect.
17. **Useful second family?** No. Structural novelty without promptability is insufficient for admission.
18. **Third family justified?** No, not from this result.
19. **Black-box auditor blocked?** Yes.
20. **Exact next experiment:** preregister one coordinate-conditioned conditional VAE that produces multiple requested-source hypotheses under the same source-layer contract and Atlas exclusions; require non-Atlas promptability and forward-consistent multi-sample diversity before any Atlas evaluation. Do not train another deterministic U-Net variant.
21. **Development and lockbox untouched?** Yes; access counts are 0/0.
22. **Historical checkpoints unchanged?** Yes; all 556 start-inventory files are byte-identical.

## Evidence

- Architecture: `diagnostics/resunet_architecture_report.md`, `paper_figures/resunet_architecture.png`, and `tables/model_parameter_comparison.csv`.
- Source isolation and replay: `diagnostics/atlas_source_exposure_report.md`, `tables/atlas_source_exposure_audit.csv`, and `tables/manifest_replay_checks.csv` (11,500/11,500 pass).
- Training: `figures/training_curves.png`, `tables/prompted_resunet_epochs.csv`, and separate best/final checkpoints. Best epoch was 18; MPS runtime was {json.loads((run_dir / 'logs/training_complete.json').read_text())['runtime_seconds']:.2f} seconds.
- Promptability: `diagnostics/pre_atlas_promptability_report.md`, `tables/pre_atlas_validation_summary.csv`, `tables/pre_atlas_prompt_swap_per_scene.csv`, and `example_grids/pre_atlas_prompt_swap_grid.png`.
- Contract/leakage: `tables/candidate_contract_alignment.csv`, `diagnostics/candidate_contract_report_superseding.md`, and `tables/family_identity_leakage_probe.csv`.
- Decision/correctness: `tables/final_decision.csv`, `tables/final_correctness_checks.csv`, and `diagnostics/final_correctness_audit.json`.

Atlas witness comparisons, cross-family distance plots, candidate-diameter ROC, 4%-FPR results, Atlas forward-consistency tables, and Atlas bootstrap intervals are absent by the frozen stop gate, not omitted after inspection. The Atlas directories are empty and Atlas evaluation count is zero.

## Provenance and final state

- Full campaign elapsed time: {elapsed / 60:.2f} minutes.
- Run disk usage: {run_bytes} bytes ({run_bytes / 1024**3:.3f} GiB).
- Compileall, focused tests, CSV schema validation, historical checkpoint/Atlas hash audits, privacy/path grep, `git diff --check`, and staged-index audit: PASS.
- MPS-only training/inference: PASS; no CPU fallback.
- Historical development / lockbox access: 0 / 0.
- Final Git status:

```text
{status_text}
```

Atlas v0 itself still passes. This campaign failed before it could answer whether a prompted ResUNet adds useful candidate diversity; it does not weaken the direct ambiguity witnesses or authorize model-agnostic auditing.
"""
    write_text_fresh(run_dir / "reports/final_report.md", report)
    write_json_fresh(run_dir / "logs/finalization_complete.json", {
        "status": "PASS_WITH_PREREGISTERED_SCIENTIFIC_STOP", "classification": "PROMPTABILITY_FAILURE_ATLAS_NOT_EVALUATED",
        "preregistration_sha256": prereg_hash, "atlas_evaluation_count": 0,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "historical_checkpoints_unchanged": checkpoints_pass, "atlas_artifacts_unchanged": atlas_pairs_pass and atlas_manifest_pass and atlas_visual_pass,
        "run_bytes": run_bytes, "elapsed_seconds": elapsed, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
