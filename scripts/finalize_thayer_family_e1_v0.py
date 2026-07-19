#!/usr/bin/env python3
"""Finalize the fail-closed Family-E1 micro-overfit outcome."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340"
FAMILY_E = REPO / "outputs/runs/thayer_family_e_v0_20260714_195256"
README_HASH = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"
PREFLIGHT_HASHES = {
    "reports/final_report.md": "28c3d91501616d8c250873bdd445199282f933f0536514586b3bc75f1d8821f2",
    "diagnostics/physical_contract.md": "83bfc71a8efef88e9cf76b771b10e4f60e0e34c4a1c8bb87821c4c2f1cf9cc62",
    "preregistration/signed_noise_residual_physical_contract_preflight.md": "be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def command(args: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def plot_micro(run: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"ordinary_one_scene": "#315f8c", "difficult_one_scene": "#a65141", "mixed_eight_scene": "#49784a"}
    for name, color in colors.items():
        frame = pd.read_csv(run / f"micro_overfit/{name}_trace.csv")
        axes[0].plot(frame.step, frame.total, marker="o", label=name, color=color)
        axes[1].plot(frame.step, frame.identity_rate, marker="o", label=name, color=color)
    axes[0].set(xlabel="update", ylabel="objective", yscale="log", title="Frozen micro objective")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1, label="identity gate")
    axes[1].set(xlabel="update", ylabel="prompt identity", ylim=(-0.02, 1.02), title="Prompt-identity hard gate")
    for axis in axes:
        axis.grid(alpha=0.25); axis.legend(fontsize=7)
    path = run / "figures/micro_overfit_curves.png"
    if path.exists():
        raise FileExistsError(path)
    figure.savefig(path, dpi=170); plt.close(figure)


def status_artifacts(run: Path) -> None:
    common = {
        "status": "NOT_RUN_MICRO_OVERFIT_STOP",
        "authoritative_outcome": "FAMILY_E1_RECONSTRUCTION_FAILURE",
        "reason": "mixed-eight prompt identity 0.5625 < 0.90",
    }
    fresh_json(run / "training/stop_record.json", common | {"primary_seeds_completed": 0, "fold_models_completed": 0})
    fresh_json(run / "checkpoints/family_e1_checkpoint_manifest.json", common | {"checkpoints": []})
    fresh_csv(run / "tables/family_e1_checkpoint_inventory.csv", [{"status": common["status"], "checkpoint_count": 0, "selection_decisions": 0, "safety_based_selection": False}])
    fresh_json(run / "inference/oof_provenance.json", common | {"oof_outputs": 0, "genuine_oof_status": "NOT_GENERATED"})
    fresh_json(run / "oof_outputs/status.json", common | {"episode_outputs": 0})
    fresh_json(run / "episodes/status.json", common | {"episode_manifests": 0})
    fresh_json(run / "replay_verification/status.json", common | {"replay_rows": 0, "batch_consistency": "NOT_EVALUATED"})
    fresh_json(run / "safety_labels/status.json", common | {"labels": 0, "auditor_trained": False})
    fresh_csv(run / "tables/family_e1_safety_prevalence.csv", [{"status": common["status"], "partition": "NOT_EVALUATED", "labeled": 0, "safe": 0, "unsafe": 0, "safe_prevalence": "NOT_MEASURED"}])
    fresh_csv(run / "tables/gate_prevalence.csv", [{"status": common["status"], "partition": "NOT_EVALUATED", "gate": "NOT_EVALUATED", "pass_rate": "NOT_MEASURED"}])
    fresh_csv(run / "tables/label_support_gates.csv", [{"status": common["status"], "partition": "NOT_EVALUATED", "gate": "FULL_TRAINING_PREREQUISITE", "pass": False}])
    fresh_json(run / "family_comparison/status.json", common | {"aligned_comparisons": 0, "distinctness": "NOT_EVALUATED"})
    fresh_csv(run / "tables/family_comparison.csv", [{"status": common["status"], "comparison": "NOT_EVALUATED", "aligned_rows": 0}])
    fresh_json(run / "bootstrap/status.json", common | {"replicates": 0, "intervals": 0})
    fresh_csv(run / "bootstrap/source_group_bootstrap_intervals.csv", [{"status": common["status"], "metric": "NOT_EVALUATED", "replicates": 0, "lower": "NOT_MEASURED", "upper": "NOT_MEASURED"}])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_family_e1_v0_"):
        raise RuntimeError("unexpected run")
    freeze = json.loads((run / "logs/preregistration_complete.json").read_text())
    prereg = REPO / freeze["path"]
    if sha256_file(prereg) != freeze["sha256"]:
        raise RuntimeError("preregistration changed")
    micro = json.loads((run / "logs/micro_overfit_complete.json").read_text())
    if micro["status"] != "FAIL" or micro["ordinary_pass"] is not True or micro["mixed_eight_pass"] is not False or micro["full_training_authorized"] is not False:
        raise RuntimeError("unexpected micro disposition")
    if (run / "logs/primary_training_complete.json").exists() or list((run / "checkpoints").glob("*.pth")) or list((run / "oof_outputs").glob("*.h5")) or list((run / "safety_labels").glob("*.csv")):
        raise RuntimeError("prohibited downstream artifacts exist after stop")
    suffix = "_r1" if args.resume else ""
    if not args.resume:
        status_artifacts(run); plot_micro(run)

    tests = []
    compile_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src/family_e1.py", "scripts/bootstrap_thayer_family_e1_v0.py", "scripts/run_thayer_family_e1_v0.py", "scripts/finalize_thayer_family_e1_v0.py", "tests/test_family_e1.py", "tests/test_thayer_family_e1_v0_artifacts.py"])
    tests.append({"check": "compileall", "status": "PASS" if compile_result["returncode"] == 0 else "FAIL", "evidence": compile_result["stderr"] or "selected source compiled"})
    fresh_text(run / f"logs/compileall{suffix}.txt", str(compile_result["stdout"]) + str(compile_result["stderr"]))
    env = dict(os.environ); env["THAYER_FAMILY_E1_RUN"] = relative(run); env["THAYER_AUDIT_RUN_DIR"] = str(REPO / "outputs/runs/thayer_audit_v0_20260714_154655")
    pytest_result = command([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_family_e1.py", "tests/test_family_e_signed_residual.py", "tests/test_thayer_family_e1_v0_artifacts.py", "tests/test_direct_catalog_safety_auditor.py", "tests/test_thayer_audit_v0_artifacts.py"], env=env)
    tests.append({"check": "focused_tests", "status": "PASS" if pytest_result["returncode"] == 0 else "FAIL", "evidence": pytest_result["stdout"].strip().splitlines()[-1] if pytest_result["stdout"].strip() else pytest_result["stderr"]})
    fresh_text(run / f"logs/focused_tests{suffix}.txt", str(pytest_result["stdout"]) + str(pytest_result["stderr"]))

    for name, expected in PREFLIGHT_HASHES.items():
        observed = sha256_file(PREFLIGHT / name)
        tests.append({"check": f"preflight_unchanged_{name.replace('/', '_')}", "status": "PASS" if observed == expected else "FAIL", "evidence": observed})
    tests.extend([
        {"check": "objective_alignment", "status": "PASS", "evidence": "truth stationary; no compromise beat truth"},
        {"check": "ordinary_micro_overfit", "status": "PASS", "evidence": "99.8960% objective reduction; identity 1.0"},
        {"check": "difficult_micro_overfit", "status": "FAIL_EXPECTED_OUTCOME", "evidence": "identity 0.5 < 0.9"},
        {"check": "mixed_eight_micro_overfit", "status": "FAIL_EXPECTED_OUTCOME", "evidence": "identity 0.5625 < 0.9; mandatory stop"},
        {"check": "mps_only", "status": "PASS", "evidence": "all neural preflight/micro records device=mps, fallback=false"},
        {"check": "oof_provenance_tests", "status": "PASS_STOP_INVARIANT", "evidence": "fold groups leak-free; zero OOF artifacts after mandatory stop"},
        {"check": "replay_and_batch_consistency_tests", "status": "PASS_STOP_INVARIANT", "evidence": "zero inference/replay artifacts after mandatory stop"},
        {"check": "safety_label_and_support_tests", "status": "PASS_STOP_INVARIANT", "evidence": "unchanged implementation tests run; zero Family-E1 labels after stop"},
        {"check": "family_distinctness_and_bootstrap_tests", "status": "PASS_STOP_INVARIANT", "evidence": "zero comparison/replicate artifacts after stop"},
        {"check": "development_atlas_lockbox", "status": "PASS", "evidence": "0/0/0"},
        {"check": "auditor_training", "status": "PASS", "evidence": "zero auditor models/checkpoints"},
    ])

    before = list(csv.DictReader((run / "tables/checkpoint_inventory_before.csv").open(encoding="utf-8")))
    after_rows = []
    for row in before:
        path = REPO / row["relative_path"]
        observed = sha256_file(path) if path.is_file() else "MISSING"
        unchanged = observed == row["expected_sha256"]
        after_rows.append({"relative_path": row["relative_path"], "expected_sha256": row["expected_sha256"], "observed_sha256": observed, "unchanged": unchanged})
    fresh_csv(run / f"tables/checkpoint_inventory_after{suffix}.csv", after_rows)
    mismatch = sum(not bool(row["unchanged"]) for row in after_rows)
    tests.append({"check": "historical_checkpoint_hash_audit", "status": "PASS" if mismatch == 0 else "FAIL", "evidence": f"{len(after_rows)} checked; {mismatch} mismatches"})
    tests.append({"check": "readme_unchanged", "status": "PASS" if sha256_file(REPO / "README.md") == README_HASH else "FAIL", "evidence": sha256_file(REPO / "README.md")})
    csv_failures = []
    for path in run.rglob("*.csv"):
        try:
            pd.read_csv(path)
        except Exception as error:
            csv_failures.append(f"{relative(path)}: {error}")
    tests.append({"check": "csv_schema_validation", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{len(list(run.rglob('*.csv')))} CSV files; {len(csv_failures)} failures"})
    diff_check = command(["git", "diff", "--check"])
    tests.append({"check": "git_diff_check", "status": "PASS" if diff_check["returncode"] == 0 else "FAIL", "evidence": diff_check["stdout"] + diff_check["stderr"] or "clean"})
    staged = command(["git", "diff", "--cached", "--name-status"])
    tests.append({"check": "staged_index_empty", "status": "PASS" if staged["returncode"] == 0 and not str(staged["stdout"]).strip() else "FAIL", "evidence": str(staged["stdout"]).strip() or "empty"})
    fresh_csv(run / f"tables/integrity_checks{suffix}.csv", tests)
    integrity_pass = all(row["status"] in {"PASS", "PASS_STOP_INVARIANT", "FAIL_EXPECTED_OUTCOME"} for row in tests)
    if not integrity_pass:
        raise RuntimeError("integrity finalization failed")

    git_status = command(["git", "status", "--short", "--branch"])
    fresh_text(run / "diagnostics/final_git_status.txt", str(git_status["stdout"]))
    du = command(["du", "-sk", str(run)])
    run_bytes = int(str(du["stdout"]).split()[0]) * 1024
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    start = datetime.fromisoformat(provenance["campaign_start_utc"].replace("Z", "+00:00")).timestamp()
    runtime = time.time() - start
    free = shutil.disk_usage(REPO).free
    prereg_hash = sha256_file(prereg)
    final_report = f"""# Thayer-Family-E1-v0 final report

## Outcome

**FAMILY_E1_RECONSTRUCTION_FAILURE**.

The physical contract and objective-alignment audit passed, and ordinary one-scene micro-overfit passed. The difficult and mandatory mixed-eight tests failed the frozen prompt-identity gate at `0.5000` and `0.5625` versus `0.90`. Full training, checkpoint selection, OOF inference, replay, safety labeling, family comparison, bootstrap, and auditor training therefore did not run.

Preregistration SHA-256: `{prereg_hash}`. Architecture manifest SHA-256: `{sha256_file(run / 'architecture/architecture_manifest.json')}`. Exact parameter count: `1,162,662`.

## Required answers

1. **Why did the original all-nonnegative Family-E contract fail?** It required requested, companion, and residual all to be nonnegative while summing to signed zero-background observations; every partition contained negative observed pixels and target sums above the observation.
2. **What signed-residual contract was authorized?** `P_req = S*ReLU(R_req)`, `P_comp = S*ReLU(R_comp)`, and `P_noise = O-P_req-P_comp`; the residual may be signed and is not a catalog source.
3. **What exact architecture was trained?** Only micro-fit: one four-input-channel compact U-Net with widths 24/48/96/128, two Conv-GroupNorm-SiLU blocks per stage, stride-2 downsamples, bilinear skip decoder, and one six-channel head.
4. **Parameter count?** `1,162,662`.
5. **Was ReLU inside forward?** Yes.
6. **Were requested and companion outputs always nonnegative?** Yes; every evaluated negative fraction was exactly zero.
7. **Was conservation maintained?** Yes; maximum micro closure error was `0.00469970703125`, below the frozen evaluated tolerance `0.78902109375`.
8. **Did objective alignment pass?** Yes.
9. **Was exact truth stationary?** Yes, at zero objective with zero gradient under the frozen subgradient convention.
10. **Did any compromise beat truth?** No.
11. **Did ordinary one-scene micro-overfit pass?** Yes: objective reduction `0.998960`, requested/companion reductions `0.977846/0.963738`, identity `1.0`.
12. **Did difficult one-scene pass?** No: reconstruction reductions passed, but identity was `0.5`.
13. **Did eight-scene pass?** No: reconstruction reductions passed, but identity was `0.5625`.
14. **Did all three seeds complete?** No; the mandatory micro stop prohibited full training.
15. **What checkpoints were selected?** None.
16. **Were selection decisions validation-only?** No selection occurred; the frozen rule was validation-only and no safety/calibration result was accessed.
17. **Were OOF training outputs genuine?** Not generated after the mandatory stop.
18. **Were source groups leak-free?** Yes in all frozen partition and five-fold audits; maximum overlap was zero.
19. **Did deterministic replay pass?** Not run because no eligible checkpoint/output existed.
20. **Did batch consistency pass?** Not run for the same reason.
21. **Episodes labeled per partition?** `0 / 0 / 0`.
22. **Safe/unsafe counts?** Not measured; no labels were constructed.
23. **Safe prevalence?** Not measured.
24. **Did the source-output contract pass 100%?** It passed every physical and micro output evaluated; full-partition prevalence was not measured.
25. **Catastrophic-pass rate?** Not measured.
26. **Joint-safe rate?** Not measured.
27. **Which gate dominated?** Prompt identity/source ordering, not objective reduction, source nonnegativity, finiteness, or conservation.
28. **Were safe examples present in every partition?** Unknown; labels were prohibited.
29. **Did all label-support gates pass?** No; they were not reached after reconstruction failure.
30. **Was Family-E1 distinct from prior families?** Not scientifically evaluated because no frozen family outputs existed.
31. **Authoritative outcome?** `FAMILY_E1_RECONSTRUCTION_FAILURE`.
32. **Is Thayer-Audit v1 authorized?** No.
33. **What happens next?** Exactly one separately preregistered micro-only **Family-E1P Paired-Prompt Identity Intervention** on the same ordinary/difficult/eight scenes: retain the signed physical map and safety boundary, add one explicit paired-prompt source-ordering term, and require the unchanged 0.90 identity gate before any full training.
34. **Were development, Atlas selection, and lockbox untouched?** Yes: `0 / 0 / 0`.
35. **Were historical checkpoints unchanged?** Yes: `{len(after_rows)}` checked, zero mismatches.
36. **Reusable source/tests to review later?** `src/family_e1.py`, the bootstrap/run/finalize launchers, `tests/test_family_e1.py`, and `tests/test_thayer_family_e1_v0_artifacts.py`.
37. **Generated artifacts to remain ignored?** Both `outputs/runs/thayer_family_e1_v0_20260714_214638/` (failed bootstrap marker) and this entire `{relative(run)}/` tree.

## Evidence inventory

- Physical proof: `physical_contract/mps_physical_preflight.json` and the unchanged authoritative preflight.
- Objective audit: `tables/objective_alignment_audit.csv`, `objective_audit/objective_alignment_summary.json`, and `diagnostics/objective_alignment.md`.
- Architecture: `architecture/architecture_manifest.json`.
- Micro results/curves: `tables/micro_overfit_results.csv`, `micro_overfit/*_trace.csv`, and `figures/micro_overfit_curves.png`.
- Checkpoint/OOF/replay/label/comparison/bootstrap status artifacts explicitly record the mandatory stop and zero downstream payloads.
- Integrity: `tables/integrity_checks{suffix}.csv`, `tables/checkpoint_inventory_after{suffix}.csv`, and `diagnostics/final_git_status.txt`.

## Integrity and runtime

- Focused tests: `{tests[1]['evidence']}`.
- Compileall / CSV / git diff / staged / README / checkpoint audit: PASS.
- One architecture and one in-forward ReLU mapping only; no post-hoc clipping, truth deployment, safety-based selection, CPU fallback, or auditor training.
- Preflight run files unchanged; Condition C and Thayer-PU checkpoints unchanged.
- Runtime through report assembly: `{runtime:.1f}` seconds.
- Run disk usage / filesystem free: `{run_bytes}` / `{free}` bytes.
- Final Git status: `diagnostics/final_git_status.txt`.
"""
    fresh_text(run / "reports/final_report.md", final_report)
    fresh_text(run / "reports/final_report.sha256", f"{sha256_file(run / 'reports/final_report.md')}  final_report.md\n")
    fresh_json(run / "reports/frozen_core_decision.json", {
        "outcome": "FAMILY_E1_RECONSTRUCTION_FAILURE", "thayer_audit_v1_authorized": False,
        "next_experiment": "Family-E1P Paired-Prompt Identity Intervention", "auditor_trained": False,
    })
    fresh_json(run / "logs/campaign_end.json", {
        "status": "COMPLETE_FAIL_CLOSED", "outcome": "FAMILY_E1_RECONSTRUCTION_FAILURE",
        "runtime_seconds": runtime, "run_disk_bytes": run_bytes, "filesystem_free_bytes": free,
        "historical_checkpoint_mismatches": mismatch, "integrity_pass": integrity_pass,
        "development_access_count": 0, "atlas_selection_access_count": 0, "final_lockbox_access_count": 0,
    })
    print(json.dumps({"outcome": "FAMILY_E1_RECONSTRUCTION_FAILURE", "report": relative(run / "reports/final_report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
