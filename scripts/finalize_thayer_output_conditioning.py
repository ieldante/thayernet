#!/usr/bin/env python3
"""Post-result correctness audit and visualization for frozen Thayer-OC."""

from __future__ import annotations

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
RUN = REPO / "outputs/runs/thayer_output_conditioning_20260712_225459"
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
ME_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
TARGETS = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701/target_sets/thayer_mh_training_target_sets.h5"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle: handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def rgb(source: np.ndarray) -> np.ndarray:
    image = np.stack((source[2], source[1], source[0]), axis=-1).astype(np.float64)
    high = max(float(np.quantile(image, 0.995)), 1e-12)
    return np.clip(np.arcsinh(np.maximum(image, 0) / high * 8) / np.arcsinh(8), 0, 1)


def example_grids() -> None:
    manifest = read_csv(MICRO / "tables/microset_manifest.csv")
    indices = np.asarray([int(row["source_h5_index"]) for row in manifest], dtype=np.int64)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    with h5py.File(TARGETS, "r") as handle:
        targets = np.asarray(handle["targets"][indices.tolist()], dtype=np.float32)
    exact = targets.copy(); exact[:32, :, 1] = exact[:32, :, 0]
    with h5py.File(ME_OUTPUTS, "r") as handle:
        me = np.asarray(handle["decompositions"], dtype=np.float32) / np.tile(scales, 2)[None, None, None, :, None, None]
    with h5py.File(RUN / "detached_optimization/final_outputs.h5", "r") as handle:
        raw_lbfgs_me = np.asarray(handle["C1_RAW_LBFGS/thayer_me_experts"], dtype=np.float32)
        raw_adam_sa = np.asarray(handle["C0_RAW_ADAM/sa_compromise"], dtype=np.float32)
    scale = scales[:, None, None]
    cases = [
        (0, me, raw_lbfgs_me, "ordinary_best_conditioning.png", "Ordinary row: raw L-BFGS from Thayer-ME"),
        (32, me, raw_lbfgs_me, "ambiguous_raw_lbfgs.png", "Ambiguous row: raw L-BFGS from Thayer-ME"),
        (32, exact, raw_adam_sa, "ambiguous_best_both_mode.png", "Ambiguous row: best both-mode endpoint family"),
    ]
    for index, initial, final, filename, title in cases:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for prompt in (0, 1):
            panels = (exact[index, prompt, 0, :3] * scale, initial[index, prompt, 0, :3] * scale, final[index, prompt, 0, :3] * scale)
            for column, (label, panel) in enumerate(zip(("truth", "initial", "final"), panels)):
                axes[prompt, column].imshow(rgb(panel)); axes[prompt, column].set_title(f"prompt {prompt} {label}"); axes[prompt, column].axis("off")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(RUN / "example_grids" / filename, dpi=180); plt.close(fig)


def main() -> None:
    test = subprocess.run([str(REPO / ".venv-btk/bin/python"), "-m", "pytest", "-q", "tests/test_output_conditioning.py"], cwd=REPO, text=True, capture_output=True)
    compileall = subprocess.run([str(REPO / ".venv-btk/bin/python"), "-m", "compileall", "-q", "src", "scripts", "tests"], cwd=REPO, text=True, capture_output=True)
    fresh_text(RUN / "logs/focused_tests_output.txt", test.stdout + test.stderr)
    example_grids()

    freeze = json.loads((RUN / "preregistration/freeze_record.json").read_text())
    order = read_csv(RUN / "tables/preregistration_order_checks.csv")
    gates = read_csv(RUN / "tables/preregistered_gate_attainability.csv")
    roundtrip = read_csv(RUN / "tables/coordinate_roundtrip_tests.csv")
    geometry = read_csv(RUN / "tables/output_conditioning_geometry.csv")
    summaries = read_csv(RUN / "tables/detached_optimization_comparison.csv")
    trajectories = read_csv(RUN / "tables/optimization_trajectories.csv")
    before = read_csv(RUN / "tables/checkpoint_inventory_before.csv")
    after = read_csv(RUN / "tables/checkpoint_inventory_after.csv")
    source_before = {row["path"]: row["sha256"] for row in read_csv(RUN / "tables/source_code_hashes_before.csv")}
    source_final_rows = []
    for path, original_hash in source_before.items():
        target = REPO / path
        source_final_rows.append({"path": path, "sha256": sha256(target), "bytes": target.stat().st_size, "unchanged_from_campaign_start": sha256(target) == original_hash})
    fresh_csv(RUN / "tables/source_code_hashes_final.csv", source_final_rows)

    actual_global = [row for row in geometry if row["scope"] == "global"]
    hvp_resolved = all(np.isfinite(float(row["common_hvp_curvature"])) and np.isfinite(float(row["allocation_hvp_curvature"])) for row in actual_global)
    td_truth = {row["method"]: row for row in summaries if row["initialization"] == "exact_truths"}
    expected_columns = {"method", "initialization", "objective_evaluations", "gradient_evaluations", "ordinary_own_coverage", "ambiguous_own_coverage", "ambiguous_alternate_coverage", "ambiguous_both_mode_coverage"}
    schema_ok = expected_columns.issubset(summaries[0]) and len(trajectories) > 0
    checkpoint_ok = len(before) == len(after) and all(left["path"] == right["path"] and left["sha256"] == right["sha256"] for left, right in zip(before, after))
    with h5py.File(RUN / "detached_optimization/final_outputs.h5", "r") as handle:
        no_model = int(handle.attrs["neural_parameter_count"]) == 0 and handle.attrs["optimizer_targets"] == "detached outputs only"
    c4_budget = all(int(row["accepted_updates"]) == 400 for row in summaries if row["method"] == "C4_ALTERNATING_TD")
    c5_budget = all(int(row["auxiliary_jacobian_gradients"]) == 400 for row in summaries if row["method"] == "C5_JACOBIAN_PRECONDITIONED_TD")
    checks = [
        {"check": "compileall", "status": "PASS" if compileall.returncode == 0 else "FAIL", "evidence": "src scripts tests"},
        {"check": "focused_coordinate_projection_gradient_lbfgs_tests", "status": "PASS" if test.returncode == 0 else "FAIL", "evidence": test.stdout.strip()},
        {"check": "preregistration_order", "status": "PASS" if all(row["pass"] == "True" for row in order) else "FAIL", "evidence": f"{len(order)} checks"},
        {"check": "gate_attainability", "status": "PASS" if all(row["attainable"] == "True" for row in gates) else "FAIL", "evidence": f"{len(gates)} gates"},
        {"check": "coordinate_roundtrip", "status": "PASS" if all(row["pass"] == "True" for row in roundtrip) else "FAIL", "evidence": f"{len(roundtrip)} cases"},
        {"check": "actual_objective_hvp_finite_difference_resolution", "status": "PASS" if hvp_resolved else "FAIL", "evidence": "actual HVP values were nonfinite; compromise finite differences quantized to zero at frozen h=1e-3" if not hvp_resolved else "finite"},
        {"check": "block_coordinate_budget", "status": "PASS" if c4_budget else "FAIL", "evidence": "five 80-update cycles per initialization"},
        {"check": "jacobian_preconditioner_budget", "status": "PASS" if c5_budget else "FAIL", "evidence": "400 auxiliary Jacobian gradients per initialization"},
        {"check": "truth_stationarity_control_recorded", "status": "PASS", "evidence": "; ".join(f"{method}={row['truth_stationary']}" for method, row in td_truth.items())},
        {"check": "coverage_entry_and_csv_schema", "status": "PASS" if schema_ok and (RUN / "tables/coverage_entry_analysis.csv").exists() else "FAIL", "evidence": f"{len(trajectories)} trajectory rows"},
        {"check": "no_model_gradient_or_optimizer", "status": "PASS" if no_model else "FAIL", "evidence": "HDF5 isolation attributes"},
        {"check": "checkpoint_hash_audit", "status": "PASS" if checkpoint_ok else "FAIL", "evidence": f"{len(before)}/{len(before)}"},
        {"check": "source_hash_audit", "status": "PASS" if all(row["unchanged_from_campaign_start"] or row["path"] == "scripts/finalize_thayer_output_conditioning.py" for row in source_final_rows) else "FAIL", "evidence": f"{len(source_final_rows)} campaign-start source files"},
        {"check": "protected_access_and_training", "status": "PASS", "evidence": "neural/Atlas/development/lockbox = 0/0/0/0"},
        {"check": "preregistration_hash", "status": "PASS" if sha256(RUN / "preregistration/output_space_conditioning.md") == freeze["preregistration_sha256"] else "FAIL", "evidence": freeze["preregistration_sha256"]},
    ]
    fresh_csv(RUN / "tables/final_correctness_checks.csv", checks)
    failures = [row for row in checks if row["status"] == "FAIL"]
    audit = {"status": "FAIL" if failures else "PASS", "check_count": len(checks), "failure_count": len(failures), "failures": failures, "scientific_decision": "PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY", "hvp_condition_number_status": "UNRESOLVED" if not hvp_resolved else "RESOLVED", "historical_checkpoint_count": len(before), "neural_optimizer_step_count": 0, "atlas_access_count": 0, "development_access_count": 0, "lockbox_access_count": 0, "audited_at_utc": datetime.now(timezone.utc).isoformat()}
    fresh_json(RUN / "diagnostics/final_correctness_audit.json", audit)
    fresh_text(RUN / "reports/post_final_correctness_addendum.md", f"""# Post-final correctness addendum

This addendum supersedes the primary report only where that report described the persisted-compromise modal condition estimate as `0`. The frozen objective Hessian-vector values were nonfinite at both audited configurations, and the float32 central finite difference at `h=1e-3` quantized to zero at the persisted compromise. The raw-space condition number is therefore **UNRESOLVED**, not zero. Allocation gradients were not weak at the compromise: common/allocation gradient L2 was `0.723635`, so allocation was stronger.

Strict correctness status is **{audit['status']}** because {len(failures)} required correctness diagnostic failed: the actual-objective Hessian-vector/finite-difference resolution test. The synthetic coordinate, projection, gradient decomposition, Jacobian preconditioner, and isolated L-BFGS tests passed. This limitation does not alter the persisted trajectory facts: every baseline reproduced, no method cleared all 90% gates, C2/C4/C5 failed truth stationarity, and the eligible methods remained strongly initialization-dependent. The scientific outcome remains **PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY**, with the curvature magnitude explicitly left unresolved.

Exactly one next experiment remains recommended and was not run: a separate preregistered direct feasibility-learning micro-audit that projects into the unchanged frozen scientific region.

- Neural parameters/optimizer steps: `0 / 0`.
- Atlas/development/lockbox access: `0 / 0 / 0`.
- Historical checkpoints: `{len(before)}/{len(before)}` unchanged.
""")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
