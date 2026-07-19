#!/usr/bin/env python3
"""Create and freeze the metadata-only Thayer-FP master run.

This bootstrap never imports h5py and never opens a per-scene numerical array.
It hashes immutable files, reads the persisted CSV microset manifest, audits
gate attainability, and freezes the full protocol before later execution.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"
LG = REPO / "outputs/runs/thayer_loss_geometry_20260712_205733"
SA = REPO / "outputs/runs/thayer_scientific_alignment_20260712_220315"
OC = REPO / "outputs/runs/thayer_output_conditioning_20260712_225459"
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
MANIFEST = MICRO / "tables/microset_manifest.csv"
ME_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
ME_CHECKPOINT = MICRO / "checkpoints/thayer_me_micro_final.pth"
SA_OUTPUTS = SA / "objective_preflight/final_outputs.h5"
OC_OUTPUTS = OC / "detached_optimization/final_outputs.h5"
MH = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701"
TRAIN_SCENES = MH / "manifests/probabilistic_unet_training_scenes.h5"
TRAIN_TARGETS = MH / "target_sets/thayer_mh_training_target_sets.h5"
VALID_TARGETS = MH / "target_sets/thayer_mh_validation_target_sets.h5"
CAL_TARGETS = MH / "target_sets/thayer_mh_calibration_target_sets.h5"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
THRESHOLDS = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/manifests/forward_consistency_thresholds.json"
NOISE = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json"


EXPECTED = {
    MANIFEST: "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085",
    ME_OUTPUTS: "612d02fc72686ccde704ebf56bdb10c30af28fde3d20939afebac3fa34553446",
    ME_CHECKPOINT: "f96648ebd990dceaab89b6e10e518701fd9126a17df45cc7eb2795968a4a5757",
    SA_OUTPUTS: "a73ee3be59b54d0dacb7a82025c9d54fcb46b4d27916256096f5ecea813cb671",
    OC_OUTPUTS: "34e059b68db1c09384876a5e047443bba8ae316dfc0137e59e9645258ddd00ad",
    TRAIN_SCENES: "d6ca6f1cbcb136a075f0216460e5f6b2dcd5fefbb63894803b86069df4e5f48d",
    TRAIN_TARGETS: "7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a",
    VALID_TARGETS: "a73477ab54f8c95ee6c14a9b13574e6f65e185e9dcebdc6f158dc564e573a55e",
    CAL_TARGETS: "9f660292c957ff72cd00356b82ccf3461a2e99f8a0fdb819a6e5d20084140910",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    THRESHOLDS: "a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a",
    NOISE: "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
    REPO / "src/canonical_tensor_hash.py": "65566c01c5e6a76bc35e638423562180f370edb7b5b8bc5a3931ae2ca994bb6e",
    REPO / "src/competing_hypotheses.py": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d",
    REPO / "src/loss_geometry.py": "94d3dbd3b29a1663517073514af1c78ee0c6a25bba1571d5f9efb939465b9b3a",
    REPO / "src/models_two_expert_decoder.py": "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415",
    REPO / "src/scientific_alignment.py": "62c0f1f7704a50a66b16c0044df7e140b3fae71563f1fa7db895f1d260655b07",
    REPO / "src/output_conditioning.py": "989699a959aa03de25d45a5285de7a8abe6b1ab61dbfc255451526d41d09474a",
    REPO / "scripts/run_thayer_two_expert_micro_overfit.py": "69a4e862fdff54f4de7dc2564f35ccced29528317a266882d710b28b92ac7ec2",
    REPO / "docs/multi_hypothesis_source_contract.md": "dc3a78b65b2eda17b71887c7616189a24fbf1f367c8fb61014d6e291a2e02128",
    REPO / "docs/latent_truth_coverage.md": "4d2d53ea7ef77c09b263ee90dec50b7138b0dbe07f2b16150113c0041e589d97",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, text=True, capture_output=True).stdout.rstrip()


def checkpoint_paths() -> list[Path]:
    return sorted(path for path in (REPO / "outputs").rglob("*") if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"})


def main() -> None:
    started = datetime.now(timezone.utc)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_feasibility_projection_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    directories = (
        "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
        "projection_targets", "projection_trajectories", "checkpoints", "micro_overfit",
        "example_grids", "figures/feasibility_entry_paths",
    )
    for name in directories:
        (run / name).mkdir(parents=True, exist_ok=False)

    mismatches: list[dict[str, object]] = []
    frozen_inputs: dict[str, dict[str, object]] = {}
    for path, expected in EXPECTED.items():
        observed = sha256(path) if path.exists() else "MISSING"
        if path.exists():
            frozen_inputs[str(path.relative_to(REPO))] = {"sha256": observed, "bytes": path.stat().st_size}
        if observed != expected:
            mismatches.append({"path": str(path.relative_to(REPO)), "expected": expected, "observed": observed})

    # Reverify every source file that existed at the authoritative Thayer-OC close.
    with (OC / "tables/source_code_hashes_final.csv").open(newline="", encoding="utf-8") as handle:
        historical_source_rows = list(csv.DictReader(handle))
    for row in historical_source_rows:
        path = REPO / row["path"]
        observed = sha256(path) if path.exists() else "MISSING"
        if observed != row["sha256"]:
            mismatches.append({"path": row["path"], "expected": row["sha256"], "observed": observed, "kind": "historical_source"})

    checkpoints = checkpoint_paths()
    checkpoint_rows = [{"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in checkpoints]
    with (OC / "tables/checkpoint_inventory_after.csv").open(newline="", encoding="utf-8") as handle:
        authoritative_checkpoints = list(csv.DictReader(handle))
    current_checkpoint_map = {row["path"]: row for row in checkpoint_rows}
    for row in authoritative_checkpoints:
        current = current_checkpoint_map.get(row["path"])
        if current is None or current["sha256"] != row["sha256"]:
            mismatches.append({"path": row["path"], "expected": row["sha256"], "observed": "MISSING" if current is None else current["sha256"], "kind": "historical_checkpoint"})
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", checkpoint_rows)

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 64 or sum(row["kind"] == "ordinary" for row in manifest_rows) != 32 or sum(row["kind"] == "near_collision" for row in manifest_rows) != 32:
        mismatches.append({"path": str(MANIFEST.relative_to(REPO)), "expected": "64 rows: 32 ordinary and 32 near_collision", "observed": len(manifest_rows)})
    frozen_rows = [{
        "micro_index": row["micro_index"], "source_h5_index": row["source_h5_index"],
        "scene_id": row["scene_id"], "kind": row["kind"], "pair_id": row["pair_id"],
        "partition": row["partition"],
    } for row in manifest_rows]
    fresh_csv(run / "tables/frozen_row_ids.csv", frozen_rows)

    package_names = ("torch", "numpy", "scipy", "pandas", "h5py", "matplotlib", "btk", "galsim")
    packages = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    import torch
    mps = {
        "built": bool(torch.backends.mps.is_built()),
        "available": bool(torch.backends.mps.is_available()),
        "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1",
    }
    source_rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    fresh_csv(run / "tables/source_code_hashes_before.csv", source_rows)

    status = git("status", "--porcelain=v2").splitlines()
    staged = git("diff", "--cached", "--name-status").splitlines()
    provenance = {
        "campaign": "Thayer-FP Direct Scientific-Feasibility Projection Micro-Audit",
        "working_experiment_name": "Thayer-FP",
        "campaign_started_utc": started.isoformat(),
        "run_dir": str(run.relative_to(REPO)),
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain_v2": status,
        "staged_index": staged,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "mps": mps,
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "frozen_inputs": frozen_inputs,
        "microset_manifest_sha256": sha256(MANIFEST),
        "target_set_hashes": {"training": sha256(TRAIN_TARGETS), "validation": sha256(VALID_TARGETS), "calibration": sha256(CAL_TARGETS)},
        "persisted_compromise_output_hashes": {"thayer_me": sha256(ME_OUTPUTS), "thayer_sa": sha256(SA_OUTPUTS), "thayer_oc": sha256(OC_OUTPUTS)},
        "architecture_code_sha256": sha256(REPO / "src/models_two_expert_decoder.py"),
        "architecture_initialization_seeds": [2026071201, 2026071202],
        "training_seed": 2026071250,
        "source_layer_contract_sha256": sha256(REPO / "docs/multi_hypothesis_source_contract.md"),
        "target_set_contract_sha256": sha256(ME / "diagnostics/target_set_reuse_report.md"),
        "hard_assignment_implementation_sha256": sha256(REPO / "src/scientific_alignment.py"),
        "scientific_thresholds_sha256": sha256(THRESHOLDS),
        "coverage_contract_sha256": sha256(REPO / "docs/latent_truth_coverage.md"),
        "canonical_hash_implementation_sha256": sha256(REPO / "src/canonical_tensor_hash.py"),
        "forward_consistency_implementation_sha256": sha256(REPO / "src/competing_hypotheses.py"),
        "historical_checkpoint_count": len(checkpoint_rows),
        "authoritative_historical_checkpoint_count": len(authoritative_checkpoints),
        "per_scene_array_load_count_before_preregistration": 0,
        "detached_optimization_count_before_preregistration": 0,
        "neural_optimizer_step_count_before_preregistration": 0,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "frozen_input_mismatches": mismatches,
        "authoritative_runs": {"thayer_me": str(ME.relative_to(REPO)), "loss_geometry": str(LG.relative_to(REPO)), "thayer_sa": str(SA.relative_to(REPO)), "thayer_oc": str(OC.relative_to(REPO))},
    }
    fresh_json(run / "logs/input_provenance.json", provenance)
    fresh_text(run / "diagnostics/environment_snapshot.md", f"""# Thayer-FP environment snapshot

- Campaign start (UTC): `{started.isoformat()}`
- Run: `{run.relative_to(REPO)}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Python executable: `{sys.executable}`
- Platform: `{platform.platform()}`
- Packages: `{json.dumps(packages, sort_keys=True)}`
- MPS built / available / fallback: `{mps['built']}` / `{mps['available']}` / `{mps['fallback_enabled']}`
- Free disk bytes: `{provenance['free_disk_bytes']}`
- Historical checkpoints: `{len(checkpoint_rows)}`
- Staged index entries: `{len(staged)}`
- Per-scene arrays loaded before preregistration: `0`
- Detached optimizations before preregistration: `0`
- Neural optimizer steps before preregistration: `0`
""")
    fresh_text(run / "diagnostics/campaign_contract.md", """# Thayer-FP campaign contract

Status before preregistration: **METADATA-ONLY**.

Thayer-FP is an append-only projection and 64-scene micro-feasibility campaign. The source-layer contract, approved target sets, hard assignment, scientific thresholds, truth-coverage definition, prompt contract, 165,612-parameter Thayer-ME architecture, initialization policy, and protected-data boundary are immutable. Projection is CPU-only and model-free. Neural training is prohibited until the projection gate passes and, if authorized, must be MPS-only. Forward consistency, source-sum consistency, and prompt swap remain evaluation-only. Atlas, development, and lockbox access remain zero. Any frozen-input mismatch fails closed before per-scene loading.
""")
    if mismatches:
        fresh_json(run / "logs/frozen_input_mismatch.json", {"status": "FAIL_CLOSED", "mismatches": mismatches})
        print(json.dumps({"status": "FROZEN_INPUT_MISMATCH", "run_dir": str(run), "mismatches": mismatches}, indent=2))
        raise SystemExit(2)

    gate_rows = [
        {"gate": "exact_truth_feasibility", "range": "0/256..256/256 pairings", "threshold": "256/256", "attainable": True, "proof": "Exact approved truths previously passed every frozen scientific check and are valid nonnegative six-channel outputs."},
        {"gate": "projected_target_overall", "range": "0..1", "threshold": ">=0.95", "attainable": True, "proof": "P0 alpha=1 is exact truth for every pairing."},
        {"gate": "ordinary_projected_sets", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Duplicated exact ordinary truths attain 32/32."},
        {"gate": "ambiguous_own_projected_sets", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact own targets attain 32/32."},
        {"gate": "ambiguous_alternate_projected_sets", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact alternate targets attain 32/32."},
        {"gate": "ambiguous_both_mode_projected_sets", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact unordered approved sets attain 32/32."},
        {"gate": "interior_slack", "range": "max ratio 0..infinity", "threshold": "<=0.95 for P0 and selected targets", "attainable": True, "proof": "Exact truth has all scientific ratios zero."},
        {"gate": "projection_determinism", "range": "boolean", "threshold": "exact canonical hash reproduction", "attainable": True, "proof": "Frozen CPU grid, bisection, seeds, and canonical hash are deterministic."},
        {"gate": "projected_forward_consistency", "range": "0/32..32/32 per kind", "threshold": ">=29/32 ordinary and ambiguous", "attainable": True, "proof": "Authoritative exact truths passed frozen forward plausibility."},
        {"gate": "neural_ordinary_coverage", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact duplicated truths attain 32/32."},
        {"gate": "neural_ambiguous_own_alternate_both", "range": "0/32..32/32 each", "threshold": ">=29/32 each", "attainable": True, "proof": "Exact approved unordered sets attain 32/32."},
        {"gate": "ordinary_expert_diameter", "range": "0..infinity", "threshold": "median <=1.0", "attainable": True, "proof": "Duplicated ordinary targets have diameter zero."},
        {"gate": "prompt_and_forward", "range": "0..1", "threshold": ">=0.90 each frozen rate", "attainable": True, "proof": "Exact prompt mapping and exact decompositions attain the gates."},
    ]
    fresh_csv(run / "tables/preregistered_gate_attainability.csv", gate_rows)
    if not all(bool(row["attainable"]) for row in gate_rows):
        raise RuntimeError("unattainable gate before freeze")

    row_lines = "\n".join(f"- `{row['micro_index']}` `{row['scene_id']}` `{row['kind']}` `{row['pair_id']}` source index `{row['source_h5_index']}`" for row in frozen_rows)
    preregistration = f"""# Direct Scientific-Feasibility Projection preregistration

Working name: **Thayer-FP (Thayer Feasibility Projection)**  
Frozen at UTC: `{datetime.now(timezone.utc).isoformat()}`  
Microset manifest SHA-256: `{sha256(MANIFEST)}`  
Frozen-row table SHA-256: `{sha256(run / 'tables/frozen_row_ids.csv')}`

## Scope and immutable boundary

Thayer-FP uses exactly the 64 training-only Thayer-ME microset observations, the persisted Thayer-ME expert outputs as initial candidates, and the unchanged approved targets. It tests offline projection into the unchanged scientific region, then only if the projection gate passes tests whether the unchanged 165,612-parameter Thayer-ME can memorize those training-only representatives. Atlas, validation, calibration, development, and lockbox arrays are prohibited. Truth and constraint values may construct offline targets but may never enter model inference tensors. Projections are training-only representatives, not new astronomical truth.

## Frozen outputs, targets, assignment, and constraints

Initial candidates are file `{ME_OUTPUTS.relative_to(REPO)}`, dataset `decompositions`, hash `{sha256(ME_OUTPUTS)}`. Target file hash is `{sha256(TRAIN_TARGETS)}`. The full six-channel contract is requested g/r/z followed by companion g/r/z, zero background, exact 60 x 60 dimensions and band order, finite values, and nonnegative physical source layers. The hard two-permutation assignment is the unchanged Thayer-SA pairwise requested-reconstruction + companion-reconstruction + scientific-cost assignment; identity wins exact ties. Ordinary rows assign both experts to target zero. Ambiguous rows retain identity/swap assignment and require the unordered own/alternate set.

Scientific feasibility uses the exact nondifferentiable frozen requested-source components: symmetric image distance / 0.25; each g/r/z symmetric relative-flux error / 0.20; each applicable g-r and r-z magnitude difference / 0.20 mag; and centroid displacement / mean PSF FWHM / 0.50. Authoritative acceptance remains componentwise <=1.0. Forward consistency, source-sum consistency, and prompt swap are evaluation-only and never feasibility or training objectives.

## Guaranteed homotopy projection P0

For every expert-target assignment, evaluate `X(alpha)=(1-alpha) candidate + alpha exact_truth` on exactly 1,025 evenly spaced alpha values from zero through one. Record every feasible interval, component entry alpha, the limiting component, and whether component ratios and binary feasibility are monotone. Locate the earliest feasible boundary by 40 deterministic bisection steps. Independently locate the earliest fixed training interior with every scientific ratio <=0.95 and move an additional alpha `1e-8` inward. The scientific evaluation threshold remains 1.0. If float32 conversion violates 0.95, use the exact-truth alpha=1 anchor. Correction distance is full-decomposition L2 correction divided by original-candidate L2 plus 1e-12.

## Fixed refinement P1

P1 starts only from P0 and uses one global CPU float32 projected augmented-Lagrangian method for every pairing: Adam, 80 iterations, learning rate 2e-4, zero weight decay, dual updates every 10 iterations, penalties 10/30/100/300 in four equal blocks, elementwise nonnegative projection after every update, and seven component constraints at the fixed 0.95 training interior. A final fixed 40-step bisection toward P0 restores any numerical violation. No neural parameter enters P0 or P1. No per-scene method selection is allowed. A separate SciPy solver is omitted because the full 21,600-variable per-pair problem is not computationally proportionate and P0 already supplies an exact feasible anchor.

## Projection gates and global selection

Exact truth must be feasible for all 256 scene/prompt/expert pairings. An eligible method must be finite, contract-valid, hard-assignment valid, canonically stable, deterministic, and unchanged-threshold feasible. The primary gate requires >=95% feasible pairings overall, >=90% ordinary sets, ambiguous own, alternate, and both-mode sets, and >=90% ordinary and ambiguous all-expert forward consistency. Selection is global and lexicographic: feasible rate, both-mode rate, smaller median correction, greater median interior slack, forward-consistency retention, deterministic stability, then P0 on an exact tie. Projection failure stops before neural work.

## Frozen projected targets

Ordinary target slots are two feasible representatives of the same approved truth region and may be identical. Ambiguous target slots remain an unordered two-target set in canonical target order. Persist tensors, canonical per-sample hashes, source-truth provenance outside inference inputs, assignment metadata, and projection metadata. No constraint or truth feature is appended to blend or prompt tensors.

## Unchanged Thayer-ME micro learning

If authorized, instantiate the exact shared Condition-C-prompted encoder and two independent 46,470-parameter expert decoders, total 165,612 parameters, with expert seeds 2026071201/2026071202, training seed 2026071250, exact Condition-C encoder warm start, and phase-2 trainability. Use MPS only, AdamW, batch size 8, learning rate 1e-3, zero weight decay, no augmentation, and at most 400 epochs. Each batch has the frozen four ordinary plus two complete ambiguity-pair schedule. The only loss is direct requested-source MSE plus direct companion-source MSE under the unchanged hard two-permutation matching; ordinary rows supervise both experts to the projected ordinary slots. No scientific surrogate, forward, source-sum, prompt-swap, concentration, diversity, uncertainty, or Atlas term is allowed.

Evaluate at epoch 1 and every 20 epochs. Stop on nonfinite loss/output, MPS fallback, target-hash mismatch, output-contract violation, nonfinite assignment, five consecutive epochs with expert gradient <=1e-12, or set prompt identity below 0.50 at/after epoch 100. Save only fresh campaign-local checkpoints. Success requires >=29/32 ordinary set-level coverage with both experts covered, median ordinary diameter <=1.0, >=29/32 ambiguous own/alternate/both-mode coverage, >=0.90 set prompt identity, and >=0.90 ordinary/ambiguous forward consistency. Partial success requires every coverage category materially nonzero but at least one success gate unmet. Failure includes feasible targets that the unchanged model cannot memorize or prompt/forward collapse. Gates cannot change after training.

## Interpretation

Success supports microset sufficiency of current decoder capacity and authorizes only one separately preregistered full non-Atlas feasibility-learning campaign. Projection pass with neural failure directly implicates capacity, encoder conditioning, or neural parameterization and authorizes a controlled decoder-capacity ladder without changing targets or thresholds. Projection failure prohibits a capacity ladder. Truth-coverage improvement with failed prompt/forward gates is partial and recommends one constrained consistency correction without reintroducing the dominating forward loss.

## Frozen rows

{row_lines}
"""
    prereg_path = run / "preregistration/direct_scientific_feasibility_projection.md"
    fresh_text(prereg_path, preregistration)
    prereg_hash = sha256(prereg_path)
    fresh_json(run / "preregistration/freeze_record.json", {
        "status": "FROZEN_BEFORE_PER_SCENE_LOAD",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": prereg_hash,
        "frozen_rows_sha256": sha256(run / "tables/frozen_row_ids.csv"),
        "gate_attainability_sha256": sha256(run / "tables/preregistered_gate_attainability.csv"),
        "per_scene_array_load_count": 0,
        "detached_optimization_count": 0,
        "neural_optimizer_step_count": 0,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
    })
    fresh_json(run / "logs/preregistration_complete.json", {
        "status": "PASS", "run_dir": str(run.relative_to(REPO)), "preregistration_sha256": prereg_hash,
        "all_gates_attainable": True, "per_scene_array_load_count_before_freeze": 0,
    })
    print(json.dumps({"status": "PREREGISTERED", "run_dir": str(run), "preregistration_sha256": prereg_hash}, indent=2))


if __name__ == "__main__":
    main()
