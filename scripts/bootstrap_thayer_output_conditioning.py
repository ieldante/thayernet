#!/usr/bin/env python3
"""Create and freeze the metadata-only Thayer-OC master run.

This script is intentionally incapable of opening HDF5 datasets.  It hashes
their files and reads only the already persisted CSV microset manifest.
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
MICRO = ME / "diagnostics/micro_overfit_20260712_203540"
MANIFEST = MICRO / "tables/microset_manifest.csv"
ME_OUTPUTS = MICRO / "expert_outputs/micro_final_decompositions.h5"
SA_OUTPUTS = SA / "objective_preflight/final_outputs.h5"
TRAIN_TARGETS = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701/target_sets/thayer_mh_training_target_sets.h5"
VALID_TARGETS = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701/target_sets/thayer_mh_validation_target_sets.h5"
CAL_TARGETS = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701/target_sets/thayer_mh_calibration_target_sets.h5"
THRESHOLDS = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/manifests/forward_consistency_thresholds.json"
NOISE_CONTRACT = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/manifests/fixed_noise_contract.json"


EXPECTED = {
    MANIFEST: "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085",
    TRAIN_TARGETS: "7fc92222ff2d980c4beb787b961fa7bdaf3130c055ce842dc8fd5f600c29c19a",
    VALID_TARGETS: "a73477ab54f8c95ee6c14a9b13574e6f65e185e9dcebdc6f158dc564e573a55e",
    CAL_TARGETS: "9f660292c957ff72cd00356b82ccf3461a2e99f8a0fdb819a6e5d20084140910",
    REPO / "src/scientific_alignment.py": "62c0f1f7704a50a66b16c0044df7e140b3fae71563f1fa7db895f1d260655b07",
    REPO / "src/competing_hypotheses.py": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d",
    REPO / "src/loss_geometry.py": "94d3dbd3b29a1663517073514af1c78ee0c6a25bba1571d5f9efb939465b9b3a",
    REPO / "src/models_two_expert_decoder.py": "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415",
    REPO / "docs/multi_hypothesis_source_contract.md": "dc3a78b65b2eda17b71887c7616189a24fbf1f367c8fb61014d6e291a2e02128",
    REPO / "docs/latent_truth_coverage.md": "4d2d53ea7ef77c09b263ee90dec50b7138b0dbe07f2b16150113c0041e589d97",
    THRESHOLDS: "a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, text=True, capture_output=True).stdout.rstrip()


def checkpoint_paths() -> list[Path]:
    suffixes = {".pth", ".pt", ".ckpt"}
    return sorted(path for path in (REPO / "outputs").rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def main() -> None:
    started = datetime.now(timezone.utc)
    local_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run = REPO / f"outputs/runs/thayer_output_conditioning_{local_stamp}"
    run.mkdir(parents=True, exist_ok=False)
    directories = (
        "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
        "conditioning_geometry", "detached_optimization", "trajectories", "example_grids",
        "figures/common_vs_allocation_geometry", "figures/optimization_trajectories",
    )
    for name in directories:
        (run / name).mkdir(parents=True, exist_ok=False)

    mismatches = []
    frozen_hashes: dict[str, dict[str, object]] = {}
    for path, expected in EXPECTED.items():
        if not path.exists():
            mismatches.append({"path": str(path.relative_to(REPO)), "expected": expected, "observed": "MISSING"})
            continue
        observed = sha256(path)
        frozen_hashes[str(path.relative_to(REPO))] = {"sha256": observed, "bytes": path.stat().st_size}
        if observed != expected:
            mismatches.append({"path": str(path.relative_to(REPO)), "expected": expected, "observed": observed})
    for path in (SA_OUTPUTS, ME_OUTPUTS, NOISE_CONTRACT, REPO / "src/output_conditioning.py", REPO / "scripts/run_thayer_output_conditioning.py", REPO / "tests/test_output_conditioning.py"):
        if not path.exists():
            mismatches.append({"path": str(path.relative_to(REPO)), "expected": "EXISTS", "observed": "MISSING"})
        else:
            frozen_hashes[str(path.relative_to(REPO))] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    checkpoints = checkpoint_paths()
    checkpoint_rows = [{"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in checkpoints]
    authoritative_checkpoint_rows = list(csv.DictReader((SA / "tables/checkpoint_inventory_before.csv").open(newline="", encoding="utf-8")))
    current_checkpoint_map = {row["path"]: row for row in checkpoint_rows}
    for historical in authoritative_checkpoint_rows:
        current = current_checkpoint_map.get(historical["path"])
        if current is None:
            mismatches.append({"path": historical["path"], "expected": historical["sha256"], "observed": "MISSING_CHECKPOINT"})
        elif current["sha256"] != historical["sha256"]:
            mismatches.append({"path": historical["path"], "expected": historical["sha256"], "observed": current["sha256"]})
    fresh_csv(run / "tables/checkpoint_inventory_before.csv", checkpoint_rows)

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 64 or sum(row["kind"] == "ordinary" for row in manifest_rows) != 32 or sum(row["kind"] == "near_collision" for row in manifest_rows) != 32:
        mismatches.append({"path": str(MANIFEST.relative_to(REPO)), "expected": "64 rows: 32 ordinary + 32 near_collision", "observed": len(manifest_rows)})
    frozen_rows = [{
        "micro_index": row["micro_index"], "source_h5_index": row["source_h5_index"], "scene_id": row["scene_id"],
        "kind": row["kind"], "pair_id": row["pair_id"], "partition": row["partition"],
    } for row in manifest_rows]
    fresh_csv(run / "tables/frozen_row_ids.csv", frozen_rows)

    package_names = ("torch", "numpy", "scipy", "pandas", "h5py", "matplotlib", "btk", "galsim")
    packages = {}
    for package in package_names:
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package] = "NOT_INSTALLED"
    try:
        import torch
        mps = {"built": bool(torch.backends.mps.is_built()), "available": bool(torch.backends.mps.is_available()), "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"}
    except Exception as exc:
        mps = {"built": False, "available": False, "error": type(exc).__name__}

    source_rows = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            source_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    fresh_csv(run / "tables/source_code_hashes_before.csv", source_rows)
    status_v2 = git("status", "--porcelain=v2").splitlines()
    staged = git("diff", "--cached", "--name-status").splitlines()
    provenance = {
        "campaign": "Thayer-OC Output-Space Conditioning Audit",
        "working_experiment_name": "Thayer-OC",
        "campaign_started_utc": started.isoformat(),
        "run_dir": str(run.relative_to(REPO)),
        "branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain_v2": status_v2, "staged_index": staged,
        "python": sys.version, "platform": platform.platform(), "packages": packages, "mps": mps,
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "frozen_inputs": frozen_hashes,
        "authoritative_runs": {"loss_geometry": str(LG.relative_to(REPO)), "scientific_alignment": str(SA.relative_to(REPO)), "two_expert_decoder": str(ME.relative_to(REPO))},
        "initializations": {
            "sa_compromise": {"file": str(SA_OUTPUTS.relative_to(REPO)), "dataset": "source_sum_wrong_allocation"},
            "thayer_me_experts": {"file": str(ME_OUTPUTS.relative_to(REPO)), "dataset": "decompositions", "stored_units": "physical_detected_electrons"},
            "collapsed_means": "derived exactly from frozen targets after preregistration",
            "wrong_allocations": "derived exactly as 50/50 requested/companion at fixed total after preregistration",
            "exact_truths": "derived exactly from frozen targets after preregistration",
        },
        "historical_checkpoint_count": len(checkpoint_rows),
        "authoritative_sa_checkpoint_count_reverified": len(authoritative_checkpoint_rows),
        "additional_post_sa_checkpoint_count": len(checkpoint_rows) - len(authoritative_checkpoint_rows),
        "atlas_scene_access_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "per_scene_array_load_count_before_preregistration": 0, "detached_optimization_count_before_preregistration": 0,
        "frozen_input_mismatches": mismatches,
    }
    fresh_json(run / "logs/input_provenance.json", provenance)

    environment = f"""# Thayer-OC environment snapshot

- Campaign start (UTC): `{started.isoformat()}`
- Run: `{run.relative_to(REPO)}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Packages: `{json.dumps(packages, sort_keys=True)}`
- MPS built / available / fallback: `{mps.get('built')}` / `{mps.get('available')}` / `{mps.get('fallback_enabled', False)}`
- Free disk bytes: `{provenance['free_disk_bytes']}`
- Historical checkpoints: `{len(checkpoint_rows)}`
- Staged index entries: `{len(staged)}`
- Per-scene arrays loaded before preregistration: `0`
- Detached optimization before preregistration: `0`
"""
    fresh_text(run / "diagnostics/environment_snapshot.md", environment)
    campaign_contract = """# Thayer-OC campaign contract

Status before preregistration: **METADATA-ONLY**.

Thayer-OC is a fresh, CPU, training-free audit of detached physical output variables. Neural parameters are prohibited from every optimizer. The 64-row microset, approved targets, corrected Thayer-SA scalar objective and weights, hard two-permutation assignment, scientific thresholds, truth-coverage definitions, source-layer semantics, architecture, and protected-data boundary are immutable. Forward consistency is evaluation-only. Atlas, development, and lockbox access are zero. Historical checkpoints are read-only and byte-audited. Any frozen-input mismatch fails closed before preregistration or array loading.
"""
    fresh_text(run / "diagnostics/campaign_contract.md", campaign_contract)

    if mismatches:
        fresh_json(run / "logs/frozen_input_mismatch.json", {"status": "FAIL_CLOSED", "mismatches": mismatches})
        print(json.dumps({"status": "FROZEN_INPUT_MISMATCH", "run_dir": str(run), "mismatches": mismatches}, indent=2))
        raise SystemExit(2)

    gates = [
        {"gate": "truth_stationarity_all_rows", "range": "0..64 passing rows", "threshold": "64/64", "attainable": True, "proof": "Exact truth is representable, nonnegative, zero-loss, and stationary in authoritative Thayer-SA."},
        {"gate": "ordinary_own_coverage", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact truth previously achieved 32/32."},
        {"gate": "ambiguous_own_coverage", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact approved sets previously achieved 32/32."},
        {"gate": "ambiguous_alternate_coverage", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact approved sets previously achieved 32/32."},
        {"gate": "ambiguous_both_mode_coverage", "range": "0/32..32/32", "threshold": ">=29/32", "attainable": True, "proof": "Exact approved sets previously achieved 32/32."},
        {"gate": "global_method", "range": "one method for all scenes", "threshold": "no per-scene selection", "attainable": True, "proof": "Every fixed method shares one frozen rule and the same feasible truth point."},
        {"gate": "no_numerical_instability", "range": "false/true", "threshold": "finite throughout", "attainable": True, "proof": "Projection maps all nonfinite values to zero and enforces finite nonnegative layers."},
        {"gate": "protected_data", "range": "0+ accesses", "threshold": "0", "attainable": True, "proof": "Only the training microset is addressed by campaign code."},
    ]
    fresh_csv(run / "tables/preregistered_gate_attainability.csv", gates)

    row_lines = "\n".join(f"- `{row['micro_index']}`: `{row['scene_id']}` (source index `{row['source_h5_index']}`, `{row['kind']}`, pair `{row['pair_id'] or 'none'}`)" for row in frozen_rows)
    prereg = f"""# Preregistration: Thayer-OC output-space conditioning

Frozen at UTC `{datetime.now(timezone.utc).isoformat()}` before any per-scene HDF5 array load, detached gradient, curvature computation, or detached optimization. The authoritative 64 row IDs are listed below and identically persisted in `tables/frozen_row_ids.csv` (manifest SHA-256 `{EXPECTED[MANIFEST]}`).

## Immutable scientific scope

This campaign keeps the exact Thayer-SA corrected scalar objective: requested reconstruction + companion reconstruction + weight-1 threshold-normalized scientific surrogate, with weight-1 ordinary concentration. Prompts, experts, and scenes are averaged. Ambiguous rows retain the exact hard minimum over identity and swap. Forward, source-sum, prompt-swap, and pair-equivalence remain evaluation-only. Targets, thresholds, source semantics, normalization, architecture, hard assignment, 64-row microset, coverage implementation, and 90% gates are immutable. There is no neural fitting, model-weight update, Atlas, validation, calibration, development, or lockbox access.

Frozen code hashes are `src/scientific_alignment.py` `{frozen_hashes['src/scientific_alignment.py']['sha256']}`, `src/output_conditioning.py` `{frozen_hashes['src/output_conditioning.py']['sha256']}`, and `scripts/run_thayer_output_conditioning.py` `{frozen_hashes['scripts/run_thayer_output_conditioning.py']['sha256']}`. The Thayer-ME architecture hash is `{frozen_hashes['src/models_two_expert_decoder.py']['sha256']}`. The threshold file hash is `{frozen_hashes[str(THRESHOLDS.relative_to(REPO))]['sha256']}`.

## Exact physical coordinates and projection

For each expert, `T=S_req+S_comp`, `D=0.5(S_req-S_comp)`, `S_req=0.5T+D`, and `S_comp=0.5T-D`. COMMON perturbations add identical changes to requested and companion layers. ALLOCATION perturbations add equal and opposite changes and preserve their sum. The frozen projection decodes to physical sources, maps nonfinite values to zero, clamps each physical requested/companion pixel to `>=0`, and exactly re-encodes T/D. It uses no target beyond the ordinary supervised objective. C0 preserves the exact historical unprojected raw-space protocol; C1-C5 project after every accepted update.

## Initializations and deterministic seeds

All methods run from exactly five starts: persisted Thayer-SA `final_outputs.h5/source_sum_wrong_allocation` (file hash `{frozen_hashes[str(SA_OUTPUTS.relative_to(REPO))]['sha256']}`), persisted Thayer-ME physical expert decompositions converted by frozen scales (file hash `{frozen_hashes[str(ME_OUTPUTS.relative_to(REPO))]['sha256']}`), exact collapsed target means, exact 50/50 source-sum-preserving wrong allocations, and exact truths. No random restart is allowed. Campaign seed is `2026071304`; method seeds are C0..C5 = `2026071310`..`2026071315`.

## Optimizers and matched budgets

Every method has at most 401 corrected-objective evaluations, 400 corrected-objective gradient evaluations, 600 seconds per method/initialization, stopping gradient L2 tolerance `1e-8`, and trajectory logging at evaluation 0 and every 20 accepted/effective updates plus final. C0 is exact historical CPU float32 raw-output Adam: learning rate `1e-4`, zero weight decay, 400 updates, no projection. C1 is projected raw-space limited-memory BFGS with history 5, Armijo `c1=1e-4`, shrink 0.5, at most 8 trials, normalized-coordinate trust RMS 0.01, at most 120 accepted iterations. C2 optimizes physical T/D with projected Adam for 400 joint updates: per-band `lr_T[b]=1e-4*normalization_scale[b]`, `lr_D[b]=5e-4*normalization_scale[b]`, zero weight decay. C3 is projected joint physical-T/D L-BFGS with the C1 line search and physical trust RMS `0.01*median(normalization_scale)`. C4 has five frozen cycles, each 40 D-only Adam steps, 20 T-only steps, and 20 joint steps, using the C2 learning-rate formulas (400 total updates). C5 is projected physical-T/D Adam with the C2 rates; before each step its unchanged-objective gradients are multiplied by the median-normalized inverse absolute local scientific-surrogate Jacobian, `clip(median_positive(|J|)/(|J|+1e-8),0.1,10)`. C5 may use 400 auxiliary surrogate-Jacobian gradients but no coverage-adaptive information.

## Geometry, trajectories, and stopping

At exact truth and the persisted Thayer-SA compromise, raw/common/allocation gradients, per-band and per-expert norms, hard-assignment margins, positivity saturation, deterministic common/allocation Hessian-vector curvatures, finite-difference agreement at `h=1e-3`, modal curvature ratio, and a two-mode local condition estimate are fixed diagnostics. No dense Hessian is allowed. Every trajectory records the corrected objective and components, exact frozen coverage and distances, smooth maximum, common/allocation step and gradient norms, assignments and margins, evaluation-only forward consistency, expert diameter, and projection fraction. Nonfinite objectives, gradients, or outputs stop that run as numerical instability. No condition is retuned after results.

## Frozen success and interpretation

A method passes only if every truth-start row remains stationary/fully covered and, for every non-truth initialization, final ordinary own, ambiguous own, alternate, and both-mode coverage each reach at least 29/32 (>=90%), while objective, assignment, thresholds, targets, and protected-data boundaries remain unchanged. One global method must pass; per-scene method selection is forbidden. Partial success requires a materially nonzero coverage increase of at least 0.20 absolute over C0 for the same initialization while a 90% gate remains unmet. Failure includes no material improvement, truth instability, insufficient allocation conditioning, assignment barrier, projection barrier, or basin extremity. Exactly one primary category will be selected from the eight specified categories, with `MIXED CAUSE` only under direct multiple-mechanism evidence. Exactly one next experiment will be recommended and not run.

## Gate attainability

Every rate gate is an integer 29/32 requirement and exact persisted truths previously achieved 32/32 for all four coverage metrics. The corrected objective has exact truth at zero with zero gradient. The T/D map is bijective before projection and exact truths are feasible nonnegative points. Therefore every numerical and coverage gate is mathematically attainable. The detailed audit is `tables/preregistered_gate_attainability.csv`.

## Frozen row IDs

{row_lines}
"""
    prereg_path = run / "preregistration/output_space_conditioning.md"
    fresh_text(prereg_path, prereg)
    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze = {
        "status": "FROZEN_BEFORE_ANY_PER_SCENE_ARRAY_LOAD_OR_DETACHED_OPTIMIZATION",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": sha256(prereg_path),
        "frozen_row_ids_sha256": sha256(run / "tables/frozen_row_ids.csv"),
        "gate_attainability_sha256": sha256(run / "tables/preregistered_gate_attainability.csv"),
        "analysis_implementation_sha256": frozen_hashes["scripts/run_thayer_output_conditioning.py"]["sha256"],
        "coordinate_implementation_sha256": frozen_hashes["src/output_conditioning.py"]["sha256"],
        "corrected_objective_implementation_sha256": frozen_hashes["src/scientific_alignment.py"]["sha256"],
        "per_scene_array_load_count": 0, "detached_optimization_count": 0,
    }
    fresh_json(run / "preregistration/freeze_record.json", freeze)
    fresh_json(run / "logs/preregistration_complete.json", {**freeze, "run_dir": str(run)})
    print(json.dumps({"status": "PREREGISTERED", "run_dir": str(run), "preregistration_sha256": freeze["preregistration_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
