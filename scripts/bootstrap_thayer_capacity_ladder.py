#!/usr/bin/env python3
"""Create the metadata-only Thayer-CL run and freeze its preregistration."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs/runs"
PREFIX = "thayer_capacity_ladder_"

FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
ME = REPO / "outputs/runs/thayer_two_expert_decoder_20260712_203121"

MICROSET = ME / "diagnostics/micro_overfit_20260712_203540/tables/microset_manifest.csv"
P0_TARGETS = FP / "projection_targets/projected_target_sets_final.h5"
P0_HASHES = FP / "tables/projected_target_hashes_final.csv"
FP_CHECKPOINT = FP / "checkpoints/thayer_fp_micro_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
FORWARD_THRESHOLDS = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340/manifests/forward_consistency_thresholds.json"

EXPECTED = {
    MICROSET: "9622abda0bc44468761ccec56cf22ed7a5ab410dfecf919bb9e4bc4a45700085",
    P0_TARGETS: "d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181",
    P0_HASHES: "b45e26f95f7ecf7fc117f3b4660901224060fd2a9ef9eecd1a408ba5e693a65b",
    FP_CHECKPOINT: "3b673487a3f69dadbde6131521218335fd59a3542d679c5a85fc001ebf90b724",
    NORMALIZATION: "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    FORWARD_THRESHOLDS: "a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a",
    REPO / "src/models_two_expert_decoder.py": "9931c81b42aa4463ef9715223f768c787d40c373519043b68167645f7708f415",
    REPO / "src/scientific_alignment.py": "62c0f1f7704a50a66b16c0044df7e140b3fae71563f1fa7db895f1d260655b07",
    REPO / "src/competing_hypotheses.py": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d",
    REPO / "src/canonical_tensor_hash.py": "65566c01c5e6a76bc35e638423562180f370edb7b5b8bc5a3931ae2ca994bb6e",
    REPO / "src/coordinate_prompt.py": "a45b47ab2fb078d7624a4389a7634a6b1d898ade95d8de10899b6da590d82732",
    REPO / "src/prompt_semantics.py": "c604301d15e9254f6f2e72858710a2b8d035a466382d46281853f5065afac536",
    REPO / "src/feasibility_projection.py": "6bc81fa4dc3807b6163f604ed76bf81b006cba128fa6e3ab190344c0420387c9",
    REPO / "scripts/run_thayer_two_expert_micro_overfit.py": "69a4e862fdff54f4de7dc2564f35ccced29528317a266882d710b28b92ac7ec2",
    REPO / "scripts/train_thayer_feasibility_projection_micro.py": "30764623541a5bbb54e45aa96daa169ec9dc5c5399b637601a86ab4709d3e7e8",
    REPO / "docs/multi_hypothesis_source_contract.md": "dc3a78b65b2eda17b71887c7616189a24fbf1f367c8fb61014d6e291a2e02128",
    REPO / "docs/latent_truth_coverage.md": "4d2d53ea7ef77c09b263ee90dec50b7138b0dbe07f2b16150113c0041e589d97",
}

CONTRACT_HASH_ROLES = {
    "microset_manifest": MICROSET,
    "projected_target_tensor": P0_TARGETS,
    "projected_target_canonical_hash_table": P0_HASHES,
    "shared_prompted_encoder_and_linear_head": REPO / "src/models_two_expert_decoder.py",
    "coordinate_prompt_generation": REPO / "src/coordinate_prompt.py",
    "prompt_semantics": REPO / "src/prompt_semantics.py",
    "prompt_identity_truth_coverage_inverse_normalization": REPO / "scripts/run_thayer_two_expert_micro_overfit.py",
    "hard_projected_target_assignment": REPO / "scripts/train_thayer_feasibility_projection_micro.py",
    "scientific_threshold_surrogate": REPO / "src/scientific_alignment.py",
    "truth_coverage_and_forward_evaluation": REPO / "src/competing_hypotheses.py",
    "scientific_region_projection": REPO / "src/feasibility_projection.py",
    "canonical_tensor_hash": REPO / "src/canonical_tensor_hash.py",
    "source_layer_contract": REPO / "docs/multi_hypothesis_source_contract.md",
    "truth_coverage_contract": REPO / "docs/latent_truth_coverage.md",
    "inverse_normalization_scales": NORMALIZATION,
    "forward_evaluation_thresholds": FORWARD_THRESHOLDS,
}

SUBDIRS = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "output_contract", "manifests", "checkpoints", "conditions", "micro_overfit",
    "example_grids",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, check=True, text=True, capture_output=True).stdout.rstrip()


def write_text_fresh(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def checkpoint_inventory() -> list[dict[str, object]]:
    paths = sorted(
        path
        for path in (REPO / "outputs").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    return [
        {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def source_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.glob("**/*.py")):
            if path.is_file():
                rows.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def preregistration(started: str) -> str:
    return f"""# Contract-Compliant Decoder-Capacity Ladder preregistration

Working name: **Thayer-CL (Thayer Capacity Ladder)**  
Frozen at UTC: `{started}`  
Microset manifest SHA-256: `{EXPECTED[MICROSET]}`  
Frozen P0 target-file SHA-256: `{EXPECTED[P0_TARGETS]}`  
Frozen P0 canonical-hash table SHA-256: `{EXPECTED[P0_HASHES]}`

## Scope and ordering

This is a 64-row, training-only output-contract and decoder-width diagnosis. This freeze precedes every Thayer-CL per-scene tensor load, output-domain statistic, model construction, synthetic head fit, and neural optimizer step. Atlas, validation, calibration, development, and lockbox arrays are prohibited. Historical artifacts are immutable. Every new artifact is campaign-local, timestamped, append-only, and collision-refusing.

The campaign first reproduces Thayer-FP, then audits whether the already-frozen contracts identify exactly one physically nonnegative neural output mapping. **If no unique mapping exists, the campaign stops before architecture construction, synthetic fitting, sanity gates, or ladder training.** No mapping may be selected after observing any training result.

## Frozen rows, targets, and contracts

The exact 64 rows are those in the hashed Thayer-ME microset manifest: 32 ordinary rows followed by the 32 observations in the first 16 validated near-collision pairs. The only learning targets are the exact P0 tensors in `projection_targets/projected_target_sets_final.h5`; their per-sample canonical hashes, hard assignment, 0.95 construction slack, source-truth provenance, and non-input status are unchanged. Scientific acceptance remains componentwise image/0.25, g/r/z flux/0.20, applicable colors/0.20 mag, and centroid/(0.5 mean-PSF-FWHM) <=1.0. Prompt, source-layer, hard two-permutation assignment, canonical-hash, truth-coverage, and forward-evaluation implementations are frozen by the metadata snapshot.

Inference tensors remain normalized g/r/z blend plus the Condition-C Gaussian coordinate prompt. Truth values, constraint values, target indices, pair IDs, and projection metadata are forbidden inputs.

## Physical and normalized domains; output-mapping uniqueness gate

Normalized channels are requested g/r/z followed by companion g/r/z and are inverse-normalized only by multiplication with the positive frozen training scales. Physical source contributions must be finite and nonnegative, use the exact band order and 60 x 60 dimensions, and have zero-background semantics.

The historical Thayer-ME mapping is the identity from a linear six-channel head to normalized outputs, followed by scale multiplication. It is unclipped and permits physical negatives. The detached Thayer-OC clamp is an audit projection and is not a frozen neural output mapping. The audit must determine from pre-existing contracts—not training outcomes—whether exactly one mapping satisfies all of: identical across conditions, differentiable or subdifferentiable, represents zero and the full P0 range within frozen tolerance, has finite gradients near zero, introduces no implicit clipping, and makes physical negatives impossible. Zero eligible mappings or more than one mathematically eligible mapping fails uniqueness. No new mapping is chosen in this preregistration.

If uniqueness passes, exact-truth and P0 normalized-to-physical-to-normalized round trips must pass within the frozen float32 inversion tolerance `0.0009765625`, and coverage may not depend on post-hoc clipping. Any negative physical output causes a synchronous run-local stop record before further optimization.

## Shared encoder and prospective width-only ladder

Conditional on the output-mapping uniqueness gate, every condition uses the exact shared encoder `4->16->32->64`, the same two-convolution decoder blocks, 3x3 kernels, GroupNorm, SiLU, bilinear skips, independent expert seeds, and one six-channel 1x1 head. Only decoder block widths differ:

| Condition | dec2 | dec1 | Parameters/expert | Total system |
|---|---:|---:|---:|---:|
| L0 CURRENT | 32 | 16 | 46,470 | 165,612 |
| L1 MODERATE | 80 | 40 | 176,646 | 425,964 |
| L2 LARGE COMPACT | 160 | 80 | 554,886 | 1,182,444 |
| L3 ORIGINAL-SCALE CLASS | 224 | 112 | 1,002,630 | 2,077,932 |

No depth, attention, latent variable, expert count, input channel, encoder width, skip connection, activation family, normalization family, output mapping, or head semantics may vary.

## Initialization, objective, assignment, and budget

Expert seeds are fixed per condition as `(2026071201 + 1000*k, 2026071202 + 1000*k)` for condition index `k=0..3`; shared encoder warm-start tensors are exact Condition-C copies. The common batch-order seed is `2026071250`. The only objective is mean requested-source MSE plus companion-source MSE to P0 targets under the unchanged hard identity/swap minimum; ordinary rows supervise both slots. No scientific, forward, source-sum, prompt-swap, concentration, diversity, uncertainty, adversarial, perceptual, or Atlas loss is allowed.

If authorized, use MPS-only AdamW, learning rate `1e-3`, weight decay `0`, batch size `8`, no scheduler, gradient-norm clipping `5.0`, no augmentation, and exactly the same at-most-400-epoch budget and batch order for every condition. Evaluate epoch 0, after the first optimizer step, epoch 1, and every 20 epochs. Use one primary seed for all conditions; replicate only the first passing condition and the immediately smaller condition with seeds offset by `+10000` and `+20000`. No condition receives extra selection steps.

## Preflight and sanity gates

After uniqueness only, every condition must pass target-range, zero/low/high representability, nonsaturation, finite-gradient, and no-clipping checks. A head-only CPU diagnostic may fit zero, constant positive, sparse positive, and one frozen P0 crop. These are parameterization tests, not neural campaign training. Then MPS-only one-ordinary, one-ambiguous, and fixed four-ordinary-plus-four-ambiguous gates require nonnegativity from step zero, substantial loss reduction, one-scene truth coverage, ambiguous both-mode coverage, and prompt fidelity. If every size fails either one-scene gate, stop the campaign.

## Fail-closed rules and scientific gates

Stop an individual run synchronously on any negative physical output, nonfinite value, MPS fallback, target/hash mismatch, collision, source-layer violation, hard-assignment mismatch, or frozen prompt-collapse condition. Stop the campaign on shared-input drift, output-mapping drift, ignored stop logic, or protected-data access.

A condition passes only with zero negatives/nonfinites and no coverage clipping; ordinary set-level and both-expert coverage >=29/32 with median diameter <=1.0; ambiguous own, alternate, and both-mode coverage each >=29/32; set prompt swap >=0.90 with no systematic inversion; ordinary and ambiguous forward consistency each >=0.90; no catastrophic z-band failure; and required seed reproduction. Exact P0 targets demonstrate attainability of scientific coverage and diameter gates. Output-mapping uniqueness is not established by target feasibility and must pass independently.

## Interpretation and protected boundary

L0 pass attributes the prior failure primarily to output parameterization; first pass at L1/L2/L3 supports the prospectively named capacity interpretation; all-size failure implicates more than decoder count; seed instability leaves the threshold unresolved. Microset memorization is not generalization. Full non-Atlas training, Atlas, development, lockbox, and auditor work remain unauthorized in this run under every outcome.
"""


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = RUNS / f"{PREFIX}{stamp}"
    run.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRS:
        (run / name).mkdir(exist_ok=False)

    mismatches = []
    frozen_inputs = {}
    for path, expected in EXPECTED.items():
        observed = sha256(path) if path.is_file() else "MISSING"
        frozen_inputs[str(path.relative_to(REPO))] = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": observed,
        }
        if observed != expected:
            mismatches.append({"path": str(path.relative_to(REPO)), "expected": expected, "observed": observed})

    checkpoints = checkpoint_inventory()
    with (run / "tables/checkpoint_inventory_before.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(checkpoints)

    source_rows = source_inventory()
    write_json_fresh(run / "manifests/source_code_inventory.json", source_rows)
    git_status = command("git", "status", "--short")
    staged = command("git", "diff", "--cached", "--name-status")
    freeze = json.loads((FP / "projection_targets/freeze_record_final.json").read_text())
    packages = {}
    for name in ("torch", "numpy", "h5py", "scipy", "btk", "astropy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    try:
        import torch
        mps_available = bool(torch.backends.mps.is_available())
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - environment snapshot only.
        mps_available = False
        torch_version = f"unavailable: {exc}"

    provenance = {
        "campaign": "Thayer-CL Contract-Compliant Decoder-Capacity Ladder",
        "campaign_started_utc": started,
        "branch": command("git", "branch", "--show-current"),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status_short": git_status.splitlines(),
        "staged_index": staged.splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_version": torch_version,
        "mps_available": mps_available,
        "free_disk_bytes": shutil.disk_usage(REPO).free,
        "microset_manifest_sha256": sha256(MICROSET),
        "projected_target_file_sha256": sha256(P0_TARGETS),
        "projected_target_hash_table_sha256": sha256(P0_HASHES),
        "fp_checkpoint_sha256": sha256(FP_CHECKPOINT),
        "fp_target_freeze_record": freeze,
        "expert_initialization_seeds": [2026071201, 2026071202],
        "frozen_inputs": frozen_inputs,
        "frozen_contract_hashes": {
            role: {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in CONTRACT_HASH_ROLES.items()
        },
        "frozen_input_mismatches": mismatches,
        "source_code_file_count": len(source_rows),
        "historical_checkpoint_count": len(checkpoints),
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "per_scene_tensor_load_count_before_preregistration": 0,
        "neural_model_construction_count_before_preregistration": 0,
        "optimizer_step_count_before_preregistration": 0,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)

    environment = f"""# Environment snapshot

- Campaign start UTC: `{started}`
- Repository: `{REPO}`
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Python: `{sys.version.split()[0]}`
- PyTorch: `{torch_version}`
- MPS available: `{mps_available}`
- Historical checkpoint files: `{len(checkpoints)}`
- Free disk bytes: `{provenance['free_disk_bytes']}`
- Frozen-input mismatches: `{len(mismatches)}`
- Staged index entries: `{len(provenance['staged_index'])}`

The full dirty-worktree snapshot, package versions, frozen hashes, source-code inventory, and protected-data counters are in `logs/input_provenance.json` and `manifests/source_code_inventory.json`. No per-scene tensor was loaded while creating this snapshot.
"""
    write_text_fresh(run / "diagnostics/environment_snapshot.md", environment)

    contract = f"""# Thayer-CL campaign contract

The exact 64-row Thayer-ME microset and final P0 target file are read-only. The shared prompted encoder, prompt contract, hard assignment, source-layer ordering, inverse normalization, scientific thresholds, truth coverage, canonical hashing, and forward evaluation are frozen. Decoder width is the only permitted prospective architecture variable.

The physical-output mapping is a prerequisite, not a tunable ladder condition. The campaign must stop before any model construction or fitting unless existing frozen contracts select exactly one nonnegative mapping. A mathematically admissible mapping cannot be adopted merely because it trains well. Every physical negative is an immediate synchronous stop condition.

Atlas, development, lockbox, full-data training, auditor training, stage, commit, push, merge, delete, overwrite, and historical-checkpoint mutation are prohibited. Frozen-input mismatches: `{len(mismatches)}`.
"""
    write_text_fresh(run / "diagnostics/campaign_contract.md", contract)

    prereg = run / "preregistration/contract_compliant_decoder_capacity_ladder.md"
    write_text_fresh(prereg, preregistration(started))
    prereg_hash = sha256(prereg)
    frozen_at = datetime.now(timezone.utc).isoformat()
    write_json_fresh(run / "preregistration/freeze_record.json", {
        "campaign": "Thayer-CL",
        "frozen_at_utc": frozen_at,
        "preregistration_sha256": prereg_hash,
        "per_scene_tensor_load_count": 0,
        "neural_model_construction_count": 0,
        "optimizer_step_count": 0,
        "frozen_input_mismatch_count": len(mismatches),
        "status": "FAIL_CLOSED_INPUT_MISMATCH" if mismatches else "FROZEN_BEFORE_PER_SCENE_LOAD",
    })
    write_json_fresh(run / "logs/bootstrap_complete.json", {
        "run_dir": str(run.relative_to(REPO)),
        "preregistration_sha256": prereg_hash,
        "frozen_at_utc": frozen_at,
        "frozen_input_mismatches": mismatches,
        "status": "FAIL" if mismatches else "PASS",
    })
    print(json.dumps({"run_dir": str(run), "preregistration_sha256": prereg_hash, "mismatches": mismatches}, sort_keys=True))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
