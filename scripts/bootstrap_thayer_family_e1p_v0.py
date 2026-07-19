#!/usr/bin/env python3
"""Freeze a fresh append-only Family-E1P paired-prompt campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = REPO / "outputs/runs/thayer_family_e1_v0_20260714_214715"
HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
INPUTS = {
    "readme": REPO / "README.md",
    "family_e1_model": REPO / "src/family_e1.py",
    "family_e1p_instrumentation": REPO / "src/family_e1p.py",
    "family_e1p_runner": REPO / "scripts/run_thayer_family_e1p_v0.py",
    "prompt_constructor": REPO / "scripts/thayer_select_prompt_ablation_common.py",
    "original_family_e1_runner": REPO / "scripts/run_thayer_family_e1_v0.py",
    "original_family_e1_micro_results": ORIGINAL_RUN / "tables/micro_overfit_results.csv",
    "family_e1_training_selector": ORIGINAL_RUN / "manifests/training_manifest.csv",
    "upstream_training_manifest": HIERARCHICAL / "manifests/v2_r_training_scene_manifest.csv",
    "upstream_training_tensors": HIERARCHICAL / "manifests/v2_r_training_scenes.h5",
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
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout


def checkpoint_inventory() -> list[dict[str, object]]:
    rows = []
    for path in sorted((REPO / "outputs/runs").glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            rows.append(
                {
                    "relative_path": relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def main() -> None:
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged.strip():
        raise RuntimeError("staged index must be empty")
    for name, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    now = datetime.now(ZoneInfo("America/New_York"))
    run = REPO / "outputs/runs" / f"thayer_family_e1p_v0_{now:%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    utc = now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    inventory = checkpoint_inventory()
    git_status = command(["git", "status", "--short", "--branch"])

    preregistration = f"""# Thayer-Family-E1P-v0 preregistration

Frozen UTC: `{utc}`

Status: **FROZEN BEFORE MODEL CONSTRUCTION, OPTIMIZER CONSTRUCTION, TRAINING-TENSOR LOAD, OR FITTING**.

## Sole question

Can the unchanged Family-E1 architecture learn requested-source identity when every microset observation is presented exactly twice, once with the stored prompt and requested/companion targets and once with the companion-coordinate prompt and exchanged targets? This is a prompt-conditioning intervention only.

The prior Family-E1 runner already constructed exactly this paired tensor in `augmented_prompt_views` and its frozen preregistration explicitly required both prompt views. Therefore this campaign is a deterministic paired-prompt replication plus internal influence tracing, not a new optimization treatment. This fact must be part of the scientific disposition.

## Frozen model, map, objective, optimizer, and prompt

- Model source SHA-256: `{hashes['family_e1_model']}`; exact expected trainable parameters: `1,162,662`.
- Input remains normalized observed g/r/z plus the unchanged unit-peak sigma-2 Gaussian coordinate prompt.
- Output remains six raw channels mapped in `forward` to nonnegative requested and companion sources by ReLU and the signed residual `O-requested-companion`.
- Objective remains requested L1 `1.0`, companion L1 `1.0`, relative flux `0.25`, centroid `0.10`, and color `0.10`; no paired, contrastive, ordering, safety, or other new loss is added.
- MPS AdamW remains learning rate `3e-3`, weight decay `1e-4`, gradient clip `5.0`, full augmented micro-batch, and no scheduler.
- Prompt encoding, band scales, target semantics, conservation rule, architecture, parameter count, and all scientific thresholds remain unchanged.

## Frozen microsets and budgets

Only training-selector difficult index `6` for `2,000` updates with seed `2026071512`, and mixed indices `[0,3,5,6,18,51,73,81]` for `3,000` updates with seed `2026071513`. The ordinary scene, full training, validation, calibration, OOF, safety labels, auditor, development, Atlas selection, and lockbox are prohibited.

Each selected blend appears in ordered views `[A scenes, B scenes]`. The observed three-channel tensor must be byte-identical across a pair. Only the prompt channel changes. Requested and companion targets exchange exactly.

## Frozen measures

- Prompt identity is the unchanged strict rate at which normalized requested-prediction MSE is smaller to its requested target than to its companion target. Gate: `>=0.90` separately for difficult and mixed-eight.
- Prompt swap is the scene rate for which both A and B views pass identity.
- Requested-source error is mean band-normalized pixel L1.
- Companion leakage is the mean nonnegative companion coefficient fraction from a two-template least-squares decomposition of the requested prediction against requested and companion truths.
- Same-observation A/B comparisons are normalized-pixel L1 difference, normalized integrated-flux difference, soft-centroid distance in pixels, log-color difference, and cosine similarity. L1/flux/centroid/color response ratios use the corresponding true A/B difference as denominator.
- Encoder and decoder tracing reports paired prompt-activation RMS, feature modulation (prompt RMS / mean feature RMS), centered cross-correlation, pair cosine, and `0.5*log1p(prompt-difference power / feature power)` as a Gaussian-channel mutual-information proxy. The gradient measure is RMS of the gradient of each layer's mean-square activation with respect to the prompt input.
- Numerical indistinguishability is diagnostic only, not a new scientific gate: feature modulation `<=1e-6` and cross-correlation `>=0.999999`.

## Outcome

Success requires prompt identity `>=0.90` on both frozen microsets while the unchanged physical contract holds. Success authorizes exactly one experiment: resume Family-E1 full training with no change to architecture, parameter count, physical contract, signed residual, or objective. Failure authorizes no full training and must receive exactly one quantitative cause label from: Prompt ignored, Prompt diluted, Prompt forgotten, Prompt overwritten, Prompt too weak, Skip connections dominate, Decoder ignores prompt, or Other.

All campaign artifacts are collision-refusing and append-only. README, historical checkpoints, git index, and all prior run files remain unchanged; nothing is staged or committed.
"""
    prereg_path = run / "preregistration/family_e1p_paired_prompt_identity_intervention.md"
    fresh_text(prereg_path, preregistration)
    fresh_text(
        run / "preregistration/family_e1p_paired_prompt_identity_intervention.sha256",
        f"{sha256_file(prereg_path)}  {prereg_path.name}\n",
    )
    fresh_json(
        run / "logs/preregistration_complete.json",
        {
            "status": "FROZEN_BEFORE_MODEL_CONSTRUCTION_OR_FITTING",
            "frozen_utc": utc,
            "path": relative(prereg_path),
            "sha256": sha256_file(prereg_path),
            "model_construction_count": 0,
            "optimizer_construction_count": 0,
            "training_tensor_load_count": 0,
        },
    )
    fresh_json(
        run / "logs/input_provenance.json",
        {
            "campaign": "Thayer-Family-E1P-v0",
            "run_dir": relative(run),
            "campaign_start_local": now.isoformat(),
            "campaign_start_utc": utc,
            "git_head": command(["git", "rev-parse", "HEAD"]).strip(),
            "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
            "staged_index_empty": True,
            "authoritative_inputs": [
                {"name": name, "path": relative(path), "sha256": hashes[name]}
                for name, path in INPUTS.items()
            ],
            "historical_checkpoint_count": len(inventory),
            "historical_checkpoint_inventory_sha256": hashlib.sha256(
                json.dumps(inventory, sort_keys=True).encode()
            ).hexdigest(),
            "training_tensor_load_count": 0,
            "validation_access_count": 0,
            "calibration_access_count": 0,
            "development_access_count": 0,
            "atlas_selection_access_count": 0,
            "final_lockbox_access_count": 0,
        },
    )
    fresh_csv(run / "tables/historical_checkpoint_inventory_before.csv", inventory)
    fresh_text(run / "diagnostics/initial_git_status.txt", git_status)
    fresh_json(
        run / "logs/bootstrap_complete.json",
        {
            "status": "PASS",
            "run_dir": relative(run),
            "preregistration_sha256": sha256_file(prereg_path),
            "historical_checkpoint_count": len(inventory),
        },
    )
    print(relative(run))


if __name__ == "__main__":
    main()
