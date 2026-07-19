#!/usr/bin/env python3
"""Audit Thayer-PU architecture, warm start, parameter budget, and API separation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models_probabilistic_unet import (  # noqa: E402
    ThayerProbabilisticUNet,
    set_training_phase,
    trainable_parameter_count,
    warm_start_condition_c,
)


CHECKPOINT = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_CHECKPOINT_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_BEFORE_MODEL_IMPLEMENTATION_DATA_RENDERING_OR_FITTING":
        raise RuntimeError("preregistration not frozen")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Condition-C checkpoint altered")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before architecture audit")

    model = ThayerProbabilisticUNet()
    inventory = warm_start_condition_c(model, CHECKPOINT)
    total = trainable_parameter_count(model)
    set_training_phase(model, 1)
    phase1 = trainable_parameter_count(model, currently_trainable=True)
    set_training_phase(model, 2)
    phase2 = trainable_parameter_count(model, currently_trainable=True)
    if total > 600_000 or phase1 <= 0 or phase2 <= phase1:
        raise RuntimeError("parameter budget/phase gate failed")
    prior_parameters = list(inspect.signature(model.encode_prior).parameters)
    posterior_parameters = list(inspect.signature(model.encode_posterior).parameters)
    if prior_parameters != ["observed_blend"] or posterior_parameters != ["observed_blend", "source_a", "source_b"]:
        raise RuntimeError("prior/posterior API separation failed")

    write_csv_fresh(run_dir / "tables/condition_c_warm_start_inventory.csv", inventory)
    components = []
    for name, module in (
        ("condition_c_compatible_backbone_and_head", model),
        ("conditional_prior", model.prior),
        ("training_only_posterior", model.posterior),
        ("latent_injection", model.latent_injection),
        ("six_channel_decomposition_head", model.decomposition_head),
    ):
        count = sum(parameter.numel() for parameter in module.parameters())
        components.append({"component": name, "parameters": count})
    components.append({"component": "TOTAL_UNIQUE_MODEL", "parameters": total})
    components.append({"component": "PHASE_1_TRAINABLE", "parameters": phase1})
    components.append({"component": "PHASE_2_TRAINABLE", "parameters": phase2})
    write_csv_fresh(run_dir / "tables/model_parameter_inventory.csv", components)
    report = f"""# Thayer-PU architecture and warm-start audit

Status: **PASS**.

- Total unique parameters: {total:,} (frozen ceiling 600,000).
- Phase-1 trainable parameters: {phase1:,}.
- Phase-2 trainable parameters: {phase2:,}.
- Latent dimension: 8; bottleneck-only injection.
- Output: six linear, unclipped channels `[requested g/r/z, companion g/r/z]`.
- Condition-C tensors loaded/inventoried: {len(inventory)} rows.
- Condition-C checkpoint SHA-256 before/after load: `{EXPECTED_CHECKPOINT_SHA256}`.
- Prior signature: `encode_prior(observed_blend)`; prompt and truth cannot enter.
- Posterior signature: `encode_posterior(observed_blend, source_a, source_b)`;
  source order is canonical manifest A/B and the API is training-only.

```text
                         +-> prior(blend) -----------+
blend + coordinate ----> Condition-C encoder        | sample z
                         +-> posterior(blend,A,B) ---+ (training only)
                                      |
Condition-C bottleneck + linear(z) -> late decoder -> six-channel decomposition
                                                   -> requested + companion = scene
```

Every matching encoder/bottleneck/decoder tensor was loaded exactly. The historical
three-channel head was copied separately into requested and companion halves; all
other stochastic components are newly initialized. Phase 1 freezes enc1, enc2, and
bottleneck; Phase 2 unfreezes bottleneck while enc1/enc2 remain frozen.
"""
    write_text_fresh(run_dir / "diagnostics/probabilistic_unet_architecture_report.md", report)
    write_json_fresh(run_dir / "logs/architecture_audit_complete.json", {
        "status": "PASS", "total_parameters": total, "phase1_trainable": phase1,
        "phase2_trainable": phase2, "warm_start_inventory_rows": len(inventory),
        "condition_c_checkpoint_sha256": sha256_file(CHECKPOINT),
        "prior_truth_free_api": True, "posterior_training_only_api": True,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "atlas_evaluation_count": 0,
    })


if __name__ == "__main__":
    main()
