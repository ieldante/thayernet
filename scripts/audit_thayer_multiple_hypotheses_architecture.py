#!/usr/bin/env python3
"""Audit the preregistered Thayer-MH architecture before fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models_multiple_hypotheses import ThayerMultipleHypotheses, parameter_count, set_training_phase, warm_start_condition_c


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_CONDITION_C = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"


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
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    prereg = run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md"
    if sha256_file(prereg) != freeze["preregistration_sha256"] or any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("preregistration/checkpoint pre-fit gate failed")
    if sha256_file(CONDITION_C) != EXPECTED_CONDITION_C:
        raise RuntimeError("Condition-C checkpoint changed")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS-only architecture probe unavailable")
    model = ThayerMultipleHypotheses()
    warm = warm_start_condition_c(model, CONDITION_C)
    total = parameter_count(model)
    set_training_phase(model, 1); phase1 = parameter_count(model, trainable_only=True)
    set_training_phase(model, 2); phase2 = parameter_count(model, trainable_only=True)
    if total > 300_000:
        raise RuntimeError("parameter ceiling exceeded")
    model = model.to("mps").eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 60, 60, device="mps"), torch.zeros(1, 1, 60, 60, device="mps"))
    if output.shape != (1, 2, 6, 60, 60) or not bool(torch.isfinite(output).all().cpu()):
        raise RuntimeError("MPS architecture probe failed")
    write_csv_fresh(run_dir / "tables/condition_c_warm_start_inventory.csv", warm)
    write_csv_fresh(run_dir / "tables/model_parameter_inventory.csv", [
        {"model": "Condition C", "total_parameters": 119091, "phase1_trainable": 119091, "phase2_trainable": 119091, "ratio_to_condition_c": 1.0, "parameter_ceiling": 300000, "pass": True},
        {"model": "Thayer-MH", "total_parameters": total, "phase1_trainable": phase1, "phase2_trainable": phase2, "ratio_to_condition_c": total / 119091, "parameter_ceiling": 300000, "pass": total <= 300000},
    ])
    write_text_fresh(run_dir / "diagnostics/multiple_hypotheses_architecture.md", f"""# Thayer-MH architecture audit

Status: **PASS** before fitting.

- Total parameters: {total:,}; increase over Condition C: {total - 119091:,}; ratio {total / 119091:.6f}.
- Phase-1 / phase-2 trainable parameters: {phase1:,} / {phase2:,}.
- Parameter ceiling: 300,000.
- K=2 learned 8-dimensional tokens enter the shared 64-channel bottleneck and shared 32-channel late decoder stage.
- Both slots share every encoder, decoder, and output-head weight. Only token values distinguish them.
- Every compatible Condition-C backbone tensor loaded exactly; the 3-channel head initialized both six-channel halves.
- MPS output probe: finite `(1,2,6,60,60)`; CPU fallback disabled.

```text
blend g/r/z + coordinate prompt
              |
        shared Condition-C encoder
              |
       shared 64-channel bottleneck <--- token 1 or token 2
              |
          shared decoder trunk
              |
         late 32-channel stage <-------- same token
              |
       shared six-channel head
              |
   requested g/r/z + companion g/r/z
```
""")
    config = {"status": "FROZEN_ARCHITECTURE_AUDIT_PASS", "total_parameters": total, "phase1_trainable": phase1, "phase2_trainable": phase2, "condition_c_sha256": sha256_file(CONDITION_C), "k": 2, "token_dimension": 8, "mps_probe": "PASS", "parameter_ceiling": 300000}
    write_json_fresh(run_dir / "manifests/thayer_mh_architecture_pre_fit.json", config)
    write_json_fresh(run_dir / "logs/architecture_audit_complete.json", {**config, "warm_start_inventory_sha256": sha256_file(run_dir / "tables/condition_c_warm_start_inventory.csv")})
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
