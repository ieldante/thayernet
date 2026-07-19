#!/usr/bin/env python3
"""Audit the preregistered Thayer-ME architecture before micro fitting."""

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

from src.models_two_expert_decoder import ThayerMixtureExperts, expert_parameter_distance, parameter_count, set_training_phase, warm_start_condition_c_encoder


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
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    prereg = run_dir / "preregistration/two_expert_ambiguity_decoder.md"
    if sha256_file(prereg) != freeze["preregistration_sha256"] or any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("preregistration/checkpoint pre-fit gate failed")
    if sha256_file(CONDITION_C) != EXPECTED_CONDITION_C:
        raise RuntimeError("Condition-C checkpoint changed")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS-only architecture probe unavailable")
    model = ThayerMixtureExperts()
    warm = warm_start_condition_c_encoder(model, CONDITION_C)
    shared = parameter_count(model.encoder)
    expert_1 = parameter_count(model.expert_1)
    expert_2 = parameter_count(model.expert_2)
    total = parameter_count(model)
    set_training_phase(model, 1)
    phase1 = parameter_count(model, trainable_only=True)
    set_training_phase(model, 2)
    phase2 = parameter_count(model, trainable_only=True)
    left_storage = {parameter.data_ptr() for parameter in model.expert_1.parameters()}
    right_storage = {parameter.data_ptr() for parameter in model.expert_2.parameters()}
    distance = float(expert_parameter_distance(model))
    if (shared, expert_1, expert_2, total) != (72672, 46470, 46470, 165612):
        raise RuntimeError("parameter inventory differs from preregistration")
    if total > freeze["parameter_ceiling"] or left_storage & right_storage or not distance > 0:
        raise RuntimeError("expert independence or parameter gate failed")
    model = model.to("mps").eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 60, 60, device="mps"), torch.zeros(1, 1, 60, 60, device="mps"))
    if output.shape != (1, 2, 6, 60, 60) or not bool(torch.isfinite(output).all().cpu()):
        raise RuntimeError("MPS architecture probe failed")
    write_csv_fresh(run_dir / "tables/condition_c_warm_start_inventory.csv", warm)
    write_csv_fresh(run_dir / "tables/model_parameter_inventory.csv", [
        {"component": "shared_encoder", "parameters": shared, "ratio_to_condition_c": shared / 119091, "independent": False},
        {"component": "expert_1", "parameters": expert_1, "ratio_to_condition_c": expert_1 / 119091, "independent": True},
        {"component": "expert_2", "parameters": expert_2, "ratio_to_condition_c": expert_2 / 119091, "independent": True},
        {"component": "Thayer-ME total", "parameters": total, "ratio_to_condition_c": total / 119091, "independent": True},
        {"component": "Thayer-MH total", "parameters": 120022, "ratio_to_condition_c": 120022 / 119091, "independent": False},
    ])
    write_text_fresh(run_dir / "diagnostics/two_expert_architecture.md", f"""# Thayer-ME architecture audit

Status: **PASS BEFORE FITTING**.

- Shared encoder / expert 1 / expert 2 parameters: {shared:,} / {expert_1:,} / {expert_2:,}.
- Total parameters: {total:,}; ratio to Condition C {total / 119091:.6f}; ratio to Thayer-MH {total / 120022:.6f}.
- Phase-1 / phase-2 trainable parameters: {phase1:,} / {phase2:,}.
- Frozen ceiling: {freeze['parameter_ceiling']:,}.
- Compatible Condition-C encoder tensors loaded exactly; no decoder or output-head tensor was warm-started.
- Expert decoder storage intersection: empty. Initialization parameter distance: {distance:.6f}.
- Frozen initialization seeds: 2026071201 / 2026071202.
- MPS finite-output probe: `(1,2,6,60,60)`; fallback disabled.

```text
blend g/r/z + coordinate prompt
              |
   shared Condition-C encoder
       /                  \
independent expert 1   independent expert 2
       |                  |
 requested+companion   requested+companion
```
""")
    record = {"status": "FROZEN_ARCHITECTURE_AUDIT_PASS", "shared_parameters": shared, "expert_1_parameters": expert_1, "expert_2_parameters": expert_2, "total_parameters": total, "phase1_trainable": phase1, "phase2_trainable": phase2, "expert_parameter_storage_overlap": 0, "expert_initial_parameter_distance": distance, "mps_probe": "PASS", "parameter_ceiling": freeze["parameter_ceiling"]}
    write_json_fresh(run_dir / "manifests/thayer_me_architecture_pre_fit.json", record)
    write_json_fresh(run_dir / "logs/architecture_audit_complete.json", {**record, "warm_start_inventory_sha256": sha256_file(run_dir / "tables/condition_c_warm_start_inventory.csv")})
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
