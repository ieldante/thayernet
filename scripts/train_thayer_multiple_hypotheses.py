#!/usr/bin/env python3
"""Train the preregistered Thayer-MH K=2 decoder on MPS only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.evaluate_probabilistic_unet_pre_atlas import prompts  # noqa: E402
from src.models_multiple_hypotheses import (  # noqa: E402
    ThayerMultipleHypotheses,
    parameter_count,
    permutation_invariant_target_loss,
    prompt_swap_set_loss,
    set_training_phase,
    source_sum,
    unordered_set_distance,
    warm_start_condition_c,
)


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
SEED = 2026079601
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_fresh(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("MPS fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU fallback prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def take(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    order = np.argsort(indices)
    inverse = np.argsort(order)
    return np.asarray(dataset[indices[order].tolist()])[inverse]


class PartitionData:
    def __init__(self, run_dir: Path, partition: str) -> None:
        self.rows = [row for row in read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv") if row["partition"] == partition]
        self.scene = h5py.File(run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5", "r")
        self.target = h5py.File(run_dir / f"target_sets/thayer_mh_{partition}_target_sets.h5", "r")
        if not bool(self.scene.attrs["complete"]) or not bool(self.target.attrs["complete"]):
            raise RuntimeError(f"incomplete {partition} data")
        self.ordinary = np.asarray([i for i, row in enumerate(self.rows) if row["kind"] == "ordinary"], dtype=np.int64)
        by_pair: dict[str, list[int]] = {}
        for i, row in enumerate(self.rows):
            if row["kind"] == "near_collision":
                by_pair.setdefault(row["near_collision_pair_id"], []).append(i)
        self.pairs = np.asarray([sorted(values) for _, values in sorted(by_pair.items())], dtype=np.int64)
        if any(len(values) != 2 for values in by_pair.values()):
            raise RuntimeError("equivalence classes must contain exactly two scenes")

    def close(self) -> None:
        self.scene.close(); self.target.close()

    def batch(self, indices: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray]:
        blend = take(self.scene["blend"], indices).astype(np.float32) / scales[None, :, None, None]
        xy = take(self.scene["xy"], indices).astype(np.float64)
        prompt_a, prompt_b = prompts(xy)
        targets = take(self.target["targets"], indices).astype(np.float32)
        targets /= np.tile(scales, 2)[None, None, None, :, None, None]
        counts = take(self.target["target_count"], indices).astype(np.int64)
        return {"blend": blend, "prompt_a": prompt_a, "prompt_b": prompt_b, "targets": targets, "counts": counts}


def loss_batch(model: ThayerMultipleHypotheses, arrays: dict[str, np.ndarray], device: torch.device, *, pair_tail: bool) -> tuple[torch.Tensor, dict[str, float]]:
    blend = torch.from_numpy(np.ascontiguousarray(arrays["blend"])).to(device)
    prompt_a = torch.from_numpy(np.ascontiguousarray(arrays["prompt_a"])).to(device)
    prompt_b = torch.from_numpy(np.ascontiguousarray(arrays["prompt_b"])).to(device)
    targets = torch.from_numpy(np.ascontiguousarray(arrays["targets"])).to(device)
    counts = torch.from_numpy(np.ascontiguousarray(arrays["counts"])).to(device)
    joined_blend = torch.cat((blend, blend), dim=0)
    joined_prompt = torch.cat((prompt_a, prompt_b), dim=0)
    joined_output = model(joined_blend, joined_prompt)
    output_a, output_b = joined_output[: len(blend)], joined_output[len(blend):]
    set_a = permutation_invariant_target_loss(output_a, targets[:, 0], counts[:, 0])
    set_b = permutation_invariant_target_loss(output_b, targets[:, 1], counts[:, 1])
    forward = 0.5 * ((source_sum(output_a) - blend[:, None]).square().mean() + (source_sum(output_b) - blend[:, None]).square().mean())
    swap = prompt_swap_set_loss(output_a, output_b)
    if pair_tail:
        pair_consistency = 0.5 * (unordered_set_distance(output_a[-2:-1], output_a[-1:]).mean() + unordered_set_distance(output_b[-2:-1], output_b[-1:]).mean())
    else:
        pair_consistency = torch.zeros((), device=device)
    total = 0.5 * (set_a["loss"] + set_b["loss"]) + 0.5 * forward + 0.25 * swap + 0.05 * pair_consistency
    ordinary_mask = counts[:, 0] == 1
    ambiguous_mask = counts[:, 0] == 2
    diameter = (output_a[:, 0] - output_a[:, 1]).square().mean(dim=(1, 2, 3))
    metrics = {
        "loss": float(total.detach().cpu()), "target_set": float((0.5 * (set_a["loss"] + set_b["loss"])).detach().cpu()),
        "forward": float(forward.detach().cpu()), "prompt_swap": float(swap.detach().cpu()), "pair_consistency": float(pair_consistency.detach().cpu()),
        "ordinary_diameter": float(diameter[ordinary_mask].mean().detach().cpu()) if bool(ordinary_mask.any().cpu()) else math.nan,
        "ambiguous_diameter": float(diameter[ambiguous_mask].mean().detach().cpu()) if bool(ambiguous_mask.any().cpu()) else math.nan,
    }
    return total, metrics


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


def train_batches(data: PartitionData, rng: np.random.Generator) -> list[np.ndarray]:
    ordinary = rng.permutation(data.ordinary)
    pairs = data.pairs[rng.permutation(len(data.pairs))]
    needed = 6 * len(pairs)
    if len(ordinary) < needed:
        ordinary = np.concatenate((ordinary, rng.choice(data.ordinary, needed - len(ordinary), replace=False)))
    return [np.concatenate((ordinary[6 * i:6 * i + 6], pair)) for i, pair in enumerate(pairs)]


def validation_batches(data: PartitionData) -> list[np.ndarray]:
    if len(data.ordinary) != 1500 or len(data.pairs) != 250:
        raise RuntimeError("frozen validation cardinality mismatch")
    return [np.concatenate((data.ordinary[6 * i:6 * i + 6], pair)) for i, pair in enumerate(data.pairs)]


def checkpoint_payload(model: ThayerMultipleHypotheses, optimizer: torch.optim.Optimizer, epoch: int, row: dict[str, object], run_dir: Path) -> dict[str, object]:
    return {
        "model": "Thayer-MH", "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "metrics": row, "seed": SEED,
        "preregistration_sha256": sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md"),
        "target_inventory_sha256": sha256_file(run_dir / "tables/target_set_inventory.csv"),
        "condition_c_sha256": sha256_file(CONDITION_C), "parameter_count": parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    target_log = json.loads((run_dir / "logs/target_sets_complete.json").read_text())
    architecture = json.loads((run_dir / "logs/architecture_audit_complete.json").read_text())
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if target_log["status"] != "PASS" or architecture["status"] != "FROZEN_ARCHITECTURE_AUDIT_PASS":
        raise RuntimeError("data/architecture gate failed")
    if sha256_file(run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md") != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration changed")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint collision before fitting")
    device = require_mps()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    training = PartitionData(run_dir, "training"); validation = PartitionData(run_dir, "validation")
    model = ThayerMultipleHypotheses()
    warm = warm_start_condition_c(model, CONDITION_C)
    if len(warm) != len(read_csv(run_dir / "tables/condition_c_warm_start_inventory.csv")):
        raise RuntimeError("warm-start inventory mismatch")
    model = model.to(device)
    set_training_phase(model, 1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    epochs: list[dict[str, object]] = []
    best_epoch = -1; best_value = math.inf
    collapse_streak = 0; started = time.time()
    config = {"status": "FROZEN_BEFORE_FIT", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "ordinary_ambiguous_ratio": "6:2 observations per batch", "condition_c_sha256": sha256_file(CONDITION_C), "target_inventory_sha256": sha256_file(run_dir / "tables/target_set_inventory.csv"), "device": "mps", "fallback": False}
    config_path = run_dir / "manifests/thayer_mh_training_config_pre_fit.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("persisted pre-fit training config mismatch")
    else:
        write_json_fresh(config_path, config)
    try:
        for epoch in range(1, EPOCHS + 1):
            phase = 1 if epoch <= 5 else 2
            if epoch == 6:
                set_training_phase(model, 2)
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            model.train(); train_rows: list[dict[str, float]] = []
            for indices in train_batches(training, rng):
                arrays = training.batch(indices, scales)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_batch(model, arrays, device, pair_tail=True)
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise RuntimeError("NaN/Inf training loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                train_rows.append(metrics)
            model.eval(); validation_rows: list[dict[str, float]] = []
            with torch.no_grad():
                for indices in validation_batches(validation):
                    _, metrics = loss_batch(model, validation.batch(indices, scales), device, pair_tail=True)
                    validation_rows.append(metrics)
            train_mean = aggregate(train_rows); validation_mean = aggregate(validation_rows)
            if not all(math.isfinite(value) for value in (*train_mean.values(), *validation_mean.values())):
                raise RuntimeError("non-finite epoch metric")
            if validation_mean["ambiguous_diameter"] <= 1e-8:
                collapse_streak += 1
            else:
                collapse_streak = 0
            if collapse_streak >= 3:
                raise RuntimeError("both hypotheses collapsed on ambiguous validation for three epochs")
            row: dict[str, object] = {"epoch": epoch, "phase": phase, "trainable_parameters": parameter_count(model, trainable_only=True), **{f"train_{key}": value for key, value in train_mean.items()}, **{f"validation_{key}": value for key, value in validation_mean.items()}, "elapsed_seconds": time.time() - started, "device": "mps", "fallback": False}
            epochs.append(row)
            save_fresh(run_dir / f"checkpoints/thayer_mh_epoch_{epoch:02d}.pth", checkpoint_payload(model, optimizer, epoch, row, run_dir))
            if validation_mean["loss"] < best_value:
                best_value = validation_mean["loss"]; best_epoch = epoch
            print(json.dumps({"epoch": epoch, "phase": phase, "train_loss": train_mean["loss"], "validation_loss": validation_mean["loss"], "ordinary_diameter": validation_mean["ordinary_diameter"], "ambiguous_diameter": validation_mean["ambiguous_diameter"], "best_epoch": best_epoch, "elapsed_seconds": time.time() - started}, sort_keys=True), flush=True)
    finally:
        training.close(); validation.close()
    if best_epoch < 1:
        raise RuntimeError("no best checkpoint selected")
    best_payload = torch.load(run_dir / f"checkpoints/thayer_mh_epoch_{best_epoch:02d}.pth", map_location="cpu", weights_only=False)
    final_payload = torch.load(run_dir / f"checkpoints/thayer_mh_epoch_{EPOCHS:02d}.pth", map_location="cpu", weights_only=False)
    save_fresh(run_dir / "checkpoints/thayer_mh_best.pth", best_payload)
    save_fresh(run_dir / "checkpoints/thayer_mh_final.pth", final_payload)
    write_csv_fresh(run_dir / "tables/thayer_mh_epochs.csv", epochs)
    write_json_fresh(run_dir / "logs/training_complete.json", {"status": "PASS", "best_epoch": best_epoch, "best_validation_objective": best_value, "epoch_count": EPOCHS, "runtime_seconds": time.time() - started, "best_sha256": sha256_file(run_dir / "checkpoints/thayer_mh_best.pth"), "final_sha256": sha256_file(run_dir / "checkpoints/thayer_mh_final.pth"), "mps_only": True, "fallback": False, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0})
    print(json.dumps({"status": "PASS", "best_epoch": best_epoch, "runtime_seconds": time.time() - started}, sort_keys=True))


if __name__ == "__main__":
    main()
