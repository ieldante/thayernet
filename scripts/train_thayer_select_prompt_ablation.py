#!/usr/bin/env python3
"""Train the three aligned compact prompt-ablation conditions on MPS only."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from thayer_select_prompt_ablation_common import (
    BANDS,
    CompactSelectNet,
    PROMPT_SIGMA_PIXELS,
    TRAINING_SEED,
    gaussian_prompt_numpy,
    load_scales,
    parameter_count,
    read_csv,
    require_mps,
    seed_everything,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
)

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16
CONDITIONS = {
    "A_centered_no_prompt": {"variant": "centered", "prompted": False, "in_channels": 3},
    "B_randomized_no_prompt": {"variant": "random", "prompted": False, "in_channels": 3},
    "C_randomized_coordinate_prompt": {"variant": "random", "prompted": True, "in_channels": 4},
}


class SceneDataset(Dataset):
    def __init__(self, run_dir: Path, partition: str, variant: str, prompted: bool, scales: np.ndarray) -> None:
        if partition not in ("training", "validation"):
            raise ValueError("Training code may only open training or validation data")
        self.path = run_dir / f"manifests/{partition}_scenes.h5"
        self.rows = [row for row in read_csv(run_dir / "manifests/development_scene_definitions.csv") if row["partition"] == partition]
        self.variant = variant
        self.prompted = prompted
        self.scales = np.asarray(scales, dtype=np.float32)
        self.handle = None
        with h5py.File(self.path, "r") as handle:
            if not bool(handle.attrs["complete"]) or len(handle[f"{variant}_blend"]) != len(self.rows):
                raise RuntimeError(f"Incomplete {partition} data")

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __getitem__(self, index: int):
        handle = self._handle()
        target_index = int(self.rows[index]["target_index"])
        blend = np.asarray(handle[f"{self.variant}_blend"][index], dtype=np.float32) / self.scales[:, None, None]
        target = np.asarray(handle[f"{self.variant}_isolated"][index, target_index], dtype=np.float32) / self.scales[:, None, None]
        if self.prompted:
            xy = np.asarray(handle[f"{self.variant}_xy"][index, target_index], dtype=np.float64)
            prompt = gaussian_prompt_numpy(float(xy[0]), float(xy[1]), sigma_pixels=PROMPT_SIGMA_PIXELS)
            model_input = np.concatenate((blend, prompt[None]), axis=0)
        else:
            model_input = blend
        return torch.from_numpy(np.ascontiguousarray(model_input)), torch.from_numpy(np.ascontiguousarray(target))


def fit_normalization(run_dir: Path) -> np.ndarray:
    path = run_dir / "manifests/normalization.json"
    if path.exists():
        return load_scales(run_dir)
    with h5py.File(run_dir / "manifests/training_scenes.h5", "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("Training render is incomplete")
        blends = np.asarray(handle["random_blend"], dtype=np.float32)
    scales = np.quantile(np.abs(blends), 0.995, axis=(0, 2, 3)).astype(np.float32)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise RuntimeError("Invalid training-only normalization")
    probe = blends[:16] / scales[None, :, None, None]
    restored = probe * scales[None, :, None, None]
    max_error = float(np.max(np.abs(restored - blends[:16])))
    write_json_fresh(path, {
        "fit_partition": "training only",
        "fit_scene_variant": "randomized shared by Conditions B/C and applied unchanged to A",
        "quantile": 0.995,
        "bands": list(BANDS),
        "per_band_scale": scales.tolist(),
        "clipping": False,
        "negative_pixels_preserved": True,
        "maximum_float32_inversion_error": max_error,
    })
    return scales


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def condition_paths(run_dir: Path, condition: str) -> dict[str, Path]:
    stem = condition.lower()
    return {
        "best": run_dir / f"checkpoints/{stem}_best.pth",
        "final": run_dir / f"checkpoints/{stem}_final.pth",
        "curve": run_dir / f"tables/{stem}_epochs.csv",
        "config": run_dir / f"manifests/{stem}_training_config.json",
    }


def train_condition(run_dir: Path, condition: str, config: dict, scales: np.ndarray, device: torch.device) -> dict:
    paths = condition_paths(run_dir, condition)
    existing = [name for name, path in paths.items() if path.exists()]
    if existing:
        if len(existing) != len(paths):
            raise RuntimeError(f"Partial/colliding outputs for {condition}: {existing}")
        frozen = json.loads(paths["config"].read_text())
        if frozen.get("status") != "FROZEN":
            raise RuntimeError(f"Existing condition is not frozen: {condition}")
        print(f"{condition}: already frozen", flush=True)
        return frozen

    seed_everything(TRAINING_SEED)
    model = CompactSelectNet(config["in_channels"], BASE_CHANNELS).to(device)
    if next(model.parameters()).device.type != "mps":
        raise RuntimeError("Model is not on MPS")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_function = nn.MSELoss()
    train_dataset = SceneDataset(run_dir, "training", config["variant"], config["prompted"], scales)
    validation_dataset = SceneDataset(run_dir, "validation", config["variant"], config["prompted"], scales)
    order_generator = torch.Generator().manual_seed(TRAINING_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=order_generator, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_loss = math.inf
    best_epoch = -1
    best_state = None
    epochs = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for model_input, target in train_loader:
            model_input = model_input.to(device)
            target = target.to(device)
            if model_input.device.type != "mps" or target.device.type != "mps":
                raise RuntimeError("Unexpected device fallback")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(model_input)
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in {condition}")
            loss.backward()
            gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            if not gradients_finite:
                raise RuntimeError(f"Non-finite gradient in {condition}")
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(model_input)
            train_count += len(model_input)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for model_input, target in validation_loader:
                model_input = model_input.to(device)
                target = target.to(device)
                if model_input.device.type != "mps":
                    raise RuntimeError("Unexpected validation device fallback")
                validation_loss = loss_function(model(model_input), target)
                if not torch.isfinite(validation_loss):
                    raise RuntimeError(f"Non-finite validation loss in {condition}")
                validation_sum += float(validation_loss.detach().cpu()) * len(model_input)
                validation_count += len(model_input)
        record = {
            "condition": condition,
            "epoch": epoch,
            "training_loss": train_sum / train_count,
            "validation_loss": validation_sum / validation_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "device": "mps",
        }
        epochs.append(record)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        scheduler.step()
        print(json.dumps(record, sort_keys=True), flush=True)
    if best_state is None or best_epoch < 1:
        raise RuntimeError("No best validation checkpoint selected")
    final_state = cpu_state_dict(model)
    elapsed = time.time() - started
    final_payload = {
        "state_dict": final_state, "condition": condition, "epoch": EPOCHS,
        "selection": "final epoch", "training_seed": TRAINING_SEED,
        "input_channels": config["in_channels"], "base_channels": BASE_CHANNELS,
    }
    best_payload = {
        "state_dict": best_state, "condition": condition, "epoch": best_epoch,
        "selection": "minimum validation MSE only", "validation_loss": best_loss,
        "training_seed": TRAINING_SEED, "input_channels": config["in_channels"], "base_channels": BASE_CHANNELS,
    }
    torch.save(final_payload, paths["final"])
    torch.save(best_payload, paths["best"])
    write_csv_fresh(paths["curve"], epochs)
    frozen = {
        "status": "FROZEN", "condition": condition, "condition_label": condition,
        "scene_variant": config["variant"], "coordinate_prompt": config["prompted"],
        "input_channels": config["in_channels"], "parameter_count": parameter_count(config["in_channels"]),
        "architecture": "compact Thayer U-Net backbone, reconstruction head only",
        "adaptation": "uncertainty/recoverability heads omitted for the reconstruction-only baseline",
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "CosineAnnealingLR",
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "training_seed": TRAINING_SEED,
        "data_order_seed": TRAINING_SEED, "loss": "whole-image normalized MSE only",
        "training_scenes": len(train_dataset), "validation_scenes": len(validation_dataset),
        "calibration_scenes_used": 0, "development_test_scenes_inspected": 0, "lockbox_scenes_used": 0,
        "device": "mps", "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "runtime_seconds": elapsed,
        "best_checkpoint": str(paths["best"].relative_to(REPO)), "best_checkpoint_sha256": sha256_file(paths["best"]),
        "final_checkpoint": str(paths["final"].relative_to(REPO)), "final_checkpoint_sha256": sha256_file(paths["final"]),
        "validation_rule": "minimum validation MSE; no test or calibration information",
        "normalization_manifest_sha256": sha256_file(run_dir / "manifests/normalization.json"),
        "scene_definition_sha256": sha256_file(run_dir / "manifests/development_scene_definitions.csv"),
    }
    write_json_fresh(paths["config"], frozen)
    return frozen


def plot_curves(run_dir: Path) -> None:
    path = run_dir / "figures/training_curves.png"
    if path.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for condition in CONDITIONS:
        rows = read_csv(condition_paths(run_dir, condition)["curve"])
        epoch = [int(row["epoch"]) for row in rows]
        axes[0].plot(epoch, [float(row["training_loss"]) for row in rows], label=condition)
        axes[1].plot(epoch, [float(row["validation_loss"]) for row in rows], label=condition)
    axes[0].set(title="Training MSE", xlabel="epoch", ylabel="normalized MSE", yscale="log")
    axes[1].set(title="Validation MSE", xlabel="epoch", ylabel="normalized MSE", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "logs/data_preparation_complete.json").is_file():
        raise RuntimeError("Data preparation/replay gate is incomplete")
    if json.loads((run_dir / "logs/data_preparation_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("Data preparation did not pass")
    device = require_mps()
    scales = fit_normalization(run_dir)
    parameters = {
        "A_centered_no_prompt": parameter_count(3),
        "B_randomized_no_prompt": parameter_count(3),
        "C_randomized_coordinate_prompt": parameter_count(4),
        "prompt_first_layer_parameter_difference": parameter_count(4) - parameter_count(3),
    }
    parameter_path = run_dir / "manifests/model_parameter_counts.json"
    if not parameter_path.exists():
        write_json_fresh(parameter_path, parameters)
    summaries = []
    for condition, config in CONDITIONS.items():
        summaries.append(train_condition(run_dir, condition, config, scales, device))
    plot_curves(run_dir)
    completion = run_dir / "logs/training_complete.json"
    if not completion.exists():
        write_json_fresh(completion, {
            "status": "PASS", "device": "mps", "conditions": list(CONDITIONS),
            "all_epochs_completed": all(item["epochs"] == EPOCHS for item in summaries),
            "development_test_inspected_before_freeze": False, "lockbox_accessed": False,
            "completed_at_unix": time.time(),
        })


if __name__ == "__main__":
    main()
