#!/usr/bin/env python3
"""Train the preregistered prompted ResUNet on MPS only."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.models_prompted_resunet import PromptedResUNet, trainable_parameter_count  # noqa: E402


NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TRAINING_SEED = 2026077301
EXPECTED_PARAMETERS = 199_219


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


class ResUNetDataset(Dataset):
    def __init__(self, run_dir: Path, partition: str, scales: np.ndarray) -> None:
        if partition not in {"training", "validation"}:
            raise ValueError("training code may only open train/validation")
        self.path = run_dir / f"manifests/resunet_{partition}_scenes.h5"
        self.rows = [row for row in read_csv(run_dir / "manifests/resunet_scene_definitions.csv") if row["partition"] == partition]
        self.scales = np.asarray(scales, dtype=np.float32)
        self.handle = None
        with h5py.File(self.path, "r") as handle:
            if not bool(handle.attrs["complete"]) or len(handle["blend"]) != len(self.rows):
                raise RuntimeError(f"incomplete {partition} manifest")

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __getitem__(self, index: int):
        handle = self._handle()
        target_index = int(self.rows[index]["target_index"])
        blend = np.asarray(handle["blend"][index], dtype=np.float32) / self.scales[:, None, None]
        target = np.asarray(handle["isolated"][index, target_index], dtype=np.float32) / self.scales[:, None, None]
        xy = np.asarray(handle["xy"][index, target_index], dtype=np.float64)
        prompt = gaussian_prompt_numpy(float(xy[0]), float(xy[1]))
        model_input = np.concatenate((blend, prompt[None]), axis=0)
        return torch.from_numpy(np.ascontiguousarray(model_input)), torch.from_numpy(np.ascontiguousarray(target))


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def plot_curves(path: Path, rows: list[dict[str, object]]) -> None:
    epoch = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epoch, [float(row["training_loss"]) for row in rows], color="#315f8c")
    axes[1].plot(epoch, [float(row["validation_loss"]) for row in rows], color="#a65141")
    axes[0].set(title="Prompted ResUNet training", xlabel="epoch", ylabel="normalized MSE", yscale="log")
    axes[1].set(title="Non-Atlas validation", xlabel="epoch", ylabel="normalized MSE", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.name.startswith("thayer_prompted_resunet_diversity_"):
        raise RuntimeError("unexpected run directory")
    preparation = json.loads((run_dir / "logs/data_preparation_complete.json").read_text())
    if preparation["status"] != "PASS" or preparation["full_replay_count"] != 11_500:
        raise RuntimeError("data/replay gate failed")
    paths = {
        "best": run_dir / "checkpoints/prompted_resunet_best.pth",
        "final": run_dir / "checkpoints/prompted_resunet_final.pth",
        "epochs": run_dir / "tables/prompted_resunet_epochs.csv",
        "config": run_dir / "manifests/prompted_resunet_training_config.json",
        "figure": run_dir / "figures/training_curves.png",
        "complete": run_dir / "logs/training_complete.json",
    }
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"training-output collision: {collisions}")
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise RuntimeError("invalid frozen normalization")
    device = require_mps()
    seed_everything(TRAINING_SEED)
    model = PromptedResUNet().to(device)
    if trainable_parameter_count(model) != EXPECTED_PARAMETERS or next(model.parameters()).device.type != "mps":
        raise RuntimeError("model architecture/device gate failed")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_function = nn.MSELoss()
    training = ResUNetDataset(run_dir, "training", scales)
    validation = ResUNetDataset(run_dir, "validation", scales)
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    epoch_rows: list[dict[str, object]] = []
    unstable_consecutive = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_sum = 0.0
        training_count = 0
        for model_input, target in train_loader:
            model_input = model_input.to(device)
            target = target.to(device)
            if model_input.device.type != "mps" or target.device.type != "mps":
                raise RuntimeError("training device fallback")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(model_input)
            if prediction.device.type != "mps":
                raise RuntimeError("model output device fallback")
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            training_sum += float(loss.detach().cpu()) * len(model_input)
            training_count += len(model_input)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for model_input, target in validation_loader:
                model_input = model_input.to(device)
                target = target.to(device)
                prediction = model(model_input)
                if model_input.device.type != "mps" or prediction.device.type != "mps":
                    raise RuntimeError("validation device fallback")
                loss = loss_function(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite validation loss")
                validation_sum += float(loss.detach().cpu()) * len(model_input)
                validation_count += len(model_input)
        row = {
            "epoch": epoch,
            "training_loss": training_sum / training_count,
            "validation_loss": validation_sum / validation_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "device": "mps",
        }
        epoch_rows.append(row)
        if float(row["validation_loss"]) < best_loss:
            best_loss = float(row["validation_loss"])
            best_epoch = epoch
            best_state = cpu_state(model)
        if epoch > 1 and float(row["validation_loss"]) > 10.0 * float(epoch_rows[0]["validation_loss"]):
            unstable_consecutive += 1
        else:
            unstable_consecutive = 0
        if unstable_consecutive >= 2:
            raise RuntimeError("preregistered validation-loss instability stop")
        scheduler.step()
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_state is None or best_epoch < 1:
        raise RuntimeError("no validation-selected checkpoint")
    final_state = cpu_state(model)
    elapsed = time.time() - started
    torch.save({
        "state_dict": best_state, "model_family": "PROMPTED_RESUNET", "selection": "minimum validation MSE only",
        "epoch": best_epoch, "validation_loss": best_loss, "training_seed": TRAINING_SEED,
        "parameter_count": EXPECTED_PARAMETERS, "input_channels": 4, "output_channels": 3,
    }, paths["best"])
    torch.save({
        "state_dict": final_state, "model_family": "PROMPTED_RESUNET", "selection": "final epoch",
        "epoch": EPOCHS, "training_seed": TRAINING_SEED, "parameter_count": EXPECTED_PARAMETERS,
        "input_channels": 4, "output_channels": 3,
    }, paths["final"])
    write_csv_fresh(paths["epochs"], epoch_rows)
    config = {
        "status": "FROZEN_TRAINING_COMPLETE", "architecture": "PromptedResUNet residual-block encoder-decoder",
        "parameter_count": EXPECTED_PARAMETERS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "training_seed": TRAINING_SEED, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0, "scheduler": "CosineAnnealingLR", "loss": "whole-image normalized MSE",
        "device": "mps", "training_scenes": len(training), "validation_scenes": len(validation),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "selection_rule": "minimum validation MSE only; first exact tie",
        "normalization_sha256": sha256_file(NORMALIZATION),
        "scene_manifest_sha256": sha256_file(run_dir / "manifests/resunet_scene_definitions.csv"),
        "best_checkpoint_sha256": sha256_file(paths["best"]),
        "final_checkpoint_sha256": sha256_file(paths["final"]),
        "warm_started": False, "condition_c_weights_loaded": False,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "atlas_evaluation_count": 0, "runtime_seconds": elapsed,
    }
    write_json_fresh(paths["config"], config)
    plot_curves(paths["figure"], epoch_rows)
    write_json_fresh(paths["complete"], {
        "status": "PASS", "device": "mps", "epochs_completed": EPOCHS,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_checkpoint_sha256": config["best_checkpoint_sha256"],
        "final_checkpoint_sha256": config["final_checkpoint_sha256"],
        "runtime_seconds": elapsed, "mps_fallback": False,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "atlas_evaluation_count": 0,
    })


if __name__ == "__main__":
    main()
