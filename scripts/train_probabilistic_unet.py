#!/usr/bin/env python3
"""Train the preregistered Thayer-PU model on MPS only."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.models_probabilistic_unet import (  # noqa: E402
    ThayerProbabilisticUNet,
    decomposition_reconstruction_loss,
    free_bits_kl,
    gaussian_kl_per_dimension,
    reparameterize,
    set_training_phase,
    split_decomposition,
    swap_decomposition,
    trainable_parameter_count,
    warm_start_condition_c,
)


CONDITION_C = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json"
EXPECTED_CONDITION_C_SHA256 = "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382"
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAINING_SEED = 2026077501
VALIDATION_EPSILON_SEED = 2026077502
FREE_BITS = 0.05
PRIOR_K = 4
LOSS_WEIGHTS = {
    "requested": 1.0,
    "companion": 1.0,
    "source_sum": 0.5,
    "prompt_swap": 0.25,
    "best_of_many_prior": 0.10,
    "kl": 0.001,
}


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


class SceneDataset(Dataset):
    def __init__(self, run_dir: Path, partition: str, scales: np.ndarray) -> None:
        if partition not in {"training", "validation"}:
            raise ValueError("trainer may open only training and validation")
        self.partition = partition
        self.path = run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        self.rows = [
            row for row in read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
            if row["partition"] == partition
        ]
        self.scales = np.asarray(scales, dtype=np.float32)
        self.handle = None
        with h5py.File(self.path, "r") as handle:
            if not bool(handle.attrs["complete"]) or len(handle["blend"]) != len(self.rows):
                raise RuntimeError(f"incomplete {partition} data")

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __getitem__(self, index: int):
        handle = self._handle()
        blend = np.asarray(handle["blend"][index], dtype=np.float32) / self.scales[:, None, None]
        isolated = np.asarray(handle["isolated"][index], dtype=np.float32) / self.scales[None, :, None, None]
        xy = np.asarray(handle["xy"][index], dtype=np.float64)
        prompt_a = gaussian_prompt_numpy(float(xy[0, 0]), float(xy[0, 1]))[None]
        prompt_b = gaussian_prompt_numpy(float(xy[1, 0]), float(xy[1, 1]))[None]
        return (
            torch.from_numpy(np.ascontiguousarray(blend)),
            torch.from_numpy(np.ascontiguousarray(isolated[0])),
            torch.from_numpy(np.ascontiguousarray(isolated[1])),
            torch.from_numpy(np.ascontiguousarray(prompt_a)),
            torch.from_numpy(np.ascontiguousarray(prompt_b)),
        )


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def beta_for_epoch(epoch: int) -> float:
    if not 1 <= epoch <= EPOCHS:
        raise ValueError("invalid epoch")
    return min(1.0, (epoch - 1) / 9.0)


def fixed_validation_epsilon(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(VALIDATION_EPSILON_SEED + batch_size)
    posterior = torch.randn((batch_size, 8), generator=generator, dtype=torch.float32).to(device)
    prior = torch.randn((PRIOR_K, batch_size, 8), generator=generator, dtype=torch.float32).to(device)
    return posterior, prior


def batch_objective(
    model: ThayerProbabilisticUNet,
    blend: torch.Tensor,
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    prompt_a: torch.Tensor,
    prompt_b: torch.Tensor,
    *,
    beta: float,
    posterior_epsilon: torch.Tensor,
    prior_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prior_mean, prior_log_variance = model.encode_prior(blend)
    posterior_mean, posterior_log_variance = model.encode_posterior(blend, source_a, source_b)
    posterior_z = reparameterize(posterior_mean, posterior_log_variance, epsilon=posterior_epsilon)
    paired_output = model.decode(
        torch.cat((blend, blend), dim=0),
        torch.cat((prompt_a, prompt_b), dim=0),
        torch.cat((posterior_z, posterior_z), dim=0),
    )
    output_a, output_b = paired_output.chunk(2, dim=0)
    loss_a = decomposition_reconstruction_loss(output_a, source_a, source_b)
    loss_b = decomposition_reconstruction_loss(output_b, source_b, source_a)
    requested = 0.5 * (loss_a["requested"] + loss_b["requested"])
    companion = 0.5 * (loss_a["companion"] + loss_b["companion"])
    source_sum_loss = 0.5 * (loss_a["source_sum"] + loss_b["source_sum"])
    prompt_swap = F.mse_loss(output_a, swap_decomposition(output_b))

    prior_z = reparameterize(
        prior_mean[None].expand(PRIOR_K, -1, -1),
        prior_log_variance[None].expand(PRIOR_K, -1, -1),
        epsilon=prior_epsilon,
    )
    sample_count, batch_size = prior_z.shape[:2]
    prior_output = model.decode(
        blend[None].expand(sample_count, -1, -1, -1, -1).reshape(sample_count * batch_size, 3, 60, 60),
        prompt_a[None].expand(sample_count, -1, -1, -1, -1).reshape(sample_count * batch_size, 1, 60, 60),
        prior_z.reshape(sample_count * batch_size, 8),
    ).reshape(sample_count, batch_size, 6, 60, 60)
    prior_requested, prior_companion = prior_output[:, :, :3], prior_output[:, :, 3:]
    per_sample_prior = (
        (prior_requested - source_a[None]).square().mean(dim=(2, 3, 4))
        + (prior_companion - source_b[None]).square().mean(dim=(2, 3, 4))
        + 0.5 * (prior_requested + prior_companion - source_a[None] - source_b[None]).square().mean(dim=(2, 3, 4))
    ) / 2.5
    best_of_many = per_sample_prior.min(dim=0).values.mean()

    kl_per_dimension = gaussian_kl_per_dimension(
        posterior_mean, posterior_log_variance, prior_mean, prior_log_variance
    )
    raw_kl = kl_per_dimension.sum(dim=1).mean()
    kl_free = free_bits_kl(kl_per_dimension, FREE_BITS)
    total = (
        LOSS_WEIGHTS["requested"] * requested
        + LOSS_WEIGHTS["companion"] * companion
        + LOSS_WEIGHTS["source_sum"] * source_sum_loss
        + LOSS_WEIGHTS["prompt_swap"] * prompt_swap
        + LOSS_WEIGHTS["best_of_many_prior"] * best_of_many
        + LOSS_WEIGHTS["kl"] * beta * kl_free
    )
    components = {
        "total": total,
        "requested": requested,
        "companion": companion,
        "source_sum": source_sum_loss,
        "prompt_swap": prompt_swap,
        "best_of_many_prior": best_of_many,
        "raw_kl": raw_kl,
        "free_bits_kl": kl_free,
        "active_dimensions": (kl_per_dimension.mean(dim=0) >= 0.02).sum(),
        "prior_std_mean": torch.exp(0.5 * prior_log_variance).mean(),
        "posterior_std_mean": torch.exp(0.5 * posterior_log_variance).mean(),
    }
    return total, components


def accumulate(sums: dict[str, float], components: dict[str, torch.Tensor], count: int) -> None:
    for name, value in components.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * count


def means(sums: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / count for name, value in sums.items()}


def plot_training(path: Path, rows: list[dict[str, object]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [float(row["training_total"]) for row in rows], label="train")
    axes[0, 0].plot(epochs, [float(row["validation_total"]) for row in rows], label="validation")
    axes[0, 0].set(title="Frozen objective", yscale="log")
    axes[0, 1].plot(epochs, [float(row["training_raw_kl"]) for row in rows], label="raw KL")
    axes[0, 1].plot(epochs, [float(row["training_free_bits_kl"]) for row in rows], label="free-bits KL")
    axes[0, 1].set(title="Latent KL")
    axes[1, 0].plot(epochs, [float(row["training_active_dimensions"]) for row in rows])
    axes[1, 0].axhline(2, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Active latent dimensions", ylim=(-0.2, 8.2))
    axes[1, 1].plot(epochs, [float(row["validation_source_sum"]) for row in rows], label="sum")
    axes[1, 1].plot(epochs, [float(row["validation_prompt_swap"]) for row in rows], label="swap")
    axes[1, 1].set(title="Validation consistency", yscale="log")
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preparation = json.loads((run_dir / "logs/data_preparation_complete.json").read_text())
    if preparation["status"] != "PASS" or preparation["scene_count"] != 20_000 or preparation["replay_pass_count"] != 20_000:
        raise RuntimeError("data/replay gate failed")
    if sha256_file(CONDITION_C) != EXPECTED_CONDITION_C_SHA256:
        raise RuntimeError("Condition-C checkpoint altered")
    paths = {
        "best": run_dir / "checkpoints/thayer_pu_best.pth",
        "final": run_dir / "checkpoints/thayer_pu_final.pth",
        "epochs": run_dir / "tables/thayer_pu_epochs.csv",
        "prefit": run_dir / "manifests/thayer_pu_training_config_pre_fit.json",
        "complete": run_dir / "logs/training_complete.json",
        "figure": run_dir / "figures/training_curves.png",
    }
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"training output collision: {collisions}")
    scales = np.asarray(json.loads(NORMALIZATION.read_text())["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise RuntimeError("frozen normalization invalid")
    device = require_mps()
    seed_everything(TRAINING_SEED)
    model = ThayerProbabilisticUNet()
    warm_inventory = warm_start_condition_c(model, CONDITION_C)
    expected_inventory = read_csv(run_dir / "tables/condition_c_warm_start_inventory.csv")
    if len(warm_inventory) != len(expected_inventory) or [row["sha256"] for row in warm_inventory] != [row["sha256"] for row in expected_inventory]:
        raise RuntimeError("warm-start tensor inventory mismatch")
    model = model.to(device)
    set_training_phase(model, 1)
    if next(model.parameters()).device.type != "mps" or trainable_parameter_count(model) > 600_000:
        raise RuntimeError("model/device/parameter gate failed")
    training = SceneDataset(run_dir, "training", scales)
    validation = SceneDataset(run_dir, "validation", scales)
    if len(training) != 16_000 or len(validation) != 2_000:
        raise RuntimeError("training/validation scene count mismatch")
    write_json_fresh(paths["prefit"], {
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP", "architecture": "ThayerProbabilisticUNet",
        "total_parameters": trainable_parameter_count(model), "latent_dimension": 8,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "training_seed": TRAINING_SEED, "validation_epsilon_seed": VALIDATION_EPSILON_SEED,
        "loss_weights": LOSS_WEIGHTS, "free_bits_nats_per_dimension": FREE_BITS,
        "kl_beta_schedule": "linear 0 at epoch 1 to 1 at epoch 10; then 1",
        "best_of_many_prior_samples": PRIOR_K, "device": "mps", "mps_fallback": False,
        "training_scenes": len(training), "validation_scenes": len(validation),
        "scene_manifest_sha256": preparation["scene_manifest_sha256"],
        "normalization_sha256": sha256_file(NORMALIZATION),
        "condition_c_checkpoint_sha256": sha256_file(CONDITION_C),
        "selection_rule": "minimum validation frozen total objective; first exact tie",
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(
        training, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(TRAINING_SEED), num_workers=0,
    )
    validation_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    epoch_rows: list[dict[str, object]] = []
    high_kl_epochs = 0
    inactive_epochs = 0
    sum_instability_epochs = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        if epoch == 4:
            set_training_phase(model, 2)
        beta = beta_for_epoch(epoch)
        model.train()
        training_sums: dict[str, float] = {}
        training_count = 0
        for blend, source_a, source_b, prompt_a, prompt_b in train_loader:
            tensors = [value.to(device) for value in (blend, source_a, source_b, prompt_a, prompt_b)]
            if any(value.device.type != "mps" for value in tensors):
                raise RuntimeError("MPS training fallback detected")
            blend, source_a, source_b, prompt_a, prompt_b = tensors
            optimizer.zero_grad(set_to_none=True)
            posterior_epsilon = torch.randn((len(blend), 8), device=device)
            prior_epsilon = torch.randn((PRIOR_K, len(blend), 8), device=device)
            loss, components = batch_objective(
                model, blend, source_a, source_b, prompt_a, prompt_b, beta=beta,
                posterior_epsilon=posterior_epsilon, prior_epsilon=prior_epsilon,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training objective")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], max_norm=10.0)
            optimizer.step()
            accumulate(training_sums, components, len(blend))
            training_count += len(blend)
        training_metrics = means(training_sums, training_count)

        model.eval()
        validation_sums: dict[str, float] = {}
        validation_count = 0
        with torch.no_grad():
            for blend, source_a, source_b, prompt_a, prompt_b in validation_loader:
                blend, source_a, source_b, prompt_a, prompt_b = [
                    value.to(device) for value in (blend, source_a, source_b, prompt_a, prompt_b)
                ]
                posterior_epsilon, prior_epsilon = fixed_validation_epsilon(len(blend), device)
                _, components = batch_objective(
                    model, blend, source_a, source_b, prompt_a, prompt_b, beta=beta,
                    posterior_epsilon=posterior_epsilon, prior_epsilon=prior_epsilon,
                )
                if not all(torch.isfinite(value) for value in components.values()):
                    raise RuntimeError("non-finite validation metric")
                accumulate(validation_sums, components, len(blend))
                validation_count += len(blend)
        validation_metrics = means(validation_sums, validation_count)
        row: dict[str, object] = {
            "epoch": epoch, "phase": 1 if epoch <= 3 else 2, "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"], "device": "mps",
            "currently_trainable_parameters": trainable_parameter_count(model, currently_trainable=True),
        }
        row.update({f"training_{name}": value for name, value in training_metrics.items()})
        row.update({f"validation_{name}": value for name, value in validation_metrics.items()})
        epoch_rows.append(row)
        validation_total = float(validation_metrics["total"])
        if validation_total < best_loss:
            best_loss = validation_total
            best_epoch = epoch
            best_state = cpu_state(model)

        high_kl_epochs = high_kl_epochs + 1 if training_metrics["raw_kl"] > 100.0 else 0
        inactive_epochs = inactive_epochs + 1 if epoch > 10 and training_metrics["active_dimensions"] < 1.0 else 0
        if len(epoch_rows) > 1 and validation_metrics["source_sum"] > 10.0 * float(epoch_rows[0]["validation_source_sum"]):
            sum_instability_epochs += 1
        else:
            sum_instability_epochs = 0
        print(json.dumps(row, sort_keys=True), flush=True)
        if high_kl_epochs >= 2:
            raise RuntimeError("uncontrolled KL explosion")
        if inactive_epochs >= 3:
            raise RuntimeError("sustained zero latent use")
        if sum_instability_epochs >= 2:
            raise RuntimeError("source-sum instability")
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    final_state = cpu_state(model)
    elapsed = time.time() - started
    torch.save({
        "state_dict": best_state, "model_family": "THAYER_PU", "selection": "minimum validation frozen total objective",
        "epoch": best_epoch, "validation_total": best_loss, "training_seed": TRAINING_SEED,
        "parameter_count": trainable_parameter_count(model), "latent_dimension": 8,
        "input_channels": 4, "output_channels": 6, "posterior_training_only": True, "prior_truth_free": True,
    }, paths["best"])
    torch.save({
        "state_dict": final_state, "model_family": "THAYER_PU", "selection": "final epoch",
        "epoch": EPOCHS, "training_seed": TRAINING_SEED, "parameter_count": trainable_parameter_count(model),
        "latent_dimension": 8, "input_channels": 4, "output_channels": 6,
        "posterior_training_only": True, "prior_truth_free": True,
    }, paths["final"])
    write_csv_fresh(paths["epochs"], epoch_rows)
    plot_training(paths["figure"], epoch_rows)
    write_json_fresh(paths["complete"], {
        "status": "PASS", "device": "mps", "mps_fallback": False,
        "epochs_completed": EPOCHS, "best_epoch": best_epoch, "best_validation_total": best_loss,
        "best_checkpoint_sha256": sha256_file(paths["best"]), "final_checkpoint_sha256": sha256_file(paths["final"]),
        "runtime_seconds": elapsed, "final_training_raw_kl": training_metrics["raw_kl"],
        "final_training_active_dimensions": training_metrics["active_dimensions"],
        "final_validation_source_sum": validation_metrics["source_sum"],
        "condition_c_checkpoint_sha256_after": sha256_file(CONDITION_C),
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
    })


if __name__ == "__main__":
    main()
