"""Training and evaluation routines for synthetic deblending models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import compute_metrics


class BlendDataset(Dataset):
    """PyTorch dataset wrapping blend dictionaries from `generate_blends`."""

    def __init__(self, blends: Sequence[dict[str, Any]]) -> None:
        if len(blends) == 0:
            raise ValueError("BlendDataset requires at least one blend.")
        self.blends = list(blends)

    def __len__(self) -> int:
        return len(self.blends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.blends[idx]
        blended = np.asarray(sample["blended"], dtype=np.float32)
        target = np.asarray(sample["target"], dtype=np.float32)
        return (
            torch.from_numpy(blended.transpose(2, 0, 1)).float(),
            torch.from_numpy(target.transpose(2, 0, 1)).float(),
        )


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve `auto` to the best available PyTorch device."""
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_accelerator(device: str | torch.device = "auto") -> torch.device:
    """Resolve a full-run device and refuse silent CPU model execution.

    MPS is preferred when available, followed by CUDA. Explicit CPU requests
    are rejected; callers may continue using :func:`resolve_device` for small
    sanity checks where CPU execution is intentional.
    """

    resolved = resolve_device(device)
    if resolved.type == "cpu":
        raise RuntimeError(
            "Full model training/evaluation requires MPS or CUDA; "
            "refusing CPU fallback."
        )
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but torch.backends.mps is unavailable.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if resolved.type not in {"mps", "cuda"}:
        raise RuntimeError(
            f"Unsupported full-run device {resolved}; expected MPS or CUDA."
        )
    return resolved


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "auto",
    checkpoint_path: str | Path | None = None,
) -> tuple[list[float], list[float]]:
    """Train a model with MSE loss and return epoch losses."""
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training and validation datasets must be non-empty.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    resolved_device = resolve_accelerator(device)
    model = model.to(resolved_device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for blended, target in train_loader:
            blended = blended.to(resolved_device)
            target = target.to(resolved_device)

            optimiser.zero_grad()
            output = model(blended)
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * blended.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blended, target in val_loader:
                blended = blended.to(resolved_device)
                target = target.to(resolved_device)
                output = model(blended)
                loss = criterion(output, target)
                val_loss += loss.item() * blended.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train loss={train_loss:.4f}, val loss={val_loss:.4f}"
        )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    test_ds: Dataset,
    device: str | torch.device = "auto",
    metrics: Sequence[str] = ("mse", "mae", "psnr", "ssim"),
) -> tuple[dict[str, float], list[np.ndarray]]:
    """Evaluate a trained model and return metric averages plus outputs."""
    if len(test_ds) == 0:
        raise ValueError("Test dataset must be non-empty.")

    resolved_device = resolve_accelerator(device)
    model = model.to(resolved_device)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    metric_sums = {name: 0.0 for name in metrics}
    reconstructions: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for blended, target in loader:
            blended = blended.to(resolved_device)
            target = target.to(resolved_device)
            pred = model(blended)

            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
            values = compute_metrics(pred_np, target_np, metrics)
            for name, value in values.items():
                metric_sums[name] += value
            reconstructions.append(pred_np)

    return {name: value / len(test_ds) for name, value in metric_sums.items()}, reconstructions
