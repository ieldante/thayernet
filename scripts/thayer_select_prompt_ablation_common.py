#!/usr/bin/env python3
"""Shared, reconstruction-only components for the first Thayer-Select baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
BANDS = ("g", "r", "z")
PARTITION_COUNTS = {
    "training": 8_000,
    "validation": 1_000,
    "calibration": 1_000,
    "development_test": 1_000,
}
LOCKBOX_PARTITION = "sealed_lockbox"
IMAGE_SIZE = 60
PIXEL_SCALE_ARCSEC = 0.2
PROMPT_SIGMA_PIXELS = 2.0
TRAINING_SEED = 2026074101
SCENE_SEED_BASE = 2026074200
NOISE_SEED_BASE = 2026075200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)


def write_json_fresh(path: Path, value) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with os.fdopen(descriptor, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def gaussian_prompt_numpy(
    x_pixel: float,
    y_pixel: float,
    *,
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    sigma_pixels: float = PROMPT_SIGMA_PIXELS,
) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    prompt = np.exp(-0.5 * ((xx - x_pixel) ** 2 + (yy - y_pixel) ** 2) / sigma_pixels**2)
    maximum = float(prompt.max())
    if maximum <= 0 or not np.isfinite(maximum):
        raise RuntimeError("Invalid coordinate prompt")
    return (prompt / maximum).astype(np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU fallback is prohibited")
    torch.mps.manual_seed(seed)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class CompactSelectNet(nn.Module):
    """The compact Thayer U-Net backbone with reconstruction head only.

    The adaptation is intentionally minimal: the uncertainty and recoverability
    heads are absent, and ``in_channels`` is 3 for controls A/B and 4 for C.
    """

    def __init__(self, in_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels not in (3, 4):
            raise ValueError("in_channels must be 3 or 4")
        self.in_channels = in_channels
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.reconstruction_head = nn.Conv2d(base_channels, 3, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.in_channels:
            raise ValueError("Unexpected model input shape")
        enc1 = self.enc1(tensor)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2))
        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.reconstruction_head(dec1)


def parameter_count(in_channels: int) -> int:
    return sum(parameter.numel() for parameter in CompactSelectNet(in_channels).parameters())


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not built and available; CPU fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    return torch.device("mps")


def load_scales(run_dir: Path) -> np.ndarray:
    value = json.loads((run_dir / "manifests/normalization.json").read_text())
    scales = np.asarray(value["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise RuntimeError("Invalid frozen normalization scales")
    return scales


def normalize(array: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32) / scales[:, None, None]


def inverse_normalize(array: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32) * scales[:, None, None]
