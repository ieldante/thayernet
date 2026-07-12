"""Lightweight coordinate-conditioned Thayer-Select model interface."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from .coordinate_prompt import concatenate_image_and_prompt


class ThayerSelectOutput(TypedDict):
    reconstruction: torch.Tensor
    log_variance: torch.Tensor
    recoverability: torch.Tensor
    no_source_probability: torch.Tensor


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


class ThayerSelectNet(nn.Module):
    """Small U-Net with reconstruction, variance, and recoverability heads.

    The only inputs are three normalized ``g,r,z`` image channels and one
    Gaussian prompt channel.  Recoverability is derived from encoder features,
    so no oracle target or evaluation metric can enter its forward pass.
    """

    def __init__(
        self,
        *,
        base_channels: int = 16,
        output_mode: str = "reconstruction",
        min_log_variance: float = -10.0,
        max_log_variance: float = 4.0,
    ) -> None:
        super().__init__()
        if output_mode not in {"reconstruction", "correction"}:
            raise ValueError("output_mode must be 'reconstruction' or 'correction'")
        if not min_log_variance < max_log_variance:
            raise ValueError("invalid log-variance bounds")
        self.output_mode = output_mode
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

        self.enc1 = ConvBlock(4, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.reconstruction_head = nn.Conv2d(base_channels, 3, 1)
        self.variance_head = nn.Conv2d(base_channels, 3, 1)
        self.recoverability_head = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, 1),
        )
        self.no_source_head = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, 1),
        )

    def forward(self, image_grz: torch.Tensor, prompt: torch.Tensor) -> ThayerSelectOutput:
        inputs = concatenate_image_and_prompt(image_grz, prompt)
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2))

        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        raw_reconstruction = self.reconstruction_head(dec1)
        if self.output_mode == "correction":
            reconstruction = image_grz + raw_reconstruction
        else:
            reconstruction = raw_reconstruction

        raw_log_variance = self.variance_head(dec1)
        interval = self.max_log_variance - self.min_log_variance
        log_variance = self.min_log_variance + interval * torch.sigmoid(raw_log_variance)

        pooled = F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(1)
        recoverability = torch.sigmoid(self.recoverability_head(pooled))
        no_source_probability = torch.sigmoid(self.no_source_head(pooled))
        return {
            "reconstruction": reconstruction,
            "log_variance": log_variance,
            "recoverability": recoverability,
            "no_source_probability": no_source_probability,
        }
