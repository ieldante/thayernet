"""Compact prompted ResUNet for source-layer candidate diversity."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    """Two normalized convolutions with an identity or projected residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or stride not in (1, 2):
            raise ValueError("invalid residual-block dimensions")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = self.projection(tensor)
        output = F.silu(self.norm1(self.conv1(tensor)))
        output = self.norm2(self.conv2(output))
        return F.silu(output + residual)


class PromptedResUNet(nn.Module):
    """Residual encoder-decoder for g/r/z plus one Gaussian prompt channel."""

    input_channels = 4
    output_channels = 3

    def __init__(self) -> None:
        super().__init__()
        self.enc0 = ResidualBlock(4, 16)
        self.enc1 = ResidualBlock(16, 32, stride=2)
        self.enc2 = ResidualBlock(32, 64, stride=2)
        self.bottleneck = ResidualBlock(64, 64)
        self.dec1 = ResidualBlock(64 + 32, 32)
        self.dec0 = ResidualBlock(32 + 16, 16)
        self.reconstruction_head = nn.Conv2d(16, 3, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.input_channels:
            raise ValueError(f"expected (N,4,H,W), got {tuple(tensor.shape)}")
        if tensor.shape[-2:] != (60, 60):
            raise ValueError("the frozen source-layer contract requires 60x60 inputs")
        enc0 = self.enc0(tensor)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        latent = self.bottleneck(enc2)
        up1 = F.interpolate(latent, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        up0 = F.interpolate(dec1, size=enc0.shape[-2:], mode="bilinear", align_corners=False)
        dec0 = self.dec0(torch.cat((up0, enc0), dim=1))
        return self.reconstruction_head(dec0)


def trainable_parameter_count(model: nn.Module | None = None) -> int:
    value = PromptedResUNet() if model is None else model
    return sum(parameter.numel() for parameter in value.parameters() if parameter.requires_grad)
