"""Compact U-Net architecture for RGB image-to-image deblending."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int, norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(2):
            layers.append(
                nn.Conv2d(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not norm,
                )
            )
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net for reconstructing a target galaxy from a blended image."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels, norm=norm)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, norm=norm)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, norm=norm)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8, norm=norm)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels, norm=norm)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.out_activation(self.out_conv(dec1))
