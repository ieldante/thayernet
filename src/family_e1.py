"""Frozen Family-E1 compact U-Net, physical map, and aligned objective."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BAND_SCALES = (611.9199829101562, 1805.8800048828125, 1854.199951171875)
EXPECTED_PARAMETERS = 1_162_662
OBJECTIVE_WEIGHTS = {
    "requested_l1": 1.0,
    "companion_l1": 1.0,
    "flux": 0.25,
    "centroid": 0.10,
    "color": 0.10,
}


@dataclass(frozen=True)
class FamilyE1Output:
    raw: Tensor
    requested: Tensor
    companion: Tensor
    residual_noise: Tensor


def _normalized_block(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class _Stage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = _normalized_block(channels, channels)
        self.second = _normalized_block(channels, channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.second(self.first(value))


class _DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_convolution = _normalized_block(in_channels, out_channels)
        self.first = _normalized_block(out_channels + skip_channels, out_channels)
        self.second = _normalized_block(out_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.up_convolution(value)
        return self.second(self.first(torch.cat((value, skip), dim=1)))


class FamilyE1UNet(nn.Module):
    """One 4-channel coordinate U-Net with the physical ReLU map in forward."""

    def __init__(self, band_scales: tuple[float, float, float] = BAND_SCALES) -> None:
        super().__init__()
        self.enc0_first = _normalized_block(4, 24)
        self.enc0_second = _normalized_block(24, 24)
        self.down0 = _normalized_block(24, 48, stride=2)
        self.enc1 = _Stage(48)
        self.down1 = _normalized_block(48, 96, stride=2)
        self.enc2 = _Stage(96)
        self.down2 = _normalized_block(96, 128, stride=2)
        self.enc3 = _Stage(128)
        self.dec2 = _DecoderStage(128, 96, 96)
        self.dec1 = _DecoderStage(96, 48, 48)
        self.dec0 = _DecoderStage(48, 24, 24)
        self.source_head = nn.Conv2d(24, 6, 1, bias=True)
        self.register_buffer("band_scales", torch.tensor(band_scales, dtype=torch.float32).reshape(1, 3, 1, 1))
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
        nn.init.constant_(self.source_head.bias, 0.01)

    def forward(self, model_input: Tensor, observed: Tensor) -> FamilyE1Output:
        if model_input.ndim != 4 or model_input.shape[1:] != (4, 60, 60):
            raise ValueError(f"model_input must have shape (N,4,60,60), got {tuple(model_input.shape)}")
        if observed.ndim != 4 or observed.shape[1:] != (3, 60, 60) or len(observed) != len(model_input):
            raise ValueError(f"observed must have shape (N,3,60,60), got {tuple(observed.shape)}")
        enc0 = self.enc0_second(self.enc0_first(model_input))
        enc1 = self.enc1(self.down0(enc0))
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        decoded = self.dec0(self.dec1(self.dec2(enc3, enc2), enc1), enc0)
        raw = self.source_head(decoded)
        scales = self.band_scales.to(dtype=raw.dtype)
        requested = F.relu(raw[:, :3]) * scales
        companion = F.relu(raw[:, 3:]) * scales
        residual = observed - requested - companion
        return FamilyE1Output(raw, requested, companion, residual)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def conservation_error(output: FamilyE1Output, observed: Tensor) -> Tensor:
    return torch.max(torch.abs(output.requested + output.companion + output.residual_noise - observed))


def _soft_centroids(images: Tensor, epsilon: float) -> Tensor:
    # images: N,source,band,H,W. Coordinates are returned as N,source,band,2.
    height, width = images.shape[-2:]
    y = torch.arange(height, dtype=images.dtype, device=images.device).reshape(1, 1, 1, height, 1)
    x = torch.arange(width, dtype=images.dtype, device=images.device).reshape(1, 1, 1, 1, width)
    flux = images.sum(dim=(-2, -1))
    denominator = flux + epsilon
    cx = (images * x).sum(dim=(-2, -1)) / denominator
    cy = (images * y).sum(dim=(-2, -1)) / denominator
    return torch.stack((cx, cy), dim=-1)


def source_objective(
    requested: Tensor,
    companion: Tensor,
    requested_target: Tensor,
    companion_target: Tensor,
    *,
    band_scales: Tensor | tuple[float, float, float] = BAND_SCALES,
    epsilon: float = 1.0e-6,
) -> dict[str, Tensor]:
    tensors = (requested, companion, requested_target, companion_target)
    if any(value.ndim != 4 or value.shape[1:] != (3, 60, 60) for value in tensors):
        raise ValueError("all source tensors must have shape (N,3,60,60)")
    if not all(value.shape == requested.shape for value in tensors):
        raise ValueError("source tensor shapes must match")
    scales = torch.as_tensor(band_scales, dtype=requested.dtype, device=requested.device).reshape(1, 3, 1, 1)
    predicted = torch.stack((requested / scales, companion / scales), dim=1)
    truth = torch.stack((requested_target / scales, companion_target / scales), dim=1)
    requested_l1 = torch.mean(torch.abs(predicted[:, 0] - truth[:, 0]))
    companion_l1 = torch.mean(torch.abs(predicted[:, 1] - truth[:, 1]))
    predicted_flux = predicted.sum(dim=(-2, -1))
    truth_flux = truth.sum(dim=(-2, -1))
    flux = torch.mean(torch.abs(predicted_flux - truth_flux) / (torch.abs(truth_flux) + epsilon))
    centroid = torch.mean(torch.linalg.vector_norm(_soft_centroids(predicted, epsilon) - _soft_centroids(truth, epsilon), dim=-1) / 60.0)
    predicted_colors = torch.stack((
        torch.log(predicted_flux[:, :, 0] + epsilon) - torch.log(predicted_flux[:, :, 1] + epsilon),
        torch.log(predicted_flux[:, :, 1] + epsilon) - torch.log(predicted_flux[:, :, 2] + epsilon),
    ), dim=-1)
    truth_colors = torch.stack((
        torch.log(truth_flux[:, :, 0] + epsilon) - torch.log(truth_flux[:, :, 1] + epsilon),
        torch.log(truth_flux[:, :, 1] + epsilon) - torch.log(truth_flux[:, :, 2] + epsilon),
    ), dim=-1)
    color = torch.mean(torch.abs(predicted_colors - truth_colors))
    total = (
        OBJECTIVE_WEIGHTS["requested_l1"] * requested_l1
        + OBJECTIVE_WEIGHTS["companion_l1"] * companion_l1
        + OBJECTIVE_WEIGHTS["flux"] * flux
        + OBJECTIVE_WEIGHTS["centroid"] * centroid
        + OBJECTIVE_WEIGHTS["color"] * color
    )
    return {
        "total": total,
        "requested_l1": requested_l1,
        "companion_l1": companion_l1,
        "flux": flux,
        "centroid": centroid,
        "color": color,
    }
