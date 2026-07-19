"""Compact prompted probabilistic U-Net for full two-source decompositions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LATENT_DIMENSION = 8
INPUT_SHAPE = (60, 60)


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    """Condition-C-compatible two-convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class GaussianEncoder(nn.Module):
    """Scene-level diagonal-Gaussian encoder with a fixed small capacity."""

    def __init__(self, in_channels: int, latent_dimension: int = LATENT_DIMENSION) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dimension = latent_dimension
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.statistics = nn.Linear(64, 2 * latent_dimension)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4 or tensor.shape[1] != self.in_channels or tensor.shape[-2:] != INPUT_SHAPE:
            raise ValueError(f"expected (N,{self.in_channels},60,60), got {tuple(tensor.shape)}")
        pooled = self.features(tensor).mean(dim=(-2, -1))
        mean, log_variance = self.statistics(pooled).chunk(2, dim=1)
        return mean, log_variance.clamp(min=-8.0, max=8.0)


class ThayerProbabilisticUNet(nn.Module):
    """Condition-C warm-started decoder with truth-free prior and train-only posterior."""

    input_channels = 4
    output_channels = 6
    latent_dimension = LATENT_DIMENSION

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)
        self.decomposition_head = nn.Conv2d(16, 6, 1)
        self.latent_injection = nn.Linear(self.latent_dimension, 64)
        self.prior = GaussianEncoder(3, self.latent_dimension)
        self.posterior = GaussianEncoder(9, self.latent_dimension)

    def encode_prior(self, observed_blend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Truth-free p(z|observed blend); prompt and target are not accepted."""

        return self.prior(observed_blend)

    def encode_posterior(
        self,
        observed_blend: torch.Tensor,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-only q(z|blend,A,B) in canonical manifest source order."""

        if source_a.shape != observed_blend.shape or source_b.shape != observed_blend.shape:
            raise ValueError("posterior truth layers must match the observed blend")
        return self.posterior(torch.cat((observed_blend, source_a, source_b), dim=1))

    def decode(self, observed_blend: torch.Tensor, prompt: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if observed_blend.ndim != 4 or observed_blend.shape[1:] != (3, *INPUT_SHAPE):
            raise ValueError(f"expected blend (N,3,60,60), got {tuple(observed_blend.shape)}")
        if prompt.shape != (len(observed_blend), 1, *INPUT_SHAPE):
            raise ValueError(f"expected prompt (N,1,60,60), got {tuple(prompt.shape)}")
        if latent.shape != (len(observed_blend), self.latent_dimension):
            raise ValueError(f"expected latent (N,{self.latent_dimension}), got {tuple(latent.shape)}")
        enc1 = self.enc1(torch.cat((observed_blend, prompt), dim=1))
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2))
        injection = self.latent_injection(latent)[:, :, None, None]
        bottleneck = bottleneck + injection
        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.decomposition_head(dec1)

    def inference_prior(
        self,
        observed_blend: torch.Tensor,
        prompt: torch.Tensor,
        *,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Inference API: prior only, with explicit frozen random variates."""

        mean, log_variance = self.encode_prior(observed_blend)
        latent = reparameterize(mean, log_variance, epsilon=epsilon)
        return self.decode(observed_blend, prompt, latent)


def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor, *, epsilon: torch.Tensor) -> torch.Tensor:
    if mean.shape != log_variance.shape or epsilon.shape != mean.shape:
        raise ValueError("mean, log variance, and epsilon must have identical shape")
    return mean + torch.exp(0.5 * log_variance) * epsilon


def gaussian_kl_per_dimension(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_variance: torch.Tensor,
) -> torch.Tensor:
    tensors = (posterior_mean, posterior_log_variance, prior_mean, prior_log_variance)
    if any(tensor.shape != posterior_mean.shape for tensor in tensors):
        raise ValueError("Gaussian statistics must have identical shapes")
    prior_variance = torch.exp(prior_log_variance)
    posterior_variance = torch.exp(posterior_log_variance)
    return 0.5 * (
        prior_log_variance
        - posterior_log_variance
        + (posterior_variance + (posterior_mean - prior_mean).square()) / prior_variance
        - 1.0
    )


def free_bits_kl(kl_per_dimension: torch.Tensor, free_bits: float = 0.05) -> torch.Tensor:
    if kl_per_dimension.ndim != 2 or free_bits < 0:
        raise ValueError("expected (N,Z) KL and nonnegative free bits")
    return torch.clamp(kl_per_dimension, min=free_bits).sum(dim=1).mean()


def split_decomposition(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 4 or output.shape[1] != 6:
        raise ValueError("expected a six-channel decomposition")
    return output[:, :3], output[:, 3:]


def swap_decomposition(output: torch.Tensor) -> torch.Tensor:
    requested, companion = split_decomposition(output)
    return torch.cat((companion, requested), dim=1)


def source_sum(output: torch.Tensor) -> torch.Tensor:
    requested, companion = split_decomposition(output)
    return requested + companion


def decomposition_reconstruction_loss(
    output: torch.Tensor,
    requested_truth: torch.Tensor,
    companion_truth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    requested, companion = split_decomposition(output)
    return {
        "requested": F.mse_loss(requested, requested_truth),
        "companion": F.mse_loss(companion, companion_truth),
        "source_sum": F.mse_loss(requested + companion, requested_truth + companion_truth),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = np.asarray(tensor.detach().cpu(), dtype=np.dtype("<f4"), order="C")
    header = json.dumps({"shape": list(value.shape), "dtype": "<f4"}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(header.encode("utf-8") + b"\0" + value.tobytes(order="C"))
    return digest.hexdigest()


def warm_start_condition_c(model: ThayerProbabilisticUNet, checkpoint: Path) -> list[dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    model_state = model.state_dict()
    backbone_prefixes = ("enc1.", "enc2.", "bottleneck.", "dec2.", "dec1.")
    rows: list[dict[str, object]] = []
    for name, tensor in state.items():
        if name.startswith(backbone_prefixes):
            if name not in model_state or model_state[name].shape != tensor.shape:
                raise RuntimeError(f"Condition-C tensor mismatch: {name}")
            model_state[name].copy_(tensor)
            rows.append({
                "condition_c_tensor": name,
                "thayer_pu_tensor": name,
                "shape": "x".join(map(str, tensor.shape)),
                "sha256": tensor_sha256(tensor),
                "load_rule": "exact",
            })
    for suffix in ("weight", "bias"):
        source_name = f"reconstruction_head.{suffix}"
        target_name = f"decomposition_head.{suffix}"
        source = state[source_name]
        target = model_state[target_name]
        if target.shape[0] != 2 * source.shape[0] or target.shape[1:] != source.shape[1:]:
            raise RuntimeError("expanded decomposition head mismatch")
        target[:3].copy_(source)
        target[3:].copy_(source)
        for start, role in ((0, "requested"), (3, "companion")):
            rows.append({
                "condition_c_tensor": source_name,
                "thayer_pu_tensor": f"{target_name}[{start}:{start + 3}]",
                "shape": "x".join(map(str, source.shape)),
                "sha256": tensor_sha256(source),
                "load_rule": f"copied_to_{role}_half",
            })
    expected = {name for name in state if name.startswith(backbone_prefixes)}
    loaded = {row["condition_c_tensor"] for row in rows if row["load_rule"] == "exact"}
    if loaded != expected:
        raise RuntimeError("not every Condition-C backbone tensor was inventoried")
    model.load_state_dict(model_state)
    return rows


def set_training_phase(model: ThayerProbabilisticUNet, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.prior, model.posterior, model.latent_injection, model.dec2, model.dec1, model.decomposition_head]
    if phase == 2:
        modules.append(model.bottleneck)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module, *, currently_trainable: bool = False) -> int:
    if currently_trainable:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())

