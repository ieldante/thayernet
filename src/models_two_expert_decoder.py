"""Thayer-ME shared prompted encoder with two independent compact decoders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models_probabilistic_unet import ConvBlock, tensor_sha256


INPUT_SHAPE = (60, 60)
EXPERT_COUNT = 2
EXPERT_INITIALIZATION_SEEDS = (2026071201, 2026071202)


class SharedPromptEncoder(nn.Module):
    """Condition-C-compatible encoder receiving only blend and coordinate prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)

    def forward(self, observed_blend: torch.Tensor, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observed_blend.ndim != 4 or observed_blend.shape[1:] != (3, *INPUT_SHAPE):
            raise ValueError(f"expected blend (N,3,60,60), got {tuple(observed_blend.shape)}")
        if prompt.shape != (len(observed_blend), 1, *INPUT_SHAPE):
            raise ValueError(f"expected prompt (N,1,60,60), got {tuple(prompt.shape)}")
        enc1 = self.enc1(torch.cat((observed_blend, prompt), dim=1))
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2))
        return enc1, enc2, bottleneck


class CompactExpertDecoder(nn.Module):
    """One complete, independently parameterized six-channel decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)
        self.decomposition_head = nn.Conv2d(16, 6, 1)

    def forward(self, enc1: torch.Tensor, enc2: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=INPUT_SHAPE, mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.decomposition_head(dec1)


def _seeded_expert(seed: int) -> CompactExpertDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CompactExpertDecoder()


class ThayerMixtureExperts(nn.Module):
    """One shared representation and two independent scientific hypotheses."""

    input_channels = 4
    output_channels = 6
    expert_count = EXPERT_COUNT

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedPromptEncoder()
        self.expert_1 = _seeded_expert(EXPERT_INITIALIZATION_SEEDS[0])
        self.expert_2 = _seeded_expert(EXPERT_INITIALIZATION_SEEDS[1])

    def forward(self, observed_blend: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observed_blend, prompt)
        return torch.stack((self.expert_1(*features), self.expert_2(*features)), dim=1)


def split_decomposition(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.shape[-3] != 6:
        raise ValueError("expected six-channel decompositions")
    return output[..., :3, :, :], output[..., 3:, :, :]


def swap_decomposition(output: torch.Tensor) -> torch.Tensor:
    requested, companion = split_decomposition(output)
    return torch.cat((companion, requested), dim=-3)


def source_sum(output: torch.Tensor) -> torch.Tensor:
    requested, companion = split_decomposition(output)
    return requested + companion


def _mean_square(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).square().mean(dim=(-3, -2, -1))


def decomposition_cost(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_requested, pred_companion = split_decomposition(predicted)
    true_requested, true_companion = split_decomposition(target)
    requested = _mean_square(pred_requested, true_requested)
    companion = _mean_square(pred_companion, true_companion)
    summed = _mean_square(pred_requested + pred_companion, true_requested + true_companion)
    return requested + companion + 0.5 * summed


def permutation_invariant_target_loss(hypotheses: torch.Tensor, targets: torch.Tensor, target_count: torch.Tensor) -> dict[str, torch.Tensor]:
    """Two-assignment target loss with ordinary concentration."""

    if hypotheses.ndim != 5 or hypotheses.shape[1:3] != (2, 6):
        raise ValueError("expected hypotheses (N,2,6,H,W)")
    if targets.shape != hypotheses.shape or target_count.shape != (len(hypotheses),):
        raise ValueError("targets/count do not match hypotheses")
    if not torch.all((target_count == 1) | (target_count == 2)):
        raise ValueError("target count must be one or two")
    c00 = decomposition_cost(hypotheses[:, 0], targets[:, 0])
    c10 = decomposition_cost(hypotheses[:, 1], targets[:, 0])
    c01 = decomposition_cost(hypotheses[:, 0], targets[:, 1])
    c11 = decomposition_cost(hypotheses[:, 1], targets[:, 1])
    identity = c00 + c11
    swapped = c01 + c10
    ambiguous = torch.minimum(identity, swapped)
    ordinary = c00 + c10
    concentration = _mean_square(hypotheses[:, 0], hypotheses[:, 1])
    per_scene = torch.where(target_count == 2, ambiguous, ordinary + 0.10 * concentration)
    return {
        "loss": per_scene.mean(),
        "per_scene": per_scene,
        "identity_assignment": identity,
        "swapped_assignment": swapped,
        "identity_wins": identity <= swapped,
        "ordinary_concentration": concentration,
    }


def unordered_set_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 5 or left.shape[1:3] != (2, 6):
        raise ValueError("expected two equal (N,2,6,H,W) sets")
    identity = _mean_square(left[:, 0], right[:, 0]) + _mean_square(left[:, 1], right[:, 1])
    swapped = _mean_square(left[:, 0], right[:, 1]) + _mean_square(left[:, 1], right[:, 0])
    return torch.minimum(identity, swapped)


def prompt_swap_set_loss(prompt_a: torch.Tensor, prompt_b: torch.Tensor) -> torch.Tensor:
    return unordered_set_distance(swap_decomposition(prompt_a), prompt_b).mean()


def warm_start_condition_c_encoder(model: ThayerMixtureExperts, checkpoint: Path) -> list[dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    rows = []
    for name, target in model.encoder.state_dict().items():
        source = state[name]
        if source.shape != target.shape:
            raise RuntimeError(f"Condition-C encoder tensor mismatch: {name}")
        target.copy_(source)
        rows.append({"condition_c_tensor": name, "thayer_me_tensor": f"encoder.{name}", "shape": "x".join(map(str, source.shape)), "sha256": tensor_sha256(source), "load_rule": "exact_encoder_only"})
    return rows


def set_training_phase(model: ThayerMixtureExperts, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError("phase must be one or two")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for expert in (model.expert_1, model.expert_2):
        for parameter in expert.parameters():
            parameter.requires_grad = True
    if phase == 2:
        for parameter in model.encoder.bottleneck.parameters():
            parameter.requires_grad = True


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)


def expert_parameter_distance(model: ThayerMixtureExperts) -> torch.Tensor:
    left = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.expert_1.parameters()])
    right = torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.expert_2.parameters()])
    return torch.linalg.vector_norm(left - right)
