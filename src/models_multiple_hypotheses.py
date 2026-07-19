"""Compact K=2 prompted decoder with permutation-invariant set supervision."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models_probabilistic_unet import ConvBlock, tensor_sha256


HYPOTHESIS_COUNT = 2
TOKEN_DIMENSION = 8
INPUT_SHAPE = (60, 60)


class ThayerMultipleHypotheses(nn.Module):
    """Condition-C-compatible shared decoder distinguished only by learned tokens."""

    input_channels = 4
    output_channels = 6
    hypothesis_count = HYPOTHESIS_COUNT

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)
        self.decomposition_head = nn.Conv2d(16, 6, 1)
        self.hypothesis_tokens = nn.Parameter(torch.empty(HYPOTHESIS_COUNT, TOKEN_DIMENSION))
        self.bottleneck_token = nn.Linear(TOKEN_DIMENSION, 64)
        self.late_token = nn.Linear(TOKEN_DIMENSION, 32)
        nn.init.normal_(self.hypothesis_tokens, mean=0.0, std=0.02)
        nn.init.zeros_(self.bottleneck_token.bias)
        nn.init.zeros_(self.late_token.bias)

    def forward(self, observed_blend: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        if observed_blend.ndim != 4 or observed_blend.shape[1:] != (3, *INPUT_SHAPE):
            raise ValueError(f"expected blend (N,3,60,60), got {tuple(observed_blend.shape)}")
        if prompt.shape != (len(observed_blend), 1, *INPUT_SHAPE):
            raise ValueError(f"expected prompt (N,1,60,60), got {tuple(prompt.shape)}")
        enc1 = self.enc1(torch.cat((observed_blend, prompt), dim=1))
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2))
        batch = len(observed_blend)
        # Materialize expanded views before MPS linear/concatenation kernels. MPS
        # otherwise sees the tiny token backing buffer with a larger view descriptor.
        tokens = self.hypothesis_tokens[None].expand(batch, -1, -1).contiguous()
        bottleneck = bottleneck[:, None] + self.bottleneck_token(tokens)[:, :, :, None, None]
        bottleneck = bottleneck.reshape(batch * HYPOTHESIS_COUNT, 64, 15, 15)
        enc2_repeat = enc2[:, None].expand(-1, HYPOTHESIS_COUNT, -1, -1, -1).contiguous().reshape(batch * HYPOTHESIS_COUNT, 32, 30, 30)
        enc1_repeat = enc1[:, None].expand(-1, HYPOTHESIS_COUNT, -1, -1, -1).contiguous().reshape(batch * HYPOTHESIS_COUNT, 16, 60, 60)
        up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2_repeat), dim=1))
        late = self.late_token(tokens).reshape(batch * HYPOTHESIS_COUNT, 32, 1, 1)
        dec2 = dec2 + late
        up1 = F.interpolate(dec2, size=(60, 60), mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1_repeat), dim=1))
        output = self.decomposition_head(dec1)
        return output.reshape(batch, HYPOTHESIS_COUNT, 6, 60, 60)


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
    """Per-example component cost; predicted and target end in 6xHxW."""

    pred_requested, pred_companion = split_decomposition(predicted)
    true_requested, true_companion = split_decomposition(target)
    requested = _mean_square(pred_requested, true_requested)
    companion = _mean_square(pred_companion, true_companion)
    summed = _mean_square(pred_requested + pred_companion, true_requested + true_companion)
    return requested + companion + 0.5 * summed


def permutation_invariant_target_loss(
    hypotheses: torch.Tensor,
    targets: torch.Tensor,
    target_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """K=2 set loss with no global slot identity."""

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
        "ordinary_concentration": concentration,
    }


def unordered_set_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Two-permutation unordered MSE distance for K=2 predicted sets."""

    if left.shape != right.shape or left.ndim != 5 or left.shape[1:3] != (2, 6):
        raise ValueError("expected two equal (N,2,6,H,W) sets")
    identity = _mean_square(left[:, 0], right[:, 0]) + _mean_square(left[:, 1], right[:, 1])
    swapped = _mean_square(left[:, 0], right[:, 1]) + _mean_square(left[:, 1], right[:, 0])
    return torch.minimum(identity, swapped)


def prompt_swap_set_loss(prompt_a: torch.Tensor, prompt_b: torch.Tensor) -> torch.Tensor:
    return unordered_set_distance(swap_decomposition(prompt_a), prompt_b).mean()


def warm_start_condition_c(model: ThayerMultipleHypotheses, checkpoint: Path) -> list[dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    model_state = model.state_dict()
    prefixes = ("enc1.", "enc2.", "bottleneck.", "dec2.", "dec1.")
    rows: list[dict[str, object]] = []
    for name, tensor in state.items():
        if name.startswith(prefixes):
            if name not in model_state or model_state[name].shape != tensor.shape:
                raise RuntimeError(f"Condition-C tensor mismatch: {name}")
            model_state[name].copy_(tensor)
            rows.append({"condition_c_tensor": name, "thayer_mh_tensor": name, "shape": "x".join(map(str, tensor.shape)), "sha256": tensor_sha256(tensor), "load_rule": "exact"})
    for suffix in ("weight", "bias"):
        source_name = f"reconstruction_head.{suffix}"
        target_name = f"decomposition_head.{suffix}"
        source = state[source_name]
        target = model_state[target_name]
        target[:3].copy_(source)
        target[3:].copy_(source)
        rows.append({"condition_c_tensor": source_name, "thayer_mh_tensor": f"{target_name}[0:3]", "shape": "x".join(map(str, source.shape)), "sha256": tensor_sha256(source), "load_rule": "copied_to_requested_half"})
        rows.append({"condition_c_tensor": source_name, "thayer_mh_tensor": f"{target_name}[3:6]", "shape": "x".join(map(str, source.shape)), "sha256": tensor_sha256(source), "load_rule": "copied_to_companion_half"})
    model.load_state_dict(model_state)
    return rows


def set_training_phase(model: ThayerMultipleHypotheses, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError("phase must be one or two")
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules: list[nn.Module] = [model.dec2, model.dec1, model.decomposition_head, model.bottleneck_token, model.late_token]
    if phase == 2:
        modules.append(model.bottleneck)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.hypothesis_tokens.requires_grad = True


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)
