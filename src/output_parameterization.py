"""Fixed-L0 physical output mappings for the Thayer-OP campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models_two_expert_decoder import (
    EXPERT_INITIALIZATION_SEEDS,
    INPUT_SHAPE,
    CompactExpertDecoder,
    SharedPromptEncoder,
)


MAPPINGS = ("relu", "square", "absolute")
# Largest float32 square on the lower side of the frozen 1e-7 numerical-zero
# contract. Its float32 square root maps back to this exact float32 value, so
# all three mappings can begin byte-identically.
INITIAL_PHYSICAL_EPSILON = 9.999999406318238e-08
NUMERICAL_ZERO_TOLERANCE = 1e-7
PHYSICAL_NEGATIVE_TOLERANCE = 0.0
FINITE_VALUE_TOLERANCE = 0.0
# One physical float32 ULP at the frozen P0 maximum (33247.875 electrons).
# The separate linear normalization round trip remains exact; this tolerance
# is for nonlinear inverse-witness reconstruction.
ROUNDTRIP_PHYSICAL_ATOL = 0.00390625
STAGNATION_DERIVATIVE_TOLERANCE = 1e-6


def validate_mapping(mapping: str) -> str:
    if mapping not in MAPPINGS:
        raise ValueError(f"unsupported output mapping: {mapping}")
    return mapping


def apply_output_mapping(raw: torch.Tensor, mapping: str) -> torch.Tensor:
    """Map raw normalized head values to nonnegative normalized source layers."""

    validate_mapping(mapping)
    if mapping == "relu":
        return torch.relu(raw)
    if mapping == "square":
        return raw.square()
    return torch.abs(raw)


def raw_inverse_witness(target: torch.Tensor, mapping: str) -> torch.Tensor:
    """Return the frozen nonnegative inverse branch for representability tests."""

    validate_mapping(mapping)
    if bool(torch.any(target < 0)):
        raise ValueError("physical output mappings require nonnegative targets")
    if mapping == "square":
        return torch.sqrt(target)
    return target.clone()


def mapping_derivative(raw: torch.Tensor, mapping: str) -> torch.Tensor:
    """Return PyTorch's finite pointwise derivative convention for the mapping."""

    validate_mapping(mapping)
    if mapping == "relu":
        return (raw > 0).to(raw)
    if mapping == "square":
        return 2.0 * raw
    # torch.abs supplies subgradient zero at the cusp.
    return torch.sign(raw)


def initial_raw_bias(mapping: str) -> float:
    validate_mapping(mapping)
    if mapping == "square":
        return math.sqrt(INITIAL_PHYSICAL_EPSILON)
    return INITIAL_PHYSICAL_EPSILON


class MappedCompactExpertDecoder(CompactExpertDecoder):
    """Exact L0 expert topology with one prospectively selected head mapping."""

    def __init__(self, mapping: str) -> None:
        self.mapping = validate_mapping(mapping)
        super().__init__()
        with torch.no_grad():
            self.decomposition_head.weight.zero_()
            self.decomposition_head.bias.fill_(initial_raw_bias(mapping))

    def raw_forward(
        self,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        up2 = F.interpolate(bottleneck, size=(30, 30), mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = F.interpolate(dec2, size=INPUT_SHAPE, mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.decomposition_head(dec1)

    def forward(
        self,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        return apply_output_mapping(self.raw_forward(enc1, enc2, bottleneck), self.mapping)


def _seeded_expert(seed: int, mapping: str) -> MappedCompactExpertDecoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MappedCompactExpertDecoder(mapping)


@dataclass(frozen=True)
class PhysicalForward:
    """The single raw-to-mapped-to-physical forward path used by every consumer."""

    raw_normalized: torch.Tensor
    mapped_normalized: torch.Tensor
    physical: torch.Tensor


class MappedThayerMixtureExperts(nn.Module):
    """Exact L0 shared encoder and two independent mapped expert decoders."""

    input_channels = 4
    output_channels = 6
    expert_count = 2

    def __init__(self, mapping: str, scales_grz: torch.Tensor) -> None:
        super().__init__()
        self.mapping = validate_mapping(mapping)
        scales = torch.as_tensor(scales_grz, dtype=torch.float32)
        if scales.shape != (3,) or not bool(torch.all(torch.isfinite(scales))) or not bool(torch.all(scales > 0)):
            raise ValueError("scales_grz must contain three finite positive scales")
        self.encoder = SharedPromptEncoder()
        self.expert_1 = _seeded_expert(EXPERT_INITIALIZATION_SEEDS[0], self.mapping)
        self.expert_2 = _seeded_expert(EXPERT_INITIALIZATION_SEEDS[1], self.mapping)
        self.register_buffer("physical_scales", scales.repeat(2).view(1, 1, 6, 1, 1), persistent=True)

    def encode(
        self,
        observed_blend: torch.Tensor,
        prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(observed_blend, prompt)

    def decode_features(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> PhysicalForward:
        raw = torch.stack(
            (
                self.expert_1.raw_forward(*features),
                self.expert_2.raw_forward(*features),
            ),
            dim=1,
        )
        mapped = apply_output_mapping(raw, self.mapping)
        physical = mapped * self.physical_scales.to(dtype=mapped.dtype, device=mapped.device)
        return PhysicalForward(raw_normalized=raw, mapped_normalized=mapped, physical=physical)

    def forward_outputs(self, observed_blend: torch.Tensor, prompt: torch.Tensor) -> PhysicalForward:
        return self.decode_features(self.encode(observed_blend, prompt))

    def forward(self, observed_blend: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return self.forward_outputs(observed_blend, prompt).physical


def freeze_encoder(model: MappedThayerMixtureExperts) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.encoder.eval()
    for expert in (model.expert_1, model.expert_2):
        for parameter in expert.parameters():
            parameter.requires_grad = True


def encoder_tensor_sha256(model: MappedThayerMixtureExperts) -> str:
    """Hash the exact encoder tensor names, shapes, dtypes, and CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.encoder.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def decoder_parameter_count(model: MappedThayerMixtureExperts) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in model.expert_1.parameters()),
        sum(parameter.numel() for parameter in model.expert_2.parameters()),
    )
