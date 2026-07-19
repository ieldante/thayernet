"""Tiny convex quantile scale models for frozen Thayer-Select proxies."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCALE_SEEDS = (2026071261, 2026071262, 2026071263, 2026071264, 2026071265)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float = 0.90) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


class ConvexQuantileScale(nn.Module):
    """Four convex hinge effects and one optional positive high-high product."""

    def __init__(
        self,
        knots: np.ndarray,
        center_proxy: np.ndarray,
        *,
        scale_floor: float,
        scale_cap: float,
        interaction: bool,
        anchor: float = 0.50,
    ) -> None:
        super().__init__()
        knot_array = np.asarray(knots, dtype=np.float32)
        center_array = np.asarray(center_proxy, dtype=np.float32)
        if knot_array.shape != (4, 5) or center_array.ndim != 2 or center_array.shape[1] != 4:
            raise ValueError("expected four proxies, five knots, and an n-by-four centering array")
        if not 0 < scale_floor < scale_cap or not 0 < anchor < 1:
            raise ValueError("invalid scale bounds or derivative anchor")
        self.register_buffer("knots", torch.from_numpy(knot_array))
        self.register_buffer("anchor_mask", torch.from_numpy((knot_array <= anchor).astype(np.float32)))
        self.scale_floor = float(scale_floor)
        self.scale_cap = float(scale_cap)
        self.anchor = float(anchor)
        self.interaction = bool(interaction)
        self.intercept = nn.Parameter(torch.tensor(inverse_softplus(max(0.10, scale_floor))))
        self.start = nn.Parameter(torch.zeros(4))
        self.delta_raw = nn.Parameter(torch.full((4, 5), -2.0))
        if self.interaction:
            self.gamma_raw = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("gamma_raw", None)
        center_tensor = torch.from_numpy(center_array)
        self.register_buffer("proxy_center", center_tensor.mean(dim=0))
        self.register_buffer(
            "hinge_center",
            F.relu(center_tensor[:, :, None] - self.knots[None, :, :]).mean(dim=0),
        )

    def positive_increments(self) -> torch.Tensor:
        return F.softplus(self.delta_raw)

    def starting_slopes(self) -> torch.Tensor:
        increments = self.positive_increments()
        return F.softplus(self.start) - torch.sum(increments * self.anchor_mask, dim=1)

    def _uncentered_effects(self, proxy: torch.Tensor) -> torch.Tensor:
        hinge = F.relu(proxy[:, :, None] - self.knots[None, :, :])
        return self.starting_slopes()[None, :] * proxy + torch.sum(
            self.positive_increments()[None, :, :] * hinge, dim=2
        )

    def main_effects(self, proxy: torch.Tensor) -> torch.Tensor:
        hinge = F.relu(proxy[:, :, None] - self.knots[None, :, :])
        return self.starting_slopes()[None, :] * (proxy - self.proxy_center[None, :]) + torch.sum(
            self.positive_increments()[None, :, :] * (hinge - self.hinge_center[None, :, :]), dim=2
        )

    def interaction_effect(self, proxy: torch.Tensor) -> torch.Tensor:
        if not self.interaction:
            return torch.zeros(len(proxy), dtype=proxy.dtype, device=proxy.device)
        return (
            F.softplus(self.gamma_raw)
            * F.relu(proxy[:, 0] - self.anchor)
            * F.relu(proxy[:, 1] - self.anchor)
        )

    def forward(self, proxy: torch.Tensor) -> torch.Tensor:
        eta = self.intercept + self.main_effects(proxy).sum(dim=1) + self.interaction_effect(proxy)
        scale = self.scale_floor + F.softplus(eta)
        return torch.clamp(scale, min=self.scale_floor, max=self.scale_cap)

    def penalty(self, roughness: float = 1e-2, interaction_shrinkage: float = 1e-1) -> torch.Tensor:
        value = roughness * torch.sum(self.positive_increments() ** 2)
        if self.interaction:
            value = value + interaction_shrinkage * F.softplus(self.gamma_raw) ** 2
        return value


@dataclass
class ShapeFit:
    model: ConvexQuantileScale
    seed: int
    condition: str
    best_epoch: int
    best_validation_loss: float
    roughness: float
    interaction_shrinkage: float


def fit_shape_model(
    condition: str,
    seed: int,
    train_proxy: np.ndarray,
    train_residual: np.ndarray,
    validation_proxy: np.ndarray,
    validation_residual: np.ndarray,
    knots: np.ndarray,
    *,
    scale_floor: float,
    scale_cap: float,
    anchor: float = 0.50,
    roughness: float = 1e-2,
    interaction_shrinkage: float = 1e-1,
    learning_rate: float = 2e-3,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 20,
) -> ShapeFit:
    if condition not in ("Q1", "Q2"):
        raise ValueError("condition must be Q1 or Q2")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    x_train = torch.from_numpy(np.asarray(train_proxy, dtype=np.float32))
    y_train = torch.from_numpy(np.asarray(train_residual, dtype=np.float32))
    x_validation = torch.from_numpy(np.asarray(validation_proxy, dtype=np.float32))
    y_validation = torch.from_numpy(np.asarray(validation_residual, dtype=np.float32))
    model = ConvexQuantileScale(
        knots,
        np.asarray(train_proxy, dtype=np.float32),
        scale_floor=scale_floor,
        scale_cap=scale_cap,
        interaction=condition == "Q2",
        anchor=anchor,
    ).cpu()
    if sum(parameter.numel() for parameter in model.parameters()) > 64:
        raise RuntimeError("shape model exceeds 64-parameter ceiling")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[index])
            loss = pinball_loss(prediction, y_train[index], 0.90)
            loss = loss + model.penalty(roughness, interaction_shrinkage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(pinball_loss(model(x_validation), y_validation, 0.90).item())
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("shape fit produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return ShapeFit(model, seed, condition, best_epoch, best_loss, roughness, interaction_shrinkage)


def predict_scale(fit: ShapeFit, proxy: np.ndarray) -> np.ndarray:
    fit.model.eval()
    with torch.no_grad():
        return fit.model(torch.from_numpy(np.asarray(proxy, dtype=np.float32))).numpy().astype(float)


def payload(fit: ShapeFit, risk: str) -> dict:
    model = fit.model
    return {
        "state_dict": model.state_dict(),
        "risk": risk,
        "condition": fit.condition,
        "seed": fit.seed,
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.best_validation_loss,
        "scale_floor": model.scale_floor,
        "scale_cap": model.scale_cap,
        "anchor": model.anchor,
        "interaction": model.interaction,
        "roughness": fit.roughness,
        "interaction_shrinkage": fit.interaction_shrinkage,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": "cpu",
        "proxy_order": (
            "estimated_low_local_signal",
            "estimated_local_complexity",
            "high_output_uncertainty",
            "strong_input_output_disagreement",
        ),
        "calibration_outcomes_used": False,
        "physical_labels_used": False,
        "reconstruction_parameters": 0,
    }


def constraint_diagnostics(fit: ShapeFit, grid_size: int = 1001) -> dict:
    model = fit.model
    grid = torch.linspace(0.0, 1.0, grid_size)
    convex_violations = 0
    monotonicity_violations = 0
    with torch.no_grad():
        for index in range(4):
            proxy = torch.zeros((grid_size, 4))
            proxy[:, index] = grid
            effect = model.main_effects(proxy)[:, index].numpy()
            first = np.diff(effect)
            second = np.diff(first)
            convex_violations += int(np.sum(second < -1e-6))
            upper = grid[:-1].numpy() >= model.anchor
            monotonicity_violations += int(np.sum(first[upper] < -1e-6))
        gamma = float(F.softplus(model.gamma_raw).item()) if model.interaction else 0.0
        interaction_min = 0.0
        interaction_max = gamma * (1.0 - model.anchor) ** 2
    return {
        "convexity_violations": convex_violations,
        "upper_half_monotonicity_violations": monotonicity_violations,
        "interaction_coefficient": gamma,
        "interaction_min": interaction_min,
        "interaction_max": interaction_max,
        "interaction_nonnegative": gamma >= 0.0,
    }
