"""Small deployable residual-scale models for Thayer-Select.

This module contains no reconstruction model and accepts only already-frozen,
model-accessible feature arrays.  Physical subgroup labels are deliberately
absent from every deployable model API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCALE_SEEDS = (2026071241, 2026071242, 2026071243, 2026071244, 2026071245)
MODEL_FAMILIES = ("M1_log_linear", "M2_one_hidden", "M3_residual", "M4_partial_pool", "M5_soft_gate")
OBJECTIVES = ("O0_huber_log", "O1_q90_pinball", "O2_bounded_gaussian_nll")


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(values) - mean) / scale).astype(np.float32)


def pinball(prediction: torch.Tensor, target: torch.Tensor, quantile: float = 0.90) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


class ScaleNet(nn.Module):
    """Fixed modest scale-model families with bounded log-scale output."""

    def __init__(self, dimensions: int, family: str, width: int = 32, proxy_dimensions: int = 4) -> None:
        super().__init__()
        self.family = family
        self.proxy_dimensions = proxy_dimensions
        if family == "M1_log_linear":
            self.linear = nn.Linear(dimensions, 1)
        elif family == "M2_one_hidden":
            self.hidden = nn.Linear(dimensions, width)
            self.output = nn.Linear(width, 1)
        elif family == "M3_residual":
            self.input = nn.Linear(dimensions, width)
            self.hidden1 = nn.Linear(width, width)
            self.hidden2 = nn.Linear(width, width)
            self.skip = nn.Linear(dimensions, 1)
            self.output = nn.Linear(width, 1)
        elif family == "M4_partial_pool":
            self.global_trunk = nn.Sequential(nn.Linear(dimensions, 16), nn.ReLU(), nn.Linear(16, 1))
            self.correction = nn.Linear(proxy_dimensions, 1, bias=False)
        elif family == "M5_soft_gate":
            self.global_expert = nn.Linear(dimensions, 1)
            self.expert_delta = nn.Linear(dimensions, 3, bias=False)
            self.gate = nn.Linear(proxy_dimensions, 3)
        else:
            raise ValueError(f"unknown scale family: {family}")

    def forward(self, values: torch.Tensor, proxies: torch.Tensor) -> torch.Tensor:
        if self.family == "M1_log_linear":
            return self.linear(values).flatten()
        if self.family == "M2_one_hidden":
            return self.output(F.relu(self.hidden(values))).flatten()
        if self.family == "M3_residual":
            hidden = F.relu(self.input(values))
            residual = hidden
            hidden = F.relu(self.hidden1(hidden))
            hidden = F.relu(self.hidden2(hidden) + residual)
            return (self.output(hidden) + self.skip(values)).flatten()
        if self.family == "M4_partial_pool":
            return (self.global_trunk(values) + self.correction(proxies)).flatten()
        global_value = self.global_expert(values)
        deltas = self.expert_delta(values)
        probabilities = torch.softmax(self.gate(proxies), dim=1)
        return (global_value + torch.sum(probabilities * deltas, dim=1, keepdim=True)).flatten()

    def regularization(self, proxies: torch.Tensor | None = None, multiplier: float = 1.0) -> torch.Tensor:
        parameter = next(self.parameters())
        penalty = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
        if self.family == "M4_partial_pool":
            penalty = 0.10 * torch.mean(self.correction.weight**2)
        elif self.family == "M5_soft_gate":
            penalty = 0.10 * torch.mean(self.expert_delta.weight**2)
            penalty = penalty + 0.05 * torch.mean(self.gate.weight**2)
            if proxies is not None:
                probability = torch.softmax(self.gate(proxies), dim=1)
                entropy = -torch.sum(probability * torch.log(probability + 1e-8), dim=1).mean()
                penalty = penalty + 0.02 * (math.log(3.0) - entropy)
        return multiplier * penalty


@dataclass
class ScaleFit:
    model: ScaleNet
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    proxy_mean: np.ndarray
    proxy_scale: np.ndarray
    best_epoch: int
    best_validation_loss: float
    family: str
    objective: str
    seed: int
    scale_floor: float
    scale_cap: float
    regularization_multiplier: float = 1.0


def objective_loss(
    log_scale: torch.Tensor,
    residual: torch.Tensor,
    objective: str,
    scale_floor: float,
    scale_cap: float,
) -> torch.Tensor:
    bounded = torch.clamp(log_scale, math.log(scale_floor), math.log(scale_cap))
    scale = torch.exp(bounded)
    if objective == "O0_huber_log":
        return F.huber_loss(bounded, torch.log(torch.clamp(residual, min=scale_floor)))
    if objective == "O1_q90_pinball":
        return pinball(scale, residual, 0.90)
    if objective == "O2_bounded_gaussian_nll":
        return torch.mean(bounded + 0.5 * (residual / scale) ** 2 + 0.01 * scale)
    raise ValueError(f"unknown objective: {objective}")


def fit_scale_model(
    family: str,
    objective: str,
    seed: int,
    train_x: np.ndarray,
    train_proxy: np.ndarray,
    train_residual: np.ndarray,
    validation_x: np.ndarray,
    validation_proxy: np.ndarray,
    validation_residual: np.ndarray,
    *,
    scale_floor: float,
    scale_cap: float,
    max_epochs: int = 70,
    regularization_multiplier: float = 1.0,
) -> ScaleFit:
    """Fit one preregistered CPU model with validation-only early stopping."""

    if family not in MODEL_FAMILIES or objective not in OBJECTIVES:
        raise ValueError("unregistered scale family or objective")
    if not 0 < scale_floor < scale_cap:
        raise ValueError("invalid scale floor/cap")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    feature_mean, feature_scale = standardizer(train_x)
    proxy_mean, proxy_scale = standardizer(train_proxy)
    x_train = torch.from_numpy(transform(train_x, feature_mean, feature_scale))
    p_train = torch.from_numpy(transform(train_proxy, proxy_mean, proxy_scale))
    y_train = torch.from_numpy(np.asarray(train_residual, dtype=np.float32))
    x_validation = torch.from_numpy(transform(validation_x, feature_mean, feature_scale))
    p_validation = torch.from_numpy(transform(validation_proxy, proxy_mean, proxy_scale))
    y_validation = torch.from_numpy(np.asarray(validation_residual, dtype=np.float32))
    model = ScaleNet(x_train.shape[1], family).cpu()
    weight_decay = 1e-2 if family == "M1_log_linear" else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale, best_state = math.inf, -1, 0, None
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 512):
            index = order[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[index], p_train[index])
            loss = objective_loss(prediction, y_train[index], objective, scale_floor, scale_cap)
            loss = loss + model.regularization(p_train[index], regularization_multiplier)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x_validation, p_validation)
            value = float(objective_loss(prediction, y_validation, objective, scale_floor, scale_cap).item())
        if value < best_loss - 1e-6:
            best_loss, best_epoch, stale = value, epoch + 1, 0
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 20 and stale >= 12:
            break
    if best_state is None:
        raise RuntimeError("scale fitting produced no checkpoint")
    model.load_state_dict(best_state)
    return ScaleFit(model, feature_mean, feature_scale, proxy_mean, proxy_scale, best_epoch, best_loss,
                    family, objective, seed, scale_floor, scale_cap, regularization_multiplier)


def predict_scale(fit: ScaleFit, values: np.ndarray, proxies: np.ndarray) -> np.ndarray:
    fit.model.eval()
    x = torch.from_numpy(transform(values, fit.feature_mean, fit.feature_scale))
    p = torch.from_numpy(transform(proxies, fit.proxy_mean, fit.proxy_scale))
    with torch.no_grad():
        log_scale = fit.model(x, p).numpy()
    return np.exp(np.clip(log_scale, math.log(fit.scale_floor), math.log(fit.scale_cap)))


def fit_payload(fit: ScaleFit) -> dict:
    return {
        "state_dict": fit.model.state_dict(),
        "feature_mean": fit.feature_mean,
        "feature_scale": fit.feature_scale,
        "proxy_mean": fit.proxy_mean,
        "proxy_scale": fit.proxy_scale,
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.best_validation_loss,
        "family": fit.family,
        "objective": fit.objective,
        "seed": fit.seed,
        "scale_floor": fit.scale_floor,
        "scale_cap": fit.scale_cap,
        "regularization_multiplier": fit.regularization_multiplier,
        "device": "cpu",
        "reconstruction_parameters": 0,
        "oracle_inputs": False,
    }


def load_fit(payload: dict, dimensions: int, proxy_dimensions: int = 4) -> ScaleFit:
    model = ScaleNet(dimensions, payload["family"], proxy_dimensions=proxy_dimensions)
    model.load_state_dict(payload["state_dict"])
    return ScaleFit(model, payload["feature_mean"], payload["feature_scale"], payload["proxy_mean"],
                    payload["proxy_scale"], int(payload["best_epoch"]), float(payload["best_validation_loss"]),
                    payload["family"], payload["objective"], int(payload["seed"]),
                    float(payload["scale_floor"]), float(payload["scale_cap"]),
                    float(payload.get("regularization_multiplier", 1.0)))


def bounded_scale(values: np.ndarray, floor: float, cap: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), floor, cap)


def normalized_scores(truth: np.ndarray, central: np.ndarray, scale: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(np.asarray(truth) - np.asarray(central)) / np.maximum(np.asarray(scale), floor)


def conformal_rank(sample_count: int, coverage: float = 0.90, convention: str = "higher") -> int:
    if sample_count < 1 or not 0 < coverage < 1:
        raise ValueError("invalid conformal request")
    raw = (sample_count + 1) * coverage
    rank = math.ceil(raw) if convention == "higher" else math.floor(raw)
    return min(sample_count, max(1, int(rank)))


def conformal_quantile(values: np.ndarray, coverage: float = 0.90, convention: str = "higher") -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("scores must be nonempty and finite")
    rank = conformal_rank(len(array), coverage, convention)
    return float(np.partition(array, rank - 1)[rank - 1])


def crossfit_normalized_upper(
    truth: np.ndarray,
    central: np.ndarray,
    scale: np.ndarray,
    fold: np.ndarray,
    *,
    scale_floor: float,
    coverage: float = 0.90,
    convention: str = "higher",
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth, dtype=float)
    central = np.asarray(central, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), scale_floor)
    fold = np.asarray(fold, dtype=int)
    score = normalized_scores(truth, central, scale, scale_floor)
    upper = np.empty_like(truth)
    quantiles = np.empty_like(truth)
    for current in np.unique(fold):
        calibration = fold != current
        evaluation = fold == current
        quantile = conformal_quantile(score[calibration], coverage, convention)
        quantiles[evaluation] = quantile
        upper[evaluation] = central[evaluation] + quantile * scale[evaluation]
    return upper, quantiles


def cluster_bootstrap_indices(components: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(components).astype(str)
    unique = np.unique(labels)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    blocks = [np.flatnonzero(labels == label) for label in sampled]
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=int)
