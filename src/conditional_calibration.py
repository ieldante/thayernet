"""Prospective utilities for the Thayer-Select conditional-calibration campaign.

The module contains no reconstruction model and performs no scene generation.
All trainable objects here are small CPU-only risk or scale heads operating on
already-frozen, model-accessible features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TARGET_COVERAGE = 0.90
MIS_COVERAGE = 1.0 - TARGET_COVERAGE
MIN_CALIBRATION_SUPPORT = 100
MIN_DISTINCT_SOURCE_GROUPS = 80
LOCAL_NEIGHBORS = 400
HEAD_SEEDS = (2026071221, 2026071222, 2026071223, 2026071224, 2026071225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conformal_rank(sample_count: int, coverage: float = TARGET_COVERAGE) -> int:
    """Return the one-indexed finite-sample upper-conformal order statistic."""

    if sample_count < 1 or not 0.0 < coverage < 1.0:
        raise ValueError("invalid sample count or coverage")
    return min(sample_count, int(math.ceil((sample_count + 1) * coverage)))


def conformal_quantile(values: np.ndarray, coverage: float = TARGET_COVERAGE) -> float:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("conformal scores must be nonempty and finite")
    rank = conformal_rank(scores.size, coverage)
    return float(np.partition(scores, rank - 1)[rank - 1])


def order_statistic_resolution(sample_count: int) -> float:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return 1.0 / (sample_count + 1)


def attainable_prevalence_relative_threshold(prevalence: float, alpha: float) -> float:
    """Use a fraction of the remaining achievable AUPRC gap."""

    if not 0.0 <= prevalence <= 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("prevalence and alpha must lie in [0, 1]")
    return float(prevalence + alpha * (1.0 - prevalence))


def fixed_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("tertile source values must be finite and nonempty")
    lower, upper = np.quantile(array, [1.0 / 3.0, 2.0 / 3.0])
    if not lower < upper:
        raise ValueError("tertile boundaries are not distinct")
    return float(lower), float(upper)


def apply_tertiles(values: np.ndarray, edges: tuple[float, float], labels: tuple[str, str, str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("subgroup covariates must be finite")
    lower, upper = edges
    return np.where(array <= lower, labels[0], np.where(array <= upper, labels[1], labels[2])).astype(str)


def effective_sample_size(groups: np.ndarray) -> float:
    labels = np.asarray(groups).astype(str)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 / np.sum(probabilities**2))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            if root_left > root_right:
                root_left, root_right = root_right, root_left
            self.parent[root_right] = root_left


def group_safe_folds(source_a: np.ndarray, source_b: np.ndarray, folds: int = 5) -> np.ndarray:
    """Assign every connected source-group component to one deterministic fold."""

    if folds < 2 or len(source_a) != len(source_b):
        raise ValueError("invalid fold request")
    union = UnionFind()
    for left, right in zip(np.asarray(source_a).astype(str), np.asarray(source_b).astype(str)):
        union.union(left, right)
    assigned = []
    for value in np.asarray(source_a).astype(str):
        root = union.find(value)
        number = int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:16], 16)
        assigned.append(number % folds)
    return np.asarray(assigned, dtype=int)


def verify_fold_isolation(source_a: np.ndarray, source_b: np.ndarray, fold: np.ndarray) -> bool:
    seen: dict[str, int] = {}
    for left, right, current in zip(source_a, source_b, fold):
        for group in (str(left), str(right)):
            if group in seen and seen[group] != int(current):
                return False
            seen[group] = int(current)
    return True


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


class RiskHead(nn.Module):
    """Exactly the three frozen capacity-ablation families R0/R1/R2."""

    def __init__(self, dimensions: int, family: str, width: int = 64) -> None:
        super().__init__()
        self.family = family
        if family == "R0_linear":
            self.input = nn.Identity()
            self.hidden1 = nn.Identity()
            self.hidden2 = nn.Identity()
            self.output = nn.Linear(dimensions, 2)
        elif family == "R1_small_mlp":
            self.input = nn.Linear(dimensions, width)
            self.hidden1 = nn.ReLU()
            self.hidden2 = nn.Identity()
            self.output = nn.Linear(width, 2)
        elif family == "R2_residual_mlp":
            self.input = nn.Linear(dimensions, width)
            self.hidden1 = nn.Linear(width, width)
            self.hidden2 = nn.Linear(width, width)
            self.output = nn.Linear(width, 2)
        else:
            raise ValueError(f"unknown risk-head family: {family}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.family == "R0_linear":
            return self.output(values)
        hidden = F.relu(self.input(values))
        if self.family == "R1_small_mlp":
            return self.output(hidden)
        residual = hidden
        hidden = F.relu(self.hidden1(hidden))
        hidden = F.relu(self.hidden2(hidden) + residual)
        return self.output(hidden)


class ScaleHead(nn.Module):
    def __init__(self, dimensions: int, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).flatten()


@dataclass
class FitResult:
    model: nn.Module
    mean: np.ndarray
    scale: np.ndarray
    best_epoch: int


def pinball(prediction: torch.Tensor, target: torch.Tensor, quantile: float = TARGET_COVERAGE) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


def fit_risk_head(
    family: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    *,
    max_epochs: int = 80,
) -> FitResult:
    """Fixed CPU optimization: Huber central loss plus q=.90 pinball loss."""

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    mean, scale = standardizer(train_x)
    x_train = torch.from_numpy(transform(train_x, mean, scale))
    y_train = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    x_valid = torch.from_numpy(transform(valid_x, mean, scale))
    y_valid = torch.from_numpy(np.asarray(valid_y, dtype=np.float32))
    model = RiskHead(x_train.shape[1], family).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale, best_state = math.inf, -1, 0, None
    for epoch in range(max_epochs):
        model.train()
        indices = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch])
            loss = F.huber_loss(output[:, 0], y_train[batch]) + pinball(output[:, 1], y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(x_valid)
            loss_value = float((F.huber_loss(output[:, 0], y_valid) + pinball(output[:, 1], y_valid)).item())
        if loss_value < best_loss - 1e-5:
            best_loss, best_epoch, stale = loss_value, epoch + 1, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 20 and stale >= 12:
            break
    if best_state is None:
        raise RuntimeError("risk-head fit produced no checkpoint")
    model.load_state_dict(best_state)
    return FitResult(model, mean, scale, best_epoch)


def fit_scale_head(
    seed: int,
    train_x: np.ndarray,
    target_log_scale: np.ndarray,
    valid_x: np.ndarray,
    valid_target_log_scale: np.ndarray,
) -> FitResult:
    torch.manual_seed(seed)
    mean, scale = standardizer(train_x)
    x_train = torch.from_numpy(transform(train_x, mean, scale))
    y_train = torch.from_numpy(np.asarray(target_log_scale, dtype=np.float32))
    x_valid = torch.from_numpy(transform(valid_x, mean, scale))
    y_valid = torch.from_numpy(np.asarray(valid_target_log_scale, dtype=np.float32))
    model = ScaleHead(x_train.shape[1]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_loss, best_epoch, stale, best_state = math.inf, -1, 0, None
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(60):
        model.train()
        indices = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = F.huber_loss(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(F.huber_loss(model(x_valid), y_valid).item())
        if value < best_loss - 1e-5:
            best_loss, best_epoch, stale = value, epoch + 1, 0
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 15 and stale >= 10:
            break
    if best_state is None:
        raise RuntimeError("scale-head fit produced no checkpoint")
    model.load_state_dict(best_state)
    return FitResult(model, mean, scale, best_epoch)


def predict(model: nn.Module, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(transform(values, mean, scale))).numpy()


def deployable_mondrian_group(central: np.ndarray, scale: np.ndarray, central_edges: tuple[float, float], scale_edge: float) -> np.ndarray:
    central_bin = apply_tertiles(central, central_edges, ("central_low", "central_mid", "central_high"))
    scale_bin = np.where(np.asarray(scale) <= scale_edge, "scale_low", "scale_high")
    return np.char.add(np.char.add(central_bin.astype(str), "__"), scale_bin.astype(str))


def crossfit_bounds(
    residual: np.ndarray,
    central: np.ndarray,
    scale: np.ndarray,
    features: np.ndarray,
    fold: np.ndarray,
    method: str,
    deployable_group: np.ndarray,
    *,
    minimum_support: int = MIN_CALIBRATION_SUPPORT,
    neighbors: int = LOCAL_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-safe calibration diagnostics; each row excludes its entire fold."""

    residual = np.asarray(residual, dtype=float)
    central = np.asarray(central, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-4)
    features = np.asarray(features, dtype=float)
    fold = np.asarray(fold, dtype=int)
    result = np.empty(len(residual), dtype=float)
    support = np.zeros(len(residual), dtype=int)
    for current in np.unique(fold):
        fit = fold != current
        test_indices = np.flatnonzero(fold == current)
        global_q = conformal_quantile(residual[fit])
        normalized_global_q = conformal_quantile(residual[fit] / scale[fit])
        for index in test_indices:
            if method == "C0_global":
                result[index] = central[index] + global_q
                support[index] = int(fit.sum())
            elif method == "C1_mondrian":
                pool = fit & (deployable_group == deployable_group[index])
                if int(pool.sum()) < minimum_support:
                    pool = fit
                result[index] = central[index] + conformal_quantile(residual[pool])
                support[index] = int(pool.sum())
            elif method == "C2_normalized":
                result[index] = central[index] + normalized_global_q * scale[index]
                support[index] = int(fit.sum())
            elif method == "C3_local":
                pool_indices = np.flatnonzero(fit)
                distance = np.sum((features[pool_indices] - features[index]) ** 2, axis=1)
                count = min(neighbors, len(pool_indices))
                nearest = pool_indices[np.argpartition(distance, count - 1)[:count]]
                result[index] = central[index] + conformal_quantile(residual[nearest])
                support[index] = count
            elif method == "C4_mondrian_normalized":
                pool = fit & (deployable_group == deployable_group[index])
                if int(pool.sum()) < minimum_support:
                    pool = fit
                result[index] = central[index] + conformal_quantile(residual[pool] / scale[pool]) * scale[index]
                support[index] = int(pool.sum())
            else:
                raise ValueError(f"unknown calibration method: {method}")
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite calibrated bound")
    return result, support
