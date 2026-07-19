"""Frozen-feature observability distillation utilities for Thayer-Select.

Physical simulator variables are accepted only as supervised targets by the
training helpers in this module.  Model forward methods accept deployable
arrays only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
from torch import nn
from torch.nn import functional as F


SNR_EDGES = (11.300333, 28.630012)
OBSTRUCTION_EDGES = (0.002057, 0.095043)
FIXED_PRECISION = 0.70
FIXED_RECALL = 0.70


def patch_grid(prompt_xy: torch.Tensor, patch_size: int = 9, radius_pixels: float = 8.0) -> torch.Tensor:
    """Return an align-corners grid centered on deployable prompt coordinates."""

    if prompt_xy.ndim != 2 or prompt_xy.shape[1] != 2:
        raise ValueError("prompt_xy must have shape (batch, 2)")
    offsets = torch.linspace(-radius_pixels, radius_pixels, patch_size, device=prompt_xy.device)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None] + prompt_xy[:, None, None, :]
    return 2.0 * grid / 59.0 - 1.0


def sample_prompt_patch(feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or grid.ndim != 4:
        raise ValueError("feature/grid ranks must be 4")
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def radial_patch_summary(patch: torch.Tensor) -> torch.Tensor:
    """Mean, max, variance, and three radial means for every channel."""

    if patch.ndim != 4 or patch.shape[-1] != patch.shape[-2]:
        raise ValueError("patch must be square NCHW")
    size = patch.shape[-1]
    coordinate = torch.arange(size, device=patch.device, dtype=patch.dtype) - (size - 1) / 2
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    boundaries = (size / 6, size / 3)
    masks = (radius <= boundaries[0], (radius > boundaries[0]) & (radius <= boundaries[1]), radius > boundaries[1])
    values = [patch.mean(dim=(-2, -1)), patch.amax(dim=(-2, -1)), patch.var(dim=(-2, -1), unbiased=False)]
    for mask in masks:
        values.append(patch[:, :, mask].mean(dim=-1))
    return torch.cat(values, dim=1)


def spatial_observation_channels(blend_patch: torch.Tensor, candidate_patch: torch.Tensor) -> torch.Tensor:
    """Blend/candidate/residual plus local x/y gradient magnitudes."""

    residual = blend_patch - candidate_patch
    bases = (blend_patch, candidate_patch, residual)
    gradients = []
    for value in (blend_patch, candidate_patch):
        dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
        dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
        gradients.extend((dx, dy))
    return torch.cat((*bases, *gradients), dim=1)


def spatial_scalar_summary(channels: torch.Tensor) -> torch.Tensor:
    mean = channels.mean(dim=(-2, -1))
    variance = channels.var(dim=(-2, -1), unbiased=False)
    high_frequency = (
        (channels[..., 1:] - channels[..., :-1]).square().mean(dim=(-2, -1))
        + (channels[..., 1:, :] - channels[..., :-1, :]).square().mean(dim=(-2, -1))
    )
    maximum = channels.abs().amax(dim=(-2, -1))
    return torch.cat((mean, variance, high_frequency, maximum), dim=1)


class LinearObservabilityHead(nn.Module):
    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(scalar_dim, 3)

    def forward(self, scalar: torch.Tensor, spatial: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(scalar)


class MLPObservabilityHead(nn.Module):
    def __init__(self, scalar_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(scalar_dim, hidden), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden, 3))

    def forward(self, scalar: torch.Tensor, spatial: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(scalar)


class SpatialObservabilityHead(nn.Module):
    def __init__(self, spatial_channels: int, combined_scalar_dim: int = 0, shared: bool = False) -> None:
        super().__init__()
        width = 32
        blocks: list[nn.Module] = [nn.Conv2d(spatial_channels, width, 3, padding=1), nn.ReLU()]
        if shared:
            blocks.extend((nn.Conv2d(width, width, 3, padding=1), nn.ReLU()))
        self.trunk = nn.Sequential(*blocks)
        self.shared = shared
        self.scalar = nn.Sequential(nn.Linear(combined_scalar_dim, 48), nn.ReLU()) if combined_scalar_dim else None
        output_dim = width + (48 if self.scalar is not None else 0)
        self.output = nn.Sequential(nn.Linear(output_dim, 48), nn.ReLU(), nn.Dropout(0.10), nn.Linear(48, 3))

    def forward(self, scalar: torch.Tensor, spatial: torch.Tensor | None = None) -> torch.Tensor:
        if spatial is None:
            raise ValueError("spatial features are required")
        hidden = self.trunk(spatial).mean(dim=(-2, -1))
        if self.scalar is not None:
            hidden = torch.cat((hidden, self.scalar(scalar)), dim=1)
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass
class FittedObservabilityHead:
    model: nn.Module
    scalar_mean: np.ndarray
    scalar_scale: np.ndarray
    snr_mean: float
    snr_scale: float
    obstruction_mean: float
    obstruction_scale: float
    best_epoch: int
    validation_auroc: float


def binary_auroc(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = int(truth.sum())
    negative = len(truth) - positive
    if positive == 0 or negative == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    return float((ranks[truth == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    ordered = truth[order]
    positive = int(ordered.sum())
    if positive == 0:
        return math.nan
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positive)


def expected_calibration_error(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    truth = np.asarray(truth, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability < edges[index + 1] if index + 1 < bins else probability <= 1.0)
        if mask.any():
            result += float(mask.mean()) * abs(float(truth[mask].mean()) - float(probability[mask].mean()))
    return float(result)


def fixed_operating_metrics(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    ordered = truth[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(truth.sum()), 1)
    recall_at_precision = float(np.max(recall[precision >= FIXED_PRECISION], initial=0.0))
    precision_at_recall = float(np.max(precision[recall >= FIXED_RECALL], initial=0.0))
    return recall_at_precision, precision_at_recall


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: int = 3) -> float:
    values = []
    for label in range(classes):
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def transform_targets(snr: np.ndarray, obstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    snr = np.asarray(snr, dtype=np.float64)
    obstruction = np.asarray(obstruction, dtype=np.float64)
    if np.any(snr <= 0) or np.any(obstruction < 0) or not np.isfinite(snr).all() or not np.isfinite(obstruction).all():
        raise ValueError("invalid physical supervision targets")
    return np.log1p(snr).astype(np.float32), np.log1p(obstruction).astype(np.float32)


def target_bins(snr: np.ndarray, obstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snr_bin = np.searchsorted(np.asarray(SNR_EDGES), np.asarray(snr), side="right")
    obstruction_bin = np.searchsorted(np.asarray(OBSTRUCTION_EDGES), np.asarray(obstruction), side="right")
    joint = ((snr_bin == 0) & (obstruction_bin == 2)).astype(np.float32)
    return snr_bin.astype(int), obstruction_bin.astype(int), joint


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def fit_observability_head(
    model: nn.Module,
    train_scalar: np.ndarray,
    validation_scalar: np.ndarray,
    train_spatial: np.ndarray | None,
    validation_spatial: np.ndarray | None,
    train_snr: np.ndarray,
    validation_snr: np.ndarray,
    train_obstruction: np.ndarray,
    validation_obstruction: np.ndarray,
    train_joint: np.ndarray,
    validation_joint: np.ndarray,
    seed: int,
    max_epochs: int = 60,
    patience: int = 8,
) -> FittedObservabilityHead:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_validation, scalar_mean, scalar_scale = _standardize(
        np.asarray(train_scalar, dtype=np.float32), np.asarray(validation_scalar, dtype=np.float32)
    )
    snr_mean, snr_scale = float(np.mean(train_snr)), float(np.std(train_snr) or 1.0)
    obs_mean, obs_scale = float(np.mean(train_obstruction)), float(np.std(train_obstruction) or 1.0)
    y_train = np.column_stack(((train_snr - snr_mean) / snr_scale, (train_obstruction - obs_mean) / obs_scale, train_joint)).astype(np.float32)
    y_validation = np.column_stack(((validation_snr - snr_mean) / snr_scale, (validation_obstruction - obs_mean) / obs_scale, validation_joint)).astype(np.float32)
    tensors = [torch.from_numpy(x_train), torch.from_numpy(y_train)]
    if train_spatial is not None:
        tensors.append(torch.from_numpy(np.asarray(train_spatial, dtype=np.float32)))
    dataset = torch.utils.data.TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=192, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    prevalence = float(np.mean(train_joint))
    pos_weight = torch.tensor([(1.0 - prevalence) / max(prevalence, 1e-6)], dtype=torch.float32)
    best_state = None
    best_auc = -math.inf
    best_epoch = -1
    stale = 0
    validation_scalar_tensor = torch.from_numpy(x_validation)
    validation_spatial_tensor = None if validation_spatial is None else torch.from_numpy(np.asarray(validation_spatial, dtype=np.float32))
    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            scalar, target = batch[0], batch[1]
            spatial = batch[2] if len(batch) == 3 else None
            output = model(scalar, spatial)
            loss = 0.55 * F.mse_loss(output[:, 0], target[:, 0]) + 0.55 * F.mse_loss(output[:, 1], target[:, 1])
            loss = loss + F.binary_cross_entropy_with_logits(output[:, 2], target[:, 2], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(validation_scalar_tensor, validation_spatial_tensor)[:, 2]).numpy()
        auc = binary_auroc(validation_joint, probability)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("observability fit produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedObservabilityHead(model, scalar_mean, scalar_scale, snr_mean, snr_scale, obs_mean, obs_scale, best_epoch, best_auc)


def predict_observability(fit: FittedObservabilityHead, scalar: np.ndarray, spatial: np.ndarray | None) -> dict[str, np.ndarray]:
    standardized = (np.asarray(scalar, dtype=np.float32) - fit.scalar_mean) / fit.scalar_scale
    spatial_tensor = None if spatial is None else torch.from_numpy(np.asarray(spatial, dtype=np.float32))
    with torch.no_grad():
        output = fit.model(torch.from_numpy(standardized), spatial_tensor).numpy()
    return {
        "snr_transformed": output[:, 0] * fit.snr_scale + fit.snr_mean,
        "obstruction_transformed": output[:, 1] * fit.obstruction_scale + fit.obstruction_mean,
        "joint_probability": 1.0 / (1.0 + np.exp(-np.clip(output[:, 2], -30, 30))),
    }


def regression_metrics(
    true_transformed: np.ndarray,
    predicted_transformed: np.ndarray,
    true_raw: np.ndarray,
    edges: Iterable[float],
) -> dict[str, float]:
    prediction_raw = np.maximum(np.expm1(predicted_transformed), 0.0)
    truth_bin = np.searchsorted(np.asarray(tuple(edges)), true_raw, side="right")
    prediction_bin = np.searchsorted(np.asarray(tuple(edges)), prediction_raw, side="right")
    return {
        "spearman": float(spearmanr(true_transformed, predicted_transformed).statistic),
        "mae_transformed": float(np.mean(np.abs(true_transformed - predicted_transformed))),
        "mae_raw": float(np.mean(np.abs(true_raw - prediction_raw))),
        "bin_macro_f1": macro_f1(truth_bin, prediction_bin),
    }


def classification_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(truth))
    recall_at_precision, precision_at_recall = fixed_operating_metrics(truth, probability)
    ap = average_precision(truth, probability)
    normalized_lift = (ap - prevalence) / max(1.0 - prevalence, 1e-12)
    return {
        "prevalence": prevalence,
        "auroc": binary_auroc(truth, probability),
        "auprc": ap,
        "auprc_over_prevalence": ap / max(prevalence, 1e-12),
        "normalized_ap_lift": normalized_lift,
        "recall_at_precision_0_70": recall_at_precision,
        "precision_at_recall_0_70": precision_at_recall,
        "brier": float(np.mean((probability - truth) ** 2)),
        "ece": expected_calibration_error(truth, probability),
        "unique_scores_6dp": int(len(np.unique(np.round(probability, 6)))),
    }


def connected_component_labels(left: Iterable[str], right: Iterable[str]) -> np.ndarray:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = [(str(a), str(b)) for a, b in zip(left, right)]
    for a, b in pairs:
        union(a, b)
    roots = {root: index for index, root in enumerate(sorted({find(a) for pair in pairs for a in pair}))}
    return np.asarray([roots[find(a)] for a, _ in pairs], dtype=int)


def cluster_bootstrap_metric(
    truth: np.ndarray,
    score: np.ndarray,
    cluster: np.ndarray,
    metric: str,
    replicates: int = 300,
    seed: int = 2026071291,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = np.unique(cluster)
    indices = {group: np.flatnonzero(cluster == group) for group in groups}
    values = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        if metric == "auroc":
            value = binary_auroc(truth[selected], score[selected])
        elif metric == "normalized_ap_lift":
            prevalence = float(np.mean(truth[selected]))
            ap = average_precision(truth[selected], score[selected])
            value = (ap - prevalence) / max(1.0 - prevalence, 1e-12)
        else:
            raise ValueError(metric)
        if np.isfinite(value):
            values.append(value)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
