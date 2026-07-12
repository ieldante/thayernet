#!/usr/bin/env python3
"""Shared frozen components for Thayer-Select recoverability Phase II."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.prompt_semantics import PromptSemantics, QueryClass, associate_prompt
from src.recoverability import PHASE2_CONTRACTS, phase2_contract_success
from thayer_select_prompt_ablation_common import (
    CompactSelectNet,
    gaussian_prompt_numpy,
    read_csv,
    sha256_array,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)

FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PHASE1 / "manifests/source_split_manifest.csv"
TEACHER_CHECKPOINT = PHASE1 / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PHASE1 / "manifests/normalization.json"
IMAGE_SIZE = 60
BANDS = ("g", "r", "z")
PIXEL_SCALE_ARCSEC = 0.2
PROMPT_SIGMA_PIXELS = 2.0
SEMANTICS = PromptSemantics()
PARTITION_COUNTS = {"training": 10_000, "validation": 1_500, "calibration": 2_000}
QUERY_FRACTIONS = {
    QueryClass.VALID_SOURCE: 0.55,
    QueryClass.PERTURBED_VALID: 0.15,
    QueryClass.NULL_SOURCE: 0.20,
    QueryClass.AMBIGUOUS_SOURCE: 0.10,
}
DEVELOPMENT_COUNTS = {
    QueryClass.VALID_SOURCE: 900,
    QueryClass.PERTURBED_VALID: 300,
    QueryClass.NULL_SOURCE: 400,
    QueryClass.AMBIGUOUS_SOURCE: 400,
}
SCENE_SEEDS = {"training": 2026077100, "validation": 2026077200, "calibration": 2026077300, "development_test": 2026077400}
NOISE_SEEDS = {"training": 2026087100, "validation": 2026087200, "calibration": 2026087300, "development_test": 2026087400}
PROMPT_SEEDS = {"training": 2026097100, "validation": 2026097200, "calibration": 2026097300, "development_test": 2026097400}
TRAINING_SEED = 2026078101
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MIN_LOG_VARIANCE = -8.0
MAX_LOG_VARIANCE = 2.0
UNCERTAINTY_SATURATION_WEIGHT = 2.0
MAX_UNCERTAINTY_SATURATION_FRACTION = 0.25
PRIMARY_CONTRACT_CANDIDATE = "moderate"
CONTRACT_BALANCE_RANGE = (0.05, 0.95)
MANIFEST_SCHEMA_VERSION = "thayer-select-recoverability-manifest-v1"


def write_csv_union_fresh(path: Path, rows: list[dict]) -> None:
    """Write heterogeneous metric rows without order-dependent field loss."""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    write_csv_fresh(path, rows, fieldnames=fieldnames)


def load_scales() -> np.ndarray:
    value = json.loads(NORMALIZATION.read_text())
    scales = np.asarray(value["per_band_scale"], dtype=np.float32)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("Invalid immutable Phase-I normalization")
    return scales


def load_teacher(device: torch.device) -> CompactSelectNet:
    payload = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("condition") != "C_randomized_coordinate_prompt":
        raise RuntimeError("Unexpected frozen teacher condition")
    model = CompactSelectNet(4).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if next(model.parameters()).device.type != "mps":
        raise RuntimeError("Teacher inference must use MPS")
    return model


def query_counts(count: int) -> dict[QueryClass, int]:
    raw = {query: int(round(count * fraction)) for query, fraction in QUERY_FRACTIONS.items()}
    raw[QueryClass.VALID_SOURCE] += count - sum(raw.values())
    if sum(raw.values()) != count:
        raise RuntimeError("Query composition did not sum to partition count")
    return raw


def draw_scene_positions(rng: np.random.Generator, query: QueryClass) -> np.ndarray:
    """Draw exchangeable two-source positions in BTK tangent-plane arcsec."""

    for _ in range(10_000):
        if query is QueryClass.AMBIGUOUS_SOURCE:
            separation = rng.uniform(0.25, 1.15)
        else:
            separation = rng.uniform(0.65, 4.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        midpoint = rng.uniform(-1.4, 1.4, size=2)
        delta = 0.5 * separation * np.asarray([np.cos(angle), np.sin(angle)])
        positions = np.stack((midpoint - delta, midpoint + delta))
        if np.max(np.abs(positions)) < 2.8:
            return positions
    raise RuntimeError("Could not draw in-frame source positions")


def choose_prompt(
    source_xy: np.ndarray,
    query: QueryClass,
    seed: int,
    requested_index: int,
) -> tuple[np.ndarray, object]:
    """Construct a prompt that exactly reproduces its declared semantics."""

    rng = np.random.default_rng(seed)
    xy = np.asarray(source_xy, dtype=np.float64)
    if query is QueryClass.VALID_SOURCE:
        prompt = xy[requested_index].copy()
    elif query is QueryClass.PERTURBED_VALID:
        prompt = None
        for _ in range(10_000):
            radius = rng.uniform(0.75, 3.5)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            candidate = xy[requested_index] + radius * np.asarray([np.cos(angle), np.sin(angle)])
            if np.all(candidate >= 0.0) and candidate[0] <= IMAGE_SIZE - 1 and candidate[1] <= IMAGE_SIZE - 1:
                association = associate_prompt(xy, candidate, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
                if association.query_class is query and association.matched_index == requested_index:
                    prompt = candidate
                    break
        if prompt is None:
            raise RuntimeError("Could not construct a uniquely perturbed valid prompt")
    elif query is QueryClass.NULL_SOURCE:
        prompt = None
        for attempt in range(10_000):
            if attempt % 3 == 0:
                edge = rng.integers(0, 4)
                candidate = rng.uniform(0.0, IMAGE_SIZE - 1, size=2)
                candidate[edge // 2] = rng.uniform(0.0, 1.0) if edge % 2 == 0 else rng.uniform(IMAGE_SIZE - 2, IMAGE_SIZE - 1)
            else:
                candidate = rng.uniform(0.0, IMAGE_SIZE - 1, size=2)
            association = associate_prompt(xy, candidate, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
            if association.query_class is query:
                prompt = candidate
                break
        if prompt is None:
            raise RuntimeError("Could not construct a null prompt")
    else:
        midpoint = np.mean(xy, axis=0)
        delta = xy[1] - xy[0]
        norm = max(float(np.linalg.norm(delta)), 1e-12)
        perpendicular = np.asarray([-delta[1], delta[0]]) / norm
        prompt = midpoint + rng.uniform(-0.2, 0.2) * perpendicular
    association = associate_prompt(xy, prompt, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
    if association.query_class is not query:
        raise RuntimeError(f"Prompt semantics mismatch: requested {query}, observed {association.query_class}")
    return np.asarray(prompt, dtype=np.float64), association


def source_mask(target: np.ndarray) -> np.ndarray:
    brightness = np.max(np.abs(np.asarray(target, dtype=np.float64)), axis=0)
    peak = float(np.max(brightness))
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(brightness.shape, dtype=bool)
    return brightness > 0.01 * peak


def _centroid(image: np.ndarray) -> tuple[float, float]:
    weight = np.maximum(np.sum(np.asarray(image, dtype=np.float64), axis=0), 0.0)
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        return math.nan, math.nan
    yy, xx = np.mgrid[: image.shape[-2], : image.shape[-1]]
    return float(np.sum(xx * weight) / total), float(np.sum(yy * weight) / total)


def _color(flux_one: float, flux_two: float) -> float:
    if flux_one <= 0 or flux_two <= 0 or not np.isfinite((flux_one, flux_two)).all():
        return math.nan
    return float(-2.5 * np.log10(flux_one / flux_two))


def outcome_metrics(
    prediction: np.ndarray,
    blend: np.ndarray,
    isolated: np.ndarray,
    query_class: str | QueryClass,
    matched_index: int | None,
) -> dict[str, float | int | bool]:
    """Compute oracle outcomes in physical electron-count image units."""

    query = QueryClass(query_class)
    pred = np.asarray(prediction, dtype=np.float64)
    blend64 = np.asarray(blend, dtype=np.float64)
    truth = np.asarray(isolated, dtype=np.float64)
    valid = pred.shape == blend64.shape == truth.shape[1:] and np.isfinite(pred).all()
    predicted_abs_flux = float(np.sum(np.abs(pred))) if valid else math.inf
    blend_abs_flux = float(np.sum(np.abs(blend64))) if np.isfinite(blend64).all() else math.nan
    base = {
        "evaluation_valid": bool(valid),
        "predicted_absolute_flux": predicted_abs_flux,
        "blend_absolute_flux": blend_abs_flux,
        "predicted_to_blend_flux_ratio": predicted_abs_flux / max(blend_abs_flux, 1e-30),
        "hallucination": False,
        "forced_source_selection": False,
        "source_confusion": False,
    }
    if not valid:
        return {**base, "normalized_rmse": math.inf, "source_mse": math.inf, "source_mae": math.inf, "whole_mse": math.inf, "whole_mae": math.inf, "max_relative_flux_error": math.inf, "max_color_error_mag": math.inf, "centroid_error_pixels": math.inf, "catastrophic_failure": True}
    if query is QueryClass.NULL_SOURCE:
        ratio = predicted_abs_flux / max(blend_abs_flux, 1e-30)
        hallucination = ratio > PHASE2_CONTRACTS["moderate"].hallucination_flux_fraction
        return {**base, "hallucination": hallucination, "normalized_rmse": ratio, "source_mse": float(np.mean(pred**2)), "source_mae": float(np.mean(np.abs(pred))), "whole_mse": float(np.mean(pred**2)), "whole_mae": float(np.mean(np.abs(pred))), "max_relative_flux_error": ratio, "max_color_error_mag": math.nan, "centroid_error_pixels": math.nan, "catastrophic_failure": ratio > 1.0}
    if query is QueryClass.AMBIGUOUS_SOURCE:
        zero_mse = float(np.mean(pred**2))
        source_mses = [float(np.mean((pred - truth[index]) ** 2)) for index in range(len(truth))]
        forced = predicted_abs_flux > 0.10 * max(blend_abs_flux, 1e-30) and min(source_mses) < zero_mse
        return {**base, "forced_source_selection": forced, "normalized_rmse": predicted_abs_flux / max(blend_abs_flux, 1e-30), "source_mse": math.nan, "source_mae": math.nan, "whole_mse": zero_mse, "whole_mae": float(np.mean(np.abs(pred))), "max_relative_flux_error": math.nan, "max_color_error_mag": math.nan, "centroid_error_pixels": math.nan, "catastrophic_failure": forced}
    if matched_index is None or matched_index not in range(len(truth)):
        return {**base, "evaluation_valid": False, "normalized_rmse": math.inf, "source_mse": math.inf, "source_mae": math.inf, "whole_mse": math.inf, "whole_mae": math.inf, "max_relative_flux_error": math.inf, "max_color_error_mag": math.inf, "centroid_error_pixels": math.inf, "catastrophic_failure": True}
    target = truth[matched_index]
    alternate = truth[1 - matched_index] if len(truth) == 2 else None
    mask = source_mask(target)
    error = pred - target
    source_values = error[:, mask]
    target_values = target[:, mask]
    source_mse = float(np.mean(source_values**2)) if source_values.size else math.inf
    source_mae = float(np.mean(np.abs(source_values))) if source_values.size else math.inf
    normalized_rmse = float(np.sqrt(np.sum(source_values**2) / max(np.sum(target_values**2), 1e-30))) if source_values.size else math.inf
    target_flux = target.sum(axis=(-2, -1)); predicted_flux = pred.sum(axis=(-2, -1))
    relative_flux = np.abs(predicted_flux - target_flux) / np.maximum(np.abs(target_flux), 1e-30)
    target_colors = (_color(target_flux[0], target_flux[1]), _color(target_flux[1], target_flux[2]))
    pred_colors = (_color(predicted_flux[0], predicted_flux[1]), _color(predicted_flux[1], predicted_flux[2]))
    color_errors = [abs(a - b) if np.isfinite((a, b)).all() else math.inf for a, b in zip(target_colors, pred_colors)]
    target_centroid = _centroid(target); predicted_centroid = _centroid(pred)
    centroid_error = float(np.linalg.norm(np.asarray(target_centroid) - np.asarray(predicted_centroid))) if np.isfinite((*target_centroid, *predicted_centroid)).all() else math.inf
    confusion = bool(alternate is not None and np.mean((pred - alternate) ** 2) < np.mean((pred - target) ** 2))
    catastrophic = bool(normalized_rmse > 2.0 or np.max(relative_flux) > 1.0 or centroid_error > 5.0)
    return {
        **base,
        "normalized_rmse": normalized_rmse,
        "source_mse": source_mse,
        "source_mae": source_mae,
        "whole_mse": float(np.mean(error**2)),
        "whole_mae": float(np.mean(np.abs(error))),
        "g_relative_flux_error": float(relative_flux[0]),
        "r_relative_flux_error": float(relative_flux[1]),
        "z_relative_flux_error": float(relative_flux[2]),
        "max_relative_flux_error": float(np.max(relative_flux)),
        "g_minus_r_color_error": color_errors[0],
        "r_minus_z_color_error": color_errors[1],
        "max_color_error_mag": float(max(color_errors)),
        "centroid_error_pixels": centroid_error,
        "source_confusion": confusion,
        "catastrophic_failure": catastrophic,
    }


def add_contract_labels(metrics: dict, query_class: str | QueryClass) -> dict:
    return {
        **metrics,
        **{
            f"{name}_success": int(phase2_contract_success(metrics, query_class, contract))
            for name, contract in PHASE2_CONTRACTS.items()
        },
    }


def add_actionable_acceptance_labels(metrics: dict, query_class: str | QueryClass) -> dict:
    """Mark only reliable source reconstructions as acceptable actions.

    Null and ambiguous queries have dedicated recognition/abstention semantics;
    even a low-energy null reconstruction must not become a positive acceptance
    example for the global selective-prediction score.
    """

    labeled = add_contract_labels(metrics, query_class)
    source_query = QueryClass(query_class) in (QueryClass.VALID_SOURCE, QueryClass.PERTURBED_VALID)
    for name in PHASE2_CONTRACTS:
        labeled[f"{name}_actionable_success"] = int(source_query and bool(labeled[f"{name}_success"]))
    return labeled


def model_parameter_counts() -> dict[str, int]:
    from src.models_thayer_select import ThayerSelectNet

    r0 = sum(parameter.numel() for parameter in CompactSelectNet(4).parameters())
    r1 = sum(parameter.numel() for parameter in ThayerSelectNet(min_log_variance=MIN_LOG_VARIANCE, max_log_variance=MAX_LOG_VARIANCE).parameters())
    return {"R0": r0, "R1": r1, "R1_minus_R0": r1 - r0}
