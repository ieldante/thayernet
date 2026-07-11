"""Streaming source-artifact audit for the Galaxy10 DECaLS HDF5 cutouts.

This is a heuristic candidate finder, not an automatic cleaning tool. It reads
the image dataset in bounded HDF5 blocks, computes transparent source-quality
scores, and writes only append-only diagnostics under a fresh ``outputs/runs``
directory. It never changes the HDF5 file, checkpoints, or shared documents.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/default.yaml"
HEURISTIC_VERSION = "source_artifact_audit_v1"
FLAG_COLUMNS = (
    "flag_axis_line_artifact",
    "flag_diagonal_line_artifact",
    "flag_edge_touching_foreground",
    "flag_saturation",
    "flag_channel_color_streak",
    "flag_blank_image",
    "flag_dark_image",
    "flag_compression_blockiness",
    "flag_extreme_channel_ratio",
    "flag_large_edge_mask",
)


@dataclass(frozen=True)
class AuditSettings:
    border_width: int = 12
    edge_width: int = 12
    stream_block_size: int = 555
    progress_every: int = 500
    top_per_sheet: int = 36
    max_images: int | None = None


@dataclass(frozen=True)
class Geometry:
    outer_mask: np.ndarray
    edge_mask: np.ndarray
    diag_main_ids: np.ndarray
    diag_anti_ids: np.ndarray
    diag_main_counts: np.ndarray
    diag_anti_counts: np.ndarray
    min_profile_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a streaming heuristic source-artifact audit."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--border-width", type=int, default=12)
    parser.add_argument("--edge-width", type=int, default=12)
    parser.add_argument("--stream-block-size", type=int, default=555)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--top-per-sheet", type=int, default=36)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional development-only prefix limit; omit for the full audit.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return config


def make_run_dir(output_root: Path, stamp: str) -> Path:
    if Path(stamp).name != stamp or not stamp:
        raise ValueError("stamp must be a non-empty filename component.")
    run_dir = output_root / "runs" / f"source_artifact_audit_{stamp}"
    try:
        run_dir.resolve().relative_to((PROJECT_ROOT / "outputs/runs").resolve())
    except ValueError as exc:
        raise ValueError("Audit output must remain under ignored outputs/runs/.") from exc
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("tables", "diagnostics", "logs", "figures"):
        (run_dir / child).mkdir(exist_ok=False)
    return run_dir


def safe_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
        handle.write("\n")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def checkpoint_snapshot(checkpoint_dir: Path) -> dict[str, Any]:
    rows = []
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pth")):
            stat = path.stat()
            rows.append(
                {
                    "path": project_relative(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "mtime_iso_local": datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                }
            )
    return {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }


def compare_checkpoint_snapshots(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    old = {row["path"]: row for row in before["checkpoints"]}
    new = {row["path"]: row for row in after["checkpoints"]}
    rows = []
    for path, old_row in old.items():
        new_row = new.get(path)
        unchanged = bool(
            new_row is not None
            and old_row["size_bytes"] == new_row["size_bytes"]
            and old_row["mtime_ns"] == new_row["mtime_ns"]
        )
        rows.append(
            {
                "path": path,
                "before": old_row,
                "after": new_row,
                "unchanged": unchanged,
            }
        )
    return {
        "old_checkpoint_count": len(rows),
        "old_checkpoints_unchanged": all(row["unchanged"] for row in rows),
        "comparisons": rows,
    }


def make_split_lookup(
    n_images: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.arange(n_images)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_train = int(n_images * train_fraction)
    n_val = int(n_images * val_fraction)
    split_code = np.empty(n_images, dtype=np.int8)
    split_position = np.empty(n_images, dtype=np.int32)
    shuffled_position = np.empty(n_images, dtype=np.int32)
    for code, indices in enumerate(
        (order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :])
    ):
        split_code[indices] = code
        split_position[indices] = np.arange(len(indices), dtype=np.int32)
    shuffled_position[order] = np.arange(n_images, dtype=np.int32)
    return split_code, split_position, shuffled_position


def make_geometry(height: int, width: int, edge_width: int) -> Geometry:
    y_grid, x_grid = np.indices((height, width))
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.hypot(y_grid - center_y, x_grid - center_x)
    outer_mask = radius >= 0.25 * min(height, width)
    edge = max(1, min(edge_width, height // 4, width // 4))
    edge_mask = (
        (y_grid < edge)
        | (y_grid >= height - edge)
        | (x_grid < edge)
        | (x_grid >= width - edge)
    )
    interior_y = y_grid[1:-1, 1:-1]
    interior_x = x_grid[1:-1, 1:-1]
    interior_outer = outer_mask[1:-1, 1:-1]
    main_ids = interior_y - interior_x + width - 1
    anti_ids = interior_y + interior_x
    n_groups = height + width - 1
    main_counts = np.bincount(
        main_ids[interior_outer].ravel(), minlength=n_groups
    )
    anti_counts = np.bincount(
        anti_ids[interior_outer].ravel(), minlength=n_groups
    )
    return Geometry(
        outer_mask=outer_mask,
        edge_mask=edge_mask,
        diag_main_ids=main_ids.astype(np.int32),
        diag_anti_ids=anti_ids.astype(np.int32),
        diag_main_counts=main_counts,
        diag_anti_counts=anti_counts,
        min_profile_length=max(32, int(0.35 * min(height, width))),
    )


def border_pixels(image: np.ndarray, width: int) -> np.ndarray:
    height, image_width, channels = image.shape
    width = max(1, min(width, height // 3, image_width // 3))
    return np.concatenate(
        (
            image[:width].reshape(-1, channels),
            image[-width:].reshape(-1, channels),
            image[width:-width, :width].reshape(-1, channels),
            image[width:-width, -width:].reshape(-1, channels),
        ),
        axis=0,
    )


def axis_profile(
    response: np.ndarray,
    valid: np.ndarray,
    horizontal: bool,
    min_length: int,
) -> tuple[float, float, float, int]:
    if horizontal:
        sums = np.sum(np.where(valid, response, 0.0), axis=1)
        counts = np.sum(valid, axis=1)
    else:
        sums = np.sum(np.where(valid, response, 0.0), axis=0)
        counts = np.sum(valid, axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts >= min_length,
    )
    if not np.any(np.isfinite(means)):
        return 0.0, 0.0, 0.0, -1
    location = int(np.nanargmax(means))
    score = float(means[location])
    sampled = response[valid][::8]
    baseline = float(np.median(sampled)) if sampled.size else 0.0
    snr = score / (baseline + 1e-4)
    line_values = (
        response[location][valid[location]]
        if horizontal
        else response[:, location][valid[:, location]]
    )
    support = float(
        np.mean(line_values > max(0.008, 4.0 * baseline))
    ) if line_values.size else 0.0
    return score, snr, support, location


def diagonal_profile(
    response: np.ndarray,
    valid: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    min_length: int,
) -> tuple[float, float, float, int]:
    n_groups = len(counts)
    selected_ids = ids[valid]
    sums = np.bincount(
        selected_ids.ravel(),
        weights=response[valid].astype(np.float64).ravel(),
        minlength=n_groups,
    )
    means = np.divide(
        sums,
        counts,
        out=np.full(n_groups, np.nan, dtype=np.float64),
        where=counts >= min_length,
    )
    if not np.any(np.isfinite(means)):
        return 0.0, 0.0, 0.0, -1
    location = int(np.nanargmax(means))
    score = float(means[location])
    sampled = response[valid][::8]
    baseline = float(np.median(sampled)) if sampled.size else 0.0
    snr = score / (baseline + 1e-4)
    line_values = response[valid & (ids == location)]
    support = float(
        np.mean(line_values > max(0.008, 4.0 * baseline))
    ) if line_values.size else 0.0
    return score, snr, support, location


def orientation_scores(
    plane: np.ndarray, geometry: Geometry
) -> dict[str, float | int | str]:
    outer = geometry.outer_mask
    horizontal_response = np.abs(
        plane[1:-1, :] - 0.5 * (plane[:-2, :] + plane[2:, :])
    )
    vertical_response = np.abs(
        plane[:, 1:-1] - 0.5 * (plane[:, :-2] + plane[:, 2:])
    )
    center = plane[1:-1, 1:-1]
    main_response = np.abs(
        center - 0.5 * (plane[:-2, 2:] + plane[2:, :-2])
    )
    anti_response = np.abs(
        center - 0.5 * (plane[:-2, :-2] + plane[2:, 2:])
    )
    h = axis_profile(
        horizontal_response,
        outer[1:-1, :],
        True,
        geometry.min_profile_length,
    )
    v = axis_profile(
        vertical_response,
        outer[:, 1:-1],
        False,
        geometry.min_profile_length,
    )
    interior_outer = outer[1:-1, 1:-1]
    main = diagonal_profile(
        main_response,
        interior_outer,
        geometry.diag_main_ids,
        geometry.diag_main_counts,
        geometry.min_profile_length,
    )
    anti = diagonal_profile(
        anti_response,
        interior_outer,
        geometry.diag_anti_ids,
        geometry.diag_anti_counts,
        geometry.min_profile_length,
    )
    axis_name, axis = max(("horizontal", h), ("vertical", v), key=lambda item: item[1][0])
    diag_name, diag = max(("main_diagonal", main), ("anti_diagonal", anti), key=lambda item: item[1][0])
    all_name, all_values = max(
        ("horizontal", h),
        ("vertical", v),
        ("main_diagonal", main),
        ("anti_diagonal", anti),
        key=lambda item: item[1][0],
    )
    return {
        "axis_score": axis[0],
        "axis_snr": axis[1],
        "axis_support": axis[2],
        "axis_location": axis[3],
        "axis_orientation": axis_name,
        "diagonal_score": diag[0],
        "diagonal_snr": diag[1],
        "diagonal_support": diag[2],
        "diagonal_location": diag[3],
        "diagonal_orientation": diag_name,
        "all_score": all_values[0],
        "all_snr": all_values[1],
        "all_support": all_values[2],
        "all_location": all_values[3],
        "all_orientation": all_name,
    }


def compute_image_scores(
    image_uint8: np.ndarray,
    geometry: Geometry,
    settings: AuditSettings,
) -> dict[str, Any]:
    image = image_uint8.astype(np.float32) / 255.0
    height, width, _channels = image.shape
    gray = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )
    border = border_pixels(image, settings.border_width)
    border_background = np.median(border, axis=0)
    border_gray = (
        0.2126 * border[:, 0]
        + 0.7152 * border[:, 1]
        + 0.0722 * border[:, 2]
    )
    background_gray = float(
        0.2126 * border_background[0]
        + 0.7152 * border_background[1]
        + 0.0722 * border_background[2]
    )
    border_median = float(np.median(border_gray))
    border_mad = float(1.4826 * np.median(np.abs(border_gray - border_median)))
    signal = np.clip(gray - background_gray, 0.0, None)
    foreground_threshold = max(0.025, 4.0 * border_mad)
    foreground = signal > foreground_threshold
    foreground_count = int(foreground.sum())
    edge_foreground_count = int((foreground & geometry.edge_mask).sum())
    edge_mask_fraction_total = edge_foreground_count / float(height * width)
    edge_foreground_fraction = float(
        foreground[geometry.edge_mask].mean()
    )
    edge_touch_ratio = edge_foreground_count / max(foreground_count, 1)
    edge_width = settings.edge_width
    side_fractions = (
        float(foreground[:edge_width].mean()),
        float(foreground[-edge_width:].mean()),
        float(foreground[:, :edge_width].mean()),
        float(foreground[:, -edge_width:].mean()),
    )
    edge_sides_touched = int(sum(value >= 0.003 for value in side_fractions))

    gray_orientation = orientation_scores(gray, geometry)
    chroma_spread = np.max(image, axis=2) - np.min(image, axis=2)
    color_orientation = orientation_scores(chroma_spread, geometry)

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    x_boundary = (np.arange(width - 1) + 1) % 8 == 0
    y_boundary = (np.arange(height - 1) + 1) % 8 == 0
    boundary_values = np.concatenate((dx[:, x_boundary].ravel(), dy[y_boundary].ravel()))
    interior_values = np.concatenate((dx[:, ~x_boundary].ravel(), dy[~y_boundary].ravel()))
    boundary_mean = float(np.mean(boundary_values))
    interior_mean = float(np.mean(interior_values))
    blockiness_difference = boundary_mean - interior_mean
    blockiness_excess_ratio = blockiness_difference / (interior_mean + 1e-4)

    foreground_rgb = np.clip(image - border_background.reshape(1, 1, 3), 0.0, None)
    channel_flux = foreground_rgb.sum(axis=(0, 1), dtype=np.float64)
    sorted_flux = np.sort(channel_flux)
    channel_flux_ratio = float((sorted_flux[-1] + 1e-6) / (sorted_flux[0] + 1e-6))
    channel_dominance = float(channel_flux.max() / (channel_flux.sum() + 1e-6))
    color_outlier_fraction = float(
        np.mean((chroma_spread > 0.35) & (gray > max(0.03, background_gray + 0.01)))
    )
    saturation_any_fraction = float(np.mean(np.any(image_uint8 >= 250, axis=2)))
    saturation_all_fraction = float(np.mean(np.all(image_uint8 >= 250, axis=2)))
    p01, p99 = np.percentile(gray, (1.0, 99.0))
    return {
        "mean_luminance": float(gray.mean()),
        "std_luminance": float(gray.std()),
        "p01_luminance": float(p01),
        "p99_luminance": float(p99),
        "dynamic_range_p99_p01": float(p99 - p01),
        "border_background_r": float(border_background[0]),
        "border_background_g": float(border_background[1]),
        "border_background_b": float(border_background[2]),
        "border_noise_mad": border_mad,
        "foreground_threshold": foreground_threshold,
        "foreground_fraction": float(foreground.mean()),
        "foreground_signal_mean": float(signal.mean()),
        "edge_foreground_fraction": edge_foreground_fraction,
        "edge_mask_fraction_total": edge_mask_fraction_total,
        "edge_touch_ratio": float(edge_touch_ratio),
        "edge_sides_touched": edge_sides_touched,
        "axis_line_score": float(gray_orientation["axis_score"]),
        "axis_line_snr": float(gray_orientation["axis_snr"]),
        "axis_line_support": float(gray_orientation["axis_support"]),
        "axis_line_orientation": str(gray_orientation["axis_orientation"]),
        "axis_line_location": int(gray_orientation["axis_location"]),
        "diagonal_line_score": float(gray_orientation["diagonal_score"]),
        "diagonal_line_snr": float(gray_orientation["diagonal_snr"]),
        "diagonal_line_support": float(gray_orientation["diagonal_support"]),
        "diagonal_line_orientation": str(gray_orientation["diagonal_orientation"]),
        "diagonal_line_location": int(gray_orientation["diagonal_location"]),
        "channel_streak_score": float(color_orientation["all_score"]),
        "channel_streak_snr": float(color_orientation["all_snr"]),
        "channel_streak_support": float(color_orientation["all_support"]),
        "channel_streak_orientation": str(color_orientation["all_orientation"]),
        "channel_streak_location": int(color_orientation["all_location"]),
        "saturation_any_fraction": saturation_any_fraction,
        "saturation_all_fraction": saturation_all_fraction,
        "color_outlier_fraction": color_outlier_fraction,
        "channel_flux_r": float(channel_flux[0]),
        "channel_flux_g": float(channel_flux[1]),
        "channel_flux_b": float(channel_flux[2]),
        "channel_flux_ratio_max_min": channel_flux_ratio,
        "channel_flux_dominance": channel_dominance,
        "block_boundary_mean_difference": boundary_mean,
        "block_interior_mean_difference": interior_mean,
        "blockiness_difference": float(blockiness_difference),
        "blockiness_excess_ratio": float(blockiness_excess_ratio),
    }


def upper_threshold(
    frame: pd.DataFrame, column: str, quantile: float, floor: float
) -> float:
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    return max(floor, float(values.quantile(quantile)))


def lower_threshold(
    frame: pd.DataFrame, column: str, quantile: float, ceiling: float
) -> float:
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    return min(ceiling, float(values.quantile(quantile)))


def derive_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "axis_line_score_min": upper_threshold(frame, "axis_line_score", 0.995, 0.008),
        "axis_line_snr_min": upper_threshold(frame, "axis_line_snr", 0.95, 3.0),
        "axis_line_support_min": 0.12,
        "diagonal_line_score_min": upper_threshold(frame, "diagonal_line_score", 0.995, 0.008),
        "diagonal_line_snr_min": upper_threshold(frame, "diagonal_line_snr", 0.95, 3.0),
        "diagonal_line_support_min": 0.12,
        "edge_touch_ratio_min": upper_threshold(frame, "edge_touch_ratio", 0.99, 0.25),
        "edge_mask_fraction_min": upper_threshold(frame, "edge_mask_fraction_total", 0.99, 0.004),
        "large_edge_mask_fraction_min": upper_threshold(frame, "edge_mask_fraction_total", 0.995, 0.012),
        "saturation_fraction_min": upper_threshold(frame, "saturation_any_fraction", 0.99, 0.001),
        "channel_streak_score_min": upper_threshold(frame, "channel_streak_score", 0.995, 0.010),
        "channel_streak_snr_min": upper_threshold(frame, "channel_streak_snr", 0.95, 3.0),
        "channel_streak_support_min": 0.12,
        "color_outlier_fraction_min": upper_threshold(frame, "color_outlier_fraction", 0.995, 0.002),
        "blank_std_luminance_max": lower_threshold(frame, "std_luminance", 0.005, 0.008),
        "blank_dynamic_range_max": lower_threshold(frame, "dynamic_range_p99_p01", 0.005, 0.04),
        "dark_p99_luminance_max": lower_threshold(frame, "p99_luminance", 0.005, 0.10),
        "blockiness_excess_ratio_min": upper_threshold(frame, "blockiness_excess_ratio", 0.995, 0.20),
        "blockiness_difference_min": 0.001,
        "channel_flux_ratio_min": upper_threshold(frame, "channel_flux_ratio_max_min", 0.995, 4.0),
        "channel_flux_signal_mean_min": 0.002,
    }


def apply_flags(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    result["flag_axis_line_artifact"] = (
        (result["axis_line_score"] >= thresholds["axis_line_score_min"])
        & (result["axis_line_snr"] >= thresholds["axis_line_snr_min"])
        & (result["axis_line_support"] >= thresholds["axis_line_support_min"])
    )
    result["flag_diagonal_line_artifact"] = (
        (result["diagonal_line_score"] >= thresholds["diagonal_line_score_min"])
        & (result["diagonal_line_snr"] >= thresholds["diagonal_line_snr_min"])
        & (result["diagonal_line_support"] >= thresholds["diagonal_line_support_min"])
    )
    result["flag_edge_touching_foreground"] = (
        (result["edge_touch_ratio"] >= thresholds["edge_touch_ratio_min"])
        & (result["edge_mask_fraction_total"] >= thresholds["edge_mask_fraction_min"])
        & (result["edge_sides_touched"] >= 1)
    )
    result["flag_saturation"] = (
        result["saturation_any_fraction"] >= thresholds["saturation_fraction_min"]
    )
    result["flag_channel_color_streak"] = (
        (
            (result["channel_streak_score"] >= thresholds["channel_streak_score_min"])
            & (result["channel_streak_snr"] >= thresholds["channel_streak_snr_min"])
            & (result["channel_streak_support"] >= thresholds["channel_streak_support_min"])
        )
        | (result["color_outlier_fraction"] >= thresholds["color_outlier_fraction_min"])
    )
    result["flag_blank_image"] = (
        (
            result["std_luminance"]
            <= thresholds["blank_std_luminance_max"]
        )
        & (
            result["dynamic_range_p99_p01"]
            <= thresholds["blank_dynamic_range_max"]
        )
    ) | (result["foreground_fraction"] <= 1.0 / (256.0 * 256.0))
    result["flag_dark_image"] = (
        result["p99_luminance"] <= thresholds["dark_p99_luminance_max"]
    )
    result["flag_compression_blockiness"] = (
        result["blockiness_excess_ratio"]
        >= thresholds["blockiness_excess_ratio_min"]
    ) & (
        result["blockiness_difference"] >= thresholds["blockiness_difference_min"]
    )
    result["flag_extreme_channel_ratio"] = (
        result["channel_flux_ratio_max_min"]
        >= thresholds["channel_flux_ratio_min"]
    ) & (
        result["foreground_signal_mean"]
        >= thresholds["channel_flux_signal_mean_min"]
    )
    result["flag_large_edge_mask"] = (
        result["edge_mask_fraction_total"]
        >= thresholds["large_edge_mask_fraction_min"]
    )
    result["flag_count"] = result[list(FLAG_COLUMNS)].sum(axis=1).astype(int)
    result["flags"] = result.apply(
        lambda row: ";".join(
            column.removeprefix("flag_")
            for column in FLAG_COLUMNS
            if bool(row[column])
        ),
        axis=1,
    )
    upper_pairs = (
        ("axis_line_score", "axis_line_score_min"),
        ("diagonal_line_score", "diagonal_line_score_min"),
        ("edge_touch_ratio", "edge_touch_ratio_min"),
        ("edge_mask_fraction_total", "large_edge_mask_fraction_min"),
        ("saturation_any_fraction", "saturation_fraction_min"),
        ("channel_streak_score", "channel_streak_score_min"),
        ("color_outlier_fraction", "color_outlier_fraction_min"),
        ("blockiness_excess_ratio", "blockiness_excess_ratio_min"),
        ("channel_flux_ratio_max_min", "channel_flux_ratio_min"),
    )
    severity = np.ones(len(result), dtype=np.float64)
    for metric, threshold in upper_pairs:
        severity = np.maximum(
            severity,
            result[metric].to_numpy(dtype=np.float64)
            / max(thresholds[threshold], 1e-8),
        )
    severity = np.maximum(
        severity,
        thresholds["dark_p99_luminance_max"]
        / np.maximum(result["p99_luminance"].to_numpy(dtype=np.float64), 1e-5),
    )
    result["candidate_severity"] = severity
    result["candidate_rank_score"] = severity + 0.25 * np.maximum(
        result["flag_count"].to_numpy() - 1, 0
    )
    result["manual_review_status"] = "pending_manual_review"
    result["manual_review_label"] = ""
    result["manual_review_notes"] = ""
    result["heuristic_version"] = HEURISTIC_VERSION
    return result


def score_distributions(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = (
        "axis_line_score",
        "axis_line_snr",
        "diagonal_line_score",
        "diagonal_line_snr",
        "edge_touch_ratio",
        "edge_mask_fraction_total",
        "saturation_any_fraction",
        "channel_streak_score",
        "channel_streak_snr",
        "color_outlier_fraction",
        "std_luminance",
        "p99_luminance",
        "blockiness_excess_ratio",
        "channel_flux_ratio_max_min",
    )
    result = {}
    for metric in metrics:
        values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
        result[metric] = {
            "min": float(values.min()),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "p995": float(values.quantile(0.995)),
            "max": float(values.max()),
        }
    return result


def abbreviated_flags(row: pd.Series) -> str:
    abbreviations = {
        "flag_axis_line_artifact": "axis",
        "flag_diagonal_line_artifact": "diag",
        "flag_edge_touching_foreground": "edge",
        "flag_saturation": "sat",
        "flag_channel_color_streak": "color",
        "flag_blank_image": "blank",
        "flag_dark_image": "dark",
        "flag_compression_blockiness": "block",
        "flag_extreme_channel_ratio": "ratio",
        "flag_large_edge_mask": "edge+",
    }
    labels = [abbreviations[column] for column in FLAG_COLUMNS if bool(row[column])]
    return ",".join(labels[:4]) + ("+" if len(labels) > 4 else "")


def save_contact_sheet(
    image_lookup: dict[int, np.ndarray],
    rows: pd.DataFrame,
    path: Path,
    title: str,
    top_count: int,
) -> int:
    selected = rows.sort_values(
        ["candidate_rank_score", "flag_count"], ascending=False
    ).head(top_count)
    if selected.empty:
        return 0
    n = len(selected)
    ncols = min(6, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 3.35 * nrows),
        squeeze=False,
    )
    for axis, (_, row) in zip(axes.ravel(), selected.iterrows(), strict=False):
        index = int(row["global_index"])
        image = image_lookup[index]
        axis.imshow(image)
        axis.set_title(
            f"#{index} {row['split']} L{int(row['label'])}\n"
            f"{abbreviated_flags(row)} s={row['candidate_rank_score']:.2f}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.ravel()[n:]:
        axis.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_figure(fig, path)
    return n


def make_contact_sheets(
    h5_path: Path,
    candidates: pd.DataFrame,
    run_dir: Path,
    top_count: int,
) -> dict[str, dict[str, Any]]:
    categories = {
        "overall": pd.Series(True, index=candidates.index),
        "line_and_diagonal": candidates[
            [
                "flag_axis_line_artifact",
                "flag_diagonal_line_artifact",
                "flag_channel_color_streak",
            ]
        ].any(axis=1),
        "edge_and_large_masks": candidates[
            ["flag_edge_touching_foreground", "flag_large_edge_mask"]
        ].any(axis=1),
        "saturation_and_color": candidates[
            ["flag_saturation", "flag_extreme_channel_ratio", "flag_channel_color_streak"]
        ].any(axis=1),
        "blank_dark_blockiness": candidates[
            ["flag_blank_image", "flag_dark_image", "flag_compression_blockiness"]
        ].any(axis=1),
    }
    selected_by_category = {
        name: candidates[mask]
        .sort_values(["candidate_rank_score", "flag_count"], ascending=False)
        .head(top_count)
        for name, mask in categories.items()
    }
    selected_indices = sorted(
        {
            int(index)
            for frame in selected_by_category.values()
            for index in frame["global_index"].tolist()
        }
    )
    image_lookup: dict[int, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        images = handle["images"]
        for index in selected_indices:
            image_lookup[index] = images[index]
    outputs: dict[str, dict[str, Any]] = {}
    for name, mask in categories.items():
        subset = candidates[mask]
        path = run_dir / "figures" / f"top_{name}.png"
        count = save_contact_sheet(
            image_lookup,
            selected_by_category[name],
            path,
            f"Source Artifact Candidates: {name.replace('_', ' ').title()}",
            top_count,
        )
        outputs[name] = {
            "path": project_relative(path) if count else None,
            "shown": count,
            "available": int(len(subset)),
        }
    return outputs


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def build_report(
    run_dir: Path,
    h5_path: Path,
    settings: AuditSettings,
    dataset_info: dict[str, Any],
    thresholds: dict[str, float],
    summary: dict[str, Any],
    contact_sheets: dict[str, dict[str, Any]],
) -> str:
    flag_rows = "\n".join(
        f"| `{column}` | {summary['flag_counts'][column]:,} |"
        for column in FLAG_COLUMNS
    )
    threshold_rows = "\n".join(
        f"| `{name}` | {value:.8g} |" for name, value in thresholds.items()
    )
    contact_rows = "\n".join(
        f"- {name.replace('_', ' ')}: "
        + (
            f"`{value['path']}` ({value['shown']} shown of {value['available']} candidates)"
            if value["path"]
            else "no qualifying candidates"
        )
        for name, value in contact_sheets.items()
    )
    return f"""# Source Artifact Audit Report

## Scope and Status

This append-only audit scanned `{project_relative(h5_path)}` with bounded,
sequential HDF5 blocks. It did not load the full image cube, modify source data,
change model checkpoints, train a model, or alter shared documentation.

The flags are **heuristic candidate flags**, not ground-truth artifact labels.
Every candidate remains `pending_manual_review`; no image is automatically
excluded or called clean.

- Heuristic version: `{HEURISTIC_VERSION}`.
- Dataset SHA-256: `{dataset_info['dataset_sha256']}`.
- Images audited: {summary['images_audited']:,} of {dataset_info['total_images']:,}.
- Full-dataset audit: `{summary['full_dataset_audit']}`.
- Candidate images: {summary['candidate_count']:,} ({summary['candidate_fraction']:.2%}).
- Peak process RSS: {summary['peak_rss_bytes'] / 1024**2:.1f} MiB.
- Runtime: {summary['runtime_seconds'] / 60.0:.2f} minutes.
- HDF5 image chunks: `{dataset_info['image_chunks']}`.
- Streaming block size: {settings.stream_block_size} images.

## Heuristic Definitions

- **Axis/diagonal lines:** second-neighbor line response outside the central
  quarter-radius region, summarized by long row, column, and two 45-degree
  diagonal profiles. Score, profile S/N proxy, and line support must pass.
- **Edge touching / large edge masks:** adaptive foreground above border
  background/noise touching the outer edge band. These can also flag genuine
  large galaxies or nearby companions.
- **Saturation:** fraction of pixels with at least one channel at or above
  250/255. Bright cores and stars can be legitimate triggers.
- **Channel/color streaks:** oriented second-neighbor response of per-pixel RGB
  channel spread, plus extreme chromatic-pixel fraction.
- **Blank/dark:** very low luminance standard deviation/dynamic range or low
  99th-percentile luminance. Low-surface-brightness galaxies can be false
  positives.
- **Compression/blockiness:** excess first-difference energy at 8-pixel grid
  boundaries relative to other pixel boundaries. This is a proxy, not a JPEG
  provenance test.
- **Extreme channel ratio:** max/min positive, border-subtracted channel flux
  ratio with a minimum signal guard. Survey-composite color can trigger it.

Upper-tail thresholds use fixed absolute floors plus empirical full-audit
quantiles. Low-tail blank/dark thresholds use conservative absolute ceilings
plus the lower empirical tail. This intentionally produces a review pool,
not a scientifically certified clean sample.

## Thresholds

| Threshold | Value |
| --- | ---: |
{threshold_rows}

## Candidate Counts

| Flag | Count |
| --- | ---: |
{flag_rows}

Candidates by split: `{json.dumps(summary['candidate_counts_by_split'])}`.

## Outputs

- Candidate manifest: `tables/source_artifact_candidates.csv`.
- Thresholds: `logs/source_artifact_thresholds.json`.
- Summary: `logs/source_artifact_audit_summary.json`.
- Score distributions: `logs/source_artifact_score_distributions.json`.

Contact sheets:

{contact_rows}

## Metadata and Manual Review

The candidate table includes original HDF5 global index, deterministic
train/validation/test membership matching seed `{dataset_info['split_seed']}`,
split-local and shuffled positions, Galaxy10 label, RA, Dec, redshift, pixel
scale, every component score, all flag booleans, applied threshold columns,
and empty manual-review fields.

Recommended next step: review contact sheets without model-performance
information, annotate artifact type and confidence, then freeze an explicitly
versioned source-quality manifest. Only after that should clean-source and
artifact-stress benchmarks be generated.

## Limitations

- No validated source-artifact labels exist in the local HDF5 file.
- Genuine edge-on galaxies, diffraction-like astrophysical structures, large
  nearby sources, saturated stars, and unusual survey-composite colors can
  trigger these proxies.
- The line detector tests horizontal, vertical, and two 45-degree directions;
  other angles may be missed.
- The foreground mask uses a border background/MAD threshold and is not a full
  segmentation model.
- The blockiness score cannot distinguish compression from real repeated
  high-frequency structure.
- Candidate frequency must not be reported as the true artifact prevalence.
"""


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (PROJECT_ROOT / config.get("output_dir", "outputs")).resolve()
    try:
        output_root.relative_to((PROJECT_ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("Configured output_dir must remain under ignored outputs/.") from exc
    settings = AuditSettings(
        border_width=args.border_width,
        edge_width=args.edge_width,
        stream_block_size=args.stream_block_size,
        progress_every=args.progress_every,
        top_per_sheet=args.top_per_sheet,
        max_images=args.max_images,
    )
    if min(
        settings.border_width,
        settings.edge_width,
        settings.stream_block_size,
        settings.progress_every,
        settings.top_per_sheet,
    ) <= 0:
        raise ValueError("Audit widths, block size, progress interval, and sheet size must be positive.")
    if settings.max_images is not None and settings.max_images <= 0:
        raise ValueError("max-images must be positive when supplied.")
    run_dir = make_run_dir(output_root, stamp)
    checkpoint_dir = output_root / "checkpoints"
    before = checkpoint_snapshot(checkpoint_dir)
    safe_json(run_dir / "logs/checkpoint_integrity_before.json", before)
    start_time = time.monotonic()
    h5_path = resolve_path(Path(config["dataset_path"]))
    dataset_sha256 = file_sha256(h5_path)
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        required = ("images", "ans", "ra", "dec", "redshift", "pxscale")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError("Missing HDF5 datasets: " + ", ".join(missing))
        images = handle["images"]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images shaped (N,H,W,3), got {images.shape}.")
        total_images, height, width, _channels = images.shape
        audit_count = min(total_images, settings.max_images or total_images)
        split_code, split_position, shuffled_position = make_split_lookup(
            total_images,
            float(config["splits"]["train_frac"]),
            float(config["splits"]["val_frac"]),
            int(config["seed"]),
        )
        split_names = np.asarray(("train", "validation", "test"), dtype=object)
        geometry = make_geometry(height, width, settings.edge_width)
        dataset_info = {
            "dataset_path": project_relative(h5_path),
            "dataset_size_bytes": int(h5_path.stat().st_size),
            "dataset_sha256": dataset_sha256,
            "total_images": int(total_images),
            "audited_images": int(audit_count),
            "image_shape": [int(height), int(width), 3],
            "image_dtype": str(images.dtype),
            "image_chunks": list(images.chunks) if images.chunks else None,
            "image_compression": images.compression,
            "metadata_keys": sorted(handle.keys()),
            "split_seed": int(config["seed"]),
            "streaming": True,
            "full_image_cube_loaded": False,
            "code_sha256": {
                "config": file_sha256(config_path),
                "scripts/source_artifact_audit.py": file_sha256(Path(__file__)),
            },
            "dependency_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "h5py": h5py.__version__,
                "matplotlib": matplotlib.__version__,
                "pyyaml": importlib_metadata.version("PyYAML"),
            },
        }
        safe_json(run_dir / "logs/dataset_streaming_config.json", dataset_info)
        for block_start in range(0, audit_count, settings.stream_block_size):
            block_stop = min(audit_count, block_start + settings.stream_block_size)
            image_block = images[block_start:block_stop]
            label_block = handle["ans"][block_start:block_stop]
            ra_block = handle["ra"][block_start:block_stop]
            dec_block = handle["dec"][block_start:block_stop]
            redshift_block = handle["redshift"][block_start:block_stop]
            pxscale_block = handle["pxscale"][block_start:block_stop]
            for local_index, image in enumerate(image_block):
                global_index = block_start + local_index
                scores = compute_image_scores(image, geometry, settings)
                rows.append(
                    {
                        "global_index": global_index,
                        "split": str(split_names[split_code[global_index]]),
                        "split_index": int(split_position[global_index]),
                        "shuffled_position": int(shuffled_position[global_index]),
                        "label": int(label_block[local_index]),
                        "ra": float(ra_block[local_index]),
                        "dec": float(dec_block[local_index]),
                        "redshift": float(redshift_block[local_index]),
                        "pxscale": float(pxscale_block[local_index]),
                        **scores,
                    }
                )
                completed = global_index + 1
                if completed % settings.progress_every == 0 or completed == audit_count:
                    elapsed = time.monotonic() - start_time
                    rate = completed / max(elapsed, 1e-6)
                    print(
                        f"Audited {completed:,}/{audit_count:,} images "
                        f"({rate:.1f} images/s, RSS {peak_rss_bytes()/1024**2:.1f} MiB).",
                        flush=True,
                    )
            del image_block, label_block, ra_block, dec_block, redshift_block, pxscale_block
            gc.collect()

    score_frame = pd.DataFrame(rows)
    thresholds = derive_thresholds(score_frame)
    flagged = apply_flags(score_frame, thresholds)
    candidates = flagged[flagged["flag_count"] > 0].copy()
    candidates.sort_values(
        ["candidate_rank_score", "flag_count", "global_index"],
        ascending=[False, False, True],
        inplace=True,
    )
    for name, value in thresholds.items():
        candidates[f"threshold_{name}"] = value
    safe_csv(run_dir / "tables/source_artifact_candidates.csv", candidates)
    safe_json(run_dir / "logs/source_artifact_thresholds.json", thresholds)
    distributions = score_distributions(score_frame)
    safe_json(run_dir / "logs/source_artifact_score_distributions.json", distributions)
    contact_sheets = make_contact_sheets(
        h5_path, candidates, run_dir, settings.top_per_sheet
    )
    runtime = time.monotonic() - start_time
    flag_counts = {column: int(flagged[column].sum()) for column in FLAG_COLUMNS}
    summary = {
        "heuristic_version": HEURISTIC_VERSION,
        "run_dir": project_relative(run_dir),
        "images_audited": int(len(flagged)),
        "total_images": int(dataset_info["total_images"]),
        "full_dataset_audit": bool(len(flagged) == dataset_info["total_images"]),
        "candidate_count": int(len(candidates)),
        "candidate_fraction": float(len(candidates) / max(len(flagged), 1)),
        "flag_counts": flag_counts,
        "candidate_counts_by_split": {
            str(key): int(value)
            for key, value in candidates["split"].value_counts().sort_index().items()
        },
        "candidate_counts_by_label": {
            str(int(key)): int(value)
            for key, value in candidates["label"].value_counts().sort_index().items()
        },
        "thresholds": thresholds,
        "contact_sheets": contact_sheets,
        "runtime_seconds": runtime,
        "peak_rss_bytes": peak_rss_bytes(),
        "streaming_block_size": settings.stream_block_size,
        "manual_review_status": "pending_manual_review",
        "automatic_exclusion_recommended": False,
        "limitations": [
            "heuristic candidates are not validated artifact labels",
            "genuine morphology and survey-composite color can trigger flags",
            "line orientations are limited to horizontal, vertical, and two diagonals",
            "candidate fraction is not an artifact prevalence estimate",
        ],
    }
    safe_json(run_dir / "logs/source_artifact_audit_summary.json", summary)
    safe_text(
        run_dir / "diagnostics/source_artifact_audit_report.md",
        build_report(
            run_dir,
            h5_path,
            settings,
            dataset_info,
            thresholds,
            summary,
            contact_sheets,
        ),
    )
    after = checkpoint_snapshot(checkpoint_dir)
    comparison = compare_checkpoint_snapshots(before, after)
    safe_json(run_dir / "logs/checkpoint_integrity_after.json", after)
    safe_json(run_dir / "logs/checkpoint_integrity_comparison.json", comparison)
    if not comparison["old_checkpoints_unchanged"]:
        raise RuntimeError("Checkpoint integrity changed during source artifact audit.")
    print(
        f"Completed source artifact audit: {len(candidates):,} candidates from "
        f"{len(flagged):,} images in {runtime/60.0:.2f} minutes.",
        flush=True,
    )
    print(f"Run directory: {project_relative(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
