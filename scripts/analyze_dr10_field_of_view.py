#!/usr/bin/env python3
"""Analyze 128/192/256-pixel DR10 engineering cutouts and select raw FOV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.io import fits

try:
    from scripts.download_dr10_grz_cutouts import output_filename, validate_fits
except ModuleNotFoundError:  # direct `python scripts/...py` execution
    from download_dr10_grz_cutouts import output_filename, validate_fits

SIZES = (128, 192, 256)


def _exclusive_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _manifest_records(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    records: dict[tuple[str, str], dict[str, str]] = {}
    if not path.exists():
        return records
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("catalog_row_index", ""), row.get("source_id", ""))
            # Preserve the original measured network timing if a later resume
            # appends a zero-time validated-skip row.
            if key not in records or row.get("status") == "downloaded_valid":
                records[key] = row
    return records


def _detection_metrics(data: np.ndarray) -> dict[str, float | int | bool]:
    residuals: list[np.ndarray] = []
    for band in data:
        image = np.ascontiguousarray(band.astype(np.float32))
        background = sep.Background(image, bw=32, bh=32, fw=3, fh=3)
        residuals.append((image - background.back()) / max(float(background.globalrms), 1e-6))
    detection = np.sum(residuals, axis=0) / np.sqrt(len(residuals))
    objects, segmentation = sep.extract(
        np.ascontiguousarray(detection.astype(np.float32)),
        thresh=2.5,
        minarea=8,
        segmentation_map=True,
    )
    height, width = detection.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    result: dict[str, float | int | bool] = {
        "detected_source_count": int(len(objects)),
        "central_source_detected": False,
        "central_centroid_offset_pixels": float("nan"),
        "central_area_pixels": 0,
        "central_bbox_border_margin_pixels": float("nan"),
        "central_source_contained": False,
        "border_truncation": False,
        "neighbor_count": 0,
        "nearest_neighbor_distance_pixels": float("nan"),
        "background_like_fraction_abs_snr_lt_1": float(np.mean(np.abs(detection) < 1.0)),
        "blank_coverage_fraction": float(
            np.mean(np.all((~np.isfinite(data)) | (data == 0), axis=0))
        ),
    }
    if len(objects) == 0:
        return result

    distances = np.hypot(objects["x"] - center_x, objects["y"] - center_y)
    # A source whose centroid is farther than one quarter-frame from the
    # requested coordinate is not accepted as the requested central source.
    central_index = int(np.argmin(distances))
    if float(distances[central_index]) > min(height, width) / 4.0:
        return result
    label = central_index + 1
    mask_y, mask_x = np.nonzero(segmentation == label)
    if mask_x.size == 0:
        return result
    margin = float(
        min(mask_x.min(), mask_y.min(), width - 1 - mask_x.max(), height - 1 - mask_y.max())
    )
    touching = margin <= 1.0
    central_x = float(objects[central_index]["x"])
    central_y = float(objects[central_index]["y"])
    neighbor_distances = np.hypot(objects["x"] - central_x, objects["y"] - central_y)
    other = np.delete(neighbor_distances, central_index)
    result.update(
        central_source_detected=True,
        central_centroid_offset_pixels=float(distances[central_index]),
        central_area_pixels=int(mask_x.size),
        central_bbox_border_margin_pixels=margin,
        central_source_contained=not touching,
        border_truncation=touching,
        neighbor_count=int(len(other)),
        nearest_neighbor_distance_pixels=float(other.min()) if len(other) else float("nan"),
    )
    return result


def _visual_rgb(data: np.ndarray) -> np.ndarray:
    """Make a visualization-only signed-asinh RGB; never return scientific data."""
    channels = []
    # Display order R,G,B = z,r,g. Each uses the same per-image scalar solely
    # for contact-sheet visibility; these pixels never enter a model or table.
    scale = float(np.nanpercentile(np.abs(data), 99.5))
    scale = max(scale, 1e-6)
    for index in (2, 1, 0):
        mapped = np.arcsinh(data[index] / (0.05 * scale)) / np.arcsinh(1.0 / 0.05)
        channels.append(mapped)
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-manifest", type=Path, required=True)
    parser.add_argument(
        "--download-root",
        type=Path,
        required=True,
        help="Directory containing size_128, size_192, and size_256 outputs",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = pd.read_csv(args.engineering_manifest, dtype={"source_id": str}).head(args.limit)
    if len(sources) == 0:
        raise SystemExit("engineering manifest has no rows")
    if "selection_phase" not in sources.columns or not sources["selection_phase"].eq(
        "engineering"
    ).all():
        raise SystemExit("FOV contact sheets require selection_phase=engineering for every row")
    for role_column in ("split_role", "partition", "role"):
        if role_column in sources.columns and sources[role_column].astype(str).str.contains(
            "lockbox", case=False, na=False
        ).any():
            raise SystemExit("lockbox-labeled rows are prohibited from the FOV study")

    metrics: list[dict[str, object]] = []
    arrays: dict[tuple[str, int], np.ndarray] = {}
    for size in SIZES:
        directory = args.download_root / f"size_{size}"
        manifest = _manifest_records(directory / "download_manifest.csv")
        for _, source in sources.iterrows():
            source_row = {key: "" if pd.isna(value) else str(value) for key, value in source.items()}
            source_id = source_row.get("source_id") or source_row.get("dr8_id") or ""
            row_index = source_row.get("catalog_row_index", "")
            path = directory / output_filename(source_row)
            validation = validate_fits(path, size, "grz") if path.exists() else None
            record = manifest.get((row_index, source_id), {})
            base: dict[str, object] = {
                "source_id": source_id,
                "catalog_row_index": row_index,
                "ra": float(source_row["ra"]),
                "dec": float(source_row["dec"]),
                "size_pixels": size,
                "angular_width_arcsec": size * 0.262,
                "valid_fits": bool(validation and validation.valid),
                "validation_error": "" if validation and validation.valid else (
                    validation.error if validation else "missing file"
                ),
                "file_size_bytes": path.stat().st_size if path.exists() else 0,
                "download_elapsed_seconds": float(record.get("elapsed_seconds") or "nan"),
                "sha256": record.get("sha256", ""),
                "relative_path": str(path),
            }
            if validation and validation.valid:
                with fits.open(path, memmap=True) as hdul:
                    data = np.asarray(hdul[0].data, dtype=np.float32).copy()
                arrays[(source_id, size)] = data
                base.update(_detection_metrics(data))
            metrics.append(base)

    frame = pd.DataFrame(metrics)
    table_path = args.run_dir / "tables" / "field_of_view_metrics.csv"
    _exclusive_dataframe(table_path, frame)

    figure_dir = args.run_dir / "figures" / "field_of_view_comparison"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(sources), 5):
        subset = sources.iloc[start : start + 5]
        fig, axes = plt.subplots(len(subset), len(SIZES), figsize=(10, 3.2 * len(subset)), squeeze=False)
        for row_number, (_, source) in enumerate(subset.iterrows()):
            source_id = str(source["source_id"])
            for column, size in enumerate(SIZES):
                axis = axes[row_number, column]
                data = arrays.get((source_id, size))
                if data is None:
                    axis.text(0.5, 0.5, "missing/invalid", ha="center", va="center")
                else:
                    axis.imshow(_visual_rgb(data), origin="lower")
                axis.set_title(f"{source_id} — {size} px")
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle("Visualization-only z/r/g signed-asinh renderings")
        fig.tight_layout()
        destination = figure_dir / f"fov_contact_sheet_{start // 5 + 1:02d}.png"
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150)
        plt.close(fig)

    summaries = []
    for size, group in frame.groupby("size_pixels", sort=True):
        valid = group[group["valid_fits"] == True]  # noqa: E712
        summaries.append(
            {
                "size": int(size),
                "attempted": int(len(group)),
                "valid": int(len(valid)),
                "central_detected": int(valid.get("central_source_detected", pd.Series(dtype=bool)).sum()),
                "contained": int(valid.get("central_source_contained", pd.Series(dtype=bool)).sum()),
                "border_truncated": int(valid.get("border_truncation", pd.Series(dtype=bool)).sum()),
                "median_neighbors": float(valid.get("neighbor_count", pd.Series(dtype=float)).median()),
                "median_blank_coverage": float(valid.get("blank_coverage_fraction", pd.Series(dtype=float)).median()),
                "max_blank_coverage": float(valid.get("blank_coverage_fraction", pd.Series(dtype=float)).max()),
                "median_file_bytes": float(valid["file_size_bytes"].median()),
                "median_download_seconds": float(valid["download_elapsed_seconds"].median()),
            }
        )
    by_size = {item["size"]: item for item in summaries}
    s128 = by_size.get(128, {})
    s192 = by_size.get(192, {})
    s256 = by_size.get(256, {})
    storage_estimate = float(s256.get("median_file_bytes", float("inf"))) * 2500
    frozen_gates = {
        "all_20_strictly_valid": s256.get("valid") == len(sources),
        "central_detection_rate_ge_0.90": s256.get("central_detected", 0) >= 0.90 * len(sources),
        "containment_rate_ge_0.90": s256.get("contained", 0) >= 0.90 * len(sources),
        "border_truncation_rate_le_0.10": s256.get("border_truncated", len(sources)) <= 0.10 * len(sources),
        "max_blank_coverage_le_0.10": float(s256.get("max_blank_coverage", 1.0)) <= 0.10,
        "median_download_seconds_le_30": float(s256.get("median_download_seconds", float("inf"))) <= 30.0,
        "projected_2500_storage_lt_5GB": storage_estimate < 5_000_000_000,
        "containment_not_worse_than_192": s256.get("contained", 0) >= s192.get("contained", 0),
        "context_not_less_than_128": float(s256.get("median_neighbors", -1)) >= float(s128.get("median_neighbors", 0)),
    }
    selected = 256 if all(frozen_gates.values()) else None
    summary_json = json.dumps(summaries, indent=2, sort_keys=True)
    selection_text = (
        "256x256" if selected == 256 else "no size selected; bulk gate remains closed"
    )
    report = f"""# DR10 field-of-view study

Twenty deterministic engineering sources were requested at 128, 192, and 256
pixels using `layer=ls-dr10-south`, `bands=grz`, and 0.262 arcsec/pixel. Raw
widths are 33.536, 50.304, and 67.072 arcsec respectively.

## Machine summary

```json
{summary_json}
```

Frozen gate outcomes: `{json.dumps(frozen_gates, sort_keys=True)}`.

The source detector uses per-band SEP background/RMS estimates and a combined
2.5-sigma detection image. A central detection must lie within one quarter of
the frame. Containment means its segmentation footprint stays more than one
pixel from every border. `blank_coverage_fraction` counts pixels that are
non-finite or exactly zero in all three bands; a separate background-like
fraction counts combined |S/N| below one. Neighbor counts are transparent SEP
detections, not catalog truth.

## Decision

Selected raw cutout size: **{selection_text}**.

The 256-pixel choice is accepted only if every strict FITS validation passes,
at least 90% have a detected and contained central segmentation, at most 10%
touch a border, maximum blank coverage is at most 10%, median download time is
at most 30 seconds, projected 2,500-source image storage is below 5 GB,
containment is no worse than at 192 pixels, and detected context is no less
than at 128 pixels. These thresholds were fixed before inspecting this metrics
table. A passing 256-pixel frame retains 67.072 arcsec of context; smaller
training crops/downsampling remain possible, whereas discarded context cannot
be recovered. This decision does not declare every central object clean:
source-isolation rules are a separate downstream gate.

Contact sheets are visualization-only z/r/g signed-asinh renderings. Their
display scaling is never written back to or substituted for scientific FITS
arrays.
"""
    _exclusive_text(args.run_dir / "diagnostics" / "field_of_view_study.md", report)
    print(json.dumps({"selected_size": selected, "summaries": summaries}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
