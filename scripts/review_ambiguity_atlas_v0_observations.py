#!/usr/bin/env python3
"""Prepare a noise-normalized observed-blend gallery for the Atlas visual gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene  # noqa: E402

CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
PAIR_COUNT = 25
PER_PAGE = 5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def observed_rgb(image: np.ndarray, sky: np.ndarray) -> np.ndarray:
    standardized = image / np.sqrt(sky[:, None, None])
    channels = np.stack([standardized[2], standardized[1], standardized[0]], axis=-1)
    return np.clip(0.5 + 0.22 * np.tanh(channels / 2.5), 0.0, 1.0)


def target_rgb(image: np.ndarray, scale: float) -> np.ndarray:
    channels = np.stack([image[2], image[1], image[0]], axis=-1)
    transformed = np.arcsinh(np.maximum(channels, 0.0) / max(scale, 1e-12))
    maximum = float(np.max(transformed))
    return transformed / maximum if maximum > 0 else transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    pairs = read_csv(run_dir / "tables/atlas_pair_manifest.csv")[:PAIR_COUNT]
    definitions = read_csv(run_dir / "manifests/atlas_pool_scene_definitions.csv")
    catalog, _ = load_catsim_catalog(CATALOG)
    output_dir = run_dir / "figures/ambiguity_atlas_observed"
    output_dir.mkdir()
    inventory: list[dict[str, object]] = []
    for page_start in range(0, PAIR_COUNT, PER_PAGE):
        page_pairs = pairs[page_start : page_start + PER_PAGE]
        figure, axes = plt.subplots(len(page_pairs), 5, figsize=(12, 2.5 * len(page_pairs)), squeeze=False)
        for row_index, pair in enumerate(page_pairs):
            with np.load(run_dir / f"atlas/{pair['pair_id']}.npz", allow_pickle=False) as arrays:
                stored = {
                    "left": np.asarray(arrays["left_blend"], dtype=np.float64),
                    "right": np.asarray(arrays["right_blend"], dtype=np.float64),
                }
                isolated = {
                    "left": np.asarray(arrays["left_isolated"][0], dtype=np.float64),
                    "right": np.asarray(arrays["right_isolated"][0], dtype=np.float64),
                }
                sky = np.asarray(arrays["sky_electrons"], dtype=np.float64)
            noisy: dict[str, np.ndarray] = {}
            hashes: dict[str, str] = {}
            for side in ("left", "right"):
                definition = definitions[int(pair[f"{side}_pool_index"])]
                spec = SceneSpec(
                    scene_id=definition["scene_id"],
                    catalog_rows=(int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])),
                    positions_arcsec=(
                        (float(definition["target_x_arcsec"]), float(definition["target_y_arcsec"])),
                        (float(definition["contaminant_x_arcsec"]), float(definition["contaminant_y_arcsec"])),
                    ),
                    source_selection_seed=int(definition["source_selection_seed"]),
                    position_seed=int(definition["position_seed"]),
                    noise_seed=int(definition["noise_seed"]),
                )
                noiseless_replay = render_fixed_scene(catalog, spec, add_noise="none")
                if sha256_array(noiseless_replay.blend.astype(np.float32)) != sha256_array(stored[side].astype(np.float32)):
                    raise RuntimeError("noiseless pair replay failed")
                first = render_fixed_scene(catalog, spec, add_noise="all").blend
                second = render_fixed_scene(catalog, spec, add_noise="all").blend
                if sha256_array(first) != sha256_array(second):
                    raise RuntimeError("noisy pair replay failed")
                noisy[side] = first
                hashes[side] = sha256_array(first)
            difference = np.mean(
                np.abs(noisy["left"] - noisy["right"]) / np.sqrt(sky[:, None, None]), axis=0
            )
            target_scale = float(np.quantile(np.concatenate([isolated["left"].ravel(), isolated["right"].ravel()]), 0.995)) / 5.0
            panels = [
                observed_rgb(noisy["left"], sky),
                observed_rgb(noisy["right"], sky),
                difference,
                target_rgb(isolated["left"], target_scale),
                target_rgb(isolated["right"], target_scale),
            ]
            titles = ("left observed", "right observed", "|observed delta|/noise", "left truth", "right truth")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axis = axes[row_index, column]
                axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None, origin="lower", vmin=0 if panel.ndim == 2 else None, vmax=5 if panel.ndim == 2 else None)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(title, fontsize=9)
            axes[row_index, 0].set_ylabel(pair["pair_id"], fontsize=8)
            inventory.append(
                {
                    "pair_id": pair["pair_id"],
                    "left_noisy_blend_sha256": hashes["left"],
                    "right_noisy_blend_sha256": hashes["right"],
                    "noisy_exact_replay_pass": True,
                    "gallery_page": page_start // PER_PAGE + 1,
                    "gallery_row": row_index + 1,
                    "visual_review_status": "PENDING",
                }
            )
        figure.tight_layout()
        figure.savefig(output_dir / f"observed_atlas_visual_audit_page_{page_start // PER_PAGE + 1:02d}.png", dpi=180)
        plt.close(figure)
    write_csv_fresh(run_dir / "tables/atlas_observed_visual_audit_inventory.csv", inventory)


if __name__ == "__main__":
    main()
