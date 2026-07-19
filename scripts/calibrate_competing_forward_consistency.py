#!/usr/bin/env python3
"""Calibrate the frozen forward-consistency tolerance on 2,000 scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import SamplingFunction
from surveycodex.utilities import mean_sky_level


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.btk_scene import BAND_ORDER, STAMP_SIZE_ARCSEC, load_catsim_catalog, validated_lsst_survey  # noqa: E402
from src.competing_hypotheses import calibrate_plausibility, forward_consistency, serialize_dataclass  # noqa: E402


CALIBRATION_COUNT = 3_000
CALIBRATION_SEED = 2026071231
BATCH_SIZE = 250


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def make_definitions(run_dir: Path) -> list[dict[str, object]]:
    output = run_dir / "manifests/forward_consistency_calibration_scene_definitions.csv"
    if output.exists():
        rows = read_csv(output)
        if len(rows) != CALIBRATION_COUNT:
            raise RuntimeError("calibration definition count mismatch")
        return rows
    pool = [
        row
        for row in read_csv(run_dir / "manifests/campaign_source_partition_commitments.csv")
        if row["campaign_role"] == "calibration"
    ]
    rng = np.random.default_rng(CALIBRATION_SEED)
    rows: list[dict[str, object]] = []
    for index in range(CALIBRATION_COUNT):
        while True:
            selected = rng.choice(len(pool), size=2, replace=False)
            target, contaminant = pool[int(selected[0])], pool[int(selected[1])]
            if target["duplicate_group_id"] != contaminant["duplicate_group_id"]:
                break
        separation = float(rng.uniform(0.6, 3.0))
        angle = float(rng.uniform(0.0, 2 * np.pi))
        rows.append(
            {
                "scene_id": f"atlas_v0_forward_calibration_{index:05d}",
                "target_catalog_row": target["catalog_row"],
                "target_group": target["duplicate_group_id"],
                "contaminant_catalog_row": contaminant["catalog_row"],
                "contaminant_group": contaminant["duplicate_group_id"],
                "target_x_arcsec": 0.0,
                "target_y_arcsec": 0.0,
                "contaminant_x_arcsec": separation * np.cos(angle),
                "contaminant_y_arcsec": separation * np.sin(angle),
                "scene_seed": CALIBRATION_SEED + index,
                "noise_seed": 202607124_000 + index,
                "campaign_role": "calibration",
            }
        )
    write_csv_fresh(output, rows)
    return rows


class FixedBatchSampling(SamplingFunction):
    def __init__(self, definitions: list[dict[str, object]]) -> None:
        super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
        self.stamp_size = STAMP_SIZE_ARCSEC
        self.definitions = definitions
        self.position = 0

    def __call__(self, table: Table) -> Table:
        row = self.definitions[self.position]
        self.position += 1
        selected = table[[int(row["target_catalog_row"]), int(row["contaminant_catalog_row"])]].copy()
        selected["ra"] = [float(row["target_x_arcsec"]), float(row["contaminant_x_arcsec"])]
        selected["dec"] = [float(row["target_y_arcsec"]), float(row["contaminant_y_arcsec"])]
        return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "manifests/atlas_initial_freeze_record.json").exists():
        raise RuntimeError("initial Atlas must be frozen before calibration")
    output = run_dir / "calibration/forward_consistency_thresholds.json"
    if output.exists():
        raise FileExistsError(output)
    definitions = make_definitions(run_dir)
    catalog_path = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
    catalog, _ = load_catsim_catalog(catalog_path)
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    band_indices = [tuple(survey.available_filters).index(band) for band in BAND_ORDER]
    results = []
    rows: list[dict[str, object]] = []
    started = time.time()
    for start in range(0, CALIBRATION_COUNT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, CALIBRATION_COUNT)
        chunk = definitions[start:stop]
        generator = CatsimGenerator(
            catalog,
            FixedBatchSampling(chunk),
            survey,
            batch_size=len(chunk),
            njobs=1,
            verbose=False,
            use_bar=False,
            add_noise="all",
            seed=202607125_000 + start,
            apply_shear=False,
            augment_data=False,
        )
        batch = next(generator)
        blends = np.asarray(batch.blend_images[:, band_indices], dtype=np.float64)
        isolated = np.asarray(batch.isolated_images[:, :2][:, :, band_indices], dtype=np.float64)
        for local, definition in enumerate(chunk):
            expected = [int(definition["target_catalog_row"]), int(definition["contaminant_catalog_row"])]
            actual = list(np.asarray(batch.catalog_list[local]["catalog_row"], dtype=int))
            if actual != expected:
                raise RuntimeError("calibration source alignment failure")
            result = forward_consistency(blends[local], isolated[local], sky)
            results.append(result)
            rows.append(
                {
                    "scene_id": definition["scene_id"],
                    "campaign_role": "calibration",
                    "global_chi_square_mean": result.global_chi_square_mean,
                    "g_chi_square_mean": result.per_band_chi_square_mean[0],
                    "r_chi_square_mean": result.per_band_chi_square_mean[1],
                    "z_chi_square_mean": result.per_band_chi_square_mean[2],
                    "g_neighbor_correlation": result.residual_neighbor_correlation[0],
                    "r_neighbor_correlation": result.residual_neighbor_correlation[1],
                    "z_neighbor_correlation": result.residual_neighbor_correlation[2],
                    "relative_flux_residual": result.relative_flux_residual,
                    "blend_sha256": sha256_array(blends[local]),
                    "truth_decomposition_sha256": sha256_array(isolated[local]),
                    "finite": result.finite,
                }
            )
        print(json.dumps({"completed": stop, "total": CALIBRATION_COUNT}), flush=True)
    thresholds = calibrate_plausibility(results)
    threshold_payload = serialize_dataclass(thresholds)
    threshold_payload.update(
        {
            "status": "FROZEN_CALIBRATION_ONLY",
            "score_inputs": "observed blend, candidate decomposition, exact BTK sky/noise contract only",
            "truth_role": "known-truth full decomposition supplies the reference calibration distribution only; truth is not a score input",
            "sky_electrons_grz": sky.tolist(),
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        }
    )
    write_json_fresh(output, threshold_payload)
    write_csv_fresh(run_dir / "tables/forward_consistency_calibration.csv", rows)
    report = f"""# Forward-consistency calibration report

Status: **PASS / FROZEN_CALIBRATION_ONLY**.

The preregistered consistency score was evaluated on exactly 3,000 approved
calibration scenes using their known-truth decompositions as the reference
calibration distribution. The score itself receives only the observed blend,
candidate decomposition, and exact BTK source-plus-sky Poisson variance
contract. No target layer, candidate true error, development scene, or lockbox
scene enters the score.

- Global 99th-percentile limit: {thresholds.global_chi_square_mean:.8g}
- Per-band 99.5th-percentile limits: {thresholds.per_band_chi_square_mean}
- Absolute flux-residual 99th-percentile limit: {thresholds.absolute_relative_flux_residual:.8g}
- Runtime seconds: {time.time() - started:.3f}
"""
    write_text_fresh(run_dir / "diagnostics/forward_consistency_report.md", report)
    write_json_fresh(
        run_dir / "logs/forward_consistency_calibration_complete.json",
        {
            "status": "PASS",
            "calibration_scene_count": CALIBRATION_COUNT,
            "runtime_seconds": time.time() - started,
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        },
    )


if __name__ == "__main__":
    main()
