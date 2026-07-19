#!/usr/bin/env python3
"""Build, search, render, and replay Atlas-excluded Thayer-PU data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import SamplingFunction
from scipy.spatial import cKDTree
from surveycodex.utilities import mean_sky_level


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.btk_scene import (  # noqa: E402
    BAND_ORDER,
    STAMP_SIZE_ARCSEC,
    SceneSpec,
    load_catsim_catalog,
    render_fixed_scene,
    validated_lsst_survey,
)
from src.canonical_tensor_hash import SCHEMA_VERSION, canonical_tensor_sha256  # noqa: E402
from src.competing_hypotheses import scientific_distance  # noqa: E402


CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
PARTITIONS = ("training", "validation", "calibration")
POOL_COUNTS = {"training": 16_000, "validation": 4_000, "calibration": 4_000}
ORDINARY_COUNTS = {"training": 12_000, "validation": 1_500, "calibration": 1_500}
PAIR_COUNTS = {"training": 2_000, "validation": 250, "calibration": 250}
POOL_SEEDS = {"training": 2026077701, "validation": 2026077702, "calibration": 2026077703}
ORDINARY_SEEDS = {"training": 2026077801, "validation": 2026077802, "calibration": 2026077803}
NOISE_BASES = {"training": 2_026_079_000, "validation": 2_026_099_000, "calibration": 2_026_109_000}
NEIGHBORS = 32
BLOCK_SIZE = 4
BATCH_SIZE = 250
IMAGE_SIZE = 60
BLEND_DISTANCE_LIMIT = 1.0
TARGET_DISTANCE_LIMIT = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        if fields is None:
            fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_run(path: Path, phase: str) -> Path:
    run_dir = path.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_probabilistic_unet_"):
        raise ValueError("unexpected run directory")
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    prereg = run_dir / "preregistration/prompted_probabilistic_unet.md"
    if sha256_file(prereg) != freeze["preregistration_sha256"]:
        raise RuntimeError("preregistration altered")
    if json.loads((run_dir / "logs/architecture_audit_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("architecture audit did not pass")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before data preparation")
    prerequisites = {
        "search": run_dir / "logs/near_collision_pool_complete.json",
        "render": run_dir / "logs/near_collision_search_complete.json",
        "replay": run_dir / "logs/data_render_complete.json",
    }
    if phase in prerequisites and not prerequisites[phase].exists():
        raise RuntimeError(f"missing prerequisite: {prerequisites[phase]}")
    return run_dir


def downsample(array: np.ndarray) -> np.ndarray:
    shape = array.shape
    if shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("unexpected image shape")
    return array.reshape(*shape[:-2], 15, BLOCK_SIZE, 15, BLOCK_SIZE).mean(axis=(-3, -1))


def choose_two(pool: list[dict[str, str]], rng: np.random.Generator) -> list[dict[str, str]]:
    while True:
        chosen = rng.choice(len(pool), size=2, replace=False)
        selected = [pool[int(chosen[0])], pool[int(chosen[1])]]
        if selected[0]["duplicate_group_id"] != selected[1]["duplicate_group_id"]:
            return selected


def scene_row(
    partition: str,
    kind: str,
    index: int,
    selected: list[dict[str, str]],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, object]:
    separation_limits = (0.6, 3.0) if kind == "near_pool" else (0.8, 3.2)
    separation = float(rng.uniform(*separation_limits))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    midpoint = rng.uniform(-0.25, 0.25, size=2)
    offset = 0.5 * separation * np.asarray([np.cos(angle), np.sin(angle)])
    positions = np.stack((midpoint - offset, midpoint + offset))
    return {
        "scene_id": f"pu_{partition}_{kind}_{index:05d}",
        "partition": partition,
        "kind": kind,
        "partition_index": index,
        "scene_seed": seed,
        "source_selection_seed": seed,
        "position_seed": seed,
        "noise_seed": NOISE_BASES[partition] + (0 if kind == "ordinary" else 1_000_000) + index,
        "source_a_row": int(selected[0]["catalog_row"]),
        "source_b_row": int(selected[1]["catalog_row"]),
        "source_a_id": selected[0]["persistent_source_id"],
        "source_b_id": selected[1]["persistent_source_id"],
        "source_a_group": selected[0]["duplicate_group_id"],
        "source_b_group": selected[1]["duplicate_group_id"],
        "source_a_x_arcsec": float(positions[0, 0]),
        "source_a_y_arcsec": float(positions[0, 1]),
        "source_b_x_arcsec": float(positions[1, 0]),
        "source_b_y_arcsec": float(positions[1, 1]),
        "separation_arcsec": separation,
        "angle_radian": angle,
        "near_collision_pair_id": "",
        "near_collision_pair_side": "",
        "requested_source_contract": "both prompts A and B",
    }


def create_definitions(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    commitments = read_csv(run_dir / "manifests/approved_source_commitments.csv")
    pools = {partition: [row for row in commitments if row["campaign_partition"] == partition] for partition in PARTITIONS}
    excluded = {row["source_group"] for row in read_csv(run_dir / "tables/atlas_source_exclusion_audit.csv")}
    if any(row["duplicate_group_id"] in excluded for row in commitments):
        raise RuntimeError("Atlas source entered approved commitments")
    pool_rows: list[dict[str, object]] = []
    ordinary_rows: list[dict[str, object]] = []
    for partition in PARTITIONS:
        pool_rng = np.random.default_rng(POOL_SEEDS[partition])
        for index in range(POOL_COUNTS[partition]):
            seed = POOL_SEEDS[partition] + index
            pool_rows.append(scene_row(partition, "near_pool", index, choose_two(pools[partition], pool_rng), pool_rng, seed))
        ordinary_rng = np.random.default_rng(ORDINARY_SEEDS[partition])
        for index in range(ORDINARY_COUNTS[partition]):
            seed = ORDINARY_SEEDS[partition] + index
            ordinary_rows.append(scene_row(partition, "ordinary", index, choose_two(pools[partition], ordinary_rng), ordinary_rng, seed))
    return pool_rows, ordinary_rows


class DefinitionSampling(SamplingFunction):
    def __init__(self, definitions: list[dict[str, object]]) -> None:
        super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
        self.stamp_size = STAMP_SIZE_ARCSEC
        self.definitions = definitions
        self.position = 0

    def __call__(self, table: Table) -> Table:
        row = self.definitions[self.position]
        self.position += 1
        output = table[[int(row["source_a_row"]), int(row["source_b_row"])]].copy()
        output["ra"] = [float(row["source_a_x_arcsec"]), float(row["source_b_x_arcsec"])]
        output["dec"] = [float(row["source_a_y_arcsec"]), float(row["source_b_y_arcsec"])]
        return output


def create_pool_h5(path: Path, count: int) -> h5py.File:
    handle = h5py.File(path, "x")
    handle.create_dataset("blend", shape=(count, 3, 60, 60), dtype="f4", chunks=(1, 3, 60, 60), compression="lzf")
    handle.create_dataset("isolated", shape=(count, 2, 3, 60, 60), dtype="f4", chunks=(1, 2, 3, 60, 60), compression="lzf")
    handle.create_dataset("target_flux", shape=(count, 3), dtype="f8")
    handle.create_dataset("blend_hash", shape=(count,), dtype="S64")
    handle.create_dataset("target_hash", shape=(count,), dtype="S64")
    handle.attrs["complete"] = False
    handle.attrs["completed_count"] = 0
    return handle


def generate_pool(run_dir: Path) -> None:
    pool_rows, ordinary_rows = create_definitions(run_dir)
    write_csv_fresh(run_dir / "manifests/near_collision_pool_definitions.csv", pool_rows)
    write_csv_fresh(run_dir / "manifests/ordinary_scene_definitions.csv", ordinary_rows)
    catalog, catalog_hash = load_catsim_catalog(CATALOG)
    survey = validated_lsst_survey()
    full_bands = tuple(survey.available_filters)
    band_indices = [full_bands.index(band) for band in BAND_ORDER]
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    manifest_rows: list[dict[str, object]] = []
    started = time.time()
    for partition in PARTITIONS:
        rows = [row for row in pool_rows if row["partition"] == partition]
        h5_path = run_dir / f"features/near_pool_{partition}.h5"
        embedding_path = run_dir / f"features/near_pool_{partition}_blend_embeddings.npy"
        target_path = run_dir / f"features/near_pool_{partition}_target_embeddings.npy"
        embeddings = np.empty((len(rows), 675), dtype=np.float32)
        target_embeddings = np.empty((len(rows), 675), dtype=np.float32)
        with create_pool_h5(h5_path, len(rows)) as handle:
            for start in range(0, len(rows), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(rows))
                chunk = rows[start:stop]
                batch = next(CatsimGenerator(
                    catalog, DefinitionSampling(chunk), survey, batch_size=len(chunk), njobs=1,
                    verbose=False, use_bar=False, add_noise="none", seed=POOL_SEEDS[partition] + start,
                    apply_shear=False, augment_data=False,
                ))
                blends64 = np.asarray(batch.blend_images[:, band_indices], dtype=np.float64)
                isolated64 = np.asarray(batch.isolated_images[:, :2][:, :, band_indices], dtype=np.float64)
                blends = np.asarray(blends64, dtype=np.float32)
                isolated = np.asarray(isolated64, dtype=np.float32)
                embeddings[start:stop] = downsample(blends / np.sqrt(sky[None, :, None, None])).reshape(len(chunk), -1)
                target_embeddings[start:stop] = downsample(isolated[:, 0] / np.sqrt(sky[None, :, None, None])).reshape(len(chunk), -1)
                handle["blend"][start:stop] = blends
                handle["isolated"][start:stop] = isolated
                handle["target_flux"][start:stop] = isolated[:, 0].sum(axis=(-2, -1), dtype=np.float64)
                for local, row in enumerate(chunk):
                    index = start + local
                    expected = [int(row["source_a_row"]), int(row["source_b_row"])]
                    observed = list(np.asarray(batch.catalog_list[local]["catalog_row"], dtype=int))
                    if observed != expected or not np.array_equal(blends64[local], isolated64[local].sum(axis=0, dtype=np.float64)):
                        raise RuntimeError(f"pool source/additivity failure: {row['scene_id']}")
                    blend_hash = canonical_tensor_sha256(blends[local])
                    target_hash = canonical_tensor_sha256(isolated[local, 0])
                    handle["blend_hash"][index] = blend_hash.encode()
                    handle["target_hash"][index] = target_hash.encode()
                    manifest_rows.append({
                        "scene_id": row["scene_id"], "partition": partition, "pool_index": row["partition_index"],
                        "source_a_group": row["source_a_group"], "source_b_group": row["source_b_group"],
                        "blend_sha256": blend_hash, "source_a_sha256": target_hash,
                        "source_b_sha256": canonical_tensor_sha256(isolated[local, 1]),
                        "additivity_exact": True, "finite": True, "catalog_sha256": catalog_hash,
                    })
                handle.attrs["completed_count"] = stop
                print(json.dumps({"phase": "pool", "partition": partition, "completed": stop, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
            handle.attrs["complete"] = True
            handle.attrs["canonical_hash_schema"] = SCHEMA_VERSION
            handle.attrs["sky_electrons_grz"] = json.dumps(sky.tolist())
        with embedding_path.open("xb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        with target_path.open("xb") as handle:
            np.save(handle, target_embeddings, allow_pickle=False)
    write_csv_fresh(run_dir / "tables/near_collision_pool_render_manifest.csv", manifest_rows)
    write_json_fresh(run_dir / "logs/near_collision_pool_complete.json", {
        "status": "PASS", "pool_counts": POOL_COUNTS, "ordinary_definition_counts": ORDINARY_COUNTS,
        "pool_scene_count": len(pool_rows), "ordinary_scene_count": len(ordinary_rows),
        "atlas_source_exposure_count": 0, "canonical_hash_schema": SCHEMA_VERSION,
        "runtime_seconds": time.time() - started, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
    })


def nontrivial_rescaling(left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
    denominator = float(np.sum(right * right))
    if denominator <= np.finfo(float).tiny:
        return False, 0.0
    alpha = float(np.sum(left * right) / denominator)
    residual = float(np.linalg.norm(left - alpha * right) / (np.linalg.norm(left) + np.finfo(float).tiny))
    return residual > 0.01, residual


def search_partition(run_dir: Path, partition: str, rows: list[dict[str, str]], sky: np.ndarray, mean_psf: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    embeddings = np.load(run_dir / f"features/near_pool_{partition}_blend_embeddings.npy", allow_pickle=False)
    target_embeddings = np.load(run_dir / f"features/near_pool_{partition}_target_embeddings.npy", allow_pickle=False)
    if embeddings.shape != (len(rows), 675):
        raise RuntimeError("pool embedding alignment failure")
    tree = cKDTree(embeddings)
    distances, neighbors = tree.query(embeddings, k=NEIGHBORS, workers=-1)
    ranked: list[tuple[float, int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for left in range(len(rows)):
        left_groups = {rows[left]["source_a_group"], rows[left]["source_b_group"]}
        for embedding_distance, right_value in zip(distances[left, 1:], neighbors[left, 1:]):
            right = int(right_value)
            pair = (min(left, right), max(left, right))
            if pair in seen:
                continue
            seen.add(pair)
            if left_groups & {rows[right]["source_a_group"], rows[right]["source_b_group"]}:
                continue
            target_distance = float(
                np.linalg.norm(target_embeddings[left] - target_embeddings[right])
                / (0.5 * (np.linalg.norm(target_embeddings[left]) + np.linalg.norm(target_embeddings[right])) + 1e-12)
            )
            if target_distance <= 0.8:
                continue
            ranked.append((float(embedding_distance / target_distance), pair[0], pair[1], target_distance))
    ranked.sort()
    selected: list[dict[str, object]] = []
    used_scenes: set[int] = set()
    evaluated = 0
    with h5py.File(run_dir / f"features/near_pool_{partition}.h5", "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("incomplete near-collision pool")
        for rank, left, right, estimated_target in ranked:
            if left in used_scenes or right in used_scenes:
                continue
            left_blend = np.asarray(handle["blend"][left], dtype=np.float64)
            right_blend = np.asarray(handle["blend"][right], dtype=np.float64)
            left_isolated = np.asarray(handle["isolated"][left], dtype=np.float64)
            right_isolated = np.asarray(handle["isolated"][right], dtype=np.float64)
            evaluated += 1
            variance = np.maximum(0.5 * (left_blend + right_blend) + sky[:, None, None], 1.0)
            blend_distance = float(np.mean((left_blend - right_blend) ** 2 / variance))
            target = scientific_distance(
                left_isolated[0], right_isolated[0], mean_psf_fwhm_pixel=mean_psf,
                image_floor=1e-12, flux_floor=1e-12,
            )
            rescaling_pass, scale_residual = nontrivial_rescaling(left_blend, right_blend)
            if blend_distance > BLEND_DISTANCE_LIMIT or target.primary_normalized <= TARGET_DISTANCE_LIMIT or not rescaling_pass:
                continue
            pair_id = f"pu_{partition}_pair_{len(selected) + 1:05d}"
            selected.append({
                "near_collision_pair_id": pair_id,
                "partition": partition,
                "left_pool_index": left,
                "right_pool_index": right,
                "left_scene_id": rows[left]["scene_id"],
                "right_scene_id": rows[right]["scene_id"],
                "left_source_a_group": rows[left]["source_a_group"],
                "left_source_b_group": rows[left]["source_b_group"],
                "right_source_a_group": rows[right]["source_a_group"],
                "right_source_b_group": rows[right]["source_b_group"],
                "blend_whitened_mse": blend_distance,
                "target_primary_diameter": target.primary_normalized,
                "target_image_distance": target.image,
                "target_centroid_psf": target.centroid_psf,
                "estimated_target_distance": estimated_target,
                "rank_score": rank,
                "global_rescaling_relative_residual": scale_residual,
                "four_groups_disjoint": True,
                "pool_scenes_unique": True,
                "construction": "independent partition-specific fixed-embedding near-collision search",
            })
            used_scenes.update((left, right))
            if len(selected) >= PAIR_COUNTS[partition]:
                break
    summary = {
        "partition": partition, "pool_size": len(rows), "ranked_candidates": len(ranked),
        "exact_candidates_evaluated": evaluated, "valid_disjoint_pairs": len(selected),
        "required_pairs": PAIR_COUNTS[partition], "status": "PASS" if len(selected) == PAIR_COUNTS[partition] else "FAIL",
    }
    return selected, summary


def create_final_definitions(
    run_dir: Path,
    pool_rows: list[dict[str, str]],
    ordinary_rows: list[dict[str, str]],
    pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_partition_index = {(row["partition"], int(row["partition_index"])): row for row in pool_rows}
    output: list[dict[str, object]] = [dict(row) for row in ordinary_rows]
    near_counter = Counter()
    for pair in pairs:
        partition = str(pair["partition"])
        for side in ("left", "right"):
            pool_index = int(pair[f"{side}_pool_index"])
            row = dict(by_partition_index[(partition, pool_index)])
            local_index = near_counter[partition]
            near_counter[partition] += 1
            row["scene_id"] = f"pu_{partition}_near_{local_index:05d}"
            row["kind"] = "near_collision"
            row["partition_index"] = ORDINARY_COUNTS[partition] + local_index
            row["near_collision_pair_id"] = pair["near_collision_pair_id"]
            row["near_collision_pair_side"] = side
            output.append(row)
    expected = {
        "training": ORDINARY_COUNTS["training"] + 2 * PAIR_COUNTS["training"],
        "validation": ORDINARY_COUNTS["validation"] + 2 * PAIR_COUNTS["validation"],
        "calibration": ORDINARY_COUNTS["calibration"] + 2 * PAIR_COUNTS["calibration"],
    }
    counts = Counter(str(row["partition"]) for row in output)
    if counts != Counter(expected):
        raise RuntimeError(f"final scene count mismatch: {counts} != {expected}")
    output.sort(key=lambda row: (PARTITIONS.index(str(row["partition"])), int(row["partition_index"])))
    return output


def search_pools(run_dir: Path) -> None:
    pool_rows = read_csv(run_dir / "manifests/near_collision_pool_definitions.csv")
    ordinary_rows = read_csv(run_dir / "manifests/ordinary_scene_definitions.csv")
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    mean_psf = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in BAND_ORDER]) / 0.2)
    pairs: list[dict[str, object]] = []
    summaries = []
    started = time.time()
    for partition in PARTITIONS:
        rows = [row for row in pool_rows if row["partition"] == partition]
        selected, summary = search_partition(run_dir, partition, rows, sky, mean_psf)
        pairs.extend(selected)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    write_csv_fresh(run_dir / "tables/non_atlas_near_collision_pair_manifest.csv", pairs)
    write_csv_fresh(run_dir / "tables/non_atlas_near_collision_search_summary.csv", summaries)
    if any(row["status"] != "PASS" for row in summaries):
        write_text_fresh(run_dir / "diagnostics/non_atlas_near_collision_report.md", "# Non-Atlas near-collision search\n\nStatus: **FAIL — REQUIRED PAIR COUNT NOT ATTAINED**.\n")
        raise RuntimeError(f"near-collision pair gate failed: {summaries}")
    definitions = create_final_definitions(run_dir, pool_rows, ordinary_rows, pairs)
    write_csv_fresh(run_dir / "manifests/probabilistic_unet_scene_definitions.csv", definitions)
    report = f"""# Non-Atlas near-collision construction report

Status: **PASS**.

- Training pairs/observations: {PAIR_COUNTS['training']:,} / {2 * PAIR_COUNTS['training']:,}.
- Validation pairs/observations: {PAIR_COUNTS['validation']:,} / {2 * PAIR_COUNTS['validation']:,}.
- Calibration pairs/observations: {PAIR_COUNTS['calibration']:,} / {2 * PAIR_COUNTS['calibration']:,}.
- Pair requirements: four disjoint groups, unique pool scenes, whitened blend MSE
  <= {BLEND_DISTANCE_LIMIT}, target primary scientific distance > {TARGET_DISTANCE_LIMIT},
  and nontrivial global-rescaling residual >0.01.
- Pools, seeds, and searches are partition-specific. Atlas groups are absent.
- Atlas pairs, Atlas thresholds/results, development, and lockbox were not accessed.
"""
    write_text_fresh(run_dir / "diagnostics/non_atlas_near_collision_report.md", report)
    write_json_fresh(run_dir / "logs/near_collision_search_complete.json", {
        "status": "PASS", "pair_counts": PAIR_COUNTS, "scene_counts": dict(Counter(row["partition"] for row in definitions)),
        "pair_manifest_sha256": sha256_file(run_dir / "tables/non_atlas_near_collision_pair_manifest.csv"),
        "scene_manifest_sha256": sha256_file(run_dir / "manifests/probabilistic_unet_scene_definitions.csv"),
        "runtime_seconds": time.time() - started, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
    })


def spec_from_row(row: dict[str, str]) -> SceneSpec:
    return SceneSpec(
        scene_id=row["scene_id"],
        catalog_rows=(int(row["source_a_row"]), int(row["source_b_row"])),
        positions_arcsec=(
            (float(row["source_a_x_arcsec"]), float(row["source_a_y_arcsec"])),
            (float(row["source_b_x_arcsec"]), float(row["source_b_y_arcsec"])),
        ),
        source_selection_seed=int(row["source_selection_seed"]),
        position_seed=int(row["position_seed"]),
        noise_seed=int(row["noise_seed"]),
    )


def create_scene_h5(path: Path, count: int) -> h5py.File:
    handle = h5py.File(path, "x")
    handle.create_dataset("blend", shape=(count, 3, 60, 60), dtype="f4", chunks=(1, 3, 60, 60), compression="lzf")
    handle.create_dataset("isolated", shape=(count, 2, 3, 60, 60), dtype="f4", chunks=(1, 2, 3, 60, 60), compression="lzf")
    handle.create_dataset("xy", shape=(count, 2, 2), dtype="f8")
    handle.attrs["complete"] = False
    handle.attrs["completed_count"] = 0
    handle.attrs["canonical_hash_schema"] = SCHEMA_VERSION
    return handle


def render_data(run_dir: Path) -> None:
    definitions = read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
    excluded = {row["source_group"] for row in read_csv(run_dir / "tables/atlas_source_exclusion_audit.csv")}
    if any(row["source_a_group"] in excluded or row["source_b_group"] in excluded for row in definitions):
        raise RuntimeError("Atlas source exposure in final scene definitions")
    catalog, _ = load_catsim_catalog(CATALOG)
    manifest: list[dict[str, object]] = []
    started = time.time()
    for partition in PARTITIONS:
        rows = [row for row in definitions if row["partition"] == partition]
        path = run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        with create_scene_h5(path, len(rows)) as handle:
            for index, row in enumerate(rows):
                rendered = render_fixed_scene(catalog, spec_from_row(row), add_noise="all")
                blend = np.asarray(rendered.blend, dtype=np.float32)
                isolated = np.asarray(rendered.isolated, dtype=np.float32)
                noiseless_sum = isolated.sum(axis=0, dtype=np.float32)
                xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered.catalog], dtype=np.float64)
                prompts = [gaussian_prompt_numpy(float(x), float(y)) for x, y in xy]
                noise = blend - noiseless_sum
                hashes = {
                    "blend_sha256": canonical_tensor_sha256(blend),
                    "source_a_sha256": canonical_tensor_sha256(isolated[0]),
                    "source_b_sha256": canonical_tensor_sha256(isolated[1]),
                    "noiseless_sum_sha256": canonical_tensor_sha256(noiseless_sum),
                    "noise_realization_sha256": canonical_tensor_sha256(noise),
                    "prompt_a_sha256": canonical_tensor_sha256(prompts[0][None]),
                    "prompt_b_sha256": canonical_tensor_sha256(prompts[1][None]),
                }
                handle["blend"][index] = blend
                handle["isolated"][index] = isolated
                handle["xy"][index] = xy
                handle.attrs["completed_count"] = index + 1
                manifest.append({
                    "scene_id": row["scene_id"], "partition": partition, "partition_index": index,
                    "kind": row["kind"], "near_collision_pair_id": row["near_collision_pair_id"],
                    "near_collision_pair_side": row["near_collision_pair_side"],
                    "source_a_id": row["source_a_id"], "source_b_id": row["source_b_id"],
                    "source_a_group": row["source_a_group"], "source_b_group": row["source_b_group"],
                    "source_a_x_arcsec": row["source_a_x_arcsec"], "source_a_y_arcsec": row["source_a_y_arcsec"],
                    "source_b_x_arcsec": row["source_b_x_arcsec"], "source_b_y_arcsec": row["source_b_y_arcsec"],
                    "prompt_a_x_pixel": xy[0, 0], "prompt_a_y_pixel": xy[0, 1],
                    "prompt_b_x_pixel": xy[1, 0], "prompt_b_y_pixel": xy[1, 1],
                    "source_selection_seed": row["source_selection_seed"], "position_seed": row["position_seed"],
                    "noise_seed": row["noise_seed"], "psf_contract": "fixed LSST g/r/z 0.86/0.81/0.77 arcsec",
                    "noise_contract": "BTK add_noise=all source+sky Poisson on summed scene",
                    "band_order": "g,r,z", "units": "detected electrons per pixel", "clipping": False,
                    "background_semantics": "two noiseless zero-background source contributions", **hashes,
                })
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(json.dumps({"phase": "render", "partition": partition, "completed": index + 1, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
            handle.attrs["complete"] = True
    write_csv_fresh(run_dir / "manifests/probabilistic_unet_rendered_scene_manifest.csv", manifest)
    write_json_fresh(run_dir / "logs/data_render_complete.json", {
        "status": "PASS", "scene_count": len(definitions), "query_count": 2 * len(definitions),
        "partition_counts": dict(Counter(row["partition"] for row in definitions)),
        "ordinary_counts": ORDINARY_COUNTS, "near_collision_observation_counts": {key: 2 * value for key, value in PAIR_COUNTS.items()},
        "atlas_source_exposure_count": 0, "canonical_hash_schema": SCHEMA_VERSION,
        "render_manifest_sha256": sha256_file(run_dir / "manifests/probabilistic_unet_rendered_scene_manifest.csv"),
        "runtime_seconds": time.time() - started, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
    })


def replay_data(run_dir: Path) -> None:
    definitions = read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
    catalog, _ = load_catsim_catalog(CATALOG)
    output: list[dict[str, object]] = []
    started = time.time()
    for partition in PARTITIONS:
        rows = [row for row in definitions if row["partition"] == partition]
        path = run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs["complete"]) or int(handle.attrs["completed_count"]) != len(rows):
                raise RuntimeError("incomplete rendered HDF5")
            for index, row in enumerate(rows):
                spec = spec_from_row(row)
                noisy = render_fixed_scene(catalog, spec, add_noise="all")
                noiseless = render_fixed_scene(catalog, spec, add_noise="none")
                stored_blend = np.asarray(handle["blend"][index], dtype=np.float32)
                stored_isolated = np.asarray(handle["isolated"][index], dtype=np.float32)
                stored_xy = np.asarray(handle["xy"][index], dtype=np.float64)
                noisy_blend = np.asarray(noisy.blend, dtype=np.float32)
                noisy_isolated = np.asarray(noisy.isolated, dtype=np.float32)
                noiseless_blend = np.asarray(noiseless.blend, dtype=np.float32)
                noiseless_isolated = np.asarray(noiseless.isolated, dtype=np.float32)
                noisy_xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in noisy.catalog], dtype=np.float64)
                checks = {
                    "source_order_exact": list(np.asarray(noisy.catalog["catalog_row"], dtype=int)) == [int(row["source_a_row"]), int(row["source_b_row"])],
                    "noisy_blend_exact": np.array_equal(noisy_blend, stored_blend),
                    "isolated_exact": np.array_equal(noisy_isolated, stored_isolated),
                    "noiseless_isolated_exact": np.array_equal(noiseless_isolated, stored_isolated),
                    "noiseless_additivity_exact": np.array_equal(noiseless.blend, noiseless.isolated.sum(axis=0, dtype=np.float64)),
                    "coordinates_exact": np.array_equal(noisy_xy, stored_xy),
                    "blend_hash_exact": canonical_tensor_sha256(noisy_blend) == canonical_tensor_sha256(stored_blend),
                    "source_a_hash_exact": canonical_tensor_sha256(noisy_isolated[0]) == canonical_tensor_sha256(stored_isolated[0]),
                    "source_b_hash_exact": canonical_tensor_sha256(noisy_isolated[1]) == canonical_tensor_sha256(stored_isolated[1]),
                }
                status = "PASS" if all(checks.values()) else "FAIL"
                output.append({"scene_id": row["scene_id"], "partition": partition, **checks, "status": status})
                if status != "PASS":
                    raise RuntimeError(f"manifest replay failed: {row['scene_id']} {checks}")
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(json.dumps({"phase": "replay", "partition": partition, "completed": index + 1, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
    write_csv_fresh(run_dir / "tables/manifest_replay_tests.csv", output)
    write_json_fresh(run_dir / "logs/data_preparation_complete.json", {
        "status": "PASS", "scene_count": len(definitions), "replay_count": len(output),
        "replay_pass_count": sum(row["status"] == "PASS" for row in output),
        "atlas_source_exposure_count": 0, "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0, "atlas_evaluation_count": 0,
        "scene_manifest_sha256": sha256_file(run_dir / "manifests/probabilistic_unet_scene_definitions.csv"),
        "render_manifest_sha256": sha256_file(run_dir / "manifests/probabilistic_unet_rendered_scene_manifest.csv"),
        "replay_table_sha256": sha256_file(run_dir / "tables/manifest_replay_tests.csv"),
        "h5_sha256": {partition: sha256_file(run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5") for partition in PARTITIONS},
        "runtime_seconds": time.time() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pool", "search", "render", "replay"), required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir, args.phase)
    if args.phase == "pool":
        generate_pool(run_dir)
    elif args.phase == "search":
        search_pools(run_dir)
    elif args.phase == "render":
        render_data(run_dir)
    else:
        replay_data(run_dir)


if __name__ == "__main__":
    main()
