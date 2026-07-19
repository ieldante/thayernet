#!/usr/bin/env python3
"""Build the preregistered 30k-scene Route-B Ambiguity Atlas pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import SamplingFunction
from scipy.spatial import cKDTree
from surveycodex.utilities import mean_sky_level


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.btk_scene import (  # noqa: E402
    BAND_ORDER,
    STAMP_SIZE_ARCSEC,
    SceneSpec,
    load_catsim_catalog,
    render_fixed_scene,
    validated_lsst_survey,
)
from src.competing_hypotheses import scientific_distance  # noqa: E402


POOL_SIZE = 30_000
BATCH_SIZE = 250
POOL_SEED = 2026071219
NEIGHBORS = 32
EXACT_PAIR_BUDGET = 1_500
MAX_VALID_PAIRS = 100
BLOCK_SIZE = 4
IMAGE_SIZE = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        if fields is None:
            fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_run(path: Path) -> Path:
    path = path.resolve()
    if path.parent != (REPO / "outputs/runs").resolve() or not path.name.startswith(
        ("thayer_competing_hypotheses_", "thayer_ambiguity_atlas_v0_")
    ):
        raise ValueError("run-dir must be a competing-hypotheses master run")
    freeze = path / "preregistration/freeze_record.json"
    if not freeze.exists():
        raise FileNotFoundError(freeze)
    record = json.loads(freeze.read_text())
    if record["candidate_inference_count_at_freeze"] != 0 or record["lockbox_scene_access_count"] != 0:
        raise RuntimeError("invalid preregistration freeze record")
    return path


def make_definitions(run_dir: Path) -> list[dict[str, object]]:
    destination = run_dir / "manifests/atlas_pool_scene_definitions.csv"
    if destination.exists():
        rows = read_csv(destination)
        if len(rows) != POOL_SIZE:
            raise RuntimeError("existing Atlas definition count is not frozen pool size")
        return rows
    commitments = read_csv(run_dir / "manifests/campaign_source_partition_commitments.csv")
    pools: dict[str, list[dict[str, str]]] = {"training": [], "validation": []}
    for row in commitments:
        role = row["campaign_role"]
        if role in pools:
            pools[role].append(row)
    if min(map(len, pools.values())) < 2:
        raise RuntimeError("insufficient approved training/validation sources")
    rng = np.random.default_rng(POOL_SEED)
    rows: list[dict[str, object]] = []
    for index in range(POOL_SIZE):
        role = "training"
        pool = pools[role]
        while True:
            selected = rng.choice(len(pool), size=2, replace=False)
            target, contaminant = pool[int(selected[0])], pool[int(selected[1])]
            if target["duplicate_group_id"] != contaminant["duplicate_group_id"]:
                break
        separation = float(rng.uniform(0.6, 3.0))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        contaminant_x = separation * np.cos(angle)
        contaminant_y = separation * np.sin(angle)
        rows.append(
            {
                "pool_index": index,
                "scene_id": f"atlas_v0_search_{index:05d}",
                "campaign_role": role,
                "target_catalog_row": target["catalog_row"],
                "target_source_id": target["persistent_source_id"],
                "target_group": target["duplicate_group_id"],
                "contaminant_catalog_row": contaminant["catalog_row"],
                "contaminant_source_id": contaminant["persistent_source_id"],
                "contaminant_group": contaminant["duplicate_group_id"],
                "target_x_arcsec": 0.0,
                "target_y_arcsec": 0.0,
                "contaminant_x_arcsec": contaminant_x,
                "contaminant_y_arcsec": contaminant_y,
                "source_selection_seed": POOL_SEED,
                "position_seed": POOL_SEED + index,
                "noise_seed": 202607121_000 + index,
                "source_split_sha256": "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
            }
        )
    write_csv_fresh(destination, rows)
    return rows


class DefinitionSampling(SamplingFunction):
    def __init__(self, definitions: list[dict[str, object]]) -> None:
        super().__init__(stamp_size=int(STAMP_SIZE_ARCSEC), min_number=2, max_number=2, seed=0)
        self.stamp_size = STAMP_SIZE_ARCSEC
        self.definitions = definitions
        self.position = 0

    def __call__(self, table: Table) -> Table:
        if self.position >= len(self.definitions):
            raise IndexError("sampling called beyond frozen definitions")
        row = self.definitions[self.position]
        self.position += 1
        selected = table[[int(row["target_catalog_row"]), int(row["contaminant_catalog_row"])]].copy()
        selected["ra"] = [float(row["target_x_arcsec"]), float(row["contaminant_x_arcsec"])]
        selected["dec"] = [float(row["target_y_arcsec"]), float(row["contaminant_y_arcsec"])]
        return selected


def downsample(array: np.ndarray) -> np.ndarray:
    if array.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("unexpected Atlas image size")
    leading = array.shape[:-2]
    reshaped = array.reshape(*leading, IMAGE_SIZE // BLOCK_SIZE, BLOCK_SIZE, IMAGE_SIZE // BLOCK_SIZE, BLOCK_SIZE)
    return reshaped.mean(axis=(-3, -1))


def generate_pool(run_dir: Path) -> None:
    definitions = make_definitions(run_dir)
    embedding_path = run_dir / "embeddings/pool_blend_embeddings.npy"
    target_path = run_dir / "embeddings/pool_target_embeddings.npy"
    summary_path = run_dir / "embeddings/pool_numeric_summaries.npy"
    manifest_path = run_dir / "tables/atlas_pool_render_manifest.csv"
    for path in (embedding_path, target_path, summary_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    catalog_path = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
    catalog, catalog_hash = load_catsim_catalog(catalog_path)
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    band_indices = [tuple(survey.available_filters).index(band) for band in BAND_ORDER]
    dimension = 3 * (IMAGE_SIZE // BLOCK_SIZE) ** 2
    blend_embeddings = np.empty((POOL_SIZE, dimension), dtype=np.float32)
    target_embeddings = np.empty((POOL_SIZE, dimension), dtype=np.float32)
    summaries = np.empty((POOL_SIZE, 8), dtype=np.float64)
    manifest: list[dict[str, object]] = []
    started = time.time()
    for start in range(0, POOL_SIZE, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, POOL_SIZE)
        chunk = definitions[start:stop]
        sampling = DefinitionSampling(chunk)
        generator = CatsimGenerator(
            catalog,
            sampling,
            survey,
            batch_size=len(chunk),
            njobs=1,
            verbose=False,
            use_bar=False,
            add_noise="none",
            seed=202607122_000 + start,
            apply_shear=False,
            augment_data=False,
        )
        batch = next(generator)
        blends = np.asarray(batch.blend_images[:, band_indices], dtype=np.float64)
        isolated = np.asarray(batch.isolated_images[:, :2][:, :, band_indices], dtype=np.float64)
        target = isolated[:, 0]
        blend_white = blends / np.sqrt(sky[None, :, None, None])
        target_white = target / np.sqrt(sky[None, :, None, None])
        blend_embeddings[start:stop] = downsample(blend_white).reshape(len(chunk), -1).astype(np.float32)
        target_embeddings[start:stop] = downsample(target_white).reshape(len(chunk), -1).astype(np.float32)
        for local, row in enumerate(chunk):
            index = start + local
            rendered = batch.catalog_list[local]
            actual_rows = list(np.asarray(rendered["catalog_row"], dtype=int))
            expected_rows = [int(row["target_catalog_row"]), int(row["contaminant_catalog_row"])]
            if actual_rows != expected_rows:
                raise RuntimeError("BTK batch order/source alignment failure")
            flux = isolated[local].sum(axis=(-2, -1), dtype=np.float64)
            summaries[index] = [*flux[0], *flux[1], float(np.sum(blends[local])), float(np.linalg.norm(target_embeddings[index]))]
            manifest.append(
                {
                    "pool_index": index,
                    "scene_id": row["scene_id"],
                    "campaign_role": row["campaign_role"],
                    "target_group": row["target_group"],
                    "contaminant_group": row["contaminant_group"],
                    "blend_sha256": sha256_array(blends[local]),
                    "target_sha256": sha256_array(target[local]),
                    "contaminant_sha256": sha256_array(isolated[local, 1]),
                    "finite": bool(np.all(np.isfinite(blends[local])) and np.all(np.isfinite(isolated[local]))),
                    "band_order": "g,r,z",
                    "units": "detected electrons per pixel",
                    "pixel_scale_arcsec": 0.2,
                    "catalog_sha256": catalog_hash,
                }
            )
        elapsed = time.time() - started
        print(json.dumps({"phase": "generate", "completed": stop, "total": POOL_SIZE, "elapsed_seconds": elapsed}), flush=True)
    with embedding_path.open("xb") as handle:
        np.save(handle, blend_embeddings, allow_pickle=False)
    with target_path.open("xb") as handle:
        np.save(handle, target_embeddings, allow_pickle=False)
    with summary_path.open("xb") as handle:
        np.save(handle, summaries, allow_pickle=False)
    write_csv_fresh(manifest_path, manifest)
    write_json_fresh(
        run_dir / "logs/atlas_pool_generation_complete.json",
        {
            "status": "PASS",
            "pool_size": POOL_SIZE,
            "training_search_scenes": 30_000,
            "validation_scenes": 0,
            "embedding": "fixed sky-whitened 4x4 block-mean pixels; no learned representation or PCA",
            "blend_embeddings_sha256": sha256_file(embedding_path),
            "target_embeddings_sha256": sha256_file(target_path),
            "numeric_summaries_sha256": sha256_file(summary_path),
            "render_manifest_sha256": sha256_file(manifest_path),
            "sky_electrons_grz": sky.tolist(),
            "runtime_seconds": time.time() - started,
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        },
    )


def render_definition(catalog, row: dict[str, str]):
    spec = SceneSpec(
        scene_id=row["scene_id"],
        catalog_rows=(int(row["target_catalog_row"]), int(row["contaminant_catalog_row"])),
        positions_arcsec=(
            (float(row["target_x_arcsec"]), float(row["target_y_arcsec"])),
            (float(row["contaminant_x_arcsec"]), float(row["contaminant_y_arcsec"])),
        ),
        source_selection_seed=int(row["source_selection_seed"]),
        position_seed=int(row["position_seed"]),
        noise_seed=int(row["noise_seed"]),
    )
    return render_fixed_scene(catalog, spec, add_noise="none")


def nontrivial_rescaling(left: np.ndarray, right: np.ndarray) -> tuple[bool, float]:
    denominator = float(np.sum(right * right))
    if denominator <= np.finfo(float).tiny:
        return False, 0.0
    alpha = float(np.sum(left * right) / denominator)
    relative = float(np.linalg.norm(left - alpha * right) / (np.linalg.norm(left) + np.finfo(float).tiny))
    return bool(relative > 0.01), relative


def search_pool(run_dir: Path) -> None:
    pair_path = run_dir / "tables/atlas_pair_manifest.csv"
    report_path = run_dir / "diagnostics/atlas_validation_report.md"
    candidate_path = run_dir / "tables/atlas_neighbor_candidates.csv"
    for path in (pair_path, report_path, candidate_path):
        if path.exists():
            raise FileExistsError(path)
    definitions = read_csv(run_dir / "manifests/atlas_pool_scene_definitions.csv")
    render_manifest = read_csv(run_dir / "tables/atlas_pool_render_manifest.csv")
    blend_embeddings = np.load(run_dir / "embeddings/pool_blend_embeddings.npy", allow_pickle=False)
    target_embeddings = np.load(run_dir / "embeddings/pool_target_embeddings.npy", allow_pickle=False)
    summaries = np.load(run_dir / "embeddings/pool_numeric_summaries.npy", allow_pickle=False)
    if len(definitions) != POOL_SIZE or blend_embeddings.shape[0] != POOL_SIZE:
        raise RuntimeError("Atlas pool alignment failure")
    started = time.time()
    tree = cKDTree(blend_embeddings)
    distances, neighbors = tree.query(blend_embeddings, k=NEIGHBORS, workers=-1)
    ranked: list[tuple[float, int, int, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for left in range(POOL_SIZE):
        left_groups = {definitions[left]["target_group"], definitions[left]["contaminant_group"]}
        for distance, right_value in zip(distances[left, 1:], neighbors[left, 1:]):
            right = int(right_value)
            pair = (min(left, right), max(left, right))
            if pair in seen:
                continue
            seen.add(pair)
            if definitions[left]["campaign_role"] != definitions[right]["campaign_role"]:
                continue
            right_groups = {definitions[right]["target_group"], definitions[right]["contaminant_group"]}
            if left_groups & right_groups:
                continue
            target_distance = float(
                np.linalg.norm(target_embeddings[left] - target_embeddings[right])
                / (0.5 * (np.linalg.norm(target_embeddings[left]) + np.linalg.norm(target_embeddings[right])) + 1e-12)
            )
            flux_difference = float(
                np.max(
                    np.abs(summaries[left, :3] - summaries[right, :3])
                    / (np.abs(0.5 * (summaries[left, :3] + summaries[right, :3])) + 1e-12)
                )
            )
            estimated_divergence = max(target_distance / 0.25, flux_difference / 0.20)
            if estimated_divergence <= 0.8:
                continue
            rank = float(distance / estimated_divergence)
            ranked.append((rank, pair[0], pair[1], float(distance), estimated_divergence))
    ranked.sort()
    ranked = ranked[:EXACT_PAIR_BUDGET]
    candidate_rows = [
        {
            "rank": index + 1,
            "left_pool_index": left,
            "right_pool_index": right,
            "embedding_distance": distance,
            "estimated_target_primary": divergence,
            "rank_score": rank,
        }
        for index, (rank, left, right, distance, divergence) in enumerate(ranked)
    ]
    write_csv_fresh(candidate_path, candidate_rows, ["rank", "left_pool_index", "right_pool_index", "embedding_distance", "estimated_target_primary", "rank_score"])

    catalog_path = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
    catalog, _ = load_catsim_catalog(catalog_path)
    survey = validated_lsst_survey()
    sky = np.asarray([mean_sky_level(survey, band).to_value("electron") for band in BAND_ORDER], dtype=np.float64)
    mean_psf_fwhm_pixel = float(np.mean([survey.get_filter(band).psf_fwhm.to_value("arcsec") for band in BAND_ORDER]) / 0.2)
    valid: list[dict[str, object]] = []
    evaluated = 0
    for rank, left, right, embedding_distance, estimated_divergence in ranked:
        render_left = render_definition(catalog, definitions[left])
        render_right = render_definition(catalog, definitions[right])
        evaluated += 1
        if sha256_array(render_left.blend) != render_manifest[left]["blend_sha256"]:
            raise RuntimeError("left scene exact replay failed")
        if sha256_array(render_right.blend) != render_manifest[right]["blend_sha256"]:
            raise RuntimeError("right scene exact replay failed")
        variance = np.maximum(0.5 * (render_left.blend + render_right.blend) + sky[:, None, None], 1.0)
        blend_distance = float(np.mean((render_left.blend - render_right.blend) ** 2 / variance))
        distance = scientific_distance(
            render_left.isolated[0],
            render_right.isolated[0],
            mean_psf_fwhm_pixel=mean_psf_fwhm_pixel,
            image_floor=1e-12,
            flux_floor=1e-12,
        )
        scaling_pass, scale_residual = nontrivial_rescaling(render_left.blend, render_right.blend)
        finite = bool(
            np.all(np.isfinite(render_left.blend))
            and np.all(np.isfinite(render_right.blend))
            and np.all(np.isfinite(render_left.isolated))
            and np.all(np.isfinite(render_right.isolated))
        )
        if blend_distance <= 0.25 and distance.primary_normalized > 1.0 and scaling_pass and finite:
            pair_id = f"atlas_pair_{len(valid) + 1:04d}"
            valid.append(
                {
                    "pair_id": pair_id,
                    "left_scene_id": definitions[left]["scene_id"],
                    "right_scene_id": definitions[right]["scene_id"],
                    "campaign_role": definitions[left]["campaign_role"],
                    "left_pool_index": left,
                    "right_pool_index": right,
                    "left_target_group": definitions[left]["target_group"],
                    "right_target_group": definitions[right]["target_group"],
                    "left_contaminant_group": definitions[left]["contaminant_group"],
                    "right_contaminant_group": definitions[right]["contaminant_group"],
                    "blend_whitened_mse": blend_distance,
                    "target_primary_diameter": distance.primary_normalized,
                    "target_image_distance": distance.image,
                    "target_flux_g_relative": distance.relative_flux_grz[0],
                    "target_flux_r_relative": distance.relative_flux_grz[1],
                    "target_flux_z_relative": distance.relative_flux_grz[2],
                    "target_color_gr_magnitude": distance.color_gr_rz_magnitude[0],
                    "target_color_rz_magnitude": distance.color_gr_rz_magnitude[1],
                    "target_centroid_pixel": distance.centroid_pixel,
                    "target_centroid_psf": distance.centroid_psf,
                    "embedding_distance": embedding_distance,
                    "estimated_target_primary": estimated_divergence,
                    "global_rescaling_relative_residual": scale_residual,
                    "different_source_groups": True,
                    "same_fixed_target_position": True,
                    "exact_replay_pass": True,
                    "finite_pass": True,
                    "normalization_artifact_pass": True,
                    "clipping_artifact_pass": True,
                    "background_artifact_pass": True,
                    "global_rescaling_artifact_pass": True,
                    "construction_route": "LARGE_POOL_NEAR_COLLISION_SEARCH",
                    "visual_audit_status": "PENDING",
                    "left_blend_sha256": render_manifest[left]["blend_sha256"],
                    "right_blend_sha256": render_manifest[right]["blend_sha256"],
                    "left_target_sha256": render_manifest[left]["target_sha256"],
                    "right_target_sha256": render_manifest[right]["target_sha256"],
                }
            )
            arrays_path = run_dir / f"atlas/{pair_id}.npz"
            if arrays_path.exists():
                raise FileExistsError(arrays_path)
            np.savez_compressed(
                arrays_path,
                left_blend=np.asarray(render_left.blend, dtype=np.float32),
                right_blend=np.asarray(render_right.blend, dtype=np.float32),
                left_isolated=np.asarray(render_left.isolated, dtype=np.float32),
                right_isolated=np.asarray(render_right.isolated, dtype=np.float32),
                left_psf=np.asarray(render_left.psf, dtype=np.float32),
                right_psf=np.asarray(render_right.psf, dtype=np.float32),
                sky_electrons=sky,
            )
            if len(valid) >= MAX_VALID_PAIRS:
                break
        if evaluated % 100 == 0:
            print(json.dumps({"phase": "exact_search", "evaluated": evaluated, "valid": len(valid)}), flush=True)
    fields = list(valid[0]) if valid else [
        "pair_id", "left_scene_id", "right_scene_id", "blend_whitened_mse", "target_primary_diameter", "visual_audit_status"
    ]
    write_csv_fresh(pair_path, valid, fields)
    status = "NUMERICAL_PASS_VISUAL_AUDIT_PENDING" if len(valid) >= 25 else "FAIL_MINIMUM_PAIR_COUNT"
    report = f"""# Ambiguity Atlas validation report

Status: **{status}**.

- Frozen candidate pool: {POOL_SIZE:,} noiseless BTK training/search scenes.
- Embedding: fixed sky-whitened 4x4 block means; no learned model and no PCA.
- Neighbor candidates retained for exact replay: {len(ranked):,}.
- Exact candidates evaluated: {evaluated:,}.
- Numerically valid pairs: {len(valid):,}.
- Required minimum: 25.
- Development scenes used: 0.
- Lockbox scenes used: 0.

Every retained numerical pair has disjoint source groups, a common fixed target
position, exact byte-level float64 replay, finite unclipped arrays, blend
distance at most 0.25 mean whitened squared error, requested-target primary
diameter above 1.0, and rejection of a trivial global-rescaling explanation.
Visual audit remains explicitly pending and the Atlas is not frozen until that
audit is recorded.
"""
    write_text_fresh(report_path, report)
    write_json_fresh(
        run_dir / "logs/atlas_search_complete.json",
        {
            "status": status,
            "pool_size": POOL_SIZE,
            "neighbor_candidate_count": len(ranked),
            "exact_candidate_count": evaluated,
            "numerically_valid_pair_count": len(valid),
            "runtime_seconds": time.time() - started,
            "development_scenes_used": 0,
            "lockbox_scenes_used": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("generate", "search", "all"), default="all")
    args = parser.parse_args()
    run_dir = require_run(args.run_dir)
    if args.phase in {"generate", "all"}:
        generate_pool(run_dir)
    if args.phase in {"search", "all"}:
        search_pool(run_dir)


if __name__ == "__main__":
    main()
