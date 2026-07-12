#!/usr/bin/env python3
"""Generate fresh non-development data for hierarchical safety training.

This program never selects sealed-lockbox or development sources.  Development
generation is implemented separately and requires a frozen-policy record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene, validated_lsst_survey
from src.hierarchical_safety import (
    HierarchicalQuerySemantics,
    QueryState,
    associate_hierarchical_query,
)
from thayer_select_prompt_ablation_common import gaussian_prompt_numpy


FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
PHASE1 = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
CATALOG_PATH = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PHASE1 / "manifests/source_split_manifest.csv"
IMAGE_SIZE = 60
PIXEL_SCALE_ARCSEC = 0.2
PROMPT_SIGMA_PIXELS = 2.0
SEMANTICS = HierarchicalQuerySemantics()
SCHEMA_VERSION = "thayer-select-hierarchical-scenes-v1"
ARTIFACT_VERSION = "v2"
BASE_SEED = 2026071200
DATASETS = {
    "q_training": {"source_partition": "training", "state_counts": {"UNIQUE_VALID": 5000, "NULL": 5000, "AMBIGUOUS": 5000}},
    "q_validation": {"source_partition": "validation", "state_counts": {"UNIQUE_VALID": 668, "NULL": 666, "AMBIGUOUS": 666}},
    "r_training": {"source_partition": "training", "stratum_counts": {"natural": 7500, "low_snr": 3000, "high_overlap": 2250, "equal_flux_similar_size": 1125, "confusion_prone": 1125}},
    "r_validation": {"source_partition": "validation", "stratum_counts": {"natural": 2000}},
    "natural_calibration": {"source_partition": "calibration", "state_counts": {"UNIQUE_VALID": 4200, "NULL": 1200, "AMBIGUOUS": 600}},
    "stratified_calibration": {"source_partition": "calibration", "state_counts": {"UNIQUE_VALID": 1000, "NULL": 1000, "AMBIGUOUS": 1000}},
}


def artifact_stem(dataset: str) -> str:
    return f"{ARTIFACT_VERSION}_{dataset}"


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    frame.to_csv(path, index=False)


def dataset_count(spec: dict) -> int:
    values = spec.get("state_counts", spec.get("stratum_counts"))
    return int(sum(values.values()))


def draw_positions(rng: np.random.Generator, stratum: str, state: QueryState) -> np.ndarray:
    for _ in range(10_000):
        if state is QueryState.AMBIGUOUS:
            separation = rng.uniform(0.25, 1.15)
        elif stratum in ("high_overlap", "confusion_prone"):
            separation = rng.uniform(0.25, 0.95)
        else:
            separation = rng.uniform(0.65, 4.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        midpoint = rng.uniform(-1.4, 1.4, size=2)
        delta = 0.5 * separation * np.asarray([np.cos(angle), np.sin(angle)])
        positions = np.stack((midpoint - delta, midpoint + delta))
        if np.max(np.abs(positions)) < 2.8:
            return positions
    raise RuntimeError("Could not draw in-frame positions")


def source_size(catalog: Table, row: int) -> float:
    return max(float(catalog["a_b"][row]), float(catalog["a_d"][row]))


def choose_sources(pool: pd.DataFrame, catalog: Table, rng: np.random.Generator, stratum: str) -> tuple[pd.Series, pd.Series]:
    if stratum == "low_snr":
        magnitudes = np.asarray(catalog["r_ab"])[pool.catalog_row.to_numpy(dtype=int)]
        threshold = float(np.quantile(magnitudes, 0.75))
        candidates = pool.loc[magnitudes >= threshold]
        indices = rng.choice(candidates.index.to_numpy(), size=2, replace=False)
        return candidates.loc[indices[0]], candidates.loc[indices[1]]
    if stratum in ("equal_flux_similar_size", "confusion_prone"):
        for _ in range(10_000):
            first = pool.iloc[int(rng.integers(0, len(pool)))]
            first_row = int(first.catalog_row)
            sample_indices = rng.integers(0, len(pool), size=128)
            for index in sample_indices:
                second = pool.iloc[int(index)]
                second_row = int(second.catalog_row)
                if second.persistent_source_id == first.persistent_source_id:
                    continue
                magnitude_gap = abs(float(catalog["r_ab"][first_row]) - float(catalog["r_ab"][second_row]))
                size_one = source_size(catalog, first_row); size_two = source_size(catalog, second_row)
                size_gap = abs(math.log(max(size_one, 1e-6) / max(size_two, 1e-6)))
                if magnitude_gap <= 0.20 and size_gap <= 0.20:
                    return first, second
        raise RuntimeError(f"Could not select pair for {stratum}")
    indices = rng.choice(len(pool), size=2, replace=False)
    return pool.iloc[int(indices[0])], pool.iloc[int(indices[1])]


def prompt_for_state(xy: np.ndarray, state: QueryState, requested: int, rng: np.random.Generator, *, perturbed: bool) -> tuple[np.ndarray, object]:
    if state is QueryState.UNIQUE_VALID and not perturbed:
        prompt = xy[requested].copy()
    elif state is QueryState.UNIQUE_VALID:
        prompt = None
        for _ in range(10_000):
            radius = rng.uniform(0.75, SEMANTICS.maximum_perturbation_pixels)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            candidate = xy[requested] + radius * np.asarray([np.cos(angle), np.sin(angle)])
            if np.all(candidate >= 0.0) and np.all(candidate <= IMAGE_SIZE - 1):
                association = associate_hierarchical_query(xy, candidate, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
                if association.state is state and association.matched_index == requested:
                    prompt = candidate
                    break
        if prompt is None:
            raise RuntimeError("Could not construct perturbed unique prompt")
    elif state is QueryState.NULL:
        prompt = None
        for attempt in range(10_000):
            candidate = rng.uniform(0.0, IMAGE_SIZE - 1, size=2)
            if attempt % 3 == 0:
                axis = int(rng.integers(0, 2)); side = int(rng.integers(0, 2))
                candidate[axis] = rng.uniform(0.0, 1.0) if side == 0 else rng.uniform(IMAGE_SIZE - 2, IMAGE_SIZE - 1)
            association = associate_hierarchical_query(xy, candidate, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
            if association.state is state:
                prompt = candidate
                break
        if prompt is None:
            raise RuntimeError("Could not construct NULL prompt")
    else:
        midpoint = np.mean(xy, axis=0)
        delta = xy[1] - xy[0]
        perpendicular = np.asarray([-delta[1], delta[0]]) / max(float(np.linalg.norm(delta)), 1e-12)
        prompt = midpoint + rng.uniform(-0.2, 0.2) * perpendicular
    association = associate_hierarchical_query(xy, prompt, image_shape=(IMAGE_SIZE, IMAGE_SIZE), semantics=SEMANTICS)
    if association.state is not state or (state is QueryState.UNIQUE_VALID and association.matched_index != requested):
        raise RuntimeError(f"Prompt semantics mismatch: expected {state}, got {association}")
    return np.asarray(prompt, dtype=np.float64), association


def expanded_labels(spec: dict, rng: np.random.Generator) -> list[tuple[QueryState, str]]:
    values = []
    if "state_counts" in spec:
        for state, count in spec["state_counts"].items():
            values.extend((QueryState(state), "natural") for _ in range(count))
    else:
        for stratum, count in spec["stratum_counts"].items():
            values.extend((QueryState.UNIQUE_VALID, stratum) for _ in range(count))
    rng.shuffle(values)
    return values


def inverse_weight(dataset: str, state: QueryState) -> tuple[float, int]:
    if dataset in ("q_training", "q_validation"):
        natural = {QueryState.UNIQUE_VALID: 0.70, QueryState.NULL: 0.20, QueryState.AMBIGUOUS: 0.10}
        spec = DATASETS[dataset]["state_counts"]
        sample_fraction = spec[state.value] / sum(spec.values())
        return natural[state] / sample_fraction, 1
    if dataset == "stratified_calibration":
        return 0.0, 0
    return 1.0, 1


def create_store(path: Path, count: int) -> h5py.File:
    handle = h5py.File(path, "x")
    for name, shape, dtype in (
        ("blend", (count, 3, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("isolated", (count, 2, 3, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("xy", (count, 2, 2), "f8"),
        ("prompt_xy", (count, 2), "f8"),
        ("prompt", (count, 1, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("matched_index", (count,), "i1"),
    ):
        chunks = (1,) + shape[1:] if len(shape) > 1 else True
        handle.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compression="lzf" if len(shape) > 1 else None)
    handle.attrs["completed_count"] = 0
    handle.attrs["complete"] = False
    handle.attrs["schema_version"] = SCHEMA_VERSION
    return handle


def build_definitions(dataset: str, spec: dict, pool: pd.DataFrame, catalog: Table) -> pd.DataFrame:
    dataset_index = list(DATASETS).index(dataset)
    rng = np.random.default_rng(BASE_SEED + dataset_index * 100_000)
    records = []
    for index, (state, stratum) in enumerate(expanded_labels(spec, rng)):
        first, second = choose_sources(pool, catalog, rng, stratum)
        positions = draw_positions(rng, stratum, state)
        requested = int(rng.integers(0, 2))
        perturbed = bool(state is QueryState.UNIQUE_VALID and rng.random() < 0.30 and stratum == "natural")
        weight, applicable = inverse_weight(dataset, state)
        records.append({
            "scene_id": f"{ARTIFACT_VERSION}_{dataset}_{index:05d}", "dataset": dataset, "source_partition": spec["source_partition"],
            "dataset_index": index, "query_state": state.value, "prompt_subtype": "PERTURBED_VALID" if perturbed else state.value,
            "sampling_stratum": stratum, "inverse_sampling_weight": weight, "operational_weight_applicable": applicable,
            "requested_index_for_generation": requested,
            "source_a_row": int(first.catalog_row), "source_b_row": int(second.catalog_row),
            "source_a_id": first.persistent_source_id, "source_b_id": second.persistent_source_id,
            "source_a_group": first.duplicate_group_id, "source_b_group": second.duplicate_group_id,
            "source_a_x_arcsec": positions[0, 0], "source_a_y_arcsec": positions[0, 1],
            "source_b_x_arcsec": positions[1, 0], "source_b_y_arcsec": positions[1, 1],
            "scene_seed": BASE_SEED + dataset_index * 100_000 + index,
            "noise_seed": BASE_SEED + 10_000_000 + dataset_index * 100_000 + index,
            "prompt_seed": BASE_SEED + 20_000_000 + dataset_index * 100_000 + index,
            "schema_version": SCHEMA_VERSION,
        })
    return pd.DataFrame(records)


def render_dataset(run: Path, dataset: str, definitions: pd.DataFrame, catalog, table: Table) -> pd.DataFrame:
    h5_path = run / f"manifests/{artifact_stem(dataset)}_scenes.h5"
    survey = validated_lsst_survey()
    psf = {band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in ("g", "r", "z")}
    rows = []
    with create_store(h5_path, len(definitions)) as handle:
        for index, row in definitions.iterrows():
            positions = ((row.source_a_x_arcsec, row.source_a_y_arcsec), (row.source_b_x_arcsec, row.source_b_y_arcsec))
            scene_spec = SceneSpec(row.scene_id, (int(row.source_a_row), int(row.source_b_row)), positions, int(row.scene_seed), int(row.scene_seed), int(row.noise_seed))
            rendered = render_fixed_scene(catalog, scene_spec, add_noise="all")
            blend = np.asarray(rendered.blend, dtype=np.float32)
            isolated = np.asarray(rendered.isolated, dtype=np.float32)
            xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered.catalog], dtype=np.float64)
            prompt_rng = np.random.default_rng(int(row.prompt_seed))
            prompt_xy, association = prompt_for_state(xy, QueryState(row.query_state), int(row.requested_index_for_generation), prompt_rng, perturbed=row.prompt_subtype == "PERTURBED_VALID")
            prompt = gaussian_prompt_numpy(float(prompt_xy[0]), float(prompt_xy[1]), height=IMAGE_SIZE, width=IMAGE_SIZE, sigma_pixels=PROMPT_SIGMA_PIXELS)[None]
            matched = -1 if association.matched_index is None else int(association.matched_index)
            handle["blend"][index] = blend; handle["isolated"][index] = isolated; handle["xy"][index] = xy
            handle["prompt_xy"][index] = prompt_xy; handle["prompt"][index] = prompt; handle["matched_index"][index] = matched
            flux = isolated.sum(axis=(-2, -1), dtype=np.float64)
            total_flux = flux.sum(axis=1)
            sizes = [source_size(table, int(row.source_a_row)), source_size(table, int(row.source_b_row))]
            minors = [max(float(table["b_b"][int(source_row)]), float(table["b_d"][int(source_row)])) for source_row in (row.source_a_row, row.source_b_row)]
            ellipticity = [(major - minor) / max(major + minor, 1e-30) for major, minor in zip(sizes, minors)]
            colors = [np.asarray([float(table["g_ab"][int(source_row)]) - float(table["r_ab"][int(source_row)]), float(table["r_ab"][int(source_row)]) - float(table["z_ab"][int(source_row)])]) for source_row in (row.source_a_row, row.source_b_row)]
            if matched >= 0:
                alternate = 1 - matched
                target_peak = np.max(np.abs(isolated[matched]), axis=0)
                core = target_peak > 0.5 * float(target_peak.max())
                core_obstruction = float(np.maximum(isolated[alternate, :, core], 0).sum() / max(np.maximum(isolated[matched, :, core], 0).sum(), 1e-30))
                flux_ratio = float(total_flux[matched] / max(total_flux[alternate], 1e-30))
                size_ratio = float(sizes[matched] / max(sizes[alternate], 1e-30))
                snr_proxy = float(total_flux[matched] / max(np.sqrt(np.sum(np.abs(blend))), 1e-30))
                matched_id = row.source_a_id if matched == 0 else row.source_b_id
                matched_group = row.source_a_group if matched == 0 else row.source_b_group
            else:
                core_obstruction = flux_ratio = size_ratio = snr_proxy = math.nan
                matched_id = matched_group = ""
            rows.append({
                **row.to_dict(), "source_a_x_pixel": xy[0, 0], "source_a_y_pixel": xy[0, 1], "source_b_x_pixel": xy[1, 0], "source_b_y_pixel": xy[1, 1],
                "prompt_x_pixel": prompt_xy[0], "prompt_y_pixel": prompt_xy[1], "matched_source_index": matched,
                "matched_source_id": matched_id, "matched_source_group": matched_group,
                "nearest_distance_pixels": association.nearest_distance_pixels, "second_distance_pixels": association.second_distance_pixels,
                "candidate_source_count": association.candidate_count, "matching_radius_pixels": SEMANTICS.matching_radius_pixels,
                "matching_radius_psf_units": SEMANTICS.matching_radius_psf, "ambiguity_margin_pixels": SEMANTICS.ambiguity_margin_pixels,
                "separation_pixels": float(np.linalg.norm(xy[0] - xy[1])), "separation_psf_units": float(np.linalg.norm(xy[0] - xy[1])) / SEMANTICS.psf_fwhm_pixels,
                "source_a_magnitude_r": float(table["r_ab"][int(row.source_a_row)]), "source_b_magnitude_r": float(table["r_ab"][int(row.source_b_row)]),
                "source_a_size_arcsec": sizes[0], "source_b_size_arcsec": sizes[1], "source_a_ellipticity": ellipticity[0], "source_b_ellipticity": ellipticity[1],
                "flux_ratio": flux_ratio, "color_similarity_distance": float(np.linalg.norm(colors[0] - colors[1])), "size_ratio": size_ratio,
                "core_obstruction": core_obstruction, "snr_proxy": snr_proxy, "source_count": 2,
                "psf_fwhm_arcsec": json.dumps(psf, sort_keys=True), "noise_settings": "BTK LSST add_noise=all; explicit noise seed",
                "isolated_source_a_sha256": sha256_array(isolated[0]), "isolated_source_b_sha256": sha256_array(isolated[1]),
                "blend_sha256": sha256_array(blend), "prompt_sha256": sha256_array(prompt), "hdf5_path": relative(h5_path),
            })
            handle.attrs["completed_count"] = index + 1
            if (index + 1) % 100 == 0 or index + 1 == len(definitions):
                handle.flush(); print(f"{dataset}: {index + 1}/{len(definitions)}", flush=True)
        handle.attrs["complete"] = True
    return pd.DataFrame(rows)


def replay_audit(run: Path, dataset: str, manifest: pd.DataFrame, catalog) -> list[dict]:
    indices = sorted(set([0, len(manifest) // 2, len(manifest) - 1]))
    rows = []
    with h5py.File(run / f"manifests/{artifact_stem(dataset)}_scenes.h5", "r") as handle:
        for index in indices:
            row = manifest.iloc[index]
            positions = ((row.source_a_x_arcsec, row.source_a_y_arcsec), (row.source_b_x_arcsec, row.source_b_y_arcsec))
            spec = SceneSpec(row.scene_id, (int(row.source_a_row), int(row.source_b_row)), positions, int(row.scene_seed), int(row.scene_seed), int(row.noise_seed))
            replay = render_fixed_scene(catalog, spec, add_noise="all")
            status = np.array_equal(np.asarray(replay.blend, dtype=np.float32), np.asarray(handle["blend"][index])) and np.array_equal(np.asarray(replay.isolated, dtype=np.float32), np.asarray(handle["isolated"][index]))
            rows.append({"dataset": dataset, "scene_id": row.scene_id, "index": index, "status": "PASS" if status else "FAIL"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != (REPO / "outputs/runs").resolve() or not run.name.startswith("thayer_select_hierarchical_safety_"):
        raise RuntimeError("Unexpected run directory")
    audit = json.loads((run / "logs/partition_drift_complete.json").read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("Partition drift gate did not pass")
    if (run / "logs/data_preparation_complete.json").exists():
        raise FileExistsError("Data preparation already completed")
    for doc in ("hierarchical_query_semantics.md", "hierarchical_recoverability_contract.md", "hierarchical_safety_policy.md", "hierarchical_safety_experiment.md"):
        if not (REPO / "docs" / doc).is_file():
            raise RuntimeError(f"Missing preregistration: {doc}")

    split = pd.read_csv(SOURCE_SPLIT)
    forbidden = set(split.loc[split.partition.isin(("development_test", "sealed_lockbox", "engineering_excluded")), "duplicate_group_id"])
    table = Table.read(CATALOG_PATH, format="fits")
    catalog, observed_catalog_hash = load_catsim_catalog(CATALOG_PATH)
    if observed_catalog_hash != sha256_file(CATALOG_PATH):
        raise RuntimeError("Loaded catalog hash mismatch")
    replay_rows = []
    inventories = []
    started = time.time()
    for dataset, spec in DATASETS.items():
        pool = split[(split.partition == spec["source_partition"]) & (split.engineering_excluded == 0)].copy()
        if set(pool.duplicate_group_id) & forbidden:
            raise RuntimeError(f"Forbidden group in {dataset} pool")
        definitions = build_definitions(dataset, spec, pool, table)
        write_csv_fresh(run / f"manifests/{artifact_stem(dataset)}_scene_definitions.csv", definitions)
        manifest = render_dataset(run, dataset, definitions, catalog, table)
        write_csv_fresh(run / f"manifests/{artifact_stem(dataset)}_scene_manifest.csv", manifest)
        replay_rows.extend(replay_audit(run, dataset, manifest, catalog))
        inventories.append({
            "dataset": dataset, "source_partition": spec["source_partition"], "scenes": len(manifest),
            "unique_scene_ids": manifest.scene_id.nunique(), "unique_source_ids": len(set(manifest.source_a_id) | set(manifest.source_b_id)),
            "query_state_counts": json.dumps(manifest.query_state.value_counts().sort_index().to_dict(), sort_keys=True),
            "sampling_stratum_counts": json.dumps(manifest.sampling_stratum.value_counts().sort_index().to_dict(), sort_keys=True),
            "artifact_version": ARTIFACT_VERSION,
            "manifest_sha256": sha256_file(run / f"manifests/{artifact_stem(dataset)}_scene_manifest.csv"),
            "hdf5_sha256": sha256_file(run / f"manifests/{artifact_stem(dataset)}_scenes.h5"),
        })
    replay = pd.DataFrame(replay_rows)
    write_csv_fresh(run / "tables/training_calibration_replay_audit.csv", replay)
    write_csv_fresh(run / "tables/fresh_dataset_inventory.csv", pd.DataFrame(inventories))
    if not (replay.status == "PASS").all():
        raise RuntimeError("Deterministic replay audit failed")
    write_json_fresh(run / "logs/data_preparation_complete.json", {
        "status": "PASS", "artifact_version": ARTIFACT_VERSION, "datasets": {name: dataset_count(spec) for name, spec in DATASETS.items()},
        "total_scenes": sum(dataset_count(spec) for spec in DATASETS.values()), "runtime_seconds": time.time() - started,
        "source_split_sha256": sha256_file(SOURCE_SPLIT), "development_sources_used": False,
        "lockbox_sources_or_scenes_used": False, "completed_at_unix": time.time(),
    })
    print(relative(run))


if __name__ == "__main__":
    main()
