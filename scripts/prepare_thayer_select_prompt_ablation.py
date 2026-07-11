#!/usr/bin/env python3
"""Freeze and render the group-safe development data for prompt ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from btk_scene import SceneSpec, file_sha256, gaussian_prompt, load_catsim_catalog, render_fixed_scene, validated_lsst_survey
from thayer_select_prompt_ablation_common import (
    BANDS,
    CATALOG,
    FOUNDATION,
    IMAGE_SIZE,
    LOCKBOX_PARTITION,
    NOISE_SEED_BASE,
    PARTITION_COUNTS,
    PIXEL_SCALE_ARCSEC,
    SCENE_SEED_BASE,
    gaussian_prompt_numpy,
    read_csv,
    sha256_array,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)

FOUNDATION_SPLIT = FOUNDATION / "manifests/btk_engineering_source_groups.csv"
REPLAY_METADATA = FOUNDATION / "data/btk_engineering_pilot/double_002.json"
REPLAY_ARRAYS = FOUNDATION / "data/btk_engineering_pilot/double_002.npz"
EXPECTED_NOISE_SEED = 2026072301
PARTITION_ORDER = ["training", "validation", "calibration", "development_test", LOCKBOX_PARTITION]


def command_output(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return f"command: {' '.join(arguments)}\nreturncode: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def prepare_provenance(run_dir: Path, replay_log: Path) -> None:
    if not replay_log.is_file():
        raise FileNotFoundError(replay_log)
    replay = json.loads(replay_log.read_text())
    if replay.get("status") != "PASS" or not all(replay.get("checks", {}).values()):
        raise RuntimeError("Explicit-seed replay gate did not pass")

    metadata = json.loads(REPLAY_METADATA.read_text())
    with np.load(REPLAY_ARRAYS, allow_pickle=False) as arrays:
        noiseless = np.asarray(arrays["blend_noiseless"])
        isolated = np.asarray(arrays["isolated_sources"])
        noisy = np.asarray(arrays["blend_noisy"])
        noise = np.asarray(arrays["noise_realization"])
        prompts = np.asarray(arrays["prompts"])
    regenerated_prompts = np.asarray(
        [gaussian_prompt(noiseless.shape[-2:], float(row["x_peak"]), float(row["y_peak"])) for row in metadata["sources"]]
    )
    foundation_rows = [row for row in read_csv(FOUNDATION / "tables/btk_scene_manifest.csv") if row["scene_id"] == "double_002"]
    replay_checks = {
        "utility_replay_pass": True,
        "source_ids_match": all(
            str(source["engineering_source_id"]).startswith(
                f"catsim:{metadata['catalog_sha256']}:row:{int(source['catalog_row'])}:"
            )
            for source in metadata["sources"]
        ),
        "source_coordinates_match": all(
            np.array_equal(np.asarray([source["ra"], source["dec"]]), np.asarray(requested))
            for source, requested in zip(metadata["sources"], metadata["requested_positions_arcsec"])
        ),
        "isolated_additivity_exact": np.array_equal(isolated.sum(axis=0), noiseless),
        "noise_reconstruction_exact": np.array_equal(noiseless + noise, noisy),
        "prompt_maps_exact": np.array_equal(prompts, regenerated_prompts),
        "explicit_noise_seed_match": int(metadata["noise_seed"]) == EXPECTED_NOISE_SEED,
        "catalog_hash_match": file_sha256(REPO / metadata["catalog_path"]) == metadata["catalog_sha256"],
        "arrays_hash_match": file_sha256(REPLAY_ARRAYS) == metadata["arrays_sha256"],
        "metadata_hash_match": bool(foundation_rows) and all(
            file_sha256(REPLAY_METADATA) == row["metadata_sha256"] for row in foundation_rows
        ),
    }
    replay_result = {
        "status": "PASS" if all(replay_checks.values()) else "FAIL",
        "checks": replay_checks,
        "utility_log": str(replay_log.relative_to(REPO)),
        "expected_noise_seed": EXPECTED_NOISE_SEED,
        "strict_absolute_tolerance": 1e-10,
        "strict_relative_tolerance": 1e-12,
    }
    replay_verification_path = run_dir / "logs/explicit_seed_replay_verification.json"
    if replay_verification_path.exists():
        if json.loads(replay_verification_path.read_text()) != replay_result:
            raise RuntimeError("Existing replay verification differs from recomputation")
    else:
        write_json_fresh(replay_verification_path, replay_result)
    if replay_result["status"] != "PASS":
        raise RuntimeError("Independent explicit-seed replay verification failed")

    git_initial_path = run_dir / "logs/git_initial.txt"
    if not git_initial_path.exists():
        write_text_fresh(git_initial_path, "".join([
            command_output(["git", "branch", "--show-current"]), "\n",
            command_output(["git", "rev-parse", "HEAD"]), "\n",
            command_output(["git", "status", "--short", "--branch"]), "\n",
        ]))
    import astropy
    import btk
    import galsim
    import matplotlib
    import scipy
    import skimage
    import surveycodex
    import torch
    package_record = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "h5py": h5py.__version__,
        "astropy": astropy.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "matplotlib": matplotlib.__version__,
        "btk": btk.__version__,
        "galsim": galsim.__version__,
        "surveycodex": version("surveycodex"),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    environment_path = run_dir / "manifests/environment.json"
    if environment_path.exists():
        if json.loads(environment_path.read_text()) != package_record:
            raise RuntimeError("Existing environment record differs from current environment")
    else:
        write_json_fresh(environment_path, package_record)
    if not package_record["mps_available"]:
        raise RuntimeError("MPS unavailable")

    historical = read_csv(FOUNDATION / "tables/preexisting_checkpoint_hashes_before.csv")
    current = []
    for row in historical:
        path = REPO / row["relative_path"]
        observed = sha256_file(path)
        current.append({**row, "observed_sha256": observed, "status": "PASS" if observed == row["sha256"] else "FAIL"})
    historical_path = run_dir / "tables/historical_checkpoint_hashes_before.csv"
    if not historical_path.exists():
        write_csv_fresh(historical_path, current)
    if len(current) != 18 or any(row["status"] != "PASS" for row in current):
        raise RuntimeError("Historical checkpoint integrity failed before campaign")

    foundation_inputs = [
        CATALOG,
        FOUNDATION_SPLIT,
        FOUNDATION / "tables/btk_split_summary.csv",
        FOUNDATION / "tables/btk_scene_unit_tests.csv",
        FOUNDATION / "tables/btk_fresh_process_replay.csv",
        REPLAY_METADATA,
        REPLAY_ARRAYS,
        FOUNDATION / "logs/final_explicit_seed_replay.json",
        FOUNDATION / "manifests/campaign_file_hashes.csv",
    ]
    foundation_path = run_dir / "tables/btk_foundation_inputs.csv"
    if not foundation_path.exists():
        write_csv_fresh(
            foundation_path,
            [{"relative_path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in foundation_inputs],
        )
    simulator_paths = [REPO / "src/btk_scene.py", REPO / "scripts/run_btk_engineering_pilot.py", Path(__file__).resolve(), REPO / "scripts/thayer_select_prompt_ablation_common.py"]
    simulator_hash_path = run_dir / "tables/simulator_code_hashes_initial.csv"
    if not simulator_hash_path.exists():
        write_csv_fresh(
            simulator_hash_path,
            [{"relative_path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in simulator_paths],
        )


def freeze_and_validate_split(run_dir: Path) -> tuple[list[dict], Table]:
    destination = run_dir / "manifests/source_split_manifest.csv"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(FOUNDATION_SPLIT, destination)
    split_rows = read_csv(destination)
    catalog = Table.read(CATALOG)
    if len(split_rows) != len(catalog):
        raise RuntimeError("Split row count does not match catalog")

    ids: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    engineering_count = 0
    for row in split_rows:
        partition = row["partition"]
        ids[row["persistent_source_id"]].add(partition)
        groups[row["duplicate_group_id"]].add(partition)
        counts[partition] += 1
        engineering_count += int(row["engineering_excluded"])
    leakage_ids = [key for key, value in ids.items() if len(value) != 1]
    leakage_groups = [key for key, value in groups.items() if len(value) != 1]
    expected = set(PARTITION_ORDER + ["engineering_excluded"])
    checks = {
        "row_count": len(split_rows),
        "unique_source_count": len(ids),
        "source_cross_partition_count": len(leakage_ids),
        "group_cross_partition_count": len(leakage_groups),
        "engineering_excluded_count": engineering_count,
        "partitions_present": sorted(counts),
        "expected_partitions_present": expected.issubset(set(counts)),
        "status": "PASS" if not leakage_ids and not leakage_groups and expected.issubset(set(counts)) else "FAIL",
    }
    write_json_fresh(run_dir / "diagnostics/group_integrity_report.json", checks)
    write_csv_fresh(
        run_dir / "tables/cross_partition_source_check.csv",
        [{"check": "persistent_source_identity", "crossing_count": len(leakage_ids), "status": "PASS" if not leakage_ids else "FAIL"},
         {"check": "duplicate_group_identity", "crossing_count": len(leakage_groups), "status": "PASS" if not leakage_groups else "FAIL"}],
    )
    if checks["status"] != "PASS":
        raise RuntimeError("Source/group leakage detected")

    rmag = np.asarray(catalog["r_ab"], dtype=float)
    size = np.maximum(np.asarray(catalog["a_b"], dtype=float), np.asarray(catalog["a_d"], dtype=float))
    minor = np.maximum(np.asarray(catalog["b_b"], dtype=float), np.asarray(catalog["b_d"], dtype=float))
    ellipticity = np.divide(size - minor, size + minor, out=np.zeros_like(size), where=(size + minor) > 0)
    morphology = np.asarray(catalog["fluxnorm_bulge"], dtype=float) / np.maximum(
        np.asarray(catalog["fluxnorm_bulge"], dtype=float) + np.asarray(catalog["fluxnorm_disk"], dtype=float), 1e-30
    )
    assignments = np.asarray([row["partition"] for row in split_rows])
    development_mask = np.isin(assignments, PARTITION_ORDER)
    balance_rows = []
    for partition in PARTITION_ORDER:
        mask = assignments == partition
        record = {"partition": partition, "source_count": int(mask.sum()), "fraction": float(mask.sum() / development_mask.sum())}
        for name, values in (("r_ab", rmag), ("size_arcsec", size), ("ellipticity", ellipticity), ("bulge_fraction", morphology)):
            record[f"{name}_mean"] = float(np.mean(values[mask]))
            record[f"{name}_std"] = float(np.std(values[mask]))
            record[f"{name}_ks_vs_development"] = float(ks_2samp(values[mask], values[development_mask]).statistic)
        balance_rows.append(record)
    write_csv_fresh(run_dir / "tables/source_split_balance_summary.csv", balance_rows)
    write_text_fresh(
        run_dir / "reports/lockbox_policy.md",
        "# Sealed CatSim lockbox policy\n\n"
        "The sealed-lockbox source assignment is metadata-only. This campaign may hash and count those assignments, "
        "but it must not sample them into a scene, render them, open any lockbox scene array, plot them, evaluate them, "
        "or use them for model selection. No lockbox scene manifest or array file is created. Only a later explicitly "
        "authorized campaign may unseal it.\n",
    )
    return split_rows, catalog


def draw_positions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(10_000):
        positions = rng.uniform(-2.1, 2.1, size=(2, 2))
        separation = float(np.linalg.norm(positions[0] - positions[1]))
        if 0.8 <= separation <= 3.2:
            return positions[0], positions[1]
    raise RuntimeError("Could not draw valid positions")


def freeze_scene_definitions(run_dir: Path, split_rows: list[dict], catalog: Table) -> None:
    pools: dict[str, np.ndarray] = {}
    row_to_split = {int(row["catalog_row"]): row for row in split_rows}
    for partition in PARTITION_COUNTS:
        pools[partition] = np.asarray(
            [int(row["catalog_row"]) for row in split_rows if row["partition"] == partition and row["engineering_excluded"] == "0"],
            dtype=np.int64,
        )
        if len(pools[partition]) < 2:
            raise RuntimeError(f"Insufficient source pool for {partition}")

    records = []
    global_index = 0
    for partition, count in PARTITION_COUNTS.items():
        for local_index in range(count):
            scene_seed = SCENE_SEED_BASE + global_index
            noise_seed = NOISE_SEED_BASE + global_index
            rng = np.random.default_rng(scene_seed)
            selected = rng.choice(pools[partition], size=2, replace=False)
            target_index = int(rng.integers(0, 2))
            pos_a, pos_b = draw_positions(rng)
            random_positions = np.stack([pos_a, pos_b])
            centered_positions = random_positions - random_positions[target_index]
            target_row = int(selected[target_index])
            alternate_index = 1 - target_index
            alternate_row = int(selected[alternate_index])
            source_entries = [row_to_split[int(value)] for value in selected]
            separation = float(np.linalg.norm(random_positions[0] - random_positions[1]) / PIXEL_SCALE_ARCSEC)
            size_values = [max(float(catalog["a_b"][row]), float(catalog["a_d"][row])) for row in selected]
            minor_values = [max(float(catalog["b_b"][row]), float(catalog["b_d"][row])) for row in selected]
            ellipticity = [(major - minor) / max(major + minor, 1e-30) for major, minor in zip(size_values, minor_values)]
            records.append({
                "scene_id": f"{partition}_{local_index:05d}", "partition": partition, "partition_index": local_index,
                "scene_seed": scene_seed, "noise_seed": noise_seed,
                "source_a_row": int(selected[0]), "source_b_row": int(selected[1]),
                "source_a_id": source_entries[0]["persistent_source_id"], "source_b_id": source_entries[1]["persistent_source_id"],
                "source_a_group": source_entries[0]["duplicate_group_id"], "source_b_group": source_entries[1]["duplicate_group_id"],
                "target_index": target_index, "target_source_id": row_to_split[target_row]["persistent_source_id"],
                "target_source_group": row_to_split[target_row]["duplicate_group_id"],
                "alternate_source_id": row_to_split[alternate_row]["persistent_source_id"],
                "alternate_source_group": row_to_split[alternate_row]["duplicate_group_id"],
                "random_a_x_arcsec": random_positions[0, 0], "random_a_y_arcsec": random_positions[0, 1],
                "random_b_x_arcsec": random_positions[1, 0], "random_b_y_arcsec": random_positions[1, 1],
                "centered_a_x_arcsec": centered_positions[0, 0], "centered_a_y_arcsec": centered_positions[0, 1],
                "centered_b_x_arcsec": centered_positions[1, 0], "centered_b_y_arcsec": centered_positions[1, 1],
                "separation_pixels": separation,
                "source_a_g_ab": float(catalog["g_ab"][selected[0]]), "source_a_r_ab": float(catalog["r_ab"][selected[0]]), "source_a_z_ab": float(catalog["z_ab"][selected[0]]),
                "source_b_g_ab": float(catalog["g_ab"][selected[1]]), "source_b_r_ab": float(catalog["r_ab"][selected[1]]), "source_b_z_ab": float(catalog["z_ab"][selected[1]]),
                "source_a_size_arcsec": size_values[0], "source_b_size_arcsec": size_values[1],
                "source_a_ellipticity": ellipticity[0], "source_b_ellipticity": ellipticity[1],
                "simulator_version": "BTK 1.0.9 / GalSim 2.8.4 / surveycodex 1.2.0",
            })
            global_index += 1
    write_csv_fresh(run_dir / "manifests/development_scene_definitions.csv", records)
    summary = {
        "counts": PARTITION_COUNTS,
        "total_development_scenes": sum(PARTITION_COUNTS.values()),
        "lockbox_scene_count": 0,
        "source_count_per_scene": 2,
        "bands": list(BANDS),
        "target_selection": "seeded uniform A/B",
        "random_position_rule": "exchangeable iid uniform square draws, rejected unless separation is 0.8-3.2 arcsec",
        "centered_position_rule": "translate the aligned randomized pair so the requested source is exactly centered; separation is unchanged",
        "noise_rule": "one explicit BTK add_noise='all' realization per condition variant and scene seed",
    }
    write_json_fresh(run_dir / "manifests/scene_freeze_summary.json", summary)


def h5_create(path: Path, count: int) -> h5py.File:
    if path.exists():
        return h5py.File(path, "r+")
    handle = h5py.File(path, "x")
    shapes = {
        "random_blend": (count, 3, IMAGE_SIZE, IMAGE_SIZE),
        "random_isolated": (count, 2, 3, IMAGE_SIZE, IMAGE_SIZE),
        "centered_blend": (count, 3, IMAGE_SIZE, IMAGE_SIZE),
        "centered_isolated": (count, 2, 3, IMAGE_SIZE, IMAGE_SIZE),
    }
    for name, shape in shapes.items():
        handle.create_dataset(name, shape=shape, dtype="f4", chunks=(1,) + shape[1:], compression="lzf")
    handle.create_dataset("random_xy", shape=(count, 2, 2), dtype="f8")
    handle.create_dataset("centered_xy", shape=(count, 2, 2), dtype="f8")
    handle.create_dataset("source_flux", shape=(count, 2, 3), dtype="f8")
    for name in ("random_blend_hash", "centered_blend_hash", "random_isolated_a_hash", "random_isolated_b_hash", "centered_isolated_a_hash", "centered_isolated_b_hash", "prompt_a_hash", "prompt_b_hash"):
        handle.create_dataset(name, shape=(count,), dtype="S64")
    handle.attrs["completed_count"] = 0
    handle.attrs["complete"] = False
    handle.flush()
    return handle


def render_partition(run_dir: Path, partition: str, definitions: list[dict], catalog) -> None:
    path = run_dir / f"manifests/{partition}_scenes.h5"
    with h5_create(path, len(definitions)) as handle:
        start = int(handle.attrs["completed_count"])
        if bool(handle.attrs["complete"]):
            if start != len(definitions):
                raise RuntimeError(f"Corrupt completed HDF5 for {partition}")
            print(f"{partition}: already complete", flush=True)
            return
        for local_index in range(start, len(definitions)):
            row = definitions[local_index]
            catalog_rows = (int(row["source_a_row"]), int(row["source_b_row"]))
            random_positions = ((float(row["random_a_x_arcsec"]), float(row["random_a_y_arcsec"])), (float(row["random_b_x_arcsec"]), float(row["random_b_y_arcsec"])))
            centered_positions = ((float(row["centered_a_x_arcsec"]), float(row["centered_a_y_arcsec"])), (float(row["centered_b_x_arcsec"]), float(row["centered_b_y_arcsec"])))
            common = dict(scene_id=row["scene_id"], catalog_rows=catalog_rows, source_selection_seed=int(row["scene_seed"]), position_seed=int(row["scene_seed"]), noise_seed=int(row["noise_seed"]))
            random_render = render_fixed_scene(catalog, SceneSpec(positions_arcsec=random_positions, **common), add_noise="all")
            centered_render = render_fixed_scene(catalog, SceneSpec(positions_arcsec=centered_positions, **common), add_noise="all")
            random_blend = np.asarray(random_render.blend, dtype=np.float32)
            random_isolated = np.asarray(random_render.isolated, dtype=np.float32)
            centered_blend = np.asarray(centered_render.blend, dtype=np.float32)
            centered_isolated = np.asarray(centered_render.isolated, dtype=np.float32)
            random_xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in random_render.catalog], dtype=np.float64)
            centered_xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in centered_render.catalog], dtype=np.float64)
            if list(np.asarray(random_render.catalog["catalog_row"], dtype=int)) != list(catalog_rows):
                raise RuntimeError("Random render source mismatch")
            if list(np.asarray(centered_render.catalog["catalog_row"], dtype=int)) != list(catalog_rows):
                raise RuntimeError("Centered render source mismatch")
            handle["random_blend"][local_index] = random_blend
            handle["random_isolated"][local_index] = random_isolated
            handle["centered_blend"][local_index] = centered_blend
            handle["centered_isolated"][local_index] = centered_isolated
            handle["random_xy"][local_index] = random_xy
            handle["centered_xy"][local_index] = centered_xy
            handle["source_flux"][local_index] = random_isolated.sum(axis=(-2, -1), dtype=np.float64)
            prompt_a = gaussian_prompt_numpy(*random_xy[0])
            prompt_b = gaussian_prompt_numpy(*random_xy[1])
            hashes = {
                "random_blend_hash": sha256_array(random_blend), "centered_blend_hash": sha256_array(centered_blend),
                "random_isolated_a_hash": sha256_array(random_isolated[0]), "random_isolated_b_hash": sha256_array(random_isolated[1]),
                "centered_isolated_a_hash": sha256_array(centered_isolated[0]), "centered_isolated_b_hash": sha256_array(centered_isolated[1]),
                "prompt_a_hash": sha256_array(prompt_a), "prompt_b_hash": sha256_array(prompt_b),
            }
            for name, value in hashes.items():
                handle[name][local_index] = value.encode()
            handle.attrs["completed_count"] = local_index + 1
            handle.flush()
            if (local_index + 1) % 100 == 0 or local_index + 1 == len(definitions):
                print(f"{partition}: rendered {local_index + 1}/{len(definitions)}", flush=True)
        survey = validated_lsst_survey()
        psf_fwhm = {band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in BANDS}
        handle.attrs["psf_fwhm_json"] = json.dumps(psf_fwhm, sort_keys=True)
        handle.attrs["complete"] = True
        handle.flush()


def export_render_manifest(run_dir: Path, definitions: list[dict]) -> None:
    output = run_dir / "manifests/rendered_scene_manifest.csv"
    if output.exists():
        return
    by_partition: dict[str, list[dict]] = defaultdict(list)
    for row in definitions:
        by_partition[row["partition"]].append(row)
    records = []
    for partition, rows in by_partition.items():
        with h5py.File(run_dir / f"manifests/{partition}_scenes.h5", "r") as handle:
            if not bool(handle.attrs["complete"]) or int(handle.attrs["completed_count"]) != len(rows):
                raise RuntimeError(f"Incomplete data store for {partition}")
            psf_fwhm = json.loads(handle.attrs["psf_fwhm_json"])
            mean_fwhm_pixels = float(np.mean(list(psf_fwhm.values())) / PIXEL_SCALE_ARCSEC)
            for index, row in enumerate(rows):
                target = int(row["target_index"]); alternate = 1 - target
                flux = np.asarray(handle["source_flux"][index])
                target_flux = float(np.sum(flux[target])); alternate_flux = float(np.sum(flux[alternate]))
                target_size = float(row[f"source_{'a' if target == 0 else 'b'}_size_arcsec"])
                alternate_size = float(row[f"source_{'a' if alternate == 0 else 'b'}_size_arcsec"])
                record = dict(row)
                record.update({
                    "target_x_pixel": float(handle["random_xy"][index, target, 0]), "target_y_pixel": float(handle["random_xy"][index, target, 1]),
                    "alternate_x_pixel": float(handle["random_xy"][index, alternate, 0]), "alternate_y_pixel": float(handle["random_xy"][index, alternate, 1]),
                    "centered_target_x_pixel": float(handle["centered_xy"][index, target, 0]), "centered_target_y_pixel": float(handle["centered_xy"][index, target, 1]),
                    "separation_psf_units": float(row["separation_pixels"]) / mean_fwhm_pixels,
                    "target_flux_grz": target_flux, "alternate_flux_grz": alternate_flux,
                    "flux_ratio_target_to_alternate": target_flux / max(alternate_flux, 1e-30),
                    "size_ratio_target_to_alternate": target_size / max(alternate_size, 1e-30),
                    "psf_fwhm_arcsec": json.dumps(psf_fwhm, sort_keys=True),
                    "random_blend_sha256": handle["random_blend_hash"][index].decode(), "centered_blend_sha256": handle["centered_blend_hash"][index].decode(),
                    "random_isolated_a_sha256": handle["random_isolated_a_hash"][index].decode(), "random_isolated_b_sha256": handle["random_isolated_b_hash"][index].decode(),
                    "centered_isolated_a_sha256": handle["centered_isolated_a_hash"][index].decode(), "centered_isolated_b_sha256": handle["centered_isolated_b_hash"][index].decode(),
                    "prompt_a_sha256": handle["prompt_a_hash"][index].decode(), "prompt_b_sha256": handle["prompt_b_hash"][index].decode(),
                    "hdf5_path": str((run_dir / f"manifests/{partition}_scenes.h5").relative_to(REPO)),
                })
                records.append(record)
    write_csv_fresh(output, records)


def replay_stratified(run_dir: Path, definitions: list[dict], catalog) -> None:
    output = run_dir / "tables/scene_replay_checks.csv"
    if output.exists():
        return
    by_partition: dict[str, list[dict]] = defaultdict(list)
    for row in definitions:
        by_partition[row["partition"]].append(row)
    results = []
    for partition, rows in by_partition.items():
        separations = np.asarray([float(row["separation_pixels"]) for row in rows])
        chosen = sorted(set(int(value) for value in np.quantile(np.argsort(separations), [0.05, 0.35, 0.65, 0.95])))
        with h5py.File(run_dir / f"manifests/{partition}_scenes.h5", "r") as handle:
            for index in chosen:
                row = rows[index]
                catalog_rows = (int(row["source_a_row"]), int(row["source_b_row"]))
                for variant in ("random", "centered"):
                    positions = tuple((float(row[f"{variant}_{role}_x_arcsec"]), float(row[f"{variant}_{role}_y_arcsec"])) for role in ("a", "b"))
                    spec = SceneSpec(row["scene_id"], catalog_rows, positions, int(row["scene_seed"]), int(row["scene_seed"]), int(row["noise_seed"]))
                    rerender = render_fixed_scene(catalog, spec, add_noise="all")
                    observed_blend = np.asarray(rerender.blend, dtype=np.float32)
                    observed_isolated = np.asarray(rerender.isolated, dtype=np.float32)
                    blend_exact = np.array_equal(observed_blend, np.asarray(handle[f"{variant}_blend"][index]))
                    isolated_exact = np.array_equal(observed_isolated, np.asarray(handle[f"{variant}_isolated"][index]))
                    xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rerender.catalog])
                    xy_exact = np.array_equal(xy, np.asarray(handle[f"{variant}_xy"][index]))
                    prompt_exact = True
                    if variant == "random":
                        for source_index, name in enumerate(("prompt_a_hash", "prompt_b_hash")):
                            prompt_exact &= sha256_array(gaussian_prompt_numpy(*xy[source_index])) == handle[name][index].decode()
                    checks = {
                        "source_ids": list(np.asarray(rerender.catalog["catalog_row"], dtype=int)) == list(catalog_rows),
                        "source_coordinates": xy_exact, "isolated_arrays": isolated_exact, "noisy_blend": blend_exact,
                        "prompt_maps": prompt_exact, "noise_seed": int(spec.noise_seed) == int(row["noise_seed"]),
                    }
                    results.append({"partition": partition, "scene_id": row["scene_id"], "variant": variant, **checks, "status": "PASS" if all(checks.values()) else "FAIL"})
    write_csv_fresh(output, results)
    if any(row["status"] != "PASS" for row in results):
        raise RuntimeError("Stratified scene replay failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--phase", choices=("freeze", "render", "all"), default="all")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("thayer_select_prompt_ablation_"):
        raise RuntimeError("Expected an existing fresh prompt-ablation run directory")
    if args.phase in ("freeze", "all"):
        if args.replay_log is None:
            raise RuntimeError("--replay-log is required during freeze")
        prepare_provenance(run_dir, args.replay_log.resolve())
        split_rows, catalog_table = freeze_and_validate_split(run_dir)
        freeze_scene_definitions(run_dir, split_rows, catalog_table)
    if args.phase in ("render", "all"):
        definitions = read_csv(run_dir / "manifests/development_scene_definitions.csv")
        if any(row["partition"] == LOCKBOX_PARTITION for row in definitions):
            raise RuntimeError("Sealed lockbox scene request detected")
        catalog, _ = load_catsim_catalog(CATALOG)
        for partition in PARTITION_COUNTS:
            rows = [row for row in definitions if row["partition"] == partition]
            if len(rows) != PARTITION_COUNTS[partition]:
                raise RuntimeError(f"Scene count mismatch for {partition}")
            render_partition(run_dir, partition, rows, catalog)
        export_render_manifest(run_dir, definitions)
        replay_stratified(run_dir, definitions, catalog)
        write_json_fresh(run_dir / "logs/data_preparation_complete.json", {"status": "PASS", "completed_at_unix": time.time(), "lockbox_scenes_rendered": 0})


if __name__ == "__main__":
    main()
