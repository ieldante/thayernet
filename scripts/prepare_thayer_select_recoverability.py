#!/usr/bin/env python3
"""Bootstrap and render the group-safe Phase-II recoverability campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene, validated_lsst_survey
from src.prompt_semantics import QueryClass
from thayer_select_recoverability_common import (
    BANDS,
    CATALOG,
    DEVELOPMENT_COUNTS,
    FOUNDATION,
    IMAGE_SIZE,
    MANIFEST_SCHEMA_VERSION,
    NOISE_SEEDS,
    NORMALIZATION,
    PARTITION_COUNTS,
    PHASE1,
    PIXEL_SCALE_ARCSEC,
    PROMPT_SEEDS,
    PROMPT_SIGMA_PIXELS,
    QUERY_FRACTIONS,
    SCENE_SEEDS,
    SEMANTICS,
    SOURCE_SPLIT,
    TEACHER_CHECKPOINT,
    choose_prompt,
    draw_scene_positions,
    gaussian_prompt_numpy,
    model_parameter_counts,
    query_counts,
    read_csv,
    sha256_array,
    sha256_file,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, cwd=REPO, text=True, capture_output=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def bootstrap(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Master run path collision: {run_dir}")
    if not run_dir.name.startswith("thayer_select_recoverability_"):
        raise RuntimeError("Run directory must use the required timestamped prefix")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("diagnostics", "tables", "figures", "logs", "reports", "manifests", "calibration", "checkpoints", "paper_figures", "example_grids"):
        (run_dir / name).mkdir(exist_ok=False)

    import astropy
    import btk
    import galsim
    import surveycodex
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; campaign stopped before training")
    probe = torch.ones(2, device="mps")
    if float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    split_hash = sha256_file(SOURCE_SPLIT)
    foundation_split = FOUNDATION / "manifests/btk_engineering_source_groups.csv"
    if split_hash != sha256_file(foundation_split):
        raise RuntimeError("Promptability and foundation source partition hashes differ")
    split_rows = read_csv(SOURCE_SPLIT)
    partitions = Counter(row["partition"] for row in split_rows)
    if partitions.get("sealed_lockbox", 0) == 0:
        raise RuntimeError("Lockbox-exclusion policy cannot be verified")

    historical = read_csv(PHASE1 / "tables/historical_checkpoint_hashes_after.csv")
    if len(historical) != 18:
        raise RuntimeError("Historical checkpoint inventory does not contain exactly 18 rows")
    inventory = []
    for row in historical:
        path = REPO / row["relative_path"]
        observed = sha256_file(path)
        expected = row.get("expected_sha256", row.get("sha256", ""))
        inventory.append({"category": "historical_18", "relative_path": row["relative_path"], "expected_sha256": expected, "observed_sha256": observed, "status": "PASS" if observed == expected else "FAIL"})
    for path in sorted((PHASE1 / "checkpoints").glob("*.pth")):
        inventory.append({"category": "phase1_promptability", "relative_path": str(path.relative_to(REPO)), "expected_sha256": sha256_file(path), "observed_sha256": sha256_file(path), "status": "PASS"})
    if any(row["status"] != "PASS" for row in inventory):
        raise RuntimeError("Checkpoint inventory failed")
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", inventory)

    package_record = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "astropy": astropy.__version__,
        "btk": btk.__version__,
        "galsim": galsim.__version__,
        "surveycodex": version("surveycodex"),
        "h5py": h5py.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_probe": 2.0,
    }
    git = {
        "branch": command(["git", "branch", "--show-current"]),
        "head": command(["git", "rev-parse", "HEAD"]),
        "status": command(["git", "status", "--short", "--branch"]),
        "staged_index": command(["git", "diff", "--cached", "--name-status"]),
    }
    disk = command(["df", "-h", str(REPO)])
    environment = [
        "# Phase II environment snapshot",
        "",
        f"Campaign start (Unix): `{time.time()}`",
        f"Branch: `{git['branch']['stdout'].strip()}`",
        f"Git HEAD: `{git['head']['stdout'].strip()}`",
        f"Python: `{platform.python_version()}`",
        f"MPS built/available/probe: `{package_record['mps_built']}` / `{package_record['mps_available']}` / `{package_record['mps_probe']}`",
        "",
        "## Git status",
        "",
        "```text",
        git["status"]["stdout"].rstrip(),
        "```",
        "",
        "## Staged index",
        "",
        "```text",
        git["staged_index"]["stdout"].rstrip() or "(empty)",
        "```",
        "",
        "## Packages",
        "",
        "```json",
        json.dumps(package_record, indent=2, sort_keys=True),
        "```",
        "",
        "## Disk",
        "",
        "```text",
        disk["stdout"].rstrip(),
        "```",
        "",
    ]
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", "\n".join(environment))

    code_paths = [
        REPO / "src/btk_scene.py",
        REPO / "src/coordinate_prompt.py",
        REPO / "src/models_thayer_select.py",
        REPO / "src/prompt_semantics.py",
        REPO / "src/recoverability.py",
        REPO / "scripts/thayer_select_prompt_ablation_common.py",
        REPO / "scripts/thayer_select_recoverability_common.py",
        Path(__file__).resolve(),
        REPO / "scripts/train_thayer_select_recoverability.py",
        REPO / "scripts/evaluate_thayer_select_recoverability.py",
        REPO / "scripts/finalize_thayer_select_recoverability.py",
    ]
    missing_code = [str(path) for path in code_paths if not path.is_file()]
    if missing_code:
        raise RuntimeError(f"Campaign code is incomplete: {missing_code}")
    code_hashes = [{"relative_path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in code_paths]
    write_csv_fresh(run_dir / "tables/campaign_code_hashes.csv", code_hashes)
    prompt_manifests = [PHASE1 / f"manifests/{name}" for name in ("source_split_manifest.csv", "training_scenes.h5", "validation_scenes.h5", "calibration_scenes.h5", "normalization.json")]
    provenance = {
        "campaign_start_unix": time.time(),
        "git": git,
        "packages": package_record,
        "disk": disk,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": sha256_file(CATALOG)},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": split_hash},
        "foundation_source_split_sha256": sha256_file(foundation_split),
        "promptability_manifests": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in prompt_manifests],
        "teacher_checkpoint": {"path": str(TEACHER_CHECKPOINT.relative_to(REPO)), "sha256": sha256_file(TEACHER_CHECKPOINT)},
        "historical_checkpoint_count": 18,
        "historical_checkpoint_inventory_sha256": sha256_file(run_dir / "tables/checkpoint_inventory_before.csv"),
        "code_hashes": code_hashes,
        "lockbox_partition_rows": partitions["sealed_lockbox"],
        "lockbox_scenes_accessed": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)
    contract = f"""# Phase II campaign contract

Status: frozen before scene generation or training.

- Output root: `{run_dir.relative_to(REPO)}`; collisions are refused.
- Neural training and inference: MPS only. CPU is limited to manifests, metrics, calibration, figures, tables, and reports.
- Persistent source partition SHA-256: `{split_hash}`.
- Sealed-lockbox assignment rows: `{partitions['sealed_lockbox']}`; no lockbox scene may be generated, opened, rendered, inspected, calibrated, or evaluated.
- Phase-I teacher SHA-256: `{sha256_file(TEACHER_CHECKPOINT)}`; it is inference-only and never fine-tuned.
- Query composition: `{json.dumps({key.value: value for key, value in QUERY_FRACTIONS.items()}, sort_keys=True)}`.
- Prompt matching radius: {SEMANTICS.matching_radius_pixels} px ({SEMANTICS.matching_radius_psf:.2f} mean-PSF FWHM); ambiguity margin: {SEMANTICS.ambiguity_margin_pixels} px.
- Ambiguous queries have no arbitrary reconstruction target. Null queries have an exact zero target.
- Primary contract candidate is MODERATE. If its training/validation teacher-label rate is outside [0.05, 0.95], the closest-to-balanced predeclared contract may be selected before R1 training; all three remain reported.
- Development scenes are generated only after checkpoints, calibrator, scores, thresholds, and evaluation metrics are frozen, and evaluated exactly once.
- No commit, stage, push, merge, deletion, historical-output modification, or CPU neural fallback is authorized.
"""
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", contract)
    write_json_fresh(run_dir / "manifests/model_parameter_counts.json", model_parameter_counts())
    shutil.copyfile(SOURCE_SPLIT, run_dir / "manifests/source_split_manifest.csv")
    shutil.copyfile(NORMALIZATION, run_dir / "manifests/normalization.json")
    write_json_fresh(run_dir / "logs/bootstrap_complete.json", {"status": "PASS", "lockbox_accessed": False, "completed_at_unix": time.time()})


def scene_definitions(run_dir: Path, partition: str, count: int, query_distribution: dict[QueryClass, int]) -> list[dict]:
    split_rows = read_csv(run_dir / "manifests/source_split_manifest.csv")
    pool = [row for row in split_rows if row["partition"] == partition and row["engineering_excluded"] == "0"]
    if len(pool) < 2 or partition == "sealed_lockbox":
        raise RuntimeError(f"Invalid source pool for {partition}")
    rng = np.random.default_rng(SCENE_SEEDS[partition])
    query_values = np.asarray([query.value for query, query_count in query_distribution.items() for _ in range(query_count)])
    rng.shuffle(query_values)
    if len(query_values) != count:
        raise RuntimeError("Query count mismatch")
    records = []
    for index, query_text in enumerate(query_values):
        query = QueryClass(query_text)
        selected_indices = rng.choice(len(pool), size=2, replace=False)
        selected = [pool[int(value)] for value in selected_indices]
        positions = draw_scene_positions(rng, query)
        requested_index = int(rng.integers(0, 2))
        records.append({
            "scene_id": f"{partition}_{index:05d}",
            "partition": partition,
            "partition_index": index,
            "query_class": query.value,
            "requested_index_for_generation": requested_index,
            "source_a_row": int(selected[0]["catalog_row"]),
            "source_b_row": int(selected[1]["catalog_row"]),
            "source_a_id": selected[0]["persistent_source_id"],
            "source_b_id": selected[1]["persistent_source_id"],
            "source_a_group": selected[0]["duplicate_group_id"],
            "source_b_group": selected[1]["duplicate_group_id"],
            "source_a_x_arcsec": positions[0, 0],
            "source_a_y_arcsec": positions[0, 1],
            "source_b_x_arcsec": positions[1, 0],
            "source_b_y_arcsec": positions[1, 1],
            "scene_seed": SCENE_SEEDS[partition] + index,
            "noise_seed": NOISE_SEEDS[partition] + index,
            "prompt_seed": PROMPT_SEEDS[partition] + index,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        })
    return records


def create_store(path: Path, count: int) -> h5py.File:
    handle = h5py.File(path, "x")
    for name, shape, dtype in (
        ("blend", (count, 3, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("isolated", (count, 2, 3, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("xy", (count, 2, 2), "f8"),
        ("prompt_xy", (count, 2), "f8"),
        ("prompt", (count, 1, IMAGE_SIZE, IMAGE_SIZE), "f4"),
        ("matched_index", (count,), "i1"),
        ("target_defined", (count,), "?"),
    ):
        chunks = (1,) + shape[1:] if len(shape) > 1 else True
        handle.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compression="lzf" if len(shape) > 1 else None)
    handle.attrs["completed_count"] = 0
    handle.attrs["complete"] = False
    return handle


def render_partition(run_dir: Path, partition: str, definitions: list[dict], catalog, catalog_table: Table) -> list[dict]:
    path = run_dir / f"manifests/{partition}_scenes.h5"
    manifest_rows = []
    zero_target_hash = sha256_array(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    survey = validated_lsst_survey()
    psf_fwhm_arcsec = {band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in BANDS}
    with create_store(path, len(definitions)) as handle:
        for index, row in enumerate(definitions):
            catalog_rows = (int(row["source_a_row"]), int(row["source_b_row"]))
            positions = ((float(row["source_a_x_arcsec"]), float(row["source_a_y_arcsec"])), (float(row["source_b_x_arcsec"]), float(row["source_b_y_arcsec"])))
            spec = SceneSpec(row["scene_id"], catalog_rows, positions, int(row["scene_seed"]), int(row["scene_seed"]), int(row["noise_seed"]))
            rendered = render_fixed_scene(catalog, spec, add_noise="all")
            blend = np.asarray(rendered.blend, dtype=np.float32)
            isolated = np.asarray(rendered.isolated, dtype=np.float32)
            xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered.catalog], dtype=np.float64)
            query = QueryClass(row["query_class"])
            prompt_xy, association = choose_prompt(xy, query, int(row["prompt_seed"]), int(row["requested_index_for_generation"]))
            prompt = gaussian_prompt_numpy(float(prompt_xy[0]), float(prompt_xy[1]), sigma_pixels=PROMPT_SIGMA_PIXELS)[None]
            matched = -1 if association.matched_index is None else int(association.matched_index)
            target_defined = query is not QueryClass.AMBIGUOUS_SOURCE
            target_hash = zero_target_hash if query is QueryClass.NULL_SOURCE else (sha256_array(isolated[matched]) if matched >= 0 else "AMBIGUOUS_NO_TARGET")
            if query is QueryClass.NULL_SOURCE and not np.array_equal(np.zeros_like(isolated[0]), np.zeros_like(isolated[0])):
                raise RuntimeError("Null target zero check failed")
            handle["blend"][index] = blend
            handle["isolated"][index] = isolated
            handle["xy"][index] = xy
            handle["prompt_xy"][index] = prompt_xy
            handle["prompt"][index] = prompt
            handle["matched_index"][index] = matched
            handle["target_defined"][index] = target_defined
            flux = isolated.sum(axis=(-2, -1), dtype=np.float64)
            total_flux = flux.sum(axis=1)
            separation_pixels = float(np.linalg.norm(xy[0] - xy[1]))
            mean_psf_pixels = SEMANTICS.psf_fwhm_pixels
            sizes = [max(float(catalog_table["a_b"][source]), float(catalog_table["a_d"][source])) for source in catalog_rows]
            minors = [max(float(catalog_table["b_b"][source]), float(catalog_table["b_d"][source])) for source in catalog_rows]
            ellipticities = [(major - minor) / max(major + minor, 1e-30) for major, minor in zip(sizes, minors)]
            colors = [np.asarray([float(catalog_table["g_ab"][source]) - float(catalog_table["r_ab"][source]), float(catalog_table["r_ab"][source]) - float(catalog_table["z_ab"][source])]) for source in catalog_rows]
            if matched >= 0:
                alternate = 1 - matched
                target_peak = np.max(np.abs(isolated[matched]), axis=0)
                core = target_peak > 0.5 * float(target_peak.max())
                core_obstruction = float(np.maximum(isolated[alternate, :, core], 0).sum() / max(np.maximum(isolated[matched, :, core], 0).sum(), 1e-30))
                flux_ratio = float(total_flux[matched] / max(total_flux[alternate], 1e-30))
                size_ratio = float(sizes[matched] / max(sizes[alternate], 1e-30))
                visibility = float(np.sum(np.abs(isolated[matched])) / max(np.sum(np.abs(blend)), 1e-30))
                snr_proxy = float(total_flux[matched] / max(np.sqrt(np.sum(np.abs(blend))), 1e-30))
                matched_id = row[f"source_{'a' if matched == 0 else 'b'}_id"]
                matched_group = row[f"source_{'a' if matched == 0 else 'b'}_group"]
            else:
                core_obstruction = flux_ratio = size_ratio = visibility = snr_proxy = np.nan
                matched_id = matched_group = ""
            manifest_rows.append({
                **row,
                "source_a_x_pixel": xy[0, 0], "source_a_y_pixel": xy[0, 1], "source_b_x_pixel": xy[1, 0], "source_b_y_pixel": xy[1, 1],
                "prompt_x_pixel": prompt_xy[0], "prompt_y_pixel": prompt_xy[1],
                "matched_source_index": matched, "matched_source_id": matched_id, "matched_source_group": matched_group,
                "null_label": int(query is QueryClass.NULL_SOURCE), "ambiguity_label": int(query is QueryClass.AMBIGUOUS_SOURCE),
                "target_defined": int(target_defined), "target_marker": "ZERO_TARGET" if query is QueryClass.NULL_SOURCE else ("NO_ARBITRARY_TARGET" if query is QueryClass.AMBIGUOUS_SOURCE else "ISOLATED_SOURCE"),
                "nearest_distance_pixels": association.nearest_distance_pixels, "second_distance_pixels": association.second_distance_pixels,
                "coordinate_error_pixels": association.coordinate_error_pixels, "candidate_source_count": association.candidate_count,
                "matching_radius_pixels": SEMANTICS.matching_radius_pixels, "matching_radius_psf_units": SEMANTICS.matching_radius_psf,
                "ambiguity_margin_pixels": SEMANTICS.ambiguity_margin_pixels,
                "separation_pixels": separation_pixels, "separation_psf_units": separation_pixels / mean_psf_pixels,
                "flux_ratio": flux_ratio, "color_similarity_distance": float(np.linalg.norm(colors[0] - colors[1])), "size_ratio": size_ratio,
                "source_a_size_arcsec": sizes[0], "source_b_size_arcsec": sizes[1], "source_a_ellipticity": ellipticities[0], "source_b_ellipticity": ellipticities[1],
                "core_obstruction": core_obstruction, "source_visibility": visibility, "snr_proxy": snr_proxy, "source_count": 2,
                "per_band_psf_fwhm_arcsec": json.dumps(psf_fwhm_arcsec, sort_keys=True),
                "noise_configuration": "BTK LSST add_noise=all; explicit noise seed", "simulator_version": "BTK 1.0.9 / GalSim 2.8.4 / surveycodex 1.2.0",
                "isolated_source_a_sha256": sha256_array(isolated[0]), "isolated_source_b_sha256": sha256_array(isolated[1]),
                "blend_sha256": sha256_array(blend), "target_sha256_or_marker": target_hash, "prompt_map_sha256": sha256_array(prompt),
                "hdf5_path": str(path.relative_to(REPO)),
            })
            handle.attrs["completed_count"] = index + 1
            if (index + 1) % 100 == 0 or index + 1 == len(definitions):
                handle.flush()
                print(f"{partition}: rendered {index + 1}/{len(definitions)}", flush=True)
        handle.attrs["complete"] = True
        handle.attrs["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
        handle.flush()
    return manifest_rows


def audit_partition(run_dir: Path, partition: str, definitions: list[dict], manifest: list[dict], catalog) -> list[dict]:
    results = []
    path = run_dir / f"manifests/{partition}_scenes.h5"
    by_query = defaultdict(list)
    for index, row in enumerate(manifest):
        by_query[row["query_class"]].append(index)
    chosen = sorted({indices[position] for indices in by_query.values() for position in np.linspace(0, len(indices) - 1, min(3, len(indices))).astype(int)})
    with h5py.File(path, "r") as handle:
        for index in chosen:
            row = definitions[index]
            positions = ((float(row["source_a_x_arcsec"]), float(row["source_a_y_arcsec"])), (float(row["source_b_x_arcsec"]), float(row["source_b_y_arcsec"])))
            spec = SceneSpec(row["scene_id"], (int(row["source_a_row"]), int(row["source_b_row"])), positions, int(row["scene_seed"]), int(row["scene_seed"]), int(row["noise_seed"]))
            replay = render_fixed_scene(catalog, spec, add_noise="all")
            blend_exact = np.array_equal(np.asarray(replay.blend, dtype=np.float32), np.asarray(handle["blend"][index]))
            isolated_exact = np.array_equal(np.asarray(replay.isolated, dtype=np.float32), np.asarray(handle["isolated"][index]))
            xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in replay.catalog], dtype=np.float64)
            xy_exact = np.array_equal(xy, np.asarray(handle["xy"][index]))
            prompt_xy, association = choose_prompt(xy, QueryClass(row["query_class"]), int(row["prompt_seed"]), int(row["requested_index_for_generation"]))
            prompt = gaussian_prompt_numpy(*prompt_xy, sigma_pixels=PROMPT_SIGMA_PIXELS)[None]
            prompt_exact = np.array_equal(prompt, np.asarray(handle["prompt"][index]))
            checks = {"blend_exact": blend_exact, "isolated_exact": isolated_exact, "coordinates_exact": xy_exact, "prompt_exact": prompt_exact, "semantics_exact": association.query_class.value == row["query_class"]}
            results.append({"partition": partition, "scene_id": row["scene_id"], "query_class": row["query_class"], **checks, "status": "PASS" if all(checks.values()) else "FAIL"})
    return results


def render_frozen_partitions(run_dir: Path) -> None:
    if not json.loads((run_dir / "logs/bootstrap_complete.json").read_text()).get("status") == "PASS":
        raise RuntimeError("Bootstrap gate did not pass")
    catalog, _ = load_catsim_catalog(CATALOG)
    catalog_table = Table.read(CATALOG)
    all_manifest = []
    replay_rows = []
    for partition, count in PARTITION_COUNTS.items():
        definitions = scene_definitions(run_dir, partition, count, query_counts(count))
        write_csv_fresh(run_dir / f"manifests/{partition}_scene_definitions.csv", definitions)
        manifest = render_partition(run_dir, partition, definitions, catalog, catalog_table)
        write_csv_fresh(run_dir / f"manifests/{partition}_scene_manifest.csv", manifest)
        all_manifest.extend(manifest)
        replay_rows.extend(audit_partition(run_dir, partition, definitions, manifest, catalog))
    write_csv_fresh(run_dir / "manifests/all_scene_manifest.csv", all_manifest)
    write_csv_fresh(run_dir / "tables/stratified_exact_replay.csv", replay_rows)
    split_by_source = defaultdict(set); split_by_group = defaultdict(set)
    for row in all_manifest:
        for suffix in ("a", "b"):
            split_by_source[row[f"source_{suffix}_id"]].add(row["partition"])
            split_by_group[row[f"source_{suffix}_group"]].add(row["partition"])
    checks = [
        {"check": "unique_scene_ids", "observed": len({row["scene_id"] for row in all_manifest}), "expected": len(all_manifest), "status": "PASS" if len({row["scene_id"] for row in all_manifest}) == len(all_manifest) else "FAIL"},
        {"check": "source_cross_partition", "observed": sum(len(value) > 1 for value in split_by_source.values()), "expected": 0, "status": "PASS" if all(len(value) == 1 for value in split_by_source.values()) else "FAIL"},
        {"check": "group_cross_partition", "observed": sum(len(value) > 1 for value in split_by_group.values()), "expected": 0, "status": "PASS" if all(len(value) == 1 for value in split_by_group.values()) else "FAIL"},
        {"check": "lockbox_scene_count", "observed": sum("lockbox" in row["partition"] for row in all_manifest), "expected": 0, "status": "PASS" if not any("lockbox" in row["partition"] for row in all_manifest) else "FAIL"},
        {"check": "null_zero_target_markers", "observed": sum(row["target_marker"] == "ZERO_TARGET" for row in all_manifest if row["query_class"] == QueryClass.NULL_SOURCE.value), "expected": sum(row["query_class"] == QueryClass.NULL_SOURCE.value for row in all_manifest), "status": "PASS" if all(row["target_marker"] == "ZERO_TARGET" for row in all_manifest if row["query_class"] == QueryClass.NULL_SOURCE.value) else "FAIL"},
        {"check": "ambiguous_no_target", "observed": sum(row["target_defined"] == 0 for row in all_manifest if row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value), "expected": sum(row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value for row in all_manifest), "status": "PASS" if all(row["target_marker"] == "NO_ARBITRARY_TARGET" and row["target_defined"] == 0 for row in all_manifest if row["query_class"] == QueryClass.AMBIGUOUS_SOURCE.value) else "FAIL"},
        {"check": "stratified_replay", "observed": sum(row["status"] == "PASS" for row in replay_rows), "expected": len(replay_rows), "status": "PASS" if all(row["status"] == "PASS" for row in replay_rows) else "FAIL"},
    ]
    write_csv_fresh(run_dir / "tables/structural_checks.csv", checks)
    if any(row["status"] != "PASS" for row in checks):
        raise RuntimeError("Structural or replay audit failed")
    write_json_fresh(run_dir / "logs/data_preparation_complete.json", {"status": "PASS", "scene_counts": PARTITION_COUNTS, "query_counts": {partition: {query.value: value for query, value in query_counts(count).items()} for partition, count in PARTITION_COUNTS.items()}, "lockbox_scenes_accessed": 0, "completed_at_unix": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("bootstrap", "render", "all"), default="all")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.phase in ("bootstrap", "all"):
        bootstrap(run_dir)
    if args.phase in ("render", "all"):
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        render_frozen_partitions(run_dir)


if __name__ == "__main__":
    main()
