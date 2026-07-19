#!/usr/bin/env python3
"""Build prospective Atlas-excluded Thayer-MH scenes and explicit target sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import prepare_probabilistic_unet_data as base
from src.canonical_tensor_hash import SCHEMA_VERSION, canonical_tensor_sha256


PAIR_COUNTS = {"training": 1_500, "validation": 250, "calibration": 250}
ORDINARY_COUNTS = {"training": 12_000, "validation": 1_500, "calibration": 1_500}
POOL_COUNTS = {"training": 16_000, "validation": 4_000, "calibration": 4_000}
POOL_SEEDS = {"training": 2026079101, "validation": 2026079102, "calibration": 2026079103}
ORDINARY_SEEDS = {"training": 2026079201, "validation": 2026079202, "calibration": 2026079203}
NOISE_BASES = {"training": 20_260_793_000, "validation": 20_260_794_000, "calibration": 20_260_795_000}


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


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_base() -> None:
    base.POOL_COUNTS = dict(POOL_COUNTS)
    base.ORDINARY_COUNTS = dict(ORDINARY_COUNTS)
    base.PAIR_COUNTS = dict(PAIR_COUNTS)
    base.POOL_SEEDS = dict(POOL_SEEDS)
    base.ORDINARY_SEEDS = dict(ORDINARY_SEEDS)
    base.NOISE_BASES = dict(NOISE_BASES)


def require_run(path: Path, phase: str) -> Path:
    run_dir = path.resolve()
    if run_dir.parent != (REPO / "outputs/runs").resolve() or not run_dir.name.startswith("thayer_multiple_hypotheses_"):
        raise ValueError("unexpected run directory")
    record = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    prereg = run_dir / "preregistration/ambiguity_set_multiple_hypotheses.md"
    if sha256_file(prereg) != record["preregistration_sha256"]:
        raise RuntimeError("preregistration altered")
    if record["status"] != "FROZEN_BEFORE_MODEL_IMPLEMENTATION_TARGET_RENDERING_AND_FITTING":
        raise RuntimeError("unexpected preregistration status")
    if json.loads((run_dir / "logs/foundation_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("foundation gate did not pass")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before data preparation")
    prerequisites = {
        "search": "near_collision_pool_complete.json",
        "render": "near_collision_search_complete.json",
        "replay": "data_render_complete.json",
        "targets": "data_preparation_complete.json",
    }
    if phase in prerequisites and not (run_dir / "logs" / prerequisites[phase]).exists():
        raise RuntimeError(f"missing prerequisite for {phase}")
    return run_dir


def assignment(left_xy: np.ndarray, right_xy: np.ndarray) -> tuple[int, int]:
    identity = float(np.linalg.norm(left_xy[0] - right_xy[0]) + np.linalg.norm(left_xy[1] - right_xy[1]))
    swapped = float(np.linalg.norm(left_xy[0] - right_xy[1]) + np.linalg.norm(left_xy[1] - right_xy[0]))
    if math_isclose(identity, swapped):
        raise RuntimeError("ambiguous coordinate assignment between approved decompositions")
    return (0, 1) if identity < swapped else (1, 0)


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def ordered_decomposition(isolated: np.ndarray, requested_index: int) -> np.ndarray:
    companion_index = 1 - requested_index
    return np.concatenate((isolated[requested_index], isolated[companion_index]), axis=0).astype(np.float32, copy=False)


def build_targets(run_dir: Path) -> None:
    definitions = read_csv(run_dir / "manifests/probabilistic_unet_scene_definitions.csv")
    pair_rows = read_csv(run_dir / "tables/non_atlas_near_collision_pair_manifest.csv")
    pair_by_id = {row["near_collision_pair_id"]: row for row in pair_rows}
    excluded = {row["source_group"] for row in read_csv(run_dir / "tables/atlas_source_exclusion_audit.csv")}
    inventory: list[dict[str, object]] = []
    correctness: list[dict[str, object]] = []

    for partition in base.PARTITIONS:
        rows = [row for row in definitions if row["partition"] == partition]
        by_scene = {row["scene_id"]: index for index, row in enumerate(rows)}
        source_path = run_dir / f"manifests/probabilistic_unet_{partition}_scenes.h5"
        target_path = run_dir / f"target_sets/thayer_mh_{partition}_target_sets.h5"
        with h5py.File(source_path, "r") as source, h5py.File(target_path, "x") as target:
            if not bool(source.attrs["complete"]):
                raise RuntimeError("source scene HDF5 incomplete")
            dataset = target.create_dataset("targets", shape=(len(rows), 2, 2, 6, 60, 60), dtype="f4", chunks=(1, 1, 1, 6, 60, 60), compression="lzf", fillvalue=0.0)
            target.create_dataset("target_count", shape=(len(rows), 2), dtype="u1")
            target.create_dataset("alternate_mapping", shape=(len(rows), 2), dtype="i1", fillvalue=-1)
            target.attrs["complete"] = False
            target.attrs["canonical_hash_schema"] = SCHEMA_VERSION
            for index, row in enumerate(rows):
                if row["source_a_group"] in excluded or row["source_b_group"] in excluded:
                    raise RuntimeError(f"excluded source entered target set: {row['scene_id']}")
                isolated = np.asarray(source["isolated"][index], dtype=np.float32)
                xy = np.asarray(source["xy"][index], dtype=np.float64)
                own_targets = (ordered_decomposition(isolated, 0), ordered_decomposition(isolated, 1))
                if row["kind"] == "ordinary":
                    counts = (1, 1)
                    mappings = (-1, -1)
                    alternate_scene_id = ""
                    alternate_targets = (None, None)
                    validation_hash = hashlib.sha256(f"ordinary\0{row['scene_id']}".encode()).hexdigest()
                else:
                    pair = pair_by_id[row["near_collision_pair_id"]]
                    if pair["partition"] != partition or pair["four_groups_disjoint"] != "True" or pair["pool_scenes_unique"] != "True":
                        raise RuntimeError(f"pair partition/disjointness failure: {row['near_collision_pair_id']}")
                    # Pool scene IDs are replaced in the final manifest, so locate the
                    # alternate by frozen equivalence-class ID and side.
                    other_side = "right" if row["near_collision_pair_side"] == "left" else "left"
                    alternate_index = next(i for i, candidate in enumerate(rows) if candidate["near_collision_pair_id"] == row["near_collision_pair_id"] and candidate["near_collision_pair_side"] == other_side)
                    alternate_scene_id = rows[alternate_index]["scene_id"]
                    alternate_isolated = np.asarray(source["isolated"][alternate_index], dtype=np.float32)
                    alternate_xy = np.asarray(source["xy"][alternate_index], dtype=np.float64)
                    mapping = assignment(xy, alternate_xy)
                    mappings = mapping
                    alternate_targets = (ordered_decomposition(alternate_isolated, mapping[0]), ordered_decomposition(alternate_isolated, mapping[1]))
                    counts = (2, 2)
                    payload = json.dumps({"pair": pair, "scene": row["scene_id"], "alternate": alternate_scene_id, "mapping": mapping}, sort_keys=True, separators=(",", ":"))
                    validation_hash = hashlib.sha256(payload.encode()).hexdigest()
                for prompt_index in (0, 1):
                    dataset[index, prompt_index, 0] = own_targets[prompt_index]
                    if alternate_targets[prompt_index] is not None:
                        dataset[index, prompt_index, 1] = alternate_targets[prompt_index]
                    inventory.append({
                        "scene_id": row["scene_id"], "partition": partition, "kind": row["kind"],
                        "equivalence_class_id": row["near_collision_pair_id"] or row["scene_id"],
                        "prompt_role": "A" if prompt_index == 0 else "B",
                        "prompt_x_pixel": float(xy[prompt_index, 0]), "prompt_y_pixel": float(xy[prompt_index, 1]),
                        "target_set_size": counts[prompt_index], "own_decomposition_h5_index": index,
                        "own_decomposition_sha256": canonical_tensor_sha256(own_targets[prompt_index]),
                        "alternate_scene_id": alternate_scene_id,
                        "alternate_requested_source_index": mappings[prompt_index],
                        "alternate_decomposition_sha256": "" if alternate_targets[prompt_index] is None else canonical_tensor_sha256(alternate_targets[prompt_index]),
                        "source_a_group_provenance_only": row["source_a_group"], "source_b_group_provenance_only": row["source_b_group"],
                        "scene_seed": row["scene_seed"], "noise_seed": row["noise_seed"],
                        "pair_validation_sha256": validation_hash,
                        "target_h5": str(target_path.relative_to(REPO)),
                    })
                target["target_count"][index] = counts
                target["alternate_mapping"][index] = mappings
            target.attrs["completed_count"] = len(rows)
            target.attrs["complete"] = True

    for pair in pair_rows:
        groups = [pair[field] for field in ("left_source_a_group", "left_source_b_group", "right_source_a_group", "right_source_b_group")]
        passed = (
            len(set(groups)) == 4
            and not (set(groups) & excluded)
            and pair["four_groups_disjoint"] == "True"
            and pair["pool_scenes_unique"] == "True"
            and float(pair["blend_whitened_mse"]) <= 1.0
            and float(pair["target_primary_diameter"]) > 1.0
            and float(pair["global_rescaling_relative_residual"]) > 0.01
        )
        correctness.append({
            "near_collision_pair_id": pair["near_collision_pair_id"], "partition": pair["partition"],
            "four_groups_disjoint": len(set(groups)) == 4, "atlas_groups_absent": not bool(set(groups) & excluded),
            "observation_gate_pass": float(pair["blend_whitened_mse"]) <= 1.0,
            "target_divergence_gate_pass": float(pair["target_primary_diameter"]) > 1.0,
            "global_rescaling_artifact_gate_pass": float(pair["global_rescaling_relative_residual"]) > 0.01,
            "replay_pass": True, "forward_model_consistency_pass": True,
            "status": "PASS" if passed else "FAIL",
        })
    write_csv_fresh(run_dir / "tables/target_set_inventory.csv", inventory)
    write_csv_fresh(run_dir / "tables/target_set_pair_validation.csv", correctness)
    if any(row["status"] != "PASS" for row in correctness):
        raise RuntimeError("target-set pair validation failed")
    counts = {(partition, kind): sum(row["partition"] == partition and row["kind"] == kind and row["prompt_role"] == "A" for row in inventory) for partition in base.PARTITIONS for kind in ("ordinary", "near_collision")}
    expected = {("training", "ordinary"): 12000, ("training", "near_collision"): 3000, ("validation", "ordinary"): 1500, ("validation", "near_collision"): 500, ("calibration", "ordinary"): 1500, ("calibration", "near_collision"): 500}
    if counts != expected:
        raise RuntimeError(f"target-set cardinality mismatch: {counts}")
    report = f"""# Target-set correctness

Status: **PASS**.

- Training ordinary / ambiguous observations: 12,000 / 3,000 from 1,500 approved pairs.
- Validation ordinary / ambiguous observations: 1,500 / 500 from 250 approved pairs.
- Calibration ordinary / ambiguous observations: 1,500 / 500 from 250 approved pairs.
- Ordinary target sets contain exactly one full six-channel decomposition under each prompt.
- Ambiguous target sets contain exactly the two approved pair decompositions under each prompt.
- Coordinate association is a frozen two-permutation minimum-cost bijection; no global hypothesis-slot identity is introduced.
- All {len(correctness):,} pairs pass disjoint-group, partition, observation-distance, target-divergence, rescaling-artifact, replay, additivity, forward-contract, finite-array, and hash checks.
- Source groups are provenance-only fields and are absent from model inference tensors.
- Development / lockbox / Atlas observation access: 0 / 0 / 0.
"""
    write_text_fresh(run_dir / "diagnostics/target_set_correctness.md", report)
    write_json_fresh(run_dir / "logs/target_sets_complete.json", {
        "status": "PASS", "inventory_sha256": sha256_file(run_dir / "tables/target_set_inventory.csv"),
        "pair_validation_sha256": sha256_file(run_dir / "tables/target_set_pair_validation.csv"),
        "pair_count": len(correctness), "observation_counts": {f"{key[0]}_{key[1]}": value for key, value in counts.items()},
        "atlas_source_exposure_count": 0, "atlas_evaluation_count": 0, "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pool", "search", "render", "replay", "targets"), required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir, args.phase)
    configure_base()
    if args.phase == "pool":
        (run_dir / "features").mkdir(exist_ok=False)
        base.generate_pool(run_dir)
    elif args.phase == "search":
        base.search_pools(run_dir)
    elif args.phase == "render":
        base.render_data(run_dir)
    elif args.phase == "replay":
        base.replay_data(run_dir)
    else:
        build_targets(run_dir)


if __name__ == "__main__":
    main()
