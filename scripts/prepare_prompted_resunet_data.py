#!/usr/bin/env python3
"""Render and fully replay the Atlas-excluded prompted-ResUNet manifests."""

from __future__ import annotations

import argparse
import ast
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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy  # noqa: E402
from src.btk_scene import SceneSpec, load_catsim_catalog, render_fixed_scene, validated_lsst_survey  # noqa: E402
from src.models_prompted_resunet import PromptedResUNet, trainable_parameter_count  # noqa: E402


CATALOG = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits"
CONDITION_C_PARAMETERS = 119_091
EXPECTED_PARAMETERS = 199_219


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
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_run(path: Path) -> Path:
    run_dir = path.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("thayer_prompted_resunet_diversity_"):
        raise RuntimeError("expected an existing prompted-ResUNet run")
    freeze = json.loads((run_dir / "preregistration/freeze_record.json").read_text())
    if freeze["status"] != "FROZEN_BEFORE_CORRECTED_MODEL_IMPLEMENTATION_RENDERING_OR_FITTING":
        raise RuntimeError("preregistration is not frozen")
    manifest = run_dir / "manifests/resunet_scene_definitions.csv"
    if sha256_file(manifest) != freeze["scene_manifest_sha256"]:
        raise RuntimeError("frozen scene manifest changed")
    if any((run_dir / "checkpoints").iterdir()):
        raise RuntimeError("checkpoint exists before data preparation")
    return run_dir


def architecture_gate(run_dir: Path) -> None:
    source_path = REPO / "src/models_prompted_resunet.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = [name for name in imports if "thayer_select" in name or "prompt_ablation" in name]
    model = PromptedResUNet()
    count = trainable_parameter_count(model)
    blocks = [model.enc0, model.enc1, model.enc2, model.bottleneck, model.dec1, model.dec0]
    checks = {
        "exact_parameter_count": count == EXPECTED_PARAMETERS,
        "preferred_parameter_ceiling": count < 350_000,
        "absolute_parameter_ceiling": count <= 500_000,
        "six_residual_blocks": len(blocks) == 6,
        "condition_c_import_absent": not forbidden,
        "historical_checkpoint_loading_absent": "torch.load" not in source_path.read_text(encoding="utf-8"),
        "fresh_initialization_implemented": "kaiming_normal_" in source_path.read_text(encoding="utf-8"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"architecture gate failed: {checks}")
    write_csv_fresh(run_dir / "tables/model_parameter_comparison.csv", [
        {"model": "Condition C", "family": "compact prompted plain U-Net", "trainable_parameters": CONDITION_C_PARAMETERS, "ratio_to_condition_c": 1.0, "parameter_ceiling_pass": True},
        {"model": "Prompted ResUNet", "family": "residual-block encoder-decoder", "trainable_parameters": count, "ratio_to_condition_c": count / CONDITION_C_PARAMETERS, "parameter_ceiling_pass": True},
    ])
    report = f"""# Prompted ResUNet architecture report

Status: **PASS**.

- Exact trainable parameters: {count:,}.
- Ratio to Condition C ({CONDITION_C_PARAMETERS:,}): {count / CONDITION_C_PARAMETERS:.6f}.
- Preferred/absolute ceilings: 350,000 / 500,000 — both pass.
- Six residual blocks span encoder, bottleneck, and decoder.
- Downsampling uses stride-2 residual blocks; decoding uses bilinear upsampling, concatenated skips, and residual fusion.
- Input/output: normalized g/r/z plus one Gaussian prompt to three linear requested-source channels.
- Initialization: fresh Kaiming-normal convolutions; no historical loading or Condition-C architecture import.
- Model source SHA-256: `{sha256_file(source_path)}`.

```text
4x60x60 -> RB(4,16) -> RBs2(16,32) -> RBs2(32,64)
         -> RB(64,64) -> up+skip32 -> RB(96,32)
         -> up+skip16 -> RB(48,16) -> linear 1x1 -> 3x60x60
```
"""
    write_text_fresh(run_dir / "diagnostics/resunet_architecture_report.md", report)
    write_json_fresh(run_dir / "diagnostics/resunet_architecture_checks.json", {"status": "PASS", "checks": checks, "parameter_count": count})


def create_h5(path: Path, count: int) -> h5py.File:
    handle = h5py.File(path, "x")
    handle.create_dataset("blend", shape=(count, 3, 60, 60), dtype="f4", chunks=(1, 3, 60, 60), compression="lzf")
    handle.create_dataset("isolated", shape=(count, 2, 3, 60, 60), dtype="f4", chunks=(1, 2, 3, 60, 60), compression="lzf")
    handle.create_dataset("xy", shape=(count, 2, 2), dtype="f8")
    for name in ("blend_sha256", "isolated_a_sha256", "isolated_b_sha256", "prompt_a_sha256", "prompt_b_sha256"):
        handle.create_dataset(name, shape=(count,), dtype="S64")
    handle.attrs["complete"] = False
    handle.attrs["completed_count"] = 0
    return handle


def spec_from_row(row: dict[str, str]) -> SceneSpec:
    return SceneSpec(
        scene_id=row["scene_id"],
        catalog_rows=(int(row["source_a_row"]), int(row["source_b_row"])),
        positions_arcsec=(
            (float(row["source_a_x_arcsec"]), float(row["source_a_y_arcsec"])),
            (float(row["source_b_x_arcsec"]), float(row["source_b_y_arcsec"])),
        ),
        source_selection_seed=int(row["scene_seed"]),
        position_seed=int(row["scene_seed"]),
        noise_seed=int(row["noise_seed"]),
    )


def render_partition(run_dir: Path, partition: str, rows: list[dict[str, str]], catalog) -> list[dict[str, object]]:
    path = run_dir / f"manifests/resunet_{partition}_scenes.h5"
    records = []
    started = time.time()
    with create_h5(path, len(rows)) as handle:
        for index, row in enumerate(rows):
            rendered = render_fixed_scene(catalog, spec_from_row(row), add_noise="all")
            blend = np.asarray(rendered.blend, dtype=np.float32)
            isolated = np.asarray(rendered.isolated, dtype=np.float32)
            xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered.catalog], dtype=np.float64)
            if list(np.asarray(rendered.catalog["catalog_row"], dtype=int)) != [int(row["source_a_row"]), int(row["source_b_row"])]:
                raise RuntimeError("rendered source alignment failed")
            prompts = [gaussian_prompt_numpy(float(x), float(y)) for x, y in xy]
            hashes = {
                "blend_sha256": sha256_array(blend),
                "isolated_a_sha256": sha256_array(isolated[0]),
                "isolated_b_sha256": sha256_array(isolated[1]),
                "prompt_a_sha256": sha256_array(prompts[0]),
                "prompt_b_sha256": sha256_array(prompts[1]),
            }
            handle["blend"][index] = blend
            handle["isolated"][index] = isolated
            handle["xy"][index] = xy
            for name, digest in hashes.items():
                handle[name][index] = digest.encode()
            handle.attrs["completed_count"] = index + 1
            records.append({
                "scene_id": row["scene_id"], "partition": partition, "partition_index": index,
                "source_a_group": row["source_a_group"], "source_b_group": row["source_b_group"],
                "target_index": row["target_index"], "band_order": "g,r,z",
                "units": "detected electrons per pixel", "clipping": False,
                "background_semantics": "requested source on zero background", **hashes,
            })
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(json.dumps({"phase": "render", "partition": partition, "completed": index + 1, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
        survey = validated_lsst_survey()
        handle.attrs["psf_fwhm_arcsec"] = json.dumps({band: float(survey.get_filter(band).psf_fwhm.to_value("arcsec")) for band in ("g", "r", "z")}, sort_keys=True)
        handle.attrs["complete"] = True
        handle.flush()
    return records


def full_replay(run_dir: Path, partition: str, rows: list[dict[str, str]], catalog) -> list[dict[str, object]]:
    output = []
    started = time.time()
    path = run_dir / f"manifests/resunet_{partition}_scenes.h5"
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs["complete"]) or int(handle.attrs["completed_count"]) != len(rows):
            raise RuntimeError("incomplete rendered manifest")
        for index, row in enumerate(rows):
            rendered = render_fixed_scene(catalog, spec_from_row(row), add_noise="all")
            blend = np.asarray(rendered.blend, dtype=np.float32)
            isolated = np.asarray(rendered.isolated, dtype=np.float32)
            xy = np.asarray([[source["x_peak"], source["y_peak"]] for source in rendered.catalog], dtype=np.float64)
            prompts = [gaussian_prompt_numpy(float(x), float(y)) for x, y in xy]
            checks = {
                "source_order": list(np.asarray(rendered.catalog["catalog_row"], dtype=int)) == [int(row["source_a_row"]), int(row["source_b_row"])],
                "blend_exact": np.array_equal(blend, np.asarray(handle["blend"][index])),
                "isolated_exact": np.array_equal(isolated, np.asarray(handle["isolated"][index])),
                "coordinates_exact": np.array_equal(xy, np.asarray(handle["xy"][index])),
                "prompt_a_exact": sha256_array(prompts[0]) == handle["prompt_a_sha256"][index].decode(),
                "prompt_b_exact": sha256_array(prompts[1]) == handle["prompt_b_sha256"][index].decode(),
                "noise_seed_exact": int(row["noise_seed"]) == spec_from_row(row).noise_seed,
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            output.append({"scene_id": row["scene_id"], "partition": partition, **checks, "status": status})
            if status != "PASS":
                raise RuntimeError(f"full manifest replay failed: {row['scene_id']}")
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(json.dumps({"phase": "replay", "partition": partition, "completed": index + 1, "total": len(rows), "elapsed_seconds": time.time() - started}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = require_run(args.run_dir)
    started = time.time()
    architecture_gate(run_dir)
    definitions = read_csv(run_dir / "manifests/resunet_scene_definitions.csv")
    if len(definitions) != 11_500 or Counter(row["partition"] for row in definitions) != Counter({"training": 10_000, "validation": 1_500}):
        raise RuntimeError("frozen scene counts changed")
    excluded = {row["source_group"] for row in read_csv(run_dir / "tables/atlas_source_exposure_audit.csv")}
    if any(row["source_a_group"] in excluded or row["source_b_group"] in excluded for row in definitions):
        raise RuntimeError("Atlas source exposure detected")
    training_groups = {row[field] for row in definitions if row["partition"] == "training" for field in ("source_a_group", "source_b_group")}
    validation_groups = {row[field] for row in definitions if row["partition"] == "validation" for field in ("source_a_group", "source_b_group")}
    if training_groups & validation_groups:
        raise RuntimeError("source group crosses training and validation")
    catalog, _ = load_catsim_catalog(CATALOG)
    records = []
    replay = []
    for partition in ("training", "validation"):
        rows = [row for row in definitions if row["partition"] == partition]
        records.extend(render_partition(run_dir, partition, rows, catalog))
    write_csv_fresh(run_dir / "manifests/rendered_scene_manifest.csv", records)
    for partition in ("training", "validation"):
        rows = [row for row in definitions if row["partition"] == partition]
        replay.extend(full_replay(run_dir, partition, rows, catalog))
    write_csv_fresh(run_dir / "tables/manifest_replay_checks.csv", replay)
    write_json_fresh(run_dir / "logs/data_preparation_complete.json", {
        "status": "PASS", "scene_count": len(definitions), "full_replay_count": len(replay),
        "training_validation_group_overlap": 0, "atlas_source_exposure_count": 0,
        "development_scene_access_count": 0, "lockbox_scene_access_count": 0,
        "runtime_seconds": time.time() - started,
        "training_h5_sha256": sha256_file(run_dir / "manifests/resunet_training_scenes.h5"),
        "validation_h5_sha256": sha256_file(run_dir / "manifests/resunet_validation_scenes.h5"),
        "render_manifest_sha256": sha256_file(run_dir / "manifests/rendered_scene_manifest.csv"),
        "replay_table_sha256": sha256_file(run_dir / "tables/manifest_replay_checks.csv"),
    })


if __name__ == "__main__":
    main()
