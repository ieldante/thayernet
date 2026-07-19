#!/usr/bin/env python3
"""Generate one deterministic eight-scene THAYER-D3-PV1-A1 cache candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.thayer_select_prompt_ablation_common import gaussian_prompt_numpy
from src.canonical_tensor_hash import canonical_tensor_sha256
from src.d3_audit_layer_pv1 import canonical_json_bytes
from src.d3_tensor_hash_contract_r1 import CANONICAL_DTYPE, canonical_nchw_tensor_hash
from src.models_two_expert_decoder import SharedPromptEncoder
from src.output_parameterization import encoder_tensor_sha256


SCENE_IDS = (
    "pu_training_ordinary_00000", "pu_training_ordinary_00008",
    "pu_training_ordinary_00016", "pu_training_ordinary_00024",
    "pu_training_near_00000", "pu_training_near_00008",
    "pu_training_near_00016", "pu_training_near_00024",
)
SOURCE_INDICES = (0, 8, 16, 24, 12000, 12008, 12016, 12024)
TARGET_INDICES = (0, 8, 16, 24, 32, 40, 48, 56)
SCALES = np.asarray((611.9199829101562, 1805.8800048828125, 1854.199951171875), dtype=np.float32)
SCENES = REPO / "outputs/runs/thayer_multiple_hypotheses_20260712_190701/manifests/probabilistic_unet_training_scenes.h5"
TARGETS = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216/projection_targets/projected_target_sets_final.h5"
ENCODER = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth"
EXPECTED_INPUT_HASHES = {
    SCENES: "d6ca6f1cbcb136a075f0216460e5f6b2dcd5fefbb63894803b86069df4e5f48d",
    TARGETS: "d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181",
    ENCODER: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_contained(root: Path, target: Path) -> Path:
    root = root.resolve(strict=True)
    resolved = target.resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError("cache output escapes candidate root")
    return resolved


def prompt_arrays(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.stack([gaussian_prompt_numpy(float(row[0, 0]), float(row[0, 1])) for row in xy])[:, None]
    right = np.stack([gaussian_prompt_numpy(float(row[1, 0]), float(row[1, 1])) for row in xy])[:, None]
    return left.astype(np.float32), right.astype(np.float32)


def canonical_batch_hash(tensor: torch.Tensor, member: str, prompt: str) -> str:
    return canonical_nchw_tensor_hash(
        tensor,
        semantic_axis_order="NCHW",
        semantic_member_name=member,
        prompt_identity=prompt,
        expert_identity="not_applicable",
        band_order=None,
        canonical_dtype=CANONICAL_DTYPE,
        ordered_sample_ids=SCENE_IDS,
    ).canonical_semantic_tensor_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-relative", required=True)
    parser.add_argument("--generation-id", choices=("A", "B"), required=True)
    args = parser.parse_args()
    root = args.candidate_root.resolve(strict=True)
    output = ensure_contained(root, root / args.output_relative)
    output.mkdir(parents=True, exist_ok=False)
    (output / "scenes").mkdir()

    for path, expected in EXPECTED_INPUT_HASHES.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen cache input changed: {path}")
    with h5py.File(SCENES, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("scene input is incomplete")
        blends = np.asarray(handle["blend"][list(SOURCE_INDICES)], dtype=np.float32)
        xy = np.asarray(handle["xy"][list(SOURCE_INDICES)], dtype=np.float64)
    with h5py.File(TARGETS, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("target input is incomplete")
        targets = np.asarray(handle["targets_physical"][list(TARGET_INDICES)], dtype=np.float32)
    prompt_a, prompt_b = prompt_arrays(xy)
    normalized = np.ascontiguousarray(blends / SCALES[None, :, None, None])

    payload = torch.load(ENCODER, map_location="cpu", weights_only=False)
    checkpoint_state = payload["state_dict"]
    encoder = SharedPromptEncoder().cpu().eval()
    encoder_state = {name: checkpoint_state[name].detach().cpu() for name in encoder.state_dict()}
    encoder.load_state_dict(encoder_state, strict=True)
    if any(parameter.requires_grad is False for parameter in encoder.parameters()):
        raise RuntimeError("unexpected preexisting encoder parameter flag")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    blend_tensor = torch.from_numpy(normalized)
    prompt_a_tensor = torch.from_numpy(np.ascontiguousarray(prompt_a))
    prompt_b_tensor = torch.from_numpy(np.ascontiguousarray(prompt_b))
    with torch.no_grad():
        joined = encoder(
            torch.cat((blend_tensor, blend_tensor), dim=0),
            torch.cat((prompt_a_tensor, prompt_b_tensor), dim=0),
        )
    features = {
        "prompt_a": tuple(value[: len(SCENE_IDS)].detach().cpu().contiguous() for value in joined),
        "prompt_b": tuple(value[len(SCENE_IDS) :].detach().cpu().contiguous() for value in joined),
    }
    levels = ("enc1", "enc2", "bottleneck")
    batch_hashes = {
        f"{prompt}.{level}": canonical_batch_hash(tensor, f"cached_features.{level}", prompt)
        for prompt, values in features.items()
        for level, tensor in zip(levels, values)
    }
    scene_records = []
    for index, scene_id in enumerate(SCENE_IDS):
        member_hashes = {
            f"{prompt}.{level}": canonical_tensor_sha256(tensor[index], layout="CHW")
            for prompt, values in features.items()
            for level, tensor in zip(levels, values)
        }
        target_chw = np.ascontiguousarray(targets[index].reshape(24, 60, 60))
        target_hash = canonical_tensor_sha256(target_chw, layout="CHW")
        combined_feature_hash = hashlib.sha256(canonical_json_bytes(member_hashes)).hexdigest()
        scene_payload = {
            "schema_version": "thayer-d3-pv1-a1-scene-cache-v1",
            "scene_id": scene_id,
            "order_index": index,
            "prompt_a": tuple(value[index : index + 1] for value in features["prompt_a"]),
            "prompt_b": tuple(value[index : index + 1] for value in features["prompt_b"]),
            "target_physical": torch.from_numpy(targets[index : index + 1]),
            "encoder_tensor_sha256": encoder_tensor_sha256(type("Container", (), {"encoder": encoder})()),
        }
        scene_path = output / "scenes" / f"{index:02d}_{scene_id}.pt"
        torch.save(scene_payload, scene_path)
        scene_records.append({
            "scene_id": scene_id,
            "order_index": index,
            "source_hdf5_index": SOURCE_INDICES[index],
            "target_hdf5_index": TARGET_INDICES[index],
            "feature_shapes": {f"{prompt}.{level}": list(tensor[index : index + 1].shape) for prompt, values in features.items() for level, tensor in zip(levels, values)},
            "feature_dtypes": {f"{prompt}.{level}": str(tensor.dtype) for prompt, values in features.items() for level, tensor in zip(levels, values)},
            "canonical_member_feature_hashes": member_hashes,
            "canonical_scene_feature_sha256": combined_feature_hash,
            "target_shape": list(targets[index].shape),
            "target_dtype": str(targets.dtype),
            "canonical_scene_target_sha256": target_hash,
            "bundle_path": scene_path.relative_to(root).as_posix(),
            "bundle_file_sha256": sha256_file(scene_path),
        })
    target_batch_chw = torch.from_numpy(np.ascontiguousarray(targets.reshape(8, 24, 60, 60)))
    target_batch_hash = canonical_nchw_tensor_hash(
        target_batch_chw,
        semantic_axis_order="NCHW",
        semantic_member_name="targets.ordered_scene_prompt_slot_channel",
        prompt_identity="both_prompts",
        expert_identity="not_applicable",
        band_order=None,
        canonical_dtype=CANONICAL_DTYPE,
        ordered_sample_ids=SCENE_IDS,
    ).canonical_semantic_tensor_sha256
    complete_feature_hash = hashlib.sha256(canonical_json_bytes(batch_hashes)).hexdigest()
    cache_path = output / "ordered_cache.pt"
    target_path = output / "ordered_targets.npy"
    torch.save({
        "schema_version": "thayer-d3-pv1-a1-ordered-cache-v1",
        "scene_ids": SCENE_IDS,
        "prompt_a": features["prompt_a"],
        "prompt_b": features["prompt_b"],
        "encoder_tensor_sha256": encoder_tensor_sha256(type("Container", (), {"encoder": encoder})()),
    }, cache_path)
    with target_path.open("xb") as handle:
        np.save(handle, targets, allow_pickle=False)
    manifest = {
        "schema_version": "thayer-d3-pv1-a1-cache-generation-v1",
        "protocol_identifier": "THAYER-D3-PV1-A1",
        "generation_id": args.generation_id,
        "process_id": os.getpid(),
        "device_policy": "CPU_DETERMINISTIC_NO_GRAD",
        "scene_ids": list(SCENE_IDS),
        "source_hdf5_indices": list(SOURCE_INDICES),
        "target_hdf5_indices": list(TARGET_INDICES),
        "feature_batch_shapes": {f"{prompt}.{level}": list(tensor.shape) for prompt, values in features.items() for level, tensor in zip(levels, values)},
        "feature_batch_dtypes": {f"{prompt}.{level}": str(tensor.dtype) for prompt, values in features.items() for level, tensor in zip(levels, values)},
        "canonical_batch_feature_member_hashes": batch_hashes,
        "complete_ordered_batch_feature_sha256": complete_feature_hash,
        "target_batch_shape": list(targets.shape),
        "target_batch_dtype": str(targets.dtype),
        "complete_ordered_batch_target_sha256": target_batch_hash,
        "encoder_checkpoint_sha256": EXPECTED_INPUT_HASHES[ENCODER],
        "encoder_tensor_sha256": encoder_tensor_sha256(type("Container", (), {"encoder": encoder})()),
        "normalization_scales_grz": SCALES.tolist(),
        "normalization_clipping": False,
        "optimizer_constructions": 0,
        "backward_passes": 0,
        "optimizer_steps": 0,
        "scenes": scene_records,
        "ordered_cache_path": cache_path.relative_to(root).as_posix(),
        "ordered_cache_file_sha256": sha256_file(cache_path),
        "ordered_targets_path": target_path.relative_to(root).as_posix(),
        "ordered_targets_file_sha256": sha256_file(target_path),
        "status": "PASS",
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PASS", "generation_id": args.generation_id,
        "feature_sha256": complete_feature_hash, "target_sha256": target_batch_hash,
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
