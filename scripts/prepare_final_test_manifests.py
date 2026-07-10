"""Prepare locked, metadata-only final-test manifests.

This script performs no training, model loading, inference, metric comparison,
or image rendering.  It freezes source identities and deterministic synthetic
blend parameters for later evaluation after all model choices are complete.

Every invocation creates a new timestamped directory under ``outputs/runs`` and
refuses to overwrite an existing path.  The emitted manifests contain no image
arrays.  CSV/JSON pairs, per-sample fingerprints, SHA-256 checksums, and
read-only file modes make later accidental mutation detectable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy.ndimage import binary_dilation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import blend as gd_blend
from src import utils as gd_utils


GENERATOR_VERSION = "thayer_provisional_grouped_final_manifest_v2"
SCHEMA_VERSION = "thayer_provisional_grouped_final_manifest_schema_v2"
MANIFEST_ROLE = "provisional_locked_manifest_prep"
SOURCE_POOL_SELECTION_METHOD = (
    "seed42_test_tail_exact_ra_dec_group_exclusion_first_representative_v1"
)
PROVISIONAL_REASON = (
    "Exact RA/Dec coordinate groups are excluded across train, validation, and "
    "the development-test prefix, but the pending exact-image/perceptual "
    "near-duplicate audit has not yet been applied. Regenerate or independently "
    "verify before final paper evaluation."
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# These suite seeds were chosen specifically for the locked final manifests.
# They are intentionally far outside the seed/offset families used by the
# current development evaluations (42-based offsets and 20260708-based stress
# seeds).  Do not reuse them for development evaluation or training.
FINAL_SUITE_SEEDS: dict[str, int] = {
    "normal_final_test": 910_300_101,
    "hard_stress_final_test": 910_300_211,
    "compact_bright_final_test": 910_300_307,
    "high_core_obstruction_final_test": 910_300_401,
    "halo_artifact_stress_final_test": 910_300_503,
}

FOREGROUND_MASK_PARAMETERS: dict[str, Any] = {
    "foreground_mask_algorithm": "src.blend.estimate_central_source_mask",
    "background_border_width": 20,
    "central_mask_dilation_iters": 14,
    "central_mask_soft_sigma": 8.0,
    "central_mask_aperture_radius": 120.0,
    "central_mask_aperture_soft_edge": 40.0,
    "central_mask_output_cutoff": 0.008,
    "foreground_intensity_cutoff": 0.002,
    "source_area_mask_threshold": 0.1,
}

EVALUATION_MASK_PARAMETERS: dict[str, Any] = {
    "affected_mask_threshold": 0.02,
    "core_aperture_fraction": 0.18,
    "core_percentile": 85.0,
    "halo_dilation_iters": 5,
}


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    display_name: str
    seed: int
    source_count: int
    max_shift: int
    brightness_range: tuple[float, float]
    blur_range: tuple[float, float]
    noise_range: tuple[float, float]
    rotation_range: tuple[float, float] = (0.0, 0.0)
    min_size_ratio: float | None = None
    max_size_ratio: float | None = None
    min_mask_fraction: float | None = None
    min_core_obstruction: float | None = None
    relaxed_min_size_ratio: float | None = None
    max_attempt_multiplier: int = 1
    relaxation_order: str = "none"
    artifact_sampling_mode: str = "none"


MANIFEST_COLUMNS = [
    "sample_id",
    "suite",
    "suite_display_name",
    "manifest_role",
    "source_split_name",
    "source_pool_name",
    "source_pool_selection_method",
    "source_pool_split_offset",
    "source_pool_size",
    "target_source_index",
    "contaminant_source_index",
    "contaminant_source_indices",
    "target_coordinate_group_sha256",
    "contaminant_coordinate_group_sha256",
    "target_class_label",
    "contaminant_class_label",
    "suite_random_seed",
    "sample_random_seed",
    "numpy_bit_generator",
    "generator_attempt",
    "accepted_sample_index",
    "shift_dx_pixels",
    "shift_dy_pixels",
    "shift_l1_pixels",
    "shift_l2_pixels",
    "size_ratio",
    "brightness_scale",
    "blur_sigma",
    "noise_std",
    "rotation_enabled",
    "rotation_angle_degrees",
    "generation_difficulty",
    "target_area_pixels",
    "target_radius_pixels",
    "contaminant_area_pixels",
    "contaminant_radius_pixels",
    "affected_mask_threshold",
    "affected_mask_fraction",
    "core_aperture_fraction",
    "core_percentile",
    "core_obstruction_fraction",
    "identity_affected_mae",
    "blend_severity_score",
    "halo_dilation_iters",
    "halo_band_fraction",
    "foreground_mask_algorithm",
    "background_border_width",
    "central_mask_dilation_iters",
    "central_mask_soft_sigma",
    "central_mask_aperture_radius",
    "central_mask_aperture_soft_edge",
    "central_mask_output_cutoff",
    "foreground_intensity_cutoff",
    "source_area_mask_threshold",
    "sampling_max_shift",
    "sampling_brightness_low",
    "sampling_brightness_high",
    "sampling_blur_low",
    "sampling_blur_high",
    "sampling_noise_low",
    "sampling_noise_high",
    "sampling_rotation_low",
    "sampling_rotation_high",
    "constraint_min_size_ratio",
    "constraint_max_size_ratio",
    "constraint_min_mask_fraction",
    "constraint_min_core_obstruction",
    "generation_constraints_met",
    "generation_relaxed",
    "source_artifact_flags_available",
    "source_artifact_filter_applied",
    "artifact_sampling_mode",
    "exact_coordinate_grouping_audited",
    "perceptual_duplicate_audit_applied",
    "manifest_provisional_reason",
    "generator_version",
    "generator_combined_sha256",
    "sample_fingerprint_sha256",
]

FORBIDDEN_ARRAY_KEYS = {
    "image",
    "images",
    "target",
    "contaminant",
    "blended",
    "pixels",
    "array",
    "raw_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create locked final-test manifests without model inference."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--n-per-suite", type=int, default=1000)
    parser.add_argument(
        "--development-test-source-count",
        type=int,
        default=1000,
        help="Test-split prefix already used by standard development evaluations.",
    )
    parser.add_argument(
        "--locked-source-count",
        type=int,
        default=1000,
        help="Number of test-split sources reserved after the development prefix.",
    )
    parser.add_argument(
        "--hard-source-count",
        type=int,
        default=800,
        help="Locked source-pool prefix used by hard stress, matching dev protocol size.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def make_run_dir(output_root: Path, stamp: str) -> Path:
    if not stamp or Path(stamp).name != stamp or stamp in {".", ".."}:
        raise ValueError("stamp must be a non-empty filename component.")
    run_dir = output_root / "runs" / f"final_test_manifest_prep_{stamp}"
    try:
        run_dir.resolve().relative_to((PROJECT_ROOT / "outputs/runs").resolve())
    except ValueError as exc:
        raise ValueError(
            "Manifest output must remain under ignored outputs/runs/."
        ) from exc
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for child in ("manifests", "tables", "diagnostics", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    return run_dir


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def safe_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return canonical_json(value)
    if value is None:
        return ""
    return value


def safe_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != MANIFEST_COLUMNS:
                raise ValueError("Manifest row keys do not match the locked schema.")
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def split_indices(
    n_samples: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ``src.data.split_dataset`` while retaining global indices."""
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def coordinate_group_id(ra: float, dec: float) -> str:
    """Hash an exact finite float64 RA/Dec pair without rounding."""
    ra_value = float(ra)
    dec_value = float(dec)
    if not math.isfinite(ra_value) or not math.isfinite(dec_value):
        return ""
    exact_key = f"{ra_value.hex()}|{dec_value.hex()}"
    return hashlib.sha256(exact_key.encode("ascii")).hexdigest()


def load_source_pool(
    dataset_path: Path,
    split_seed: int,
    train_frac: float,
    val_frac: float,
    development_count: int,
    locked_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Build a coordinate-group-isolated test pool and load it without rendering."""
    with h5py.File(dataset_path, "r") as handle:
        required = ("images", "ans", "ra", "dec")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(
                "Coordinate-group-aware manifests require datasets: "
                + ", ".join(missing)
            )
        n_samples = int(handle["images"].shape[0])
        train_idx, val_idx, test_idx = split_indices(
            n_samples, split_seed, train_frac, val_frac
        )
        if development_count + locked_count > len(test_idx):
            raise ValueError(
                f"Requested at least {locked_count} locked sources after a "
                f"{development_count}-source development prefix, but "
                f"the test split contains only {len(test_idx)} sources."
            )
        development_idx = test_idx[:development_count].copy()
        candidate_tail_idx = test_idx[development_count:].copy()
        ra = handle["ra"][:]
        dec = handle["dec"][:]
        all_group_ids = np.asarray(
            [coordinate_group_id(ra[i], dec[i]) for i in range(n_samples)],
            dtype="U64",
        )

        reference_idx = np.concatenate([train_idx, val_idx, development_idx])
        blocked_group_ids = {
            str(all_group_ids[int(index)])
            for index in reference_idx
            if str(all_group_ids[int(index)])
        }
        eligible_representatives: list[int] = []
        selected_group_ids: set[str] = set()
        exclusion_counts = {
            "nonfinite_ra_dec": 0,
            "coordinate_group_present_in_train_validation_or_development": 0,
            "later_duplicate_within_candidate_tail": 0,
        }
        for index_value in candidate_tail_idx:
            index = int(index_value)
            group_id = str(all_group_ids[index])
            if not group_id:
                exclusion_counts["nonfinite_ra_dec"] += 1
            elif group_id in blocked_group_ids:
                exclusion_counts[
                    "coordinate_group_present_in_train_validation_or_development"
                ] += 1
            elif group_id in selected_group_ids:
                exclusion_counts["later_duplicate_within_candidate_tail"] += 1
            else:
                selected_group_ids.add(group_id)
                eligible_representatives.append(index)
        if len(eligible_representatives) < locked_count:
            raise RuntimeError(
                f"Only {len(eligible_representatives)} coordinate-group-isolated "
                f"test-tail sources remain; {locked_count} were requested."
            )
        locked_idx = np.asarray(
            eligible_representatives[:locked_count], dtype=np.int64
        )
        locked_group_ids = all_group_ids[locked_idx].copy()

        group_members: dict[str, list[int]] = {}
        for index, group_id_value in enumerate(all_group_ids):
            group_id = str(group_id_value)
            if group_id:
                group_members.setdefault(group_id, []).append(index)
        split_role = np.full(n_samples, "unassigned", dtype="U16")
        split_role[train_idx] = "train"
        split_role[val_idx] = "validation"
        split_role[test_idx] = "test"
        cross_split_groups = 0
        for members in group_members.values():
            if len({str(split_role[index]) for index in members}) > 1:
                cross_split_groups += 1
        assignment_records = [
            {
                "global_source_index": index,
                "coordinate_group_sha256": str(all_group_ids[index]),
                "historical_seed42_split": str(split_role[index]),
            }
            for index in range(n_samples)
        ]
        coordinate_assignment_hash = hashlib.sha256(
            canonical_json(assignment_records).encode("utf-8")
        ).hexdigest()

        sort_order = np.argsort(locked_idx)
        sorted_indices = locked_idx[sort_order]
        images_sorted = handle["images"][sorted_indices]
        labels_sorted = handle["ans"][sorted_indices]
        inverse = np.empty_like(sort_order)
        inverse[sort_order] = np.arange(len(sort_order))
        images = images_sorted[inverse]
        labels = labels_sorted[inverse]
        metadata_keys = sorted(key for key in ("ra", "dec", "redshift", "pxscale") if key in handle)
        dataset_shape = [int(x) for x in handle["images"].shape]
        dataset_dtype = str(handle["images"].dtype)

    split_audit = {
        "dataset_path": project_relative(dataset_path),
        "dataset_size_bytes": int(dataset_path.stat().st_size),
        "dataset_images_shape": dataset_shape,
        "dataset_images_dtype": dataset_dtype,
        "metadata_keys_present": metadata_keys,
        "metadata_keys_used_for_coordinate_grouping": ["ra", "dec"],
        "coordinate_grouping_method": "exact_finite_float64_ra_dec_pair_python_hex_sha256_v1",
        "source_pool_selection_method": SOURCE_POOL_SELECTION_METHOD,
        "coordinate_group_assignment_sha256": coordinate_assignment_hash,
        "coordinate_groups_total": int(len(group_members)),
        "coordinate_duplicate_groups_total": int(
            sum(len(members) > 1 for members in group_members.values())
        ),
        "coordinate_groups_crossing_historical_seed42_splits": int(
            cross_split_groups
        ),
        "split_seed": split_seed,
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_index_sha256": {
            "train": array_sha256(train_idx),
            "validation": array_sha256(val_idx),
            "test": array_sha256(test_idx),
        },
        "development_test_prefix_count": development_count,
        "development_test_prefix_sha256": array_sha256(development_idx),
        "locked_test_pool_split_offset": development_count,
        "candidate_test_tail_count": int(len(candidate_tail_idx)),
        "coordinate_group_eligible_representatives": int(
            len(eligible_representatives)
        ),
        "coordinate_group_exclusion_counts": exclusion_counts,
        "locked_test_pool_count": locked_count,
        "locked_test_pool_global_index_sha256": array_sha256(locked_idx),
        "locked_test_pool_coordinate_group_sha256": array_sha256(
            locked_group_ids.astype("S64")
        ),
        "locked_coordinate_groups_unique": bool(
            len(set(str(x) for x in locked_group_ids)) == len(locked_group_ids)
        ),
        "train_validation_overlap": int(
            len(np.intersect1d(train_idx, val_idx, assume_unique=True))
        ),
        "train_test_overlap": int(
            len(np.intersect1d(train_idx, test_idx, assume_unique=True))
        ),
        "validation_test_overlap": int(
            len(np.intersect1d(val_idx, test_idx, assume_unique=True))
        ),
        "locked_pool_train_overlap": int(
            len(np.intersect1d(locked_idx, train_idx, assume_unique=True))
        ),
        "locked_pool_validation_overlap": int(
            len(np.intersect1d(locked_idx, val_idx, assume_unique=True))
        ),
        "locked_pool_development_prefix_overlap": int(
            len(np.intersect1d(locked_idx, development_idx, assume_unique=True))
        ),
        "locked_coordinate_group_train_overlap": int(
            len(set(str(x) for x in locked_group_ids) & set(str(x) for x in all_group_ids[train_idx]))
        ),
        "locked_coordinate_group_validation_overlap": int(
            len(set(str(x) for x in locked_group_ids) & set(str(x) for x in all_group_ids[val_idx]))
        ),
        "locked_coordinate_group_development_prefix_overlap": int(
            len(set(str(x) for x in locked_group_ids) & set(str(x) for x in all_group_ids[development_idx]))
        ),
        "provisional_status": MANIFEST_ROLE,
        "perceptual_duplicate_audit_applied": False,
        "provisional_reason": PROVISIONAL_REASON,
    }
    return images, labels, locked_idx, locked_group_ids, all_group_ids, split_audit


def suite_specs(config: dict[str, Any], hard_source_count: int) -> list[SuiteSpec]:
    normal = config["blending"]
    return [
        SuiteSpec(
            name="normal_final_test",
            display_name="Normal final test",
            seed=FINAL_SUITE_SEEDS["normal_final_test"],
            source_count=-1,
            max_shift=int(normal["max_shift"]),
            brightness_range=tuple(float(x) for x in normal["brightness_range"]),
            blur_range=tuple(float(x) for x in normal["blur_range"]),
            noise_range=tuple(float(x) for x in normal["noise_range"]),
            rotation_range=tuple(float(x) for x in normal["rotation_range"]),
        ),
        SuiteSpec(
            name="hard_stress_final_test",
            display_name="Hard stress final test",
            seed=FINAL_SUITE_SEEDS["hard_stress_final_test"],
            source_count=hard_source_count,
            max_shift=18,
            brightness_range=(0.8, 1.4),
            blur_range=(0.0, 0.15),
            noise_range=(0.0, 0.006),
            min_size_ratio=0.75,
            min_mask_fraction=0.01,
            relaxed_min_size_ratio=0.5,
            max_attempt_multiplier=40,
            relaxation_order="core_mask_size",
        ),
        SuiteSpec(
            name="compact_bright_final_test",
            display_name="Compact bright contaminant final test",
            seed=FINAL_SUITE_SEEDS["compact_bright_final_test"],
            source_count=-1,
            max_shift=24,
            brightness_range=(1.35, 1.95),
            blur_range=(0.0, 0.08),
            noise_range=(0.0, 0.006),
            min_size_ratio=0.15,
            max_size_ratio=0.90,
            min_mask_fraction=0.004,
            max_attempt_multiplier=120,
            relaxation_order="compact_size_brightness_mask",
        ),
        SuiteSpec(
            name="high_core_obstruction_final_test",
            display_name="High core obstruction final test",
            seed=FINAL_SUITE_SEEDS["high_core_obstruction_final_test"],
            source_count=-1,
            max_shift=12,
            brightness_range=(0.9, 1.5),
            blur_range=(0.0, 0.12),
            noise_range=(0.0, 0.006),
            min_size_ratio=0.70,
            min_mask_fraction=0.01,
            min_core_obstruction=0.82,
            max_attempt_multiplier=140,
            relaxation_order="core_mask_brightness",
        ),
        SuiteSpec(
            name="halo_artifact_stress_final_test",
            display_name="Halo/artifact stress final test",
            seed=FINAL_SUITE_SEEDS["halo_artifact_stress_final_test"],
            source_count=-1,
            max_shift=28,
            brightness_range=(0.85, 1.45),
            blur_range=(0.05, 0.25),
            noise_range=(0.0, 0.006),
            min_size_ratio=0.70,
            min_mask_fraction=0.015,
            max_attempt_multiplier=100,
            relaxation_order="core_mask_brightness",
            artifact_sampling_mode="halo_oriented_proxy_no_source_artifact_flags",
        ),
    ]


def target_core_mask(
    target: np.ndarray,
    aperture_fraction: float,
    core_percentile: float,
) -> np.ndarray:
    gray = target.mean(axis=-1)
    height, width = gray.shape
    center_y, center_x = height / 2.0, width / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = aperture_fraction * min(height, width)
    aperture = np.hypot(y_grid - center_y, x_grid - center_x) <= radius
    values = gray[aperture]
    threshold = float(np.percentile(values, core_percentile))
    mask = aperture & (gray >= threshold)
    return mask if np.any(mask) else aperture


def finite_or_none(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def sample_seed(suite_seed: int, attempt: int) -> int:
    state = np.random.SeedSequence([suite_seed, attempt]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0])


def make_candidate(
    spec: SuiteSpec,
    attempt: int,
    source_images: np.ndarray,
    source_labels: np.ndarray,
    source_global_indices: np.ndarray,
    source_coordinate_group_ids: np.ndarray,
    source_pool_offset: int,
    generator_hash: str,
) -> dict[str, Any]:
    seed = sample_seed(spec.seed, attempt)
    rng = np.random.default_rng(seed)
    source_count = len(source_images) if spec.source_count < 0 else spec.source_count
    target_local, contaminant_local = rng.choice(
        source_count, size=2, replace=False
    ).astype(int)
    target = source_images[target_local].astype(np.float32) / 255.0
    contaminant = source_images[contaminant_local].astype(np.float32) / 255.0
    dx = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
    dy = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
    brightness = float(rng.uniform(*spec.brightness_range))
    blur_sigma = float(rng.uniform(*spec.blur_range))
    noise_std = float(rng.uniform(*spec.noise_range))
    rotation = (
        float(rng.uniform(*spec.rotation_range))
        if spec.rotation_range != (0.0, 0.0)
        else 0.0
    )
    blended, info = gd_blend.blend_pair(
        target=target,
        contaminant=contaminant,
        shift=(dx, dy),
        rotation=rotation,
        brightness=brightness,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        rng=rng,
    )
    affected_threshold = float(EVALUATION_MASK_PARAMETERS["affected_mask_threshold"])
    affected = gd_utils.affected_region_mask(
        target, blended, threshold=affected_threshold
    )
    core = target_core_mask(
        target,
        float(EVALUATION_MASK_PARAMETERS["core_aperture_fraction"]),
        float(EVALUATION_MASK_PARAMETERS["core_percentile"]),
    )
    core_obstruction = (
        float(np.logical_and(affected, core).sum() / core.sum())
        if np.any(core)
        else 0.0
    )
    halo = binary_dilation(
        affected,
        iterations=int(EVALUATION_MASK_PARAMETERS["halo_dilation_iters"]),
    ) & ~affected
    mask_fraction = float(affected.mean())
    identity_affected_mae = (
        gd_utils.masked_mae(blended, target, affected)
        if np.any(affected)
        else None
    )
    severity = (
        mask_fraction * float(identity_affected_mae) * (1.0 + core_obstruction)
        if identity_affected_mae is not None
        else None
    )
    size_ratio = finite_or_none(info["size_ratio"])

    finite_size = size_ratio is not None
    size_ok = (
        True
        if spec.min_size_ratio is None
        else finite_size and size_ratio >= spec.min_size_ratio
    )
    if spec.max_size_ratio is not None:
        size_ok = size_ok and finite_size and size_ratio <= spec.max_size_ratio
    mask_ok = (
        True
        if spec.min_mask_fraction is None
        else mask_fraction >= spec.min_mask_fraction
    )
    core_ok = (
        True
        if spec.min_core_obstruction is None
        else core_obstruction >= spec.min_core_obstruction
    )
    constraints_met = bool(size_ok and mask_ok and core_ok)
    relaxed_ok = bool(mask_ok and finite_size)
    if spec.relaxed_min_size_ratio is not None:
        relaxed_ok = relaxed_ok and size_ratio >= spec.relaxed_min_size_ratio

    row: dict[str, Any] = {
        "sample_id": "",
        "suite": spec.name,
        "suite_display_name": spec.display_name,
        "manifest_role": MANIFEST_ROLE,
        "source_split_name": "test",
        "source_pool_name": "coordinate_group_isolated_test_tail",
        "source_pool_selection_method": SOURCE_POOL_SELECTION_METHOD,
        "source_pool_split_offset": int(source_pool_offset),
        "source_pool_size": int(source_count),
        "target_source_index": int(source_global_indices[target_local]),
        "contaminant_source_index": int(source_global_indices[contaminant_local]),
        "contaminant_source_indices": [
            int(source_global_indices[contaminant_local])
        ],
        "target_coordinate_group_sha256": str(
            source_coordinate_group_ids[target_local]
        ),
        "contaminant_coordinate_group_sha256": str(
            source_coordinate_group_ids[contaminant_local]
        ),
        "target_class_label": int(source_labels[target_local]),
        "contaminant_class_label": int(source_labels[contaminant_local]),
        "suite_random_seed": int(spec.seed),
        "sample_random_seed": seed,
        "numpy_bit_generator": "PCG64",
        "generator_attempt": int(attempt),
        "accepted_sample_index": -1,
        "shift_dx_pixels": dx,
        "shift_dy_pixels": dy,
        "shift_l1_pixels": int(abs(dx) + abs(dy)),
        "shift_l2_pixels": float(math.hypot(dx, dy)),
        "size_ratio": size_ratio,
        "brightness_scale": brightness,
        "blur_sigma": blur_sigma,
        "noise_std": noise_std,
        "rotation_enabled": bool(abs(rotation) > 1e-12),
        "rotation_angle_degrees": rotation,
        "generation_difficulty": str(info["generation_difficulty"]),
        "target_area_pixels": float(info["target_area"]),
        "target_radius_pixels": float(info["target_radius"]),
        "contaminant_area_pixels": float(info["contaminant_area"]),
        "contaminant_radius_pixels": float(info["contaminant_radius"]),
        "affected_mask_threshold": affected_threshold,
        "affected_mask_fraction": mask_fraction,
        "core_aperture_fraction": float(
            EVALUATION_MASK_PARAMETERS["core_aperture_fraction"]
        ),
        "core_percentile": float(EVALUATION_MASK_PARAMETERS["core_percentile"]),
        "core_obstruction_fraction": core_obstruction,
        "identity_affected_mae": identity_affected_mae,
        "blend_severity_score": severity,
        "halo_dilation_iters": int(
            EVALUATION_MASK_PARAMETERS["halo_dilation_iters"]
        ),
        "halo_band_fraction": float(halo.mean()),
        **FOREGROUND_MASK_PARAMETERS,
        "sampling_max_shift": int(spec.max_shift),
        "sampling_brightness_low": float(spec.brightness_range[0]),
        "sampling_brightness_high": float(spec.brightness_range[1]),
        "sampling_blur_low": float(spec.blur_range[0]),
        "sampling_blur_high": float(spec.blur_range[1]),
        "sampling_noise_low": float(spec.noise_range[0]),
        "sampling_noise_high": float(spec.noise_range[1]),
        "sampling_rotation_low": float(spec.rotation_range[0]),
        "sampling_rotation_high": float(spec.rotation_range[1]),
        "constraint_min_size_ratio": spec.min_size_ratio,
        "constraint_max_size_ratio": spec.max_size_ratio,
        "constraint_min_mask_fraction": spec.min_mask_fraction,
        "constraint_min_core_obstruction": spec.min_core_obstruction,
        "generation_constraints_met": constraints_met,
        "generation_relaxed": False,
        "source_artifact_flags_available": False,
        "source_artifact_filter_applied": False,
        "artifact_sampling_mode": spec.artifact_sampling_mode,
        "exact_coordinate_grouping_audited": True,
        "perceptual_duplicate_audit_applied": False,
        "manifest_provisional_reason": PROVISIONAL_REASON,
        "generator_version": GENERATOR_VERSION,
        "generator_combined_sha256": generator_hash,
        "sample_fingerprint_sha256": "",
    }
    row["_relaxed_ok"] = relaxed_ok
    return row


def relaxation_key(spec: SuiteSpec, row: dict[str, Any]) -> tuple[float, ...]:
    if spec.relaxation_order == "compact_size_brightness_mask":
        return (
            -abs(float(row["size_ratio"]) - 0.55),
            float(row["brightness_scale"]),
            float(row["affected_mask_fraction"]),
        )
    if spec.relaxation_order == "core_mask_size":
        return (
            float(row["core_obstruction_fraction"]),
            float(row["affected_mask_fraction"]),
            float(row["size_ratio"]),
        )
    return (
        float(row["core_obstruction_fraction"]),
        float(row["affected_mask_fraction"]),
        float(row["brightness_scale"]),
    )


def finalize_row(row: dict[str, Any], accepted_index: int, relaxed: bool) -> dict[str, Any]:
    row = dict(row)
    row.pop("_relaxed_ok")
    row["accepted_sample_index"] = int(accepted_index)
    row["sample_id"] = f"{row['suite']}_{accepted_index:06d}"
    row["generation_relaxed"] = bool(relaxed)
    row["sample_fingerprint_sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in row.items() if key != "sample_fingerprint_sha256"}
        ).encode("utf-8")
    ).hexdigest()
    ordered = {key: row[key] for key in MANIFEST_COLUMNS}
    return ordered


def generate_suite(
    spec: SuiteSpec,
    n_samples: int,
    source_images: np.ndarray,
    source_labels: np.ndarray,
    source_global_indices: np.ndarray,
    source_coordinate_group_ids: np.ndarray,
    source_pool_offset: int,
    generator_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_attempts = max(n_samples, spec.max_attempt_multiplier * n_samples)
    accepted: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    attempt = 0
    while attempt < max_attempts and len(accepted) < n_samples:
        attempt += 1
        candidate = make_candidate(
            spec,
            attempt,
            source_images,
            source_labels,
            source_global_indices,
            source_coordinate_group_ids,
            source_pool_offset,
            generator_hash,
        )
        if spec.min_size_ratio is None and spec.min_mask_fraction is None:
            accepted.append(candidate)
        elif candidate["generation_constraints_met"]:
            accepted.append(candidate)
        elif candidate["_relaxed_ok"]:
            relaxed.append(candidate)

    strict_count = len(accepted)
    if len(accepted) < n_samples:
        relaxed.sort(key=lambda row: relaxation_key(spec, row), reverse=True)
        accepted.extend(relaxed[: n_samples - len(accepted)])
    if len(accepted) < n_samples:
        raise RuntimeError(
            f"Could not generate {n_samples} samples for {spec.name}; "
            f"got {len(accepted)} after {attempt} attempts."
        )
    rows = [
        finalize_row(row, index, relaxed=index >= strict_count)
        for index, row in enumerate(accepted[:n_samples])
    ]
    diag = {
        "suite": spec.name,
        "display_name": spec.display_name,
        "requested": n_samples,
        "accepted": len(rows),
        "strictly_accepted": strict_count,
        "relaxed_candidates_used": int(sum(row["generation_relaxed"] for row in rows)),
        "attempts": attempt,
        "suite_seed": spec.seed,
        "source_count": int(
            len(source_images) if spec.source_count < 0 else spec.source_count
        ),
        "mean_affected_mask_fraction": float(
            np.mean([row["affected_mask_fraction"] for row in rows])
        ),
        "mean_core_obstruction_fraction": float(
            np.mean([row["core_obstruction_fraction"] for row in rows])
        ),
        "mean_size_ratio": float(
            np.mean([row["size_ratio"] for row in rows if row["size_ratio"] is not None])
        ),
        "zero_affected_masks": int(
            sum(row["affected_mask_fraction"] == 0 for row in rows)
        ),
        "artifact_sampling_mode": spec.artifact_sampling_mode,
    }
    return rows, diag


def source_hashes(config_path: Path) -> tuple[dict[str, str], str]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/blend.py",
        PROJECT_ROOT / "src/data.py",
        PROJECT_ROOT / "src/utils.py",
        config_path,
    ]
    hashes = {project_relative(path): sha256_file(path) for path in paths}
    combined = hashlib.sha256(canonical_json(hashes).encode("utf-8")).hexdigest()
    return hashes, combined


def manifest_metadata(
    spec: SuiteSpec,
    n_samples: int,
    source_pool_offset: int,
    locked_source_count: int,
    split_audit: dict[str, Any],
    source_hash_map: dict[str, str],
    generator_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_combined_sha256": generator_hash,
        "generator_source_sha256": source_hash_map,
        "manifest_role": MANIFEST_ROLE,
        "paper_ready": False,
        "provisional_reason": PROVISIONAL_REASON,
        "suite": spec.name,
        "suite_display_name": spec.display_name,
        "sample_count": n_samples,
        "suite_seed": spec.seed,
        "source_split_name": "test",
        "source_pool_name": "coordinate_group_isolated_test_tail",
        "source_pool_selection_method": SOURCE_POOL_SELECTION_METHOD,
        "source_pool_split_offset": source_pool_offset,
        "locked_source_count": locked_source_count,
        "source_pool_global_index_sha256": split_audit[
            "locked_test_pool_global_index_sha256"
        ],
        "source_pool_coordinate_group_sha256": split_audit[
            "locked_test_pool_coordinate_group_sha256"
        ],
        "coordinate_group_assignment_sha256": split_audit[
            "coordinate_group_assignment_sha256"
        ],
        "coordinate_grouping_method": split_audit["coordinate_grouping_method"],
        "suite_spec": asdict(spec),
        "foreground_mask_parameters": FOREGROUND_MASK_PARAMETERS,
        "evaluation_mask_parameters": EVALUATION_MASK_PARAMETERS,
        "contains_raw_image_arrays": False,
        "qualitative_examples_inspected": False,
        "model_inference_performed": False,
        "exact_coordinate_grouping_audited": True,
        "perceptual_duplicate_audit_applied": False,
        "use_restriction": (
            "Do not use this provisional manifest for model selection or final paper "
            "evaluation. First apply or independently verify the pending exact-image "
            "and perceptual near-duplicate exclusions, then regenerate or formally "
            "approve the same source pool."
        ),
    }


def schema_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_columns_in_order": MANIFEST_COLUMNS,
        "row_count_policy": "exactly the configured n_per_suite",
        "null_policy": (
            "Only optional constraints and non-finite size-derived fields may be null; "
            "the generated locked manifests are validated to have finite size and nonempty affected masks."
        ),
        "raw_array_policy": "No image, pixel, tensor, mask-array, or raw cutout fields are permitted.",
        "identity_fields": [
            "sample_id",
            "suite",
            "target_source_index",
            "contaminant_source_index",
            "suite_random_seed",
            "sample_random_seed",
            "sample_fingerprint_sha256",
        ],
        "source_group_fields": [
            "source_pool_selection_method",
            "target_coordinate_group_sha256",
            "contaminant_coordinate_group_sha256",
            "exact_coordinate_grouping_audited",
            "perceptual_duplicate_audit_applied",
        ],
        "blend_parameter_fields": [
            "shift_dx_pixels",
            "shift_dy_pixels",
            "size_ratio",
            "brightness_scale",
            "blur_sigma",
            "noise_std",
            "rotation_enabled",
            "rotation_angle_degrees",
        ],
        "evaluation_fields": [
            "affected_mask_threshold",
            "affected_mask_fraction",
            "core_obstruction_fraction",
            "blend_severity_score",
            "halo_band_fraction",
        ],
        "reproducibility_note": (
            "Each row's sample_random_seed independently reproduces source selection, "
            "blend parameter sampling, and noise realization with NumPy PCG64."
        ),
        "provisional_status": MANIFEST_ROLE,
        "paper_ready": False,
        "provisional_reason": PROVISIONAL_REASON,
    }


def validate_manifests(
    manifests: dict[str, list[dict[str, Any]]],
    manifest_paths: dict[str, dict[str, Path]],
    n_per_suite: int,
    locked_indices: np.ndarray,
    locked_coordinate_group_ids: np.ndarray,
    all_coordinate_group_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    development_idx: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    locked_set = set(int(x) for x in locked_indices)
    train_set = set(int(x) for x in train_idx)
    val_set = set(int(x) for x in val_idx)
    development_set = set(int(x) for x in development_idx)
    locked_group_set = set(str(x) for x in locked_coordinate_group_ids if str(x))
    train_group_set = set(
        str(all_coordinate_group_ids[int(x)])
        for x in train_idx
        if str(all_coordinate_group_ids[int(x)])
    )
    val_group_set = set(
        str(all_coordinate_group_ids[int(x)])
        for x in val_idx
        if str(all_coordinate_group_ids[int(x)])
    )
    development_group_set = set(
        str(all_coordinate_group_ids[int(x)])
        for x in development_idx
        if str(all_coordinate_group_ids[int(x)])
    )
    all_ids: list[str] = []
    all_fingerprints: list[str] = []
    all_sample_seeds: list[int] = []
    suite_rows: list[dict[str, Any]] = []

    for suite, rows in manifests.items():
        csv_path = manifest_paths[suite]["csv"]
        json_path = manifest_paths[suite]["json"]
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        with json_path.open("r", encoding="utf-8") as handle:
            json_payload = json.load(handle)
        json_rows = json_payload["samples"]
        ids = [str(row["sample_id"]) for row in rows]
        fingerprints = [str(row["sample_fingerprint_sha256"]) for row in rows]
        seeds = [int(row["sample_random_seed"]) for row in rows]
        source_indices = [
            int(index)
            for row in rows
            for index in (
                row["target_source_index"],
                row["contaminant_source_index"],
            )
        ]
        source_group_ids = [
            str(group_id)
            for row in rows
            for group_id in (
                row["target_coordinate_group_sha256"],
                row["contaminant_coordinate_group_sha256"],
            )
        ]
        signatures = [
            (
                row["target_source_index"],
                row["contaminant_source_index"],
                row["shift_dx_pixels"],
                row["shift_dy_pixels"],
                row["brightness_scale"],
                row["blur_sigma"],
                row["noise_std"],
                row["rotation_angle_degrees"],
            )
            for row in rows
        ]
        checks = {
            "suite": suite,
            "expected_rows": n_per_suite,
            "csv_rows": len(csv_rows),
            "json_rows": len(json_rows),
            "schema_column_count": len(MANIFEST_COLUMNS),
            "csv_schema_exact": bool(csv_rows and list(csv_rows[0].keys()) == MANIFEST_COLUMNS),
            "json_schema_exact": all(
                set(row.keys()) == set(MANIFEST_COLUMNS) for row in json_rows
            ),
            "sample_ids_unique": len(ids) == len(set(ids)),
            "sample_fingerprints_unique": len(fingerprints) == len(set(fingerprints)),
            "sample_seeds_unique": len(seeds) == len(set(seeds)),
            "generation_signatures_unique": len(signatures) == len(set(signatures)),
            "csv_json_fingerprint_parity": [
                row["sample_fingerprint_sha256"] for row in csv_rows
            ]
            == [row["sample_fingerprint_sha256"] for row in json_rows],
            "sources_in_locked_pool": all(index in locked_set for index in source_indices),
            "sources_outside_train": all(index not in train_set for index in source_indices),
            "sources_outside_validation": all(index not in val_set for index in source_indices),
            "sources_outside_development_prefix": all(
                index not in development_set for index in source_indices
            ),
            "coordinate_groups_in_locked_pool": all(
                group_id in locked_group_set for group_id in source_group_ids
            ),
            "coordinate_groups_outside_train": all(
                group_id not in train_group_set for group_id in source_group_ids
            ),
            "coordinate_groups_outside_validation": all(
                group_id not in val_group_set for group_id in source_group_ids
            ),
            "coordinate_groups_outside_development_prefix": all(
                group_id not in development_group_set
                for group_id in source_group_ids
            ),
            "coordinate_group_ids_match_global_indices": all(
                row["target_coordinate_group_sha256"]
                == str(all_coordinate_group_ids[int(row["target_source_index"])])
                and row["contaminant_coordinate_group_sha256"]
                == str(
                    all_coordinate_group_ids[
                        int(row["contaminant_source_index"])
                    ]
                )
                for row in rows
            ),
            "provisional_status_explicit": all(
                row["manifest_role"] == MANIFEST_ROLE
                and row["exact_coordinate_grouping_audited"] is True
                and row["perceptual_duplicate_audit_applied"] is False
                for row in rows
            ),
            "target_contaminant_distinct": all(
                row["target_source_index"] != row["contaminant_source_index"]
                for row in rows
            ),
            "nonempty_affected_masks": all(
                float(row["affected_mask_fraction"]) > 0 for row in rows
            ),
            "finite_size_ratios": all(row["size_ratio"] is not None for row in rows),
            "no_raw_array_columns": not any(
                key.lower() in FORBIDDEN_ARRAY_KEYS for key in MANIFEST_COLUMNS
            ),
            "raw_array_values_absent": all(
                not isinstance(value, np.ndarray)
                for row in rows
                for value in row.values()
            ),
        }
        checks["passed"] = all(
            value
            for key, value in checks.items()
            if key not in {"suite", "expected_rows", "csv_rows", "json_rows", "schema_column_count"}
        ) and len(rows) == n_per_suite and len(csv_rows) == n_per_suite and len(json_rows) == n_per_suite
        suite_rows.append(checks)
        all_ids.extend(ids)
        all_fingerprints.extend(fingerprints)
        all_sample_seeds.extend(seeds)

    combined = {
        "schema_version": SCHEMA_VERSION,
        "suite_count": len(manifests),
        "total_rows": sum(len(rows) for rows in manifests.values()),
        "all_sample_ids_unique": len(all_ids) == len(set(all_ids)),
        "all_sample_fingerprints_unique": len(all_fingerprints)
        == len(set(all_fingerprints)),
        "all_sample_seeds_unique": len(all_sample_seeds) == len(set(all_sample_seeds)),
        "train_validation_overlap": len(train_set & val_set),
        "locked_train_overlap": len(locked_set & train_set),
        "locked_validation_overlap": len(locked_set & val_set),
        "locked_development_prefix_overlap": len(locked_set & development_set),
        "locked_coordinate_groups_unique": len(locked_coordinate_group_ids)
        == len(locked_group_set),
        "locked_coordinate_group_train_overlap": len(
            locked_group_set & train_group_set
        ),
        "locked_coordinate_group_validation_overlap": len(
            locked_group_set & val_group_set
        ),
        "locked_coordinate_group_development_prefix_overlap": len(
            locked_group_set & development_group_set
        ),
        "all_suites_passed": all(row["passed"] for row in suite_rows),
        "manifest_role": MANIFEST_ROLE,
        "paper_ready": False,
        "perceptual_duplicate_audit_applied": False,
        "provisional_reason": PROVISIONAL_REASON,
    }
    combined["passed"] = bool(
        combined["all_suites_passed"]
        and combined["all_sample_ids_unique"]
        and combined["all_sample_fingerprints_unique"]
        and combined["all_sample_seeds_unique"]
        and combined["train_validation_overlap"] == 0
        and combined["locked_train_overlap"] == 0
        and combined["locked_validation_overlap"] == 0
        and combined["locked_development_prefix_overlap"] == 0
        and combined["locked_coordinate_groups_unique"]
        and combined["locked_coordinate_group_train_overlap"] == 0
        and combined["locked_coordinate_group_validation_overlap"] == 0
        and combined["locked_coordinate_group_development_prefix_overlap"] == 0
    )
    return suite_rows, combined


def write_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def protocol_markdown(
    run_dir: Path,
    suite_diagnostics: list[dict[str, Any]],
    split_audit: dict[str, Any],
    checksums: dict[str, str],
    generator_hash: str,
) -> str:
    suite_lines = [
        "| Suite | Samples | Seed | Sources | Attempts | Relaxed | Mean affected fraction | Mean core obstruction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite_diagnostics:
        suite_lines.append(
            "| {suite} | {accepted} | {seed} | {sources} | {attempts} | {relaxed} | {mask:.6f} | {core:.6f} |".format(
                suite=row["suite"],
                accepted=row["accepted"],
                seed=row["suite_seed"],
                sources=row["source_count"],
                attempts=row["attempts"],
                relaxed=row["relaxed_candidates_used"],
                mask=row["mean_affected_mask_fraction"],
                core=row["mean_core_obstruction_fraction"],
            )
        )
    checksum_lines = [f"- `{path}`: `{digest}`" for path, digest in checksums.items()]
    return f"""# Provisional Locked Final-Test Manifest Protocol

## Status

Status: `{MANIFEST_ROLE}` — **not ready for final paper evaluation**.

This run prepared frozen, metadata-only provisional manifests. It performed no
model loading, training, inference, model comparison, or image rendering. No
qualitative example was inspected. Exact RA/Dec coordinate-group leakage is
excluded, but the full exact-image/perceptual near-duplicate audit was still in
progress when these files were generated. The provisional source pool must be
regenerated with those exclusions or independently verified against the audit
before any final evaluation.

Run: `{project_relative(run_dir)}`

Generator version: `{GENERATOR_VERSION}`  
Combined generator/source hash: `{generator_hash}`

## Why this run exists

The current normal and hard-stress evaluations are development benchmarks: they
have already informed model choices. A final paper claim therefore needs a newly
frozen, leakage-audited manifest whose samples are not consulted during further
development. This run establishes the schema, deterministic seeds, coordinate
grouping, and append-only infrastructure, but it is deliberately provisional.

## Source reservation

The repository's canonical split shuffles global Galaxy10 indices with seed
`{split_audit['split_seed']}` before a 70/15/15 split. Existing scripted normal
evaluations use the first 1,000 sources in the resulting test split, while hard
stress uses the first 800 of that development pool. A parallel audit found
`{split_audit['coordinate_groups_crossing_historical_seed42_splits']}` exact
RA/Dec coordinate groups crossing the historical random-index splits. This run
therefore scans the test tail after position
`{split_audit['locked_test_pool_split_offset']}`, excludes every coordinate
group represented in train, validation, or the development-test prefix, keeps
only the first representative of a coordinate group, and freezes the first
`{split_audit['locked_test_pool_count']}` eligible representatives.

- Train/validation/test overlap: `0 / 0 / 0` across pairwise checks.
- Locked-pool overlap with train: `{split_audit['locked_pool_train_overlap']}`.
- Locked-pool overlap with validation: `{split_audit['locked_pool_validation_overlap']}`.
- Locked-pool overlap with the scripted development-test prefix: `{split_audit['locked_pool_development_prefix_overlap']}`.
- Locked coordinate-group overlap with train: `{split_audit['locked_coordinate_group_train_overlap']}`.
- Locked coordinate-group overlap with validation: `{split_audit['locked_coordinate_group_validation_overlap']}`.
- Locked coordinate-group overlap with the development prefix: `{split_audit['locked_coordinate_group_development_prefix_overlap']}`.
- Candidate tail sources excluded because their coordinate group appeared in a reference split: `{split_audit['coordinate_group_exclusion_counts']['coordinate_group_present_in_train_validation_or_development']}`.
- Locked source-index SHA-256: `{split_audit['locked_test_pool_global_index_sha256']}`.
- Locked coordinate-group SHA-256: `{split_audit['locked_test_pool_coordinate_group_sha256']}`.
- Coordinate-group assignment SHA-256: `{split_audit['coordinate_group_assignment_sha256']}`.

This exact-coordinate reservation is supported by the repository's current
scripted pipeline. It does not yet exclude exact-image duplicates with missing
or differing coordinates, perceptual near-duplicates, or access from unknown
external notebooks. Those limitations are why the status remains provisional.

## Provisionally frozen suites and seeds

The five suite seeds are new and deliberately outside the seed families used by
the current development scripts. Each accepted row also records a unique
`sample_random_seed`; rerunning that seed with NumPy PCG64 reproduces source
selection, blend parameters, and the noise realization independently of earlier
rows.

{os.linesep.join(suite_lines)}

The halo/artifact suite is a halo-oriented proxy using controlled blur and broad
contaminant settings. The local dataset has no validated source-artifact quality
flags, so this run did not claim to select known artifact-bearing sources and did
not silently invent such labels.

## Manifest contents

Each row stores global target and contaminant indices, source split and pool,
class labels, shifts, size ratio, brightness, blur, noise, rotation state,
foreground-mask constants, affected-mask threshold, core obstruction, blend
severity, suite and sample seeds, generator version/hash, and a sample
fingerprint. No raw image, blend, target, contaminant, tensor, or mask arrays are
stored.

Each row also stores hashed target/contaminant coordinate-group identifiers and
an explicit `perceptual_duplicate_audit_applied = false` flag.

The JSON manifests preserve native types; the CSV manifests provide portable
tabular copies. `manifest_schema.json` defines the locked column order.
`manifest_index.json` records counts and SHA-256 checksums. Manifest and schema
files are made read-only after validation; checksums remain the authoritative
tamper-evidence mechanism.

## Locking rules

1. Do not render, inspect, or rank qualitative examples from these manifests
   while designing models.
2. Do not train on any locked-pool source or generated sample.
3. Do not tune thresholds, clipping, masks, or selection criteria using final
   results.
4. Do not run final evaluation from these provisional files until the completed
   exact-image/perceptual audit has been applied and the pool regenerated or
   independently verified.
5. Report all predeclared suites and negative results; do not regenerate a suite
   because its metrics are unfavorable.
6. If any manifest file changes, the SHA-256 mismatch invalidates the final run.

## Manifest SHA-256 checksums

{os.linesep.join(checksum_lines)}
"""


def validation_markdown(
    suite_rows: list[dict[str, Any]],
    combined: dict[str, Any],
    read_only_paths: list[Path],
) -> str:
    rows = [
        "| Suite | CSV rows | JSON rows | Schema | IDs | Fingerprints | Seeds | Sources locked | Dev-prefix excluded | Passed |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in suite_rows:
        rows.append(
            "| {suite} | {csv_rows} | {json_rows} | {schema} | {ids} | {fingerprints} | {seeds} | {locked} | {dev} | {passed} |".format(
                suite=row["suite"],
                csv_rows=row["csv_rows"],
                json_rows=row["json_rows"],
                schema="pass" if row["csv_schema_exact"] and row["json_schema_exact"] else "fail",
                ids="pass" if row["sample_ids_unique"] else "fail",
                fingerprints="pass" if row["sample_fingerprints_unique"] else "fail",
                seeds="pass" if row["sample_seeds_unique"] else "fail",
                locked="pass" if row["sources_in_locked_pool"] else "fail",
                dev="pass" if row["sources_outside_development_prefix"] else "fail",
                passed="PASS" if row["passed"] else "FAIL",
            )
        )
    modes = [oct(stat.S_IMODE(path.stat().st_mode)) for path in read_only_paths]
    return f"""# Provisional Final-Test Manifest Schema and Validation

Mechanical validation: `{'PASS' if combined['passed'] else 'FAIL'}`  
Manifest role: `{combined['manifest_role']}`  
Ready for final paper evaluation: `{combined['paper_ready']}`

{os.linesep.join(rows)}

Combined checks:

- Suites: `{combined['suite_count']}`.
- Total rows: `{combined['total_rows']}`.
- Sample IDs unique across suites: `{combined['all_sample_ids_unique']}`.
- Sample fingerprints unique across suites: `{combined['all_sample_fingerprints_unique']}`.
- Per-sample seeds unique across suites: `{combined['all_sample_seeds_unique']}`.
- Locked/train overlap: `{combined['locked_train_overlap']}`.
- Locked/validation overlap: `{combined['locked_validation_overlap']}`.
- Locked/development-prefix overlap: `{combined['locked_development_prefix_overlap']}`.
- Locked coordinate groups unique: `{combined['locked_coordinate_groups_unique']}`.
- Locked coordinate-group/train overlap: `{combined['locked_coordinate_group_train_overlap']}`.
- Locked coordinate-group/validation overlap: `{combined['locked_coordinate_group_validation_overlap']}`.
- Locked coordinate-group/development-prefix overlap: `{combined['locked_coordinate_group_development_prefix_overlap']}`.
- Exact-coordinate grouping applied: `True`.
- Exact-image/perceptual near-duplicate exclusions applied: `{combined['perceptual_duplicate_audit_applied']}`.
- Raw image arrays stored: `False`.
- Manifest/schema read-only files checked: `{len(read_only_paths)}`.
- Read-only modes: `{', '.join(sorted(set(modes)))}`.

CSV/JSON row counts, column order, fingerprints, source membership, split
exclusion, target/contaminant distinction, nonempty affected masks, finite size
ratios, coordinate-group isolation, and generation-signature uniqueness were
validated programmatically. Mechanical `PASS` does not override the provisional
status: `{combined['provisional_reason']}`
"""


def main() -> int:
    args = parse_args()
    if args.n_per_suite <= 0:
        raise ValueError("n-per-suite must be positive.")
    if min(
        args.development_test_source_count,
        args.locked_source_count,
        args.hard_source_count,
    ) <= 1:
        raise ValueError("Source counts must be greater than one.")
    if args.hard_source_count > args.locked_source_count:
        raise ValueError("hard-source-count cannot exceed locked-source-count.")

    config_path = resolve(args.config)
    output_root = resolve(args.output_root)
    config = load_config(config_path)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(output_root, stamp)

    source_hash_map, generator_hash = source_hashes(config_path)
    dataset_path = resolve(Path(config["dataset_path"]))
    dataset_provenance = {
        "path": project_relative(dataset_path),
        "size_bytes": int(dataset_path.stat().st_size),
        "sha256": sha256_file(dataset_path),
    }
    dependency_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "pyyaml": importlib_metadata.version("PyYAML"),
        "scipy": importlib_metadata.version("scipy"),
    }
    print("Mode: metadata-only manifest generation (no model device required).", flush=True)
    print("Model loading/training/inference: disabled.", flush=True)
    print("Qualitative image inspection/rendering: disabled.", flush=True)
    print(f"Manifest role: {MANIFEST_ROLE} (not paper-ready).", flush=True)
    print(f"Run directory: {project_relative(run_dir)}", flush=True)
    print(
        "Building an exact-RA/Dec-group-isolated test-tail source pool.",
        flush=True,
    )
    (
        source_images,
        source_labels,
        locked_indices,
        locked_coordinate_group_ids,
        all_coordinate_group_ids,
        split_audit,
    ) = load_source_pool(
        dataset_path=dataset_path,
        split_seed=int(config["seed"]),
        train_frac=float(config["splits"]["train_frac"]),
        val_frac=float(config["splits"]["val_frac"]),
        development_count=args.development_test_source_count,
        locked_count=args.locked_source_count,
    )
    split_audit["dataset_provenance"] = dataset_provenance
    safe_write_json(run_dir / "logs/split_and_source_pool_audit.json", split_audit)
    safe_write_json(
        run_dir / "logs/generator_source_hashes.json",
        {
            "generator_version": GENERATOR_VERSION,
            "generator_source_sha256": source_hash_map,
            "generator_combined_sha256": generator_hash,
            "dataset_provenance": dataset_provenance,
            "dependency_versions": dependency_versions,
        },
    )

    specs = suite_specs(config, args.hard_source_count)
    safe_write_json(
        run_dir / "logs/generation_config.json",
        {
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "manifest_role": MANIFEST_ROLE,
            "paper_ready": False,
            "provisional_reason": PROVISIONAL_REASON,
            "source_pool_selection_method": SOURCE_POOL_SELECTION_METHOD,
            "n_per_suite": args.n_per_suite,
            "development_test_source_count": args.development_test_source_count,
            "locked_source_count": args.locked_source_count,
            "hard_source_count": args.hard_source_count,
            "suite_specs": [asdict(spec) for spec in specs],
            "foreground_mask_parameters": FOREGROUND_MASK_PARAMETERS,
            "evaluation_mask_parameters": EVALUATION_MASK_PARAMETERS,
            "development_seed_families_not_reused": [
                "split seed 42",
                "42-based evaluation offsets (including 4042 and 9042-9044)",
                "v0.3/ResUNet offset families near 4,500-5,500",
                "stress seed 20260708 and 20260808-20260810",
            ],
            "model_inference_performed": False,
            "qualitative_examples_inspected": False,
        },
    )

    manifests: dict[str, list[dict[str, Any]]] = {}
    manifest_paths: dict[str, dict[str, Path]] = {}
    suite_diagnostics: list[dict[str, Any]] = []
    for spec in specs:
        print(
            f"Generating provisional locked metadata for {spec.name} with seed {spec.seed}.",
            flush=True,
        )
        rows, diag = generate_suite(
            spec=spec,
            n_samples=args.n_per_suite,
            source_images=source_images,
            source_labels=source_labels,
            source_global_indices=locked_indices,
            source_coordinate_group_ids=locked_coordinate_group_ids,
            source_pool_offset=args.development_test_source_count,
            generator_hash=generator_hash,
        )
        manifests[spec.name] = rows
        suite_diagnostics.append(diag)
        csv_path = run_dir / "manifests" / f"{spec.name}.csv"
        json_path = run_dir / "manifests" / f"{spec.name}.json"
        safe_write_csv(csv_path, rows)
        safe_write_json(
            json_path,
            {
                "manifest_metadata": manifest_metadata(
                    spec,
                    args.n_per_suite,
                    args.development_test_source_count,
                    args.locked_source_count,
                    split_audit,
                    source_hash_map,
                    generator_hash,
                ),
                "samples": rows,
            },
        )
        manifest_paths[spec.name] = {"csv": csv_path, "json": json_path}

    schema_path = run_dir / "manifests/manifest_schema.json"
    safe_write_json(schema_path, schema_payload())
    train_idx, val_idx, test_idx = split_indices(
        int(split_audit["dataset_images_shape"][0]),
        int(config["seed"]),
        float(config["splits"]["train_frac"]),
        float(config["splits"]["val_frac"]),
    )
    development_idx = test_idx[: args.development_test_source_count]
    suite_validation, combined_validation = validate_manifests(
        manifests=manifests,
        manifest_paths=manifest_paths,
        n_per_suite=args.n_per_suite,
        locked_indices=locked_indices,
        locked_coordinate_group_ids=locked_coordinate_group_ids,
        all_coordinate_group_ids=all_coordinate_group_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        development_idx=development_idx,
    )
    if not combined_validation["passed"]:
        safe_write_json(
            run_dir / "diagnostics/manifest_validation_failure.json",
            {"suites": suite_validation, "combined": combined_validation},
        )
        raise RuntimeError("Provisional locked manifest validation failed; see diagnostics.")

    manifest_files = [
        path
        for paths in manifest_paths.values()
        for path in (paths["csv"], paths["json"])
    ]
    checksums = {
        project_relative(path): sha256_file(path)
        for path in [*manifest_files, schema_path]
    }
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_combined_sha256": generator_hash,
        "manifest_role": MANIFEST_ROLE,
        "paper_ready": False,
        "provisional_reason": PROVISIONAL_REASON,
        "source_pool_selection_method": SOURCE_POOL_SELECTION_METHOD,
        "coordinate_group_assignment_sha256": split_audit[
            "coordinate_group_assignment_sha256"
        ],
        "perceptual_duplicate_audit_applied": False,
        "run_directory": project_relative(run_dir),
        "total_samples": int(sum(len(rows) for rows in manifests.values())),
        "suite_diagnostics": suite_diagnostics,
        "files": [
            {
                "path": path,
                "sha256": digest,
                "locked_read_only": True,
            }
            for path, digest in checksums.items()
        ],
        "model_inference_performed": False,
        "qualitative_examples_inspected": False,
    }
    index_path = run_dir / "manifests/manifest_index.json"
    safe_write_json(index_path, index_payload)
    checksums[project_relative(index_path)] = sha256_file(index_path)
    checksum_path = run_dir / "manifests/SHA256SUMS"
    safe_write_text(
        checksum_path,
        "\n".join(
            f"{digest}  {Path(path).name}" for path, digest in checksums.items()
        ),
    )

    validation_json_path = run_dir / "tables/manifest_validation.json"
    validation_csv_path = run_dir / "tables/manifest_validation.csv"
    safe_write_json(
        validation_json_path,
        {"suites": suite_validation, "combined": combined_validation},
    )
    write_validation_csv(validation_csv_path, suite_validation)
    safe_write_json(
        run_dir / "tables/suite_generation_summary.json", suite_diagnostics
    )

    read_only_paths = [
        *manifest_files,
        schema_path,
        index_path,
        checksum_path,
    ]
    for path in read_only_paths:
        path.chmod(0o444)
    if not all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in read_only_paths):
        raise RuntimeError("One or more locked manifest files are not read-only.")

    safe_write_text(
        run_dir / "diagnostics/final_test_manifest_protocol.md",
        protocol_markdown(
            run_dir, suite_diagnostics, split_audit, checksums, generator_hash
        ),
    )
    safe_write_text(
        run_dir / "diagnostics/manifest_schema_validation_report.md",
        validation_markdown(suite_validation, combined_validation, read_only_paths),
    )
    safe_write_json(
        run_dir / "logs/completion_status.json",
        {
            "status": "complete_provisional",
            "manifest_role": MANIFEST_ROLE,
            "validation_passed": True,
            "paper_ready": False,
            "perceptual_duplicate_audit_applied": False,
            "provisional_reason": PROVISIONAL_REASON,
            "manifest_files_read_only": True,
            "model_inference_performed": False,
            "qualitative_examples_inspected": False,
            "run_directory": project_relative(run_dir),
        },
    )
    print(
        f"Mechanical validation: PASS ({combined_validation['total_rows']} rows).",
        flush=True,
    )
    print(
        "Paper readiness: BLOCKED pending exact-image/perceptual duplicate audit.",
        flush=True,
    )
    print("All manifest/schema files locked read-only with SHA-256 checksums.", flush=True)
    print(f"Completed: {project_relative(run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
