#!/usr/bin/env python3
"""Build deterministic, duplicate-safe grouped development blend manifests.

This script consumes a verified grouped *source* split and creates metadata-only
blend manifests for grouped training, validation, and development evaluation.
It performs no model training or inference.  Every accepted row is independently
replayable: ``sample_seed`` is passed directly to the one ``blend_pair`` call
that creates the saved blend hash, while source/parameter sampling uses the
separate ``candidate_seed``.

The manifests are deliberately labelled as development infrastructure, not as
a locked final-paper test.  Output paths are timestamped and collision-safe.
No raw image arrays are written to the manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import traceback
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import blend as gd_blend
from src import utils as gd_utils


GENERATOR_VERSION = "thayer_grouped_blend_manifest_development_v1"
SCHEMA_VERSION = "thayer_grouped_blend_manifest_schema_v1"
MANIFEST_ROLE = "duplicate_safe_grouped_development_benchmark"
NOT_FINAL_NOTE = (
    "Fresh duplicate-safe grouped development manifests. These are not a locked "
    "final-paper test and may be used for grouped retraining/evaluation only."
)
SEED_DERIVATION = "sha256_utf8_pipe_delimited_first_uint64_little_endian_v1"
ARRAY_HASH_METHOD = "sha256(dtype_ascii+shape_ascii+contiguous_bytes)_v1"

DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/manifests"
DEFAULT_BASE_SEED = 2_026_071_001

AFFECTED_THRESHOLD = 0.02
CORE_APERTURE_FRACTION = 0.18
CORE_PERCENTILE = 85.0
HALO_DILATION_ITERS = 5

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


@dataclass(frozen=True)
class BlendSpec:
    name: str
    filename: str
    display_name: str
    source_split: str
    n_samples: int
    max_shift: int
    brightness_range: tuple[float, float]
    blur_range: tuple[float, float]
    noise_range: tuple[float, float]
    rotation_range: tuple[float, float] = (0.0, 0.0)
    min_size_ratio: float | None = None
    max_size_ratio: float | None = None
    min_mask_fraction: float = 0.0
    min_core_obstruction: float | None = None
    max_attempts_per_sample: int = 1
    component: str = "normal"


MANIFEST_COLUMNS = [
    # Canonical evaluator contract.
    "sample_id",
    "suite",
    "source_split",
    "target_source_index",
    "target_group_id",
    "target_label",
    "contaminant_source_index",
    "contaminant_group_id",
    "contaminant_label",
    "sample_seed",
    "attempt_index",
    "shift_x",
    "shift_y",
    "brightness_scale",
    "blur_sigma",
    "noise_std",
    "rotation_degrees",
    "affected_threshold",
    "size_ratio",
    "mask_fraction",
    "core_obstruction_fraction",
    "blend_severity_score",
    "target_pxscale",
    "contaminant_pxscale",
    "blend_sha256",
    "affected_mask_sha256",
    "core_mask_sha256",
    "halo_mask_sha256",
    "generator_version",
    "generator_sha256",
    "dataset_sha256",
    "source_split_manifest_sha256",
    # Additional provenance, source identity, sampling, and replay fields.
    "schema_version",
    "manifest_role",
    "development_not_final",
    "suite_display_name",
    "training_component",
    "accepted_sample_index",
    "candidate_seed",
    "numpy_bit_generator",
    "seed_derivation",
    "source_selection_draws",
    "target_exact_sha256",
    "contaminant_exact_sha256",
    "target_coordinate_group_key",
    "contaminant_coordinate_group_key",
    "target_ra",
    "target_dec",
    "target_redshift",
    "contaminant_ra",
    "contaminant_dec",
    "contaminant_redshift",
    "target_group_size",
    "contaminant_group_size",
    "target_image_sha256",
    "contaminant_image_sha256",
    "shift_l1_pixels",
    "shift_l2_pixels",
    "rotation_enabled",
    "generation_difficulty",
    "target_area_pixels",
    "target_radius_pixels",
    "contaminant_area_pixels",
    "contaminant_radius_pixels",
    "pixel_size_ratio",
    "angular_size_ratio",
    "affected_pixel_count",
    "core_pixel_count",
    "core_affected_pixel_count",
    "noncore_affected_pixel_count",
    "halo_pixel_count",
    "identity_affected_mse",
    "identity_affected_mae",
    "halo_dilation_iters",
    "target_blur_or_noise_correction_present",
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
    "constraints_met",
    "target_contaminant_groups_distinct",
    "target_contaminant_sources_distinct",
    "replay_validation_selected",
    "replay_validation_status",
    "replay_max_abs_difference",
    "array_hash_method",
    "manifest_builder_sha256",
    "generator_combined_sha256",
    "sample_fingerprint_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic grouped training/validation/development blend "
            "manifests from a verified grouped source split."
        )
    )
    parser.add_argument(
        "--source-split-manifest",
        type=Path,
        required=True,
        help="Path to grouped source_split_manifest.csv.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override config dataset_path.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--audit-run-dir",
        type=Path,
        default=None,
        help="Existing master audit run that receives collision-safe small copies.",
    )
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--n-normal-test", type=int, default=1000)
    parser.add_argument("--n-hard-stress-test", type=int, default=1000)
    parser.add_argument("--n-compact-bright-test", type=int, default=1000)
    parser.add_argument("--n-high-core-test", type=int, default=1000)
    parser.add_argument(
        "--include-halo",
        action="store_true",
        help="Also create the optional halo/artifact-proxy development stress suite.",
    )
    parser.add_argument("--n-halo-test", type=int, default=1000)
    parser.add_argument(
        "--replay-samples-per-manifest",
        type=int,
        default=12,
        help="Number of rows per manifest regenerated and hash-checked in-process.",
    )
    parser.add_argument(
        "--source-cache-size",
        type=int,
        default=1024,
        help="LRU cache size for raw HDF5 cutouts (1024 is about 192 MiB).",
    )
    parser.add_argument(
        "--preload-all-images",
        action="store_true",
        help=(
            "Sequentially preload the read-only uint8 image array into host RAM "
            "before generation. This changes only I/O strategy, not blend values."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def validate_stamp(stamp: str) -> str:
    if not stamp or Path(stamp).name != stamp:
        raise ValueError("stamp must be a non-empty filename component")
    if not all(char.isdigit() or char == "_" for char in stamp):
        raise ValueError("stamp may contain only digits and underscores")
    return stamp


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def derive_seed(*parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return project_relative(value)
    return value


def exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def exclusive_json(path: Path, payload: Any) -> None:
    exclusive_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True))


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(jsonable(value))
    return value


def exclusive_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            missing = [column for column in columns if column not in row]
            if missing:
                raise ValueError(f"CSV row is missing columns: {missing}")
            writer.writerow({column: csv_value(row[column]) for column in columns})


def safe_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite audit copy: {destination}")
    with destination.open("xb") as output, source.open("rb") as input_handle:
        shutil.copyfileobj(input_handle, output)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def first_present(columns: Iterable[str], aliases: tuple[str, ...]) -> str:
    available = set(columns)
    for alias in aliases:
        if alias in available:
            return alias
    raise KeyError(f"Missing required grouped-split column; expected one of {aliases}")


def load_grouped_source_split(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Grouped source manifest has no header: {path}")
        raw_rows = list(reader)
        columns = reader.fieldnames
    if not raw_rows:
        raise ValueError("Grouped source manifest is empty")

    aliases = {
        "source_index": ("source_index", "global_source_index", "index"),
        "split": ("split", "source_split", "split_name"),
        "group_id": ("group_id", "duplicate_group_id", "source_group_id"),
        "label": ("label", "class_label", "ans"),
        "ra": ("ra",),
        "dec": ("dec",),
        "redshift": ("redshift",),
        "pxscale": ("pxscale", "pixel_scale"),
        "exact_sha256": ("exact_sha256", "image_sha256", "pixel_sha256"),
        "coordinate_group_key": (
            "coordinate_group_key",
            "coordinate_group_id",
            "coordinate_group_sha256",
        ),
        "group_size": ("group_size", "duplicate_group_size"),
    }
    mapping = {name: first_present(columns, choices) for name, choices in aliases.items()}
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        split_value = str(raw[mapping["split"]]).strip().lower()
        if split_value not in SPLIT_ALIASES:
            raise ValueError(f"Unknown split label {split_value!r}")
        row = {
            "source_index": int(raw[mapping["source_index"]]),
            "split": SPLIT_ALIASES[split_value],
            "group_id": str(raw[mapping["group_id"]]).strip(),
            "label": int(raw[mapping["label"]]),
            "ra": optional_float(raw[mapping["ra"]]),
            "dec": optional_float(raw[mapping["dec"]]),
            "redshift": optional_float(raw[mapping["redshift"]]),
            "pxscale": optional_float(raw[mapping["pxscale"]]),
            "exact_sha256": str(raw[mapping["exact_sha256"]]).strip(),
            "coordinate_group_key": str(
                raw[mapping["coordinate_group_key"]]
            ).strip(),
            "group_size": int(raw[mapping["group_size"]]),
        }
        if not row["group_id"] or not row["exact_sha256"]:
            raise ValueError("group_id and exact_sha256 must be non-empty")
        rows.append(row)
    return rows, mapping


def values_cross_splits(rows: list[dict[str, Any]], key: str, allow_empty: bool) -> int:
    assignments: dict[str, set[str]] = {}
    for row in rows:
        value = str(row[key])
        if not value and allow_empty:
            continue
        assignments.setdefault(value, set()).add(str(row["split"]))
    return sum(len(splits) > 1 for splits in assignments.values())


def validate_source_split(
    rows: list[dict[str, Any]], dataset_size: int
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    findings: list[dict[str, Any]] = []

    def add(check: str, expected: Any, observed: Any, passed: bool, detail: str) -> None:
        findings.append(
            {
                "scope": "source_split_input",
                "check": check,
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "severity_if_failed": "blocker",
                "detail": detail,
            }
        )

    indices = [int(row["source_index"]) for row in rows]
    add("row_count_matches_dataset", dataset_size, len(rows), len(rows) == dataset_size, "")
    add(
        "source_indices_unique",
        dataset_size,
        len(set(indices)),
        len(indices) == len(set(indices)) == dataset_size,
        "",
    )
    add(
        "source_indices_cover_dataset",
        "0..N-1",
        f"{min(indices)}..{max(indices)}",
        set(indices) == set(range(dataset_size)),
        "",
    )
    for key, allow_empty in (
        ("group_id", False),
        ("exact_sha256", False),
        ("coordinate_group_key", True),
    ):
        crossing = values_cross_splits(rows, key, allow_empty)
        add(
            f"{key}_cross_split_count",
            0,
            crossing,
            crossing == 0,
            "Every duplicate/coordinate group must remain inside one source split.",
        )
    counts = Counter(str(row["split"]) for row in rows)
    add(
        "required_splits_present",
        ["train", "validation", "test"],
        dict(counts),
        all(counts[name] >= 2 for name in ("train", "validation", "test")),
        "Each role needs at least two sources and two groups.",
    )
    pools = {
        split: sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: int(row["source_index"]),
        )
        for split in ("train", "validation", "test")
    }
    for split, pool in pools.items():
        groups = {str(row["group_id"]) for row in pool}
        add(
            f"{split}_has_distinct_groups",
            ">=2",
            len(groups),
            len(groups) >= 2,
            "",
        )
    if not all(bool(row["passed"]) for row in findings):
        failed = [row["check"] for row in findings if not row["passed"]]
        raise RuntimeError(f"Grouped source split integrity failed: {failed}")
    return findings, pools


class SourceImageCache:
    """Small explicit LRU cache for random HDF5 source reads."""

    def __init__(self, images: h5py.Dataset | np.ndarray, max_items: int) -> None:
        if max_items < 0:
            raise ValueError("source-cache-size must be non-negative")
        self.images = images
        self.max_items = max_items
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __call__(self, index: int) -> np.ndarray:
        index = int(index)
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value.astype(np.float32) / 255.0
        raw = np.asarray(self.images[index], dtype=np.uint8)
        if self.max_items > 0:
            self.cache[index] = raw
            if len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return raw.astype(np.float32) / 255.0


def component_schedule(n_samples: int, base_seed: int, suite: str) -> list[str]:
    normal = int(round(n_samples * 0.50))
    high = int(round(n_samples * 0.30))
    size = n_samples - normal - high
    schedule = ["normal"] * normal + ["high_overlap_core"] * high + [
        "brightness_size"
    ] * size
    rng = np.random.default_rng(derive_seed(base_seed, suite, "component_schedule"))
    rng.shuffle(schedule)
    return schedule


def suite_specs(args: argparse.Namespace, config: dict[str, Any]) -> list[BlendSpec]:
    normal = config["blending"]

    def base(
        name: str,
        filename: str,
        display: str,
        split: str,
        count: int,
        component: str = "normal",
    ) -> BlendSpec:
        return BlendSpec(
            name=name,
            filename=filename,
            display_name=display,
            source_split=split,
            n_samples=count,
            max_shift=int(normal["max_shift"]),
            brightness_range=tuple(float(x) for x in normal["brightness_range"]),
            blur_range=tuple(float(x) for x in normal["blur_range"]),
            noise_range=tuple(float(x) for x in normal["noise_range"]),
            rotation_range=tuple(float(x) for x in normal["rotation_range"]),
            min_mask_fraction=0.0,
            max_attempts_per_sample=16,
            component=component,
        )

    specs = [
        base(
            "train",
            "train_blends.csv",
            "Grouped balanced training",
            "train",
            args.n_train,
            "balanced_schedule",
        ),
        base(
            "validation",
            "val_blends.csv",
            "Grouped balanced validation",
            "validation",
            args.n_validation,
            "balanced_schedule",
        ),
        base(
            "normal_test",
            "normal_test_blends.csv",
            "Grouped normal development test",
            "test",
            args.n_normal_test,
        ),
        BlendSpec(
            "hard_stress_test",
            "hard_stress_test_blends.csv",
            "Grouped hard-stress development test",
            "test",
            args.n_hard_stress_test,
            18,
            (0.8, 1.4),
            (0.0, 0.15),
            (0.0, 0.006),
            min_size_ratio=0.75,
            min_mask_fraction=0.01,
            max_attempts_per_sample=500,
            component="hard_stress",
        ),
        BlendSpec(
            "compact_bright_test",
            "compact_bright_test_blends.csv",
            "Grouped compact-bright development test",
            "test",
            args.n_compact_bright_test,
            24,
            (1.35, 1.95),
            (0.0, 0.08),
            (0.0, 0.006),
            min_size_ratio=0.15,
            max_size_ratio=0.90,
            min_mask_fraction=0.004,
            max_attempts_per_sample=900,
            component="compact_bright",
        ),
        BlendSpec(
            "high_core_obstruction_test",
            "high_core_obstruction_test_blends.csv",
            "Grouped high-core-obstruction development test",
            "test",
            args.n_high_core_test,
            12,
            (0.9, 1.5),
            (0.0, 0.12),
            (0.0, 0.006),
            min_size_ratio=0.70,
            min_mask_fraction=0.01,
            min_core_obstruction=0.82,
            max_attempts_per_sample=1200,
            component="high_core_obstruction",
        ),
    ]
    if args.include_halo:
        specs.append(
            BlendSpec(
                "halo_artifact_stress_test",
                "halo_artifact_stress_test_blends.csv",
                "Grouped halo/artifact-proxy development stress",
                "test",
                args.n_halo_test,
                28,
                (0.85, 1.45),
                (0.05, 0.25),
                (0.0, 0.006),
                min_size_ratio=0.70,
                min_mask_fraction=0.015,
                max_attempts_per_sample=700,
                component="halo_blur_proxy",
            )
        )
    return specs


def spec_for_component(parent: BlendSpec, component: str) -> BlendSpec:
    if component == "normal":
        return BlendSpec(**{**asdict(parent), "component": component})
    if component == "high_overlap_core":
        return BlendSpec(
            **{
                **asdict(parent),
                "max_shift": 18,
                "brightness_range": (0.8, 1.4),
                "blur_range": (0.0, 0.15),
                "noise_range": (0.0, 0.006),
                "rotation_range": (0.0, 0.0),
                "min_size_ratio": 0.75,
                "max_size_ratio": None,
                "min_mask_fraction": 0.01,
                "min_core_obstruction": 0.66,
                "max_attempts_per_sample": 700,
                "component": component,
            }
        )
    if component == "brightness_size":
        return BlendSpec(
            **{
                **asdict(parent),
                "max_shift": 36,
                "brightness_range": (1.05, 1.5),
                "blur_range": (0.0, 0.15),
                "noise_range": (0.0, 0.006),
                "rotation_range": (0.0, 0.0),
                "min_size_ratio": 0.75,
                "max_size_ratio": None,
                "min_mask_fraction": 0.01,
                "min_core_obstruction": None,
                "max_attempts_per_sample": 500,
                "component": component,
            }
        )
    raise ValueError(f"Unknown balanced component: {component}")


def replay_indices(n_samples: int, requested: int) -> set[int]:
    count = min(max(requested, 0), n_samples)
    if count == 0:
        return set()
    return {int(value) for value in np.linspace(0, n_samples - 1, count, dtype=int)}


def choose_distinct_group_sources(
    rng: np.random.Generator,
    pool: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    target_local = int(rng.integers(0, len(pool)))
    target = pool[target_local]
    for draw in range(1, 10_001):
        contaminant_local = int(rng.integers(0, len(pool)))
        contaminant = pool[contaminant_local]
        if (
            contaminant["source_index"] != target["source_index"]
            and contaminant["group_id"] != target["group_id"]
        ):
            return target, contaminant, draw
    raise RuntimeError("Could not select a contaminant from a distinct duplicate group")


def generate_candidate(
    spec: BlendSpec,
    accepted_index: int,
    attempt_index: int,
    base_seed: int,
    pool: list[dict[str, Any]],
    load_image: Callable[[int], np.ndarray],
    provenance: dict[str, str],
    replay_selected: bool,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    candidate_seed = derive_seed(
        base_seed, spec.name, accepted_index, attempt_index, "candidate"
    )
    sample_seed = derive_seed(
        base_seed, spec.name, accepted_index, attempt_index, "blend_noise"
    )
    rng = np.random.default_rng(candidate_seed)
    target_source, contaminant_source, source_draws = choose_distinct_group_sources(
        rng, pool
    )
    target = load_image(int(target_source["source_index"]))
    contaminant = load_image(int(contaminant_source["source_index"]))
    shift_x = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
    shift_y = int(rng.integers(-spec.max_shift, spec.max_shift + 1))
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
        shift=(shift_x, shift_y),
        rotation=rotation,
        brightness=brightness,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        rng=np.random.default_rng(sample_seed),
    )
    affected = gd_utils.affected_region_mask(
        target, blended, threshold=AFFECTED_THRESHOLD
    )
    core = gd_utils.evaluation_core_mask_p85_v1(
        target,
        aperture_fraction=CORE_APERTURE_FRACTION,
        core_percentile=CORE_PERCENTILE,
    )
    halo = gd_utils.halo_band_mask_manhattan_v1(
        affected, dilation_iters=HALO_DILATION_ITERS
    )
    core_affected = affected & core
    noncore_affected = affected & ~core
    mask_fraction = float(affected.mean())
    core_obstruction = float(core_affected.sum() / max(int(core.sum()), 1))
    size_ratio = float(info["size_ratio"])
    identity_affected_mse = gd_utils.masked_mse(blended, target, affected)
    identity_affected_mae = gd_utils.masked_mae(blended, target, affected)
    severity = (
        mask_fraction * identity_affected_mae * (1.0 + core_obstruction)
        if math.isfinite(identity_affected_mae)
        else float("nan")
    )
    target_pxscale = target_source["pxscale"]
    contaminant_pxscale = contaminant_source["pxscale"]
    angular_ratio = (
        size_ratio * float(contaminant_pxscale) / float(target_pxscale)
        if target_pxscale not in (None, 0.0)
        and contaminant_pxscale is not None
        and math.isfinite(size_ratio)
        else None
    )

    row: dict[str, Any] = {
        "sample_id": f"{spec.name}_{accepted_index:06d}",
        "suite": spec.name,
        "source_split": spec.source_split,
        "target_source_index": int(target_source["source_index"]),
        "target_group_id": str(target_source["group_id"]),
        "target_label": int(target_source["label"]),
        "contaminant_source_index": int(contaminant_source["source_index"]),
        "contaminant_group_id": str(contaminant_source["group_id"]),
        "contaminant_label": int(contaminant_source["label"]),
        "sample_seed": int(sample_seed),
        "attempt_index": int(attempt_index),
        "shift_x": shift_x,
        "shift_y": shift_y,
        "brightness_scale": brightness,
        "blur_sigma": blur_sigma,
        "noise_std": noise_std,
        "rotation_degrees": rotation,
        "affected_threshold": AFFECTED_THRESHOLD,
        "size_ratio": size_ratio if math.isfinite(size_ratio) else None,
        "mask_fraction": mask_fraction,
        "core_obstruction_fraction": core_obstruction,
        "blend_severity_score": severity if math.isfinite(severity) else None,
        "target_pxscale": target_pxscale,
        "contaminant_pxscale": contaminant_pxscale,
        "blend_sha256": array_sha256(blended),
        "affected_mask_sha256": array_sha256(affected.astype(np.uint8)),
        "core_mask_sha256": array_sha256(core.astype(np.uint8)),
        "halo_mask_sha256": array_sha256(halo.astype(np.uint8)),
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": provenance["generator_sha256"],
        "dataset_sha256": provenance["dataset_sha256"],
        "source_split_manifest_sha256": provenance[
            "source_split_manifest_sha256"
        ],
        "schema_version": SCHEMA_VERSION,
        "manifest_role": MANIFEST_ROLE,
        "development_not_final": NOT_FINAL_NOTE,
        "suite_display_name": spec.display_name,
        "training_component": spec.component,
        "accepted_sample_index": accepted_index,
        "candidate_seed": int(candidate_seed),
        "numpy_bit_generator": "PCG64",
        "seed_derivation": SEED_DERIVATION,
        "source_selection_draws": source_draws,
        "target_exact_sha256": str(target_source["exact_sha256"]),
        "contaminant_exact_sha256": str(contaminant_source["exact_sha256"]),
        "target_coordinate_group_key": str(
            target_source["coordinate_group_key"]
        ),
        "contaminant_coordinate_group_key": str(
            contaminant_source["coordinate_group_key"]
        ),
        "target_ra": target_source["ra"],
        "target_dec": target_source["dec"],
        "target_redshift": target_source["redshift"],
        "contaminant_ra": contaminant_source["ra"],
        "contaminant_dec": contaminant_source["dec"],
        "contaminant_redshift": contaminant_source["redshift"],
        "target_group_size": int(target_source["group_size"]),
        "contaminant_group_size": int(contaminant_source["group_size"]),
        "target_image_sha256": array_sha256(target),
        "contaminant_image_sha256": array_sha256(contaminant),
        "shift_l1_pixels": int(abs(shift_x) + abs(shift_y)),
        "shift_l2_pixels": float(math.hypot(shift_x, shift_y)),
        "rotation_enabled": bool(abs(rotation) > 1e-12),
        "generation_difficulty": str(info["generation_difficulty"]),
        "target_area_pixels": float(info["target_area"]),
        "target_radius_pixels": float(info["target_radius"]),
        "contaminant_area_pixels": float(info["contaminant_area"]),
        "contaminant_radius_pixels": float(info["contaminant_radius"]),
        "pixel_size_ratio": size_ratio if math.isfinite(size_ratio) else None,
        "angular_size_ratio": angular_ratio,
        "affected_pixel_count": int(affected.sum()),
        "core_pixel_count": int(core.sum()),
        "core_affected_pixel_count": int(core_affected.sum()),
        "noncore_affected_pixel_count": int(noncore_affected.sum()),
        "halo_pixel_count": int(halo.sum()),
        "identity_affected_mse": (
            identity_affected_mse if math.isfinite(identity_affected_mse) else None
        ),
        "identity_affected_mae": (
            identity_affected_mae if math.isfinite(identity_affected_mae) else None
        ),
        "halo_dilation_iters": HALO_DILATION_ITERS,
        "target_blur_or_noise_correction_present": bool(
            blur_sigma > 0.0 or noise_std > 0.0
        ),
        "sampling_max_shift": spec.max_shift,
        "sampling_brightness_low": spec.brightness_range[0],
        "sampling_brightness_high": spec.brightness_range[1],
        "sampling_blur_low": spec.blur_range[0],
        "sampling_blur_high": spec.blur_range[1],
        "sampling_noise_low": spec.noise_range[0],
        "sampling_noise_high": spec.noise_range[1],
        "sampling_rotation_low": spec.rotation_range[0],
        "sampling_rotation_high": spec.rotation_range[1],
        "constraint_min_size_ratio": spec.min_size_ratio,
        "constraint_max_size_ratio": spec.max_size_ratio,
        "constraint_min_mask_fraction": spec.min_mask_fraction,
        "constraint_min_core_obstruction": spec.min_core_obstruction,
        "constraints_met": False,
        "target_contaminant_groups_distinct": bool(
            target_source["group_id"] != contaminant_source["group_id"]
        ),
        "target_contaminant_sources_distinct": bool(
            target_source["source_index"] != contaminant_source["source_index"]
        ),
        "replay_validation_selected": replay_selected,
        "replay_validation_status": "pending" if replay_selected else "not_sampled",
        "replay_max_abs_difference": None,
        "array_hash_method": ARRAY_HASH_METHOD,
        "manifest_builder_sha256": provenance["manifest_builder_sha256"],
        "generator_combined_sha256": provenance["generator_combined_sha256"],
        "sample_fingerprint_sha256": "",
    }
    return row, target, contaminant, blended, affected, core, halo


def constraints_met(spec: BlendSpec, row: dict[str, Any]) -> bool:
    ratio = row["size_ratio"]
    if ratio is None:
        return False
    if float(row["mask_fraction"]) <= 0.0:
        return False
    if float(row["mask_fraction"]) < spec.min_mask_fraction:
        return False
    if spec.min_size_ratio is not None and float(ratio) < spec.min_size_ratio:
        return False
    if spec.max_size_ratio is not None and float(ratio) > spec.max_size_ratio:
        return False
    if (
        spec.min_core_obstruction is not None
        and float(row["core_obstruction_fraction"]) < spec.min_core_obstruction
    ):
        return False
    return True


def exact_replay_check(
    row: dict[str, Any],
    target: np.ndarray,
    contaminant: np.ndarray,
    blended: np.ndarray,
    affected: np.ndarray,
    core: np.ndarray,
    halo: np.ndarray,
) -> float:
    replayed, _ = gd_blend.blend_pair(
        target=target,
        contaminant=contaminant,
        shift=(int(row["shift_x"]), int(row["shift_y"])),
        rotation=float(row["rotation_degrees"]),
        brightness=float(row["brightness_scale"]),
        blur_sigma=float(row["blur_sigma"]),
        noise_std=float(row["noise_std"]),
        rng=np.random.default_rng(int(row["sample_seed"])),
    )
    replay_affected = gd_utils.affected_region_mask(
        target, replayed, threshold=float(row["affected_threshold"])
    )
    replay_core = gd_utils.evaluation_core_mask_p85_v1(target)
    replay_halo = gd_utils.halo_band_mask_manhattan_v1(
        replay_affected, dilation_iters=int(row["halo_dilation_iters"])
    )
    expected_hashes = {
        "blend": array_sha256(blended),
        "affected": array_sha256(affected.astype(np.uint8)),
        "core": array_sha256(core.astype(np.uint8)),
        "halo": array_sha256(halo.astype(np.uint8)),
    }
    replay_hashes = {
        "blend": array_sha256(replayed),
        "affected": array_sha256(replay_affected.astype(np.uint8)),
        "core": array_sha256(replay_core.astype(np.uint8)),
        "halo": array_sha256(replay_halo.astype(np.uint8)),
    }
    if expected_hashes != replay_hashes:
        raise RuntimeError(
            f"Exact replay hash mismatch for {row['sample_id']}: "
            f"expected={expected_hashes}, replay={replay_hashes}"
        )
    return float(np.max(np.abs(replayed.astype(np.float64) - blended.astype(np.float64))))


def finalize_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(row)
    finalized["sample_fingerprint_sha256"] = hashlib.sha256(
        canonical_json(
            jsonable(
                {
                    key: value
                    for key, value in finalized.items()
                    if key != "sample_fingerprint_sha256"
                }
            )
        ).encode("utf-8")
    ).hexdigest()
    if set(finalized) != set(MANIFEST_COLUMNS):
        missing = sorted(set(MANIFEST_COLUMNS) - set(finalized))
        extra = sorted(set(finalized) - set(MANIFEST_COLUMNS))
        raise ValueError(f"Manifest schema mismatch: missing={missing}, extra={extra}")
    return {column: finalized[column] for column in MANIFEST_COLUMNS}


def generate_manifest(
    parent_spec: BlendSpec,
    base_seed: int,
    pool: list[dict[str, Any]],
    load_image: Callable[[int], np.ndarray],
    provenance: dict[str, str],
    replay_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if parent_spec.n_samples < 0:
        raise ValueError("Sample counts must be non-negative")
    selected_replay = replay_indices(parent_spec.n_samples, replay_count)
    schedule = (
        component_schedule(parent_spec.n_samples, base_seed, parent_spec.name)
        if parent_spec.component == "balanced_schedule"
        else [parent_spec.component] * parent_spec.n_samples
    )
    rows: list[dict[str, Any]] = []
    attempts_total = 0
    for accepted_index, component in enumerate(schedule):
        spec = (
            spec_for_component(parent_spec, component)
            if parent_spec.component == "balanced_schedule"
            else parent_spec
        )
        accepted_row: dict[str, Any] | None = None
        for attempt_index in range(spec.max_attempts_per_sample):
            attempts_total += 1
            candidate = generate_candidate(
                spec,
                accepted_index,
                attempt_index,
                base_seed,
                pool,
                load_image,
                provenance,
                accepted_index in selected_replay,
            )
            row, target, contaminant, blended, affected, core, halo = candidate
            if not constraints_met(spec, row):
                continue
            row["constraints_met"] = True
            if accepted_index in selected_replay:
                difference = exact_replay_check(
                    row, target, contaminant, blended, affected, core, halo
                )
                row["replay_validation_status"] = "exact_hash_match"
                row["replay_max_abs_difference"] = difference
            accepted_row = finalize_fingerprint(row)
            break
        if accepted_row is None:
            raise RuntimeError(
                f"Could not create strict {parent_spec.name}/{component} sample "
                f"{accepted_index} after {spec.max_attempts_per_sample} independent attempts"
            )
        rows.append(accepted_row)
        if (accepted_index + 1) % 100 == 0 or accepted_index + 1 == parent_spec.n_samples:
            print(
                f"{parent_spec.name}: accepted {accepted_index + 1}/"
                f"{parent_spec.n_samples} (candidate attempts={attempts_total})",
                flush=True,
            )

    mask_values = [float(row["mask_fraction"]) for row in rows]
    core_values = [float(row["core_obstruction_fraction"]) for row in rows]
    size_values = [float(row["size_ratio"]) for row in rows]
    summary = {
        "suite": parent_spec.name,
        "filename": parent_spec.filename,
        "display_name": parent_spec.display_name,
        "source_split": parent_spec.source_split,
        "n_samples": len(rows),
        "candidate_attempts": attempts_total,
        "rejected_candidates": attempts_total - len(rows),
        "component_counts": dict(Counter(str(row["training_component"]) for row in rows)),
        "unique_target_sources": len({int(row["target_source_index"]) for row in rows}),
        "unique_contaminant_sources": len(
            {int(row["contaminant_source_index"]) for row in rows}
        ),
        "unique_source_groups": len(
            {
                str(value)
                for row in rows
                for value in (row["target_group_id"], row["contaminant_group_id"])
            }
        ),
        "mean_mask_fraction": float(np.mean(mask_values)) if rows else None,
        "mean_core_obstruction_fraction": float(np.mean(core_values)) if rows else None,
        "mean_size_ratio": float(np.mean(size_values)) if rows else None,
        "replay_rows_checked": sum(
            row["replay_validation_status"] == "exact_hash_match" for row in rows
        ),
        "manifest_spec": asdict(parent_spec),
    }
    return rows, summary


def integrity_rows_for_manifest(
    spec: BlendSpec, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def check(name: str, expected: Any, observed: Any, passed: bool, detail: str = "") -> None:
        output.append(
            {
                "scope": spec.name,
                "check": name,
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "severity_if_failed": "blocker",
                "detail": detail,
            }
        )

    check("row_count", spec.n_samples, len(rows), len(rows) == spec.n_samples)
    ids = [str(row["sample_id"]) for row in rows]
    check("sample_ids_unique", len(rows), len(set(ids)), len(ids) == len(set(ids)))
    seeds = [int(row["sample_seed"]) for row in rows]
    check("sample_seeds_unique", len(rows), len(set(seeds)), len(seeds) == len(set(seeds)))
    check(
        "all_sources_in_expected_split",
        spec.source_split,
        sorted({str(row["source_split"]) for row in rows}),
        all(row["source_split"] == spec.source_split for row in rows),
    )
    check(
        "target_contaminant_sources_distinct",
        True,
        sum(bool(row["target_contaminant_sources_distinct"]) for row in rows),
        all(bool(row["target_contaminant_sources_distinct"]) for row in rows),
    )
    check(
        "target_contaminant_groups_distinct",
        True,
        sum(bool(row["target_contaminant_groups_distinct"]) for row in rows),
        all(bool(row["target_contaminant_groups_distinct"]) for row in rows),
    )
    check(
        "all_constraints_met",
        True,
        sum(bool(row["constraints_met"]) for row in rows),
        all(bool(row["constraints_met"]) for row in rows),
    )
    selected = [row for row in rows if row["replay_validation_selected"]]
    check(
        "selected_replays_exact",
        len(selected),
        sum(row["replay_validation_status"] == "exact_hash_match" for row in selected),
        all(row["replay_validation_status"] == "exact_hash_match" for row in selected),
        "sample_seed directly seeds the single replay blend_pair call",
    )
    return output


def global_manifest_integrity(
    manifests: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, expected: Any, observed: Any, passed: bool, detail: str = "") -> None:
        rows.append(
            {
                "scope": "all_manifests",
                "check": name,
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "severity_if_failed": "blocker",
                "detail": detail,
            }
        )

    role_sources: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    role_groups: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    role_target_sources: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    role_contaminant_sources = {key: set() for key in role_target_sources}
    all_ids: list[str] = []
    all_seeds: list[int] = []
    for manifest_rows in manifests.values():
        for row in manifest_rows:
            split = str(row["source_split"])
            target = int(row["target_source_index"])
            contaminant = int(row["contaminant_source_index"])
            role_sources[split].update((target, contaminant))
            role_target_sources[split].add(target)
            role_contaminant_sources[split].add(contaminant)
            role_groups[split].update(
                (str(row["target_group_id"]), str(row["contaminant_group_id"]))
            )
            all_ids.append(str(row["sample_id"]))
            all_seeds.append(int(row["sample_seed"]))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        source_overlap = role_sources[left] & role_sources[right]
        group_overlap = role_groups[left] & role_groups[right]
        cross_role = (
            role_target_sources[left] & role_contaminant_sources[right]
        ) | (role_contaminant_sources[left] & role_target_sources[right])
        add(f"{left}_{right}_source_overlap", 0, len(source_overlap), not source_overlap)
        add(f"{left}_{right}_group_overlap", 0, len(group_overlap), not group_overlap)
        add(
            f"{left}_{right}_target_contaminant_role_overlap",
            0,
            len(cross_role),
            not cross_role,
        )
    add("all_sample_ids_unique", len(all_ids), len(set(all_ids)), len(all_ids) == len(set(all_ids)))
    add("all_sample_seeds_unique", len(all_seeds), len(set(all_seeds)), len(all_seeds) == len(set(all_seeds)))
    return rows


def verify_hdf5_metadata(
    rows: list[dict[str, Any]], handle: h5py.File
) -> list[dict[str, Any]]:
    labels = np.asarray(handle["ans"][:]).reshape(-1)
    pxscale = np.asarray(handle["pxscale"][:]).reshape(-1)
    label_mismatch = sum(int(row["label"]) != int(labels[row["source_index"]]) for row in rows)
    pxscale_mismatch = 0
    for row in rows:
        manifest_value = row["pxscale"]
        dataset_value = float(pxscale[row["source_index"]])
        if manifest_value is None or not np.isclose(
            float(manifest_value), dataset_value, rtol=0.0, atol=1e-12
        ):
            pxscale_mismatch += 1
    return [
        {
            "scope": "source_split_input",
            "check": "labels_match_hdf5",
            "expected": 0,
            "observed": label_mismatch,
            "passed": label_mismatch == 0,
            "severity_if_failed": "blocker",
            "detail": "Grouped source indices must address the same HDF5 rows.",
        },
        {
            "scope": "source_split_input",
            "check": "pxscale_matches_hdf5",
            "expected": 0,
            "observed": pxscale_mismatch,
            "passed": pxscale_mismatch == 0,
            "severity_if_failed": "blocker",
            "detail": "",
        },
    ]


def main() -> int:
    args = parse_args()
    stamp = validate_stamp(args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    config_path = resolve(args.config)
    config = load_config(config_path)
    dataset_path = resolve(args.dataset_path or Path(config["dataset_path"]))
    source_split_path = resolve(args.source_split_manifest)
    output_root = resolve(args.output_root)
    allowed_output_root = (PROJECT_ROOT / "data/manifests").resolve()
    try:
        output_root.relative_to(allowed_output_root)
    except ValueError as exc:
        raise ValueError("Output must remain under data/manifests") from exc
    run_dir = output_root / f"grouped_blends_{stamp}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite grouped blend run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        for name, count in (
            ("n_train", args.n_train),
            ("n_validation", args.n_validation),
            ("n_normal_test", args.n_normal_test),
            ("n_hard_stress_test", args.n_hard_stress_test),
            ("n_compact_bright_test", args.n_compact_bright_test),
            ("n_high_core_test", args.n_high_core_test),
            ("n_halo_test", args.n_halo_test),
        ):
            if count < 0:
                raise ValueError(f"{name} must be non-negative")

        builder_path = Path(__file__).resolve()
        blend_path = PROJECT_ROOT / "src/blend.py"
        utils_path = PROJECT_ROOT / "src/utils.py"
        print(f"Hashing dataset: {project_relative(dataset_path)}", flush=True)
        source_hashes = {
            "manifest_builder_sha256": sha256_file(builder_path),
            "generator_sha256": sha256_file(blend_path),
            "utils_sha256": sha256_file(utils_path),
            "config_sha256": sha256_file(config_path),
            "dataset_sha256": sha256_file(dataset_path),
            "source_split_manifest_sha256": sha256_file(source_split_path),
        }
        source_hashes["generator_combined_sha256"] = hashlib.sha256(
            canonical_json(source_hashes).encode("utf-8")
        ).hexdigest()
        provenance = dict(source_hashes)

        source_rows, alias_mapping = load_grouped_source_split(source_split_path)
        with h5py.File(dataset_path, "r") as handle:
            required_keys = {"images", "ans", "ra", "dec", "redshift", "pxscale"}
            missing = sorted(required_keys - set(handle.keys()))
            if missing:
                raise KeyError(f"Dataset lacks required keys: {missing}")
            images = handle["images"]
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"Expected images (N,H,W,3), got {images.shape}")
            dataset_size = int(images.shape[0])
            input_checks, pools = validate_source_split(source_rows, dataset_size)
            metadata_checks = verify_hdf5_metadata(source_rows, handle)
            if not all(bool(row["passed"]) for row in metadata_checks):
                raise RuntimeError("Grouped source metadata does not match HDF5")
            image_source: h5py.Dataset | np.ndarray = images
            if args.preload_all_images:
                estimated_bytes = int(np.prod(images.shape) * images.dtype.itemsize)
                print(
                    "Sequentially preloading read-only uint8 source images "
                    f"({estimated_bytes / (1024 ** 3):.2f} GiB).",
                    flush=True,
                )
                image_source = np.asarray(images[:], dtype=np.uint8)
                if image_source.shape != images.shape:
                    raise RuntimeError(
                        f"Preloaded image shape {image_source.shape} != HDF5 {images.shape}"
                    )
            cache = SourceImageCache(image_source, args.source_cache_size)

            start_payload = {
                "status": "running",
                "created_at": datetime.now().astimezone().isoformat(),
                "manifest_role": MANIFEST_ROLE,
                "development_not_final": NOT_FINAL_NOTE,
                "dataset_path": project_relative(dataset_path),
                "source_split_manifest": project_relative(source_split_path),
                "output_directory": project_relative(run_dir),
                "base_seed": args.base_seed,
                "preload_all_images": bool(args.preload_all_images),
                "source_column_alias_mapping": alias_mapping,
                "hashes": source_hashes,
                "argv": sys.argv,
            }
            exclusive_json(run_dir / "generation_started.json", start_payload)

            specs = suite_specs(args, config)
            manifests: dict[str, list[dict[str, Any]]] = {}
            summaries: list[dict[str, Any]] = []
            integrity = input_checks + metadata_checks
            for spec in specs:
                print(
                    f"Generating {spec.name} ({spec.n_samples} rows, "
                    f"source split={spec.source_split})",
                    flush=True,
                )
                rows, summary = generate_manifest(
                    spec,
                    args.base_seed,
                    pools[spec.source_split],
                    cache,
                    provenance,
                    args.replay_samples_per_manifest,
                )
                manifests[spec.name] = rows
                summaries.append(summary)
                manifest_path = run_dir / spec.filename
                exclusive_csv(manifest_path, rows, MANIFEST_COLUMNS)
                integrity.extend(integrity_rows_for_manifest(spec, rows))
            integrity.extend(global_manifest_integrity(manifests))

        passed = all(bool(row["passed"]) for row in integrity)
        integrity_columns = [
            "scope",
            "check",
            "expected",
            "observed",
            "passed",
            "severity_if_failed",
            "detail",
        ]
        exclusive_csv(
            run_dir / "manifest_integrity_check.csv", integrity, integrity_columns
        )
        if not passed:
            failed = [f"{row['scope']}:{row['check']}" for row in integrity if not row["passed"]]
            raise RuntimeError(f"Grouped manifest integrity checks failed: {failed}")

        file_hashes = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.glob("*.csv"))
        }
        summary_payload = {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "schema_version": SCHEMA_VERSION,
            "manifest_columns_in_order": MANIFEST_COLUMNS,
            "generator_version": GENERATOR_VERSION,
            "manifest_role": MANIFEST_ROLE,
            "paper_ready_final_test": False,
            "development_not_final": NOT_FINAL_NOTE,
            "output_directory": project_relative(run_dir),
            "dataset_path": project_relative(dataset_path),
            "source_split_manifest": project_relative(source_split_path),
            "base_seed": args.base_seed,
            "preload_all_images": bool(args.preload_all_images),
            "seed_derivation": SEED_DERIVATION,
            "sample_seed_replay_contract": (
                "Pass sample_seed directly to np.random.default_rng and make one "
                "src.blend.blend_pair call with the row's explicit sources/parameters."
            ),
            "array_hash_method": ARRAY_HASH_METHOD,
            "affected_mask_definition": "mean(abs(blended-target), RGB) > 0.02",
            "core_mask_definition": "evaluation_core_mask_p85_v1",
            "halo_mask_definition": "halo_band_mask_manhattan_v1, 5 iterations",
            "source_column_alias_mapping": alias_mapping,
            "source_split_counts": dict(Counter(row["split"] for row in source_rows)),
            "source_split_group_counts": {
                split: len({row["group_id"] for row in pool})
                for split, pool in pools.items()
            },
            "suite_summaries": summaries,
            "integrity_check_count": len(integrity),
            "integrity_checks_passed": sum(bool(row["passed"]) for row in integrity),
            "all_integrity_checks_passed": passed,
            "source_hashes": source_hashes,
            "manifest_file_sha256": file_hashes,
            "provenance": {
                "output_sha256": file_hashes,
                "source_sha256": source_hashes,
            },
            "contains_raw_image_arrays": False,
            "qualitative_examples_inspected": False,
            "known_scope_limitations": [
                "Galaxy10 DECaLS inputs are RGB display cutouts, not calibrated FITS flux images.",
                "The historical generator applies optional blur to the target and noise after compositing.",
                "Generator difficulty is a parameter label; blend_severity_score is measured damage, not model difficulty.",
                "These fresh grouped manifests are development infrastructure and not a locked final-paper test.",
            ],
        }
        exclusive_json(run_dir / "manifest_summary.json", summary_payload)

        # Small, uniquely named provenance copies inside the master audit run.
        if args.audit_run_dir is not None:
            audit_dir = resolve(args.audit_run_dir)
            allowed_audit_root = (PROJECT_ROOT / "outputs/runs").resolve()
            try:
                audit_dir.relative_to(allowed_audit_root)
            except ValueError as exc:
                raise ValueError("audit-run-dir must remain under outputs/runs") from exc
            if not audit_dir.is_dir():
                raise FileNotFoundError(f"Audit run does not exist: {audit_dir}")
            prefix = f"grouped_blends_{stamp}"
            safe_copy(
                run_dir / "manifest_summary.json",
                audit_dir / "manifests" / f"{prefix}_manifest_summary.json",
            )
            safe_copy(
                run_dir / "manifest_integrity_check.csv",
                audit_dir / "tables" / f"{prefix}_manifest_integrity_check.csv",
            )
            exclusive_json(
                audit_dir / "manifests" / f"{prefix}_location.json",
                {
                    "grouped_blend_manifest_directory": project_relative(run_dir),
                    "manifest_summary_sha256": sha256_file(
                        run_dir / "manifest_summary.json"
                    ),
                    "manifest_integrity_check_sha256": sha256_file(
                        run_dir / "manifest_integrity_check.csv"
                    ),
                    "manifest_role": MANIFEST_ROLE,
                    "paper_ready_final_test": False,
                },
            )

        print(f"Grouped blend manifests complete: {run_dir}", flush=True)
        return 0
    except Exception as exc:
        failure_path = run_dir / "generation_failure.json"
        if not failure_path.exists():
            exclusive_json(
                failure_path,
                {
                    "status": "failed",
                    "failed_at": datetime.now().astimezone().isoformat(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "output_directory_preserved": project_relative(run_dir),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
