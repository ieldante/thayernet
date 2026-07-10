#!/usr/bin/env python3
"""Audit historical v0.2 source exposure in the grouped development tests.

This is deliberately a CPU-only metadata/CSV aggregation. It never loads a model,
opens a checkpoint, or performs tensor inference. Exposure is evaluated primarily
at the duplicate-safe source ``group_id`` level. Direct row-index exposure is also
reported as a sensitivity check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_RUN = PROJECT_ROOT / "outputs/runs/research_correctness_audit_20260710_092241"
DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"
DEFAULT_HISTORICAL_CONFIG = (
    PROJECT_ROOT / "outputs/runs/weighted_residual_20260709_030245/logs/run_config.yaml"
)

IDENTITY_METHOD = "identity"
V02_METHOD = "v02_moderate_old_split"
METHODS = (IDENTITY_METHOD, V02_METHOD)

SOURCE_COLUMNS = (
    "suite",
    "manifest_suite",
    "manifest_row_index",
    "source_split",
    "target_source_index",
    "target_group_id",
    "contaminant_source_index",
    "contaminant_group_id",
    "sample_seed",
    "attempt_index",
)

REQUIRED_METRIC_COLUMNS = (
    "affected_mse_clipped",
    "affected_mae_clipped",
    "core_affected_pixel_count",
    "core_affected_mse_clipped",
    "halo_pixel_count",
    "halo_mse_clipped",
    "worse_than_identity_clipped",
)

PREFERRED_SUITE_ORDER = (
    "normal",
    "hard_stress",
    "compact_bright",
    "high_core_obstruction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_AUDIT_RUN)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--historical-run-config", type=Path, default=DEFAULT_HISTORICAL_CONFIG
    )
    parser.add_argument(
        "--per-sample-table",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/tables/grouped_existing_v02_per_sample_metrics.csv.",
    )
    parser.add_argument(
        "--source-split-manifest",
        type=Path,
        default=None,
        help=(
            "Duplicate-safe source manifest. If omitted, resolve it from "
            "RUN_DIR/logs/grouped_existing_v02_provenance.json."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def write_text_exclusive(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_json_exclusive(payload: dict[str, Any], path: Path) -> None:
    write_text_exclusive(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(("true", "false"))
    if invalid.any():
        values = sorted(normalized.loc[invalid].unique().tolist())
        raise ValueError(f"Could not parse Boolean values: {values}")
    return normalized.eq("true")


def resolve_source_manifest(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve(explicit)
    provenance_path = run_dir / "logs/grouped_existing_v02_provenance.json"
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    return Path(provenance["input_provenance"]["source_split_manifest"]["path"])


def historical_source_sets(
    dataset_path: Path, historical_config_path: Path
) -> tuple[dict[str, Any], set[int], set[int]]:
    with historical_config_path.open("r", encoding="utf-8") as handle:
        historical = yaml.safe_load(handle)

    settings = historical["settings"]
    config = historical["config"]
    seed = int(config["seed"])
    train_frac = float(config["splits"]["train_frac"])
    val_frac = float(config["splits"]["val_frac"])
    train_subset = int(settings["train_source_subset"])
    val_subset = int(settings["val_source_subset"])

    with h5py.File(dataset_path, "r") as handle:
        n_sources = int(handle["images"].shape[0])

    indices = np.arange(n_sources, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train_partition = int(n_sources * train_frac)
    n_val_partition = int(n_sources * val_frac)
    train_partition = indices[:n_train_partition]
    val_partition = indices[n_train_partition : n_train_partition + n_val_partition]

    if train_subset > len(train_partition) or val_subset > len(val_partition):
        raise ValueError("Historical source subset exceeds its reconstructed partition.")

    train_used = {int(value) for value in train_partition[:train_subset]}
    val_used = {int(value) for value in val_partition[:val_subset]}
    if train_used & val_used:
        raise AssertionError("Historical train and validation row-index subsets overlap.")

    metadata = {
        "dataset_n_sources": n_sources,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": float(config["splits"]["test_frac"]),
        "historical_train_partition_n": n_train_partition,
        "historical_validation_partition_n": n_val_partition,
        "historical_train_sources_used_n": len(train_used),
        "historical_validation_sources_used_n": len(val_used),
        "historical_variant": historical.get("variant_name"),
        "historical_best_checkpoint": historical.get("best_checkpoint"),
    }
    return metadata, train_used, val_used


def source_level_samples(metrics: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        metrics,
        ("sample_id", "method", *SOURCE_COLUMNS, *REQUIRED_METRIC_COLUMNS),
        "grouped per-sample metric table",
    )

    if metrics.duplicated(["sample_id", "method"]).any():
        duplicates = metrics.loc[
            metrics.duplicated(["sample_id", "method"], keep=False),
            ["sample_id", "method"],
        ]
        raise ValueError(
            "Metric table has duplicate sample_id/method rows; examples: "
            f"{duplicates.head(5).to_dict(orient='records')}"
        )

    methods = set(metrics["method"].astype(str).unique())
    missing_methods = sorted(set(METHODS) - methods)
    if missing_methods:
        raise ValueError(f"Metric table is missing required methods: {missing_methods}")

    comparison = metrics.loc[metrics["method"].isin(METHODS)].copy()
    sample_sets = {
        method: set(comparison.loc[comparison["method"] == method, "sample_id"].astype(str))
        for method in METHODS
    }
    if sample_sets[IDENTITY_METHOD] != sample_sets[V02_METHOD]:
        raise ValueError("Identity and v0.2 sample_id sets are not aligned.")

    variability = metrics.groupby("sample_id", sort=False)[list(SOURCE_COLUMNS)].nunique(
        dropna=False
    )
    inconsistent = variability.gt(1).any(axis=1)
    if inconsistent.any():
        examples = variability.loc[inconsistent].head(5).index.tolist()
        raise ValueError(f"Source metadata differs across methods for samples: {examples}")

    samples = (
        metrics.loc[:, ["sample_id", *SOURCE_COLUMNS]]
        .drop_duplicates(subset=["sample_id"], keep="first")
        .copy()
    )
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["suite"] = samples["suite"].astype(str)
    for column in ("target_source_index", "contaminant_source_index"):
        numeric = pd.to_numeric(samples[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} contains non-integer values.")
        samples[column] = numeric.astype(np.int64)
    return samples


def add_exposure_flags(
    samples: pd.DataFrame,
    source_manifest: pd.DataFrame,
    old_train_rows: set[int],
    old_val_rows: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(source_manifest, ("source_index", "group_id"), "source split manifest")
    manifest = source_manifest.loc[:, ["source_index", "group_id"]].copy()
    manifest["source_index"] = pd.to_numeric(manifest["source_index"], errors="raise").astype(
        np.int64
    )
    manifest["group_id"] = manifest["group_id"].astype(str)
    if manifest["source_index"].duplicated().any():
        raise ValueError("Source split manifest repeats source_index values.")
    expected = set(range(len(manifest)))
    observed = set(int(value) for value in manifest["source_index"])
    if observed != expected:
        raise ValueError("Source split manifest does not cover source indices 0..N-1 exactly.")

    index_to_group = manifest.set_index("source_index")["group_id"]
    for role in ("target", "contaminant"):
        index_column = f"{role}_source_index"
        group_column = f"{role}_group_id"
        expected_groups = samples[index_column].map(index_to_group)
        mismatch = expected_groups.astype(str).ne(samples[group_column].astype(str))
        if mismatch.any():
            examples = samples.loc[
                mismatch, ["sample_id", index_column, group_column]
            ].head(5)
            raise ValueError(
                f"{role} group IDs disagree with source split manifest: "
                f"{examples.to_dict(orient='records')}"
            )

    old_train_groups = set(
        manifest.loc[manifest["source_index"].isin(old_train_rows), "group_id"]
    )
    old_val_groups = set(
        manifest.loc[manifest["source_index"].isin(old_val_rows), "group_id"]
    )
    old_union_groups = old_train_groups | old_val_groups
    old_union_rows = old_train_rows | old_val_rows

    result = samples.copy()
    references = {
        "old_train": (old_train_groups, old_train_rows),
        "old_validation": (old_val_groups, old_val_rows),
        "old_train_or_validation": (old_union_groups, old_union_rows),
    }
    for reference, (groups, rows) in references.items():
        for role in ("target", "contaminant"):
            result[f"{role}_{reference}_exposed"] = result[f"{role}_group_id"].isin(groups)
            result[f"{role}_{reference}_row_exact"] = result[f"{role}_source_index"].isin(rows)
        result[f"either_role_{reference}_exposed"] = (
            result[f"target_{reference}_exposed"]
            | result[f"contaminant_{reference}_exposed"]
        )
        result[f"both_roles_{reference}_exposed"] = (
            result[f"target_{reference}_exposed"]
            & result[f"contaminant_{reference}_exposed"]
        )
        result[f"either_role_{reference}_row_exact"] = (
            result[f"target_{reference}_row_exact"]
            | result[f"contaminant_{reference}_row_exact"]
        )
        result[f"both_roles_{reference}_row_exact"] = (
            result[f"target_{reference}_row_exact"]
            & result[f"contaminant_{reference}_row_exact"]
        )

    result["clean_neither_old_train_or_validation"] = ~result[
        "either_role_old_train_or_validation_exposed"
    ]
    result["clean_neither_old_train_or_validation_row_exact"] = ~result[
        "either_role_old_train_or_validation_row_exact"
    ]

    group_sizes = manifest.groupby("group_id", sort=False).size()
    metadata = {
        "old_train_group_n": len(old_train_groups),
        "old_validation_group_n": len(old_val_groups),
        "old_train_or_validation_group_n": len(old_union_groups),
        "old_train_validation_group_overlap_n": len(old_train_groups & old_val_groups),
        "old_train_group_expanded_source_rows_n": int(group_sizes.loc[list(old_train_groups)].sum()),
        "old_validation_group_expanded_source_rows_n": int(
            group_sizes.loc[list(old_val_groups)].sum()
        ),
        "old_train_or_validation_group_expanded_source_rows_n": int(
            group_sizes.loc[list(old_union_groups)].sum()
        ),
    }
    return result, metadata


def ordered_scopes(samples: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    suites = list(dict.fromkeys(samples["suite"].astype(str).tolist()))
    ordered = [suite for suite in PREFERRED_SUITE_ORDER if suite in suites]
    ordered.extend(sorted(set(suites) - set(ordered)))
    return [("all_suites", samples), *[(suite, samples[samples["suite"] == suite]) for suite in ordered]]


def exposure_rates(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in ordered_scopes(samples):
        n = len(frame)
        if n == 0:
            continue
        unique_groups = set(frame["target_group_id"]) | set(frame["contaminant_group_id"])
        row: dict[str, Any] = {
            "scope": scope,
            "n_samples": n,
            "unique_target_source_groups": int(frame["target_group_id"].nunique()),
            "unique_contaminant_source_groups": int(
                frame["contaminant_group_id"].nunique()
            ),
            "unique_source_groups_either_role": len(unique_groups),
            "unique_target_source_indices": int(frame["target_source_index"].nunique()),
            "unique_contaminant_source_indices": int(
                frame["contaminant_source_index"].nunique()
            ),
        }
        for reference in (
            "old_train",
            "old_validation",
            "old_train_or_validation",
        ):
            for role_label, flag_column in (
                ("target", f"target_{reference}_exposed"),
                ("contaminant", f"contaminant_{reference}_exposed"),
                ("either_role", f"either_role_{reference}_exposed"),
                ("both_roles", f"both_roles_{reference}_exposed"),
            ):
                count = int(frame[flag_column].sum())
                row[f"{reference}_{role_label}_exposed_n"] = count
                row[f"{reference}_{role_label}_exposed_fraction"] = count / n

            exposed_group_ids = set(
                frame.loc[frame[f"target_{reference}_exposed"], "target_group_id"]
            ) | set(
                frame.loc[
                    frame[f"contaminant_{reference}_exposed"], "contaminant_group_id"
                ]
            )
            row[f"{reference}_unique_source_groups_exposed"] = len(exposed_group_ids)

            for role_label, flag_column in (
                ("target", f"target_{reference}_row_exact"),
                ("contaminant", f"contaminant_{reference}_row_exact"),
                ("either_role", f"either_role_{reference}_row_exact"),
                ("both_roles", f"both_roles_{reference}_row_exact"),
            ):
                count = int(frame[flag_column].sum())
                row[f"{reference}_{role_label}_row_exact_n"] = count
                row[f"{reference}_{role_label}_row_exact_fraction"] = count / n

        clean_n = int(frame["clean_neither_old_train_or_validation"].sum())
        clean_row_n = int(
            frame["clean_neither_old_train_or_validation_row_exact"].sum()
        )
        row["clean_neither_old_train_or_validation_n"] = clean_n
        row["clean_neither_old_train_or_validation_fraction"] = clean_n / n
        row["clean_neither_old_train_or_validation_row_exact_n"] = clean_row_n
        row["clean_neither_old_train_or_validation_row_exact_fraction"] = (
            clean_row_n / n
        )
        row["group_expansion_additional_old_train_or_validation_exposed_n"] = int(
            (
                frame["either_role_old_train_or_validation_exposed"]
                & ~frame["either_role_old_train_or_validation_row_exact"]
            ).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def finite_mean(series: pd.Series) -> tuple[int, float]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric)
    return int(valid.sum()), float(np.mean(numeric[valid])) if valid.any() else float("nan")


def aggregate_method(frame: pd.DataFrame) -> dict[str, Any]:
    affected_n, affected_mse = finite_mean(frame["affected_mse_clipped"])
    affected_mae_n, affected_mae = finite_mean(frame["affected_mae_clipped"])

    core_values = pd.to_numeric(frame["core_affected_mse_clipped"], errors="coerce")
    core_pixels = pd.to_numeric(frame["core_affected_pixel_count"], errors="coerce")
    core_valid = core_values.notna() & np.isfinite(core_values) & core_pixels.gt(0)

    halo_values = pd.to_numeric(frame["halo_mse_clipped"], errors="coerce")
    halo_pixels = pd.to_numeric(frame["halo_pixel_count"], errors="coerce")
    halo_valid = halo_values.notna() & np.isfinite(halo_values) & halo_pixels.gt(0)

    worse = bool_series(frame["worse_than_identity_clipped"])
    return {
        "n_samples": len(frame),
        "affected_valid_n": affected_n,
        "affected_mse_clipped_macro": affected_mse,
        "affected_mae_valid_n": affected_mae_n,
        "affected_mae_clipped_macro": affected_mae,
        "core_valid_n": int(core_valid.sum()),
        "core_affected_mse_clipped_macro": (
            float(core_values.loc[core_valid].mean()) if core_valid.any() else float("nan")
        ),
        "halo_valid_n": int(halo_valid.sum()),
        "halo_mse_clipped_macro": (
            float(halo_values.loc[halo_valid].mean()) if halo_valid.any() else float("nan")
        ),
        "worse_than_identity_count": int(worse.sum()),
        "worse_than_identity_fraction": float(worse.mean()) if len(worse) else float("nan"),
    }


def metrics_by_exposure(metrics: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    comparison = metrics.loc[metrics["method"].isin(METHODS)].copy()
    comparison["sample_id"] = comparison["sample_id"].astype(str)
    comparison = comparison.merge(
        samples.loc[
            :,
            [
                "sample_id",
                "either_role_old_train_exposed",
                "either_role_old_validation_exposed",
                "either_role_old_train_or_validation_exposed",
                "clean_neither_old_train_or_validation",
            ],
        ],
        on="sample_id",
        validate="many_to_one",
    )

    subset_definitions = (
        ("all", None),
        ("clean_neither_old_train_or_validation", "clean_neither_old_train_or_validation"),
        ("old_train_exposed", "either_role_old_train_exposed"),
        ("old_validation_exposed", "either_role_old_validation_exposed"),
        (
            "old_train_or_validation_exposed",
            "either_role_old_train_or_validation_exposed",
        ),
    )

    rows: list[dict[str, Any]] = []
    for scope, sample_scope in ordered_scopes(samples):
        scope_ids = set(sample_scope["sample_id"])
        scope_metrics = comparison[comparison["sample_id"].isin(scope_ids)]
        scope_n = len(scope_ids)
        for subset_name, flag_column in subset_definitions:
            if flag_column is None:
                subset = scope_metrics
            else:
                subset = scope_metrics[scope_metrics[flag_column].astype(bool)]
            subset_ids = set(subset["sample_id"])
            if not subset_ids:
                continue

            method_frames = {
                method: subset[subset["method"] == method].sort_values("sample_id")
                for method in METHODS
            }
            expected_ids = set(method_frames[IDENTITY_METHOD]["sample_id"])
            if expected_ids != subset_ids or set(method_frames[V02_METHOD]["sample_id"]) != subset_ids:
                raise ValueError(
                    f"Sample alignment failed for scope={scope}, subset={subset_name}."
                )
            if any(len(frame) != len(subset_ids) for frame in method_frames.values()):
                raise ValueError(
                    f"Method row count failed for scope={scope}, subset={subset_name}."
                )

            aggregates = {
                method: aggregate_method(frame) for method, frame in method_frames.items()
            }
            identity_mse = aggregates[IDENTITY_METHOD]["affected_mse_clipped_macro"]
            model_mse = aggregates[V02_METHOD]["affected_mse_clipped_macro"]
            ratio = (
                identity_mse / model_mse
                if np.isfinite(identity_mse) and np.isfinite(model_mse) and model_mse > 0
                else float("nan")
            )

            identity_aligned = method_frames[IDENTITY_METHOD].set_index("sample_id")
            model_aligned = method_frames[V02_METHOD].set_index("sample_id")
            direct_worse = (
                pd.to_numeric(model_aligned["affected_mse_clipped"], errors="coerce")
                > pd.to_numeric(identity_aligned["affected_mse_clipped"], errors="coerce")
            )
            reported_worse = bool_series(
                model_aligned["worse_than_identity_clipped"]
            )
            if not np.array_equal(direct_worse.to_numpy(), reported_worse.to_numpy()):
                raise ValueError(
                    f"Stored worse-than-identity flags disagree with aligned metrics for "
                    f"scope={scope}, subset={subset_name}."
                )

            for method in METHODS:
                aggregate = aggregates[method]
                row = {
                    "scope": scope,
                    "exposure_subset": subset_name,
                    "exposure_definition": "duplicate_safe_group_identity",
                    "method": method,
                    "n_samples": aggregate["n_samples"],
                    "scope_n_samples": scope_n,
                    "subset_fraction_of_scope": len(subset_ids) / scope_n,
                    **aggregate,
                    "identity_to_method_affected_mse_macro_ratio": (
                        1.0 if method == IDENTITY_METHOD else ratio
                    ),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame.loc[:, columns].copy()
    for column in table.columns:
        if "fraction" in column or "ratio" in column or "mse" in column:
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
            )
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *rows))


def build_report(
    historical: dict[str, Any],
    group_metadata: dict[str, Any],
    rates: pd.DataFrame,
    metrics: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    all_rates = rates[rates["scope"] == "all_suites"].iloc[0]
    clean_n = int(all_rates["clean_neither_old_train_or_validation_n"])
    total_n = int(all_rates["n_samples"])
    union_n = int(all_rates["old_train_or_validation_either_role_exposed_n"])
    row_union_n = int(all_rates["old_train_or_validation_either_role_row_exact_n"])
    additional_n = int(
        all_rates["group_expansion_additional_old_train_or_validation_exposed_n"]
    )

    rate_view = rates[
        [
            "scope",
            "n_samples",
            "old_train_either_role_exposed_n",
            "old_train_either_role_exposed_fraction",
            "old_validation_either_role_exposed_n",
            "old_validation_either_role_exposed_fraction",
            "old_train_or_validation_either_role_exposed_n",
            "old_train_or_validation_either_role_exposed_fraction",
            "clean_neither_old_train_or_validation_n",
            "clean_neither_old_train_or_validation_fraction",
        ]
    ]

    all_model_metrics = metrics[
        (metrics["scope"] == "all_suites") & (metrics["method"] == V02_METHOD)
    ][
        [
            "exposure_subset",
            "n_samples",
            "affected_mse_clipped_macro",
            "affected_mae_clipped_macro",
            "core_valid_n",
            "core_affected_mse_clipped_macro",
            "halo_valid_n",
            "halo_mse_clipped_macro",
            "identity_to_method_affected_mse_macro_ratio",
            "worse_than_identity_count",
        ]
    ]

    return f"""# Historical v0.2 exposure in the grouped development test

## Verdict

Part 7 remains a diagnostic, not a source-independent generalization result. The grouped blend manifests are internally duplicate-safe for the new protocol, but the evaluated checkpoint was trained and selected with a separate historical row-index split. At the duplicate-safe source-identity level, {union_n:,}/{total_n:,} grouped evaluation samples ({union_n / total_n:.2%}) contain a target or contaminant group represented in the old training or validation subset. The conservative clean-neither subset contains {clean_n:,}/{total_n:,} samples ({clean_n / total_n:.2%}). A v0.2 retrain using the grouped training and validation manifests is required.

Validation exposure is distinct from gradient-training exposure: validation sources were not optimizer inputs, but they influenced checkpoint selection and model-development decisions. It therefore still prevents the old-checkpoint evaluation from being a fully untouched source-level test.

## Exact historical split reconstruction

- Historical run: `weighted_residual_20260709_030245`, variant `{historical['historical_variant']}`.
- Dataset sources: `{historical['dataset_n_sources']:,}`.
- Shuffle: `numpy.random.default_rng({historical['seed']}).shuffle(indices)`.
- Historical row-index partitions: train `{historical['historical_train_partition_n']:,}`, validation `{historical['historical_validation_partition_n']:,}`.
- Sources actually available to weighted v0.2 blend generation: first `{historical['historical_train_sources_used_n']:,}` shuffled train rows and first `{historical['historical_validation_sources_used_n']:,}` shuffled validation rows.
- Duplicate-safe identities represented: training `{group_metadata['old_train_group_n']:,}` groups, validation `{group_metadata['old_validation_group_n']:,}` groups; `{group_metadata['old_train_validation_group_overlap_n']:,}` groups occur in both old roles because the historical split was row-based.
- Group expansion covers `{group_metadata['old_train_group_expanded_source_rows_n']:,}` dataset rows for old training identities and `{group_metadata['old_validation_group_expanded_source_rows_n']:,}` rows for old validation identities.

## Exposure rates

The primary exposure definition expands each historical row through the duplicate-safe `group_id` (exact-pixel and exact-coordinate grouping). This is stricter than direct row-index matching. Direct row matching finds {row_union_n:,} train-or-validation-exposed samples; group expansion adds {additional_n:,} samples whose selected row was different but whose source identity was represented historically.

{markdown_table(rate_view, list(rate_view.columns))}

Target, contaminant, either-role, both-role, unique-source, direct-row sensitivity, and clean-neither fields are all retained in `{paths['rates'].name}`. Samples were deduplicated by `sample_id` before exposure counting, so the identity, threshold, and learned-model rows do not triple-count source exposure.

## Clipped metric sensitivity

All metrics below are macro means over aligned samples. The identity/model ratio is mean identity affected MSE divided by mean model affected MSE. Core and halo means include only samples with non-empty corresponding masks; their valid counts are reported explicitly.

{markdown_table(all_model_metrics, list(all_model_metrics.columns))}

The clean-neither subset is a useful conservative diagnostic under the exact-pixel/exact-coordinate grouping definition. It is not a final paper result: the grouped benchmark was constructed during this audit, perceptual near-duplicate grouping remains a separate limitation, and the old model was not trained under the grouped protocol.

## Interpretation and required action

- The new grouped test is suitable for comparing models trained with the same grouped source protocol.
- Evaluating the old v0.2 checkpoint on it cannot establish source-independent generalization because historical source exposure is substantial.
- Metrics on train-exposed, validation-exposed, union-exposed, and clean-neither subsets quantify sensitivity but do not repair the protocol mismatch.
- Retrain v0.2 Moderate on the grouped train/validation manifests, then evaluate its best checkpoint on the identical grouped test manifests.
- Do not call Part 7 final and do not use it alone to keep or replace the historical 32x development claim.

## Output files

- `{paths['rates']}`
- `{paths['metrics']}`
- `{paths['provenance']}`
"""


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    dataset_path = resolve(args.dataset)
    historical_config_path = resolve(args.historical_run_config)
    per_sample_path = resolve(args.per_sample_table) if args.per_sample_table else (
        run_dir / "tables/grouped_existing_v02_per_sample_metrics.csv"
    )
    source_manifest_path = resolve_source_manifest(run_dir, args.source_split_manifest)

    paths = {
        "rates": run_dir / "tables/grouped_existing_v02_historical_exposure_rates.csv",
        "metrics": run_dir / "tables/grouped_existing_v02_metrics_by_historical_exposure.csv",
        "report": run_dir / "diagnostics/grouped_existing_v02_historical_exposure_report.md",
        "provenance": run_dir
        / "logs/grouped_existing_v02_historical_exposure_provenance.json",
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {collisions}")

    for path in (
        dataset_path,
        historical_config_path,
        per_sample_path,
        source_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    historical, old_train_rows, old_val_rows = historical_source_sets(
        dataset_path, historical_config_path
    )
    metric_frame = pd.read_csv(per_sample_path, low_memory=False)
    samples = source_level_samples(metric_frame)
    source_manifest = pd.read_csv(source_manifest_path, low_memory=False)
    if len(source_manifest) != historical["dataset_n_sources"]:
        raise ValueError("Source manifest row count does not match dataset source count.")

    exposed_samples, group_metadata = add_exposure_flags(
        samples, source_manifest, old_train_rows, old_val_rows
    )
    rates = exposure_rates(exposed_samples)
    metrics = metrics_by_exposure(metric_frame, exposed_samples)

    expected_sample_n = len(samples)
    if int(rates.loc[rates["scope"] == "all_suites", "n_samples"].iloc[0]) != expected_sample_n:
        raise AssertionError("All-suite exposure count does not reconcile to unique samples.")
    all_metric_rows = metrics[
        (metrics["scope"] == "all_suites") & (metrics["exposure_subset"] == "all")
    ]
    if set(all_metric_rows["method"]) != set(METHODS):
        raise AssertionError("All-suite metric rows do not contain exactly both required methods.")
    if not all(all_metric_rows["n_samples"].eq(expected_sample_n)):
        raise AssertionError("All-suite metric counts do not reconcile to unique samples.")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_kind": "cpu_only_csv_metadata_aggregation_no_model_or_checkpoint_io",
        "exposure_definition_primary": "duplicate_safe_group_identity",
        "exposure_definition_sensitivity": "direct_historical_row_index",
        "historical_split": historical,
        "group_expansion": group_metadata,
        "inputs": {
            "dataset": {
                "path": str(dataset_path),
                "size_bytes": dataset_path.stat().st_size,
            },
            "historical_run_config": {
                "path": str(historical_config_path),
                "sha256": sha256_file(historical_config_path),
            },
            "per_sample_table": {
                "path": str(per_sample_path),
                "sha256": sha256_file(per_sample_path),
                "rows": len(metric_frame),
                "unique_samples": expected_sample_n,
            },
            "source_split_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
                "rows": len(source_manifest),
            },
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "validation": {
            "identity_v02_sample_sets_aligned": True,
            "source_metadata_constant_across_methods": True,
            "per_sample_sources_match_source_manifest_groups": True,
            "historical_train_validation_row_sets_disjoint": True,
            "all_suite_counts_reconciled": True,
            "stored_worse_flags_match_aligned_metric_comparison": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    report = build_report(historical, group_metadata, rates, metrics, paths)
    write_csv_exclusive(rates, paths["rates"])
    write_csv_exclusive(metrics, paths["metrics"])
    write_text_exclusive(report, paths["report"])
    write_json_exclusive(provenance, paths["provenance"])

    all_rates = rates[rates["scope"] == "all_suites"].iloc[0]
    clean_n = int(all_rates["clean_neither_old_train_or_validation_n"])
    union_n = int(all_rates["old_train_or_validation_either_role_exposed_n"])
    print(f"Unique grouped evaluation samples: {expected_sample_n}")
    print(f"Old train-or-validation source-identity exposed: {union_n}")
    print(f"Clean neither old train nor validation: {clean_n}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
