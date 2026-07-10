"""Append-only post-processing for a completed Thayer-BR v0.3 Delta run.

The training script deliberately keeps each experiment self-contained.  This
finalizer adds comparisons with the earlier v0.3 Color/Structure run only when
the two deterministic evaluation datasets are demonstrably aligned.  It never
modifies an existing artifact: every output is a new supplemental file opened
with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUN_ROOT = (PROJECT_ROOT / "outputs" / "runs").resolve()

DELTA_TABLE_CANDIDATES = (
    "v03_delta_per_sample_metrics.csv",
    "v03_color_per_sample_metrics.csv",
)
PRIOR_TABLE_CANDIDATES = (
    "v03_color_per_sample_metrics.csv",
    "v03_delta_per_sample_metrics.csv",
)

KEY_COLUMNS = ("suite", "index")
INTEGER_METADATA_COLUMNS = ("shift_x", "shift_y", "shift_distance")
CATEGORICAL_METADATA_COLUMNS = ("generation_difficulty", "training_component")
FLOAT_METADATA_COLUMNS = (
    "brightness",
    "blur_sigma",
    "noise_std",
    "rotation",
    "target_radius",
    "contaminant_radius",
    "size_ratio",
    "mask_fraction",
    "core_obstruction_fraction",
    "blend_severity_score",
    "core_affected_fraction",
    "halo_band_fraction",
)
BASELINE_ALIGNMENT_COLUMNS = (
    "identity_whole_mse",
    "identity_affected_mse",
    "identity_affected_delta_e2000_mean",
    "br_v02_moderate_affected_mse",
    "br_v02_moderate_affected_delta_e2000_mean",
)

METADATA_ATOL = 1e-9
METADATA_RTOL = 1e-8
BASELINE_ATOL = 1e-7
BASELINE_RTOL = 1e-5

SUITE_DISPLAY_ORDER = (
    "normal",
    "hard_stress",
    "compact_bright",
    "high_core_obstruction",
    "halo_band",
    "color_saturation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely add prior-v0.3 comparisons and missing figures to a completed "
            "Thayer-BR v0.3 Delta run."
        )
    )
    parser.add_argument(
        "--delta-run-dir",
        type=Path,
        required=True,
        help="Completed outputs/runs/br_v03_delta_candidate_* directory.",
    )
    parser.add_argument(
        "--prior-color-run-dir",
        "--prior-color-structure-run-dir",
        "--prior-v03-color-run-dir",
        dest="prior_color_run_dir",
        type=Path,
        default=None,
        help=(
            "Optional completed v0.3 Color/Structure run. When supplied, its "
            "per-sample metrics are merged only after strict alignment checks."
        ),
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def require_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def require_delta_under_outputs(run_dir: Path) -> None:
    try:
        run_dir.relative_to(OUTPUT_RUN_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Delta run must be inside the ignored outputs/runs directory: "
            f"{run_dir}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_completed_run(run_dir: Path, role: str) -> None:
    integrity_path = run_dir / "logs" / "checkpoint_integrity_comparison.json"
    if not integrity_path.exists():
        raise FileNotFoundError(
            f"{role} run has no completed checkpoint-integrity comparison: "
            f"{integrity_path}"
        )
    integrity = load_json(integrity_path)
    if not bool(integrity.get("old_checkpoints_unchanged", False)):
        raise RuntimeError(
            f"{role} run did not record unchanged historical checkpoints: "
            f"{integrity_path}"
        )


def validate_experiment_identity(run_dir: Path, role: str) -> dict[str, Any]:
    loss_path = run_dir / "logs" / "loss_config.yaml"
    config_path = run_dir / "logs" / "run_config.yaml"
    if not loss_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"{role} run is missing loss_config.yaml or run_config.yaml in logs/."
        )
    loss_config = load_yaml(loss_path)
    run_config = load_yaml(config_path)
    if not isinstance(loss_config, dict) or not isinstance(run_config, dict):
        raise ValueError(f"{role} run configuration files must contain mappings.")

    color_weight = float(loss_config.get("color_loss_weight", float("nan")))
    title = str(run_config.get("experiment_title", "")).strip()
    title_source = "logs/run_config.yaml"
    if not title and role == "Prior Color/Structure":
        # The completed legacy v0.3 run predates the experiment_title field in
        # run_config.yaml. Recover it only from that run's immutable diagnostic
        # header; do not infer identity from a caller-provided label alone.
        legacy_report = run_dir / "diagnostics/v03_color_structure_report.md"
        if legacy_report.exists():
            first_line = legacy_report.read_text(encoding="utf-8").splitlines()[0]
            title = first_line.lstrip("#").strip()
            title_source = project_relative(legacy_report)
    if role == "Delta":
        if not np.isclose(color_weight, 0.10, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Expected Delta color_loss_weight=0.10, found {color_weight!r}."
            )
        if "delta" not in title.lower() and "delta" not in run_dir.name.lower():
            raise ValueError(
                f"Run does not identify itself as Delta: title={title!r}, dir={run_dir.name!r}."
            )
    elif role == "Prior Color/Structure":
        if not np.isclose(color_weight, 0.05, atol=1e-12, rtol=0.0):
            raise ValueError(
                "Expected prior Color/Structure color_loss_weight=0.05, "
                f"found {color_weight!r}."
            )
        lower_title = title.lower()
        if "color" not in lower_title and "structure" not in lower_title:
            raise ValueError(
                f"Prior run does not identify Color/Structure: title={title!r}."
            )
    else:
        raise KeyError(f"Unknown experiment role: {role}")

    return {
        "experiment_title": title,
        "experiment_title_source": title_source,
        "color_loss_weight": color_weight,
        "loss_config_path": project_relative(loss_path),
        "run_config_path": project_relative(config_path),
    }


def locate_unique_table(run_dir: Path, candidates: tuple[str, ...], label: str) -> Path:
    table_dir = run_dir / "tables"
    found = [table_dir / name for name in candidates if (table_dir / name).exists()]
    if not found:
        raise FileNotFoundError(
            f"Could not find {label} consolidated per-sample table. Tried: "
            + ", ".join(str(table_dir / name) for name in candidates)
        )
    if len(found) > 1:
        raise RuntimeError(
            f"Ambiguous {label} per-sample tables; refusing to guess: "
            + ", ".join(str(path) for path in found)
        )
    return found[0]


def csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def find_method_prefix(
    columns: list[str],
    candidates: tuple[str, ...],
    label: str,
) -> str:
    required_suffixes = (
        "affected_mse",
        "affected_delta_e2000_mean",
        "affected_lab_chroma_mae",
        "affected_gradient_error",
        "halo_band_mse",
    )
    matches = [
        prefix
        for prefix in candidates
        if all(f"{prefix}{suffix}" in columns for suffix in required_suffixes)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} method prefix, found {matches!r}."
        )
    return matches[0]


def normalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [col for col in KEY_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(f"{label} table is missing key columns: {missing}")
    normalized = frame.copy()
    if normalized["suite"].isna().any():
        raise ValueError(f"{label} table has missing suite values.")
    normalized["suite"] = normalized["suite"].astype(str)
    numeric_index = pd.to_numeric(normalized["index"], errors="raise")
    if numeric_index.isna().any() or not np.all(
        np.isclose(numeric_index.to_numpy(dtype=float), np.round(numeric_index), atol=0.0)
    ):
        raise ValueError(f"{label} table index values must be finite integers.")
    normalized["index"] = numeric_index.astype(np.int64)
    duplicates = normalized.duplicated(list(KEY_COLUMNS), keep=False)
    if bool(duplicates.any()):
        rows = normalized.loc[duplicates, list(KEY_COLUMNS)].head(10).to_dict("records")
        raise ValueError(f"{label} table contains duplicate suite/index keys: {rows}")
    return normalized


def row_keys(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return list(zip(frame["suite"].tolist(), frame["index"].astype(int).tolist()))


def compare_numeric_column(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    *,
    atol: float,
    rtol: float,
    exact_integer: bool = False,
) -> dict[str, Any]:
    left_values = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float)
    right_values = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float)
    if exact_integer:
        if np.isnan(left_values).any() or np.isnan(right_values).any():
            raise ValueError(f"Alignment column {column} contains missing integer values.")
        matched = left_values == right_values
    else:
        matched = np.isclose(
            left_values,
            right_values,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
    if not bool(np.all(matched)):
        positions = np.flatnonzero(~matched)[:10]
        examples = [
            {
                "suite": str(left.iloc[pos]["suite"]),
                "index": int(left.iloc[pos]["index"]),
                "delta": float(left_values[pos]),
                "prior": float(right_values[pos]),
            }
            for pos in positions
        ]
        raise ValueError(
            f"Per-sample alignment failed for {column}; mismatches include {examples}."
        )
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    max_abs_diff = (
        float(np.max(np.abs(left_values[finite] - right_values[finite])))
        if bool(np.any(finite))
        else 0.0
    )
    return {
        "column": column,
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference": max_abs_diff,
    }


def align_prior_metrics(
    delta: pd.DataFrame,
    prior: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    delta = normalize_keys(delta, "Delta")
    prior = normalize_keys(prior, "prior Color/Structure")
    delta_keys = row_keys(delta)
    prior_keys = row_keys(prior)
    delta_key_set = set(delta_keys)
    prior_key_set = set(prior_keys)
    if delta_key_set != prior_key_set:
        missing_prior = sorted(delta_key_set - prior_key_set)[:10]
        extra_prior = sorted(prior_key_set - delta_key_set)[:10]
        raise ValueError(
            "Delta and prior suite/index key sets differ; "
            f"missing in prior={missing_prior}, extra in prior={extra_prior}."
        )

    prior_positions = {key: pos for pos, key in enumerate(prior_keys)}
    aligned_prior = prior.iloc[[prior_positions[key] for key in delta_keys]].reset_index(drop=True)
    delta = delta.reset_index(drop=True)
    if row_keys(delta) != row_keys(aligned_prior):
        raise AssertionError("Internal suite/index alignment ordering failure.")

    required_columns = (
        *INTEGER_METADATA_COLUMNS,
        *CATEGORICAL_METADATA_COLUMNS,
        *FLOAT_METADATA_COLUMNS,
        *BASELINE_ALIGNMENT_COLUMNS,
    )
    for column in required_columns:
        if column not in delta.columns or column not in aligned_prior.columns:
            raise KeyError(f"Strict alignment requires shared column: {column}")

    checks: list[dict[str, Any]] = []
    for column in INTEGER_METADATA_COLUMNS:
        checks.append(
            compare_numeric_column(
                delta,
                aligned_prior,
                column,
                atol=0.0,
                rtol=0.0,
                exact_integer=True,
            )
        )
    for column in CATEGORICAL_METADATA_COLUMNS:
        delta_values = delta[column].fillna("<NA>").astype(str).to_numpy()
        prior_values = aligned_prior[column].fillna("<NA>").astype(str).to_numpy()
        matched = delta_values == prior_values
        if not bool(np.all(matched)):
            pos = int(np.flatnonzero(~matched)[0])
            raise ValueError(
                f"Per-sample alignment failed for {column} at "
                f"{delta.iloc[pos]['suite']}/{int(delta.iloc[pos]['index'])}: "
                f"Delta={delta_values[pos]!r}, prior={prior_values[pos]!r}."
            )
        checks.append({"column": column, "comparison": "exact normalized string"})
    for column in FLOAT_METADATA_COLUMNS:
        checks.append(
            compare_numeric_column(
                delta,
                aligned_prior,
                column,
                atol=METADATA_ATOL,
                rtol=METADATA_RTOL,
            )
        )
    for column in BASELINE_ALIGNMENT_COLUMNS:
        checks.append(
            compare_numeric_column(
                delta,
                aligned_prior,
                column,
                atol=BASELINE_ATOL,
                rtol=BASELINE_RTOL,
            )
        )

    suites = list(dict.fromkeys(delta["suite"].tolist()))
    return aligned_prior, {
        "status": "strictly_aligned",
        "row_count": int(len(delta)),
        "suite_count": int(len(suites)),
        "suites": suites,
        "metadata_atol": METADATA_ATOL,
        "metadata_rtol": METADATA_RTOL,
        "baseline_atol": BASELINE_ATOL,
        "baseline_rtol": BASELINE_RTOL,
        "checks": checks,
    }


def safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    result = np.full(num.shape, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    result[valid] = num[valid] / den[valid]
    return pd.Series(result, index=numerator.index)


def add_delta_aliases(delta: pd.DataFrame, delta_prefix: str) -> pd.DataFrame:
    if delta_prefix == "br_v03_delta_":
        return delta.copy()
    method_columns = [col for col in delta.columns if col.startswith(delta_prefix)]
    aliases = delta[method_columns].rename(
        columns={
            col: f"br_v03_delta_{col[len(delta_prefix):]}" for col in method_columns
        }
    )
    collisions = sorted(set(aliases.columns) & set(delta.columns))
    if collisions:
        raise RuntimeError(f"Delta alias columns already exist: {collisions[:10]}")
    return pd.concat([delta.reset_index(drop=True), aliases.reset_index(drop=True)], axis=1)


def merge_prior_metrics(
    delta_with_aliases: pd.DataFrame,
    aligned_prior: pd.DataFrame,
    prior_prefix: str,
) -> pd.DataFrame:
    prior_method_columns = [
        column for column in aligned_prior.columns if column.startswith(prior_prefix)
    ]
    if not prior_method_columns:
        raise KeyError(f"No prior method columns use prefix {prior_prefix!r}.")
    prior_metrics = aligned_prior[prior_method_columns].rename(
        columns={
            col: f"br_v03_color_structure_{col[len(prior_prefix):]}"
            for col in prior_method_columns
        }
    )
    collisions = sorted(set(prior_metrics.columns) & set(delta_with_aliases.columns))
    if collisions:
        raise RuntimeError(f"Prior supplemental columns already exist: {collisions[:10]}")

    merged = pd.concat(
        [delta_with_aliases.reset_index(drop=True), prior_metrics.reset_index(drop=True)],
        axis=1,
    )
    delta_mse = merged["br_v03_delta_affected_mse"]
    prior_mse = merged["br_v03_color_structure_affected_mse"]
    delta_de = merged["br_v03_delta_affected_delta_e2000_mean"]
    prior_de = merged["br_v03_color_structure_affected_delta_e2000_mean"]
    comparison = pd.DataFrame(
        {
            "delta_beats_prior_v03_color_affected_mse": delta_mse < prior_mse,
            "delta_to_prior_v03_color_affected_mse_ratio": safe_ratio_series(
                delta_mse, prior_mse
            ),
            "delta_beats_prior_v03_color_delta_e2000": delta_de < prior_de,
            "delta_to_prior_v03_color_delta_e2000_ratio": safe_ratio_series(
                delta_de, prior_de
            ),
        }
    )
    return pd.concat([merged, comparison], axis=1)


def finite_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_ratio_value(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def win_rate(frame: pd.DataFrame, left: str, right: str) -> float:
    left_values = pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    return float(np.mean(left_values[valid] < right_values[valid])) if np.any(valid) else float("nan")


def build_prior_comparison_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for suite, frame in merged.groupby("suite", sort=False):
        identity_mse = finite_mean(frame, "identity_affected_mse")
        delta_mse = finite_mean(frame, "br_v03_delta_affected_mse")
        v02_mse = finite_mean(frame, "br_v02_moderate_affected_mse")
        prior_mse = finite_mean(frame, "br_v03_color_structure_affected_mse")
        delta_de = finite_mean(frame, "br_v03_delta_affected_delta_e2000_mean")
        v02_de = finite_mean(frame, "br_v02_moderate_affected_delta_e2000_mean")
        prior_de = finite_mean(
            frame, "br_v03_color_structure_affected_delta_e2000_mean"
        )
        rows.append(
            {
                "suite": suite,
                "n": int(len(frame)),
                "delta_affected_mse": delta_mse,
                "v02_moderate_affected_mse": v02_mse,
                "prior_v03_color_affected_mse": prior_mse,
                "delta_to_v02_affected_mse_ratio": safe_ratio_value(delta_mse, v02_mse),
                "delta_to_prior_v03_color_affected_mse_ratio": safe_ratio_value(
                    delta_mse, prior_mse
                ),
                "prior_v03_color_to_v02_affected_mse_ratio": safe_ratio_value(
                    prior_mse, v02_mse
                ),
                "delta_improvement_vs_identity": safe_ratio_value(identity_mse, delta_mse),
                "v02_moderate_improvement_vs_identity": safe_ratio_value(
                    identity_mse, v02_mse
                ),
                "prior_v03_color_improvement_vs_identity": safe_ratio_value(
                    identity_mse, prior_mse
                ),
                "delta_vs_prior_v03_color_affected_mse_win_rate": win_rate(
                    frame,
                    "br_v03_delta_affected_mse",
                    "br_v03_color_structure_affected_mse",
                ),
                "delta_vs_v02_affected_mse_win_rate": win_rate(
                    frame,
                    "br_v03_delta_affected_mse",
                    "br_v02_moderate_affected_mse",
                ),
                "delta_delta_e2000": delta_de,
                "v02_moderate_delta_e2000": v02_de,
                "prior_v03_color_delta_e2000": prior_de,
                "delta_to_v02_delta_e2000_ratio": safe_ratio_value(delta_de, v02_de),
                "delta_to_prior_v03_color_delta_e2000_ratio": safe_ratio_value(
                    delta_de, prior_de
                ),
                "delta_vs_prior_v03_color_delta_e2000_win_rate": win_rate(
                    frame,
                    "br_v03_delta_affected_delta_e2000_mean",
                    "br_v03_color_structure_affected_delta_e2000_mean",
                ),
                "delta_gradient_error": finite_mean(
                    frame, "br_v03_delta_affected_gradient_error"
                ),
                "prior_v03_color_gradient_error": finite_mean(
                    frame, "br_v03_color_structure_affected_gradient_error"
                ),
                "delta_chroma_error": finite_mean(
                    frame, "br_v03_delta_affected_lab_chroma_mae"
                ),
                "prior_v03_color_chroma_error": finite_mean(
                    frame, "br_v03_color_structure_affected_lab_chroma_mae"
                ),
                "delta_halo_band_mse": finite_mean(
                    frame, "br_v03_delta_halo_band_mse"
                ),
                "prior_v03_color_halo_band_mse": finite_mean(
                    frame, "br_v03_color_structure_halo_band_mse"
                ),
                "delta_worse_than_identity_count": int(
                    (
                        pd.to_numeric(frame["br_v03_delta_affected_mse"], errors="coerce")
                        > pd.to_numeric(frame["identity_affected_mse"], errors="coerce")
                    ).sum()
                ),
                "prior_v03_color_worse_than_identity_count": int(
                    (
                        pd.to_numeric(
                            frame["br_v03_color_structure_affected_mse"],
                            errors="coerce",
                        )
                        > pd.to_numeric(frame["identity_affected_mse"], errors="coerce")
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def ordered_suites(frame: pd.DataFrame) -> list[str]:
    observed = list(dict.fromkeys(frame["suite"].astype(str).tolist()))
    return [suite for suite in SUITE_DISPLAY_ORDER if suite in observed] + [
        suite for suite in observed if suite not in SUITE_DISPLAY_ORDER
    ]


def figure_to_png(fig: plt.Figure, dpi: int = 220) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_normal_stress_improvement(
    frame: pd.DataFrame,
    include_prior: bool,
) -> bytes:
    suites = ["normal", "hard_stress"]
    missing = [suite for suite in suites if suite not in set(frame["suite"])]
    if missing:
        raise ValueError(f"Improvement chart requires suites missing from Delta: {missing}")
    methods = [
        ("br_v02_moderate_affected_mse", "BR v0.2 Moderate", "#c28b2c"),
    ]
    if include_prior:
        methods.append(
            (
                "br_v03_color_structure_affected_mse",
                "BR v0.3 Color/Structure",
                "#7b5fa3",
            )
        )
    methods.append(("br_v03_delta_affected_mse", "BR v0.3 Delta", "#2c8f7b"))

    x = np.arange(len(suites), dtype=float)
    width = 0.22 if len(methods) == 3 else 0.30
    offsets = np.linspace(
        -width * (len(methods) - 1) / 2,
        width * (len(methods) - 1) / 2,
        len(methods),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (column, label, color), offset in zip(methods, offsets):
        values = []
        for suite in suites:
            subset = frame[frame["suite"] == suite]
            identity = finite_mean(subset, "identity_affected_mse")
            model = finite_mean(subset, column)
            values.append(safe_ratio_value(identity, model))
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1fx", padding=2, fontsize=8)
    ax.set_xticks(x, ["Normal held-out", "Hard stress"])
    ax.set_ylabel("Affected-region improvement vs identity")
    ax.set_title("Normal vs Stress Improvement Ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    return figure_to_png(fig)


def render_three_way_variant_chart(merged: pd.DataFrame) -> bytes:
    suites = ordered_suites(merged)
    methods = (
        ("br_v02_moderate_affected_mse", "BR v0.2 Moderate", "#c28b2c"),
        (
            "br_v03_color_structure_affected_mse",
            "BR v0.3 Color/Structure",
            "#7b5fa3",
        ),
        ("br_v03_delta_affected_mse", "BR v0.3 Delta", "#2c8f7b"),
    )
    x = np.arange(len(suites), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.5, 1.35 * len(suites)), 4.8))
    for method_index, (column, label, color) in enumerate(methods):
        values = [
            finite_mean(merged[merged["suite"] == suite], column) for suite in suites
        ]
        offset = (method_index - 1) * width
        ax.bar(x + offset, values, width, color=color, label=label)
    ax.set_xticks(x, [suite.replace("_", " ") for suite in suites], rotation=20, ha="right")
    ax.set_ylabel("Affected-region MSE (lower is better)")
    ax.set_title("BR v0.2 Moderate vs v0.3 Color/Structure vs v0.3 Delta")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    return figure_to_png(fig)


def select_color_artifact_grid(
    delta_run_dir: Path,
    frame: pd.DataFrame,
    include_prior: bool,
) -> tuple[bytes | None, dict[str, Any]]:
    delta_de = pd.to_numeric(
        frame["br_v03_delta_affected_delta_e2000_mean"], errors="coerce"
    )
    v02_de = pd.to_numeric(
        frame["br_v02_moderate_affected_delta_e2000_mean"], errors="coerce"
    )
    v02_ratio = safe_ratio_series(delta_de, v02_de)
    worsened = v02_ratio > 1.0
    score = v02_ratio.copy()
    prior_ratio = pd.Series(np.nan, index=frame.index, dtype=float)
    if include_prior:
        prior_de = pd.to_numeric(
            frame["br_v03_color_structure_affected_delta_e2000_mean"], errors="coerce"
        )
        prior_ratio = safe_ratio_series(delta_de, prior_de)
        worsened = worsened | (prior_ratio > 1.0)
        score = pd.concat([v02_ratio, prior_ratio], axis=1).max(axis=1, skipna=True)

    candidates = frame.loc[worsened].copy()
    if candidates.empty:
        return None, {
            "status": "not_generated",
            "reason": "No evaluated sample had worse Delta E 2000 than v0.2 or the supplied prior run.",
        }
    candidates["_color_artifact_score"] = score.loc[candidates.index]
    key_to_row = {
        (str(row["suite"]), int(row["index"])): row
        for _, row in candidates.iterrows()
    }

    stored_candidates: list[tuple[float, Path, Path, pd.Series]] = []
    grid_dir = delta_run_dir / "example_grids"
    for json_path in sorted(grid_dir.glob("*.json")):
        try:
            payload = load_json(json_path)
            key = (str(payload["suite"]), int(payload["index"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        row = key_to_row.get(key)
        png_path = json_path.with_suffix(".png")
        if row is not None and png_path.exists():
            stored_candidates.append(
                (float(row["_color_artifact_score"]), json_path, png_path, row)
            )

    top_row = candidates.sort_values("_color_artifact_score", ascending=False).iloc[0]
    if not stored_candidates:
        return None, {
            "status": "not_generated",
            "reason": (
                "Color-worsened samples exist, but the completed run retained no "
                "qualitative source grid for any such suite/index. Predictions were "
                "not reconstructed or fabricated during post-processing."
            ),
            "strongest_metric_only_candidate": {
                "suite": str(top_row["suite"]),
                "index": int(top_row["index"]),
                "delta_to_v02_delta_e2000_ratio": float(v02_ratio.loc[top_row.name]),
                "delta_to_prior_delta_e2000_ratio": (
                    float(prior_ratio.loc[top_row.name]) if include_prior else None
                ),
            },
        }

    stored_candidates.sort(
        key=lambda item: (item[0], "failure" in item[1].name.lower(), item[1].name),
        reverse=True,
    )
    artifact_score, source_json, source_png, row = stored_candidates[0]
    image = plt.imread(source_png)
    suite = str(row["suite"])
    index = int(row["index"])
    delta_value = float(row["br_v03_delta_affected_delta_e2000_mean"])
    v02_value = float(row["br_v02_moderate_affected_delta_e2000_mean"])
    annotation = (
        f"Affected Delta E 2000 (lower is better): v0.2={v02_value:.3f}, "
        f"Delta={delta_value:.3f}, Delta/v0.2={delta_value / v02_value:.2f}x"
    )
    prior_value: float | None = None
    if include_prior:
        prior_value = float(row["br_v03_color_structure_affected_delta_e2000_mean"])
        annotation += (
            f", prior v0.3={prior_value:.3f}, Delta/prior={delta_value / prior_value:.2f}x"
        )

    height, width = image.shape[:2]
    figure_width = 15.5
    figure_height = max(3.0, figure_width * height / width + 1.0)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"Delta Color-Artifact Candidate — {suite.replace('_', ' ')}, sample {index}\n"
        "The source grid's BR v0.3 panel is the Delta candidate.",
        fontsize=11,
    )
    fig.text(0.5, 0.02, annotation, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    png_bytes = figure_to_png(fig, dpi=180)
    return png_bytes, {
        "status": "generated_from_existing_grid",
        "suite": suite,
        "index": index,
        "selection_score": artifact_score,
        "source_grid_json": project_relative(source_json),
        "source_grid_png": project_relative(source_png),
        "delta_affected_delta_e2000_mean": delta_value,
        "v02_moderate_affected_delta_e2000_mean": v02_value,
        "delta_to_v02_delta_e2000_ratio": delta_value / v02_value,
        "prior_v03_color_affected_delta_e2000_mean": prior_value,
        "delta_to_prior_delta_e2000_ratio": (
            delta_value / prior_value if prior_value is not None else None
        ),
        "note": (
            "This is an explicitly relabeled copy of an existing completed-run "
            "qualitative grid; no prediction was regenerated."
        ),
    }


def realized_training_mix(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "training_composition.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing training composition: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Training composition must be a mapping: {path}")
    result: dict[str, Any] = {"source": project_relative(path), "splits": {}}
    for split in ("train", "validation"):
        section = payload.get(split)
        if not isinstance(section, dict):
            raise KeyError(f"Training composition is missing {split!r}.")
        counts = section.get("counts")
        if not isinstance(counts, dict) or not counts:
            raise KeyError(f"Training composition {split!r} has no counts mapping.")
        numeric_counts = {str(key): int(value) for key, value in counts.items()}
        total = int(section.get("total", sum(numeric_counts.values())))
        if total <= 0 or sum(numeric_counts.values()) != total:
            raise ValueError(
                f"Training composition {split!r} counts do not sum to total={total}."
            )
        fractions = {key: value / total for key, value in numeric_counts.items()}
        result["splits"][split] = {
            "total": total,
            "counts": numeric_counts,
            "fractions": fractions,
            "artifact_note": section.get("target_distribution", {}).get("artifact_note"),
        }
    return result


def percent_mix_text(split: dict[str, Any]) -> str:
    labels = (
        ("normal_clean", "normal clean"),
        ("high_overlap_core", "high-overlap/core"),
        ("compact_bright", "compact bright"),
        ("brightness_size", "brightness/size"),
        ("low_overlap_easy", "low-overlap/easy"),
    )
    parts = []
    for key, label in labels:
        fraction = float(split["fractions"].get(key, 0.0))
        count = int(split["counts"].get(key, 0))
        parts.append(f"{100.0 * fraction:.1f}% {label} ({count:,})")
    return ", ".join(parts)


def summary_markdown(summary: pd.DataFrame | None) -> str:
    if summary is None:
        return "Prior Color/Structure run was not supplied; no cross-run metric merge was attempted."
    rows = [
        "| Suite | Delta affected MSE | v0.2 Moderate | Prior v0.3 Color | Delta/prior | Delta win rate vs prior | Delta DE2000 | Prior DE2000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        rows.append(
            "| {suite} | {delta:.6f} | {v02:.6f} | {prior:.6f} | {ratio:.3f} | {win:.1%} | {delta_de:.3f} | {prior_de:.3f} |".format(
                suite=str(row["suite"]).replace("_", " "),
                delta=float(row["delta_affected_mse"]),
                v02=float(row["v02_moderate_affected_mse"]),
                prior=float(row["prior_v03_color_affected_mse"]),
                ratio=float(row["delta_to_prior_v03_color_affected_mse_ratio"]),
                win=float(row["delta_vs_prior_v03_color_affected_mse_win_rate"]),
                delta_de=float(row["delta_delta_e2000"]),
                prior_de=float(row["prior_v03_color_delta_e2000"]),
            )
        )
    return "\n".join(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def safe_write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text.rstrip() + "\n")


def safe_write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(json_ready(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def safe_write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def safe_write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def preflight_destinations(paths: list[Path]) -> None:
    duplicates = sorted(
        str(path) for path in set(paths) if paths.count(path) > 1
    )
    if duplicates:
        raise RuntimeError(f"Duplicate planned destinations: {duplicates}")
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Post-processing is append-only; refusing because destinations exist: "
            + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        if not path.parent.exists() or not path.parent.is_dir():
            raise FileNotFoundError(
                f"Completed Delta run is missing output directory: {path.parent}"
            )


def build_report(
    *,
    delta_run_dir: Path,
    prior_run_dir: Path | None,
    delta_table_path: Path,
    prior_table_path: Path | None,
    experiment_info: dict[str, Any],
    prior_info: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
    training_mix: dict[str, Any],
    observed_suites: list[str],
    summary: pd.DataFrame | None,
    outputs: list[Path],
    color_artifact: dict[str, Any],
) -> str:
    train_mix = training_mix["splits"]["train"]
    val_mix = training_mix["splits"]["validation"]
    alignment_text = (
        f"Strict validation passed for {alignment['row_count']:,} rows across "
        f"{alignment['suite_count']} suites. Suite/index key sets matched exactly; "
        "integer shifts and categorical metadata matched exactly; blend metadata "
        f"matched with atol={METADATA_ATOL:g}, rtol={METADATA_RTOL:g}; identity and "
        f"v0.2 baseline metrics matched with atol={BASELINE_ATOL:g}, "
        f"rtol={BASELINE_RTOL:g}."
        if alignment is not None
        else "No prior run was supplied, so cross-run alignment was not attempted."
    )
    prior_source = (
        f"- Prior run: `{project_relative(prior_run_dir)}`\n"
        f"- Prior table: `{project_relative(prior_table_path)}`\n"
        f"- Prior experiment: {prior_info['experiment_title']} "
        f"(`color_loss_weight={prior_info['color_loss_weight']}`)"
        if prior_run_dir is not None and prior_table_path is not None and prior_info is not None
        else "- Prior run: not supplied"
    )
    artifact_text = (
        f"Generated from `{color_artifact['source_grid_png']}` for "
        f"{color_artifact['suite']}/{color_artifact['index']}."
        if color_artifact.get("status") == "generated_from_existing_grid"
        else str(color_artifact.get("reason", "No color-artifact grid was generated."))
    )
    missing_suites = [
        name
        for name in ("clean_source_filtered_normal", "artifact_heavy")
        if name not in observed_suites
    ]
    return f"""# Thayer-BR v0.3 Delta Candidate Post-processing Report

## Append-only status

This post-processing pass did not modify the completed training outputs, historical metrics, or any checkpoint. It created only the new supplemental artifacts listed below. Existing destinations are always treated as errors and are never overwritten.

## Sources

- Delta run: `{project_relative(delta_run_dir)}`
- Delta table: `{project_relative(delta_table_path)}`
- Delta experiment: {experiment_info['experiment_title']} (`color_loss_weight={experiment_info['color_loss_weight']}`)
{prior_source}

## Cross-run alignment

{alignment_text}

The prior v0.3 metrics were merged only after this validation. They are stored under the unambiguous `br_v03_color_structure_*` prefix; the new candidate is stored under `br_v03_delta_*` in the supplemental table.

## Corrected realized training distribution

The realized Delta training mix was **{percent_mix_text(train_mix)}**. This is 40/25/20/10/5, not the 35/25/20/10/5 text present in the original generated report. The validation mix was {percent_mix_text(val_mix)}.

No artifact/outlier bucket was used. The run's composition log states: {train_mix.get('artifact_note') or 'no source-quality artifact note was recorded'}

## Evaluation-suite availability

Observed suites: {', '.join(observed_suites)}.

Unavailable suites: {', '.join(missing_suites) if missing_suites else 'none'}. Clean-source filtering and artifact-heavy evaluation were not fabricated during post-processing because the completed run contains no source-quality artifact flags or retained predictions for new suites.

## Prior v0.3 comparison

{summary_markdown(summary)}

This supplemental comparison does not by itself promote Delta to current best. Thayer-BR v0.2 Moderate remains the reference until the full success criteria and qualitative evidence are adjudicated.

## Explicit color-artifact grid

{artifact_text}

## New artifacts

{chr(10).join(f'- `{project_relative(path)}`' for path in outputs)}
"""


def main() -> int:
    args = parse_args()
    delta_run_dir = resolve_input_path(args.delta_run_dir)
    prior_run_dir = (
        resolve_input_path(args.prior_color_run_dir)
        if args.prior_color_run_dir is not None
        else None
    )
    require_directory(delta_run_dir, "Delta run directory")
    require_delta_under_outputs(delta_run_dir)
    validate_completed_run(delta_run_dir, "Delta")
    delta_info = validate_experiment_identity(delta_run_dir, "Delta")

    if prior_run_dir is not None:
        require_directory(prior_run_dir, "Prior Color/Structure run directory")
        if prior_run_dir == delta_run_dir:
            raise ValueError("Delta and prior Color/Structure run directories must differ.")
        validate_completed_run(prior_run_dir, "Prior Color/Structure")
        prior_info = validate_experiment_identity(prior_run_dir, "Prior Color/Structure")
    else:
        prior_info = None

    delta_table_path = locate_unique_table(
        delta_run_dir, DELTA_TABLE_CANDIDATES, "Delta"
    )
    delta = normalize_keys(
        pd.read_csv(delta_table_path, low_memory=False), "Delta"
    ).reset_index(drop=True)
    delta_prefix = find_method_prefix(
        list(delta.columns), ("br_v03_delta_", "br_v03_color_"), "Delta"
    )
    delta_working = add_delta_aliases(delta, delta_prefix)

    prior_table_path: Path | None = None
    alignment: dict[str, Any] | None = None
    merged: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    if prior_run_dir is not None:
        prior_table_path = locate_unique_table(
            prior_run_dir, PRIOR_TABLE_CANDIDATES, "prior Color/Structure"
        )
        prior_columns = csv_columns(prior_table_path)
        prior_prefix = find_method_prefix(
            prior_columns,
            ("br_v03_color_", "br_v03_color_structure_"),
            "prior Color/Structure",
        )
        required_prior_columns = set(
            (
                *KEY_COLUMNS,
                *INTEGER_METADATA_COLUMNS,
                *CATEGORICAL_METADATA_COLUMNS,
                *FLOAT_METADATA_COLUMNS,
                *BASELINE_ALIGNMENT_COLUMNS,
            )
        )
        required_prior_columns.update(
            col for col in prior_columns if col.startswith(prior_prefix)
        )
        prior = pd.read_csv(
            prior_table_path,
            usecols=lambda col: col in required_prior_columns,
            low_memory=False,
        )
        aligned_prior, alignment = align_prior_metrics(delta, prior)
        merged = merge_prior_metrics(delta_working, aligned_prior, prior_prefix)
        summary = build_prior_comparison_summary(merged)

    figure_frame = merged if merged is not None else delta_working
    training_mix = realized_training_mix(delta_run_dir)
    observed_suites = ordered_suites(figure_frame)

    improvement_png = render_normal_stress_improvement(
        figure_frame, include_prior=merged is not None
    )
    variant_png = render_three_way_variant_chart(merged) if merged is not None else None
    color_grid_png, color_artifact = select_color_artifact_grid(
        delta_run_dir,
        figure_frame,
        include_prior=merged is not None,
    )

    tables_dir = delta_run_dir / "tables"
    paper_figures_dir = delta_run_dir / "paper_figures"
    example_grids_dir = delta_run_dir / "example_grids"
    diagnostics_dir = delta_run_dir / "diagnostics"
    logs_dir = delta_run_dir / "logs"

    merged_path = tables_dir / "v03_delta_per_sample_metrics_with_prior_color.csv"
    summary_path = tables_dir / "v03_delta_comparison_summary_with_prior_color.csv"
    improvement_path = paper_figures_dir / "normal_vs_stress_improvement_ratio_chart.png"
    variant_path = (
        paper_figures_dir / "v02_vs_v03_color_vs_v03_delta_variant_comparison.png"
    )
    color_grid_path = example_grids_dir / "delta_color_artifact.png"
    color_grid_json_path = example_grids_dir / "delta_color_artifact.json"
    report_path = diagnostics_dir / "v03_delta_postprocess_report.md"
    manifest_path = logs_dir / "v03_delta_postprocess_manifest.json"

    outputs: list[Path] = [improvement_path, report_path, manifest_path]
    if merged is not None:
        outputs.extend([merged_path, summary_path, variant_path])
    if color_grid_png is not None:
        outputs.extend([color_grid_path, color_grid_json_path])
    preflight_destinations(outputs)

    manifest = {
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "append_only_no_overwrite",
        "delta_run_dir": project_relative(delta_run_dir),
        "prior_color_run_dir": (
            project_relative(prior_run_dir) if prior_run_dir is not None else None
        ),
        "sources": {
            "delta_per_sample_table": {
                "path": project_relative(delta_table_path),
                "sha256": sha256_file(delta_table_path),
            },
            "prior_per_sample_table": (
                {
                    "path": project_relative(prior_table_path),
                    "sha256": sha256_file(prior_table_path),
                }
                if prior_table_path is not None
                else None
            ),
        },
        "experiment_info": delta_info,
        "prior_experiment_info": prior_info,
        "alignment": alignment,
        "realized_training_mix": training_mix,
        "observed_suites": observed_suites,
        "unavailable_suites": [
            suite
            for suite in ("clean_source_filtered_normal", "artifact_heavy")
            if suite not in observed_suites
        ],
        "color_artifact_grid": color_artifact,
        "outputs": [project_relative(path) for path in outputs],
    }
    report = build_report(
        delta_run_dir=delta_run_dir,
        prior_run_dir=prior_run_dir,
        delta_table_path=delta_table_path,
        prior_table_path=prior_table_path,
        experiment_info=delta_info,
        prior_info=prior_info,
        alignment=alignment,
        training_mix=training_mix,
        observed_suites=observed_suites,
        summary=summary,
        outputs=outputs,
        color_artifact=color_artifact,
    )

    if merged is not None and summary is not None and variant_png is not None:
        safe_write_csv(merged_path, merged)
        safe_write_csv(summary_path, summary)
        safe_write_bytes(variant_path, variant_png)
    safe_write_bytes(improvement_path, improvement_png)
    if color_grid_png is not None:
        safe_write_bytes(color_grid_path, color_grid_png)
        safe_write_json(color_grid_json_path, color_artifact)
    safe_write_text(report_path, report)
    safe_write_json(manifest_path, manifest)

    print(f"Delta post-processing complete: {project_relative(delta_run_dir)}")
    print(f"Created {len(outputs)} append-only artifacts; no existing file was modified.")
    if alignment is not None:
        print(
            f"Prior Color/Structure alignment passed for {alignment['row_count']} rows "
            f"across {alignment['suite_count']} suites."
        )
    else:
        print("No prior Color/Structure run supplied; cross-run comparison was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
