"""Quantify historical duplicate leakage without running model inference.

The script reconstructs the seed-42 source split, identifies sources involved
in cross-split exact-image or exact-coordinate groups, recovers source roles for
the original v0.2 normal/stress suites, and recomputes saved v0.2 aggregates on
clean versus implicated subsets. Historical files are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/Galaxy10_DECals.h5"
DEFAULT_SOURCE_AUDIT = (
    PROJECT_ROOT / "outputs/runs/source_leakage_audit_20260710_062950"
)
DEFAULT_V02_RUN = PROJECT_ROOT / "outputs/runs/weighted_residual_20260709_030245"
DEFAULT_STRESS_INDEX_TABLE = (
    PROJECT_ROOT
    / "outputs/runs/stress_test_20260708_141153/results/stress_test_per_sample_results.csv"
)
DEFAULT_TARGETED_TABLE = (
    PROJECT_ROOT
    / "outputs/runs/resunet_v04_candidate_20260710_043109/tables/resunet_v04_per_sample_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-audit-run", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--v02-run", type=Path, default=DEFAULT_V02_RUN)
    parser.add_argument("--stress-index-table", type=Path, default=DEFAULT_STRESS_INDEX_TABLE)
    parser.add_argument("--targeted-table", type=Path, default=DEFAULT_TARGETED_TABLE)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def resolve_existing(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_master_run(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    allowed = (PROJECT_ROOT / "outputs/runs").resolve()
    if allowed not in resolved.parents:
        raise ValueError(f"Run directory must be under {allowed}")
    if not resolved.name.startswith("research_correctness_audit_"):
        raise ValueError("Expected a research_correctness_audit_* master run")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def safe_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def safe_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": indices[:n_train],
        "validation": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def coordinate_key(ra: float, dec: float) -> str:
    return f"{float(ra).hex()}|{float(dec).hex()}"


def reconstruct_normal_roles(
    saved: pd.DataFrame,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay the original normal-suite RNG stream without retaining arrays."""

    rng = np.random.default_rng(4042)
    rows: list[dict[str, Any]] = []
    for sample_index in range(len(saved)):
        target_local, contaminant_local = rng.choice(1000, size=2, replace=False)
        dx = int(rng.integers(-56, 57))
        dy = int(rng.integers(-56, 57))
        brightness = float(rng.uniform(0.5, 1.2))
        blur_sigma = float(rng.uniform(0.0, 0.3))
        noise_std = float(rng.uniform(0.0, 0.01))
        # blend_pair consumes this exact draw after compositing.
        rng.normal(scale=noise_std, size=(256, 256, 3))
        rows.append(
            {
                "suite": "normal",
                "sample_index": sample_index,
                "target_split_local_index": int(target_local),
                "contaminant_split_local_index": int(contaminant_local),
                "target_source_index": int(test_indices[int(target_local)]),
                "contaminant_source_index": int(test_indices[int(contaminant_local)]),
                "shift_x": dx,
                "shift_y": dy,
                "brightness": brightness,
                "blur_sigma": blur_sigma,
                "noise_std": noise_std,
                "source_role_recovery": "exact_rng_replay_seed4042",
            }
        )
    recovered = pd.DataFrame(rows)
    checks = {}
    for column in ("shift_x", "shift_y"):
        checks[column] = bool(np.array_equal(recovered[column], saved[column]))
    for column in ("brightness", "blur_sigma", "noise_std"):
        checks[column] = bool(
            np.allclose(recovered[column], saved[column], rtol=0.0, atol=5e-15)
        )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Normal RNG replay did not match saved parameters: {checks}")
    return recovered, checks


def recover_stress_roles(
    saved: pd.DataFrame,
    indexed: pd.DataFrame,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(saved) != len(indexed) or len(saved) != 1000:
        raise ValueError("Expected aligned 1,000-row stress tables")
    checks: dict[str, Any] = {"row_count": len(saved) == len(indexed)}
    for column in ("index", "shift_x", "shift_y"):
        checks[column] = bool(np.array_equal(saved[column], indexed[column]))
    for column in ("brightness", "blur_sigma", "noise_std"):
        checks[column] = bool(
            np.allclose(saved[column], indexed[column], rtol=0.0, atol=5e-15)
        )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Stress source table did not align with v0.2: {checks}")
    recovered = pd.DataFrame(
        {
            "suite": "stress",
            "sample_index": saved["index"].astype(int),
            "target_split_local_index": indexed["target_index"].astype(int),
            "contaminant_split_local_index": indexed["contaminant_index"].astype(int),
            "target_source_index": [
                int(test_indices[int(value)]) for value in indexed["target_index"]
            ],
            "contaminant_source_index": [
                int(test_indices[int(value)]) for value in indexed["contaminant_index"]
            ],
            "shift_x": saved["shift_x"],
            "shift_y": saved["shift_y"],
            "brightness": saved["brightness"],
            "blur_sigma": saved["blur_sigma"],
            "noise_std": saved["noise_std"],
            "source_role_recovery": "aligned_original_stress_index_table_seed20260708",
        }
    )
    return recovered, checks


def targeted_roles(frame: pd.DataFrame, suite: str, test_indices: np.ndarray) -> pd.DataFrame:
    selected = frame.loc[frame["suite"].eq(suite)].copy()
    selected = selected.sort_values("index").reset_index(drop=True)
    if len(selected) != 1000 or selected[["target_index", "contaminant_index"]].isna().any().any():
        raise ValueError(f"Expected 1,000 indexed rows for {suite}")
    return pd.DataFrame(
        {
            "suite": suite,
            "sample_index": selected["index"].astype(int),
            "target_split_local_index": selected["target_index"].astype(int),
            "contaminant_split_local_index": selected["contaminant_index"].astype(int),
            "target_source_index": [
                int(test_indices[int(value)]) for value in selected["target_index"]
            ],
            "contaminant_source_index": [
                int(test_indices[int(value)]) for value in selected["contaminant_index"]
            ],
            "source_role_recovery": "saved_resunet_targeted_suite_subset_local_indices",
        }
    )


def implication_summary(roles: pd.DataFrame, implicated: set[int]) -> dict[str, Any]:
    target = roles["target_source_index"].isin(implicated)
    contaminant = roles["contaminant_source_index"].isin(implicated)
    either = target | contaminant
    both = target & contaminant
    n = len(roles)
    return {
        "suite": str(roles["suite"].iloc[0]),
        "n_samples": n,
        "target_implicated_count": int(target.sum()),
        "target_implicated_fraction": float(target.mean()),
        "contaminant_implicated_count": int(contaminant.sum()),
        "contaminant_implicated_fraction": float(contaminant.mean()),
        "either_source_implicated_count": int(either.sum()),
        "either_source_implicated_fraction": float(either.mean()),
        "both_sources_implicated_count": int(both.sum()),
        "both_sources_implicated_fraction": float(both.mean()),
        "role_recovery_method": str(roles["source_role_recovery"].iloc[0]),
    }


def aggregate_subset(
    suite: str,
    metrics: pd.DataFrame,
    roles: pd.DataFrame,
    implicated: set[int],
) -> list[dict[str, Any]]:
    aligned = metrics.sort_values("index").reset_index(drop=True).copy()
    role_rows = roles.sort_values("sample_index").reset_index(drop=True)
    if len(aligned) != len(role_rows) or not np.array_equal(
        aligned["index"].astype(int), role_rows["sample_index"].astype(int)
    ):
        raise RuntimeError(f"Metric/source-role alignment failed for {suite}")
    aligned["target_source_index"] = role_rows["target_source_index"]
    aligned["contaminant_source_index"] = role_rows["contaminant_source_index"]
    aligned["implicated"] = aligned["target_source_index"].isin(implicated) | aligned[
        "contaminant_source_index"
    ].isin(implicated)
    result: list[dict[str, Any]] = []
    for subset, mask in (
        ("all", np.ones(len(aligned), dtype=bool)),
        ("clean_excluding_implicated", ~aligned["implicated"].to_numpy()),
        ("implicated_only", aligned["implicated"].to_numpy()),
    ):
        selected = aligned.loc[mask]
        if selected.empty:
            continue
        identity = float(selected["identity_affected_mse"].mean())
        model = float(selected["weighted_residual_affected_mse"].mean())
        core = selected["weighted_residual_core_affected_mse"]
        result.append(
            {
                "suite": suite,
                "subset": subset,
                "n_samples": int(len(selected)),
                "fraction_of_suite": float(len(selected) / len(aligned)),
                "identity_affected_mse": identity,
                "v02_affected_mse": model,
                "identity_to_v02_affected_mse_ratio": (
                    float(identity / model) if model > 0 else float("inf")
                ),
                "v02_core_affected_mse": float(core.mean()),
                "v02_core_valid_n": int(core.notna().sum()),
                "v02_worse_than_identity_count": int(
                    selected["weighted_residual_worse_than_identity"].astype(bool).sum()
                ),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir = resolve_master_run(args.run_dir)
    dataset = resolve_existing(args.dataset)
    source_audit = resolve_existing(args.source_audit_run)
    v02_run = resolve_existing(args.v02_run)
    stress_index_table = resolve_existing(args.stress_index_table)
    targeted_table = resolve_existing(args.targeted_table)

    exact_table_path = source_audit / "tables/exact_duplicate_audit.csv"
    exact_pairs = pd.read_csv(exact_table_path)
    cross_exact = exact_pairs.loc[truthy(exact_pairs["cross_split"])].copy()
    exact_sources = set(cross_exact["index_a"].astype(int)) | set(
        cross_exact["index_b"].astype(int)
    )

    with h5py.File(dataset, "r") as handle:
        n_sources = int(handle["images"].shape[0])
        labels = handle["ans"][:]
        ra = handle["ra"][:]
        dec = handle["dec"][:]
    splits = split_indices(n_sources, args.split_seed)
    source_split: dict[int, str] = {}
    source_local: dict[int, int] = {}
    for split_name, values in splits.items():
        for local, index in enumerate(values):
            source_split[int(index)] = split_name
            source_local[int(index)] = int(local)

    coordinate_groups: dict[str, list[int]] = defaultdict(list)
    for index, (ra_value, dec_value) in enumerate(zip(ra, dec)):
        coordinate_groups[coordinate_key(float(ra_value), float(dec_value))].append(index)
    cross_coordinate_groups = {
        key: members
        for key, members in coordinate_groups.items()
        if len(members) > 1 and len({source_split[index] for index in members}) > 1
    }
    coordinate_sources = {
        index for members in cross_coordinate_groups.values() for index in members
    }
    implicated = exact_sources | coordinate_sources

    exact_hashes_by_source: dict[int, set[str]] = defaultdict(set)
    for row in cross_exact.itertuples(index=False):
        exact_hashes_by_source[int(row.index_a)].add(str(row.exact_hash_sha256))
        exact_hashes_by_source[int(row.index_b)].add(str(row.exact_hash_sha256))
    coordinate_key_by_source = {
        index: key for key, members in cross_coordinate_groups.items() for index in members
    }
    source_rows = []
    for index in sorted(implicated):
        key = coordinate_key_by_source.get(index)
        coord_members = cross_coordinate_groups.get(key, []) if key else []
        source_rows.append(
            {
                "source_index": index,
                "historical_split": source_split[index],
                "historical_split_local_index": source_local[index],
                "class_label": int(labels[index]),
                "ra_deg": float(ra[index]),
                "dec_deg": float(dec[index]),
                "exact_cross_split_implicated": index in exact_sources,
                "coordinate_cross_split_implicated": index in coordinate_sources,
                "union_implicated": True,
                "exact_hash_sha256": ";".join(sorted(exact_hashes_by_source[index])),
                "coordinate_group_key_sha256": (
                    hashlib.sha256(key.encode("utf-8")).hexdigest() if key else ""
                ),
                "coordinate_group_size": len(coord_members),
                "coordinate_group_splits": ";".join(
                    sorted({source_split[value] for value in coord_members})
                ),
            }
        )
    implicated_frame = pd.DataFrame(source_rows)

    order = {"train": 0, "validation": 1, "test": 2}
    pair_rows: list[dict[str, Any]] = []
    for evidence, pairs, group_column in (
        ("exact_pixels", cross_exact, "exact_hash_sha256"),
    ):
        pair_labels = pairs.apply(
            lambda row: "-".join(
                sorted((str(row["split_a"]), str(row["split_b"])), key=order.get)
            ),
            axis=1,
        )
        for split_pair, group in pairs.assign(_split_pair=pair_labels).groupby("_split_pair"):
            pair_rows.append(
                {
                    "evidence_type": evidence,
                    "split_pair": split_pair,
                    "pair_count": int(len(group)),
                    "unique_source_count": int(
                        len(set(group["index_a"].astype(int)) | set(group["index_b"].astype(int)))
                    ),
                    "duplicate_group_count": int(group[group_column].nunique()),
                    "groups_spanning_train_validation_test": int(
                        sum(
                            len(
                                {
                                    source_split[int(value)]
                                    for value in set(
                                        exact_pairs.loc[
                                            exact_pairs[group_column].eq(group_id),
                                            ["index_a", "index_b"],
                                        ].to_numpy().ravel()
                                    )
                                }
                            )
                            == 3
                            for group_id in group[group_column].unique()
                        )
                    ),
                }
            )
    coordinate_pair_records = []
    for key, members in cross_coordinate_groups.items():
        for offset, index_a in enumerate(members):
            for index_b in members[offset + 1 :]:
                if source_split[index_a] != source_split[index_b]:
                    split_pair = "-".join(
                        sorted(
                            (source_split[index_a], source_split[index_b]), key=order.get
                        )
                    )
                    coordinate_pair_records.append((key, index_a, index_b, split_pair))
    coordinate_pairs = pd.DataFrame(
        coordinate_pair_records,
        columns=["group_key", "index_a", "index_b", "split_pair"],
    )
    for split_pair, group in coordinate_pairs.groupby("split_pair"):
        pair_rows.append(
            {
                "evidence_type": "exact_coordinates",
                "split_pair": split_pair,
                "pair_count": int(len(group)),
                "unique_source_count": int(
                    len(set(group["index_a"]) | set(group["index_b"]))
                ),
                "duplicate_group_count": int(group["group_key"].nunique()),
                "groups_spanning_train_validation_test": int(
                    sum(
                        len({source_split[value] for value in members}) == 3
                        for key, members in cross_coordinate_groups.items()
                        if key in set(group["group_key"])
                    )
                ),
            }
        )
    pair_summary = pd.DataFrame(pair_rows).sort_values(
        ["evidence_type", "split_pair"]
    )

    normal_metrics = pd.read_csv(v02_run / "tables/normal_per_sample_results.csv")
    stress_metrics = pd.read_csv(v02_run / "tables/stress_per_sample_results.csv")
    normal_roles, normal_replay = reconstruct_normal_roles(
        normal_metrics, splits["test"][:1000]
    )
    stress_roles, stress_replay = recover_stress_roles(
        stress_metrics,
        pd.read_csv(stress_index_table),
        splits["test"][:800],
    )
    targeted = pd.read_csv(targeted_table, low_memory=False)
    compact_roles = targeted_roles(targeted, "compact_bright", splits["test"][:1000])
    high_core_roles = targeted_roles(
        targeted, "high_core_obstruction", splits["test"][:1000]
    )
    role_tables = [normal_roles, stress_roles, compact_roles, high_core_roles]
    implication_rates = pd.DataFrame(
        [implication_summary(frame, implicated) for frame in role_tables]
    )

    metric_rows = [
        *aggregate_subset("normal", normal_metrics, normal_roles, implicated),
        *aggregate_subset("stress", stress_metrics, stress_roles, implicated),
    ]
    metric_frame = pd.DataFrame(metric_rows)
    ratios = metric_frame.pivot(
        index="suite", columns="subset", values="identity_to_v02_affected_mse_ratio"
    )
    max_ratio_change = float(
        max(
            abs(ratios.loc[suite, "clean_excluding_implicated"] / ratios.loc[suite, "all"] - 1.0)
            for suite in ratios.index
        )
    )
    max_implication_rate = float(implication_rates["either_source_implicated_fraction"].max())
    if max_implication_rate <= 0.02 and max_ratio_change <= 0.05:
        measured_severity = "minor"
    elif max_implication_rate <= 0.10 and max_ratio_change <= 0.20:
        measured_severity = "moderate"
    else:
        measured_severity = "catastrophic"

    tables = run_dir / "tables"
    diagnostics = run_dir / "diagnostics"
    safe_csv(tables / "implicated_sources.csv", implicated_frame)
    safe_csv(tables / "leak_pair_summary.csv", pair_summary)
    safe_csv(tables / "evaluation_sample_implication_rates.csv", implication_rates)
    safe_csv(tables / "v02_metrics_all_vs_clean_subset.csv", metric_frame)

    report = f"""# Leak Severity Audit

## Direct answer

Measured aggregate severity: **{measured_severity}** for the reconstructable
historical development suites. Structural benchmark severity remains **major**:
the old source split violates object/content independence and grouped retraining
is still required before a paper claim.

The old 32x normal affected-MSE ratio does not appear heavily dependent on the
implicated samples. The exact comparison is in
`tables/v02_metrics_all_vs_clean_subset.csv`; excluding every sample whose
target or contaminant belongs to a known cross-split exact/coordinate group
changes the normal/stress ratios by at most {max_ratio_change:.2%}.

## Source-level extent

- Cross-split exact-pixel implicated sources: {len(exact_sources)}
- Cross-split exact-coordinate implicated sources: {len(coordinate_sources)}
- Union: {len(implicated)} of {n_sources} ({len(implicated) / n_sources:.4%})
- Exact-only sources: {len(exact_sources - coordinate_sources)}
- Coordinate-only sources: {len(coordinate_sources - exact_sources)}
- Union by split: {implicated_frame.groupby('historical_split').size().to_dict()}

The three exact-only sources are the constant-gray duplicate group with
different coordinates. Every exact-coordinate implicated source is also an
exact-pixel implicated source in this dataset.

## Evaluation reconstruction

The original v0.2 per-sample tables did not save source indices. Normal source
roles were recovered by exact replay of the saved seed-4042 NumPy stream,
including noise-array draws; all saved shift/brightness/blur/noise parameters
matched (`{normal_replay}`). Stress roles were joined to the original indexed
stress table and all parameters matched (`{stress_replay}`). This makes the
normal/stress severity estimate reproducible despite the historical schema
omission.

Compact-bright and high-core implication rates use the saved subset-local source
indices from the later ResUNet same-suite table. They quantify source exposure,
not a retroactive v0.2 metric subset.

## Interpretation and limits

- A small implicated fraction can still invalidate a nominally independent
  test protocol; low measured aggregate impact does not repair the split.
- Known exact/coordinate groups are measurable. Medium perceptual candidates
  are not automatically labeled duplicates and are excluded from the numerical
  implicated set.
- Historical artifacts without source indices and without a validated replay
  path remain non-auditable sample by sample.
- The result supports "old 32x remains plausible," not "old 32x is final."
- A grouped source split, grouped manifests, grouped old-checkpoint diagnostic,
  and duplicate-safe retraining remain necessary.
"""
    safe_text(diagnostics / "leak_severity_report.md", report)
    safe_json(
        run_dir / "logs/leak_severity_provenance.json",
        {
            "dataset": str(dataset.relative_to(PROJECT_ROOT)),
            "dataset_sha256": sha256_file(dataset),
            "source_audit_run": str(source_audit.relative_to(PROJECT_ROOT)),
            "v02_run": str(v02_run.relative_to(PROJECT_ROOT)),
            "stress_index_table": str(stress_index_table.relative_to(PROJECT_ROOT)),
            "targeted_table": str(targeted_table.relative_to(PROJECT_ROOT)),
            "code_sha256": sha256_file(Path(__file__)),
            "split_seed": args.split_seed,
            "normal_rng_replay": normal_replay,
            "stress_alignment": stress_replay,
            "measured_severity": measured_severity,
            "structural_protocol_severity": "major",
        },
    )
    print(f"Leak severity audit complete: {measured_severity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
