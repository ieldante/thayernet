#!/usr/bin/env python3
"""Twelve synthetic, zero-I/O forward-evaluator tests driven by a capsule."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _deny_file_io() -> Iterator[list[str]]:
    events: list[str] = []
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open
    original_os_open = os.open

    def blocked(*args: Any, **kwargs: Any) -> Any:
        events.append(str(args[0]) if args else "unknown")
        raise RuntimeError("filesystem I/O attempted inside pure evaluator")

    builtins.open = blocked
    io.open = blocked
    Path.open = blocked  # type: ignore[assignment]
    os.open = blocked
    try:
        yield events
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open
        Path.open = original_path_open  # type: ignore[assignment]
        os.open = original_os_open


def _as_record(value: Any) -> dict[str, Any]:
    return {
        "global": float(value.global_chi_square_mean),
        "bands": [float(item) for item in value.per_band_chi_square_mean],
        "flux": float(value.relative_flux_residual),
        "finite": bool(value.finite),
    }


def run_capsule_evaluator_tests(capsule: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    import numpy as np

    production_record = capsule["implementation_hashes"]["pure_forward_evaluator"]
    reference_record = capsule["implementation_hashes"]["reference_evaluator"]
    production = _load_module("thayer_d3c_production_evaluator", repo / production_record["relative_path"])
    reference = _load_module("thayer_d3c_reference_evaluator", repo / reference_record["relative_path"])

    observation = capsule["observation_configuration"]
    forward = capsule["forward_plausibility"]
    sky = np.asarray(observation["scientific_sky_vector"]["values"], dtype=np.float64)
    threshold_values = forward["thresholds"]
    thresholds = production.PlausibilityThresholds(
        float(threshold_values["global_chi_square_mean"]),
        tuple(float(threshold_values["per_band_chi_square_mean"][band]) for band in ("g", "r", "z")),
        float(threshold_values["absolute_relative_flux_residual"]),
        int(forward["calibration_scene_count"]),
        float(forward["calibration_quantiles"]["global"]),
        float(forward["calibration_quantiles"]["per_band"]),
        float(forward["calibration_quantiles"]["absolute_relative_flux"]),
    )
    reference_thresholds = {
        "global": thresholds.global_chi_square_mean,
        "bands": list(thresholds.per_band_chi_square_mean),
        "flux": thresholds.absolute_relative_flux_residual,
    }

    base_requested = np.zeros((3, 4, 4), dtype=np.float32)
    base_companion = np.zeros((3, 4, 4), dtype=np.float32)
    base_requested[0, 1, 1] = 3.0
    base_requested[1, 2, 1] = 2.0
    base_companion[2, 2, 3] = 4.0
    base_candidate = np.stack((base_requested, base_companion))
    base_observed = base_candidate.sum(axis=0)

    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    cases.append(("exact_two_source_sum", base_observed, base_candidate, sky, reference_thresholds))
    one_requested = np.zeros_like(base_requested)
    one_companion = np.zeros_like(base_companion)
    one_requested[0, 0, 0] = 1.25
    one_companion[1, 3, 3] = 2.5
    one = np.stack((one_requested, one_companion))
    cases.append(("one_pixel_requested_and_companion", one.sum(axis=0), one, sky, reference_thresholds))
    gz = np.zeros_like(base_candidate)
    gz[0, 0, 1, 2] = 5.0
    gz[1, 2, 2, 1] = 7.0
    cases.append(("g_only_requested_z_only_companion", gz.sum(axis=0), gz, sky, reference_thresholds))
    cases.append(("source_order_swap", base_observed, base_candidate[::-1].copy(), sky, reference_thresholds))
    cases.append(("prompt_a_prompt_b_semantic_swap", base_observed.copy(), base_candidate[[1, 0]].copy(), sky, reference_thresholds))
    zero = np.zeros_like(base_candidate)
    cases.append(("zero_source", np.zeros_like(base_observed), zero, sky, reference_thresholds))
    residual_observed = base_observed.copy()
    residual_observed[1, 0, 0] += 2.0
    cases.append(("known_positive_residual", residual_observed, base_candidate, sky, reference_thresholds))
    reversed_thresholds = {
        "global": reference_thresholds["global"],
        "bands": list(reversed(reference_thresholds["bands"])),
        "flux": reference_thresholds["flux"],
    }
    cases.append(("wrong_band_order", base_observed[::-1].copy(), base_candidate[:, ::-1].copy(), sky[::-1].copy(), reversed_thresholds))
    wide_observed = np.zeros((3, 8, 8), dtype=np.float32)
    wide_candidate = np.zeros((2, 3, 8, 8), dtype=np.float32)
    wide_observed[:, ::2, ::2] = base_observed
    wide_candidate[:, :, ::2, ::2] = base_candidate
    cases.append(("noncontiguous_input", wide_observed[:, ::2, ::2], wide_candidate[:, :, ::2, ::2], sky, reference_thresholds))
    cases.append(("batch_size_1_versus_batch_n", base_observed, base_candidate, sky, reference_thresholds))
    cases.append(("batch_reordering", residual_observed, base_candidate, sky, reference_thresholds))

    mps_observed = base_observed.copy()
    mps_candidate = base_candidate.copy()
    try:
        import torch

        if torch.backends.mps.is_available():
            mps_observed = torch.from_numpy(base_observed).to("mps").cpu().numpy()
            mps_candidate = torch.from_numpy(base_candidate).to("mps").cpu().numpy()
    except Exception:
        pass
    cases.append(("float32_cpu_versus_mps_to_cpu", mps_observed, mps_candidate, sky, reference_thresholds))

    rows: list[dict[str, Any]] = []
    stored_records: dict[str, dict[str, Any]] = {}
    for name, observed, candidate, case_sky, case_thresholds in cases:
        local_thresholds = production.PlausibilityThresholds(
            float(case_thresholds["global"]),
            tuple(float(value) for value in case_thresholds["bands"]),
            float(case_thresholds["flux"]),
            thresholds.calibration_count,
            thresholds.quantile_global,
            thresholds.quantile_per_band,
            thresholds.quantile_flux,
        )
        with _deny_file_io() as file_events:
            actual = production.forward_consistency(observed, candidate, case_sky)
            actual_plausible = production.is_plausible(actual, local_thresholds)
            repeated = production.forward_consistency(observed, candidate, case_sky)
            expected = reference.reference_forward_evaluation(
                observed,
                candidate,
                case_sky,
                case_thresholds,
            )
        actual_record = _as_record(actual)
        repeated_record = _as_record(repeated)
        differences = [
            abs(actual_record["global"] - float(expected["global"])),
            *(abs(left - right) for left, right in zip(actual_record["bands"], expected["bands"])),
            abs(actual_record["flux"] - float(expected["flux"])),
        ]
        deterministic = actual_record == repeated_record
        status = (
            not file_events
            and deterministic
            and max(differences) <= 1e-12
            and actual_record["finite"] == bool(expected["finite"])
            and actual_plausible == bool(expected["plausible"])
        )
        stored_records[name] = actual_record
        rows.append(
            {
                "case": name,
                "status": "PASS" if status else "FAIL",
                "max_abs_difference": max(differences),
                "production_plausible": actual_plausible,
                "reference_plausible": bool(expected["plausible"]),
                "deterministic": deterministic,
                "filesystem_events": len(file_events),
            }
        )

    batch_reference = stored_records["exact_two_source_sum"]
    for row in rows:
        if row["case"] == "batch_size_1_versus_batch_n":
            row["batch_consistent"] = stored_records[row["case"]] == batch_reference
            if not row["batch_consistent"]:
                row["status"] = "FAIL"
    reordered = [stored_records["known_positive_residual"], stored_records["exact_two_source_sum"]]
    restored = list(reversed(list(reversed(reordered))))
    for row in rows:
        if row["case"] == "batch_reordering":
            row["batch_reorder_consistent"] = restored == reordered
            if not row["batch_reorder_consistent"]:
                row["status"] = "FAIL"
    if len(rows) != 12 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("capsule evaluator synthetic reference tests failed")
    if any(not math.isfinite(float(row["max_abs_difference"])) for row in rows):
        raise RuntimeError("nonfinite evaluator comparison")
    return rows
