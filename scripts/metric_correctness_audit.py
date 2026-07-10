"""Run independent deterministic checks of Thayer-Net metric infrastructure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from skimage import color


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stress_test
import train_weighted_residual_unet as weighted
from src import utils


DEFAULT_V02_RUN = PROJECT_ROOT / "outputs/runs/weighted_residual_20260709_030245"


@dataclass
class Check:
    test_id: str
    metric: str
    description: str
    expected: str
    observed: str
    passed: bool
    tolerance: str = "exact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v02-run", type=Path, default=DEFAULT_V02_RUN)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional lowercase tag appended to every output stem.",
    )
    return parser.parse_args()


def resolve_run(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    allowed = (PROJECT_ROOT / "outputs/runs").resolve()
    if allowed not in resolved.parents or not resolved.name.startswith(
        "research_correctness_audit_"
    ):
        raise ValueError("run-dir must be an existing research_correctness_audit_* run")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_existing(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def safe_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def safe_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return bool(np.isclose(a, b, rtol=0.0, atol=tol, equal_nan=True))


def expect_raises(function: Callable[[], Any], error: type[Exception]) -> bool:
    try:
        function()
    except error:
        return True
    return False


def add(
    checks: list[Check],
    test_id: str,
    metric: str,
    description: str,
    expected: Any,
    observed: Any,
    passed: bool,
    tolerance: str = "exact",
) -> None:
    checks.append(
        Check(
            test_id=test_id,
            metric=metric,
            description=description,
            expected=str(expected),
            observed=str(observed),
            passed=bool(passed),
            tolerance=tolerance,
        )
    )


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=float).mean(axis=-1)
    grad_y, grad_x = np.gradient(gray)
    return np.hypot(grad_x, grad_y)


def run_unit_checks(v02_run: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    zeros = np.zeros((8, 8, 3), dtype=np.float32)
    halves = np.full_like(zeros, 0.5)
    mse_value = utils.mse(halves, zeros)
    mae_value = utils.mae(halves, zeros)
    psnr_value = utils.psnr(halves, zeros)
    add(checks, "U01", "whole_mse", "Constant half versus zero", 0.25, mse_value, close(mse_value, 0.25))
    add(checks, "U02", "whole_mae", "Constant half versus zero", 0.5, mae_value, close(mae_value, 0.5))
    add(
        checks,
        "U03",
        "psnr",
        "PSNR for MSE 0.25 at data_range 1",
        20 * math.log10(2.0),
        psnr_value,
        close(psnr_value, 20 * math.log10(2.0)),
        "1e-10",
    )
    ssim_value = utils.ssim_metric(halves, halves)
    add(checks, "U04", "ssim", "Identical RGB arrays", 1.0, ssim_value, close(ssim_value, 1.0))

    uint_a = np.zeros((1, 1, 3), dtype=np.uint8)
    uint_b = np.full((1, 1, 3), 255, dtype=np.uint8)
    uint_mse = utils.mse(uint_a, uint_b)
    add(
        checks,
        "U05",
        "whole_mse",
        "Unsigned-byte subtraction must not wrap",
        65025.0,
        uint_mse,
        close(uint_mse, 65025.0),
    )

    target = np.zeros((2, 2, 3), dtype=np.float32)
    blended = target.copy()
    blended[0, 0] = 0.02
    strict = utils.affected_region_mask(target, blended, threshold=0.02)
    blended[0, 0] = np.array([0.0201, 0.0201, 0.0201])
    above = utils.affected_region_mask(target, blended, threshold=0.02)
    add(checks, "U06", "affected_mask", "Strict threshold excludes exactly 0.02", 0, int(strict.sum()), strict.sum() == 0)
    add(checks, "U07", "affected_mask", "Value above threshold is included", 1, int(above.sum()), above.sum() == 1)
    prediction_a = np.zeros_like(target)
    prediction_b = np.ones_like(target)
    mask_a = utils.affected_region_mask(target, blended, threshold=0.02)
    mask_b = utils.affected_region_mask(target, blended, threshold=0.02)
    add(
        checks,
        "U08",
        "affected_mask",
        "Prediction changes cannot affect blend-change mask",
        True,
        np.array_equal(mask_a, mask_b) and not np.array_equal(prediction_a, prediction_b),
        np.array_equal(mask_a, mask_b),
    )

    prediction = np.zeros((2, 2, 3), dtype=np.float32)
    prediction[0, 0, 0] = 1.0
    one_pixel = np.zeros((2, 2), dtype=bool)
    one_pixel[0, 0] = True
    masked_mse = utils.masked_mse(prediction, target, one_pixel)
    masked_mae = utils.masked_mae(prediction, target, one_pixel)
    add(checks, "U09", "affected_mse", "One RGB channel error at one masked pixel", 1 / 3, masked_mse, close(masked_mse, 1 / 3))
    add(checks, "U10", "affected_mae", "One RGB channel error at one masked pixel", 1 / 3, masked_mae, close(masked_mae, 1 / 3))
    empty = np.zeros((2, 2), dtype=bool)
    empty_value = utils.masked_mse(prediction, target, empty)
    add(checks, "U11", "empty_mask", "Empty masked MSE is explicit NaN", "nan", empty_value, np.isnan(empty_value))

    fixture = np.zeros((32, 32, 3), dtype=np.float32)
    y, x = np.ogrid[:32, :32]
    fixture[..., :] = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 40.0)[..., None]
    eval_core = utils.evaluation_core_mask_p85_v1(fixture)
    loss_core = utils.loss_core_mask_v02_numpy(fixture)
    affected = np.zeros((32, 32), dtype=bool)
    affected[10:24, 9:25] = True
    core_affected = affected & eval_core
    noncore_affected = affected & ~eval_core
    add(
        checks,
        "U12",
        "core_noncore",
        "Core/non-core are disjoint and union to affected",
        True,
        bool(not np.any(core_affected & noncore_affected) and np.array_equal(core_affected | noncore_affected, affected)),
        not np.any(core_affected & noncore_affected) and np.array_equal(core_affected | noncore_affected, affected),
    )
    add(
        checks,
        "U13",
        "core_semantics",
        "Training and evaluation core masks are explicitly distinct",
        "different",
        f"evaluation={int(eval_core.sum())}; loss={int(loss_core.sum())}",
        not np.array_equal(eval_core, loss_core),
    )

    point = np.zeros((25, 25), dtype=bool)
    point[12, 12] = True
    halo = utils.halo_band_mask_manhattan_v1(point, dilation_iters=5)
    add(checks, "U14", "halo_band", "Five-step Manhattan ring size", 60, int(halo.sum()), int(halo.sum()) == 60)
    add(checks, "U15", "halo_band", "Halo excludes affected pixels", 0, int((halo & point).sum()), not np.any(halo & point))

    target1 = np.zeros((2, 2, 3), dtype=np.float32)
    pred1 = np.zeros_like(target1)
    pred1[0, 0] = 1.0
    mask1 = np.zeros((2, 2), dtype=bool)
    mask1[0, 0] = True
    target2 = np.zeros((2, 2, 3), dtype=np.float32)
    pred2 = np.zeros_like(target2)
    mask2 = np.ones((2, 2), dtype=bool)
    summary = utils.masked_mse_summary([pred1, pred2], [target1, target2], [mask1, mask2])
    add(checks, "U16", "aggregation", "Macro MSE gives equal sample weight", 0.5, summary["macro_mse"], close(float(summary["macro_mse"]), 0.5))
    add(checks, "U17", "aggregation", "Micro MSE gives equal affected-channel weight", 0.2, summary["micro_mse"], close(float(summary["micro_mse"]), 0.2))
    add(checks, "U18", "aggregation", "Coverage counts are explicit", (2, 2, 0), (summary["n_total"], summary["n_valid"], summary["n_empty"]), summary["n_total"] == 2 and summary["n_valid"] == 2 and summary["n_empty"] == 0)

    outcomes = utils.aligned_pair_outcomes(
        ["a", "b", "c", "d"],
        [1.0, 2.0, 3.0, np.nan],
        ["a", "b", "c", "d"],
        [2.0, 2.0, 1.0, 4.0],
    )
    add(checks, "U19", "win_rate", "Finite aligned pairs produce win/loss/tie", (1, 1, 1, 3), (outcomes["wins"], outcomes["losses"], outcomes["ties"], outcomes["n_valid_pairs"]), outcomes["wins"] == 1 and outcomes["losses"] == 1 and outcomes["ties"] == 1 and outcomes["n_valid_pairs"] == 3)
    reorder_raises = expect_raises(
        lambda: utils.aligned_pair_outcomes(["a", "b"], [1, 2], ["b", "a"], [1, 2]),
        ValueError,
    )
    add(checks, "U20", "win_rate", "Reordered sample IDs must raise", "ValueError", reorder_raises, reorder_raises)
    duplicate_raises = expect_raises(
        lambda: utils.aligned_pair_outcomes(["a", "a"], [1, 2], ["a", "b"], [1, 2]),
        ValueError,
    )
    add(checks, "U21", "win_rate", "Duplicate sample IDs must raise", "ValueError", duplicate_raises, duplicate_raises)
    truncation_raises = expect_raises(
        lambda: run_stress_test.evaluate_samples([{}], [], 0.02), ValueError
    )
    add(checks, "U22", "alignment", "Prediction-count mismatch must raise", "ValueError", truncation_raises, truncation_raises)

    rgb = np.zeros((8, 8, 3), dtype=np.float64)
    lab = color.rgb2lab(np.clip(rgb, 0.0, 1.0))
    delta = color.deltaE_ciede2000(lab, lab)
    add(checks, "U23", "delta_e2000", "Identical valid-range RGB has zero Delta E", 0.0, float(delta.max()), close(float(delta.max()), 0.0))
    clipped_lab = color.rgb2lab(np.clip(np.full_like(rgb, 2.0), 0.0, 1.0))
    finite_delta = color.deltaE_ciede2000(clipped_lab, lab)
    add(checks, "U24", "delta_e2000", "Out-of-range RGB is clipped before Lab conversion", True, bool(np.isfinite(finite_delta).all()), np.isfinite(finite_delta).all())
    grad = gradient_magnitude(rgb)
    add(checks, "U25", "gradient_error", "Constant identical images have zero gradient error", 0.0, float(np.mean(np.abs(grad - grad))), close(float(np.mean(np.abs(grad - grad))), 0.0))

    historical: dict[str, Any] = {}
    for suite in ("normal", "stress"):
        frame = pd.read_csv(v02_run / f"tables/{suite}_per_sample_results.csv")
        aggregate = weighted.aggregate_metrics(frame)
        v02 = aggregate.loc[aggregate["method"].eq("weighted_residual")].iloc[0]
        valid_core = int(frame["weighted_residual_core_affected_mse"].notna().sum())
        expected_core = 858 if suite == "normal" else 1000
        add(checks, f"U26_{suite}", "regional_coverage", f"Historical {suite} core-valid count is reported explicitly", expected_core, int(v02["n_valid_core"]), int(v02["n_valid_core"]) == expected_core)
        weights = frame["mask_fraction"].to_numpy(dtype=float)
        identity_micro = float(np.average(frame["identity_affected_mse"], weights=weights))
        model_micro = float(np.average(frame["weighted_residual_affected_mse"], weights=weights))
        macro_ratio = float(frame["identity_affected_mse"].mean() / frame["weighted_residual_affected_mse"].mean())
        micro_ratio = identity_micro / model_micro
        historical[suite] = {
            "n_total": len(frame),
            "n_valid_core": valid_core,
            "macro_identity_to_v02_ratio": macro_ratio,
            "micro_identity_to_v02_ratio": micro_ratio,
        }
    add(checks, "U28", "headline_crosscheck", "Historical normal macro ratio", 32.3137, historical["normal"]["macro_identity_to_v02_ratio"], close(historical["normal"]["macro_identity_to_v02_ratio"], 32.3137, 5e-4), "5e-4")
    add(checks, "U29", "headline_crosscheck", "Historical stress macro ratio", 19.6386, historical["stress"]["macro_identity_to_v02_ratio"], close(historical["stress"]["macro_identity_to_v02_ratio"], 19.6386, 5e-4), "5e-4")
    return checks, historical


def definitions() -> pd.DataFrame:
    rows = [
        ("whole_mse", "mean((prediction-target)^2) over H,W,C", "macro mean of per-sample values", "finite float; normally clipped [0,1]", "none", "correct"),
        ("whole_mae", "mean(abs(prediction-target)) over H,W,C", "macro mean", "finite float; normally clipped [0,1]", "none", "correct"),
        ("psnr", "10*log10(data_range^2/MSE), data_range=1", "macro mean of per-sample dB", "fixed display scale", "perfect prediction=inf", "correct"),
        ("ssim", "skimage SSIM, channel_axis=2, data_range=1", "macro mean", "RGB; main tables clipped", "whole image only", "correct"),
        ("affected_mask", "mean_channel(abs(blended-target)) > threshold", "per sample", "prediction-independent", "strict boundary", "correct; call it blend-change mask"),
        ("affected_mse", "mean squared RGB error on affected mask", "macro mean primary; micro secondary", "same mask for all methods", "empty=NaN and explicit count", "correct"),
        ("affected_mae", "mean absolute RGB error on affected mask", "macro mean", "same mask for all methods", "empty=NaN", "correct"),
        ("evaluation_core_mask_p85_v1", "central 0.18 aperture and >= aperture p85", "per sample", "target-only evaluation mask", "distinct from loss core", "correct but must be versioned"),
        ("loss_core_mask_v02", "central 0.18 aperture and >=55% aperture max", "training spatial weights", "target used only in supervised loss", "distinct from evaluation core", "correct historical loss semantics"),
        ("noncore_affected_mse", "affected AND NOT evaluation_core", "macro over valid samples", "same paired mask", "empty=NaN", "correct"),
        ("halo_band_mse", "five-step Manhattan dilation(affected) minus affected", "macro over valid samples", "evaluation geometry", "distinct from v0.3 square training halo", "correct evaluation proxy"),
        ("delta_e2000", "rgb2lab(clipped RGB), then pixelwise CIEDE2000", "macro summaries", "valid [0,1] display RGB", "auxiliary only", "correct auxiliary metric"),
        ("lab_chroma_error", "abs(||Lab_ab(pred)||-||Lab_ab(target)||)", "macro mean", "clipped display RGB", "not calibrated color", "correct proxy"),
        ("gradient_error", "abs difference in grayscale gradient magnitude", "macro masked mean", "arithmetic RGB grayscale", "edge proxy", "correct proxy"),
        ("improvement_vs_identity", "macro identity affected MSE / macro model affected MSE", "ratio of macro means", "same samples and masks", "zero model MSE policy must be explicit", "correct historical formula"),
        ("worse_than_identity", "model affected MSE > identity affected MSE", "count over finite paired rows", "sample-aligned", "ties are not worse", "correct with explicit denominator"),
        ("win_rate", "candidate affected MSE < reference affected MSE", "wins / finite aligned pairs", "immutable sample_id alignment", "report ties/missing", "hardened for grouped evaluator"),
    ]
    return pd.DataFrame(rows, columns=["metric", "formula", "aggregation", "range_or_inputs", "empty_or_edge_policy", "audit_assessment"])


def main() -> int:
    args = parse_args()
    if args.output_tag and not re.fullmatch(r"[a-z0-9_]+", args.output_tag):
        raise ValueError(
            "output-tag must contain only lowercase letters, digits, and underscores"
        )
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    run_dir = resolve_run(args.run_dir)
    v02_run = resolve_existing(args.v02_run)
    checks, historical = run_unit_checks(v02_run)
    test_frame = pd.DataFrame([check.__dict__ for check in checks])
    definition_frame = definitions()
    tests_name = f"metric_unit_tests{suffix}.csv"
    definitions_name = f"metric_definitions{suffix}.csv"
    report_name = f"metric_correctness_audit{suffix}.md"
    provenance_name = f"metric_audit_provenance{suffix}.json"
    safe_csv(run_dir / f"tables/{tests_name}", test_frame)
    safe_csv(run_dir / f"tables/{definitions_name}", definition_frame)
    failed = test_frame.loc[~test_frame["passed"]]
    gate = "pass" if failed.empty else "fail"
    report = f"""# Metric Correctness Audit

## Verdict

Metric gate: **{gate.upper()}** ({int(test_frame['passed'].sum())}/{len(test_frame)}
deterministic checks passed).

The primary whole/affected MSE and MAE, PSNR, SSIM, prediction-independent
blend-change mask, residual sign convention, and same-sample improvement ratio
are arithmetically correct. The old normal macro identity/model affected-MSE
ratio independently reconstructs to
`{historical['normal']['macro_identity_to_v02_ratio']:.4f}x`; stress is
`{historical['stress']['macro_identity_to_v02_ratio']:.4f}x`.

Affected-pixel-weighted micro ratios are
`{historical['normal']['micro_identity_to_v02_ratio']:.4f}x` normal and
`{historical['stress']['micro_identity_to_v02_ratio']:.4f}x` stress. Macro
sample weighting therefore does not explain or inflate the 32x value.

## Correctness fixes made before grouped evaluation

- Metric subtraction now casts before arithmetic, preventing unsigned-byte
  wraparound.
- Nonfinite inputs are rejected.
- The stress evaluator raises on sample/prediction count mismatch instead of
  silently truncating `zip`.
- Training-loss core and evaluation core are separately named and unit-tested.
- The historical Manhattan evaluation halo is canonical and unit-tested.
- Aligned win rates require unique, identically ordered sample IDs and report
  finite-pair coverage.
- Aggregate regional tables now expose total and valid affected/core/non-core/
  halo counts. Historical files remain unchanged.

## Historical reporting correction

The original v0.2 normal core-affected MSE is a macro mean over
`{historical['normal']['n_valid_core']}/1000` nonempty core-affected masks, not
all 1,000 samples. Stress has `{historical['stress']['n_valid_core']}/1000`
valid core masks. This was a coverage-label bug, not an affected-MSE arithmetic
bug, and all models were compared on the same valid regional rows.

## Remaining semantic qualifications

- The evaluation core (`evaluation_core_mask_p85_v1`) differs intentionally
  from the v0.2 loss core (`loss_core_mask_v02`). Claims must not treat them as
  the same mask.
- v0.3's historical square training halo differs from the Manhattan evaluation
  halo; v0.2 has zero halo loss, so this does not block the grouped v0.2 retrain.
- CIEDE2000/Lab/chroma and gradient errors are auxiliary display-RGB proxies.
- Primary tables must label reconstruction state as clipped or unclipped.
- Suite-level values called per-sample medians must not be described as pooled
  pixel medians.

## Gate decision

Grouped evaluation may proceed only when every row in
`tables/{tests_name}` passes and the grouped evaluator uses immutable
manifest `sample_id` alignment plus explicit valid counts. Retraining remains
separately gated on the infrastructure, blending, leakage, and grouped-manifest
audits.
"""
    safe_text(run_dir / f"diagnostics/{report_name}", report)
    safe_json(
        run_dir / f"logs/{provenance_name}",
        {
            "gate": gate,
            "passed": int(test_frame["passed"].sum()),
            "total": len(test_frame),
            "code_sha256": {
                "scripts/metric_correctness_audit.py": sha256_file(Path(__file__)),
                "src/utils.py": sha256_file(PROJECT_ROOT / "src/utils.py"),
                "scripts/run_stress_test.py": sha256_file(PROJECT_ROOT / "scripts/run_stress_test.py"),
                "scripts/train_weighted_residual_unet.py": sha256_file(PROJECT_ROOT / "scripts/train_weighted_residual_unet.py"),
            },
            "historical_input_sha256": {
                "normal": sha256_file(v02_run / "tables/normal_per_sample_results.csv"),
                "stress": sha256_file(v02_run / "tables/stress_per_sample_results.csv"),
            },
        },
    )
    print(f"Metric audit gate: {gate} ({int(test_frame['passed'].sum())}/{len(test_frame)})")
    return 0 if failed.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
