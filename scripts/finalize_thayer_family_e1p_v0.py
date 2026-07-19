#!/usr/bin/env python3
"""Finalize the append-only Family-E1P paired-prompt scientific report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_thayer_family_e1p_v0 import ORIGINAL_RUN, REPO, relative, sha256_file, validate_run  # noqa: E402


EXTERNAL_CHECKPOINT_BASELINE = (
    REPO
    / "outputs/runs/thayer_select_prompt_ablation_20260711_164329/tables/historical_checkpoint_hashes_after.csv"
)


def fresh_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def fresh_json(path: Path, value: object) -> None:
    fresh_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fresh_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False, env=env)


def plot_diagnosis(run: Path, results: pd.DataFrame, layers: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = ["difficult", "mixed eight"]
    x = np.arange(2)
    width = 0.33
    axes[0].bar(x - width / 2, results.prompt_identity, width, label="identity", color="#315f8c")
    axes[0].bar(x + width / 2, results.prompt_swap, width, label="pair swap", color="#a65141")
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1, label="identity gate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate")
    axes[0].set_title("Required paired-prompt gates")
    axes[0].legend(fontsize=8)

    chosen = ["enc0_first", "enc0_second", "enc1", "enc2", "enc3", "dec2_second", "dec1_second", "dec0_second", "requested_output"]
    colors = {"difficult_one_scene": "#a65141", "mixed_eight_scene": "#49784a"}
    for condition, color in colors.items():
        frame = layers[(layers.condition == condition) & (layers.phase == "final")].set_index("layer").loc[chosen]
        axes[1].plot(np.arange(len(chosen)), frame.feature_modulation, marker="o", color=color, label=condition.replace("_", " "))
    axes[1].set_xticks(np.arange(len(chosen)), chosen, rotation=55, ha="right", fontsize=7)
    axes[1].set_ylabel("paired feature modulation")
    axes[1].set_yscale("log")
    axes[1].set_title("Prompt signal survives all levels")
    axes[1].legend(fontsize=7)

    difficult = decomposition[decomposition.condition == "difficult_one_scene"]
    mixed = decomposition[decomposition.condition == "mixed_eight_scene"]
    positions = np.arange(len(mixed))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].scatter([-1], difficult.contrast_cosine, color="#a65141", marker="D", label="difficult")
    axes[2].scatter(positions, mixed.contrast_cosine, color="#49784a", label="mixed scenes")
    axes[2].set_xticks([-1, *positions], ["difficult", *mixed.family_e1_index.astype(str)])
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].set_ylabel("cosine(predicted, true prompt contrast)")
    axes[2].set_title("Only mixed index 81 learns identity direction")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    path = run / "figures/prompt_conditioning_diagnosis.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def checkpoint_audit(run: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    before = pd.read_csv(run / "tables/historical_checkpoint_inventory_before.csv")
    run_rows: list[dict[str, object]] = []
    for row in before.itertuples(index=False):
        path = REPO / row.relative_path
        observed = sha256_file(path) if path.is_file() else "MISSING"
        run_rows.append(
            {
                "relative_path": row.relative_path,
                "expected_sha256": row.sha256,
                "observed_sha256": observed,
                "unchanged": observed == row.sha256,
            }
        )
    external = pd.read_csv(EXTERNAL_CHECKPOINT_BASELINE)
    external_rows: list[dict[str, object]] = []
    for row in external.itertuples(index=False):
        path = REPO / row.relative_path
        observed = sha256_file(path) if path.is_file() else "MISSING"
        external_rows.append(
            {
                "relative_path": row.relative_path,
                "expected_sha256": row.expected_sha256,
                "observed_sha256": observed,
                "unchanged": observed == row.expected_sha256,
            }
        )
    return run_rows, external_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    provenance = validate_run(run)
    status = json.loads((run / "logs/micro_overfit_complete.json").read_text())
    if status["status"] != "FAIL" or status["full_training_authorized"] is not False:
        raise RuntimeError("unexpected Family-E1P micro disposition")
    started = time.time()
    results = pd.read_csv(run / "tables/micro_overfit_results.csv").set_index("condition")
    layers = pd.read_csv(run / "tables/layerwise_prompt_trace.csv")
    scenes = pd.read_csv(run / "tables/per_scene_pair_metrics.csv")
    views = pd.read_csv(run / "tables/per_view_identity_diagnostics.csv")
    decomposition = pd.read_csv(run / "tables/common_contrast_diagnostics.csv")
    difficult = results.loc["difficult_one_scene"]
    mixed = results.loc["mixed_eight_scene"]

    comparison_fields = [
        "initial_total",
        "final_total",
        "objective_reduction",
        "initial_requested_l1",
        "final_requested_l1",
        "requested_l1_reduction",
        "initial_companion_l1",
        "final_companion_l1",
        "companion_l1_reduction",
        "prompt_identity_rate",
        "maximum_conservation_error",
        "maximum_conservation_tolerance",
        "head_update_norm",
        "nonfinal_decoder_update_norm",
    ]
    prior = pd.read_csv(ORIGINAL_RUN / "tables/micro_overfit_results.csv").set_index("condition")
    replication_rows = []
    for condition in ("difficult_one_scene", "mixed_eight_scene"):
        for prior_field in comparison_fields:
            current_field = "prompt_identity" if prior_field == "prompt_identity_rate" else prior_field
            old_value = float(prior.loc[condition, prior_field])
            new_value = float(results.loc[condition, current_field])
            replication_rows.append(
                {
                    "condition": condition,
                    "metric": current_field,
                    "prior_value": old_value,
                    "paired_intervention_value": new_value,
                    "exact_numeric_match": old_value == new_value,
                    "absolute_difference": abs(old_value - new_value),
                }
            )
    fresh_csv(run / "tables/prior_family_e1_replication.csv", replication_rows)
    exact_replication = all(bool(row["exact_numeric_match"]) for row in replication_rows)

    mixed_decomposition = decomposition[decomposition.condition == "mixed_eight_scene"]
    difficult_decomposition = decomposition[decomposition.condition == "difficult_one_scene"].iloc[0]
    pair_counts = views.groupby(["condition", "family_e1_index"]).identity_correct.sum()
    difficult_exactly_one = int((pair_counts.loc["difficult_one_scene"] == 1).sum())
    mixed_exactly_one = int((pair_counts.loc["mixed_eight_scene"] == 1).sum())
    mixed_both = int((pair_counts.loc["mixed_eight_scene"] == 2).sum())
    taxonomy_rows = [
        {"cause": "Prompt ignored", "primary": False, "disposition": "REJECTED", "evidence": f"mixed requested-output modulation={mixed['cross_prompt_l1_response_ratio']:.6f} of truth by aggregate L1; all prompt gradients nonzero"},
        {"cause": "Prompt diluted", "primary": False, "disposition": "PARTIAL_DIFFICULT_ONLY", "evidence": "difficult enc0_second modulation 0.449193 falls to enc3 0.043556, but mixed enc3 remains 0.783484"},
        {"cause": "Prompt forgotten", "primary": False, "disposition": "REJECTED", "evidence": "no layer is numerically indistinguishable in either condition"},
        {"cause": "Prompt overwritten", "primary": False, "disposition": "REJECTED", "evidence": "decoder and requested output retain prompt-dependent activation and nonzero input gradients"},
        {"cause": "Prompt too weak", "primary": True, "disposition": "PRIMARY", "evidence": f"identity-aligned mixed contrast median gain={mixed_decomposition.contrast_gain.median():.6g}, cosine={mixed_decomposition.contrast_cosine.median():.6g}; leakage={mixed.companion_leakage:.6f}"},
        {"cause": "Skip connections dominate", "primary": False, "disposition": "REJECTED", "evidence": "mixed bottleneck modulation is already strong; difficult skips restore rather than erase prompt modulation"},
        {"cause": "Decoder ignores prompt", "primary": False, "disposition": "REJECTED", "evidence": f"requested-output feature modulation is {difficult['cross_prompt_l1_response_ratio']:.6f} / {mixed['cross_prompt_l1_response_ratio']:.6f} of true L1 response"},
        {"cause": "Other", "primary": False, "disposition": "NOT_NEEDED", "evidence": "the frozen taxonomy is resolved by Prompt too weak when restricted to the identity-aligned component"},
    ]
    fresh_csv(run / "tables/failure_taxonomy.csv", taxonomy_rows)
    fresh_json(
        run / "diagnostics/failure_classification.json",
        {
            "classification": "Prompt too weak",
            "qualification": "identity-aligned prompt component too weak despite strong generic prompt modulation",
            "prompt_ignored": False,
            "first_numerically_indistinguishable_layer": {
                "difficult_one_scene": "NONE",
                "mixed_eight_scene": "NONE",
            },
            "difficult_contrast_gain": float(difficult_decomposition.contrast_gain),
            "difficult_contrast_cosine": float(difficult_decomposition.contrast_cosine),
            "mixed_median_contrast_gain": float(mixed_decomposition.contrast_gain.median()),
            "mixed_median_contrast_cosine": float(mixed_decomposition.contrast_cosine.median()),
            "mixed_exactly_one_correct_pairs": mixed_exactly_one,
            "mixed_both_correct_pairs": mixed_both,
        },
    )
    plot_diagnosis(run, results.reset_index(), layers, decomposition)

    historical_rows, external_rows = checkpoint_audit(run)
    fresh_csv(run / "tables/historical_checkpoint_inventory_after.csv", historical_rows)
    fresh_csv(run / "tables/external_historical_checkpoint_inventory_after.csv", external_rows)
    historical_mismatches = sum(not bool(row["unchanged"]) for row in historical_rows + external_rows)
    new_states = sorted((run / "micro_overfit").glob("*_final_state.pth"))
    fresh_csv(
        run / "tables/fresh_micro_state_inventory.csv",
        [
            {"relative_path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "micro_only": True}
            for path in new_states
        ],
    )

    compile_result = command(
        [
            str(REPO / ".venv-btk/bin/python"),
            "-m",
            "compileall",
            "-q",
            "src/family_e1.py",
            "src/family_e1p.py",
            "scripts/bootstrap_thayer_family_e1p_v0.py",
            "scripts/run_thayer_family_e1p_v0.py",
            "scripts/analyze_thayer_family_e1p_v0.py",
            "scripts/finalize_thayer_family_e1p_v0.py",
            "tests/test_family_e1p.py",
            "tests/test_thayer_family_e1p_v0_artifacts.py",
        ]
    )
    fresh_text(run / "logs/compileall.txt", compile_result.stdout + compile_result.stderr)
    test_env = dict(os.environ)
    test_env["THAYER_FAMILY_E1P_RUN"] = relative(run)
    tests_result = command(
        [
            str(REPO / ".venv-btk/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_family_e1p.py",
            "tests/test_family_e1.py",
            "tests/test_family_e_signed_residual.py",
            "tests/test_thayer_family_e1p_v0_artifacts.py",
        ],
        env=test_env,
    )
    fresh_text(run / "logs/focused_tests.txt", tests_result.stdout + tests_result.stderr)
    readme_expected = next(
        item["sha256"] for item in provenance["authoritative_inputs"] if item["name"] == "readme"
    )
    readme_current = sha256_file(REPO / "README.md")
    staged_result = command(["git", "diff", "--cached", "--name-status"])
    diff_result = command(["git", "diff", "--check"])
    input_mismatches = 0
    for item in provenance["authoritative_inputs"]:
        path = REPO / item["path"]
        input_mismatches += int(not path.is_file() or sha256_file(path) != item["sha256"])
    csv_failures = []
    for path in run.rglob("*.csv"):
        try:
            pd.read_csv(path)
        except Exception as error:
            csv_failures.append(f"{relative(path)}: {error}")
    integrity_rows = [
        {"check": "compileall", "status": "PASS" if compile_result.returncode == 0 else "FAIL", "evidence": compile_result.stderr or "selected sources compiled"},
        {"check": "focused_tests", "status": "PASS" if tests_result.returncode == 0 else "FAIL", "evidence": tests_result.stdout.strip().splitlines()[-1] if tests_result.stdout.strip() else tests_result.stderr},
        {"check": "authoritative_inputs_unchanged", "status": "PASS" if input_mismatches == 0 else "FAIL", "evidence": f"{len(provenance['authoritative_inputs'])} checked; {input_mismatches} mismatches"},
        {"check": "historical_checkpoints_unchanged", "status": "PASS" if historical_mismatches == 0 else "FAIL", "evidence": f"{len(historical_rows) + len(external_rows)} checked; {historical_mismatches} mismatches"},
        {"check": "fresh_micro_states_only", "status": "PASS" if len(new_states) == 2 else "FAIL", "evidence": f"{len(new_states)} fresh micro-only states"},
        {"check": "readme_unchanged", "status": "PASS" if readme_current == readme_expected else "FAIL", "evidence": readme_current},
        {"check": "staged_index_empty", "status": "PASS" if staged_result.returncode == 0 and not staged_result.stdout.strip() else "FAIL", "evidence": staged_result.stdout.strip() or "empty"},
        {"check": "git_diff_check", "status": "PASS" if diff_result.returncode == 0 else "FAIL", "evidence": diff_result.stdout + diff_result.stderr or "clean"},
        {"check": "csv_schema", "status": "PASS" if not csv_failures else "FAIL", "evidence": f"{len(list(run.rglob('*.csv')))} CSV files; {len(csv_failures)} failures"},
        {"check": "no_downstream_science", "status": "PASS" if all(status[key] == 0 for key in ("validation_access_count", "calibration_access_count", "oof_outputs", "safety_labels", "auditor_models")) else "FAIL", "evidence": "validation/calibration/OOF/labels/auditor all zero"},
        {"check": "exact_prior_paired_replication", "status": "PASS" if exact_replication else "FAIL", "evidence": f"{len(replication_rows)} scientific values compared"},
    ]
    fresh_csv(run / "tables/integrity_checks.csv", integrity_rows)
    if not all(row["status"] == "PASS" for row in integrity_rows):
        raise RuntimeError("Family-E1P integrity finalization failed")

    difficult_layer = layers[(layers.condition == "difficult_one_scene") & (layers.phase == "final")].set_index("layer")
    mixed_layer = layers[(layers.condition == "mixed_eight_scene") & (layers.phase == "final")].set_index("layer")
    difficult_drop = 1.0 - difficult_layer.loc["enc3", "feature_modulation"] / difficult_layer.loc["enc0_second", "feature_modulation"]
    difficult_gradient_attenuation = difficult_layer.loc["enc0_second", "gradient_wrt_prompt_input_rms"] / difficult_layer.loc["enc3", "gradient_wrt_prompt_input_rms"]
    mixed_scenes = scenes[scenes.condition == "mixed_eight_scene"]
    tests_summary = tests_result.stdout.strip().splitlines()[0] if tests_result.stdout.strip() else "tests passed"
    report = f"""# Thayer-Family-E1P-v0 scientific report

## Outcome

**FAIL — prompt identity remains below the frozen 0.90 gate. Family-E1 full training is not authorized.**

The difficult scene reached prompt identity `{difficult.prompt_identity:.4f}` and prompt-swap pair success `{difficult.prompt_swap:.4f}`. The mixed-eight set reached `{mixed.prompt_identity:.4f}` and `{mixed.prompt_swap:.4f}`. These exactly reproduce the prior Family-E1 paired micro results: all `{len(replication_rows)}` compared scientific values match numerically with zero difference.

The required single failure classification is **Prompt too weak**, qualified precisely as: **the source-identity-aligned prompt component is too weak even though generic prompt modulation survives throughout the network**. This is not a reconstruction-capacity failure and not literal prompt disappearance.

## Critical protocol finding

The proposed intervention was already present in Family-E1. The prior runner duplicated each blend into A/B prompt views and swapped requested/companion targets, and its frozen preregistration explicitly specified that pairing. This campaign therefore changed no training tensor semantics; it is an instrumented deterministic replication. Paired examples alone do not repair Family-E1 identity.

## Required measurements

| Condition | Prompt identity | Prompt swap | Requested-source error | Companion leakage | Gate |
|---|---:|---:|---:|---:|---|
| difficult | {difficult.prompt_identity:.4f} | {difficult.prompt_swap:.4f} | {difficult.requested_source_error:.6f} | {difficult.companion_leakage:.6f} | FAIL |
| mixed eight | {mixed.prompt_identity:.4f} | {mixed.prompt_swap:.4f} | {mixed.requested_source_error:.6f} | {mixed.companion_leakage:.6f} | FAIL |

Prompt swap is the stricter scene-level event that both prompt views select their requested source. The difficult pair had exactly one correct view. In mixed eight, `{mixed_exactly_one}/8` scenes had exactly one correct view and only `{mixed_both}/8` had both correct; the sole full success was Family-E1 index `81`.

For each identical observation, prediction A/B differences were:

| Condition | L1 diff / truth / ratio | Flux diff / truth / ratio | Centroid px / truth / ratio | Color diff / truth / ratio | Prediction cosine / truth cosine |
|---|---|---|---|---|---|
| difficult | {difficult.cross_prompt_l1_difference:.6f} / {difficult.truth_cross_prompt_l1_difference:.6f} / {difficult.cross_prompt_l1_response_ratio:.3f} | {difficult.cross_prompt_flux_difference:.3f} / {difficult.truth_cross_prompt_flux_difference:.3f} / {difficult.cross_prompt_flux_response_ratio:.3f} | {difficult.cross_prompt_centroid_difference_pixels:.3f} / {difficult.truth_cross_prompt_centroid_difference_pixels:.3f} / {difficult.cross_prompt_centroid_response_ratio:.3f} | {difficult.cross_prompt_color_difference:.3f} / {difficult.truth_cross_prompt_color_difference:.3f} / {difficult.cross_prompt_color_response_ratio:.3f} | {difficult.cross_prompt_cosine_similarity:.3f} / {difficult.truth_cross_prompt_cosine_similarity:.3f} |
| mixed eight | {mixed.cross_prompt_l1_difference:.6f} / {mixed.truth_cross_prompt_l1_difference:.6f} / {mixed.cross_prompt_l1_response_ratio:.3f} | {mixed.cross_prompt_flux_difference:.3f} / {mixed.truth_cross_prompt_flux_difference:.3f} / {mixed.cross_prompt_flux_response_ratio:.3f} | {mixed.cross_prompt_centroid_difference_pixels:.3f} / {mixed.truth_cross_prompt_centroid_difference_pixels:.3f} / {mixed.cross_prompt_centroid_response_ratio:.3f} | {mixed.cross_prompt_color_difference:.3f} / {mixed.truth_cross_prompt_color_difference:.3f} / {mixed.cross_prompt_color_response_ratio:.3f} | {mixed.cross_prompt_cosine_similarity:.3f} / {mixed.truth_cross_prompt_cosine_similarity:.3f} |

The mixed aggregate L1 ratio `{mixed.cross_prompt_l1_response_ratio:.3f}` is not evidence of correct identity: its per-scene median is `{mixed_scenes.cross_prompt_l1_response_ratio.median():.3f}`, while the median cosine between the predicted A-B contrast and the true source A-B contrast is only `{mixed_decomposition.contrast_cosine.median():.6f}` and the median signed contrast gain is `{mixed_decomposition.contrast_gain.median():.6f}`. Index 81 alone has contrast cosine `{mixed_decomposition.loc[mixed_decomposition.family_e1_index == 81, 'contrast_cosine'].iloc[0]:.6f}` and gain `{mixed_decomposition.loc[mixed_decomposition.family_e1_index == 81, 'contrast_gain'].iloc[0]:.6f}`. Thus most prompt-driven output changes are nearly orthogonal to requested-source identity.

## Prompt influence trace

No encoder, decoder, head, or source output met the frozen numerical-indistinguishability diagnostic in either condition. Every per-layer gradient with respect to the prompt input was nonzero.

| Condition/layer | Feature modulation | Cross-correlation | MI proxy (nats) | Prompt-gradient RMS |
|---|---:|---:|---:|---:|
| difficult enc0_second | {difficult_layer.loc['enc0_second', 'feature_modulation']:.6f} | {difficult_layer.loc['enc0_second', 'cross_correlation']:.6f} | {difficult_layer.loc['enc0_second', 'mutual_information_proxy_nats']:.6f} | {difficult_layer.loc['enc0_second', 'gradient_wrt_prompt_input_rms']:.3e} |
| difficult enc3 | {difficult_layer.loc['enc3', 'feature_modulation']:.6f} | {difficult_layer.loc['enc3', 'cross_correlation']:.6f} | {difficult_layer.loc['enc3', 'mutual_information_proxy_nats']:.6f} | {difficult_layer.loc['enc3', 'gradient_wrt_prompt_input_rms']:.3e} |
| difficult requested output | {difficult_layer.loc['requested_output', 'feature_modulation']:.6f} | {difficult_layer.loc['requested_output', 'cross_correlation']:.6f} | {difficult_layer.loc['requested_output', 'mutual_information_proxy_nats']:.6f} | {difficult_layer.loc['requested_output', 'gradient_wrt_prompt_input_rms']:.3e} |
| mixed enc0_second | {mixed_layer.loc['enc0_second', 'feature_modulation']:.6f} | {mixed_layer.loc['enc0_second', 'cross_correlation']:.6f} | {mixed_layer.loc['enc0_second', 'mutual_information_proxy_nats']:.6f} | {mixed_layer.loc['enc0_second', 'gradient_wrt_prompt_input_rms']:.3e} |
| mixed enc3 | {mixed_layer.loc['enc3', 'feature_modulation']:.6f} | {mixed_layer.loc['enc3', 'cross_correlation']:.6f} | {mixed_layer.loc['enc3', 'mutual_information_proxy_nats']:.6f} | {mixed_layer.loc['enc3', 'gradient_wrt_prompt_input_rms']:.3e} |
| mixed requested output | {mixed_layer.loc['requested_output', 'feature_modulation']:.6f} | {mixed_layer.loc['requested_output', 'cross_correlation']:.6f} | {mixed_layer.loc['requested_output', 'mutual_information_proxy_nats']:.6f} | {mixed_layer.loc['requested_output', 'gradient_wrt_prompt_input_rms']:.3e} |

The difficult prompt is diluted internally: modulation falls `{difficult_drop:.1%}` from enc0_second to enc3 and the layer-energy prompt gradient attenuates by `{difficult_gradient_attenuation:.1f}x`; decoder skips restore visible prompt dependence by dec0. That cannot be the campaign-wide primary cause because mixed-eight retains strong bottleneck modulation (`{mixed_layer.loc['enc3', 'feature_modulation']:.3f}`) and stronger requested-output modulation (`{mixed_layer.loc['requested_output', 'feature_modulation']:.3f}`) yet still fails identity. The failure is semantic strength/alignment, not the first layer going numerically blind.

Final objective prompt-gradient RMS was `{pd.read_csv(run / 'tables/prompt_gradient_metrics.csv').set_index('condition').loc['difficult_one_scene', 'objective_prompt_gradient_rms']:.3e}` difficult and `{pd.read_csv(run / 'tables/prompt_gradient_metrics.csv').set_index('condition').loc['mixed_eight_scene', 'objective_prompt_gradient_rms']:.3e}` mixed-eight.

## Why conditioning, not reconstruction capacity, is the remaining bottleneck

- Total-objective reduction was `{difficult.objective_reduction:.6f}` difficult and `{mixed.objective_reduction:.6f}` mixed; unchanged requested/companion reduction gates passed in both.
- The same architecture achieves both-view identity on index 81 and previously on the ordinary scene, demonstrating representational reach without changing capacity.
- Requested/companion outputs stayed nonnegative and finite, and the signed residual conserved the observation. Maximum closure error/tolerance was `{difficult.maximum_conservation_error:.6g}/{difficult.maximum_conservation_tolerance:.6g}` difficult and `{mixed.maximum_conservation_error:.6g}/{mixed.maximum_conservation_tolerance:.6g}` mixed.
- What fails is role assignment: companion leakage remains `{difficult.companion_leakage:.3f}` / `{mixed.companion_leakage:.3f}`, identity margins are near zero for seven mixed scenes, and their predicted prompt contrasts are not aligned with the true source contrasts.

Therefore reconstruction capacity is not the gate that remains. Family-E1 learns prompt-responsive features and scalar changes, but the identity-aligned component is too weak to override scene-specific source ordering/mixing.

## Classification and authorization

Primary classification: **Prompt too weak** (identity-aligned component). Prompt ignored, forgotten, overwritten, skip-dominated, and decoder-ignored are rejected by the nonzero layerwise modulation/gradients. Prompt dilution is a real secondary difficult-scene observation but cannot explain the mixed-eight failure.

The success clause is not activated. **Do not resume Family-E1 full training.** No full training, validation, calibration, OOF generation, safety labeling, auditor work, or alternate experiment was authorized or run.

## Integrity

- Exact architecture/parameter count: unchanged FamilyE1UNet, `1,162,662`; no new model modules or loss terms.
- MPS only; CPU fallback false; same optimizer, weights, thresholds, seeds, and update budgets.
- Focused tests: `{tests_summary}`; compileall, CSV schemas, and git diff checks passed.
- README SHA-256 remains `{readme_current}`.
- `{len(historical_rows) + len(external_rows)}` historical checkpoints were rehashed with zero mismatches; the two new files are explicitly micro-only states inside this run.
- Git index empty; nothing staged or committed. Authoritative inputs unchanged. Validation/calibration/development/Atlas/lockbox access and OOF/label/auditor counts are all zero.

## Evidence inventory

- Frozen contract: `preregistration/family_e1p_paired_prompt_identity_intervention.md`.
- Aggregate and per-scene metrics: `tables/micro_overfit_results.csv`, `tables/per_scene_pair_metrics.csv`.
- Per-view role and contrast diagnosis: `tables/per_view_identity_diagnostics.csv`, `tables/common_contrast_diagnostics.csv`.
- Full layer trace and gradients: `tables/layerwise_prompt_trace.csv`, `tables/prompt_gradient_metrics.csv`.
- Failure taxonomy: `tables/failure_taxonomy.csv`, `diagnostics/failure_classification.json`.
- Exact prior replication: `tables/prior_family_e1_replication.csv`.
- Visual summary: `figures/prompt_conditioning_diagnosis.png`.
- Integrity: `tables/integrity_checks.csv` and checkpoint inventories.
"""
    fresh_text(run / "reports/final_report.md", report)
    fresh_text(run / "reports/final_report.sha256", f"{sha256_file(run / 'reports/final_report.md')}  final_report.md\n")
    fresh_json(
        run / "reports/frozen_core_decision.json",
        {
            "outcome": "FAMILY_E1P_PROMPT_IDENTITY_FAILURE",
            "classification": "Prompt too weak",
            "qualification": "identity-aligned component too weak despite generic prompt responsiveness",
            "difficult_prompt_identity": float(difficult.prompt_identity),
            "mixed_eight_prompt_identity": float(mixed.prompt_identity),
            "identity_gate": 0.90,
            "full_training_authorized": False,
            "authorized_experiment": None,
            "architecture_change": False,
            "objective_change": False,
        },
    )
    git_status = command(["git", "status", "--short", "--branch"])
    fresh_text(run / "diagnostics/final_git_status.txt", git_status.stdout + git_status.stderr)
    run_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    fresh_json(
        run / "logs/campaign_end.json",
        {
            "status": "COMPLETE_FAIL_CLOSED",
            "outcome": "FAMILY_E1P_PROMPT_IDENTITY_FAILURE",
            "classification": "Prompt too weak",
            "full_training_authorized": False,
            "runtime_seconds_finalization": time.time() - started,
            "run_disk_bytes": run_bytes,
            "filesystem_free_bytes": shutil.disk_usage(REPO).free,
            "historical_checkpoint_mismatches": historical_mismatches,
            "staged_index_empty": not staged_result.stdout.strip(),
            "validation_access_count": 0,
            "calibration_access_count": 0,
            "development_access_count": 0,
            "atlas_selection_access_count": 0,
            "final_lockbox_access_count": 0,
            "oof_outputs": 0,
            "safety_labels": 0,
            "auditor_models": 0,
        },
    )
    print(json.dumps({"outcome": "FAMILY_E1P_PROMPT_IDENTITY_FAILURE", "report": relative(run / "reports/final_report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
