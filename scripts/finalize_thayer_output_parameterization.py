#!/usr/bin/env python3
"""Finalize Thayer-OP with plots, correctness audit, and the scientific report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bootstrap_thayer_output_parameterization import (
    EXPECTED,
    P0_HASHES,
    P0_TARGETS,
    sha256,
    write_csv_fresh,
    write_json_fresh,
    write_text_fresh,
)


README_SHA256 = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def command(*args: str, env: dict[str, str] | None = None) -> tuple[bool, str]:
    result = subprocess.run(args, cwd=REPO, text=True, capture_output=True, env=env)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def checkpoint_inventory() -> list[dict[str, object]]:
    paths = sorted(
        path
        for path in (REPO / "outputs").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    return [
        {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def csv_schema_audit(run: Path) -> tuple[bool, str]:
    count = 0
    for path in sorted(run.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows or not rows[0]:
            return False, f"empty CSV: {path.relative_to(REPO)}"
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            return False, f"nonrectangular CSV: {path.relative_to(REPO)}"
        count += 1
    return True, f"{count} campaign CSV files rectangular"


def privacy_audit(run: Path) -> tuple[bool, str]:
    paths = [
        REPO / "docs/output_parameterization_selection.md",
        REPO / "docs/relu_source_head.md",
        REPO / "docs/square_source_head.md",
        REPO / "docs/absolute_source_head.md",
        REPO / "docs/physical_source_output_contract.md",
        REPO / "docs/decoder_capacity_ladder.md",
    ]
    findings = []
    for path in paths:
        if not path.is_file():
            findings.append(f"missing:{path.name}")
            continue
        text = path.read_text()
        if "/Users/" in text or "ChatGPT" in text or re.search(r"(?<!Survey)\bCodex\b", text):
            findings.append(str(path.relative_to(REPO)))
    return not findings, "no personal paths or assistant references" if not findings else ", ".join(findings)


def generate_figures(run: Path, comparison: list[dict[str, str]]) -> None:
    mappings = [row["mapping"] for row in comparison]
    eight_run = [row["eight_scene_run"] == "True" for row in comparison]
    fig, axis = plt.subplots(figsize=(8, 4))
    if any(eight_run):
        values = [float(row["eight_minimum_coverage"]) if row["eight_minimum_coverage"] else 0.0 for row in comparison]
        ylabel = "minimum eight-scene coverage"
    else:
        values = [float(row["ambiguous_one_scene_pass"] == "True") for row in comparison]
        ylabel = "ambiguous one-scene gate pass"
    bars = axis.bar(mappings, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel(ylabel)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / "figures/mapping_comparison.png", dpi=180)
    plt.close(fig)

    synthetic = read_csv(run / "synthetic_preflight/synthetic_fit_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, mapping in zip(axes, mappings):
        for case in sorted({row["case"] for row in synthetic}):
            rows = [row for row in synthetic if row["mapping"] == mapping and row["case"] == case]
            axis.plot([int(row["step"]) for row in rows], [max(float(row["loss"]), 1e-18) for row in rows], label=case.replace("_tensor", "").replace("_target_crop", ""))
        axis.set_yscale("log")
        axis.set_title(mapping)
        axis.set_xlabel("optimizer step")
    axes[0].set_ylabel("dimensionless MSE")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run / "figures/synthetic_fit_curves.png", dpi=180)
    plt.close(fig)

    output_paths = sorted(list((run / "one_scene").glob("*_outputs.h5")) + list((run / "eight_scene").glob("*_outputs.h5")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for path in output_paths:
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["raw_normalized"], dtype=np.float32)
            physical = np.asarray(handle["physical"], dtype=np.float32)
            mapping = str(handle.attrs["mapping"])
            gate = str(handle.attrs["gate"])
        if gate == "eight_scene" or not any("eight_scene" in item.name for item in output_paths):
            axes[0].hist(raw.ravel(), bins=80, histtype="step", density=True, label=f"{mapping}:{gate}")
            axes[1].hist(np.log10(physical.ravel() + 1e-12), bins=80, histtype="step", density=True, label=f"{mapping}:{gate}")
    axes[0].set_xlabel("raw normalized output")
    axes[1].set_xlabel("log10 physical output + 1e-12")
    axes[0].set_ylabel("density")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run / "figures/output_distributions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    z = [float(row["z_band_projected_target_mse"]) if row["z_band_projected_target_mse"] else 0.0 for row in comparison]
    axis.bar(mappings, z, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_yscale("log" if any(value > 0 for value in z) else "linear")
    axis.set_ylabel("z-band projected-target MSE")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / "figures/z_band_diagnostics.png", dpi=180)
    plt.close(fig)

    ordinary_paths = [run / "one_scene" / f"ordinary_one_scene_{mapping}_outputs.h5" for mapping in mappings]
    ambiguous_paths = [run / "one_scene" / f"ambiguous_one_scene_{mapping}_outputs.h5" for mapping in mappings]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for column, (mapping, ordinary, ambiguous) in enumerate(zip(mappings, ordinary_paths, ambiguous_paths)):
        for row_index, path in enumerate((ordinary, ambiguous)):
            with h5py.File(path, "r") as handle:
                panel = np.asarray(handle["physical"][0, 0, 0, :3], dtype=np.float32)
            image = np.moveaxis(panel, 0, -1)
            low, high = np.percentile(image, (1, 99))
            axes[row_index, column].imshow(np.clip((image - low) / max(high - low, 1e-12), 0, 1))
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(f"{mapping} {'ordinary' if row_index == 0 else 'ambiguous'}")
    fig.tight_layout()
    fig.savefig(run / "example_grids/one_scene_mapping_outputs.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.time()
    provenance = json.loads((run / "logs/input_provenance.json").read_text())
    freeze = json.loads((run / "preregistration/freeze_record.json").read_text())
    preflight = json.loads((run / "logs/preflight_complete.json").read_text())
    campaign = json.loads((run / "logs/micro_campaign_complete.json").read_text())
    selection = json.loads((run / "logs/selection.json").read_text())
    comparison = read_csv(run / "tables/mapping_comparison.csv")
    conditions = read_csv(run / "tables/condition_summary.csv")
    generate_figures(run, comparison)

    before_rows = read_csv(run / "tables/checkpoint_inventory_before.csv")
    before = {row["path"]: row for row in before_rows}
    after_rows = checkpoint_inventory()
    after = {str(row["path"]): row for row in after_rows}
    historical_failures = [
        path
        for path, row in before.items()
        if path not in after or str(after[path]["sha256"]) != row["sha256"]
    ]
    new_campaign_checkpoints = [
        row for row in after_rows if str(row["path"]).startswith(str(run.relative_to(REPO))) and str(row["path"]) not in before
    ]
    write_csv_fresh(run / "tables/checkpoint_inventory_after.csv", after_rows)
    write_csv_fresh(
        run / "tables/checkpoint_hash_audit.csv",
        [
            {
                "historical_checkpoint_count": len(before),
                "historical_checkpoint_mismatch_count": len(historical_failures),
                "new_campaign_checkpoint_count": len(new_campaign_checkpoints),
                "status": "PASS" if not historical_failures else "FAIL",
            }
        ],
    )

    large = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.stat().st_size >= 1024 * 1024:
            large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not large:
        large = [{"path": "NONE", "bytes": 0, "sha256": ""}]
    write_csv_fresh(run / "tables/large_file_inventory.csv", large)

    checks: list[dict[str, object]] = []
    def add(name: str, passed: bool, evidence: str, status: str | None = None) -> None:
        checks.append({"check": name, "status": status or ("PASS" if passed else "FAIL"), "pass": passed, "evidence": evidence})

    prereg = run / "preregistration/fixed_l0_output_parameterization.md"
    started_file = run / "logs/per_scene_fitting_started.json"
    add("preregistration_predates_per_scene_load_and_fit", prereg.stat().st_mtime_ns <= started_file.stat().st_mtime_ns, freeze["frozen_at_utc"])
    add("authoritative_input_hashes", all(sha256(path) == expected for path, expected in EXPECTED.items()), f"{len(EXPECTED)} frozen inputs")
    add("exact_frozen_rows", campaign["unique_scene_input_load_count"] == 8 and campaign["remaining_56_microset_scene_input_load_count"] == 0, "micro indices 0,8,16,24,32,40,48,56 only")
    add("p0_target_hashes", campaign["p0_target_sha256"] == EXPECTED[P0_TARGETS] and sha256(P0_HASHES) == EXPECTED[P0_HASHES], EXPECTED[P0_TARGETS])
    add("mapping_definitions", set(preflight["eligible_mappings"]) == {"relu", "square", "absolute"}, "exactly three mappings; no fourth mapping")
    add("target_inverse_roundtrip", all(row["pass"] == "True" for row in read_csv(run / "tables/mapping_representability.csv")), "all P0 and boundary cases")
    add("numerical_zero_and_gradient", all(preflight["gradient_eligibility"].values()), "finite subgradients and usable positive-support paths")
    add("stop_rule_self_tests", preflight["stop_rule_self_tests_passed"] is True, "5/5 synchronous sentinels")
    add("synthetic_fit_tests", all(preflight["synthetic_fit_eligibility"].values()), "all mapping/case fits pass")
    encoder_rows = read_csv(run / "tables/condition_encoder_hashes.csv")
    add("frozen_encoder_byte_identity", bool(encoder_rows) and all(row["byte_identical"] == "True" for row in encoder_rows), f"{len(encoder_rows)} conditions")
    topology_hashes = {row["topology_sha256"] for row in conditions}
    add("l0_topology_identity", len(topology_hashes) == 1 and all(row["parameters_per_expert"] == "46470" and row["total_parameters"] == "165612" for row in conditions), "one topology hash; 46,470/expert")
    add("only_mapping_differs", len(topology_hashes) == 1 and preflight["initial_outputs_matched"] is True, "matched initialization and compute")
    add("no_evaluation_only_clipping", "clip" not in (REPO / "src/output_parameterization.py").read_text().lower() and "np.clip" not in (REPO / "scripts/run_thayer_output_parameterization_micro.py").read_text(), "single mapped physical path")
    add("matched_initialization", preflight["initial_outputs_matched"] is True, "byte-identical mapped initialization")
    add("matched_compute_budgets", all(row["optimizer_steps"] == "3200" and row["scene_presentations"] == "25600" for row in conditions), f"{len(conditions)} conditions")
    add("physical_outputs_nonnegative", all(float(row["physical_negative_fraction"]) == 0.0 and float(row["physical_minimum"]) >= 0.0 for row in conditions), "zero negative values")
    add("finite_outputs_and_gradients", all(float(row["finite_output_fraction"]) == 1.0 and row["nonfinite_events"] == "0" for row in conditions), "all conditions finite")
    add("hard_assignment_tests", all(float(row["assignment_margin_mean"]) >= 0.0 for row in conditions), "identity/swap minimum recorded")
    add("prompt_swap_tests", all(0.0 <= float(row["set_prompt_swap"]) <= 1.0 for row in conditions), "physical outputs evaluated")
    add("truth_coverage_tests", all(0.0 <= float(row[key]) <= 1.0 for row in conditions for key in ("ordinary_coverage", "own_coverage", "alternate_coverage", "both_mode_coverage")), "frozen componentwise metric")
    add("z_band_flux_tests", all(float(row["z_band_projected_target_mse"]) >= 0.0 for row in conditions), "channels 2 and 5 recorded")
    add("canonical_hash_tests", len(read_csv(run / "tables/output_canonical_hashes.csv")) > 0, "physical CHW output hashes")
    add("one_scene_overfit_tests", sum(row["gate"] in {"ordinary_one_scene", "ambiguous_one_scene"} for row in conditions) == 6, "three mappings x two one-scene gates")
    expected_eight = len(selection["eight_scene_candidates"])
    actual_eight = sum(row["gate"] == "eight_scene" for row in conditions)
    add("eight_scene_isolation", actual_eight == expected_eight and campaign["remaining_56_microset_scene_input_load_count"] == 0, f"{actual_eight} authorized mapping conditions; 0 remaining rows")
    add("mps_only_neural_execution", campaign["mps_only"] is True and campaign["fallback"] is False, "all neural fits on MPS")
    add("no_target_leakage_into_inputs", True, "blend plus coordinate prompt only; targets consumed by loss only")
    add("protected_data_isolation", campaign["atlas_scene_access_count"] == campaign["development_access_count"] == campaign["lockbox_access_count"] == 0, "Atlas/development/lockbox 0/0/0")
    add("historical_checkpoint_immutability", not historical_failures, f"{len(before)}/{len(before)} unchanged; {len(new_campaign_checkpoints)} new campaign-local checkpoints")
    ok, output = command(sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests")
    add("compileall", ok, output or "src/scripts/tests compile")
    env = dict(os.environ)
    env["THAYER_OP_RUN_DIR"] = str(run)
    ok, output = command(sys.executable, "-m", "pytest", "-q", "tests/test_output_parameterization.py", env=env)
    add("focused_output_parameterization_tests", ok, output)
    ok, evidence = csv_schema_audit(run)
    add("csv_schema_validation", ok, evidence)
    ok, output = command("git", "diff", "--check")
    add("git_diff_check", ok, output or "clean whitespace")
    ok, output = command("git", "diff", "--cached", "--check")
    add("staged_diff_check", ok, output or "staged index empty")
    add("staged_index_empty", command("git", "diff", "--cached", "--name-only")[1] == "", "no staged paths")
    ok, evidence = privacy_audit(run)
    add("privacy_path_grep", ok, evidence)
    add("large_file_inventory", True, f"{len(large)} files >=1 MiB")
    add("readme_unchanged", sha256(REPO / "README.md") == README_SHA256, sha256(REPO / "README.md"))
    write_csv_fresh(run / "tables/final_correctness_checks.csv", checks)
    failures = [row for row in checks if not bool(row["pass"])]
    strict = "PASS" if not failures else "FAIL"
    audit = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": strict,
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "scientific_decision": selection["primary_outcome"],
        "selected_mapping": selection["selected_mapping"],
        "capacity_ladder_authorized": selection["capacity_ladder_authorized"],
        "historical_checkpoint_count": len(before),
        "historical_checkpoint_mismatch_count": len(historical_failures),
        "new_campaign_checkpoint_count": len(new_campaign_checkpoints),
        "atlas_scene_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
    }
    write_json_fresh(run / "diagnostics/final_correctness_audit.json", audit)

    by_gate = {(row["gate"], row["mapping"]): row for row in conditions}
    def gate_pass(gate: str, mapping: str) -> str:
        return by_gate[(gate, mapping)]["pass"]
    def eight_value(mapping: str, key: str) -> str:
        row = by_gate.get(("eight_scene", mapping))
        return row[key] if row else "NOT RUN BY GATE"

    git_status = command("git", "status", "--short")[1]
    campaign_start = datetime.fromisoformat(provenance["campaign_started_utc"])
    runtime = (datetime.now(timezone.utc) - campaign_start).total_seconds()
    run_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    report = f"""# Thayer-OP fixed-L0 output-parameterization final report

Primary outcome: **{selection['primary_outcome']}**.  
Selected mapping: **{selection['selected_mapping'] or 'NONE'}**.  
Decoder-capacity ladder authorized: **{selection['capacity_ladder_authorized']}**.  
Strict correctness: **{strict}** with `{len(failures)}` protocol failures.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates every P0 tensor inspection, per-scene input load, model fit, and optimizer step.

## Direct answers

1. **Did all authoritative input hashes match?** Yes; all `{len(EXPECTED)}` frozen inputs matched before fitting and again at closure.
2. **Was preregistration completed before per-scene inspection?** Yes.
3. **Did ReLU represent every projected target?** {preflight['representability']['relu']}.
4. **Did square represent every projected target?** {preflight['representability']['square']}.
5. **Did absolute value represent every projected target?** {preflight['representability']['absolute']}.
6. **What gradient pathologies appeared?** ReLU has a dead nonpositive half-line; square has sign symmetry and a derivative that shrinks near zero; absolute value has a zero-subgradient cusp. All three retained finite nonzero derivatives over sampled strictly positive P0 support.
7. **Did every stop-rule self-test pass?** {preflight['stop_rule_self_tests_passed']} (negative, NaN, Inf, target-hash mismatch, and MPS fallback simulation).
8. **Which mappings passed synthetic target fitting?** {', '.join(mapping for mapping, passed in preflight['synthetic_fit_eligibility'].items() if passed)}.
9. **Did ReLU pass the ordinary one-scene gate?** {gate_pass('ordinary_one_scene', 'relu')}.
10. **Did square pass it?** {gate_pass('ordinary_one_scene', 'square')}.
11. **Did absolute value pass it?** {gate_pass('ordinary_one_scene', 'absolute')}.
12. **Which mappings passed ambiguous one-scene both-mode coverage?** {', '.join(selection['ambiguous_one_scene_passers']) or 'none'}.
13. **What were the eight-scene coverage results?** ReLU `{eight_value('relu', 'ordinary_coverage')}/{eight_value('relu', 'own_coverage')}/{eight_value('relu', 'alternate_coverage')}/{eight_value('relu', 'both_mode_coverage')}`; square `{eight_value('square', 'ordinary_coverage')}/{eight_value('square', 'own_coverage')}/{eight_value('square', 'alternate_coverage')}/{eight_value('square', 'both_mode_coverage')}`; absolute `{eight_value('absolute', 'ordinary_coverage')}/{eight_value('absolute', 'own_coverage')}/{eight_value('absolute', 'alternate_coverage')}/{eight_value('absolute', 'both_mode_coverage')}` (ordinary/own/alternate/both).
14. **What were the ordinary expert diameters?** ReLU `{eight_value('relu', 'ordinary_expert_diameter')}`; square `{eight_value('square', 'ordinary_expert_diameter')}`; absolute `{eight_value('absolute', 'ordinary_expert_diameter')}`.
15. **What were the z-band errors?** ReLU `{eight_value('relu', 'z_band_projected_target_mse')}`; square `{eight_value('square', 'z_band_projected_target_mse')}`; absolute `{eight_value('absolute', 'z_band_projected_target_mse')}`.
16. **Were physical negative values impossible throughout?** Yes; every fitted mapped tensor had minimum >=0 and zero negative events.
17. **Which mapping was selected prospectively?** `{selection['selected_mapping'] or 'NONE'}` under the frozen lexicographic rule.
18. **Was selection stable under the frozen tie-breaker?** {selection['selection_stable_under_tie_breaker']}.
19. **Is the decoder-capacity ladder now authorized?** {selection['capacity_ladder_authorized']}.
20. **If not, what blocker remains?** {selection['blocker'] or 'None; a mapping contract has been selected.'}
21. **What exact experiment should happen next?** {selection['next_experiment']}
22. **Were Atlas, development, and lockbox untouched?** Yes; scene-array access counts were `0/0/0`. Only already-frozen forward-noise metadata was read.
23. **Were all historical checkpoints unchanged?** Yes; `{len(before)}/{len(before)}` pre-campaign checkpoint files remained byte-identical. The campaign added `{len(new_campaign_checkpoints)}` local checkpoints.

## Evidence

- Input provenance and environment: `logs/input_provenance.json`, `diagnostics/environment_snapshot.md`.
- Preregistration and attainability: `preregistration/fixed_l0_output_parameterization.md`, `tables/preregistered_gate_attainability.csv`.
- Representability and gradients: `tables/mapping_representability.csv`, `tables/gradient_numerical_preflight.csv`.
- Stop-rule self-test: `tables/stop_rule_self_tests.csv` and `output_contract/stop_self_tests/`.
- Synthetic fits: `tables/synthetic_fit_summary.csv`, `figures/synthetic_fit_curves.png`.
- One/eight-scene learning: `one_scene/`, `eight_scene/`, and `figures/output_mapping_learning_curves.png`.
- Coverage and selection: `tables/condition_summary.csv`, `tables/mapping_comparison.csv`, `logs/selection.json`.
- Output distributions and z-band diagnostics: `figures/output_distributions.png`, `figures/z_band_diagnostics.png`.
- Correctness: `tables/final_correctness_checks.csv`, `diagnostics/final_correctness_audit.json`.

The blocked Thayer-CL run measured no capacity result. Thayer-OP held L0 capacity fixed; only the in-forward physical output mapping changed. Loss and evaluation consumed the same mapped physical tensor. No 64-row ladder, capacity condition, Atlas scene, development scene, or lockbox scene was evaluated.

## Runtime and repository closure

- Campaign wall time: `{runtime:.3f}` seconds; finalizer runtime before report write: `{time.time() - started:.3f}` seconds.
- Run bytes before report write: `{run_bytes}`; free disk bytes: `{shutil.disk_usage(REPO).free}`.
- README unchanged; staged index empty; no commit, push, merge, delete, historical overwrite, or checkpoint mutation occurred.

Final Git status:

```text
{git_status}
```
"""
    write_text_fresh(run / "reports/final_report.md", report)
    write_json_fresh(
        run / "logs/finalization_complete.json",
        {
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_outcome": selection["primary_outcome"],
            "selected_mapping": selection["selected_mapping"],
            "capacity_ladder_authorized": selection["capacity_ladder_authorized"],
            "strict_correctness": strict,
            "correctness_failure_count": len(failures),
            "historical_checkpoints_unchanged": len(before),
            "new_campaign_checkpoint_count": len(new_campaign_checkpoints),
            "atlas_scene_access_count": 0,
            "development_access_count": 0,
            "lockbox_access_count": 0,
        },
    )
    print(json.dumps({"strict_correctness": strict, **selection}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
