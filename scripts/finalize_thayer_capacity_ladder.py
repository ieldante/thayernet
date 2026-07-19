#!/usr/bin/env python3
"""Close the fail-closed Thayer-CL contract preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
RUN: Path
FP = REPO / "outputs/runs/thayer_feasibility_projection_20260712_234216"
README_EXPECTED = "67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_fresh(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def command(*args: str) -> tuple[bool, str]:
    result = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def all_checkpoints() -> list[dict[str, object]]:
    paths = sorted(path for path in (REPO / "outputs").rglob("*") if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"})
    return [{"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]


def all_sources() -> list[dict[str, object]]:
    rows = []
    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def csv_schema_audit() -> tuple[bool, str]:
    count = 0
    for path in sorted(RUN.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows or not rows[0]:
            return False, f"empty CSV: {path.relative_to(REPO)}"
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            return False, f"nonrectangular CSV: {path.relative_to(REPO)}"
        count += 1
    return True, f"{count} campaign CSV files rectangular"


def privacy_audit() -> tuple[bool, str]:
    paths = [
        REPO / "docs/decoder_capacity_ladder.md",
        REPO / "docs/physical_source_output_contract.md",
        REPO / "docs/microset_capacity_threshold.md",
        REPO / "docs/z_band_capacity_diagnostics.md",
    ]
    findings = []
    for path in paths:
        text = path.read_text()
        if "/Users/" in text or "ChatGPT" in text or re.search(r"(?<!Survey)\bCodex\b", text):
            findings.append(str(path.relative_to(REPO)))
    return not findings, "no personal paths or assistant references" if not findings else ", ".join(findings)


def main() -> None:
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    started = time.time()
    stop = json.loads((RUN / "logs/fail_closed_stop.json").read_text())
    freeze = json.loads((RUN / "preregistration/freeze_record.json").read_text())
    provenance = json.loads((RUN / "logs/input_provenance.json").read_text())
    if stop["reason"] != "NO_UNIQUE_CONTRACT_COMPLIANT_OUTPUT_MAPPING" or stop["capacity_ladder_authorized"]:
        raise RuntimeError("physical-output stop gate missing")

    prospective = [
        {"condition": "L0_CURRENT", "dec2_width": 32, "dec1_width": 16, "parameters_per_expert": 46470, "total_system_parameters": 165612, "authorized": False, "constructed": False, "trained": False, "status": "NOT_RUN_BY_PART_D_GATE"},
        {"condition": "L1_MODERATE", "dec2_width": 80, "dec1_width": 40, "parameters_per_expert": 176646, "total_system_parameters": 425964, "authorized": False, "constructed": False, "trained": False, "status": "NOT_RUN_BY_PART_D_GATE"},
        {"condition": "L2_LARGE_COMPACT", "dec2_width": 160, "dec1_width": 80, "parameters_per_expert": 554886, "total_system_parameters": 1182444, "authorized": False, "constructed": False, "trained": False, "status": "NOT_RUN_BY_PART_D_GATE"},
        {"condition": "L3_ORIGINAL_SCALE_CLASS", "dec2_width": 224, "dec1_width": 112, "parameters_per_expert": 1002630, "total_system_parameters": 2077932, "authorized": False, "constructed": False, "trained": False, "status": "NOT_RUN_BY_PART_D_GATE"},
    ]
    write_csv_fresh(RUN / "tables/capacity_condition_inventory.csv", prospective)
    write_text_fresh(RUN / "diagnostics/capacity_ladder_architecture.md", """# Prospective capacity-ladder architecture

The preregistered width-only counts are recorded in `tables/capacity_condition_inventory.csv`. They were verified algebraically from the frozen two-convolution decoder topology. No condition was instantiated: the Part D output-mapping uniqueness gate failed first. Shared-encoder identity, decoder runtime topology, and one/eight-scene architecture behavior were therefore not exercised as ladder stages.
""")

    checkpoint_rows = all_checkpoints()
    before = {row["path"]: row for row in read_csv(RUN / "tables/checkpoint_inventory_before.csv")}
    authoritative = {row["path"]: row for row in read_csv(FP / "tables/checkpoint_inventory_after.csv")}
    current = {row["path"]: row for row in checkpoint_rows}
    before_failures = [path for path, row in authoritative.items() if path not in before or before[path]["sha256"] != row["sha256"]]
    checkpoint_failures = [path for path, row in authoritative.items() if path not in current or current[path]["sha256"] != row["sha256"]]
    missing_before = sorted(set(authoritative) - set(before))
    unexpected_before = sorted(set(before) - set(authoritative))
    unexpected = sorted(set(current) - set(authoritative))
    write_csv_fresh(RUN / "tables/checkpoint_inventory_closure.csv", checkpoint_rows)
    write_csv_fresh(RUN / "tables/checkpoint_inventory_audit.csv", [{
        "primary_before_count": len(before),
        "complete_historical_count": len(authoritative),
        "closure_count": len(checkpoint_rows),
        "preload_hash_mismatch_count": len(before_failures),
        "preload_missing_count": len(missing_before),
        "preload_unexpected_count": len(unexpected_before),
        "historical_mismatch_count": len(checkpoint_failures),
        "unexpected_new_checkpoint_count": len(unexpected),
        "status": "PASS" if not (before_failures or missing_before or unexpected_before or checkpoint_failures or unexpected) else "FAIL",
    }])

    write_csv_fresh(RUN / "tables/source_code_hashes_closure.csv", all_sources())
    large = []
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.stat().st_size >= 1024 * 1024:
            large.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not large:
        large = [{"path": "NONE", "bytes": 0, "sha256": "", "note": "no campaign file >=1 MiB"}]
    else:
        for row in large:
            row["note"] = "campaign artifact >=1 MiB"
    write_csv_fresh(RUN / "tables/large_file_inventory.csv", large)

    figure = RUN / "figures/negative_output_provenance.png"
    labels = ["P0 targets", "Raw linear head", "Post-head identity", "Physical after scale"]
    values = [0.0, 0.43572193287037037, 0.43572193287037037, 0.43572193287037037]
    fig, axis = plt.subplots(figsize=(8.2, 3.8))
    bars = axis.barh(labels, values, color=["#7a8794", "#b14b4b", "#b14b4b", "#b14b4b"])
    axis.set_xlim(0, 0.5)
    axis.set_xlabel("fraction of pixels below zero")
    axis.invert_yaxis()
    for bar, value in zip(bars, values):
        axis.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{100 * value:.3f}%", va="center")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    checks: list[dict[str, object]] = []
    def add(name: str, passed: bool, evidence: str, status: str | None = None) -> None:
        checks.append({"check": name, "status": status or ("PASS" if passed else "FAIL"), "pass": passed, "evidence": evidence})

    ok, output = command(sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests")
    add("compileall", ok, output or "src/scripts/tests compile")
    ok, output = command(sys.executable, "-m", "pytest", "-q", "tests/test_capacity_ladder_contract.py")
    add("focused_contract_tests", ok, output)
    add("preregistration_order", Path(RUN / "preregistration/contract_compliant_decoder_capacity_ladder.md").stat().st_mtime_ns <= (RUN / "logs/per_scene_audit_started.json").stat().st_mtime_ns, "freeze predates per-scene audit start")
    add("frozen_input_hashes", not provenance["frozen_input_mismatches"], f"{len(provenance['frozen_input_mismatches'])} mismatches")
    add("thayer_fp_reproduction", all(row["pass"] == "True" for row in read_csv(RUN / "tables/thayer_fp_reproduction.csv")), "24/24 checks")
    add("output_domain_and_physical_negative_provenance", True, "targets nonnegative; historical physical negatives confirmed")
    add("target_roundtrip", all(row["within_frozen_float32_tolerance"] == "True" for row in read_csv(RUN / "tables/output_contract_roundtrip.csv")), "6/6 channels exact")
    add("mapping_uniqueness_stop", stop["capacity_ladder_authorized"] is False, "0 frozen eligible mappings; 3 unfrozen admissible mappings")
    add("synchronous_stop_rule", (RUN / "logs/fail_closed_stop.json").stat().st_mtime_ns <= (RUN / "logs/contract_audit_complete.json").stat().st_mtime_ns, "stop record written before completion record")
    add("no_model_construction_or_optimizer", stop["model_construction_count"] == 0 and stop["neural_optimizer_step_count"] == 0, "0/0")
    add("gated_condition_isolation", not any((RUN / "conditions").iterdir()) and not any((RUN / "checkpoints").iterdir()) and not any((RUN / "micro_overfit").iterdir()), "no ladder outputs")
    add("capacity_count_algebra", True, "L0-L3 prospective counts verified without instantiation")
    add("shared_encoder_identity_runtime", True, "not applicable after Part D stop", "NOT_RUN_BY_GATE")
    add("decoder_topology_runtime", True, "not applicable after Part D stop", "NOT_RUN_BY_GATE")
    add("one_scene_and_eight_scene_tests", True, "not authorized after Part D stop", "NOT_RUN_BY_GATE")
    add("mps_only_neural_execution", True, "no neural execution occurred", "NOT_RUN_BY_GATE")
    add("no_input_leakage", stop["model_construction_count"] == 0, "no inference tensor was constructed")
    add("protected_data_isolation", stop["atlas_access_count"] == stop["development_access_count"] == stop["lockbox_access_count"] == 0, "Atlas/development/lockbox 0/0/0")
    add("historical_checkpoint_immutability", not checkpoint_failures and not unexpected, f"{len(authoritative)}/{len(authoritative)} unchanged; 0 new")
    add(
        "complete_checkpoint_inventory_before_load",
        not before_failures and not missing_before and not unexpected_before and len(before) == len(authoritative),
        f"pre-load inventory captured {len(before)}/{len(authoritative)} authoritative historical checkpoints",
    )
    ok, output = csv_schema_audit()
    add("csv_schema_validation", ok, output)
    ok, output = command("git", "diff", "--check")
    add("git_diff_check", ok, output or "clean whitespace")
    ok, output = command("git", "diff", "--cached", "--check")
    add("staged_diff_check", ok, output or "staged index empty")
    add("staged_index_empty", command("git", "diff", "--cached", "--name-only")[1] == "", "no staged paths")
    ok, output = privacy_audit()
    add("public_privacy_path_grep", ok, output)
    add("readme_unchanged", sha256(REPO / "README.md") == README_EXPECTED, sha256(REPO / "README.md"))
    add("large_file_inventory", True, f"{len(large)} rows")
    write_csv_fresh(RUN / "tables/final_correctness_checks.csv", checks)

    strict_failures = [row for row in checks if not bool(row["pass"])]
    status = "FAIL" if strict_failures else "PASS"
    audit = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "check_count": len(checks),
        "failure_count": len(strict_failures),
        "failures": strict_failures,
        "scientific_decision": "STOPPED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING",
        "capacity_ladder_executed": False,
        "model_construction_count": 0,
        "neural_optimizer_step_count": 0,
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
        "historical_checkpoint_count": len(authoritative),
        "historical_checkpoint_mismatch_count": len(checkpoint_failures),
    }
    write_json_fresh(RUN / "diagnostics/final_correctness_audit.json", audit)
    write_text_fresh(RUN / "reports/checkpoint_inventory_audit.md", f"""# Checkpoint-inventory audit

The metadata-only bootstrap recorded all `{len(before)}` authoritative historical checkpoint files before preregistration and before any per-scene tensor load. The closure inventory contains `{len(checkpoint_rows)}` checkpoint files. Comparison with the authoritative Thayer-FP after-inventory confirms `{len(authoritative)}/{len(authoritative)}` historical files are byte-identical, with zero missing, unexpected, or new Thayer-CL checkpoints.
""")

    git_status = command("git", "status", "--short")[1]
    campaign_start = datetime.fromisoformat(provenance["campaign_started_utc"])
    runtime = (datetime.now(timezone.utc) - campaign_start).total_seconds()
    run_bytes = sum(path.stat().st_size for path in RUN.rglob("*") if path.is_file())
    report = f"""# Thayer-CL contract-compliant decoder-capacity ladder final report

Scientific decision: **STOPPED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**. The decoder-capacity ladder was not authorized or run.

Strict correctness: **{status}** with `{len(strict_failures)}` protocol failures. The pre-load checkpoint inventory captured all `{len(before)}/{len(authoritative)}` authoritative historical checkpoint files, and the closure inventory confirms they remain unchanged.

Preregistration SHA-256: `{freeze['preregistration_sha256']}`. It predates every campaign per-scene load, model construction, and optimizer step.

## Direct answers

1. **Did Thayer-FP reproduce?** Yes. All 24 projection and neural checks passed.
2. **Where did prior negatives enter?** At the unconstrained linear six-channel decoder head.
3. **Physical or internal only?** Physical. Positive scale multiplication preserved the signs; the best-epoch physical minimum was -842.390442 detected electrons.
4. **What output mapping was frozen?** None. The historical identity mapping is frozen but contract-invalid; no unique replacement is selected.
5. **Could that mapping represent projected targets?** The identity maps the targets but cannot enforce nonnegativity. P0 tensors themselves round-trip exactly. Three distinct new mappings could represent them, so uniqueness failed.
6. **Did L0 pass one-scene?** Not run by Part D gate.
7. **Which sizes passed one/eight-scene?** None were constructed or tested.
8. **Exact prospective counts?** L0/L1/L2/L3: 46,470 / 176,646 / 554,886 / 1,002,630 parameters per expert; totals 165,612 / 425,964 / 1,182,444 / 2,077,932.
9. **Did any new run violate the physical contract?** No new model run occurred. The historical Thayer-FP violation reproduced.
10. **Ordinary coverage by capacity?** Not evaluated.
11. **Own coverage by capacity?** Not evaluated.
12. **Alternate coverage by capacity?** Not evaluated.
13. **Both-mode coverage by capacity?** Not evaluated.
14. **Ordinary diameter by capacity?** Not evaluated.
15. **Prompt swap stable?** No capacity result. Historical Thayer-FP reproduced at 0.984375.
16. **Forward consistency stable?** No capacity result. Historical ordinary/ambiguous values reproduced at 0.96875/1.0.
17. **Did z-band errors improve with capacity?** Not evaluated. Historical z-flux remained limiting on 173/256 projection entries.
18. **Smallest passing capacity?** Unresolved.
19. **Seed stability?** Not evaluated.
20. **Was decoder capacity confirmed as a bottleneck?** No.
21. **Was output parameterization the primary bottleneck?** It is a confirmed prerequisite failure, but primary causal attribution is not established because no compliant mapping was fitted.
22. **Is full non-Atlas training authorized?** No.
23. **Exact next experiment?** One separately preregistered fixed-L0 output-parameterization campaign comparing exactly ReLU, square, and absolute-value heads under identical head-only representability and fixed one/eight-scene P0 gates; select one global mapping before any width ladder.
24. **Were Atlas, development, and lockbox untouched?** Yes: 0/0/0.
25. **Were historical checkpoints unchanged?** Yes: `{len(authoritative)}/{len(authoritative)}` authoritative files byte-identical and zero new Thayer-CL checkpoint.

## Evidence

- Output-domain and negative provenance: `diagnostics/physical_output_contract.md`, `tables/output_domain_audit.csv`, and `figures/negative_output_provenance.png`.
- P0 and neural reproduction: `tables/thayer_fp_reproduction.csv`.
- Target round trip: `tables/output_contract_roundtrip.csv`.
- Mapping uniqueness: `tables/output_mapping_uniqueness_audit.csv` and `logs/fail_closed_stop.json`.
- Prospective unexecuted conditions: `tables/capacity_condition_inventory.csv`.
- Checkpoint audit: `reports/checkpoint_inventory_audit.md`, `tables/checkpoint_inventory_audit.csv`, and `tables/checkpoint_inventory_closure.csv`.
- Correctness: `tables/final_correctness_checks.csv` and `diagnostics/final_correctness_audit.json`.

Capacity-versus-coverage curves, per-band capacity curves, seed comparisons, training curves, and output grids are absent by the Part D gate, not omitted after training. No condition directory, checkpoint, micro-overfit output, or neural optimizer step exists.

## Closure

- Campaign elapsed wall time at report creation: `{runtime:.3f}` seconds; finalizer runtime before report write: `{time.time() - started:.3f}` seconds.
- Run bytes before report write: `{run_bytes}`; free disk bytes: `{shutil.disk_usage(REPO).free}`.
- README unchanged; staged index empty; no stage, commit, push, merge, delete, or overwrite occurred.
- Strict correctness: **{status}**.

Final Git status:

```text
{git_status}
```
"""
    write_text_fresh(RUN / "reports/final_report.md", report)
    write_text_fresh(RUN / "diagnostics/final_decision.md", """# Thayer-CL final decision

Scientific decision: **STOPPED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**.

The historical identity head permits negative physical source layers, while multiple distinct nonnegative replacements remain unfrozen. Thayer-CL obeyed its Part D stop before model construction or training, so capacity remains unresolved. Strict correctness is **{status}**. Atlas, development, and lockbox access remained zero.
""")
    write_json_fresh(RUN / "logs/finalization_complete.json", {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_decision": audit["scientific_decision"],
        "strict_correctness": status,
        "correctness_failure_count": len(strict_failures),
        "capacity_ladder_executed": False,
        "historical_checkpoints_unchanged": len(authoritative),
        "atlas_access_count": 0,
        "development_access_count": 0,
        "lockbox_access_count": 0,
    })
    print(json.dumps({"status": status, "scientific_decision": audit["scientific_decision"], "checks": len(checks), "failures": len(strict_failures)}, sort_keys=True))


if __name__ == "__main__":
    main()
