#!/usr/bin/env python3
"""Start the append-only Thayer-PU campaign and enforce its hard provenance gates."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
RESUNET = REPO / "outputs/runs/thayer_prompted_resunet_diversity_20260712_154122"
PROMPT_RUN = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"
SOURCE_SPLIT = PROMPT_RUN / "manifests/source_split_manifest.csv"
CONDITION_C = PROMPT_RUN / "checkpoints/c_randomized_coordinate_prompt_best.pth"
NORMALIZATION = PROMPT_RUN / "manifests/normalization.json"
EXPECTED = {
    CATALOG: "cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46",
    SOURCE_SPLIT: "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
    CONDITION_C: "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    RESUNET / "checkpoints/prompted_resunet_best.pth": "9595f4c99ad5e57cd7f67e3134a56aefa99746d5486c6df3482137a6bd20ef45",
    RESUNET / "checkpoints/prompted_resunet_final.pth": "90d0be0e698cbc1b8371702e5dc50ace874bfe88f946fb8effaf42c3fd228ec4",
    ATLAS / "tables/ambiguity_witness_inventory.csv": "af812eeb25c1bffced8bcc8988eb280c9b4ce3dfde56e2ff2ef94b5f38ef83a9",
    ATLAS / "tables/atlas_ambiguity_scores.csv": "9be3a9bbf876910c5982c6cb33c65037fee92ebdf702a04246366321f399dab5",
    ATLAS / "manifests/fixed_noise_contract.json": "3ce4435330da83eace363ceee3856612e100f43b63d2493aed7441992494ec7b",
    ATLAS / "tables/fixed_psf_configuration.csv": "396d6f8cea74b6f906da1965a968330fe7c9ac45dabb7c9b24a81252ae69b15d",
}
SUBDIRECTORIES = (
    "diagnostics", "tables", "figures", "logs", "reports", "preregistration",
    "manifests", "checkpoints", "features", "prior_samples", "posterior_samples",
    "candidate_sets", "atlas_evaluation", "example_grids", "paper_figures",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, cwd=REPO, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_run() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / f"outputs/runs/thayer_probabilistic_unet_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(exist_ok=False)
    return run_dir


def verify_fixed_inputs() -> dict[str, object]:
    staged = command(["git", "diff", "--cached", "--name-only"]).splitlines()
    if staged:
        raise RuntimeError(f"staged files are prohibited: {staged}")
    for path, expected in EXPECTED.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen input altered: {path.relative_to(REPO)}")

    freeze_path = ATLAS / "manifests/atlas_initial_freeze_record.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    pair_manifest = ATLAS / "tables/atlas_pair_manifest.csv"
    visual_audit = ATLAS / "tables/atlas_initial_visual_audit.csv"
    if freeze["status"] != "FROZEN_INITIAL_ATLAS_PASS" or freeze["pair_count"] != 25:
        raise RuntimeError("Atlas freeze is unresolved")
    if sha256_file(pair_manifest) != freeze["numerical_manifest_sha256"]:
        raise RuntimeError("Atlas pair manifest altered")
    if sha256_file(visual_audit) != freeze["visual_audit_sha256"]:
        raise RuntimeError("Atlas visual audit altered")

    split_summary = json.loads((ATLAS / "manifests/source_partition_summary.json").read_text())
    if split_summary["source_split_sha256"] != EXPECTED[SOURCE_SPLIT]:
        raise RuntimeError("source split provenance unresolved")
    if split_summary["sealed_lockbox_rows_committed"] != 0 or split_summary["development_rows_committed"] != 0:
        raise RuntimeError("lockbox/development exclusion unresolved")
    return {"freeze": freeze, "staged_index_empty": True}


def checkpoint_inventory() -> list[dict[str, object]]:
    reference = read_csv(ATLAS / "tables/checkpoint_inventory_before.csv")
    expected_by_path = {row["path"]: row["sha256"] for row in reference}
    for path in (RESUNET / "checkpoints/prompted_resunet_best.pth", RESUNET / "checkpoints/prompted_resunet_final.pth"):
        expected_by_path[str(path.relative_to(REPO))] = EXPECTED[path]
    rows: list[dict[str, object]] = []
    for relative, expected in sorted(expected_by_path.items()):
        path = REPO / relative
        observed = sha256_file(path)
        rows.append({
            "path": relative,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "status": "PASS" if observed == expected else "FAIL",
        })
    if len(rows) != 558 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("historical checkpoint inventory changed")
    return rows


def source_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_name in ("src", "scripts", "tests"):
        for path in sorted((REPO / root_name).rglob("*.py")):
            rows.append({
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    hard = verify_fixed_inputs()
    run_dir = create_run()
    started = datetime.now(timezone.utc).isoformat()
    checkpoints = checkpoint_inventory()
    write_csv_fresh(run_dir / "tables/checkpoint_inventory_before.csv", checkpoints)
    write_csv_fresh(run_dir / "tables/source_code_hashes_before.csv", source_inventory())

    import btk
    import galsim
    import torch

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU neural fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")

    disk = shutil.disk_usage(REPO)
    status = command(["git", "status", "--porcelain=v2", "--untracked-files=all"]).splitlines()
    atlas_paths = {
        "freeze_record": ATLAS / "manifests/atlas_initial_freeze_record.json",
        "pair_manifest": ATLAS / "tables/atlas_pair_manifest.csv",
        "visual_audit": ATLAS / "tables/atlas_initial_visual_audit.csv",
        "witness_inventory": ATLAS / "tables/ambiguity_witness_inventory.csv",
        "metric_table": ATLAS / "tables/atlas_ambiguity_scores.csv",
        "noise_contract": ATLAS / "manifests/fixed_noise_contract.json",
        "psf_contract": ATLAS / "tables/fixed_psf_configuration.csv",
    }
    provenance = {
        "campaign": "Thayer-PU prompted probabilistic U-Net multi-hypothesis feasibility",
        "campaign_started_utc": started,
        "run_dir": str(run_dir.relative_to(REPO)),
        "branch": command(["git", "branch", "--show-current"]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_v2": status,
        "staged_index_empty": hard["staged_index_empty"],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "torch": torch.__version__, "btk": btk.__version__,
            "galsim": galsim.__version__, "astropy": package_version("astropy"),
            "h5py": package_version("h5py"), "scipy": package_version("scipy"),
            "scikit_image": package_version("scikit-image"), "surveycodex": package_version("surveycodex"),
        },
        "mps": {"built": torch.backends.mps.is_built(), "available": torch.backends.mps.is_available(), "probe": "PASS", "fallback_enabled": False},
        "free_disk_bytes": disk.free,
        "source_catalog": {"path": str(CATALOG.relative_to(REPO)), "sha256": sha256_file(CATALOG)},
        "source_split": {"path": str(SOURCE_SPLIT.relative_to(REPO)), "sha256": sha256_file(SOURCE_SPLIT)},
        "condition_c": {
            "architecture_source": "scripts/thayer_select_prompt_ablation_common.py",
            "architecture_sha256": sha256_file(REPO / "scripts/thayer_select_prompt_ablation_common.py"),
            "checkpoint": str(CONDITION_C.relative_to(REPO)),
            "checkpoint_sha256": sha256_file(CONDITION_C),
            "parameter_count": 119091,
        },
        "source_layer_contract": {"path": "src/btk_scene.py", "sha256": sha256_file(REPO / "src/btk_scene.py")},
        "prompt_implementation": {"path": "scripts/thayer_select_prompt_ablation_common.py", "sha256": sha256_file(REPO / "scripts/thayer_select_prompt_ablation_common.py")},
        "normalization": {"path": str(NORMALIZATION.relative_to(REPO)), "sha256": sha256_file(NORMALIZATION)},
        "fixed_atlas_artifacts": {name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for name, path in atlas_paths.items()},
        "historical_checkpoint_count": len(checkpoints),
        "historical_checkpoint_inventory": "tables/checkpoint_inventory_before.csv",
        "source_code_inventory": "tables/source_code_hashes_before.csv",
        "development_scene_access_count": 0,
        "lockbox_scene_access_count": 0,
        "atlas_evaluation_count": 0,
    }
    write_json_fresh(run_dir / "logs/input_provenance.json", provenance)

    environment = f"""# Thayer-PU environment snapshot

- Campaign start UTC: {started}
- Branch / HEAD: `{provenance['branch']}` / `{provenance['git_head']}`
- Staged index: empty
- Python: `{sys.version.split()[0]}`
- PyTorch: `{torch.__version__}`
- MPS built / available / execution probe: `{torch.backends.mps.is_built()}` / `{torch.backends.mps.is_available()}` / `PASS`
- BTK / GalSim: `{btk.__version__}` / `{galsim.__version__}`
- Free disk: {disk.free} bytes
- Source split SHA-256: `{sha256_file(SOURCE_SPLIT)}`
- Condition-C checkpoint SHA-256: `{sha256_file(CONDITION_C)}`
- Historical checkpoints verified: {len(checkpoints)} / {len(checkpoints)}
- Frozen Atlas pair count: {hard['freeze']['pair_count']}
- Development / final-lockbox scene access: 0 / 0

The exact package inventory, initial Git status, and input hashes are in
`logs/input_provenance.json`. Neural work is MPS-only; hashing, manifests,
candidate filtering, metrics, and reports are CPU-only.
"""
    write_text_fresh(run_dir / "diagnostics/environment_snapshot.md", environment)
    contract = """# Thayer-PU campaign contract

Status: **PART A HARD GATES PASS**.

This append-only feasibility campaign tests whether a compact coordinate-conditioned
probabilistic U-Net can generate prompt-faithful, forward-consistent two-source
decompositions whose plausible diversity expands on non-Atlas near-collisions and,
only after every non-Atlas gate passes, on the frozen Ambiguity Atlas v0.

Historical runs, checkpoints, source splits, Atlas pairs, Atlas metrics, controls,
witness semantics, development scenes, and final lockbox scenes are immutable.
Atlas-related source groups are prohibited from training, validation, calibration,
and non-Atlas pair generation. Posterior truth inputs are training-only; inference
uses the truth-free conditional prior. Posterior results never stand in for prior
hypothesis quality. Arbitrary diversity and forward-inconsistent samples do not
count as ambiguity. No black-box auditor or catalog policy is authorized.

All campaign outputs are fresh, timestamped, collision-refusing, and confined to
this master run. Any staged index, provenance mismatch, MPS fallback, exposure,
hash failure, failed non-Atlas gate, or output collision stops the campaign before
the next protected stage.
"""
    write_text_fresh(run_dir / "diagnostics/campaign_contract.md", contract)
    write_json_fresh(run_dir / "logs/part_a_complete.json", {
        "status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(REPO)), "staged_index_empty": True,
        "historical_checkpoint_count": len(checkpoints), "mps_probe": "PASS",
        "atlas_artifacts_unchanged": True, "source_split_unchanged": True,
        "condition_c_unchanged": True, "lockbox_exclusion_resolved": True,
    })
    print(run_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
