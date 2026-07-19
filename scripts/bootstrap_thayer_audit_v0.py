#!/usr/bin/env python3
"""Bootstrap and preregister the append-only Thayer-Audit v0 campaign."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import scipy
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.direct_catalog_safety_auditor import (
    FALSE_SUBTRACTION_LIMIT,
    IMAGE_CHANNEL_CLIP,
    MEAN_PSF_FWHM_PIXELS,
    PostAuditSafetyNetwork,
    PreAuditQueryNetwork,
    SCALAR_FEATURE_NAMES,
    SCIENTIFIC_CENTROID_LIMIT_PSF,
    SCIENTIFIC_COLOR_LIMIT_MAG,
    SCIENTIFIC_FLUX_LIMIT,
    SCIENTIFIC_IMAGE_LIMIT,
    SOURCE_SUPPORT_FRACTION,
    trainable_parameter_count,
)


HIERARCHICAL = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
OBSERVABILITY = REPO / "outputs/runs/thayer_select_observability_distillation_20260712_035843"
ATLAS = REPO / "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627"
THAYER_PU = REPO / "outputs/runs/thayer_probabilistic_unet_20260712_163340"
D3 = REPO / "outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200"
PROMPT_ABLATION = REPO / "outputs/runs/thayer_select_prompt_ablation_20260711_164329"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
CONDITION_C = PROMPT_ABLATION / "checkpoints/c_randomized_coordinate_prompt_best.pth"
SOURCE_SPLIT = PROMPT_ABLATION / "manifests/source_split_manifest.csv"
NORMALIZATION = PROMPT_ABLATION / "manifests/normalization.json"
CONDITION_C_SCENES = PROMPT_ABLATION / "manifests/development_scene_definitions.csv"

RUN_SUBDIRECTORIES = (
    "diagnostics", "logs", "reports", "preregistration", "manifests", "provenance",
    "frozen_models", "episodes", "features", "models", "calibration", "thresholds",
    "tables", "figures", "family_holdout", "bootstrap", "atlas_diagnostic",
    "checkpoints", "replay_verification",
)

EXPECTED_INPUT_HASHES = {
    "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/reports/final_report.md": "4f10b3982f5997729baf70adf4ad6ef14a2d4dd5fb923fb362559af70fa9b358",
    "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/reports/final_report_addendum.md": "6dc42cb286d3fe65181fd4187cd0b1da5370e12b409339dcc0c85b52518d7257",
    "docs/prospective_hierarchical_feasibility.md": "8b18ae4671c9bdcb05b52de617840afbc6f4cdce5ba0c2ebd304f6a5b08d7f6b",
    "outputs/runs/thayer_select_observability_distillation_20260712_035843/reports/final_report.md": "335e92106eaf51636762d0b3d9f9441aff305eb5c14d63b058dcb5c6bd248337",
    "outputs/runs/thayer_select_observability_distillation_20260712_035843/reports/final_report_addendum.md": "2b39c40fb1cb1ecd97dadb013286d0beab0a573b388564faa0326f8a18228aec",
    "docs/observable_regime_distillation.md": "add33851086821a0013cb7ae57357389ba40ea68684bc66297474e68925ecb23",
    "outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/reports/final_report.md": "ac6a937055d743339db3ce97d01ccc31b9f16ba89e9012ef57611db0a7677f1b",
    "docs/ambiguity_atlas_v0.md": "2947146a99a78784afefbd7980927f3c1433ca56f10559045129531daa5ca84e",
    "outputs/runs/thayer_probabilistic_unet_20260712_163340/reports/final_report.md": "675c1cd5bc939b470cab83e663642a296aabf8b3812e54ed00f9948fa7b622d8",
    "docs/thayer_probabilistic_unet.md": "59838f41a3f29aaee02cb79ddfb6a0b30ce4e5d5e1b427276dc6dda6ac5e15fc",
    "outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/reports/final_report.md": "8dd2ea12f9e2bbb3e8f90f5c34e946107d3c79a15f14ab8246a1d1fb583a052a",
    "outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/scientific_run/authoritative_pv1a1_20260714_182005_552915/reports/final_outcome.json": "f2695e956f3a5ecd328146898828f7885f8cd5fff71a0ac2794a46d4b8a5382b",
    "outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/audit/post_science_final_audit.json": "7ec8dc4d741568e15c0e8f1ebebae1caebd4a4d2ea1c56d656c1cefdf935ae7b",
    "outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth": "e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382",
    "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/source_split_manifest.csv": "98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27",
    "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/normalization.json": "940f062c01acd982f48e62d8ac283cbf4f3990a21b54cb78c5d6cb0abcb2b92a",
    "outputs/runs/thayer_select_prompt_ablation_20260711_164329/manifests/development_scene_definitions.csv": "4da021327aae9285ee1fda3464faa2365a792a8c680fe9e30e66c6ed4475ebda",
    "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/tables/fresh_dataset_inventory.csv": "bfe6acc6a44061a87c18c85778563203d5717054198ac1fd1a0039ba1ab1184c",
    "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/tables/frozen_feature_inventory.csv": "c3b715d1d2041af38388dcb99d3fd5aef1fd1844f9ff33bd59d539419f68526d",
    "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/tables/label_provenance_audit.csv": "c7ad8f5903d1d22f0a90fb7ed2452a753363f01ad01d2336032ddd808f4bb1ca",
    "src/hierarchical_feasibility.py": "bb05950df723d506713741fd4ce410bbb7267004a68bffc93bdf211c94b6bba3",
    "src/hierarchical_safety.py": "cccf04416c3ff87ee233ed15ac30cde1782299894d494337886a5f0f2932f0a3",
    "src/competing_hypotheses.py": "e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d",
    "docs/physical_source_output_contract.md": "3fbd8c019a0489106ec0be8efc1cbe0a152c36fd022e928673813c9bab74303f",
    "docs/d3_threshold_contract.md": "ac6c4585d214008c03b19b6b61b69dee999242d02d7e1cb724caf5fffa7320e3",
}

CODE_PATHS = (
    REPO / "src/direct_catalog_safety_auditor.py",
    Path(__file__).resolve(),
    REPO / "scripts/run_thayer_audit_v0.py",
    REPO / "tests/test_direct_catalog_safety_auditor.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def write_text_fresh(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_fresh(path: Path, payload: object) -> None:
    write_text_fresh(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv_fresh(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def command(arguments: list[str]) -> dict[str, object]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def validate_run_path(run: Path, *, must_exist: bool) -> None:
    expected_parent = (REPO / "outputs/runs").resolve()
    if run.parent.resolve() != expected_parent or not run.name.startswith("thayer_audit_v0_"):
        raise RuntimeError("Thayer-Audit run path is outside the append-only namespace")
    if must_exist and not run.is_dir():
        raise FileNotFoundError(run)
    if not must_exist and run.exists():
        raise FileExistsError(run)


def verify_authoritative_inputs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, expected in EXPECTED_INPUT_HASHES.items():
        path = REPO / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        status = "PASS" if observed == expected else "FAIL_CHANGED_AUTHORITATIVE_INPUT"
        rows.append({"relative_path": name, "expected_sha256": expected, "observed_sha256": observed,
                     "size_bytes": path.stat().st_size, "status": status})
        if observed != expected:
            raise RuntimeError(f"authoritative input changed: {name}")
    inventory = pd.read_csv(HIERARCHICAL / "tables/fresh_dataset_inventory.csv", keep_default_na=False)
    frozen = pd.read_csv(HIERARCHICAL / "tables/frozen_feature_inventory.csv", keep_default_na=False)
    for row in inventory.itertuples(index=False):
        manifest = HIERARCHICAL / f"manifests/v2_{row.dataset}_scene_manifest.csv"
        scenes = HIERARCHICAL / f"manifests/v2_{row.dataset}_scenes.h5"
        for path, expected in ((manifest, row.manifest_sha256), (scenes, row.hdf5_sha256)):
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(f"hierarchical input changed: {relative(path)}")
            rows.append({"relative_path": relative(path), "expected_sha256": expected, "observed_sha256": observed,
                         "size_bytes": path.stat().st_size, "status": "PASS"})
    for row in frozen.itertuples(index=False):
        path = REPO / row.reconstruction_file
        observed = sha256_file(path)
        if observed != row.reconstruction_file_sha256:
            raise RuntimeError(f"frozen reconstruction output changed: {relative(path)}")
        rows.append({"relative_path": relative(path), "expected_sha256": row.reconstruction_file_sha256,
                     "observed_sha256": observed, "size_bytes": path.stat().st_size, "status": "PASS"})
    return rows


def checkpoint_inventory(run: Path) -> pd.DataFrame:
    rows = []
    for suffix in ("*.pth", "*.pt", "*.ckpt"):
        for path in sorted((REPO / "outputs/runs").rglob(suffix)):
            if run not in path.parents and path.is_file():
                rows.append({"relative_path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows).drop_duplicates("relative_path").sort_values("relative_path", kind="stable")
    return frame.reset_index(drop=True)


def bootstrap(run: Path) -> None:
    validate_run_path(run, must_exist=False)
    staged = command(["git", "diff", "--cached", "--name-status"])
    if staged["returncode"] != 0 or str(staged["stdout"]).strip():
        raise RuntimeError("staged index is not empty")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; neural CPU fallback is prohibited")
    probe = torch.ones(2, device="mps")
    if probe.device.type != "mps" or float(probe.sum().cpu()) != 2.0:
        raise RuntimeError("MPS execution probe failed")
    authoritative = verify_authoritative_inputs()

    run.mkdir(parents=True, exist_ok=False)
    for name in RUN_SUBDIRECTORIES:
        (run / name).mkdir(exist_ok=False)
    started = datetime.now(timezone.utc)
    checkpoints = checkpoint_inventory(run)
    write_csv_fresh(run / "tables/checkpoint_inventory_before.csv", checkpoints)
    write_csv_fresh(run / "provenance/authoritative_input_inventory.csv", pd.DataFrame(authoritative))
    code_hashes = [{"relative_path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                   for path in CODE_PATHS]
    write_csv_fresh(run / "provenance/campaign_code_hashes_before.csv", pd.DataFrame(code_hashes))

    git = {
        "branch": command(["git", "branch", "--show-current"]),
        "head": command(["git", "rev-parse", "HEAD"]),
        "status": command(["git", "status", "--short", "--branch"]),
        "staged_index": staged,
    }
    packages = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "h5py": h5py.__version__, "blending-toolkit": package_version("blending-toolkit"),
        "GalSim": package_version("GalSim"), "surveycodex": package_version("surveycodex"),
        "astropy": package_version("astropy"), "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(), "mps_execution_probe": 2.0,
    }
    disk = shutil.disk_usage(REPO)
    provenance = {
        "campaign": "Thayer-Audit-v0",
        "meaning": "Direct Pre-Deblendability and Post-Reconstruction Safety Audit",
        "campaign_start_utc": started.isoformat(),
        "run_dir": relative(run),
        "git": git,
        "packages": packages,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "mps_required_for_auditor_training": True,
        "cpu_allowed_for_artifact_metrics_calibration_bootstrap_reports": True,
        "authoritative_inputs": authoritative,
        "condition_c_checkpoint": {"path": relative(CONDITION_C), "sha256": sha256_file(CONDITION_C)},
        "partition_manifest": {"path": relative(SOURCE_SPLIT), "sha256": sha256_file(SOURCE_SPLIT)},
        "base_fit_scene_manifest": {"path": relative(CONDITION_C_SCENES), "sha256": sha256_file(CONDITION_C_SCENES)},
        "normalization": {"path": relative(NORMALIZATION), "sha256": sha256_file(NORMALIZATION)},
        "scientific_threshold_contract": {"path": "docs/d3_threshold_contract.md", "sha256": sha256_file(REPO / "docs/d3_threshold_contract.md")},
        "physical_output_contract": {"path": "docs/physical_source_output_contract.md", "sha256": sha256_file(REPO / "docs/physical_source_output_contract.md")},
        "query_semantics": {"path": "src/hierarchical_feasibility.py", "sha256": sha256_file(REPO / "src/hierarchical_feasibility.py")},
        "historical_checkpoint_count": int(len(checkpoints)),
        "historical_checkpoint_inventory": "tables/checkpoint_inventory_before.csv",
        "historical_checkpoint_inventory_sha256": sha256_file(run / "tables/checkpoint_inventory_before.csv"),
        "readme_sha256": sha256_file(REPO / "README.md"),
        "development_outcome_access_count": 0,
        "final_lockbox_outcome_access_count": 0,
        "atlas_selection_access_count": 0,
        "staged_index_empty": True,
    }
    write_json_fresh(run / "logs/input_provenance.json", provenance)
    write_json_fresh(run / "logs/bootstrap_complete.json", {
        "status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_input_count": len(authoritative), "historical_checkpoint_count": len(checkpoints),
    })
    write_text_fresh(run / "diagnostics/environment_snapshot.md", f"""# Thayer-Audit v0 environment snapshot

- Start UTC: `{started.isoformat()}`
- Branch: `{str(git['branch']['stdout']).strip()}`
- Git HEAD: `{str(git['head']['stdout']).strip()}`
- Condition-C SHA-256: `{sha256_file(CONDITION_C)}`
- Source partition SHA-256: `{sha256_file(SOURCE_SPLIT)}`
- Query-semantics SHA-256: `{sha256_file(REPO / 'src/hierarchical_feasibility.py')}`
- Scientific-threshold SHA-256: `{sha256_file(REPO / 'docs/d3_threshold_contract.md')}`
- MPS built / available / probe: `{packages['mps_built']}` / `{packages['mps_available']}` / `2.0`
- Free disk bytes: `{disk.free}`
- Historical checkpoint files: `{len(checkpoints)}`

## Package versions

```json
{json.dumps(packages, indent=2, sort_keys=True)}
```

## Initial Git status

```text
{str(git['status']['stdout']).rstrip()}
```

## Staged index

```text
{str(staged['stdout']).rstrip() or '(empty)'}
```
""")
    write_text_fresh(run / "diagnostics/campaign_contract.md", f"""# Thayer-Audit v0 campaign contract

Status: **FROZEN BOOTSTRAP BOUNDARY** before episode construction, labels, fitting, calibration, threshold selection, or Atlas diagnostic.

- D3 remains an authoritative negative result for exactly one frozen two-expert decoder setup. No D3 repair, retry, extension, or reinterpretation is authorized.
- The audit hypothesis is separate: a truth-free external layer may abstain before reconstruction and reject a proposed reconstruction afterward.
- Only duplicate-safe training, validation, and calibration partitions are eligible. Development outcomes and final-lockbox outcomes are prohibited.
- PRE-AUDIT receives normalized observed g/r/z plus the Gaussian prompt only.
- POST-AUDIT receives normalized blend, prompt, frozen reconstruction, residual, and the frozen {len(SCALAR_FEATURE_NAMES)}-scalar deployable feature vector only.
- Truth, identities, groups, model-family identity, physical difficulty, SNR, obstruction, separation, flux ratio, generator parameters, targets, errors, gradients, optimizer state, and training trajectories never enter auditor inference inputs.
- Reconstruction checkpoints and outputs are immutable. Auditor training is MPS-only; CPU is limited to manifests, labels, metrics, calibration, bootstrap, plots, and reports.
- All artifacts are fresh, append-only, and collision-refusing. No stage, commit, push, merge, delete, move, historical overwrite, or README edit is authorized.
- Atlas v0 is a post-freeze development diagnostic only and cannot determine architecture, checkpoint, calibration, threshold, campaign success, or outcome.
- Exact conditional coverage is not claimed.
""")
    print(json.dumps({"status": "PASS", "run_dir": str(run), "started_utc": started.isoformat()}, sort_keys=True))


def preregister(run: Path) -> None:
    validate_run_path(run, must_exist=True)
    if json.loads((run / "logs/bootstrap_complete.json").read_text())["status"] != "PASS":
        raise RuntimeError("bootstrap did not pass")
    forbidden = (run / "logs/episode_construction_complete.json", run / "logs/training_complete.json", run / "logs/calibration_complete.json")
    if any(path.exists() for path in forbidden):
        raise RuntimeError("preregistration attempted after data or fitting")
    path = run / "preregistration/direct_hierarchical_catalog_safety_auditor.md"
    if path.exists():
        raise FileExistsError(path)
    pre_parameters = trainable_parameter_count(PreAuditQueryNetwork())
    post_parameters = trainable_parameter_count(PostAuditSafetyNetwork())
    if pre_parameters > 100_000 or post_parameters > 350_000:
        raise RuntimeError("architecture parameter ceiling violated")
    text = f"""# Direct hierarchical catalog-safety auditor preregistration

Frozen UTC: `{datetime.now(timezone.utc).isoformat()}`. This document is written and hashed before episode construction, new safety-label construction, auditor fitting, calibration access, threshold selection, held-family evaluation, bootstrap, or Atlas-v0 policy evaluation.

## Scientific hypothesis and interpretation boundary

Two freshly initialized 46,470-parameter expert decoders under D3's frozen square mapping, hard assignment, direct reconstruction loss, optimizer, and 5,000-step budget did not learn both approved hidden modes. That is the narrow D3 result. It does not test whether an external observer-only audit layer can classify unsupported queries or unsafe frozen reconstructions. Thayer-Audit v0 tests that binary operational catalog-safety hypothesis without retraining, repairing, or changing any deblender.

The only success outcome is `DIRECT_AUDITOR_FEASIBILITY_PASS`. `DIRECT_AUDITOR_PARTIAL`, `DIRECT_AUDITOR_FAILURE`, and `DATA_OR_PROVENANCE_FAILURE` retain the exact meanings supplied in the campaign brief. Held-family strong pass is separate and is required before any deblender-agnostic claim.

## Partitions, frozen families, and OOF rule

Only source-split SHA-256 `{sha256_file(SOURCE_SPLIT)}` partitions `training`, `validation`, and `calibration` may be used. Development and sealed final-lockbox outcomes are unavailable to every selection, fit, calibrator, threshold, and success decision.

Eligible-family rules are frozen: immutable checkpoint/output hash; exact Gaussian prompt and g/r/z detected-electron source-layer semantics; aligned scene/prompt IDs; sufficient training, validation, and calibration coverage; passed promptability, or explicit negative/failure-domain designation; and no development/lockbox inference. Condition C is the expected primary valid family. R0/R1 remain the same architecture cluster, so their seeds/checkpoints cannot manufacture family diversity. Thayer-PU is stochastic-candidate evidence but is eligible for core family rotation only if complete aligned out-of-fit train, validation, and calibration requested-source outputs already exist under one frozen sampling rule; they are not generated merely to increase family count.

Auditor-training reconstruction rows must be persisted predictions from a base-model fold that excluded both episode source groups from fitting and validation-based checkpoint selection. For Condition C, the historical 8,000-scene fit plus 1,000-scene selection manifest defines one immutable historical base fold; only later training-partition episode rows with both groups absent from that fold are eligible. This held-out-fold subset is source-group safer than reusing in-sample outputs, but it is not described as a complete K-fold base-model cross-fit. Any overlap, missing identity, or in-sample row fails closed. No reconstruction model may be trained to fill a fold.

Validation uses frozen validation outputs for architecture/checkpoint selection only. Calibration fits temperatures and the operating threshold only. Source IDs, duplicate groups, family, and seed are grouping/evaluation metadata and never model inputs.

## Episode schema and hierarchical targets

Every row records scene ID, source-group IDs, partition, prompt ID/semantics, family/seed metadata, exact upstream tensor/output hashes, blend, Gaussian prompt, frozen requested-source reconstruction, residual, deployable scalars, supervision/evaluation labels, and provenance. Inference arrays and truth-derived supervision are stored separately and alignment-hashed.

PRE-AUDIT maps exact frozen query semantics to three labels: `UNIQUE_VALID -> VALID`, `NULL -> NULL_OR_WRONG`, and `AMBIGUOUS -> AMBIGUOUS_OR_UNSUPPORTED`. No heterogeneous composite label is created.

POST-AUDIT is fitted only on true `VALID` rows. `UNSAFE_TO_CATALOG` is the OR of the following frozen valid-query failures:

- requested-source symmetric image distance greater than `{SCIENTIFIC_IMAGE_LIMIT}`;
- any g/r/z symmetric relative-flux distance greater than `{SCIENTIFIC_FLUX_LIMIT}`;
- either applicable g-r or r-z color error greater than `{SCIENTIFIC_COLOR_LIMIT_MAG}` mag;
- applicable centroid displacement greater than `{SCIENTIFIC_CENTROID_LIMIT_PSF}` mean-PSF FWHM (mean PSF `{MEAN_PSF_FWHM_PIXELS:.9f}` pixels);
- source confusion (candidate closer by image MSE to the alternate isolated source);
- physical source-output failure: wrong shape, nonfinite value, or any negative detected-electron contribution;
- false-subtraction fraction greater than `{FALSE_SUBTRACTION_LIMIT}` on requested-source support above `{SOURCE_SUPPORT_FRACTION}` of peak outside alternate-source support; empty protected support is not applicable;
- catastrophic reconstruction MSE worse than the observed-blend identity baseline.

`catastrophic` is the OR of image/flux/color/centroid/source-confusion failures and is reported separately from the broader unsafe label. Strict greater-than thresholds preserve the frozen inclusive safe boundary. Null/ambiguous rows never receive valid-query safety labels.

## Deployable input contract

PRE-AUDIT receives exactly four image channels: blend g/r/z divided by the frozen training-only band scales and the Gaussian prompt. POST-AUDIT receives exactly ten image channels: normalized blend g/r/z, prompt, normalized proposed reconstruction g/r/z, and normalized observation-minus-reconstruction residual g/r/z. Normalized image values are deterministically finite-mapped then clipped to `[-{IMAGE_CHANNEL_CLIP}, {IMAGE_CHANNEL_CLIP}]` for auditor numerical stability; the reconstruction itself is not changed.

POST-AUDIT additionally receives exactly {len(SCALAR_FEATURE_NAMES)} deployable scalars: reconstruction band fluxes; residual band L1/L2; reconstruction peaks and sparsities; observation-to-reconstruction and prompt-to-reconstruction centroid displacement; reconstruction/residual and reconstruction/observation band ratios; and finite/nonnegative indicators. Scalar names and order are frozen in `src/direct_catalog_safety_auditor.py`. Scalars are standardized by training-only mean and standard deviation, with zero scale replaced by one.

No target, mask, true error, source/family identity, difficulty, SNR, obstruction, separation, flux ratio, morphology, generator parameter, future outcome, gradient, optimizer state, or D3 trajectory enters either network. Prompt jitter and disagreement are omitted because complete deployment-time coverage is not established on every compared partition/family. Consequently A2-D is unavailable and cannot be selected.

## Fixed architectures and training

A1 has 3x3 stride-2 convolution widths 16/32/64, GroupNorm, SiLU, global average pooling, one 64-unit hidden layer, and three outputs. It has `{pre_parameters}` trainable parameters (ceiling 100,000).

A2 has four 3x3 stride-2 blocks with widths 24/48/96/96, GroupNorm, SiLU, global average pooling, one 32-unit scalar MLP, concatenation into the exact 128-unit representation, one 128-unit fusion layer, and one unsafe logit. It has `{post_parameters}` trainable parameters (ceiling 350,000).

Exactly seeds `2026071501`, `2026071502`, and `2026071503` are used. Training is MPS-only AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, at most 30 epochs, patience 6, gradient clipping 5.0. A1 uses training-prevalence inverse-frequency weighted three-class cross-entropy. A2 uses training-prevalence inverse-frequency weighted binary cross-entropy on valid rows only. A missing class is recorded as a degenerate scientific limitation; its present class retains unit weight and metrics requiring two classes are undefined rather than invented.

Per-seed A1 checkpoint selection is lexicographic: maximum validation macro-F1, maximum ambiguous recall, minimum validation cross-entropy, earliest epoch. Per-seed A2 selection is maximum validation AUPRC, maximum validation AUROC, minimum Brier, earliest epoch. Calibration, development, Atlas, and lockbox never select a checkpoint. The frozen final predictor is the unweighted mean of the three selected seed logits. B0 accepts every true-valid row. B1 replays the existing frozen hierarchical catastrophic scalar score on aligned rows as a reference ranking only; it cannot redefine the new label or threshold.

## Calibration and threshold policy

After all checkpoints freeze, calibration-only A1 logits receive one positive temperature. A2 logits receive one positive temperature; Platt and isotonic are diagnostic only, and isotonic cannot replace the primary calibration. The probability calibrators never change rankings.

The final policy predicts the calibrated A1 argmax, abstains for either invalid class, then accepts a predicted-valid request only when calibrated A2 unsafe probability is at most the selected threshold. Threshold candidates are the fail-closed value below the minimum calibration score plus every attainable unique calibration probability. Select maximum true-valid accepted coverage, breaking ties toward the larger threshold, subject simultaneously to: unsafe-rate reduction >=50%; catastrophic-rate reduction >=50%; true-valid accepted coverage >=50%; null acceptance <=5%; ambiguity acceptance <=10%. If no candidate satisfies all constraints, freeze the fail-closed below-minimum threshold, report zero post-gate acceptance, and classify FINAL POLICY FAIL without relaxing a gate.

## Held-family, bootstrap, Atlas, success, and stopping

If two genuinely eligible aligned families exist, leave-one-family-out models exclude the held family from training, validation selection, and calibration; family identity remains absent. Otherwise all held-family metrics are `UNRESOLVED`, and no deblender-agnostic claim is permitted. Fewer than two families does not stop the one-family core.

Use 300 deterministic connected-source-group bootstrap replicates. Report percentile intervals for macro-F1, invalid recalls, AUROC, AUPRC, Brier, ECE, coverage, accepted unsafe rate, catastrophic reduction, and invalid acceptance. Physical difficulty is post-freeze analysis only. No subgroup-conditional guarantee is claimed.

Only after architectures, seed checkpoints, temperatures, and the threshold freeze may the frozen policy be evaluated on existing Atlas-v0 pairs and existing matched controls. Report abstention rates, odds ratio, scores, and source/pair-group intervals where supported. Atlas does not determine campaign success.

PRE PASS requires validation/calibration macro-F1 >=0.85/0.82; null recall >=0.95 on both; ambiguity recall >=0.80 on both; and every class recall >=0.70. POST PASS requires validation/calibration AUROC >=0.90/0.85; validation AUPRC at least prevalence+0.15; calibrated Brier below the constant-prevalence baseline; calibrated ECE <=0.10; and three-seed validation AUROC SD <=0.03. FINAL PASS requires every threshold constraint plus source-group bootstrap lower bound for unsafe-rate reduction >0 and no eligible family worse than its accept-all-valid unsafe baseline. Held-family strong pass uses the exact >=0.80 AUROC, >=0.40 coverage, >=0.25 reduction, and <=0.20 ambiguity-acceptance gates supplied in the brief.

`DIRECT_AUDITOR_FEASIBILITY_PASS` requires PRE, POST, and FINAL PASS. A pass authorizes exactly the separately preregistered `Thayer-Audit Prospective Holdout v1`; it is not run here. PARTIAL/FAILURE recommends exactly one next experiment. D3 restart or a capacity ladder is prohibited.

Stop before fitting on any changed authoritative hash, nonempty staged index, MPS failure, missing/overlapping group identity, in-sample reconstruction row, train/validation/calibration group overlap, truth-derived inference feature, family-ID input, development/lockbox outcome access, label/input misalignment, parameter-ceiling failure, nonfinite training tensor, or historical checkpoint mutation. A failed scientific success gate does not authorize repair or retuning.

## Numerical attainability audit

All F1, recall, AUROC, AUPRC, coverage, rate, and ECE gates lie in [0,1]; Brier noninferiority has an attainable perfect value 0; seed SD has attainable value 0. AUPRC >= prevalence+0.15 is mathematically attainable iff unsafe prevalence <=0.85. Because new labels do not exist before this hash, this condition is re-audited immediately after label construction and cannot be changed; prevalence >0.85 makes POST PASS unattainable under the frozen gate. A 50% relative reduction is attainable whenever baseline risk is positive; baseline risk zero makes the reduction gate fail rather than divide by zero. The simultaneous threshold constraints have the constructive attainable case of accepting at least half of valid safe rows and no unsafe/invalid rows. Their empirical joint attainability is audited before threshold selection. Bootstrap lower bound >0 is attainable under strict separation. Parameter ceilings are attained by the counts above. Three seeds, 30 epochs, patience 6, batch 128, and 300 bootstrap replicates are positive finite budgets.

Development outcome access count and final-lockbox outcome access count must remain exactly zero. Atlas selection access count must remain zero. Historical reconstruction checkpoints and outputs remain immutable.
"""
    write_text_fresh(path, text)
    digest = sha256_file(path)
    record = {
        "status": "FROZEN_BEFORE_EPISODES_OR_FITTING",
        "path": relative(path),
        "sha256": digest,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "pre_audit_parameters": pre_parameters,
        "post_audit_parameters": post_parameters,
        "development_outcomes_used": False,
        "final_lockbox_outcomes_used": False,
        "atlas_used_for_selection": False,
    }
    write_json_fresh(run / "preregistration/direct_hierarchical_catalog_safety_auditor.sha256.json", record)
    write_text_fresh(run / "preregistration/direct_hierarchical_catalog_safety_auditor.sha256", digest + "\n")
    write_json_fresh(run / "logs/preregistration_complete.json", record)
    print(json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("bootstrap", "preregister"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == "bootstrap":
        bootstrap(run)
    else:
        preregister(run)


if __name__ == "__main__":
    main()
