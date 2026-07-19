# Thayer-Select controlled-simulation foundation — final report

Run: `outputs/runs/thayer_select_btk_foundation_20260711_152613`  
Campaign: 2026-07-11 15:26:13–15:42:37 EDT (16m24s)  
Branch/HEAD: `thayer-select` / `1a95152`  
Overall required CatSim correctness gates: **PASS**  
COSMOS execution: **BLOCKED** (no local small sample; no multi-GB download)  

## Executive result

The DR10 direct-source route is finally and explicitly rejected in
`docs/dr10_source_extraction_decision.md`. An isolated Python 3.9.6 BTK
environment was created without touching the Python 3.14 main environment.
BTK 1.0.9, GalSim 2.8.4, and surveycodex 1.2.0 produced 20 single-source and 20
two-source g,r,z engineering scenes from the official 86,273-row v1.0.9 CatSim
catalog.

All 200 simulator checks pass. Noiseless addition is exact, the noisy image is
the noiseless summed scene plus one stored BTK observation realization,
coordinates and external identities are stable, A/B prompts share an unchanged
blend, and all 40 scenes replay exactly in fresh processes. Evidence is
`tables/btk_scene_unit_tests.csv`, `tables/btk_scene_manifest.csv`, and
`diagnostics/btk_simulator_correctness_audit.md`.

A group-safe five-way CatSim assignment excludes all 60 engineering identities
and has zero source/group/engineering overlap. An optional three-epoch MPS
engineering smoke run then used only those already excluded identities. Losses
fell and prompt identity worked, but empty-prompt flux and output-range behavior
remain warnings. This was **only engineering smoke training; no scientific or
full Thayer-Select training has begun**.

## Required questions

1. **PASS — DR10 negative decision finalized.** Acquisition succeeded, but
   direct isolated-source extraction remains blocked; see
   `docs/dr10_source_extraction_decision.md` and the immutable historical
   `outputs/runs/dr10_model_probe_20260711_160018/reports/final_report.md`.

2. **PASS — isolated BTK environment created.** Path `.venv-btk`; installation,
   dependency, import, and freeze evidence is in
   `diagnostics/btk_environment_report.md` and `logs/btk_install_raw.log`.

3. **PASS — exact versions:** Python 3.9.6, blending-toolkit 1.0.9, GalSim
   2.8.4, surveycodex 1.2.0. See `logs/btk_import_smoke.txt` and
   `logs/btk_environment_freeze.txt`.

4. **PASS — CatSim catalog found and validated.** Official v1.0.9-tag file,
   13,124,160 bytes, 86,273 rows, 19 columns, hash
   `cc72782f8c4d8c549b85c0224db6d471e2ddeb0b9db73b103df714f59b746b46`;
   see `tables/btk_catalog_schema.csv` and `diagnostics/btk_api_audit.md`.

5. **PASS — 20 single-source scenes rendered.** Exact single-source identity
   and replay rows are in `tables/btk_scene_unit_tests.csv`.

6. **PASS — 20 two-source scenes rendered.** Arrays/metadata are under
   `data/btk_engineering_pilot/`; normalized query rows are in
   `tables/btk_scene_manifest.csv`.

7. **PASS — isolated-source additivity.** 20/20 double scenes have maximum
   absolute error 0 at predeclared float64 tolerance `1e-10`; 20/20 singles are
   also exact. Evidence: `tables/btk_scene_unit_tests.csv`.

8. **PASS — one-noise-realization contract.** 20/20 noisy doubles exactly equal
   noiseless plus the one saved source+sky Poisson observation realization;
   isolated arrays are unchanged. Evidence: unit-test table and
   `diagnostics/btk_api_audit.md`.

9. **PASS — A/B prompts share one unchanged blend.** 20/20 pairs have two
   normalized query rows with one arrays hash/blend key and separate prompt and
   target keys; see `tables/btk_scene_manifest.csv`.

10. **PASS — fresh-process replay.** 40/40 scenes reproduce source selection,
    positions, isolated arrays, noiseless blend, noise realization, and noisy
    blend exactly; see `tables/btk_fresh_process_replay.csv` and
    `logs/final_explicit_seed_replay.json`.

11. **PASS — approximate DR10-conditioned simulation is feasible.** It is not
    exact DR10 reproduction. Parameters and limitations are in
    `tables/dr10_like_simulation_parameters.csv` and
    `docs/dr10_like_simulation_contract.md`.

12. **PASS — five-way source split designed and group-safe.** Counts are
    60,339 training; 8,630 validation; 6,901 calibration; 6,033 development
    test; and 4,310 sealed lockbox. Exact-position duplicate groups do not
    cross; engineering leakage is zero; maximum recorded KS distance is below
    0.016. Evidence: `manifests/btk_engineering_source_groups.csv` and
    `tables/btk_split_summary.csv`.

13. **BLOCKED — COSMOS RealGalaxy execution.** APIs exist, but no local
    RealGalaxy catalog/stamps or clearly small official sample exists. The
    official downloader states almost 6 GB unpacked, so nothing was downloaded
    or rendered. See `diagnostics/cosmos_realgalaxy_feasibility.md`.

14. **PASS — tiny smoke run authorized.** Every prerequisite passed before
    execution; see `diagnostics/thayer_select_smoke_gate_checklist.md`.

15. **PASS — tiny smoke run completed.** 500 training plus 100 validation
    two-source scenes, equal A/B querying, three epochs, MPS, one fixed
    three-layer convolutional model. Training loss 0.008748→0.004687 and
    validation loss 0.005927→0.003996. Prompt A/B correct-source MSE is lower
    than swapped-source MSE. See `reports/thayer_select_btk_smoke_results.json`
    and `tables/thayer_select_btk_smoke_epochs.csv`.

16. **NOT RUN — actual scientific training has not begun.** Only the explicitly
    authorized engineering smoke run occurred. Its checkpoints are not a
    scientific result, benchmark, production model, current best model, or
    calibrated result.

17. **BLOCKED — first full experiment still requires a frozen, human-reviewed
    scientific generator/trainer/config and explicit authorization.** The smoke
    also leaves an empty-prompt absolute-flux ratio of 0.185, only 67.2% of
    queried outputs inside the central 0.1–99.9% training-target range, and no
    three/four-source, uncertainty, recoverability, calibration, COSMOS-shift,
    or DR10-shift experiment. These are not correctness failures in the BTK
    scene foundation, but they prevent calling the smoke scientifically ready.

18. **PASS — exact next permitted command:** run another explicit-seed fresh
    replay, not full training:

```bash
cd <REPOSITORY_ROOT>
.venv-btk/bin/python scripts/run_btk_engineering_pilot.py \
  --run-dir outputs/runs/thayer_select_btk_foundation_20260711_152613 \
  --catalog outputs/runs/thayer_select_btk_foundation_20260711_152613/data/input_catalog_btk_v1.0.9.fits \
  --replay-metadata outputs/runs/thayer_select_btk_foundation_20260711_152613/data/btk_engineering_pilot/double_002.json \
  --expected-noise-seed 2026072301 \
  > outputs/runs/thayer_select_btk_foundation_20260711_152613/logs/next_explicit_seed_replay_double_002.json
```

## Lockbox and split isolation

1. **Path discovery:** the DR10 manifest and policy were located at
   `data/manifests/dr10_grouped_source_split_20260711_024415/`, including
   `source_split_manifest.csv` and `lockbox_policy.md`.
2. **Metadata-only exclusion:** **Yes.** Policy/aggregate summaries and only
   split/source/group identity columns were used to establish
   `future_lockbox` exclusion and verify no group crosses sealed/unsealed roles.
   No individual lockbox identity was persisted by this campaign.
3. **Underlying DR10 lockbox source-content access:** **No.** No linked FITS,
   image, catalog record, path, photometry, morphology, coordinate, statistic,
   hash, sample, or rendering from a lockbox row was accessed.

The new CatSim sealed-lockbox assignment was written but no sealed-lockbox
source was sampled, rendered, plotted, visualized, or used by the smoke run.

## Integrity, disk, and repository state

- Run disk usage: 141 MB; BTK environment: 301 MB; run data: 110 MB;
  campaign-local checkpoints: 40 KB.
- Pre-existing checkpoint integrity: **PASS, 18/18 byte-for-byte identical**;
  the before and after CSVs compare exactly. The two smoke checkpoints are new,
  unique, and confined to this run.
- Authorized source/docs: nine paths; initial/final hashes are in
  `tables/authorized_file_hashes.csv`. The only pre-existing authorized file
  changed was `docs/recoverability_operational_definition.md`, with a localized
  16-line clarification; its original bytes are preserved under
  `diagnostics/pre_edit_snapshots/`.
- Historical outputs changed: **No.** Historical DR10 reports/runs and all
  historical checkpoints were read-only.
- Final Git status: branch aligned with `origin/thayer-select`; one modified
  tracked doc, eight untracked source/docs files, and untracked `.venv-btk/`.
  The campaign run is ignored by the existing `outputs/` rule. `.venv-btk/` is
  not ignored and must never be staged.
- Staged index: empty. No commit, stage, push, pull, fetch, merge, rebase,
  cherry-pick, branch switch/create, stash, reset, restore, clean, deletion, or
  historical overwrite occurred.
- Unsupported scientific claims: none. The simulator, split, and smoke results
  are limited to the recorded engineering evidence.

Complete command/status evidence is `logs/important_commands.md`; final Git
evidence is `logs/repository_status_final.txt`; future commit separation is
`diagnostics/commit_plan.md`.
