# Thayer-Select promptability baseline — final report

Run: `outputs/runs/thayer_select_prompt_ablation_20260711_164329`  
Scientific question: Does coordinate conditioning let a compact model reconstruct an arbitrary requested galaxy after the centered-target shortcut is removed?  
Decision: **SUCCESS**  
Rationale: Condition C substantially reduced mean and 1%-trimmed randomized reconstruction risk, won 720/1000 whole-image comparisons, achieved 98.0% prompt-swap success with 0.2% output collapse, and was competitive with the centered control without leakage; the near-even source-region win count and empty-prompt hallucination remain important non-uniformity and no-harm caveats.

## Executive result

All source, simulator, replay, MPS, manifest, and historical-checkpoint gates passed. Condition B is interpreted as an identifiability/centrality control, not a production deblender. Conditions B and C used byte-identical randomized development scenes and requested identities; Condition A used the aligned centered-position variant. No ratio across the A and B scene variants is reported.

The paired randomized source-region MSE effect was C−B = **-1.00927e+06** (95% bootstrap CI -1.97662e+06 to -292977); C won 499/1000 scenes. Prompt-swap success was **0.98**, output-collapse rate was **0.002**, and changing only the prompt changed the output in **1** of scenes.

The distribution is heavy-tailed and is reported explicitly: median source MSE was 9343.04 for B and 11346.9 for C; the 1%-trimmed means were 284164 and 133976. C won 720/1000 whole-image comparisons. Thus the mean improvement is not described as uniform per scene.

## Required answers

1. **Explicit-seed replay:** PASS. Source IDs, source coordinates, isolated arrays, noiseless/noisy blends, prompt maps, seed `2026072301`, and catalog/metadata/array hashes matched exactly (`logs/explicit_seed_replay_verification.json`).
2. **Group-safe source split:** PASS. 86,273 persistent source identities and exact-position duplicate groups have zero cross-partition crossings (`diagnostics/group_integrity_report.json`).
3. **Lockbox sealed:** YES. Zero lockbox scene definitions, renders, arrays, plots, or evaluations were created. Only assignment metadata was counted/hashed.
4. **Training on MPS:** YES. All three models completed 20/20 epochs on MPS with no CPU neural fallback.
5. **Exact parameter counts:** A=118,947; B=118,947; C=119,091. The sole prompt-related difference is **+144** first-layer weights.
6. **Randomization cost:** B randomized source MSE was 2.02931e+06; A centered source MSE was 1.82721e+06. The aligned absolute difference B−A was 202105; this is not presented as a cross-manifest ratio.
7. **Prompt recovery:** C randomized source MSE was 1.02004e+06, an absolute paired change of -1.00927e+06 from B, with CI above.
8. **Prompt swapping:** Source-swap success=0.98; output-collapse=0.002; prompt sensitivity ratio=1.607.
9. **Coordinate versus brightness/centrality:** Condition-C requested-identity evidence is in `tables/prompt_swap_per_scene.csv`; Condition-B tendencies are: closer to central 483/1000, brighter 399/1000, larger 516/1000, and average over either source 953/1000.
10. **Failure modes:** The principal observed weaknesses are quantified by prompt collapse, swap failures, stratified source confusion, and small-offset sensitivity (`tables/centrality_stratified_summary.csv`, `tables/prompt_swap_per_scene.csv`).
11. **Basic no-harm tests:** Empty-prompt hallucination rate=1 under the predeclared diagnostic definition (absolute predicted flux >10% of requested-source flux); isolated-source source MSE=905164; wrong-prompt confusion rate=0.99. These are engineering diagnostics, not calibrated abstention results.
12. **Promptability classification:** **SUCCESS**. Condition C substantially reduced mean and 1%-trimmed randomized reconstruction risk, won 720/1000 whole-image comparisons, achieved 98.0% prompt-swap success with 0.2% output collapse, and was competitive with the centered control without leakage; the near-even source-region win count and empty-prompt hallucination remain important non-uniformity and no-harm caveats.
13. **Ready for recoverability/abstention:** Yes—promptability passed, so a separately frozen recoverability/abstention campaign is justified.
14. **Exact next experiment:** Freeze a new group-safe campaign on the same promptable Condition-C backbone, then add recoverability prediction and bounded uncertainty using calibration-only threshold selection; keep the current development test and lockbox untouched.

## Primary metric table (macro per scene)

| Condition | Whole MSE | Source MSE | Source MAE | PSNR | SSIM | Centroid error (px) | Worse than input |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_centered_no_prompt | 261612 | 1.82721e+06 | 162.518 | 20.4967 | 0.191311 | 2.5596 | 26 |
| B_randomized_no_prompt | 276782 | 2.02931e+06 | 174.497 | 23.2453 | 0.2478 | 6.03618 | 35 |
| C_randomized_coordinate_prompt | 223204 | 1.02004e+06 | 154.345 | 25.9188 | 0.384605 | 3.82446 | 20 |


Micro source/core/non-core aggregations are in `tables/primary_metrics_micro.csv`; per-sample metrics, flux/color errors, win/loss/tie inputs, and identity comparisons are in `tables/primary_metrics_per_sample.csv`. Quantiles and trimmed distribution summaries are in `tables/paired_distribution_summary.csv`.

## Figures and diagnostic evidence

- Training curves: `figures/training_curves.png`
- Defining prompt-swap grid: `figures/prompt_swap_flagship_grid.png`
- Prompt-swap table and summary: `tables/prompt_swap_per_scene.csv`, `reports/prompt_swap_summary.json`
- Centrality stratification: `tables/centrality_stratification_per_scene.csv`, `tables/centrality_stratified_summary.csv`
- No-harm diagnostics: `tables/no_harm_per_sample.csv`, `tables/no_harm_summary.csv`

## Provenance and integrity

- Source split and scene hashes: `manifests/source_split_manifest.csv`, `manifests/rendered_scene_manifest.csv`, `tables/btk_foundation_inputs.csv`
- New checkpoint hashes: the three frozen training configs and `tables/campaign_file_hashes.csv`
- Historical checkpoints: PASS, 18/18 unchanged (`tables/historical_checkpoint_hashes_before.csv`, `tables/historical_checkpoint_hashes_after.csv`)
- Training runtime: 23.12 minutes; campaign elapsed wall time: 35.96 minutes
- Run disk usage before final hash inventory: 2.630 GiB
- Git status: `logs/git_final.txt`; Git diff whitespace check: PASS
- Compileall, relevant unittests, CSV/schema, privacy/path, formula, and MPS audits: PASS

No uncertainty, recoverability, abstention, calibration selection, COSMOS, DR10, or lockbox experiment was performed.
