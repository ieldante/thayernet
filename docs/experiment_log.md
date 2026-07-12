# Experiment Log

> **Interpretive status (2026-07-10):** historical promotion language below
> records decisions made on the development suites; it is not a locked-final
> claim. The correctness audit found 29 exact-image pairs crossing the old
> row-index partitions. Exact-pixel and exact-coordinate grouped development
> infrastructure and one authorized v0.2 retrain are now complete. A fresh
> untouched final source pool remains blocked; historical metrics are preserved.

This log records the main formal experiments and the development work that made
them interpretable. Raw outputs, generated run directories, and saved model
checkpoint files remain under `outputs/` and are not committed.

## 2026-07-11 — Thayer-Select Phase I freeze and Phase II launch

The group-safe BTK promptability baseline completed Conditions A/B/C for 20
epochs on MPS. Condition C added exactly 144 parameters, reduced randomized
mean requested-source MSE from `2.029e6` to `1.020e6`, and achieved 98.0%
prompt-swap success with 0.2% output collapse. Source-region results remained
heavy-tailed, and empty-prompt hallucination was 100% under the declared rule.
The run `outputs/runs/thayer_select_prompt_ablation_20260711_164329` is frozen.

Phase II was launched with fresh 10,000/1,500/2,000 train/validation/calibration
scenes, four explicit query classes, empirical teacher-derived contract labels,
R0 reconstruction-only and R1 bounded-recoverability conditions, calibration-
only score mapping, and a development manifest generated only after full
freeze. The lockbox remains metadata-only and sealed.

### Phase II result

The authoritative run is
`outputs/runs/thayer_select_recoverability_20260711_191518`; an earlier
`..._191127` run is preserved as a pre-training CSV-schema incident. R0 and R1
completed all 20 MPS epochs. Append-only incident records preserve the
actionable-label correction, two uncertainty-saturation stops, a reporting-only
NumPy-boolean serialization resume that did not rerun inference, and a privacy
scanner self-match correction.

PERMISSIVE became primary under the predeclared imbalance fallback. Isotonic
calibration achieved AUROC 0.8746, AUPRC 0.2475, and Brier 0.0456. Development
risk declined modestly with abstention, but ambiguity ranking and useful-
coverage catastrophic-error gates failed. The campaign is PARTIAL SUCCESS.
Ambiguity feasibility found zero qualifying pairs among 77,671 candidate edges;
the full atlas and lockbox remain unauthorized.

## Development Phase: Pipeline and Evaluation Setup

- Data loading uses the local Galaxy10 DECaLS HDF5 file.
- HDF5 row indices are split into train, validation, and test arrays before
  blending, so the same row index cannot cross partitions. Pixel-identical and
  same-object duplicate rows can and do cross the historical split.
- Naive whole-cutout addition was replaced with foreground-only contaminant
  extraction to reduce rectangular cutout/background artifacts.
- Halo-aware masking was added to retain diffuse contaminant light while
  suppressing cutout-boundary leakage.
- Identity and threshold baselines were added as transparent non-learning
  references.
- Affected-region MSE and MAE were added because whole-image metrics can hide
  errors when most pixels are unchanged.
- Legacy easy/medium/hard generator labels were reframed as
  `generation_difficulty` metadata, not as true model difficulty.

Current terminology:

- `generation_difficulty`: legacy generator metadata from sampled shift,
  brightness, blur, noise, and size ratio.
- `blend_severity_score` / `blend_severity_bin`: measured image-level blend
  damage.
- `core_obstruction_fraction` / `core_overlap_bin`: how much the contaminant
  affects the target core.
- `model_failure_score`: model affected-region MSE.
- `model_improvement_ratio`: identity affected-region MSE divided by model
  affected-region MSE.

A high-severity blend is not always hardest for the model. Some large obvious
contaminants are severe but separable. Some low-severity blends are hard because
they overlap the target core or destroy important target structure.

## Experiment 1: Thayer-Direct

### Setup

- Dataset: Galaxy10 DECaLS from `data/Galaxy10_DECals.h5`.
- Split seed: 42.
- Task: direct reconstruction, `blended -> clean target`.
- Train/validation/test blends: 5,000 / 800 / 800.
- Epochs: 20.
- Batch size: 8.
- Model: compact U-Net.
- Rotation: disabled.
- Saved model checkpoint:
  `outputs/checkpoints/unet_direct_5000train_800val_800test_20ep.pth`.

### Earlier Normal Held-Out Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.005224 | 0.964264 | 0.062555 | 0.180238 | 1.00x |
| threshold | 0.029782 | 0.054440 | 0.067528 | 0.207066 | 0.93x |
| Thayer-Direct | 0.000566 | 0.976648 | 0.004428 | 0.041573 | 14.13x |

### Interpretation

Thayer-Direct clearly beats identity and threshold baselines on controlled
normal held-out blends. This experiment establishes that learned reconstruction
is useful in the controlled synthetic setting. The result should be reported as
an earlier 800-blend normal evaluation; it is not necessarily the exact same
generated normal set as the later 1,000-blend comparable table.

## Experiment 1b: Thayer-Direct Hard Stress Test

### Setup

- Run directory: `outputs/runs/stress_test_20260708_145221`.
- Model: Thayer-Direct saved model checkpoint from Experiment 1.
- Stress blends: 1,000.
- Source subset: first 800 images from the held-out test partition.
- Seed: 20260708.
- `max_shift`: 18.
- `brightness_range`: 0.8 to 1.4.
- `blur_range`: 0.0 to 0.15.
- `noise_range`: 0.0 to 0.006.
- Rotation: disabled.
- `min_size_ratio`: 0.75.
- `min_mask_fraction`: 0.01.
- Affected-region threshold: 0.02.

### Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.958915 | 0.075541 | 0.202969 | 1.00x | 0/1000 |
| threshold | 0.031344 | 0.045878 | 0.082746 | 0.229405 | 0.91x | 990/1000 |
| Thayer-Direct | 0.000937 | 0.972558 | 0.009390 | 0.060231 | 8.04x | 13/1000 |

### Interpretation

Thayer-Direct still beats identity on the hard stress distribution, but
affected-region improvement drops from about `14.13x` in the earlier normal
evaluation to `8.04x`. This is the expected direction under a harder synthetic
distribution with more concentrated overlap, brighter contaminants, and
similar-size sources.

## Experiment 2: Thayer-Residual

### Setup

- Run directory: `outputs/runs/residual_unet_20260708_154947`.
- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Train/validation/held-out blends: 5,000 / 800 / 800.
- Epochs: 20.
- Batch size: 8.
- Model: compact U-Net with linear residual output head.
- Saved model checkpoint:
  `outputs/checkpoints/unet_residual_5000train_800val_800test_20ep_20260708_154947.pth`.

### Normal Held-Out Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.005224 | 0.964264 | 0.062555 | 0.180238 | 1.00x | 0/800 |
| threshold | 0.029782 | 0.054440 | 0.067528 | 0.207066 | 0.93x | 0/800 |
| Thayer-Direct | 0.000566 | 0.976648 | 0.004428 | 0.041573 | 14.13x | 5/800 |
| Thayer-Residual | 0.000390 | 0.981015 | 0.004039 | 0.045027 | 15.49x | 1/800 |

Thayer-Residual beats Thayer-Direct on `310/800` normal held-out cases.
Thayer-Direct has slightly better affected-region MAE on the normal aggregate,
so residual prediction is not universally better.

### Hard Stress-Test Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.958915 | 0.075541 | 0.202969 | 1.00x | 0/1000 |
| threshold | 0.031344 | 0.045878 | 0.082746 | 0.229405 | 0.91x | 990/1000 |
| Thayer-Direct | 0.000937 | 0.972558 | 0.009390 | 0.060231 | 8.04x | 13/1000 |
| Thayer-Residual | 0.000700 | 0.977410 | 0.007069 | 0.058334 | 10.69x | 0/1000 |

Thayer-Residual beats Thayer-Direct on `667/1000` stress cases and reduces
worse-than-identity stress failures from `13/1000` to `0/1000`.

### Interpretation

Residual prediction improves aggregate affected-region MSE on both the earlier
normal held-out evaluation and the hard stress set. The improvement is clearest
under stress, where predicting what to subtract avoids some of the direct
model's worst regressions.

## Experiment 3: Thayer-BR v0.1

### Setup

- Run directory: `outputs/runs/balanced_residual_20260708_184632`.
- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Model name: Thayer-BR v0.1 (Balanced Residual U-Net).
- Train/validation blends: 8,000 / 1,000.
- Normal held-out/stress test blends: 1,000 / 1,000.
- Epochs: 20.
- Batch size: 8.
- Composition target: 50% normal/random, 30% high-overlap/core-obstruction,
  20% brightness/size stress.
- Saved best model checkpoint:
  `outputs/checkpoints/unet_residual_balanced_hard_20260708_184632.pth`.
- Saved final model checkpoint:
  `outputs/checkpoints/unet_residual_balanced_hard_final_20260708_184632.pth`.
- Best validation loss: `0.000378` at epoch 18.
- Final train/validation loss: `0.000336 / 0.000383`.

Actual accepted training components:

| Component | Requested | Accepted | Notes |
| --- | ---: | ---: | --- |
| Normal/random | 4,000 | 4,000 | Default controlled blend sampling |
| High-overlap/core-obstruction | 2,400 | 2,400 | Mean core obstruction `0.928`; relaxed candidates used when needed |
| Brightness/size stress | 1,600 | 1,600 | Mean brightness `1.278`; mean size ratio `1.057` |

### Current 1,000-Blend Normal Held-Out Metrics

| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.006168 | 0.020058 | 23.710 | 0.962503 | 0.068122 | 0.188809 | 1.00x | 0/1000 |
| threshold | 0.031229 | 0.147082 | 15.211 | 0.052671 | 0.073101 | 0.215194 | 0.93x | 926/1000 |
| Thayer-Direct | 0.000564 | 0.012206 | 33.494 | 0.976820 | 0.004236 | 0.040871 | 16.08x | 5/1000 |
| Thayer-Residual | 0.000452 | 0.008192 | 34.568 | 0.980729 | 0.004431 | 0.046584 | 15.37x | 3/1000 |
| Thayer-BR v0.1 | 0.000248 | 0.006708 | 37.245 | 0.983863 | 0.002451 | 0.033412 | 27.79x | 1/1000 |

Thayer-BR v0.1 beats Thayer-Residual on `91.3%` of normal cases and
beats Thayer-Direct on `76.1%` of normal cases.

### Hard Stress-Test Metrics

| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.021927 | 22.495 | 0.958915 | 0.075541 | 0.202969 | 1.00x | 0/1000 |
| threshold | 0.031344 | 0.148270 | 15.153 | 0.045878 | 0.082746 | 0.229405 | 0.91x | 990/1000 |
| Thayer-Direct | 0.000937 | 0.012772 | 30.917 | 0.972558 | 0.009390 | 0.060231 | 8.04x | 13/1000 |
| Thayer-Residual | 0.000700 | 0.008341 | 32.389 | 0.977410 | 0.007069 | 0.058334 | 10.69x | 0/1000 |
| Thayer-BR v0.1 | 0.000440 | 0.006731 | 34.417 | 0.981289 | 0.004587 | 0.045293 | 16.47x | 0/1000 |

Thayer-BR v0.1 beats Thayer-Residual on `87.9%` of stress cases and
beats Thayer-Direct on `93.1%` of stress cases.

### Interpretation

Thayer-BR v0.1 improves stress robustness relative to Thayer-Residual and does
not degrade normal aggregate affected-region MSE in the current 1,000-blend
comparable evaluation. This suggests that training distribution matters:
targeted hard-case sampling is more useful than simply adding more random
blends.

The result remains a controlled synthetic result. Thayer-Direct and
Thayer-Residual can still win on individual samples, so the paper should report
both aggregate improvements and remaining per-sample tradeoffs.

## Historical Interpretation at That Stage

- Thayer-Direct proves that learned models beat simple baselines
  in this controlled benchmark.
- Hard stress testing reveals a robustness drop for Thayer-Direct.
- Thayer-Residual improves stress robustness by learning contaminant signal
  to subtract.
- Thayer-BR v0.1 improved both normal and stress aggregate affected-region MSE
  in the prior same-run evaluation.
- Thayer-BR v0.2 Moderate improves further with affected/core-weighted residual
  loss and is now the current best controlled synthetic checkpoint.
- Remaining failures involve ambiguity, target-core overlap, lost target detail,
  foreground-extraction limitations, and model-specific tradeoffs.

## Historical Next Steps at That Stage

- Preserve exact generated evaluation sets and global source indices for future
  reproducibility.
- Finalize paper figures and captions.
- Write the LaTeX report.
- Preserve the Thayer-BR v0.2 Moderate checkpoint as the current best model and
  treat the stronger weighted variant as an ablation.
- Consider hybrid direct/residual models only if more modeling is needed.
- Improve preprocessing, foreground extraction diagnostics, sky/noise realism,
  and core-obstruction-balanced evaluation.


## Experiment 4: Thayer-BR v0.2 Weighted Residual Loss

Status: completed. Run directory: `outputs/runs/weighted_residual_20260709_030245`.

### Training Settings

- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Variant: `moderate`.
- Train/validation blends: 12,000 / 1,000.
- Normal held-out/stress test blends: 1,000 / 1,000.
- Epochs: 20.
- Batch size: 8.
- Composition target: 50% normal/random, 30% high-overlap/core-obstruction, 20% brightness/size stress.
- Actual training composition: [{"component": "train_normal", "requested": 6000, "accepted": 6000}, {"component": "train_high_overlap_core", "requested": 3600, "accepted": 3600, "attempts": 4300, "mean_mask_fraction": 0.09456119113498264, "mean_core_obstruction_fraction": 0.9302031100882925, "mean_size_ratio": 1.0653481550951234, "mean_brightness": 1.103194680445825, "relaxed_candidates_used": 683}, {"component": "train_brightness_size", "requested": 2400, "accepted": 2400, "attempts": 2662, "mean_mask_fraction": 0.09421144485473633, "mean_core_obstruction_fraction": 0.7093364132426144, "mean_size_ratio": 1.057729401183628, "mean_brightness": 1.2758697390398546, "relaxed_candidates_used": 256}].
- Loss formula: `sum(weight * (predicted_residual - true_residual)^2) / sum(weight)`, normalized over channels.
- Loss weights: background `1.0`, affected extra `3.0`, core affected extra `2.0`.
- Affected threshold: `0.02`.
- Core mask: central aperture `0.18` with bright-core fraction `0.55`.
- Saved best model checkpoint: `outputs/checkpoints/unet_residual_weighted_br_20260709_030245_best.pth`.
- Saved final model checkpoint: `outputs/checkpoints/unet_residual_weighted_br_20260709_030245_final.pth`.
- Best validation loss: 0.001040 at epoch 17.
- Final train/validation loss: 0.000915 / 0.001086.

### Normal Held-Out Metrics

| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Core affected MSE | Non-core affected MSE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.006168 | 0.020058 | 23.710 | 0.962503 | 0.068122 | 0.029349 | 0.069706 | 1.00x | 0/1000 |
| threshold | 0.031229 | 0.147082 | 15.211 | 0.052671 | 0.073101 | 0.040219 | 0.074698 | 0.93x | 926/1000 |
| Thayer-Direct | 0.000564 | 0.012206 | 33.494 | 0.976820 | 0.004236 | 0.018294 | 0.002730 | 16.08x | 5/1000 |
| Thayer-Residual | 0.000452 | 0.008192 | 34.568 | 0.980729 | 0.004431 | 0.008187 | 0.003964 | 15.37x | 3/1000 |
| Thayer-BR v0.1 | 0.000248 | 0.006708 | 37.245 | 0.983863 | 0.002451 | 0.007002 | 0.001969 | 27.79x | 1/1000 |
| Thayer-BR v0.2 Moderate | 0.000230 | 0.007213 | 37.392 | 0.983676 | 0.002108 | 0.004361 | 0.001837 | 32.31x | 0/1000 |

### Hard Stress-Test Metrics

| Method | Whole MSE | Whole MAE | PSNR | SSIM | Affected MSE | Core affected MSE | Non-core affected MSE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.021927 | 22.495 | 0.958915 | 0.075541 | 0.085131 | 0.072979 | 1.00x | 0/1000 |
| threshold | 0.031344 | 0.148270 | 15.153 | 0.045878 | 0.082746 | 0.090047 | 0.080026 | 0.91x | 990/1000 |
| Thayer-Direct | 0.000937 | 0.012772 | 30.917 | 0.972558 | 0.009390 | 0.036538 | 0.003649 | 8.04x | 13/1000 |
| Thayer-Residual | 0.000700 | 0.008341 | 32.389 | 0.977410 | 0.007069 | 0.015913 | 0.005217 | 10.69x | 0/1000 |
| Thayer-BR v0.1 | 0.000440 | 0.006731 | 34.417 | 0.981289 | 0.004587 | 0.013848 | 0.002810 | 16.47x | 0/1000 |
| Thayer-BR v0.2 Moderate | 0.000386 | 0.007295 | 34.961 | 0.981315 | 0.003847 | 0.009533 | 0.002785 | 19.64x | 0/1000 |

### Core and Model-Win Summary

- Stress Thayer-BR v0.1 affected MSE: 0.004587.
- Stress Thayer-BR v0.2 Moderate affected MSE: 0.003847.
- Stress Thayer-BR v0.1 core affected MSE: 0.013848.
- Stress Thayer-BR v0.2 Moderate core affected MSE: 0.009533.
- Stress worse-than-identity cases: v0.1 `0`, v0.2 `0`.
- Model win rates: `[{'split': 'normal', 'n': 1000, 'weighted_vs_balanced_win_rate': 0.726, 'weighted_vs_direct_win_rate': 0.836, 'weighted_vs_old_residual_win_rate': 0.948, 'weighted_to_balanced_aggregate_mse_ratio': 0.8600717023402843, 'weighted_to_direct_aggregate_mse_ratio': 0.49766017631485787, 'weighted_to_old_residual_aggregate_mse_ratio': 0.47579030565864255, 'weighted_worse_than_identity_count': 0, 'balanced_worse_than_identity_count': 1}, {'split': 'stress', 'n': 1000, 'weighted_vs_balanced_win_rate': 0.708, 'weighted_vs_direct_win_rate': 0.921, 'weighted_vs_old_residual_win_rate': 0.935, 'weighted_to_balanced_aggregate_mse_ratio': 0.8385196758185709, 'weighted_to_direct_aggregate_mse_ratio': 0.40964755164923333, 'weighted_to_old_residual_aggregate_mse_ratio': 0.5441293977659549, 'weighted_worse_than_identity_count': 0, 'balanced_worse_than_identity_count': 0}]`.
- Evaluation-seed identity/model affected-MSE ratio: normal
  `32.02 +/- 1.21x`; stress `19.55 +/- 0.30x`. Models were not independently
  retrained.

### Strong-Weight Variant Check

Because the moderate weighted model clearly improved over Thayer-BR v0.1, a
second stronger-weight variant was trained in
`outputs/runs/weighted_residual_20260709_043745` with affected extra weight `5`
and core affected extra weight `4`.

The stronger variant also beat Thayer-BR v0.1, but it did not beat the moderate
variant on aggregate affected-region MSE:

| Variant | Normal affected MSE | Normal core MSE | Stress affected MSE | Stress core MSE | Stress non-core MSE | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thayer-BR v0.1 | 0.002451 | 0.007002 | 0.004587 | 0.013848 | 0.002810 | 1 normal / 0 stress |
| Thayer-BR v0.2 Moderate, 3/2 extra weights | 0.002108 | 0.004361 | 0.003847 | 0.009533 | 0.002785 | 0 normal / 0 stress |
| Thayer-BR v0.2 Strong, 5/4 extra weights | 0.002306 | 0.004412 | 0.004030 | 0.009344 | 0.003017 | 0 normal / 0 stress |

Strong weighting slightly improved stress core affected MSE relative to the
moderate variant, but worsened normal aggregate, stress aggregate, and stress
non-core affected MSE. Thayer-BR v0.2 Moderate is therefore the better overall
weighted v0.2 model.

Combined variant comparison artifacts:

- `outputs/runs/weighted_residual_20260709_030245/tables/weighted_variant_comparison.csv`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/weighted_variant_comparison.png`

### Interpretation

Verdict: `strong_new_best` for the moderate weighted variant. Thayer-BR v0.2 Moderate
should become the current best model; the stronger weighted variant is
a useful ablation showing that more core weighting is not better overall.

Coherence/suspicion status: None.

Paper figures:

- `outputs/runs/weighted_residual_20260709_030245/paper_figures/affected_region_mse_bar.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/normal_vs_stress_improvement_ratio.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/core_affected_mse_comparison.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/noncore_affected_mse_comparison.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/weighted_vs_br_v01_per_sample_scatter.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/hist_weighted_to_br_v01_affected_mse_ratio.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/worse_than_identity_count_chart.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/stress_performance_by_core_overlap_bin.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/stress_performance_by_blend_severity_bin.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/qualitative_weighted_improves_over_br_v01.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/qualitative_br_v01_beats_weighted.png`
- `outputs/runs/weighted_residual_20260709_030245/paper_figures/qualitative_weighted_remaining_failure.png`

## Size/Visual Audit: Apparent Size, Halo Bands, and Visual Disagreement

Status: completed. Run directory:
`outputs/runs/size_visual_audit_20260709_102251`.

This was an evaluation-only audit. It did not train, retrain, or modify
checkpoints.

### Apparent-Size Findings

The audit estimated apparent target and contaminant sizes with border
background subtraction, central-source foreground masks, foreground area,
equivalent radius, bounding boxes, flux proxies, and centroid estimates.

The apparent contaminant/target equivalent-radius ratio varied substantially:

- p5 approximately `0.49`.
- median approximately `1.06`.
- p95 approximately `2.37`.

Affected-error dependence on apparent size ratio was weak for the learned
models in this audit:

- Thayer-Direct: `-0.165`.
- Thayer-Residual: `0.052`.
- Thayer-BR v0.1: `-0.143`.
- Thayer-BR v0.2 Moderate: `-0.123`.

Interpretation: the result is not obviously explained by a simple apparent-size
shortcut, but a future size-normalized benchmark is still recommended.

### Centrality and Core Obstruction

Centrality/core obstruction remains important, but the stress set is skewed
toward near-center high-overlap examples. Thayer-BR v0.2 Moderate performs
best in high core-obstruction cases, while small low/medium-obstruction bins
show mixed behavior and should not be over-interpreted.

### Halo-Band and Visual Findings

Aggregate halo-band MSE improved for Thayer-BR v0.2 Moderate relative to
Thayer-BR v0.1:

| Evaluation | BR v0.1 halo MSE | BR v0.2 Moderate halo MSE |
| --- | ---: | ---: |
| Normal | 0.000300 | 0.000250 |
| Stress | 0.000359 | 0.000320 |

However, selected individual v0.2 Moderate outputs show broad low-level or
halo-like artifacts. The audit saved visual-vs-metric disagreement candidates,
including v0.2 broad-error cases, v0.1 wins, direct-looking-cleaner cases,
strong v0.2 successes, and ambiguous targets.

### Historical Stopping Point at That Stage

Thayer-BR v0.2 Moderate is the current best model for the controlled synthetic
benchmark. Thayer-BR v0.1 remains a historical ablation showing the value of
balanced hard-case training, and Thayer-BR v0.2 Strong remains a weighting
ablation showing that stronger affected/core weights are not better overall.

The next modeling-related benchmark should be evaluation-only at first:
size-normalized held-out blends with current checkpoints, before any
size-normalized retraining.

## Experiment 5: Thayer-BR v0.3 Color/Structure Candidate

Status: completed as an ablation/tradeoff. Run directory:
`outputs/runs/br_v03_delta_color_20260709_185630`.

### Setup

- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Architecture: unchanged compact residual U-Net from Thayer-BR v0.2.
- Train/validation blends: 8,000 / 1,000.
- Epochs: 20.
- Batch size: 8.
- Realized training distribution: 40% normal clean, 25% high-overlap/core
  obstruction, 20% compact bright contaminants, 10% brightness/size stress,
  5% low-overlap/easy stabilizer, and no artifact bucket because no safe
  source-quality artifact flags were available. The intended 5% artifact
  bucket was redistributed to normal clean blends.
- Loss formula: `residual_mse + 0.5*recon_l1 + affected_core_loss +
  0.10*gradient_loss + 0.05*color_proxy_loss + 0.05*halo_band_loss`.
- Color loss: differentiable RGB chroma plus color-direction proxy.
- Color metrics: Lab metrics and Delta E 2000 through scikit-image for
  evaluation only.
- Saved best checkpoint:
  `outputs/checkpoints/unet_br_v03_delta_color_20260709_185630_best.pth`.
- Saved final checkpoint:
  `outputs/checkpoints/unet_br_v03_delta_color_20260709_185630_final.pth`.
- Best validation loss: `0.006891` at epoch 20.
- Final train/validation loss: `0.006542 / 0.006891`.

### Main Comparison to Thayer-BR v0.2 Moderate

| Suite | v0.2 affected MSE | v0.3 affected MSE | v0.3/v0.2 | v0.2 Delta E 2000 | v0.3 Delta E 2000 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Normal | 0.002025 | 0.002590 | 1.28 | 6.415 | 6.623 | worse primary MSE and color |
| Hard stress | 0.003648 | 0.004772 | 1.31 | 7.670 | 8.768 | worse primary MSE and color |
| Compact bright | 0.006514 | 0.006325 | 0.97 | 9.447 | 10.005 | slight compact-MSE gain, worse color |
| High core obstruction | 0.004312 | 0.005443 | 1.26 | 8.135 | 9.443 | worse core-stress result |
| Halo band | 0.003730 | 0.004745 | 1.27 | 7.861 | 8.660 | worse affected MSE |
| Color saturation | 0.005227 | 0.006341 | 1.21 | 9.769 | 10.550 | worse affected MSE and color |

v0.3 improved a few secondary slices: compact-bright affected MSE was slightly
lower than v0.2 Moderate, compact-bright gradient error improved, normal
halo-band MSE was marginally lower, and color-saturation chroma/gradient
metrics improved slightly. These gains did not hold on the primary normal and
hard-stress affected/core metrics.

### Interpretation

Verdict: `visual_tradeoff`. Thayer-BR v0.3 Color/Structure Candidate should
not replace Thayer-BR v0.2 Moderate as the current best model. It is useful as
an ablation showing that low-weight color/edge auxiliaries can help targeted
compact or chroma/edge slices, but they weakened aggregate affected-region MSE
and Delta E 2000 on the main normal and stress suites.

Delta E 2000 is a perceptual visual metric under a standard RGB/sRGB-like
assumption. Galaxy10 DECaLS RGB cutouts are survey composite images, not
guaranteed true human-color photographs, so Delta E is visual-quality evidence,
not the primary scientific metric. Affected-region MSE remains the primary
metric.

Generated artifacts:

- `outputs/runs/br_v03_delta_color_20260709_185630/tables/v03_color_suite_metrics.csv`
- `outputs/runs/br_v03_delta_color_20260709_185630/tables/v03_color_per_sample_metrics.csv`
- `outputs/runs/br_v03_delta_color_20260709_185630/tables/v03_color_comparison_summary.csv`
- `outputs/runs/br_v03_delta_color_20260709_185630/tables/v03_color_metric_summary.csv`
- `outputs/runs/br_v03_delta_color_20260709_185630/diagnostics/color_metric_implementation.md`
- `outputs/runs/br_v03_delta_color_20260709_185630/diagnostics/v03_color_structure_report.md`

Checkpoint integrity checks before and after the run confirmed that the 10
pre-existing comparison checkpoints were unchanged.

### Delta Follow-up (Ablation/Tradeoff)

The stronger-color follow-up, Thayer-BR v0.3 Delta Candidate, completed in
`outputs/runs/br_v03_delta_candidate_20260710_031425`. It used the same
architecture and realized 40/25/20/10/5 training distribution as the first
v0.3 candidate, increased
the differentiable color-proxy weight from `0.05` to `0.10`, and retained the
`0.5` reconstruction-L1, `1.0` affected/core, `0.10` gradient, and `0.05`
halo-band terms. Training ran for 20 epochs on 8,000/1,000 train/validation
blends at batch size 8. The best epoch was 20 with validation loss `0.007056`;
final train/validation loss was `0.006866 / 0.007056`.

- Best checkpoint:
  `outputs/checkpoints/unet_br_v03_delta_candidate_20260710_031425_best.pth`.
- Final checkpoint:
  `outputs/checkpoints/unet_br_v03_delta_candidate_20260710_031425_final.pth`.

Same-run comparison with Thayer-BR v0.2 Moderate:

| Suite | v0.2 affected MSE | Delta affected MSE | Delta/v0.2 | Delta E 2000, v0.2 -> Delta | Delta win rate | Worse than identity, Delta/v0.2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.002025 | 0.002275 | 1.123 | 6.415 -> 6.148 | 35.3% | 0/0 |
| Hard stress | 0.003648 | 0.004035 | 1.106 | 7.670 -> 7.951 | 36.0% | 1/0 |
| Compact bright | 0.006514 | 0.005428 | 0.833 | 9.447 -> 8.921 | 67.2% | 0/0 |
| High core obstruction | 0.004312 | 0.004665 | 1.082 | 8.135 -> 8.523 | 36.5% | 0/1 |
| Halo band | 0.003730 | 0.004048 | 1.085 | 7.861 -> 7.928 | 41.6% | 0/0 |
| Color saturation | 0.005227 | 0.005528 | 1.057 | 9.769 -> 9.842 | 45.7% | 0/0 |

Delta improved on the first v0.3 Color/Structure candidate in aggregate
affected MSE and Delta E 2000 on all six shared suites. Relative to v0.2
Moderate, however, the primary normal and stress affected MSE worsened by
about 12.3% and 10.6%, and stress added one worse-than-identity case. The
targeted gains were real: compact-bright affected MSE improved by about 16.7%,
Lab chroma error improved on all six suites, and halo-band MSE improved on all
six suites. Delta E improved on normal and compact-bright blends but worsened
slightly on the other four aggregate suites. The qualitative bank contains
both a muted-color improvement and an explicit Delta color-artifact example,
so the visual evidence also supports a tradeoff rather than universal progress.

Verdict: `visual_tradeoff`. Delta is a useful compact/color/halo ablation, not
the current best model. Thayer-BR v0.2 Moderate remains current best. The run
did not fabricate clean-source or artifact-heavy results: validated source
quality flags are unavailable. The separate non-training plan in
`docs/clean_benchmark_plan.md` specifies how those suites should be constructed
later. Integrity checks confirmed that all 12 checkpoints present before the
Delta run were unchanged.

## Experiment 6: Thayer-ResUNet v0.4 Candidate

Status: completed as an architecture ablation. Run directory:
`outputs/runs/resunet_v04_candidate_20260710_043109`.

### Setup and Training

- Task: predict `residual = blended - target`; reconstruct with
  `blended - predicted_residual`.
- Architecture: U-Net encoder/decoder and skip connections with residual
  two-convolution blocks at each scale.
- Trainable parameters: `2,014,595`, versus `1,927,075` for the standard v0.2
  U-Net, an increase of `4.54%`.
- Loss: the proven v0.2 Moderate normalized weighted residual MSE, with
  background/affected-extra/core-extra weights `1/3/2` and no color or halo
  auxiliary loss.
- Training distribution: 50% normal, 30% high-overlap/core obstruction, and
  20% brightness/size stress.
- Train/validation blends: 8,000 / 1,000; epochs: 20; batch size: 8.
- Best validation loss: `0.001076` at epoch 19.
- Final train/validation loss: `0.000792 / 0.001082`; final validation affected
  MSE: `0.003086`.
- Best checkpoint:
  `outputs/checkpoints/unet_resunet_v04_candidate_20260710_043109_best.pth`.
- Final checkpoint:
  `outputs/checkpoints/unet_resunet_v04_candidate_20260710_043109_final.pth`.

### Same-Run Comparison to Thayer-BR v0.2 Moderate

| Suite | v0.2 affected MSE | ResUNet affected MSE | ResUNet/v0.2 | ResUNet win rate | Worse than identity, ResUNet/v0.2 | Core-MSE ratio | Halo-MSE ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.002132 | 0.002118 | 0.993 | 55.2% | 0/0 | 0.965 | 0.703 |
| Hard stress | 0.003929 | 0.003950 | 1.005 | 51.7% | 1/0 | 1.027 | 0.873 |
| Compact bright | 0.006840 | 0.005427 | 0.793 | 68.1% | 0/0 | 0.965 | 0.850 |
| High core obstruction | 0.004243 | 0.004193 | 0.988 | 50.3% | 0/0 | 1.055 | 0.931 |
| Halo band | 0.003869 | 0.003757 | 0.971 | 53.3% | 1/1 | 1.000 | 0.773 |
| Color saturation | 0.005098 | 0.004999 | 0.981 | 55.5% | 0/0 | 1.113 | 0.848 |

The architecture has useful targeted behavior. Compact-bright affected MSE
improved by about 20.7%, and halo-band MSE improved on every evaluated suite,
including about 22.7% on the halo-band stress suite. Although no color loss was
used, affected Delta E 2000 also improved on all six suites. The aggregate
normal affected MSE was effectively tied with v0.2 and slightly lower.

The strict gate nevertheless failed. Hard-stress affected MSE was about 0.5%
worse, hard-stress core MSE was about 2.7% worse, and one stress sample became
worse than identity where v0.2 had none. On the high-core suite, aggregate
affected MSE improved slightly but core affected MSE worsened by about 5.5%; on
the color-saturation suite core affected MSE worsened by about 11.3%. Saved
qualitative grids include a ResUNet improvement, a failure/tradeoff, and a case
where Thayer-Direct still wins, so the aggregate gains are not universal.

### Interpretation and Stopping Decision

Verdict: `architecture_ablation`. The compact-contaminant, halo, and color
results make ResUNet scientifically useful as a targeted tradeoff, but it does
not clearly beat Thayer-BR v0.2 Moderate under the required stress/core and
worse-than-identity criteria. Thayer-BR v0.2 Moderate therefore remains current
best.

The conditional Part B2 loss tuning was not run. Core+ and Halo-safe were
authorized only if the baseline ResUNet clearly beat v0.2 Moderate; because it
failed that gate, additional training would not have been a controlled
follow-up under the preregistered decision rule. No Core+ or Halo-safe result is
claimed. Clean-source filtering was also unavailable for this run and remains
part of the separate clean benchmark plan.

Generated result tables and diagnostics:

- `outputs/runs/resunet_v04_candidate_20260710_043109/tables/resunet_v04_suite_metrics.csv`
- `outputs/runs/resunet_v04_candidate_20260710_043109/tables/resunet_v04_per_sample_metrics.csv`
- `outputs/runs/resunet_v04_candidate_20260710_043109/tables/resunet_v04_comparison_summary.csv`
- `outputs/runs/resunet_v04_candidate_20260710_043109/diagnostics/resunet_v04_candidate_report.md`

Checkpoint integrity verification confirmed that all 14 checkpoints present
before the ResUNet experiment were unchanged.

## Benchmark-Defensibility Audit Pass

Status: completed after the Delta and baseline ResUNet runs. No Core+, Halo-safe,
or additional architecture training was launched. All full model inference used
MPS; CPU work was limited to hashing, manifest generation, CSV aggregation,
plots, and documentation.

### Locked-Manifest Preparation

Run: `outputs/runs/final_test_manifest_prep_20260710_061737`.

- Reserved 1,000 unique coordinate groups from the historical test tail after
  excluding groups present in train, validation, or the first 1,000 development
  test sources.
- Created normal, hard-stress, compact-bright, high-core-obstruction, and
  halo/artifact-proxy manifests with 1,000 metadata-only rows each.
- Stored global source indices, blend parameters, seeds, masks, severity,
  generator hashes, sample fingerprints, schemas, and checksums; no raw arrays.
- All replay/schema/checksum tests pass, and manifest files are read-only.
- No model was run and no qualitative sample was inspected for model selection.
- Status remains `provisional_locked_manifest_prep` and `paper_ready = false`.

**Superseding status:** this pool is no longer final-eligible. The later grouped
resplit maps its 1,000 sources to 683 train, 173 validation, and 144 test, and
the grouped manifests actually use 499 in training plus 91 in validation. The
files remain preserved as historical infrastructure; a fresh untouched final
pool is required after the model and protocol are frozen.

The earlier random-index draft at
`outputs/runs/final_test_manifest_prep_20260710_060845` is preserved and marked
blocked. A later conservative-exclusion setup was stopped before generation
after independent leakage review cleared the proposed perceptual candidates.

### Source-Leakage Audit

Run: `outputs/runs/source_leakage_audit_20260710_062950`.

- Row-index partitions are disjoint: 12,415 train / 2,660 validation / 2,661
  test, with zero pairwise intersections.
- Twenty-three auditable artifacts contain 21,060 indexed blend rows with zero
  target/contaminant role-containment failures.
- Raw-pixel hashing finds 60 duplicate groups / 62 exact pairs overall and 28
  groups / 29 pairs crossing train/validation/test.
- RA/Dec finds 59 duplicate-coordinate groups and 27 cross-split pairs.
- No local no-duplicate Galaxy10 file was found; RA/Dec, redshift, and pixel
  scale exist, but no object-ID field is available.
- The provisional final-tail pool has no sustained exact, coordinate, or
  reviewed perceptual links to train, validation, or the development prefix.

Verdict at that point: major blocker for the original random-index protocol.
Historical model metrics remained development evidence, not a leakage-cleared
final claim. The grouped protocol described below subsequently resolved this
training gate, but not the need for an untouched final partition.

### Unblended Preservation and Unaffected Regions

Corrected run: `outputs/runs/preservation_null_tests_20260710_063312`.

| Model | Mean unblended-input MSE | SSIM | p99 MSE | MSE > 0.001 |
| --- | ---: | ---: | ---: | ---: |
| Thayer-BR v0.2 Moderate | 0.00002646 | 0.998626 | 0.00026892 | 3/1,000 |
| Thayer-BR v0.3 Delta | 0.00000120 | 0.999933 | 0.00000735 | 0/1,000 |
| Thayer-ResUNet v0.4 | 0.00002144 | 0.998909 | 0.00006481 | 0/1,000 |

The Delta preservation ablation preserves unblended inputs best but remains
worse on the primary blended affected-region metric. ResUNet remains the
architecture ablation documented above. The v0.2 tail shows false subtraction
of bright off-center sources and target structure. Heuristic artifact
candidates account for 23/1,000 null inputs and show elevated null error for all
three models.

The corrected unaffected-region table reports target error, model output change
versus the blend, and paired excess target error over identity. This avoids
calling the affected-mask complement pure model damage when it still contains
sub-threshold blend changes, blur, or noise.

### Clipping Audit

Corrected run: `outputs/runs/clipping_audit_20260710_063312`.

- Aggregate whole-image MSE changes by at most 0.96% after clipping.
- Aggregate affected-region MSE changes by at most 0.16%; rankings do not
  change.
- Ten of 6,000 paired model/sample rows have absolute affected-MSE clipping
  gains above `0.0001`; none exceeds a 10% relative gain.
- Per-sample pixel statistics retain low/high clipping fractions, conditional
  excursion magnitudes, and magnitude-qualified residual signs.

### Source-Artifact Audit

Run: `outputs/runs/source_artifact_audit_20260710_061059`.

The streaming heuristic audit flags 356/17,736 sources (2.01%) for manual
review: 178 saturation, 104 color-streak, 89 large-edge-mask, 52 edge-touching,
10 axis-line, and 3 blank flags, with overlapping categories. Contact sheets
show both clear artifacts and legitimate-morphology false positives. No source
was removed and no flag is treated as validated ground truth.

All 16 checkpoints remained unchanged throughout the completed audit runs.
Thayer-BR v0.2 Moderate remains the current best development-benchmark model;
no new model is promoted.

## Research Correctness Audit and Grouped v0.2 Retrain

Master run: `outputs/runs/research_correctness_audit_20260710_092241/`.

### Infrastructure, blending, and metrics

- Confirmed 29 pixel-identical and 27 exact-coordinate pairs crossing the
  historical row split, implicating 57/17,736 sources (`0.321%`).
- Historical evaluation implication was 13/1,000 normal and 12/1,000 stress;
  excluding implicated rows changed improvement ratios by at most about
  `0.31%`. Measured aggregate severity is minor, while protocol severity is
  major.
- Passed 29/29 deterministic metric checks, including independent arithmetic,
  prediction-independent masks, empty-mask coverage, clipping separation,
  color-range checks, and sample-ID alignment.
- Replayed 150/150 stratified generator checks and later all 13,000 grouped
  manifest rows exactly. A first precision-mismatch attempt is preserved rather
  than hidden; round-trip CSV float parsing fixed replay.
- Confirmed the model receives only blended RGB. The residual represents a
  blend-to-target correction field, not necessarily pure contaminant flux.
- Documented display-RGB rather than calibrated-flux compositing, target
  centrality, input clipping, padded-mask size compression, ignored pixel-scale
  mismatch, and repeated-source statistical dependence.

### Grouped source and blend infrastructure

Source split:
`data/manifests/grouped_source_split_20260710_100907/`.

- 12,417 train, 2,660 validation, and 2,659 development-test sources.
- Exact-pixel and exact-coordinate groups remain wholly inside one partition.
- Zero cross-split source, group, exact-pixel, or exact-coordinate overlap.
- Near-duplicate identity is not exhaustively proven.

Blend manifests: `data/manifests/grouped_blends_20260710_103233/`.

- 8,000 train, 1,000 validation, and four 1,000-row development-test suites.
- Both source roles stay inside their assigned partition; 71/71 integrity
  checks and 13,000/13,000 exact replays pass.
- These are grouped development manifests, not a locked final benchmark.

### Existing-checkpoint diagnostic

The historical v0.2 checkpoint remained strong on grouped tests, but 54.575%
of rows exposed an old training/validation source group after repartitioning.
On the 45.425% clean-neither subset it still achieved `31.53x` normal,
`18.18x` hard, `11.68x` compact-bright, and `18.27x` high-core affected-MSE
ratios. This is plausibility evidence, not a source-independent result.

### Thayer-BR v0.2 Moderate Grouped Retrain

Run: `outputs/runs/br_v02_moderate_grouped_retrain_20260710_110917/`.

- MPS, seed 3042, batch size 8, 20 epochs, 8,000/1,000 grouped train/validation
  blends, historical v0.2 U-Net, affected/core extra weights 3/2.
- Best epoch 20: train loss `0.0010825181`, validation loss `0.0011635236`,
  validation affected MSE `0.0033365143`.
- Best checkpoint SHA-256:
  `eea442ff21bdfbdd74815d7b292e786f187dc9a63fea73d4adde98a4b082802b`.

| Grouped development suite | Affected MSE | Identity/model ratio | Core affected MSE | Halo MSE | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.00231890 | 28.8127x | 0.00497364 | 0.000435626 | 0/1000 |
| Hard stress | 0.00458983 | 15.8025x | 0.0115079 | 0.000640123 | 3/1000 |
| Compact bright | 0.00872771 | 9.18304x | 0.0118618 | 0.000778985 | 2/1000 |
| High core obstruction | 0.00491680 | 15.8378x | 0.0123239 | 0.000548833 | 1/1000 |

The grouped retrain remains strong but is worse than the historical checkpoint
on the identical suites. The comparison confounds split repair with training
budget: the historical model used 12,000 blends and the requested grouped run
used 8,000. One seed does not establish training-seed robustness.

No optional second seed was launched because the earlier provisional final pool
was demonstrably reused by grouped train/validation. Final-test independence is
the next infrastructure blocker. All 16 pre-existing checkpoints remained
unchanged; the grouped best/final checkpoints are separate timestamped files.

## 2026-07-11 — Frozen-representation recoverability ablation

Run: `outputs/runs/thayer_select_frozen_head_ablation_20260711_220756/`.

- Frozen Phase-II R1 pooled-bottleneck features: 13,500 x 64 across training,
  validation, and calibration; MPS encoder-only extraction.
- Moderate positives: 41/10,000 training, 5/1,500 validation, 30/2,000
  calibration.
- Validation AUROC/AUPRC: H0 0.985/0.265, H1 0.983/0.516, H2 0.984/0.548,
  H3 0.986/0.532, H4 0.989/0.561.
- H2-H1 AUPRC difference: +0.033, paired 95% CI [-0.015, +0.184].
- H4-H2 AUPRC difference: +0.012, paired 95% CI [-0.249, +0.275].
- Validation-selected H2 calibration raw AUROC/AUPRC: 0.514/0.032.
- H2 ambiguity gap remained +0.073; catastrophic rejection AUROC 0.654; null
  hallucination rejection AUROC 0.948.
- H2 isotonic calibration produced four values and an 87.6% largest plateau;
  temperature avoided zero thresholds but still collapsed nominal 95%, 90%,
  and 80% coverage to 100% realized coverage.
- Oracle validation AUROC/AUPRC: 0.795/0.023; analysis-only.
- Decision: **NO CLEAR IMPROVEMENT**. The original automatic nonlinear gate was
  preserved and superseded because it ignored paired-CI materiality and
  calibration stability.
- Zero development and lockbox access; zero reconstruction inference; all
  historical checkpoints unchanged.

## 2026-07-11 — Hierarchical recoverability-policy campaign

Run: `outputs/runs/thayer_select_hierarchical_safety_20260711_225657/`.

- Pre-training drift audit: zero label mismatches across 40,500 contract-row
  checks; maximum physical-covariate |SMD| `0.0535`; source-reuse |SMD|
  `0.2762`; validation had 5 moderate and 37 permissive actionable positives.
- Fresh non-development data: 15,000 Q-train, 2,000 Q-validation, 15,000
  R-train, 2,000 R-validation, 6,000 natural calibration, and 3,000 stratified
  diagnostic calibration scenes. Eighteen replay probes passed.
- Frozen Condition-C features: 64 global + 112 multiscale prompt-local + 18
  reconstruction-summary values; exact repeated MPS extraction; zero trainable
  reconstruction parameters.
- Query gate: F_COMBINED small-MLP five-seed ensemble; validation macro
  F1/AUPRC `0.8811`/`0.9230`; NULL recall `0.9985`; AMBIGUOUS recall `0.8889`;
  inversion removed in all seeds.
- Valid-risk selections: image/flux used F_COMBINED small MLPs; centroid used
  F_RECON_SUMMARY small MLP. Five-seed upper-risk Spearman means were `0.734`,
  `0.816`, and `0.956`. Confusion AUROC/AUPRC means were `0.859`/`0.217` at
  2.3% prevalence.
- Split-conformal natural coverage: image `0.9000`, flux `0.9002`, centroid
  `0.9002`; stratified diagnostic coverage `0.897`, `0.901`, `0.907`.
- Frozen policy calibration was degenerate: 1/4,200 natural valid accepted and
  0/1,000 stratified valid accepted.
- Fresh development: 3,000 scenes, manifest SHA-256
  `9ccb1626dcc158f43951ee15e03b6c00c3bcb01fc31e396a7d32e980d4ce51aa`,
  read-only and evaluated once. Query gate accepted 66.65% valid, 0% NULL, and
  9.2% AMBIGUOUS; full policy accepted one valid and no invalid scenes.
- Diagnostic hierarchical catastrophic rates at 95/90/80/70% valid coverage:
  `0.8253`/`0.8156`/`0.7931`/`0.7643`, not materially better than R1.
- Decision: **FAILURE** for the complete campaign; successful query-validity
  and tail-ranking subcomponents do not compensate for unusable coverage.
- Condition C and every historical checkpoint remained unchanged; development
  was not retuned; the lockbox remained untouched.
