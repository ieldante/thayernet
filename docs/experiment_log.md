# Experiment Log

This log records the main formal experiments and the development work that made
them interpretable. Raw outputs, generated run directories, and saved model
checkpoint files remain under `outputs/` and are not committed.

## Development Phase: Pipeline and Evaluation Setup

- Data loading uses the local Galaxy10 DECaLS HDF5 file.
- Original images are split into train, validation, and test subsets before
  blending, so the same source image cannot appear in both training and
  evaluation examples.
- Naive whole-cutout addition was replaced with foreground-only contaminant
  extraction to avoid rectangular cutout/background artifacts.
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

## Current Interpretation

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

## Next Steps

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
- Weighted multi-seed improvement: normal 32.02 +/- 1.21x; stress 19.55 +/- 0.30x.

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

### Current Stopping Point

Thayer-BR v0.2 Moderate is the current best model for the controlled synthetic
benchmark. Thayer-BR v0.1 remains a historical ablation showing the value of
balanced hard-case training, and Thayer-BR v0.2 Strong remains a weighting
ablation showing that stronger affected/core weights are not better overall.

The next modeling-related benchmark should be evaluation-only at first:
size-normalized held-out blends with current checkpoints, before any
size-normalized retraining.
