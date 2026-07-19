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

## 2026-07-12 — Hierarchical protocol corrective audit

Run: `outputs/runs/thayer_select_hierarchical_safety_20260712_001405/`.

- Part A provenance passed on MPS availability, Condition-C hash, source-split
  identity, empty staged index, and lockbox exclusion metadata.
- Reconstructed `moderate_actionable_success` for all 13,500 persisted Phase-II
  rows: zero actionable-label and zero underlying-contract mismatches.
- Positive counts remained 41 training, 5 validation, and 30 calibration.
- Confirmed scientific heterogeneity: 2,543 successful NULL outcomes and all
  1,350 AMBIGUOUS rows are actionable negatives; mild and catastrophic valid
  failures share the same negative class.
- Confirmed provenance mismatch: training/validation outcomes use frozen
  Condition C, calibration outcomes use Phase-II R1.
- The historical run lacked the required pre-fit preregistration, original-
  contract truth-table/postmortem artifacts, and explicit frozen-reconstructor
  report. Retrospective preregistration was refused.
- Decision: stop before new inference/training. No development reevaluation,
  policy change, checkpoint change, or lockbox access occurred.

## 2026-07-12 — Prospective hierarchical-safety feasibility

Run: `outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/`.

- Preregistration SHA-256: `f2184c169c9161e920988d32b217e56b78bb4688a65a6a0023944f9e73dec9d2`,
  hashed before fitting.
- Fresh scenes: 12,000 Q-train, 2,000 Q-validation, 12,000 UNIQUE_VALID
  risk-train, 2,000 risk-validation, 4,000 natural calibration; no development
  or lockbox scenes.
- Uniform Condition-C checkpoint SHA-256:
  `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`.
- Query five-seed macro F1 `0.872 ± 0.010`; NULL/AMBIGUOUS recalls
  `1.000`/`0.877`; inversion removed in every seed.
- Image/flux/centroid validation Spearman `0.860`/`0.867`/`0.949`, transferring
  to natural calibration at `0.870`/`0.858`/`0.954`.
- Confusion validation/calibration AUROC `0.866`/`0.844`.
- Catastrophic validation AUROC/AUPRC `0.987`/`0.997`, but formal FAIL because
  the frozen `1.25 × prevalence` AUPRC gate equals `1.0206`.
- Query vector-scaling ECE `0.0266`; continuous marginal coverage ~`0.900`;
  image/flux subgroup coverage minimum `0.691`, so calibration PARTIAL.
- Decision: **PARTIAL SUCCESS**. Superseding correctness audit PASS; two
  bookkeeping incidents are preserved with append-only corrections.
- No operational policy, development evaluation, lockbox access, checkpoint
  change, stage, commit, or push.

## 2026-07-12 — Prospective conditional-calibration correction

- Run: `outputs/runs/thayer_select_conditional_calibration_20260712_021556/`.
- Preregistration SHA-256:
  `95a67082acbe0921af7db64f3d78c5280d18f58442f71b0de16364228bc8494d`.
- Exact original-sag reproduction: image/flux minimum `0.691429`.
- All frozen subgroups supported; no extra calibration scenes generated.
- Selected marginal coverage: image `0.9029`, flux `0.8982`, centroid `0.9007`.
- Selected worst supported coverage: image `0.6373`, flux `0.6839`, centroid
  `0.8882`.
- Natural-calibration Spearman: image `0.8700`, flux `0.8617`, centroid
  `0.9520`.
- Catastrophic sanity: validation AUROC/AUPRC `0.9872`/`0.9971`; attainable
  AUPRC gate `0.9541`, PASS.
- Decision: image FAIL, flux FAIL, centroid PASS, overall **FAILURE**.
- Condition C stayed frozen; zero reconstruction inference, development
  access, lockbox access, policy construction, staging, or commits.

## 2026-07-12 — Partially pooled scale correction

- Run: `outputs/runs/thayer_select_scale_correction_20260712_024957/`.
- Preregistration SHA-256:
  `4d2d6701e3cfe0847a0b88bb5ae04ca8f3ef514ce8747e3b123670bb57c80d96`.
- Exact baseline reproduction: image `0.9029`/`0.6373`, flux
  `0.8982`/`0.6839`, centroid `0.9007`/`0.8882` marginal/worst coverage.
- Five connected-source-component folds produced held-out-only risk
  predictions with zero source overlap and zero calibration target leakage.
- Selected O0 Huber log-residual scale objective for image and flux.
- Partially pooled marginal/worst coverage: image `0.9189`/`0.5492`, flux
  `0.9218`/`0.6788`.
- Median-width inflation: image `1.336x`, flux `1.055x`; ranking remained
  `0.877`/`0.866`.
- Worst-subgroup bootstrap 95% lower bounds: image `0.477`, flux `0.614`.
- Non-deployable oracle worst coverage: image `0.914`, flux `0.905`; diagnostic
  only and marginally overcovering.
- Decision: image FAIL, flux FAIL, centroid PASS, overall **FAILURE**.
- One integrity-only continuation incident is preserved: Pandas initially
  treated literal `NULL` as NA, then a generic helper used logit rather than
  documented probability ensembling. Both were corrected without refitting or
  changing calibration results.
- Zero reconstruction inference, development/lockbox access, policy
  construction, historical overwrite, staging, commit, or push.

## 2026-07-12 — Shape-constrained quantile scale correction

- Authoritative run:
  `outputs/runs/thayer_select_shape_constrained_quantile_20260712_033406/`.
- Preregistration SHA-256:
  `93c047c6cacd3db51340860ed1d0f5e086b78b4b4fd0277bc4ed88cc083c575c`.
- Training-only OOF proxy endpoint checks reproduced within `1e-10`; global
  monotonicity was rejected.
- Q1/Q2 satisfied every convexity and upper-half monotonicity constraint; Q2's
  corrected positive interaction was bounded and nonnegative.
- Validation selected Q1 for image and flux. Q2 improved worst supported
  validation-cell coverage by `0.000` for both risks.
- Selected marginal/worst calibration coverage: image `0.9221`/`0.5440`, flux
  `0.9221`/`0.5907`.
- Median-width inflation: image `1.723x`, flux `1.303x`; p95 widths `3.012`
  and `10.163`.
- Worst-subgroup bootstrap 95% lower bounds: image `0.473`, flux `0.522`.
- Decision: image FAIL, flux FAIL, centroid PASS, overall **FAILURE**.
- Integrity audit PASS: 18 tests, zero constraint violations, exact post-fit
  centering, 456 historical checkpoints unchanged, and zero
  neural/development/lockbox access.
- A prior restart at `..._032938` is superseded: its centering constant was
  initialized before optimization rather than recomputed from the learned
  basis. No result from that run is authoritative. The defect was corrected,
  regression-tested, and the full campaign restarted without overwriting it.

## 2026-07-12 — Observable-regime distillation

- Run:
  `outputs/runs/thayer_select_observability_distillation_20260712_035843/`.
- Preregistration SHA-256:
  `8ec5b644fba2658f32eeac43edea0e5f8d4e3301a6b108cf3d1f932722dddbff`.
- All conditional, oracle, partially pooled, and corrected Q1/Q2 baselines
  reproduced at `1e-10` before fitting.
- A0 exact four-proxy validation AUROC: `0.7113`.
- Selected A3 five-seed validation AUROC: `0.9014 ± 0.0035`; normalized AP
  lift `0.3725`; source-component bootstrap AUROC lower bound `0.8827`.
- Validation/calibration SNR Spearman: `0.883`/`0.889`; obstruction Spearman:
  `0.456`/`0.479`.
- Joint-hard validation/calibration AUROC: `0.906`/`0.880`; AUPRC:
  `0.430`/`0.325`.
- Frozen gate failures: recall at precision 0.70 `0.0835 < 0.30`;
  calibration Brier `0.1397 > 0.0642`; calibration ECE `0.2191 > 0.15`.
- Decision: **OBSERVATIONAL INFORMATION LIMIT — FAILURE**. GroupDRO, new
  quantile fitting, and predicted/multigroup calibration were not run.
- Condition C unchanged; 536 historical checkpoints unchanged; 22 relevant
  tests passed; development/lockbox accesses zero.
- Two pre-fit append-only incidents are preserved: calibration access ordering
  was corrected before execution, and object-dtype scene IDs required a
  fit-only loader continuation after feature extraction but before fitting.

## 2026-07-12 — Explicit-PSF provenance and variation audit

- Authoritative run:
  `outputs/runs/thayer_select_psf_conditioning_20260712_043442/`.
- Audited 12,000 training, 2,000 validation, and 4,000 natural-calibration
  scenes; 54,000 scene-band provenance rows.
- Exact renderer: BTK default SurveyCodex LSST PSF, implemented as an
  axisymmetric GalSim Kolmogorov-plus-Airy convolution at 0.2 arcsec/pixel.
- Fixed g/r/z FWHM: `0.86`/`0.81`/`0.77` arcsec.
- Unique combined scene configurations: `1`; effective count: `1.0`;
  within-band scene variation: zero up to floating-point aggregation roundoff.
- Sampled native-grid PSF replay passed for 27 scene-band checks across all
  partitions.
- Decision: **PSF NON-INFORMATIVE BY CONSTRUCTION**.
- The campaign stopped before preregistration, fitting, controls, risk or
  calibration continuation, development, or lockbox access.
- Exactly one next experiment: prospectively generate scenes with realistic
  varying PSFs.
- Three incomplete attempts are preserved and superseded; all stopped before
  preregistration or fitting, and none altered historical artifacts.

## 2026-07-12 — Competing-hypothesis recoverability feasibility

- Authoritative run:
  `outputs/runs/thayer_competing_hypotheses_20260712_131111/`.
- Preregistration SHA-256:
  `692b4194da0486b8240fcda8227d36df9b1654187dd5c670d60c69b8c5fd5a4b`.
- Compatible inventory: Condition C, R0, and reconstruction-only R1 map to the
  common source-layer contract but share one architecture cluster; cross-family
  auditing stopped before training.
- Route-B Atlas pool: 30,000 scenes from approved training/validation groups;
  100 numerical pairs; first 25 frozen after exact replay and five-page visual
  artifact audit.
- Forward score: calibrated on exactly 2,000 calibration scenes; global 99th-
  percentile threshold 1.031580046990072.
- Empirical witnesses: 49/50 noisy observations for constructed truth
  decompositions; 18/50 for same-cluster model candidates.
- Atlas deblender behavior: all 75 pair/model rows contained at least one unsafe
  noisy requested reconstruction; Condition C returned nearly the same
  mean-scene answer on 25/25 divergent pairs, R0 on 16/25, and R1 on 1/25.
- Correctness: 180 tests and 23 subtests passed across the isolated main/BTK
  environments; 556 historical checkpoints, source split, and catalog were
  unchanged; development and lockbox access counts were zero.
- Decision: **PARTIAL SUCCESS**. No auditor, catalog policy, or cross-family
  claim was produced.

## 2026-07-12 — Ambiguity Atlas v0 and competing-hypothesis recoverability

- Run: `outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/`.
- Preregistration SHA-256:
  `2b54bf035f5a51721b4d012faa84060bb926a81610463fcb393c16d5f3f39185`.
- Fresh scenes: 30,000 training/search, 2,000 validation, and 3,000 calibration;
  development and lockbox access counts remained zero.
- Route 1: 100/100 numerical candidates passed; 25 were visually reviewed and
  frozen. Route 2: 25/25 final bounded optimization pairs passed; 600 trials
  are preserved.
- Constructed witnesses: 50/50. Same-cluster model-candidate witnesses: 19/50.
- Operational baseline: diameter AUROC 0.4712 and zero recall at 4% control
  false positives; R1 unsafe-confidence AUROC 0.9176.
- Decision: **ATLAS PASS; AMBIGUITY-WITNESS DETECTOR FAIL; AUDITOR BLOCKED**.

## 2026-07-12 — Prompted ResUNet candidate-diversity feasibility

- Authoritative run:
  `outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/`.
- Preregistration SHA-256:
  `d412f2071e49bf53ccf4633021d2ced8f43ffe32a160b537542be3ab10798884`.
- One earlier run stopped before rendering or fitting after an internal decoder
  channel/parameter-count inconsistency was caught; its artifacts are preserved.
- Final architecture: 199,219 trainable parameters, six residual blocks,
  fresh initialization, no Condition-C weight import.
- Source isolation: all 59 Atlas/targeted-feasibility groups excluded; 10,000
  training and 1,500 validation scenes; 11,500/11,500 full replays passed.
- MPS training: 20/20 epochs; best epoch 18; best normalized validation MSE
  `0.03862`; no CPU fallback.
- Pre-Atlas result: prompt-swap `0.3947 < 0.80`; individual requested identity
  `0.695 < 0.75`; output collapse `0.00067`; whole-image MSE ratio to Condition C
  `1.1205`.
- Decision: **PROMPTABILITY FAILURE — ATLAS EVALUATION NOT AUTHORIZED**.
- Atlas, development, and lockbox access counts: 0/0/0. No auditor or catalog
  policy was trained.

## 2026-07-12 — Thayer-PU prompted probabilistic U-Net

- Run: `outputs/runs/thayer_probabilistic_unet_20260712_163340/`.
- Preregistration SHA-256:
  `eb62db24da7c77f35f56d1187f561f88a2e63e2acd89c01c859c1fd2213b2b09`.
- Canonical per-sample hash audit: 11/11 pass.
- Source isolation: 59 Atlas-related groups excluded; 24,000 collision-pool
  scenes and 20,000 final scenes; 20,000/20,000 final replays passed.
- Model: 170,278 parameters, latent dimension 8, Condition-C warm start,
  truth-free prior, training-only posterior, six-channel full decomposition.
- Training: 30/30 MPS epochs; best epoch 27; no fallback.
- Non-Atlas gates: latent use, promptability, prior/posterior gap, forward
  consistency, and selective control concentration all passed.
- Promptability: majority-of-16 swaps 0.9875; individual prior identity 0.99384;
  best-of-16 identity 0.99425; collapse 0.00106.
- Atlas one-pass: witnesses 24/50; AUROC 0.856, bootstrap interval 0.751–0.942;
  recall at 4% control false positives 0.32; safe-control witnesses 0.08;
  own/alternate truth coverage 0/0.
- Decision: **PARTIAL SUCCESS**. Atlas evaluation count 1; development and
  lockbox access 0/0; no post-Atlas tuning or auditor training.

## 2026-07-12 — Thayer-PF posterior/decoder sufficiency gate

- Run: `outputs/runs/thayer_flow_prior_20260712_182516/`.
- Persisted Thayer-PU and Atlas metrics reproduced without new Atlas inference.
- Frozen truth-coverage metric synthetic audit: pass.
- Part D: K=32; 256 ordinary scenes; all 250 validation near-collision pairs;
  MPS inference with fallback prohibited.
- Own-truth coverage: ordinary 0%; near-collision 0%.
- Cross-decoded paired alternate coverage: 0%; alternate identity 1.76%.
- Forward-consistent sample fractions: 0.930 ordinary, 1.000 near-own, and
  1.000 near-cross.
- Decision: **FAILURE — DECODER/POSTERIOR INSUFFICIENT; FLOW PROHIBITED**.
- Flow fitting / Atlas / development / lockbox access counts: 0/0/0/0.

## 2026-07-12 — Thayer-MH ambiguity-set decoder

- Run: `outputs/runs/thayer_multiple_hypotheses_20260712_190701/`.
- Expanded Atlas-related exclusion: 36,288 groups; development and lockbox
  commitments remained zero.
- Targets: 12,000/3,000 training, 1,500/500 validation, and 1,500/500
  calibration ordinary/ambiguous observations; 2,000 validated pairs.
- Architecture: shared K=2 token decoder, 120,022 parameters; 30 MPS epochs;
  best validation epoch 27; no fallback.
- Promptability: token-0/1 and set-level prompt swap all 0.992; requested MSE
  ratio to Condition C 0.864.
- Coverage: ordinary own 0%; near own 0%; near alternate 0%; both-mode 0%.
- Forward-consistent fractions: ordinary 0.933; near-collision 1.000.
- Decision: **FAILURE — NON-ATLAS SET COVERAGE FAILED; ATLAS PROHIBITED**.
- Atlas / development / lockbox access counts: 0/0/0.

## 2026-07-12 — Thayer-ME two-expert capacity gate

- Run: `outputs/runs/thayer_two_expert_decoder_20260712_203121/`.
- Thayer-MH promptability, reconstruction, coverage, forward consistency, and
  zero Atlas inference reproduced exactly from persisted artifacts.
- Exact Thayer-MH scenes and targets reused; all 2,000 pair gates and hashes
  passed without regeneration.
- Architecture: 72,672 shared encoder parameters plus two disjoint 46,470-
  parameter expert decoders; 165,612 total.
- Microset: 32 ordinary and 32 ambiguous training-only observations from 16
  pairs; 400 MPS epochs; no fallback.
- Expert-1/expert-2/set prompt swap: 0.969/0.969/0.953.
- Ordinary/ambiguous forward consistency: 0.969/1.000.
- Ordinary, own, alternate, and both-mode truth coverage: 0/0/0/0; median
  ordinary expert diameter 5.166.
- Decision: **REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE**.
- Full fit / Atlas / development / lockbox access counts: 0/0/0/0.

## 2026-07-12 — Thayer-LG frozen loss-geometry audit

- Created and preregistered
  `outputs/runs/thayer_loss_geometry_20260712_205733/` before per-scene loss
  inspection.
- Reproduced all persisted Thayer-ME micro-overfit gates and frozen input
  hashes without model inference.
- Passed exact-truth output-contract, prompt mapping, forward plausibility,
  own/alternate/both-mode coverage, and ordinary concentration sanity checks.
- Found truth objective-optimal on 10/64 rows; a compromise beat truth on
  54/64 rows and all 32 ambiguous rows.
- Found forward loss dominant at truth, frequent source/forward gradient
  conflict, collapsed-mean assignment ties, and immediate coverage loss along
  a lower-objective truth-to-trained path.
- Detached full-objective optimization from truth lowered loss and destroyed
  coverage. No model weights, Atlas, development rows, or lockbox rows were
  touched.

## 2026-07-12 — Thayer-SA scientific-alignment correction

- Created `outputs/runs/thayer_scientific_alignment_20260712_220315/` and
  reproduced the Thayer-LG diagnosis before freezing the objective and gates.
- Froze preregistration SHA-256
  `6ef3bc2505a5677e3acade93a818566105c26368f532dca40b60121f839ddc26`.
- Surrogate Spearman/Kendall/threshold agreement: 0.990679/0.957683/1.000000;
  exact truth value and gradient were zero.
- Exact truth remained at full coverage. Final own/alternate/both-mode coverage
  was 0/0/0 from trained output, 0.281/0.281/0.250 from collapsed mean, and
  0.438/0.438/0.312 from wrong source allocation.
- Decision: **FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED**. Assignment and
  neural stages were not reached; no checkpoint, Atlas, development, or
  lockbox access occurred.

## 2026-07-12 — Thayer-OC output-space conditioning

- Created `outputs/runs/thayer_output_conditioning_20260712_225459/`; froze
  preregistration SHA-256
  `4202c5ddc9b9733138168b2acc650334e1ef10b002f7799071a3a12bc827e484`
  before per-scene loads or detached optimization.
- Reproduced 16/16 Thayer-SA and Thayer-ME baselines and passed 6/6 coordinate
  round-trip/projection cases.
- Compared raw Adam, raw L-BFGS, T/D Adam, T/D L-BFGS, alternating T/D, and
  threshold-Jacobian-preconditioned T/D under frozen budgets.
- No global method passed. Best ordinary coverage was 0.438; best ambiguous
  own/alternate/both-mode coverages were 0.844/0.875/0.812, from different
  method/initialization combinations.
- C2, C4, and C5 moved exact truth outside full coverage and were ineligible.
- Scientific decision: **PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY**.
- Strict correctness: **FAIL**; the actual-objective HVP/finite-difference
  condition estimate was unresolved. Neural training and Atlas/development/
  lockbox access remained zero; 593/593 checkpoints were unchanged.

## 2026-07-12 — Thayer-FP direct scientific-feasibility projection

- Froze and hashed the 64-row projection/micro-learning protocol before every
  per-scene load or optimization.
- Reproduced all authoritative Thayer-ME, Thayer-SA, and Thayer-OC baselines;
  left the prior HVP status unresolved without a new curvature claim.
- Projected all 256 expert/prompt pairings with P0. Final projected target-set
  coverage and forward consistency were 100% in every required category.
- Median P0 alpha was 0.999979 and median normalized correction was 0.946369;
  flux-z was most often limiting.
- P1 reduced correction but missed the strict 0.95 target interior on three
  pairings, so P0 was frozen globally.
- Trained unchanged Thayer-ME for 400 MPS-only epochs with direct
  requested/companion loss. All four truth-coverage rates remained zero;
  ordinary diameter was 3.564 and output nonnegativity failed.
- Decision: **FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT
  MEMORIZE THEM**. No Atlas, development, or lockbox access occurred.

## 2026-07-13 — Thayer-CL output-contract preflight

- Created `outputs/runs/thayer_capacity_ladder_20260713_005215/` and froze
  preregistration SHA-256
  `b3d77b7726f5f117c1fa70946730cc213d1db3015798c9bcfa058b3ceb03ed23`
  before all per-scene loads, model construction, and optimization.
- Reproduced 24/24 Thayer-FP checks, including complete P0 feasibility, median
  alpha/correction, 173/256 z-flux limiting entries, and the diagnostic neural
  trajectory.
- Traced negatives to the unconstrained linear head. Positive inverse-
  normalization scales preserved them in physical source layers.
- Found zero contract-selected compliant mappings and three distinct
  admissible but unfrozen replacements. Stopped before L0-L3 construction,
  synthetic fitting, or neural training.
- Decision: **FAIL-CLOSED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**.
  Strict correctness is **FAIL** because the initial checkpoint inventory was
  incomplete before per-scene loading; a complete closure audit verified
  594/594 historical checkpoints unchanged. Atlas, development, and lockbox
  access counts remained zero.

## 2026-07-13 — Thayer-CL strict metadata-correct rerun

- Preserved the first append-only `20260713_005215` attempt and created
  `outputs/runs/thayer_capacity_ladder_20260713_013132/` to correct only its
  incomplete pre-load checkpoint inventory.
- Froze preregistration SHA-256
  `d44778017b45a1a21109b19cd5e623c76b0a2353128c10a5ad8a7edbfb27820c`
  after inventorying all 594 historical checkpoints and before any per-scene
  tensor load, model construction, or optimizer step.
- Reproduced all 24 Thayer-FP checks and the same physical-negative provenance:
  the raw linear head emitted negative values that positive inverse-
  normalization scales preserved in the metric-facing physical source layers.
- Found zero frozen eligible mappings and three distinct unfrozen admissible
  mappings. The campaign again stopped at Part D before L0-L3 construction,
  preflight fitting, or neural training.
- Decision: **FAIL-CLOSED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING**.
  Strict correctness is **PASS** with 27/27 checks; all 594 historical
  checkpoints remained byte-identical. Atlas, development, and lockbox access
  counts remained zero.

## 2026-07-13 — Thayer-OP fixed-L0 output parameterization

- Created `outputs/runs/thayer_output_parameterization_20260713_023120/` and
  froze preregistration SHA-256
  `c6abcb8ba70888bc9a14477968933713c0729a4e32065f7f2becfcec9c468597`
  before per-scene loading and fitting.
- Held the Condition-C encoder, two 46,470-parameter L0 expert decoders, P0
  targets, hard assignment, initialization policy, optimizer, order, and step
  budget fixed; only ReLU, square, or absolute value changed in forward.
- All three mappings passed full target representability, gradient/numerical
  preflight, five stop-rule self-tests, and five synthetic MPS fits per mapping.
  Physical negative and nonfinite event counts remained zero.
- Each mapping received 3,200 MPS optimizer steps on the same ordinary scene
  and 3,200 on the same ambiguous scene. Every mapping finished with zero
  ordinary coverage and zero ambiguous both-mode coverage.
- Decision: **NO MAPPING PASSES**. Stopped before eight-scene fitting, selected
  no mapping, and did not authorize the decoder-capacity ladder. Atlas,
  development, and lockbox access counts remained zero.

## 2026-07-13 — Thayer-RI repository-integrity and fixed-feature audit

- Started from an empty staged index and preserved the existing working tree.
- Installed a strict exact-path Python access guard and closed a ten-module
  local execution graph. All 391 high-risk static occurrences were classified.
- Independent production/reference comparisons passed 13 groups with zero
  mismatches. Seven corrected nonnegative golden cases and seven differential
  truth-injection variants passed.
- Exact lineage selected training scene row 12000 and P0 row 32 only.
  Ordinary, eight-scene, remaining-microset, Atlas, development, and lockbox
  access counts were zero.
- No result-changing production defect was found and no production source was
  edited. Both experts and both final heads received gradients and updates;
  the encoder hash remained unchanged.
- D0: square passed all three coverage gates; ReLU and absolute value failed.
- D1: square passed all three coverage gates with frozen rank-six heads.
- D2: square's 204-parameter final-head-only readout reduced target loss by
  about 28.5% but retained zero own, alternate, and both-mode coverage.
- D3 and tangent diagnostics were not authorized by the frozen progression
  rule. Primary outcome: **FROZEN-FEATURE CONDITIONING BARRIER**.
- All five executed condition/mapping pairs received exactly 5,000 MPS steps.
  All 600 inventoried historical checkpoints matched before and after.

## 2026-07-13 — Thayer-D3 square full-L0 fixed-feature diagnostic

- Created the append-only Thayer-D3 run and froze preregistration SHA-256
  `08fe5d9bf97ca98e0cb79b161082e62c6022f014f23cfba9cad389f4aca2deda`
  before every tensor load.
- Reproduced all 54 persisted evaluation rows for square D0, D1, and D2. D0
  and D1 retained 100% own/alternate/both coverage; D2 retained 0%/0%/0%.
- Matched the joined cache, one-scene payload, square initial-state artifact,
  P0 hashes, square checkpoint, and D0-D2 endpoints. Initial raw, mapped,
  physical, penultimate, assignment, and target-loss values were exact.
- Found that the D1 endpoint artifact lacks `penultimate_expert_1` and
  `penultimate_expert_2`. Preserved the output artifact without reconstructing
  or rerunning D1.
- Decision: **FROZEN-INPUT PROVENANCE FAILURE — D3 NOT RUN**. No optimizer,
  autograd trace, decoder update, tangent diagnostic, broader scene, Atlas,
  development, or lockbox access occurred.

## 2026-07-13 — Thayer-D1R square D1 endpoint replay

- Created `outputs/runs/thayer_d1_endpoint_replay_20260713_113715/` and froze
  preregistration SHA-256
  `0ebd166e1ea33b306d4dc78fe748dcd0240ada5d24dfd9d4d282ac763507faf2`
  before every scientific tensor load.
- Reproduced the authoritative 54-row D1 metadata, exact initial state, six
  cached feature hashes, four P0 hashes, rank-six frozen heads, and all 600
  historical checkpoint hashes.
- Optimized only two detached `[2,16,60,60]` paired-prompt tensors with the
  frozen AdamW `0.03`, clip `5.0`, 5,000-step MPS protocol. No neural parameter,
  encoder, or decoder body entered the optimizer.
- Matched all 54 physical trajectory hashes. The final objective was
  `3.1026115010490685e-09`; raw, mapped, and physical outputs were byte-identical
  and own/alternate/both-mode coverage was 100%/100%/100%.
- Persisted four named prompt/expert endpoint tensors plus output, assignment,
  optimizer-provenance, schema, and canonical-hash artifacts.
- A restricted fresh process reproduced the endpoint outputs and metrics. All
  13 batch/serialization checks passed with zero difference.
- Decision: **SUCCESS — D1 ENDPOINT PERSISTED AND REPLAYED**. D3 was not run.
  A separate square-only D3 campaign is authorized; broader scenes, Atlas,
  development, lockbox, eight-scene fitting, and capacity scaling remain closed.

## 2026-07-13 — Thayer-D3R authoritative full-L0 retry

- Created `outputs/runs/thayer_full_l0_d3r_20260713_121652/` and froze
  preregistration SHA-256
  `8a995ec98e162cab54e69d0efdcae6dd340a7c0f0082280548cb28ae28cacb72`
  before scientific tensor loading.
- Passed the exact-path guard self-test, all frozen input byte hashes, and all
  600 historical checkpoint hashes.
- Stopped before optimizer construction after a Matplotlib import dependency
  attempted prohibited cache deletion and PyTorch's tempfile probe could not
  complete under the no-delete contract.
- Decision: **EXECUTION-READINESS FAILURE — D3 NOT RUN**. No one-step trace,
  decoder update, trajectory, tangent diagnostic, broader scene, Atlas array,
  development, or lockbox access occurred. No retry was attempted.

## 2026-07-13 — Thayer-D3B runtime bootstrap readiness

- Created `outputs/runs/thayer_d3_runtime_readiness_20260713_125352/` and froze
  preregistration SHA-256
  `6c543ed014771b6a39d49d017a16533c962f7217b7eac1054d4b9b959ecc6b55`
  before all third-party imports.
- Reproduced the Matplotlib lock-file and PyTorch tempfile cleanup operations
  under bootstrap scratch with exact call stacks.
- Passed all guard self-tests and four final process modes: two cold, one warm
  cache reuse, and one fresh process after shutdown. Every process emitted
  `READY_FOR_SCIENTIFIC_TENSOR_LOAD`.
- Strict phases recorded zero deletion, cache write, blocked read, protected
  access, or Matplotlib import. Shutdown cleanup stayed inside scratch.
- The pure forward evaluator matched an independent reference in all ten
  synthetic cases with zero file I/O. All 21 D1R prerequisites, eleven named
  metadata containers, four scientific source hashes, and 600 historical
  checkpoint hashes matched.
- Decision: **READINESS PASS — D3 NOT RUN**. No scientific tensor, model,
  optimizer, decoder forward, JVP, VJP, Atlas, development, or lockbox access
  occurred. D3 remains scientifically unknown.

## 2026-07-13 — Thayer-D3B superseding exact-environment readiness

- Preserved the earlier `thayer_d3_runtime_readiness_20260713_125352` record
  but rejected its readiness conclusion because its preregistered shared
  runtime paths did not exactly match the per-process launch roots.
- Created `outputs/runs/thayer_d3_runtime_readiness_20260713_130859/` and froze
  preregistration SHA-256
  `6a2a067a273de952acfad473b67d72d761e52a583abb29ebb505b4792a48d4f8`
  before all third-party imports, including exact environment values for both
  cold processes, warm reuse, the post-shutdown process, and postprocessing.
- Passed the final guard self-test, two cold processes, warm-cache reuse, one
  fresh process after shutdown, and isolated metadata postprocessing. Every
  readiness process emitted exactly `READY_FOR_SCIENTIFIC_TENSOR_LOAD`.
- Strict phases recorded zero deletion attempts, cache writes, blocked events,
  protected access, or Matplotlib imports. Shutdown removed only generated
  runtime scratch after the readiness status was flushed.
- The pure evaluator matched the independent reference in all ten synthetic
  cases with zero file I/O. All 21 D1R prerequisites, eleven named metadata
  containers, four scientific source hashes, and 600 historical checkpoint
  hashes matched.
- Decision: **READINESS PASS — D3 NOT RUN**. No scientific tensor, model,
  optimizer, decoder forward, JVP, VJP, Atlas, development, or lockbox access
  occurred. D3 remains scientifically unknown.

## 2026-07-13 — Thayer-D3B clean authoritative closure

- Preserved `thayer_d3_runtime_readiness_20260713_130859` after its runtime
  passed but its independent closure audit failed: the bootstrap inventory did
  not record `CUBLAS_WORKSPACE_CONFIG`, so exact dictionary comparison failed.
- Added that already-frozen variable to the bootstrap inventory without
  changing its value or any scientific path, then created
  `outputs/runs/thayer_d3_runtime_readiness_20260713_131306/`.
- Froze preregistration SHA-256
  `7beeefeafa2496aa6171304ca0cf2656ffa2351adc5ee1cbd44370f38526e3f2`
  before all third-party imports. Every observed process environment exactly
  matched the preregistered map.
- Passed the final guard self-test, both cold processes, warm-cache reuse, one
  fresh process after shutdown, isolated metadata postprocessing, pure
  evaluator tests, 21 D1R prerequisites, eleven metadata containers, four
  source hashes, and all 600 historical checkpoint hashes.
- Decision: **READINESS PASS — D3 NOT RUN**. Strict phases had zero deletion,
  cache write, blocked read, protected access, or Matplotlib import. No
  scientific tensor, model, optimizer, decoder forward, JVP, VJP, Atlas,
  development, or lockbox access occurred. D3 remains scientifically unknown.

## 2026-07-13 — Thayer-D3B authoritative process-inventory closure

- Preserved `thayer_d3_runtime_readiness_20260713_134646` as non-authoritative.
  Its first closure passed, but it did not persist every required initial,
  bootstrap, strict-end, and post-shutdown inventory or independently validate
  the complete postprocessor lifecycle from its access log.
- Created
  `outputs/runs/thayer_d3_runtime_readiness_20260713_135017/` and froze
  preregistration SHA-256
  `c5272757ce125d6603b615005b33002bb7aee6703790788a2d725c0f7c1106f6`
  before all third-party imports.
- Passed 16/16 guard self-tests and the 26-check final closure. The primary,
  both cold, warm-cache, and shutdown-audited scientific processes emitted the
  exact readiness marker. Their strict phases had zero deletion, cache or
  bytecode write, blocked read, protected access, or Matplotlib import, and
  their frozen runtime inventories did not change.
- The separate Matplotlib/Agg postprocessor emitted its independent marker,
  allowed zero scientific reads, imported zero project modules, and confined
  both lifecycle operations to its own disposable runtime. The production
  forward evaluator matched the independent reference on all twelve synthetic
  cases with zero file I/O.
- All 21 D1R prerequisites, eleven named metadata containers, exact scientific
  code hashes, and all 600 historical checkpoints matched. No scientific
  tensor, model, optimizer, decoder forward, JVP, VJP, Atlas, development, or
  lockbox access occurred. Decision: **READINESS PASS — D3 NOT RUN**. One
  separately preregistered square-only one-scene D3 campaign is operationally
  permitted; D3 remains scientifically unknown.

## 2026-07-13 — Thayer-D3A preregistration completeness stop

- Created `outputs/runs/thayer_authoritative_d3_20260713_145040/` with a
  standard-library-only orchestrator and froze preregistration SHA-256
  `209aa4a2fdad6917536a010cb497eeb437c5b1a0994f948d279b60b63972e899`.
- Matched 27/27 runtime-readiness hashes, 11/11 scientific-container hashes,
  and 600/600 historical checkpoints without deserializing a tensor.
- Found that the isolated evidence did not persist the scientific sky vector
  and plausibility thresholds required by the pure forward evaluator and full
  truth-coverage gate. Synthetic readiness fixtures were not substituted and
  prohibited historical paths were not reopened.
- Decision: **PREREGISTRATION INCOMPLETE — D3 NOT RUN**. Third-party imports,
  tensor loads, models, optimizers, decoder forwards, D3 steps, postprocessing,
  and protected-data accesses were all zero.

## 2026-07-13 — Thayer-D3C scientific contract capsule

- Created the corrected authoritative record
  `outputs/runs/thayer_d3_scientific_capsule_20260713_155637/` and
  froze preregistration SHA-256
  `bb3ce7d68a42f88f52c54aa485448a929020782708aa251d38abb0f2ecdf819f`
  before extracting any sky or threshold value.
- Enumerated 97 required D3 scientific dependencies from the exact bootstrap,
  evaluator, reference evaluator, truth-coverage, assignment, mapping, runtime,
  artifact, and row-identity contracts. All 97 resolved with zero conflicts.
- Built capsule SHA-256
  `8a76ccdfa659a7291f0f9b73e0cb4d4c8adfb317b9902fc8ad5763e6d17b7d21`
  and schema SHA-256
  `42a974a7ef2b48a7108ef350d2d119c3955f3df325411784c9a22da9cf975f40`.
- Passed the strict schema, hash chain, all 16 corruption tests, all 12
  production/reference evaluator cases, zero evaluator I/O, and four
  cwd/environment process modes. The capsule-only launcher emitted both
  required authorization markers.
- Decision: **SCIENTIFIC CONTRACT CAPSULE PASS — D3 NOT RUN**. No scientific
  tensor, model, optimizer, decoder forward, gradient, or D3 step occurred.
- Preserved the earlier preclosure run
  `outputs/runs/thayer_d3_scientific_capsule_20260713_153815/` as
  non-authoritative after finding that its generic small-JSON guard did not
  recursively enforce selected-value rank inside mappings. The corrected run
  proves rank-1 acceptance, rank-2 rejection, and 65-scalar rejection.

## 2026-07-13 — Thayer-D3E executable contract

- Created authoritative append-only run
  `outputs/runs/thayer_d3_executable_contract_20260713_164320/` and froze
  preregistration SHA-256
  `b5a69f70c0f24f287da1f70a4a33e876fe9c8186be7c4e3c0eea67804bf1eede`
  before container-member inspection or model and optimizer construction.
- Reproduced the exact nine capsule-v1 requirements missing from the actual
  consumer, then built one canonical 180-entry registry with identical builder,
  validator, preflight, consumer, and runtime-access sets.
- Passed capsule-v2 schema and hash chain, every scientific container header,
  exact two-expert L0 construction and state loading, production-shape
  synthetic MPS forward, assignment/loss/evaluator references, one AdamW step,
  checkpoint reload and fresh-process replay, and all 25 consumer corruption
  tests.
- Built executable bundle SHA-256
  `884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`.
  The actual consumer and future launcher emitted the required authorization
  markers.
- Decision: **EXECUTABLE D3 CONTRACT PASS — SCIENTIFIC D3 NOT RUN**. Scientific
  array loads, scientific D3 steps, and Atlas, development, and lockbox access
  were zero. D3 remains scientifically unknown.

## 2026-07-13 — Thayer-D3S scientific attempt

- Froze no partial preregistration and loaded no scientific tensor.
- Matched executable-bundle SHA-256
  `884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`
  and confirmed the 180-entry registry.
- Found that the bundle lacks the required expert-activity/death gate,
  prompt-collapse stop, optional tangent protocol, complete primary-outcome
  mapping, and semantic-state rules.
- Decision: **EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN**. D3 and capacity
  remain unknown; broader data and the capacity ladder remain closed.

## 2026-07-13 — Thayer-D3P policy closure

- Created authoritative append-only run
  `outputs/runs/thayer_d3_policy_contract_20260713_173955/` and froze
  preregistration SHA-256
  `6edc2bbbfa1d98172dfbfdae6f28bf983099fab1200245f80e055590eab4543c`
  before policy definitions, fixtures, launcher changes, or bundle v3.
- Reproduced the exact five-family bundle-v2 regression and enumerated the
  complete 16-policy consumer surface.
- Passed 76 synthetic fixtures, all 106 launcher policy branches, 256/256
  outcome combinations, semantic-state collision and replay tests, exact
  six-set equality, and 30/30 bundle corruptions.
- Built bundle-v3 SHA-256
  `30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`.
  The actual launcher emitted all four readiness markers.
- Decision: **D3 POLICY CONTRACT PASS — SCIENTIFIC D3 NOT RUN**. Scientific
  tensor, model, optimizer, decoder-forward, and D3-step counts were zero;
  Atlas, development, and lockbox remained untouched.

## 2026-07-13 — Thayer-D3I

- Reproduced the historical launcher regression with six pre-fix failures.
- Added append-only bridge-v4 orchestration and worker/postprocess sources.
- Passed the actual synthetic full stack, fresh-process replay, exact flow
  closure, and 25/25 integration corruptions.
- Froze bridge SHA-256
  `3ab6e4a525297f48cc7fd9428651c604aa1236ed0a4425f9953c5b5772345dc5`.
- Mandatory science loaded eight allowlisted containers / 91 members and then
  stopped before model construction on `D3I-D1-MEMBER-CONTRACT`; two external
  bytecode-cache reads were also blocked.
- Outcome: `IMPLEMENTATION_OR_CONTRACT_FAILURE`; authorization: `none`.
  Historical checkpoints were 600/600 unchanged; protected access was zero.

## 2026-07-13 — Thayer-D3I41

- Reproduced the v4 dtype and lazy-serialization defects before correction.
- Candidate 001 stopped before bridge creation on a Correction-B helper
  reference and was preserved; candidate 002 passed 58 tests, 25 inherited and
  18 new corruptions, synthetic/replay/isolation/flow, and MPS gates.
- Scientific continuation loaded 8 containers / 91 members and passed all four
  corrected dtype contracts, then stopped before model construction on
  `V41_MEMBER_INVENTORY_HASH_DOMAIN_ERROR`.
- Outcome: `IMPLEMENTATION_OR_CONTRACT_FAILURE`; authorization: `none`;
  600/600 historical checkpoints and all frozen sources remained unchanged.

## 2026-07-13 — Thayer-D3I41R1

- Froze a 30-row independent ledger; exact required tests passed 19/19.
- Production-schema checkpoint probes passed in two cold and one warm process.
- Candidate 001 failed worker import and was superseded within scope.
- Candidate 002 stopped on an unrelated append-only log collision before worker
  launch; no scientific payload was loaded and authorization remains `none`.
## 2026-07-14 — Thayer-Audit v0

- Run: `outputs/runs/thayer_audit_v0_20260714_154655/`.
- Preregistration SHA-256:
  `3ca55b23997c8bfb0d6be2d395096020ab04df1d730f043d04a0b7c6d6a9f1c2`.
- Source-group-safe historical held-out fold: 7,055 PRE and 7,025 POST training
  episodes; zero base-fit or auditor-partition group overlap.
- Fixed MPS auditors: A1 28,307 parameters; A2 155,209 parameters; three seeds
  2026071501–2026071503; zero reconstruction training.
- PRE validation/calibration macro-F1 `0.8947/0.7980`, null recall
  `1.0000/0.9988`, ambiguous recall `0.9009/0.9100`.
- POST unsafe prevalence `1.0`; AUROC undefined; no threshold met the joint
  coverage/risk constraints. Fail-closed accepted coverage was `0.0`.
- Atlas/control abstention was `1.0/1.0` after policy freeze and was not used
  for success.
- Append-only outcome mapping correction: **DIRECT_AUDITOR_PARTIAL** because
  query detection was useful while coverage, POST ranking, and family gates
  failed. No prospective v1 authorization.
- D3 remained unchanged; truth-only features were supervision/evaluation only;
  development outcomes and final lockbox access counts remained zero.

## 2026-07-14 — Thayer-PU Eligibility v1

- Run: `outputs/runs/thayer_pu_eligibility_v1_20260714_213113/`.
- Preregistration SHA-256:
  `6f5cd5de57e7810aab947c9c59e955bb09215abb5d32251dff663cbf753d578c`.
- Froze epoch-27 Thayer-PU checkpoint SHA-256 `c1d17a…557e`, K=16 seeds
  2026077600–2026077615, truth-free mean aggregation, and MPS batch size 8.
- Built 3,998/793/2,800 source-group-disjoint fit/selection-excluded source
  manifests without retraining.
- Prompt identity passed at 1.0 majority and 1.0 individual success. Repeated
  batch-8 and batch-4/batch-8 hashes were exact.
- All 24 scenes failed exact single-scene/batched candidate and deployed hashes;
  full inference stopped. Outcome: `THAYER_PU_DEPLOYMENT_INELIGIBLE`.
- Condition C replayed 12,493 unsafe / 0 safe with zero substantive mismatch.
  No auditor, safety labels for Thayer-PU, Atlas selection, development, or
  lockbox evaluation occurred.

## 2026-07-14 — Thayer-Audit Family-E v0

- Run: `outputs/runs/thayer_family_e_v0_20260714_195256/`.
- Froze preregistration SHA-256
  `256bffe3bc53b572b7596bba844f0afdbf4abf3c4cb1d8906fc0ad08663d8881`
  before model construction or tensor loading.
- Froze 10,000/2,000/2,000 group-disjoint valid-query manifests and five
  2,000-row connected-source-group OOF folds.
- Synthetic MPS simplex allocation passed nonnegativity, conservation
  (maximum error `4.76837158203125e-07`), finite-gradient, low-flux, and
  zero-source checks.
- Full frozen-target audit found observed negative fractions
  `0.486877/0.481794/0.482363` and target-sum exceedances in every episode.
- Stopped before architecture construction. Outcome:
  `DATA_OR_IMPLEMENTATION_FAILURE`.
- No training, checkpoint, reconstruction, replay, safety label, bootstrap,
  family comparison, auditor, development, Atlas-selection, or lockbox access
  occurred.

## 2026-07-14 — Family-E1 signed-noise-residual preflight

- Run:
  `outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/`.
- Froze preregistration SHA-256
  `be546f7f1aa2ec04f1a76f84bc5305c87521d5b89331c681dc3cdf18a5293d3b`
  before new source-tensor loading.
- Froze one in-forward ReLU requested/companion mapping and signed algebraic
  observation-closure residual; no alternative mapping was tested.
- Synthetic MPS gates passed without CPU fallback.
- Full 10,000/2,000/2,000 audit passed: zero mapped-source negatives,
  finite residuals with both signs, source round-trip within tolerance, and
  float32 closure error at most `0.015625`.
- Outcome: `SIGNED_NOISE_RESIDUAL_CONTRACT_PASS`.
- No model, optimizer, checkpoint, reconstruction, safety label, bootstrap,
  auditor, development, Atlas-selection, or lockbox access occurred.

## 2026-07-14 — Thayer-Family-E1-v0

- Run: `outputs/runs/thayer_family_e1_v0_20260714_214715/`.
- Preregistration SHA-256:
  `33c65102ec946cb980709fe66ca3728e85e0066c844354932af49d31c2aa65d5`.
- Constructed exactly one 1,162,662-parameter 4-channel compact coordinate
  U-Net with a six-channel in-forward ReLU source head and derived signed
  residual.
- Physical and objective-alignment audits passed; truth was stationary and no
  compromise beat truth.
- Ordinary micro-overfit passed with `0.998960` objective reduction and 1.0
  prompt identity.
- Difficult and mixed-eight prompt identity was `0.50/0.5625`, below 0.90;
  outcome `FAMILY_E1_RECONSTRUCTION_FAILURE`.
- Full training, OOF, replay, safety labeling, family comparison, bootstrap,
  and auditor training did not run. Protected-data access stayed `0/0/0`.
