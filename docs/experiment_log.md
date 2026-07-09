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
- Thayer-BR v0.1 improves both normal and stress
  aggregate affected-region MSE in the current same-run evaluation.
- Remaining failures involve ambiguity, target-core overlap, lost target detail,
  foreground-extraction limitations, and model-specific tradeoffs.

## Next Steps

- Preserve exact generated evaluation sets and global source indices for future
  reproducibility.
- Finalize paper figures and captions.
- Write the LaTeX report.
- Consider affected-region-weighted loss or a hybrid direct/residual model only
  if more modeling is needed.
- Improve preprocessing, foreground extraction diagnostics, sky/noise realism,
  and core-obstruction-balanced evaluation.
