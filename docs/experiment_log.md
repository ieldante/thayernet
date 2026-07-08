# Experiment Log

This log records the main local experiment checkpoints. Raw outputs,
checkpoints, and generated run directories remain under `outputs/` and are not
committed.

## Checkpoint 1: Direct U-Net on Normal Held-Out Blends

Date: 2026-07-08.

### Settings

- Dataset: Galaxy10 DECaLS from `data/Galaxy10_DECals.h5`.
- Split seed: 42.
- Task: direct reconstruction, `blended -> clean target`.
- Train/validation/test blends: 5,000 / 800 / 800.
- Epochs: 20.
- Batch size: 8.
- Model: compact U-Net.
- Rotation: disabled.
- Checkpoint: `outputs/checkpoints/unet_direct_5000train_800val_800test_20ep.pth`.

### Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.005224 | 0.964264 | 0.062555 | 0.180238 | 1.00x |
| threshold | 0.029782 | 0.054440 | 0.067528 | 0.207066 | 0.93x |
| direct U-Net | 0.000566 | 0.976648 | 0.004428 | 0.041573 | 14.13x |

### Interpretation

The direct U-Net clearly beats identity and threshold baselines on controlled
normal held-out blends. Affected-region metrics are the headline result because
they evaluate only pixels changed by the contaminant; whole-image metrics are
less diagnostic when most pixels are unchanged.

The legacy `generation_difficulty` labels from sampled parameters were useful
for early inspection, but they did not represent final model difficulty.
Measured blend severity and model failure metrics were added after this
checkpoint.

## Checkpoint 2: Direct U-Net Hard-Case Stress Test

Date: 2026-07-08.

### Settings

- Run directory: `outputs/runs/stress_test_20260708_145221`.
- Model: direct U-Net checkpoint from Checkpoint 1.
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

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.958915 | 0.075541 | 0.202969 | 1.00x |
| threshold | 0.031344 | 0.045878 | 0.082746 | 0.229405 | 0.91x |
| direct U-Net | 0.000937 | 0.972558 | 0.009390 | 0.060231 | 8.04x |

### Interpretation

The direct U-Net still beats identity on the stress distribution, but
affected-region improvement drops from about 14.13x on normal held-out blends to
about 8.04x. This is the expected direction for a harder distribution with more
overlap and brighter contaminants.

The stress-test analysis separates measured blend severity from model failure.
Some high-severity blends are easy for the model when the contaminant is obvious,
while some lower-severity blends can be difficult when the target core is
obstructed or the contaminant resembles target structure.

## Checkpoint 3: Residual U-Net

Date: 2026-07-08.

### Settings

- Run directory: `outputs/runs/residual_unet_20260708_154947`.
- Task: residual prediction, `blended -> blended - target`.
- Reconstruction: `blended - predicted_residual`.
- Train/validation/held-out blends: 5,000 / 800 / 800.
- Epochs: 20.
- Batch size: 8.
- Model: compact U-Net with linear residual output head.
- Checkpoint:
  `outputs/checkpoints/unet_residual_5000train_800val_800test_20ep_20260708_154947.pth`.

### Normal Held-Out Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.005224 | 0.964264 | 0.062555 | 0.180238 | 1.00x | 0/800 |
| threshold | 0.029782 | 0.054440 | 0.067528 | 0.207066 | 0.93x | 0/800 |
| direct U-Net | 0.000566 | 0.976648 | 0.004428 | 0.041573 | 14.13x | 5/800 |
| residual U-Net | 0.000390 | 0.981015 | 0.004039 | 0.045027 | 15.49x | 1/800 |

Residual beats direct on 310/800 normal held-out cases. Direct has slightly
better affected-region MAE on the normal aggregate, so residual is not
universally better.

### Hard Stress-Test Metrics

| Method | Whole MSE | Whole SSIM | Affected MSE | Affected MAE | Improvement vs identity | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.007602 | 0.958915 | 0.075541 | 0.202969 | 1.00x | 0/1000 |
| threshold | 0.031344 | 0.045878 | 0.082746 | 0.229405 | 0.91x | 0/1000 |
| direct U-Net | 0.000937 | 0.972558 | 0.009390 | 0.060231 | 8.04x | 13/1000 |
| residual U-Net | 0.000700 | 0.977410 | 0.007069 | 0.058334 | 10.69x | 0/1000 |

Residual beats direct on 667/1000 stress cases and reduces worse-than-identity
stress failures from 13/1000 to 0/1000.

### Interpretation

Residual prediction improves aggregate affected-region MSE on both normal
held-out blends and the hard stress set. The improvement is clearest under
stress, where residual prediction appears to preserve target structure better and
avoid the direct model's worst regressions. Direct reconstruction still wins on
some individual cases, so future work should analyze when each formulation is
preferable rather than treating residual prediction as a universal replacement.

## Terminology Correction: Difficulty vs Severity vs Model Failure

Earlier notebook cells and static figures used easy/medium/hard labels from the
blend generator. These labels are now treated as legacy
`generation_difficulty` metadata, not as model-failure categories.

Current terminology:

- `generation_difficulty`: legacy generator metadata assigned from sampled
  shift, brightness, blur, noise, and source-size ratio.
- `blend_severity_score` / `blend_severity_bin`: measured image damage based on
  affected mask fraction, identity affected error, and optionally core
  obstruction.
- `core_obstruction_fraction` / `core_overlap_bin`: how much the contaminant
  affects the target core.
- `model_failure_score`: model affected-region MSE.
- `model_improvement_ratio`: identity affected-region MSE divided by model
  affected-region MSE.

A high-severity blend is not always hardest for the model. Some large obvious
contaminants are severe but easy to subtract. Some lower-severity blends are
hard if they hit the target core or mimic target structure.

Earlier figures may display the generator's legacy easy/medium/hard metadata.
These labels are retained for provenance but are not treated as model-failure
categories.

## Next Steps

- Build a core-obstruction-balanced evaluation set.
- Train or fine-tune on a balanced hard-case residual dataset.
- Test an affected-region-weighted loss.
- Improve foreground extraction and residual-output diagnostics.
- Add preprocessing checks for halo leakage, masked background artifacts, and
  source-size imbalance.
- Move toward more realistic sky, noise, PSF, and background simulations.
- Write the final report using the direct, stress, and residual checkpoints as
  the first complete experimental arc.
