# Thayer-Net Experiment Summary

> **Development-benchmark status (2026-07-10):** recorded historical metrics
> remain unchanged. An exact-pixel/exact-coordinate grouped retrain now gives
> `28.81x` lower normal affected MSE than identity. It is a grouped development
> result, not a final claim; multi-seed historical results vary evaluation, not
> training, seeds.

## Overview

Thayer-Net is a compact U-Net research testbed for controlled synthetic galaxy
deblending. It uses Galaxy10 DECaLS cutouts, constructs synthetic blends with
known clean targets, and evaluates whether learned models can reconstruct the
target galaxy more accurately than simple non-learning baselines.

The current project state supports a careful but limited claim: compact U-Nets
substantially improve target reconstruction on controlled synthetic blends, and
residual prediction with balanced hard-case training plus moderate
affected/core-weighted loss gives the strongest current result. These results
should not be interpreted as validated real-survey performance.

## Model Naming

- **Thayer-Net:** project name and model family.
- **Thayer-Direct:** direct reconstruction U-Net experiment.
- **Thayer-Residual:** residual prediction U-Net experiment.
- **Thayer-BR v0.1 (Balanced Residual U-Net):** previous balanced hard-case
  residual U-Net, trained on 8,000 synthetic blends with a 50/30/20 mix of
  normal, high-overlap/core-obstruction, and brightness/size-stress cases.
- **Thayer-BR v0.2 Moderate:** current best model family, a balanced residual
  U-Net trained with moderate affected/core-weighted residual loss. The original
  checkpoint is historical; a grouped-retrained checkpoint is the defensible
  development reference.
- **Thayer-BR v0.2 Strong:** stronger weighted-loss ablation, not the current
  best model.
- **Thayer-BR v0.3 Delta:** preservation/color/perceptual tradeoff ablation;
  not the current reference.
- **Thayer-ResUNet v0.4:** controlled architecture ablation with targeted
  compact/halo gains; not the current reference.

## Current Development Reference

Current best model family: **Thayer-BR v0.2 Moderate**.

| Grouped development suite | Identity affected MSE | Grouped retrain affected MSE | Lower affected MSE vs identity |
| --- | ---: | ---: | ---: |
| Normal | 0.066814 | 0.002319 | 28.81x |
| Hard stress | 0.072531 | 0.004590 | 15.80x |
| Compact bright | 0.080147 | 0.008728 | 9.18x |
| High core obstruction | 0.077871 | 0.004917 | 15.84x |

These are source-group-disjoint development estimates, not final-paper values.
The old checkpoint's better score on these grouped manifests is diagnostic only
because `54.575%` of rows expose a historical train/validation source group.

The original row-split development result (`32.3x` normal and `19.6x` hard) and
its evaluation-seed audit (`32.02 +/- 1.21x`, `19.55 +/- 0.30x`) are preserved
as historical evidence only. The latter varied evaluation seeds, not training
seeds.

A fresh untouched group-disjoint source pool is still required for the future
final-paper test.

## Development Phase

Early synthetic blending risked rectangular cutout and double-background
artifacts when whole contaminant cutouts were added directly to target images.
The pipeline was improved to extract only foreground contaminant light before
addition. Halo-aware masks were added so diffuse galaxy outskirts were retained
while cutout boundaries were suppressed.

Evaluation also changed during development. Whole-image metrics alone can hide
deblending failures because most pixels are unchanged. Affected-region metrics
were added to evaluate only pixels where the contaminant changed the target.
Legacy easy/medium/hard labels from the generator were kept as
`generation_difficulty` metadata, but current analysis separates generation
difficulty, measured blend severity, target-core obstruction, and model failure.

## Formal Experiments

### Experiment 1: Thayer-Direct

Thayer-Direct maps `blended -> target`. It tests whether a compact learned
model can reconstruct the clean target better than identity and threshold
baselines on normal held-out synthetic blends.

An earlier 800-blend normal evaluation found identity affected-region MSE
`0.062555`, Thayer-Direct affected-region MSE `0.004428`, and about `14.13x`
improvement. On the current 1,000-blend same-run normal evaluation, Thayer-Direct
affected-region MSE is `0.004236`, corresponding to `16.08x` improvement over
identity.

### Experiment 1b: Thayer-Direct Hard Stress Test

Hard stress testing was added to probe smaller shifts, brighter contaminants,
similar-size sources where possible, blur/noise perturbations, and core overlap.
On this 1,000-blend stress set, Thayer-Direct affected-region MSE is `0.009390`,
or `8.04x` improvement over identity. The drop from the earlier normal result
shows that the normal held-out score did not fully characterize overlap
robustness.

### Experiment 2: Thayer-Residual

Thayer-Residual predicts `residual = blended - target` and reconstructs with
`target_hat = blended - predicted_residual`. This objective can preserve target
light by learning what contaminant signal to subtract rather than redrawing the
entire galaxy.

On the hard stress test, residual prediction improves affected-region MSE from
`0.009390` for Thayer-Direct to `0.007069`, or `10.69x` improvement over
identity. It also reduces worse-than-identity stress cases from `13/1000` for
Thayer-Direct to `0/1000`.

### Experiment 3: Thayer-BR v0.1

Thayer-BR v0.1 (Balanced Residual U-Net) keeps the residual objective but
changes the training distribution: 8,000 training blends with 50%
normal/random blends, 30% high-overlap/core-obstruction blends, and 20%
brightness/size stress blends.
Validation used 1,000 blends, batch size was 8, and training ran for 20 epochs.
The best saved model checkpoint occurred at epoch 18 with validation loss
`0.000378`; final train/validation loss was `0.000336 / 0.000383`.

This experiment tests whether targeted hard-case sampling improves stress
robustness without sacrificing normal held-out performance. In the current
same-run evaluation, it improves both normal and stress affected-region MSE.

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement | Worse-than-identity stress cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.068122 | 1.00x | 0.075541 | 1.00x | 0/1000 |
| Threshold | 0.073101 | 0.93x | 0.082746 | 0.91x | 990/1000 |
| Thayer-Direct | 0.004236 | 16.08x | 0.009390 | 8.04x | 13/1000 |
| Thayer-Residual | 0.004431 | 15.37x | 0.007069 | 10.69x | 0/1000 |
| Thayer-BR v0.1 | 0.002451 | 27.79x | 0.004587 | 16.47x | 0/1000 |

Thayer-BR v0.1 beats Thayer-Residual on `91.3%` of normal cases and `87.9%`
of stress cases. It beats Thayer-Direct on `76.1%` of normal cases and `93.1%`
of stress cases. Thayer-Direct and Thayer-Residual still win on some individual
samples, so the result should be presented as an aggregate robustness
improvement, not as a universal per-sample dominance claim.

### Experiment 4: Thayer-BR v0.2 Moderate

Thayer-BR v0.2 Moderate keeps the residual U-Net architecture and balanced
hard-case training idea, but changes the objective to a normalized weighted
residual loss. Affected pixels receive extra weight `3`, and affected target
core pixels receive extra weight `2`.

The moderate run used 12,000 training blends and 1,000 validation blends with
the same 50/30/20 training composition target: normal/random, high-overlap/core
obstruction, and brightness/size stress.

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement | Stress core MSE | Worse-than-identity stress cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.068122 | 1.00x | 0.075541 | 1.00x | 0.085131 | 0/1000 |
| Thayer-BR v0.1 | 0.002451 | 27.79x | 0.004587 | 16.47x | 0.013848 | 0/1000 |
| Thayer-BR v0.2 Moderate | 0.002108 | ~32.3x | 0.003847 | ~19.6x | 0.009533 | 0/1000 |

Relative to Thayer-BR v0.1, v0.2 Moderate lowers normal affected MSE by about
14%, stress affected MSE by about 16%, and stress core MSE by about 31%. It also
removes the remaining normal worse-than-identity case in the tested setup.

Thayer-BR v0.2 Strong used larger affected/core extra weights `5/4`. It
slightly improved stress core MSE relative to Moderate, but worsened aggregate
normal affected MSE, aggregate stress affected MSE, and stress non-core affected
MSE. It is therefore an ablation rather than the current best model.

## Evaluation Robustness Audit

Audit run: `outputs/runs/evaluation_audit_20260708_220833`.

The audits loaded existing checkpoints for evaluation only. Checkpoint size and
modified-time records were unchanged before and after the audit passes.

The affected-region mask is computed from
`abs(blended - target).mean(axis=-1) > threshold`, so the mask is independent of
model predictions. In the v0.1-era audit, Thayer-BR v0.1 remained the best
audited method across thresholds `0.005`, `0.01`, `0.02`, and `0.04`. Thayer-BR
v0.2 Moderate was not included in that audit.

Halo sensitivity was tested by dilating the affected mask by `0`, `1`, `3`,
`5`, and `9` pixels. Thayer-BR v0.1 remained best at every tested radius, but
some lower-ranked methods changed order. This supports only the narrower v0.1
mask-robustness claim, not a direct v0.2 claim or complete rank stability.

The v0.2 Moderate evaluation-seed audit found mean identity/model
affected-MSE ratios of `32.02 +/- 1.21x` on normal blends and
`19.55 +/- 0.30x` on stress blends. The checkpoint was not independently
retrained.

Core-region metrics support the aggregate result, but core-obstructed pixels
remain the hardest region. Thayer-BR v0.2 Moderate improves stress core MSE
from `0.013848` for v0.1 to `0.009533`, while non-core errors are already much
smaller.

The residual logic audit confirmed the intended sign convention:
`residual = blended - target` and `reconstruction = blended - predicted_residual`.
The `~32.3x` normal and `~19.6x` stress same-run identity/model affected-MSE
ratios, together with the `32.02 +/- 1.21x` and `19.55 +/- 0.30x`
evaluation-seed results, should be presented as controlled synthetic metrics.
They should not be presented as RMSE ratios, real-survey performance,
training-seed robustness, or universal per-sample dominance.

## Limitations

These results are for controlled synthetic blends and should not be interpreted
as validated real-survey performance. The current setup does not fully
model PSF variation, sky background mismatch, detector artifacts, source
crowding, physically correlated galaxy environments, or the full ambiguity of
real overlapping astronomical sources.

## Recommended Next Experiment

The next scientific step is to document the v0.2 Moderate result, preserve exact
generated evaluation sets and global source indices for future reproducibility,
and run a size-normalized held-out benchmark before making stronger claims about
size-invariant deblending.
