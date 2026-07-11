# Thayer-BR v0.2 Moderate: Weighted Balanced Residual U-Net

> **Historical release status (2026-07-10):** v0.2 remains the current model
> family. The values below are original row-split development results. A later
> grouped retrain gives `28.81x` normal; a fresh locked final partition is still
> required before a final-paper effect-size claim.

## Summary

This historical release introduced the current-best Thayer-Net model family on
the controlled synthetic Galaxy10 DECaLS-style development benchmark. It
reconstructs an unblended target reference from a synthetic two-galaxy blend.
The later grouped checkpoint is the defensible development reference.

The model builds on Thayer-BR v0.1, the balanced hard-case residual U-Net, by
adding an affected/core-weighted residual loss. The architecture and residual
prediction formulation stay the same: the model predicts a blend-to-target
correction field, and the reconstruction subtracts that field from the blended
image. This field can include target-blur, noise, and clipping correction and is
not pure contaminant light.

The moderate weighted loss improves affected-region and core reconstruction
relative to Thayer-BR v0.1 while eliminating the remaining worse-than-identity
normal cases in the tested setup. These are controlled synthetic benchmark
results, not full real-sky survey deployment claims.

## Headline Results

Original development checkpoint versus identity:

| Original development evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Lower affected MSE vs identity |
| --- | ---: | ---: | ---: |
| Normal | 0.068122 | 0.002108 | ~32.3x |
| Hard stress | 0.075541 | 0.003847 | ~19.6x |

Multi-seed audit:

| Evaluation | Improvement mean +/- std |
| --- | ---: |
| Normal | 32.02 +/- 1.21x |
| Stress | 19.55 +/- 0.30x |

## What Changed From v0.1

Thayer-BR v0.2 Moderate keeps the same balanced residual architecture and
formulation as Thayer-BR v0.1. The change is the training objective: errors in
contaminated pixels and affected target-core pixels receive higher loss weight
than unchanged background pixels.

This matches the main scientific objective better than a uniform pixel loss.
Most pixels in each blend are unchanged, so a uniform loss can spend too much
optimization pressure on already-correct background. The weighted residual loss
pushes the same U-Net to focus more directly on pixels where contaminant light
changed the target.

| Metric | Thayer-BR v0.1 | Thayer-BR v0.2 Moderate | Direction |
| --- | ---: | ---: | --- |
| Normal affected MSE | 0.002451 | 0.002108 | lower is better |
| Stress affected MSE | 0.004587 | 0.003847 | lower is better |
| Stress core MSE | 0.013848 | 0.009533 | lower is better |
| Normal worse-than-identity | 1/1000 | 0/1000 | lower is better |
| Stress worse-than-identity | 0/1000 | 0/1000 | lower is better |

Approximate improvement over Thayer-BR v0.1:

- Normal affected MSE is about 14% lower.
- Stress affected MSE is about 16% lower.
- Stress core MSE is about 31% lower.

## Why Moderate Weighting Won

The moderate variant used affected/core extra weights `3/2`. It improved
aggregate affected-region MSE on both normal and stress evaluations and
improved stress core affected MSE.

The stronger variant used affected/core extra weights `5/4`. It slightly
improved stress core MSE relative to Moderate, but worsened aggregate normal
affected MSE, aggregate stress affected MSE, and stress non-core affected MSE.

The conclusion is that stronger weighting is not automatically better. Moderate
weighting is the current best setting because it improves the intended
contaminated/core regions without over-focusing on core pixels at the expense of
the broader affected region.

## What Was Audited

The v0.2 Moderate result is supported by several evaluation-only checks:

- Affected masks are based on `abs(blended - target).mean(axis=-1) > threshold`,
  not on model prediction error.
- In the earlier v0.1-era audit, Thayer-BR v0.1 remained best across
  affected-mask thresholds `0.005`, `0.01`, `0.02`, and `0.04`; v0.2 Moderate
  was not included.
- Thayer-BR v0.1 also remained best across mask dilation radii `0`, `1`, `3`,
  `5`, and `9`, although lower-ranked methods changed order. This does not
  establish direct v0.2 mask robustness or complete rank stability.
- Evaluation/blend-seed variation supports the reported affected-MSE ratios;
  the models were not independently retrained.
- Residual logic was verified: `true_residual = blended - target` and
  `reconstruction = blended - predicted_residual`.
- Checkpoint integrity checks confirmed that old checkpoints were not modified
  by evaluation/audit runs.
- Apparent-size audit found broad contaminant/target size variation but weak
  learned-model affected-error dependence on size ratio.
- Halo-band audit found improved aggregate halo-band MSE for v0.2 Moderate
  relative to v0.1, while selected individual cases still show broad low-level
  artifacts.
- Visual-vs-metric disagreement examples were collected for paper caveats and
  appendix figures.

## Caveats

- Results are for controlled synthetic Galaxy10 DECaLS-style blends only.
- The benchmark is not full real-sky survey validation.
- The model should not be used as a production astronomy pipeline without
  substantial external validation.
- Apparent-size normalization has not yet been run as a full benchmark.
- Foreground and halo extraction remain approximate.
- Target-core overlap remains the hardest region.
- Thayer-BR v0.2 Moderate can show broad low-level or halo-like artifacts in
  selected individual cases even though aggregate halo-band error improved.
- Per-sample winners vary; v0.1, direct, or residual models can look better on
  some examples.

## Recommended Use in Paper

Use the Thayer-BR v0.2 Moderate grouped retrain as the current development
reference: `28.81x` normal, `15.80x` hard, `9.18x` compact-bright, and `15.84x`
high-core lower affected MSE than identity. It is not a final-paper result.

Report Thayer-BR v0.1 as the previous balanced residual checkpoint and
Thayer-BR v0.2 Strong as a
weighting ablation, not as the main model.

A final paper headline should be withheld until a fresh untouched group-disjoint
test is run. Retain `32.02 +/- 1.21x` normal and `19.55 +/- 0.30x` stress only as
the historical row-split evaluation-seed audit, and state that it varied
evaluation seeds rather than independently retraining checkpoints. The old
checkpoint on grouped manifests is exposure-confounded and diagnostic only.

The paper should include both positive examples and counterexamples. In
particular, it should not hide cases where v0.2 Moderate has broad low-level
artifacts or where another model looks cleaner despite worse aggregate metrics.
