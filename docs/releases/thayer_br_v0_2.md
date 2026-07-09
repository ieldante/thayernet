# Thayer-BR v0.2 Moderate: Weighted Balanced Residual U-Net

## Summary

Thayer-BR v0.2 Moderate is the current best Thayer-Net model on the controlled
synthetic Galaxy10 DECaLS-style deblending benchmark. It reconstructs a clean
target galaxy from a synthetic two-galaxy blend and is evaluated against known
clean targets.

The model builds on Thayer-BR v0.1, the balanced hard-case residual U-Net, by
adding an affected/core-weighted residual loss. The architecture and residual
prediction formulation stay the same: the model predicts contaminant residual
light, and the reconstruction subtracts that residual from the blended image.

The moderate weighted loss improves affected-region and core reconstruction
relative to Thayer-BR v0.1 while eliminating the remaining worse-than-identity
normal cases in the tested setup. These are controlled synthetic benchmark
results, not full real-sky survey deployment claims.

## Headline Results

Current best model versus identity:

| Evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Improvement |
| --- | ---: | ---: | ---: |
| Normal held-out | 0.068122 | 0.002108 | ~32.3x |
| Hard stress test | 0.075541 | 0.003847 | ~19.6x |

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
- The earlier balanced/weighted model ranking is robust across affected-mask
  thresholds `0.005`, `0.01`, `0.02`, and `0.04`.
- The ranking is robust across mask dilation radii `0`, `1`, `3`, `5`, and `9`,
  which checks sensitivity to halo inclusion.
- Multi-seed evaluation supports the large improvement ratios.
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

Use Thayer-BR v0.2 Moderate as the current best model.

Report Thayer-BR v0.1 as the previous balanced residual checkpoint and
Thayer-BR v0.2 Strong as a
weighting ablation, not as the main model.

The strongest headline should use the multi-seed robustness numbers:
`32.02 +/- 1.21x` normal improvement and `19.55 +/- 0.30x` stress improvement.
The same-run single-seed tables are still useful for detailed model comparison.

The paper should include both positive examples and counterexamples. In
particular, it should not hide cases where v0.2 Moderate has broad low-level
artifacts or where another model looks cleaner despite worse aggregate metrics.
