# Professor Briefing

## 30-Second Version

Thayer-Net is a controlled synthetic galaxy deblending benchmark using Galaxy10
DECaLS cutouts. The best current model is **Thayer-BR v0.2 Moderate**, a
balanced residual U-Net with an affected/core-weighted residual loss. It
improves affected-region MSE by about `32.3x` on normal held-out blends and
about `19.6x` on hard stress blends in the same-run comparison, with multi-seed
results of `32.02 +/- 1.21x` normal and `19.55 +/- 0.30x` stress.

This is controlled synthetic evidence, not a claim of full real-sky
survey-grade deblending.

## 2-Minute Version

The project builds a synthetic deblending benchmark where the clean target is
known. Instead of adding whole rectangular contaminant cutouts, the pipeline
extracts foreground contaminant light with halo-aware masks, which avoids
trivial pasted-patch cues. Evaluation focuses on affected-region metrics because
whole-image metrics are dominated by unchanged pixels.

The model progression is:

- Thayer-Direct maps `blended -> target`.
- Thayer-Residual predicts `blended - target` and reconstructs by subtraction.
- Thayer-BR v0.1 keeps residual prediction but trains on a 50/30/20 mix of
  normal, high-overlap/core-obstruction, and brightness/size stress blends.
- Thayer-BR v0.2 Moderate keeps that residual setup and adds moderate
  affected/core-weighted loss.

The v0.2 Moderate result improves on v0.1: normal affected MSE changes from
`0.002451` to `0.002108`, stress affected MSE from `0.004587` to `0.003847`,
and stress core MSE from `0.013848` to `0.009533`. The stronger weighted
variant slightly improves stress core MSE but worsens aggregate normal/stress
performance, so it is an ablation rather than the main model.

## Key Numbers

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement | Stress core MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.068122 | 1.00x | 0.075541 | 1.00x | 0.085131 |
| Thayer-Direct | 0.004236 | 16.08x | 0.009390 | 8.04x | 0.036538 |
| Thayer-Residual | 0.004431 | 15.37x | 0.007069 | 10.69x | 0.015913 |
| Thayer-BR v0.1 | 0.002451 | 27.79x | 0.004587 | 16.47x | 0.013848 |
| Thayer-BR v0.2 Moderate | 0.002108 | ~32.3x | 0.003847 | ~19.6x | 0.009533 |

## If Asked About 32x

The `32x` result is affected-region improvement over identity, not whole-image
improvement. It compares identity affected MSE `0.068122` with v0.2 Moderate
affected MSE `0.002108` on normal held-out blends.

The number is large because identity preserves unchanged pixels but does not
remove contaminant light. Affected-region MSE isolates the pixels where the
contaminant changed the target. The result was audited with mask-threshold
checks, dilation/halo checks, residual logic checks, multi-seed evaluation, and
checkpoint integrity checks.

## If Asked About Size Normalization

The current benchmark normalizes pixel values, not apparent galaxy size. A
size/visual audit found broad contaminant/target apparent radius ratios:
approximately `0.49` at p5, `1.06` at the median, and `2.37` at p95.

The learned-model affected-error dependence on size ratio was weak in that
audit, so the current result is not obviously just a size shortcut. Still, a
future size-normalized held-out benchmark is recommended before making stronger
claims about size-invariant deblending.

## If Asked Why v0.2 Sometimes Looks Worse Visually

The metric and visual judgment can disagree. v0.2 Moderate often lowers
affected and core MSE, but some individual examples show broad low-level or
halo-like artifacts in error maps. The aggregate halo-band audit improved
relative to v0.1, but the selected counterexamples should still be shown as
limitations.

The safest phrasing is: v0.2 Moderate is best in aggregate, not universally
best on every sample.

## If Asked Whether This Generalizes Outside DECaLS

Not yet. The result is for controlled synthetic Galaxy10 DECaLS-style blends
with known clean targets. It does not validate the model on arbitrary survey
images, crowded real fields, different PSFs, real sky backgrounds, detector
artifacts, or physically correlated source environments.

## If Asked Whether To Stop Modeling

The current stopping point is good for writing: Thayer-BR v0.2 Moderate is a
clear current best model.

Thayer-BR v0.1 is a useful balanced-training ablation, and Strong is a useful
loss-weighting ablation.

The next modeling-related step should be evaluation-only first: create a
size-normalized held-out benchmark and evaluate current checkpoints. New
training should wait until that benchmark clarifies whether size normalization
changes the model ranking.

## Questions To Bring

- Is the v0.2 Moderate result enough to stop modeling and focus on the paper?
- Should the paper foreground weighted loss as the final model improvement or
  present it as an ablation after v0.1?
- How much space should be given to visual counterexamples?
- Is the size-normalized benchmark required before submission, or acceptable as
  future work?
