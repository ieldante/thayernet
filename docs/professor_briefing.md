# Professor Briefing

For the concise post-audit update, see
[`advisor_update_grouped_audit.md`](advisor_update_grouped_audit.md).

> **Development-benchmark status (2026-07-10):** the original `32.3x` and
> `19.6x` figures are affected-MSE ratios versus identity. A grouped retrain now
> gives `28.81x` normal and `15.80x` hard on group-disjoint development suites.
> The multi-seed historical check varies evaluation seeds, not training seeds.

## 30-Second Version

Thayer-Net is a controlled synthetic galaxy deblending benchmark using Galaxy10
DECaLS cutouts. The current development reference is the **Thayer-BR v0.2
Moderate grouped retrain**, which achieved `28.81x` normal, `15.80x` hard,
`9.18x` compact-bright, and `15.84x` high-core lower affected-region MSE than
identity. The original row-split result (`32.3x` normal, `19.6x` hard) is now
historical evidence only. The grouped result uses one training seed and is not
a final-paper estimate.

This is controlled synthetic evidence, not a claim of full real-sky
survey-grade deblending.

## 2-Minute Version

The project builds a synthetic deblending benchmark where the clean target is
known. Instead of adding whole rectangular contaminant cutouts, the pipeline
extracts foreground contaminant light with halo-aware masks, which reduces
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

## Original Development-Split Numbers

| Model | Normal affected MSE | Normal identity/model MSE ratio | Stress affected MSE | Stress identity/model MSE ratio | Stress core MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.068122 | 1.00x | 0.075541 | 1.00x | 0.085131 |
| Thayer-Direct | 0.004236 | 16.08x | 0.009390 | 8.04x | 0.036538 |
| Thayer-Residual | 0.004431 | 15.37x | 0.007069 | 10.69x | 0.015913 |
| Thayer-BR v0.1 | 0.002451 | 27.79x | 0.004587 | 16.47x | 0.013848 |
| Thayer-BR v0.2 Moderate | 0.002108 | ~32.3x | 0.003847 | ~19.6x | 0.009533 |

## If Asked About 32x

The historical `32x` result is an affected-region identity/model MSE ratio, not
an RMSE or whole-image ratio. It compares identity affected MSE `0.068122` with
the original v0.2 Moderate affected MSE `0.002108` on row-split development
blends. It must not be presented as the grouped or final estimate.

The number is large because identity preserves unchanged pixels but does not
remove contaminant light. Affected-region MSE isolates the pixels where the
contaminant changed the target. Residual logic, evaluation-seed sensitivity,
and checkpoint integrity were audited for v0.2. The mask-threshold and
dilation/halo sensitivity audit predates v0.2 and supports the v0.1-era result,
not direct v0.2 mask robustness.

## If Asked About Size Normalization

The current benchmark normalizes pixel values, not apparent galaxy size. A
size/visual audit found broad contaminant/target apparent radius ratios:
approximately `0.49` at p5, `1.06` at the median, and `2.37` at p95.

The learned-model affected-error dependence on size ratio was weak in that
historical audit, so the original development result is not obviously just a
size shortcut. A future prototype should use grouped development sources only;
the untouched final pool must not be used to design it.

## If Asked Why v0.2 Sometimes Looks Worse Visually

The metric and visual judgment can disagree. v0.2 Moderate often lowers
affected and core MSE, but some individual examples show broad low-level or
halo-like artifacts in error maps. The aggregate halo-band audit improved
relative to v0.1, but the selected counterexamples should still be shown as
limitations.

The safest phrasing is: v0.2 Moderate remains the strongest model family and
the grouped retrain is the current development reference, but neither is
universally best on every sample or a final-paper result.

## If Asked Whether This Generalizes Outside DECaLS

Not yet. The result is for controlled synthetic Galaxy10 DECaLS-style blends
with known unblended target references. It does not validate the model on arbitrary survey
images, crowded real fields, different PSFs, real sky backgrounds, detector
artifacts, or physically correlated source environments.

## If Asked Whether To Stop Modeling

The grouped evaluation and retrain are complete. No more training should be run
in the current cleanup phase. The grouped v0.2 checkpoint is the defensible
development reference, while the old checkpoint on grouped manifests is only a
historical-exposure diagnostic. A fresh untouched final source partition must
be defined before any final-paper evaluation.

Thayer-BR v0.1 and Strong remain training/loss ablations. Thayer-BR v0.3 Delta
is a preservation/color tradeoff ablation, and ResUNet v0.4 is an architecture
ablation; neither replaces grouped v0.2 Moderate.

## Questions To Bring

- Is the grouped v0.2 Moderate result sufficient to freeze model development
  and focus on a final-safe benchmark and paper?
- Should the paper foreground weighted loss as the development-model improvement or
  present it as an ablation after v0.1?
- How much space should be given to visual counterexamples?
- Should the next protocol use new independent data or a new four-way grouped
  split followed by retraining?
