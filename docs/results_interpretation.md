# Results Interpretation

## High-Level Takeaway

Thayer-BR v0.2 Moderate is the current best Thayer-Net model on the controlled
synthetic Galaxy10 DECaLS-style deblending benchmark. In the same-run
comparison, it improves affected-region MSE from `0.068122` to `0.002108` on
normal held-out blends, about `32.3x`, and from `0.075541` to `0.003847` on the
hard stress test, about `19.6x`.

The multi-seed audit supports the same conclusion: `32.02 +/- 1.21x` normal
improvement and `19.55 +/- 0.30x` stress improvement. These are controlled
synthetic benchmark results, not full real-sky survey validation.

## Model Progression

Thayer-Direct establishes that learned reconstruction is useful in the
controlled benchmark. It maps `blended -> target` and strongly beats identity
and threshold baselines, but its stress performance drops under harder overlap
and contaminant conditions.

Thayer-Residual changes the task to residual prediction. It predicts
`blended - target` and reconstructs with `blended - predicted_residual`. This
helps stress robustness because the model learns what contaminant light to
subtract rather than redrawing the whole galaxy.

Thayer-BR v0.1 keeps residual prediction but changes the training distribution
to include more high-overlap/core-obstruction and brightness/size stress cases.
It was the previous best model and showed that targeted hard-case sampling
matters.

Thayer-BR v0.2 Moderate keeps the same residual U-Net family and adds moderate
affected/core-weighted residual loss. It is the current best model.

Thayer-BR v0.2 Strong is a stronger weighting ablation. It slightly improves
stress core MSE relative to Moderate, but worsens aggregate normal affected MSE,
aggregate stress affected MSE, and stress non-core affected MSE. It should not
be treated as the main model.

## Why Affected-Region MSE Matters

Affected-region MSE evaluates only pixels where the blend differs from the
target above the affected threshold. This isolates the pixels where contaminant
light changed the clean target.

Whole-image metrics are still useful, but they can make identity look
deceptively strong because most pixels in each synthetic blend are unchanged.
Affected-region MSE is therefore the primary metric for deblending quality in
this benchmark.

## Why SSIM Improves More Modestly

SSIM is computed over the whole image. In these cutouts, large regions are
unchanged background or target light that the identity baseline already
preserves. A model can substantially improve the contaminated region while
showing a smaller whole-image SSIM change.

This does not make SSIM useless; it means SSIM should be interpreted as a global
image-quality metric, not as the main measure of contaminant removal.

## Why Residual Prediction Helps

Residual prediction asks the model to learn contaminant signal:

```text
true_residual = blended - target
reconstruction = blended - predicted_residual
```

This gives the model a direct path for preserving already-correct target light.
It can focus on subtracting the extra contaminant contribution. That is why
Thayer-Residual improves stress performance relative to Thayer-Direct even
though direct reconstruction still wins on some individual samples.

## Why Balanced Training Helps

Random blend sampling under-represents some of the cases most relevant to
deblending failure, especially core overlap and bright/similar-size
contaminants. Thayer-BR v0.1 makes those cases common during optimization
through a 50/30/20 mix of normal, high-overlap/core-obstruction, and
brightness/size stress blends.

The improvement from Thayer-Residual to Thayer-BR v0.1 shows that training
distribution matters even when architecture and residual formulation are held
fixed.

## Why Weighted Loss Helps

Thayer-BR v0.2 Moderate changes the loss so contaminated pixels and affected
target-core pixels receive more emphasis than unchanged background pixels. The
moderate setting uses affected/core extra weights `3/2`.

This better matches the metric and scientific objective. The model is not
rewarded primarily for background pixels that were already easy; more training
pressure is placed on the regions where contaminant light changed the target.

Compared with Thayer-BR v0.1, v0.2 Moderate lowers:

- normal affected MSE from `0.002451` to `0.002108`;
- stress affected MSE from `0.004587` to `0.003847`;
- stress core MSE from `0.013848` to `0.009533`.

## Why Strong Weighting Is Not Best

Thayer-BR v0.2 Strong uses affected/core extra weights `5/4`. It slightly
improves stress core MSE relative to Moderate (`0.009344` versus `0.009533`),
but worsens normal affected MSE (`0.002306` versus `0.002108`), stress affected
MSE (`0.004030` versus `0.003847`), and stress non-core affected MSE.

This shows that the weighting objective has a tradeoff. More core emphasis can
over-focus the model on core pixels and reduce broader affected-region quality.
Moderate weighting is the best current balance.

## Why Size Ratio Does Not Currently Threaten the Result

The size/visual audit found that apparent contaminant/target radius ratio varies
substantially: approximately `0.49` at the 5th percentile, `1.06` at the median,
and `2.37` at the 95th percentile. That is enough variation to justify a future
size-normalized benchmark.

However, learned-model affected-error dependence on apparent size ratio was
weak in the audit (`-0.17` direct, `0.05` residual, `-0.14` BR v0.1, `-0.12`
BR v0.2). The current evidence does not show that the headline ranking is just
a size shortcut.

The right conclusion is cautious: keep the current controlled benchmark result,
and add a size-normalized held-out benchmark before making stronger
size-invariance claims.

## Why Some Visual Examples Still Look Weird

Lower affected-region MSE does not always match visual preference. A model can
reduce large core errors while introducing broad low-level residual patterns
that are visible in error maps. Conversely, a direct model can sometimes look
cleaner while having worse affected-region MSE.

The halo-band audit did not show an aggregate v0.2 halo penalty. Normal
halo-band MSE improved from `0.000300` for BR v0.1 to `0.000250` for v0.2
Moderate; stress halo-band MSE improved from `0.000359` to `0.000320`.

Selected individual examples still show broad low-level artifacts and
visual-vs-metric disagreements. These examples should appear in the paper as
limitations and diagnostic figures.

## Limitations and Future Work

The current result is limited to controlled synthetic Galaxy10 DECaLS-style
blends with known clean targets. It does not prove performance on arbitrary
survey imagery, crowded real fields, physically correlated sources, spatially
varying PSFs, or realistic sky-background conditions.

Recommended next steps:

- preserve exact generated evaluation sets and source indices;
- run a size-normalized held-out benchmark;
- keep halo-band and visual-vs-metric diagnostics in future comparisons;
- improve sky, PSF, and noise realism;
- report counterexamples alongside headline metrics.
