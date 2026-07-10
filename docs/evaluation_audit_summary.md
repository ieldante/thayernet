# Evaluation Audit Summary

> **Historical audit scope:** this page records the earlier evaluation-only
> audit of the original row-index checkpoint. The later correctness campaign
> completed a grouped retrain at `28.81x` normal and `15.80x` hard lower
> affected MSE than identity. Neither set of values is a locked final result.

## Why This Audit Was Needed

The development-set MSE ratios are large: Thayer-BR v0.2 Moderate has `32.3x`
lower affected-region MSE than identity on normal blends and `19.6x` lower MSE
on hard stress, corresponding to about `5.7x` and `4.4x` lower RMSE. Large ratios need
diagnostic checks because affected-region metrics, mask definitions, residual
sign conventions, and synthetic blend construction can all create misleading
conclusions if implemented incorrectly.

The audits described on this page were evaluation-only. They did not train,
retrain, or modify saved checkpoints.

## Mask Correctness

Affected-region masks are computed from the synthetic blend and known target:

```text
abs(blended - target).mean(axis=-1) > threshold
```

The mask is therefore based on where the contaminant changed the target, not on
model prediction error. This avoids rewarding or penalizing models by changing
the evaluation region after prediction.

## Threshold Sensitivity

The earlier robustness audit evaluated affected-mask thresholds:

```text
0.005, 0.01, 0.02, 0.04
```

Thayer-BR v0.1 (`balanced_residual`) remained best among the methods included at
every threshold. Thayer-BR v0.2 was trained later and was not part of this
mask-threshold audit, so the table is not direct evidence for v0.2 robustness.

## Dilation and Halo Sensitivity

The audit also dilated affected masks by radii:

```text
0, 1, 3, 5, 9
```

Thayer-BR v0.1 remained best at every tested radius. Direct and the earlier
residual model swap order at some radii, so only the v0.1 lead is stable. This
audit predates v0.2 and should not be cited as its mask-dilation evidence.

## Evaluation-Seed Sensitivity

The v0.2 Moderate multi-seed audit found:

| Evaluation | Identity/model affected-MSE ratio, mean +/- std |
| --- | ---: |
| Normal | 32.02 +/- 1.21x |
| Stress | 19.55 +/- 0.30x |

This varies blend-generation/evaluation seeds, not independent training seeds.
It supports the development ranking across several generated sets but does not
establish retraining robustness.

## Residual Logic

Residual reconstruction was checked directly:

```text
true_residual = blended - target
predicted_residual = model(blended)
reconstruction = blended - predicted_residual
```

This confirms that the residual models are not silently being evaluated as
direct reconstruction models and that the sign convention matches the training
target.

## Apparent-Size Audit

The size/visual audit estimated apparent foreground size for target and
contaminant cutouts using border background subtraction, central-source masks,
foreground area, equivalent radius, flux proxies, and centroid estimates.

The contaminant/target apparent radius ratio varied substantially:

- p5 approximately `0.49`;
- median approximately `1.06`;
- p95 approximately `2.37`.

Learned-model affected-error dependence on apparent size ratio was weak in this
audit. The result is not obviously explained by a simple size shortcut.
However, the size range is wide enough that a future size-normalized benchmark
is recommended.

## Visual-vs-Metric Disagreement

The audit selected examples where metric ranking and visual inspection may
disagree:

- v0.2 Moderate beats v0.1 by affected MSE but shows broad low-level error;
- v0.1 or direct beats v0.2 on individual samples;
- direct can look visually cleaner in some cases;
- v0.2 can reduce core error while leaving non-core or halo-band artifacts.

These examples should be included as qualitative caveats, not filtered away.

## Halo-Band Audit

Halo-band error was measured in a ring around the affected mask by dilating the
affected region and subtracting the original mask. This checks for broad,
low-level error outside the directly contaminated pixels.

Aggregate halo-band MSE improved for v0.2 Moderate relative to BR v0.1:

| Evaluation | BR v0.1 halo MSE | BR v0.2 Moderate halo MSE |
| --- | ---: | ---: |
| Normal | 0.000300 | 0.000250 |
| Stress | 0.000359 | 0.000320 |

This means there is no aggregate halo-band penalty in the audited sets. The
qualitative caveat remains: selected individual v0.2 outputs can show broad
low-level artifacts.

## Source Leakage Audit

The full source audit supersedes the earlier row-index-only split check. The
70/15/15 row partitions are disjoint and auditable target/contaminant roles stay
inside their assigned arrays, but the HDF5 file contains duplicate source rows:

- 29 raw-pixel-identical pairs cross train/validation/test;
- 27 of those are also exact-coordinate duplicated objects;
- no local no-duplicate HDF5 variant or object-ID field was found.

Therefore, the normal/stress results are development benchmarks, not a
leakage-cleared final test. The exact tables and limitations are in
`docs/source_leakage_audit.md`.

## Preservation and Clipping Audits

On 1,000 unblended inputs, v0.2 mean reconstruction MSE is `0.00002646`, with
3/1,000 tail cases above `0.001` and visible false subtraction in the saved
grid. The Delta preservation ablation has much lower mean null MSE
(`0.00000120`) but remains worse on normal/stress affected-region MSE, so this
is a preservation tradeoff rather than a current-best result.

The corrected unaffected-region audit distinguishes target error, output change
relative to the blend, and paired excess target error over identity. The mask
complement is not pure model damage because it includes sub-threshold blend
change, blur, and noise.

Clipping changes macro affected MSE by at most 0.16% and does not change model
rankings. Ten of 6,000 per-sample model rows gain more than `0.0001` absolute
affected MSE from clipping, but none gains more than 10% relatively.

## Remaining Caveats

- Controlled synthetic Galaxy10 DECaLS-style blends only.
- No full real-sky survey validation.
- Full apparent-size-normalized benchmark has not yet been run.
- Foreground extraction and halo masking are approximate.
- Stress evaluation is intentionally skewed toward high-overlap cases.
- Core-obstructed pixels remain the hardest region.
- Per-sample winners vary.
- Historical train/validation/test rows are not duplicate-object disjoint.
- Galaxy10 DECaLS inputs are RGB display cutouts, not calibrated FITS flux
  images.

## Safe Claims

Safe claims:

- Thayer-BR v0.2 Moderate is the current best model family on the controlled
  synthetic development benchmark; the grouped retrain is the defensible
  development reference.
- Moderate affected/core-weighted residual loss improves aggregate affected and
  core metrics relative to Thayer-BR v0.1.
- Evaluation-seed variation supports the v0.2 development ranking; the earlier
  mask-sensitivity audit supports the v0.1 lead among methods it included.
- The result should be interpreted within the tested Galaxy10 DECaLS-style
  synthetic setting.

## Claims to Avoid

Avoid claiming:

- validated survey-grade deblending performance;
- general performance on arbitrary real-sky images;
- universal per-sample dominance;
- that apparent-size variation is irrelevant;
- that low MSE always corresponds to the visually preferred reconstruction;
- independent training-seed robustness;
- a leakage-cleared final-paper effect size;
- that identity or threshold is a competitive astronomical deblender;
- survey-grade or calibrated-flux performance from RGB display cutouts.
