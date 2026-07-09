# Evaluation Audit Summary

## Why This Audit Was Needed

The newest headline numbers are large: Thayer-BR v0.2 Moderate improves
affected-region MSE by about `32.3x` on normal held-out blends and `19.6x` on
the hard stress test in the same-run comparison. Large improvements need
diagnostic checks because affected-region metrics, mask definitions, residual
sign conventions, and synthetic blend construction can all create misleading
conclusions if implemented incorrectly.

The audits were evaluation-only. They did not train, retrain, or modify saved
checkpoints.

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

The balanced/weighted model ranking remained stable across these thresholds.
This supports the result against the concern that the conclusion depends on one
arbitrary affected-pixel cutoff.

## Dilation and Halo Sensitivity

The audit also dilated affected masks by radii:

```text
0, 1, 3, 5, 9
```

The ranking remained stable under these halo-inclusive masks. This matters
because galaxy outskirts and low-level contaminant light can extend beyond the
default affected region.

## Multi-Seed Stability

The v0.2 Moderate multi-seed audit found:

| Evaluation | Improvement mean +/- std |
| --- | ---: |
| Normal | 32.02 +/- 1.21x |
| Stress | 19.55 +/- 0.30x |

This supports the result as more than a single favorable random evaluation set.

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

## Remaining Caveats

- Controlled synthetic Galaxy10 DECaLS-style blends only.
- No full real-sky survey validation.
- Full apparent-size-normalized benchmark has not yet been run.
- Foreground extraction and halo masking are approximate.
- Stress evaluation is intentionally skewed toward high-overlap cases.
- Core-obstructed pixels remain the hardest region.
- Per-sample winners vary.

## Safe Claims

Safe claims:

- Thayer-BR v0.2 Moderate is the current best model on this controlled
  synthetic benchmark.
- Moderate affected/core-weighted residual loss improves aggregate affected and
  core metrics relative to Thayer-BR v0.1.
- Multi-seed and mask-sensitivity audits support the robustness of the
  controlled benchmark result.
- The result should be interpreted within the tested Galaxy10 DECaLS-style
  synthetic setting.

## Claims to Avoid

Avoid claiming:

- validated survey-grade deblending performance;
- general performance on arbitrary real-sky images;
- universal per-sample dominance;
- that apparent-size variation is irrelevant;
- that low MSE always corresponds to the visually preferred reconstruction.
