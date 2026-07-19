# Thayer-Select partially pooled scale correction

## Scope and outcome

The prospective run
`outputs/runs/thayer_select_scale_correction_20260712_024957/` completed with
**FAILURE**. It used only the frozen 12,000-row risk-training partition, the
2,000-row risk-validation partition, and 2,800 UNIQUE_VALID natural-calibration
rows. Condition C, query semantics, risks, selected risk-head families, source
partitions, and physical subgroup boundaries remained frozen. No development
or lockbox data was generated, opened, rendered, or evaluated.

The preregistration SHA-256 is
`4d2d6701e3cfe0847a0b88bb5ae04ca8f3ef514ce8747e3b123670bb57c80d96`.
All gates were audited as attainable before fitting. The exact prior selected
results reproduced before any scale fit: image marginal/worst coverage
`0.9029`/`0.6373`, flux `0.8982`/`0.6839`, and centroid
`0.9007`/`0.8882`.

## Leakage control and deployable features

The authoritative training risk predictions were in-sample. Five deterministic
connected-source-component folds therefore refit the exact selected head form
on four folds and predicted only the held-out fold. Only concatenated held-out
absolute residuals trained scale models. Validation selected objectives and
models; natural calibration was retained for source-group-cross-fitted
normalized conformal correction and final diagnostics.

Deployable S0-S4 features contained frozen risk/query outputs, frozen global
and prompt-local latents, frozen reconstruction summaries, and observed-blend
quality proxies. No true SNR, obstruction, separation, flux ratio, source ID,
source truth, generator difficulty, morphology, or physical subgroup label
entered a deployable array. Existing features made neural extraction
unnecessary, so no reconstruction inference occurred.

## Results

O0 Huber loss on log absolute residual was selected for both risks. Small
nonlinear models predicted validation residual scale better than log-linear
models, but this did not transfer into the required difficult-subgroup
coverage.

| Risk | Marginal coverage | Worst supported coverage | Low-SNR/high-obstruction | Median-width inflation | p95 width | Calibration Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image | 0.9189 | 0.5492 | 0.5492 | 1.336x | 2.846 | 0.877 |
| Flux | 0.9218 | 0.6788 | 0.6788 | 1.055x | 13.595 | 0.866 |

Image coverage worsened by `0.0881` relative to the authoritative failure and
flux changed by `-0.0052`. Source-component-bootstrap 95% lower bounds were
`0.477` and `0.614`, below the frozen `0.75` stability gate. Seed variability,
scale caps, p95 widths, and ranking were acceptable; coverage transfer was not.

The non-deployable hard physical-group oracle reached worst-subgroup coverage
`0.914` image and `0.905` flux, but also broke the marginal `[0.88, 0.92]` gate
and cannot determine success. The regime dependence is real, but the tested
continuous deployable proxies did not recover the required correction.

## Decision

IMAGE_RISK FAIL, FLUX_RISK FAIL, CENTROID_RISK remains PASS, and overall
**FAILURE**. No full hierarchical-policy campaign is authorized. Exactly one
corrective experiment is recommended, not run here: a separately
preregistered train/validation/calibration-only monotone quantile scale model
using a shape-constrained additive function of the same four deployable
proxies, with the same out-of-fold targets, gates, and natural-calibration
audit.
