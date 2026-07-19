# Thayer-Select conditional-calibration experiment

## Scope and outcome

The prospective run
`outputs/runs/thayer_select_conditional_calibration_20260712_021556/` is complete
with **FAILURE**. It used only the authoritative feasibility training,
validation, and natural-calibration partitions. Condition C stayed byte
identical, no reconstruction inference or fine-tuning occurred, and all new
risk and scale heads ran on CPU. No development manifest or scene was created
or accessed, and the lockbox remained sealed.

The preregistration SHA-256 was
`95a67082acbe0921af7db64f3d78c5280d18f58442f71b0de16364228bc8494d`.
All gates were audited for mathematical attainability before fitting. The
catastrophic AUPRC threshold used 75% of the remaining achievable gap rather
than a prevalence multiplier.

## Reproduction and support

Exact float32 replay of the frozen feasibility CPU heads reproduced the
original image/flux subgroup minimum of `0.691429`. Every frozen physical
subgroup and intersection passed the minimum support requirement of 100
calibration rows and 80 distinct source groups, so no additional calibration
scenes were generated. The lowest selected groups were supported:

| Risk | Lowest supported subgroup | Coverage | Rows |
| --- | --- | ---: | ---: |
| image | low SNR + high obstruction | 0.637 | 193 |
| flux | low SNR + high obstruction | 0.684 | 193 |
| centroid | intermediate source size | 0.888 | 921 |

## Capacity and calibration results

The fixed capacity ablation compared a linear head, a one-hidden-layer
64-unit MLP, and a two-layer residual 64-unit MLP over the unchanged
F_COMBINED representation. Five seeds were used. Selected natural-calibration
Spearman means were `0.870` for image, `0.862` for flux, and `0.952` for
centroid, so rank transfer remained strong.

The frozen selection chose group-conditional normalized conformal with the
small MLP for image, group-conditional normalized conformal with the residual
MLP for flux, and normalized conformal with the small MLP for centroid.
Marginal coverage was `0.903`, `0.898`, and `0.901`, respectively. Centroid
passed its component gates with worst supported coverage `0.888`; image and
flux failed with minima `0.637` and `0.684`. Group-conditional correction did
not repair their low-SNR/high-obstruction failure.

Median and 95th-percentile widths were bounded, but rare exponential-tail
bounds still dominated means. That tail behavior remains a limitation even
for the formal centroid PASS. No exact conditional-coverage guarantee is
claimed; physical source variables were audit-only and never deployable
calibrator inputs.

## Decision

A full hierarchical-policy campaign is not authorized. The one next
experiment is a separately preregistered train/validation/calibration-only
partially pooled deployable scale-model correction using model-accessible
features, robust heavy-tail loss, and frozen pooling/shrinkage. It must reuse
the same physical subgroup audit and must not generate development data or
access the lockbox.

