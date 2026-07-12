# Hierarchical recoverability contract

Status: preregistered before hierarchical scene generation, feature extraction,
or head training.

Recoverability is not a binary training label. For `UNIQUE_VALID` prompts it is
derived from separate continuous empirical risks and a separate confusion
event. NULL and AMBIGUOUS rows have no valid-query risk target and are excluded
from every valid-risk loss.

All empirical risks are measured after inverse normalization in the simulator's
physical electron-count image units. Model heads receive only frozen
model-accessible features; truth images enter target construction and evaluation
only.

## IMAGE_RISK

For requested truth image `T` and frozen reconstruction `P`, over all bands and
pixels:

`IMAGE_RISK = sqrt(mean((P - T)^2) / mean(T^2))`.

If `mean(T^2)` is nonfinite or nonpositive, risk is infinite and the scene fails
closed. The training target is `log1p(IMAGE_RISK)`; reports retain both raw and
transformed values.

Limits are 0.40 strict, 0.75 moderate, and 1.25 permissive. This is a
requested-source NRMSE: 0 is exact and 1 means error RMS equals truth RMS.

## FLUX_RISK

For each band `b` in g, r, z:

`FLUX_RISK_b = abs(sum(P_b) - sum(T_b)) / max(abs(sum(T_b)), FLOOR_b)`.

`FLOOR_b` is frozen from Dataset-R training only as 0.1% of the median absolute
requested-source band flux, separately by band, and must be finite and positive.
The aggregate is `FLUX_RISK = max(g, r, z)`; all band values remain stored.
The head target is `log1p(FLUX_RISK)`. A nonfinite band makes the aggregate
infinite and fails closed.

Aggregate limits are 0.30 strict, 0.50 moderate, and 1.00 permissive. The floor
prevents nearly empty bands from creating arbitrarily large ratios while
remaining negligible for ordinary sources.

## CENTROID_RISK

Centroids use nonnegative integrated band weights:

`W(x,y) = max(sum_b image_b(x,y), 0)`.

`CENTROID_RISK_PIXELS` is Euclidean distance between the P and T centroids.
`CENTROID_RISK_PSF = CENTROID_RISK_PIXELS / 4.0666666667`. If either positive
weight sum is empty or nonfinite, risk is infinite and fails closed. The head
target is `log1p(CENTROID_RISK_PIXELS)`.

Pixel limits are 1.0 strict, 2.0 moderate, and 3.0 permissive, corresponding to
approximately 0.246, 0.492, and 0.738 mean-PSF FWHM.

## CONFUSION_RISK

For alternate isolated truth `A`, confusion is the binary event:

`mean((P - A)^2) < mean((P - T)^2)`.

Ties are not confusion. Nonfinite reconstruction output fails closed as
confusion. Confusion has a separate binary head and is an automatic empirical
policy failure; it is never folded into a continuous regression target.

## Applicability and missing values

- Continuous and confusion risks apply only to UNIQUE_VALID rows with a unique
  matched source and an available alternate source.
- NULL and AMBIGUOUS rows have applicability mask 0 and are absent from these
  losses. Their hidden reconstructions may be summarized for gate features but
  are never exposed when the query gate abstains.
- An applicable row with a missing/nonfinite empirical metric is retained in
  provenance, marked invalid for regression fitting, and treated as automatic
  failure by policy evaluation. It is never silently imputed as low risk.

## Derived policy violation

For a selected limit set:

```text
VIOLATION = max(
    IMAGE_RISK / IMAGE_LIMIT,
    FLUX_RISK / FLUX_LIMIT,
    CENTROID_RISK_PIXELS / CENTROID_LIMIT
)
```

`CONFUSION_RISK = true` maps violation to infinity. Each component remains
separate in every manifest, table, calibration record, and policy decision.
Optional color, size, and ellipticity risks are diagnostic-only in this
campaign; they cannot enter the primary policy without a later stability and
applicability preregistration.

For success-gate and rejection-curve reporting, a `CATASTROPHIC_VALID_FAILURE`
is preregistered as either `CONFUSION_RISK = true` or finite moderate
`VIOLATION >= 2`. This tail label is evaluation-only: no head is trained on it.
