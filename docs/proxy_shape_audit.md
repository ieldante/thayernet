# Deployable proxy-shape audit

The authoritative audit used all 12,000 training rows and persisted OOF
central predictions. Absolute residuals were reconstructed as
`abs(true_risk - persisted_OOF_central_prediction)`. Proxy definitions and
order were unchanged, sample IDs aligned exactly, and all inputs were finite.

The q=0.90 endpoint values were:

| Risk | Proxy | Lowest decile | Highest decile |
| --- | --- | ---: | ---: |
| Image | z0 estimated low local signal | 9.482565 | 1.476295 |
| Image | z1 estimated local complexity | 7.205539 | 1.454962 |
| Flux | z0 estimated low local signal | 16.811191 | 11.403268 |
| Flux | z1 estimated local complexity | 15.070414 | 10.193126 |

These reversals reject a globally monotone-increasing relationship. They
support convex main effects that can decrease before an interior minimum and
increase in the upper half. Physical subgroup membership was not used to
choose terms, knots, constraints, penalties, or models.

The complete decile, quantile, support, source-group, floor/saturation, and
z0-by-z1 tables are under the timestamped run. No validation, calibration,
development, or lockbox outcome informed the audit.
