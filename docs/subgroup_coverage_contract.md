# Thayer-Select subgroup-coverage contract

This contract governs conditional-calibration claims for IMAGE_RISK,
FLUX_RISK, and CENTROID_RISK. It does not define an accept/abstain policy.

## Frozen analysis groups

Boundaries are tertiles of the frozen 12,000-row valid-risk training
distribution, fixed before calibration fitting:

| Family | First boundary | Second boundary | Labels |
| --- | ---: | ---: | --- |
| SNR proxy | 11.300333 | 28.630012 | low, medium, high |
| core obstruction | 0.002057 | 0.095043 | low, medium, high |
| separation (PSF units) | 1.416034 | 3.277542 | close, intermediate, separated |
| symmetric flux contrast | 1.311524 | 2.667453 | near-equal, moderate contrast, high contrast |
| requested-source size (arcsec) | 0.222665 | 0.373487 | compact, intermediate, extended |

The frozen intersections are low SNR + high obstruction, low SNR + near-equal
flux, and close separation + high obstruction. Boundaries cannot be changed
after viewing coverage.

## Support and claims

A group supports a strong empirical coverage claim only with at least 100
diagnostic calibration rows and 80 distinct source groups. Effective sample
size, a Wilson interval, order-statistic resolution, and failures beyond the
bound must also be reported. Underpowered groups remain visible but are
excluded from strong claims.

SNR, obstruction, separation, flux ratio, and source size are simulator or
source metadata for analysis only. They cannot enter a risk head, scale head,
Mondrian rule, neighbor distance, or inference-time correction. Deployable
calibration grouping must be derived entirely from frozen model-accessible
features and predictions.

Marginal split conformal does not establish conditional coverage. Mondrian,
normalized, local, and cross-fitted empirical subgroup results must be named
precisely; no finite-sample exact conditional guarantee may be claimed unless
its assumptions and group construction actually justify one.

The conditional-calibration campaign found all frozen groups supported, but
image and flux coverage in the low-SNR/high-obstruction intersection was only
`0.637` and `0.684`. Those are adequately supported failures and cannot be
attributed to calibration sparsity.

