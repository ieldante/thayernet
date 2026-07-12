# Hierarchical safety policy

Status: decision structure and selection rules preregistered before head
training. Numeric fitted parameters are frozen from validation and natural
calibration before any fresh development scene is generated.

## Inputs and heads

The frozen Condition-C reconstructor supplies model-accessible global,
multi-scale prompt-local, and reconstruction-summary features. A three-class
query head predicts calibrated probabilities `p_unique`, `p_null`, and
`p_ambiguous`. UNIQUE_VALID rows alone train separate image-, flux-, and
centroid-risk median and q=0.90 upper-quantile heads plus a binary confusion
head.

Multinomial logistic regression and one small MLP are the only query candidates.
Linear quantile regression and one small MLP quantile regressor are the only
risk candidates. Validation selects families; natural calibration only fits
vector/temperature scaling, conformal residual offsets, and operational
thresholds. Isotonic is diagnostic-only.

## Calibration and thresholds

- The query gate uses vector scaling when all three parameters are identifiable;
  otherwise one temperature is fitted. Score resolution, ECE, log loss, and
  per-class reliability are reported.
- Each q=0.90 risk prediction receives a split-conformal upper-residual offset
  at miscoverage alpha=0.10 using applicable UNIQUE_VALID rows from natural
  calibration only. Marginal empirical coverage and interval width are reported
  overall and by scene regime.
- Query thresholds are chosen from the monotone empirical candidate family on
  natural calibration to maximize UNIQUE_VALID acceptance subject to NULL
  false acceptance <= 5% and AMBIGUOUS false acceptance <= 10%. Ties choose
  lower combined invalid acceptance, then stricter thresholds. If no nonzero
  candidate satisfies both constraints, the gate is degenerate and the
  campaign cannot succeed.
- The confusion-probability operational limit is 0.20. Temperature scaling is
  fitted on natural calibration; no development data may change it.
- The primary metric limits are moderate: image 0.75, maximum per-band flux
  error 0.50, and centroid error 2.0 pixels. Strict and permissive policies are
  sensitivity analyses, not alternate post-hoc success definitions.

Conformal risk control is evaluated only if exchangeability, fixed score
function, and loss boundedness are documented. Otherwise it is omitted rather
than presented as a guarantee.

## Accept or abstain

A scene is accepted only when every gate passes:

1. `p_unique >= T_unique`;
2. `p_null <= T_null`;
3. `p_ambiguous <= T_ambiguous`;
4. calibrated IMAGE_RISK upper bound `< 0.75`;
5. calibrated FLUX_RISK upper bound `< 0.50`;
6. calibrated CENTROID_RISK upper bound `< 2.0 pixels`;
7. calibrated predicted confusion probability `< 0.20`.

Every other scene abstains. Hidden reconstructions for NULL or AMBIGUOUS
queries are never exposed after gate abstention.

The reporting-only recoverability margin is the minimum of dimensionless gate
margins:

```text
(p_unique - T_unique) / (1 - T_unique)
(T_null - p_null) / T_null
(T_ambiguous - p_ambiguous) / T_ambiguous
(IMAGE_LIMIT - IMAGE_UPPER) / IMAGE_LIMIT
(FLUX_LIMIT - FLUX_UPPER) / FLUX_LIMIT
(CENTROID_LIMIT - CENTROID_UPPER) / CENTROID_LIMIT
(0.20 - p_confusion) / 0.20
```

Zero denominators fail closed. This scalar is never trained directly.

## Freeze boundary

Before development generation, one freeze record hashes the feature family,
query head, risk heads, confusion head, scalers, conformal offsets, thresholds,
scientific limits, metric code, and evaluation code. Development is evaluated
once. Seeing development results cannot authorize retuning.
