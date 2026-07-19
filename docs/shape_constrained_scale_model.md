# Shape-constrained scale model

The frozen Q1 model is a 25-parameter additive q=0.90 quantile model. Each of
the four deployable proxies has five training-quantile knots and a convex
piecewise-linear effect. Its initial slope is parameterized so that the
derivative at and above the `0.50` anchor is nonnegative while a decreasing
lower branch remains possible. Main effects are centered by their training-row
means.

Q2 adds one parameter:

```text
softplus(gamma) * relu(z0 - 0.50) * relu(z1 - 0.50)
```

The multiplication is essential: it creates a nonnegative high-high
interaction with positive mixed finite difference. The earlier subtraction
form would have been additive, non-identifiable with the intercept, and
opposite to the required upper-half behavior.

All models were trained on CPU with raw-residual q=0.90 pinball loss, five
fixed seeds, AdamW, fixed roughness and interaction penalties, and
validation-only early stopping. All finite-difference constraint tests passed.
The fitted interaction was extremely small and did not change validation-cell
coverage or the calibration decision, so Q2 remains an ablation rather than a
recommended model.
