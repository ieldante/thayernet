# Normalized conformal scale protocol

## Frozen procedure

For each applicable image or flux risk, define the raw absolute residual and
normalized score:

```text
residual_i = abs(true_risk_i - predicted_risk_i)
score_i = residual_i / max(predicted_scale_i, scale_floor)
```

Natural calibration remains a distinct partition. Connected source groups are
assigned to one of five folds. Each fold is evaluated using the finite-sample
90% score order statistic from the other four folds:

```text
rank = min(n, ceil((n + 1) * 0.90))
upper_i = predicted_risk_i + score_quantile * predicted_scale_i
```

This cross-fitted calculation prevents a row from calibrating its own bound.
It is not an exact finite-sample conditional-coverage guarantee. Calibration
outcomes cannot select features, objectives, architectures, pooling strength,
subgroup boundaries, floors, caps, or gates.

## Required reporting

Report marginal, every frozen supported subgroup, and
low-SNR/high-obstruction coverage; median, p90, p95, difficult-regime, and
worst-group widths; tail miss rate; floor/cap activation; extreme inflation;
score uniqueness; five-seed variation; and connected-source-component
bootstrap intervals. Physical subgroup labels are evaluation-only. A hard
oracle-group correction must be labeled non-deployable and cannot determine a
PASS.

The 2026-07-12 campaign produced image/flux marginal coverage
`0.9189`/`0.9218` and worst supported coverage `0.549`/`0.679`. The protocol
therefore did not authorize a full policy campaign.

