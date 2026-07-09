# Release Notes: Thayer-BR v0.2 Moderate

## Status

Thayer-BR v0.2 Moderate is the current best Thayer-Net model for the controlled
synthetic Galaxy10 DECaLS-style deblending benchmark.

## Main Change

The model keeps the balanced residual U-Net formulation from Thayer-BR v0.1 and
adds a moderate affected/core-weighted residual loss:

- affected extra weight: `3`;
- core affected extra weight: `2`;
- residual reconstruction: `blended - predicted_residual`.

## Main Result

| Evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Improvement |
| --- | ---: | ---: | ---: |
| Normal held-out | 0.068122 | 0.002108 | ~32.3x |
| Hard stress test | 0.075541 | 0.003847 | ~19.6x |

Multi-seed audit:

- Normal: `32.02 +/- 1.21x`.
- Stress: `19.55 +/- 0.30x`.

## Comparison to Thayer-BR v0.1

- Normal affected MSE: `0.002451 -> 0.002108`.
- Stress affected MSE: `0.004587 -> 0.003847`.
- Stress core MSE: `0.013848 -> 0.009533`.
- Normal worse-than-identity: `1/1000 -> 0/1000`.
- Stress worse-than-identity: unchanged at `0/1000`.

## Ablation

Thayer-BR v0.2 Strong used affected/core extra weights `5/4`. It slightly
improved stress core MSE relative to Moderate but worsened aggregate normal,
aggregate stress, and stress non-core affected MSE. It should be reported as an
ablation, not as the current best model.

## Documentation Added

- `docs/releases/thayer_br_v0_2.md`
- `docs/model_card_thayer_br_v0_2.md`
- `docs/evaluation_audit_summary.md`
- `docs/current_best_model.md`

## Caveat

This is a controlled synthetic benchmark result. It should not be presented as
validated real-sky or survey-grade deblending.
