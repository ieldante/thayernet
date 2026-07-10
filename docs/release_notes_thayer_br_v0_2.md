# Release Notes: Thayer-BR v0.2 Moderate

> **Historical release status (2026-07-10):** preserve these metrics as original
> row-split development results. A later grouped retrain achieved `28.81x`
> normal affected-MSE reduction and is the defensible development reference.
> Neither result is final, and the multi-seed audit did not retrain the model.

## Status

This historical release introduced the current-best Thayer-Net model family for
the controlled synthetic Galaxy10 DECaLS-style development benchmark. The later
grouped checkpoint is the defensible development reference.

## Main Change

The model keeps the balanced residual U-Net formulation from Thayer-BR v0.1 and
adds a moderate affected/core-weighted residual loss:

- affected extra weight: `3`;
- core affected extra weight: `2`;
- residual reconstruction: `blended - predicted_residual`.

## Main Result

| Original development evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Lower affected MSE vs identity |
| --- | ---: | ---: | ---: |
| Normal | 0.068122 | 0.002108 | ~32.3x |
| Hard stress | 0.075541 | 0.003847 | ~19.6x |

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
