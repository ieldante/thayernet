# Current Best Model: Thayer-BR v0.2 Moderate

Thayer-BR v0.2 Moderate is the current best Thayer-Net model on the controlled
synthetic Galaxy10 DECaLS-style deblending benchmark.

## Key Result

| Evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Improvement |
| --- | ---: | ---: | ---: |
| Normal held-out | 0.068122 | 0.002108 | ~32.3x |
| Hard stress test | 0.075541 | 0.003847 | ~19.6x |

Multi-seed audit:

- Normal: `32.02 +/- 1.21x`.
- Stress: `19.55 +/- 0.30x`.

## Why It Works

Thayer-BR v0.2 Moderate keeps the residual formulation:

```text
true_residual = blended - target
predicted_residual = model(blended)
reconstruction = blended - predicted_residual
```

It improves Thayer-BR v0.1 by weighting errors in contaminated pixels and
affected target-core pixels more strongly than unchanged background pixels. The
moderate weights are affected extra weight `3` and core affected extra weight
`2`.

## What Was Audited

- Affected masks are target/blend based, not prediction based.
- Mask-threshold sensitivity was checked.
- Mask dilation and halo sensitivity were checked.
- Residual sign and reconstruction logic were checked.
- Multi-seed evaluation supports the result.
- Checkpoint integrity checks passed.
- Apparent-size, centrality, halo-band, and visual-vs-metric audits were run.

## Limitations

The result is a controlled synthetic benchmark result. It is not validated
survey-grade deblending. Core-overlap cases remain hardest, apparent-size
normalization has not yet been run as a full benchmark, and some individual
v0.2 Moderate examples show broad low-level artifacts despite improved aggregate
halo-band metrics.

## Detailed Documentation

- [Release summary](releases/thayer_br_v0_2.md)
- [Model card](model_card_thayer_br_v0_2.md)
- [Evaluation audit summary](evaluation_audit_summary.md)
- [Results interpretation](results_interpretation.md)
- [Methodology](methodology.md)
- [Paper plan](paper_plan.md)
