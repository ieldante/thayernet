# Current Best Model: Thayer-BR v0.2 Moderate

> **Development-benchmark status (2026-07-10):** v0.2 remains the current best
> model family. The original `32.3x`/`19.6x` values are historical row-split
> development results. A grouped retrain gives `28.81x`/`15.80x` on grouped
> normal/hard development suites. None is final-paper performance.

Thayer-BR v0.2 Moderate is the current best Thayer-Net model family on the
controlled synthetic Galaxy10 DECaLS-style development benchmark.

## Key Result

| Grouped development suite | Identity affected MSE | Grouped retrain affected MSE | Identity/model affected-MSE ratio |
| --- | ---: | ---: | ---: |
| Normal | 0.066814 | 0.002319 | 28.81x |
| Hard stress | 0.072531 | 0.004590 | 15.80x |
| Compact bright | 0.080147 | 0.008728 | 9.18x |
| High core obstruction | 0.077871 | 0.004917 | 15.84x |

This grouped retrain is the current source-group-disjoint development
reference. One grouped training seed is not training-seed robustness, and a
fresh untouched final-paper test has not been run.

Historical context: the original row-split checkpoint scored `32.3x` normal and
`19.6x` hard. Its evaluation-seed audit gave:

- Normal: `32.02 +/- 1.21x`.
- Stress: `19.55 +/- 0.30x`.

Those values are historical development evidence only. The old checkpoint also
scores better on the grouped manifests, but that diagnostic is exposure-
confounded because `54.575%` of rows contain an old training or validation
source group.

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
- In the earlier v0.1-era comparison, mask-threshold and dilation sensitivity
  were checked and v0.1 remained best; v0.2 was not included.
- Residual sign and reconstruction logic were checked.
- Blend-generation/evaluation-seed variation supports the development result;
  it does not establish training-seed robustness.
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
