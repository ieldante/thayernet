# Model Card: Thayer-BR v0.2 Moderate

## Model Identity

- Model name: Thayer-BR v0.2 Moderate.
- Model family: Thayer-Net.
- Task: controlled synthetic galaxy deblending.
- Architecture: compact U-Net encoder-decoder with skip connections.
- Formulation: residual prediction.
- Training objective: affected/core-weighted residual MSE.
- Current status: current best model for the controlled synthetic benchmark.

## Intended Use

Thayer-BR v0.2 Moderate is intended for research and educational analysis of a
controlled synthetic galaxy deblending benchmark. The task is to reconstruct a
known clean target galaxy from a synthetic blend of a target and contaminant
cutout.

The model is useful for comparing objective design, residual prediction,
training distribution, and diagnostic metrics under controlled Galaxy10
DECaLS-style blend generation.

## Not Intended For

Thayer-BR v0.2 Moderate is not intended for:

- production astronomy pipelines;
- arbitrary survey images;
- real-sky deployment without external validation;
- replacing classical deblending tools;
- claims about survey-grade performance.

## Architecture

The model uses the same compact residual U-Net architecture as the balanced
residual experiments. It predicts contaminant residual light rather than drawing
the clean target directly.

Plain-text formulation:

```text
true_residual = blended - target
predicted_residual = model(blended)
reconstruction = blended - predicted_residual
```

The v0.2 Moderate change is the weighted residual loss:

```text
loss = sum(weight_map * (predicted_residual - true_residual)^2) / sum(weight_map)
```

The affected mask is computed from the known synthetic target/blend pair:

```text
affected_mask = mean(abs(blended - target), channel) > 0.02
```

Moderate weighting uses:

- background/base weight: `1.0`;
- affected extra weight: `3.0`;
- affected target-core extra weight: `2.0`.

The target-core term gives additional emphasis to affected pixels overlapping a
central/bright target-core mask. The normalization by summed weights prevents
larger affected masks from trivially increasing the loss scale.

## Training Data

- Source dataset: Galaxy10 DECaLS RGB cutouts.
- Blend type: synthetic foreground-only two-galaxy blends.
- Split policy: original images are split before blending.
- Training blends: 12,000.
- Validation blends: 1,000.
- Batch size: 8.
- Epochs: 20.
- Training distribution:
  - 50% normal/random blends;
  - 30% high-overlap/core-obstruction blends;
  - 20% brightness/size stress blends.

The weighted v0.2 Moderate run is stored under
`outputs/runs/weighted_residual_20260709_030245`, with the best checkpoint at
`outputs/checkpoints/unet_residual_weighted_br_20260709_030245_best.pth`.

## Evaluation Data

Evaluation used controlled synthetic blends generated from held-out source
images:

- normal held-out blends;
- hard stress-test blends;
- multi-seed synthetic evaluation;
- apparent-size and visual audit sets.

These evaluations have known clean targets. They should be interpreted as
controlled benchmark evidence, not as real-sky validation.

## Metrics

Reported metrics include:

- affected-region MSE and MAE;
- whole-image MSE and MAE;
- PSNR;
- SSIM;
- core affected MSE;
- non-core affected MSE;
- halo-band error;
- improvement ratio versus identity;
- worse-than-identity count;
- model win rates.

Affected-region MSE is the primary metric because most image pixels are
unchanged in each synthetic blend.

## Results

Current best model versus identity:

| Evaluation | Identity affected MSE | Thayer-BR v0.2 Moderate affected MSE | Improvement |
| --- | ---: | ---: | ---: |
| Normal held-out | 0.068122 | 0.002108 | ~32.3x |
| Hard stress test | 0.075541 | 0.003847 | ~19.6x |

Multi-seed audit:

| Evaluation | Improvement mean +/- std |
| --- | ---: |
| Normal | 32.02 +/- 1.21x |
| Stress | 19.55 +/- 0.30x |

Comparison to Thayer-BR v0.1:

| Metric | Thayer-BR v0.1 | Thayer-BR v0.2 Moderate |
| --- | ---: | ---: |
| Normal affected MSE | 0.002451 | 0.002108 |
| Stress affected MSE | 0.004587 | 0.003847 |
| Stress core MSE | 0.013848 | 0.009533 |
| Normal worse-than-identity | 1/1000 | 0/1000 |
| Stress worse-than-identity | 0/1000 | 0/1000 |

Approximate v0.2 Moderate improvements over v0.1 are 14% lower normal affected
MSE, 16% lower stress affected MSE, and 31% lower stress core MSE.

## Robustness Checks

Evaluation/audit checks include:

- affected-mask correctness: mask computed from blend/target difference;
- mask-threshold sensitivity at `0.005`, `0.01`, `0.02`, and `0.04`;
- mask-dilation sensitivity at radii `0`, `1`, `3`, `5`, and `9`;
- residual sign and reconstruction logic checks;
- multi-seed normal and stress evaluation;
- checkpoint size and modified-time integrity checks;
- apparent-size ratio audit;
- centrality/core-obstruction audit;
- halo-band error audit;
- visual-vs-metric disagreement grid selection.

## Limitations

- The benchmark is synthetic and controlled.
- The data distribution is Galaxy10 DECaLS-style, not arbitrary survey imagery.
- Full apparent-size normalization has not yet been run.
- Foreground extraction and halo masking are approximate.
- Core-overlap cases remain the hardest examples.
- Some per-sample outputs show broad low-level artifacts.
- Stronger core weighting was not better overall, showing that the objective
  can over-focus on core pixels.

## Ethical and Scientific Use

These results should be described as controlled benchmark evidence. They support
the conclusion that residual prediction, hard-case balanced training, and
moderate affected/core-weighted loss improve synthetic deblending performance
under the tested setup.

They should not be used to claim validated performance on real crowded fields,
production survey images, or general astronomical source-separation workloads.
