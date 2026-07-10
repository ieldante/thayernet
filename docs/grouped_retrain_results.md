# Grouped v0.2 Moderate Retrain Results

## Status

`Thayer-BR v0.2 Moderate Grouped Retrain` completed all 20 epochs and was
evaluated on exact-pixel and exact-coordinate group-disjoint development
manifests. It remains a strong restoration model, but it is not a locked final
result and does not establish training-seed robustness.

Run:
`outputs/runs/br_v02_moderate_grouped_retrain_20260710_110917/`

Best checkpoint:
`outputs/checkpoints/unet_br_v02_moderate_grouped_retrain_20260710_110917_best.pth`

Best-checkpoint SHA-256:
`eea442ff21bdfbdd74815d7b292e786f187dc9a63fea73d4adde98a4b082802b`

Training used MPS, seed 3042, batch size 8, 8,000 grouped training blends,
1,000 grouped validation blends, and the historical 1,927,075-parameter v0.2
U-Net. The loss retained background weight 1 plus affected/core extra weights
3 and 2. Epoch 20 was both best and final:

- train weighted loss: `0.0010825181`
- validation weighted loss: `0.0011635236`
- validation affected MSE: `0.0033365143`

The separately saved best and final checkpoint state tensors are identical;
their metadata correctly records `best_validation` and `final_epoch`.

## Grouped development metrics

Primary values are clipped macro means of per-sample metrics. Identity and
threshold are sanity checks, not competitive astronomical deblenders.

| Suite | Identity affected MSE | Grouped retrain affected MSE | Lower affected MSE vs identity | Core affected MSE | Halo-band MSE | Worse than identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.0668139 | 0.00231890 | 28.8127x | 0.00497364 | 0.000435626 | 0/1000 |
| Hard stress | 0.0725308 | 0.00458983 | 15.8025x | 0.0115079 | 0.000640123 | 3/1000 |
| Compact bright | 0.0801469 | 0.00872771 | 9.18304x | 0.0118618 | 0.000778985 | 2/1000 |
| High core obstruction | 0.0778714 | 0.00491680 | 15.8378x | 0.0123239 | 0.000548833 | 1/1000 |

All 4,000 suite rows replayed exactly on MPS. Output clipping reduced affected
MSE by only about `0.22%` to `0.35%` across suites, so the grouped conclusion is
not materially clipping-dependent. Pre- and post-clipping metrics, predicted
residual signs, and clipping fractions remain in the audit tables.

## Comparison with the historical checkpoint

On the identical grouped manifests, the old row-split checkpoint scored
`32.33x`, `18.15x`, `11.75x`, and `18.43x` on normal, hard, compact-bright, and
high-core. It outperformed the grouped retrain on every aggregate suite.
However, 54.575% of those rows exposed an old training or validation source
group, so this old-checkpoint evaluation is diagnostic rather than
source-independent.

The grouped retrain is about 12% worse in normal affected MSE than the old
checkpoint on the same suite, with larger gaps on some stress suites. That
difference cannot be attributed solely to leakage: the historical model used
12,000 training blends while the requested grouped retrain used 8,000, and only
one grouped training seed was run. The direct duplicate-severity audit also
found that removing implicated historical rows changed ratios by at most about
0.31%.

## Decision

The v0.2 Moderate model family remains the current development leader. For
future scientifically defensible work, use the grouped-retrained checkpoint as
the duplicate-safe development reference. Preserve the original checkpoint and
numbers as historical development evidence; do not promote them as final.

The claim should be softened from an unqualified `32x` headline to the grouped
suite result, led by `28.81x lower affected-region MSE than identity` on normal
grouped development data (about `5.37x` lower RMSE). A fresh untouched final
partition, clustered uncertainty, and independent training seeds are still
required.

