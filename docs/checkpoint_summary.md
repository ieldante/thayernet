# Checkpoint Summary

Thayer-Net is a compact U-Net-based research testbed for controlled synthetic
galaxy deblending. It uses Galaxy10 DECaLS cutouts, creates foreground-only
synthetic blends with known clean targets, and evaluates whether learned models
can recover the target galaxy better than simple baselines.

## What Checkpoint 1 Proved

The first direct-reconstruction U-Net maps `blended -> target`. On normal
held-out synthetic blends, it substantially improves affected-region error over
identity and threshold baselines. This showed that the model learns useful
contaminant removal in the controlled setting.

## What Stress Testing Showed

The hard stress test concentrates smaller shifts, brighter contaminants,
similar-or-larger contaminant sizes, and target-core obstruction where possible.
The direct U-Net still beats identity, but affected-region MSE improvement drops
from about 14.13x on normal held-out blends to about 8.04x on the stress set.
This confirmed that the normal held-out score was not enough to characterize
hard overlap behavior.

## What Residual Prediction Improved

The residual U-Net predicts the contaminant/residual layer,
`blended -> blended - target`, and reconstructs with
`blended - predicted_residual`. It improves aggregate affected-region MSE on both
normal and stress tests, with the clearest benefit on stress cases.

| Model | Normal affected MSE | Normal improvement | Stress affected MSE | Stress improvement |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.062555 | 1.00x | 0.075541 | 1.00x |
| Direct U-Net | 0.004428 | 14.13x | 0.009390 | 8.04x |
| Residual U-Net | 0.004039 | 15.49x | 0.007069 | 10.69x |

Residual prediction reduced stress worse-than-identity cases from 13/1000 to
0/1000 and beat direct reconstruction on 667/1000 stress cases. It is not
universally better: direct reconstruction still wins on some individual samples
and has slightly better affected-region MAE on the normal aggregate.

## Limitations

These results are for controlled synthetic blends and may change under more
realistic sky simulations or different blend-generation settings. The current
pipeline does not yet model all survey conditions, including PSF variation, sky
background mismatch, detector artifacts, source crowding, and physically
correlated galaxy environments.

## Recommended Next Checkpoint

The next checkpoint should test residual prediction on a core-obstruction-
balanced hard-case dataset, possibly with an affected-region-weighted loss. It
should also add stronger foreground-extraction diagnostics and more realistic
noise/background controls before any broader claims about astronomical
deblending.
