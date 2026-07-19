# Forward-consistency contract

Status: frozen feasibility contract from
`outputs/runs/thayer_competing_hypotheses_20260712_131111/`.

## Inputs and forward model

The score receives only an observed g/r/z blend, a candidate full source
decomposition, and the exact BTK observation metadata. Candidate source layers
are PSF-convolved noiseless detected electrons per pixel on the common 60 x 60,
0.2 arcsec/pixel grid. Recomposition is the unclipped sum of every candidate
layer with an explicitly zero background.

BTK adds source Poisson noise and one zero-mean sky Poisson realization after
source layers are summed. For band b and pixel p, the frozen variance is

`variance[b,p] = max(recomposed[b,p] + sky[b], 1)`.

The residual is `observed - recomposed`, and the primary score is the mean
squared residual after division by the square root of this variance. Per-band
means, neighboring-pixel residual correlation, and relative total-flux residual
are mandatory diagnostics. Target truth and candidate true error never enter
the score.

## Calibration and interpretation

Exactly 2,000 approved calibration scenes supplied known-truth decompositions
to define the reference distribution. The frozen limits are:

- global 99th percentile: 1.031580046990072;
- g/r/z 99.5th percentiles: 1.0633776056642712,
  1.061456880656522, and 1.057231166110142;
- absolute relative flux-residual 99th percentile: 0.03435249505297672.

A candidate is plausible only if every applicable limit passes. Forward
consistency is necessary, not sufficient: 49 of 50 noisy Atlas observations
retained two scientifically divergent constructed decompositions. The score
therefore cannot be described as a uniqueness test.

No development or lockbox scene was used.
