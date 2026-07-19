# Frozen Thayer-PU Atlas protocol

Frozen UTC: `2026-07-12T22:03:44.682352+00:00` before matched-control
sampling or any Thayer-PU Atlas inference.

- Selected checkpoint SHA-256: `c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e` (validation-selected epoch 27).
- K=32 truth-free prior samples; seeds: 2026077600 through 2026077631.
- The same scene-level epsilon sample is queried under prompt A and prompt B.
- Posterior samples, target-guided resampling, adaptive rejection, and post-Atlas tuning are prohibited.
- Forward tolerance SHA-256: `a479a94bc1940b5fa146bc1a3eda3aeee6c931c90f25cc3a2108197486833e0a`; calibration-only and already frozen.
- Scientific distances: image 0.25, per-band flux 0.20, color 0.20 mag,
  centroid 0.5 mean-PSF FWHM; complete-linkage cluster cut at primary distance 1.0.
- Candidate diameter is the maximum primary scientific distance among retained
  forward-consistent requested layers; fewer than two retained candidates gives zero.
- Matched control set: first 25 scenes of the frozen Atlas fresh-validation manifest.
  For each K prefix, its control 95th percentile is frozen with `higher`; positive
  classification is strict `diameter > threshold`, yielding the authoritative 4% FPR scale.
- Atlas: exactly the 25 frozen pair IDs and both observation sides (50 observations),
  exact BTK noisy replay, prior only, one execution.
- Coverage requires at least one retained sample within primary distance <=1.0 of
  own truth or the paired alternate requested truth.
- Gates: witnesses >=30/50 and >19/50; AUROC >=0.60 with pair-cluster bootstrap
  95% lower endpoint >0.5; recall at 4% control FPR >=0.10; safe-control false
  witnesses <=0.10; own coverage >=0.70; alternate coverage >=0.30; Atlas forward
  rate >=0.50.
