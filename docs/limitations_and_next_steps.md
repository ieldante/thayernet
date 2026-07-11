# Limitations and Next Steps

## Current decision

The grouped v0.2 Moderate retrain and grouped development evaluation are
complete. The retrained checkpoint has 28.8x lower normal and 15.8x lower
hard-stress affected-region MSE than identity on the grouped suites. This is
strong duplicate-safe development evidence, not a locked final-paper result.
The original v0.2 result remains a historical development result; Delta is a
compact/color/preservation tradeoff, and ResUNet v0.4 is a compact/halo
architecture ablation.

The original random-index protocol is not eligible for new claims. The source
audit found 29 pixel-identical pairs crossing train/validation/test, including
27 same-coordinate duplicated objects. The authorized grouped retrain corrected
the observed exact-pixel and exact-coordinate leakage before training. It did
not establish exhaustive near-duplicate identity resolution or final-test
independence.

## Highest-priority corrections

1. Freeze the model/checkpoint list, generator, masks, metrics, clipping policy,
   and reporting rules.
2. Create a fresh untouched final source pool after that freeze. It must be
   group-disjoint from every source group used for grouped training, validation,
   and development testing.
3. Audit exact pixels, exact coordinates, and high-confidence perceptual
   candidates for that final pool. Exact-group disjointness is not proof of an
   exhaustive near-duplicate audit.
4. Manually review the 356-source artifact candidate pool without model scores,
   then freeze versioned artifact-screened-source and artifact-stress flags.
5. Run the predeclared final comparison once and report all suites. Do not infer
   training-seed robustness from evaluation-seed variation or one grouped
   retrain.

The earlier provisional 1,000-source final pool is superseded and not
final-eligible: under the grouped split it maps to 683 train, 173 validation,
and 144 test sources, and the actual grouped train/validation blend manifests
use 499/91 of those sources (590 total). The grouped blend infrastructure itself
contains 8,000 train, 1,000 validation, and four 1,000-row test manifests, with
71/71 integrity checks and 13,000/13,000 exact replays. It is development
infrastructure, not the untouched final pool.

## Model-behavior limitations

- v0.2 has a small aggregate unblended-input error but a meaningful tail:
  3/1,000 null inputs exceed MSE `0.001`, with false subtraction visible around
  bright off-center sources and target structure.
- Delta reduces mean unblended-input MSE by about 22.1x relative to v0.2 and
  lowers paired excess target error over identity in the mask-complement region,
  but worsens normal/stress affected MSE. This is a preservation/perceptual
  tradeoff, not a new best model.
- ResUNet improves compact-bright and halo-band aggregates but does not improve
  the main stress/core gate consistently.
- Clipping has little aggregate effect and does not change rankings, but
  per-sample out-of-range statistics should remain visible.
- Source-artifact heuristics have expected false positives and must not become
  automatic exclusions without review.

## Scope limitations

Galaxy10 DECaLS inputs are RGB display cutouts, not calibrated FITS flux images.
The work studies controlled synthetic restoration of RGB cutouts. It does not
establish survey-grade deblending, calibrated photometry, or source separation
in crowded real fields. Identity and threshold are sanity checks, not strong
astronomical deblenders.

Additional realism work should follow benchmark repair: apparent-size-matched
evaluation, PSF variation, sky/background mismatch, detector artifacts,
correlated environments, and calibrated-data validation.

## Claim boundaries

Safe current wording separates the two development protocols:

> On the original random-index development suites, Thayer-BR v0.2 Moderate has
> 32.3x lower normal and 19.6x lower stress affected-region MSE than identity,
> corresponding to about 5.7x and 4.4x lower RMSE. These are development results
> from a source split with confirmed duplicate leakage. After exact-pixel and
> exact-coordinate grouping and retraining, the corresponding grouped
> development ratios are 28.8x and 15.8x. Neither protocol is an untouched
> final test; a fresh group-disjoint final pool is required for a paper claim.

Do not claim survey readiness, independent-training-seed robustness, a
leakage-cleared final result, or that heuristic artifact flags are ground truth.
