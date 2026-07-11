# Research Correctness Audit

## Verdict

The pipeline is sufficiently consistent to support a controlled, grouped
**development** benchmark after the corrections in the 2026-07-10 audit. The
affected-region arithmetic, residual sign convention, model inputs, training
loop, aligned evaluation, grouped source roles, and exact manifest replay all
passed their checks. The old `32x` result remains numerically plausible, but it
must remain labeled an original row-index development result.

One blocker remains for a final-paper claim: the earlier provisional final pool
is no longer independent. Of its 1,000 sources, 499 were used by the grouped
training blends and 91 by grouped validation. That pool is preserved but
superseded. A fresh untouched group-disjoint source partition must be allocated
after the model and analysis protocol are frozen.

The full private report is under
`outputs/runs/research_correctness_audit_20260710_092241/`.

## What was verified

- Dataset: 17,736 `256 x 256 x 3` `uint8` Galaxy10 DECaLS display-RGB cutouts.
  The HDF5 contains class labels, RA, Dec, redshift, and pixel scale; it has no
  stable object-ID field or channel-order attribute. No local
  `Galaxy10_DECals_NoDuplicated.h5` file was found.
- Model input: only blended RGB is passed to the network. Targets, masks,
  source IDs, and blend parameters are not model inputs.
- Residual semantics: the target is `blended - target`, and reconstruction is
  `blended - predicted_residual`. This is a blend-to-target correction field,
  not guaranteed pure contaminant flux.
- Metrics: 29 deterministic unit and spot checks passed, including independent
  tiny-array arithmetic, prediction-independent affected masks, empty-mask
  handling, clipping separation, Delta E range checks, and sample-ID-aligned
  win rates.
- Replay: 13,000 grouped blend rows replayed exactly, including blend,
  affected-mask, core-mask, and halo-mask hashes.
- Accelerator safety: grouped training and all tensor inference used MPS. Full
  runs refuse silent CPU fallback.
- Checkpoint safety: all 16 checkpoints that predated grouped training retained
  their exact size, nanosecond mtime, and SHA-256. The grouped best and final
  checkpoints were written to new timestamped paths.

## Leakage finding and severity

The historical seed-42 row split was disjoint by HDF5 row but not by source
identity. The audit confirmed 29 pixel-identical and 27 exact-coordinate
cross-split pairs, implicating 57 unique sources (`0.321%` of 17,736).

Only 13/1,000 historical normal rows and 12/1,000 historical stress rows used an
implicated target or contaminant. Removing them changed the affected-MSE
improvement ratios by no more than about `0.31%`. The measured aggregate effect
is therefore minor; the protocol defect is major because object independence
was not enforced.

The old checkpoint was also evaluated on the new grouped tests as a diagnostic.
Because the dataset was repartitioned, 54.575% of those test rows contained a
source group exposed to that checkpoint's historical train/validation pools.
On the 45.425% clean-neither subset, its affected-MSE ratios remained `31.53x`
normal, `18.18x` hard, `11.68x` compact-bright, and `18.27x` high-core. This
supports plausibility, not a source-independent claim.

## Grouped correction

`data/manifests/grouped_source_split_20260710_100907/` groups exact pixel hashes
and exact coordinates before assigning train, validation, and development-test
partitions. It contains 12,417/2,660/2,659 sources and has zero cross-split
source, group, exact-pixel, or exact-coordinate overlap. The grouping does not
prove exhaustive near-duplicate identity.

`data/manifests/grouped_blends_20260710_103233/` contains 8,000 training, 1,000
validation, and four 1,000-row development-test suites. Both target and
contaminant stay inside their assigned source partition. All 71 integrity
checks and all 13,000 exact replay checks passed.

## Generator and benchmark limits

The generator is internally deterministic and replayable, but it is a
computer-vision-style synthetic RGB restoration benchmark, not calibrated
FITS-band physical injection. Important retained limitations are target
centrality, target-only blur, material composite clipping in bright suites,
padded-mask size-ratio compression, ignored pixel-scale mismatch, simplified
noise/PSF behavior, and repeated source groups across blend rows. Statistical
intervals should cluster by target/contaminant group rather than treating all
rows as independent sources.

## Claim status

- Keep the old `32.3x` normal and `19.6x` hard numbers only as original
  development-split context.
- Use the grouped retrain's `28.81x` normal result as the more defensible current
  development estimate, alongside all stress-suite results.
- Do not call either result final, survey-ready, or training-seed robust.
- Do not use the superseded provisional final manifests.
- Freeze a fresh final partition before any final-paper evaluation.

