# Methodology

Thayer-Net is a controlled synthetic galaxy deblending benchmark. It starts
from Galaxy10 DECaLS cutouts, creates synthetic blends with known clean targets,
and evaluates whether compact U-Net models can reconstruct the target better
than simple baselines.

## Dataset

The dataset is Galaxy10 DECaLS, stored locally as an HDF5 file. The dataset is
not included in the repository; local copies belong under `data/`, which is
ignored except for `data/.gitkeep`.

Dataset rows are shuffled with seed 42 and split 70/15/15 into train,
validation, and test arrays before synthetic blending. This prevents reuse of
the same row index across arrays, but it does **not** guarantee object-level
independence when the HDF5 file contains duplicate or near-duplicate cutouts.
A later RA/Dec and image-fingerprint audit confirmed that duplicated source
objects cross the historical random-index partitions. Those partitions must be
treated as development splits, not a leakage-cleared final test.

The corrected grouped protocol computes exact raw-pixel hashes and uses exact
RA/Dec coordinates when available, unions every exact-pixel or exact-coordinate
match into one source group, and assigns each group wholly to train,
validation, or test. The resulting source and blend manifests verify zero
source-index, source-group, exact-pixel, exact-coordinate, and cross-role
overlap between partitions. High-confidence near-duplicate grouping remains a
separate conservative review step; morphology-only lookalikes are not merged
automatically.

Galaxy10 DECaLS images are RGB display cutouts, not calibrated FITS flux
images. The experiments therefore measure synthetic restoration of RGB
cutouts; they do not establish survey-grade flux/source separation. Although
the HDF5 file includes a pixel-scale field, the current generator does not use
it to normalize angular size or PSF. Apparent-size operations and evaluation
therefore occur in cutout-pixel coordinates.

## Synthetic Blend Generation

Each blend uses two normalized RGB cutouts from the same split:

- `target`: the clean supervised image to reconstruct.
- `contaminant`: a second galaxy whose foreground light is added to the target.

The generator records shift, brightness, blur, noise, and size-ratio metadata.
Rotation is disabled in the main formal experiments unless artifact-free
foreground extraction is specifically being tested.

In the grouped protocol, both target and contaminant source indices and group
IDs are saved with every row, along with the sampled parameters, random seed,
generator/code hashes, source hashes, and expected replay hashes. Train blends
draw only from grouped-train sources, validation blends only from
grouped-validation sources, and all evaluation suites only from grouped-test
sources. Exact replay passed for all 13,000 train, validation, and grouped-test
manifest rows.

The synthetic target is normally centered in its source cutout, while the
contaminant is shifted, and foreground extraction preferentially selects a
central component. This may provide a centrality shortcut. Size handling also
compresses some larger apparent sources rather than defining a symmetric,
pixel-scale-aware angular-size experiment. These behaviors are controlled and
replayable, but require centrality-matched and angular/size-normalized follow-up
benchmarks.

## Why Foreground-Only Blending

Adding full contaminant cutouts can create artificial rectangular boundaries and
double-background problems. A model could learn those cutout artifacts instead
of learning deblending behavior.

Thayer-Net therefore estimates contaminant foreground light before blending. The
pipeline estimates a background from border pixels, subtracts it, isolates the
central source, and adds only the foreground component to the target. This is
still synthetic, but it avoids trivial pasted-patch cues and keeps the
reconstruction target well defined.

The compositing operation is computer-vision-style addition in normalized RGB
display space, followed by clipping to the valid image range. It is not
addition of calibrated band flux. Input clipping can be material because
saturated pixels no longer retain linearly separable component information;
that blend-construction effect is distinct from optional clipping of a model's
reconstruction before metric calculation.

## Foreground Extraction and Halo-Aware Masking

Foreground extraction is deliberately conservative:

1. Estimate the image background from border pixels.
2. Subtract the background and clip negative foreground values to zero.
3. Detect the bright source using robust border statistics, Otsu-style
   thresholding, and high-percentile thresholds.
4. Select the connected component nearest the cutout center.
5. Dilate and smooth the source mask so diffuse halos are retained.
6. Apply a soft aperture so the mask fades before the image border.
7. Suppress very faint residual values to limit background leakage.

Halo-aware masking was added because aggressive masking can remove diffuse
galaxy light and make the synthetic contaminant unrealistically sharp.

## Baselines

Baselines define what the learned models must beat:

- Identity baseline: returns the blended image unchanged.
- Threshold baseline: uses simple thresholding and connected-component logic to
  remove bright regions.

The threshold baseline is intentionally simple and often performs worse than
identity because it can remove or segment bright target structure without
reconstructing hidden light. Identity and threshold are sanity checks, not
competitive astronomical deblenders.

## Thayer-Direct

Thayer-Direct is the direct reconstruction formulation:

- Input: `X = blended image`.
- Target: `Y = clean target image`.
- Objective: reconstruct the target directly.

The model is a compact encoder-decoder U-Net with skip connections. Skip
connections help preserve spatial detail while the bottleneck captures broader
context.

## Thayer-Residual

Thayer-Residual is the residual prediction formulation:

- Input: `X = blended image`.
- Target: `R = blended image - target image`.
- Reconstruction: `Y_hat = blended image - predicted_residual`.

This formulation asks the model to learn contaminant signal to remove rather
than redrawing the whole target. It can preserve unchanged target light through
the subtraction path, although Thayer-Direct still wins on some
individual samples.

## Thayer-BR v0.1

Random blends alone may under-sample the cases most relevant to deblending
failure, especially target-core overlap and similar-size bright contaminants.
Thayer-BR v0.1 uses the same residual formulation as Thayer-Residual, but
changes the training distribution to include a balanced mix of normal and
hard-case blends.

Thayer-BR v0.1 is a balanced hard-case residual U-Net trained with:

- 8,000 training blends.
- 1,000 validation blends.
- 50% normal/random blends.
- 30% high-overlap/core-obstruction blends.
- 20% brightness/size stress blends.
- Batch size 8.
- 20 epochs.

This tests whether targeted hard-case sampling improves robustness without
requiring a larger or fundamentally different model. Thayer-BR v0.1 is an
experimental research checkpoint, not a stable public model release.

## Thayer-BR v0.2 Moderate

Thayer-BR v0.2 Moderate keeps the residual U-Net architecture and balanced
hard-case training distribution, but changes the loss. It is the current best
model on the controlled synthetic development benchmark.

The residual formulation is unchanged:

```text
true_residual = blended - target
predicted_residual = model(blended)
reconstruction = blended - predicted_residual
```

The weighted residual objective is:

```text
loss = sum(weight_map * (predicted_residual - true_residual)^2) / sum(weight_map)
```

The affected mask is computed from the known blend and target:

```text
affected_mask = mean(abs(blended - target), channel) > 0.02
```

The moderate loss weights are:

- background/base weight `1.0`;
- affected extra weight `3.0`;
- affected target-core extra weight `2.0`.

This loss emphasizes contaminated pixels and affected target-core pixels while
keeping stable normalization by the summed weight map. Thayer-BR v0.2 Strong
uses larger affected/core extra weights `5/4` and is treated as an ablation
because it improves stress core MSE slightly but worsens aggregate affected
metrics relative to Moderate.

## Metrics

The project reports:

- Whole-image MSE and MAE.
- PSNR.
- SSIM.
- Affected-region masked MSE and MAE.
- Core affected MSE and non-core affected MSE.
- Halo-band MSE and MAE for broad low-level error outside the affected mask.
- Model-improvement ratio: identity affected-region MSE divided by model
  affected-region MSE.
- Worse-than-identity count: number of samples where a model has higher
  affected-region MSE than identity.

Whole-image metrics measure global reconstruction fidelity, but they can be
misleading because most pixels in each synthetic blend are unchanged. Affected
regions are pixels where the blend differs from the target by more than a fixed
threshold after averaging absolute RGB differences. Affected-region metrics
therefore focus evaluation on a prediction-independent **blend-change mask**,
sometimes described more compactly as the correction-field support. It
measures where the generated input differs from the clean target; it does not
assert that every changed pixel is contaminant flux. The mask may include
target-blur, input-clipping, and noise effects and is not a component-specific
contaminant-flux mask.

The historical original-development ratio of `32.3x` means 32.3x lower
affected-region MSE versus identity. Because RMSE is the square root of MSE, it
corresponds to about 5.7x lower affected-region RMSE, not 32.3x lower RMSE. It
is not the grouped-retrain or future final-paper estimate.

Primary masked summaries are macro averages: compute a regional metric for
each sample with a nonempty mask, then average over valid samples. Pooled-pixel
or micro averages are reported separately because they upweight samples with
larger masks. Every core, non-core, and halo summary must include valid-sample
coverage, and empty regions are excluded rather than assigned zero error.

## Apparent-Size Normalization vs Pixel Normalization

All current experiments normalize image pixel values, but they do not normalize
the apparent angular or pixel extent of each galaxy cutout. Pixel normalization
puts intensities on a common numeric scale; apparent-size normalization would
control the measured foreground radius or area of the target and contaminant
before blending.

This distinction matters because size ratio can become a shortcut cue. A model
could partly learn that a larger, off-center, or brighter structure is usually
the contaminant instead of learning a fully general deblending rule. Apparent
size variation is also realistic, so the current benchmark should not be
discarded. The right follow-up is a matched-size evaluation that asks whether
the same model ranking holds when apparent size ratio is controlled.

## Evaluation Robustness Audit

The evaluation audit in `outputs/runs/evaluation_audit_20260708_220833`
checked the current data and metric pipeline without training or modifying
checkpoints.

The original split audit confirmed that row indices are split before blending
and that normal/stress blends draw targets and contaminants only from their
assigned array. It did not test whether different rows represent the same sky
object. The later source-leakage audit found duplicate RA/Dec groups across the
historical random-index partitions, including train-to-test crossings. Thus,
the array-role logic is correct at the row-index level but object-level leakage
is present. Standard historical blend dictionaries also omit global source
indices, so old generated samples cannot be reconstructed source-by-source.

The affected-region mask audit confirmed the formula
`abs(blended - target).mean(axis=-1) > threshold`. This means the mask is based
on where synthetic blending changed the clean target, not on a model's
prediction error.

Threshold sensitivity was tested at `0.005`, `0.01`, `0.02`, and `0.04`.
Thayer-BR v0.1 stayed best across all tested thresholds on both normal and
stress sets. Mask dilation was tested at `0`, `1`, `3`, `5`, and `9` pixels,
and Thayer-BR v0.1 also stayed best under these larger halo-inclusive masks.

The multi-seed audit used three normal blend-generation/evaluation seeds and
three stress blend-generation/evaluation seeds, each with 1,000 generated
blends. It did not independently retrain the model. Thayer-BR v0.1 was the best
learned model for every audited seed. Mean improvement over identity was
`27.04 +/- 1.04x` on normal blends and `15.76 +/- 0.07x` on stress blends.

The later v0.2 Moderate evaluation-seed audit found `32.02 +/- 1.21x` lower
normal affected MSE and `19.55 +/- 0.30x` lower stress affected MSE versus
identity. It supports v0.2 Moderate as the current best development-benchmark
model, not training-seed robustness or a final leakage-cleared estimate.

Residual evaluation logic was also checked: residual targets are
`blended - target`, residual reconstructions are `blended - predicted_residual`,
and clipping is applied after subtraction for metrics and visualization. This
supports the sign convention used for the residual checkpoints.

## Size and Visual Audit

The non-training size/visual audit in
`outputs/runs/size_visual_audit_20260709_102251` regenerated original
row-index development normal and stress blends and evaluated existing
checkpoints only. It estimated apparent
source size with a transparent foreground mask based on border background
subtraction, central-source masking, foreground area, equivalent radius,
bounding boxes, flux proxy, and centroid estimates.

Across the audit sets, the contaminant-to-target apparent radius ratio was
broad: approximately `0.49` at the 5th percentile, `1.06` at the median, and
`2.37` at the 95th percentile. Correlations between apparent size ratio and
affected-region MSE were weak for the learned models in this audit, so the
current result is not obviously explained by a simple size shortcut. However,
the size range is wide enough that a size-normalized benchmark is recommended as
a future robustness check.

The same audit added a halo-band metric using a dilated affected mask. This
measures broad low-level error just outside the directly contaminated region.
Thayer-BR v0.2 Moderate improved aggregate halo-band MSE over Thayer-BR v0.1 on
both normal and stress audit sets, but selected per-sample examples still show
metric-vs-visual disagreements and halo-like broad residual patterns. These
examples should be reported as qualitative caveats rather than hidden.

The visual-vs-metric disagreement audit automatically selected examples where
per-sample MSE rankings and visual inspection may point in different directions:
v0.2 Moderate broad-error cases, v0.1 wins over v0.2, direct-looking-cleaner
cases, strong v0.2 successes, and ambiguous targets. The examples are intended
for limitations and appendix figures rather than for changing metric values.

## Difficulty, Severity, and Failure

The project separates several concepts:

- `generation_difficulty`: legacy generator metadata assigned from sampled
  shift, brightness, blur, noise, and size ratio.
- `blend_severity_score` / `blend_severity_bin`: measured image-level blend
  damage, based on affected mask fraction, identity affected error, and
  optionally core obstruction.
- `core_obstruction_fraction` / `core_overlap_bin`: how much of the target core
  is touched by affected pixels.
- `model_failure_score`: model affected-region MSE.
- `model_improvement_ratio`: identity affected-region MSE divided by model
  affected-region MSE.

Blend severity and model difficulty are not the same. A severe blend can be
easy to subtract if the contaminant is obvious and separable. A low-severity
blend can be hard if it corrupts the target core or mimics target structure.

Earlier static figures may display legacy easy/medium/hard generator labels.
Those labels are retained for provenance only.

## Why Stress Testing Was Added

Thayer-Direct performed strongly on original row-index normal development blends, but normal
sampling did not fully characterize hard overlap behavior. Stress testing
concentrates smaller shifts, brighter contaminants, similar-size sources where
possible, blur/noise perturbations, and a minimum affected mask fraction.

Thayer-Direct still beats identity on the hard stress set, but its
affected-region improvement drops to `8.04x`. Thayer-Residual improves that
stress result to `10.69x`, Thayer-BR v0.1 improves it to `16.47x`, and
Thayer-BR v0.2 Moderate improves it further to about `19.6x` in the historical
original-development evaluation. The grouped-retrain estimate is reported
separately below.

## Source-Leakage and Final-Manifest Audits

The full source audit in
`outputs/runs/source_leakage_audit_20260710_062950` reconstructed the seed-42
row split, streamed raw-pixel SHA-256 hashes, compared exact RA/Dec groups, and
screened perceptual fingerprints. Row partitions and auditable target/
contaminant role assignments pass, but the source file contains 29
pixel-identical pairs crossing train/validation/test and 27 cross-split exact
coordinate pairs. Random row splitting is therefore not object-disjoint.

The union implicates 57/17,736 sources (`0.321%`). Only 13 historical normal
and 12 historical stress rows contain an implicated target or contaminant, and
excluding them changes affected-MSE ratios by at most `0.31%`. The observed
aggregate effect is therefore minor, but the scientific-protocol failure is
major because source-level independence was not enforced.

The provisional manifest run
`outputs/runs/final_test_manifest_prep_20260710_061737` reserves 1,000 sources
from the post-development test tail after exact coordinate-group exclusion. Its
five 1,000-row suites have frozen seeds, global indices, generator hashes,
sample fingerprints, schemas, and checksums. A completed exact/perceptual
cross-check found no sustained link from that pool into the historical train,
validation, or development prefix. That pool is now superseded rather than
locked-final: the grouped campaign assigned 590 of its sources to grouped
train/validation (`499/91`). A final paper evaluation requires a newly selected
untouched source-group pool after model and protocol freeze.

The grouped source split in
`data/manifests/grouped_source_split_20260710_100907` and grouped blends in
`data/manifests/grouped_blends_20260710_103233` are exact-pixel/coordinate
group-disjoint and fully replayed. They are development infrastructure, not a
final test. An old-checkpoint diagnostic on these grouped suites is also not
source-independent: `54.575%` of rows contain a group seen in the old
historical train or validation pool. On the clean-neither subset, the old
checkpoint still gives `31.53x`, `18.18x`, `11.68x`, and `18.27x` lower
affected MSE than identity across normal, hard, compact-bright, and high-core
suites, respectively.

## Duplicate-Safe Grouped Retraining

The v0.2 Moderate grouped retrain uses the same residual U-Net, `3/2`
affected/core extra weights, 50/30/20 balanced training composition, 8,000
train blends, 1,000 validation blends, batch size 8, and 20 epochs. Training
and inference use MPS; the manifest, code, device, command, and checkpoint
hashes are recorded. Best-checkpoint evaluation uses the same replayed sample
rows and masks for identity, threshold, and model comparisons.

Clipped macro affected-region MSE and identity/model ratios are:

- normal: `0.00231890`, `28.8127x`, with `0/1000`
  worse-than-identity cases;
- hard stress: `0.00458983`, `15.8025x`, with `3/1000` cases;
- compact bright: `0.00872771`, `9.18304x`, with `2/1000` cases;
- high core obstruction: `0.00491680`, `15.8378x`, with `1/1000` case.

The grouped retrain therefore verifies a substantial duplicate-safe
development effect, but it is weaker than the historical development result
and weaker than the old checkpoint on these same suites. The latter comparator
is exposure-confounded, so it cannot supersede the grouped retrain as the
correctness result. A fresh final pool and independent grouped training seeds
are still required for final-effect and training-robustness claims.

## Preservation and Clipping Audits

The corrected MPS audit in
`outputs/runs/preservation_null_tests_20260710_063312` feeds 1,000 unblended
inputs to each residual model. These are not artifact-filtered clean sources.
It reports reconstruction MSE/MAE/SSIM, residual magnitude, evaluation-core
and non-core error, and heuristic source-artifact strata. The blended audit
reports both target error outside the affected mask and model output change
relative to the blended input, because the unaffected-mask complement still
contains sub-threshold blend changes.

The paired clipping audit in `outputs/runs/clipping_audit_20260710_063312`
saves both macro summaries and per-sample clipped/unclipped metrics. Aggregate
affected MSE changes by at most 0.16%, rankings are unchanged, and no sample has
more than a 10% relative affected-MSE reduction from clipping. Out-of-range
fractions, conditional excursion magnitudes, and residual sign magnitudes are
retained so small aggregate effects do not hide output-physics diagnostics.

This is an output-clipping result. Blend-input clipping after RGB addition is
material to the generator and remains a benchmark limitation; the two effects
must not be conflated.

Masked summaries are macro means of per-sample masked metrics, not pooled-pixel
means. The evaluation core mask uses an aperture percentile; the training loss
uses a separate 55%-of-aperture-maximum core rule.

## Limitations

These experiments use controlled synthetic blends. They do not yet capture full
survey realism, including PSF variation, sky-background mismatch, detector
artifacts, crowded fields, correlated source environments, or cases where the
true target and contaminant are not available as separate clean cutouts.

The historical and grouped suites were used for model development or
infrastructure/model validation. A final paper claim requires a newly frozen,
duplicate-aware source pool and final-test manifests that are not inspected or
used for further model selection. The prior provisional pool cannot serve this
role because 590 of its sources entered grouped training or validation.
