# Methodology

Thayer-Net is a controlled synthetic galaxy deblending benchmark. It starts
from Galaxy10 DECaLS cutouts, creates synthetic blends with known clean targets,
and evaluates whether compact U-Net models can reconstruct the target better
than simple baselines.

## Dataset

The dataset is Galaxy10 DECaLS, stored locally as an HDF5 file. The dataset is
not included in the repository; local copies belong under `data/`, which is
ignored except for `data/.gitkeep`.

Original images are split into train, validation, and test subsets before any
synthetic blending occurs. This prevents source-image leakage: the same original
galaxy cannot appear in both training and evaluation examples, even if paired
with different contaminants.

## Synthetic Blend Generation

Each blend uses two normalized RGB cutouts from the same split:

- `target`: the clean supervised image to reconstruct.
- `contaminant`: a second galaxy whose foreground light is added to the target.

The generator records shift, brightness, blur, noise, and size-ratio metadata.
Rotation is disabled in the main formal experiments unless artifact-free
foreground extraction is specifically being tested.

## Why Foreground-Only Blending

Adding full contaminant cutouts can create artificial rectangular boundaries and
double-background problems. A model could learn those cutout artifacts instead
of learning deblending behavior.

Thayer-Net therefore estimates contaminant foreground light before blending. The
pipeline estimates a background from border pixels, subtracts it, isolates the
central source, and adds only the foreground component to the target. This is
still synthetic, but it avoids trivial pasted-patch cues and keeps the
reconstruction target well defined.

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
reconstructing hidden light. Its role is to provide a transparent non-learning
reference, not a competitive survey algorithm.

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
model on the controlled synthetic benchmark.

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
therefore focus evaluation on the actual contaminant-altered area.

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

The split audit confirmed that original Galaxy10 DECaLS images are split before
blending and that normal/stress evaluation blends are generated only from the
held-out test source array. No split-level source leakage was found. A caveat is
that standard normal blend dictionaries do not save global source indices, so
historical normal samples cannot be re-proven source-by-source after generation.

The affected-region mask audit confirmed the formula
`abs(blended - target).mean(axis=-1) > threshold`. This means the mask is based
on where synthetic blending changed the clean target, not on a model's
prediction error.

Threshold sensitivity was tested at `0.005`, `0.01`, `0.02`, and `0.04`.
Thayer-BR v0.1 stayed best across all tested thresholds on both normal and
stress sets. Mask dilation was tested at `0`, `1`, `3`, `5`, and `9` pixels,
and Thayer-BR v0.1 also stayed best under these larger halo-inclusive masks.

The multi-seed audit used three independent normal seeds and three independent
stress seeds, each with 1,000 generated blends. Thayer-BR v0.1 was the best
learned model for every audited seed. Mean improvement over identity was
`27.04 +/- 1.04x` on normal blends and `15.76 +/- 0.07x` on stress blends.

The later v0.2 Moderate multi-seed audit found `32.02 +/- 1.21x` normal
improvement and `19.55 +/- 0.30x` stress improvement, supporting v0.2 Moderate
as the current best controlled synthetic model.

Residual evaluation logic was also checked: residual targets are
`blended - target`, residual reconstructions are `blended - predicted_residual`,
and clipping is applied after subtraction for metrics and visualization. This
supports the sign convention used for the residual checkpoints.

## Size and Visual Audit

The non-training size/visual audit in
`outputs/runs/size_visual_audit_20260709_102251` regenerated held-out normal and
stress blends and evaluated existing checkpoints only. It estimated apparent
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

Thayer-Direct performed strongly on normal held-out blends, but normal
sampling did not fully characterize hard overlap behavior. Stress testing
concentrates smaller shifts, brighter contaminants, similar-size sources where
possible, blur/noise perturbations, and a minimum affected mask fraction.

Thayer-Direct still beats identity on the hard stress set, but its
affected-region improvement drops to `8.04x`. Thayer-Residual improves that
stress result to `10.69x`, Thayer-BR v0.1 improves it to `16.47x`, and
Thayer-BR v0.2 Moderate improves it further to about `19.6x` in the current
evaluation.

## Limitations

These experiments use controlled synthetic blends. They do not yet capture full
survey realism, including PSF variation, sky-background mismatch, detector
artifacts, crowded fields, correlated source environments, or cases where the
true target and contaminant are not available as separate clean cutouts.
