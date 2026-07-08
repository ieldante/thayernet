# Methodology

Thayer-Net is a controlled synthetic galaxy deblending benchmark. The project
starts from Galaxy10 DECaLS cutouts with known target images, creates synthetic
foreground-only blends, and evaluates whether compact U-Nets can reconstruct the
target better than simple baselines.

## Split Before Blending

Original images are split into train, validation, and test subsets before any
synthetic blends are generated. This prevents source-image leakage: the same
original galaxy cannot appear in both training and evaluation examples, even if
paired with different contaminants.

The first-pass direct and residual experiments use:

- 5,000 training blends
- 800 validation blends
- 800 normal held-out test blends
- fixed split seed 42

## Synthetic Blending

Each blend starts with two normalized RGB cutouts:

- target: the supervised clean image to recover
- contaminant: a second galaxy whose foreground light is added to the target

The contaminant is not pasted as a full rectangular cutout. Instead,
`src/blend.py` estimates a per-channel background from border pixels, subtracts
it, isolates the centered foreground source, and adds only that foreground light
to the target. This avoids teaching the model square cutout boundaries or pasted
background noise.

## Foreground Extraction and Halo-Aware Masking

Foreground extraction follows a deliberately conservative sequence:

1. Estimate the image background from border pixels.
2. Subtract the background and clip negative foreground values to zero.
3. Detect the bright central source using robust border statistics, an Otsu-style
   threshold, and a high percentile threshold.
4. Select the connected component nearest the cutout center, with a mild
   preference for larger components.
5. Dilate and Gaussian-smooth the source mask so diffuse halos and outskirts are
   retained.
6. Apply a soft circular aperture so the mask fades before reaching image
   borders.
7. Suppress very faint residual values to limit background leakage.

This makes the synthetic task closer to overlapping light profiles than naive
image-patch addition. It is still synthetic and does not fully model sky
backgrounds, PSFs, detector effects, or source clustering.

## Blend Perturbations

The generator samples shift, brightness, blur, noise, and optional rotation.
Rotation is disabled in the main checkpoint and stress-test settings because
interpolation can introduce artifacts when foreground isolation is imperfect.

The normal held-out blends use the default controlled ranges from
`configs/default.yaml`. The hard stress test concentrates harder conditions:
small shifts, brighter contaminants, similar-or-larger contaminant sizes where
possible, no rotation, and a non-trivial affected-region mask.

## Baselines

Baselines define what the learned models must beat:

- Identity baseline: returns the blended image unchanged.
- Threshold baseline: thresholds the blended image, keeps the largest connected
  foreground component, and zeros out the rest.

The threshold baseline is intentionally simple and sometimes poor. Its value is
as a transparent non-learning reference, not as a competitive survey algorithm.

## Model Formulations

Two compact U-Net formulations are evaluated:

- Direct reconstruction: input `X = blended`; target `Y = clean target`.
- Residual prediction: input `X = blended`; target `R = blended - target`;
  reconstruction `Y_hat = blended - predicted_residual`.

The residual formulation gives unchanged target light a direct subtraction path,
which can preserve target structure better in some overlapping cases. It is not
universally better; direct reconstruction still wins on some individual samples.

## Metrics

The project reports:

- MSE
- MAE
- PSNR
- SSIM
- affected-region masked MSE
- affected-region masked MAE

Whole-image metrics measure global reconstruction fidelity, but they can be
misleading because most pixels in each synthetic blend are unchanged. The
identity baseline can therefore look strong over the whole image while still
failing to remove the contaminant.

Affected-region metrics restrict evaluation to pixels where the blended image
differs from the target by more than a threshold after averaging absolute RGB
differences. These metrics better isolate the deblending problem.

## Terminology

The old easy/medium/hard labels are retained only as legacy generation metadata.
Current analysis separates five concepts:

- `generation_difficulty`: legacy metadata assigned from sampled shift,
  brightness, blur, noise, and size ratio.
- `blend_severity_score` / `blend_severity_bin`: measured image damage from
  affected mask fraction, identity affected error, and optionally core
  obstruction.
- `core_obstruction_fraction` / `core_overlap_bin`: how much of the target core
  is touched by affected pixels.
- `model_failure_score`: model affected-region MSE.
- `model_improvement_ratio`: identity affected-region MSE divided by model
  affected-region MSE.

A high-severity blend is not always hardest for the model. Some large obvious
contaminants are severe but easy to subtract. Some lower-severity blends are
hard if they obscure the target core or mimic target structure.

Earlier figures may display the generator's legacy easy/medium/hard metadata.
These labels are retained for provenance but are not treated as model-failure
categories.

## Core Obstruction

The stress-test script estimates a simple target-core mask from bright pixels
inside a central aperture. It then computes `core_obstruction_fraction` as the
fraction of that core mask overlapped by affected pixels. The derived
`core_overlap_bin` is a diagnostic for target-core overlap, not an astrophysical
morphology classifier.

## Why Stress Testing Was Added

The direct U-Net performs strongly on normal held-out blends, improving
affected-region MSE over identity by about 14.13x. Stress testing was added to
check whether that result holds under more concentrated overlap, brighter
contaminants, and similar-size sources. The direct model still beats identity on
the stress set, but its improvement drops to about 8.04x. This expected
degradation motivated the residual-prediction comparison, which improves the
stress result to about 10.69x.

## Limitations

These results apply to controlled synthetic blends. They may change under more
realistic sky simulations, PSF variation, background mismatch, detector effects,
crowding, correlated galaxy environments, or different blend-generation
settings. The project should be read as a controlled reconstruction study and a
step toward more realistic deblending experiments, not as deployment-ready
astronomical deblending.
