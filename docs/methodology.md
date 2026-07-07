# Methodology

## Why Synthetic Blending

Real astronomical blends often do not provide clean ground-truth target images. Synthetic blending starts from isolated galaxy cutouts and creates controlled overlaps, so the target, contaminant, and blending parameters are known. This makes direct reconstruction metrics possible and allows performance to be stratified by blend difficulty.

## Split Before Blending

Original Galaxy10 DECaLS images are split into train, validation, and test subsets before any synthetic blends are generated. This prevents the same original galaxy from appearing in both training and evaluation examples, even if it is paired with different contaminants. Without this ordering, validation or test metrics could be inflated by leakage.

## Blending in Brief

Each synthetic example starts with two normalized RGB galaxy cutouts: a target and a contaminant. The target remains the supervised ground truth. The contaminant image is background-subtracted, reduced to a soft central-source foreground mask, optionally rotated, shifted, brightness-scaled, and added onto the target. Optional blur and noise are applied as controlled degradations. The output is stored as a dictionary containing the original target, original contaminant, blended image, and metadata describing the sampled perturbations and estimated source sizes.

The important design choice is that the contaminant is added as extracted foreground light, not as a full rectangular image patch. This makes the synthetic task closer to deblending overlapping light profiles and avoids teaching the model artifacts from pasted cutout boundaries.

## Full Technical Blending Procedure

The current implementation in `src/blend.py` follows this sequence:

1. Background estimation: the code estimates a per-channel background from border pixels. The median is used because it is less sensitive to bright central galaxy structure than a mean.
2. Foreground construction: the estimated background is subtracted from the contaminant and negative residuals are clipped to zero. This produces a candidate foreground image.
3. Central-source detection: the RGB foreground is converted to a grayscale intensity image. A robust border statistic and an Otsu-style threshold are combined with a high percentile threshold to identify the bright core of the central source.
4. Component selection: connected components are labeled, and the component closest to the image center is selected, with a mild preference for larger components. This assumes Galaxy10 DECaLS cutouts are centered on the labeled galaxy.
5. Halo-aware expansion: the selected binary component is dilated, then Gaussian-smoothed into a soft mask. This expands the mask beyond the bright core so faint halo light and galaxy outskirts are not immediately discarded.
6. Aperture tapering: a soft circular aperture fades the mask before it reaches the image boundaries. This is meant to preserve central diffuse structure while avoiding square or edge-shaped remnants from the original cutout.
7. Residual suppression: very faint masked values are set to zero. This reduces the chance of pasting low-level background noise as if it were contaminant galaxy light.
8. Optional contaminant rotation: if enabled, only the extracted contaminant foreground is rotated. Rotation is disabled by default because interpolation can introduce faint artifacts, especially when foreground isolation is imperfect.
9. Shift and scaling: the extracted contaminant foreground is shifted without wraparound and multiplied by a sampled brightness factor.
10. Blend formation: the shifted contaminant foreground is added to the target image and clipped to `[0, 1]`.
11. Controlled degradations: optional blur and Gaussian noise can be applied to simulate harder reconstruction settings. These are intentionally conservative in the default config.
12. Metadata recording: the blend stores shift, rotation, brightness, blur, noise, source areas, source radii, size ratio, and a heuristic difficulty label.

This procedure creates a known-input/known-target reconstruction problem: the model receives the blended image and is trained to recover the unmodified target. The contaminant is retained for visualization and analysis, but it is not the supervised output.

## TODO: Size Normalization and Related Controls

Future iterations should add a more explicit treatment of apparent galaxy size before blending:

- Estimate a stable source radius for each original image and inspect its distribution across the dataset.
- Consider normalizing target and contaminant size so the experiment can separate overlap difficulty from raw apparent-size differences.
- Add an option to stratify or match pairs by source radius, morphology label, or brightness percentile.
- Track whether size normalization improves metric stability across difficulty bins.
- Compare native-size blending against normalized-size blending as an ablation, rather than replacing the current behavior without measurement.
- Revisit point-spread-function and background matching if the project moves from controlled synthetic blends toward more observationally realistic simulations.

## Foreground Extraction

The contaminant is not pasted as a full rectangular cutout. Instead, the code estimates the image background, subtracts it, isolates the central source foreground, and adds only that foreground light to the target. This reduces artificial square boundaries and makes the synthetic blend closer to the intended physical setup: extra galaxy light superimposed on another galaxy image.

## Halo-Aware Masking

Galaxy outskirts and diffuse halos are scientifically relevant and visually important. The mask therefore starts from the central bright component, dilates it, smooths it, and applies a soft central aperture. This keeps extended light while tapering the mask before it reaches the image edges. Very faint residuals are suppressed to limit background leakage.

## Optional Rotation

Rotation can increase orientation diversity, but it is risky because interpolation can introduce faint angular or polygon-like artifacts if the foreground isolation is imperfect. For that reason, rotation is disabled in the default configuration. A wide rotation range, such as `[0.0, 180.0]`, should be treated as a stress test and inspected visually.

## Blend Difficulty Labels

Each blend stores the sampled shift, brightness, blur, noise, and estimated source-size metadata. Difficulty is assigned heuristically from:

- Smaller contaminant shifts, which increase overlap.
- Brighter contaminants, which make the target harder to recover.
- Higher blur or noise, which reduce recoverable structure.
- Large target/contaminant size differences, which can complicate separation.

The labels are analysis aids rather than physical truth. They should be checked against metric trends and visual examples.

## Baselines

Baselines define what a lightweight learned model must beat.

- Identity baseline: returns the blended image unchanged.
- Threshold baseline: thresholds the blended image, keeps the largest connected foreground component, and zeros out the rest.

These are intentionally simple. They are useful because they are fast, interpretable, and reveal whether the learned model is doing more than copying the input.

## Error Metrics

- MSE: mean squared pixel error; emphasizes large deviations.
- MAE: mean absolute pixel error; less sensitive to outliers than MSE.
- PSNR: logarithmic signal-to-error ratio derived from MSE; higher is better.
- SSIM: structural similarity; captures luminance, contrast, and structure.
- Optional IoU: foreground-mask overlap after thresholding, useful when evaluating detection-like behavior rather than full image reconstruction.

Metrics should be reported both overall and by difficulty bin. Visual inspection remains necessary because small metric differences can hide structured artifacts.

## Limitations

Synthetic blends do not fully model survey point-spread functions, sky background variation, source crowding, detector artifacts, calibration errors, or physically correlated galaxy environments. The results should be interpreted as a controlled reconstruction study and a stepping stone toward more realistic deblending experiments.
