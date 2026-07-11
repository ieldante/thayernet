# Clean Benchmark Plan

## Scope

This is a non-training benchmark proposal. It does not replace any historical
evaluation, modify a checkpoint, or retroactively change reported metrics. The
frozen planning manifest is
`outputs/runs/clean_benchmark_plan_20260710_032839/plan_manifest.json`.

All benchmark-design iterations must use only exact-pixel/exact-coordinate
grouped development sources and preserve global source/group IDs for both blend
roles. The future untouched final source pool must not be used for filter
design, manual examples, or model comparison.

The next evaluation should answer three distinct questions:

1. Do model rankings persist when both source cutouts pass explicit quality
   filters?
2. Do rankings persist when apparent target and contaminant sizes are
   controlled?
3. Which compact, bright contaminants remain difficult after source quality
   and size are separated from blend severity?

## Why Source Artifacts Matter

Synthetic blending assumes that each source cutout is a useful observation of a
single central galaxy. Border contamination, truncated sources, secondary
objects, saturation, or a failed central-source mask can violate that
assumption. Such defects can change the task in opposite ways: a conspicuous
artifact may make a contaminant easier to identify, while target corruption may
make the clean reconstruction intrinsically ambiguous. Mixing these cases into
one aggregate can therefore reward artifact recognition rather than deblending.

The source dataset has no validated artifact-quality labels. A future clean
benchmark must not silently substitute an unvalidated heuristic. It should
generate transparent flags from grouped development source pixels, audit those flags by
eye without consulting model outputs, freeze them, and report both clean and
artifact-stress results. Flagged sources remain scientifically useful as stress
cases; they are not deleted or hidden.

## Proposed Source Measurements and Filters

Use the existing size-audit conventions so the plan is reproducible: images in
`[0, 1]`, a 12-pixel border, per-channel border-median subtraction, a circular
aperture of `0.48 * min(height, width)`, and the central connected component
selected by `distance_to_center - 0.025 * sqrt(area)`. The foreground threshold
is the maximum of `0.006`, `border_median + 3*MAD`, `0.07*aperture_peak`, and
`0.35*aperture_p88`.

Test these frozen filter definitions:

- `F0_valid`: finite in-range pixels, central component area at least 8 pixels,
  and positive foreground flux.
- `F1_clean_primary`: `F0_valid`, centroid offset no more than `0.15` of the
  smaller image dimension, no component pixel within 4 pixels of an edge, no
  more than 1% hot border pixels, and off-center component flux below 20% of
  central-component flux.
- `F2_clean_strict`: `F1_clean_primary`, plus no more than 1% saturated central
  pixels and aperture foreground peak at least `0.03`.
- `F3_artifact_stress`: `F0_valid` with at least one off-center, edge-truncated,
  border-contaminated, secondary-source, or saturation flag.

Here, a hot border pixel exceeds
`max(0.006, border_median + 5*MAD)`. A secondary component must have area at
least `max(8, 0.05*central_area)` before its flux contributes to the 20% test.
The complete formulas and a preregistered sensitivity grid are recorded in the
run manifest. Before inference, manually inspect at least 50 random examples per
pool and at least 25 examples per flag. If the audit exposes poor specificity,
revise and re-freeze the rule before running any model; never tune a quality
filter based on which model benefits.

## Clean and Artifact-Stress Design

Generate separate, fixed grouped development suites:

- clean normal: 1,000 blends, with both sources from `F1_clean_primary`;
- clean hard stress: 1,000 blends, with both sources from
  `F1_clean_primary`;
- artifact-target stress: 500 blends with a flagged target and clean
  contaminant;
- artifact-contaminant stress: 500 blends with a clean target and flagged
  contaminant; and
- artifact-both stress: 500 blends with both sources flagged.

Preserve global source indices, group IDs, and split boundaries; require
different target and contaminant groups; and save every pair, seed, generation parameter,
quality flag, accepted sample, and rejected-candidate count. All compared models
must receive the same frozen samples. Report individual artifact types in
addition to pooled results, because a border defect and a second astronomical
source are not equivalent failure modes.

## Why Apparent-Size Normalization Matters

Pixel normalization only places intensities on a common numeric scale. It does
not control the number of pixels occupied by a galaxy. If apparent size ratio is
correlated with source identity, central overlap, or contaminant visibility, a
model may exploit that cue rather than learn a general separation rule.

The primary control should use pair matching, not image resizing. Estimate each
source's equivalent radius from the frozen central component, divide clean
grouped development targets into radius quartiles, and draw 250 target-contaminant pairs
from each quartile with `0.90 <= contaminant_radius / target_radius <= 1.10`.
Create 1,000 matched-size normal and 1,000 matched-size hard-stress blends. This
retains native pixels and avoids introducing interpolation artifacts.

Also report preregistered radius-ratio slices: smaller `[0.50, 0.80)`, broadly
matched `[0.80, 1.25)`, and larger `[1.25, 2.00]`. A rescaled-image benchmark,
if attempted later, should be labeled as a separate interpolation sensitivity
test rather than merged with the pair-matched result.

## Compact Bright Contaminant Benchmark

First reproduce a clean-source, legacy-compatible suite of 1,000 samples:

- per-axis shift from `-24` to `24` pixels;
- brightness from `1.35` to `1.95`;
- blur sigma from `0.0` to `0.08`;
- noise standard deviation from `0.0` to `0.006`;
- contaminant/target equivalent-radius ratio from `0.15` to `0.90`;
- affected-mask fraction at least `0.004`; and
- affected-pixel threshold `0.02`.

Do not fill a shortfall with relaxed candidates. Record rejections and enlarge
only the eligible grouped development source pool if the requested count cannot be met.

Then create a 1,200-sample diagnostic suite with 100 samples in every cell of a
`3 x 2 x 2` design: radius ratios `[0.15, 0.50)`, `[0.50, 0.70)`, and
`[0.70, 0.90]`; brightness `[1.35, 1.65)` and `[1.65, 1.95]`; and core
obstruction `[0.0, 0.50)` and `[0.50, 1.0]`. Keep the same shift, blur, noise,
mask, and source-quality rules as the legacy-compatible suite. This separates
the very-small-contaminant problem from brightness and core-overlap effects.

## Reporting and Scientific Validity

Cleaning can lower headline metrics even when it improves the benchmark. It can
remove conspicuous cues that made subtraction easy, exclude damaged blends with
large identity errors that inflated improvement ratios, and leave more subtle
low-contrast structure. Conversely, cleaning can also improve metrics by
removing irrecoverable target defects. Either direction is a composition change,
not evidence that historical numbers were wrong.

Report unfiltered, `F0_valid`, `F1_clean_primary`, `F2_clean_strict`, and
artifact-stress results side by side, with paired bootstrap 95% confidence
intervals. Include whole and affected MSE/MAE, PSNR, SSIM, core and non-core
affected MSE, halo-band MSE, gradient error, color error when available,
improvement versus identity, worse-than-identity count, and paired win rates.
Publish per-sample manifests and qualitative clean/artifact failures. A model
ranking should change only on a prespecified success criterion, never because a
post hoc filter produces a preferred result.

## Future Execution Checklist

1. Compute and tabulate flags for every grouped development source without model outputs.
2. Complete the blinded manual flag audit and freeze the source pools.
3. Save source IDs, pair IDs, seeds, parameters, and rejection logs.
4. Generate clean, artifact-stress, size-matched, and compact suites once.
5. Evaluate every comparator on the identical saved samples.
6. Verify checkpoints are unchanged before and after evaluation.
7. Report negative, niche, and ranking-changing results without altering
   historical metrics.
8. Keep the future final-paper pool untouched throughout benchmark design and
   run it only once after the full protocol is frozen.
