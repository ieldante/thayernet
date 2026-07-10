# Size-Normalized Benchmark Plan

## Motivation

This is a future **grouped development** benchmark plan, not a final-paper test
plan. Any prototype must use only the current grouped development-test source
groups and retain source/group IDs. The future untouched final pool must not be
used to design size bins, inspect examples, or select models.

The current synthetic benchmark normalizes pixel values, but it does not
normalize apparent galaxy size. Target and contaminant cutouts can have different
foreground areas and equivalent radii. That variation is realistic, but it also
raises a useful concern: a model may learn size or centrality shortcuts instead
of learning a robust deblending rule.

The size/visual audit in `outputs/runs/size_visual_audit_20260709_102251`
found broad apparent-size variation. Across the regenerated normal and stress
audit sets, the apparent contaminant/target radius ratio ranged from about
`0.49` at the 5th percentile to `2.37` at the 95th percentile.

## Pixel Normalization vs Apparent-Size Normalization

Pixel-value normalization converts image intensities to a common numeric range,
usually `[0, 1]`. It does not make galaxies the same apparent size on the image
grid.

Apparent-size normalization would instead control the measured foreground size
of the target and contaminant before blending. It asks a different question:
does the model still work when size ratio is not an easy cue?

## Proposed Method

1. Estimate foreground size for each cutout using a transparent source mask:
   border background estimate, background-subtracted brightness, central-source
   mask, foreground area, and equivalent radius.
2. Build source pools by apparent equivalent radius.
3. Crop and scale target and contaminant foregrounds into controlled apparent
   size ranges.
4. Generate matched-size blends from grouped development-test images first.
5. Evaluate existing checkpoints on the size-normalized test set without
   training.
6. Compare current benchmark metrics with size-normalized metrics.
7. Only after the evaluation result is understood, optionally train a future
   model on size-normalized or size-balanced blends.

## Prototype Evaluation

A safe first prototype would generate `200` to `500` grouped development blends with
matched apparent target and contaminant equivalent radii. It should not train or
modify checkpoints.

Suggested bins:

- target and contaminant matched within `0.8` to `1.25`
- contaminant smaller, `0.5` to `0.8`
- contaminant larger, `1.25` to `2.0`

The prototype should report affected MSE, core affected MSE, halo-band MSE,
model win rates, and qualitative examples for the same models used in the
current audit.

## Risks

- Resizing can introduce interpolation artifacts that are easier or harder than
  the original deblending problem.
- Removing size variation can reduce realism, because real surveys naturally
  contain galaxies at different apparent sizes.
- Foreground-size estimates are approximate and can be affected by neighboring
  stars, diffuse halos, or low signal-to-noise cutouts.
- A strictly size-matched benchmark may under-test real bright/large
  contaminant cases.

## Recommended Experiment

Run a size-normalized grouped development evaluation without additional
training. Do not use the future untouched final pool for this prototype.
Compare:

- current normal and stress benchmark
- size-normalized normal benchmark
- size-normalized stress or high-overlap benchmark

The key question is whether Thayer-BR v0.2 Moderate remains strong when apparent
size ratio is controlled. If it does, the current result is more robust. If it
drops substantially, report the current benchmark as size-varied and use the
size-normalized benchmark as an important ablation rather than treating the
current results as invalid.
