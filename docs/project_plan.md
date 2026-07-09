# Project Plan

## Objective

Build a controlled research pipeline for galaxy deblending using Galaxy10 DECaLS
cutouts. The project asks whether compact learned models can recover a target
galaxy from synthetic blends more accurately than simple non-learning baselines,
and which blend conditions drive failure.

## Research Questions

1. Does a lightweight U-Net improve reconstruction metrics over identity and
   threshold-based baselines?
2. How do errors change with contaminant shift, brightness, blur, noise, source
   size, and target-core overlap?
3. Which qualitative failure modes appear in severe or high-overlap blends, and
   are they tied to blend severity, core obstruction, or model affected error?

## Work Plan

- Prepare portable data loading and split original images into train,
  validation, and test subsets before blending.
- Generate synthetic blends with foreground-only contaminants, halo-aware masks,
  conservative default perturbations, and recorded legacy generation metadata.
- Evaluate simple baselines to establish non-learning reference performance.
- Train compact U-Net models with direct and residual objectives.
- Analyze metrics overall, by generation difficulty, by blend severity, by
  overlap/core obstruction, and by model-failure score.
- Document representative successes, failures, and limitations in the final
  report.

## Scope

This project is a controlled synthetic experiment, not a production survey
pipeline. The synthetic setup makes target images known and enables direct
reconstruction metrics, but it cannot capture every observational effect present
in real blended survey images.

## Current Deliverables

- Reusable source modules under `src/`.
- A clean experiment notebook under `notebooks/`.
- Research documentation under `docs/`.
- Scripted stress-test, Thayer-Residual, Thayer-BR v0.1, Thayer-BR v0.2
  Moderate, and size/visual audit workflows under `scripts/`.
- Reviewed public-safe figures under `reports/figures/`.
- A concise project summary at `docs/checkpoint_summary.md`.
- Current best model documentation at `docs/current_best_model.md`.
- Thayer-BR v0.2 Moderate model card and release summary under `docs/`.
- A LaTeX paper skeleton under `reports/paper/`.

## Current Next Steps

- Preserve exact generated evaluation sets and global source indices for future
  reproducibility.
- Finalize paper figures and captions.
- Write the LaTeX report.
- Improve foreground extraction diagnostics and preprocessing checks.
- Evaluate current checkpoints on a size-normalized held-out benchmark.
- Add more realistic sky, PSF, noise, and background simulation before making
  broader claims.
