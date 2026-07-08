# Project Plan

## Objective

Build a controlled research pipeline for galaxy deblending using Galaxy10 DECaLS cutouts. The experiment asks whether a compact learned model can recover a target galaxy from synthetic blends more accurately than simple non-learning baselines, and which blend conditions drive failure.

## Research Questions

1. Does a lightweight U-Net improve reconstruction metrics over identity and threshold-based baselines?
2. How do errors change with contaminant shift, brightness, blur, noise, and source-size ratio?
3. Which qualitative failure modes appear in severe or high-overlap blends, and are they tied to blend severity, core obstruction, or model affected error?

## Work Plan

- Prepare portable data loading and split original images into train, validation, and test subsets before blending.
- Generate synthetic blends with foreground-only contaminants, halo-aware masks, conservative default perturbations, and recorded original generation difficulty metadata.
- Evaluate simple baselines to establish non-learning reference performance.
- Train a compact U-Net on generated blends and evaluate on held-out synthetic blends.
- Analyze metrics overall, by original generation difficulty, by blend severity, by overlap/core obstruction, and by model-failure score.
- Document representative successes, failures, and limitations in the final report.

## Scope

This project is a controlled synthetic experiment, not a production survey pipeline. The synthetic setup makes target images known and enables direct reconstruction metrics, but it cannot capture every observational effect present in real blended survey images.

## Current Deliverables

- Reusable source modules under `src/`
- A clean experiment notebook under `notebooks/`
- Research documentation under `docs/`
- Scripted direct stress-test and residual U-Net evaluation workflows under `scripts/`
- Public-safe checkpoint figures under `reports/figures/`
- A concise checkpoint summary at `docs/checkpoint_summary.md`

## Current Next Steps

- Build a core-obstruction-balanced hard-case evaluation.
- Test residual training with affected-region weighting.
- Improve foreground extraction diagnostics and preprocessing checks.
- Add more realistic sky, PSF, noise, and background simulation before making
  broader claims.
