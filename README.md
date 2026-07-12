# Thayer-Net

Thayer-Net is a compact U-Net research testbed for controlled synthetic galaxy
deblending with Galaxy10 DECaLS cutouts. It asks whether learned image-to-image
models can reconstruct a known target galaxy from synthetic blends more
accurately than simple non-learning baselines, and how that behavior changes
under harder overlap conditions.

This repository is a controlled synthetic benchmark. It is not a full
survey-grade astronomical deblending pipeline.

## Thayer-Select status

Phase I promptability is complete and frozen: a four-channel compact model
(three `g,r,z` image channels plus a Gaussian coordinate prompt) selected the
requested member of a two-source BTK blend with 98.0% prompt-swap success.
Phase II recoverability prediction and selective abstention completed with a
**partial-success** classification. The calibrated score discriminated
actionable contract success on calibration data (AUROC 0.875), and selective
risk declined modestly with coverage, but ambiguous queries were scored higher
than clear valid queries and catastrophic failure did not decline at 80%
coverage. All current Thayer-Select evidence is controlled BTK development
evidence, DR10 remains a real-sky out-of-distribution benchmark, and the future
lockbox remains sealed.

## TL;DR

Current development reference: **Thayer-BR v0.2 Moderate Grouped Retrain**.
It uses exact-pixel and exact-coordinate group-disjoint train, validation, and
development-test sources.

| Grouped development suite | Identity affected MSE | Grouped retrain affected MSE | Lower affected MSE vs identity |
| --- | ---: | ---: | ---: |
| Normal | 0.066814 | 0.002319 | 28.81x |
| Hard stress | 0.072531 | 0.004590 | 15.80x |
| Compact bright | 0.080147 | 0.008728 | 9.18x |
| High core obstruction | 0.077871 | 0.004917 | 15.84x |

The normal `28.81x` figure is an MSE ratio, corresponding to about `5.37x`
lower RMSE. It comes from one grouped training seed and is not a final-paper
estimate.

| Evidence category | Status |
| --- | --- |
| Original development split | Historical only: `32.3x` normal and `19.6x` hard lower affected MSE than identity. The row split was not duplicate-safe. |
| Historical checkpoint on grouped manifests | Diagnostic only: it scores better than the grouped retrain, but `54.575%` of rows expose an old train/validation source group. |
| Grouped retrain | Current source-group-disjoint development reference: `28.81x`, `15.80x`, `9.18x`, and `15.84x` across the four suites above. |
| Future final-paper test | Not yet run. It requires a fresh untouched group-disjoint source pool after protocol freeze. |

The historical evaluation-seed audit (`32.02 +/- 1.21x` normal and
`19.55 +/- 0.30x` hard) varied blend-generation/evaluation seeds, not
independent training seeds.

Affected-region metrics are emphasized because most pixels are unchanged in
each synthetic blend. Whole-image scores are still useful, but affected-region
MSE better isolates the thresholded blend-change region. Because target-only
blur, post-composite noise, and clipping can also contribute to that region, it
must not be interpreted as a pure contaminant-flux mask.

The correctness audit found 57 implicated sources (`0.321%` of the dataset).
Removing implicated historical evaluation rows changed reported ratios by at
most about `0.31%`: a minor measured aggregate effect but a major protocol
defect. The grouped correction is complete for development work. A fresh
untouched final source pool is still required because the earlier provisional
pool was later reused by grouped training/validation.

## Model Naming

**Thayer-Net** refers to the overall project and model family.

The evaluated model variants are:

- **Thayer-Direct:** direct reconstruction U-Net, trained to map
  `blended -> target`.
- **Thayer-Residual:** residual prediction U-Net, trained to map
  `blended -> residual`, with reconstruction computed as
  `blended - predicted_residual`.
- **Thayer-BR v0.1:** previous balanced
  hard-case residual U-Net trained on 8,000 synthetic blends with a 50/30/20 mix
  of normal, high-overlap/core-obstruction, and brightness/size-stress cases.
- **Thayer-BR v0.2 Moderate:** balanced residual U-Net with moderate
  affected/core-weighted loss; the grouped retrain is the current development
  reference and the original checkpoint is historical.
- **Thayer-BR v0.2 Strong:** stronger weighted-loss ablation, not the current
  best model.
- **Thayer-BR v0.3 Delta:** color/perceptual tradeoff ablation; it improves some
  compact/color diagnostics but does not replace v0.2 Moderate.
- **Thayer-ResUNet v0.4:** small residual-architecture ablation with targeted
  compact/halo gains and aggregate tradeoffs; it is not the current reference.

## Evaluation-set sensitivity and robustness

The 2026-07-08 evaluation audit reran evaluation only; it did not train,
retrain, or modify checkpoints. The later correctness campaign separately
built grouped manifests and trained one authorized grouped v0.2 checkpoint.

- Historical multi-seed normal evaluation: `32.02 +/- 1.21x` lower affected MSE vs identity.
- Historical multi-seed stress evaluation: `19.55 +/- 0.30x` lower affected MSE vs identity.
- The grouped v0.2 Moderate retrain is the current source-group-disjoint
  development reference, not a final-paper result.
- Thayer-BR v0.1 remained best among the audited methods across affected-mask
  thresholds `0.005`--`0.04` and dilation radii `0`--`9`; v0.2 was not part of
  that earlier mask-robustness audit.
- Checkpoint integrity logs confirmed that the Thayer-Direct, Thayer-Residual,
  Thayer-BR v0.1, and Thayer-BR v0.2 Moderate checkpoints were unchanged before
  and after evaluation/audit runs.

## What Was Audited

- Affected-mask thresholds and dilation radii for the v0.1-era comparison; v0.2
  was not included.
- Multi-seed normal and hard stress blend generation/evaluation; these are not
  independent training-seed replications.
- Residual reconstruction logic and sign convention.
- Visual blend, mask, residual, and model-output diagnostics.
- Apparent-size, centrality, halo-band, and visual-vs-metric diagnostics.
- Checkpoint paths, file sizes, and modified times before and after evaluation.
- Split-before-blending logic, duplicate/source leakage, and same-runtime sample
  comparability. Splitting indices before blending does not eliminate duplicate
  objects already present in the source file.

## Limitations

The current results are for controlled synthetic blends with known targets. They
should not be interpreted as validated real-survey performance.

Galaxy10 DECaLS images are RGB display cutouts, not calibrated FITS flux
images. Results should be interpreted as synthetic deblending of Galaxy10 RGB
cutouts, not survey-grade source separation. Identity and threshold baselines
are sanity checks, not strong astronomical deblenders.

Remaining limitations include ambiguous source overlap, target-core
obstruction, target-detail loss, over-smoothing, simplified sky/noise modeling,
missing PSF variation, and synthetic foreground-extraction assumptions.

Thayer-BR v0.2 Moderate is not universally best on every individual example.
Thayer-Direct and Thayer-Residual still win on some samples, especially where
their inductive biases preserve a particular target structure better.
Thayer-BR v0.1 also wins on some v0.2 counterexamples.

The size/visual audit found broad apparent size variation and selected
halo-like artifacts in individual v0.2 outputs. Aggregate halo-band error still
improved relative to Thayer-BR v0.1, but future work should include a
size-normalized benchmark.

The audit supports a development-benchmark ranking; it does not prove
performance on real crowded survey scenes. Exact-pixel and exact-coordinate
grouping is complete, but exhaustive near-duplicate identity is not proven and
a fresh untouched final partition must be frozen before final-paper evaluation.

## Links to Deeper Docs

- [Methodology](docs/methodology.md): blend generation, masking, metrics, and
  residual reconstruction.
- [Current best model](docs/current_best_model.md): short v0.2 Moderate summary.
- [Release summary](docs/releases/thayer_br_v0_2.md): research checkpoint
  announcement for Thayer-BR v0.2 Moderate.
- [Model card](docs/model_card_thayer_br_v0_2.md): technical model details,
  intended use, limitations, and metrics.
- [Evaluation audit summary](docs/evaluation_audit_summary.md): concise audit
  trail for the headline result.
- [Source-leakage audit](docs/source_leakage_audit.md) and
  [final-test protocol](docs/final_test_protocol.md): duplicate findings and why
  the earlier provisional pool is superseded.
- [Research correctness audit](docs/research_correctness_audit.md) and
  [grouped retrain results](docs/grouped_retrain_results.md): grouped
  infrastructure, metrics, and claim status.
- [Advisor update](docs/advisor_update_grouped_audit.md): concise audit and
  grouped-retrain summary.
- [Preservation/clipping audits](docs/preservation_and_clipping_audits.md) and
  [limitations](docs/limitations_and_next_steps.md): null tests, output-range
  diagnostics, and next scientific corrections.
- [Checkpoint summary](docs/checkpoint_summary.md): experiment history,
  checkpoint-level results, and audit summary.
- [Results interpretation](docs/results_interpretation.md): how to read the
  affected-region metrics, threshold baseline, core obstruction, and caveats.
- [Paper plan](docs/paper_plan.md): recommended paper framing and claims to
  avoid.

## Research Question

Can a compact convolutional model recover the target galaxy from controlled
synthetic blends more accurately than simple image-processing baselines, and how
does performance change with overlap, contaminant brightness, blur, noise, and
apparent source size?

## Dataset

This repository does not include the dataset. Download Galaxy10 DECaLS
separately and place the HDF5 file at:

```text
data/Galaxy10_DECals.h5
```

The `data/` directory is kept in the repository with `data/.gitkeep`, while
dataset files are ignored by git.

## Method Overview

- Exact-pixel and exact-coordinate source groups are assigned wholly to train,
  validation, or grouped development-test before blends are generated.
- Synthetic blends add only extracted contaminant foreground light to the
  target, reducing rectangular cutout/background artifacts.
- Halo-aware masks preserve diffuse contaminant outskirts while tapering before
  cutout edges.
- Sanity baselines include identity reconstruction and a simple
  threshold/connected-component method; neither is a competitive astronomical
  deblender.
- Thayer-Direct, Thayer-Residual, Thayer-BR v0.1, and Thayer-BR v0.2 Moderate
  are evaluated with whole-image and affected-region metrics.
- Hard stress testing uses smaller shifts, bright contaminants, similar-size
  sources where possible, blur/noise perturbations, and a minimum affected mask
  fraction.
- Thayer-BR v0.2 Moderate adds affected/core-weighted residual loss while
  keeping the residual prediction formulation.

For implementation details, see [docs/methodology.md](docs/methodology.md).
For a concise project summary, see
[docs/checkpoint_summary.md](docs/checkpoint_summary.md).

## Repository Structure

```text
thayernet/
├── configs/                  # Portable experiment defaults
├── data/                     # Local dataset location; dataset files ignored
├── docs/                     # Methodology, experiment logs, paper planning
├── notebooks/                # Main experiment notebook
├── reports/                  # Public-safe figures and paper skeleton
├── scripts/                  # Reproducible training/evaluation scripts
├── src/                      # Reusable data, blending, model, training code
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Quickstart

Python 3.11 or 3.12 is recommended because scientific Python and PyTorch wheels
can lag newer Python releases.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/Galaxy10_DECals.h5`, then start JupyterLab:

```bash
jupyter lab
```

Open `notebooks/galaxy_deblending.ipynb` to inspect the notebook workflow.
Larger formal experiments are captured by scripts under `scripts/`.

## Reproducibility Notes

- Historical source indices were split by row. The grouped development
  manifests instead keep exact-pixel and exact-coordinate groups within one
  partition and record both source and group IDs.
- Synthetic blend generation accepts a NumPy random generator for fixed-seed
  experiments.
- Generated outputs, saved model checkpoints, cached files, and the Galaxy10
  DECaLS HDF5 file are intentionally excluded from version control.
- Existing blend objects in a live notebook session do not update after editing
  `src/blend.py`; regenerate blends after restarting or reloading.

## Current Next Steps

- Freeze a fresh untouched group-disjoint final source pool after the model,
  generator, metrics, and analysis protocol are fixed.
- Add clustered uncertainty and an independent grouped training seed; one seed
  does not establish training-seed robustness.
- Finalize paper figures from `reports/figures/` and the latest reviewed output
  figures.
- Write the LaTeX report.
- Improve preprocessing diagnostics and foreground extraction checks.
- Run a size-normalized held-out benchmark before making stronger claims about
  size-invariant deblending.
- Add more realistic sky, PSF, noise, and background simulation.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for
details.
