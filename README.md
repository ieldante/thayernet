# Learning to Unblend the Sky

This project studies whether a lightweight learned model can recover a target galaxy from controlled synthetic blends built from Galaxy10 DECaLS images. The repository focuses on a reproducible experimental pipeline: split real galaxy cutouts before blending, generate foreground-only contaminants with difficulty metadata, compare simple baselines against a small U-Net, and analyze reconstruction quality as blend conditions become harder.

## Research Question

Can a compact convolutional model recover the target galaxy from synthetic blends more accurately than simple image-processing baselines, and how does performance change with overlap, contaminant brightness, blur, noise, and apparent source size?

## Why Deblending Matters

Astronomical surveys often observe overlapping sources in crowded or deep fields. If blended light is assigned to the wrong object, downstream measurements of flux, morphology, color, and redshift can be biased. This project does not attempt full survey-grade deblending; it builds a controlled testbed for studying which blend conditions are learnable and where simple models fail.

## Dataset

This repository does not include the dataset. Download Galaxy10 DECaLS separately and place the HDF5 file at:

```text
data/Galaxy10_DECals.h5
```

The data directory is kept in the repository with `data/.gitkeep`, while dataset files are ignored by git. The notebook expects the portable path above by default.

## Method Overview

- Synthetic blend generation: pairs of original images are sampled from the same split and combined with controlled shift, brightness, blur, noise, and optional rotation.
- Foreground extraction and halo-aware masking: only the contaminant foreground is added to the target, using a soft central mask that preserves diffuse halo light while avoiding rectangular cutout artifacts.
- Baselines: identity reconstruction and thresholded connected-component segmentation provide lightweight non-learning references.
- U-Net model: a compact PyTorch U-Net maps blended RGB images to reconstructed target RGB images.
- Metrics: MSE, MAE, PSNR, and SSIM are computed over the full image, with additional affected-region MSE/MAE on pixels where the blend differs from the target.

For both a brief technical summary and a longer implementation-level explanation of the blending procedure, see `docs/methodology.md`.

## Repository Structure

```text
galaxy_deblending_project/
├── configs/                  # Portable experiment defaults
├── data/                     # Local dataset location; dataset files ignored
├── docs/                     # Project plan, methodology, dataset notes, logs
├── notebooks/                # Main experiment notebook
├── reports/                  # Future paper/report and final public figures
├── src/                      # Reusable data, blending, model, training code
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Quickstart

Python 3.11 or 3.12 is recommended because scientific Python and PyTorch wheels can lag newer Python releases.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/Galaxy10_DECals.h5`, then start JupyterLab:

```bash
jupyter lab
```

Open `notebooks/galaxy_deblending.ipynb` and run the cells in order.

## Reproducibility Notes

- Original images are split into train, validation, and test subsets before synthetic blends are generated.
- Synthetic blend generation accepts a NumPy random generator so experiments can be repeated with fixed seeds.
- Existing blend objects in a live notebook session do not update after editing `src/blend.py`; regenerate blends after restarting or explicitly reloading the module.
- Generated outputs, checkpoints, cached files, and the Galaxy10 DECaLS HDF5 file are intentionally excluded from version control.
- Whole-image metrics can make the identity baseline look stronger than it is because most pixels remain unchanged in a synthetic blend. Affected-region metrics are included to evaluate performance where the contaminant changes the image.

## Paper and Report

Paper and report: This repository currently contains the experimental pipeline and working notebook. A formal research paper/report will be added after the full set of experiments, figures, and evaluation tables are completed.

The future report and final public-safe figures will live under `reports/`. Draft PDFs, generated figures, checkpoints, and experimental outputs should not be committed unless they are final and explicitly reviewed.

## Current Status and Next Steps

The repository contains the initial data-loading, synthetic-blending, baseline, model, training, and notebook workflow. The next research steps are to run the full baseline/model comparisons, evaluate performance by difficulty bin, build failure-case visualizations, and write the final report.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
