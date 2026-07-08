# Thayer-Net

> Thayer-Net is named after Thayer Street in Providence, the street outside my dorm room where this project began. The name also reflects the model’s U-Net-based architecture.

## Learning to Unblend the Sky

Thayer-Net is a lightweight U-Net-based model for reconstructing target galaxy images from controlled synthetic blends. This project studies whether a compact learned model can recover a target galaxy from blends built using Galaxy10 DECaLS images.

## Repository Structure

```text
thayernet/
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

## Paper and Report

This repository currently contains the experimental pipeline and working notebook for the project. A formal research paper/report will be added after the full set of experiments, figures, and evaluation tables are completed.

The future report and final public-safe figures will live under `reports/`. Draft PDFs, generated figures, checkpoints, and experimental outputs should not be committed unless they are final and explicitly reviewed.

## Current Status and Next Steps

The repository contains the initial data-loading, synthetic-blending, baseline, model, training, and notebook workflow. The current direct-reconstruction U-Net has been trained on a 5,000-blend training run and evaluated against identity and threshold baselines on held-out synthetic blends.

Next steps include refining the measured difficulty labels, running a balanced hard-case stress test, saving final evaluation tables and figures, and testing a residual-prediction variant that predicts the contaminant layer rather than reconstructing the full target image directly.
