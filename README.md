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

## Current Status and Next Steps

The repository contains the initial data-loading, synthetic-blending, baseline, model, training, and notebook workflow. The current direct-reconstruction U-Net has been trained on a 5,000-blend training run and evaluated against identity and threshold baselines on held-out synthetic blends.

Next steps include refining the measured difficulty labels, running a balanced hard-case stress test, saving final evaluation tables and figures, and testing a residual-prediction variant that predicts the contaminant layer rather than reconstructing the full target image directly.
