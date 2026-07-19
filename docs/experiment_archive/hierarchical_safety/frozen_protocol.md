# Hierarchical safety campaign contract

Status: frozen before drift audit, new scene generation, feature extraction, or head training.

- Master run: `outputs/runs/thayer_select_hierarchical_safety_20260711_225657`; every output is collision-refusing and append-only.
- Reconstruction checkpoint: Condition C `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`; all reconstruction parameters remain frozen.
- Source split SHA-256: `98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27`.
- Lockbox assignment rows: `4310`; assignment metadata is used only to enforce exclusion. No sealed scene or pixel is opened.
- Neural feature extraction: MPS only. CPU is used for audits, lightweight heads, calibration, tables, and figures.
- Query states: UNIQUE_VALID, NULL, AMBIGUOUS under `thayer-select-hierarchical-query-v1`.
- Recoverability: derived from query validity, metric-specific calibrated upper bounds, confusion risk, and accept/abstain; never a monolithic training label.
- Primary risk limits: `RiskLimits(image=0.75, flux=0.5, centroid_pixels=2.0)`; strict and permissive limits are sensitivity analyses.
- Development: created only after the complete policy freezes, evaluated exactly once, never used for threshold tuning.
- Prohibited: backbone alteration, encoder/decoder fine-tuning, source-split changes, historical inference regeneration, oracle deployable inputs, development retuning, lockbox access, version-control mutation, overwrite, or deletion.
