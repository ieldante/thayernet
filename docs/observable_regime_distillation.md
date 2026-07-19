# Observable-regime distillation

The prospective run
`outputs/runs/thayer_select_observability_distillation_20260712_035843/`
completed with **OBSERVATIONAL INFORMATION LIMIT — FAILURE**. Its
preregistration SHA-256 is
`8ec5b644fba2658f32eeac43edea0e5f8d4e3301a6b108cf3d1f932722dddbff`.
All predecessor baselines reproduced at tolerance `1e-10` before frozen MPS
feature extraction or head fitting.

The campaign compared the exact four historical scalar proxies (A0), pooled
prompt-local encoder summaries (A1), prompt-centered spatial encoder and
blend/candidate/residual patches (A2), and their shared spatial combination
with frozen risk outputs (A3). Simulator SNR and obstruction appeared only in
separate supervision/evaluation tables. They never entered deployable forward
arrays. Condition C remained byte-identical with zero trainable parameters.

Validation selected A3 without calibration access. It materially improved
joint-hard ranking over A0: five-seed AUROC was `0.9014 ± 0.0035` versus
`0.7113`, and normalized AP lift was `0.3725` versus `0.1032`. SNR rank signal
was strong (validation/calibration Spearman `0.883`/`0.889`), while obstruction
was materially weaker (`0.456`/`0.479`). Joint-hard AUROC/AUPRC transferred
from `0.906`/`0.430` on validation to `0.880`/`0.325` on natural calibration.

Those ranking gains did not pass the frozen information-sufficiency contract.
Mean recall at precision 0.70 was only `0.0835` against a `0.30` gate.
Natural-calibration Brier was `0.1397`, worse than the prevalence-only
`0.0642` reference, and ECE was `0.2191` against a `0.15` maximum. Continuous
SNR/obstruction magnitudes also failed to transfer even though rank order did;
their unbounded calibration predictions produced nonfinite raw-space MAE.
No threshold or calibration repair was performed after observing this result.

The early stop therefore prohibited GroupDRO, direct upper-quantile fitting,
predicted-regime calibration, and multigroup calibration. Image and flux remain
FAIL at the authoritative corrected-Q1 joint-hard coverages `0.5440` and
`0.5907`; centroid remains PASS. No full policy is authorized.

Exactly one next data-level experiment is recommended: a separately
preregistered train/validation/natural-calibration observability study with
**explicit PSF input**. Do not add other data sources in the same experiment,
generate development scenes, or access lockbox.

