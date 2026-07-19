# Cross-deblender audit protocol

Cross-deblender claims require at least three genuinely distinct compatible
families under one observed-space source-layer contract. Multiple seeds or
checkpoints of one architecture cluster are not sufficient.

For each compatible family F, all F outputs must be excluded from auditor
training, model selection, and calibration. Model selection uses seen-family
validation only; calibration uses seen-family calibration only; every threshold
then freezes before one evaluation on fresh F outputs. No retuning follows.
Family-ID leakage probes, within-family and mixed-family references, five
auditor seeds, source-group-clustered intervals, and family macro-averaging are
mandatory.

Current status is `BLOCKED`. Condition C, R0, and reconstruction-only R1 share
one family cluster. SEP lacks a validated prompt-to-source-layer adapter,
legacy RGB U-Nets have incompatible inputs/units, and scarlet is absent. No
leave-one-family-out rotation was run and no model-agnostic claim exists.

One prospective prompted ResUNet is the next bounded addition. Its admission
requires MPS training, the frozen normalization, aligned full decompositions,
no clipping, exact checkpoint/configuration hashes, and deterministic replay.
The distinct-family count must be reassessed before Thayer-Audit training.

Atlas v0 did not reopen this gate. Its three checkpoints remain one family
cluster, and same-cluster diameter failed the operational baseline comparison.
No leave-one-family-out rotation or model-agnostic result exists.
