# Worst-group quantile training

Status: **NOT RUN — PROSPECTIVE EARLY STOP**.

The observability-distillation preregistration required its complete
information-sufficiency gate to pass before R1 group-balanced quantile loss,
R2 GroupDRO, or R3 GroupDRO plus physical auxiliary supervision could run. A3
improved joint-hard ranking, but failed fixed-precision recall, natural-
calibration Brier, and natural-calibration ECE gates. Executing any worst-group
quantile continuation would therefore have violated the frozen protocol.

No oracle physical-group identity entered an inference array, no GroupDRO
weights were applied, no direct upper-quantile model was fitted, and no new
image or flux interval exists. The authoritative corrected-Q1 results remain
the active deployable evidence: image/flux marginal coverage `0.9221`/`0.9221`
and joint-hard coverage `0.5440`/`0.5907`.

