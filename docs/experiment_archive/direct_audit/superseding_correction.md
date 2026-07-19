# Authoritative final-report addendum

This append-only addendum clarifies interpretation without changing any model,
metric, calibrator, threshold, gate, or the corrected
`DIRECT_AUDITOR_PARTIAL` outcome.

- POST unsafe prevalence is exactly 1.0. AUROC is undefined. AUPRC 1.0 is the
  mechanical value for an all-positive label set and is not evidence of
  discrimination.
- Temperature-scaled Brier and ECE are both zero because the calibrated score
  and label are both one. The constant-prevalence Brier baseline is also zero,
  so POST does **not** satisfy the required strict Brier improvement.
- The reported unsafe and catastrophic reductions of 1.0 occur only because
  the fail-closed policy accepts zero requests. They do not rescue the failed
  50% coverage gate or establish usable catalog safety.
- Atlas and matched-control abstention are both 1.0. The tabulated 1.9804 odds
  ratio is a Haldane-corrected finite number for two saturated cells with
  unequal sample sizes; the substantive Atlas-to-control odds ratio is
  unidentifiable and non-discriminative under complete abstention.
- The original machine outcome `DIRECT_AUDITOR_FAILURE` is preserved in
  `reports/frozen_core_decision.json`. The append-only mapping correction is
  authoritative because the supplied PARTIAL definition explicitly covers
  useful query detection with failed coverage/calibration. PRE's formal pass
  remains false.
