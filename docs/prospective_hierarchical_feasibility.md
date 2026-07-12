# Prospective hierarchical-safety feasibility

The prospective campaign is preserved at
`outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/` and is
classified **PARTIAL SUCCESS**. Read `reports/final_report.md` together with
the authoritative append-only `reports/final_report_addendum.md`.

The campaign preregistration SHA-256 is
`f2184c169c9161e920988d32b217e56b78bb4688a65a6a0023944f9e73dec9d2`.
It predates every fitted head. The experiment used 12,000 balanced query-
training scenes, 2,000 query-validation scenes, 12,000 UNIQUE_VALID risk-
training scenes, 2,000 natural risk-validation scenes, and 4,000 natural-
mixture calibration scenes. It generated and accessed no development or
lockbox scene.

The query gate passed: five-seed macro F1 was `0.872 ± 0.010`; NULL recall was
`1.000`, AMBIGUOUS recall `0.877`, and ambiguity inversion disappeared in all
five seeds. Validation/natural-calibration rank correlations were `0.860/0.870`
for image risk, `0.867/0.858` for flux risk, and `0.949/0.954` for centroid
risk. Confusion validation/calibration AUROC was `0.866/0.844`.

The catastrophic-valid head achieved validation AUROC/AUPRC `0.987/0.997` and
transferred at `0.987/0.997`. It nevertheless fails the frozen preregistered
AUPRC gate because `1.25 × 0.8165 = 1.0206`, above the mathematical maximum
AUPRC. The gate is defective and was not changed after results.

Calibration did not collapse: vector-scaled query ECE was `0.0266`; marginal
90% continuous-risk coverage was approximately `0.900`; risk bounds retained
2,799–2,800 unique scores. Calibration is PARTIAL because frozen image/flux
subgroups covered as little as `0.691` and raw mean widths were dominated by
rare extremes.

No full policy, operational threshold, development evaluation, or selective-
risk claim was created. The historical hierarchical campaign remains FAILURE.
The one next experiment is a separately preregistered train/validation/
calibration-only conditional-calibration correction with an attainability
preflight for every numerical gate and a bounded prevalence-adjusted AP-lift
criterion. Condition C and all heads must remain frozen; development and
lockbox remain prohibited.
