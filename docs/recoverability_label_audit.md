# Recoverability label audit

The frozen-head ablation preserved the empirical moderate reliability-contract
label exactly. It did not redefine the target after observing performance.

The label is exceptionally sparse: 0.410% positive in training, 0.333% in
validation, and 1.500% in calibration. Validation contains only five positives,
so AUPRC estimates and head differences have wide bootstrap intervals. AUPRC
must always be stated alongside the split prevalence.

Separate diagnostic targets remain available for catastrophic failure, null
hallucination, ambiguous query, and source confusion. They were not collapsed
into one opaque learned input. Generator variables—SNR, PSF-normalized
separation, flux ratio, size ratio, core obstruction, color similarity, and
source count—were used only in an explicitly non-deployable oracle analysis.

The automated boundary audit found 629/10,000 training, 77/1,500 validation,
and 104/2,000 calibration samples within 5% of at least one moderate contract
threshold. Strict/moderate/permissive contract status changed for 320/10,000,
37/1,500, and 110/2,000 samples, respectively. The validation boundary fraction
was 5.13%; the contract-change fraction was 2.47%. This is meaningful threshold
sensitivity, though it does not prove that any individual empirical label is
wrong.

The analysis-only oracle achieved validation AUROC 0.795 and AUPRC 0.023 at
0.333% prevalence. Thus generator-known scene difficulty predicts some label
variation but does not explain the much stronger validation ranking in the
frozen representation. Conversely, the selected latent head's calibration
AUROC/AUPRC fell to 0.514/0.032, showing that validation performance was not
stable enough to isolate representation, objective, and label effects cleanly.

The manual-review gallery was generated only from training, validation, and
calibration inputs. No development or lockbox scene was opened. Its purpose is
to route threshold-near and model/label-disagreement cases for later contract
review; it is not a basis for post hoc relabeling in this campaign.

The next experiment should preregister a revised reliability target with
failure-specific supervision before training. It must preserve a held-out
selection policy and must not use development or lockbox results to choose the
new thresholds.

Training/validation and calibration label provenance is not identical:
training/validation use the persisted Phase-I teacher outcomes used for
Phase-II supervision, while calibration uses persisted Phase-II R1 outcomes.
No new R1 reconstruction inference was authorized to harmonize these labels.
This target-domain mismatch is part of the label/objective limitation and must
be removed by design in the next experiment, not patched after seeing results.
