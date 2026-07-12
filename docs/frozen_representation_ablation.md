# Frozen-representation recoverability ablation

Run: `outputs/runs/thayer_select_frozen_head_ablation_20260711_220756/`.

This controlled diagnostic froze the authoritative Phase-II R1 model and
extracted the 64-value adaptive-average-pooled bottleneck representation for
10,000 training, 1,500 validation, and 2,000 calibration scenes. Every model
parameter was non-trainable, the model was in evaluation mode, a repeated
64-sample extraction was bit-identical, no parameter received a gradient, and
the original checkpoint hash remained unchanged. The encoder-only pass ran on
MPS. All head fitting, calibration, bootstrap statistics, and plots ran on CPU.
The decoder, reconstruction head, uncertainty head, and deployed
recoverability head were not executed during feature extraction.

The primary inputs were the pooled latent values only. They contain prompt
information through the original fourth input channel. Pixel uncertainty,
reconstruction statistics, and generator-known variables were excluded. H0 was
unweighted logistic regression; H1 was balanced logistic regression; H2 and H3
used fixed hidden widths 32 and 32/16. Validation-only selection chose balanced
minibatch sampling. H4 appended five cross-band centroid-shift summaries
computed from the observable input blend. A separate generator-metadata oracle
was analysis-only and is not deployable.

The moderate target was preserved despite extreme imbalance: 41/10,000
training, 5/1,500 validation, and 30/2,000 calibration positives. On validation,
H0 AUROC/AUPRC was 0.985/0.265, H1 was 0.983/0.516, H2 was 0.984/0.548, and H3
was 0.986/0.532. H2 did not materially beat H1: its paired AUPRC difference was
+0.033 with a 95% bootstrap interval of [-0.015, +0.184]. The strict
point-margin definition did not classify the representation as approximately
linearly accessible because 0.033 exceeded the predeclared 0.02 AUPRC margin;
the interval and calibration behavior nevertheless do not support a confident
nonlinear-sufficiency claim.

The validation-selected H2 head failed the stability test on calibration: raw
AUROC/AUPRC fell to 0.514/0.032. Its ambiguous-minus-valid score gap remained
inverted at +0.073. Catastrophic-source rejection AUROC was 0.654, while null
hallucination rejection was stronger at 0.948. Temperature scaling avoided an
exact zero threshold but inherited saturated MLP scores and realized 100%
coverage at nominal 95%, 90%, and 80%. Isotonic reduced 2,000 calibration
scores to four values with an 87.6% largest plateau.

H4 improved validation AUPRC by only +0.012 versus H2, with a paired 95%
interval of [-0.249, +0.275]. The cross-band centroid augmentation therefore
did not establish independent value. The oracle reached validation
AUROC/AUPRC 0.795/0.023 and was informative but substantially weaker in AUPRC
than the latent heads.

The authoritative outcome is **NO CLEAR IMPROVEMENT**. Balanced frozen heads
show that the representation contains useful information, but the combination
of extreme label scarcity, heterogeneous failure mechanisms, head instability,
and calibration ties prevents an operational selective-abstention claim. The
single recommended next experiment is to redesign and preregister the moderate
reliability-contract target and its failure-specific labels before any further
model or representation change.

Label provenance is intentionally heterogeneous because no new reconstruction
pass was authorized: training/validation use the persisted Phase-I teacher
outcomes that supervised Phase II, while calibration uses persisted frozen-R1
outcomes. The validation-to-calibration degradation may therefore combine head
instability, extreme scarcity, and teacher/R1 target-domain mismatch. This is
documented in `reports/postfinal_label_provenance_addendum.md` inside the run.

Development was not evaluated, the future lockbox remained sealed, and the
scientific reconstruction backbone was not retrained.
