# Direct hierarchical catalog-safety auditor preregistration

Frozen UTC: `2026-07-14T19:47:06.107520+00:00`. This document is written and hashed before episode construction, new safety-label construction, auditor fitting, calibration access, threshold selection, held-family evaluation, bootstrap, or Atlas-v0 policy evaluation.

## Scientific hypothesis and interpretation boundary

Two freshly initialized 46,470-parameter expert decoders under D3's frozen square mapping, hard assignment, direct reconstruction loss, optimizer, and 5,000-step budget did not learn both approved hidden modes. That is the narrow D3 result. It does not test whether an external observer-only audit layer can classify unsupported queries or unsafe frozen reconstructions. Thayer-Audit v0 tests that binary operational catalog-safety hypothesis without retraining, repairing, or changing any deblender.

The only success outcome is `DIRECT_AUDITOR_FEASIBILITY_PASS`. `DIRECT_AUDITOR_PARTIAL`, `DIRECT_AUDITOR_FAILURE`, and `DATA_OR_PROVENANCE_FAILURE` retain the exact meanings supplied in the campaign brief. Held-family strong pass is separate and is required before any deblender-agnostic claim.

## Partitions, frozen families, and OOF rule

Only source-split SHA-256 `98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27` partitions `training`, `validation`, and `calibration` may be used. Development and sealed final-lockbox outcomes are unavailable to every selection, fit, calibrator, threshold, and success decision.

Eligible-family rules are frozen: immutable checkpoint/output hash; exact Gaussian prompt and g/r/z detected-electron source-layer semantics; aligned scene/prompt IDs; sufficient training, validation, and calibration coverage; passed promptability, or explicit negative/failure-domain designation; and no development/lockbox inference. Condition C is the expected primary valid family. R0/R1 remain the same architecture cluster, so their seeds/checkpoints cannot manufacture family diversity. Thayer-PU is stochastic-candidate evidence but is eligible for core family rotation only if complete aligned out-of-fit train, validation, and calibration requested-source outputs already exist under one frozen sampling rule; they are not generated merely to increase family count.

Auditor-training reconstruction rows must be persisted predictions from a base-model fold that excluded both episode source groups from fitting and validation-based checkpoint selection. For Condition C, the historical 8,000-scene fit plus 1,000-scene selection manifest defines one immutable historical base fold; only later training-partition episode rows with both groups absent from that fold are eligible. This held-out-fold subset is source-group safer than reusing in-sample outputs, but it is not described as a complete K-fold base-model cross-fit. Any overlap, missing identity, or in-sample row fails closed. No reconstruction model may be trained to fill a fold.

Validation uses frozen validation outputs for architecture/checkpoint selection only. Calibration fits temperatures and the operating threshold only. Source IDs, duplicate groups, family, and seed are grouping/evaluation metadata and never model inputs.

## Episode schema and hierarchical targets

Every row records scene ID, source-group IDs, partition, prompt ID/semantics, family/seed metadata, exact upstream tensor/output hashes, blend, Gaussian prompt, frozen requested-source reconstruction, residual, deployable scalars, supervision/evaluation labels, and provenance. Inference arrays and truth-derived supervision are stored separately and alignment-hashed.

PRE-AUDIT maps exact frozen query semantics to three labels: `UNIQUE_VALID -> VALID`, `NULL -> NULL_OR_WRONG`, and `AMBIGUOUS -> AMBIGUOUS_OR_UNSUPPORTED`. No heterogeneous composite label is created.

POST-AUDIT is fitted only on true `VALID` rows. `UNSAFE_TO_CATALOG` is the OR of the following frozen valid-query failures:

- requested-source symmetric image distance greater than `0.25`;
- any g/r/z symmetric relative-flux distance greater than `0.2`;
- either applicable g-r or r-z color error greater than `0.2` mag;
- applicable centroid displacement greater than `0.5` mean-PSF FWHM (mean PSF `4.066666667` pixels);
- source confusion (candidate closer by image MSE to the alternate isolated source);
- physical source-output failure: wrong shape, nonfinite value, or any negative detected-electron contribution;
- false-subtraction fraction greater than `0.2` on requested-source support above `0.01` of peak outside alternate-source support; empty protected support is not applicable;
- catastrophic reconstruction MSE worse than the observed-blend identity baseline.

`catastrophic` is the OR of image/flux/color/centroid/source-confusion failures and is reported separately from the broader unsafe label. Strict greater-than thresholds preserve the frozen inclusive safe boundary. Null/ambiguous rows never receive valid-query safety labels.

## Deployable input contract

PRE-AUDIT receives exactly four image channels: blend g/r/z divided by the frozen training-only band scales and the Gaussian prompt. POST-AUDIT receives exactly ten image channels: normalized blend g/r/z, prompt, normalized proposed reconstruction g/r/z, and normalized observation-minus-reconstruction residual g/r/z. Normalized image values are deterministically finite-mapped then clipped to `[-20.0, 20.0]` for auditor numerical stability; the reconstruction itself is not changed.

POST-AUDIT additionally receives exactly 25 deployable scalars: reconstruction band fluxes; residual band L1/L2; reconstruction peaks and sparsities; observation-to-reconstruction and prompt-to-reconstruction centroid displacement; reconstruction/residual and reconstruction/observation band ratios; and finite/nonnegative indicators. Scalar names and order are frozen in `src/direct_catalog_safety_auditor.py`. Scalars are standardized by training-only mean and standard deviation, with zero scale replaced by one.

No target, mask, true error, source/family identity, difficulty, SNR, obstruction, separation, flux ratio, morphology, generator parameter, future outcome, gradient, optimizer state, or D3 trajectory enters either network. Prompt jitter and disagreement are omitted because complete deployment-time coverage is not established on every compared partition/family. Consequently A2-D is unavailable and cannot be selected.

## Fixed architectures and training

A1 has 3x3 stride-2 convolution widths 16/32/64, GroupNorm, SiLU, global average pooling, one 64-unit hidden layer, and three outputs. It has `28307` trainable parameters (ceiling 100,000).

A2 has four 3x3 stride-2 blocks with widths 24/48/96/96, GroupNorm, SiLU, global average pooling, one 32-unit scalar MLP, concatenation into the exact 128-unit representation, one 128-unit fusion layer, and one unsafe logit. It has `155209` trainable parameters (ceiling 350,000).

Exactly seeds `2026071501`, `2026071502`, and `2026071503` are used. Training is MPS-only AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, at most 30 epochs, patience 6, gradient clipping 5.0. A1 uses training-prevalence inverse-frequency weighted three-class cross-entropy. A2 uses training-prevalence inverse-frequency weighted binary cross-entropy on valid rows only. A missing class is recorded as a degenerate scientific limitation; its present class retains unit weight and metrics requiring two classes are undefined rather than invented.

Per-seed A1 checkpoint selection is lexicographic: maximum validation macro-F1, maximum ambiguous recall, minimum validation cross-entropy, earliest epoch. Per-seed A2 selection is maximum validation AUPRC, maximum validation AUROC, minimum Brier, earliest epoch. Calibration, development, Atlas, and lockbox never select a checkpoint. The frozen final predictor is the unweighted mean of the three selected seed logits. B0 accepts every true-valid row. B1 replays the existing frozen hierarchical catastrophic scalar score on aligned rows as a reference ranking only; it cannot redefine the new label or threshold.

## Calibration and threshold policy

After all checkpoints freeze, calibration-only A1 logits receive one positive temperature. A2 logits receive one positive temperature; Platt and isotonic are diagnostic only, and isotonic cannot replace the primary calibration. The probability calibrators never change rankings.

The final policy predicts the calibrated A1 argmax, abstains for either invalid class, then accepts a predicted-valid request only when calibrated A2 unsafe probability is at most the selected threshold. Threshold candidates are the fail-closed value below the minimum calibration score plus every attainable unique calibration probability. Select maximum true-valid accepted coverage, breaking ties toward the larger threshold, subject simultaneously to: unsafe-rate reduction >=50%; catastrophic-rate reduction >=50%; true-valid accepted coverage >=50%; null acceptance <=5%; ambiguity acceptance <=10%. If no candidate satisfies all constraints, freeze the fail-closed below-minimum threshold, report zero post-gate acceptance, and classify FINAL POLICY FAIL without relaxing a gate.

## Held-family, bootstrap, Atlas, success, and stopping

If two genuinely eligible aligned families exist, leave-one-family-out models exclude the held family from training, validation selection, and calibration; family identity remains absent. Otherwise all held-family metrics are `UNRESOLVED`, and no deblender-agnostic claim is permitted. Fewer than two families does not stop the one-family core.

Use 300 deterministic connected-source-group bootstrap replicates. Report percentile intervals for macro-F1, invalid recalls, AUROC, AUPRC, Brier, ECE, coverage, accepted unsafe rate, catastrophic reduction, and invalid acceptance. Physical difficulty is post-freeze analysis only. No subgroup-conditional guarantee is claimed.

Only after architectures, seed checkpoints, temperatures, and the threshold freeze may the frozen policy be evaluated on existing Atlas-v0 pairs and existing matched controls. Report abstention rates, odds ratio, scores, and source/pair-group intervals where supported. Atlas does not determine campaign success.

PRE PASS requires validation/calibration macro-F1 >=0.85/0.82; null recall >=0.95 on both; ambiguity recall >=0.80 on both; and every class recall >=0.70. POST PASS requires validation/calibration AUROC >=0.90/0.85; validation AUPRC at least prevalence+0.15; calibrated Brier below the constant-prevalence baseline; calibrated ECE <=0.10; and three-seed validation AUROC SD <=0.03. FINAL PASS requires every threshold constraint plus source-group bootstrap lower bound for unsafe-rate reduction >0 and no eligible family worse than its accept-all-valid unsafe baseline. Held-family strong pass uses the exact >=0.80 AUROC, >=0.40 coverage, >=0.25 reduction, and <=0.20 ambiguity-acceptance gates supplied in the brief.

`DIRECT_AUDITOR_FEASIBILITY_PASS` requires PRE, POST, and FINAL PASS. A pass authorizes exactly the separately preregistered `Thayer-Audit Prospective Holdout v1`; it is not run here. PARTIAL/FAILURE recommends exactly one next experiment. D3 restart or a capacity ladder is prohibited.

Stop before fitting on any changed authoritative hash, nonempty staged index, MPS failure, missing/overlapping group identity, in-sample reconstruction row, train/validation/calibration group overlap, truth-derived inference feature, family-ID input, development/lockbox outcome access, label/input misalignment, parameter-ceiling failure, nonfinite training tensor, or historical checkpoint mutation. A failed scientific success gate does not authorize repair or retuning.

## Numerical attainability audit

All F1, recall, AUROC, AUPRC, coverage, rate, and ECE gates lie in [0,1]; Brier noninferiority has an attainable perfect value 0; seed SD has attainable value 0. AUPRC >= prevalence+0.15 is mathematically attainable iff unsafe prevalence <=0.85. Because new labels do not exist before this hash, this condition is re-audited immediately after label construction and cannot be changed; prevalence >0.85 makes POST PASS unattainable under the frozen gate. A 50% relative reduction is attainable whenever baseline risk is positive; baseline risk zero makes the reduction gate fail rather than divide by zero. The simultaneous threshold constraints have the constructive attainable case of accepting at least half of valid safe rows and no unsafe/invalid rows. Their empirical joint attainability is audited before threshold selection. Bootstrap lower bound >0 is attainable under strict separation. Parameter ceilings are attained by the counts above. Three seeds, 30 epochs, patience 6, batch 128, and 300 bootstrap replicates are positive finite budgets.

Development outcome access count and final-lockbox outcome access count must remain exactly zero. Atlas selection access count must remain zero. Historical reconstruction checkpoints and outputs remain immutable.
