# Hierarchical Thayer-Select safety campaign final report

## Outcome

**Campaign classification: FAILURE under the frozen gates.** The query-state subproblem succeeded and valid-only tail ranking was strong, but the complete calibrated policy was operationally degenerate and did not improve on the historical R1 ranking at useful valid-scene coverage. It accepted 1/2,000 UNIQUE_VALID development scenes (0.05%) and no invalid scenes. This is not a deployable safety policy.

The overall project did not change. Recoverability is now represented as a derived hierarchy rather than a monolithic target. Condition C stayed frozen, development was generated only after policy freeze and evaluated once, and the lockbox remained untouched.

## Required answers

1. **Did partition drift explain the prior calibration collapse?** No physical source/scene shift did: maximum physical |SMD| was 0.0535. Source-reuse frequency shifted (|SMD| 0.276), reducing effective independence. The main causes were sparse heterogeneous labels (5 moderate/37 permissive validation positives), calibration underpowering, and isotonic ties.
2. **Were query semantics applied consistently?** Yes: zero mismatches across 40,500 historical contract checks, exact three-state unit tests passed, and fresh manifests replayed deterministically.
3. **Did the UNIQUE/NULL/AMBIGUOUS gate work?** Yes. Balanced validation macro F1/AUPRC were 0.881/0.923; recalls were UNIQUE 0.757, NULL 0.998, AMBIGUOUS 0.889.
4. **Was ambiguity inversion removed?** Yes in all five query-head seeds. Development query-gate acceptance was 9.2% for AMBIGUOUS versus 66.6% for UNIQUE_VALID.
5. **Which prompt-local feature family worked best?** F_COMBINED (global + multiscale prompt-local + reconstruction summary) with a small MLP. Standalone F_PROMPT_LOCAL was the best purely prompt-local family but underperformed the combination.
6. **How well were continuous risks predicted?** Five-seed means are:

| task | median Spearman | upper Spearman | pinball | top-10% recall | catastrophic AUROC | catastrophic AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| centroid | 0.955 | 0.956 | 0.033 | 0.823 | 0.934 | 0.984 |
| flux | 0.841 | 0.816 | 0.093 | 0.423 | 0.933 | 0.977 |
| image | 0.853 | 0.734 | 0.066 | 0.494 | 0.710 | 0.883 |
7. **Did upper-quantile predictions achieve calibrated coverage?** Yes marginally: natural coverage was image 0.900, flux 0.900, centroid 0.900; stratified diagnostic coverage was image 0.897, flux 0.901, centroid 0.907.
8. **Did confusion-risk ranking improve?** Yes diagnostically: five-seed AUROC mean 0.859 and AUPRC mean 0.217 at 2.3% validation prevalence, versus the prior 0.654 catastrophic-rejection AUROC. It did not rescue policy coverage.
9. **Did the hierarchy produce nondegenerate thresholds?** No. Natural calibration accepted 1/4,200 valid scenes; stratified diagnostic calibration accepted 0/1,000.
10. **What happened to NULL false acceptance?** Reconstruction-only exposed 100%; the query gate and full policy both accepted 0.0%/0.0%. Condition-C exposed hallucination fell from 5.4% to 0 because all NULL queries abstained.
11. **What happened to AMBIGUOUS false acceptance?** Reconstruction-only exposed 100%; query-gate/full-policy acceptance became 9.2%/0.0%. Exposed forced-source behavior fell from 19.2% to 1.6% and 0.
12. **Catastrophic valid failures at 95/90/80/70% diagnostic coverage?** Hierarchy: 95%: 0.825, 90%: 0.816, 80%: 0.793, 70%: 0.764. Random: 95%: 0.831, 90%: 0.830, 80%: 0.830, 70%: 0.835. The gain is negligible at 95%, modest by 70%, and statistically similar to R1.
13. **What reconstruction performance was sacrificed?** None by the safety heads: they never modify Condition C. Selection sacrifices essentially all coverage at the frozen operating point. Mean Condition-C valid image/flux/centroid risks were 2.257/11.735/4.057.
14. **Were results stable across head seeds?** Query classification was stable (macro-F1 SD 0.0026). Risk rank correlations were stable; raw linear-space widths were not, motivating log-space ensemble calibration. Confusion AUROC ranged 0.817-0.888.
15. **Did the frozen backbone remain unchanged?** Yes: Condition-C SHA-256 remained `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`, with zero trainable reconstruction parameters and exact repeated feature extraction.
16. **Was development evaluated only once?** Yes. Manifest SHA-256 `9ccb1626dcc158f43951ee15e03b6c00c3bcb01fc31e396a7d32e980d4ce51aa`; evaluation count 1; no retuning.
17. **Was the lockbox untouched?** Yes: zero lockbox groups/scenes/files and no sealed pixel access.
18. **SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE**, because the frozen complete policy is operationally degenerate and does not materially beat R1 at useful coverage, despite successful query validity and risk-ranking subcomponents.
19. **Ready for targeted Ambiguity Atlas construction?** Ready only for a separate targeted pilot, not the full Atlas: use simulator optimization to find close decision boundaries, matched source-pair construction, and multi-hypothesis truth sets. Do not use development or lockbox scenes.
20. **Exact next experiment?** A preregistered train/validation/calibration-only *risk-limit feasibility and conditional-conformal audit*: verify aperture flux-risk scaling and log-space tail stability, require at least 70% valid calibration coverage at a fixed catastrophic-risk budget, and compare hierarchy versus R1 before generating any new development set. Keep Condition C frozen.

## Correctness and provenance

- Final correctness audit: **PASS**.
- Fresh non-development scenes: 43,000; development scenes: 3,000.
- Approximate measured campaign runtime through freeze: 640.0 seconds, excluding one-time development and final audits.
- Run disk usage: 7.95 GiB.
- Historical checkpoints: unchanged.
- Development reporting correction: R1 macro and operating-point outcomes were recomputed from already persisted R1 outputs; no new inference or second evaluation occurred.

## Artifact index

- Drift: `diagnostics/partition_drift_report_superseding_source_reuse.md`, `tables/partition_drift_audit.csv`
- Query gate: `tables/query_gate_candidate_comparison.csv`, `figures/query_gate_confusion_matrices.png`, `figures/query_gate_per_class_pr.png`
- Risks/calibration: `tables/risk_head_seed_stability.csv`, `tables/conformal_calibration_summary.csv`, `figures/valid_risk_regression.png`, `figures/conformal_quantile_coverage.png`
- Development: `tables/development_per_sample.csv`, `tables/development_valid_operating_points_superseding_r1_outcomes.csv`, `figures/catastrophic_failure_rejection_curves.png`
- Galleries: `example_grids/development_accepted_rejected_gallery.png`
- Freeze/audit: `manifests/hierarchical_policy_freeze.json`, `manifests/hierarchical_policy_freeze_superseding_nondegeneracy.json`, `diagnostics/final_correctness_audit.json`

## Final git status

```text
## thayer-select...origin/thayer-select
?? docs/hierarchical_query_semantics.md
?? docs/hierarchical_recoverability_contract.md
?? docs/hierarchical_safety_experiment.md
?? docs/hierarchical_safety_policy.md
?? scripts/calibrate_hierarchical_safety.py
?? scripts/correct_hierarchical_sample_metadata.py
?? scripts/extract_hierarchical_safety_features.py
?? scripts/finalize_hierarchical_safety.py
?? scripts/prepare_hierarchical_safety_data.py
?? scripts/run_hierarchical_development_evaluation.py
?? scripts/run_hierarchical_safety.py
?? scripts/train_hierarchical_query_gate.py
?? scripts/train_hierarchical_risk_heads.py
?? src/hierarchical_safety.py
?? tests/test_hierarchical_query_gate.py
?? tests/test_hierarchical_safety.py
```
