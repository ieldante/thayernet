# Thayer-ME two-expert ambiguity decoder final report

Decision: **FAILURE — REPRESENTATIONAL OR LOSS IMPLEMENTATION FAILURE; FULL TRAINING AND ATLAS PROHIBITED**.

Preregistration SHA-256: `c5e0c4bb80ccf58346b9c5053d4ac607f7316d2e823aac629537f09742ed4c62`. It predates model implementation and the isolated micro fit. Thayer-MH baseline reproduction and exact target-set reuse passed before the freeze.

## Direct answers

1. **Was the Thayer-MH failure reproduced?** Yes. Prompt swap 0.992, reconstruction ratio 0.864391, all four coverage rates 0, forward fractions 0.933333/1.0, and zero Atlas inference reproduced from persisted artifacts.
2. **Were target sets reused unchanged?** Yes. Exact read-only Thayer-MH tensors and hashes were used; nothing was regenerated or copied.
3. **Were Atlas source groups excluded?** Yes. All 2,000 pair gates and the expanded exclusion commitment passed.
4. **Did the micro-overfit capacity gate pass?** No.
5. **Could the two experts represent both approved modes on the tiny set?** No. Own, alternate, and both-mode coverage were all 0.
6. **What was the final parameter count?** 165,612: 72,672 shared encoder plus 46,470 per expert.
7. **Were expert decoder parameters independent?** Yes; storage overlap was zero and initialization seeds differed.
8. **Did both experts remain active?** Yes diagnostically. Final expert gradient norms were 0.124945 / 0.100587; activity did not produce truth coverage.
9. **Did promptability pass?** The micro promptability gate passed; full non-Atlas validation promptability was not run.
10. **What was set-level prompt-swap success?** 0.953125 on the microset.
11. **Did ordinary controls remain concentrated?** No. Median ordinary expert diameter was 5.165995, above 1.0.
12. **What was ordinary own-truth coverage?** 0.000000 for both experts covering both prompts.
13. **What was ordinary false-witness rate?** Not evaluated; the earlier micro-capacity gate failed.
14. **Did non-Atlas own-truth coverage become nonzero?** No; micro ambiguous own coverage was 0.
15. **Did alternate-truth coverage become nonzero?** No.
16. **Did both-mode coverage become nonzero?** No.
17. **Were both experts forward-consistent?** The frozen micro aggregate gates passed: ordinary 0.968750, ambiguous 1.000000. One ordinary scene still failed the all-expert criterion.
18. **Did near-collision diameter exceed ordinary diameter?** Not evaluated as a protected validation/control gate.
19. **Was Atlas evaluation authorized?** No.
20. **Did Atlas own-truth coverage become nonzero?** Not evaluated.
21. **Did Atlas alternate-truth coverage become nonzero?** Not evaluated.
22. **Did Atlas both-mode coverage become nonzero?** Not evaluated.
23. **Did witness count improve beyond 24/50?** Not evaluated.
24. **Did AUROC remain above or improve over 0.856?** Not evaluated.
25. **Did 4%-FPR recall remain above or improve over 0.32?** Not evaluated.
26. **Did controls remain bounded?** Not evaluated beyond the failed micro ordinary-concentration result.
27. **Was the campaign SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE** at the micro-overfit gate.
28. **What exact experiment should happen next?** Run one training-free frozen loss-geometry audit on the persisted micro targets and outputs, decomposing normalized objective terms against image, flux, color, centroid, and primary scientific distance. Do not fit a model or change coverage thresholds.
29. **Were final lockbox and unauthorized development data untouched?** Yes; access counts 0/0.
30. **Were all historical checkpoints unchanged?** Yes; 574/574 campaign-start files are byte-identical.

## Evidence and interpretation

- Baseline reproduction: `tables/baseline_reproduction.csv`.
- Target reuse: `tables/target_set_reuse_audit.csv` and `diagnostics/target_set_reuse_report.md`.
- Architecture and parameters: `diagnostics/two_expert_architecture.md`, `tables/model_parameter_inventory.csv`, and `paper_figures/thayer_me_architecture.png`.
- Micro isolation and results: `diagnostics/micro_overfit_20260712_203540/tables/microset_manifest.csv`, `micro_overfit_report.md`, gate tables, per-scene table, and persisted expert outputs.
- Specialization curves: `figures/micro_training_and_specialization.png`; ordinary and ambiguous examples are in `example_grids/`.
- Atlas galleries, witness comparisons, ROC curves, bootstrap intervals, calibration tables, and full-training checkpoints are absent by gate, not silently omitted after evaluation.

The independent experts remained trainable, prompt-sensitive, and largely forward-consistent, yet failed even the isolated training-set scientific coverage test. Parameter sharing alone is therefore not an adequate explanation for Thayer-MH's compromise. The present result cannot distinguish limited function class from misalignment between the normalized training loss and the frozen scientific coverage geometry; the latter must be audited before any capacity change.

## Correctness, provenance, and repository state

- Correctness audit: FAIL; 22 checks, 1 failures.
- Focused campaign/Atlas contract tests: 45 passed in 1.76s.
- Compileall, CSV/schema validation, `git diff --check`, staged-index audit, privacy/path grep, historical-checkpoint audit, and frozen-Atlas hash audit: see correctness table.
- Micro runtime: 136.45 seconds; run size at report creation: 23799743 bytes.
- Full fit / Atlas / development / lockbox access counts: 0 / 0 / 0 / 0.
- No full checkpoint, Atlas protocol, auditor, catalog policy, development result, lockbox result, or production claim exists.

Final Git status:

```text
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/ambiguity_atlas_v0.md
?? docs/ambiguity_set_supervision.md
?? docs/atlas_candidate_diversity.md
?? docs/atlas_expert_hypotheses.md
?? docs/atlas_flow_hypotheses.md
?? docs/atlas_set_hypotheses.md
?? docs/atlas_stochastic_hypotheses.md
?? docs/catalog_safety_coverage.md
?? docs/competing_hypothesis_recoverability.md
?? docs/conditional_calibration_experiment.md
?? docs/conditional_flow_prior_contract.md
?? docs/cross_deblender_audit_protocol.md
?? docs/deployable_scale_model.md
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/expert_specialization_contract.md
?? docs/explicit_psf_conditioning.md
?? docs/forward_consistency_contract.md
?? docs/gate_attainability_protocol.md
?? docs/latent_truth_coverage.md
?? docs/micro_overfit_capacity_gate.md
?? docs/model_family_diversity_contract.md
?? docs/multi_hypothesis_source_contract.md
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/partially_pooled_scale_correction.md
?? docs/permutation_invariant_decomposition_loss.md
?? docs/predicted_multigroup_calibration.md
?? docs/prior_posterior_gap.md
?? docs/prompted_resunet_candidate_family.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/subgroup_coverage_contract.md
?? docs/thayer_audit_failure_taxonomy.md
?? docs/thayer_audit_overview.md
?? docs/thayer_flow_prior.md
?? docs/thayer_multiple_hypotheses.md
?? docs/thayer_probabilistic_unet.md
?? docs/thayer_two_expert_decoder.md
?? docs/worst_group_quantile_training.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/audit_canonical_tensor_hash.py
?? scripts/audit_probabilistic_unet_architecture.py
?? scripts/audit_thayer_flow_prior_foundation.py
?? scripts/audit_thayer_multiple_hypotheses_architecture.py
?? scripts/audit_thayer_multiple_hypotheses_foundation.py
?? scripts/audit_thayer_two_expert_architecture.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_multiple_hypotheses.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/bootstrap_thayer_two_expert_decoder.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/evaluate_ambiguity_evidence_baselines.py
?? scripts/evaluate_deblenders_on_ambiguity_atlas.py
?? scripts/evaluate_probabilistic_unet_hypotheses.py
?? scripts/evaluate_probabilistic_unet_pre_atlas.py
?? scripts/evaluate_prompted_resunet_validation.py
?? scripts/evaluate_thayer_flow_prior_sufficiency.py
?? scripts/evaluate_thayer_multiple_hypotheses_pre_atlas.py
?? scripts/finalize_competing_hypotheses.py
?? scripts/finalize_conditional_calibration.py
?? scripts/finalize_probabilistic_unet.py
?? scripts/finalize_prompted_resunet_diversity.py
?? scripts/finalize_thayer_flow_prior.py
?? scripts/finalize_thayer_multiple_hypotheses.py
?? scripts/finalize_thayer_two_expert_decoder.py
?? scripts/found_thayer_two_expert_decoder.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/prepare_probabilistic_unet_data.py
?? scripts/prepare_prompted_resunet_data.py
?? scripts/prepare_thayer_multiple_hypotheses_data.py
?? scripts/preregister_thayer_probabilistic_unet.py
?? scripts/review_ambiguity_atlas.py
?? scripts/review_ambiguity_atlas_v0_observations.py
?? scripts/run_conditional_calibration.py
?? scripts/run_observability_distillation.py
?? scripts/run_probabilistic_unet_atlas.py
?? scripts/run_psf_conditioning.py
?? scripts/run_scale_correction.py
?? scripts/run_shape_constrained_quantile.py
?? scripts/run_thayer_two_expert_micro_overfit.py
?? scripts/train_probabilistic_unet.py
?? scripts/train_prompted_resunet_diversity.py
?? scripts/train_thayer_multiple_hypotheses.py
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/models_multiple_hypotheses.py
?? src/models_probabilistic_unet.py
?? src/models_prompted_resunet.py
?? src/models_two_expert_decoder.py
?? src/observability_distillation.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_canonical_tensor_hash.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_multiple_hypotheses.py
?? tests/test_observability_distillation.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_shape_constrained_quantile.py
?? tests/test_two_expert_decoder.py
```
