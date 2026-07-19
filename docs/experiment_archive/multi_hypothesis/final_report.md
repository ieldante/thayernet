# Thayer-MH ambiguity-set decoder final report

Decision: **FAILURE — NON-ATLAS AMBIGUITY-SET TRUTH COVERAGE FAILED; ATLAS PROHIBITED**.

Preregistration SHA-256: `2281d1d30e2bb8e7cee1db996eaa2d6c013210f6f1092564d64ac1f209869d92`. It predates implementation, target rendering, and fitting. The selected checkpoint was chosen only by the frozen validation objective. Atlas, historical development, and final lockbox access counts are 0/0/0.

## Direct answers

1. **Previous posterior/decoder failure reproduced?** Yes: ordinary own, near own, and cross alternate posterior coverage reproduced at 0%; forward fractions reproduced at 0.929870605 / 1.0 / 1.0.
2. **Atlas groups excluded?** Yes: 36,288 groups spanning frozen pairs, targeted feasibility, controls, and the historical candidate pool.
3. **Target sets constructed?** 12,000/3,000 training ordinary/ambiguous observations; 1,500/500 validation; 1,500/500 calibration.
4. **All near-collision target sets validated?** Yes, 2,000/2,000 pairs.
5. **Final parameter count?** 120,022, or 931 above Condition C.
6. **Both hypotheses prompt-faithful?** Token-0/1 rates: 0.992 / 0.992.
7. **Prompt swap pass?** True; set-level observed 0.992.
8. **Ordinary controls concentrated?** False.
9. **Ordinary false-witness rate?** NOT_EVALUATED.
10. **Non-Atlas own-truth coverage nonzero?** 0.0.
11. **Non-Atlas alternate coverage nonzero?** 0.0.
12. **Non-Atlas both-mode coverage nonzero?** 0.0.
13. **Both hypotheses forward-consistent?** Ordinary / near fractions: 0.9333333333333333 / 1.0.
14. **Near diameter exceeded ordinary?** NOT_EVALUATED.
15. **Atlas evaluation authorized?** No.
16. **Atlas own-truth coverage nonzero?** Not evaluated.
17. **Atlas alternate coverage nonzero?** Not evaluated.
18. **Atlas both-mode coverage nonzero?** Not evaluated.
19. **Witness count improve over 24/50?** Not evaluated.
20. **AUROC remain above 0.856?** Not evaluated.
21. **4%-FPR recall remain above 0.32?** Not evaluated.
22. **Safe-control false witnesses bounded?** Not evaluated on Atlas controls.
23. **Campaign classification?** FAILURE.
24. **Exact next experiment?** Preregister one K=2 separate-expert decoder experiment with a shared prompt encoder and two compact expert decoders, retaining permutation-invariant approved-target matching, ordinary concentration, and every current exclusion and forward gate.
25. **Final lockbox and unauthorized development untouched?** Yes, 0/0 accesses.
26. **Historical checkpoints unchanged?** Yes, 560/560 byte-identical.

## Evidence and interpretation

The campaign created and replayed the requested prospective target sets, preserved prompt semantics, trained the compact shared K=2 architecture for all 30 MPS epochs, and selected one checkpoint by validation loss. The mandatory non-Atlas gate then failed at **non-Atlas ambiguity-set truth coverage failed**. Accordingly, no one-time Atlas protocol was frozen and no Atlas inference, calibration, gallery, ROC, bootstrap comparison, or post-Atlas tuning exists.

This result does not show that the approved ambiguity sets are invalid. It shows that the current shared decoder/token mechanism and frozen loss schedule did not satisfy the preregistered operational representation gates. The two outputs are candidate hypotheses, not probabilities or a complete posterior; absence of a covered second mode does not prove uniqueness.

## Correctness and repository state

- Correctness audit: PASS (16 checks; 0 failures).
- Focused tests: `35 passed in 1.71s`.
- Compileall / `git diff --check` / staged index: PASS / PASS / empty.
- Run size: 10429531763 bytes.
- Atlas / development / lockbox accesses: 0 / 0 / 0.

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
?? docs/explicit_psf_conditioning.md
?? docs/forward_consistency_contract.md
?? docs/gate_attainability_protocol.md
?? docs/latent_truth_coverage.md
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
?? docs/worst_group_quantile_training.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/audit_canonical_tensor_hash.py
?? scripts/audit_probabilistic_unet_architecture.py
?? scripts/audit_thayer_flow_prior_foundation.py
?? scripts/audit_thayer_multiple_hypotheses_architecture.py
?? scripts/audit_thayer_multiple_hypotheses_foundation.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_multiple_hypotheses.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
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
?? scripts/train_probabilistic_unet.py
?? scripts/train_prompted_resunet_diversity.py
?? scripts/train_thayer_multiple_hypotheses.py
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/models_multiple_hypotheses.py
?? src/models_probabilistic_unet.py
?? src/models_prompted_resunet.py
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
```
