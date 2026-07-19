# Thayer-Audit v0 final report

## Outcome

**DIRECT_AUDITOR_PARTIAL**. PRE-AUDIT pass: `False`. POST-AUDIT pass: `False`. FINAL POLICY pass: `False`. Held-family status: `UNRESOLVED_ONE_ELIGIBLE_FAMILY`. Integrity audit: `PASS`.

The preregistration SHA-256 is `3ca55b23997c8bfb0d6be2d395096020ab04df1d730f043d04a0b7c6d6a9f1c2`. The frozen outcome authorizes prospective Audit/Atlas v1: `False`. Exactly one recommendation follows: Run exactly one prospective physically compliant frozen-deblender family-diversity audit before another catalog-policy attempt.

## Required answers

1. **What narrow hypothesis did D3 falsify?** Two freshly initialized 46,470-parameter expert decoders under the frozen square mapping, hard assignment, direct reconstruction objective, optimizer, and 5,000-step budget did not learn the two approved hidden modes.
2. **Why does D3 failure not invalidate the audit layer?** D3 trained a truth-mode generator; this campaign tests separate truth-free classifiers over observed requests and frozen proposed outputs.
3. **Which frozen deblender families were eligible?** Condition C alone was core-eligible. R0/R1 share its family cluster; Thayer-PU lacked complete aligned out-of-fit outputs under one deployment sampling rule; prompted ResUNet failed promptability.
4. **Were training episodes truly OOF?** `14080` persisted rows came from a historical held-out base fold: both source groups were absent from Condition-C fitting and validation-based checkpoint selection. This is not claimed as a complete K-fold base-model cross-fit.
5. **Were source groups leak-free?** Yes; zero train/validation/calibration overlap and zero base-fit overlap entered auditor training.
6. **What inputs did PRE-AUDIT receive?** Normalized observed g/r/z plus the Gaussian prompt—four channels.
7. **What inputs did POST-AUDIT receive?** Ten image channels (blend 3, prompt 1, reconstruction 3, residual 3) plus 25 deployable scalars.
8. **Did any truth-only feature enter the auditor?** No. Truth was confined to supervision/evaluation labels.
9. **What were PRE validation/calibration metrics?** Macro-F1 `0.8947` / `0.7980`.
10. **What were null and ambiguous recall?** Validation `1.0000` / `0.9009`; calibration `0.9988` / `0.9100`.
11. **What were POST AUROC/AUPRC?** Validation `nan` / `1.0`; calibration `nan` / `1.0`.
12. **Did calibration improve Brier and ECE?** See `tables/post_audit_metrics.csv`; calibrated Brier/ECE are `0.000000` / `0.000000` against constant-prevalence Brier `0.000000`.
13. **What threshold was selected?** `0.9999999999999999` with status `NO_FEASIBLE_THRESHOLD_FAIL_CLOSED`.
14. **What accepted coverage resulted?** Validation `0.0000`; calibration `0.0000`.
15. **How much did unsafe rate fall?** Validation `1.0000`; calibration `1.0000`.
16. **How much did catastrophic rate fall?** Validation `1.0000`; calibration `1.0000`.
17. **What were null and ambiguity acceptance rates?** Calibration `0.0000` / `0.0000`.
18. **Did risk fall monotonically as coverage decreased?** `True` for empirical unsafe risk nondecreasing with coverage on the attainable calibration curve.
19. **Did held-family generalization pass?** No evaluation was available; deblender-agnostic generalization remains unproven.
20. **Did disagreement features materially help?** Not evaluated; A2-D was ineligible because disagreement lacked complete partition/family coverage.
21. **How did the frozen policy behave on Atlas v0 and controls?** Abstention `1.0000` vs `1.0000`, odds ratio `1.9804`; this is development-only.
22. **Did the direct audit layer pass?** `DIRECT_AUDITOR_PARTIAL`; the complete feasibility gate did not pass unless that token is `DIRECT_AUDITOR_FEASIBILITY_PASS`.
23. **Is a prospective sealed Audit/Atlas v1 authorized?** `False`.
24. **What exactly should happen next?** Run exactly one prospective physically compliant frozen-deblender family-diversity audit before another catalog-policy attempt.
25. **Were development and final lockbox untouched?** Yes; outcome access counts are 0/0.
26. **Were all historical checkpoints unchanged?** Yes: `743` audited, `0` mismatches.
27. **What reusable code/tests should eventually be committed?** `src/direct_catalog_safety_auditor.py`, `scripts/bootstrap_thayer_audit_v0.py`, `scripts/run_thayer_audit_v0.py`, `tests/test_direct_catalog_safety_auditor.py`, and `tests/test_thayer_audit_v0_artifacts.py`, after normal review.
28. **What generated artifacts should remain ignored?** This entire `outputs/runs/thayer_audit_v0_20260714_154655` tree: episodes, features, auditor checkpoints, calibration, thresholds, bootstrap, Atlas diagnostic, tables, figures, logs, diagnostics, provenance, and reports.

## Artifact inventory and integrity

- Frozen family inventory: `tables/frozen_deblender_family_inventory.csv`.
- OOF provenance and episode schema: `tables/audit_episode_inventory.csv` plus `episodes/*_manifest.csv`.
- Target prevalence and feature contract: `tables/audit_target_prevalence.csv`, `tables/deployable_feature_inventory.csv`.
- Architecture and training curves: `tables/auditor_architecture_inventory.csv`, `figures/training_curves.png`.
- Calibration and risk coverage: `tables/post_audit_metrics.csv`, `figures/post_calibration_reliability.png`, `figures/risk_coverage_curve.png`.
- Held-family result: `family_holdout/status.md`.
- Bootstrap: `bootstrap/source_group_bootstrap_intervals.csv`; unsafe-reduction lower 95% endpoint `1.0`.
- Atlas diagnostic: `atlas_diagnostic/summary.json`.
- Runtime: `475.2` seconds; run disk usage `2500069319` bytes; filesystem free `450664169472` bytes.
- Focused tests: `................                                                         [100%]
16 passed in 2.16s`
- Compileall / CSV / git diff / staged / README / checkpoint audit: `True` / `True` / `True` / `True` / `True` / `True`.

## Final Git status

```text
## thayer-select...origin/thayer-select
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/absolute_source_head.md
?? docs/allowlisted_file_access_contract.md
?? docs/ambiguity_atlas_v0.md
?? docs/ambiguity_set_supervision.md
?? docs/atlas_candidate_diversity.md
?? docs/atlas_expert_hypotheses.md
?? docs/atlas_flow_hypotheses.md
?? docs/atlas_set_hypotheses.md
?? docs/atlas_stochastic_hypotheses.md
?? docs/authoritative_full_l0_d3.md
?? docs/authoritative_square_full_l0_d3.md
?? docs/cached_encoder_feature_contract.md
?? docs/catalog_safety_coverage.md
?? docs/competing_hypothesis_recoverability.md
?? docs/conditional_calibration_experiment.md
?? docs/conditional_flow_prior_contract.md
?? docs/cross_deblender_audit_protocol.md
?? docs/d1_endpoint_persistence.md
?? docs/d1_reproducibility.md
?? docs/d3_artifact_contract.md
?? docs/d3_capsule_validation.md
?? docs/d3_control_policy_contract.md
?? docs/d3_dtype_contract.md
?? docs/d3_executable_bundle.md
?? docs/d3_executable_bundle_v3.md
?? docs/d3_executable_contract.md
?? docs/d3_expert_activity_policy.md
?? docs/d3_feature_trajectory.md
?? docs/d3_final_artifact_contract.md
?? docs/d3_final_integration_and_science.md
?? docs/d3_full_decoder_feature_trajectory.md
?? docs/d3_i41r1_dtype_contract.md
?? docs/d3_i41r1_independent_contract_compliance.md
?? docs/d3_i41r1_production_checkpoint_prewarm.md
?? docs/d3_i41r1_scientific_execution.md
?? docs/d3_l0_architecture_contract.md
?? docs/d3_l0_capacity_diagnosis.md
?? docs/d3_outcome_mapping.md
?? docs/d3_penultimate_feature_trajectory.md
?? docs/d3_policy_branch_coverage.md
?? docs/d3_prompt_collapse_policy.md
?? docs/d3_requirement_registry.md
?? docs/d3_runtime_bootstrap_contract.md
?? docs/d3_runtime_readiness.md
?? docs/d3_scientific_artifact_contract.md
?? docs/d3_scientific_contract_capsule.md
?? docs/d3_scientific_dependency_schema.md
?? docs/d3_scientific_execution_result.md
?? docs/d3_scientific_worker_contract.md
?? docs/d3_semantic_state_contract.md
?? docs/d3_sky_vector_contract.md
?? docs/d3_stop_event_precedence.md
?? docs/d3_synthetic_full_stack_preflight.md
?? docs/d3_tangent_policy.md
?? docs/d3_tensor_member_contract.md
?? docs/d3_threshold_contract.md
?? docs/d3_v41_contract_token_normalization.md
?? docs/d3_v41_scientific_execution.md
?? docs/d3_v41_serialization_bootstrap.md
?? docs/d3_v4_execution_bridge.md
?? docs/d3_v4_launcher_integration.md
?? docs/decoder_capacity_ladder.md
?? docs/decoder_execution_trace.md
?? docs/deployable_scale_model.md
?? docs/differentiable_scientific_distance.md
?? docs/direct_catalog_safety_auditor.md
?? docs/direct_scientific_feasibility_projection.md
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/expert_specialization_contract.md
?? docs/explicit_psf_conditioning.md
?? docs/feasible_target_learning.md
?? docs/feature_endpoint_artifact_contract.md
?? docs/fixed_feature_decoder_audit.md
?? docs/forward_consistency_as_gate.md
?? docs/forward_consistency_contract.md
?? docs/frozen_loss_geometry_audit.md
?? docs/full_l0_fixed_feature_d3.md
?? docs/gate_attainability_protocol.md
?? docs/held_deblender_family_evaluation.md
?? docs/independent_scientific_oracles.md
?? docs/l0_decoder_optimization_diagnosis.md
?? docs/latent_truth_coverage.md
?? docs/loss_scientific_alignment.md
?? docs/micro_capacity_after_projection.md
?? docs/micro_overfit_capacity_gate.md
?? docs/microset_capacity_threshold.md
?? docs/model_family_diversity_contract.md
?? docs/multi_hypothesis_source_contract.md
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/output_parameterization_selection.md
?? docs/output_space_conditioning_audit.md
?? docs/output_space_optimization_audit.md
?? docs/partially_pooled_scale_correction.md
?? docs/permutation_invariant_decomposition_loss.md
?? docs/physical_source_output_contract.md
?? docs/pre_and_post_deblend_audit.md
?? docs/predicted_multigroup_calibration.md
?? docs/prior_posterior_gap.md
?? docs/prompted_resunet_candidate_family.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/pure_forward_evaluator_contract.md
?? docs/relu_source_head.md
?? docs/repository_integrity_audit.md
?? docs/scientific_alignment_micro_overfit.md
?? docs/scientific_alignment_objective.md
?? docs/scientific_basin_geometry.md
?? docs/scientific_d3_result.md
?? docs/scientific_gradient_preconditioning.md
?? docs/scientific_postprocessing_isolation.md
?? docs/scientific_process_isolation.md
?? docs/scientific_region_projection_contract.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/source_allocation_null_space.md
?? docs/source_total_allocation_coordinates.md
?? docs/square_decoder_optimization.md
?? docs/square_l0_decoder_reachability.md
?? docs/square_source_head.md
?? docs/subgroup_coverage_contract.md
?? docs/thayer_audit_failure_taxonomy.md
?? docs/thayer_audit_overview.md
?? docs/thayer_audit_v0.md
?? docs/thayer_flow_prior.md
?? docs/thayer_multiple_hypotheses.md
?? docs/thayer_probabilistic_unet.md
?? docs/thayer_two_expert_decoder.md
?? docs/worst_group_quantile_training.md
?? docs/z_band_capacity_diagnostics.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/audit_canonical_tensor_hash.py
?? scripts/audit_probabilistic_unet_architecture.py
?? scripts/audit_thayer_capacity_ladder_contract.py
?? scripts/audit_thayer_flow_prior_foundation.py
?? scripts/audit_thayer_loss_geometry.py
?? scripts/audit_thayer_loss_geometry_hvp.py
?? scripts/audit_thayer_multiple_hypotheses_architecture.py
?? scripts/audit_thayer_multiple_hypotheses_foundation.py
?? scripts/audit_thayer_scientific_alignment.py
?? scripts/audit_thayer_two_expert_architecture.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_audit_v0.py
?? scripts/bootstrap_thayer_authoritative_d3.py
?? scripts/bootstrap_thayer_authoritative_d3_from_capsule.py
?? scripts/bootstrap_thayer_capacity_ladder.py
?? scripts/bootstrap_thayer_d3_executable_contract.py
?? scripts/bootstrap_thayer_d3_readiness.py
?? scripts/bootstrap_thayer_d3_scientific_capsule.py
?? scripts/bootstrap_thayer_feasibility_projection.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_loss_geometry.py
?? scripts/bootstrap_thayer_multiple_hypotheses.py
?? scripts/bootstrap_thayer_output_conditioning.py
?? scripts/bootstrap_thayer_output_parameterization.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/bootstrap_thayer_scientific_alignment.py
?? scripts/bootstrap_thayer_two_expert_decoder.py
?? scripts/build_ambiguity_atlas.py
?? scripts/build_d3_executable_capsule_v2.py
?? scripts/build_d3_scientific_capsule.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/close_thayer_feasibility_projection.py
?? scripts/close_thayer_output_conditioning.py
?? scripts/close_thayer_output_parameterization_micro.py
?? scripts/d3_capsule_evaluator_selftest.py
?? scripts/d3_scientific_capsule_guard.py
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
?? scripts/finalize_thayer_capacity_ladder.py
?? scripts/finalize_thayer_d3_executable_contract.py
?? scripts/finalize_thayer_d3_policy_contract.py
?? scripts/finalize_thayer_d3_readiness.py
?? scripts/finalize_thayer_d3_scientific_capsule.py
?? scripts/finalize_thayer_feasibility_projection.py
?? scripts/finalize_thayer_feasibility_projection_selection.py
?? scripts/finalize_thayer_fixed_feature_prestart_stop.py
?? scripts/finalize_thayer_flow_prior.py
?? scripts/finalize_thayer_loss_geometry.py
?? scripts/finalize_thayer_multiple_hypotheses.py
?? scripts/finalize_thayer_output_conditioning.py
?? scripts/finalize_thayer_output_parameterization.py
?? scripts/finalize_thayer_scientific_alignment.py
?? scripts/finalize_thayer_two_expert_decoder.py
?? scripts/found_thayer_two_expert_decoder.py
?? scripts/generate_thayer_d3_pv1a1_cache.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/postprocess_thayer_d3_readiness.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/prepare_probabilistic_unet_data.py
?? scripts/prepare_prompted_resunet_data.py
?? scripts/prepare_thayer_multiple_hypotheses_data.py
?? scripts/preregister_thayer_probabilistic_unet.py
?? scripts/preregister_thayer_scientific_alignment.py
?? scripts/record_thayer_fixed_feature_prestart_stop.py
?? scripts/replay_thayer_d3_synthetic_checkpoint.py
?? scripts/review_ambiguity_atlas.py
?? scripts/review_ambiguity_atlas_v0_observations.py
?? scripts/run_conditional_calibration.py
?? scripts/run_observability_distillation.py
?? scripts/run_probabilistic_unet_atlas.py
?? scripts/run_psf_conditioning.py
?? scripts/run_scale_correction.py
?? scripts/run_shape_constrained_quantile.py
?? scripts/run_thayer_audit_v0.py
?? scripts/run_thayer_authoritative_d3_v2.py
?? scripts/run_thayer_capsule_authoritative_d3.py
?? scripts/run_thayer_d3_executable_contract.py
?? scripts/run_thayer_d3_policy_contract.py
?? scripts/run_thayer_d3_postprocess_readiness.py
?? scripts/run_thayer_d3_postprocess_v4.py
?? scripts/run_thayer_d3_pv1a1_readiness.py
?? scripts/run_thayer_d3_pv1a1_scientific.py
?? scripts/run_thayer_d3_readiness.py
?? scripts/run_thayer_d3_scientific_readiness.py
?? scripts/run_thayer_d3_synthetic_preflight.py
?? scripts/run_thayer_feasibility_projection.py
?? scripts/run_thayer_loss_geometry_audit.py
?? scripts/run_thayer_output_conditioning.py
?? scripts/run_thayer_output_parameterization_micro.py
?? scripts/run_thayer_output_parameterization_preflight.py
?? scripts/run_thayer_scientific_d3.py
?? scripts/run_thayer_scientific_d3_process_v4.py
?? scripts/run_thayer_scientific_d3_process_v41.py
?? scripts/run_thayer_scientific_d3_process_v41r1.py
?? scripts/run_thayer_scientific_d3_v4.py
?? scripts/run_thayer_scientific_d3_v41.py
?? scripts/run_thayer_scientific_d3_v41r1.py
?? scripts/run_thayer_two_expert_micro_overfit.py
?? scripts/supersede_thayer_d3_readiness_closure.py
?? scripts/supersede_thayer_feasibility_projection_target_assembly.py
?? scripts/supersede_thayer_fixed_feature_checkpoint_inventory.py
?? scripts/supersede_thayer_fixed_feature_closure_privacy.py
?? scripts/supersede_thayer_two_expert_correctness_audit.py
?? scripts/supplement_thayer_d3_readiness_closure.py
?? scripts/supplement_thayer_loss_geometry_regression.py
?? scripts/thayer_d3_runtime_guard.py
?? scripts/train_probabilistic_unet.py
?? scripts/train_prompted_resunet_diversity.py
?? scripts/train_thayer_feasibility_projection_micro.py
?? scripts/train_thayer_multiple_hypotheses.py
?? scripts/validate_d3_capsule_independence.py
?? scripts/validate_d3_executable_capsule_v2.py
?? scripts/validate_d3_scientific_capsule.py
?? scripts/validate_thayer_d3_i41r1_candidate.py
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/d3_artifact_metadata.py
?? src/d3_audit_layer_pv1.py
?? src/d3_checkpoint_adapter_v41r1.py
?? src/d3_contract_tokens_v41.py
?? src/d3_contract_tokens_v41r1.py
?? src/d3_control_policy.py
?? src/d3_executable_contract.py
?? src/d3_execution_bridge_v4.py
?? src/d3_execution_mode_contract_r1.py
?? src/d3_hash_callsite_r1.py
?? src/d3_policy_engine.py
?? src/d3_policy_preflight.py
?? src/d3_policy_registry.py
?? src/d3_protocol_pv1a1.py
?? src/d3_requirement_registry.py
?? src/d3_semantic_checkpoint_path_r1.py
?? src/d3_state_machine.py
?? src/d3_tensor_hash_contract_r1.py
?? src/direct_catalog_safety_auditor.py
?? src/feasibility_projection.py
?? src/loss_geometry.py
?? src/models_multiple_hypotheses.py
?? src/models_probabilistic_unet.py
?? src/models_prompted_resunet.py
?? src/models_two_expert_decoder.py
?? src/observability_distillation.py
?? src/output_conditioning.py
?? src/output_parameterization.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/scientific_alignment.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_canonical_tensor_hash.py
?? tests/test_capacity_ladder_contract.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_d3_audit_layer_pv1.py
?? tests/test_d3_contract_tokens_v41.py
?? tests/test_d3_contract_tokens_v41r1.py
?? tests/test_d3_executable_contract.py
?? tests/test_d3_execution_bridge_v4.py
?? tests/test_d3_policy_contract.py
?? tests/test_d3_protocol_pv1a1.py
?? tests/test_d3_protocol_pv1a1_complete.py
?? tests/test_d3_pv1a1_scientific_entrypoint.py
?? tests/test_d3_readiness_process_isolation.py
?? tests/test_d3_scientific_capsule.py
?? tests/test_d3_semantic_checkpoint_path_r1.py
?? tests/test_d3_serialization_v41r1.py
?? tests/test_direct_catalog_safety_auditor.py
?? tests/test_feasibility_projection.py
?? tests/test_loss_geometry.py
?? tests/test_multiple_hypotheses.py
?? tests/test_observability_distillation.py
?? tests/test_output_conditioning.py
?? tests/test_output_parameterization.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_scientific_alignment.py
?? tests/test_shape_constrained_quantile.py
?? tests/test_thayer_audit_v0_artifacts.py
?? tests/test_thayer_d3_i41r1_independent_validator.py
?? tests/test_thayer_d3_i41r1_integration.py
?? tests/test_thayer_d3_i41r1_worker_launch.py
?? tests/test_thayer_scientific_d3_v4.py
?? tests/test_thayer_scientific_d3_v41.py
?? tests/test_two_expert_decoder.py
```
