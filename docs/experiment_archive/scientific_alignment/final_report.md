# Thayer-SA scientific-alignment micro-overfit final report

Decision: **FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED**. The campaign stopped at the preregistered detached output-space preflight. Assignment auditing and neural micro-overfit were not authorized.

Preregistration SHA-256: `6ef3bc2505a5677e3acade93a818566105c26368f532dca40b60121f839ddc26`. It predates the official surrogate audit, official output-space preflight, and any neural fit. Neural optimizer steps, model checkpoints, Atlas evaluations, development accesses, and lockbox accesses are all zero.

## Direct answers

1. **Was the loss-geometry diagnosis reproduced?** Yes. All 13 frozen reproduction checks passed, including 54/64 compromises beating truth and all 32/32 ambiguous compromises beating truth.
2. **Did the differentiable scientific surrogate match the frozen metric?** Yes. Spearman was 0.990679, Kendall 0.957683, and threshold-side agreement 1.000000.
3. **Did exact truth receive numerical-near-zero surrogate loss?** Yes: 0.0, with zero gradient.
4. **Did truth beat all compromise configurations?** Yes under the corrected objective and surrogate.
5. **Did free-output optimization remain at truth?** Yes. Loss and tensor RMS remained 0, with all frozen coverage rates 1.0.
6. **Did free-output optimization move compromise outputs toward truth?** Directionally but insufficiently. Loss and mean scientific distance fell for trained, collapsed, and wrong-allocation starts, but none reached the required coverage hierarchy.
7. **Did hard set assignment remain stable enough?** Not evaluated. The earlier output-space gate failed, so assignment auditing was not reached.
8. **Was neural micro-overfit authorized?** No.
9. **Did ordinary truth coverage become high?** No neural result exists. Final detached-preflight ordinary coverage was 0.03125 from trained output, 0.09375 from collapsed mean, and 0.03125 from wrong allocation.
10. **Did ordinary expert diameter fall below 1.0?** Not evaluated as a neural gate.
11. **Did ambiguous own-truth coverage become high?** No. Detached-preflight finals were 0.00000, 0.28125, and 0.43750.
12. **Did alternate-truth coverage become high?** No; the same three finals were 0.00000, 0.28125, and 0.43750.
13. **Did both-mode coverage become high?** No; the same three finals were 0.00000, 0.25000, and 0.31250.
14. **Did prompt swap remain strong without an explicit prompt-swap loss?** Not evaluated after training because training was prohibited.
15. **Did forward consistency remain strong without a forward loss?** No neural conclusion exists. Detached trained-output optimization retained mean per-expert forward-consistent fraction 1.00000, but this is not a model result.
16. **Did source-sum consistency remain strong without a source-sum loss?** Not evaluated after neural training.
17. **Did gradient conflict disappear or materially decline?** The former forward/source conflict is absent by construction because forward loss is evaluation-only. No post-training gradient comparison exists.
18. **Did objective ranking align with scientific distance?** Canonically yes, but favorable ranking did not yield reliable coverage-reaching optimization.
19. **Was the campaign SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE**.
20. **Is full non-Atlas training now authorized?** No.
21. **What exact experiment should happen next?** Run one preregistered, training-free output-space conditioning campaign that keeps the same targets, thresholds, architecture, hard assignment, and initializations while testing a near-truth smooth component geometry; require detached coverage entry before neural fitting.
22. **Were Atlas, development, and lockbox untouched?** Yes: 0 / 0 / 0 accesses.
23. **Were all historical checkpoints unchanged?** Yes. No Thayer-SA checkpoint exists, and every campaign-start checkpoint is byte-identical.

## Evidence inventory

- Baseline reproduction: `tables/loss_geometry_reproduction.csv`.
- Preregistration and attainability: `preregistration/scientific_alignment_micro_overfit.md`, `preregistration/freeze_record.json`, and `tables/preregistered_gate_attainability.csv`.
- Surrogate tests and alignment: `tables/scientific_surrogate_unit_tests.csv`, `tables/surrogate_alignment_summary.csv`, and `figures/surrogate_vs_frozen_metric.png`.
- Loss and gradient scales: `tables/loss_gradient_scale_audit.csv`.
- Official detached paths: `tables/output_space_preflight_trajectories.csv`, `tables/output_space_preflight_gates.csv`, `objective_preflight/final_outputs.h5`, and `figures/output_space_preflight_paths.png`.
- Output examples: `example_grids/ordinary_preflight_outputs.png` and `example_grids/ambiguous_preflight_outputs.png`.
- Assignment, micro-overfit, post-training geometry, and neural forward-evaluation artifacts are absent by the failed prerequisite, not silently omitted after fitting.

## Provenance and closure

- Runtime including finalization: 120.598 seconds.
- Run size at report creation: 111405633 bytes; free disk: 457723432960 bytes.
- Correctness checks: see `tables/final_correctness_checks.csv` and `diagnostics/final_correctness_audit.json`.
- Neural execution: 0 optimizer steps; no CPU fallback and no MPS training launch.
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
?? docs/differentiable_scientific_distance.md
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/expert_specialization_contract.md
?? docs/explicit_psf_conditioning.md
?? docs/forward_consistency_as_gate.md
?? docs/forward_consistency_contract.md
?? docs/frozen_loss_geometry_audit.md
?? docs/gate_attainability_protocol.md
?? docs/latent_truth_coverage.md
?? docs/loss_scientific_alignment.md
?? docs/micro_overfit_capacity_gate.md
?? docs/model_family_diversity_contract.md
?? docs/multi_hypothesis_source_contract.md
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/output_space_optimization_audit.md
?? docs/partially_pooled_scale_correction.md
?? docs/permutation_invariant_decomposition_loss.md
?? docs/predicted_multigroup_calibration.md
?? docs/prior_posterior_gap.md
?? docs/prompted_resunet_candidate_family.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/scientific_alignment_micro_overfit.md
?? docs/scientific_alignment_objective.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/source_allocation_null_space.md
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
?? scripts/audit_thayer_loss_geometry.py
?? scripts/audit_thayer_loss_geometry_hvp.py
?? scripts/audit_thayer_multiple_hypotheses_architecture.py
?? scripts/audit_thayer_multiple_hypotheses_foundation.py
?? scripts/audit_thayer_scientific_alignment.py
?? scripts/audit_thayer_two_expert_architecture.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_loss_geometry.py
?? scripts/bootstrap_thayer_multiple_hypotheses.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/bootstrap_thayer_scientific_alignment.py
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
?? scripts/finalize_thayer_loss_geometry.py
?? scripts/finalize_thayer_multiple_hypotheses.py
?? scripts/finalize_thayer_scientific_alignment.py
?? scripts/finalize_thayer_two_expert_decoder.py
?? scripts/found_thayer_two_expert_decoder.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/prepare_probabilistic_unet_data.py
?? scripts/prepare_prompted_resunet_data.py
?? scripts/prepare_thayer_multiple_hypotheses_data.py
?? scripts/preregister_thayer_probabilistic_unet.py
?? scripts/preregister_thayer_scientific_alignment.py
?? scripts/review_ambiguity_atlas.py
?? scripts/review_ambiguity_atlas_v0_observations.py
?? scripts/run_conditional_calibration.py
?? scripts/run_observability_distillation.py
?? scripts/run_probabilistic_unet_atlas.py
?? scripts/run_psf_conditioning.py
?? scripts/run_scale_correction.py
?? scripts/run_shape_constrained_quantile.py
?? scripts/run_thayer_loss_geometry_audit.py
?? scripts/run_thayer_two_expert_micro_overfit.py
?? scripts/supersede_thayer_two_expert_correctness_audit.py
?? scripts/supplement_thayer_loss_geometry_regression.py
?? scripts/train_probabilistic_unet.py
?? scripts/train_prompted_resunet_diversity.py
?? scripts/train_thayer_multiple_hypotheses.py
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/loss_geometry.py
?? src/models_multiple_hypotheses.py
?? src/models_probabilistic_unet.py
?? src/models_prompted_resunet.py
?? src/models_two_expert_decoder.py
?? src/observability_distillation.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/scientific_alignment.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_canonical_tensor_hash.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_loss_geometry.py
?? tests/test_multiple_hypotheses.py
?? tests/test_observability_distillation.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_scientific_alignment.py
?? tests/test_shape_constrained_quantile.py
?? tests/test_two_expert_decoder.py
```
