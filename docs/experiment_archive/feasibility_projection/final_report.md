# Thayer-FP direct scientific-feasibility projection final report

Scientific decision: **FAILURE — PROJECTED TARGETS FEASIBLE; UNCHANGED THAYER-ME CANNOT MEMORIZE THEM**.

Strict correctness: **FAIL** with one protocol failure. Negative model outputs were observed at epoch 1, but the preregistered output-contract stop rule was not enforced and training continued to epoch 400. The later trajectory is diagnostic rather than fully protocol-valid. Projection evidence is unaffected.

Preregistration SHA-256: `c826734eab4d299b875aa7d69529816e6ca1db1cdefa63400deea334bc29e4d8`. It predates every per-scene load, detached projection, and neural optimizer step.

## Direct answers

1. **Did every authoritative baseline reproduce?** Yes. Every Thayer-ME and Thayer-SA check and all corrected Thayer-OC trajectory, stationarity, gradient-ratio, and unresolved-HVP checks reproduced.
2. **Were all exact truths feasible?** Yes, 256/256 scene/prompt/expert pairings.
3. **Were feasibility constraints frozen and unchanged?** Yes. Targets, hard assignment, thresholds, source layers, canonical hashes, and coverage definitions were unchanged.
4. **Were homotopy paths monotone?** Binary feasibility was monotone for 256/256 paths. Individual scientific-component ratios were monotone on 249/256 and nonmonotone on 7/256.
5. **How close to truth did each candidate need to move?** Interior alpha ranged `0.999459948` to `0.999999981`, with median `0.999979483`. Median normalized correction was `0.946369`.
6. **Which constraint was most often limiting?** `flux_z` on `173/256` pairings (`67.578%`).
7. **Did nearest-feasible refinement reduce correction?** Yes, P1 reduced median correction from `0.946369` to `0.885394`, but three P1 pairings exceeded the strict 0.95 training interior by about 1e-6, so P1 was ineligible.
8. **Ordinary feasible projected targets?** 100%.
9. **Ambiguous own-mode feasible targets?** 100%.
10. **Alternate-mode feasible targets?** 100%.
11. **Both-mode feasible target sets?** 100%.
12. **Did projected outputs remain forward-consistent?** Yes, 100% ordinary and 100% ambiguous under the final P0 set.
13. **Which method was frozen?** `P0_HOMOTOPY_INTERIOR`.
14. **Was unchanged Thayer-ME micro training authorized?** Yes, after the final strict projection gate.
15. **Did unchanged Thayer-ME memorize the projected targets?** No.
16. **Did ordinary coverage exceed 90%?** No; `0.000000`.
17. **Did ambiguous own coverage exceed 90%?** No; `0.000000`.
18. **Did alternate coverage exceed 90%?** No; `0.000000`.
19. **Did both-mode coverage exceed 90%?** No; `0.000000`.
20. **Did ordinary expert diameter fall below 1.0?** No; `3.563524`.
21. **Did prompt swap remain strong?** Yes; set prompt swap `0.984375`.
22. **Did forward consistency remain scientifically acceptable?** Yes diagnostically; ordinary/ambiguous `0.968750/1.000000`.
23. **Is existing 46k-per-expert capacity sufficient?** Not established; the unchanged model failed the microset test.
24. **Is a capacity ladder justified?** Yes, because target projection passed while neural learning failed. The strict stop-rule failure must be corrected prospectively.
25. **What exact experiment should happen next?** One separately preregistered controlled decoder-capacity ladder on the same 64 rows and frozen P0 targets, varying only expert-decoder capacity and enforcing nonnegative-output stopping from epoch 0.
26. **Were Atlas, development, and lockbox untouched?** Yes, access counts `0/0/0`.
27. **Were all historical checkpoints unchanged?** Yes, `593/593` historical checkpoints remained byte-identical; Thayer-FP added one campaign-local checkpoint.

## Projection evidence

- Final feasible-set contract: `docs/scientific_region_projection_contract.md` and `projection_targets/freeze_record_final.json`.
- Homotopy paths: `projection_trajectories/homotopy_paths.csv.gz`, `tables/homotopy_projection_summary.csv`, and `figures/feasibility_entry_paths/`.
- Alpha and correction distributions: `figures/alpha_correction_distributions.png`.
- Limiting constraints: `tables/limiting_constraint_frequency.csv`.
- Final method comparison: `tables/projection_method_comparison_final_superseding.csv`.
- Frozen target tensors and hashes: `projection_targets/projected_target_sets_final.h5` and `tables/projected_target_hashes_final.csv`.

The target-set assembly and strict-interior serialization corrections are preserved in append-only addenda. The final P0 selection supersedes the malformed ordinary-set and near-0.95 P1 artifacts without deleting them or changing a gate.

## Neural evidence and capacity conclusion

The MPS-only direct reconstruction run reached best epoch `395` and projection-target loss `0.00195134478`. Coverage stayed zero in all categories. Ordinary diameter was `3.563524` and negative-output fraction was `0.435717`. Prompt mapping and forward consistency remained strong, so the result directly implicates decoder capacity, shared-encoder conditioning, or output parameterization rather than target feasibility. Training curves are in `figures/micro_training_and_coverage.png`; ordinary and ambiguous grids are in `example_grids/`.

## Correctness, provenance, and closure

- Correctness checks: `28` total, `1` failure.
- Focused tests: `43 passed in 1.90s`.
- Compileall, CSV/schema validation, git diff checks, privacy/path grep, large-file inventory, projected-target integrity, architecture identity, no-input-leakage, and checkpoint audit are recorded in `tables/final_correctness_checks.csv`.
- Prior actual-objective HVP status remains `UNRESOLVED`; Thayer-FP made no curvature or condition-number claim.
- Campaign wall time at finalization: `1059.798` seconds; neural runtime `126.737` seconds.
- Run bytes at finalization: `175278965`; free disk bytes `456805765120`.
- Final target file SHA-256: `d58ef71e988de8584a78865f00747b931c1e65f6e406e437cebdca60a049b181`.
- Final checkpoint SHA-256: `3b673487a3f69dadbde6131521218335fd59a3542d679c5a85fc001ebf90b724`.
- README unchanged; staged index empty; no commit, stage, push, merge, delete, or overwrite occurred.

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
?? docs/direct_scientific_feasibility_projection.md
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/expert_specialization_contract.md
?? docs/explicit_psf_conditioning.md
?? docs/feasible_target_learning.md
?? docs/forward_consistency_as_gate.md
?? docs/forward_consistency_contract.md
?? docs/frozen_loss_geometry_audit.md
?? docs/gate_attainability_protocol.md
?? docs/latent_truth_coverage.md
?? docs/loss_scientific_alignment.md
?? docs/micro_capacity_after_projection.md
?? docs/micro_overfit_capacity_gate.md
?? docs/model_family_diversity_contract.md
?? docs/multi_hypothesis_source_contract.md
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/output_space_conditioning_audit.md
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
?? docs/scientific_basin_geometry.md
?? docs/scientific_gradient_preconditioning.md
?? docs/scientific_region_projection_contract.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/source_allocation_null_space.md
?? docs/source_total_allocation_coordinates.md
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
?? scripts/bootstrap_thayer_feasibility_projection.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_loss_geometry.py
?? scripts/bootstrap_thayer_multiple_hypotheses.py
?? scripts/bootstrap_thayer_output_conditioning.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/bootstrap_thayer_scientific_alignment.py
?? scripts/bootstrap_thayer_two_expert_decoder.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/close_thayer_output_conditioning.py
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
?? scripts/finalize_thayer_feasibility_projection.py
?? scripts/finalize_thayer_feasibility_projection_selection.py
?? scripts/finalize_thayer_flow_prior.py
?? scripts/finalize_thayer_loss_geometry.py
?? scripts/finalize_thayer_multiple_hypotheses.py
?? scripts/finalize_thayer_output_conditioning.py
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
?? scripts/run_thayer_feasibility_projection.py
?? scripts/run_thayer_loss_geometry_audit.py
?? scripts/run_thayer_output_conditioning.py
?? scripts/run_thayer_two_expert_micro_overfit.py
?? scripts/supersede_thayer_feasibility_projection_target_assembly.py
?? scripts/supersede_thayer_two_expert_correctness_audit.py
?? scripts/supplement_thayer_loss_geometry_regression.py
?? scripts/train_probabilistic_unet.py
?? scripts/train_prompted_resunet_diversity.py
?? scripts/train_thayer_feasibility_projection_micro.py
?? scripts/train_thayer_multiple_hypotheses.py
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/feasibility_projection.py
?? src/loss_geometry.py
?? src/models_multiple_hypotheses.py
?? src/models_probabilistic_unet.py
?? src/models_prompted_resunet.py
?? src/models_two_expert_decoder.py
?? src/observability_distillation.py
?? src/output_conditioning.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/scientific_alignment.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_canonical_tensor_hash.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_feasibility_projection.py
?? tests/test_loss_geometry.py
?? tests/test_multiple_hypotheses.py
?? tests/test_observability_distillation.py
?? tests/test_output_conditioning.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_scientific_alignment.py
?? tests/test_shape_constrained_quantile.py
?? tests/test_two_expert_decoder.py
```
