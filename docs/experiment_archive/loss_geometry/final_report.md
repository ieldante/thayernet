# Thayer-LG frozen loss-geometry audit final report

Decision: **MIXED CAUSE**. Supported secondary categories are **OBJECTIVE MISALIGNMENT**, **LOSS-SCALE DOMINANCE**, **GRADIENT CONFLICT**, **PERMUTATION-MATCHING PATHOLOGY**, and descriptive **SCIENTIFIC-THRESHOLD EXTREMITY**.

Preregistration SHA-256: `405d460f3663fffdfc11f62235cd107e77fba2354ec5aac536378975807b057d`. It predates per-scene numerical inspection. This campaign performed no model inference or fitting, no neural-weight gradient or optimizer step, and no Atlas, development, or lockbox evaluation.

## Direct answers

1. **Was the Thayer-ME micro-overfit failure reproduced?** Yes, all 10 persisted aggregate metrics and every manifest/target hash reproduced.
2. **Could exact truths be represented?** Yes, for all 64 rows and both prompts under the frozen six-channel contract.
3. **Did exact truths pass every coverage metric?** Yes: own, alternate, both-mode, ordinary duplication, prompt mapping, and forward plausibility all passed.
4. **What loss did exact truth receive?** Mean scene objective 0.029376837541: ordinary 0.029400177300, ambiguous 0.029353497783.
5. **Did exact approved sets beat collapsed truth means?** Yes. A1 averaged 0.029353497783 versus A4 0.029364644433, and A1 was lower on 32/32 ambiguous rows. The margin was small.
6. **Did trained outputs receive lower loss than truth?** Ambiguous trained outputs did on 32/32 rows: 0.026542228181 versus 0.029353497783. Ordinary trained outputs were lower on 21/32 rows but had a higher mean because of outliers.
7. **Which terms dominated raw loss?** Forward-to-observed MSE dominated. At trained outputs it supplied 76.496% of ordinary and 86.711% of ambiguous total objective on average.
8. **Which terms dominated gradient magnitude?** At exact ordinary truth, forward supplied 100% of summed weighted term-gradient L2; at exact ambiguous truth it supplied 98.1%. At trained outputs it remained the largest mean contribution (42.5% ordinary, 49.2% ambiguous).
9. **Which gradients conflicted?** Set matching versus forward was negative on 63.281% of ordinary and 51.562% of ambiguous evaluations; severe conflict occurred on 31.250% and 25.000%.
10. **Did hard assignment create unstable or flat regions?** Yes at collapsed means: identity and swap tied on every baseline row, and 35.938% of perturbations at scale <=1e-5 flipped assignment relative to the deterministic tie choice.
11. **What happened on truth-to-compromise paths?** At alpha 0.05 toward trained outputs the mean objective fell from 0.029377 to 0.029047 while combined coverage fell from 1.0 to 0.094. The objective minimum was near alpha 0.5 with zero coverage.
12. **Was source-sum-preserving light transfer cheap?** Locally yes but not flat. A 5% transfer raised mean loss by only about 0.000256 while preserving forward consistency; reverse transfer already reduced coverage to 0.594. Positive transfer lost coverage at 20%.
13. **What local flat directions were found?** Float64 HVPs found no direction below the preregistered 1e-4 weak-curvature gate. Source-light exchange was weakest (median 1.16e-4 ordinary, 1.26e-4 ambiguous). The hard-assignment tie is nonsmooth rather than a smooth zero-curvature null space.
14. **Did direct full-objective optimization converge toward truth?** No. From exact truth it lowered objective 0.029377 to 0.029000, raised mean scientific distance to 8.265, reduced ordinary coverage to 0.03125, and reduced every ambiguous coverage rate to zero.
15. **Which diagnostic objective aligned best?** D2, source reconstruction/set matching plus ordinary concentration, ended with the lowest mean scientific distance (2.295) and 0.5625 both-mode coverage under the fixed 40-step protocol. This is diagnostic, not a selected replacement.
16. **Were ordinary and ambiguous rows comparably weighted?** Their medians were comparable (0.027438 versus 0.026442 at trained outputs) under identical pixel/prompt denominators. Ordinary mean loss was 1.238 times ambiguous because of realized outliers, not an unidentified factor-of-2/3/pixel-count bug.
17. **Primary problem?** Mixed: direct objective misalignment plus forward-term scale dominance and gradient conflict, with hard-assignment instability and a narrow scientific boundary. Output-contract and coverage-metric defects were rejected.
18. **Exactly one next experiment?** Prospectively rerun only the same 64-row Thayer-ME micro-overfit gate using source-set reconstruction plus ordinary concentration and a preregistered differentiable surrogate of the unchanged scientific distance; retain forward consistency solely as an evaluation gate. Do not run it in this campaign.
19. **Were Atlas, development, and lockbox untouched?** Yes: 0/0/0 scene or inference accesses. Only previously frozen forward/noise contract metadata was reused.
20. **Were historical checkpoints unchanged?** Yes, 575/575 were byte-identical and Thayer-LG created no checkpoint.

## Evidence inventory

- Baseline reproduction: `tables/micro_overfit_reproduction.csv` and `tables/trained_objective_reproduction.csv`.
- Truth representability: `tables/truth_representability_audit.csv`.
- Canonical losses and rankings: `tables/canonical_loss_decomposition.csv`, `tables/loss_term_scale_summary.csv`, and `tables/objective_ranking_summary.csv`.
- Gradients: `tables/gradient_norms.csv`, `tables/gradient_cosines.csv`, and `figures/gradient_cosine_heatmap.png`.
- Assignment geometry: `tables/assignment_geometry.csv` and `figures/assignment_margin_distributions.png`.
- Paths and curvature: `tables/objective_path_metrics.csv`, `figures/objective_paths/mean_objective_paths.png`, and `tables/local_curvature_hvp.csv`.
- Detached optimization: `tables/output_space_optimization_trajectories.csv` and `output_space_optimization/final_outputs.h5`.
- Loss/science regression and scale audit: `tables/loss_science_regression_full.csv` and `tables/numerical_scale_audit.csv`.
- Correctness and provenance: `tables/final_correctness_checks.csv`, `tables/frozen_input_hash_audit_final.csv`, and `diagnostics/final_correctness_audit.json`.

## Provenance and closure

- Correctness: PASS (26 checks; 0 failures); focused tests and compileall passed.
- Runtime: numerical audit 68.852 seconds; finalization 3.751 seconds.
- Run size at report creation: 272821539 bytes; free disk: 458994737152 bytes.
- Historical checkpoints: 575 unchanged.
- Model inference / model-gradient / model-optimizer steps: 0 / 0 / 0.
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
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/expert_specialization_contract.md
?? docs/explicit_psf_conditioning.md
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
?? scripts/audit_thayer_two_expert_architecture.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_loss_geometry.py
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
?? scripts/finalize_thayer_loss_geometry.py
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
?? tests/test_shape_constrained_quantile.py
?? tests/test_two_expert_decoder.py
```
