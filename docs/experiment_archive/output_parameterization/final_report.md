# Thayer-OP fixed-L0 output-parameterization final report

Primary outcome: **NO MAPPING PASSES**.  
Selected mapping: **NONE**.  
Decoder-capacity ladder authorized: **False**.  
Strict correctness: **PASS** with `0` protocol failures.

Preregistration SHA-256: `c6abcb8ba70888bc9a14477968933713c0729a4e32065f7f2becfcec9c468597`. It predates every P0 tensor inspection, per-scene input load, model fit, and optimizer step.

## Direct answers

1. **Did all authoritative input hashes match?** Yes; all `19` frozen inputs matched before fitting and again at closure.
2. **Was preregistration completed before per-scene inspection?** Yes.
3. **Did ReLU represent every projected target?** True.
4. **Did square represent every projected target?** True.
5. **Did absolute value represent every projected target?** True.
6. **What gradient pathologies appeared?** ReLU has a dead nonpositive half-line; square has sign symmetry and a derivative that shrinks near zero; absolute value has a zero-subgradient cusp. All three retained finite nonzero derivatives over sampled strictly positive P0 support.
7. **Did every stop-rule self-test pass?** True (negative, NaN, Inf, target-hash mismatch, and MPS fallback simulation).
8. **Which mappings passed synthetic target fitting?** absolute, relu, square.
9. **Did ReLU pass the ordinary one-scene gate?** False.
10. **Did square pass it?** False.
11. **Did absolute value pass it?** False.
12. **Which mappings passed ambiguous one-scene both-mode coverage?** none.
13. **What were the eight-scene coverage results?** ReLU `NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE`; square `NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE`; absolute `NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE/NOT RUN BY GATE` (ordinary/own/alternate/both).
14. **What were the ordinary expert diameters?** ReLU `NOT RUN BY GATE`; square `NOT RUN BY GATE`; absolute `NOT RUN BY GATE`.
15. **What were the z-band errors?** ReLU `NOT RUN BY GATE`; square `NOT RUN BY GATE`; absolute `NOT RUN BY GATE`.
16. **Were physical negative values impossible throughout?** Yes; every fitted mapped tensor had minimum >=0 and zero negative events.
17. **Which mapping was selected prospectively?** `NONE` under the frozen lexicographic rule.
18. **Was selection stable under the frozen tie-breaker?** True.
19. **Is the decoder-capacity ladder now authorized?** False.
20. **If not, what blocker remains?** one-scene ambiguous truth-mode memorization
21. **What exact experiment should happen next?** Run one fixed-feature L0 expert-decoder optimization audit on the frozen ambiguous scene, retaining the same hard assignment and mapping while comparing the neural decoder trajectory with direct cached-feature output optimization.
22. **Were Atlas, development, and lockbox untouched?** Yes; scene-array access counts were `0/0/0`. Only already-frozen forward-noise metadata was read.
23. **Were all historical checkpoints unchanged?** Yes; `594/594` pre-campaign checkpoint files remained byte-identical. The campaign added `6` local checkpoints.

## Evidence

- Input provenance and environment: `logs/input_provenance.json`, `diagnostics/environment_snapshot.md`.
- Preregistration and attainability: `preregistration/fixed_l0_output_parameterization.md`, `tables/preregistered_gate_attainability.csv`.
- Representability and gradients: `tables/mapping_representability.csv`, `tables/gradient_numerical_preflight.csv`.
- Stop-rule self-test: `tables/stop_rule_self_tests.csv` and `output_contract/stop_self_tests/`.
- Synthetic fits: `tables/synthetic_fit_summary.csv`, `figures/synthetic_fit_curves.png`.
- One/eight-scene learning: `one_scene/`, `eight_scene/`, and `figures/output_mapping_learning_curves.png`.
- Coverage and selection: `tables/condition_summary.csv`, `tables/mapping_comparison.csv`, `logs/selection.json`.
- Output distributions and z-band diagnostics: `figures/output_distributions.png`, `figures/z_band_diagnostics.png`.
- Correctness: `tables/final_correctness_checks.csv`, `diagnostics/final_correctness_audit.json`.

The blocked Thayer-CL run measured no capacity result. Thayer-OP held L0 capacity fixed; only the in-forward physical output mapping changed. Loss and evaluation consumed the same mapped physical tensor. No 64-row ladder, capacity condition, Atlas scene, development scene, or lockbox scene was evaluated.

## Runtime and repository closure

- Campaign wall time: `802.096` seconds; finalizer runtime before report write: `4.705` seconds.
- Run bytes before report write: `10497005`; free disk bytes: `456279199744`.
- README unchanged; staged index empty; no commit, push, merge, delete, historical overwrite, or checkpoint mutation occurred.

Final Git status:

```text
M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/absolute_source_head.md
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
?? docs/decoder_capacity_ladder.md
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
?? docs/predicted_multigroup_calibration.md
?? docs/prior_posterior_gap.md
?? docs/prompted_resunet_candidate_family.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/relu_source_head.md
?? docs/scientific_alignment_micro_overfit.md
?? docs/scientific_alignment_objective.md
?? docs/scientific_basin_geometry.md
?? docs/scientific_gradient_preconditioning.md
?? docs/scientific_region_projection_contract.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/source_allocation_null_space.md
?? docs/source_total_allocation_coordinates.md
?? docs/square_source_head.md
?? docs/subgroup_coverage_contract.md
?? docs/thayer_audit_failure_taxonomy.md
?? docs/thayer_audit_overview.md
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
?? scripts/bootstrap_thayer_capacity_ladder.py
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
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/close_thayer_feasibility_projection.py
?? scripts/close_thayer_output_conditioning.py
?? scripts/close_thayer_output_parameterization_micro.py
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
?? scripts/finalize_thayer_feasibility_projection.py
?? scripts/finalize_thayer_feasibility_projection_selection.py
?? scripts/finalize_thayer_flow_prior.py
?? scripts/finalize_thayer_loss_geometry.py
?? scripts/finalize_thayer_multiple_hypotheses.py
?? scripts/finalize_thayer_output_conditioning.py
?? scripts/finalize_thayer_output_parameterization.py
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
?? scripts/run_thayer_output_parameterization_micro.py
?? scripts/run_thayer_output_parameterization_preflight.py
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
?? tests/test_two_expert_decoder.py
```
