# Thayer-OC output-space conditioning final report

Decision: **PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY**.

Preregistration SHA-256: `4202c5ddc9b9733138168b2acc650334e1ef10b002f7799071a3a12bc827e484`. It predates every per-scene HDF5 load, detached gradient, curvature evaluation, and optimizer action in this campaign.

## Direct answers

1. **Was preregistration completed before any per-scene numerical inspection?** Yes.
2. **Did every authoritative baseline reproduce?** Yes; 16/16 frozen checks passed.
3. **Was the T/D transformation exact?** Yes; 6/6 round-trip/projection cases passed.
4. **Did exact truth remain stationary under every method?** No.
5. **How ill-conditioned was raw output space?** The persisted compromise two-mode local condition estimate was `0`; this is directional, not a dense-Hessian condition number.
6. **Were allocation gradients weaker than common-mode gradients?** No; the compromise common/allocation gradient ratio was `0.723635`.
7. **Did raw L-BFGS outperform raw Adam?** Yes by the frozen minimum-coverage comparison.
8. **Did total/allocation coordinates help?** Yes.
9. **Did alternating optimization help?** No.
10. **Did threshold/Jacobian preconditioning help?** Yes.
11. **Which method achieved the highest ordinary coverage?** `C4_ALTERNATING_TD` from `collapsed_means` at `0.437500`.
12. **Which achieved the highest ambiguous own coverage?** `C1_RAW_LBFGS` from `thayer_me_experts` at `0.843750`.
13. **Which achieved the highest alternate coverage?** `C1_RAW_LBFGS` from `thayer_me_experts` at `0.875000`.
14. **Which achieved the highest both-mode coverage?** `C0_RAW_ADAM` from `sa_compromise` at `0.812500`.
15. **Did any method clear every frozen 90% gate?** No.
16. **Did assignment instability explain residual failures?** Not primary under the frozen diagnostics.
17. **Did positivity projection explain residual failures?** Not primary under the frozen diagnostics.
18. **What was the primary problem?** `SCIENTIFIC-BASIN EXTREMITY`.
19. **What exact neural experiment, if any, is now justified?** Run one separate preregistered direct feasibility-learning micro-audit that projects into the unchanged frozen scientific region.
20. **Were neural training, Atlas, development, and lockbox all untouched?** Yes: neural parameters/optimizer steps and protected accesses were all zero.
21. **Were all historical checkpoints unchanged?** Yes; 593/593 were audited.

## Evidence

- Baseline reproduction: `tables/baseline_reproduction.csv`.
- Coordinates: `tables/coordinate_roundtrip_tests.csv` and `diagnostics/output_coordinate_contract.md`.
- Gradient, curvature, assignment, and projection geometry: `tables/output_conditioning_geometry.csv` and `diagnostics/output_conditioning_report.md`.
- Detached comparison and truth stationarity: `tables/detached_optimization_comparison.csv` and `tables/conditioning_method_success_gates.csv`.
- Full trajectories and coverage entry: `tables/optimization_trajectories.csv`, `tables/coverage_entry_analysis.csv`, and `figures/optimization_trajectories/`.
- Checkpoint/provenance: `tables/checkpoint_inventory_before.csv`, `tables/checkpoint_inventory_after.csv`, and `logs/input_provenance.json`.

The scientific surrogate alignment passed, exact truth remained the zero-loss stationary solution, and no threshold, target, architecture, scalar-objective weight, or hard-assignment rule changed. This audit tests conditioning only. Forward consistency remains evaluation-only.

Exactly one next experiment is recommended and was not run: **Run one separate preregistered direct feasibility-learning micro-audit that projects into the unchanged frozen scientific region.**

## Closure

- Runtime: `795.328` seconds.
- Run bytes at report creation: `634623528`.
- Free disk bytes: `457011146752`.
- Historical checkpoints unchanged: `True`.
- Neural parameter count in optimizers: `0`.
- Neural training / Atlas / development / lockbox accesses: `0 / 0 / 0 / 0`.

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
?? scripts/bootstrap_thayer_output_conditioning.py
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
?? scripts/run_thayer_output_conditioning.py
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
?? src/output_conditioning.py
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
?? tests/test_output_conditioning.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_scientific_alignment.py
?? tests/test_shape_constrained_quantile.py
?? tests/test_two_expert_decoder.py
```
