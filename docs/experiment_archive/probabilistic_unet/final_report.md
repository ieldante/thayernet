# Thayer-PU prompted probabilistic U-Net final report

Decision: **PARTIAL SUCCESS — STOCHASTIC DIAMETER IMPROVED; ATLAS TRUTH COVERAGE FAILED**.

Preregistration SHA-256: `eb62db24da7c77f35f56d1187f561f88a2e63e2acd89c01c859c1fd2213b2b09`.
Frozen Atlas protocol SHA-256: `6ce3e8754a8db44efafc401a44a2920cd52e65690ac7e76f61ad076299a73be0`.
The preregistration predates fitting; the Atlas protocol predates matched-control
thresholds and the single Atlas inference pass.

## Direct answers

1. **Canonical hashing fixed?** Yes. Campaign schema `thayer-per-sample-tensor-sha256-v1`; 11/11 invariance and sensitivity tests passed. Historical hashes were not changed.
2. **Atlas groups excluded?** Yes. All 59 frozen Atlas and targeted-feasibility groups were excluded from training, validation, calibration, and non-Atlas pair generation.
3. **Condition-C weights warm-started?** Every matching `enc1`, `enc2`, `bottleneck`, `dec2`, and `dec1` tensor loaded exactly. The historical three-channel reconstruction head was copied into both requested and companion halves; every tensor and hash is in `tables/condition_c_warm_start_inventory.csv`.
4. **Final parameter count?** 170,278 total; 97,606 trainable in phase 1 and 153,286 in phase 2.
5. **Posterior truth only during training?** Yes. It receives canonical source A/B truth only in the training-only API and diagnostics.
6. **Prior truth-free?** Yes. `p(z|blend)` accepts only the observed three-channel blend; prompts and truth are absent from its API.
7. **Promptability pass?** Yes.
8. **Majority-of-K prompt-swap success?** 0.987500.
9. **Best-of-K requested success?** 0.994250.
10. **Posterior collapse?** No. Every frozen latent-use gate passed.
11. **Active dimensions?** 4/8 on the 256-scene latent audit; final training reported 3.3765 batch-averaged active dimensions.
12. **Prior/posterior gap?** Prior best-of-16/posterior MSE ratio 0.914304; posterior-minus-prior identity gap -9.375e-05. Both passed.
13. **Forward-consistent prior fraction?** 0.951844 on non-Atlas validation and 1.000000 on Atlas.
14. **Controls concentrated?** Yes. Ordinary false witnesses were 0.059333, within the 0.10 gate.
15. **Greater non-Atlas near-collision diversity?** Yes. Near/matched-control median diameter ratio 1.264508, pair-cluster bootstrap lower endpoint 1.204412.
16. **Atlas authorized?** Yes, only after all non-Atlas gates passed. It was evaluated exactly once.
17. **Atlas model-generated witnesses?** 24/50.
18. **Improved over 19/50?** Yes, by 5; the frozen 30/50 target failed.
19. **AUROC improved over 0.4712?** Yes: 0.8560, bootstrap 95% interval [0.75118, 0.9416].
20. **Recall at 4% control FPR nonzero?** Yes: 0.3200.
21. **Correct-target coverage?** 0.0000.
22. **Paired alternate-truth coverage?** 0.0000.
23. **Safe-control false-witness rate?** 0.0800.
24. **SUCCESS, PARTIAL SUCCESS, or FAILURE?** **PARTIAL SUCCESS.** Promptability, latent use, prior quality, forward consistency, control concentration, AUROC, and low-FPR recall passed. The 30/50 witness target and both Atlas truth-coverage gates failed.
25. **Exact next experiment?** Preregister one focused conditional normalizing-flow prior correction on the frozen Thayer-PU representation, retaining every current non-Atlas and Atlas gate.
26. **Historical development and lockbox untouched?** Yes; access counts 0/0.
27. **Historical checkpoints unchanged?** Yes; 558/558 campaign-start historical files are byte-identical. Condition C remains `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`.

## Evidence and figures

- Architecture and parameters: `diagnostics/probabilistic_unet_architecture_report.md`, `tables/model_parameter_inventory.csv`, `paper_figures/thayer_pu_architecture.png`.
- Training and latent use: `figures/training_curves.png`, `tables/thayer_pu_epochs.csv`, `tables/latent_kl_per_dimension.csv`, `example_grids/latent_interpolation_grid.png`.
- Prior/posterior and prompt swaps: `diagnostics/prior_posterior_gap_report.md`, `example_grids/pre_atlas_promptability_grid.png`.
- Ordinary and near-collision samples: `example_grids/ordinary_control_prior_samples.png`, `example_grids/near_collision_prior_samples.png`.
- Forward consistency and concentration: `figures/forward_consistency_plausible_counts.png`, `figures/non_atlas_control_concentration.png`.
- Atlas samples and metrics: `example_grids/atlas_prior_sample_gallery.png`, `paper_figures/atlas_candidate_diameter_roc.png`, `paper_figures/atlas_sample_efficiency.png`, `paper_figures/atlas_witness_and_coverage.png`.
- Canonical hash and provenance: `diagnostics/canonical_hash_contract.md`, `tables/canonical_hash_tests.csv`, `logs/input_provenance.json`.

The K-prefix witness curve is 0, 1, 2, 8, 12, and 24 for K=1,2,4,8,16,32.
The candidate family becomes operationally discriminative, but none of its
retained Atlas samples approaches either frozen truth. Forward consistency alone
therefore does not establish posterior correctness or target coverage.

## Correctness, runtime, and repository state

- Correctness audit: PASS_WITH_PREREGISTERED_ATLAS_GATE_FAILURES; 24 checks, 0 failures.
- Focused 16-test campaign/Atlas suite and main-environment compileall: PASS.
- CSV/schema validation, `git diff --check`, staged-index audit, privacy/path grep, historical-checkpoint audit, and frozen-Atlas hash audit: PASS.
- Training runtime: 52.86 minutes; full campaign elapsed: 97.50 minutes.
- Run disk usage: 8886075462 bytes (8.276 GiB).
- Atlas/development/lockbox access counts: 1/0/0. Post-Atlas tuning: none.
- No black-box auditor, catalog admission policy, development evaluation, lockbox evaluation, or formal posterior-correctness claim was created.

Final Git status:

```text
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/ambiguity_atlas_v0.md
?? docs/atlas_candidate_diversity.md
?? docs/atlas_stochastic_hypotheses.md
?? docs/catalog_safety_coverage.md
?? docs/competing_hypothesis_recoverability.md
?? docs/conditional_calibration_experiment.md
?? docs/cross_deblender_audit_protocol.md
?? docs/deployable_scale_model.md
?? docs/empirical_ambiguity_certificate.md
?? docs/empirical_ambiguity_witness.md
?? docs/explicit_psf_conditioning.md
?? docs/forward_consistency_contract.md
?? docs/gate_attainability_protocol.md
?? docs/model_family_diversity_contract.md
?? docs/multi_hypothesis_source_contract.md
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/partially_pooled_scale_correction.md
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
?? docs/thayer_probabilistic_unet.md
?? docs/worst_group_quantile_training.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/audit_canonical_tensor_hash.py
?? scripts/audit_probabilistic_unet_architecture.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/evaluate_ambiguity_evidence_baselines.py
?? scripts/evaluate_deblenders_on_ambiguity_atlas.py
?? scripts/evaluate_probabilistic_unet_hypotheses.py
?? scripts/evaluate_probabilistic_unet_pre_atlas.py
?? scripts/evaluate_prompted_resunet_validation.py
?? scripts/finalize_competing_hypotheses.py
?? scripts/finalize_conditional_calibration.py
?? scripts/finalize_probabilistic_unet.py
?? scripts/finalize_prompted_resunet_diversity.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/prepare_probabilistic_unet_data.py
?? scripts/prepare_prompted_resunet_data.py
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
?? src/canonical_tensor_hash.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
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
?? tests/test_observability_distillation.py
?? tests/test_probabilistic_unet.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_shape_constrained_quantile.py
```
