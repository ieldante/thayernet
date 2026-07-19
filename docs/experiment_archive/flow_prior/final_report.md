# Thayer-PF conditional flow-prior truth-coverage final report

Decision: **FAILURE — POSTERIOR/DECODER INSUFFICIENT; FLOW PROHIBITED**.

Posterior/decoder preregistration SHA-256: `2fc96556e2db411de97f75a029c443721ccbb7cfd3dbcf19162462d0e2cc6fb7`. It predates
all Part-D inference. No flow preregistration was created because the mandatory
sufficiency gate failed.

## Direct answers

1. **Was the frozen truth-coverage metric correct?** Yes. All 9 independent synthetic cases passed without changing the frozen threshold.
2. **Did posterior samples cover the own truth?** No. K=32 coverage was 0% on 512 ordinary prompt evaluations and 0% on 500 near-collision own-posterior evaluations.
3. **Did cross-decoding demonstrate alternate-truth representability?** No. Coverage was 0%; alternate identity was 0.017625.
4. **Was a prior correction scientifically justified?** No. The decoder/posterior bottleneck gate failed.
5. **Was the decoder and posterior frozen?** Yes. The selected Thayer-PU checkpoint remained byte-identical and no parameters were trained.
6. **What flow architecture and mixture base were used?** None; flow implementation was prohibited by gate.
7. **What was the added parameter count?** 0.
8. **Did both mixture components remain active?** Not applicable; no mixture was implemented.
9. **Did the flow assign mass to both posterior modes?** Not applicable.
10. **Did it avoid excessive mode bridging?** Not applicable.
11. **Did non-Atlas own-truth coverage improve?** No flow result exists; posterior own-truth coverage itself was zero.
12. **Did non-Atlas alternate-truth coverage become nonzero?** No; cross-decode coverage was zero.
13. **Did forward consistency remain valid?** Yes diagnostically: ordinary / near-own / near-cross sample fractions were 0.929871 / 1.000000 / 1.000000.
14. **Did safe controls remain concentrated?** Persisted Thayer-PU control concentration reproduced; no new prior was evaluated.
15. **Was Atlas evaluation authorized?** No.
16. **How many Atlas witnesses were produced?** No new Atlas inference. The unchanged persisted Thayer-PU result is 24/50.
17. **Did witness count improve over 24/50?** Not evaluated.
18. **Did AUROC improve over 0.856?** Not evaluated.
19. **Did recall at 4% FPR improve over 0.32?** Not evaluated.
20. **Did Atlas own-truth coverage become nonzero?** Not evaluated.
21. **Did Atlas alternate-truth coverage become nonzero?** Not evaluated.
22. **What fraction of the posterior-prior gap was closed?** 0 by intervention: no flow was fitted; this is not a claim that the representations are equivalent.
23. **Did safe-control false witnesses remain bounded?** No new control inference was run; the persisted 0.08 Thayer-PU rate reproduced.
24. **Was the model SUCCESS, PARTIAL SUCCESS, or FAILURE?** **FAILURE** at the posterior/decoder sufficiency gate. Thayer-PF is not a model artifact.
25. **What exact experiment should happen next?** Preregister one ambiguity-set decoder-training experiment that presents both non-Atlas near-collision decompositions under each observationally equivalent condition while preserving prompt identity and forward consistency.
26. **Were final lockbox and unauthorized development data untouched?** Yes; access counts 0/0.
27. **Were all historical checkpoints unchanged?** Yes; 560/560 files are byte-identical.

## Scientific evidence

- Persisted non-Atlas and Atlas baselines reproduced exactly from immutable tables and logs; Atlas scenes were not reopened.
- Ordinary posterior coverage: 0.000000; median best distance 6.371040.
- Near-own posterior coverage: 0.000000; median best distance 7.826188.
- Near-cross alternate coverage: 0.000000; median best distance 7.832268.
- Near-own identity remained 0.978000, but cross alternate identity was only 0.017625.

Forward consistency did not rescue the gate. It establishes observation-level
recomposition within tolerance, not recovery of either known scientific truth.
No latent teachers, flow curves, mixture diagnostics, mode plots, prior samples,
Atlas galleries, or new ROC/sample-efficiency curves exist because the campaign
stopped before those stages.

## Correctness, provenance, and repository state

- Correctness audit: PASS_WITH_PREREGISTERED_POSTERIOR_DECODER_GATE_FAILURE; 18 checks, 0 failures.
- Focused unit tests: `28 passed in 1.83s`.
- Compileall, CSV validation, checkpoint/Atlas/source-partition hashes, `git diff --check`, empty staged index, public privacy grep, and large-file inventory: PASS.
- Campaign runtime through finalization: 28.34 seconds recorded in the active scripts; run size at report creation: 1638146 bytes.
- Flow fitting / Atlas / development / lockbox access counts: 0 / 0 / 0 / 0.

Final Git status:

```text
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/ambiguity_atlas_v0.md
?? docs/atlas_candidate_diversity.md
?? docs/atlas_flow_hypotheses.md
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
?? docs/thayer_probabilistic_unet.md
?? docs/worst_group_quantile_training.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/audit_canonical_tensor_hash.py
?? scripts/audit_probabilistic_unet_architecture.py
?? scripts/audit_thayer_flow_prior_foundation.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/bootstrap_thayer_flow_prior.py
?? scripts/bootstrap_thayer_probabilistic_unet.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/evaluate_ambiguity_evidence_baselines.py
?? scripts/evaluate_deblenders_on_ambiguity_atlas.py
?? scripts/evaluate_probabilistic_unet_hypotheses.py
?? scripts/evaluate_probabilistic_unet_pre_atlas.py
?? scripts/evaluate_prompted_resunet_validation.py
?? scripts/evaluate_thayer_flow_prior_sufficiency.py
?? scripts/finalize_competing_hypotheses.py
?? scripts/finalize_conditional_calibration.py
?? scripts/finalize_probabilistic_unet.py
?? scripts/finalize_prompted_resunet_diversity.py
?? scripts/finalize_thayer_flow_prior.py
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
