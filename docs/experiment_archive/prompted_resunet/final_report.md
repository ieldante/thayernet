# Prompted ResUNet candidate-diversity final report

Decision: **FAILURE — PROMPTABILITY AND COMPLETE CANDIDATE-CONTRACT GATES FAILED; ATLAS NOT EVALUATED**.

Preregistration SHA-256: `d412f2071e49bf53ccf4633021d2ced8f43ffe32a160b537542be3ab10798884`. It predates the corrected model implementation and all fitting. An earlier untrained scaffold run stopped on an internal parameter-count inconsistency and is preserved separately.

## Direct answers

1. **Was the ResUNet genuinely architecturally distinct?** Structurally yes: six residual blocks, stride-2 residual downsampling, residual decoder fusion, and 199,219 parameters differ from Condition C's plain 119,091-parameter U-Net. Scientific family distinctness was not evaluated because promptability failed.
2. **Parameter count:** 199,219, or 1.672830 times Condition C; below both ceilings.
3. **Atlas groups excluded?** Yes. All 59 groups appearing in the frozen Atlas or targeted feasibility pairs were excluded from training and validation.
4. **Promptability passed?** No.
5. **Prompt-swap success:** 0.394667 (39.47%), below the frozen 0.80 gate.
6. **Reconstruction versus Condition C:** whole-image MSE 959480 versus 856307, ratio 1.120486; source-region MSE ratio 1.787554. ResUNet PSNR/SSIM were 25.4681/0.3859 versus 25.8006/0.3741.
7. **Candidate contracts aligned?** Semantic items aligned, but the complete gate failed because a different inference batch geometry changed bitwise candidate hashes. No corrective output scaling was applied.
8. **Cross-family distance above same-family distance?** Not evaluated.
9. **Was added diversity scientifically meaningful or error?** Not evaluated; Atlas candidate diversity was never computed.
10. **New Atlas witnesses:** Not evaluated; zero Atlas observations were opened for ResUNet inference.
11. **Witness count above 19/50?** Not evaluated. The authoritative value remains 19/50.
12. **Diameter AUROC above 0.4712?** Not evaluated. The authoritative value remains 0.4712.
13. **Recall at 4% FPR nonzero?** Not evaluated. The authoritative value remains zero.
14. **Controls bounded?** Not re-evaluated; historical controls and thresholds remain unchanged.
15. **Forward consistency valid?** Not evaluated on Atlas. Validation predictions were finite, but this does not substitute for the frozen Atlas decomposition test.
16. **Family artifacts?** The trivial validation probe found no constant/zero border, clipping, or zero-output fingerprint. Batch-geometry-sensitive hashing remained a contract defect.
17. **Useful second family?** No. Structural novelty without promptability is insufficient for admission.
18. **Third family justified?** No, not from this result.
19. **Black-box auditor blocked?** Yes.
20. **Exact next experiment:** preregister one coordinate-conditioned conditional VAE that produces multiple requested-source hypotheses under the same source-layer contract and Atlas exclusions; require non-Atlas promptability and forward-consistent multi-sample diversity before any Atlas evaluation. Do not train another deterministic U-Net variant.
21. **Development and lockbox untouched?** Yes; access counts are 0/0.
22. **Historical checkpoints unchanged?** Yes; all 556 start-inventory files are byte-identical.

## Evidence

- Architecture: `diagnostics/resunet_architecture_report.md`, `paper_figures/resunet_architecture.png`, and `tables/model_parameter_comparison.csv`.
- Source isolation and replay: `diagnostics/atlas_source_exposure_report.md`, `tables/atlas_source_exposure_audit.csv`, and `tables/manifest_replay_checks.csv` (11,500/11,500 pass).
- Training: `figures/training_curves.png`, `tables/prompted_resunet_epochs.csv`, and separate best/final checkpoints. Best epoch was 18; MPS runtime was 608.09 seconds.
- Promptability: `diagnostics/pre_atlas_promptability_report.md`, `tables/pre_atlas_validation_summary.csv`, `tables/pre_atlas_prompt_swap_per_scene.csv`, and `example_grids/pre_atlas_prompt_swap_grid.png`.
- Contract/leakage: `tables/candidate_contract_alignment.csv`, `diagnostics/candidate_contract_report_superseding.md`, and `tables/family_identity_leakage_probe.csv`.
- Decision/correctness: `tables/final_decision.csv`, `tables/final_correctness_checks.csv`, and `diagnostics/final_correctness_audit.json`.

Atlas witness comparisons, cross-family distance plots, candidate-diameter ROC, 4%-FPR results, Atlas forward-consistency tables, and Atlas bootstrap intervals are absent by the frozen stop gate, not omitted after inspection. The Atlas directories are empty and Atlas evaluation count is zero.

## Provenance and final state

- Full campaign elapsed time: 22.54 minutes.
- Run disk usage: 1463964517 bytes (1.363 GiB).
- Compileall, focused tests, CSV schema validation, historical checkpoint/Atlas hash audits, privacy/path grep, `git diff --check`, and staged-index audit: PASS.
- MPS-only training/inference: PASS; no CPU fallback.
- Historical development / lockbox access: 0 / 0.
- Final Git status:

```text
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/ambiguity_atlas_v0.md
?? docs/atlas_candidate_diversity.md
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
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/partially_pooled_scale_correction.md
?? docs/predicted_multigroup_calibration.md
?? docs/prompted_resunet_candidate_family.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/subgroup_coverage_contract.md
?? docs/thayer_audit_failure_taxonomy.md
?? docs/thayer_audit_overview.md
?? docs/worst_group_quantile_training.md
?? scripts/audit_ambiguity_atlas_v0.py
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/bootstrap_prompted_resunet_diversity.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/evaluate_ambiguity_evidence_baselines.py
?? scripts/evaluate_deblenders_on_ambiguity_atlas.py
?? scripts/evaluate_prompted_resunet_validation.py
?? scripts/finalize_competing_hypotheses.py
?? scripts/finalize_conditional_calibration.py
?? scripts/finalize_prompted_resunet_diversity.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/prepare_prompted_resunet_data.py
?? scripts/review_ambiguity_atlas.py
?? scripts/review_ambiguity_atlas_v0_observations.py
?? scripts/run_conditional_calibration.py
?? scripts/run_observability_distillation.py
?? scripts/run_psf_conditioning.py
?? scripts/run_scale_correction.py
?? scripts/run_shape_constrained_quantile.py
?? scripts/train_prompted_resunet_diversity.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/models_prompted_resunet.py
?? src/observability_distillation.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_observability_distillation.py
?? tests/test_prompted_resunet.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_shape_constrained_quantile.py
```

Atlas v0 itself still passes. This campaign failed before it could answer whether a prompted ResUNet adds useful candidate diversity; it does not weaken the direct ambiguity witnesses or authorize model-agnostic auditing.
