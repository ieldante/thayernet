# Ambiguity Atlas v0 and Competing-Hypothesis Recoverability final report

Decision: **FAILURE AFTER ATLAS PASS — CANDIDATE-DIAMETER DETECTION FAILED; AUDITOR BLOCKED**.

Preregistration SHA-256: `2b54bf035f5a51721b4d012faa84060bb926a81610463fcb393c16d5f3f39185`. It predates new candidate inference
and Atlas optimization. Historical development and final lockbox scenes were
never opened, rendered, or evaluated.

## Direct answers

1. **Compatible deblenders:** three reproducible checkpoints but only one
   meaningfully distinct family cluster, `THAYER_COMPACT_PROMPTED_UNET`.
2. **Fixed forward contract:** yes. Exact g/r/z source addition, noise replay,
   band order, 0.2 arcsec pixels, and fixed 0.86/0.81/0.77 arcsec PSFs passed.
   The global consistency limit frozen on 3000
   calibration scenes is 1.0310823.
3. **Large-pool search:** yes; 100 genuine
   numerical near-collisions were found in 30,000 training/search scenes.
4. **Targeted optimization:** yes; 25/25
   bounded catalog-parameter optimizations produced valid feasibility pairs.
5. **Atlas pairs passing every frozen validation gate:** 25. The 25
   optimized pairs are separate route-feasibility evidence, not silently added
   to the frozen initial Atlas.
6. **Observation similarity:** frozen-pair mean squared whitened distance spans
   8.81738e-05 to
   0.000390451, far below
   the frozen 0.25 limit. The observed panels are strongly noise-dominated.
7. **Requested-truth difference:** primary scientific diameter spans
   5.348 to
   22.787 times
   the frozen scientific limit.
8. **Deblender behavior:** every family had at least one unsafe noisy requested
   reconstruction on all 25 pairs. Nearly-identical noisy pair outputs occurred
   on 15/25 for Condition C,
   0/25 for R0, and
   0/25 for R1.
9. **Confidently wrong:** not under R1's private diagnostic. Reconstructions were
   unsafe, but R1 recoverability was low (median 0.000126712,
   maximum 0.00026552); this narrow result does not rehabilitate
   the historically unstable confidence head.
10. **Average/prior-like outputs:** yes for Condition C on
    15/25 noisy pairs by the
    frozen output-diameter criterion; not for R0 or R1 under that criterion.
11. **Forward-consistent candidates per scene:** constructed sets retained two
    on 50/50 observations. Same-cluster model sets retained
    two on 19/50 and one on 31/50.
12. **Did plausible-set diameter identify Atlas cases?** Only partially:
    19/50 model-candidate observations formed witnesses.
13. **Did diameter beat confidence and residual?** No. Diameter AUROC is
    0.4712 with recall
    0.0000
    at the frozen control threshold; forward residual is
    0.5000 and R1
    unsafe-confidence is 0.9176.
14. **Black-box auditor authorized?** No: the diameter gate failed and fewer
    than three distinct families exist.
15. **Held-out-family transfer:** not trained or evaluated.
16. **Catastrophic false-safe rate by coverage:** not evaluated; no policy was
    authorized at 95/90/80/70/50% coverage.
17. **Atlas witnesses incorrectly accepted:** not evaluated as a policy. Direct
    constructed witnesses exist on 50/50 observations.
18. **Safe outputs rejected:** not evaluated; no admission rule exists.
19. **Accepted-catalog flux and centroid bias:** not evaluated.
20. **Operational definition:** finite competing explanations are viable direct
    evidence of non-identifiability for exhibited cases, but the available
    same-cluster candidate diameter is not a viable operational recoverability
    detector. Absence of a witness remains non-probative.
21. **Exact next experiment:** preregister and train one compact prompted
    ResUNet under the frozen BTK normalization/source-layer contract, validate
    deterministic full-decomposition replay, and rerun only the frozen 25-pair
    Atlas behavior/candidate-diversity audit. Do not train Thayer-Audit yet.
22. **Historical development and lockbox:** untouched; access counts are 0/0.
23. **Historical checkpoints:** all 556 files are byte-identical
    to the campaign-start inventory.

## Evidence inventory

- Family inventory: `tables/deblender_family_inventory.csv`.
- Forward audit/tests: `diagnostics/forward_model_audit.md` and
  `tables/forward_model_unit_tests.csv`.
- Search/optimization: `tables/atlas_pair_manifest.csv`,
  `tables/targeted_optimization_pair_manifest.csv`, and
  `optimization/counterfactual_optimization_trajectories.csv`.
- Atlas validation/gallery: `tables/atlas_pair_validation.csv`,
  `figures/ambiguity_atlas/`, and `figures/ambiguity_atlas_observed/`.
- Model behavior and plausible sets: `tables/atlas_deblender_behavior.csv`,
  `tables/plausible_candidate_sets.csv` if present, and
  `tables/model_candidate_witness_inventory.csv`.
- Baseline comparison: `tables/ambiguity_evidence_baselines.csv` and
  `diagnostics/ambiguity_evidence_baseline_report.md`.
- Figures: observation/truth galleries, forward-consistency distributions,
  plausible-set sizes, and deblender output-diameter plots. Transfer matrices,
  coverage curves, catalog-bias curves, and bootstrap intervals are absent by
  gate, not silently omitted after evaluation.

## Correctness and provenance

- Compileall: PASS.
- Main contract tests: PASS
  (`14 passed in 1.31s`).
- BTK contract tests: PASS
  (`17 passed in 1.42s`).
- CSV/schema validation: 34 files checked, 0 failures.
- `git diff --check`: PASS; staged index:
  empty.
- Privacy/path grep: PASS.
- Source split/catalog: unchanged;
  historical checkpoints: unchanged.
- Run disk usage: 299292	<REPOSITORY_ROOT>/outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627 KiB. Development/lockbox access: 0/0.
- No auditor tensor, model, threshold, held-out-family evaluation, catalog
  policy, or final-survey claim was created.

The Atlas exhibits finite competing explanations and therefore falsifies
practical uniqueness for its frozen cases. It does not prove uniqueness where
no witness is found, establish high-information ambiguity frequency, or support
model-agnostic auditing.

## Final repository state

```text
 M docs/current_status.md
 M docs/experiment_log.md
 M docs/limitations_and_next_steps.md
 M docs/model_card_thayer_select.md
 M docs/project_roadmap.md
?? docs/ambiguity_atlas_v0.md
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
?? docs/normalized_conformal_scale_protocol.md
?? docs/observable_regime_distillation.md
?? docs/partially_pooled_scale_correction.md
?? docs/predicted_multigroup_calibration.md
?? docs/proxy_shape_audit.md
?? docs/psf_information_sufficiency.md
?? docs/psf_provenance_audit.md
?? docs/shape_constrained_quantile_scale_correction.md
?? docs/shape_constrained_scale_model.md
?? docs/subgroup_coverage_contract.md
?? docs/thayer_audit_failure_taxonomy.md
?? docs/thayer_audit_overview.md
?? docs/worst_group_quantile_training.md
?? scripts/bootstrap_competing_hypotheses.py
?? scripts/build_ambiguity_atlas.py
?? scripts/calibrate_competing_forward_consistency.py
?? scripts/evaluate_ambiguity_evidence_baselines.py
?? scripts/evaluate_deblenders_on_ambiguity_atlas.py
?? scripts/finalize_competing_hypotheses.py
?? scripts/finalize_conditional_calibration.py
?? scripts/optimize_ambiguity_atlas_v0.py
?? scripts/prepare_ambiguity_atlas_v0.py
?? scripts/review_ambiguity_atlas.py
?? scripts/review_ambiguity_atlas_v0_observations.py
?? scripts/run_conditional_calibration.py
?? scripts/run_observability_distillation.py
?? scripts/run_psf_conditioning.py
?? scripts/run_scale_correction.py
?? scripts/run_shape_constrained_quantile.py
?? src/competing_hypotheses.py
?? src/conditional_calibration.py
?? src/observability_distillation.py
?? src/psf_conditioning.py
?? src/scale_correction.py
?? src/shape_constrained_quantile.py
?? tests/test_ambiguity_atlas.py
?? tests/test_competing_hypotheses.py
?? tests/test_conditional_calibration.py
?? tests/test_observability_distillation.py
?? tests/test_psf_conditioning.py
?? tests/test_scale_correction.py
?? tests/test_shape_constrained_quantile.py
```
