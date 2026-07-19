# Pre-archive repository state

This record freezes the repository state observed before the canonical archive
was edited. It is an audit record, not a claim that ignored local artifacts are
safe to publish.

## Git state

| Field | Value |
| --- | --- |
| Observation time (UTC) | `2026-07-19T21:57:20Z` |
| Repository | `<REPOSITORY_ROOT>` |
| Current branch | `thayer-select` |
| Current HEAD | `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174` |
| Upstream | `origin/thayer-select` |
| Backup branch | `backup/pre-thayer-canonical-archive-20260719T215720Z` |
| Backup target | `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174` |
| Branch switched | no |
| Staged entries | `0` |
| Git index empty | yes (`git diff --cached --quiet` returned `0`) |
| Modified tracked files | `5` |
| Untracked, non-ignored files | `394` |

The five pre-existing tracked modifications were
`docs/current_status.md`, `docs/experiment_log.md`,
`docs/limitations_and_next_steps.md`, `docs/model_card_thayer_select.md`, and
`docs/project_roadmap.md`. They and all pre-existing untracked files are
user-owned work. The archive audit preserves their scientific content and does
not attribute their authorship to this curation pass.

## Workspace inventory boundary

The workspace occupied approximately `66 GiB`, of which `outputs/runs/` was
approximately `56 GiB`, `data/` `5.2 GiB`, `data_exploration/` `2.5 GiB`,
`outputs/checkpoints/` `134 MiB`, and `reports/` `23 MiB`. Local virtual
environments, `.git`, Python caches, notebook checkpoints, and tool caches are
environmental rather than research artifacts and are excluded from research
file counts.

Within that boundary the initial scan found:

| Category | Count |
| --- | ---: |
| Research files | 35,060 |
| Files below `outputs/runs/` | 31,492 |
| Top-level run directories | 124 |
| Run-tree `final_report.md` files | 78 |
| Run-tree protocol-named Markdown/JSON files | 60 |
| Run-tree `final_manifest.json` or `manifest.json` files | 13 |
| Documentation/report files | 392 |
| Source and script files | 402 |
| Test files | 111 |
| Notebooks | 2 |
| Configuration files | 20 |
| JSON/CSV/TSV artifacts | 5,108 |
| Figures | 851 |
| HDF5 files | 88 |
| NPY/NPZ files | 1,005 |
| PT/PTH/checkpoint-like files | 769 |
| Log/text/output-like files | 1,113 |
| Files larger than 5 MB (5,000,000 bytes) | 340 |
| Files larger than 50 MB (50,000,000 bytes) | 88 |
| Files larger than 100 MB | 75 |

Counts overlap where a file satisfies more than one category. Exact sizes,
SHA-256 values, dispositions, and duplicate groups for files above 5 MB are
recorded in [`large_file_audit.csv`](large_file_audit.csv). The canonical
artifact and reproducibility dispositions are recorded in
[`artifact_index.md`](artifact_index.md) and
[`reproducibility_index.md`](reproducibility_index.md).

## Initial safety findings

- Raw Galaxy10 HDF5 data existed at both `data/Galaxy10_DECals.h5` and
  `data_exploration/data/Galaxy10_DECals.h5`; each was 2,735,267,419 bytes.
  Both are protected from ordinary staging by ignore rules and are local-only.
- The largest artifact was
  `outputs/runs/br_v02_moderate_grouped_retrain_20260710_110917/replay_cache/train_blended_float32.npy`
  at 6,291,456,128 bytes. It is a generated replay cache and is local-only.
- Raw HDF5 scene tensors, NPY/NPZ arrays, checkpoints, generated observations,
  caches, optimizer state, and logs are non-committable unless a later row in
  the audit gives a narrow, explicit compact-artifact justification.
- `.gitignore` already excludes `data/*`, `outputs/`, HDF5/FITS/NPY/NPZ/PT/PTH,
  logs, caches, notebook checkpoints, and local environments. Git LFS was not
  introduced.
- Local branches `thayer-br-0.2` and `thayer-br-0.3` were merged into current
  history and their configured upstreams were `[gone]`. They were inspected as
  stale-branch candidates but were not deleted or otherwise changed by this
  archive task.

## Complete initial run-directory inventory

Every immediate child of `outputs/runs/` present at the observation time is
listed below. Repeated fail-closed launch attempts remain distinct here even
when the experiment ledger groups them as one engineering history.

```text
balanced_residual_20260708_184454
balanced_residual_20260708_184544
balanced_residual_20260708_184632
br_v02_moderate_grouped_retrain_20260710_110917
br_v03_delta_candidate_20260709_203034
br_v03_delta_candidate_20260710_031425
br_v03_delta_color_20260709_185630
clean_benchmark_plan_20260710_032839
clipping_audit_20260710_063312
clipping_audit_20260710_075442
dr10_foundation_20260711_024415
dr10_model_probe_20260711_155820
dr10_model_probe_20260711_160018
evaluation_audit_20260708_215421
evaluation_audit_20260708_220833
final_checkpoint_integrity_20260710_065316
final_test_manifest_prep_20260710_060845
final_test_manifest_prep_20260710_061737
final_test_manifest_prep_conservative_exclusion_setup_20260710_063746
preservation_null_tests_20260710_063312
preservation_null_tests_20260710_075442
research_correctness_audit_20260710_092241
residual_unet_20260708_154947
resunet_v04_candidate_20260710_043109
size_visual_audit_20260709_102251
source_artifact_audit_20260710_061059
source_leakage_audit_20260710_060927
source_leakage_audit_20260710_062157
source_leakage_audit_20260710_062950
stress_test_20260708_051648
stress_test_20260708_141153
stress_test_20260708_145221
thayer_ambiguity_atlas_v0_20260712_145627
thayer_audit_v0_20260714_154655
thayer_authoritative_d3_20260713_145040
thayer_authoritative_scientific_d3_20260714_070916
thayer_capacity_ladder_20260713_005215
thayer_capacity_ladder_20260713_013132
thayer_capsule_authoritative_d3_20260713_161342
thayer_competing_hypotheses_20260712_131111
thayer_d1_endpoint_replay_20260713_113715
thayer_d3_alignment_r1_20260714_020758
thayer_d3_executable_contract_20260713_162704
thayer_d3_executable_contract_20260713_164243
thayer_d3_executable_contract_20260713_164320
thayer_d3_hash_r1_20260714_012539
thayer_d3_i41r1_20260713_221426
thayer_d3_integration_science_20260713_182315
thayer_d3_onego_20260713_224729
thayer_d3_policy_contract_20260713_173955
thayer_d3_protocol_readiness_r1_20260714_074016
thayer_d3_pv1_readiness_r1_20260714_161723
thayer_d3_pv1a1_entrypoint_r3_20260714_175200
thayer_d3_pv1a1_readiness_r2_20260714_165947
thayer_d3_runtime_readiness_20260713_125352
thayer_d3_runtime_readiness_20260713_130859
thayer_d3_runtime_readiness_20260713_131306
thayer_d3_runtime_readiness_20260713_134646
thayer_d3_runtime_readiness_20260713_135017
thayer_d3_scientific_capsule_20260713_153815
thayer_d3_scientific_capsule_20260713_155637
thayer_d3_semantic_path_r1_20260714_024423
thayer_d3_v41_science_20260713_200621
thayer_external_photometry_convergence_correction_v0_20260718_205638
thayer_external_photometry_preflight_v0_20260718_154852
thayer_external_photometry_scene_stratification_v0_20260719_011606
thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954
thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340
thayer_family_e1_v0_20260714_214638
thayer_family_e1_v0_20260714_214715
thayer_family_e1p_v0_20260714_225228
thayer_family_e_v0_20260714_195256
thayer_feasibility_projection_20260712_234216
thayer_final_authoritative_d3_20260713_181323
thayer_final_authoritative_d3_policy_preflight_20260713_181323
thayer_fixed_feature_audit_20260713_025737
thayer_flow_prior_20260712_182516
thayer_flux_free_identifiability_v0_20260715_152950
thayer_flux_free_identifiability_v0_20260715_183310
thayer_full_l0_d3_20260713_101720
thayer_full_l0_d3r_20260713_121652
thayer_identifiability_v1_20260715_003220
thayer_loss_geometry_20260712_205733
thayer_model_9_preparation_v0_20260715_172217
thayer_multiple_hypotheses_20260712_190701
thayer_output_conditioning_20260712_225412
thayer_output_conditioning_20260712_225459
thayer_output_parameterization_20260713_022852
thayer_output_parameterization_20260713_022924
thayer_output_parameterization_20260713_023120
thayer_probabilistic_unet_20260712_163340
thayer_prompted_resunet_diversity_20260712_153854
thayer_prompted_resunet_diversity_20260712_153913
thayer_prompted_resunet_diversity_20260712_154122
thayer_psf_diverse_flux_identifiability_v0_20260717_081646
thayer_pu_batch_r1_20260714_224244
thayer_pu_eligibility_v1_20260714_213113
thayer_repository_integrity_20260713_031653
thayer_scientific_alignment_20260712_220315
thayer_scientific_d3_20260713_170508
thayer_select_btk_foundation_20260711_152613
thayer_select_conditional_calibration_20260712_021556
thayer_select_frozen_head_ablation_20260711_220756
thayer_select_hierarchical_feasibility_20260712_010729
thayer_select_hierarchical_safety_20260711_225657
thayer_select_hierarchical_safety_20260712_001405
thayer_select_observability_distillation_20260712_035843
thayer_select_prompt_ablation_20260711_164329
thayer_select_psf_conditioning_20260712_043319
thayer_select_psf_conditioning_20260712_043342
thayer_select_psf_conditioning_20260712_043415
thayer_select_psf_conditioning_20260712_043442
thayer_select_recoverability_20260711_191127
thayer_select_recoverability_20260711_191518
thayer_select_recoverability_seed_replication_20260711_203115
thayer_select_root_cause_analysis_20260711
thayer_select_scale_correction_20260712_024957
thayer_select_shape_constrained_quantile_20260712_032007
thayer_select_shape_constrained_quantile_20260712_032938
thayer_select_shape_constrained_quantile_20260712_033406
thayer_two_expert_decoder_20260712_203038
thayer_two_expert_decoder_20260712_203121
weighted_residual_20260709_030245
weighted_residual_20260709_043745
```

## Preservation rule

No local output was deleted, rewritten, force-added, or moved. The backup ref
protects only the pre-audit Git history; ignored scientific outputs remain
local files and require the retention policy in the data-use map. No push was
performed.
