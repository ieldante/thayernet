#!/usr/bin/env python3
"""Create the compact, text-first Thayer experiment archive.

Only explicitly allowlisted compact authorities are copied. Dense arrays,
HDF5, checkpoints, logs, endpoint histories, and files above 5 MB are rejected.
Machine-specific home/repository paths are tokenized in archive text copies;
the original SHA-256 and the archive-copy SHA-256 are both recorded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "docs" / "experiment_archive"
MAX_BYTES = 5_000_000
BANNED_SUFFIXES = {
    ".bin", ".ckpt", ".fit", ".fits", ".fts", ".h5", ".hdf5",
    ".joblib", ".jsonl", ".log", ".npy", ".npz", ".optimizer",
    ".parquet", ".pickle", ".pkl", ".pt", ".pth", ".safetensors",
}
TEXT_SUFFIXES = {".md", ".json", ".csv", ".py", ".txt", ".yaml", ".yml", ".toml"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".png"}
HOME_PREFIX = str(Path.home())


@dataclass(frozen=True)
class Campaign:
    name: str
    authority: str
    supersession: str
    files: tuple[tuple[str, str], ...]
    note: str = ""


CAMPAIGNS = (
    Campaign("grouped_correctness", "HISTORICAL_FOUNDATION_AUTHORITY", "CURRENT_FOR_GROUPED_BENCHMARK_HISTORY", (
        ("outputs/runs/research_correctness_audit_20260710_092241/reports/final_report.md", "final_report.md"),
        ("outputs/runs/research_correctness_audit_20260710_092241/manifests/grouped_blends_20260710_103233_manifest_summary.json", "final_manifest.json"),
        ("outputs/runs/research_correctness_audit_20260710_092241/tables/grouped_retrain_comparison_summary.csv", "summary_table.csv"),
        ("outputs/runs/research_correctness_audit_20260710_092241/figures/grouped_existing_vs_retrain.png", "key_figure.png"),
    )),
    Campaign("btk_foundation", "ENGINEERING_FOUNDATION_AUTHORITY", "CURRENT_FOR_SIMULATION_FOUNDATION", (
        ("outputs/runs/thayer_select_btk_foundation_20260711_152613/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_select_btk_foundation_20260711_152613/tables/btk_scene_manifest.csv", "final_manifest.csv"),
        ("outputs/runs/thayer_select_btk_foundation_20260711_152613/tables/btk_split_summary.csv", "summary_table.csv"),
    )),
    Campaign("coordinate_prompting", "AUTHORITATIVE_PROMPTABILITY_RESULT", "CURRENT_WITHIN_CONDITION_C", (
        ("outputs/runs/thayer_select_prompt_ablation_20260711_164329/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_select_prompt_ablation_20260711_164329/tables/primary_metrics_macro.csv", "summary_table.csv"),
        ("outputs/runs/thayer_select_prompt_ablation_20260711_164329/reports/prompt_swap_summary.json", "final_manifest.json"),
    )),
    Campaign("recoverability_head", "HISTORICAL_PARTIAL_RESULT", "SUPERSEDED_AS_SAFETY_AUTHORITY_BY_LATER_IDENTIFIABILITY_WORK", (
        ("outputs/runs/thayer_select_recoverability_20260711_191518/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_select_recoverability_20260711_191518/tables/risk_coverage_operating_points.csv", "summary_table.csv"),
    ), "The source risk_coverage_summary.json is omitted because it is not a manifest and contains non-finite Infinity tokens. The development-example gallery is also omitted under the fail-closed protected-visualization rule."),
    Campaign("hierarchical_safety", "VALID_NEGATIVE_RESULT", "SUPERSEDED_BY_CORRECTIVE_PROTOCOL_REPORT_FOR_POLICY_STATUS", (
        ("outputs/runs/thayer_select_hierarchical_safety_20260711_225657/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_select_hierarchical_safety_20260712_001405/reports/final_report.md", "superseding_correction.md"),
        ("outputs/runs/thayer_select_hierarchical_safety_20260711_225657/diagnostics/campaign_contract.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_select_hierarchical_safety_20260711_225657/tables/development_metrics_macro_superseding_r1_outcomes.csv", "summary_table.csv"),
        ("outputs/runs/thayer_select_hierarchical_safety_20260711_225657/figures/null_ambiguous_false_accept_curves_superseding.png", "key_figure.png"),
    )),
    Campaign("ambiguity_atlas", "AUTHORITATIVE_DIAGNOSTIC_RESULT", "ATLAS_WITNESS_COUNT_CORRECTED_IN_FINAL_REPORT", (
        ("outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/preregistration/ambiguity_atlas_v0.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/tables/atlas_pair_manifest.csv", "final_manifest.csv"),
        ("outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/tables/final_decision.csv", "summary_table.csv"),
        ("outputs/runs/thayer_ambiguity_atlas_v0_20260712_145627/figures/atlas_deblender_output_diameter.png", "key_figure.png"),
    )),
    Campaign("prompted_resunet", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_THIS_MODEL_FAMILY", (
        ("outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/preregistration/prompted_resunet_candidate_diversity.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_prompted_resunet_diversity_20260712_154122/tables/final_decision.csv", "summary_table.csv"),
    )),
    Campaign("probabilistic_unet", "VALID_PARTIAL_RESULT", "ELIGIBILITY_REINTERPRETED_BY_LATER_PU_BATCH_CORRECTION", (
        ("outputs/runs/thayer_probabilistic_unet_20260712_163340/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_probabilistic_unet_20260712_163340/preregistration/frozen_atlas_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_probabilistic_unet_20260712_163340/tables/final_scientific_decision.csv", "summary_table.csv"),
        ("outputs/runs/thayer_probabilistic_unet_20260712_163340/paper_figures/atlas_sample_efficiency.png", "key_figure.png"),
    )),
    Campaign("pu_batch_correction", "SUPERSEDING_DEPLOYMENT_EXECUTOR_CORRECTION", "CURRENT_FOR_PU_ELIGIBILITY_AND_LABEL_SUPPORT", (
        ("outputs/runs/thayer_pu_batch_r1_20260714_224244/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_pu_batch_r1_20260714_224244/preregistration/thayer_pu_batch_invariance_repair.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_pu_batch_r1_20260714_224244/reports/final_decision.json", "final_manifest.json"),
        ("outputs/runs/thayer_pu_batch_r1_20260714_224244/eligibility_continuation/tables/combined_family_label_support.csv", "summary_table.csv"),
    )),
    Campaign("flow_prior", "VALID_NEGATIVE_PREFIT_RESULT", "CURRENT_FOR_FLOW_BRANCH", (
        ("outputs/runs/thayer_flow_prior_20260712_182516/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_flow_prior_20260712_182516/preregistration/posterior_decoder_sufficiency.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_flow_prior_20260712_182516/tables/posterior_decoder_sufficiency_summary.csv", "summary_table.csv"),
    )),
    Campaign("multi_hypothesis", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_SHARED_K2_DECODER", (
        ("outputs/runs/thayer_multiple_hypotheses_20260712_190701/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_multiple_hypotheses_20260712_190701/preregistration/ambiguity_set_multiple_hypotheses.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_multiple_hypotheses_20260712_190701/tables/non_atlas_near_collision_search_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_multiple_hypotheses_20260712_190701/figures/training_curves.png", "key_figure.png"),
    )),
    Campaign("two_expert", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_SEPARATE_EXPERT_BRANCH", (
        ("outputs/runs/thayer_two_expert_decoder_20260712_203121/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_two_expert_decoder_20260712_203121/preregistration/two_expert_ambiguity_decoder.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540/tables/microset_manifest.csv", "final_manifest.csv"),
        ("outputs/runs/thayer_two_expert_decoder_20260712_203121/diagnostics/micro_overfit_20260712_203540/tables/micro_overfit_gates.csv", "summary_table.csv"),
    )),
    Campaign("loss_geometry", "AUTHORITATIVE_MECHANISTIC_RESULT", "CURRENT_FOR_FULL_OBJECTIVE_DIAGNOSIS", (
        ("outputs/runs/thayer_loss_geometry_20260712_205733/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_loss_geometry_20260712_205733/preregistration/frozen_loss_geometry_audit.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_loss_geometry_20260712_205733/tables/gradient_alignment_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_loss_geometry_20260712_205733/figures/gradient_cosine_heatmap.png", "key_figure.png"),
    )),
    Campaign("scientific_alignment", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_SURROGATE_OBJECTIVE_BRANCH", (
        ("outputs/runs/thayer_scientific_alignment_20260712_220315/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_scientific_alignment_20260712_220315/preregistration/scientific_alignment_micro_overfit.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_scientific_alignment_20260712_220315/reports/post_final_protocol_addendum.md", "superseding_correction.md"),
        ("outputs/runs/thayer_scientific_alignment_20260712_220315/tables/surrogate_alignment_summary.csv", "summary_table.csv"),
    )),
    Campaign("output_conditioning", "VALID_PARTIAL_RESULT", "CURRENT_FOR_OUTPUT_SPACE_CONDITIONING", (
        ("outputs/runs/thayer_output_conditioning_20260712_225459/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_output_conditioning_20260712_225459/preregistration/output_space_conditioning.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_output_conditioning_20260712_225459/tables/detached_optimization_comparison.csv", "summary_table.csv"),
    )),
    Campaign("feasibility_projection", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_PROJECTED_TARGET_BRANCH", (
        ("outputs/runs/thayer_feasibility_projection_20260712_234216/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_feasibility_projection_20260712_234216/preregistration/direct_scientific_feasibility_projection.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_feasibility_projection_20260712_234216/tables/projection_method_comparison_final.csv", "summary_table.csv"),
    )),
    Campaign("output_parameterization", "VALID_NEGATIVE_RESULT", "CURRENT_FOR_FIXED_L0_MAPPING_COMPARISON", (
        ("outputs/runs/thayer_output_parameterization_20260713_023120/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_output_parameterization_20260713_023120/preregistration/fixed_l0_output_parameterization.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_output_parameterization_20260713_023120/tables/mapping_comparison.csv", "summary_table.csv"),
    ), "The training-microset model-output grid is noncentral generated imagery and remains local-only."),
    Campaign("fixed_feature_ladder", "AUTHORITATIVE_MECHANISTIC_RESULT", "CURRENT_FOR_D0_D2; D3_REQUIRES_SEPARATE_AUTHORITY", (
        ("outputs/runs/thayer_repository_integrity_20260713_031653/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_repository_integrity_20260713_031653/tables/d0_d3_summary_superseding_v2.csv", "summary_table.csv"),
        ("outputs/runs/thayer_repository_integrity_20260713_031653/preregistration/feature_cache_protocol_clarification_superseding_v4.md", "frozen_protocol.md"),
    )),
    Campaign("d3_pv1a1", "AUTHORITATIVE_LOCAL_L0_RESULT", "MIXED_CAUSE_MAPPER_DOES_NOT_SUPERSEDE_UNKNOWN_UNRUN_BRANCHES", (
        ("outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/protocol_bundle/THAYER-D3-PV1-A1/d3_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_d3_pv1a1_entrypoint_r3_20260714_175200/scientific_run/authoritative_pv1a1_20260714_182005_552915/reports/final_outcome.json", "final_manifest.json"),
    )),
    Campaign("direct_audit", "VALID_PARTIAL_RESULT", "POST_INTERPRETATION_CLARIFIED_BY_APPEND_ONLY_ADDENDUM", (
        ("outputs/runs/thayer_audit_v0_20260714_154655/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_audit_v0_20260714_154655/preregistration/direct_hierarchical_catalog_safety_auditor.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_audit_v0_20260714_154655/reports/final_report_addendum.md", "superseding_correction.md"),
        ("outputs/runs/thayer_audit_v0_20260714_154655/tables/final_policy_metrics.csv", "summary_table.csv"),
        ("outputs/runs/thayer_audit_v0_20260714_154655/figures/risk_coverage_curve.png", "key_figure.png"),
    )),
    Campaign("family_e_invalid", "ENGINEERING_INVALID_NEGATIVE_PREFLIGHT", "SUPERSEDED_ONLY_AS_PHYSICAL_CONTRACT_BY_SIGNED_RESIDUAL", (
        ("outputs/runs/thayer_family_e_v0_20260714_195256/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_family_e_v0_20260714_195256/preregistration/family_e_nonnegative_flux_conserving_eligibility.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_family_e_v0_20260714_195256/architecture/architecture_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_family_e_v0_20260714_195256/tables/micro_overfit_results.csv", "summary_table.csv"),
    )),
    Campaign("signed_residual_preflight", "AUTHORITATIVE_PHYSICAL_CONTRACT_CORRECTION", "CURRENT_FOR_SIGNED_NOISE_RESIDUAL_CONTRACT", (
        ("outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/preregistration/signed_noise_residual_physical_contract_preflight.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340/tables/gate_results.csv", "summary_table.csv"),
    )),
    Campaign("family_e1", "VALID_NEGATIVE_RESULT", "PROMPT_IDENTITY_DIAGNOSED_BY_E1P", (
        ("outputs/runs/thayer_family_e1_v0_20260714_214715/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_family_e1_v0_20260714_214715/preregistration/family_e1_nonnegative_source_signed_residual_model.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_family_e1_v0_20260714_214715/architecture/architecture_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_family_e1_v0_20260714_214715/tables/micro_overfit_results.csv", "summary_table.csv"),
        ("outputs/runs/thayer_family_e1_v0_20260714_214715/figures/micro_overfit_curves.png", "key_figure.png"),
    )),
    Campaign("family_e1p", "AUTHORITATIVE_PROMPT_DIAGNOSIS", "CURRENT_FOR_E1_PROMPT_MECHANISM", (
        ("outputs/runs/thayer_family_e1p_v0_20260714_225228/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_family_e1p_v0_20260714_225228/preregistration/family_e1p_paired_prompt_identity_intervention.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_family_e1p_v0_20260714_225228/manifests/difficult_one_scene_paired_scene_manifest.csv", "final_manifest.csv"),
        ("outputs/runs/thayer_family_e1p_v0_20260714_225228/tables/micro_overfit_results.csv", "summary_table.csv"),
        ("outputs/runs/thayer_family_e1p_v0_20260714_225228/figures/prompt_conditioning_diagnosis.png", "key_figure.png"),
    )),
    Campaign("recoverability_nullspace", "AUTHORITATIVE_UNRESTRICTED_IDENTIFIABILITY_RESULT", "CURRENT_FOR_UNRESTRICTED_SOURCE_ALLOCATION", (
        ("reports/thayer_recoverability_v0.md", "final_report.md"),
    )),
    Campaign("oracle_flux_identifiability", "HISTORICAL_CONDITIONAL_AUTHORITY", "SUPERSEDED_FOR_NONORACLE_INFERENCE_BY_FLUX_FREE_RESULT", (
        ("outputs/runs/thayer_identifiability_v1_20260715_003220/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_identifiability_v1_20260715_003220/preregistration/prior_and_analysis_freeze.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_identifiability_v1_20260715_003220/logs/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_identifiability_v1_20260715_003220/tables_final/full_identifiability_metrics.csv", "summary_table.csv"),
    )),
    Campaign("flux_free_invalid", "FAIL_CLOSED_INVALID_CAMPAIGN", "NO_SCIENTIFIC_0_OF_8_EVIDENCE; SUPERSEDED_BY_MODEL9_FOUNDATION_AND_VALID_RUN", (
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_152950/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_152950/preregistration/frozen_protocol.md", "frozen_protocol.md"),
    )),
    Campaign("model9_foundation", "ENGINEERING_SOLVER_AUTHORITY", "CURRENT_FOR_MODEL9_IMPLEMENTATION_FOUNDATION", (
        ("outputs/runs/thayer_model_9_preparation_v0_20260715_172217/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_model_9_preparation_v0_20260715_172217/preregistration/frozen_protocol.json", "frozen_protocol.json"),
        ("outputs/runs/thayer_model_9_preparation_v0_20260715_172217/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_model_9_preparation_v0_20260715_172217/engineering_validation/synthetic_validation_r2.json", "summary_table.json"),
    )),
    Campaign("flux_free_identifiability", "AUTHORITATIVE_NONORACLE_BASELINE", "CURRENT_FOR_SINGLE_OBSERVATION_FLUX_FREE_CONTRACT", (
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/preregistration/frozen_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/tables/scene_family_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/tables/comparison_to_oracle_flux_v1.csv", "comparison_to_oracle.csv"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/tables/oracle_information_audit.csv", "oracle_information_audit.csv"),
        ("outputs/runs/thayer_flux_free_identifiability_v0_20260715_183310/figures/flux_free_recoverability_frontier.png", "key_figure.png"),
    )),
    Campaign("psf_diverse_identifiability", "AUTHORITATIVE_PAIRED_OBSERVATION_RESULT", "CURRENT_FOR_PSF_DIVERSITY_INTERVENTION", (
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/preregistration/frozen_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/tables/scene_family_condition_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/tables/condition_transition_matrix.csv", "condition_transition_matrix.csv"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/tables/psf_diversity_metrics.csv", "psf_diversity_metrics.csv"),
        ("outputs/runs/thayer_psf_diverse_flux_identifiability_v0_20260717_081646/figures/recoverability_frontier_s1_s2_p2.png", "key_figure.png"),
    )),
    Campaign("external_photometry_preflight", "PREFLIGHT_EVIDENCE", "SUPERSEDED_BY_500_EVALUATION_CORRECTION", (
        ("outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/preregistration/frozen_preflight_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/tables/preflight_condition_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_external_photometry_preflight_v0_20260718_154852/figures/information_source_comparison.png", "key_figure.png"),
    )),
    Campaign("external_photometry_correction", "AUTHORITATIVE_TWO_SCENE_CORRECTION", "SUPERSEDED_IN_POPULATION_COUNT_BY_EIGHT_SCENE_CORRECTION", (
        ("outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638/preregistration/frozen_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_external_photometry_convergence_correction_v0_20260718_205638/tables/convergence_correction_summary.csv", "summary_table.csv"),
    )),
    Campaign("scene_stratification", "EXPLORATORY_SIX_SCENE_AUTHORITY", "SUPERSEDED_FOR_COMPLETE_RATE_BY_CONVERGENCE_CORRECTION", (
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/preregistration/frozen_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/tables/summary_table.csv", "summary_table.csv"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/tables/external_photometry_measurements.csv", "external_photometry_measurements.csv"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/analysis/run_campaign.py", "run_campaign.py"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/analysis/finalize_campaign.py", "finalize_campaign.py"),
        ("outputs/runs/thayer_external_photometry_scene_stratification_v0_20260719_011606/figures/scene_response_ranking.png", "key_figure.png"),
    )),
    Campaign("scene_stratification_correction", "AUTHORITATIVE_COMPLETE_EIGHT_SCENE_EXPLORATORY_RESULT", "CURRENT_FOR_PHASE_I_PHOTOMETRY_RESPONSE", (
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/reports/final_report.md", "final_report.md"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/preregistration/frozen_protocol.md", "frozen_protocol.md"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/manifests/final_manifest.json", "final_manifest.json"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/tables/corrected_eight_scene_summary.csv", "summary_table.csv"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/tables/corrected_feature_ranking.csv", "corrected_feature_ranking.csv"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/tables/corrected_decision_rule.csv", "corrected_decision_rule.csv"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/tables/comparison_to_p2.csv", "comparison_to_p2.csv"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/analysis/run_campaign.py", "run_campaign.py"),
        ("outputs/runs/thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954/figures/corrected_information_source_comparison.png", "key_figure.png"),
    )),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_copy(source: Path, target: Path) -> str:
    """Copy one artifact, tokenizing machine paths in compact text."""
    if source.suffix.lower() not in TEXT_SUFFIXES:
        shutil.copy2(source, target)
        return "byte_exact"
    original = source.read_bytes()
    original.decode("utf-8", errors="strict")
    transformed = original.replace(str(ROOT).encode(), b"<REPOSITORY_ROOT>")
    transformed = transformed.replace(HOME_PREFIX.encode(), b"<USER_HOME>")
    if transformed == original:
        shutil.copy2(source, target)
        if target.suffix.lower() == ".json":
            json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
        return "byte_exact"
    target.write_bytes(transformed)
    if target.suffix.lower() == ".json":
        json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    return "machine_paths_tokenized"


def reject_nonfinite(value: str) -> None:
    """Reject JavaScript-style NaN/Infinity tokens in purported JSON."""
    raise ValueError(f"non-finite token is not strict JSON: {value}")


def main() -> None:
    # This directory is fully generated by this script. Rebuild it from the
    # explicit allowlist so removed mappings cannot survive as stale evidence.
    if ARCHIVE_ROOT.exists():
        if ARCHIVE_ROOT.name != "experiment_archive" or ARCHIVE_ROOT.parent != ROOT / "docs":
            raise RuntimeError(f"refusing to replace unexpected archive path: {ARCHIVE_ROOT}")
        shutil.rmtree(ARCHIVE_ROOT)
    ARCHIVE_ROOT.mkdir(parents=True)
    index_rows: list[tuple[str, str, str]] = []
    for campaign in CAMPAIGNS:
        destination = ARCHIVE_ROOT / campaign.name
        destination.mkdir(parents=True, exist_ok=True)
        provenance_rows: list[tuple[str, str, int, int, str, str, str]] = []
        for source_name, archive_name in campaign.files:
            source = ROOT / source_name
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.suffix.lower() in BANNED_SUFFIXES:
                raise ValueError(f"banned archive suffix: {source}")
            if source.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError(f"archive suffix is not allowlisted: {source}")
            size = source.stat().st_size
            if size > MAX_BYTES:
                raise ValueError(f"archive source exceeds 5 MB: {source} ({size})")
            target = destination / archive_name
            transformation = archive_copy(source, target)
            provenance_rows.append(
                (
                    source_name,
                    target.relative_to(ROOT).as_posix(),
                    size,
                    target.stat().st_size,
                    digest(source),
                    digest(target),
                    transformation,
                )
            )
        lines = [
            "# Source provenance",
            "",
            f"- Authority status: `{campaign.authority}`",
            f"- Supersession status: `{campaign.supersession}`",
            "- Bulky originals remain local-only: yes",
            "- Copy policy: allowlisted compact artifacts; machine paths are tokenized where present",
            "",
            "| Original path | Original SHA-256 | Original bytes | Archive path | Archive SHA-256 | Archive bytes | Transformation |",
            "| --- | --- | ---: | --- | --- | ---: | --- |",
        ]
        if campaign.note:
            lines.insert(6, f"- Note: {campaign.note}")
        lines.extend(
            f"| `{src}` | `{original_sha}` | {original_size} | `{dst}` | `{archive_sha}` | {archive_size} | `{transformation}` |"
            for src, dst, original_size, archive_size, original_sha, archive_sha, transformation in provenance_rows
        )
        lines.append("")
        (destination / "SOURCE_PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")
        index_rows.append((campaign.name, campaign.authority, campaign.supersession))
    readme = [
        "# Curated experiment archive",
        "",
        "This archive contains compact copies of selected reports, protocols, manifests, tables, and figures. Copies are byte-exact unless their provenance row says machine-specific paths were tokenized. Raw data, HDF5, NPY/NPZ, checkpoints, dense endpoint histories, caches, and logs remain local-only.",
        "",
        "| Campaign | Authority | Supersession |",
        "| --- | --- | --- |",
    ]
    readme.extend(f"| [{name}]({name}/) | `{authority}` | `{supersession}` |" for name, authority, supersession in index_rows)
    readme.append("")
    (ARCHIVE_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
