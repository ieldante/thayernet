#!/usr/bin/env python3
"""Build metadata-only inventories for the canonical research archive.

This utility never imports project models, opens scientific array contents, or
executes a campaign. It reads directory metadata and compact text authorities;
large files are streamed only to calculate SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs" / "research_archive"
RUNS = ROOT / "outputs" / "runs"
LARGE_THRESHOLD = 5_000_000
FIFTY_MB = 50_000_000
HUNDRED_MB = 100_000_000
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv-btk",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
}


# Claim-bearing metadata that cannot be recovered reliably from directory
# names is recorded explicitly. Missing values remain NOT_RECORDED rather than
# being inferred retrospectively. Repeated launch-only attempts without a
# distinct scientific result retain the conservative generated defaults.
RUN_OVERRIDES: dict[str, dict[str, str]] = {
    "thayer_identifiability_v1_20260715_003220": {
        "campaign_type": "scientific",
        "predecessor": "Unrestricted null-space and Family-E1/E1P diagnosis; an exact single run predecessor was not recorded.",
        "hypothesis": "Does realistic prompt-centered morphology eliminate the unrestricted allocation null space under progressively stronger priors?",
        "exact_authorized_change": "Added the training-free L0-L7 prior ladder: nonnegativity, exact per-source flux, TV, Sersic, bulge+disk, shared color, and weak morphology density; no neural change.",
        "data_accessed": "Exact A/B source images, signed noise, coordinates, and 16 catalog rows for 8 authorized training scenes.",
        "protected_data_status": "Intentional oracle truth access; zero development, Atlas, or lockbox access.",
        "model_or_solver": "Analytic [I I] system for L0-L3; GalSim Sersic/bulge+disk bounded least squares for parametric levels.",
        "scene_or_sample_count": "8 scenes.",
        "number_of_starts": "16 deterministic starts per scene/family.",
        "compute_budget": "Maximum 400 evaluations per start.",
        "metrics": "Rank/nullity, exact-solution count, condition, scientific diameter, support, prompt identity, and residual.",
        "final_outcome": "PASS; strict UNIQUE 7/8 under the oracle-photometry contract. L4 resolved 4 scenes, L5 added 3; Scene 51 was out of structural support.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Oracle per-source flux and truth-derived noise, 8 selected training scenes, and one out-of-support case.",
        "downstream_consequence": "Established a conditional upper bound and motivated a nonoracle flux-free audit.",
        "authority_status": "HISTORICAL_CONDITIONAL_ORACLE_AUTHORITY",
        "supersession_status": "RETAINED_AS_ORACLE_UPPER_BOUND; NONORACLE_INTERPRETATION_QUALIFIED_BY_FLUX_FREE_RESULT",
    },
    "thayer_flux_free_identifiability_v0_20260715_152950": {
        "campaign_type": "engineering_invalid",
        "predecessor": "thayer_identifiability_v1_20260715_003220",
        "hypothesis": "Would the 7/8 oracle-contract uniqueness result survive when individual source fluxes are free?",
        "exact_authorized_change": "Intended removal of oracle per-source flux constraints; no scientific execution was authorized because Model-9 preparation was absent.",
        "data_accessed": "Historical hashes and readiness metadata only; no scientific arrays or observations.",
        "protected_data_status": "Zero scientific, catalog-truth, development, Atlas, or lockbox access.",
        "model_or_solver": "None instantiated.",
        "scene_or_sample_count": "8 intended; 0 executed.",
        "number_of_starts": "0 scientific starts.",
        "compute_budget": "0 scientific evaluations.",
        "metrics": "Authorization, readiness, and integrity checks only.",
        "final_outcome": "FLUX_FREE_INVALID; uniqueness counts are not estimable and this run supplies no 0/8 evidence.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Missing Model-9 foundation, authorization token, frozen protocol, and tests.",
        "downstream_consequence": "Mandated the Model-9 preparation campaign.",
        "authority_status": "FAIL_CLOSED_ENGINEERING_HISTORY_NO_SCIENTIFIC_AUTHORITY",
        "supersession_status": "SCIENTIFIC_QUESTION_SUPERSEDED_BY_VALID_20260715_183310_RUN",
    },
    "thayer_model_9_preparation_v0_20260715_172217": {
        "campaign_type": "engineering",
        "predecessor": "thayer_flux_free_identifiability_v0_20260715_152950",
        "hypothesis": "Can an oracle-resistant differentiable structured solver and frozen protocol be validated using only synthetic fixtures?",
        "exact_authorized_change": "Implemented differentiable L4/L5 rendering, free fluxes, signed residuals, optimizer, fixtures, tests, validator, and frozen flux-free protocol.",
        "data_accessed": "Synthetic fixtures and historical checkpoint hashes only.",
        "protected_data_status": "Zero scientific-observation, isolated-array, development, Atlas, or lockbox access.",
        "model_or_solver": "Model-9 float64 PyTorch renderer/autograd Jacobian with SciPy bounded TRF.",
        "scene_or_sample_count": "0 scientific scenes; synthetic fixture count NOT_RECORDED.",
        "number_of_starts": "5 synthetic optimizer starts; future protocol froze 16.",
        "compute_budget": "Synthetic validation budget NOT_RECORDED; future maximum 500 evaluations.",
        "metrics": "Renderer agreement, PSF normalization, closure, finite-difference Jacobian, synthetic rank/nullity, replay, and foundation tests.",
        "final_outcome": "MODEL_9_FOUNDATION_READY.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Engineering fixtures only; no scientific-scene result.",
        "downstream_consequence": "Authorized the valid flux-free campaign.",
        "authority_status": "ENGINEERING_SOLVER_AUTHORITY",
        "supersession_status": "CURRENT_FOR_MODEL9_FOUNDATION",
    },
    "thayer_flux_free_identifiability_v0_20260715_183310": {
        "campaign_type": "scientific",
        "predecessor": "thayer_model_9_preparation_v0_20260715_172217; oracle comparator thayer_identifiability_v1_20260715_003220",
        "hypothesis": "Are L4/L5 morphologies sufficient to identify both sources from one blend when per-source fluxes are free?",
        "exact_authorized_change": "Removed true per-source fluxes and truth-derived noise; fitted nonnegative source fluxes to the noisy blend with an observation-derived noise model.",
        "data_accessed": "8 frozen g/r/z blends, coordinates, known PSFs, geometry, and plug-in noise estimates; exactly 16 authorized blend/coordinate-row reads.",
        "protected_data_status": "No isolated images, true flux/noise/morphology, catalog truth, development, Atlas, or lockbox data.",
        "model_or_solver": "Model-9 L4/L5 bounded TRF with PyTorch rendering and Jacobian.",
        "scene_or_sample_count": "8 scenes x 2 families.",
        "number_of_starts": "16 starts per fit; 256 endpoints.",
        "compute_budget": "Maximum 500 evaluations; 222 successful and 34 failed/capped endpoints.",
        "metrics": "Local rank/nullity, condition, Hessian/gradient, endpoint classes, flux/morphology diameters, support, identity, boundary, and replay.",
        "final_outcome": "FLUX_FREE_UNIQUENESS_COLLAPSES; L4, L5, and union strict uniqueness all 0/8; best-family outcomes 6 NEAR_UNIQUE and 2 PARTIALLY_IDENTIFIABLE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "8 simulated scenes, finite multistart, near-unique is not a proof of ambiguity, and reconstruction accuracy was not established.",
        "downstream_consequence": "Motivated the controlled S2/P2 acquisition audit; PriorNet/POST remained unauthorized.",
        "authority_status": "AUTHORITATIVE_NONORACLE_S1_BASELINE",
        "supersession_status": "SUPERSEDES_INVALID_RUN_SCIENCE; QUALIFIES_BUT_DOES_NOT_ERASE_ORACLE_RESULT",
    },
    "thayer_psf_diverse_flux_identifiability_v0_20260717_081646": {
        "campaign_type": "scientific",
        "predecessor": "thayer_flux_free_identifiability_v0_20260715_183310",
        "hypothesis": "Does a second known PSF-diverse observation remove the flux-free ambiguity?",
        "exact_authorized_change": "Added one shared-latent second observation and compared same-PSF S2 with PSF-diverse P2 under fixed fitting/classification rules.",
        "data_accessed": "Original observation plus generated second blends from the same CatSim rows, exact coordinates, known normalized PSFs, and plug-in noise.",
        "protected_data_status": "Catalog used only inside simulation; isolated layers discarded; zero development, Atlas, or lockbox access.",
        "model_or_solver": "Joint two-observation Model-9 likelihood with stacked Jacobian and bounded TRF.",
        "scene_or_sample_count": "8 scenes x 2 families x 2 new conditions = 32 fits.",
        "number_of_starts": "16 starts per fit; 512 endpoints.",
        "compute_budget": "Maximum 500 evaluations; 64 capped failures.",
        "metrics": "S1/S2/P2 classes, singular values, condition, rank/nullity/Hessian, diameters, attribution, PSF distances, and replay.",
        "final_outcome": "PSF_DIVERSE_CONDITIONING_IMPROVES_WITHOUT_UNIQUENESS; strict unique 0/8; composite P2-vs-S1 gain 15/16, only 3/16 PSF-specific after S2.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "8 simulated scenes, 64 capped fits, and most composite gains were not uniquely attributable to PSF diversity.",
        "downstream_consequence": "Motivated the external-photometry comparison.",
        "authority_status": "AUTHORITATIVE_PAIRED_OBSERVATION_RESULT",
        "supersession_status": "CURRENT_FOR_P2; S1_REMAINS_COMPARATOR",
    },
    "thayer_external_photometry_preflight_v0_20260718_154852": {
        "campaign_type": "preflight",
        "predecessor": "thayer_psf_diverse_flux_identifiability_v0_20260717_081646",
        "hypothesis": "Does 5%-precision external total or per-band source photometry reduce multiplicity enough to justify a full campaign?",
        "exact_authorized_change": "Added Gaussian total-flux or per-band source-photometry residuals to unchanged single-observation Level-5 fits on two scenes.",
        "data_accessed": "Scenes 0/6 blends, coordinates, PSF/noise, and catalog photometry used only to generate noisy external measurements.",
        "protected_data_status": "No isolated-image inference; zero development, Atlas, or lockbox access.",
        "model_or_solver": "Model-9 Level-5 single-observation photometry likelihood with bounded TRF.",
        "scene_or_sample_count": "2 scenes x 2 photometry conditions = 4 fits.",
        "number_of_starts": "4 starts per fit; 16 endpoints.",
        "compute_budget": "Maximum 150 evaluations; 5 capped endpoints.",
        "metrics": "Classes/diameters versus S1/P2, rank/nullity/condition, gradients/objectives, optimizer success, replay, and promise category.",
        "final_outcome": "PREFLIGHT_OPTIMIZATION_LIMITED; Scene 0 showed moderate total-photometry promise and Scene 6 no gain.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "2 scenes, 4 starts, reduced budget, and Scene 0 per-band had 0/4 successful starts.",
        "downstream_consequence": "Required a 500-evaluation correction before expansion.",
        "authority_status": "PROVISIONAL_PREFLIGHT_EVIDENCE",
        "supersession_status": "FIT_INTERPRETATIONS_SUPERSEDED_BY_20260718_205638; MEASUREMENTS_RETAINED",
    },
    "thayer_external_photometry_convergence_correction_v0_20260718_205638": {
        "campaign_type": "correction",
        "predecessor": "thayer_external_photometry_preflight_v0_20260718_154852",
        "hypothesis": "Do the two-scene preflight conclusions survive an authoritative optimization budget?",
        "exact_authorized_change": "Changed only max_nfev from 150 to 500; reused exact fits, starts, and byte-identical measurements.",
        "data_accessed": "Exact preflight measurements and the same Scenes 0/6 blends and coordinates.",
        "protected_data_status": "No measurement regeneration or added truth access; zero protected-set access.",
        "model_or_solver": "Unchanged Model-9 Level-5 photometry likelihood and bounded TRF.",
        "scene_or_sample_count": "2 scenes x 2 conditions = 4 fits.",
        "number_of_starts": "4 starts per fit; 16 endpoints.",
        "compute_budget": "Maximum 500 evaluations; 2 capped endpoints.",
        "metrics": "Corrected classes, objectives/gradients, rank/nullity/condition, preflight/S1/P2 comparison, and replay.",
        "final_outcome": "EXTERNAL_PHOTOMETRY_TARGETED_CAMPAIGN_JUSTIFIED; Scene 0 benefited under both forms, Scene 6 did not; total and per-band were materially similar.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Only 2 scenes; Scene 0 per-band retained 2/4 capped starts.",
        "downstream_consequence": "Authorized targeted scene stratification.",
        "authority_status": "AUTHORITATIVE_TWO_SCENE_CORRECTION",
        "supersession_status": "SUPERSEDES_PREFLIGHT_OPTIMIZATION_INTERPRETATION_ONLY",
    },
    "thayer_external_photometry_scene_stratification_v0_20260719_011606": {
        "campaign_type": "scientific",
        "predecessor": "thayer_external_photometry_convergence_correction_v0_20260718_205638",
        "hypothesis": "Which prespecified scene properties explain why external photometry helps some scenes but not others?",
        "exact_authorized_change": "Extended total/per-band photometry to the remaining 6 scenes, reused Scenes 0/6, and computed 15 post-fit exploratory descriptors/statistics.",
        "data_accessed": "All 8 training scenes; six new blends/coordinates/catalog-generated measurements; isolated arrays and catalog morphology only after fitting for descriptors.",
        "protected_data_status": "Truth-derived descriptors restricted to exploratory evaluation; zero development, Atlas, or lockbox access.",
        "model_or_solver": "Model-9 Level-5 photometry solver plus post-fit midpoint-Gini depth-2 exploratory analysis.",
        "scene_or_sample_count": "6 new scenes x 2 conditions = 12 fits; 48 new plus 16 reused endpoints.",
        "number_of_starts": "4 starts per new fit.",
        "compute_budget": "Maximum 500 evaluations; total-photometry Scenes 5/18 unresolved.",
        "metrics": "Helpful labels versus P2, classes/condition/diameters, 15-feature AUC, permutation p, BH q, Spearman, tree, LOO, and CI.",
        "final_outcome": "SCENE_STRATIFICATION_PRIMARY_RESPONSE_PARTIALLY_RESOLVED; 6/8 interpretable and helpful in 3/6; Scenes 5/18 unresolved.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Only 6 interpretable labels, 4-start photometry versus 16-start P2, and truth-derived descriptors.",
        "downstream_consequence": "Required a 2,000-evaluation correction restricted to Scenes 5/18.",
        "authority_status": "AUTHORITATIVE_FOR_SIX_RESOLVED_LABELS; FEATURE_ANALYSIS_EXPLORATORY",
        "supersession_status": "SCENES_5_18_AND_AGGREGATE_SUPERSEDED_BY_20260719_030954",
    },
    "thayer_external_photometry_stratification_convergence_correction_v0_20260719_030954": {
        "campaign_type": "correction",
        "predecessor": "thayer_external_photometry_scene_stratification_v0_20260719_011606",
        "hypothesis": "Can increased budget alone resolve Scenes 5/18 and complete the exploratory response map?",
        "exact_authorized_change": "Changed only maximum evaluations from 500 to 2,000 for total-photometry Scenes 5/18; reused exact starts and measurements.",
        "data_accessed": "Exact predecessor measurements, blends, and coordinates for Scenes 5/18.",
        "protected_data_status": "No regeneration, isolated-source inference, resolved-scene rerun, development, Atlas, or lockbox access.",
        "model_or_solver": "Unchanged Model-9 Level-5 total-photometry likelihood with bounded TRF.",
        "scene_or_sample_count": "2 corrected scenes.",
        "number_of_starts": "4 starts per scene; 8 endpoints.",
        "compute_budget": "Maximum 2,000 evaluations; Scene 5 success 1/4, Scene 18 success 4/4.",
        "metrics": "Corrected classes/objectives/gradients/rank/nullity/condition, replay, perturbation, helpful rate/CI, AUC, permutation/BH q, tree, and LOO.",
        "final_outcome": "STRATIFICATION_DATASET_COMPLETE; Scene 5 helpful/near, Scene 18 not helpful/non; corrected 4/8, exact CI 15.7%-84.3%; exploratory AUC 1.0, p .0286, q .4286, LOO .75.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Scene 5 retained 3/4 capped starts; n=8 discovery set; truth-derived descriptors; unequal starts versus P2; no independent validation.",
        "downstream_consequence": "Motivated and froze independent-scene Phase-II validation.",
        "authority_status": "AUTHORITATIVE_CORRECTED_PHASE_I_AGGREGATE; FEATURE_INFERENCE_EXPLORATORY",
        "supersession_status": "SUPERSEDES_ONLY_SCENES_5_18_UNRESOLVED_LABELS_AND_INCOMPLETE_AGGREGATE",
    },
}


RUN_OVERRIDES.update({
    "research_correctness_audit_20260710_092241": {
        "campaign_type": "correction",
        "predecessor": "Historical BR v0.2 Moderate under the random-row split.",
        "hypothesis": "A source-group correction will retain a meaningful learned advantage over identity.",
        "exact_authorized_change": "Grouped sources by exact-pixel-hash/coordinate connected components, created grouped manifests, and retrained the unchanged v0.2 Moderate model on 8,000 rather than 12,000 blends.",
        "data_accessed": "17,736 Galaxy10 sources; grouped split 12,417/2,660/2,659; 8,000 train, 1,000 validation, and four 1,000-scene development suites.",
        "protected_data_status": "No untouched final pool remained; the provisional pool was superseded after 590/1,000 sources entered grouped train/validation.",
        "model_or_solver": "1,927,075-parameter residual U-Net on MPS.",
        "scene_or_sample_count": "8,000 train, 1,000 validation, 4 x 1,000 evaluation scenes.",
        "number_of_starts": "One training seed (3042).",
        "compute_budget": "20 epochs, batch 8.",
        "metrics": "Normal/hard/compact/high-core affected-MSE ratios 28.8127/15.8025/9.18304/15.8378; 0/3/2/1 scenes worse than identity.",
        "final_outcome": "Preferred duplicate-safe development reference; a final-test blocker remains.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "RGB display cutouts, simplified generator, one seed, smaller training budget, and no independent final set.",
        "downstream_consequence": "Downgraded historical 32.3x/19.6x values to context and made grouped development values authoritative.",
        "authority_status": "HISTORICAL_GROUPED_DEVELOPMENT_AUTHORITY",
        "supersession_status": "SUPERSEDES_RANDOM_ROW_DEVELOPMENT_CLAIMS_AND_PROVISIONAL_FINAL_POOL",
    },
    "thayer_select_btk_foundation_20260711_152613": {
        "campaign_type": "engineering",
        "predecessor": "Blocked DR10 source-extraction route.",
        "hypothesis": "BTK/GalSim can provide replayable, single-noise-realization, source-resolved g/r/z scenes.",
        "exact_authorized_change": "Installed isolated BTK 1.0.9 environment and rendered CatSim singles/doubles with exact source addition and a frozen split.",
        "data_accessed": "Official 86,273-row CatSim catalog.",
        "protected_data_status": "Source split 60,339/8,630/6,901/6,033/4,310; future lockbox not used.",
        "model_or_solver": "BTK 1.0.9, GalSim 2.8.4, surveycodex 1.2.0; engineering U-Net smoke only.",
        "scene_or_sample_count": "20 singles, 20 doubles; smoke 500 train/100 validation.",
        "number_of_starts": "One smoke fit.",
        "compute_budget": "3 smoke epochs.",
        "metrics": "200/200 checks, 20/20 exact additivity, 40/40 fresh replay.",
        "final_outcome": "Controlled CatSim foundation PASS; COSMOS route blocked.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Engineering-scale smoke, fixed simulator contract, and no real-sky claim.",
        "downstream_consequence": "Authorized the coordinate-prompt ablation.",
        "authority_status": "ENGINEERING_FOUNDATION_AUTHORITY",
        "supersession_status": "CURRENT_CONTROLLED_SIMULATION_FOUNDATION",
    },
    "thayer_select_prompt_ablation_20260711_164329": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_btk_foundation_20260711_152613",
        "hypothesis": "A coordinate prompt can select an arbitrary requested component after removing the centered-target shortcut.",
        "exact_authorized_change": "Compared centered/no prompt, randomized/no prompt, and randomized plus Gaussian coordinate prompt; Condition C added 144 first-layer weights.",
        "data_accessed": "8,000 train, 1,000 validation, 1,000 calibration, 1,000 development scenes.",
        "protected_data_status": "Development evaluated after freeze; lockbox scene access 0.",
        "model_or_solver": "Compact U-Net; conditions A/B 118,947 parameters, C 119,091.",
        "scene_or_sample_count": "11,000 frozen scenes; primary randomized comparison on 1,000.",
        "number_of_starts": "Three model fits, one per condition.",
        "compute_budget": "20 epochs each, batch 8, MPS.",
        "metrics": "B/C source MSE 2.02931e6/1.02004e6; paired delta -1.00927e6, CI [-1.97662e6,-292977]; prompt swap .98; collapse .002.",
        "final_outcome": "SUCCESS for promptability; source recovery not established.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Heavy-tailed source error, only 499/1,000 source-region wins, and empty-prompt hallucination 1.0 under the declared rule.",
        "downstream_consequence": "Motivated recoverability prediction rather than a source-recovery claim.",
        "authority_status": "AUTHORITATIVE_CONDITION_C_PROMPTABILITY_RESULT",
        "supersession_status": "CURRENT_FOR_CONDITION_C_PROMPTABILITY",
    },
    "thayer_select_recoverability_20260711_191518": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_prompt_ablation_20260711_164329",
        "hypothesis": "A bounded uncertainty/recoverability head can rank unsafe reconstructions and reduce risk at retained coverage.",
        "exact_authorized_change": "Trained R0 reconstruction control and R1 recoverability/uncertainty outputs with calibration-only isotonic selection.",
        "data_accessed": "10,000 train, 1,500 validation, 2,000 calibration, and one-time 2,000 development scenes.",
        "protected_data_status": "Development evaluated once after freeze; lockbox access 0.",
        "model_or_solver": "R0 119,091 parameters; R1 123,368.",
        "scene_or_sample_count": "15,500 including development.",
        "number_of_starts": "One R0 and one R1 primary fit.",
        "compute_budget": "20 epochs each, batch 8.",
        "metrics": "AUROC .8746, AUPRC .2475, Brier .1010 to .0456, risk-coverage area .4697, normalized RMSE C/R1 1.6557/.9887.",
        "final_outcome": "PARTIAL SUCCESS; ambiguity gate failed.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Controlled development only, sparse/heterogeneous labels, and initially unknown seed stability.",
        "downstream_consequence": "Authorized a fixed-protocol seed replication.",
        "authority_status": "HISTORICAL_PARTIAL_RECOVERABILITY_RESULT",
        "supersession_status": "191127_IS_SCHEMA_INCIDENT; STABILITY_INTERPRETATION_SUPERSEDED_BY_SEED_REPLICATION",
    },
    "thayer_select_recoverability_seed_replication_20260711_203115": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_recoverability_20260711_191518",
        "hypothesis": "R1 ranking and calibration persist across frozen training seeds.",
        "exact_authorized_change": "Trained exactly two additional seeds with unchanged manifests, objectives, calibration, and gates.",
        "data_accessed": "Same frozen train/validation/calibration/development partitions as R1.",
        "protected_data_status": "Frozen development evaluated under the replication protocol; lockbox access 0.",
        "model_or_solver": "Unchanged R1 recoverability model.",
        "scene_or_sample_count": "Same 15,500-scene contract as predecessor.",
        "number_of_starts": "Two new starts; three-seed comparison total.",
        "compute_budget": "20 epochs per new seed.",
        "metrics": "AUROC .8746-.9076, AUPRC .1388-.3026, Brier .0154-.0456; every 80%-coverage threshold was 0 and accepted all 2,000 scenes.",
        "final_outcome": "UNSTABLE RESULT.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Tied calibration created degenerate agreement and no material catastrophic rejection.",
        "downstream_consequence": "Blocked R1 policy and lockbox advancement.",
        "authority_status": "AUTHORITATIVE_SEED_STABILITY_CORRECTION",
        "supersession_status": "SUPERSEDES_SINGLE_SEED_OPERATIONAL_INTERPRETATION",
    },
    "thayer_select_frozen_head_ablation_20260711_220756": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_recoverability_seed_replication_20260711_203115",
        "hypothesis": "Stable recoverability information may already be accessible in the frozen latent representation.",
        "exact_authorized_change": "Extracted 64 pooled bottleneck features and compared H0/H1/H2/H4 frozen heads.",
        "data_accessed": "10,000 train, 1,500 validation, 2,000 calibration feature rows.",
        "protected_data_status": "Development/lockbox access 0/0.",
        "model_or_solver": "Logistic and MLP heads over frozen R1 features.",
        "scene_or_sample_count": "13,500.",
        "number_of_starts": "Four candidate head families; stochastic-start count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Positive labels 41/10,000, 5/1,500, 30/2,000; H0 AUROC/AUPRC .985/.265; H2 .984/.549; ambiguity remained inverted.",
        "final_outcome": "NO CLEAR IMPROVEMENT.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Extreme label sparsity and validation-to-calibration instability.",
        "downstream_consequence": "Motivated label hierarchy rather than a backbone change.",
        "authority_status": "VALID_NEGATIVE_FROZEN_HEAD_RESULT",
        "supersession_status": "CURRENT_FOR_FROZEN_HEAD_DIAGNOSTIC",
    },
    "thayer_select_hierarchical_safety_20260711_225657": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_frozen_head_ablation_20260711_220756",
        "hypothesis": "Decomposing query validity, continuous risks, and catastrophe can yield a useful selective policy.",
        "exact_authorized_change": "Separated three-state query, risk, catastrophe, conformal-interval, and hierarchical-acceptance components.",
        "data_accessed": "Historical train/validation/calibration rows plus fresh one-time development.",
        "protected_data_status": "Development evaluated once after policy freeze; lockbox access 0.",
        "model_or_solver": "Frozen Condition C plus multiple query/risk heads.",
        "scene_or_sample_count": "40,500 historical semantic checks; 2,000 UNIQUE_VALID development scenes plus invalid-query cohorts.",
        "number_of_starts": "Five query-head seeds; other candidate count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Query macro-F1/AUPRC .881/.923, recalls .757/.998/.889, and 1/2,000 accepted valid development scenes.",
        "final_outcome": "FAILURE under frozen complete-policy gates.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Sparse heterogeneous targets, calibration-provenance mismatch, and operationally zero coverage.",
        "downstream_consequence": "Required a provenance/protocol corrective audit.",
        "authority_status": "VALID_NEGATIVE_HIERARCHICAL_POLICY_RESULT",
        "supersession_status": "SCIENTIFIC_FAILURE_RETAINED; CERTIFICATION_QUALIFIED_BY_20260712_001405",
    },
    "thayer_select_hierarchical_safety_20260712_001405": {
        "campaign_type": "correction",
        "predecessor": "thayer_select_hierarchical_safety_20260711_225657",
        "hypothesis": "Did the historical hierarchy comply with the later-required frozen sequence and coherent labels?",
        "exact_authorized_change": "Provenance and Boolean-label reconstruction only; no new fit.",
        "data_accessed": "13,500 persisted train/validation/calibration rows.",
        "protected_data_status": "No new development or lockbox inference.",
        "model_or_solver": "Audit only.",
        "scene_or_sample_count": "13,500.",
        "number_of_starts": "0.",
        "compute_budget": "0 optimizer steps.",
        "metrics": "Exact reconstruction of the original moderate composite with zero Boolean mismatches.",
        "final_outcome": "Stopped before training; the historical campaign is not protocol-certifiable.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Assesses compliance, not predictive performance.",
        "downstream_consequence": "Required a prospective uniform-provenance feasibility study.",
        "authority_status": "SUPERSEDING_PROTOCOL_COMPLIANCE_CORRECTION",
        "supersession_status": "OVERRIDES_PROTOCOL_COMPLIANCE_CLAIMS_ONLY; HISTORICAL_METRICS_RETAINED",
    },
})


RUN_OVERRIDES.update({
    "dr10_foundation_20260711_024415": {
        "campaign_type": "engineering",
        "predecessor": "Grouped RGB audit and proposal for realistic source-flux injection.",
        "hypothesis": "DR10 coadds and model/residual products can supply isolated, single-noise-realization, PSF-compatible contaminants.",
        "exact_authorized_change": "Downloaded/audited 2,500 southern DR10 cutouts and froze source-quality/lockbox rules.",
        "data_accessed": "8,689,370-row catalog and 2,494 valid pilot FITS.",
        "protected_data_status": "A 100-source future lockbox was allocated and untouched; no blend manifests created.",
        "model_or_solver": "FITS/WCS/source-quality audit; no neural fit.",
        "scene_or_sample_count": "2,500 candidate outcomes.",
        "number_of_starts": "NOT_APPLICABLE",
        "compute_budget": "Download/audit only.",
        "metrics": "2,022 accepted, 381 review, 97 rejected.",
        "final_outcome": "Foundation data acquired, but isolated-source suitability remained unresolved pending the model probe.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Real coadds contain correlated noise and uncertain source isolation/PSFs.",
        "downstream_consequence": "Required the DR10 model/residual probe.",
        "authority_status": "ENGINEERING_REAL_SKY_FOUNDATION",
        "supersession_status": "BTK_REPLACES_CONTROLLED_TRAINING_ROUTE; DR10_REMAINS_FUTURE_OOD_ROUTE",
    },
    "dr10_model_probe_20260711_160018": {
        "campaign_type": "preflight",
        "predecessor": "dr10_foundation_20260711_024415",
        "hypothesis": "Observed/model/residual triplets can yield acceptable isolated contaminants.",
        "exact_authorized_change": "Inspected 20 triplets under three source-extraction options.",
        "data_accessed": "20 observed/model/residual triplets; 60 extraction-option evaluations.",
        "protected_data_status": "Future 100-source lockbox untouched; no blend manifests.",
        "model_or_solver": "SEP plus alignment, closure, central-isolation, and PSF audits.",
        "scene_or_sample_count": "20 triplets; 60 options.",
        "number_of_starts": "NOT_APPLICABLE",
        "compute_budget": "Audit only.",
        "metrics": "Alignment 20/20, closure 15/20, central-only isolation 0/20, acceptable options 0/60.",
        "final_outcome": "REAL-CUTOUT BLENDING BLOCKED.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Summed scene models, correlated coadd noise, inseparable wings, and unverified local PSFs.",
        "downstream_consequence": "Selected BTK/GalSim controlled rendering.",
        "authority_status": "VALID_NEGATIVE_REAL_CUTOUT_PREFLIGHT",
        "supersession_status": "160018_TABLES_SUPERSEDE_155820_SCALARS; BTK_GOVERNS_CONTROLLED_DATA",
    },
    "thayer_select_hierarchical_feasibility_20260712_010729": {
        "campaign_type": "preflight",
        "predecessor": "thayer_select_hierarchical_safety_20260712_001405",
        "hypothesis": "Uniform reconstruction provenance and applicability masks make hierarchy components learnable without policy construction.",
        "exact_authorized_change": "Prospective train/validation/calibration-only component study using one frozen Condition-C checkpoint.",
        "data_accessed": "32,000 rows.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Query ensemble and separate image/flux/centroid/confusion/catastrophe heads.",
        "scene_or_sample_count": "32,000.",
        "number_of_starts": "Five seeds.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Query macro-F1 .872+/- .010; image/flux/centroid rank correlations about .86/.86/.95; minimum subgroup coverage .691.",
        "final_outcome": "PARTIAL SUCCESS; no policy authorized.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Subgroup calibration sag and heavy-tail interval widths.",
        "downstream_consequence": "Required a focused conditional-calibration correction.",
        "authority_status": "VALID_PARTIAL_COMPONENT_FEASIBILITY_RESULT",
        "supersession_status": "ADDENDUM_CHANGES_CATASTROPHIC_COMPONENT_TO_FAIL_ONLY",
    },
    "thayer_select_conditional_calibration_20260712_021556": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_hierarchical_feasibility_20260712_010729",
        "hypothesis": "Larger risk heads and conditional conformal methods repair subgroup undercoverage.",
        "exact_authorized_change": "Compared increased-capacity heads and C0-C4 deployable conditional calibration methods.",
        "data_accessed": "Frozen train/validation/natural-calibration artifacts from the 32,000-row campaign.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0; no new reconstruction inference.",
        "model_or_solver": "Risk MLPs plus normalized/Mondrian conformal methods.",
        "scene_or_sample_count": "Inherited 32,000-row contract.",
        "number_of_starts": "Multiple candidates/seeds; exact consolidated count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Marginal image/flux/centroid coverage .902857/.898214/.900714; worst subgroup .637306/.683938/.888165.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Marginal calibration did not imply conditional coverage.",
        "downstream_consequence": "Motivated residual-scale correction.",
        "authority_status": "VALID_NEGATIVE_CONDITIONAL_CALIBRATION_RESULT",
        "supersession_status": "CURRENT_FOR_CONDITIONAL_CALIBRATION_BRANCH",
    },
    "thayer_select_scale_correction_20260712_024957": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_conditional_calibration_20260712_021556",
        "hypothesis": "OOF residual-scale modeling and partial pooling repair conditional coverage.",
        "exact_authorized_change": "Used five connected-source-group OOF folds, log-linear/MLP scale models, and pooled/soft-gated corrections.",
        "data_accessed": "Same frozen train/validation/calibration rows.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Scale regressors plus normalized conformal calibration.",
        "scene_or_sample_count": "Inherited campaign rows.",
        "number_of_starts": "Five OOF folds; candidate-start count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Image worst subgroup .637306 to .549223; flux .683938 to .678756; marginal .918929/.921786.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Interval inflation without conditional repair.",
        "downstream_consequence": "Motivated a shape-constrained scale study.",
        "authority_status": "VALID_NEGATIVE_SCALE_CORRECTION_RESULT",
        "supersession_status": "CURRENT_FOR_PARTIALLY_POOLED_SCALE_BRANCH",
    },
    "thayer_select_shape_constrained_quantile_20260712_033406": {
        "campaign_type": "scientific",
        "predecessor": "thayer_select_scale_correction_20260712_024957",
        "hypothesis": "Shape-constrained quantile scale with interactions repairs tail/subgroup behavior.",
        "exact_authorized_change": "Corrected post-optimization centering and compared Q1 with mixed-interaction Q2.",
        "data_accessed": "Same frozen calibration contract.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Shape-constrained quantile scale models.",
        "scene_or_sample_count": "Inherited campaign rows.",
        "number_of_starts": "NOT_RECORDED",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Image/flux worst coverage .544041/.590674; marginal .922143/.922143; Q2 validation gain 0.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Deployable features were insufficient for conditional tail correction.",
        "downstream_consequence": "Motivated observability distillation.",
        "authority_status": "VALID_NEGATIVE_SHAPE_CONSTRAINED_RESULT",
        "supersession_status": "032007_FORMULA_INCIDENT_AND_032938_PREOPT_CENTERING_SUPERSEDED",
    },
    "thayer_select_observability_distillation_20260712_035843": {
        "campaign_type": "scientific",
        "predecessor": "Repeated calibration corrections.",
        "hypothesis": "Deployable observables contain enough information to identify high-risk/undercovered cases.",
        "exact_authorized_change": "Distilled failure/scale signal into observable-only candidates A0-A3.",
        "data_accessed": "Frozen train/validation/calibration features; no new reconstruction.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Observable-only audit classifiers/regressors.",
        "scene_or_sample_count": "Inherited 32,000-row foundation.",
        "number_of_starts": "Candidate models; exact count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Selected A3 validation AUROC .9014+/- .0035, calibration .87997; recall at precision .70 only .08346; Brier .1397 vs prevalence .06418.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Strong ranking but inadequate high-precision recall/calibration.",
        "downstream_consequence": "Reframed missing signal as an information issue and triggered PSF-provenance audit.",
        "authority_status": "VALID_NEGATIVE_OBSERVABILITY_RESULT",
        "supersession_status": "CURRENT_FOR_OBSERVABLE_DISTILLATION_BRANCH",
    },
    "thayer_select_psf_conditioning_20260712_043442": {
        "campaign_type": "preflight",
        "predecessor": "thayer_select_observability_distillation_20260712_035843",
        "hypothesis": "Per-scene PSF variation may provide missing recoverability information.",
        "exact_authorized_change": "Audited PSF provenance/configuration diversity before fitting.",
        "data_accessed": "18,000 frozen scene/configuration records.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Provenance audit only.",
        "scene_or_sample_count": "18,000.",
        "number_of_starts": "0.",
        "compute_budget": "0 fits.",
        "metrics": "One combined PSF configuration; g/r/z FWHM .86/.81/.77 arcsec.",
        "final_outcome": "PSF NON-INFORMATIVE BY CONSTRUCTION.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Could not evaluate benefits of actual PSF diversity.",
        "downstream_consequence": "Motivated explicit ambiguity witnesses and later controlled P2 acquisition.",
        "authority_status": "AUTHORITATIVE_PSF_PROVENANCE_PREFLIGHT",
        "supersession_status": "043319_043342_043415_LAUNCH_HISTORY; 043442_AUTHORITATIVE",
    },
})


RUN_OVERRIDES.update({
    "thayer_capacity_ladder_20260713_013132": {
        "campaign_type": "preflight",
        "predecessor": "thayer_feasibility_projection_20260712_234216",
        "hypothesis": "Decoder capacity can be tested only after selecting one unique nonnegative output mapping.",
        "exact_authorized_change": "Preflighted identity/ReLU/square/absolute representation and output contracts before building L0-L3.",
        "data_accessed": "Frozen P0/input hashes; no scientific fit.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Mapping audit only; prospective decoder sizes.",
        "scene_or_sample_count": "No fitted scenes.",
        "number_of_starts": "0.",
        "compute_budget": "0 optimizer steps.",
        "metrics": "All mappings represent P0; identity is physically invalid; no unique replacement was authorized.",
        "final_outcome": "STOPPED — NO UNIQUE CONTRACT-COMPLIANT OUTPUT MAPPING.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "No capacity evidence was produced.",
        "downstream_consequence": "Required direct mapping comparison at fixed L0.",
        "authority_status": "ENGINEERING_PREFLIGHT_STOP",
        "supersession_status": "SUPERSEDES_20260713_005215_INCOMPLETE_CHECKPOINT_INVENTORY",
    },
    "thayer_audit_v0_20260714_154655": {
        "campaign_type": "scientific",
        "predecessor": "Condition C family inventory and authoritative D3 diagnosis.",
        "hypothesis": "Separate PRE query-state and POST truth-free safety auditors can yield a useful fail-closed catalog policy.",
        "exact_authorized_change": "Trained A1 PRE on blend+prompt and A2 POST on blend/prompt/reconstruction/residual plus 25 scalars, with temperature calibration and a frozen threshold.",
        "data_accessed": "PRE 7,055/2,000/4,000 train/validation/calibration; POST 7,025/2,000 plus 4,000 policy calibration.",
        "protected_data_status": "Existing Atlas used only as post-freeze diagnostic; final-lockbox outcome access 0.",
        "model_or_solver": "A1 28,307 parameters; A2 155,209.",
        "scene_or_sample_count": "Episode counts as listed in data_accessed.",
        "number_of_starts": "Three PRE and three POST seeds.",
        "compute_budget": "Up to 30 epochs per seed.",
        "metrics": "PRE macro-F1 validation/calibration .894711/.797993; POST unsafe prevalence 1.0, AUROC undefined, AUPRC 1.0, accepted coverage 0.",
        "final_outcome": "DIRECT_AUDITOR_PARTIAL; PRE formal pass false, POST/final policy false.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Only Condition C was core-eligible and every POST example was unsafe.",
        "downstream_consequence": "Required a distinct physical family with safe-positive support.",
        "authority_status": "VALID_PARTIAL_DIRECT_AUDIT_RESULT",
        "supersession_status": "ADDENDUM_CORRECTS_FINAL_CLASS_TOKEN; SATURATED_POST_METRICS_NONDISCRIMINATIVE",
    },
    "thayer_pu_eligibility_v1_20260714_213113": {
        "campaign_type": "preflight",
        "predecessor": "thayer_audit_v0_20260714_154655 family-diversity blocker.",
        "hypothesis": "Frozen PU can act as a reproducible second family under one truth-free K=16 deployment rule.",
        "exact_authorized_change": "Used mean K=16 prior samples with fixed seeds and batch-8 MPS, gated by single-versus-batched replay.",
        "data_accessed": "24 preflight scenes; 3,998/793/2,800 source manifests prepared but not fully inferred.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Frozen epoch-27 PU checkpoint.",
        "scene_or_sample_count": "24 preflight scenes.",
        "number_of_starts": "16 latent seeds per scene; no training.",
        "compute_budget": "384 candidate scene-seed evaluations plus deployment aggregation.",
        "metrics": "Batch-1 candidate/deployed hashes failed 24/24; batch 4 equaled batch 8; max deltas .005859375/.000766754.",
        "final_outcome": "THAYER_PU_DEPLOYMENT_INELIGIBLE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Exact replay gate failed before label construction.",
        "downstream_consequence": "Required a batch-geometry repair.",
        "authority_status": "SUPERSEDED_EXECUTOR_PREFLIGHT_RESULT",
        "supersession_status": "BATCH_R1_SUPERSEDES_DEPLOYMENT_STATUS; INCIDENT_RETAINED",
    },
    "thayer_pu_batch_r1_20260714_224244": {
        "campaign_type": "correction",
        "predecessor": "thayer_pu_eligibility_v1_20260714_213113",
        "hypothesis": "The PU mismatch is batch-geometry numerics repairable by a fixed padded executor without model changes.",
        "exact_authorized_change": "Forced every MPS neural call to eight rows, padded short chunks with zero dummies, then stripped dummy outputs.",
        "data_accessed": "24 diagnostic scenes, then full 3,998/793/2,800 partitions.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Unchanged frozen PU and K=16 deployment rule.",
        "scene_or_sample_count": "7,591 complete outputs.",
        "number_of_starts": "16 latent seeds per episode; no training.",
        "compute_budget": "7,591 x 16 deployed samples plus diagnostics.",
        "metrics": "Corrected preflight/replay exact; safe/unsafe train 0/3998, validation 0/793, calibration 0/2800.",
        "final_outcome": "THAYER_PU_ELIGIBLE_BUT_LABEL_COLLAPSED.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Deployment consistency was repaired but no safe-positive support existed.",
        "downstream_consequence": "PU remained unusable for POST and motivated a physically distinct family.",
        "authority_status": "SUPERSEDING_DEPLOYMENT_EXECUTOR_CORRECTION",
        "supersession_status": "SUPERSEDES_INELIGIBLE_STATUS_ONLY; 24_OF_24_FAILURE_INCIDENT_RETAINED",
    },
    "thayer_family_e_v0_20260714_195256": {
        "campaign_type": "engineering_invalid",
        "predecessor": "Condition C and repaired PU all-unsafe populations.",
        "hypothesis": "A nonnegative exactly flux-conserving source/residual simplex can produce a physically distinct safer family.",
        "exact_authorized_change": "Defined per-band requested/companion/residual softmax fractions multiplied by the raw observation.",
        "data_accessed": "10,000 train, 2,000 validation, 2,000 calibration frozen episodes.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Analytic physical preflight; prospective 1,162,737-parameter U-Net not instantiated.",
        "scene_or_sample_count": "14,000.",
        "number_of_starts": "0.",
        "compute_budget": "0 optimizer steps.",
        "metrics": "Negative-observation fractions .486877/.481794/.482363; every episode had target-sum exceedance; synthetic conservation error 4.768e-7.",
        "final_outcome": "DATA_OR_IMPLEMENTATION_FAILURE: all-nonnegative simplex incompatible with signed observations.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Physical contract was structurally infeasible for signed zero-background data.",
        "downstream_consequence": "Authorized exactly one signed-residual preflight.",
        "authority_status": "ENGINEERING_INVALID_PHYSICAL_CONTRACT",
        "supersession_status": "FAILURE_REMAINS_CORRECT_FOR_ITS_CONTRACT; E1_CHANGES_CONTRACT_ONLY",
    },
    "thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340": {
        "campaign_type": "preflight",
        "predecessor": "thayer_family_e_v0_20260714_195256",
        "hypothesis": "Allowing only the observational closure residual to be signed restores representability while sources remain nonnegative.",
        "exact_authorized_change": "Defined P_req=S*ReLU, P_comp=S*ReLU, and P_noise=O-P_req-P_comp.",
        "data_accessed": "Same 10,000/2,000/2,000 frozen targets.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Analytic inverse-coordinate witnesses and synthetic MPS checks; no model.",
        "scene_or_sample_count": "14,000.",
        "number_of_starts": "0.",
        "compute_budget": "Sequential representability audit only.",
        "metrics": "Mapped-source negatives 0; requested/companion errors within tolerance; float32 closure <=.015625; signed residual roughly balanced.",
        "final_outcome": "SIGNED_NOISE_RESIDUAL_CONTRACT_PASS.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Establishes representability, not learnability, identity, or safety.",
        "downstream_consequence": "Authorized Family-E1 model eligibility.",
        "authority_status": "AUTHORITATIVE_PHYSICAL_CONTRACT_CORRECTION",
        "supersession_status": "SUPERSEDES_ONLY_THE_PHYSICAL_REPRESENTATION_ELEMENT",
    },
    "thayer_family_e1_v0_20260714_214715": {
        "campaign_type": "scientific",
        "predecessor": "thayer_family_e1_signed_noise_residual_preflight_v0_20260714_202340",
        "hypothesis": "A compact coordinate U-Net can learn nonnegative requested/companion sources under the corrected physical map.",
        "exact_authorized_change": "Used a four-channel compact U-Net with six-channel ReLU source head and signed residual closure.",
        "data_accessed": "Micro indices: ordinary 16, difficult 6, mixed 0/3/5/6/18/51/73/81; full manifests audited but full training not run.",
        "protected_data_status": "Validation/calibration/development/Atlas/lockbox inference 0.",
        "model_or_solver": "1,162,662-parameter FamilyE1UNet.",
        "scene_or_sample_count": "Three micro conditions with 1, 1, and 8 scenes; 20 prompt views.",
        "number_of_starts": "Three independent micro fits/seeds.",
        "compute_budget": "1,500 / 2,000 / 3,000 updates.",
        "metrics": "Ordinary objective reduction .998960 and identity 1.0; difficult identity .5; mixed identity .5625; source-negative fractions 0.",
        "final_outcome": "FAMILY_E1_RECONSTRUCTION_FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Micro-only; reconstruction losses passed while source ordering failed.",
        "downstream_consequence": "Authorized a micro-only paired-prompt intervention.",
        "authority_status": "VALID_NEGATIVE_FAMILY_E1_RESULT",
        "supersession_status": "214638_FAILED_BOOTSTRAP; 214715_AUTHORITATIVE",
    },
    "thayer_family_e1p_v0_20260714_225228": {
        "campaign_type": "scientific",
        "predecessor": "thayer_family_e1_v0_20260714_214715",
        "hypothesis": "An explicit paired-prompt source-ordering intervention repairs requested identity.",
        "exact_authorized_change": "Instrumented and deterministically replicated difficult/mixed fits; audit showed paired A/B targets already existed, so training semantics were unchanged.",
        "data_accessed": "Difficult index 6 and mixed indices 0/3/5/6/18/51/73/81.",
        "protected_data_status": "Validation/calibration/development/Atlas/lockbox access 0.",
        "model_or_solver": "Unchanged 1,162,662-parameter Family-E1 micro states plus layerwise prompt tracing.",
        "scene_or_sample_count": "8 unique scenes, 9 condition entries, 18 prompt views in two replicated conditions.",
        "number_of_starts": "Two deterministic replications.",
        "compute_budget": "2,000 and 3,000 updates, matching E1.",
        "metrics": "Difficult/mixed identity .5/.5625; prompt swap 0/.125; all 28 scientific values reproduced; nonzero prompt modulation/gradients at every layer.",
        "final_outcome": "FAIL: generic prompt signal survived but identity-aligned prompt semantics were too weak.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Generic prompt sensitivity is not source-allocation information.",
        "downstream_consequence": "Motivated the observation-level null-space analysis.",
        "authority_status": "AUTHORITATIVE_PROMPT_IDENTITY_DIAGNOSIS",
        "supersession_status": "COMPLEMENTS_E1; CORRECTS_MISSING_PAIRED_EXAMPLES_MECHANISM",
    },
})


RUN_OVERRIDES.update({
    "thayer_competing_hypotheses_20260712_131111": {
        "campaign_type": "scientific",
        "predecessor": "Calibration/observability dead end.",
        "hypothesis": "Large scientific failures may reflect multiple observation-consistent truths rather than only poor calibration.",
        "exact_authorized_change": "Searched a 30,000-scene pool for near-collisions, optimized 25 pairs, and inventoried deblender families.",
        "data_accessed": "Training/search pool only: 30,000 scenes, 100 numerical candidates, 25 optimized pairs.",
        "protected_data_status": "Historical development/lockbox access 0/0; later Atlas assets prepared under freeze.",
        "model_or_solver": "Forward optimizer plus Condition C/R0/R1 candidate inventory.",
        "scene_or_sample_count": "25 pairs, 50 truth decompositions, 150 model-candidate decompositions.",
        "number_of_starts": "Optimization-start count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Witnesses for 25/25 pairs; 49/50 constructed observations retained two truths; 150/150 candidate decompositions unsafe.",
        "final_outcome": "PARTIAL SUCCESS; only one distinct model-family cluster.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Cross-family audit was impossible.",
        "downstream_consequence": "Motivated the formal Atlas and operational detector test.",
        "authority_status": "AUTHORITATIVE_AMBIGUITY_FEASIBILITY_RESULT",
        "supersession_status": "ATLAS_SUPERSEDES_PROVISIONAL_OPERATIONAL_EVALUATION_NOT_WITNESS_EXISTENCE",
    },
    "thayer_ambiguity_atlas_v0_20260712_145627": {
        "campaign_type": "scientific",
        "predecessor": "thayer_competing_hypotheses_20260712_131111",
        "hypothesis": "Optimized near-collision pairs form an ambiguity benchmark and model-candidate diameter can detect them.",
        "exact_authorized_change": "Froze 25 Atlas pairs, validated constructed truths, and evaluated three same-cluster checkpoints with matched controls.",
        "data_accessed": "30,000-scene search pool, 100 near collisions, 25 pairs.",
        "protected_data_status": "Atlas opened once after freeze; historical development/lockbox access 0/0.",
        "model_or_solver": "Constrained truth optimizer plus same-cluster deterministic deblenders.",
        "scene_or_sample_count": "25 pairs / 50 prompt-observations.",
        "number_of_starts": "Optimizer-start count NOT_RECORDED.",
        "compute_budget": "NOT_RECORDED",
        "metrics": "Whitened MSE 8.817e-5 to 3.90451e-4; diameter 5.348-22.787x; constructed witnesses 50/50; model witnesses 19/50; AUROC .4712; recall at 4% FPR 0.",
        "final_outcome": "FAILURE AFTER ATLAS PASS: witnesses exist, operational detector failed.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Finite exhibited cases, one family cluster, and no prevalence estimate.",
        "downstream_consequence": "Motivated a genuinely distinct or stochastic model family.",
        "authority_status": "AUTHORITATIVE_AMBIGUITY_DIAGNOSTIC_RESULT",
        "supersession_status": "FINAL_REPORT_CORRECTED_COUNTS_GOVERN; PU_IS_NEW_FAMILY_NOT_CORRECTION",
    },
    "thayer_prompted_resunet_diversity_20260712_154122": {
        "campaign_type": "scientific",
        "predecessor": "Atlas family-diversity gate.",
        "hypothesis": "An architecturally distinct ResUNet yields promptable, meaningfully different candidates.",
        "exact_authorized_change": "Replaced the plain U-Net with a six-block residual encoder/decoder and used no Condition-C warm start.",
        "data_accessed": "10,000 train and 1,500 validation scenes; all 59 Atlas/feasibility groups excluded.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0; Atlas unopened.",
        "model_or_solver": "199,219-parameter PromptedResUNet.",
        "scene_or_sample_count": "11,500.",
        "number_of_starts": "One seed (2026077301).",
        "compute_budget": "20 epochs; best epoch 18.",
        "metrics": "Prompt swap .394667; whole/source MSE ratios versus Condition C 1.120486/1.787554.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Failed promptability and batch-geometry candidate-hash contract.",
        "downstream_consequence": "Atlas inference prohibited; motivated a stochastic latent family.",
        "authority_status": "VALID_NEGATIVE_PROMPTED_RESUNET_RESULT",
        "supersession_status": "153854_AND_153913_ENGINEERING_ONLY; 154122_AUTHORITATIVE",
    },
    "thayer_probabilistic_unet_20260712_163340": {
        "campaign_type": "scientific",
        "predecessor": "Deterministic Atlas detector and Prompted ResUNet failures.",
        "hypothesis": "Stochastic latent candidates broaden plausible diameter and recover alternate truths.",
        "exact_authorized_change": "Added an 8-dimensional posterior/prior latent path with training-only truth posterior and compatible Condition-C warm start.",
        "data_accessed": "16,000 train, 2,000 validation, calibration/non-Atlas controls, and one frozen 25-pair Atlas pass.",
        "protected_data_status": "Atlas one pass after freeze; development/lockbox access 0/0.",
        "model_or_solver": "170,278-parameter probabilistic U-Net; 97,606 then 153,286 trainable by phase.",
        "scene_or_sample_count": "18,000 core train/validation plus controls and 50 Atlas prompt-observations.",
        "number_of_starts": "One training seed; stochastic K varied by diagnostic.",
        "compute_budget": "30 epochs.",
        "metrics": "Prompt swap majority/best-K .9875/.99425; forward fraction .951844 non-Atlas and 1.0 Atlas; witnesses 24/50; AUROC .856; recall at 4% FPR .32; own/alternate truth coverage 0/0.",
        "final_outcome": "PARTIAL SUCCESS for diversity; no truth coverage.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Diameter diversity without truth coverage and safe-control false-witness rate .08.",
        "downstream_consequence": "Required posterior/decoder sufficiency before a flow fit.",
        "authority_status": "VALID_PARTIAL_STOCHASTIC_FAMILY_RESULT",
        "supersession_status": "CURRENT_FOR_PU_SCIENCE; BATCH_R1_GOVERNS_DEPLOYMENT_STATUS",
    },
    "thayer_flow_prior_20260712_182516": {
        "campaign_type": "preflight",
        "predecessor": "thayer_probabilistic_unet_20260712_163340",
        "hypothesis": "A flow prior can help only if the frozen posterior/decoder already represents the truths.",
        "exact_authorized_change": "Applied a K=32 posterior/decoder sufficiency gate before constructing any flow.",
        "data_accessed": "512 ordinary prompt evaluations and 500 near-collision own/cross evaluations.",
        "protected_data_status": "No new Atlas, development, or lockbox access.",
        "model_or_solver": "Frozen PU posterior/decoder; no flow built.",
        "scene_or_sample_count": "1,012 evaluations, K=32 each.",
        "number_of_starts": "32 posterior samples per evaluation; 0 optimizer starts.",
        "compute_budget": "0 training steps.",
        "metrics": "Ordinary-own, near-own, and cross-alternate coverage all 0; alternate identity .017625.",
        "final_outcome": "FAILURE — POSTERIOR/DECODER INSUFFICIENT.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Does not test an independently trained flow/decoder architecture.",
        "downstream_consequence": "Prohibited the flow branch and motivated explicit set supervision.",
        "authority_status": "VALID_NEGATIVE_PREFIT_RESULT",
        "supersession_status": "CURRENT_FOR_FLOW_BRANCH",
    },
    "thayer_multiple_hypotheses_20260712_190701": {
        "campaign_type": "scientific",
        "predecessor": "thayer_flow_prior_20260712_182516",
        "hypothesis": "Explicit K=2 set supervision can retain both approved truths.",
        "exact_authorized_change": "Trained a compact shared two-hypothesis decoder on ordinary and ambiguous target sets.",
        "data_accessed": "12,000/3,000 train ordinary/ambiguous; 1,500/500 validation; 1,500/500 calibration.",
        "protected_data_status": "Atlas/development/lockbox access 0/0/0.",
        "model_or_solver": "120,022-parameter K=2 model.",
        "scene_or_sample_count": "19,000 observations.",
        "number_of_starts": "One model fit.",
        "compute_budget": "30 MPS epochs.",
        "metrics": "Prompt token identity .992/.992; set swap .992; own/alternate/both coverage 0/0/0.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Non-Atlas truth coverage failed before Atlas evaluation.",
        "downstream_consequence": "Motivated independent experts on a microset.",
        "authority_status": "VALID_NEGATIVE_MULTI_HYPOTHESIS_RESULT",
        "supersession_status": "CURRENT_FOR_SHARED_K2_DECODER",
    },
    "thayer_two_expert_decoder_20260712_203121": {
        "campaign_type": "scientific",
        "predecessor": "thayer_multiple_hypotheses_20260712_190701",
        "hypothesis": "Independent decoders avoid shared-head compromise and can memorize both modes.",
        "exact_authorized_change": "Used two independent 46,470-parameter decoders with one shared 72,672-parameter encoder.",
        "data_accessed": "32 ordinary plus 32 ambiguous persisted micro rows.",
        "protected_data_status": "Validation/calibration/Atlas/development/lockbox access all 0.",
        "model_or_solver": "165,612 total parameters; 92,940 phase-1 trainable.",
        "scene_or_sample_count": "64.",
        "number_of_starts": "One model fit with distinct expert initialization seeds.",
        "compute_budget": "400/400 epochs.",
        "metrics": "Own/alternate/both coverage 0/0/0; prompt swap .953125; forward .96875/1.0.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "One microset and one objective/initialization family.",
        "downstream_consequence": "Motivated separate representation, loss, assignment, and optimization diagnosis.",
        "authority_status": "VALID_NEGATIVE_TWO_EXPERT_RESULT",
        "supersession_status": "203038_LAUNCH_HISTORY; 203121_AND_CORRECTNESS_ADDENDUM_GOVERN",
    },
    "thayer_loss_geometry_20260712_205733": {
        "campaign_type": "scientific",
        "predecessor": "thayer_two_expert_decoder_20260712_203121",
        "hypothesis": "The frozen objective may prefer compromises or contain conflicting/nonsmooth gradients.",
        "exact_authorized_change": "Evaluated truths, trained outputs, collapsed means, assignments, HVPs, and output-space paths without model fitting.",
        "data_accessed": "Persisted 64-row microset.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Analytic/autodiff objective and curvature diagnostics; no neural optimizer.",
        "scene_or_sample_count": "64.",
        "number_of_starts": "0 neural starts; multiple frozen output configurations/path probes.",
        "compute_budget": "0 neural steps.",
        "metrics": "Ambiguous trained/truth objective .0265422/.0293535; trained lower on 32/32; forward term 76.496%/86.711%; set-forward gradient conflict 63.281%/51.562%.",
        "final_outcome": "MIXED CAUSE: exact truth representable, aggregate objective often prefers compromise.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Local/micro objective diagnosis, not a trained corrective model.",
        "downstream_consequence": "Motivated a scientifically aligned surrogate.",
        "authority_status": "AUTHORITATIVE_MECHANISTIC_LOSS_RESULT",
        "supersession_status": "CURRENT_FOR_FULL_OBJECTIVE_DIAGNOSIS",
    },
    "thayer_scientific_alignment_20260712_220315": {
        "campaign_type": "scientific",
        "predecessor": "thayer_loss_geometry_20260712_205733",
        "hypothesis": "A corrected source-specific surrogate makes truth stable and moves compromises into coverage.",
        "exact_authorized_change": "Replaced the misaligned loss in a detached output-space preflight; neural fitting remained gated.",
        "data_accessed": "Same 64 micro rows.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Differentiable scientific surrogate and detached output optimizers.",
        "scene_or_sample_count": "64.",
        "number_of_starts": "Trained, collapsed, wrong-allocation, and truth initialization families; 0 neural starts.",
        "compute_budget": "Detached fits only; neural budget unused.",
        "metrics": "Surrogate Spearman .990679, Kendall .957683; truth stationary; compromise starts improved but did not enter full gates.",
        "final_outcome": "FAILURE.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Output-space preflight only; smoke detached fits preceded freeze.",
        "downstream_consequence": "Motivated conditioning and alternative-coordinate audits.",
        "authority_status": "VALID_NEGATIVE_SURROGATE_ALIGNMENT_RESULT",
        "supersession_status": "POST_FINAL_ADDENDUM_MAKES_STRICT_CORRECTNESS_FAIL; MEASUREMENTS_RETAINED",
    },
    "thayer_output_conditioning_20260712_225459": {
        "campaign_type": "scientific",
        "predecessor": "thayer_scientific_alignment_20260712_220315",
        "hypothesis": "Coordinate transforms, L-BFGS, alternating optimization, or preconditioning can reach the narrow scientific basin.",
        "exact_authorized_change": "Compared six methods C0-C5 across five initialization families.",
        "data_accessed": "64-row microset.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Detached output-space Adam, L-BFGS, alternating, transformed, and preconditioned optimization.",
        "scene_or_sample_count": "64 per method-initialization aggregate.",
        "number_of_starts": "30 method x initialization cells.",
        "compute_budget": "Up to 400 updates per cell.",
        "metrics": "Best ordinary .4375, own .84375, alternate .875, both .8125; no method passed all gates.",
        "final_outcome": "PARTIAL SUCCESS — SCIENTIFIC-BASIN EXTREMITY.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "No neural result; some transformed methods failed truth stationarity.",
        "downstream_consequence": "Motivated direct projection into the strict scientific interior.",
        "authority_status": "VALID_PARTIAL_OUTPUT_CONDITIONING_RESULT",
        "supersession_status": "CORRECTNESS_ADDENDUM_CONTROLS_RAW_CONDITION_AND_STRICT_STATUS",
    },
    "thayer_feasibility_projection_20260712_234216": {
        "campaign_type": "scientific",
        "predecessor": "thayer_output_conditioning_20260712_225459",
        "hypothesis": "Projecting targets into strict scientific interiors removes threshold extremity and permits two-expert memorization.",
        "exact_authorized_change": "Constructed P0 homotopy-interior and P1 nearest-feasible targets and trained unchanged two-expert model on the selected target.",
        "data_accessed": "64 rows, 256 scene/prompt/expert pairings.",
        "protected_data_status": "Development/Atlas/lockbox access 0/0/0.",
        "model_or_solver": "Homotopy/projection solver plus unchanged two-expert model.",
        "scene_or_sample_count": "256 projection pairings.",
        "number_of_starts": "256 projection paths plus one neural fit.",
        "compute_budget": "Two-expert model 400 epochs.",
        "metrics": "P0 feasibility 256/256; median alpha .999979483; median correction .946369; z flux limiting 173/256; neural coverage all 0.",
        "final_outcome": "Projection success but unchanged neural failure; trajectory after negative-output stop is diagnostic only.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Neural trajectory after epoch 1 was not authoritative.",
        "downstream_consequence": "Required a contract-compliant output mapping before a capacity ladder.",
        "authority_status": "VALID_PROJECTION_RESULT_AND_INVALID_NEURAL_CONTINUATION",
        "supersession_status": "CORRECTED_P0_AUTHORITATIVE; P1_REJECTED_BY_ADDENDA",
    },
    "thayer_output_parameterization_20260713_023120": {
        "campaign_type": "scientific",
        "predecessor": "thayer_capacity_ladder_20260713_013132",
        "hypothesis": "ReLU, square, or absolute mapping can pass physical, optimization, and scientific gates at fixed L0.",
        "exact_authorized_change": "Fit each mapping separately on one ordinary and one ambiguous scene.",
        "data_accessed": "Two training micro scenes.",
        "protected_data_status": "All data outside selected micro scenes, including Atlas/development/lockbox, had zero access.",
        "model_or_solver": "Fixed-L0 two-expert decoder under three output mappings.",
        "scene_or_sample_count": "2 scenes x 3 mappings.",
        "number_of_starts": "Six fits.",
        "compute_budget": "3,200 steps per fit.",
        "metrics": "Every final coverage 0; square lowest ordinary/ambiguous losses 1.16478e-5/3.114e-6.",
        "final_outcome": "NO MAPPING PASSES.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "One start per mapping/scene and a fixed finite budget.",
        "downstream_consequence": "Motivated separate output, feature, head, and full-decoder reachability tests.",
        "authority_status": "VALID_NEGATIVE_OUTPUT_MAPPING_RESULT",
        "supersession_status": "022852_INVALID_JSON; 022924_UNATTAINABLE_GATE; 023120_AUTHORITATIVE",
    },
    "thayer_repository_integrity_20260713_031653": {
        "campaign_type": "scientific",
        "predecessor": "Output-parameterization failure and stopped first fixed-feature audit.",
        "hypothesis": "A D0-D3 decomposition can localize the first reachability barrier without changing targets or mapping.",
        "exact_authorized_change": "Tested D0 free raw logits, D1 free penultimate features, D2 final head only, and gated D3 under prospective mappings.",
        "data_accessed": "One exact ambiguous training scene: source row 12000 / projected P0 row 32.",
        "protected_data_status": "Ordinary remainder, eight-scene set, Atlas, development, and lockbox access all 0.",
        "model_or_solver": "Direct tensors, free features, and frozen heads.",
        "scene_or_sample_count": "One scene/two prompts.",
        "number_of_starts": "Five executed conditions: D0 x3, D1 square, D2 square.",
        "compute_budget": "5,000 MPS steps each.",
        "metrics": "D0 square final 3.614716e-10 with all coverage 1; D1 3.1026115e-9 with all 1; D2 4.45156e-6, residual capture .2853197, all 0, design ranks 17/17.",
        "final_outcome": "FROZEN-FEATURE CONDITIONING BARRIER; D0/D1 pass, D2 fails, D3 separately gated.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "One scene; no full-decoder or capacity conclusion.",
        "downstream_consequence": "Persisted the D1 endpoint and motivated square-only full-L0 D3.",
        "authority_status": "AUTHORITATIVE_D0_D2_MECHANISTIC_RESULT",
        "supersession_status": "SUPERSEDES_STOPPED_FIXED_FEATURE_AUDIT; IN_RUN_SUPERSEDING_TABLES_GOVERN",
    },
    "thayer_d3_pv1a1_entrypoint_r3_20260714_175200": {
        "campaign_type": "scientific",
        "predecessor": "thayer_d3_pv1a1_readiness_r2_20260714_165947, blocked only by the absent entrypoint.",
        "hypothesis": "Fresh-seeded L0 can satisfy the full one-scene predicate and authorize eight-scene work or a capacity ladder.",
        "exact_authorized_change": "Implemented the missing exact-command entrypoint and executed the frozen R3 command once without scientific-rule changes.",
        "data_accessed": "One primary ambiguous scene/two prompt views/four prompt-expert candidates; eight authorized training indices read by frozen cache/protocol machinery.",
        "protected_data_status": "Atlas/development/lockbox/broader-scene access 0/0/0/0.",
        "model_or_solver": "Fresh-seeded two-expert L0 with 92,940 decoder optimizer parameters.",
        "scene_or_sample_count": "One primary scientific scene.",
        "number_of_starts": "One run-once start.",
        "compute_budget": "5,000 optimizer steps; 10,108 forwards; 5,054 target losses.",
        "metrics": "Own max distance .7153766 PASS; alternate 3.6737798 FAIL; both FAIL; prompt and all four forward candidates PASS; five assignment flips; capture 1.0.",
        "final_outcome": "MIXED_CAUSE; branch NONE; no downstream authorization.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "One fresh-seeded local L0 setup; eight-scene and capacity ladders unrun.",
        "downstream_consequence": "No further D3 experiment authorized.",
        "authority_status": "AUTHORITATIVE_LOCAL_L0_D3_RESULT",
        "supersession_status": "SUPERSEDES_STALE_D3_NOT_RUN_STATUS; UNRUN_BRANCHES_REMAIN_UNKNOWN",
    },
})


NON_RUN_CAMPAIGN_RECORDS: tuple[dict[str, str], ...] = (
    {
        "run_directory": "non_run_campaign:Thayer-Recoverability-v0",
        "campaign_name": "Thayer-Recoverability-v0",
        "timestamp": "NOT_RECORDED",
        "campaign_type": "scientific",
        "predecessor": "Family-E1/E1P prompt-identity diagnosis and the fixed-feature D0-D3 audit.",
        "hypothesis": "Is observation-level source identity unique under the unrestricted Family-E1 output contract?",
        "exact_authorized_change": "Performed analytic and direct nonnegative allocation analysis only; no reconstruction model, checkpoint, or campaign rerun.",
        "data_accessed": "Family-E1 training indices 0,3,5,6,18,51,73,81; 8 unique scenes and 9 condition entries; verified observation/source hashes.",
        "protected_data_status": "No training/validation/calibration/development/Atlas/lockbox array, model, or checkpoint access; report-level source tensors were used under the frozen training-scene authority.",
        "model_or_solver": "Analytic [I I] forward map plus direct nonnegative allocation witnesses.",
        "scene_or_sample_count": "8 unique scenes; 9 condition entries.",
        "number_of_starts": "32 direct nonnegative starts per scene; 256 fits.",
        "compute_budget": "One projection step per start.",
        "metrics": "Forward-map rank/nullity, exact witnesses, scientific diameter, overlap, flux ratio, separation, color similarity, and PSF overlap.",
        "final_outcome": "All 8 scenes FUNDAMENTALLY_UNIDENTIFIABLE; fixed-residual rank/nullity 10,800/10,800 and free-signed-residual nullity 21,600.",
        "expected_or_surprising": "NOT_RECORDED",
        "primary_limitation": "Unrestricted direct output and selected simulated training scenes; it does not prove every structurally constrained inverse is non-identifiable.",
        "downstream_consequence": "Shifted the program to explicit structural/information contracts and the oracle-to-flux-free ladder.",
        "authority_status": "AUTHORITATIVE_UNRESTRICTED_IDENTIFIABILITY_RESULT",
        "supersession_status": "CURRENT_FOR_UNRESTRICTED_ALLOCATION; STALE_D3_BACKGROUND_SENTENCE_SUPERSEDED_BY_R3_MIXED_CAUSE",
        "final_report_path": "reports/thayer_recoverability_v0.md",
        "protocol_path": "",
        "manifest_path": "",
        "summary_table_paths": "",
        "key_figure_path": "",
        "commit_artifact": "curated_compact_copy",
        "notes": "Non-run analytic campaign retained because the authoritative report is repository-level rather than under outputs/runs.",
    },
)


D3_ENGINEERING_PREFIXES = (
    "thayer_authoritative_d3_", "thayer_full_l0_d3_",
    "thayer_d1_endpoint_replay_", "thayer_full_l0_d3r_",
    "thayer_d3_runtime_readiness_", "thayer_d3_scientific_capsule_",
    "thayer_capsule_authoritative_d3_", "thayer_d3_executable_contract_",
    "thayer_scientific_d3_", "thayer_d3_policy_contract_",
    "thayer_final_authoritative_d3_", "thayer_final_authoritative_d3_policy_preflight_",
    "thayer_d3_integration_science_", "thayer_d3_v41_science_",
    "thayer_d3_i41r1_", "thayer_d3_onego_", "thayer_d3_hash_r1_",
    "thayer_d3_alignment_r1_", "thayer_d3_semantic_path_r1_",
    "thayer_authoritative_scientific_d3_", "thayer_d3_protocol_readiness_r1_",
    "thayer_d3_pv1_readiness_r1_", "thayer_d3_pv1a1_readiness_r2_",
    "thayer_d3_pv1a1_entrypoint_r3_",
)

LAUNCH_HISTORY_PREFIXES = (
    "thayer_prompted_resunet_diversity_",
    "thayer_two_expert_decoder_",
    "thayer_output_conditioning_",
    "thayer_output_parameterization_",
    "thayer_family_e1_v0_",
    "thayer_select_psf_conditioning_",
    "thayer_select_shape_constrained_quantile_",
)


def grouped_run_override(name: str) -> dict[str, str]:
    """Return conservative metadata for repeated non-scientific launch groups."""
    if name.startswith(D3_ENGINEERING_PREFIXES):
        return {
            "campaign_type": "engineering_invalid",
            "predecessor": "Authoritative D0-D2 fixed-feature barrier and persisted D1 endpoint.",
            "hypothesis": "The exact full-L0 D3 test can be made reproducible and protocol-complete.",
            "exact_authorized_change": "Engineering closure only: endpoint persistence, path/runtime guards, capsules, thresholds, policies, tensor hashes, dtype/serialization, semantic paths, initialization authority, or entrypoint repair.",
            "data_accessed": "Mostly frozen metadata, synthetic fixtures, and one-scene caches; ALIGN/SEMANTIC attempts reached only one scientific update.",
            "protected_data_status": "Atlas/development/lockbox access 0/0/0.",
            "model_or_solver": "Prospective 92,940-trainable-parameter L0 decoder; no complete scientific campaign in this grouped record.",
            "scene_or_sample_count": "No complete distinct scientific campaign; grouped engineering attempt.",
            "number_of_starts": "Repeated launch candidate; not an independent scientific start.",
            "compute_budget": "No complete 5,000-step run; at most one optimizer step in ALIGN/SEMANTIC attempts.",
            "metrics": "Engineering readiness, hashes, tests, and stop reasons only.",
            "final_outcome": "FAIL-CLOSED ENGINEERING HISTORY; no authoritative D3 scientific result from this launch.",
            "expected_or_surprising": "NOT_RECORDED",
            "primary_limitation": "Implementation/protocol closure only.",
            "downstream_consequence": "Contributed to the final R3 entrypoint campaign.",
            "authority_status": "ENGINEERING_HISTORY_NO_DISTINCT_SCIENTIFIC_AUTHORITY",
            "supersession_status": "R3_SUPERSEDES_D3_NOT_RUN_CURRENT_STATUS; INCIDENT_HISTORY_RETAINED",
        }
    if name.startswith(LAUNCH_HISTORY_PREFIXES):
        return {
            "campaign_type": "engineering_invalid",
            "predecessor": "See the later authoritative run with the same campaign stem.",
            "hypothesis": "Launch the frozen campaign without changing its scientific question.",
            "exact_authorized_change": "Scaffold, serialization, schema, contract, or launch repair only.",
            "data_accessed": "No distinct completed scientific data use; see the run-local access audit.",
            "protected_data_status": "No independent scientific authority; protected results must be read from the later valid run only.",
            "model_or_solver": "Campaign did not complete a distinct authoritative fit.",
            "scene_or_sample_count": "0 distinct authoritative scientific scenes.",
            "number_of_starts": "0 authoritative scientific starts.",
            "compute_budget": "Stopped before a distinct authoritative outcome.",
            "metrics": "Engineering stop reason only.",
            "final_outcome": "ENGINEERING_INVALID_OR_INCOMPLETE LAUNCH; no distinct scientific result.",
            "expected_or_surprising": "NOT_RECORDED",
            "primary_limitation": "Launch/contract failure.",
            "downstream_consequence": "Superseded by the later valid same-stem run.",
            "authority_status": "ENGINEERING_HISTORY_NO_DISTINCT_SCIENTIFIC_AUTHORITY",
            "supersession_status": "LATER_VALID_SAME_STEM_RUN_GOVERNS_SCIENCE",
        }
    return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_files() -> list[Path]:
    paths: list[Path] = []
    for directory, names, filenames in os.walk(ROOT):
        names[:] = [name for name in names if name not in EXCLUDED_DIRS and not name.startswith(".venv-btk-")]
        base = Path(directory)
        paths.extend(base / filename for filename in filenames)
    return sorted(paths)


def ignored_paths(relative_paths: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(relative_paths) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "git check-ignore failed")
    return {line for line in result.stdout.splitlines() if line}


def classify_large(path: Path) -> tuple[str, str, str, str]:
    rel = path.relative_to(ROOT).as_posix()
    suffix = path.suffix.lower()
    if rel.startswith("data/") or rel.startswith("data_exploration/data/"):
        return (
            "raw_dataset_or_catalog",
            "PROTECTED_LOCAL_DATA",
            "EXCLUDE_LOCAL_ONLY",
            "Raw datasets and downloaded catalogs are not repository artifacts.",
        )
    if suffix in {".pt", ".pth", ".ckpt"} or "/checkpoints/" in rel:
        return (
            "checkpoint_or_optimizer_state",
            "LOCAL_MODEL_STATE",
            "EXCLUDE_LOCAL_ONLY",
            "Checkpoints and optimizer state are retained locally and represented by hashes only.",
        )
    if "cache" in rel.lower() or "embedding" in rel.lower():
        return (
            "cache_or_dense_embedding",
            "DERIVED_BULKY_ARTIFACT",
            "EXCLUDE_LOCAL_ONLY",
            "Regenerable cache or dense embedding; compact provenance is sufficient.",
        )
    if suffix in {".h5", ".hdf5", ".npy", ".npz"}:
        return (
            "scientific_tensor_or_generated_observation",
            "POTENTIALLY_PROTECTED_OR_DERIVED",
            "EXCLUDE_LOCAL_ONLY",
            "Dense scientific tensors remain local; commit compact tables, reports, and manifests.",
        )
    if suffix in {".jsonl", ".log", ".out"} or "/logs/" in rel:
        return (
            "log_or_access_trace",
            "MAY_CONTAIN_MACHINE_PATHS",
            "EXCLUDE_LOCAL_ONLY",
            "Bulky operational logs are represented by compact validation summaries.",
        )
    if suffix in {".png", ".jpg", ".jpeg", ".pdf"}:
        return (
            "large_figure",
            "PUBLICATION_REVIEW_REQUIRED",
            "EXCLUDE_UNLESS_EXPLICITLY_JUSTIFIED",
            "Only selected compact central figures belong in the curated archive.",
        )
    return (
        "other_large_artifact",
        "REVIEW_REQUIRED",
        "EXCLUDE_UNLESS_EXPLICITLY_JUSTIFIED",
        "Large artifact has no automatic scientific justification for Git inclusion.",
    )


def build_large_file_audit() -> None:
    paths = [path for path in research_files() if path.stat().st_size > LARGE_THRESHOLD]
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    ignored = ignored_paths(relative)
    hashes: dict[str, str] = {}
    duplicate_members: dict[str, list[str]] = defaultdict(list)
    for path, rel in zip(paths, relative, strict=True):
        digest = sha256(path)
        hashes[rel] = digest
        duplicate_members[digest].append(rel)
    duplicate_ids = {
        digest: f"DUP-{index:03d}"
        for index, digest in enumerate(
            sorted(digest for digest, members in duplicate_members.items() if len(members) > 1),
            start=1,
        )
    }
    output = ARCHIVE / "large_file_audit.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "size_bytes",
            "size_mb_decimal",
            "sha256",
            "category",
            "protected_status",
            "git_ignored",
            "above_50_mb",
            "above_100_mb",
            "duplicate_group",
            "commit_disposition",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, rel in sorted(zip(paths, relative, strict=True), key=lambda item: (-item[0].stat().st_size, item[1])):
            size = path.stat().st_size
            category, protected, disposition, reason = classify_large(path)
            writer.writerow(
                {
                    "path": rel,
                    "size_bytes": size,
                    "size_mb_decimal": f"{size / 1_000_000:.6f}",
                    "sha256": hashes[rel],
                    "category": category,
                    "protected_status": protected,
                    "git_ignored": "yes" if rel in ignored else "no",
                    "above_50_mb": "yes" if size > FIFTY_MB else "no",
                    "above_100_mb": "yes" if size > HUNDRED_MB else "no",
                    "duplicate_group": duplicate_ids.get(hashes[rel], ""),
                    "commit_disposition": disposition,
                    "reason": reason,
                }
            )


def best_path(run: Path, patterns: tuple[str, ...]) -> str:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(run.rglob(pattern))
    candidates = [
        path
        for path in candidates
        if path.is_file()
        and "corruption_cases" not in path.parts
        and "access_guard" not in path.parts
        and "__pycache__" not in path.parts
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: (len(path.relative_to(run).parts), path.as_posix()))
    return candidates[0].relative_to(ROOT).as_posix()


def compact_paths(run: Path, suffix: str, tokens: tuple[str, ...], limit: int = 3) -> str:
    candidates = [
        path
        for path in run.rglob(f"*{suffix}")
        if path.is_file()
        and "corruption_cases" not in path.parts
        and any(token in path.name.lower() for token in tokens)
    ]
    candidates.sort(key=lambda path: (len(path.relative_to(run).parts), path.as_posix()))
    return ";".join(path.relative_to(ROOT).as_posix() for path in candidates[:limit])


def extract_outcome(report_relative: str) -> str:
    if not report_relative:
        return "No final scientific report found; treat as engineering, incomplete, or superseded evidence."
    text = (ROOT / report_relative).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    section_start: int | None = None
    for index, line in enumerate(lines[:80]):
        if re.match(r"^##\s+(Decision|Outcome|Campaign outcome|Status)", line, flags=re.IGNORECASE):
            section_start = index + 1
            break
    if section_start is None:
        return "Outcome not safely extracted; see the final report and authority matrix."
    paragraph: list[str] = []
    for line in lines[section_start : section_start + 30]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#") and paragraph:
            break
        if stripped.startswith("|") or stripped.startswith("-"):
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    result = re.sub(r"[*_`]", "", " ".join(paragraph))
    return result[:500] if result else "See final report for outcome."


def campaign_type(name: str, has_report: bool) -> str:
    lower = name.lower()
    if "correction" in lower:
        return "correction"
    if any(token in lower for token in ("preflight", "readiness", "capsule", "contract", "integrity", "audit", "manifest", "foundation", "preparation")):
        return "engineering" if not has_report else "preflight_or_engineering"
    if any(token in lower for token in ("identifiability", "photometry", "stratification", "loss_geometry", "conditioning", "parameterization", "feasibility", "hypotheses", "probabilistic", "flow_prior", "family_e", "recoverability", "residual", "stress", "unet", "decoder")):
        return "scientific"
    return "scientific_or_engineering"


def curated_run_roots() -> set[str]:
    """Return run roots explicitly represented by archive provenance rows."""
    roots: set[str] = set()
    archive_root = ROOT / "docs" / "experiment_archive"
    if not archive_root.is_dir():
        return roots
    for provenance in archive_root.rglob("SOURCE_PROVENANCE.md"):
        for line in provenance.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\| `([^`]+)` \|", line)
            if not match:
                continue
            parts = Path(match.group(1)).parts
            if len(parts) >= 3 and parts[:2] == ("outputs", "runs"):
                roots.add(Path(*parts[:3]).as_posix())
    return roots


def build_experiment_ledger() -> None:
    runs = sorted(path for path in RUNS.iterdir() if path.is_dir())
    curated_roots = curated_run_roots()
    names_by_stem: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        stem = re.sub(r"_\d{8}(?:_\d{6})?$", "", run.name)
        names_by_stem[stem].append(run.name)
    output = ARCHIVE / "experiment_ledger.csv"
    fields = [
        "run_directory",
        "campaign_name",
        "timestamp",
        "campaign_type",
        "predecessor",
        "hypothesis",
        "exact_authorized_change",
        "data_accessed",
        "protected_data_status",
        "model_or_solver",
        "scene_or_sample_count",
        "number_of_starts",
        "compute_budget",
        "metrics",
        "final_outcome",
        "expected_or_surprising",
        "primary_limitation",
        "downstream_consequence",
        "authority_status",
        "supersession_status",
        "final_report_path",
        "protocol_path",
        "manifest_path",
        "summary_table_paths",
        "key_figure_path",
        "commit_artifact",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            match = re.search(r"(\d{8})(?:_(\d{6}))?$", run.name)
            timestamp = ""
            if match:
                timestamp = match.group(1)
                if match.group(2):
                    timestamp += "T" + match.group(2)
            report = best_path(run, ("final_report.md", "*final*report*.md", "*audit_report.md"))
            protocol = best_path(run, ("frozen_protocol.md", "frozen_*protocol.md", "*protocol*.md", "*preregistration*.json"))
            manifest = best_path(run, ("final_manifest.json", "manifest.json", "*manifest*.json", "*manifest*.csv"))
            tables = compact_paths(run, ".csv", ("summary", "result", "comparison", "metric", "decision"))
            figure = best_path(run, ("*summary*.png", "*comparison*.png", "*ranking*.png", "*.png"))
            stem = re.sub(r"_\d{8}(?:_\d{6})?$", "", run.name)
            siblings = names_by_stem[stem]
            later_sibling = siblings[-1] != run.name
            authority = "REPORT_PRESENT_REVIEW_AUTHORITY_MATRIX" if report else "ENGINEERING_OR_INCOMPLETE_RECORD"
            supersession = "LATER_SAME_STEM_EXISTS_REVIEW_SUPERSESSION_LEDGER" if later_sibling else "REVIEW_SUPERSESSION_LEDGER"
            if "correction" in stem:
                supersession = "CORRECTION_NAMED_REVIEW_SCOPE_IN_SUPERSESSION_LEDGER"
            record = {
                    "run_directory": run.relative_to(ROOT).as_posix(),
                    "campaign_name": stem,
                    "timestamp": timestamp,
                    "campaign_type": campaign_type(run.name, bool(report)),
                    "predecessor": "See supersession ledger; not inferred from directory ordering.",
                    "hypothesis": "See frozen protocol and final report." if report else "Not recoverable from a compact final authority.",
                    "exact_authorized_change": "See frozen protocol." if protocol else "Not recorded in a compact frozen protocol.",
                    "data_accessed": "See input provenance/access audit in the run tree.",
                    "protected_data_status": "Verify from authority; raw and protected tensors remain local-only.",
                    "model_or_solver": "See final report/source implementation.",
                    "scene_or_sample_count": "See final report/manifest.",
                    "number_of_starts": "See final report/manifest.",
                    "compute_budget": "See protocol/manifest.",
                    "metrics": "See compact result tables and final report.",
                    "final_outcome": extract_outcome(report),
                    "expected_or_surprising": "Not retrospectively inferred.",
                    "primary_limitation": "See final report; no missing value is fabricated.",
                    "downstream_consequence": "See canonical timeline and supersession ledger.",
                    "authority_status": authority,
                    "supersession_status": supersession,
                    "final_report_path": report,
                    "protocol_path": protocol,
                    "manifest_path": manifest,
                    "summary_table_paths": tables,
                    "key_figure_path": figure,
                    "commit_artifact": "curated_compact_copy" if run.relative_to(ROOT).as_posix() in curated_roots else "ledger_only",
                    "notes": "Metadata locator row. Repeated launch attempts remain separate; scientific authority, predecessor, and causal relations are resolved only in the canonical map and supersession ledger.",
                }
            grouped = grouped_run_override(run.name)
            if grouped:
                record.update(grouped)
                record["notes"] = "Grouped engineering-history metadata; this launch did not produce a distinct authoritative scientific result."
            override = RUN_OVERRIDES.get(run.name)
            if override:
                record.update(override)
                record["notes"] = "Scientifically enriched from the final report, manifest, frozen protocol, and authority audit; missing values remain NOT_RECORDED."
            writer.writerow(record)
        for record in NON_RUN_CAMPAIGN_RECORDS:
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-files", action="store_true")
    parser.add_argument("--experiment-ledger", action="store_true")
    args = parser.parse_args()
    if not args.large_files and not args.experiment_ledger:
        args.large_files = args.experiment_ledger = True
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if args.large_files:
        build_large_file_audit()
    if args.experiment_ledger:
        build_experiment_ledger()


if __name__ == "__main__":
    main()
