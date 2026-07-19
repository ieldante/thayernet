# Claim authority matrix

## Authority rule

Claims are resolved in this order: final scientific report; final manifest;
frozen protocol or preregistration; superseding correction report; compact
result table; source and tests; current-status documentation; planning notes.
An append-only correction may supersede only the field it explicitly corrects.
It does not erase the historical report or silently widen the claim.

## Canonical authority chain

| Topic | Current authority | Historical or subordinate evidence | Canonical interpretation |
| --- | --- | --- | --- |
| Grouped neural benchmark | [`grouped_correctness/final_report.md`](../experiment_archive/grouped_correctness/final_report.md) | Earlier random-row results and historical checkpoints | Group-disjoint development benchmark only; no final-paper or real-survey claim. |
| Coordinate promptability | [`coordinate_prompting/final_report.md`](../experiment_archive/coordinate_prompting/final_report.md) | No-prompt and randomized-prompt controls | Condition C achieved 98.0% prompt-swap success in its frozen development contract. This demonstrates promptability, not correct source allocation. |
| Recoverability-score policy | [`recoverability_head/final_report.md`](../experiment_archive/recoverability_head/final_report.md) | Seed replication and later hierarchy work | Partial historical result. AUROC and risk ranking did not establish safe selective reconstruction. |
| Hierarchical safety | Corrective report in [`hierarchical_safety/superseding_correction.md`](../experiment_archive/hierarchical_safety/superseding_correction.md) read with the original report | Original hierarchy final report | Failed, non-deployable historical policy; retrospective preregistration was refused. |
| Ambiguity witnesses | [`ambiguity_atlas/final_report.md`](../experiment_archive/ambiguity_atlas/final_report.md) | Provisional Atlas counts | Constructed witnesses worked, but the operational witness detector failed its gate. The corrected final witness count governs. |
| Stochastic PU family | [`probabilistic_unet/final_report.md`](../experiment_archive/probabilistic_unet/final_report.md) | Atlas stochastic tables | Stochastic diameter and discrimination improved, but own- and alternate-truth coverage were zero. |
| PU deployment eligibility | [`pu_batch_correction/final_report.md`](../experiment_archive/pu_batch_correction/final_report.md) | Earlier single-versus-batch ineligibility report | Corrected executor made PU eligible, but all 7,591 complete outputs were unsafe: eligible-but-label-collapsed, not deployable. |
| Loss geometry | [`loss_geometry/final_report.md`](../experiment_archive/loss_geometry/final_report.md) | ME micro-overfit outputs | Exact truths were representable, while the full objective often preferred compromise and could destroy truth coverage. |
| D0–D3 | [`fixed_feature_ladder/final_report.md`](../experiment_archive/fixed_feature_ladder/final_report.md) and [`d3_pv1a1/final_report.md`](../experiment_archive/d3_pv1a1/final_report.md) | Multiple fail-closed engineering attempts | D0 and D1 passed and D2 failed in the fixed-feature ladder. The later local L0 trajectory ended at budget exhaustion and mapped to `MIXED_CAUSE`; unrun eight-scene and capacity-ladder branches remain `UNKNOWN`. |
| Physical source contract | [`signed_residual_preflight/final_report.md`](../experiment_archive/signed_residual_preflight/final_report.md) | [`family_e_invalid/final_report.md`](../experiment_archive/family_e_invalid/final_report.md) | Two nonnegative source layers require a signed residual/noise layer to conserve a signed noisy observation. This fixes feasibility, not identity or recovery. |
| Family-E1 prompt diagnosis | [`family_e1p/final_report.md`](../experiment_archive/family_e1p/final_report.md) | Family-E1 micro result | Generic prompt effects survived, but source-identity-aligned prompt contrast was too weak: prompt-swap 0/1 difficult and 1/8 mixed. |
| Unrestricted source allocation | [`recoverability_nullspace/final_report.md`](../experiment_archive/recoverability_nullspace/final_report.md) | Family-E1P and fixed-feature results | Every frozen scene has an exact 10,800-dimensional two-source allocation null space under the unrestricted output contract. |
| Morphology plus oracle source flux | [`oracle_flux_identifiability/final_report.md`](../experiment_archive/oracle_flux_identifiability/final_report.md) | Oracle metrics table | Conditional result only: 7/8 unique when exact per-source g/r/z flux and truth-derived signed-noise information are supplied. |
| Flux-free morphology | [`flux_free_identifiability/final_report.md`](../experiment_archive/flux_free_identifiability/final_report.md) | The first invalid flux-free launch is engineering history only | 0/8 strict unique across the frozen Level-4/5 union. Local full rank in individual fits does not overturn global multiplicity or strict-gate failures. |
| PSF-diverse paired observation | [`psf_diverse_identifiability/final_report.md`](../experiment_archive/psf_diverse_identifiability/final_report.md) | S1 and same-PSF S2 controls | 0/8 unique; 15/16 P2 family fits passed the composite information/geometry improvement rule. Only 3/16 were attributed specifically to PSF diversity after the S2 control. |
| External total photometry | [`scene_stratification_correction/final_report.md`](../experiment_archive/scene_stratification_correction/final_report.md) | Preflight, 500-evaluation correction, and six-scene stratification | Helpful relative to P2 for Scenes 0, 5, 51, 73; not helpful for 3, 6, 18, 81. Descriptive rate 4/8, exact 95% CI 15.7%–84.3%. |
| Low-|ΔB/T| rule | Corrected feature table in [`scene_stratification_correction/summary_table.csv`](../experiment_archive/scene_stratification_correction/summary_table.csv) plus the final report | Six-scene exploratory rule | Exploratory only: AUC 1.0, exact p=0.0286, BH q=0.4286, leave-one-out balanced accuracy 0.75, n=8. The feature is truth-derived/post-fit and is not yet a deployable acquisition input. |
| PRE audit | [`direct_audit/final_report.md`](../experiment_archive/direct_audit/final_report.md) | Query-validity and ambiguity classifiers | Useful research component but missed the frozen formal macro-F1 gate. Not a validated operational gate. |
| Identifiability audit | Flux-free, P2, and photometry authorities above | Rank, singular values, multi-start endpoints, replay and perturbations | Strongest functioning audit layer within the frozen simulation contract. It supports abstention/research decisions, not a survey deployment claim. |
| POST audit | [`direct_audit/superseding_correction.md`](../experiment_archive/direct_audit/superseding_correction.md) and PU batch correction | Saturated all-unsafe labels | Not operational. Zero accepted coverage and an all-unsafe reconstruction population provide no meaningful safe-positive class. |

## Claim wording controls

| Class | Allowed wording | Disallowed wording |
| --- | --- | --- |
| Frozen-contract proof | “Within the frozen eight-scene simulated contract…” | “Galaxy deblending is impossible.” |
| Cross-contract transition | “The oracle-information contract was 7/8 unique; the later nonoracle flux-free contract was 0/8 strict unique.” The comparison must note differences in renderer/noise handling and diameter gates, so it is not a clean one-variable causal ablation. | “Removing photometry alone caused the 7/8 to 0/8 collapse,” or “Morphology never works.” |
| PSF result | “P2 improved the preregistered composite geometry metric in 15/16 fits but restored uniqueness in 0/8 scenes.” | “PSF diversity is useless,” or “PSF diversity improved 15/16 fits” without the S2-control qualification. |
| Photometry result | “Total source photometry was helpful relative to P2 in four of eight discovery scenes.” | “Photometry will solve half of survey blends.” |
| Stratification | “Low |ΔB/T| is a candidate explanatory rule requiring independent validation.” | “Low |ΔB/T| is a validated acquisition policy.” |
| Audit policy | “The current identifiability audit can fail closed in the frozen research setting.” | “The deployed system safely decides when to reconstruct.” |

## Missing authority gaps

1. No executed independent-scene validation report exists.
2. No population-prevalence estimate exists beyond the selected eight-scene
   mechanistic set.
3. No real-survey closure report exists for the structured solver or audit.
4. No validated operational proxy replaces truth-derived `|ΔB/T|` for deciding
   whether to acquire photometry.
5. No direct reconstruction-accuracy campaign has yet established an accurate
   source estimate under a target proven unique without oracle source flux.
6. No POST classifier has a nondegenerate safe/unsafe label population.
7. The full D3 capacity ladder and eight-scene D3 branches were not executed;
   their status remains unknown rather than negative.
