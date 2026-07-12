# Current project status

Phase I promptability is complete and frozen. Coordinate prompting removed the
centered-target shortcut, halved mean requested-source MSE relative to the
randomized unprompted control, and achieved 98% prompt-swap success. It did not
learn no-source semantics: empty prompts hallucinated under the declared
criterion in 100% of cases.

Phase II recoverability and selective abstention completed under the frozen,
append-only protocol with **PARTIAL SUCCESS**. R0 and R1 completed 20 MPS
epochs; R1 added 4,277 parameters, kept log variance inside `[-8, 2]`, and used
isotonic calibration selected by calibration-only cross-validation. Calibration
AUROC/AUPRC were `0.8746`/`0.2475`; Brier improved from `0.1010` raw to `0.0456`
calibrated. All-query permissive-contract risk declined from `0.9560` at full
coverage to `0.9379` at 70%, but catastrophic failure increased at 80% and 70%
coverage, and ambiguous queries received a higher mean score (`0.1017`) than
clear valid queries (`0.0614`). R1 null hallucination was `8.25%`, versus
`7.5%` for frozen Phase-I C and `2.25%` for R0 on identical new null-coordinate
scenes. This does not meet the success gates.

The source split and lockbox policy are unchanged; the lockbox remained sealed.
Current evidence is controlled BTK development evidence. DR10 is still a
real-sky out-of-distribution benchmark and is not evidence of calibrated survey
performance. The next authorized experiment is two fixed-protocol R1 training-
seed replications; a full Ambiguity Atlas and lockbox evaluation are not
authorized.

## Superseding status — frozen-representation diagnostic (2026-07-11)

The fixed-protocol seed replication and root-cause analysis are complete. The
subsequent frozen-head ablation is recorded in
`outputs/runs/thayer_select_frozen_head_ablation_20260711_220756/` and is
classified **NO CLEAR IMPROVEMENT**. Balanced heads improved validation AUPRC,
but the validation-selected H2 head degraded to calibration AUROC/AUPRC
0.514/0.032, kept ambiguous prompts above valid prompts, and did not reject
catastrophic source failures reliably. Temperature scaling retained more
resolution than isotonic but still produced 100% realized coverage at several
nominal operating points because the selected MLP scores saturated.

The encoder remained fully frozen, no reconstruction backbone was retrained,
development was not evaluated, and the lockbox remained sealed. The only next
recommended experiment is a preregistered redesign of the moderate
reliability-contract target and its failure-specific labels.

## Superseding status — hierarchical safety policy (2026-07-11)

The hierarchical campaign is complete in
`outputs/runs/thayer_select_hierarchical_safety_20260711_225657/` and is
classified **FAILURE under the frozen gates**. This classification does not
erase successful subcomponents: the five-seed UNIQUE_VALID/NULL/AMBIGUOUS gate
reached balanced validation macro F1/AUPRC `0.881`/`0.923`, recalled NULL at
`99.85%` and AMBIGUOUS at `88.89%`, and removed ambiguity-over-valid inversion
in every seed. Valid-only image, flux, and centroid heads also ranked the tail,
and split conformal upper bounds reached approximately 90% natural-calibration
coverage.

The complete policy was not operational. It accepted only 1/4,200 natural-
calibration valid scenes, 0/1,000 stratified-diagnostic valid scenes, and
1/2,000 fresh development valid scenes. On development, query-gate-only false
acceptance was `0%` for NULL and `9.2%` for AMBIGUOUS, but the full policy's
near-zero coverage made its zero invalid acceptance unusable. At 95%, 90%,
80%, and 70% diagnostic valid coverage, hierarchical catastrophic rates were
`0.825`, `0.816`, `0.793`, and `0.764`; these were not materially better than
the historical R1 ranking.

Condition C remained byte-identical and fully frozen. The fresh 3,000-scene
development manifest was generated after policy freeze and evaluated exactly
once. The lockbox remained sealed. Recoverability is now documented as a
derived policy, not a monolithic training label, but that policy is not ready
for deployment or lockbox evaluation.

## Protocol correction — 2026-07-12

An append-only compliance audit found that the historical hierarchical run did
not contain the required pre-fit preregistration or full original-contract
postmortem. Those omissions cannot be repaired retrospectively after its
one-time development evaluation. The exact persisted moderate composite was
subsequently reconstructed for all 13,500 Phase-II rows with zero Boolean
reapplication mismatches, but it confirmed a heterogeneous target and a
training/validation-versus-calibration reconstruction-provenance change. No new
inference or training was performed. The historical complete-policy result
remains FAILURE, and a new prospectively preregistered train/validation/calibration-only
feasibility campaign is required before another development set is authorized.

## Prospective feasibility status — 2026-07-12

The prospective train/validation/calibration-only run
`outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729/` is
complete with **PARTIAL SUCCESS**. Its preregistration was hashed before every
fit, Condition C was the only reconstructor across all 32,000 scenes, and the
applicability audit found zero prior-style logical defects. Query validity and
all continuous valid risks were strongly learnable; score calibration remained
noncollapsed.

The frozen catastrophic AUPRC gate was impossible at the observed 81.65%
prevalence, and image/flux conditional calibration remained uneven despite
90% marginal coverage. Those gates were not changed. No full policy or
development manifest was created, the lockbox remained sealed, and the
historical hierarchy remains FAILURE. A future full-policy campaign is not yet
authorized.
