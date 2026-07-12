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
