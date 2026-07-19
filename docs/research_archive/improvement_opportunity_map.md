# Improvement opportunity map

Scores use `1` (low) to `5` (high). Compute and engineering are costs; risk is
the risk of an uninterpretable or misleading result. “Phase II” means the work
belongs in the independent validation phase, not that it is already authorized
for execution by this archive task.

| Rank | Opportunity | Scientific value | Evidence gained | Compute cost | Engineering cost | Risk | Poster necessity | Paper necessity | Phase II status | Why now / success condition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Independent-scene validation | 5 | 5 | 4 | 3 | 2 | 4 | 5 | Primary | Blinded scenes excluding all eight discovery sources; frozen rule, starts, budgets, and thresholds. Success is an interpretable estimate of helpfulness and fixed-rule accuracy. |
| 2 | Equal-start intervention reanalysis | 5 | 5 | 4 | 2 | 2 | 4 | 5 | Mandatory design correction | Use identical starts and outcome-blind escalation for S1, P2, and photometry. This removes the current 16-start-versus-4-start basin-search asymmetry. |
| 3 | Larger population validation | 5 | 5 | 5 | 3 | 2 | 2 | 5 | Publication extension | Random rather than stratum-balanced sampling for a prevalence estimate; approximately 384 scenes are needed for about ±5 percentage-point precision near 50%. |
| 4 | Structured reconstruction accuracy | 5 | 5 | 4 | 3 | 3 | 4 | 5 | Required before `RECONSTRUCT` | Establish truth accuracy only where the target is strict-unique under an inference-legal contract. Identifiability alone is not reconstruction accuracy. |
| 5 | Validated reconstruct/acquire/abstain policy | 5 | 5 | 5 | 4 | 4 | 5 | 5 | Operational Phase II/III | Measure false authorization, useful acquisition, and abstention. No current evidence authorizes `RECONSTRUCT`. |
| 6 | Observable acquisition feature | 5 | 4 | 3 | 4 | 4 | 4 | 5 | Required translational bridge | Replace truth-derived/post-fit `|ΔB/T|` with a prespecified observable estimator. Validate it without reusing discovery outcomes. |
| 7 | Morphology misspecification | 5 | 4 | 4 | 3 | 3 | 2 | 5 | Robustness | Perturb family/support away from the generating catalog and quantify false uniqueness/out-of-support rates. |
| 8 | Prompt-coordinate error | 4 | 4 | 3 | 2 | 2 | 3 | 4 | Robustness | Inject prespecified astrometric offsets and measure PRE, prompt identity, endpoint multiplicity, and policy transitions. |
| 9 | PSF error and broader PSF diversity | 4 | 4 | 4 | 3 | 3 | 3 | 4 | Robustness/comparator | Separate known PSF diversity from PSF misspecification; retain same-PSF S2 and equal budgets. |
| 10 | Noise misspecification | 4 | 4 | 3 | 2 | 3 | 2 | 4 | Robustness | Vary variance, correlation, background, and tails while preserving a frozen truth-use boundary. |
| 11 | Real-data closure | 5 | 5 | 5 | 5 | 5 | 3 | 5 | Later phase | Requires calibrated data, source/measurement contract, injection or external truth strategy, and no hidden simulator oracle. |
| 12 | Independent implementation audit | 4 | 5 | 3 | 4 | 2 | 1 | 5 | Parallel validation | Reimplement renderer/Jacobian, clustering, diameters, and decision logic from protocols and compare compact outputs/hashes. |
| 13 | Future POST audit | 4 | 4 | 4 | 4 | 5 | 2 | 4 | Blocked | Begin only after an eligible reconstructor supplies a nondegenerate safe/unsafe population. Require prospective family holdout. |
| 14 | Computational acceleration | 3 | 3 | 2 | 4 | 3 | 1 | 3 | Enabler | Cache rendering/PSFs, parallelize deterministic starts, and use a symmetric escalation rule without changing the objective or endpoint search depth. |
| 15 | Full D3 capacity ladder | 3 | 3 | 5 | 5 | 5 | 1 | 2 | Optional, not Phase II critical | R3 supports a local mixed cause, not general capacity. Revisit only with a complete noncontradictory protocol and a clear link to the scientific validation target. |

## Highest-priority design corrections

1. **Equalize search effort.** P2/S1 used 16 starts per scene/family, while
   external photometry used four. Because “helpful” depends partly on the
   number of endpoint classes discovered, Phase II must use matched starts and
   budgets or a symmetric, outcome-blind escalation rule.
2. **Treat unresolved fits as unresolved.** Scene 5's frozen helpful label is
   preserved, but only 1/4 corrected starts reported optimizer success and 3/4
   hit 2,000 evaluations. A one-class set found under unresolved starts is not
   a proof of global uniqueness.
3. **Separate explanatory and operational features.** `|ΔB/T|` uses
   simulator/catalog truth after fitting. It may be tested as a mechanistic
   hypothesis, but it cannot directly drive acquisition in deployed data.
4. **Separate S1, S2, and P2 questions.** The `15/16` composite P2-versus-S1
   result is not pure PSF causality. Same-PSF S2 is required, and an operational
   question must also ask whether photometry improves over S1—not only P2.
5. **Do not equate strict-gate failure with demonstrated multiplicity.** Six of
   eight flux-free scenes are best-family `NEAR_UNIQUE`: one class, nullity
   zero, often zero diameters, but a strict numerical gate fails. Preserve this
   distinction in future endpoints.

# What Could Be Proven Next?

## Immediate with existing data

Can establish or audit more tightly without new scientific execution:

- the exact frozen-contract outcomes and their hashes;
- the analytic unrestricted allocation null space;
- local/global divergence examples;
- the complete eight-scene Phase I intervention map;
- the cross-contract differences that prevent a pure “remove flux only” causal
  interpretation;
- start-count, convergence, and truth-derived-feature limitations;
- reproducibility of compact tables through an independent implementation.

Cannot establish with the existing eight scenes:

- population ambiguity or helpfulness prevalence;
- a universal acquisition rule;
- a deployable low-|ΔB/T| policy;
- real-survey accuracy;
- a safe `RECONSTRUCT` action;
- POST performance with a meaningful safe-positive class.

## Independent validation

A preregistered, blinded independent sample could prove or falsify:

- whether the exact low-|ΔB/T| threshold predicts the frozen helpful label;
- whether total-photometry benefit generalizes beyond the discovery scenes;
- the helpful-rate estimate, with an honest interval from a random sample;
- whether P2 gains remain mostly conditioning rather than uniqueness;
- whether an equal-budget global audit produces stable scene labels;
- the false-authorization and abstention behavior of a research policy.

## Robustness validation

Prespecified perturbation campaigns could establish stability under:

- coordinate error;
- PSF mismatch and broader PSF diversity;
- variance/background/noise-correlation misspecification;
- structural-family and support misspecification;
- unseen morphology and brightness strata.

The primary robustness endpoint should be a change in strict authorization or
endpoint multiplicity, not condition number alone.

## Operational validation

An operational study could eventually establish three actions with measured
error rates:

- **RECONSTRUCT:** strict inference-legal uniqueness *and* independently
  validated source accuracy;
- **ACQUIRE PHOTOMETRY:** a validated observable feature predicts a beneficial
  transition and the post-acquisition audit passes;
- **DON'T EVEN TRY:** non-identifiable, out-of-support, invalid, unstable, or
  unresolved cases fail closed.

For zero observed false reconstruction authorizations, about 59 independent
negative cases give a one-sided 95% upper bound near 5%; about 299 give a bound
near 1%. These are planning approximations, not achieved guarantees.
