# Phase II independent validation plan

**Status: planned, not executed.** This document freezes a defensible design;
it does not authorize compute or data access.

## Frozen hypotheses

Primary mechanistic hypothesis:

> On independent CatSim/BTK scenes, the discovery threshold
> `|ΔB/T| <= 0.004913344229132974` is associated with the frozen total-source-
> photometry helpful label relative to P2.

The rule is explanatory, not operational: `ΔB/T` uses simulator/catalog truth
and is unavailable as a direct real-data acquisition feature.

Key secondary hypotheses:

1. Total source photometry reduces endpoint multiplicity or all nonzero
   scientific diameters relative to P2 in a nonzero subset of scenes.
2. Photometry's global effect is not reducible to condition-number change.
3. P2 often improves local information geometry relative to S1 without
   restoring strict uniqueness.
4. The helpful label and strict-uniqueness label remain distinct.

## Independent sample

- Exclude Phase I scenes `0, 3, 5, 6, 18, 51, 73, 81`.
- Exclude their exact source rows, source groups, pixel-duplicate groups,
  exact-coordinate groups, and any pair/campaign derivative.
- Freeze candidate rows before rendering and record source/catalog hashes.
- Do not select scenes using the Phase I outcome model after labels are known.
- For fixed-rule accuracy, use a prospectively balanced sample across the
  frozen low/high `|ΔB/T|` strata.
- For population helpfulness, use a separate random sample or report sampling-
  weighted estimates; a balanced/enriched cohort is not a prevalence sample.

## Sample-size tiers

| Tier | Independent scenes | Allocation | Purpose |
| --- | ---: | --- | --- |
| Minimal mechanistic | 32 | Ideally 16 below and 16 above the frozen threshold | Feasibility, fixed-rule confusion matrix, and wide exact intervals; not publication-grade prevalence |
| Moderate | 64 | Approximately 32/32 | Main confirmatory mechanistic test; roughly 99% exact-binomial planning power for 0.75 versus 0.50 fixed-rule accuracy, ignoring clustering and exclusions |
| Publication mechanistic | 128 | 64/64 | Stable stratum sensitivity/specificity and interaction estimates |
| Population precision extension | Approximately 384 random scenes | Natural prevalence | Approximate ±5 percentage-point 95% margin near a 50% helpful rate; compute and clustering may require more |

The power statements are planning approximations. The final sample calculation
must be frozen before access and should account for source-group clustering,
unresolved fits, exclusions, and the chosen exact primary test.

## Conditions and structural family

All scenes receive three paired conditions using the same source rows,
coordinates, first observation, parameter bounds, and decision logic:

1. **S1 baseline:** one noisy observation, source flux free.
2. **P2 comparator:** S1 plus an independently seeded exposure with the frozen
   PSF-diverse acquisition.
3. **External total photometry:** S1 plus one noisy total g+r+z measurement per
   source with frozen 5% relative uncertainty.

The primary structural family is the validated Level-5 bulge+disk Model-9
family because it is the family used by the external-photometry discovery
chain. Level 4 may be included only as a prespecified secondary family; it may
not be selected per scene from validation outcomes.

Per-band photometry is secondary. It is not a required primary condition
because the discovery campaign did not resolve total-versus-per-band behavior
for all eight scenes at the corrected budget.

## Frozen solver and global-search budget

- Use 16 deterministic, matched physical starts per scene and condition.
- Use the same start generator/order across S1, P2, and photometry.
- Freeze a common 500-evaluation first stage.
- If escalation is used, apply a symmetric, outcome-blind rule to all three
  conditions for that scene; the recommended ceiling is 2,000 evaluations.
- Do not stop a condition merely because a one-class endpoint has appeared.
- Preserve all endpoints, optimizer statuses, gradients, boundary contacts,
  and replay hashes.
- Mark a scene `UNRESOLVED` if the frozen convergence rule is not met. In the
  intent-to-audit primary analysis, unresolved scenes cannot authorize
  reconstruction and do not silently disappear. An interpretable-only
  sensitivity analysis may be reported separately.

This matched budget corrects the Phase I asymmetry: S1/P2 used 16 starts per
scene/family while external photometry used four.

## Frozen metrics and classifications

Use the Model-9 definitions without post-hoc changes:

- local rank, nullity, smallest singular value, and condition;
- accepted endpoint count and endpoint-class count;
- requested/companion image diameter;
- morphology diameter;
- flux-allocation diameter;
- prompt-identity stability, source collapse, gradient and support gates;
- `UNIQUE`, `NEAR_UNIQUE`, `PARTIALLY_IDENTIFIABLE`,
  `NON_IDENTIFIABLE`, `OUT_OF_SUPPORT`, `UNRESOLVED`.

Strict `UNIQUE` requires one class, full active rank/nullity zero, condition
`<= 1e6`, gradient `<= 1e-5`, all frozen Model-9 diameters `<= 1e-3`, stable
prompt identity, and no zero-source collapse.

The frozen **Photometry Helpful** label is relative to P2:

- fewer endpoint classes; or
- no diameter worsening and at least 50% reduction of every nonzero P2
  diameter.

It does not mean strict uniqueness, better than S1, or accurate source
reconstruction. Report those questions separately.

## Primary endpoint and statistical test

Primary endpoint: association between the frozen binary low/high `|ΔB/T|`
rule and the binary frozen total-photometry-helpful label under matched search
effort.

Primary analysis:

1. Publish the 2×2 confusion matrix.
2. Report balanced accuracy, sensitivity, specificity, positive/negative
   predictive value, and exact or source-group bootstrap 95% intervals.
3. Use one two-sided exact Fisher/conditional permutation test for the frozen
   association.

No threshold is estimated from Phase II.

## Secondary endpoints

- Descriptive helpful rate with exact Clopper–Pearson interval in a random or
  sampling-weighted cohort.
- Strict-uniqueness rates under S1, P2, and photometry.
- Paired endpoint-class change and paired diameter change.
- S1→P2 and P2→photometry condition-number and minimum-singular-value changes.
- Local/global discordance rate.
- Convergence/unresolved rate by condition and stratum.
- Photometry helpfulness relative to S1 as an operationally relevant secondary
  question.
- If Level 4 or per-band photometry is included, its result is secondary.

Use paired permutation or Wilcoxon tests for prespecified continuous paired
changes and McNemar/exact paired tests for binary transitions where applicable.
Use Holm correction across confirmatory secondary hypotheses. Use BH only for
explicitly exploratory feature ranking.

## Confidence intervals and clustering

- Exact binomial intervals for simple proportions.
- Source-group bootstrap intervals when rows share catalog/source ancestry.
- Report both scene-level and source-group-effective sample sizes.
- Do not use naive row-level intervals if source groups repeat.
- Preserve unresolved outcomes in denominators for intent-to-audit policy
  summaries.

## Failure and stop criteria

The primary mechanistic hypothesis fails if the frozen association test does
not reject at the prespecified alpha, or if the confidence interval is
incompatible with useful discrimination. The operational interpretation fails
regardless of statistical association if:

- the feature cannot be estimated from allowed observables;
- helpful labels are unstable under matched starts/budgets;
- unresolved fits exceed the frozen tolerance;
- false reconstruction authorization is nonzero beyond the frozen safety
  bound;
- external information does not yield strict uniqueness plus accurate
  reconstruction where `RECONSTRUCT` is proposed.

Stop fail-closed on protected-data exposure, manifest overlap with discovery
sources, failed replay/hash checks, an unapproved contract change, or a
material scientific implementation error.

## Runtime estimate

Archived timing anchors:

- 16 endpoints at 150 evaluations: 562.87 s, about 35.2 s/endpoint;
- 16 endpoints at 500 evaluations: 854.86 s, about 53.4 s/endpoint;
- eight hard endpoints at 2,000 evaluations: 4,357.36 s, about 544.7
  s/endpoint.

With three conditions and 16 starts, each scene requires 48 fits.

| Tier | Fits | Serial time at 53.4 s/fit | Serial hard-case bound at 544.7 s/fit |
| --- | ---: | ---: | ---: |
| 32 scenes | 1,536 | ~22.8 h | ~232 h |
| 64 scenes | 3,072 | ~45.6 h | ~465 h |
| 128 scenes | 6,144 | ~91.1 h | ~930 h |
| 384 scenes | 18,432 | ~273 h | ~2,789 h |

These are rough compute anchors, not promises. They exclude data generation,
validation, I/O, and retries. Parallel execution may reduce wall time only if
start identity, deterministic ordering, atomic outputs, and resource isolation
are preserved.

## Compute-reduction strategies

- Cache deterministic rendering, PSFs, coordinates, sigma maps, and initial
  parameter vectors by hash.
- Parallelize starts/scenes with fixed seeds and one atomic record per fit.
- Use the symmetric 500→2,000 escalation rule rather than granting more search
  only to an apparently favorable condition.
- Pilot runtime on non-analysis scenes without viewing scientific outcomes.
- Precompute Jacobian sparsity/analytic derivatives only after exact numerical
  equivalence tests.
- Do not reduce the number of starts based on discovered endpoint counts.

## No post-hoc tuning

After independent-scene access, do not change the `|ΔB/T|` threshold,
uncertainty, family, PSFs, starts, budgets, endpoint tolerances, clustering,
diameters, support/gradient gates, escalation logic, or primary test. Any
necessary correction must be append-only, explain its scope, preserve the
original outcome, and be evaluated as a new protocol—not folded into the
confirmatory result.

## Future decision-policy evaluation

- `RECONSTRUCT`: require inference-legal strict uniqueness and separately
  validated source accuracy.
- `ACQUIRE PHOTOMETRY`: require an observable acquisition feature plus a
  validated beneficial transition.
- `DON'T EVEN TRY`: assign non-identifiable, out-of-support, invalid, unstable,
  or unresolved requests.

POST remains future work until reconstructions produce a nondegenerate safe and
unsafe label population.
