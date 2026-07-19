# Project roadmap

The canonical scientific history is in the
[complete research program map](thayer_research_program_map.md). This roadmap
contains only current decisions. The long pre-curation roadmap is preserved
unchanged in
[`project_roadmap_pre_canonical_archive.md`](research_archive/project_roadmap_pre_canonical_archive.md).

## Complete — Phase I mechanistic map

- Reconciled the early neural, prompting, recoverability, ambiguity, model-
  family, loss-geometry, output-contract, D0–D3, and audit branches.
- Proved unrestricted additive non-identifiability in the frozen direct-output
  contract.
- Mapped conditional oracle, flux-free S1, same-PSF S2, PSF-diverse P2, and
  external-photometry interventions across the selected eight scenes.
- Classified PRE as useful but not fully validated, identifiability as the
  strongest functioning audit, and POST as blocked by label collapse.
- Curated compact authorities while keeping raw/protected/bulky artifacts
  local-only.

## Next — Phase II independent validation

Follow the frozen
[Phase-II validation plan](research_archive/phase_ii_validation_plan.md).

1. Create a fresh, group-disjoint scene sample excluding Scenes 0, 3, 5, 6,
   18, 51, 73, and 81.
2. Freeze the structural family, total-photometry uncertainty, start count,
   optimizer budget, endpoint clustering, diameters, and decision thresholds.
3. Compare single-observation flux-free S1, total external photometry, and P2
   with matched starts and budgets.
4. Test the prespecified photometry-benefit hypothesis and low-`|ΔB/T|`
   stratum without post-hoc threshold tuning.
5. Report exact confidence intervals, multiplicity-adjusted inference, failed
   optimizations, out-of-support cases, and all abstentions.

Minimal, moderate, and publication-grade sample sizes and compute-reduction
rules are specified in the plan. The existing eight scenes may inform runtime,
metric freezing, and stratum definitions only; they are not validation data.

## After independent validation — robustness

Only if the primary Phase-II contract remains scientifically useful:

- perturb prompt coordinates;
- introduce PSF estimation error;
- test noise misspecification and correlation;
- test structural/morphology misspecification;
- audit an independent implementation of the structured solver and
  identifiability logic;
- evaluate direct reconstruction accuracy for independently unique targets.

## Later — operational and real-data closure

- Replace truth-derived routing features with validated observable quantities.
- Measure false authorization, unnecessary acquisition, and abstention costs
  for `RECONSTRUCT`, `ACQUIRE PHOTOMETRY`, and `DON’T EVEN TRY`.
- Revisit POST only after an eligible reconstructor supplies a nondegenerate
  independently verified safe/unsafe population.
- Perform survey-specific unit, PSF, morphology, catalog, and source-only
  closure before any real-data performance or safety claim.

## Deferred / unresolved

- Full D3 capacity and eight-scene ladders.
- Larger population-prevalence estimation.
- Operational external-photometry acquisition policy.
- Real-survey deployment.

These branches remain `UNKNOWN`, `EXPLORATORY`, or `BLOCKED`; absence of a run
is not a negative result.
