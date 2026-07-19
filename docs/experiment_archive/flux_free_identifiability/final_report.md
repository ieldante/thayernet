# Thayer-Flux-Free-Identifiability-v0 final report

## Campaign outcome

**FLUX_FREE_UNIQUENESS_COLLAPSES**

The primary endpoint is **0/8** scenes `UNIQUE` under at least one
permitted flux-free structural family. Level 4 is unique for **0/8**
and Level 5 for **0/8**. The earlier **7/8** result does
 not survive at the declared 6/8 threshold
after isolated-truth per-source g/r/z fluxes are removed.

## Exact findings

The solver received only the frozen blended g/r/z observation, the requested
and companion coordinates from the frozen two-source coordinate contract, the
known normalized LSST g/r/z PSFs, 60x60 geometry at 0.2 arcsec/pixel, the
observation-only BTK Poisson plug-in sigma map, the family identifier, and the
pre-frozen Level-4/5 bounds. Individual requested and companion fluxes were
free nonnegative unbounded-above likelihood parameters.

The solver did **not** receive isolated source images or masks, true per-source
fluxes, the exact signed noise realization, true morphology parameters or
labels, B/T truth, catalog rows, truth initialization, protected-set data, or
outcome-dependent family selection. All 16 starts per scene/family are
retained; no failing run was discarded.

- Scene 0: L4 `NON_IDENTIFIABLE`; L5 `NEAR_UNIQUE`; minimum unique family `none`.
- Scene 3: L4 `PARTIALLY_IDENTIFIABLE`; L5 `NON_IDENTIFIABLE`; minimum unique family `none`.
- Scene 5: L4 `NON_IDENTIFIABLE`; L5 `NEAR_UNIQUE`; minimum unique family `none`.
- Scene 6: L4 `PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE`; minimum unique family `none`.
- Scene 18: L4 `NON_IDENTIFIABLE`; L5 `NEAR_UNIQUE`; minimum unique family `none`.
- Scene 51: L4 `PARTIALLY_IDENTIFIABLE`; L5 `PARTIALLY_IDENTIFIABLE`; minimum unique family `none`.
- Scene 73: L4 `PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE`; minimum unique family `none`.
- Scene 81: L4 `NEAR_UNIQUE`; L5 `PARTIALLY_IDENTIFIABLE`; minimum unique family `none`.

All fitted source layers are nonnegative, PSFs are normalized, residuals are
signed, and each solver invocation has a machine-readable eight-input
provenance trace. Deterministic renderer/Jacobian replay and the frozen
numerical perturbation check are reported per scene/family.

## Why non-unique scenes fail

- Scene 0: Level 4 NON_IDENTIFIABLE: several isolated tolerance-quality source solutions are present; Level 5 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails.
- Scene 3: Level 4 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold; Level 5 NON_IDENTIFIABLE: several isolated tolerance-quality source solutions are present.
- Scene 5: Level 4 NON_IDENTIFIABLE: several isolated tolerance-quality source solutions are present; Level 5 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails.
- Scene 6: Level 4 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold; Level 5 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails.
- Scene 18: Level 4 NON_IDENTIFIABLE: several isolated tolerance-quality source solutions are present; Level 5 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails.
- Scene 51: Level 4 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold; Level 5 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold.
- Scene 73: Level 4 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold; Level 5 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails.
- Scene 81: Level 4 NEAR_UNIQUE: one solution class remains but conditioning, gradient, identity, or a strict diameter rule fails; Level 5 PARTIALLY_IDENTIFIABLE: residual null/classes remain but requested-source and allocation diameters stay below the strict uniqueness threshold.

`OUT_OF_SUPPORT` means the admissible likelihood set is empty and is never
counted as uniqueness. Scene 51 remains the required stress case; its prior
oracle-flux evidence identified a sub-support unresolved bulge, and the
flux-free classification above does not reinterpret empty support as source
identifiability.

## Interpretation

The findings are conditional structural identifiability results for these
eight frozen simulated observations and the declared exact-coordinate,
known-PSF, fixed-noise-convention contract. They are not observation-only
unrestricted-output uniqueness and do not establish generalization to real
survey data.

The present evidence does not yet justify treating morphology-aware fitting as a broadly unique target across this frozen set.
PriorNet training remains **unauthorized**. The POST audit layer may return
only after a direct structural solver establishes reconstruction accuracy and
produces scientifically valid outputs; identifiability alone is insufficient.

## Integrity

Preparation and compatibility tests passed before science; all 600 historical
checkpoints matched both before and after execution. Historical reports,
README, and the Git index remain unchanged. Scientific accesses were exactly
16 authorized blend/coordinate row reads total (two per scene, one per family),
with zero isolated-source array reads, zero catalog-row reads, zero
development, zero Atlas-array, and zero lockbox access. No neural network was
loaded or trained. Nothing was staged or committed.

The workspace had 382 pre-existing porcelain entries before this campaign.
They remain user-owned. Campaign-created files are this fresh ignored run tree
and `scripts/run_thayer_flux_free_identifiability_v0.py`; no prior output was
deleted or rewritten.

## Exactly one recommended next experiment

Run **Thayer-PSF-Diverse-Flux-Identifiability-v0** to test whether a second known PSF observation supplies enough independent information to remove the restored allocation ambiguity. PriorNet is not authorized by this
recommendation.
