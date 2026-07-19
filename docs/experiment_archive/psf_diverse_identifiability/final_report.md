# Thayer-PSF-Diverse-Flux-Identifiability-v0 final report

## Campaign outcome

**PSF_DIVERSE_CONDITIONING_IMPROVES_WITHOUT_UNIQUENESS**

The primary P2 union endpoint is **0/8 UNIQUE**. P2 Level 4 is
**0/8** and Level 5 is **0/8**.
The same-PSF S2 control is L4 **0/8**, L5
**0/8**, union **0/8**. The authoritative S1
counts remain L4 **0/8**, L5
**0/8**, union **0/8**.

## Solver information and paired acquisition

The solver received only the two blended g/r/z observations, frozen requested
and companion coordinates, both known normalized PSFs, 60x60 geometry at 0.2
arcsec/pixel, separate observation-derived BTK Poisson plug-in sigma maps, and
the frozen Level-4/5 support. One latent morphology-and-flux parameter vector
jointly explained both observations. No isolated image, true per-source flux,
catalog parameter, truth morphology/family, mask, truth initialization, or true
noise realization entered inference.

Observation A is the unchanged authoritative blend. A2 and B were freshly
generated from the identical two CatSim rows and coordinates through BTK
`CatsimGenerator(add_noise='all')`, with the preregistered independent second
seed. A2 used PSF A; B used PSF B. Simulator-only catalog state and generated
isolated layers were discarded before the paired observations were sealed.

PSF A has nominal g/r/z FWHM 0.86/0.81/0.77 arcsec. PSF B is the preregistered
0.70/0.68/0.66 arcsec, e=0.10, 30-degree GalSim construction. Across bands,
relative kernel L2 distances span
`0.22`--
`0.2993`,
Fourier-transfer relative distances span
`0.22`--
`0.2993`,
and cross-correlations span
`0.9828`--
`0.99`.

## Scene-level transitions (S1 -> S2 -> P2)

- Scene 0: L4 `NON_IDENTIFIABLE -> NON_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE -> NON_IDENTIFIABLE -> NON_IDENTIFIABLE`; P2 minimum unique family `none`.
- Scene 3: L4 `PARTIALLY_IDENTIFIABLE -> NON_IDENTIFIABLE -> NON_IDENTIFIABLE`; L5 `NON_IDENTIFIABLE -> NON_IDENTIFIABLE -> NEAR_UNIQUE`; P2 minimum unique family `none`.
- Scene 5: L4 `NON_IDENTIFIABLE -> NON_IDENTIFIABLE -> NON_IDENTIFIABLE`; L5 `NEAR_UNIQUE -> NEAR_UNIQUE -> NON_IDENTIFIABLE`; P2 minimum unique family `none`.
- Scene 6: L4 `PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE -> NEAR_UNIQUE -> NEAR_UNIQUE`; P2 minimum unique family `none`.
- Scene 18: L4 `NON_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE -> PARTIALLY_IDENTIFIABLE -> NEAR_UNIQUE`; P2 minimum unique family `none`.
- Scene 51: L4 `PARTIALLY_IDENTIFIABLE -> NON_IDENTIFIABLE -> NON_IDENTIFIABLE`; L5 `PARTIALLY_IDENTIFIABLE -> NEAR_UNIQUE -> NON_IDENTIFIABLE`; P2 minimum unique family `none`.
- Scene 73: L4 `PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE`; L5 `NEAR_UNIQUE -> NON_IDENTIFIABLE -> NON_IDENTIFIABLE`; P2 minimum unique family `none`.
- Scene 81: L4 `NEAR_UNIQUE -> NEAR_UNIQUE -> NEAR_UNIQUE`; L5 `PARTIALLY_IDENTIFIABLE -> PARTIALLY_IDENTIFIABLE -> NEAR_UNIQUE`; P2 minimum unique family `none`.

Family-level causal attributions: `{"ADDITIONAL_EXPOSURE_ONLY": 2, "BOTH_EXPOSURE_AND_PSF_DIVERSITY": 0, "INCONCLUSIVE_OPTIMIZATION": 0, "NO_MEANINGFUL_GAIN": 11, "PSF_DIVERSITY_SPECIFIC": 3}`.
An improvement is not attributed to PSF diversity merely because P2 exceeds
S1; S2 is the mandatory repeated-exposure control.

## Rank, conditioning, endpoint geometry, and flux allocation

All local ranks, nullities, singular spectra, Hessian spectra, conditions,
endpoint-class counts, requested/companion image diameters, flux-allocation
diameters, morphology diameters, boundary contacts, prompt-swap tests, replay
hashes, and perturbation results are in the full metrics and information-gain
tables. Relative to S1, the P2 joint minimum-singular ratios span
`0.6082`--`6354` and joint condition ratios
span `0.0002012`--`2.808`.
`15/16` P2 family fits meet the preregistered >=5%
information/geometry improvement rule. Fitted per-source g/r/z allocations are
reported for every condition without oracle component photometry.

All **512** planned endpoints are retained. Optimizer-declared failures:
**64**. Starts reaching the 500-evaluation ceiling: **64**.
Good joint image likelihood alone was not treated as source uniqueness.

## Scientific interpretation

PSF diversity restored uniqueness: **no under the frozen campaign threshold**.
The same-PSF control shows whether any gain is explained by an added exposure;
the exact causal breakdown above is authoritative. These are conditional
structural-identifiability findings for eight frozen simulated scenes, not a
claim of unrestricted source recovery or real-survey generalization.

A morphology-aware reconstruction target is **not yet broadly justified as unique across the frozen set**.
Independent observations are **not shown to restore uniqueness**.
The audit layer may return **only after** a direct structural reconstruction
experiment produces accurate, scientifically valid outputs; identifiability
alone does not authorize POST. PriorNet remains unauthorized.

## Integrity

All inference audits pass: `32/32`.
Protected development, Atlas tensor, lockbox, and historical isolated-HDF5
access are zero. README, HEAD, the Git index, historical reports, the completed
predecessor run, and all 600 checkpoints remain unchanged. Nothing was staged
or committed.

The authorized post-science report-only finalization amendment in
`preregistration/post_science_finalization_amendment.json` coerces serialized
numeric strings such as `"inf"` only for report arithmetic. It does not alter
the protocol, observations, completed fits, thresholds, or scientific outcome.

## Exactly one recommended next experiment

Run **Thayer-External-Photometry-vs-PSF-Diversity-v0** to compare independent per-source photometry against PSF diversity as a targeted missing information source. No neural training is recommended.
