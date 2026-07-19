# Competing-Hypothesis Recoverability preregistration

Status: **FROZEN_BEFORE_NEW_CANDIDATE_INFERENCE_OR_TRAINING**  
Frozen at: `2026-07-12T18:56:29.488500+00:00`  
Working project title: **Don't Even Try**

## Scope decision and family definitions

Only one meaningfully distinct compatible family cluster is currently
available: `THAYER_COMPACT_PROMPTED_UNET`, represented by frozen Condition C,
R0, and reconstruction-only R1 controls. SEP has no validated source-layer
adapter, legacy RGB models have incompatible contracts, and scarlet is absent.
Consequently, Atlas and finite-candidate ambiguity feasibility are active;
leave-one-family-out training, held-out-family evaluation, and model-agnostic
claims are blocked until at least three genuinely distinct compatible families
exist. Absence of a witness will never be treated as proof of uniqueness.

## Source commitments and scene counts

The immutable parent split is SHA-256 `98ccf4d2662b6fbef803b5b4a187769521759093c17d5a118afb38ee1035ae27`. Historical
`development_test` and `sealed_lockbox` identities are omitted from campaign
commitments and their scenes are never opened, sampled, rendered, normalized,
or inspected. Existing training groups are deterministically divided by group
hash into campaign training and a prospective 10% feasibility-audit holdout;
existing validation and calibration groups retain their roles. Approved row
counts are `{"audit_evaluation": 5998, "calibration": 6901, "training": 54341, "validation": 8630}`.

Frozen requested scene counts are 30,000 noiseless TRAIN/SEARCH scenes, 2,000
fresh validation scenes, and 3,000 fresh calibration scenes. If implementation
or resource validation cannot support these counts, the run stops before
claims; it does not silently substitute a smaller analysis. Atlas v0 uses
two-source scenes only. Targets and contaminants always come from the same
campaign role and duplicate group is the atomic isolation unit.

## Candidate-output contract

Every candidate is a float source-layer tensor `(K,3,60,60)` with g/r/z band
order, 0.2 arcsec pixels, PSF-convolved noiseless detected electrons per pixel,
no clipping, and no sky/background duplicated into any layer. Its full
decomposition is the sum of its K layers plus an explicitly zero background.
The requested layer is selected only by the declared prompt-to-source mapping.
Family/checkpoint/path metadata is provenance only and is prohibited from any
auditor tensor. Each output stores scene ID, candidate ID, requested layer,
decomposition hash, measurements, runtime, finite status, and configuration
hash. Expected-deterministic models must replay byte-identically or within
`1e-6` maximum absolute float32 error.

## Frozen forward model and measurement distance

The forward model is BTK 1.0.9 / GalSim 2.8.4 LSST g/r/z rendering as frozen in
`src/btk_scene.py`: source layers are already PSF-convolved, so recomposition is
their unclipped sum. Observation noise is source Poisson plus one zero-mean sky
Poisson realization. Per-pixel variance for candidate consistency is
`max(recomposed_noiseless + sky_electrons_per_pixel, 1.0)` separately by band,
where the exact surveycodex mean sky level is stored in scene metadata.
Residual is `observed - recomposed`; whitened residual is residual divided by
the square root of that variance. The primary consistency score is the mean
squared whitened residual. Per-band means, 8-neighbor residual correlation,
and relative total-flux residual are mandatory diagnostics. Truth is never an
input to this score.

Plausibility thresholds are fit without candidate outcome errors: use the
calibration distribution of the known-truth full decomposition under the same
observation/noise contract. Freeze the finite-sample conservative 99th
percentile of the global score, 99.5th percentile per band, and 99th percentile
of absolute relative flux residual. A candidate passes only if all applicable
limits pass. The threshold procedure, not its future numeric values, is frozen
here; calibration may not use audit-evaluation families or scenes.

## Scientific distances and empirical witness

For requested layers A and B, define image distance as
`||A-B||_2 / (0.5*(||A||_2+||B||_2)+training_flux_floor)`. Per-band flux
distance is absolute flux difference divided by the absolute mean flux plus a
training-only floor. Colors are AB-like `-2.5 log10(F1/F2)` and are not
applicable for non-positive flux. Centroids use nonnegative band-summed source
weights after subtracting only the training-frozen zero floor; nonpositive
total flux is not applicable. Shape distance is diagnostic until its validity
gate passes.

Frozen component limits are image distance 0.25, any-band relative flux 0.20,
either color 0.20 mag, or centroid distance 0.5 mean-PSF FWHM. The primary
diameter is the maximum applicable component divided by its limit. An empirical
ambiguity witness requires at least two forward-consistent decompositions with
primary diameter greater than 1.0 and a passed unit/clipping/translation/
serialization/background artifact audit. It certifies ambiguity only within
the finite candidate family and frozen forward model. No witness never
certifies uniqueness.

## Ambiguity Atlas

Route 1 generates exactly 30,000 noiseless training scenes, embeds raw
noise-whitened pixels (PCA only if fit on training), and rejects same-group
pairs and global rescalings. Route 2 prospectively performs bounded
counterfactual optimization over contaminant choice from a finite approved
pool, position, flux scale, and orientation using exact BTK/GalSim rendering.
Both routes preserve seeds, bounds, replay hashes, and route-specific results.

A valid pair has different requested-source duplicate groups, exact replay,
mean squared whitened blend difference at most 0.25, primary truth diameter
greater than 1.0, and passed numerical plus visual artifact audit. The initial
Atlas freezes only with at least 25 genuine pairs. At least one frozen
deblender must then show either confidence inversion, essentially identical
outputs on divergent truths, or a scientifically unsafe output on the set.

## Failure labels and black-box inputs

Labels retain positive/negative/not-applicable semantics for QUERY_NULL,
QUERY_AMBIGUOUS, SOURCE_CONFUSION, CATASTROPHIC_IMAGE, CATASTROPHIC_FLUX,
CATASTROPHIC_CENTROID, COLOR_UNSAFE, SHAPE_UNSAFE, ATLAS_NON_IDENTIFIABLE, and
SAFE_CANDIDATE. Truth is allowed only to form these labels and evaluation
metrics.

If family compatibility is later reopened prospectively, deployable auditor
inputs are limited to observed blend, coordinate prompt, candidate requested
layer, candidate full decomposition, blend-minus-recomposition residual,
forward score, candidate measurements, plausible-set diameter, and legitimate
observational metadata. Target truth, family/checkpoint/path/architecture
identity, private activations, gradients, training loss, true errors, source
IDs, true SNR/obstruction, and generator variables are forbidden. Frozen
ablations A0--A5 follow the campaign brief. A compact two-stream CNN with
masked failure-specific heads and five seeds is the only allowed primary
auditor; no broad search is allowed.

## Calibration, coverage, intervals, and success gates

If cross-family work becomes attainable in a future preregistered extension,
each family is excluded from training, model selection, and calibration;
thresholds freeze on seen-family validation/calibration and the held-out family
is evaluated once. Accepted coverages are 95, 90, 80, 70, and 50%. Confidence
intervals are source-duplicate-group cluster bootstraps with 2,000 resamples,
and families are macro-averaged. Random rejection and oracle ranking bounds are
reported.

Atlas feasibility passes only with at least 25 valid replayable pairs, all
distance/divergence/artifact gates, and at least one deblender failure. Witness
feasibility passes only if diameter beats both self-confidence and forward
residual in training/validation selection and calibration-frozen evaluation,
is stable across five seeds where a stochastic component exists, and has
useful recall at a frozen 5% false-positive rate. Cross-family audit is
currently mathematically unattainable because fewer than three distinct
families exist and is therefore not an active success gate. Its future 80%
coverage relative false-safe reduction must be selected from training-only
prevalence before held-out-family evaluation and may never be changed post hoc.

## Critical ablations and correctness stops

Required future ablations are candidate shuffling, removal of forward residual,
candidate, blend, prompt, or diameter, same-family seeds only, removal of
distinct families, family-ID leakage, normalization/border probes, held-out
failure severity, and separate Atlas evaluation. Any target leakage, family ID
in tensors, group overlap, calibration reuse, candidate/prompt misalignment,
forward/noise formula failure, post-evaluation tuning, historical development
access, lockbox access, historical-checkpoint mutation, staged-index mutation,
or path collision is a fail-closed stop.

## Attainability audit

- Source counts exceed the requested scene counts without replacement at the
  duplicate-group level; exact scene reuse is not required.
- The forward score has finite variance floor and defined calibration
  quantiles.
- The witness thresholds can be passed or failed by finite arrays and do not
  require inaccessible metadata.
- The Atlas minimum is finite and both construction routes use approved
  training/validation sources only.
- Atlas and witness gates are independently attainable with the current single
  family cluster plus optimized decompositions.
- Cross-family transfer is explicitly blocked, not assigned an impossible
  success requirement.
- No final-development, lockbox, survey, uniqueness, or model-agnostic claim is
  authorized by this preregistration.
