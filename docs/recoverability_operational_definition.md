# Operational recoverability for Thayer-Select

Status: initial benchmark specification, not a calibrated claim. Thresholds
must be frozen on the calibration partition and reported unchanged on the
development-test partition. The future lockbox is excluded from all work in
this document until a separately authorized final evaluation.

## Observable input and hidden truth

For sample (i), the model observes only a normalized three-channel DR10
coadd blend (x_i\in\mathbb{R}^{3\times H\times W}) in ordered (g,r,z)
bands and a Gaussian prompt (p_i\in\mathbb{R}^{1\times H\times W}) centered
on the requested sky coordinate. A future, separately validated version may
also observe aligned inverse-variance channels. Morphology labels, source IDs,
partition labels, clean targets, blend-generator parameters, affected masks,
and evaluation errors are not model inputs.

During supervised evaluation only, the benchmark also has the clean requested
target (y_i), the injected source-only contaminant, and generator-derived
affected/core masks. These hidden quantities define reconstruction error and
failure labels. They may train an output head on the training partition and
set a decision rule on the calibration partition, but they may not enter a
forward pass at inference time.

## What “recoverable” means

Recoverability is a decision-relative property: under a frozen data
distribution, model version, normalization, reconstruction loss, and scientific
failure rule, an observation is recoverable when the requested reconstruction
meets that rule. The predicted recoverability score is an estimate of this
property using model-accessible information. It is not synonymous with neural
confidence, low pixel variance, visual plausibility, or a high softmax value.

The benchmark will report at least the following hidden-truth measurements:

- affected-region normalized MSE and absolute error in physical flux after
  inverse normalization;
- requested-source aperture-flux fractional error in each band and jointly;
- false-subtraction flux in target core pixels that were not injected by the
  contaminant;
- centroid displacement and color error of the reconstruction;
- whole-image error as a guard against rectangular leakage and background
  alteration.

An initial catastrophic-reconstruction candidate is any sample with one or
more of: affected-region normalized MSE above 0.10, absolute requested-source
flux error above 25% in any scientifically usable band, or a centroid error
above a frozen pixel threshold. A false-subtraction candidate removes more
than 20% of the clean target core flux outside the injected affected mask or
creates a comparably large negative residual there. These numerical values are
provisional benchmark candidates; they are not final until their units,
apertures, denominators, and edge cases are frozen using training and
calibration data only.

## Prompt failures

An empty prompt is an exact all-zero prompt channel with an explicit validity
flag set false at prompt construction. It is not encoded by NaN propagation or
an arbitrary off-frame coordinate. A wrong prompt is a finite non-empty prompt
whose requested location does not correspond to the supervised target. Wrong
prompts include another detected source, blank sky, and an offset location.

The interface must represent both cases. They receive dedicated benchmark
labels and are never silently remapped to the central source. A model that
returns a plausible central galaxy for an empty or wrong prompt has failed the
identity-selection task even if its image looks realistic.

## Coverage, risk, and calibration

At a score threshold (t), coverage is the fraction of finite scored samples
with score at least (t). Selective risk is the mean frozen reconstruction
loss among those retained samples. Reports must show the full risk-coverage
curve, the number of non-finite/excluded samples, catastrophic-failure rate,
false-subtraction rate, and morphology/brightness/separation-stratified
coverage. Empty selection is reported explicitly and never treated as zero
risk.

The calibration partition is used once the model and loss are frozen to map
raw scores to an empirical probability or choose operating thresholds for
predeclared risk/coverage targets. Development testing evaluates those fixed
choices and may reveal a need for another development cycle. It must not be
relabelled as calibration after inspection. The future lockbox is untouched by
both activities and is used only for a final, separately authorized assessment.

Calibration quality will be assessed with reliability diagrams, bin counts,
expected/maximum calibration error with predeclared bins, Brier score, and
confidence intervals. A pixel log-variance head is called “predicted
uncertainty” until these tests pass; it is not called calibrated uncertainty.

## Candidate supervision targets

Several candidates should be compared without choosing solely by visual appeal:

1. A binary label for passing all frozen catastrophic and false-subtraction
   thresholds.
2. A continuous monotone transform of affected-region physical-flux loss,
   with separate reporting of the binary scientific failures.
3. A vector or multi-task target for flux error, centroid error, false
   subtraction, and prompt validity, combined into an operating decision only
   after calibration.
4. A probability of remaining below a predeclared loss threshold, estimated by
   proper scoring rules and checked for conditional calibration across source
   brightness, morphology, color, separation, and obstruction.
5. A conservative upper prediction bound on reconstruction risk. This is
   useful only if its empirical coverage is validated; a raw variance channel
   is not such a bound.

Selection among these targets will prioritize scientific failure detection,
proper scoring behavior, stable subgroup coverage, and reproducibility.

## Ambiguity benchmark

The Ambiguity benchmark tests information-theoretic limits rather than only
ordinary model error. It constructs or mines pairs with near-identical
model-observable blends and prompts but materially different hidden requested
targets. Near-identity is measured in the exact normalized observable tensor,
with tolerances frozen before development-test inspection. Hidden-target
divergence is measured in physical-flux reconstruction metrics.

For each matched pair, the benchmark records observable distance,
hidden-target distance, reconstruction errors for both members, predicted
recoverability scores, and whether the score appropriately declines. Pairs and
all source groups remain within one source partition; no target or contaminant
crosses roles. A high score on both members of an observationally
indistinguishable, target-divergent pair is evidence of overconfidence. Failure
on such pairs does not by itself imply a better architecture exists; it may
identify an irreducible ambiguity that should be rejected by the operational
selection rule.

## Versioning and prohibited shortcuts

Every recoverability result must identify the source split, blend-generator
version, normalization statistics and training-only provenance, prompt
definition, loss version, failure-threshold version, model hash, and calibration
mapping. No morphology label, clean target statistic, generator mask, source
identity, or oracle difficulty metric may enter the model input or score at
inference. No threshold may be changed after development-test or lockbox
inspection without declaring a new benchmark version.

## Provisional deterministic metric specification v0.2

This section supersedes any earlier placeholder such as “a frozen pixel
threshold.” It makes the candidate failure rule executable without claiming
that its numerical values are calibrated or final.

Let `M_aff` be the nonempty generator-derived affected mask, `M_core` the
nonempty clean-target core mask, `y[b,p]` the inverse-normalized clean target,
and `yhat[b,p]` the inverse-normalized reconstruction, all in nanomaggies per
pixel. A missing/empty required mask, a nonfinite value in a scored mask, or a
normalization inverse failure is an invalid evaluation and is counted as
catastrophic, never silently dropped.

Affected-region NMSE is

```text
sum_{b,p in M_aff} (yhat[b,p] - y[b,p])^2
-------------------------------------------------
max(sum_{b,p in M_aff} y[b,p]^2, epsilon_flux^2)
```

with provisional `epsilon_flux = 1e-12` nanomaggies. If the unguarded
denominator is at or below `epsilon_flux^2`, the NMSE is reported as undefined
and the sample fails rather than gaining an artificial favorable score.

For band `b`, core aperture flux is `F_b = sum_{p in M_core} y[b,p]` after the
same fixed background convention used by the generator. With no verified
IVAR, background scale is `sigma_b = 1.4826 * MAD` on the declared clean-target
background mask and aperture noise is `sigma_b * sqrt(|M_core|)`. A band is
scientifically usable when every required value is finite, at least 64
background pixels exist, `sigma_b > 0`, and `abs(F_b)/aperture_noise >= 5`.
With verified IVAR, an alternative version may replace this noise expression,
but it cannot be mixed into v0.2. No usable band is an explicit failure.

Per usable band, flux fractional error is
`abs(sum_Mcore(yhat)-F_b) / max(abs(F_b), epsilon_flux)`. Any usable band above
0.25 fails. The centroid uses nonnegative background-subtracted weights
`w[p]=max(sum_over_usable_bands(y[b,p]),0)` within `M_core`. If total weight is
not positive, centroid is undefined and fails. Otherwise Euclidean target-to-
reconstruction centroid displacement above 2.0 pixels (0.524 arcsec) fails.

Protected target pixels are `M_protected = M_core AND NOT M_aff`. False
subtraction has an explicit positive-removal sign:

```text
sum_{b,p in M_protected} max(y[b,p] - yhat[b,p], 0)
----------------------------------------------------------------
max(sum_{b,p in M_protected} max(y[b,p], 0), epsilon_flux)
```

An empty protected mask is recorded as not applicable, not zero evidence; if
nonempty, a fraction above 0.20 fails. Added flux is reported separately with
`max(yhat-y,0)` and cannot cancel removed flux.

The provisional v0.2 binary recoverable label is true only when the evaluation
is valid, at least one usable band exists, affected NMSE is at most 0.10, all
usable-band flux errors are at most 0.25, centroid error is at most 2.0 pixels,
and every applicable false-subtraction fraction is at most 0.20. Failures
combine by logical OR and every component remains in the result table.

For an empty prompt, the supervised requested target is an exact zero source
map and `M_core` is a frozen Gaussian-prompt aperture; predicted positive
aperture flux above five background-noise standard deviations is false
selection. For a wrong prompt, evaluation uses the source identity actually
requested at that coordinate; if no catalog/detection-matched source is
declared, it uses the same zero-target false-selection rule. It is prohibited
to recenter the prompt onto the nearest source during evaluation.

## Provisional deterministic metric specification v0.3

Version v0.3 supersedes v0.2 and resolves three incomplete definitions while
leaving all provisional numerical failure thresholds unchanged.

For the centroid term, define separate nonnegative weights on `M_core`:

```text
w_y[p]    = max(sum_over_usable_bands y[b,p], 0)
w_yhat[p] = max(sum_over_usable_bands yhat[b,p], 0)
c_y       = sum_p (x_p,y_p) w_y[p]    / sum_p w_y[p]
c_yhat    = sum_p (x_p,y_p) w_yhat[p] / sum_p w_yhat[p]
```

Both weight sums must be finite and strictly positive. Otherwise the centroid
metric is undefined and the sample fails. The reported displacement is
`sqrt((c_yhat.x-c_y.x)^2 + (c_yhat.y-c_y.y)^2)` in pixels and fails above 2.0.

An empty-prompt benchmark row stores a hidden evaluation coordinate `q_i` and
an evaluation-aperture mask in benchmark metadata; neither is model input.
`q_i` must be finite, inside the image, and fixed before evaluation. The mask
is exactly the set of pixel centers within `2*sigma_prompt` pixels of `q_i`,
using the same frozen `sigma_prompt` recorded for nonempty Gaussian prompts.
The visible prompt channel is nevertheless all zero because its validity flag
is false. For each band with at least 64 finite background pixels and positive
robust background scale `sigma_b`, define
`S_b = max(sum_{p in aperture} yhat[b,p], 0) /
(sigma_b*sqrt(|aperture|))`. False selection occurs if **any** such band has
`S_b > 5`; having no evaluable band is an invalid evaluation and fails.

For a finite wrong prompt at coordinate `q_i`, a source is considered matched
only if the frozen segmentation label at the nearest integer pixel is nonzero,
or, when that pixel is unlabeled, exactly one detected centroid lies within
2.0 pixels. More than one centroid in that radius is ambiguous and fails the
evaluation rather than being chosen by order. A matched wrong prompt is scored
against that matched source's clean target and core mask. An unmatched wrong
prompt uses the exact zero target and the empty-prompt aperture/any-band `5
sigma` rule above. Evaluation never recenters the model prompt.

All error quantities and all thresholds must be finite and nonnegative;
`epsilon_flux` and required noise scales must be strictly positive. Negative
NMSE, fractional error, centroid displacement, or false-subtraction values are
invalid and fail. The code-level threshold metadata version is
`recoverability-thresholds-provisional-v0.3`. These remain candidate scientific
rules to be frozen with training/calibration data, not calibrated claims.

## Provisional deterministic metric specification v0.4

Version v0.4 supersedes v0.3 and freezes one internally consistent benchmark
target convention. For a valid requested-source sample, `x` is the blended
coadd, `y` is the complete clean target coadd that existed before contaminant
injection (including its original target background and noise realization),
and `yhat` is the model's complete reconstructed target coadd. A model
configured internally as a correction model must expose
`yhat = x + predicted_correction` before evaluation. Metrics never mix a
source-only truth with a full-coadd truth.

For empty or deliberately wrong prompts, the declared task is abstention from
pixel modification: the truth correction is exactly zero, equivalently the
truth reconstruction is `y = x`. A wrong prompt is any valid nonempty prompt
whose coordinate is not the manifest-declared target identity/coordinate for
that row; it is not silently reassigned to another source. Normal source A and
source B reconstruction are represented by separately declared valid-target
rows. Empty/wrong rows are stress cases and may use only source groups from the
same partition. This convention is a benchmark choice, not a claim that it is
the only scientifically useful output parameterization.

### Dimensioned valid-prompt metrics

Arrays have units nanomaggies per pixel and pixel indices are dimensionless.
Two distinct guards are used:

- `epsilon_pixel = 1e-12` nanomaggies per pixel;
- `epsilon_aperture = 1e-12` nanomaggies for summed aperture flux.

For nonempty `M_aff` with `N_aff` selected band-pixels, affected NMSE is

```text
sum_{b,p in M_aff} (yhat[b,p]-y[b,p])^2
-----------------------------------------------------------
max(sum_{b,p in M_aff} y[b,p]^2, N_aff*epsilon_pixel^2)
```

The affected absolute error is
`mean_{b,p in M_aff} abs(yhat[b,p]-y[b,p])` in nanomaggies per
pixel. Both are undefined and fail when required values are nonfinite or the
unguarded NMSE denominator is at most `N_aff*epsilon_pixel^2`.

For band `b`, target aperture flux is `F_b=sum_{p in M_core} y[b,p]`.
Correlated coadd noise is estimated empirically rather than as
`sigma_pixel*sqrt(Npix)`: translate the exact core-aperture shape to
deterministically seeded positions wholly contained in the frozen background
mask, exclude every detected-source segmentation, subtract the median blank-
aperture sum, and require at least 64 valid placements. Define
`sigma_ap,b = 1.4826*MAD(blank_aperture_sums)` and require it to be finite and
positive. A galaxy band is usable only when `F_b > 0` and
`F_b/sigma_ap,b >= 5`. No usable band is an invalid catastrophic evaluation.

Per usable band, flux fractional error is
`abs(sum_Mcore(yhat[b])-F_b)/max(F_b,epsilon_aperture)`. Joint aperture-flux
error is the sum of absolute aperture-flux errors divided by
`max(sum_b F_b,epsilon_aperture)` over usable bands. The per-band 0.25 failure
threshold remains provisional. Colors are reported only when both component
bands are usable and positive: `g-r=-2.5*log10(F_g/F_r)` and
`r-z=-2.5*log10(F_r/F_z)`; prediction-minus-truth absolute color error is in
magnitudes. Missing color pairs are reported as not applicable, never zero.

The separate target/reconstruction centroid definition from v0.3 remains in
force. Outside the injected affected mask, define a whole-image/background
guard as
`RMSE(yhat-y)/sigma_background`, where `sigma_background` is the robust
per-pixel scale on the same frozen source-excluded clean-target background
mask. Nonfinite/nonpositive scale fails; a provisional value above 0.25 fails.
The false-subtraction definition and 0.20 threshold from v0.3 remain, using
the distinct aperture-flux guard.

The valid-prompt recoverable label is the logical AND of valid evaluation,
one or more usable bands, affected NMSE at most 0.10, every usable-band flux
error at most 0.25, centroid displacement at most 2.0 pixels, outside-affected
RMSE at most 0.25 background sigma, and every applicable false-subtraction
fraction at most 0.20. Absolute error, joint flux error, and color error are
mandatory reported diagnostics but do not gain an uncalibrated extra cutoff.

### Empty/wrong prompt metric

Each stress row stores a hidden in-frame evaluation coordinate and the exact
`2*sigma_prompt` aperture mask; these are evaluation metadata, not model input.
Its frozen background mask is built from the input coadd with all detected-
source segmentations excluded. The same at-least-64 empirical translated blank
apertures define `sigma_ap,b`. With `dhat=yhat-x`, false selection occurs when
`abs(sum_aperture dhat[b])/sigma_ap,b > 5` in any evaluable band, or when the
outside-aperture correction exceeds 0.25 background sigma. No evaluable band,
missing clean identity metadata, or a nonfinite quantity fails. Wrong prompts
are generated only for rows with a predeclared valid target truth and core
mask; their truth remains the zero correction rather than an opportunistically
recentered alternate source.

### Risk denominator and invalid rows

The primary selective risk is the mean binary scientific-failure indicator,
so every row has finite loss 0 or 1 and every invalid evaluation is assigned
1. Model scores must be finite in `[0,1]`; a nonfinite score is recorded as a
model failure and mapped to the finite sentinel `-1` solely for deterministic
ranking. Coverage always divides by all benchmark rows. The utility refuses
nonfinite scores/losses rather than silently excluding them. Empty selection
has coverage zero and undefined risk; full coverage includes invalid rows.
Continuous reconstruction losses are reported separately and require their
own predeclared finite catastrophic cap before use in a risk curve.

### Executable Ambiguity benchmark proposal

For each row, concatenate the fixed-normalized observable image and prompt and
define observable distance as tensor RMS:
`d_obs(i,j)=||z_i-z_j||_2/sqrt(number_of_tensor_elements)`. Define hidden-target
distance as the square root of summed physical-flux squared difference divided
by the square root of
`max(sum(y_i^2)+sum(y_j^2), N*epsilon_pixel^2)`. Candidate near-observable and
divergent-target cutoffs are respectively `d_obs <= 1e-3` and
`d_hidden >= 0.25`; these numbers remain provisional.

Pairs must be from distinct source groups within one partition and the same
shape/prompt-validity stratum. Enumerate all eligible pairs when feasible or
use a deterministic exact nearest-neighbor index; sort candidates by
`(d_obs, source_id_i, source_id_j)` and greedily select disjoint pairs. Freeze
the transform, distance code, cutoffs, and pairing list using training and
calibration only, then report it unchanged on development test. The lockbox is
excluded. This makes the ambiguity proposal executable without asserting that
its provisional distance cutoffs are scientifically final.
