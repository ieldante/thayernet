# Thayer-Recoverability-v0 scientific report

## Decision

**Primary conclusion: observation-level source identity is not unique under the
declared Family-E1 output contract. All eight unique frozen observations are
classified `FUNDAMENTALLY_UNIDENTIFIABLE` as source-pair decompositions.**

This is evidence for explanation **(A), insufficient information in the
observation for a unique unrestricted two-image decomposition**, rather than a
claim that increasing coordinate-prompt strength inside the present network
would make the inverse problem identifiable. The prompt is not numerically
ignored: Family-E1P already showed nonzero prompt modulation and gradients at
every traced layer. The missing information is source allocation. A coordinate
labels a component but does not determine how the blended light must be divided
between two nonnegative source images.

This conclusion has an important boundary. The current architecture also has
an optimization/inductive-bias limitation: it failed on several geometrically
easy scenes, while it succeeded on index 81. That affects whether it learns the
simulator's chosen decomposition. It cannot, however, remove the exact
observation null space demonstrated here. Thus architecture is not exonerated
as a supervised fitter, but it is not the primary answer to the stated
information-sufficiency question.

## Evidence boundary

The audit used only Family-E1 training indices `[0, 3, 5, 6, 18, 51, 73, 81]`.
Index 6 is both the frozen difficult one-scene case and one member of the
mixed-eight set, so there are eight unique observations and nine condition
entries, not nine independent scenes. Observation and isolated-source hashes
were verified against the frozen Family-E1P and upstream manifests before any
calculation.

The authoritative reports establish the following prior facts.

- Family-E1 passed physical closure, objective alignment, and ordinary
  one-scene overfit, but prompt identity was `0.5000` on the difficult scene
  and `0.5625` on mixed eight.
- Family-E1P reproduced all 28 compared values exactly. Generic prompt
  modulation survived, but the source-identity-aligned contrast was weak;
  prompt swap was `0/1` on the difficult case and `1/8` on mixed eight.
- In the separate fixed-feature audit, square D0 and D1 reached 100% own,
  alternate, and both-mode coverage, while D2 reached 0%. Those results show
  output and free-feature reachability plus a fixed-feature decoder barrier;
  they do not establish observation uniqueness.
- D3 was not validly completed in the authoritative combined campaign. Its
  scientific classification remains unknown, so no D3 capacity inference is
  used here.

No reconstruction model or checkpoint was loaded, constructed, optimized, or
modified. No training, validation, calibration, development, Atlas, or lockbox
array was accessed. Prompts and thresholds were unchanged.

## Definitions

For source images `A,B >= 0`, observed image `O`, and the frozen signed noise
realization `R0 = O-A-B`, the strongest-case inverse problem grants the analyst
the exact noise and requires

```text
S_A + S_B + R0 = O.
```

Its complete nonnegative exact-fit set is

```text
(S_A, S_B) = (A + delta, B - delta),
              with -A <= delta <= B elementwise.
```

Every isolated tensor is strictly positive at every stored pixel, so this set
has local dimension `10,800` per scene. Family-E1 is even less constrained when
the signed residual is free: every nonnegative source pair is feasible with
`R = O-S_A-S_B`.

The following scene metrics are reported.

- **Overlap fraction:** `sum(min(A,B))/min(sum(A),sum(B))`, the fraction of the
  fainter source's positive light lying under positive light from the other
  source.
- **Flux ratio:** symmetric total flux ratio `min(F_A,F_B)/max(F_A,F_B)`.
- **Centroid separation:** catalog-rendered peak separation in pixels and in
  the fixed mean PSF FWHM (`4.0667` pixels).
- **Color similarity:** cosine of the two rendered g/r/z flux vectors.
- **PSF overlap:** mean analytic Gaussian-PSF cosine
  `exp(-d^2/(4 sigma_b^2))` using the frozen g/r/z FWHM values
  `0.86/0.81/0.77` arcsec and `0.2` arcsec pixels. It is explicitly a
  FWHM-matched approximation to the fixed Kolmogorov-plus-Airy PSF.
- **Ambiguity score:** descriptive geometric mean of overlap fraction,
  symmetric flux ratio, color cosine, and PSF overlap. It is a ranking
  diagnostic only and is not used as a gate.

The only scientific distinction threshold is the unchanged empirical
ambiguity-certificate primary diameter. It is the maximum of symmetric image
distance divided by `0.25`, per-band symmetric flux distance divided by
`0.20`, applicable color distance divided by `0.20` mag, and centroid distance
divided by `0.5` mean-PSF FWHM. Diameter `>1.0` is scientifically distinct. No
threshold was introduced or changed for this audit.

## Scene geometry

| Family-E1 index | Membership | E1P prompt swap | Overlap | Flux ratio | Separation px / PSF | Color cosine | PSF overlap | Ambiguity score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | mixed eight | 0 | 0.6102 | 0.0852 | 6.759 / 1.662 | 0.9592 | 2.250e-2 | 0.1830 |
| 3 | mixed eight | 0 | 0.5941 | 0.0931 | 12.074 / 2.969 | 0.9674 | 7.849e-6 | 0.02546 |
| 5 | mixed eight | 0 | 0.01434 | 0.8211 | 19.641 / 4.830 | 0.9363 | 9.384e-14 | 1.794e-4 |
| 6 | difficult + mixed eight | 0 | 0.8500 | 0.7854 | 1.252 / 0.308 | 0.9722 | 0.8762 | 0.8684 |
| 18 | mixed eight | 0 | 0.6100 | 0.3472 | 3.300 / 0.812 | 0.6573 | 0.4004 | 0.4859 |
| 51 | mixed eight | 0 | 0.03565 | 0.9318 | 13.773 / 3.387 | 0.9949 | 2.645e-7 | 0.009670 |
| 73 | mixed eight | 0 | 0.3220 | 0.6244 | 6.669 / 1.640 | 0.8939 | 2.484e-2 | 0.2585 |
| 81 | mixed eight | 1 | 0.01960 | 0.8940 | 22.129 / 5.442 | 0.9908 | 3.821e-17 | 2.854e-5 |

Index 6 is the clearest conventional ambiguity: separation is only `0.308`
PSF FWHM, rendered-source overlap is `0.850`, PSF overlap is `0.876`, and the
noise-weighted source-template cosine is `0.887`. Index 81 is the opposite
extreme and is the sole Family-E1P prompt-swap success. Nevertheless, scene
geometry alone does not decide unrestricted output-space uniqueness: indices
5, 51, and 81 have negligible PSF overlap but retain the same allocation null
space.

## Fisher information, Jacobian rank, and local Hessian

Two analyses are intentionally separated.

First, an optimistic oracle-template approximation grants the two true source
shapes and fits only their fractional amplitudes. Per-band noise is estimated
from the robust MAD of `O-A-B`. Its Jacobian has two columns, the whitened true
templates. This is an upper-bound diagnostic: the actual observation does not
provide those templates.

| Index | Fisher eigenvalue min / max | Jacobian rank / null | Jacobian condition | Hessian condition | Shape-only Hessian condition | Worst CRLB amplitude std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 45.56 / 1.081e4 | 2 / 0 | 15.40 | 237.17 | 1.567 | 0.1481 |
| 3 | 6.240 / 2.268e3 | 2 / 0 | 19.07 | 363.52 | 1.288 | 0.4003 |
| 5 | 19.00 / 21.39 | 2 / 0 | 1.061 | 1.126 | 1.002 | 0.2294 |
| 6 | 103.29 / 1.726e3 | 2 / 0 | 4.088 | 16.708 | 16.651 | 0.09839 |
| 18 | 5.171 / 45.00 | 2 / 0 | 2.950 | 8.704 | 2.185 | 0.4398 |
| 51 | 82.63 / 101.74 | 2 / 0 | 1.110 | 1.231 | 1.013 | 0.1100 |
| 73 | 163.88 / 367.99 | 2 / 0 | 1.498 | 2.245 | 1.630 | 0.07811 |
| 81 | 2.865e4 / 3.649e4 | 2 / 0 | 1.129 | 1.274 | 1.004 | 0.005908 |

All oracle-template amplitude problems are rank two. Index 6 is locally
collinear even after column normalization; indices 0 and 3 have large raw
condition numbers mainly because their flux ratios are extreme, not because
their normalized shapes are collinear. Indices 5, 51, 73, and 81 are well
conditioned if their true shapes are supplied. This is evidence that a
physical source-shape prior could help; it is not evidence that the unrestricted
source images are identifiable.

For the actual direct-output inverse, vectorize each `3x60x60` source. The
two-source observation Jacobian is `[I I]`:

| Quantity | Two sources, fixed residual | Family-E1 sources + signed residual |
| --- | ---: | ---: |
| Parameter dimension | 21,600 | 32,400 |
| Observation dimension | 10,800 | 10,800 |
| Jacobian rank | 10,800 | 10,800 |
| Null-space dimension | 10,800 | 21,600 |
| Jacobian condition number | infinity | infinity |
| Local data-Hessian rank | 10,800 | 10,800 |
| Local data-Hessian null dimension | 10,800 | 21,600 |
| Local data-Hessian condition number | infinity | infinity |

These values apply to every scene. The prompt does not enter the additive
forward operator, so it cannot change these ranks. It can remove a discrete
component-label swap, but not the continuous allocation direction
`(delta,-delta)`.

## Direct output-space optimization and perturbation sensitivity

For each scene, 32 independent nonnegative source-pair starts were optimized
directly against the observation while holding the actual signed-noise
realization fixed. One projected-gradient step maps each start onto
`S_A+S_B=A+B`. All final pairs therefore have the same observation likelihood.
The maximum relative post-projection data objective over all 256 fits was
`1.165e-33`, while endpoint-to-truth primary scientific diameters ranged up to
`1.773`–`8.254` by scene.

Prompt-conditioned witnesses were then constructed along the stricter path
`(A+lambda B,(1-lambda)B)` in each prompt direction. Both component centroids
had to remain closer to their own frozen prompt than to the other prompt.
`lambda_crit` is the smallest companion-light fraction whose requested source
crosses the unchanged diameter-1 boundary; `--` means this particular transfer
direction remains within the scientific boundary even at its prompt-preserving
endpoint. The other prompt direction can still certify the source-pair
decomposition.

| Index | Max projected data objective | Direct-fit max diameter | Prompt-A witness diameter / `lambda_crit` | Prompt-B witness diameter / `lambda_crit` | Both views certified | Scene classification |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 7.074e-34 | 8.254 | 0.544 / -- | 1.606 / 0.0117 | no | FUNDAMENTALLY_UNIDENTIFIABLE |
| 3 | 4.325e-34 | 3.265 | 4.835 / 0.0119 | 0.837 / -- | no | FUNDAMENTALLY_UNIDENTIFIABLE |
| 5 | 4.272e-36 | 5.403 | 4.819 / 0.0952 | 4.595 / 0.1307 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |
| 6 | 3.656e-34 | 1.773 | 3.579 / 0.1993 | 4.241 / 0.1299 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |
| 18 | 1.299e-35 | 4.573 | 5.371 / 0.0957 | 6.378 / 0.0215 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |
| 51 | 7.658e-36 | 3.400 | 3.630 / 0.1866 | 3.494 / 0.1621 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |
| 73 | 4.994e-35 | 3.783 | 3.437 / 0.2122 | 5.027 / 0.0695 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |
| 81 | 1.165e-33 | 5.234 | 5.416 / 0.0910 | 5.111 / 0.1138 | yes | FUNDAMENTALLY_UNIDENTIFIABLE |

The prompt-conditioned witness objective never exceeded `3.883e-33`. Six of
eight scenes have scientifically distinct witnesses in both prompt directions.
Indices 0 and 3 have a sub-threshold witness in one direction because the
requested source is much brighter than its companion; the reverse prompt has a
diameter `>1` exact witness after transferring only `1.17%` or `1.19%` of the
brighter component. Thus every source-pair decomposition has at least one
scientifically distinct, prompt-consistent, observation-identical alternative.

Sensitivity is therefore singular, not merely large. Along `(delta,-delta)`,
the source pair changes while `Delta O=0`, so `||Delta source||/||Delta O||` is
infinite. Numerically, adding a dimensionless linear tilt of only `+/-1e-12` to
the otherwise flat allocation objective selects opposite ends of the
prompt-preserving interval. The resulting scene-level scientific jump is
`1.606`–`6.378`, while the unperturbed data objective is unchanged to at most
`3.883e-33`.

## Per-scene classification

The categories are applied as follows: `UNIQUE` requires a singleton
observation-consistent solution; `NEAR_UNIQUE` requires no exact null space but
poor conditioning; `AMBIGUOUS` denotes multiple exact solutions not yet shown
to differ beyond the inherited scientific boundary; and
`FUNDAMENTALLY_UNIDENTIFIABLE` requires an exact, prompt-consistent alternative
with primary diameter `>1` for at least one component of the requested source
pair.

All eight unique scenes are `FUNDAMENTALLY_UNIDENTIFIABLE`. Prompt A alone is
only certified `AMBIGUOUS` for index 0, and prompt B alone is only certified
`AMBIGUOUS` for index 3; the opposite prompt direction supplies the
scientifically distinct source-pair witness. No scene is `UNIQUE` or
`NEAR_UNIQUE` because every full output-space Jacobian has an exact 10,800-
dimensional null space.

## Answer to the causal question

Family-E1 is not failing merely because the network numerically ignores a weak
prompt. Family-E1P rejects that mechanism: prompt effects survive throughout
the network, and the architecture can express correct two-view identity on
index 81 and the ordinary control. The stricter result here is that a
coordinate prompt carries no information about the continuous allocation of
overlapping blended light. Under the declared nonnegative source-image plus
signed-residual contract, the requested source is therefore not uniquely
recoverable from these observations.

The evidence supports the following layered interpretation.

1. **Information limit:** exact nonuniqueness is present in every scene, even
   when the true signed-noise realization is granted. This is the decisive
   answer to the scientific question.
2. **Scene severity:** index 6 is difficult even in the optimistic oracle-
   template problem; indices 5, 51, 73, and 81 are not. The information limit
   is therefore structural and not reducible to PSF overlap alone.
3. **Architecture limitation:** failures on low-score, well-conditioned
   indices 5 and 51 show that the current architecture/optimizer does not
   reliably learn the simulator's supervised allocation convention. This can
   explain empirical misses, but a better architecture could only choose a
   convention or impose a prior; it could not make the unrestricted observation
   inverse unique.

Accordingly, the prompt is **semantically insufficient for unique source
allocation**, not merely too small in activation magnitude. The requested
identity is fundamentally unrecoverable under the current observation-only,
unrestricted output definition.

## Exactly one recommended experiment

Run **Thayer-Recoverability-Sersic-v1**, a training-free, preregistered
multi-start fit on these same eight frozen observations in which each source is
restricted to one prompt-centered, two-component Sérsic/bulge-disk forward
model with no generator-truth initialization. Retain the current prompts,
noise model, scientific thresholds, Fisher/rank diagnostics, and exact-witness
test. This single experiment will determine whether a physically explicit
single-galaxy morphology prior removes the source-allocation witnesses; a
unique full-rank result would identify missing inductive bias, while persistent
diameter-`>1` witnesses would extend the information-limit conclusion to that
plausible galaxy family.

## Reproducibility and integrity

The analysis is implemented in
`scripts/analyze_thayer_recoverability_v0.py`, SHA-256
`c5b9ae09969130098433ad337b3fb679da987e1bd76688b83bbc0d19d75d621e`.
Two independent executions produced byte-identical 39,985-byte JSON output,
SHA-256
`341ca014147046055dd250e5d5306f25d139c63206ca7d70f77b6821ceeb4d91`.
The script imports only array/data utilities and the already frozen scientific-
model optimizer steps, checkpoint writes, and protected-data accesses were all
zero. Exactly eight training-selector rows and eight training-manifest rows
were materialized. Nothing was staged or committed.
