# Nonnegative flux-conserving deblender

Family-E v0 tested one exact simplex allocation. For every band and pixel,
softmax fractions `a_req`, `a_comp`, and `a_res` sum to one, and outputs
are `P_i = a_i O`. When `O >= 0`, all outputs are nonnegative and their sum
equals `O` to floating-point tolerance.

That construction is mathematically incompatible with a signed
zero-background observation. If `O < 0` at a pixel, at least one summand must
be negative for an exact sum; under the multiplicative simplex, every positive
fraction produces a negative contribution. Likewise, two fixed nonnegative
source targets cannot be represented where their sum exceeds `O`.

The frozen BTK source targets are finite and nonnegative, but the observed
blend includes signed noise after background removal. Family-E v0 measured
this incompatibility in all required partitions and stopped before building a
model. No clipping, sky offset, softplus transformation, or truth-based
renormalization was used.

A future contract must resolve noise semantics prospectively. The one
recommended preflight is to retain nonnegative requested and companion layers
while allowing a signed residual/noise layer
`P_noise = O - P_req - P_comp`. That is a separate hypothesis and is not an
authorized continuation of Family-E v0.

## Signed-noise-residual correction

The recommended correction was tested prospectively and passed. Physical
requested and companion layers use in-forward ReLU, while a signed
non-source residual closes the raw observation identity. All 14,000 frozen
targets were representable with zero mapped-source negatives. This resolves
the physical contract contradiction but does not establish neural learnability
or catalog safety.
