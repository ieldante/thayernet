# Deployable residual-scale model

## Contract

A deployable Thayer-Select scale model consumes only quantities available from
the observed blend, prompt, and frozen model. Its output is a positive bounded
estimate of raw image- or flux-risk residual magnitude. True outcomes are used
only to form training and validation targets. Physical simulator metadata and
frozen subgroup labels are never inference inputs.

The scale-correction campaign used S0 frozen risk/query outputs, S1 global and
prompt-local latents, S2 reconstruction summaries, S3 observed-blend quality
proxies, and their S4 combination. S3 includes background scale, local
signal/background, variance, gradient, high-frequency, concentration,
cross-band centroid, and structural-disagreement summaries. The local
signal/background quantity is an observed proxy, not true SNR.

## Models and bounds

The fixed comparison included global constant, ridge log-linear,
one-hidden-layer, residual MLP, partially pooled, and three-expert soft-gated
models. Image scales were bounded to `[0.001, 5]`; flux scales to
`[0.001, 25]`. Every trainable condition used five fixed seeds and CPU only.

The partial-pooling model combined a 16-unit global trunk with four strongly
regularized continuous corrections for estimated low local signal, local
complexity, output uncertainty, and input/output disagreement. The soft gate
used the same deployable proxies with at most three experts and explicit
shrinkage and entropy penalties. Neither model accepted subgroup identity.

## Current evidence

The partially pooled model predicted validation residual scale with Spearman
`0.648` image and `0.723` flux and retained natural-calibration risk ranking.
Nevertheless, normalized conformal worst-subgroup coverage was only `0.549`
and `0.679`. These scale models are research artifacts, not authorized policy
components. Their bounded outputs and good ranking do not establish reliable
conditional intervals.

