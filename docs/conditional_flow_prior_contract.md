# Conditional flow-prior contract

The intended Thayer-PF intervention was narrow: replace only the frozen
Thayer-PU Gaussian conditional prior with a compact conditional rational-
quadratic-spline flow using a two-component Gaussian-mixture base. The decoder,
posterior, prompt path, normalization, source-layer semantics, forward model,
noise model, Atlas protocol, and truth-coverage metric would remain frozen.

That intervention was conditional on a pre-fit sufficiency result. Posterior
samples had to cover own truths, and latents transferred between each approved
non-Atlas near-collision pair had to decode the paired alternate truth under the
other observation while remaining forward-consistent. The preregistered floors
were 70% ordinary and near-own coverage, 30% cross alternate coverage, 50%
forward-consistent sample fractions, and 70% relevant source identity.

The coverage gates failed at 0%, 0%, and 0%; cross alternate identity was 1.76%.
Therefore the flow architecture, mixture base, parameter ceiling, latent-teacher
pools, likelihood objective, checkpoints, and Atlas sampling protocol were not
created. This document records a blocked contract, not a model specification
that may be implemented without a new prospective campaign.
