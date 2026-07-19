# Micro capacity after projection

Thayer-FP separates feasible-target construction from neural capacity. P0
produced valid targets for every frozen scene, prompt, and expert assignment,
so direct constrained feasibility is resolved for this 64-scene microset.

The unchanged Thayer-ME then failed its capacity test. At the best training
checkpoint, ordinary and ambiguous own/alternate/both-mode coverage were all
zero; ordinary expert diameter was 3.564 rather than at most 1.0. Set prompt
swap was 0.984 and ordinary/ambiguous forward consistency was 0.969/1.000, so
the failure was not prompt collapse or catastrophic recomposition failure.
The final output also violated nonnegativity on 43.6% of pixels.

Current 46,470-parameter expert decoders are not established as sufficient on
the microset. Decoder capacity, shared-encoder conditioning, or neural output
parameterization is directly implicated. The one next experiment is a
controlled, preregistered decoder-capacity ladder that preserves the same P0
targets, thresholds, assignment, prompt contract, data boundary, and
evaluation gates.
