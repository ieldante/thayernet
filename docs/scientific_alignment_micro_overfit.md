# Thayer-SA scientific-alignment micro-overfit

The authoritative run is
`outputs/runs/thayer_scientific_alignment_20260712_220315/`. Thayer-SA was a
prospective micro-overfit-only correction for the frozen Thayer-ME architecture
and 64-scene training microset. The Thayer-LG diagnosis reproduced before the
objective and gates were frozen.

The differentiable scientific surrogate passed unit, rank, threshold-side,
exact-truth, and perturbation tests. Exact truth was a zero-loss stationary
point and remained fully covered under the official detached CPU optimizer.
However, trained, collapsed, and source-sum-preserving compromise starts did
not enter at least 90% ordinary and ambiguous truth coverage, and random
bounded outputs barely moved under the frozen protocol. The decision is
`FAILURE — CORRECTED OBJECTIVE STILL MISALIGNED`.

The hard-assignment audit and MPS neural micro-overfit were not reached. No
checkpoint was created, and full non-Atlas training is not authorized. Atlas,
development, and lockbox access remained zero. The one next experiment is a
training-free output-space conditioning campaign that keeps the same thresholds,
targets, architecture, and hard assignment while prospectively testing a
near-truth smooth component geometry; it must pass detached optimization before
any neural fitting.

## Protocol addendum

Pre-freeze detached-output optimizer smoke checks were used during numerical
implementation debugging. The persisted official protocol was frozen before it
ran, and no neural fit or scientific gate changed, but the strict requirement
that preregistration predate every detached optimization was not met. The
superseding correctness status is therefore FAIL with one protocol violation;
the scientific failure and prohibition on training are unchanged.
