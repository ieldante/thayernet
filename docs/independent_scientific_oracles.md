# Independent Scientific Oracles

Thayer-RI compared production behavior with an audit-only reference
implementation that imports none of the production loss, assignment, coverage,
prompt-swap, or canonical-hash helpers.

Thirteen comparison groups passed with zero mismatches. The checks included
1,000 deterministic cases for each output mapping, 1,000 hard two-permutation
assignments, requested/companion splitting, prompt swap, canonical hashing of
noncontiguous inputs, Gaussian prompt construction, forward evaluation,
scientific distance, batch-order invariance, MPS-to-CPU canonicalization, and a
full-evaluator truth-coverage case.

Seven preregistered golden cases also passed after a malformed negative-valued
audit fixture was preserved and superseded by a valid nonnegative fixture.
Differential truth injection accepted the exact P0 tensor and expert-order
permutation, while rejecting collapsed, source-swapped, band-swapped,
prompt-swapped, and sum-preserving incorrect allocations.

These results support the implementation contract on the audited one-scene
path. They do not establish performance beyond that path.
