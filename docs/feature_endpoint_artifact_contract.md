# Feature Endpoint Artifact Contract

A downstream fixed-feature diagnostic may treat a D1 feature endpoint as
authoritative only when the artifact is semantically named and independently
replayable.

The `thayer-d1-endpoint-v1` schema requires four CHW float32 arrays:

- `penultimate_prompt_a_expert_1`
- `penultimate_prompt_a_expert_2`
- `penultimate_prompt_b_expert_1`
- `penultimate_prompt_b_expert_2`

Every array must record shape, dtype, byte order, memory order, finiteness,
canonical per-sample SHA-256, and artifact-array SHA-256. The manifest must also
bind the endpoint to exact frozen-head hashes, P0 target hashes, square mapping,
loss, hard assignment, evaluator and execution-code hashes, optimizer protocol,
raw/mapped/physical output hashes, assignment costs, and scientific metrics.

Completeness requires a fresh process that loads no initial features or
optimizer state and instantiates no encoder or decoder body. That process must
regenerate the exact output hashes and scientific result through immutable
heads. Batch size, batch position, contiguous/noncontiguous layout,
save/reload, CPU canonicalization, and device-to-CPU transfer must also pass.

An output-only artifact is not a substitute for an optimized feature endpoint,
and a feasible D1 endpoint is not assumed unique. Failure of any schema or
replay requirement blocks downstream use without implying a decoder-capacity
result.
