# D3 Capsule Validation

The Thayer-D3C validator checks the strict JSON schema, every required field,
band and channel order, units, finite values, threshold/operator semantics,
code hashes, artifact byte sizes and hashes, runtime hashes, placeholders,
implicit defaults, protected runtime paths, and the completeness marker.

All 16 corruption tests were rejected. All 12 synthetic production/reference
evaluator cases passed deterministically with zero filesystem events inside
evaluator calls. Capsule validation and preflight also passed from the
repository root, a fresh working directory, a process with the relevant
environment variables cleared, and the frozen scientific runtime.

The capsule-only preflight emits exactly:

`ALL_SCIENTIFIC_DEPENDENCIES_SELF_CONTAINED`

`READY_FOR_AUTHORITATIVE_D3_PREREGISTRATION`

These markers authorize only a separately preregistered D3 campaign. The
preflight loads no scientific tensor, constructs no model or optimizer, and
does not execute D3.

