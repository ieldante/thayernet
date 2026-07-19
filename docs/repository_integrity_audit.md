# Repository-Integrity Audit

## Scope and result

The Thayer-RI campaign audited the exact Thayer-OP one-scene execution graph
under an exact-path allowlist. It did not scan unrelated source, data,
checkpoints, or outputs. The closed local graph contained ten modules covering
the loader, output mapping, physical loss, hard assignment, scientific
coverage, prompt swap, and canonical hashing.

The audit found no result-changing defect on the active production path. All
391 high-risk static occurrences received an explicit disposition: 151 were
correct on the execution path and 240 were outside it. No production source
correction was made.

## Integrity controls

- The primary preregistration was hashed before any per-scene tensor load.
- Every guarded Python path decision was recorded; denied accesses remained denied.
- Only the frozen ambiguous training scene and its exact P0 row were loaded.
- Ordinary, eight-scene, remaining-microset, Atlas, development, and lockbox
  access counts remained zero.
- The staged index remained empty throughout the scientific run.
- A 600-checkpoint exact inventory matched before and after the campaign with
  zero byte or hash differences.
- Historical artifacts were never overwritten or deleted; failed audit
  attempts remain as superseded evidence.

The full append-only record is under
`outputs/runs/thayer_repository_integrity_20260713_031653/`. This is an
integrity and one-scene diagnostic result, not a deployment, catalog, or
model-selection claim.

## D3 runtime-readiness reuse

Thayer-D3B reused the exact path-guard principles, closed import graph, code
hash inventory, and independent forward-evaluator contract without loading the
one-scene arrays. The strict scientific graph was narrowed to four exact source
modules and contains no Matplotlib edge. All four source hashes matched the
repository-integrity inventory.

Metadata-only checks again matched all 600 historical checkpoints. No
scientific container was deserialized, and Atlas, development, and lockbox
access remained zero. The readiness result is operational only; it does not
alter the repository-integrity scientific conclusions. The authoritative
readiness record persisted initial, bootstrap, strict-end, and post-shutdown
inventories for every process and passed its 26-check closure under
`outputs/runs/thayer_d3_runtime_readiness_20260713_135017/`.
