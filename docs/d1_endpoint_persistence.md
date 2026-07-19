# D1 Endpoint Persistence

Thayer-D1R reconstructs the missing successful square-D1 penultimate endpoint
for the exact authoritative ambiguous scene. It is an artifact-reconstruction
campaign, not D3 or decoder training.

The replay used the frozen paired prompt tensors, restored rank-six final
heads, square mapping, P0 targets, direct requested-plus-companion objective,
hard two-permutation assignment, AdamW learning rate `0.03`, gradient clip
`5.0`, and exactly 5,000 MPS steps. Only two detached `[2,16,60,60]`
penultimate tensors were optimized; no neural parameter entered the optimizer.

All 54 scheduled physical tensors matched the repository-integrity trajectory
exactly. The final objective was `3.1026115010490685e-09`, the combined physical
hash was `79ce0af8503282208538c2717efe024bd3df83808f3fead07ff624cda85b9229`,
and own, alternate, and both-mode coverage were each 100%.

The endpoint archive now names every semantic tensor explicitly for prompt A
and B and expert 1 and 2. A separate restricted process regenerated every raw,
mapped, and physical output from only the endpoint, frozen heads, isolated P0
targets, square mapping, evaluator, and manifest. It reproduced all hashes,
metrics, assignments, prompt semantics, and expert diameter. Batch size,
position, noncontiguous layout, serialization, CPU canonicalization, and
MPS-to-CPU transfer were invariant with zero difference.

This resolves the artifact blocker that stopped the earlier D3 campaign. It
does not supply a D3 result or a decoder-capacity conclusion. It authorizes
only a separately preregistered one-scene square-only D3 campaign. Ordinary,
eight-scene, remaining-microset, Atlas, development, and lockbox data remained
untouched.

D3 was not run in Thayer-D1R.

## Thayer-D3R disposition

The subsequent authoritative retry froze a complete D1R reference but stopped
during guarded runtime bootstrap, before optimizer construction. A dependency
attempted a prohibited cache deletion and the temporary-directory probe could
not complete under the no-delete contract. D1R remains valid and complete;
the retry supplies no D3 trajectory or capacity result.

## Thayer-D3A disposition

The later D3A preregistration matched the D1R endpoint container and manifest
hashes without deserializing them. It stopped because the isolated evidence did
not contain the exact forward-gate sky and plausibility-threshold values. D1R
remains complete and unchanged, was not used for supervision or initialization,
and supplies no D3 or decoder-capacity result.

## Thayer-D3C downstream use

Thayer-D3C references the D1 endpoint archive only by immutable path, byte
size, expected member names, schema, and SHA-256. It did not deserialize the
endpoint. The complete capsule now carries the scientific sky, thresholds,
units, rules, and code hashes that D1R did not aim to package.

D1R remains an evaluation-only reference and is not initialization,
supervision, loss, tuning, or selection input. Capsule completion permits one
separately preregistered D3 campaign but supplies no D3 or capacity result.

## Thayer-D3E endpoint member contract

Capsule v2 adds the previously absent D1 endpoint-manifest and final-objective
evidence requirements. A header-only audit verified the four expected
prompt/expert endpoint members as float32 tensors of shape `[16,60,60]`, with
exact endianness and canonical hashes. The values were not deserialized and the
endpoint remained evaluation-only. This closes the executable member contract,
not the scientific D3 question.

## Thayer-D3S status

The D1 endpoint remained diagnostic-only and was not deserialized by D3S. The
campaign stopped before preregistration because the executable bundle lacks
required trajectory gate definitions. D1 persistence remains valid, but it
does not authorize an incompletely specified D3 run.

## Thayer-D3I endpoint load

The D1 endpoint container hash and four canonical member hashes matched. The
v4 worker nevertheless stopped because it compared NumPy's `float32` display
string to the v2 dtype token `<f4` rather than comparing normalized dtype
identity. No endpoint value was changed, and the endpoint remains
diagnostic-only.

## V4.1 dtype validation

All four D1 endpoint members preserved display token `float32`, expected token
`<f4`, canonical actual `<f4`, canonical expected `<f4`, and PASS status. No
endpoint value or hash changed. The later member-inventory failure occurred
after these checks and does not alter D1 persistence evidence.

R1 did not open D1 payloads. D1 evidence and hashes remain frozen; the campaign
stopped during candidate orchestration before independent payload authorization.
