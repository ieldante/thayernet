# D3 Executable Contract

Thayer-D3E completed the executable-contract audit in
`outputs/runs/thayer_d3_executable_contract_20260713_164320/`.

Status: **EXECUTABLE D3 CONTRACT PASS — SCIENTIFIC D3 NOT RUN**.

The preceding capsule-driven attempt stopped before tensor deserialization,
model construction, optimizer construction, or a decoder forward because its
actual consumer required nine entries that capsule v1 did not declare. Capsule
v1 was scientifically complete for the values in its own schema, but it was
consumer-incomplete. Its producer, base validator, and hash-chain validator
agreed with one another and therefore could not detect dependencies declared
only by the downstream consumer.

Thayer-D3E resolved that drift with one canonical 180-requirement registry.
The capsule-v2 builder, validator, preflight, and actual consumer each derive
their required set from that registry. Exact set equality and complete runtime
access closure both passed. Capsule v2 includes the D0, D1, and D2 evidence
references; the D1 endpoint manifest and objective evidence; the complete L0
construction and initial-state contract; and member-level shape, dtype,
endianness, role, and hash expectations.

The exact two-expert square L0 architecture instantiated with 46,470 trainable
parameters per expert. Both initial states loaded strictly. A deterministic
production-shape synthetic MPS execution passed forward, production/reference
assignment and loss comparisons, pure-evaluator comparison with zero file I/O,
one AdamW backward and update, checkpoint save/reload, and fresh-process replay.
All 25 corrupted capsules failed before model execution with the expected
canonical requirement identifier. The actual consumer emitted both
`ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED` and
`READY_FOR_AUTHORITATIVE_D3_EXECUTION`.

No scientific scene, target, cached-feature, or D1 endpoint value was loaded.
No scientific decoder forward or D3 optimizer step occurred. Atlas,
development, and lockbox access were zero. The result authorizes exactly one
new separately preregistered authoritative D3 campaign using executable bundle
SHA-256
`884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`.
It does not provide a D3 scientific result, authorize broader data, or
authorize a decoder-capacity ladder.

## Thayer-D3S downstream regression

The downstream bundle-only preregistration audit found that the 180-entry
contract does not freeze the required expert-activity/death gate,
prompt-collapse stop, optional tangent protocol, six-category outcome mapping,
or complete semantic-state rules. Thayer-D3S therefore stopped with
**EXECUTABLE BUNDLE REGRESSION — D3 NOT RUN** before third-party imports or
scientific tensor loads. The executable-contract pass remains a synthetic
readiness result, not an executable scientific-trajectory contract.

## Thayer-D3P policy closure

Thayer-D3P preserved bundle v2 and its 180-entry registry and added a separate
16-policy executable registry. The actual launcher now delegates policy
preflight to one pure engine. Seventy-six scalar/event fixtures executed all
106 declared branches; all 30 bundle-v3 corruptions were rejected.

Bundle-v3 SHA-256 is
`30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`.
This closes the control-policy defect without producing a scientific D3
trajectory.

## V4 bridge result

Bridge v4 resolves this bundle only through bundle v3 and preserves its
architecture, artifact, optimizer, and numerical contracts. Synthetic
execution passed. In scientific mode, literal comparison of NumPy's `float32`
display token to the contract token `<f4` caused a frozen implementation stop
before model construction; the bundle-v2 architecture was not scientifically
executed.
