# D3 Executable Bundle

The append-only Thayer-D3E bundle is
`outputs/runs/thayer_d3_executable_contract_20260713_164320/future_d3_bundle/d3_executable_bundle_v2.json`.
Its required SHA-256 is
`884d35e7ee385b2bb0b7d0e65aaf6bc18121fa209db6a17cc101ef12e32f4045`.

The bundle freezes capsule v2 and its schema, manifest, and hash chain; the
180-item requirement registry; scientific artifact file and member contracts;
the exact L0 constructor, source, initial states, and square mapping; production
assignment, loss, evaluator, runtime guard, and optimizer identities; the
synthetic preflight result; and the required consumer markers. A separate
schema, manifest, and checksum accompany the bundle.

`scripts/run_thayer_authoritative_d3_v2.py` accepts exactly one bundle path,
one expected bundle SHA-256, and one fresh output directory. It does not query
historical runs for configuration. Its preflight validates the bundle, capsule,
registry, artifact members, model construction, initial states, runtime, and
the frozen synthetic result before emitting
`READY_FOR_AUTHORITATIVE_D3_EXECUTION`.

The marker authorizes a future separately preregistered scientific D3 campaign;
it is not evidence that D3 was run or that the decoder passes any scientific
gate. The next campaign must freeze this exact bundle hash before loading any
scientific value.

## Thayer-D3S completeness finding

The bundle hash matched, but the separately governed D3S campaign requires
execution definitions that are absent from the 180-entry registry, most
critically the expert-activity/death gate. No threshold or decision mapping was
inferred. D3S stopped before preregistration and science; this bundle must not
be used for a scientific D3 trajectory.

## Append-only bundle v3

Thayer-D3P leaves bundle v2 unchanged and adds bundle v3 with executable expert
activity/death, prompt-collapse, tangent, outcome, semantic-state, precedence,
authorization, safety, persistence, and diagnostic policies. The actual
launcher executed all 106 policy branches synthetically and emitted the four
v3 readiness markers.

The required bundle-v3 SHA-256 is
`30ac88c635774d0fb4518bedde66fa459d67b1c1a323816c12d1e37b4614b61c`.
Only a new separately preregistered scientific D3 campaign may use it.
