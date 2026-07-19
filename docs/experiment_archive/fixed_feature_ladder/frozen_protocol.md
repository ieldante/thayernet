# Feature-Cache Protocol Clarification — Superseding v4

- The MPS encoder is not bitwise batch-size invariant; v1 remains failed and no exact-identity claim is made.
- Maximum feature difference across the tested batch contracts: `8.34465026855e-07` normalized feature units.
- Maximum propagated physical-output difference across all restored mappings: `0.00103759765625` detected electrons.
- Frozen physical numerical tolerance: `0.00390625` detected electrons.
- Scientific assignment/coverage metrics identical across tested batch contracts: `True`.
- Decision: `PASS`.

All D0-D3 conditions must load the single persisted joined-A/B batch-size-2 cache. This clarification records a bounded MPS-kernel numerical property; it does not change the encoder, data, targets, mappings, loss, assignment, thresholds, or model capacity.
