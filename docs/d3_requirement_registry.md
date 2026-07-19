# D3 Requirement Registry

The canonical D3 registry is implemented by
`src/d3_requirement_registry.py` and persisted for Thayer-D3E as
`requirement_registry/d3_requirement_registry.json`.

- Requirement count: `180`.
- Registry SHA-256:
  `a1af885bc8e1c6b6bc33395920eb4b279151e51663444e6e303c2f1cfc34660f`.
- Builder required set: `180`.
- Validator required set: `180`.
- Preflight required set: `180`.
- Consumer required set: `180`.
- Runtime accessed-or-validated set: `180`.
- Missing, extra, unaccessed, or undeclared entries: `0`.

Each entry has one canonical identifier, category, type, source, consumer,
validation rule, required/optional status, and failure policy. Components call
the registry API for their required identifiers; they do not maintain private
lists or recover configuration from historical runs. The campaign verifies
set equality before capsule construction and again from the actual consumer's
runtime access record.

The nine requirements absent from capsule v1 are:

1. `capsule_artifact_d1_endpoint_manifest`
2. `capsule_artifact_d0_persisted_evidence`
3. `capsule_artifact_d1_persisted_evidence`
4. `capsule_artifact_d2_persisted_evidence`
5. `capsule_frozen_l0_decoder_topology_code`
6. `capsule_frozen_decoder_parameter_count`
7. `capsule_frozen_decoder_initialization_seeds`
8. `capsule_d1_final_objective_evidence`
9. `capsule_member_shape_dtype_endianness_expectations`

The executable-contract result is append-only. Capsule v1 and its historical
validators remain unchanged evidence of the confirmed producer-consumer drift.
