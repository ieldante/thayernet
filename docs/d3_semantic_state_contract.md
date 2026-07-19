# D3 Semantic-State Contract

The required states are `initial`, `one_step`, `lowest_objective`,
`closest_to_d1`, `first_own_coverage`, `first_alternate_coverage`,
`first_both_mode_coverage`, `success`, `terminal_failure`,
`budget_exhausted`, and `final`.

Single-occurrence states reject duplicates. Lowest-objective and closest-to-D1
states retain the lower metric; ties choose the earliest evaluation and then
the lexicographically lower payload SHA-256. Payload names include the semantic
state, evaluation index, and hash prefix and are created exclusively.

The manifest records semantic prompt, expert, and member names; scalar metrics;
optimizer and payload hashes; assignment and event records; and terminal
status. An unreached state remains explicit with reason, terminal campaign
status, and last eligible evaluation. Payloads are append-only. Only the
manifest pointer may be atomically replaced inside the fresh run. Fresh-process
hash replay passed for every reached state.
