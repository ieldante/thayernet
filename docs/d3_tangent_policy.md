# D3 Optional Tangent Policy

Tangent diagnostics are disabled by default and may run only after the
authoritative trajectory, checkpoint manifest, and primary outcome are frozen.
They cannot change checkpoints, primary metrics, or scientific success or
failure.

The central finite-difference relative steps are `0.001`, `0.0003`, and
`0.0001`. JVP/VJP relative tolerance is `0.0001`, with absolute floor `1e-12`.
The protocol uses eight Rademacher probes, seed `20260713`, six semantic tensor
roles, and at most 64 forward-equivalent evaluations.

Unavailable JVP/VJP, finite-difference mismatch, sign or scale mismatch,
insufficient precision, or a condition-number claim produces
`TANGENT_DIAGNOSTIC_UNRESOLVED`. Tangent failure is nonterminal. Capacity
authorization may use tangent evidence only when the protocol passes.
