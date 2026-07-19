# D3 scientific execution result

Primary outcome: **IMPLEMENTATION_OR_CONTRACT_FAILURE**.

The v4 integration itself passed and froze correctly. Authoritative continuation
loaded eight exact allowlisted containers containing 91 counted members, all
with matching container hashes. It then stopped before D0/D1/D2 reproduction,
model construction, initial alignment, optimizer construction, one-step tracing,
or trajectory execution.

The immediate worker error was `D3I-D1-MEMBER-CONTRACT`: the D1 endpoint array
is little-endian float32, but the frozen worker compared `str(dtype) == "<f4"`;
NumPy renders that dtype as `float32`. The strict guard also blocked two
non-protected external bytecode-cache reads. Bundle-v3 precedence selected
`ACCESS_GUARD_VIOLATION`; the outcome engine assigned
`IMPLEMENTATION_OR_CONTRACT_FAILURE`; downstream authorization is `none`.

Consequently, D3 own, alternate, and both-mode coverage remain scientifically
unknown. No L0 capacity, optimization, assignment, or square-mapping conclusion
follows. Atlas, development, lockbox, ordinary/eight-scene, and
remaining-microset access counts are zero.
