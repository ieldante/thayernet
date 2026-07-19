# Pure Forward-Evaluator Contract

The production forward evaluator for future D3 readiness is
`src.competing_hypotheses.forward_consistency`. It receives the observed blend,
candidate source layers, and sky-electron configuration as explicit arguments.
It performs no path discovery, manifest loading, scene lookup, plotting import,
or global partition access.

Thayer-D3B compares this production function with an independently implemented
reference on twelve synthetic cases:

1. exact two-source sum;
2. one-pixel requested and companion sources;
3. g-only versus z-only sources;
4. source-order swap;
5. prompt-view swap;
6. zero source;
7. known positive residual;
8. wrong band order;
9. noncontiguous input;
10. batch size one versus a multi-item loop;
11. batch reordering; and
12. float32 CPU versus equivalent MPS-to-CPU values.

Every comparison had zero numerical difference. Repeated calls were
deterministic, and the access guard recorded zero filesystem events during all
evaluator calls. The evaluator source hash remained
`e66111b2853c2b954efaa35880ee74d99736c03dc75197fd474fdc390271ca6d`.

This validates path independence and formula agreement on synthetic arrays. It
does not evaluate a scientific scene or supply a D3 result.
