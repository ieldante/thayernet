# Post-final protocol addendum

This addendum supersedes only the primary report's unqualified preregistration
correctness claim. The scientific decision remains **FAILURE — CORRECTED
OBJECTIVE STILL MISALIGNED**, and neural fitting and full non-Atlas training
remain prohibited.

Before the preregistration freeze, unpersisted CPU detached-output optimizer
smoke checks were run while debugging numerical conditioning of the implemented
surrogate. They used the frozen microset. No neural model was fitted, no neural
parameter received a gradient, and no objective coefficient, scientific
threshold, coverage gate, target, architecture, or protected-data boundary was
changed as a result. The official 400-step evidence protocol and all reported
decision gates were frozen before their persisted execution.

Nevertheless, if detached output optimization is included in the word
"fitting," the requirement that preregistration predate every fitting action was
not satisfied. The authoritative correctness status is therefore **FAIL** with
one protocol failure, rather than the primary audit's PASS. The official
preflight remains useful as a frozen replication of the same failure, but this
run cannot be described as fully prospective under the strictest boundary.

- Primary report SHA-256:
  `e6f3abd33f934849902d127434a08538819d5ea4317a73bd5929e5708f75706c`.
- Primary correctness-audit SHA-256:
  `304795f081e1782af50d3700c9eae9ed9fb77c5a61c024beefc3e7c147a202df`.
- Neural optimizer steps / checkpoints: 0 / 0.
- Atlas / development / lockbox accesses: 0 / 0 / 0.
- Historical checkpoints: 575/575 unchanged.

The exact next experiment remains one new, collision-free, training-free
output-space conditioning campaign. It must freeze and hash every objective,
optimizer, and gate before any detached optimization is executed.
