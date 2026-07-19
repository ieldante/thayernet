# Family-E physical output contract

Status: **synthetic construction PASS; frozen-target representability FAIL**.

| Check | Result |
|---|---|
| Preregistration frozen before tensor load/model construction | PASS |
| MPS softmax allocation on nonnegative observations | PASS |
| Nonnegative requested/companion/residual on synthetic fixture | PASS |
| Maximum synthetic conservation error | `4.76837158203125e-07` |
| Finite gradients | PASS |
| Zero-source absolute error | `0.0` |
| Raw observations nonnegative | FAIL in all partitions |
| Requested plus companion targets bounded by observation | FAIL in all partitions |
| Post-hoc clipping/offset/background repair | not used |
| Model constructed | no |

The raw observations have signed zero-background noise. The exact frozen
failure token is
`SIGNED_ZERO_BACKGROUND_OBSERVATIONS_INCOMPATIBLE_WITH_NONNEGATIVE_EXACT_SIMPLEX_CONSERVATION`.
See the run-local
`diagnostics/physical_output_contract.md` and
`physical_contract/target_representability.json` for counts.

## Family-E1 correction result

The training-free signed-noise-residual contract passed. Requested and
companion sources remain nonnegative; only the residual/noise closure may be
signed. Maximum source round-trip error was `0.5`, below its frozen
partition tolerance, and maximum float32 conservation error was `0.015625`.
The original all-nonnegative simplex outcome remains a valid failure; it was
not overwritten or reinterpreted.
