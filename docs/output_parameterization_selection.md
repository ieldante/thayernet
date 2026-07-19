# Thayer-OP output-parameterization selection

Thayer-OP is the fixed-L0 output-parameterization campaign in
`outputs/runs/thayer_output_parameterization_20260713_023120/`. Its
preregistration SHA-256 is
`c6abcb8ba70888bc9a14477968933713c0729a4e32065f7f2becfcec9c468597`
and predates all per-scene tensor loading and fitting. All authoritative input
hashes matched.

The campaign held the Condition-C shared encoder, two independently
initialized 46,470-parameter L0 expert decoders, P0 targets, hard assignment,
prompt contract, optimizer, batch order, and step budget fixed. The only model
change was the in-forward physical mapping: ReLU, square, or absolute value.
Training loss and every evaluation consumed the same mapped physical tensor;
there was no evaluation-only clamp or alternate inverse-normalization path.

All three mappings represented every frozen P0 target. All gradient and
numerical preflights were usable, all five fail-closed stop-rule self-tests
passed, and all 15 synthetic output-head fits passed on MPS. The matched
initial physical value was `9.999999406318238e-08` in normalized units. No
physical negative or nonfinite value occurred in any real condition, and the
frozen encoder remained byte-identical.

The final one-scene results were:

| Mapping | Ordinary coverage | Ordinary diameter | Ambiguous own | Ambiguous alternate | Ambiguous both | Ordinary z-MSE | Ambiguous z-MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReLU | 0% | 10.0000 | 0% | 0% | 0% | 2.57033e-4 | 2.11664e-6 |
| Square | 0% | 0.9160 | 0% | 0% | 0% | 4.89752e-6 | 4.16324e-7 |
| Absolute value | 0% | 0.8475 | 0% | 0% | 0% | 4.53913e-6 | 8.88526e-7 |

Both experts remained active and prompt swap passed, but no mapping placed both
ordinary experts in the target region and no mapping recovered both ambiguous
truth modes at the frozen final endpoint. The primary outcome is **NO MAPPING
PASSES**. The frozen stop rule therefore prevented eight-scene fitting; the
remaining 56 microset rows were not loaded.

No output mapping is selected, and the decoder-capacity ladder is not
authorized. The remaining blocker is one-scene truth-mode memorization through
the frozen encoder, L0 expert-decoder, and hard-assignment optimization path,
not target representability or physical nonnegativity. Exactly one next
diagnostic is authorized: run a fixed-feature L0 expert-decoder optimization
audit on the frozen ambiguous scene, retaining the same hard assignment and
mapping while comparing the neural decoder trajectory with direct
cached-feature output optimization.

The blocked Thayer-CL campaign measured no capacity result, and Thayer-OP also
made no capacity comparison. Atlas, development, and lockbox data remained
untouched. All historical checkpoints remained unchanged.

## Repository-integrity fixed-feature closure

Thayer-RI independently validated the active mapping, loss, assignment, and
coverage path and found no result-changing production defect. Under the exact
one-scene fixed-feature ladder, square alone passed direct raw-logit reachability
(D0) and free-penultimate reachability with its frozen rank-six head (D1).
ReLU and absolute value failed D0. Square then failed the frozen-penultimate,
final-head-only D2 coverage gate. This does not select or promote square as a
general output parameterization; it only identifies the mapping that remained
reachable longest in this one-scene diagnostic. D3 was not authorized.

## Thayer-D3 follow-up

Square remained the sole eligible mapping, and Thayer-D3 changed neither the
mapping nor its scientific contract. The campaign stopped before optimization
because the persisted successful D1 artifact did not contain its optimized
penultimate tensors. This is an evidence-persistence failure, not a new mapping
comparison or a square decoder-training result. Square remains unselected for
broader use.

## Thayer-D3R square follow-up

The authoritative square-only retry stopped before decoder optimization due to
a terminal access-guard/runtime-bootstrap event. It does not change mapping
selection: square remains diagnostically reachable in D0/D1 but unselected for
broader use, and no neural decoder-training practicality result was produced.

## Thayer-D3A square follow-up

D3A preserved square as the sole diagnostic mapping and changed no mapping
contract. It stopped at preregistration because the isolated evidence lacked
the exact forward-gate sky and plausibility-threshold values. This supplies no
new mapping comparison or neural square-optimization result; square remains
unselected for broader use.
