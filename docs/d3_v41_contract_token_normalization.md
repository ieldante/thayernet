# D3 v4.1 contract-token normalization

Thayer-D3I41 preserves the frozen bundle-v3, bundle-v2, capsule-v1, model,
optimizer, policy, threshold, artifact, mapping, assignment, and execution
budget authorities. Its dtype correction changes only comparison semantics:
the original expected token `<f4` remains frozen, while actual and expected
tokens are independently canonicalized through NumPy dtype semantics before
comparison.

`float32`, `numpy.float32`, `numpy.dtype("float32")`, `=f4`, and `<f4`
canonicalize to `<f4` on the current little-endian platform. Big-endian
`>f4`, `float64`, and `int32` remain distinct. Structured and object dtypes
remain unsupported. Shapes, member names, semantic roles, values, hashes,
units, and thresholds retain separate strict checks.

Candidate 002 passed all four previously failing D1 comparisons. Each row
preserved original actual `float32`, original expected `<f4`, and canonical
actual/expected `<f4`. The later campaign stop was unrelated to dtype
equivalence: a post-validation inventory reporter called a CHW-only hash on a
higher-rank tensor. No source-frozen retry was permitted.

Reusable implementation and tests are in `src/d3_contract_tokens_v41.py`,
`tests/test_d3_contract_tokens_v41.py`, and
`tests/test_thayer_scientific_d3_v41.py`. Generated evidence remains under
`outputs/runs/thayer_d3_v41_science_20260713_200621/`.

## R1 status

R1 implemented direct NumPy dtype-object equality, complete result fields, and
compound-dtype rejection. Exact tests passed, but an unrelated candidate-log
collision stopped the campaign before independent eligibility or science.
