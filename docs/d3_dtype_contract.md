# D3 NumPy dtype contract

The frozen D1 dtype contract uses NumPy storage tokens, including `<f4` for
native little-endian IEEE float32 on the current platform. User-interface
display strings such as `float32`, `dtype.name`, and `repr(dtype)` are not
contract comparison values.

`canonical_numpy_dtype_token(value)` accepts NumPy dtype objects, NumPy scalar
types, arrays or objects exposing `.dtype`, and supported dtype strings. It
uses `numpy.dtype(value.dtype).str` for dtype-bearing objects and
`numpy.dtype(value).str` otherwise. The returned token preserves kind, item
size, and byte order. Structured and object dtypes are rejected.

`numpy_dtype_contract_equal(actual, expected)` returns a typed record with the
original actual/expected tokens, canonical actual/expected tokens, equality,
platform byte order, and a failure reason. Only canonical tokens are compared.
The original expected token is retained for reporting and is never rewritten
to match an observed display string.

V4.1 verified that all four authoritative D1 endpoint arrays display
`float32` while canonicalizing to the unchanged `<f4` contract. Wrong dtype
and wrong endianness negative tests remained rejecting.

R1 now uses direct NumPy dtype-object equality and a complete 15-field result;
canonical strings are reporting-only. Component tests passed, but the R1
candidate never became independently eligible for scientific loading.
