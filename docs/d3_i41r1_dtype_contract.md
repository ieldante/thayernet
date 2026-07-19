# D3 v4.1 R1 dtype contract

R1 implements `coerce_numpy_dtype`, a frozen 15-field immutable
`NumpyDTypeContractResult`, and `numpy_dtype_contract_equal`. Equality is
decided only by `actual_dtype == expected_dtype`; canonical `.str` tokens are
reporting-only. Structured, object-containing, and subarray dtypes are rejected
by default, while the frozen expected token remains `<f4`.

All five CSV-required dtype regressions and the inherited dtype suite passed.
No scientific member was loaded in this campaign, so these results establish
component compliance only—not an authoritative scientific D3 result.
