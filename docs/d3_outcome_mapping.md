# D3 Scientific Outcome Mapping

The executable outcome categories are:

1. `L0_FULL_DECODER_SUCCESS`
2. `DECODER_OPTIMIZATION_BARRIER`
3. `DECODER_PARAMETERIZATION_CAPACITY_BARRIER`
4. `HARD_ASSIGNMENT_BARRIER`
5. `SQUARE_MAPPING_OPTIMIZATION_BARRIER`
6. `MIXED_CAUSE`
7. `MECHANISM_UNRESOLVED`
8. `IMPLEMENTATION_OR_CONTRACT_FAILURE`
9. `NO_SCIENTIFIC_RESULT`

Precedence is implementation or contract failure, no authoritative trajectory,
clean full success, one independently supported mechanism, at least two
supported mechanisms, then unresolved. Conflicting success and failure
evidence fails closed as an implementation or contract failure.

All 256 boolean evidence combinations mapped to exactly one category in the
synthetic audit, and every category was reached. Unresolved, implementation
failure, and no scientific result authorize no downstream campaign. The
capacity ladder is possible only from the capacity-barrier category with D0/D1
passes, no implementation defect, and valid tangent evidence if tangent
evidence was used.
