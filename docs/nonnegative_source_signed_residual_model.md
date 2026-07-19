# Nonnegative-source signed-residual model

Family-E1 preserves the raw signed zero-background observation and separates
catalog-source semantics from observational closure:

`P_req = S * ReLU(R_req)`

`P_comp = S * ReLU(R_comp)`

`P_noise = O - P_req - P_comp`

ReLU is inside the model forward path. Requested and companion are the only
catalog-source layers and are nonnegative by construction. `P_noise` is
derived, may have either sign, and is not subject to catalog-source
nonnegativity. Observations, targets, and mapped sources are not clipped or
offset, and truth is not used at inference.

The physical and objective audits passed, but that does not establish model
eligibility. The v0 micro campaign exposed a prompt-ordering failure on the
mandatory eight-scene set, so no full Family-E1 model was frozen.
