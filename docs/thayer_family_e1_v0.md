# Thayer-Family-E1-v0

The authoritative model-eligibility run is
`outputs/runs/thayer_family_e1_v0_20260714_214715/`. Its outcome is
**FAMILY_E1_RECONSTRUCTION_FAILURE**.

The signed-residual precondition passed before model construction. Family-E1
used one 1,162,662-parameter coordinate U-Net, in-forward ReLU requested and
companion layers, and the exact derived signed residual
`O - P_req - P_comp`. Catalog-source outputs stayed nonnegative and finite,
and observation conservation remained within the frozen tolerance.

Objective alignment passed: exact truth was stationary and no compromise beat
truth. Ordinary one-scene micro-overfit passed. The difficult and mandatory
mixed-eight tests failed prompt identity at 0.50 and 0.5625, respectively,
despite strong objective and source-L1 reductions. The preregistered stop
therefore prohibited full three-seed training, fold fitting, OOF outputs,
replay, safety labels, family comparison, bootstrap, and auditor training.

Thayer-Audit v1 is not authorized. The single next experiment is a separately
preregistered micro-only **Family-E1P Paired-Prompt Identity Intervention** on
the same frozen micro scenes, retaining the physical contract and requiring
the unchanged 0.90 prompt-identity gate before any full training.

Development, Atlas selection, and the final lockbox were untouched. Nothing
was staged or committed.
