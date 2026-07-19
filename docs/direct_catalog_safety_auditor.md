# Direct catalog-safety auditor

Thayer-Audit v0 implements a hierarchical external safety policy around an
unchanged frozen deblender.

PRE-AUDIT receives four channels—observed g/r/z and a Gaussian coordinate
prompt—and predicts `VALID`, `NULL_OR_WRONG`, or
`AMBIGUOUS_OR_UNSUPPORTED`. It may abstain before any reconstruction is
requested.

POST-AUDIT applies only after PRE predicts valid. It receives ten image
channels—blend g/r/z, prompt, proposed reconstruction g/r/z, and
observation-minus-reconstruction residual g/r/z—plus 25 deployable scalar
diagnostics. It predicts whether the proposal is unsafe to catalog and may
reject the proposed reconstruction.

Neither network receives clean truth, target masks, true error, source or
duplicate-group identity, deblender-family identity, physical difficulty,
true SNR, obstruction, separation, flux ratio, morphology, generator
parameters, gradients, optimizer state, or D3 trajectory features. Family and
source-group fields exist only for provenance, leakage checks, grouping, and
evaluation.

The v0 result does not establish a usable catalog policy. Query classification
was informative, but all eligible Condition-C valid reconstructions were
unsafe under the frozen safety contract, leaving POST with no safe comparison
class. The fail-closed threshold accepted nothing. Exact conditional coverage
is not claimed.
