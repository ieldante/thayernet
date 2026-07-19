# Pre- and post-deblend audit

The operational order is:

1. PRE-AUDIT examines only the blend and coordinate prompt.
2. A null/wrong or ambiguous/unsupported prediction abstains immediately.
3. A predicted-valid request is sent to the unchanged frozen deblender.
4. POST-AUDIT examines the blend, prompt, proposal, residual, and deployable
   diagnostics.
5. An unsafe prediction abstains; only a request passing both gates may enter
   a catalog.

Thayer-Audit v0 tested this order without training or changing a
reconstruction model. Its training predictions came from a historical held-out
base fold: both source groups in every selected training episode were absent
from Condition-C fitting and validation-based checkpoint selection. This is a
source-group-safe held-out-fold design, not a claim of complete K-fold
base-model cross-fitting.

The PRE gate produced useful unsupported-query detection but narrowly failed
its formal calibration macro-F1 requirement. The POST domain contained only
unsafe valid reconstructions. Consequently no threshold could simultaneously
retain at least 50% valid coverage and halve unsafe/catastrophic rates. The
policy froze fail-closed at zero accepted coverage.

D3's multi-hypothesis decoder failure does not cause this outcome: D3 and the
external auditor test different hypotheses. The binding v0 limitation is the
lack of a physically compliant, scientifically safe comparison domain and a
second aligned frozen family.
