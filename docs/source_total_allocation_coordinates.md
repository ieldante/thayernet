# Source-total and allocation coordinates

Thayer-OC defines physical requested and companion source layers by

`T = S_req + S_comp`

`D = 0.5 * (S_req - S_comp)`

with exact inverse

`S_req = 0.5 * T + D`

`S_comp = 0.5 * T - D`.

Common-mode changes alter requested and companion layers equally. Allocation-
mode changes are equal and opposite and preserve total source light. The
coordinate audit passed exact truths, persisted Thayer-SA and Thayer-ME
outputs, collapsed means, source-sum-preserving wrong allocations, and a
deterministic random valid tensor with zero recorded float64 round-trip error.

The frozen target-independent projection decodes physical sources, replaces
nonfinite values by zero, clamps requested and companion pixels at zero, and
then re-encodes T/D. It uses no truth information beyond the unchanged
supervised objective. Projection affected 8.19% of pixels in the persisted
Thayer-SA compromise and 42.09% in the persisted Thayer-ME outputs.

The transformation is algebraically exact, but optimizer stationarity is a
separate numerical requirement. Float32 physical decode and normalization left
a `3.36e-7` exact-truth objective residual for the T/D conditions. Adam-based
T/D methods amplified that residual and left full truth coverage; they are
ineligible. T/D L-BFGS retained truth coverage but did not pass the global
coverage gates. A neural T/D head is therefore not authorized by this audit.
